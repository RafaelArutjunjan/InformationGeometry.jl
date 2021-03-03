

"""
    InformNames(DS::AbstractDataSet, sys::ODESystem, observables::Vector{<:Int})
Copy the state names saved in `ODESystem` to `DS`.
"""
function InformNames(DS::AbstractDataSet, sys::ODESystem, observables::AbstractVector{<:Int})
    newxnames = xnames(DS) == CreateSymbolNames(xdim(DS),"x") ? [string(sys.iv.name)] : xnames(DS)
    newynames = ynames(DS) == CreateSymbolNames(ydim(DS),"y") ? string.(sys.states[observables]) : ynames(DS)
    InformNames(DS, newxnames, newynames)
end

# No ObservationFunction, therefore try to use sys to infer state names of ODEsys
function DataModel(DS::AbstractDataSet, sys::Union{ODESystem,ODEFunction}, u0::Union{AbstractArray{<:Number},Function},
                        observables::Union{AbstractVector{<:Int},BitArray}=collect(1:length(u0)), args...; tol::Real=1e-7,
                        meth::OrdinaryDiffEqAlgorithm=GetMethod(tol), Domain::Union{HyperCube,Bool}=false, kwargs...)
    newDS = (typeof(observables) <: AbstractVector{<:Int} && sys isa ODESystem) ? InformNames(DS, sys, observables) : DS
    DataModel(newDS, GetModel(sys, u0, observables; tol=tol, Domain=Domain, meth=meth), args...; kwargs...)
end

function GetModel(func::Function, u0::Union{AbstractArray{<:Number},Function}, observables::Union{Function,AbstractVector{<:Int},BitArray}=collect(1:length(u0)); tol::Real=1e-7,
                    meth::OrdinaryDiffEqAlgorithm=GetMethod(tol), Domain::Union{HyperCube,Bool}=false, inplace::Bool=true)
    GetModel(ODEFunction{inplace}(func), u0, observables; tol=tol, Domain=Domain, meth=meth, inplace=inplace)
end

# Allow option of passing Domain for parameters as keyword
function GetModel(sys::ODESystem, u0::Union{AbstractArray{<:Number},Function}, observables::Union{AbstractVector{<:Int},BitArray,Function}=collect(1:length(u0));
                tol::Real=1e-7, Domain::Union{HyperCube,Bool}=false, meth::OrdinaryDiffEqAlgorithm=GetMethod(tol), inplace::Bool=true)
    # Is there some optimization that can be applied here? Modollingtoolkitize(sys) or something?
    Model = GetModel(ODEFunction{inplace}(sys), u0, observables; tol=tol, Domain=Domain, meth=meth, inplace=inplace)
    if Model isa ModelMap       Model = Model.Map    end
    pnames = [string(x.name) for x in sys.ps]
    ylen = if observables isa Function      # ObservationFunction
        # Might still fail if states u are a Matrix.
        num = GetArgLength(z->observables(z,0.1))
        length(observables(ones(num),0.1))
    else
        observables isa BitArray ? sum(observables) : length(observables)
    end
    plen = if Domain isa HyperCube
        length(Domain)
    elseif u0 isa AbstractArray     # Vector / Matrix
        # initial conditions given as array means the parameters are only the sys.ps
        length(pnames)
    else        # SplitterFunction
        # May well fail depending on how splitter function is implemented
        GetArgLength(u0)
    end
    xyp = (1, ylen, plen)
    Domain = isa(Domain, Bool) ? FullDomain(xyp[3]) : Domain

    pnames = plen - length(pnames) > 0 ? vcat(CreateSymbolNames(plen - length(pnames), "u"), pnames) : pnames
    # new(Map, InDomain, Domain, xyp, pnames, StaticOutput, inplace, CustomEmbedding)
    ModelMap(Model, θ->true, Domain, xyp, pnames, Val(false), Val(false), Val(true))
end

# Vanilla version with constant array of initial conditions and vector of observables.
function GetModel(func::ODEFunction{T}, u0::AbstractArray{<:Number}, observables::Union{AbstractVector{<:Int},BitArray}=collect(1:length(u0)); tol::Real=1e-7,
                    meth::OrdinaryDiffEqAlgorithm=GetMethod(tol), Domain::Union{HyperCube,Bool}=false, inplace::Bool=true) where T
    @assert T == inplace
    u0 = PromoteStatic(u0, inplace)

    function GetSol(θ::AbstractVector{<:Number}, u0::AbstractArray{<:Number}; tol::Real=tol, max_t::Number=10., meth::OrdinaryDiffEqAlgorithm=GetMethod(tol), kwargs...)
        odeprob = ODEProblem(func, u0, (0., max_t), θ)
        solve(odeprob, meth; reltol=tol, abstol=tol, kwargs...)
    end
    function Model(t::Number, θ::AbstractVector{<:Number}; observables::Union{AbstractVector{<:Int},BitArray}=observables, u0::AbstractArray{<:Number}=u0,
                                                        tol::Real=tol, max_t::Number=t, meth::OrdinaryDiffEqAlgorithm=GetMethod(tol), FullSol::Bool=false, kwargs...)
        sol = GetSol(θ, u0; tol=tol, max_t=t, meth=meth, save_everystep=false, save_start=false, save_end=true, kwargs...)
        FullSol && return sol
        sol.u[end][observables]
    end
    function Model(ts::AbstractVector{<:Number}, θ::AbstractVector{<:Number}; observables::Union{AbstractVector{<:Int},BitArray}=observables, u0::AbstractArray{<:Number}=u0,
                                                tol::Real=tol, max_t::Number=maximum(ts), meth::OrdinaryDiffEqAlgorithm=GetMethod(tol), FullSol::Bool=false, kwargs...)
        # sol = GetSol(θ; tol=tol, max_t=max_t, meth=meth, tstops=ts, save_everywhere=false, kwargs...)
        # FullSol && return sol
        ## Slow method:        Reduction(map(t->sol(t)[observables], ts))
        # [sol.u[findnext(x->x==t,sol.t,i)][observables] for (i,t) in enumerate(ts)] |> Reduction
        sol = GetSol(θ, u0; tol=tol, max_t=max_t, meth=meth, saveat=ts, kwargs...)
        FullSol && return sol
        length(sol.u) != length(ts) && throw("ODE integration failed, try using a lower tolerance value.")
        [sol.u[i][observables] for i in 1:length(ts)] |> Reduction
    end
    MakeCustom(Model, Domain)
end

"""
`ObservationFunction` should be of the form `F(u,t) -> Vector`. Note that the independent variable, i.e. `t` MUST be passed as the second argument.
"""
function GetModel(func::ODEFunction{T}, u0::AbstractArray{<:Number}, ObservationFunction::Function; tol::Real=1e-7,
                    meth::OrdinaryDiffEqAlgorithm=GetMethod(tol), Domain::Union{HyperCube,Bool}=false, inplace::Bool=true) where T
    @assert T == inplace
    u0 = PromoteStatic(u0, inplace)

    function GetSol(θ::AbstractVector{<:Number}, u0::AbstractArray{<:Number}; tol::Real=tol, max_t::Number=10., meth::OrdinaryDiffEqAlgorithm=GetMethod(tol), kwargs...)
        odeprob = ODEProblem(func, u0, (0., max_t), θ)
        solve(odeprob, meth; reltol=tol, abstol=tol, kwargs...)
    end
    function Model(t::Number, θ::AbstractVector{<:Number}; ObservationFunction::Function=ObservationFunction, u0::AbstractArray{<:Number}=u0,
                                                tol::Real=tol, max_t::Number=t, meth::OrdinaryDiffEqAlgorithm=GetMethod(tol), FullSol::Bool=false, kwargs...)
        sol = GetSol(θ, u0; tol=tol, max_t=t, meth=meth, save_everystep=false, save_start=false, save_end=true, kwargs...)
        FullSol && return sol
        ObservationFunction(sol.u[end], t)
    end
    function Model(ts::AbstractVector{<:Number}, θ::AbstractVector{<:Number}; ObservationFunction::Function=ObservationFunction, u0::AbstractArray{<:Number}=u0,
                                            tol::Real=tol, max_t::Number=maximum(ts), meth::OrdinaryDiffEqAlgorithm=GetMethod(tol), FullSol::Bool=false, kwargs...)
        sol = GetSol(θ, u0; tol=tol, max_t=max_t, meth=meth, saveat=ts, kwargs...)
        FullSol && return sol
        length(sol.u) != length(ts) && throw("ODE integration failed, try using a lower tolerance value.")
        [ObservationFunction(sol.u[i], sol.t[i]) for i in 1:length(ts)] |> Reduction
    end
    MakeCustom(Model, Domain)
end

"""
`SplitterFunction` should be of the form `F(θ) -> (u0, p)`, i.e. the output is a tuple whose first entry is the initial condition for the ODE model and the second entry constitutes the parameters which go on to enter the `ODEFunction`.
Typically, a fair bit of additional performance can be gained from ensuring that `SplitterFunction` outputs the initial condition `u0` as type `MVector` or `MArray`, if it has less than ~100 components.
"""
function GetModel(func::ODEFunction{T}, SplitterFunction::Function, observables::Union{AbstractVector{<:Int},BitArray}=collect(1:length(u0)); tol::Real=1e-7,
                    meth::OrdinaryDiffEqAlgorithm=GetMethod(tol), Domain::Union{HyperCube,Bool}=false, inplace::Bool=true) where T
    @assert T == inplace

    function GetSol(θ::AbstractVector{<:Number}, SplitterFunction::Function; tol::Real=tol, max_t::Number=10., meth::OrdinaryDiffEqAlgorithm=GetMethod(tol), kwargs...)
        u0, p = SplitterFunction(θ);        odeprob = ODEProblem(func, u0, (0., max_t), p)
        solve(odeprob, meth; reltol=tol, abstol=tol, kwargs...)
    end
    function Model(t::Number, θ::AbstractVector{<:Number}; observables::Union{AbstractVector{<:Int},BitArray}=observables, SplitterFunction::Function=SplitterFunction,
                                tol::Real=tol, max_t::Number=t, meth::OrdinaryDiffEqAlgorithm=GetMethod(tol), FullSol::Bool=false, kwargs...)
        sol = GetSol(θ, SplitterFunction; tol=tol, max_t=t, meth=meth, save_everystep=false, save_start=false, save_end=true, kwargs...)
        FullSol && return sol
        sol.u[end][observables]
    end
    function Model(ts::AbstractVector{<:Number}, θ::AbstractVector{<:Number}; observables::Union{AbstractVector{<:Int},BitArray}=observables, SplitterFunction::Function=SplitterFunction,
                                tol::Real=tol, max_t::Number=maximum(ts), meth::OrdinaryDiffEqAlgorithm=GetMethod(tol), FullSol::Bool=false, kwargs...)
        sol = GetSol(θ, SplitterFunction; tol=tol, max_t=max_t, meth=meth, saveat=ts, kwargs...)
        FullSol && return sol
        length(sol.u) != length(ts) && throw("ODE integration failed, try using a lower tolerance value.")
        [sol.u[i][observables] for i in 1:length(ts)] |> Reduction
    end
    MakeCustom(Model, Domain)
end

"""
`SplitterFunction` should be of the form `F(θ) -> (u0, p)`, i.e. the output is a tuple whose first entry is the initial condition for the ODE model and the second entry constitutes the parameters which go on to enter the `ODEFunction`.
Typically, a fair bit of additional performance can be gained from ensuring that `SplitterFunction` outputs the initial condition `u0` as type `MVector` or `MArray`, if it has less than ~100 components.

`ObservationFunction` should be of the form `F(u,t) -> Vector`. Note that the time `t` must be passed as the second argument.
"""
function GetModel(func::ODEFunction{T}, SplitterFunction::Function, ObservationFunction::Function; tol::Real=1e-7,
                    meth::OrdinaryDiffEqAlgorithm=GetMethod(tol), Domain::Union{HyperCube,Bool}=false, inplace::Bool=true) where T
    @assert T == inplace

    function GetSol(θ::AbstractVector{<:Number}, SplitterFunction::Function; tol::Real=tol, max_t::Number=10., meth::OrdinaryDiffEqAlgorithm=GetMethod(tol), kwargs...)
        u0, p = SplitterFunction(θ);        odeprob = ODEProblem(func, u0, (0., max_t), p)
        solve(odeprob, meth; reltol=tol, abstol=tol, kwargs...)
    end
    function Model(t::Number, θ::AbstractVector{<:Number}; ObservationFunction::Function=ObservationFunction, SplitterFunction::Function=SplitterFunction,
                                                    tol::Real=tol, max_t::Number=t, meth::OrdinaryDiffEqAlgorithm=GetMethod(tol), FullSol::Bool=false, kwargs...)
        sol = GetSol(θ, SplitterFunction; tol=tol, max_t=t, meth=meth, save_everystep=false, save_start=false, save_end=true, kwargs...)
        FullSol && return sol
        ObservationFunction(sol.u[end], sol.t[end])
    end
    function Model(ts::AbstractVector{<:Number}, θ::AbstractVector{<:Number}; ObservationFunction::Function=ObservationFunction, SplitterFunction::Function=SplitterFunction,
                                            tol::Real=tol, max_t::Number=maximum(ts), meth::OrdinaryDiffEqAlgorithm=GetMethod(tol), FullSol::Bool=false, kwargs...)
        sol = GetSol(θ, SplitterFunction; tol=tol, max_t=max_t, meth=meth, saveat=ts, kwargs...)
        FullSol && return sol
        length(sol.u) != length(ts) && throw("ODE integration failed, try using a lower tolerance value.")
        [ObservationFunction(sol.u[i], sol.t[i]) for i in 1:length(ts)] |> Reduction
    end
    MakeCustom(Model, Domain)
end



Getxyp(DM::AbstractDataModel) = Getxyp(Data(DM), Predictor(DM))
Getxyp(DS::AbstractDataSet, model::Function) = (xdim(DS),ydim(DS),pdim(DS,model))
Getxyp(DS::AbstractDataSet, M::ModelMap) = M.xyp

SymbolicArguments(args...) = SymbolicArguments(Getxyp(args...))
function SymbolicArguments(xyp::Tuple{Int,Int,Int})
    @variables x[1:xyp[1]] y[1:xyp[2]] θ[1:xyp[3]]
    X = xyp[1] == 1 ? x[1] : x;         Y = xyp[2] == 1 ? y[1] : y
    X, Y, θ
end

ToExpr(DS::AbstractDataSet, model::Function; timeout::Real=5) = ToExpr(model, (xdim(DS),ydim(DS),pdim(DS,model)); timeout=timeout)
ToExpr(DS::AbstractDataSet, M::ModelMap; timeout::Real=5) = ToExpr(M.Map, M.xyp; timeout=timeout)
ToExpr(M::ModelMap; timeout::Real=5) = ToExpr(M.Map, M.xyp; timeout=timeout)

function ToExpr(model::Function, xyp::Tuple{Int,Int,Int}; timeout::Real=5)
    X, Y, θ = SymbolicArguments(xyp)

    # Add option for models which are already inplace
    function TryOptim(model,X,θ)
        try
            model(X,θ)
        catch;
            @warn "Unable to convert given function to symbolic expression."
        end
    end
    modelexpr = nothing
    task = @async(TryOptim(model,X,θ))
    if timedwait(()->istaskdone(task), timeout) == :timed_out
        @async(Base.throwto(task, DivideError())) # kill task
    else
        modelexpr = fetch(task)
    end;    modelexpr
end

PrintModel(DM::AbstractDataModel) = "y(x;θ) = $(ToExpr(Data(DM), Predictor(DM)))"
# PrintdModel(DM::AbstractDataModel) = "(dy)(x;θ) = $(ToExpr(Data(DM), dPredictor(DM)))"


function Optimize(DM::AbstractDataModel; inplace::Bool=false, timeout::Real=5, parallel::Bool=false)
    Optimize(Data(DM), Predictor(DM); inplace=inplace, timeout=timeout, parallel=parallel)
end
function Optimize(DS::AbstractDataSet, model::ModelOrFunction; inplace::Bool=false, timeout::Real=5, parallel::Bool=false)
    Optimize(model, Getxyp(DS, model); inplace=inplace, timeout=timeout, parallel=parallel)
end
function Optimize(M::ModelMap, xyp::Tuple{Int,Int,Int}; inplace::Bool=false, timeout::Real=5, parallel::Bool=false)
    xyp != M.xyp && throw("xyp inconsistent.")
    model, dmodel = Optimize(M.Map, xyp; inplace=inplace, timeout=timeout, parallel=parallel)
    ModelMap(model, M), ModelMap(dmodel, M)
end
function Optimize(model::Function, xyp::Tuple{Int,Int,Int}; inplace::Bool=false, timeout::Real=5, parallel::Bool=false)
    modelexpr = ToExpr(model, xyp; timeout=timeout)
    modelexpr == nothing && return nothing, nothing

    X, Y, θ = SymbolicArguments(xyp)

    # Need to make sure that modelexpr is of type Vector{Num}, not just Num
    modelexpr = xyp[2] == 1 ? [ModelingToolkit.simplify(modelexpr)] : ModelingToolkit.simplify(modelexpr)
    derivative = ModelingToolkit.jacobian(modelexpr, θ; simplify=true)

    ExprToModelMap(X, θ, modelexpr; inplace=inplace, parallel=parallel, IsJacobian=false), ExprToModelMap(X, θ, derivative; inplace=inplace, parallel=parallel, IsJacobian=true)
end

function ExprToModelMap(X::Union{Num,AbstractVector{<:Num}}, P::AbstractVector{Num}, modelexpr::Union{Num,AbstractArray{<:Num}};
                                                        inplace::Bool=false, parallel::Bool=false, IsJacobian::Bool=false)
    parallelization = parallel ? ModelingToolkit.MultithreadedForm() : ModelingToolkit.SerialForm()
    OptimizedModel = try
        build_function(modelexpr, X, P; expression=Val{false}, parallel=parallelization)[inplace ? 2 : 1]
    catch;
        build_function(modelexpr, X, P; expression=Val{false}, parallel=parallelization)
    end
    ### Pretty Function names
    if IsJacobian
        # THROWING AWAY KWARGS HERE!
        SymbolicModelJacobian(x::Union{Number,AbstractVector{<:Number}}, θ::AbstractVector{<:Number}; kwargs...) = OptimizedModel(x, θ)
        function SymbolicModelJacobian!(y::Union{Number,AbstractMatrix{<:Number}}, x::Union{Number,AbstractVector{<:Number}}, θ::AbstractVector{<:Number}; kwargs...)
            OptimizedModel(y, x, θ)
        end
        return inplace ? SymbolicModelJacobian! : SymbolicModelJacobian
    else
        # THROWING AWAY KWARGS HERE!
        SymbolicModel(x::Union{Number,AbstractVector{<:Number}}, θ::AbstractVector{<:Number}; kwargs...) = OptimizedModel(x, θ)
        SymbolicModel!(y::Union{Number,AbstractVector{<:Number}}, x::Union{Number,AbstractVector{<:Number}}, θ::AbstractVector{<:Number}; kwargs...) = OptimizedModel(y, x, θ)
        return inplace ? SymbolicModel! : SymbolicModel
    end
end


function OptimizedDM(DM::AbstractDataModel)
    model, dmodel = Optimize(DM)
    # Very simple models (ydim=1) typically slower after simplification using ModelingToolkit.jl
    if dmodel != nothing
        return DataModel(Data(DM), Predictor(DM), dmodel, MLE(DM), LogLikeMLE(DM))
    else
        # Get warning from Optimize() that symbolic optimization was unsuccessful
        return DM
    end
end
