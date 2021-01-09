


function InformNames(DS::AbstractDataSet, sys::ODESystem, observables::Vector{Int})
    newxnames = xnames(DS) == CreateSymbolNames(xdim(DS),"x") ? [string(sys.iv.name)] : xnames(DS)
    newynames = ynames(DS) == CreateSymbolNames(ydim(DS),"y") ? string.(sys.states[observables]) : ynames(DS)
    InformNames(DS, newxnames, newynames)
end


function DataModel(DS::AbstractDataSet, sys::ODESystem, u0::AbstractVector{<:Number}, observables::Vector{Int}, args...; tol::Real=1e-6, kwargs...)
    DataModel(InformNames(DS, sys, observables), GetModel(sys, u0, observables; tol=tol), args...; kwargs...)
end


# Allow option of passing Domain for parameters as keyword
function GetModel(sys::ODESystem, u0::AbstractVector{<:Number}, observables::Vector{Int}; tol::Real=1e-6, Domain::Union{HyperCube,Bool}=false, inplace::Bool=true)
    # Is there some optimization that can be applied here? Modollingtoolkitize() or something?
    func = ODEFunction{inplace}(sys)
    u0 = inplace ? MVector{length(u0)}(u0) : SVector{length(u0)}(u0)

    # Do not need ts
    function GetSol(ts::AbstractVector{<:Real}, θ::AbstractVector{<:Number}; observables::Vector{Int}=observables, tol::Real=tol, max_t::Real=maximum(ts)+1e-5,
                                                                            meth::OrdinaryDiffEqAlgorithm=Tsit5(), kwargs...)
        odeprob = ODEProblem(func, u0, (0., max_t), θ)
        solve(odeprob, meth; reltol=tol, abstol=tol, kwargs...)
    end

    function Model(t::Real, θ::AbstractVector{<:Number}; observables::Vector{Int}=observables, tol::Real=tol, max_t::Real=t,
                                                                            meth::OrdinaryDiffEqAlgorithm=Tsit5(), FullSol::Bool=false, kwargs...)
        sol = GetSol([t], θ; observables=observables, tol=tol, max_t=t, meth=meth, save_everystep=false,save_start=false,save_end=true, kwargs...)
        FullSol && return sol
        sol.u[end][observables]
    end
    function Model(ts::AbstractVector{<:Real}, θ::AbstractVector{<:Number}; observables::Vector{Int}=observables, tol::Real=tol, max_t::Real=maximum(ts)+1e-5,
                                                                            meth::OrdinaryDiffEqAlgorithm=Tsit5(), FullSol::Bool=false, kwargs...)
        sol = GetSol(ts, θ; observables=observables, tol=tol, max_t=max_t, meth=meth, kwargs...)
        FullSol && return sol
        Reduction(map(t->sol(t)[observables], ts))
    end
    function FastModel(t::Real, θ::AbstractVector{<:Number}; observables::Vector{Int}=observables, tol::Real=tol, max_t::Real=t,
                                                                            meth::OrdinaryDiffEqAlgorithm=Tsit5(), FullSol::Bool=false, kwargs...)
        Model(t, θ; observables=observables, tol=tol, max_t=max_t, meth=meth,FullSol=FullSol, kwargs)
    end
    function FastModel(ts::AbstractVector{<:Real}, θ::AbstractVector{<:Number}; observables::Vector{Int}=observables, tol::Real=tol, max_t::Real=maximum(ts)+1e-5,
                                                                            meth::OrdinaryDiffEqAlgorithm=Tsit5(), FullSol::Bool=false, kwargs...)
        sol = GetSol(ts, θ; observables=observables, tol=tol, max_t=max_t, meth=meth, tstops=ts, save_start=false, save_end=false, save_everywhere=false, kwargs...)
        FullSol && return sol
        [sol.u[findnext(x->x==t,sol.t,i)][observables] for (i,t) in enumerate(ts)] |> Reduction
    end

    pnames = [string(x.name) for x in sys.ps]
    xyp = (1, length(observables), length(sys.ps))
    Domain = isa(Domain, Bool) ? FullDomain(xyp[3]) : Domain

    # new(Map, InDomain, Domain, xyp, ParamNames, StaticOutput, inplace, CustomEmbedding)
    ModelMap(FastModel, θ->true, Domain, xyp, pnames, Val(false), Val(false), Val(true))
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
    modelexpr = xyp[2] == 1 ? [simplify(modelexpr)] : simplify(modelexpr)
    derivative = ModelingToolkit.jacobian(modelexpr, θ; simplify=true)

    ExprToModelMap(X, θ, modelexpr; inplace=inplace, parallel=parallel, IsJacobian=false), ExprToModelMap(X, θ, derivative; inplace=inplace, parallel=parallel, IsJacobian=true)
end

function ExprToModelMap(X::Union{Num,AbstractVector{<:Num}}, P::AbstractVector{Num}, modelexpr::Union{Num,AbstractArray{<:Num}}; inplace::Bool=false, parallel::Bool=false, IsJacobian::Bool=false)
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
