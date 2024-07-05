

function CompleteObservationFunction(PreObservationFunction::Function)
    numargs = MaximalNumberOfArguments(PreObservationFunction)
    if numargs == 1
        return (u::Union{Number,AbstractArray{<:Number}}, t::Real, θ::AbstractVector{<:Number}) -> PreObservationFunction(u)
    elseif numargs == 2
        return (u::Union{Number,AbstractArray{<:Number}}, t::Real, θ::AbstractVector{<:Number}) -> PreObservationFunction(u, t)
    elseif numargs == 3
        return (u::Union{Number,AbstractArray{<:Number}}, t::Real, θ::AbstractVector{<:Number}) -> PreObservationFunction(u, t, θ)
    else
        throw("Given ObservationFunction should accept either arguments (u) or (u,t) or (u,t,θ). Got function which accepts $numargs arguments.")
    end
end


"""
    InformNames(DS::AbstractDataSet, sys::ODESystem, observables::AbstractVector{<:Int})
Copy the state names saved in `ODESystem` to `DS`.
"""
function InformNames(DS::AbstractDataSet, sys::ModelingToolkit.AbstractSystem, observables::Union{Int,AbstractVector{<:Int},BoolArray})
    newxnames = xnames(DS) == CreateSymbolNames(xdim(DS),"x") ? [string(ModelingToolkit.get_iv(sys))] : xnames(DS)
    newynames = ynames(DS) == CreateSymbolNames(ydim(DS),"y") ? string.(ModelingToolkit.get_states(sys)[observables]) : ynames(DS)
    InformNames(DS, newxnames, newynames)
end


# No ObservationFunction, therefore try to use sys to infer state names of ODEsys
# Extend for other DEFunctions in the future
function DataModel(DS::AbstractDataSet, sys::Union{ModelingToolkit.AbstractSystem,SciMLBase.AbstractDiffEqFunction}, u0::Union{Number,AbstractArray{<:Number},Function},
                        observables::Union{Int,AbstractVector{<:Int},BoolArray,Function}=1:length(u0), args...; tol::Real=1e-7, Domain::Union{HyperCube,Nothing}=nothing, kwargs...)
    newDS = (observables isa Union{Int,AbstractVector{<:Int}} && sys isa ModelingToolkit.AbstractSystem) ? InformNames(DS, sys, observables) : DS
    DataModel(newDS, GetModel(sys, u0, observables; tol=tol, Domain=Domain, kwargs...), args...)
end


# """
# Given `func` is converted to `ODEFunction`. Will probably deprecate this in the future.
# """
# function GetModel(func::Function, u0::Union{AbstractArray{<:Number},Function}, observables::Union{Function,AbstractVector{<:Int},BoolArray}=1:length(u0); tol::Real=1e-7,
#                     meth::AbstractODEAlgorithm=GetMethod(tol), Domain::Union{HyperCube,Nothing}=nothing, inplace::Bool=true)
#     GetModel(ODEFunction{inplace}(func), u0, observables; tol=tol, Domain=Domain, meth=meth, inplace=inplace)
# end




function GetModel(sys::ModelingToolkit.AbstractSystem, u0::Union{Number,AbstractArray{<:Number},Function}, observables::Union{Int,AbstractVector{<:Int},BoolArray,Function}=1:length(u0);
                Domain::Union{HyperCube,Nothing}=nothing, inplace::Bool=true, pnames::AbstractVector{<:String}=string.(ModelingToolkit.get_ps(sys)), InDomain::Union{Function,Nothing}=nothing, name::Union{String,Symbol}=ModelingToolkit.getname(sys), kwargs...)
    # Is there some optimization that can be applied here? Modellingtoolkitize(sys) or something?
    # sys = Sys isa Catalyst.ReactionSystem ? convert(ODESystem, Sys) : Sys
    
    Model = if sys isa ModelingToolkit.AbstractODESystem
        odefunc = ODEFunction{inplace}(structural_simplify(sys); jac = true)
        GetModel(odefunc, u0, observables; Domain=Domain, inplace=inplace, kwargs...)
    else
        throw("Not programmed for $(typeof(sys)) yet, please convert to a ModelingToolkit.AbstractODESystem first.")
    end
    !isnothing(Domain) && (@assert length(pnames) ≤ length(Domain))
    ylen = if observables isa Function      # ObservationFunction
        # Might still fail if states u are a Matrix.
        argnum = MaximalNumberOfArguments(observables)
        F = if argnum==1  z->observables(z)   elseif argnum==2  z->observables(z,0.1)
            elseif argnum==3 z->observables(z,0.1,GetStartP(length(pnames)))    else throw("Error") end
        num = GetArgLength(F)
        length(F(ones(num)))
    else
        observables isa BoolArray ? sum(observables) : length(observables)
    end
    plen = if Domain isa HyperCube
        length(Domain)
    elseif u0 isa Union{Number,AbstractArray}     # Vector / Matrix
        # initial conditions given as array means the parameters are only the ps in sys
        length(pnames)
    else        # SplitterFunction
        # May well fail depending on how splitter function is implemented
        GetArgLength(u0)
    end
    xyp = (1, ylen, plen)
    Domain = isnothing(Domain) ? FullDomain(xyp[3], 1e5) : Domain

    pnames = length(pnames) == length(Domain) ? pnames : CreateSymbolNames(plen, "θ")
    # new(Map, InDomain, Domain, xyp, pnames, inplace, CustomEmbedding)
    ModelMap(Model.Map, InDomain, Domain, xyp, pnames, Val(false), Val(true), name, (Model.Meta isa Tuple ? (sys, (Model.Meta[2:end])...) : Model.Meta))
end




# Although specialized methods for constant initial condition and specification of observed components in terms of arrays is faster than using functions, also use general method here.
function GetModel(func::SciMLBase.AbstractDiffEqFunction{T}, Initial, Observables; kwargs...) where T
    SplitterFunction = if Initial isa Function
        Initial
    else
        # u0 = PromoteStatic(Initial, T)
        # θ -> (u0, θ)
        θ -> (Initial, θ)
    end
    GetModel(func, SplitterFunction, (Observables isa Function ? Observables : (u -> view(u,Observables))); kwargs...)
end

function GetModel(func::SciMLBase.AbstractDiffEqFunction, SplitterFunction::Function, ObservationFunction::Function; kwargs...)
    throw("If you see this error, it is most likely because no specialized method has been implemented for $(typeof(func)) yet.")
end



# Retain argument-type specific docstrings but allow for robust

"""
    GetModel(func::ODEFunction, u0::AbstractArray, ObservationFunction::Function; tol::Real=1e-7, meth::AbstractODEAlgorithm=Tsit5(), Domain::Union{HyperCube,Nothing}=nothing, inplace::Bool=true)
Returns a `ModelMap` which evolves the given system of ODEs from the initial configuration `u0` and afterwards applies the `ObservationFunction` to produce its predictions.

`ObservationFunction` should either be of the form `F(u) -> Vector` or `F(u,t) -> Vector` or `F(u,t,θ) -> Vector`.
Internally, the `ObservationFunction` is automatically wrapped as `F(u,t,θ)` if it is not already defined to accept three arguments.

A `Domain` can be supplied to constrain the parameters of the model to particular ranges which can be helpful in the fitting process.
"""
GetModel(func::AbstractODEFunction{T}, u0::Union{Number,AbstractArray{<:Number}}, ObservationFunc::Function; kwargs...) where T = GetModelFastOrRobust(func,u0,ObservationFunc; kwargs...)

"""
    GetModel(func::ODEFunction, SplitterFunction::Function, observables::Union{AbstractVector{<:Int},BoolArray}; tol::Real=1e-7, meth::AbstractODEAlgorithm=Tsit5(), Domain::Union{HyperCube,Nothing}=nothing, inplace::Bool=true)
Returns a `ModelMap` which evolves the given system of ODEs and returns `u[observables]` to produce its predictions.
Here, the initial conditions for the ODEs are produced from the parameters `θ` using the `SplitterFunction` which for instance allows one to estimate them from data.

`SplitterFunction` should be of the form `F(θ) -> (u0, p)`, i.e. the output is a `Tuple` whose first entry is the initial condition for the ODE model and the second entry constitutes the parameters which go on to enter the `ODEFunction`.
Typically, a fair bit of performance can be gained from ensuring that `SplitterFunction` outputs the initial condition `u0` as type `MVector` or `MArray`, if it has less than ~100 components.

A `Domain` can be supplied to constrain the parameters of the model to particular ranges which can be helpful in the fitting process.
"""
GetModel(func::AbstractODEFunction{T}, Splitter::Function, Observables::Union{Int,AbstractVector{<:Int},BoolArray}=1; kwargs...) where T = GetModelFastOrRobust(func, Splitter, Observables; kwargs...)

"""
    GetModel(func::AbstractODEFunction{T}, SplitterFunction::Function, PreObservationFunction::Function; tol::Real=1e-7, meth::AbstractODEAlgorithm=Tsit5(), Domain::Union{HyperCube,Nothing}=nothing, inplace::Bool=true)
Returns a `ModelMap` which evolves the given system of ODEs and afterwards applies the `ObservationFunction` to produce its predictions.
Here, the initial conditions for the ODEs are produced from the parameters `θ` using the `SplitterFunction` which for instance allows one to estimate them from data.

`SplitterFunction` should be of the form `F(θ) -> (u0, p)`, i.e. the output is a `Tuple` whose first entry is the initial condition for the ODE model and the second entry constitutes the parameters which go on to enter the `ODEFunction`.
Typically, a fair bit of additional performance can be gained from ensuring that `SplitterFunction` outputs the initial condition `u0` as type `MVector` or `MArray`, if it has less than ~100 components.

`ObservationFunction` should either be of the form `F(u) -> Vector` or `F(u,t) -> Vector` or `F(u,t,θ) -> Vector`.
Internally, the `ObservationFunction` is automatically wrapped as `F(u,t,θ)` if it is not already defined to accept three arguments.

!!! note
    The vector `θ` passed to `ObservationFunction` is the same `θ` that is passed to `SplitterFunction`, i.e. before splitting.
    This is because `ObservationFunction` might also depend on the initial conditions in general.

A `Domain` can be supplied to constrain the parameters of the model to particular ranges which can be helpful in the fitting process.
"""
GetModel(func::AbstractODEFunction{T}, Splitter::Function, ObservationFunc::Function; kwargs...) where T = GetModelFastOrRobust(func, Splitter, ObservationFunc; kwargs...)



# Promote to Dual only if time is dual and second argument is not to avoid changing tag
ConditionalConvert(type::Type{ForwardDiff.Dual{T}}, var::Union{Number,AbstractVector{<:Number}}) where T = convert.(type, var)
ConditionalConvert(type::Type{ForwardDiff.Dual{T}}, var::Union{ForwardDiff.Dual{S},AbstractVector{<:ForwardDiff.Dual{S}}}) where {T,S} = convert.(type, var)
ConditionalConvert(type::Type, var::Union{Number,AbstractVector{<:Number}}) = var

# Vanilla version with constant array of initial conditions and vector of observables.
function GetModelFast(func::AbstractODEFunction{T}, u0::Union{Number,AbstractArray{<:Number}}, Observables::Union{Int,AbstractVector{<:Int},BoolArray}=1:length(u0); tol::Real=1e-7,
                    meth::AbstractODEAlgorithm=GetMethod(tol), Domain::Union{HyperCube,Nothing}=nothing, inplace::Bool=true, callback=nothing, Kwargs...) where T
    @warn "This method for solving ODEs will throw errors when applying time-derivatives or trying to evaluate at t < 0! Alternatively use keyword robust=true."
    @assert T == inplace
    CB = callback
    # u0 = PromoteStatic(u0, inplace)
    # If observable only has single component, don't pass vector to getindex() in second arg
    observables = length(Observables) == 1 ? Observables[1] : Observables

    function GetSol(θ::AbstractVector{<:Number}, u0::Union{Number,AbstractArray{<:Number}}; tol::Real=tol, max_t::Ttype=10., meth::AbstractODEAlgorithm=meth, callback=nothing, kwargs...) where Ttype <: Number
        odeprob = ODEProblem(func, ConditionalConvert(Ttype,u0), (zero(max_t), max_t), ConditionalConvert(Ttype,θ))
        solve(odeprob, meth; reltol=tol, abstol=tol, callback=CallbackSet(callback,CB), Kwargs..., kwargs...)
    end
    function ODEmodel(t::Number, θ::AbstractArray{<:Number}; observables::Union{Int,AbstractVector{<:Int},BoolArray}=observables, u0::Union{Number,AbstractArray{<:Number}}=u0,
                                                        tol::Real=tol, max_t::Number=t, meth::AbstractODEAlgorithm=meth, FullSol::Bool=false, kwargs...)
        FullSol && return GetSol(θ, u0; tol=tol, max_t=max_t, meth=meth, kwargs...)
        sol = GetSol(θ, u0; tol=tol, max_t=max_t, meth=meth, save_everystep=false, save_start=false, save_end=true, kwargs...)
        sol.u[end][observables]
    end
    function ODEmodel(ts::AbstractVector{<:Number}, θ::AbstractVector{<:Number}; observables::Union{Int,AbstractVector{<:Int},BoolArray}=observables, u0::Union{Number,AbstractArray{<:Number}}=u0,
                                                tol::Real=tol, max_t::Number=maximum(ts), meth::AbstractODEAlgorithm=meth, FullSol::Bool=false, kwargs...)
        FullSol && return GetSol(θ, u0; tol=tol, max_t=max_t, meth=meth, tstops=ts, kwargs...)
        sol = GetSol(θ, u0; tol=tol, max_t=max_t, meth=meth, saveat=ts, kwargs...)
        length(sol.u) != length(ts) && throw("ODE integration failed, maybe try using a lower tolerance value. θ=$θ.")
        [sol.u[i][observables] for i in 1:length(ts)] |> Reduction
    end
    MakeCustom(ODEmodel, Domain; Meta=(func, u0, observables, callback), verbose=false)
end

function GetModelFast(func::AbstractODEFunction{T}, u0::Union{Number,AbstractArray{<:Number}}, PreObservationFunction::Function; tol::Real=1e-7,
                    meth::AbstractODEAlgorithm=GetMethod(tol), Domain::Union{HyperCube,Nothing}=nothing, inplace::Bool=true, callback=nothing, Kwargs...) where T
    @warn "This method for solving ODEs will throw errors when applying time-derivatives or trying to evaluate at t < 0! Alternatively use keyword robust=true."
    @assert T == inplace
    CB = callback
    # u0 = PromoteStatic(u0, inplace)
    ObservationFunction = CompleteObservationFunction(PreObservationFunction)

    function GetSol(θ::AbstractVector{<:Number}, u0::Union{Number,AbstractArray{<:Number}}; tol::Real=tol, max_t::Ttype=10., meth::AbstractODEAlgorithm=meth, callback=nothing, kwargs...) where Ttype <: Number
        odeprob = ODEProblem(func, ConditionalConvert(Ttype,u0), (zero(max_t), max_t), ConditionalConvert(Ttype,θ))
        solve(odeprob, meth; reltol=tol, abstol=tol, callback=CallbackSet(callback,CB), Kwargs..., kwargs...)
    end
    function ODEmodel(t::Number, θ::AbstractVector{<:Number}; ObservationFunction::Function=ObservationFunction, u0::Union{Number,AbstractArray{<:Number}}=u0,
                                                tol::Real=tol, max_t::Number=t, meth::AbstractODEAlgorithm=meth, FullSol::Bool=false, kwargs...)
        FullSol && return GetSol(θ, u0; tol=tol, max_t=max_t, meth=meth, kwargs...)
        sol = GetSol(θ, u0; tol=tol, max_t=max_t, meth=meth, save_everystep=false, save_start=false, save_end=true, kwargs...)
        ObservationFunction(sol.u[end], t, θ)
    end
    function ODEmodel(ts::AbstractVector{<:Number}, θ::AbstractVector{<:Number}; ObservationFunction::Function=ObservationFunction, u0::Union{Number,AbstractArray{<:Number}}=u0,
                                            tol::Real=tol, max_t::Number=maximum(ts), meth::AbstractODEAlgorithm=meth, FullSol::Bool=false, kwargs...)
        FullSol && return GetSol(θ, u0; tol=tol, max_t=max_t, meth=meth, kwargs...)
        sol = GetSol(θ, u0; tol=tol, max_t=max_t, meth=meth, saveat=ts, kwargs...)
        length(sol.u) != length(ts) && throw("ODE integration failed, maybe try using a lower tolerance value. θ=$θ.")
        [ObservationFunction(sol.u[i], sol.t[i], θ) for i in 1:length(ts)] |> Reduction
    end
    MakeCustom(ODEmodel, Domain; Meta=(func, u0, ObservationFunction, callback), verbose=false)
end

function GetModelFast(func::AbstractODEFunction{T}, SplitterFunction::Function, Observables::Union{Int,AbstractVector{<:Int},BoolArray}=1; tol::Real=1e-7,
                    meth::AbstractODEAlgorithm=GetMethod(tol), Domain::Union{HyperCube,Nothing}=nothing, inplace::Bool=true, callback=nothing, Kwargs...) where T
    @warn "This method for solving ODEs will throw errors when applying time-derivatives or trying to evaluate at t < 0! Alternatively use keyword robust=true."
    @assert T == inplace
    CB = callback
    # If observable only has single component, don't pass vector to getindex() in second arg
    observables = length(Observables) == 1 ? Observables[1] : Observables

    function GetSol(θ::AbstractVector{<:Number}, SplitterFunction::Function; tol::Real=tol, max_t::Ttype=10., meth::AbstractODEAlgorithm=meth, callback=nothing, kwargs...) where Ttype <: Number
        u0, p = SplitterFunction(θ);        odeprob = ODEProblem(func, ConditionalConvert(Ttype,u0), (zero(max_t), max_t), ConditionalConvert(Ttype,p))
        solve(odeprob, meth; reltol=tol, abstol=tol, callback=CallbackSet(callback,CB), Kwargs..., kwargs...)
    end
    function ODEmodel(t::Number, θ::AbstractVector{<:Number}; observables::Union{Int,AbstractVector{<:Int},BoolArray}=observables, SplitterFunction::Function=SplitterFunction,
                                tol::Real=tol, max_t::Number=t, meth::AbstractODEAlgorithm=meth, FullSol::Bool=false, kwargs...)
        FullSol && return GetSol(θ, SplitterFunction; tol=tol, max_t=max_t, meth=meth, kwargs...)
        sol = GetSol(θ, SplitterFunction; tol=tol, max_t=max_t, meth=meth, save_everystep=false, save_start=false, save_end=true, kwargs...)
        sol.u[end][observables]
    end
    function ODEmodel(ts::AbstractVector{<:Number}, θ::AbstractVector{<:Number}; observables::Union{Int,AbstractVector{<:Int},BoolArray}=observables, SplitterFunction::Function=SplitterFunction,
                                tol::Real=tol, max_t::Number=maximum(ts), meth::AbstractODEAlgorithm=meth, FullSol::Bool=false, kwargs...)
        FullSol && return GetSol(θ, SplitterFunction; tol=tol, max_t=max_t, meth=meth, tstops=ts, kwargs...)
        sol = GetSol(θ, SplitterFunction; tol=tol, max_t=max_t, meth=meth, saveat=ts, kwargs...)
        length(sol.u) != length(ts) && throw("ODE integration failed, maybe try using a lower tolerance value. θ=$θ.")
        [sol.u[i][observables] for i in 1:length(ts)] |> Reduction
    end
    MakeCustom(ODEmodel, Domain; Meta=(func, SplitterFunction, observables, callback), verbose=false)
end

function GetModelFast(func::AbstractODEFunction{T}, SplitterFunction::Function, PreObservationFunction::Function; tol::Real=1e-7,
                    meth::AbstractODEAlgorithm=GetMethod(tol), Domain::Union{HyperCube,Nothing}=nothing, inplace::Bool=true, callback=nothing, Kwargs...) where T
    @warn "This method for solving ODEs will throw errors when applying time-derivatives or trying to evaluate at t < 0! Alternatively use keyword robust=true."
    @assert T == inplace
    CB = callback
    ObservationFunction = CompleteObservationFunction(PreObservationFunction)

    function GetSol(θ::AbstractVector{<:Number}, SplitterFunction::Function; tol::Real=tol, max_t::Ttype=10., meth::AbstractODEAlgorithm=meth, callback=nothing, kwargs...) where Ttype <: Number
        u0, p = SplitterFunction(θ);        odeprob = ODEProblem(func, ConditionalConvert(Ttype,u0), (zero(max_t), max_t), ConditionalConvert(Ttype,p))
        solve(odeprob, meth; reltol=tol, abstol=tol, callback=CallbackSet(callback, CB), Kwargs..., kwargs...)
    end
    function ODEmodel(t::Number, θ::AbstractVector{<:Number}; ObservationFunction::Function=ObservationFunction, SplitterFunction::Function=SplitterFunction,
                                                    tol::Real=tol, max_t::Number=t, meth::AbstractODEAlgorithm=meth, FullSol::Bool=false, kwargs...)
        FullSol && return GetSol(θ, SplitterFunction; tol=tol, max_t=max_t, meth=meth, kwargs...)
        sol = GetSol(θ, SplitterFunction; tol=tol, max_t=max_t, meth=meth, save_everystep=false, save_start=false, save_end=true, kwargs...)
        ObservationFunction(sol.u[end], sol.t[end], θ)
    end
    function ODEmodel(ts::AbstractVector{<:Number}, θ::AbstractVector{<:Number}; ObservationFunction::Function=ObservationFunction, SplitterFunction::Function=SplitterFunction,
                                            tol::Real=tol, max_t::Number=maximum(ts), meth::AbstractODEAlgorithm=meth, FullSol::Bool=false, kwargs...)
        FullSol && return GetSol(θ, SplitterFunction; tol=tol, max_t=max_t, meth=meth, tstops=ts, kwargs...)
        sol = GetSol(θ, SplitterFunction; tol=tol, max_t=max_t, meth=meth, saveat=ts, kwargs...)
        length(sol.u) != length(ts) && throw("ODE integration failed, maybe try using a lower tolerance value. θ=$θ.")
        [ObservationFunction(sol.u[i], sol.t[i], θ) for i in 1:length(ts)] |> Reduction
    end
    MakeCustom(ODEmodel, Domain; Meta=(func, SplitterFunction, ObservationFunction, callback), verbose=false)
end

"""
Switch between possibly slightly faster method which does not allow for time autodifferentiation vs 'robust' method which additionally includes backward integration.
"""
function GetModelFastOrRobust(func::AbstractODEFunction{T}, Splitter, ObservationFunc; robust::Bool=true, kwargs...) where T
    if robust
        GetModelRobust(func, Splitter, ObservationFunc; kwargs...)
    else
        GetModelFast(func, Splitter, ObservationFunc; kwargs...)
    end
end

GetModelRobust(func::ODESystem, A, B; kwargs...) = GetModelRobust(ODEFunction(func), A, B; kwargs...)
function GetModelRobust(func::AbstractODEFunction, u0, Observables; kwargs...)
    SplitterFunction = if u0 isa Function
        u0
    elseif u0 isa Union{Number, AbstractArray}
        (θ->(u0, θ))
    else
        throw("SplitterFunction must be either Function, AbstractArray or Number.")
    end
    ObservationFunction = if Observables isa Function
        Observables
    elseif Observables isa Union{Int,AbstractVector{<:Int},BoolArray}
        observables = length(Observables) == 1 ? Observables[1] : Observables
        u->view(u,observables)
    else
        throw("ObservationFunction must be either Function, AbstractArray or Number.")
    end
    GetModelRobust(func, SplitterFunction, ObservationFunction; kwargs...)
end
function GetModelRobust(func::AbstractODEFunction{T}, SplitterFunction::Function, PreObservationFunction::Function; tol::Real=1e-7,
                    meth::AbstractODEAlgorithm=GetMethod(tol), Domain::Union{HyperCube,Nothing}=nothing, inplace::Bool=true, callback=nothing, Kwargs...) where T
    @assert T == inplace
    CB = callback
    ObservationFunction = CompleteObservationFunction(PreObservationFunction)

    function _ODEmodel(ts::AbstractVector{<:Number}, θ::AbstractVector{<:Number}; ObservationFunction::Function=ObservationFunction, SplitterFunction::Function=SplitterFunction,
                                            tol::Real=tol, max_t::Ttype=maximum(ts), meth::AbstractODEAlgorithm=meth, callback=nothing, FullSol::Bool=false, kwargs...) where Ttype <: Number
        u0, p = SplitterFunction(θ);        odeprob = ODEProblem(func, ConditionalConvert(Ttype,u0), (zero(max_t), max_t), ConditionalConvert(Ttype,p))

        sol = solve(odeprob, meth; reltol=tol, abstol=tol, saveat=ts, callback=CallbackSet(callback, CB), Kwargs..., kwargs...)
        FullSol ? sol : Reduction([ObservationFunction(sol(t), t, θ) for t in ts])
    end
    # ts sorted ascending, smallest element is first
    function _ODEmodelbacksorted(ts::AbstractVector{<:Number}, θ::AbstractVector{<:Number}; max_t::Number=maximum(ts), FullSol::Bool=false, kwargs...)
        @assert !FullSol "Cannot provide FullSol for backwards integration."
        lastind = findfirst(x->x≥0.0, ts)
        if isnothing(lastind)
            _ODEmodel(ts, θ; max_t=ts[1], kwargs...)
        else
            [_ODEmodel(view(ts,1:lastind-1), θ; max_t=ts[1], kwargs...); _ODEmodel(view(ts,lastind:length(ts)), θ; max_t=max_t, kwargs...)]
        end
    end
    function _ODEmodelback(ts::AbstractVector{<:Number}, θ::AbstractVector{<:Number}; ObservationFunction::Function=ObservationFunction, SplitterFunction::Function=SplitterFunction,
                                            tol::Real=tol, max_t::Ttype=maximum(ts), meth::AbstractODEAlgorithm=meth, callback=nothing, FullSol::Bool=false, kwargs...) where Ttype <: Number
        @assert !FullSol "Cannot provide FullSol for backwards integration."
        u0, p = SplitterFunction(θ)
        negTs = map(x->x<0.0, ts);  min_t = minimum(ts)

        odeprob1 = ODEProblem(func, ConditionalConvert(Ttype,u0), (zero(min_t), min_t), ConditionalConvert(Ttype,p))
        odeprob2 = ODEProblem(func, ConditionalConvert(Ttype,u0), (zero(max_t), max_t), ConditionalConvert(Ttype,p))

        sol1 = solve(odeprob1, meth; reltol=tol, abstol=tol, saveat=ts[negTs], callback=CallbackSet(callback, CB), Kwargs..., kwargs...)
        sol2 = solve(odeprob2, meth; reltol=tol, abstol=tol, saveat=ts[.!negTs], callback=CallbackSet(callback, CB), Kwargs..., kwargs...)
        [ObservationFunction((t < 0.0 ? sol1(t) : sol2(t)) , t, θ) for t in ts] |> Reduction
    end

    ODEmodel(ts::AbstractVector{<:Number}, θ::AbstractVector{<:Number}; kwargs...) = all(x->x≥0.0, ts) ? _ODEmodel(ts, θ; kwargs...) : (issorted(ts) ? _ODEmodelbacksorted(ts, θ; kwargs...) : _ODEmodelback(ts, θ; kwargs...))
    ODEmodel(t::Number, θ::AbstractVector{<:Number}; kwargs...) = ODEmodel([t], θ; kwargs...)
    MakeCustom(ODEmodel, Domain; Meta=(func, SplitterFunction, ObservationFunction, callback), verbose=false)
end

@deprecate GetModelNaive(func, split, obs; kwargs...) GetModelRobust(func, split, obs; kwargs...) false

"""
    ModifyODEmodel(DM::AbstractDataModel, NewObservationFunc::Function) -> ModelMap
Constructs a new `ModelMap` with new observation function `f(u,t,θ)` from a given ODE-based `DataModel`.
"""
ModifyODEmodel(DM::AbstractDataModel, NewObservationFunc::Function) = ModifyODEmodel(DM, Predictor(DM), NewObservationFunc)
function ModifyODEmodel(DM::AbstractDataModel, Model::ModelMap, NewObservationFunc::Function)
    @assert !isnothing(Model.Meta) "It appears as though the given model is not an ODEmodel."
    @assert length(Model.Meta) == 4
    # Model.Map isa DEModel && return ModifyDEModel(Model, NewObservationFunc)
    Eval = try
        Model(WoundX(DM)[1], MLE(DM); FullSol=true).u[1]
    catch;
        throw("It appears as though the given model is not an ODEmodel.")
    end
    F = CompleteObservationFunction(NewObservationFunc)
    out = F(Eval, WoundX(DM)[1], MLE(DM))
    function NewODEmodel(x::AbstractVector{<:Number}, θ::AbstractVector{<:Number}; FullSol=false, kwargs...)
        FullSol && return Model.Map(x, θ; FullSol=true, kwargs...)
        sol = Model.Map(x, θ; FullSol=true, saveat=x, kwargs...)
        length(sol.u) != length(x) && throw("ODE integration failed, maybe try using a lower tolerance value. θ=$θ.")
        [F(sol.u[i], sol.t[i], θ) for i in 1:length(x)] |> Reduction
    end
    function NewODEmodel(x::Number, θ::AbstractVector{<:Number}; FullSol=false, kwargs...)
        FullSol && return Model.Map(x, θ; FullSol=true, kwargs...)
        sol = Model.Map(x, θ; FullSol=true, save_everystep=false, save_start=false, save_end=true, kwargs...)
        F(sol.u[end], sol.t[end], θ)
    end
    ModelMap(NewODEmodel, InDomain(Model), Domain(Model), (Model.xyp[1], length(out), Model.xyp[3]), Model.pnames, Model.inplace, Model.CustomEmbedding, name(Model), (Model.Meta[1:end-2]..., F, Model.Meta[end]))
end


"""
    IsODEParameter(DM::AbstractDataModel, SplitterFunc::Function; factor::Real=0.1, ADmode::Val=Val(:ForwardDiff), kwargs...) -> BitVector
Does the parameter enter the ODEFunction? Initial value parameters do not count.
"""
function IsODEParameter(DM::AbstractDataModel, SplitterFunc::Function; factor::Real=0.1, ADmode::Val=Val(:ForwardDiff), kwargs...)
    Jac = GetJac(ADmode, θ->SplitterFunc(θ)[2])
    CompareCols(Jac(MLE(DM) + factor*rand(pdim(DM))), Jac(MLE(DM) + factor*rand(pdim(DM))))
end
"""
    IsInitialParameter(DM::AbstractDataModel, SplitterFunc::Function; factor::Real=0.1, ADmode::Val=Val(:ForwardDiff), kwargs...) -> BitVector
Is the parameter an initial value parameter for the ODE model?
"""
function IsInitialParameter(DM::AbstractDataModel, SplitterFunc::Function; factor::Real=0.1, ADmode::Val=Val(:ForwardDiff), kwargs...)
    Jac = GetJac(ADmode, θ->SplitterFunc(θ)[1])
    CompareCols(Jac(MLE(DM) + factor*rand(pdim(DM))), Jac(MLE(DM) + factor*rand(pdim(DM))))
end
