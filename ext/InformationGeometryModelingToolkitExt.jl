module InformationGeometryModelingToolkitExt


using InformationGeometry, ModelingToolkit, Optim, LineSearches

import ModelingToolkit: AbstractODESystem

import InformationGeometry: StringOrSymb, BoolArray
import InformationGeometry: xnames, ynames, xdim, ydim, Xnames, Ynames, CreateSymbolNames
import InformationGeometry: InformNames, GetModel, DataModel


"""
    InformNames(DS::AbstractDataSet, sys::ODESystem, observables::AbstractVector{<:Int})
Copy the state names saved in `ODESystem` to `DS`.
"""
function InformNames(DS::AbstractDataSet, sys::ModelingToolkit.AbstractSystem, observables::Union{Int,AbstractVector{<:Int},BoolArray})
    newxnames = xnames(DS) == CreateSymbolNames(xdim(DS),"x") ? [string(ModelingToolkit.get_iv(sys))] : Xnames(DS)
    newynames = ynames(DS) == CreateSymbolNames(ydim(DS),"y") ? string.((try ModelingToolkit.get_unknowns(sys) catch; ModelingToolkit.get_states(sys) end)[observables]) : Ynames(DS)
    InformNames(DS, newxnames, newynames)
end

# No ObservationFunction, therefore try to use sys to infer state names of ODEsys
# Extend for other DEFunctions in the future
function DataModel(DS::AbstractDataSet, sys::ModelingToolkit.AbstractSystem, u0::Union{Number,AbstractArray{<:Number},Function},
                        observables::Union{Int,AbstractVector{<:Int},BoolArray,Function}=1:length(u0), args...; tol::Real=1e-7, Domain::Union{HyperCube,Nothing}=nothing, OptimTol::Real=tol*1e-2, OptimMeth=LBFGS(;linesearch=LineSearches.BackTracking()), kwargs...)
    newDS = observables isa Union{Int,AbstractVector{<:Int}} ? InformNames(DS, sys, observables) : DS
    DataModel(newDS, GetModel(sys, u0, observables; tol, Domain, kwargs...), args...; OptimMeth, OptimTol)
end


import InformationGeometry: pnames, Domain, MaximalNumberOfArguments, GetArgLength, GetStartP, ModelMap

function GetModel(sys::ModelingToolkit.AbstractSystem, u0::Union{Number,AbstractArray{<:Number},Function}, observables::Union{Int,AbstractVector{<:Int},BoolArray,Function}=1:length(u0); startp::Union{Nothing,AbstractVector} = nothing,
                Domain::Union{HyperCube,Nothing}=nothing, inplace::Bool=true, pnames::AbstractVector{<:StringOrSymb}=string.(ModelingToolkit.get_ps(sys)), InDomain::Union{Function,Nothing}=nothing, name::StringOrSymb=ModelingToolkit.getname(sys), kwargs...)
    # Is there some optimization that can be applied here? Modellingtoolkitize(sys) or something?
    # sys = Sys isa Catalyst.ReactionSystem ? convert(ODESystem, Sys) : Sys
    
    Model = if sys isa ModelingToolkit.AbstractODESystem
        odefunc = ODEFunction{inplace}((ModelingToolkit.iscomplete(sys) ? sys : structural_simplify(sys)); jac = true)
        GetModel(odefunc, u0, observables; Domain=Domain, inplace=inplace, kwargs...)
    else
        throw("Not programmed for $(typeof(sys)) yet, please convert to a ModelingToolkit.AbstractODESystem first.")
    end
    if !isnothing(Domain) && (length(pnames) != length(Domain))
        @warn "Dimensionality of given Domain HyperCube inconsistent with given number of parameter names. Dropping given names and using default parameter names."
        pnames = CreateSymbolNames(length(Domain), "θ")
    end
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
    plen = if !isnothing(startp)
        length(startp)
    elseif Domain isa HyperCube
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

    length(pnames) != plen && (pnames = CreateSymbolNames(plen, "θ"))
    # new(Map, InDomain, Domain, xyp, pnames, inplace, CustomEmbedding)
    ModelMap(Model.Map, InDomain, Domain, xyp, pnames, Val(false), Val(true), name, (Model.Meta isa Tuple ? (sys, (Model.Meta[2:end])...) : Model.Meta))
end


import InformationGeometry: GetModelRobust
GetModelRobust(func::ODESystem, A, B; kwargs...) = GetModelRobust(ODEFunction(func), A, B; kwargs...)


import InformationGeometry: ExpTransform, LogTransform, Exp10Transform, Log10Transform

# Parameter transforms
ExpTransform(Sys::ODESystem, idxs::AbstractVector{<:Bool}=trues(length(ModelingToolkit.parameters(Sys))); kwargs...) = SystemTransform(Sys, exp, idxs; kwargs...)
LogTransform(Sys::ODESystem, idxs::AbstractVector{<:Bool}=trues(length(ModelingToolkit.parameters(Sys))); kwargs...) = SystemTransform(Sys, log, idxs; kwargs...)
Exp10Transform(Sys::ODESystem, idxs::AbstractVector{<:Bool}=trues(length(ModelingToolkit.parameters(Sys))); kwargs...) = SystemTransform(Sys, exp10, idxs; kwargs...)
Log10Transform(Sys::ODESystem, idxs::AbstractVector{<:Bool}=trues(length(ModelingToolkit.parameters(Sys))); kwargs...) = SystemTransform(Sys, log10, idxs; kwargs...)

import InformationGeometry: SystemTransform
"""
    SystemTransform(Sys::ODESystem, F::Function, idxs::AbstractVector{<:Bool}=trues(length(parameters(Sys)))) -> ODESystem
Transforms the parameters of an `ODESystem` according to a component-wise function `F`.
"""
function SystemTransform(Sys::AbstractODESystem, F::Function, idxs::AbstractVector{<:Bool}=trues(length(ModelingToolkit.parameters(Sys))))
    SubstDict = Dict(parameters(Sys) .=> [(idxs[i] ? F(x) : x) for (i,x) in enumerate(parameters(Sys))])
    NewEqs = [(equations(Sys)[i].lhs ~ substitute(equations(Sys)[i].rhs, SubstDict)) for i in eachindex(equations(Sys))]
    # renamed "states" to "unknowns": https://github.com/SciML/ModelingToolkit.jl/pull/2432
    ODESystem(NewEqs, independent_variables(Sys)[1], try ModelingToolkit.unknowns(Sys) catch; ModelingToolkit.states(Sys) end, ModelingToolkit.parameters(Sys); name=nameof(Sys))
end
export SystemTransform




"""
    ODESystemTimeRetardation(Sys::ODESystem) -> ODESystem
Applies [`TimeRetardation`](@ref) the to given `ODESystem` by multiplying all equations with the sigmodial derivative of the time retardation transformation.
The new parameters `T_shift` and `r` are appended to the ODE parameters.
"""
function InformationGeometry.ODESystemTimeRetardation(Sys::AbstractODESystem)
    t = independent_variables(Sys)[1]
    @parameters T_shift r_coupling
    RetFactor = exp10(r_coupling * t) / (exp10(r_coupling * t) + exp10(r_coupling * T_shift))
    NewEqs = [(equations(Sys)[i].lhs ~ equations(Sys)[i].rhs * RetFactor) for i in eachindex(equations(Sys))]
    # renamed "states" to "unknowns": https://github.com/SciML/ModelingToolkit.jl/pull/2432
    ODESystem(NewEqs, t, try ModelingToolkit.unknowns(Sys) catch; ModelingToolkit.states(Sys) end, [ModelingToolkit.parameters(Sys); [T_shift, r_coupling]]; name=Symbol("Time-Retarded " * string(nameof(Sys))))
end

import InformationGeometry: MakeSymbolicPars
# MakeMTKParameters(S::Symbol) = eval(Meta.parse("@parameters "*string(S)))[1];      MakeMTKParameters(S::AbstractArray{<:Symbol}) = [MakeMTKParameters(s) for s in S]
MakeSymbolicPars(X::AbstractVector{<:Symbol}) = eval(ModelingToolkit._parse_vars(:parameters, Real, X, ModelingToolkit.toparam))



end # module
