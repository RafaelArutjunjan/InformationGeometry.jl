

struct ParamTrafo{F<:Function} <: AbstractVector{F}
    Trafos::AbstractVector{F}
    ConditionNames::AbstractVector{<:Symbol}
    function ParamTrafo(Trafos::AbstractVector{<:Function}, ConditionNames::AbstractVector{<:Symbol})
        @assert allunique(ConditionNames)
        new{eltype(Trafos)}(Trafos, ConditionNames)
    end
end
(P::ParamTrafo)(θ::AbstractVector{<:Number}; Cond::Union{Nothing,Symbol}=nothing) = _ExecuteParamTrafo(P, θ, Cond)
(P::ParamTrafo)(θ::AbstractVector{<:Number}, i::Int; kwargs...) = P.Trafos[i](θ)

_ExecuteParamTrafo(P::ParamTrafo, θ::AbstractVector{<:Number}, ::Union{Nothing,Val{nothing}}) = [P.Trafos[i](θ) for i in eachindex(P.Trafos)]
function _ExecuteParamTrafo(P::ParamTrafo, θ::AbstractVector{<:Number}, Cond::Symbol)
    ind = findfirst(x -> x === Cond, P)
    @assert !isnothing(ind) "Unable to find condition $Cond in given ParamTrafo: $(P.ConditionNames)"
    P.Trafos[ind](θ)
end

# Forwarding:
for F in [:length, :size, :firstindex, :lastindex, :getindex, :keys, :values]
    @eval Base.$F(P::ParamTrafo, args...) = $F(P.Trafos, args...)
end

const ParameterTransformations = ParamTrafo


# function TryToInferPnames()
# end



"""
    ConditionGrid(DMs::AbstractVector{<:AbstractDataModel}, Trafos::AbstractVector{<:Function}, mle::AbstractVector; Domain::Union{Nothing,Cuboid}=nothing, 
                    SkipOptim::Bool=false, pnames::AbstractVector{<:StringOrSymb}=CreateSymbolNames(length(mle)), name::StringOrSymb="")
Implements condition grid inspired by R package dMod.
Connects different given `DataModel`s via a vector of parameter transformations, which read from the same collective vector of outer parameter values and compute from them the individual parameter configurations of the respective `DataModel`s from this at every step.
Thus, this allows for easily connecting different datasets with distinct models while performing simultaneous inference with shared parameters between the models.
"""
struct ConditionGrid <: AbstractDataModel
    DMs::AbstractVector{<:AbstractDataModel}
    Trafos::ParamTrafo
    LogPriorFn::Union{Function,Nothing}
    MLE::AbstractVector{<:Number}
    pnames::AbstractVector{Symbol}
    Domain::Union{Nothing,Cuboid}
    name::Symbol
    LogLikelihoodFn::Function
    ScoreFn::Function
    FisherInfoFn::Function
    LogLikeMLE::Number
    ConditionGrid(DMs::AbstractVector{<:AbstractDataModel}, Trafo::AbstractVector{<:Function}, mle::AbstractVector, LogPriorFn::Union{Nothing,Function}=nothing; kwargs...) = ConditionGrid(DMs, Trafo, LogPriorFn, mle; kwargs...)
    function ConditionGrid(DMs::AbstractVector{<:AbstractDataModel}, 
        trafo::AbstractVector{<:Function}=(C=[0;cumsum(pdim.(DMs))];   Inds=[1+C[i-1]:C[i] for i in 2:length(C)];   [ViewElements(inds) for inds in Inds]), 
        LogPriorFn::Union{Function,Nothing}=nothing, 
        mle::AbstractVector=reduce(vcat, MLE.(DMs));
        ADmode::Val=Val(:ForwardDiff),
        pnames::AbstractVector{<:StringOrSymb}=CreateSymbolNames(length(mle)), 
        name::StringOrSymb=Symbol(),
        Domain::Union{Nothing,Cuboid}=nothing,
        Trafos::ParamTrafo=(trafo isa ParamTrafo ? trafo : ParamTrafo(trafo, Symbol.(InformationGeometry.name.(DMs)))),
        # LogLikelihoodFn::Function=θ::AbstractVector->mapreduce(loglikelihood, +, DMs, Trafos(θ)) + EvalLogPrior(LogPriorFn, θ),
        LogLikelihoodFn::Function=(θ::AbstractVector->sum(loglikelihood(DM)(Trafos[i](θ)) for (i,DM) in enumerate(DMs)) + EvalLogPrior(LogPriorFn, θ)),
        ScoreFn::Function=MergeOneArgMethods(GetGrad(ADmode, LogLikelihoodFn), GetGrad!(ADmode,LogLikelihoodFn)), # θ::AbstractVector->mapreduce(Score, +, DMs, [T(θ) for T in Trafos]) + EvalLogPriorGrad(LogPriorFn, θ),
        FisherInfoFn::Function=MergeOneArgMethods(GetHess(ADmode,Negate(LogLikelihoodFn)), GetHess!(ADmode,Negate(LogLikelihoodFn))), # θ::AbstractVector->mapreduce(FisherMetric, +, DMs, [T(θ) for T in Trafos]) - EvalLogPriorHess(LogPriorFn, θ),
        LogLikeMLE::Number=LogLikelihoodFn(mle), SkipOptim::Bool=false, SkipTests::Bool=false, kwargs...)
        # Check pnames correct?
        # Condition names already unique from ParamTrafo
        @assert length(Trafos) == length(DMs)
        @assert allunique(InformationGeometry.name.(DMs))
        !isnothing(Domain) && @assert length(Domain) == length(mle)

        @warn "The ConditionGrid functionality is still experimental, use with caution!"

        Mle = if SkipOptim
            @warn "ConditionGrid: Not performing optimization out of the box!"
            mle
        else
            InformationGeometry.minimize(Negate(LogLikelihoodFn), mle, Domain; kwargs...)
        end            

        new(DMs, Trafos, LogPriorFn, Mle, Symbol.(pnames), Domain, Symbol(name), LogLikelihoodFn, ScoreFn, FisherInfoFn, LogLikeMLE)
    end
end

# For SciMLBase.remake
ConditionGrid(;
DMs::AbstractVector{<:AbstractDataModel}=DataModel[],
Trafos::Union{AbstractVector{<:Function}, ParamTrafo}=Function[],
LogPriorFn::Union{Function,Nothing}=nothing,
MLE::AbstractVector{<:Number}=Float64[],
pnames::AbstractVector{Symbol}=Symbol[],
Domain::Union{Nothing,Cuboid}=nothing,
name::Symbol=Symbol(),
LogLikelihoodFn::Function=x->-Inf,
ScoreFn::Function=x->[-Inf],
FisherInfoFn::Function=x->[-Inf],
LogLikeMLE::Number=-Inf, 
SkipOptim::Bool=true, kwargs...) = ConditionGrid(DMs, Trafos, LogPriorFn, MLE; pnames, Domain, name, LogLikelihoodFn, ScoreFn, FisherInfoFn, LogLikeMLE, SkipOptim, kwargs...)

Base.getindex(CG::ConditionGrid, i) = getindex(Conditions(CG), i)

# Forwarding:
for F in [:length, :size, :firstindex, :lastindex, :keys, :values, :getindex]
    @eval Base.$F(CG::ConditionGrid, args...) = $F(Conditions(CG), args...)
end

MLE(CG::ConditionGrid) = CG.MLE
pdim(CG::ConditionGrid) = length(MLE(CG))
DOF(CG::ConditionGrid, mle::AbstractVector=MLE(CG)) = length(mle) - sum(map(NumberOfErrorParameters, Conditions(CG), CG.Trafos(mle)))

LogLikeMLE(CG::ConditionGrid) = CG.LogLikeMLE
pnames(CG::ConditionGrid) = Pnames(CG) .|> string
Pnames(CG::ConditionGrid) = CG.pnames
name(CG::ConditionGrid) = CG.name
LogPrior(CG::ConditionGrid) = CG.LogPriorFn
Domain(CG::ConditionGrid) = CG.Domain
InDomain(CG::ConditionGrid) = nothing

loglikelihood(CG::ConditionGrid) = CG.LogLikelihoodFn
loglikelihood(CG::ConditionGrid, θ::AbstractVector{<:Number}) = loglikelihood(CG)(θ)

Score(CG::ConditionGrid) = CG.ScoreFn
Score(CG::ConditionGrid, θ::AbstractVector{<:Number}) = Score(CG)(θ)

FisherMetric(CG::ConditionGrid) = CG.FisherInfoFn
FisherMetric(CG::ConditionGrid, θ::AbstractVector{<:Number}) = FisherMetric(CG)(θ)

# Disable Boundaries for Optimization
GetDomain(CG::ConditionGrid) = Domain(CG)
GetInDomain(CG::ConditionGrid) = nothing
GetConstraintFunc(CG::ConditionGrid, startp::AbstractVector{<:Number}=Float64[]; kwargs...) = (nothing, nothing, nothing)

# Return nothing instead of producing MethodErrors
Data(CG::ConditionGrid) = nothing
Conditions(CG::ConditionGrid) = CG.DMs

GetDomainSafe(DM::DataModel; maxval::Real=1e2) = isnothing(GetDomain(DM)) ? FullDomain(length(MLE(DM)), maxval) : GetDomain(DM)
function MultistartFit(CG::ConditionGrid; dof=DOF(CG), maxval::Real=1e2, Domain::Union{Nothing,HyperCube}=Domain(CG), kwargs...)
    if isnothing(Domain)
        pdim(CG) != sum(pdim.(Conditions(CG))) && throw(ArgumentError("Domain HyperCube or Distribution for sampling must be specified as second argument for ConditionGrids."))
        @warn "Using naively constructed Domain for Multistart. If you get an error, try specifying the Domain manually!"
        Domain = reduce(vcat, [GetDomainSafe(DM; maxval) for DM in Conditions(CG)])
    end
    MultistartFit(CG, Domain; dof, Domain, maxval, kwargs...)
end


## Prediction Functions
function EmbeddingMap(CG::ConditionGrid, θ::AbstractVector{<:Number}, S::Symbol; kwargs...)
    i = findfirst(x-> x === S, CG.Trafos.ConditionNames);    EmbeddingMap(CG, θ, WoundX(Conditions(CG)[i]), S, i; kwargs...)
end
function EmbeddingMatrix(CG::ConditionGrid, θ::AbstractVector{<:Number}, S::Symbol; kwargs...)
    i = findfirst(x-> x === S, CG.Trafos.ConditionNames);    EmbeddingMatrix(CG, θ, WoundX(Conditions(CG)[i]), S, i; kwargs...)
end
function EmbeddingMap!(Y::AbstractVector{<:Number}, CG::ConditionGrid, θ::AbstractVector{<:Number}, S::Symbol; kwargs...)
    i = findfirst(x-> x === S, CG.Trafos.ConditionNames);    EmbeddingMap!(Y, CG, θ, WoundX(Conditions(CG)[i]), S, i; kwargs...)
end
function EmbeddingMatrix!(J::AbstractMatrix{<:Number}, CG::ConditionGrid, θ::AbstractVector{<:Number}, S::Symbol; kwargs...)
    i = findfirst(x-> x === S, CG.Trafos.ConditionNames);    EmbeddingMatrix!(J, CG, θ, WoundX(Conditions(CG)[i]), S, i; kwargs...)
end

function EmbeddingMap(CG::ConditionGrid, θ::AbstractVector{<:Number}, woundX::AbstractVector, S::Symbol, i::Int=findfirst(x-> x===S, CG.Trafos.ConditionNames); kwargs...)
    EmbeddingMap(Conditions(CG)[i], CG.Trafos[i](θ), woundX; kwargs...)
end
function EmbeddingMatrix(CG::ConditionGrid, θ::AbstractVector{<:Number}, woundX::AbstractVector, S::Symbol, i::Int=findfirst(x-> x===S, CG.Trafos.ConditionNames); kwargs...)
    EmbeddingMatrix(Conditions(CG)[i], CG.Trafos[i](θ), woundX; kwargs...)
end
function EmbeddingMap!(Y::AbstractVector{<:Number}, CG::ConditionGrid, θ::AbstractVector{<:Number}, woundX::AbstractVector, S::Symbol, i::Int=findfirst(x-> x===S, CG.Trafos.ConditionNames); kwargs...)
    EmbeddingMap!(Y, Conditions(CG)[i], CG.Trafos[i](θ), woundX; kwargs...)
end
function EmbeddingMatrix!(J::AbstractMatrix{<:Number}, CG::ConditionGrid, θ::AbstractVector{<:Number}, woundX::AbstractVector, S::Symbol, i::Int=findfirst(x-> x===S, CG.Trafos.ConditionNames); kwargs...)
    EmbeddingMatrix!(J, Conditions(CG)[i], CG.Trafos[i](θ), woundX; kwargs...)
end



SymbolicParamTrafo(CG::ConditionGrid, GenericNames::Bool=true) = (GenericNames ? SymbolicArguments((1,1,pdim(CG)))[end] : MakeSymbolicPars(Pnames(CG))) |> CG.Trafos .|> collect
function ParamTrafoString(CG::ConditionGrid, GenericNames::Bool=true, args...; kwargs...)
   Shorten(S::AbstractString) = @view S[findfirst('[', S):end]
   if GenericNames
      "[" * join("θ->" .* Shorten.(string.(SymbolicParamTrafo(CG, GenericNames, args...; kwargs...))), ", ") * "]"
   else
      "θ ⟼ " * Shorten(string(SymbolicParamTrafo(CG, GenericNames, args...; kwargs...)))
   end
end


# Plotrecipe: plot all dms individually
RecipesBase.@recipe function f(CG::ConditionGrid, mle::AbstractVector{<:Number}=MLE(CG))
    plot_title --> string(name(CG))
    layout --> length(Conditions(CG))
    for i in eachindex(Conditions(CG))
        @series begin
            subplot := i
            dof --> DOF(CG)
            Conditions(CG)[i], CG.Trafos[i](mle)
        end
    end
end


function ConditionSpecificProfiles(CG::ConditionGrid, P::AbstractProfiles; OffsetResults::Bool=true, kwargs...)
   Plt = RecipesBase.plot(P; lw=2);    PopulatedInds = IsPopulated(P);   k = 0
   for i in 1:pdim(CG)
      if PopulatedInds[i]
         k += 1
         for j in eachindex(Conditions(CG))
            L = map(Negloglikelihood(Conditions(CG)[j])∘CG.Trafos[j], Trajectories(P)[i])
            OffsetResults && (L .-= minimum(L))
            RecipesBase.plot!(Plt, getindex.(Trajectories(P)[i], i), L; color=j+2, label="Contribution "*string(name(Conditions(CG)[j])), lw=1.5, legend=true, subplot=k, kwargs...)
         end
      end
   end;  Plt
end

function ConditionSpecificWaterFalls(CG::ConditionGrid, R::AbstractMultistartResults; BiLog::Bool=true, Trafo::Function=(BiLog ? InformationGeometry.BiLog : identity), OffsetResults::Bool=true, kwargs...)
   Plt = RecipesBase.plot(; xlabel="Run (sorted)", ylabel=(BiLog ? "BiLog(" : "")*"CostFunction"*(BiLog ? ")" : ""))
   for j in eachindex(Conditions(CG))
       L = map(Negloglikelihood(Conditions(CG)[j])∘CG.Trafos[j], R.FinalPoints)
       L = @view L[1:findlast(isfinite, L)]
       OffsetResults && (L .-= minimum(L))
       BiLog && (L .= Trafo.(L))
       RecipesBase.plot!(Plt, L; label="Contribution "*string(name(Conditions(CG)[j])), lw=1.5, color=j, legend=true, kwargs...)
   end;  Plt
end