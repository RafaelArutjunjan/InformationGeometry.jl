


struct ParamTrafo{F<:Function} <: AbstractVector{F}
    Trafos::AbstractVector{F}
    ConditionNames::AbstractVector{<:Symbol}
    function ParamTrafo(Trafos::AbstractVector{<:Function}, ConditionNames::AbstractVector{<:Symbol})
        @assert allunique(ConditionNames)
        new{eltype(Trafos)}(Trafos, ConditionNames)
    end
end
(P::ParamTrafo)(θ::AbstractVector{<:Number}; Cond::Union{Nothing,Symbol}=nothing) = _ExecuteParamTrafo(P, θ, Cond)

_ExecuteParamTrafo(P::ParamTrafo, θ::AbstractVector{<:Number}, ::Union{Nothing,Val{nothing}}) = [P.Trafos[i](θ) for i in eachindex(P.Trafos)]
function _ExecuteParamTrafo(P::ParamTrafo, θ::AbstractVector{<:Number}, Cond::Symbol)
    ind = findfirst(x -> x === Cond, P)
    @assert !isnothing(ind) "Unable to find condition $Cond in given ParamTrafo: $(P.ConditionNames)"
    P.Trafos[ind](θ)
end
# Should behave like AbstractVector{<:Function}?
Base.getindex(P::ParamTrafo, i) = getindex(P.Trafos, i)
# Forwarding:
for F in [:length, :size, :firstindex, :lastindex, :keys, :values]
    @eval Base.$F(P::ParamTrafo) = $F(P.Trafos)
end



# function TryToInferPnames()
# end



"""
    ConditionGrid(DMs::AbstractVector{<:AbstractDataModel}, Trafos::AbstractVector{<:Function})
Implements condition grid inspired by R package dMod.
Connects different given `DataModel`s via a vector of transformations, which read from the same collective vector of parameter values and compute the parameter configurations of the respective `DataModel`s from this.
Thus, this allows for easily connecting different datasets with distinct models while performing simultaneous inference with shared parameters between the models.
"""
struct ConditionGrid <: AbstractDataModel
    DMs::AbstractVector{<:AbstractDataModel}
    Trafos::Union{AbstractVector{<:Function}, ParamTrafo}
    LogPriorFn::Union{Function,Nothing}
    MLE::AbstractVector{<:Number}
    pnames::AbstractVector{Symbol}
    name::Symbol
    LogLikelihoodFn::Function
    ScoreFn::Function
    FisherMetricFn::Function
    LogLikeMLE::Number
    function ConditionGrid(DMs::AbstractVector{<:AbstractDataModel}, 
        Trafo::AbstractVector{<:Function}=(C=[0;cumsum(pdim.(DMs))];   Inds=[1+C[i-1]:C[i] for i in 2:length(C)];   [ViewElements(inds) for inds in Inds]), 
        LogPriorFn::Union{Function,Nothing}=nothing, 
        mle::AbstractVector=reduce(vcat, MLE.(DMs)),
        pnames::AbstractVector{<:StringOrSymb}=CreateSymbolNames(length(mle)), 
        Name::Symbol=Symbol(),
        LogLikelihoodFn::Function=θ::AbstractVector->mapreduce(loglikelihood, +, DMs, Trafo(θ)) + EvalLogPrior(LogPriorFn, θ),
        ScoreFn::Function=GetGrad(LogLikelihoodFn), # θ::AbstractVector->mapreduce(Score, +, DMs, [T(θ) for T in Trafos]) + EvalLogPriorGrad(LogPriorFn, θ),
        FisherMetricFn::Function=GetHess(Negate(LogLikelihoodFn)), # θ::AbstractVector->mapreduce(FisherMetric, +, DMs, [T(θ) for T in Trafos]) - EvalLogPriorHess(LogPriorFn, θ),
        LogLikeMLE::Number=LogLikelihoodFn(mle); name::StringOrSymb=Name, SkipOptim::Bool=false, kwargs...)
        # Check pnames correct?
        # Condition names already unique from ParamTrafo
        @assert length(Trafo) == length(DMs)
        @assert allunique(InformationGeometry.name.(DMs))

        @warn "The condition grid functionality is still experimental, use with caution!"

        Mle = if SkipOptim
            @warn "ConditionGrid: Not performing optimization out of the box!"
            mle
        else
            InformationGeometry.minimize(Negate(LogLikelihoodFn), mle; kwargs...)
        end            

        new(DMs, (Trafo isa ParamTrafo ? Trafo : ParamTrafo(Trafo, Symbol.(InformationGeometry.name.(DMs)))), LogPriorFn, Mle, Symbol.(pnames), Symbol(name), LogLikelihoodFn, ScoreFn, FisherMetricFn, LogLikeMLE)
    end
end

# For SciMLBase.remake
ConditionGrid(;
DMs::AbstractVector{<:AbstractDataModel}=DataModel[],
Trafos::Union{AbstractVector{<:Function}, ParamTrafo}=Function[],
LogPriorFn::Union{Function,Nothing}=nothing,
MLE::AbstractVector{<:Number}=Float64[],
pnames::AbstractVector{Symbol}=Symbol[],
name::Symbol=Symbol(),
LogLikelihoodFn::Function=x->-Inf,
ScoreFn::Function=x->[-Inf],
FisherMetricFn::Function=x->[-Inf],
LogLikeMLE::Number=-Inf, kwargs...) = ConditionGrid(DMs, Trafos, LogPriorFn, MLE, pnames, name, LogLikelihoodFn, ScoreFn, FisherMetricFn, LogLikeMLE; kwargs...)

Base.getindex(CG::ConditionGrid, i) = getindex(CG.DMs, i)

# Forwarding:
for F in [:length, :size, :firstindex, :lastindex, :keys, :values]
    @eval Base.$F(CG::ConditionGrid) = $F(CG.DMs)
end

MLE(CG::ConditionGrid) = CG.MLE
pdim(CG::ConditionGrid) = length(MLE(CG))
DOF(CG::ConditionGrid, mle::AbstractVector=MLE(CG)) = length(mle) - sum(map(NumberOfErrorParameters, CG.DMs, CG.Trafos(mle)))

LogLikeMLE(CG::ConditionGrid) = CG.LogLikeMLE
pnames(CG::ConditionGrid) = Pnames(CG) .|> string
Pnames(CG::ConditionGrid) = CG.pnames
name(CG::ConditionGrid) = CG.name
LogPrior(CG::ConditionGrid) = CG.LogPriorFn

loglikelihood(CG::ConditionGrid) = CG.LogLikelihoodFn
loglikelihood(CG::ConditionGrid, θ::AbstractVector{<:Number}) = loglikelihood(CG)(θ)

Score(CG::ConditionGrid) = CG.ScoreFn
Score(CG::ConditionGrid, θ::AbstractVector{<:Number}) = Score(CG)(θ)

FisherMetric(CG::ConditionGrid) = CG.FisherMetricFn
FisherMetric(CG::ConditionGrid, θ::AbstractVector{<:Number}) = FisherMetric(CG)(θ)

# Disable Boundaries for Optimization
GetDomain(CG::ConditionGrid) = nothing
GetInDomain(CG::ConditionGrid) = nothing
GetConstraintFunc(CG::ConditionGrid, startp::AbstractVector{<:Number}=Float64[]; kwargs...) = (nothing, nothing, nothing)

# Return nothing instead of producing MethodErrors
Data(CG::ConditionGrid) = nothing


GetDomainSafe(DM::DataModel; maxval::Real=1e2) = isnothing(GetDomain(DM)) ? FullDomain(length(MLE(DM)), maxval) : GetDomain(DM)
MultistartFit(CG::ConditionGrid; maxval::Real=1e2, Domain::HyperCube=(@info "Using naively constructed Domain for Multistart."; reduce(vcat, [GetDomainSafe(DM; maxval) for DM in CG.DMs])), kwargs...) = MultistartFit(CG, Domain; Domain, maxval, kwargs...)


## Prediction Functions
function EmbeddingMap(CG::ConditionGrid, θ::AbstractVector{<:Number}, S::Symbol; kwargs...)
    i = findfirst(x-> x === S, CG.Trafos.ConditionNames);    EmbeddingMap(CG, θ, WoundX(CG.DMs[i]), S, i; kwargs...)
end
function EmbeddingMatrix(CG::ConditionGrid, θ::AbstractVector{<:Number}, S::Symbol; kwargs...)
    i = findfirst(x-> x === S, CG.Trafos.ConditionNames);    EmbeddingMatrix(CG, θ, WoundX(CG.DMs[i]), S, i; kwargs...)
end
function EmbeddingMap!(Y::AbstractVector{<:Number}, CG::ConditionGrid, θ::AbstractVector{<:Number}, S::Symbol; kwargs...)
    i = findfirst(x-> x === S, CG.Trafos.ConditionNames);    EmbeddingMap!(Y, CG, θ, WoundX(CG.DMs[i]), S, i; kwargs...)
end
function EmbeddingMatrix!(J::AbstractMatrix{<:Number}, CG::ConditionGrid, θ::AbstractVector{<:Number}, S::Symbol; kwargs...)
    i = findfirst(x-> x === S, CG.Trafos.ConditionNames);    EmbeddingMatrix!(J, CG, θ, WoundX(CG.DMs[i]), S, i; kwargs...)
end

function EmbeddingMap(CG::ConditionGrid, θ::AbstractVector{<:Number}, woundX::AbstractVector, S::Symbol, i::Int=findfirst(x-> x===S, CG.Trafos.ConditionNames); kwargs...)
    EmbeddingMap(CG.DMs[i], CG.Trafos[i](θ), woundX; kwargs...)
end
function EmbeddingMatrix(CG::ConditionGrid, θ::AbstractVector{<:Number}, woundX::AbstractVector, S::Symbol, i::Int=findfirst(x-> x===S, CG.Trafos.ConditionNames); kwargs...)
    EmbeddingMatrix(CG.DMs[i], CG.Trafos[i](θ), woundX; kwargs...)
end
function EmbeddingMap!(Y::AbstractVector{<:Number}, CG::ConditionGrid, θ::AbstractVector{<:Number}, woundX::AbstractVector, S::Symbol, i::Int=findfirst(x-> x===S, CG.Trafos.ConditionNames); kwargs...)
    EmbeddingMap!(Y, CG.DMs[i], CG.Trafos[i](θ), woundX; kwargs...)
end
function EmbeddingMatrix!(J::AbstractMatrix{<:Number}, CG::ConditionGrid, θ::AbstractVector{<:Number}, woundX::AbstractVector, S::Symbol, i::Int=findfirst(x-> x===S, CG.Trafos.ConditionNames); kwargs...)
    EmbeddingMatrix!(J, CG.DMs[i], CG.Trafos[i](θ), woundX; kwargs...)
end


function Base.summary(CG::ConditionGrid)
    # Also use "RuntimeGeneratedFunction" string from build_function in ModelingToolkit.jl
    Name = string(name(CG))
    string(TYPE_COLOR, "Condition Grid",
    NO_COLOR, (length(Name) > 0 ? " "*ColoredString(name(CG)) : ""),
    " with pdim=", string(pdim(CG)), " containing ", string(length(CG.DMs)), " submodels")
end
# Multi-line display when used on its own in REPL
function Base.show(io::IO, ::MIME"text/plain", CG::ConditionGrid)
    LogPr = !isnothing(LogPrior(CG)) ? LogPrior(CG)(MLE(CG)) : nothing
    print(io, Base.summary(CG));    println(io, ": "*ColoredString(name.(CG.DMs)))
    println(io, "Maximal value of log-likelihood: "*string(round(LogLikeMLE(CG); sigdigits=5)))
    isnothing(LogPr) || println(io, "Log prior at MLE: "*string(round(LogPr; sigdigits=5)))
end
# Single line display
Base.show(io::IO, CG::ConditionGrid) = println(io, Base.summary(CG))

# Plotrecipe: plot all dms individually
RecipesBase.@recipe function f(CG::ConditionGrid, mle::AbstractVector{<:Number}=MLE(CG))
    plot_title --> string(name(CG))
    layout --> length(CG.DMs)
    for i in eachindex(CG.DMs)
        @series begin
            subplot := i
            dof --> DOF(CG)
            CG.DMs[i], CG.Trafos[i](mle)
        end
    end
end

