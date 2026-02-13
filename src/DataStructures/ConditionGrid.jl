

struct ParameterTransformations{F<:Function} <: AbstractParameterTransformations{F}
    Trafos::AbstractVector{F}
    IsLinear::AbstractVector{<:Bool}
    # Jacobians::AbstractVector{<:Union{UniformScaling,AbstractMatrix}} ## Precompute Jacobians for linear trafos?
    ConditionNames::AbstractVector{<:Symbol}
    function ParameterTransformations(Trafos::AbstractVector{<:Function}, ConditionNames::AbstractVector{<:Symbol}, testp::AbstractVector{<:Number}; ADmode::Val=Val(:ForwardDiff))
        Jac = DerivableFunctionsBase._GetJac(ADmode);   testp2 = testp .+ 0.5rand(length(testp))
        IsLinear = [Jac(Trafos[i], testp) ≈ Jac(Trafos[i], testp2) for i in eachindex(Trafos)]
        ParameterTransformations(Trafos, IsLinear, ConditionNames; ADmode)
    end
    function ParameterTransformations(Trafos::AbstractVector{<:Function}, ConditionNames::AbstractVector{<:Symbol}; IsLinear::AbstractVector{<:Bool}=falses(length(ConditionNames)), ADmode::Val=Val(:ForwardDiff))
        ParameterTransformations(Trafos, IsLinear, ConditionNames; ADmode)
    end
    function ParameterTransformations(Trafos::AbstractVector{<:Function}, IsLinear::AbstractVector{<:Bool}, ConditionNames::AbstractVector{<:Symbol}; ADmode::Val=Val(:ForwardDiff))
        @assert length(Trafos) == length(ConditionNames) == length(IsLinear)
        @assert allunique(ConditionNames)
        new{eltype(Trafos)}(Trafos, IsLinear, ConditionNames)
    end
end
const ParamTrafo = ParameterTransformations
# No nesting
ParameterTransformations(Trafos::ParameterTransformations, args...; kwargs...) = ParameterTransformations(Trafos.Trafos, args...; kwargs...)

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

Base.ComposedFunction(P::ParamTrafo, inner::Function) = ParamTrafo(map(x->x∘inner, P.Trafos), P.ConditionNames)
Base.ComposedFunction(outer::Function, P::ParamTrafo) = ParamTrafo(map(x->outer∘x, P.Trafos), P.ConditionNames)

ConditionNames(P::ParamTrafo) = P.ConditionNames
Trafos(P::ParamTrafo) = P.Trafos
Trafos(X::AbstractVector{<:Function}) = X

IsLinear(P::ParamTrafo) = P.IsLinear

IsIdentityTrafo(P::ParamTrafo) = IsIdentityTrafo(Trafos(P))
IsIdentityTrafo(P::AbstractVector{<:Function}) = all(isequal(identity), P)


# function TryToInferPnames()
# end

# Get Vector of inds for viewing into concatenated vector made of individual lengths
IndsVecFromLengths(Lenghts::AbstractVector{<:Int}) = (C=[0;cumsum(Lenghts)];   [1+C[i-1]:C[i] for i in 2:length(C)])

"""
    ConditionGrid(DMs::AbstractVector{<:AbstractDataModel}, Trafos::AbstractVector{<:Function}, mle::AbstractVector, LogPriorFn::Union{Nothing,Function}=nothing; Domain::Union{Nothing,Cuboid}=nothing, 
                    SkipOptim::Bool=false, pnames::AbstractVector{<:StringOrSymb}=CreateSymbolNames(length(mle)), name::StringOrSymb="")
Implements condition grid inspired by R package dMod.
Connects different given `DataModel`s via a vector of parameter transformations, which read from the same collective vector of outer parameter values and compute from them the individual parameter configurations of the respective `DataModel`s from this at every step.
Thus, this allows for easily connecting different datasets with distinct models while performing simultaneous inference with shared parameters between the models.
"""
struct ConditionGrid <: AbstractConditionGrid
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
    CostHessianFn::Function
    LogLikeMLE::Number
    ConditionGrid(DMs::AbstractVector{<:AbstractDataModel}, Trafo::AbstractVector{<:Function}, mle::AbstractVector, LogPriorFn::Union{Nothing,Function}=nothing; kwargs...) = ConditionGrid(DMs, Trafo, LogPriorFn, mle; kwargs...)
    function ConditionGrid(DMs::AbstractVector{<:AbstractDataModel}, 
        trafo::AbstractVector{<:Function}=[ViewElements(inds) for inds in IndsVecFromLengths(pdim.(DMs))], 
        LogPriorFn::Union{Function,Nothing}=nothing, 
        mle::AbstractVector=reduce(vcat, MLE.(DMs));
        ADmode::Val=Val(:ForwardDiff),
        pnames::AbstractVector{<:StringOrSymb}=CreateSymbolNames(length(mle)), 
        name::StringOrSymb=Symbol(),
        Domain::Union{Nothing,Cuboid}=nothing,
        Trafos::ParamTrafo=(trafo isa ParamTrafo ? trafo : ParamTrafo(trafo, Symbol.(InformationGeometry.name.(DMs)), mle)),
        # LogLikelihoodFn::Function=θ::AbstractVector->mapreduce(loglikelihood, +, DMs, Trafos(θ)) + EvalLogPrior(LogPriorFn, θ),
        LogLikelihoodFn::Function=(θ::AbstractVector->sum(loglikelihood(DM)(Trafos[i](θ)) for (i,DM) in enumerate(DMs)) + EvalLogPrior(LogPriorFn, θ)),
        ScoreFn::Function=MergeOneArgMethods(GetGrad(ADmode,LogLikelihoodFn), GetGrad!(ADmode,LogLikelihoodFn)), # θ::AbstractVector->mapreduce(Score, +, DMs, [T(θ) for T in Trafos]) + EvalLogPriorGrad(LogPriorFn, θ),
        CostHessianFn::Function=MergeOneArgMethods(GetHess(ADmode, Negate(LogLikelihoodFn)),GetHess!(ADmode, Negate(LogLikelihoodFn))),
        FisherInfoFn::Function=GetFisherInfoFn(DMs, Trafos, LogPriorFn; ADmode),
        LogLikeMLE::Union{Nothing,Number}=nothing, SkipOptim::Bool=false, SkipTests::Bool=false, verbose::Bool=true, kwargs...)
        # Check pnames correct?
        # Condition names already unique from ParamTrafo
        @assert length(Trafos) == length(DMs)
        @assert allunique(InformationGeometry.name.(DMs))
        !isnothing(Domain) && @assert length(Domain) == length(mle)

        Mle = if SkipOptim
            verbose && @info "ConditionGrid: Not performing optimization!"
            mle
        else
            InformationGeometry.minimize((Negate(LogLikelihoodFn), NegateBoth(ScoreFn), FisherInfoFn), mle, Domain; kwargs...)
        end
        logLikeMLE = isnothing(LogLikeMLE) ? LogLikelihoodFn(Mle) : LogLikeMLE

        new(DMs, Trafos, LogPriorFn, Mle, Symbol.(pnames), Domain, Symbol(name), LogLikelihoodFn, ScoreFn, FisherInfoFn, CostHessianFn, logLikeMLE)
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
CostHessianFn::Function=x->[-Inf],
LogLikeMLE::Union{Nothing,Number}=-Inf, 
SkipOptim::Bool=true, SkipTests::Bool=true, kwargs...) = ConditionGrid(DMs, Trafos, LogPriorFn, MLE; pnames, Domain, name, LogLikelihoodFn, ScoreFn, FisherInfoFn, CostHessianFn, LogLikeMLE, SkipOptim, SkipTests, kwargs...)

Base.getindex(CG::ConditionGrid, i) = getindex(Conditions(CG), i)

# Forwarding:
for F in [:length, :size, :firstindex, :lastindex, :keys, :values, :getindex]
    @eval Base.$F(CG::ConditionGrid, args...) = $F(Conditions(CG), args...)
end

MLE(CG::ConditionGrid) = CG.MLE
pdim(CG::ConditionGrid) = length(MLE(CG))
DOF(CG::ConditionGrid, mle::AbstractVector=MLE(CG)) = length(mle) - sum(map(NumberOfErrorParameters, Conditions(CG), Trafos(CG)(mle)))

LogLikeMLE(CG::ConditionGrid) = CG.LogLikeMLE
pnames(CG::ConditionGrid) = Pnames(CG) .|> string
Pnames(CG::ConditionGrid) = CG.pnames
name(CG::ConditionGrid) = CG.name
LogPrior(CG::ConditionGrid) = CG.LogPriorFn
Domain(CG::ConditionGrid) = CG.Domain
InDomain(CG::ConditionGrid) = nothing

HasPrior(CG::ConditionGrid) = !isnothing(LogPrior(CG))


loglikelihood(CG::ConditionGrid) = CG.LogLikelihoodFn
loglikelihood(CG::ConditionGrid, θ::AbstractVector{<:Number}) = loglikelihood(CG)(θ)

Score(CG::ConditionGrid) = CG.ScoreFn
Score(CG::ConditionGrid, θ::AbstractVector{<:Number}) = Score(CG)(θ)

FisherMetric(CG::ConditionGrid) = CG.FisherInfoFn
FisherMetric(CG::ConditionGrid, θ::AbstractVector{<:Number}) = FisherMetric(CG)(θ)

CostHessian(CG::ConditionGrid) = CG.CostHessianFn
AutoMetric(CG::ConditionGrid) = CostHessian(CG)
AutoMetric!(CG::ConditionGrid) = CostHessian(CG)

# Disable Boundaries for Optimization
GetDomain(CG::ConditionGrid) = Domain(CG)
GetInDomain(CG::ConditionGrid) = nothing
GetConstraintFunc(CG::ConditionGrid, startp::AbstractVector{<:Number}=Float64[]; kwargs...) = (nothing, nothing, nothing)

# Return nothing instead of producing MethodErrors
Data(CG::ConditionGrid) = nothing
Conditions(CG::ConditionGrid) = CG.DMs
ConditionNames(CG::ConditionGrid) = ConditionNames(Trafos(CG))
Trafos(CG::ConditionGrid) = CG.Trafos

DataspaceDim(CG::ConditionGrid) = sum(length∘ydata, Conditions(CG))

ydim(CG::ConditionGrid) = sum(ydim, Conditions(CG))
Ynames(CG::ConditionGrid) = mapreduce(ydim, vcat, Conditions(CG))

xdim(CG::ConditionGrid) = (xd=xdim.(Conditions(CG));  @assert all(isequal(xd[1]), xd);  xd[1])
Xnames(CG::ConditionGrid) = (xd=Xnames.(Conditions(CG));  @assert all(isequal(xd[1]), xd);  xd[1])

xnames(CG::ConditionGrid) = Xnames(CG) .|> string
ynames(CG::ConditionGrid) = Ynames(CG) .|> string


function GetDomainSafe(DM::AbstractDataModel; maxval::Real=1e2, verbose::Bool=true)
    !isnothing(GetDomain(DM)) && return GetDomain(DM)
    verbose && @warn "Making Domain [-$maxval, $maxval]^$(pdim(DM)) for $(typeof(DM)) $(name(DM))"
    FullDomain(length(MLE(DM)), maxval)
end
function MultistartFit(CG::AbstractConditionGrid; maxval::Real=1e2, Domain::Union{Nothing,HyperCube}=Domain(CG), verbose::Bool=true, kwargs...)
    if isnothing(Domain)
        pdim(CG) != sum(pdim.(Conditions(CG))) && throw(ArgumentError("Domain HyperCube or Distribution for sampling must be specified as second argument for ConditionGrids."))
        @warn "Using naively constructed Domain for Multistart. If you get an error, try specifying the Domain manually!"
        Domain = reduce(vcat, [GetDomainSafe(DM; maxval, verbose) for DM in Conditions(CG)])
    end
    MultistartFit(CG, Domain; Domain, maxval, verbose, kwargs...)
end


## Prediction Functions
function EmbeddingMap(CG::AbstractConditionGrid, θ::AbstractVector{<:Number}, S::Symbol; kwargs...)
    i = findfirst(x-> x === S, ConditionNames(CG));    EmbeddingMap(CG, θ, WoundX(Conditions(CG)[i]), S, i; kwargs...)
end
function EmbeddingMatrix(CG::AbstractConditionGrid, θ::AbstractVector{<:Number}, S::Symbol; kwargs...)
    i = findfirst(x-> x === S, ConditionNames(CG));    EmbeddingMatrix(CG, θ, WoundX(Conditions(CG)[i]), S, i; kwargs...)
end
function EmbeddingMap!(Y::AbstractVector{<:Number}, CG::AbstractConditionGrid, θ::AbstractVector{<:Number}, S::Symbol; kwargs...)
    i = findfirst(x-> x === S, ConditionNames(CG));    EmbeddingMap!(Y, CG, θ, WoundX(Conditions(CG)[i]), S, i; kwargs...)
end
function EmbeddingMatrix!(J::AbstractMatrix{<:Number}, CG::AbstractConditionGrid, θ::AbstractVector{<:Number}, S::Symbol; kwargs...)
    i = findfirst(x-> x === S, ConditionNames(CG));    EmbeddingMatrix!(J, CG, θ, WoundX(Conditions(CG)[i]), S, i; kwargs...)
end

function EmbeddingMap(CG::AbstractConditionGrid, θ::AbstractVector{<:Number}, woundX::AbstractVector, S::Symbol, i::Int=findfirst(x-> x===S, ConditionNames(CG)); kwargs...)
    EmbeddingMap(Conditions(CG)[i], Trafos(CG)[i](θ), woundX; kwargs...)
end
function EmbeddingMatrix(CG::AbstractConditionGrid, θ::AbstractVector{<:Number}, woundX::AbstractVector, S::Symbol, i::Int=findfirst(x-> x===S, ConditionNames(CG)); kwargs...)
    EmbeddingMatrix(Conditions(CG)[i], Trafos(CG)[i](θ), woundX; kwargs...)
end
function EmbeddingMap!(Y::AbstractVector{<:Number}, CG::AbstractConditionGrid, θ::AbstractVector{<:Number}, woundX::AbstractVector, S::Symbol, i::Int=findfirst(x-> x===S, ConditionNames(CG)); kwargs...)
    EmbeddingMap!(Y, Conditions(CG)[i], Trafos(CG)[i](θ), woundX; kwargs...)
end
function EmbeddingMatrix!(J::AbstractMatrix{<:Number}, CG::AbstractConditionGrid, θ::AbstractVector{<:Number}, woundX::AbstractVector, S::Symbol, i::Int=findfirst(x-> x===S, ConditionNames(CG)); kwargs...)
    EmbeddingMatrix!(J, Conditions(CG)[i], Trafos(CG)[i](θ), woundX; kwargs...)
end

## Create Master Methods which simply concatenate?
function EmbeddingMap(CG::AbstractConditionGrid, θ::AbstractVector{<:Number}; verbose::Bool=false, kwargs...)
    verbose && @warn "EmbeddingMap: Simply concatenating all predictions for all conditions."
    reduce(vcat, [EmbeddingMap(CG, θ, S; kwargs...) for S in ConditionNames(CG)])
end

function EmbeddingMatrix(CG::AbstractConditionGrid, θ::AbstractVector{<:Number}; verbose::Bool=false, ADmode::Val=Val(:ForwardDiff), kwargs...)
    verbose && @warn "EmbeddingMatrix: Simply concatenating all Jacobians for all conditions."
    reduce(vcat, [EmbeddingMatrix(CG, θ, S; kwargs...) * GetJac(ADmode, Trafos(CG)[i])(θ) for (i,S) in enumerate(ConditionNames(CG))])
end


function TrafoLength(P::ParamTrafo; max::Int=MaxArgLen)
    for i in 1:max
        try P(ones(i)); return i catch; end
    end;    max
end

function SymbolicParamTrafo(CG::AbstractConditionGrid; GenericNames::Bool=true)
    Pars = GenericNames ? SymbolicArguments((1,1,pdim(CG)))[end] : MakeSymbolicPars(Pnames(CG))
    SymbolicParamTrafo(Trafos(CG), Pars)
end
function SymbolicParamTrafo(P::ParamTrafo, Pars::Union{AbstractArray{<:Num}, Symbolics.Arr}=SymbolicArguments((1,1,TrafoLength(P)))[end]; GenericNames::Bool=true)
    [F === identity ? "θ" : F(Pars) for F in P]
end
function ParamTrafoString(CG::Union{AbstractConditionGrid,ParamTrafo}, args...; GenericNames::Bool=true, kwargs...)
   Shorten(S::AbstractString) = !startswith(S, "θ") ? (@view S[findfirst('[', S):end]) : S
   if GenericNames
      "[" * join("θ->" .* Shorten.(string.(SymbolicParamTrafo(CG, args...; GenericNames, kwargs...))), ", ") * "]"
   else
      "θ ⟼ " * Shorten(string(SymbolicParamTrafo(CG, args...; GenericNames, kwargs...)))
   end
end


# Plotrecipe: plot all dms individually
RecipesBase.@recipe function f(CG::AbstractConditionGrid, mle::AbstractVector{<:Number}=MLE(CG); idxs=1:length(Conditions(CG)), ConditionNames=view(InformationGeometry.ConditionNames(CG), idxs), 
                        Confnum=1, Fisher=any(Confnum .> 0) ? FisherMetric(CG, mle) : nothing, ADmode=Val(:ForwardDiff))
    ConditionNames isa Symbol && (ConditionNames = [ConditionNames])
    @assert ConditionNames isa AbstractVector{<:Symbol} && idxs isa Union{<:Int,AbstractArray{<:Int}}
    @assert all(∈(InformationGeometry.ConditionNames(CG)), ConditionNames) "Given condition names $ConditionNames not all in $(InformationGeometry.ConditionNames(CG))"
    @assert isnothing(Fisher) || (size(Fisher,1) == size(Fisher,2) == length(mle))
    plot_title --> string(name(CG))
    layout --> length(ConditionNames)
    size --> PlotSizer(length(ConditionNames))
    InvFisher = if isnothing(Fisher) || IsIdentityTrafo(Trafos(CG))
        nothing
    else
        NotPosDef(Fisher) && @warn "Given Fisher information not positive definite, using pseudo-inverse for plot."
        pinv(Fisher)
    end
    for (i,Name) in enumerate(ConditionNames)
        @series begin
            subplot := i
            dof --> DOF(CG)
            Confnum --> Confnum
            ind = findfirst(isequal(Name),ConditionNames)
            if !isnothing(Fisher)
                Fisher --> if Trafos(CG)[ind] == identity
                        Fisher
                    else
                        J = DerivableFunctionsBase._GetJac(ADmode)(Trafos(CG)[ind], mle)
                        inv(J * InvFisher * transpose(J))
                    end
            else
                Fisher --> nothing
            end
            Conditions(CG)[ind], Trafos(CG)[ind](mle)
        end
    end
end

## Order of data and fits is often messed up in nested plot recipes, so plot each individually and layout instead
function RecipesBase.plot(CG::AbstractConditionGrid, mle::AbstractVector{<:Number}=MLE(CG); idxs=1:length(Conditions(CG)), ConditionNames=view(InformationGeometry.ConditionNames(CG), idxs), dof=DOF(CG),
                        Confnum=1, Fisher=any(Confnum .> 0) ? FisherMetric(CG, mle) : nothing, ADmode=Val(:ForwardDiff), kwargs...)
    ConditionNames isa Symbol && (ConditionNames = [ConditionNames])
    @assert ConditionNames isa AbstractVector{<:Symbol} && idxs isa Union{<:Int,AbstractArray{<:Int}}
    @assert all(∈(InformationGeometry.ConditionNames(CG)), ConditionNames) "Given condition names $ConditionNames not all in $(InformationGeometry.ConditionNames(CG))"
    @assert isnothing(Fisher) || (size(Fisher,1) == size(Fisher,2) == length(mle))
    InvFisher = if isnothing(Fisher) || IsIdentityTrafo(Trafos(CG))
        nothing
    else
        NotPosDef(Fisher) && @warn "Given Fisher information not positive definite, using pseudo-inverse for plot."
        pinv(Fisher)
    end
    Plts = []
    for Name in ConditionNames
        ind = findfirst(isequal(Name),ConditionNames)
        FisherSub = if !isnothing(Fisher)
                if Trafos(CG)[ind] == identity
                    Fisher
                else
                    J = DerivableFunctionsBase._GetJac(ADmode)(Trafos(CG)[ind], mle)
                    inv(J * InvFisher * transpose(J))
                end
            else
                nothing
            end
        push!(Plts, RecipesBase.plot(Conditions(CG)[ind], Trafos(CG)[ind](mle); Confnum, dof, Fisher=FisherSub, plot_title="", kwargs...))
    end
    RecipesBase.plot(Plts...; plot_title=string(name(CG)), layout=length(ConditionNames), size=PlotSizer(length(ConditionNames)))
end



function ConditionSpecificProfiles(CG::AbstractConditionGrid, P::AbstractProfiles; idxs::AbstractVector{<:Int}=1:pdim(CG), OffsetResults::Bool=true, OffsetResultsBy::Union{Nothing,Number}=nothing, Trafo::Function=identity, Interpolate::Bool=false, Interp=QuadraticInterpolation, kwargs...)
    Plt = RecipesBase.plot(P; lw=2, label="", Interpolate);    PopulatedInds = IsPopulated(P);   k = 0
    for i in idxs
        if PopulatedInds[i]
            k += 1
            for j in eachindex(Conditions(CG))
                L = map(Trafo∘Negloglikelihood(Conditions(CG)[j])∘Trafos(CG)[j], Trajectories(P)[i])
                OffsetResults && (isnothing(OffsetResultsBy) ? (L .-= minimum(L)) : (L .-= OffsetResultsBy))
                X, Y = if Interpolate
                    F = Interp(L, getindex.(Trajectories(P)[i], i));   xran = range(F.t[1], F.t[end]; length=300);   (xran, F.(xran))
                else
                    getindex.(Trajectories(P)[i], i), L
                end
                RecipesBase.plot!(Plt, X, Y; color=j+2, label=ApplyTrafoNames("Contribution "*string(name(Conditions(CG)[j])), Trafo), lw=1.5, legend=true, subplot=k, kwargs...)
            end
        end
    end;  Plt
end

function ConditionSpecificWaterFalls(CG::AbstractConditionGrid, R::AbstractMultistartResults; BiLog::Bool=true, Trafo::Function=(BiLog ? InformationGeometry.BiLog : identity), OffsetResults::Bool=true, OffsetResultsBy::Union{Nothing,Number}=nothing, kwargs...)
    Plt = RecipesBase.plot(; xlabel="Run (sorted)", ylabel=ApplyTrafoNames("CostFunction", Trafo))
    for j in eachindex(Conditions(CG))
        L = map(Negloglikelihood(Conditions(CG)[j])∘Trafos(CG)[j], R.FinalPoints)
        L = @view L[1:findlast(isfinite, L)]
        OffsetResults && (isnothing(OffsetResultsBy) ? (L .-= minimum(L)) : (L .-= OffsetResultsBy))
        RecipesBase.plot!(Plt, Trafo.(L); label=ApplyTrafoNames("Contribution "*string(name(Conditions(CG)[j])), Trafo), lw=1.5, color=j, legend=true, kwargs...)
    end;  Plt
end



"""
    SplitObservablesIntoConditions(DM::DataModel, Structure::AbstractVector{<:AbstractVector{<:Int}}=[[i] for i in 1:ydim(DM)]) -> ConditionGrid
Takes `DataModel` with ydim > 1 and artificially splits the different observed components it into `ConditionGrid` with different conditions according to given `Structure`.
"""
function SplitObservablesIntoConditions(DM::AbstractDataModel, Structure::AbstractVector{<:AbstractVector{<:Int}}=[[i] for i in 1:ydim(DM)]; kwargs...)
    @assert length(Conditions(DM)) == 1 "Given DataModel already has multiple conditions! Split each condition separately."
    @assert ydim(DM) > 1 "Not enough observables for splitting, ydim=1!"
    @assert xdim(DM) == 1
    @assert all(s->allunique(s) && all(1 .≤ s .≤ ydim(DM)), Structure)
    # No double counting of observables
    @assert all(isempty, [Si ∩ Sj for (i,Si) in enumerate(Structure), (j,Sj) in enumerate(Structure) if i > j])
    # Check if any observables missing
    sort(reduce(union, Structure)) != 1:ydim(DM) && @warn "SplitObservablesIntoConditions: Missing components $(setdiff(1:ydim(DM), reduce(union, Structure))) in given Structure."

    DMs = [DataModel(SubDataSetComponent(Data(DM), Structure[i]), 
        SubComponentModel(Predictor(DM), Structure[i], xdim(DM), ydim(DM)),
        SubComponentDModel(dPredictor(DM), Structure[i], xdim(DM), ydim(DM)),
        MLE(DM), true; name=Symbol(Structure[i])) for i in eachindex(Structure)]
    ConditionGrid(DMs, InformationGeometry.ParamTrafo([identity for i in eachindex(Structure)], Symbol.(Structure)), LogPrior(DM), MLE(DM); name=name(DM), SkipOptim=true, SkipTests=true)
end
