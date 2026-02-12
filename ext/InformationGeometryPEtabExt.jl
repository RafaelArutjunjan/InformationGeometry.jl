module InformationGeometryPEtabExt


using InformationGeometry, PEtab, DataFrames, SciMLBase, ModelingToolkitBase, ForwardDiff, RecipesBase

InformationGeometry.MLE(P::PEtabODEProblem) = PEtab.get_x(P)
InformationGeometry.HyperCube(P::PEtabODEProblem) = HyperCube(P.lower_bounds, P.upper_bounds)
InformationGeometry.MLE(R::PEtabOptimisationResult) = R.xmin

InformationGeometry.pdim(M::PEtabModel) = length(ModelingToolkitBase.parameters(M.sys_mutated))
InformationGeometry.pdim(P::PEtabODEProblem) = InformationGeometry.pdim(P.model_info.model)


import InformationGeometry: AbstractPEtabBasedConditionGrid
## Wrap DataModel or ConditionGrid together with its contituent PEtabODEProblem for later reference or modification
struct PEtabConditionGrid <: AbstractPEtabBasedConditionGrid
    DM::AbstractDataModel
    P::PEtabODEProblem
    PEtabConditionGrid(P::PEtabModel; kwargs...) = PEtabConditionGrid(PEtabODEProblem(P); kwargs...) 
    PEtabConditionGrid(P::PEtabODEProblem; kwargs...) = PEtabConditionGrid(ConditionGrid(P; kwargs...), P)
    PEtabConditionGrid(DM::AbstractDataModel, P::PEtabODEProblem) = new(DM, P)
end
const PEtabDataModel = PEtabConditionGrid

for F in [:pdim, :DOF, :getindex]
    @eval InformationGeometry.$F(P::AbstractPEtabBasedConditionGrid, args...; kwargs...) = InformationGeometry.$F(P.DM, args...; kwargs...)
end
# Only single arg
for F in [:MLE, :LogLikeMLE, :pdim, :DOF, :name, :LogPrior, :HasPrior, :Domain, :InDomain, :loglikelihood, :Score, :FisherMetric, :AutoMetric, :CostHessian,
        :Data, :Conditions, :ConditionNames, :Trafos, :DataspaceDim, :Predictor, :dPredictor, :pnames, :Pnames,
        :xdata, :ydata, :dims, :Npoints, :xdim, :ydim, 
        :logdetInvCov, :WoundX, :WoundY, :WoundYmasked, :WoundInvCov, :HasEstimatedUncertainties,
        :xnames, :ynames, :Xnames, :Ynames, :xdist, :ydist, :dist, :HasXerror, :HasMissingValues,
        :ReconstructDataMatrices, :SplitErrorParams, :SkipXs, :GetOnlyModelParams, :GetDomain, :GetInDomain,
        :length, :size, :firstindex, :lastindex, :keys, :values, :getindex]
    @eval InformationGeometry.$F(P::AbstractPEtabBasedConditionGrid; kwargs...) = InformationGeometry.$F(P.DM; kwargs...)
end

InformationGeometry.GetConstraintFunc(CG::AbstractPEtabBasedConditionGrid, startp::AbstractVector{<:Number}=Float64[]; kwargs...) = (nothing, nothing, nothing)

# Pass remake to DataModel / ConditionGrid part
SciMLBase.remake(PDM::PEtabConditionGrid; P::PEtabODEProblem=PDM.P, kwargs...) = PEtabConditionGrid(remake(PDM.DM; kwargs...), P)

Base.summary(P::PEtabConditionGrid) = Base.summary(P.DM)
# Multi-line display when used on its own in REPL
Base.show(io::IO, m::MIME"text/plain", P::PEtabConditionGrid) = Base.show(io, m, P.DM)
# Single line display
Base.show(io::IO, P::PEtabConditionGrid) = Base.show(io, P.DM)


for F in [:plot, :plot!]
    @eval RecipesBase.$F(CG::AbstractPEtabBasedConditionGrid, mle::AbstractVector{<:Number}=MLE(CG), args...; kwargs...) = RecipesBase.$F(CG.DM, mle, args...; kwargs...)
end

## Different order than GetUniqueConditions
GetAllSimulationConditions(petab_prob::PEtabODEProblem) = petab_prob.model_info.simulation_info.conditionids[:simulation]


GetUniqueConditions(M::PEtabModel; CondID=:simulationConditionId) = Symbol.(M.petab_tables[:measurements][!, CondID]) |> unique
# GetAllUniqueObservables(M::PEtabModel; ObsID=:observableId) = Symbol.(M.petab_tables[:measurements][!, ObsID]) |> unique

function GetObservablesInCondition(M::PEtabModel, C::Symbol; ObsID=:observableId, CondID=:simulationConditionId)
    sdf = copy(M.petab_tables[:measurements]);    sdf[!, ObsID] .= Symbol.(sdf[!, ObsID]);  sdf[!, CondID] .= Symbol.(sdf[!, CondID])
    unique(sdf[!, ObsID][sdf[!, CondID] .=== C])
end
GetObservablesInConditionDict(M::PEtabModel; ObsID=:observableId, CondID=:simulationConditionId) = Dict{Symbol,Vector{Symbol}}([C => GetObservablesInCondition(M, C; ObsID, CondID) for C in GetUniqueConditions(M; CondID)])


tryfloat(x::Number) = float(x)
tryfloat(x) = x

using InformationGeometry: MissingToNan

function Long2WidePEtabMeasurements(M::PEtabModel; ObsID=:observableId, CondID=:simulationConditionId, ObsidsInCondDict::Dict{Symbol,<:AbstractVector{Symbol}}=GetObservablesInConditionDict(M; ObsID, CondID), kwargs...)
    sdf = CreateSymbolDF(M)
    Dict([k => Long2WidePEtabMeasurements(sdf, k; UniqueObsids=ObsidsInCondDict[k], ObsID, CondID, kwargs...) for k in keys(ObsidsInCondDict)])
end
function Long2WidePEtabMeasurements(sdf::AbstractDataFrame, CondName::Symbol; ObsID=:observableId, Time=:time, Meas=:measurement, CondID=:simulationConditionId, 
                        UniqueObsids::AbstractVector{<:Symbol}=Symbol[])
    measurements = @view sdf[sdf[!, CondID] .=== CondName, :]
    @assert all(∈(unique(measurements[!,ObsID])), UniqueObsids) "Need to provide correct $(ObsID)s for given $CondID."
    gdf = [select(measurements[measurements[!,ObsID] .=== ID,:], [Time, Meas]) for ID in UniqueObsids]
    sort(reduce((args...; kwargs...)->rightjoin(args...; on=Time, kwargs...), [DataFrame(float.(MissingToNan.(Matrix(df))), [Time, UniqueObsids[i]]) for (i,df) in enumerate(gdf)]), Time)
end


function Long2WidePEtabMeasurementsWithErrors(M::PEtabModel; ObsID=:observableId, CondID=:simulationConditionId, ObsidsInCondDict::Dict{Symbol,<:AbstractVector{Symbol}}=GetObservablesInConditionDict(M; ObsID, CondID), kwargs...)
    sdf = CreateSymbolDF(M)
    Dict([k => Long2WidePEtabMeasurementsWithErrors(sdf, k; UniqueObsids=ObsidsInCondDict[k], ObsID, CondID, kwargs...) for k in keys(ObsidsInCondDict)])
end
function Long2WidePEtabMeasurementsWithErrors(sdf::AbstractDataFrame, CondName::Symbol; ObsID=:observableId, Time=:time, Meas=:measurement, CondID=:simulationConditionId, NoiseParam=:noiseParameters, 
                        UniqueObsids::AbstractVector{<:Symbol}=Symbol[])
    measurements = @view sdf[sdf[!, CondID] .=== CondName, :]
    @assert all(∈(unique(measurements[!,ObsID])), UniqueObsids) "Need to provide correct $(ObsID)s for given $CondID."
    gdf = [select(measurements[measurements[!,ObsID] .=== ID,:], [Time, Meas, NoiseParam]) for ID in UniqueObsids]
    sort(reduce((args...; kwargs...)->rightjoin(args...; on=Time, kwargs...), [DataFrame([vec(col) for col in eachcol(df)], [Time, UniqueObsids[i], Symbol("sd_"*string(UniqueObsids[i]))]) for (i,df) in enumerate(gdf)]), Time)
end

import PEtab: PEtabODEProblemInfo, ModelInfo
## Change via Pull Request:
# import PEtab: _get_nllh, _get_grad, _get_hess
# const GetNllh = PEtab._get_nllh
# const GetNllhGrads = PEtab._get_grad
# const GetNllhHesses = PEtab._get_hess

#### Debugging:
import InformationGeometry: GetNllh, GetNllhGrads, GetNllhHesses, GetFixedDataUncertainty, GetConditionData, GetDataSets, GetModelFunction
import InformationGeometry: SplitParamsIntoCategories, DataSet

# Pass model from PEtabODEProblem
for F in [:GetUniqueConditions, :GetAllUniqueObservables, :GetObservablesInCondition, :GetObservablesInConditionDict, :GetDataSets, :CreateSymbolDF, 
            :Long2WidePEtabMeasurements, :Long2WidePEtabMeasurementsWithErrors]
    @eval $F(P::PEtabODEProblem, args...; kwargs...) = $F(P.model_info.model, args...; kwargs...)
end

include(joinpath(@__DIR__, "PEtabFix/ModifiedMethods.jl"))

GetNllh(M::PEtabODEProblem, args...; kwargs...) = GetNllh(M.probinfo, M.model_info, args...; kwargs...)

GetNllhGrads(M::PEtabODEProblem, args...; gradient_method=M.probinfo.gradient_method, kwargs...) = GetNllhGrads(Val(gradient_method), M.probinfo, M.model_info, PEtab._get_prior(M.model_info)[2], args...; kwargs...)
GetNllhHesses(M::PEtabODEProblem, args...; kwargs...) = GetNllhHesses(M.probinfo, M.model_info, PEtab._get_prior(M.model_info)[3], args...; kwargs...)



SplitParamsIntoCategories(P::PEtabODEProblem, args...; kwargs...) = SplitParamsIntoCategories(P.model_info, args...; kwargs...)
SplitParamsIntoCategories(model_info::PEtab.ModelInfo, args...; kwargs...) = SplitParamsIntoCategories(model_info.xindices, args...; kwargs...)
function SplitParamsIntoCategories(xindices::PEtab.ParameterIndices)
    LogDict = xindices.xscale
    Merger(S::Symbol) = (X=LogDict[S];  X === :lin ? S : Symbol(string(X)*"_"*string(S)))
    (Merger.(xindices.xids[:dynamic]), Merger.(xindices.xids[:noise]), Merger.(xindices.xids[:nondynamic]), Merger.(xindices.xids[:observable]))
end
GetDynamicParams(args...; kwargs...) = SplitParamsIntoCategories(args...; kwargs...)[1]
GetErrorParams(args...; kwargs...) = SplitParamsIntoCategories(args...; kwargs...)[2]
GetNondynamicParams(args...; kwargs...) = SplitParamsIntoCategories(args...; kwargs...)[3]
GetObservableParams(args...; kwargs...) = SplitParamsIntoCategories(args...; kwargs...)[4]

GetDynamicParamInds(X::PEtabODEProblem) = InformationGeometry.GetNamesSymb(get_x(X)) .∈ Ref(SplitParamsIntoCategories(X)[1])
GetErrorParamInds(X::PEtabODEProblem) = InformationGeometry.GetNamesSymb(get_x(X)) .∈ Ref(SplitParamsIntoCategories(X)[2])
GetNondynamicParamInds(X::PEtabODEProblem) = InformationGeometry.GetNamesSymb(get_x(X)) .∈ Ref(SplitParamsIntoCategories(X)[3])
GetObservableParamInds(X::PEtabODEProblem) = InformationGeometry.GetNamesSymb(get_x(X)) .∈ Ref(SplitParamsIntoCategories(X)[4])


HasErrorModel(P::PEtabODEProblem, args...; kwargs...) = HasErrorModel(P.model_info.model, args...; kwargs...)
function HasErrorModel(P::PEtabModel, CondName::Symbol; CondID=:simulationConditionId, NoiseParam=:noiseParameters)
    df = @view P.petab_tables[:measurements][Symbol.(P.petab_tables[:measurements][!,CondID]) .=== CondName, :]
    @assert !isempty(df) "Condition Name wrong? Got $CondName."
    IsFloat(x::Number) = true;  IsFloat(x) = Meta.parse(x) isa Number
    !all(IsFloat, df[!, NoiseParam])
end


# for F in [:GetNllh]
#     @eval $F(M::PEtabODEProblem, args...; kwargs...) = $F(M.probinfo, M.model_info, args...; kwargs...)
# end


# GetNllhGrads(M::PEtabODEProblem, args...; gradient_method=M.probinfo.gradient_method, kwargs...) = GetNllhGrads(Val(gradient_method), M.probinfo, M.model_info, PEtab._get_prior(M.model_info)[2], args...; kwargs...)
# GetNllhHesses(M::PEtabODEProblem, args...; kwargs...) = GetNllhHesses(M.probinfo, M.model_info, PEtab._get_prior(M.model_info)[3], args...; kwargs...)


import InformationGeometry: DataModel, DataSet, CompositeDataSet, ModelMap, ConditionGrid, Negate, Negate!!, NegateBoth, ydim, MergeOneArgMethods
import InformationGeometry: Identity2Splitter

# Get fixed uncertainties based on error parameter values from P
function GetFixedDataUncertainty(P::PEtabModel, observablesDF::AbstractDataFrame=P.petab_tables[:observables], ObsNames::AbstractVector{<:Symbol}=Symbol.(observablesDF[!,:observableId]); Mle::AbstractVector=Float64[],
                        FixedError::Bool=true, ObsID=:observableId, CondID=:simulationConditionId, Formula=:noiseFormula, ObsTrafo=:observableTransformation, NoiseDist=:noiseDistribution, Pscale=:parameterScale, ParamID=:parameterId, debug::Bool=false, kwargs...)
    Odf = @view (observablesDF[[findfirst(isequal(O),Symbol.(observablesDF[!,ObsID])) for O in ObsNames], :])
    # @assert all(Symbol.(Odf[!,ObsTrafo]) .=== :lin)
    @assert all(Symbol.(Odf[!,NoiseDist]) .=== :normal)
    ParsedError = Tuple(Meta.parse.(X) for X in Odf[!, Formula])
    all(x->isa(x,Number), ParsedError) && return collect(ParsedError)
    function FindLineInParamTable(Symb::Symbol)
        S = deepcopy(Symb)
        i = findfirst(isequal(S), Symbol.(P.petab_tables[:parameters][!,ParamID]))
        if isnothing(i) # Try exchanging "noiseParameter1_" prefix for "sd_"
            s = string(Symb)
            S = startswith(s, "noiseParameter") ? Symbol("sd"*s[findfirst('_', s):length(s)]) : Symb
            i = findfirst(isequal(S), Symbol.(P.petab_tables[:parameters][!,ParamID]))
        end
        if isnothing(i) #debugging
            @warn "Parameter $s (or alternatively $S) not found in parameters table"
            if debug
                println(P.petab_tables[:parameters])
                println(observablesDF)
            end
        end;    (i, S)
    end
    GetFullParameterName(x::Number) = x;    GetFullParameterName(E::Expr) = E
    function GetFullParameterName(Symb::Symbol)
        i, S = FindLineInParamTable(Symb)
        Symbol(P.petab_tables[:parameters][i,Pscale]) === :lin ? S : Symbol(string(Symbol(P.petab_tables[:parameters][i,Pscale]))*"_"*string(S))
    end
    GetInverseTrafo(x) = (@warn "GetInverseTrafo: Got $x, trying to continue.";  x)
    GetInverseTrafo(x::Number) = identity
    function GetInverseTrafo(Symb::Symbol)
        i, S = FindLineInParamTable(Symb)
        Traf = Symbol(P.petab_tables[:parameters][i,Pscale])
        Traf === :lin && return identity
        Traf === :log10 && return exp10
        Traf === :log || Traf === :ln && return exp
        throw("Do not know how to invert Trafo $Traf yet.")
    end
    GetInds(x::Symbol) = [findfirst(isequal(x), InformationGeometry.GetNamesSymb(Mle))];    GetInds(x::Number) = x
    GetInds(x) = (@warn "GetInds: Got $x, trying to continue.";  x)
    MakeConstError(X::AbstractVector{<:Int}, S::Symbol) = (R=Mle[X];   length(R) == 1 ? R[1] : R);  MakeConstError(x::Number, S) = x
    NameOrValue = GetFullParameterName.(ParsedError)
    IndOrValue = GetInds.(NameOrValue)
    ## For debugging
    # return ParsedError, NameOrValue, IndOrValue, MakeConstError.(IndOrValue, ParsedError), GetInverseTrafo.(ParsedError)
    FixedError && !all(isa.(IndOrValue,Number)) && @warn "Approximating data uncertainties as fixed in dataset object although the error parameters are estimated for $(ObsNames[collect(.!isa.(IndOrValue,Number))])."
    FixedError && return [Transform(Value) for (Transform, Value) in Iterators.zip(GetInverseTrafo.(ParsedError), MakeConstError.(IndOrValue, ParsedError))]
end


function GetConditionData(P::PEtabODEProblem, M::PEtabModel=P.model_info.model, sdf::AbstractDataFrame=CreateSymbolDF(M), CondName::Symbol=sdf[!,:simulationConditionId][1]; Time=:time, ObsID=:observableId, CondID=:simulationConditionId, NoiseParam=:noiseParameters, 
                        ObsidsInCondDict::Dict{Symbol,<:AbstractVector{Symbol}}=GetObservablesInConditionDict(M; ObsID, CondID),
                        FixedError::Bool=true, verbose::Bool=false, debug::Bool=false, Mle=MLE(P))
    cdf = sdf[sdf[!, CondID] .=== CondName, :]
    verbose && @info "Starting Condition $CondName."
    # df = Long2WidePEtabMeasurementsWithErrors(cdf; UniqueObsids=GetObservablesInCondition(M, CondName; ObsID, CondID))
    df = Long2WidePEtabMeasurementsWithErrors(sdf, CondName; UniqueObsids=ObsidsInCondDict[CondName], Time, CondID)
    Xdf = MissingToNan.(df[!,[Time]]);    YdfE = broadcast(x->ismissing(x) ? NaN : x, df[!,Not(Time)])
    Ydf = broadcast(x->ismissing(x) ? NaN : x, YdfE[!,map(!startswith("sd_"), names(YdfE))])
    Sdf = if HasErrorModel(M, CondName; CondID, NoiseParam)
        if FixedError
            # Approximate error model as constant from best fit
            ConstError = try
                GetFixedDataUncertainty(M, M.petab_tables[:observables], Symbol.(names(Ydf)); FixedError, debug, Mle)
            catch;
                debug && @warn "Tried getting fixed errors but could not because of $E. Continuing with uncertainties=1."
                ones(length(names(Ydf)))
            end
            if length(ConstError) == length(names(Ydf)) || eltype(ConstError) <: Number
                DataFrame(reduce(hcat, [fill(x,size(Ydf,1)) for x in ConstError]), [Symbol("sd_"*yn) for yn in string.(names(Ydf))])
            else
                throw("Do not know how to handle case $ConstError yet.")
            end
        else
            # Reuse already constructed modelmap with kwarg Error=true to avoid double compilation?
            σErrorModel = GetModelFunction(P; cid=CondName, Error=true)
            # is σ^(-1)
            yinverrormod(x::AbstractVector,y,p) = inv.(σErrorModel(x,p))
            yinverrormod(x::Number,y,p) = yinverrormod([x], y, p)
        end
    else
        MissingToNan.(YdfE[!,map(startswith("sd_"), names(YdfE))])
    end
    if Sdf isa AbstractDataFrame
        (any(ismissing, eachrow(df)) ? DataSet : CompositeDataSet)(Xdf, Ydf, Sdf; xnames=names(Xdf), ynames=names(Ydf), name=CondName)
    else
        if !any(ismissing, eachrow(df))
            ## Need identity splitter
            DataSetUncertain(Xdf, Ydf, Sdf, Identity2Splitter, Mle; xnames=names(Xdf), ynames=names(Ydf), name=CondName)
        else
            @warn "Throwing away any rows with missing or NaN currently"
            keep = map(x->all(isfinite∘MissingToNan, x), eachrow(df))
            bigkeep = reduce(vcat, [repeat(n, size(Ydf,2)) for n in keep])
            ## Need to modify error model for missings...

            throw("Not programmed for reading error models with missing values yet.")
            DataSetUncertain(Xdf[keep], Ydf[keep,:], (x,y,p)->Sdf(x,y,p)[bigkeep], Identity2Splitter, Mle; xnames=names(Xdf), ynames=names(Ydf), name=CondName)
            ### Try version with throwing away all rows containing any missing value
        end
    end
end

CreateSymbolDF(M::PEtabModel; ObsID=:observableId, CondID=:simulationConditionId) = (sdf = copy(M.petab_tables[:measurements]);    sdf[!, ObsID] .= Symbol.(sdf[!, ObsID]);  sdf[!, CondID] .= Symbol.(sdf[!, CondID]);     sdf)
# Get vector of all condition datasets
function GetDataSets(P::PEtabODEProblem, M::PEtabModel=P.model_info.model; ObsID=:observableId, CondID=:simulationConditionId, ObsidsInCondDict::Dict{Symbol,<:AbstractVector{Symbol}}=GetObservablesInConditionDict(M; ObsID, CondID), UniqueConds=collect(keys(ObsidsInCondDict)), FixedError::Bool=true, Mle=MLE(P), verbose::Bool=false)
    sdf = CreateSymbolDF(M; ObsID, CondID)
    [GetConditionData(P, M, sdf, C; ObsID, CondID, ObsidsInCondDict, FixedError, Mle, verbose) for C in UniqueConds]
end

import InformationGeometry: StringOrSymb, NicifyPEtabNames
# Still need to broadcast apply latexstring, apply EscapeStr before
NicifyPEtabNames(PNames::AbstractVector{Symbol}; kwargs...) = NicifyPEtabNames(string.(PNames); kwargs...)
NicifyPEtabNames(PNames::StringOrSymb; kwargs...) = NicifyPEtabNames([string(PNames)]; kwargs...)[1]
function NicifyPEtabNames(PNames::AbstractVector{<:AbstractString}; Textifier::AbstractString="text")
    # Don't look for underscore after log, may already be escaped
    IsLog10 = [startswith(x, "log10") for x in PNames]
    IsLog = [!startswith(x, "log10") && startswith(x, "log") for x in PNames]
    Clean = [IsLog10[i] || IsLog[i] ? PNames[i][findfirst('_',PNames[i])+1:end] : PNames[i] for i in eachindex(PNames)]
    length(Textifier) > 0 && (Clean = "\\"*Textifier*"{" .* Clean .* "}")
    [(IsLog[i] ? "\\log(" : "")* (IsLog10[i] ? "\\log_{10}(" : "") * Clean[i] *(IsLog10[i] || IsLog[i] ? ")" : "") for i in eachindex(PNames)]
end


InformationGeometry.DataSet(M::PEtabModel, C::Symbol=Symbol(M.petab_tables[:conditions][1,1]); ObsID=:observableId, CondID=:simulationConditionId, FixedError::Bool=true) = GetConditionData(M, CreateSymbolDF(M; ObsID, CondID), C; ObsID, CondID, FixedError)


InformationGeometry.ConditionGrid(P::PEtabModel; kwargs...) = InformationGeometry.ConditionGrid(PEtabODEProblem(P); kwargs...)
function InformationGeometry.ConditionGrid(P::PEtabODEProblem, Mle::AbstractVector=MLE(P); ObsID=:observableId, CondID=:simulationConditionId, ADmode::Val=Val(:FiniteDifferences), SkipOptim::Bool=true, FixedError::Bool=true, verbose::Bool=false, SortConditions::Bool=false, kwargs...)

    ObsidsInCondDict = GetObservablesInConditionDict(P.model_info.model; ObsID, CondID)
    UniqueConds = GetUniqueConditions(P; CondID) # keep original order
    SortConditions && sort!(UniqueConds)
    DSs = GetDataSets(P; ObsID, CondID, ObsidsInCondDict, UniqueConds, FixedError, Mle, verbose)

    PNames = InformationGeometry.GetNamesSymb(Mle) # .|> string |> NicifyPEtabNames
    NewModel = ModelMap(GetModelFunction(P; cid=UniqueConds[1], ObsidsInCondDict), HyperCube(P), (1, ydim(DSs[1]), length(Mle)); startp=Mle, pnames=PNames, inplace=false, IsCustom=true)
    
    @assert Mle ∈ NewModel.Domain

    LogLikelihoodFn = Negate(P.nllh)
    ScoreFn = MergeOneArgMethods(Negate(P.grad), Negate!!(P.grad!));    FisherInfoFn = MergeOneArgMethods(P.hess, P.hess!)
    # Check if there is a prior or nothing
    LogPriorFn = P.prior
    if length(UniqueConds) == 1
        DataModel(DSs[1], NewModel, convert(Vector,Mle), LogPriorFn; LogLikelihoodFn, ScoreFn, FisherInfoFn, 
                    ADmode, name=Symbol(P.model_info.model.name), SkipOptim, verbose, kwargs...) |> x->PEtabConditionGrid(x,P)
    else
        # NewModels = [remake(NewModel; Map=GetModelFunction(P; cid=C), xyp=(1, ydim(DSs[i]), length(Mle))) for (i,C) in enumerate(UniqueConds)]
        # Give Prior to ConditionGrid, not individual DMs
        DMs = [DataModel(DSs[i], remake(NewModel; Map=GetModelFunction(P; cid=C, ObsidsInCondDict), xyp=(1, ydim(DSs[i]), length(Mle))), convert(Vector,Mle), nothing; 
                            ADmode, LogLikelihoodFn=Negate(GetNllh(P; cids=[C])),
                            ScoreFn=((Sc!,Sc)=GetNllhGrads(P; cids=[C]);  MergeOneArgMethods(Negate(Sc), Negate!!(Sc!))),
                            FisherInfoFn=((Fi!,Fi)=GetNllhHesses(P; cids=[C]);   MergeOneArgMethods(Fi, Fi!)), # Set FIM = true later!
                            name=C, SkipTests=true, SkipOptim=true, verbose) for (i,C) in enumerate(UniqueConds)]
        InformationGeometry.ConditionGrid(DMs, [identity for i in eachindex(UniqueConds)], LogPriorFn, convert(Vector,Mle); 
                    ADmode, Domain=HyperCube(P), pnames=PNames, name=Symbol(P.model_info.model.name), LogLikelihoodFn, ScoreFn, FisherInfoFn, SkipOptim, verbose, kwargs...) |> x->PEtabConditionGrid(x,P)
    end
end

# Currently overload with identical behaviour
InformationGeometry.DataModel(P::PEtabODEProblem, Mle::AbstractVector=MLE(P); kwargs...) = InformationGeometry.ConditionGrid(P, Mle; kwargs...)

"""
    SameStructContents(A::T, B::T; verbose::Bool=false) where T
Compare all fields of an arbitrary struct individually.
"""
function SameStructContents(A::T, B::T; verbose::Bool=false) where T
    Names = fieldnames(T)
    for Name in Names
        if !isequal(getproperty(A, Name), getproperty(B,Name))
            verbose && println("Test failed due to $Name: $(getproperty(A, Name)) != $(getproperty(B, Name))")
            return false
        end
    end;    true
end

Base.isequal(A::PEtab.ObservableNoiseMap, B::PEtab.ObservableNoiseMap) = SameStructContents(A, B)
import Base.==
==(A::PEtab.ObservableNoiseMap, B::PEtab.ObservableNoiseMap) = SameStructContents(A, B)

"""
    uniqueindices(X::AbstractVector)
Returns indices of unique elements of `X`, i.e. `X[uniqueindices(X)] == unique(X)`.
"""
uniqueindices(X::AbstractVector) = unique(i -> X[i], eachindex(X))


function GetModelFunction(petab_prob::PEtabODEProblem; cid::Symbol=Symbol(petab_prob.model_info.model.petab_tables[:conditions][1,1]), ObsID=:observableId, CondID=:simulationConditionId, Error::Bool=false, verbose::Bool=true,
                ObsidsInCondDict::Dict{Symbol,<:AbstractVector{Symbol}}=GetObservablesInConditionDict(petab_prob.model_info.model; ObsID, CondID))
    @unpack model_info, probinfo = petab_prob
    @unpack conditionids = petab_prob.model_info.simulation_info

    ## conditionids[:experiment] always contains concatenation of pre_equilibration_id and simulation_id, unless pre_equilibration is :None
    ## If no pre_equilibration, then experiment_id == simulation_id
    ## conditionids[:Simulation]
    all_pre_equilibrations = conditionids[:pre_equilibration]
    all_simulations = conditionids[:simulation]

    # Map pure simulationConditionId to merged pre_equilibration_id * simulationConditionId, which appears in simulation_info.imeasurements
    PreEquilMergedCidDict = Dict([pre_equil === :None ? simu => simu : simu => Symbol(string(pre_equil)*string(simu)) for (pre_equil, simu) in zip(all_pre_equilibrations,all_simulations)])

    ####### Check if observable maps and noise maps consistent within condition, so that values can be interpolated between measurements
    ## Dict for unique index of each measurement in given condition
    imeasurementsCid = model_info.simulation_info.imeasurements[PreEquilMergedCidDict[cid]]
    observable_idCid = model_info.petab_measurements.observable_id[imeasurementsCid]
    UniqueObsInds = uniqueindices(observable_idCid)
    ## Representative inds to take observable and noise maps from for interpolation, given obsid
    ObsidToRepresentativeMeasurementIndDict = Dict(observable_idCid[UniqueObsInds] .=> UniqueObsInds)

    ### Boolvectors of equal indices
    ObservableInds = [map(isequal(Observable), observable_idCid) for Observable in observable_idCid[UniqueObsInds]]

    ## Check that all observable maps are consistent per observable
    CanInterpolateObservables = all(allequal(i->petab_prob.model_info.xindices.xobservable_maps[i], imeasurementsCid[ObservableInds[j]]) for j in eachindex(ObservableInds))
    # [map(i->petab_prob.model_info.xindices.xobservable_maps[i], @view imeasurementsCid[ObservableInds[j]]) for j in eachindex(ObservableInds)]

    ## Check that all noise maps are consistent per observable
    CanInterpolateUncertainties = all(allequal(i->petab_prob.model_info.xindices.xnoise_maps[i], imeasurementsCid[ObservableInds[j]]) for j in eachindex(ObservableInds))
    # [map(i->petab_prob.model_info.xindices.xnoise_maps[i], @view imeasurementsCid[ObservableInds[j]]) for j in eachindex(ObservableInds)]
    
    UniqueObsids = ObsidsInCondDict[cid]

    GetPredictions(x::Number, θ::AbstractVector; kwargs...) = GetPredictions([x], θ; kwargs...)
    GetPredictions(x::AbstractVector{<:Int}, θ::AbstractVector; kwargs...) = GetPredictions(float.(x), θ; kwargs...)
    function GetPredictions(ts::AbstractVector{<:AbstractFloat}, θ::AbstractVector{T}; Error::Bool=Error)::Vector{T} where T<:Number
        # odesols = PEtab.solve_all_conditions(collect(θ), petab_prob, petab_prob.probinfo.solver.solver)
        # sol = odesols[cid]
        Res = Matrix{T}(undef, length(UniqueObsids), length(ts))
        Xnom = collect(petab_prob.xnominal)
        x = collect(θ)

        # Has pre_equilibration or not
        ind = findfirst(isequal(cid), all_simulations)
        sol = PEtab.get_odesol(x, petab_prob; condition = (all_pre_equilibrations[ind] === :None ? cid : (all_pre_equilibrations[ind] => cid)))

        xdynamic, xobservable, xnoise, xnondynamic = PEtab.split_x(x, model_info.xindices)
        xnondynamic_ps = PEtab.transform_x(xnondynamic, model_info.xindices,
                                        :xnondynamic, probinfo.cache)
        if !Error
            verbose && !CanInterpolateObservables && @warn "Cannot interpolate observables!"
            xobservable_ps = PEtab.transform_x(xobservable, model_info.xindices,
                                            :xobservable, probinfo.cache)
            for (i,obsid) in enumerate(UniqueObsids)
                imeasurement = ObsidToRepresentativeMeasurementIndDict[obsid]
                mapxobservables = model_info.xindices.xobservable_maps[imeasurement]
                for (j,t) in enumerate(ts)
                    Res[i,j] = PEtab._h(sol(t), t, sol.prob.p, xobservable_ps, xnondynamic_ps,
                            model_info.model, mapxobservables, obsid, Xnom)
                end
            end
        else
            verbose && !CanInterpolateUncertainties && @warn "Cannot interpolate uncertainties!"
            xnoise_ps = PEtab.transform_x(xnoise, model_info.xindices,
                                            :xnoise, probinfo.cache)
            for (i,obsid) in enumerate(UniqueObsids)
                imeasurement = ObsidToRepresentativeMeasurementIndDict[obsid]
                mapxnoise = model_info.xindices.xnoise_maps[imeasurement]
                for (j,t) in enumerate(ts)
                    Res[i,j] = PEtab._sd(sol(t), t, sol.prob.p, xnoise_ps, xnondynamic_ps, 
                            model_info.model, mapxnoise, obsid, Xnom)
                end
            end
        end;    vec(Res)
    end
end


end # module