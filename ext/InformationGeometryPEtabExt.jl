module InformationGeometryPEtabExt


using InformationGeometry, PEtab, DataFrames, ModelingToolkit, ForwardDiff

InformationGeometry.MLE(P::PEtabODEProblem) = PEtab.get_x(P)
InformationGeometry.HyperCube(P::PEtabODEProblem) = HyperCube(P.lower_bounds, P.upper_bounds)
InformationGeometry.MLE(R::PEtabOptimisationResult) = R.xmin


GetUniqueConditions(M::PEtabModel; CondID=:simulationConditionId) = Symbol.(M.petab_tables[:measurements][!, CondID]) |> unique
GetAllUniqueObservables(M::PEtabModel; ObsID=:observableId) = Symbol.(M.petab_tables[:measurements][!, ObsID]) |> unique

function GetObservablesInCondition(M::PEtabModel, C::Symbol; ObsID=:observableId, CondID=:simulationConditionId)
    sdf = copy(M.petab_tables[:measurements]);    sdf[!, ObsID] .= Symbol.(sdf[!, ObsID]);  sdf[!, CondID] .= Symbol.(sdf[!, CondID])
    unique(sdf[!, ObsID][sdf[!, CondID] .=== C])
end
GetObservablesInConditionDict(M::PEtabModel; ObsID=:observableId, CondID=:simulationConditionId) = Dict{Symbol,Vector{Symbol}}([C => GetObservablesInCondition(M, C; ObsID, CondID) for C in GetUniqueConditions(M; CondID)])


# Pass model from PEtabODEProblem
for F in [:GetUniqueConditions, :GetAllUniqueObservables, :GetObservablesInCondition, :GetObservablesInConditionDict, :GetDataSets, :CreateSymbolDF, Symbol("InformationGeometry.DataSet")]
    @eval $F(P::PEtabODEProblem, args...; kwargs...) = $F(P.model_info.model, args...; kwargs...)
end

tryfloat(x::Number) = float(x)
tryfloat(x) = x
MissingToNan(x::Number) = x
MissingToNan(x::Missing) = NaN


function Long2WidePEtabMeasurements(measurements::AbstractDataFrame, ObsID=:observableId, Time=:time, Meas=:measurement; UniqueObsids::AbstractVector{<:Symbol}=unique(measurements[!, ObsID]))
    gdf = [select(measurements[measurements[!,ObsID] .=== ID,:], [Time, Meas]) for ID in UniqueObsids]
    sort(reduce((args...; kwargs...)->rightjoin(args...; on=Time, kwargs...), [DataFrame(float.(Matrix(df)), [Time, UniqueObsids[i]]) for (i,df) in enumerate(gdf)]), Time)
end


import PEtab: PEtabODEProblemInfo, ModelInfo
## Change via Pull Request:
# import PEtab: _get_nllh, _get_grad, _get_hess
# GetNllh = _get_nllh
# GetNllhGrads = _get_grad
# GetNllhHesses = _get_hess
# const GetNllh = PEtab._get_nllh
# const GetNllhGrads = PEtab._get_grad
# const GetNllhHesses = PEtab._get_hess

include(joinpath(@__DIR__, "PEtabFix/ModifiedMethods.jl"))

GetNllh(M::PEtabODEProblem, args...; kwargs...) = GetNllh(M.probinfo, M.model_info, args...; kwargs...)

GetNllhGrads(M::PEtabODEProblem, args...; gradient_method=M.probinfo.gradient_method, kwargs...) = GetNllhGrads(Val(gradient_method), M.probinfo, M.model_info, PEtab._get_prior(M.model_info)[2], args...; kwargs...)
GetNllhHesses(M::PEtabODEProblem, args...; kwargs...) = GetNllhHesses(M.probinfo, M.model_info, PEtab._get_prior(M.model_info)[3], args...; kwargs...)



# for F in [:GetNllh]
#     @eval $F(M::PEtabODEProblem, args...; kwargs...) = $F(M.probinfo, M.model_info, args...; kwargs...)
# end


# GetNllhGrads(M::PEtabODEProblem, args...; gradient_method=M.probinfo.gradient_method, kwargs...) = GetNllhGrads(Val(gradient_method), M.probinfo, M.model_info, PEtab._get_prior(M.model_info)[2], args...; kwargs...)
# GetNllhHesses(M::PEtabODEProblem, args...; kwargs...) = GetNllhHesses(M.probinfo, M.model_info, PEtab._get_prior(M.model_info)[3], args...; kwargs...)


import InformationGeometry: DataModel, DataSet, CompositeDataSet, ModelMap, ConditionGrid, Negate, Negate!!, NegateBoth, ydim, MergeOneArgMethods

# Get fixed uncertainties based on error parameter values from P
function GetDataUncertainty(P::PEtabModel, observablesDF::AbstractDataFrame=P.petab_tables[:observables], ObsNames::AbstractVector{<:Symbol}=Symbol.(observablesDF[!,:observableId]); Mle::AbstractVector=Float64[],
                        FixedError::Bool=true, ObsID=:observableId, CondID=:simulationConditionId, Formula=:noiseFormula, ObsTrafo=:observableTransformation, NoiseDist=:noiseDistribution, Pscale=:parameterScale, ParamID=:parameterId, debug::Bool=true, kwargs...)
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
    MakeConstError(X::AbstractVector{<:Int}, S::Symbol) = (R=Mle[X];   length(R) == 1 ? R[1] : R);  MakeConstError(x::Number, S) = x
    NameOrValue = GetFullParameterName.(ParsedError)
    IndOrValue = GetInds.(NameOrValue)
    ## For debugging
    # return ParsedError, NameOrValue, IndOrValue, MakeConstError.(IndOrValue, ParsedError), GetInverseTrafo.(ParsedError)
    FixedError && !all(isa.(IndOrValue,Number)) && @warn "Approximating data uncertainties as fixed although the error parameters are estimated for $(ObsNames[collect(.!isa.(IndOrValue,Number))])."
    FixedError && return [Transform(Value) for (Transform, Value) in Iterators.zip(GetInverseTrafo.(ParsedError), MakeConstError.(IndOrValue, ParsedError))]
end

# Get all data for single condition
function GetConditionData(P::PEtabODEProblem, M::PEtabModel, sdf::AbstractDataFrame=CreateSymbolDF(M), C::Symbol=sdf[!,:simulationConditionId][1]; ObsID=:observableId, CondID=:simulationConditionId, verbose::Bool=false, Mle=MLE(P))
    cdf = sdf[sdf[!, CondID] .=== C, :]
    verbose && @info "Starting Condition $C."
    df = Long2WidePEtabMeasurements(cdf; UniqueObsids=GetObservablesInCondition(M, C; ObsID, CondID))
    Xdf, Ydf = MissingToNan.(float.(df[:,1:1])), MissingToNan.(float.(df[:,2:end]))
    Constructor = !any(ismissing.(Matrix(df))) ? DataSet : CompositeDataSet

    try 
        ConstError = GetDataUncertainty(M, M.petab_tables[:observables], Symbol.(names(Ydf)); Mle)
        SigmaDf = if length(ConstError) == length(names(Ydf)) || eltype(ConstError) <: Number
            DataFrame(reduce(hcat, [fill(x,size(Ydf,1)) for x in ConstError]), :auto)
        else
            throw("Do not know how to handle case $ConstError yet.")
        end
        Constructor(Xdf, Ydf, SigmaDf; xnames=names(Xdf), ynames=names(Ydf), name=C)
    catch E;
        @warn "Tried getting errors but could not because of $E."
        Constructor(Xdf, Ydf; xnames=names(Xdf), ynames=names(Ydf), name=C)
    end
end

CreateSymbolDF(M::PEtabModel; ObsID=:observableId, CondID=:simulationConditionId) = (sdf = copy(M.petab_tables[:measurements]);    sdf[!, ObsID] .= Symbol.(sdf[!, ObsID]);  sdf[!, CondID] .= Symbol.(sdf[!, CondID]);     sdf)
# Get vector of all condition datasets
function GetDataSets(P::PEtabODEProblem, M::PEtabModel=P.model_info.model; ObsID=:observableId, CondID=:simulationConditionId, UniqueConds=GetUniqueConditions(M; CondID), Mle=MLE(P))
    sdf = CreateSymbolDF(M; ObsID, CondID)
    [GetConditionData(P, M, sdf, C; ObsID, CondID, Mle) for C in UniqueConds]
end

function NicifyPEtabNames(PNames::AbstractVector{<:AbstractString})
    IsLog10 = [startswith(x, "log10_") for x in PNames]
    IsLog = [startswith(x, "log_") for x in PNames]
    Clean = [IsLog10[i] || IsLog[i] ? PNames[i][findfirst('_',PNames[i])+1:end] : PNames[i] for i in eachindex(PNames)]
    [(IsLog[i] ? "\\log(" : "")* (IsLog10[i] ? "\\log_{10}(" : "") * Clean[i] *(IsLog10[i] || IsLog[i] ? ")" : "") for i in eachindex(PNames)] .|> latexstring
end


InformationGeometry.DataSet(M::PEtabModel, C::Symbol=Symbol(M.petab_tables[:conditions][1,1]); ObsID=:observableId, CondID=:simulationConditionId) = GetConditionData(M, CreateSymbolDF(M; ObsID, CondID), C; ObsID, CondID)


InformationGeometry.DataModel(P::PEtabModel; kwargs...) = InformationGeometry.DataModel(PEtabODEProblem(P); kwargs...)
function InformationGeometry.DataModel(P::PEtabODEProblem, Mle::AbstractVector=MLE(P); ObsID=:observableId, CondID=:simulationConditionId, ADmode::Val=Val(:FiniteDifferences), SkipOptim::Bool=true, kwargs...)
    UniqueConds = GetUniqueConditions(P; CondID)
    DSs = GetDataSets(P; ObsID, CondID, UniqueConds, Mle)
    DS = DSs[1]

    PNames = InformationGeometry.GetNamesSymb(Mle) # .|> string |> NicifyPEtabNames
    NewModel = ModelMap(GetModelFunction(P), HyperCube(P), (1, ydim(DS), length(Mle)); startp=Mle, pnames=PNames, inplace=false, IsCustom=true)
    
    @assert Mle ∈ NewModel.Domain

    LogLikelihoodFn = Negate(P.nllh) # Negate(GetNllh(P; cids=[:all]))
    # Overload with in-place methods from PEtab.jl later!
    ScoreFn = MergeOneArgMethods(Negate(P.grad), Negate!!(P.grad!));    FisherInfoFn = MergeOneArgMethods(P.hess, P.hess!)
    # Check if there is a prior or nothing
    LogPriorFn = P.prior
    if length(UniqueConds) == 1
        DataModel(DS, NewModel, convert(Vector,Mle), LogPriorFn; LogLikelihoodFn, ScoreFn, FisherInfoFn, 
                ADmode, name=Symbol(P.model_info.model.name), SkipOptim, kwargs...)
    else
        # NewModels = [remake(NewModel; Map=GetModelFunction(P; cond=C), xyp=(1, ydim(DSs[i]), length(Mle))) for (i,C) in enumerate(UniqueConds)]
        # Give Prior to ConditionGrid, not individual DMs
        DMs = [DataModel(DSs[i], remake(NewModel; Map=GetModelFunction(P; cond=C), xyp=(1, ydim(DSs[i]), length(Mle))), convert(Vector,Mle), nothing; 
                            ADmode, LogLikelihoodFn=Negate(GetNllh(P; cids=[C])), name=C, SkipTests=true, SkipOptim=true) for (i,C) in enumerate(UniqueConds)]
        InformationGeometry.ConditionGrid(DMs, [identity for i in eachindex(UniqueConds)], LogPriorFn, convert(Vector,Mle); ADmode, pnames=PNames, name=Symbol(P.model_info.model.name), LogLikelihoodFn, ScoreFn, FisherInfoFn, SkipOptim, kwargs...)
    end
end

function GetModelFunction(petab_prob::PEtabODEProblem; cond::Symbol=Symbol(petab_prob.model_info.model.petab_tables[:conditions][1,1]), ObsID=:observableId, CondID=:simulationConditionId)
    @unpack model_info, probinfo = petab_prob

    # Decide all observables which should be returned here and mind their order!
    ObsidsInCondDict = GetObservablesInConditionDict(model_info.model; ObsID, CondID)
    # Make dict of unique obsids per condition and apply later to get local UniqueObsids

    function GetPredictions(xs, θ::AbstractVector; cid::Symbol=cond, Error::Bool=false)
        UniqueObsids = ObsidsInCondDict[cid]
        odesols = PEtab.solve_all_conditions(collect(θ), petab_prob, petab_prob.probinfo.solver.solver)

        Res = Matrix{eltype(xs)}(undef, length(xs), length(UniqueObsids))
        Xnom = collect(petab_prob.xnominal)
        sol = odesols[cid]
        for (i,obsid) in enumerate(UniqueObsids)
            # idata = findall(cids .=== cid .&& obsids .=== obsid)
            @unpack xobservable, xnondynamic, xnoise = probinfo.cache
            xnondynamic_ps = PEtab.transform_x(xnondynamic, model_info.xindices,
                                            :xnondynamic, probinfo.cache)
            if !Error
                xobservable_ps = PEtab.transform_x(xobservable, model_info.xindices,
                                                :xobservable, probinfo.cache)
                mapxobservables = model_info.xindices.mapxobservable
                Res[:,i] .= reduce(vcat,[PEtab._h(sol(t), t, sol.prob.p, xobservable_ps, xnondynamic_ps,
                            model_info.model.h, mapxobservables[1], Symbol(obsid), Xnom) for t in xs])
            else
                xnoise_ps = PEtab.transform_x(xnoise, model_info.xindices,
                                                :xnoise, probinfo.cache)
                mapxnoise = model_info.xindices.mapxnoise
                Res[:,i] .= reduce(vcat,[PEtab._sd(sol(t), t, sol.prob.p, xnoise_ps, xnondynamic_ps, 
                            model_info.model.sd, mapxnoise[1], Symbol(obsid), Xnom) for t in xs])
            end
        end
        # Slow!
        reduce(vcat, [view(Res,i,:) for i in axes(Res,1)])
    end
end

# using SnoopPrecompile

# SnoopPrecompile.@precompile_all_calls begin
    
# end

end # module