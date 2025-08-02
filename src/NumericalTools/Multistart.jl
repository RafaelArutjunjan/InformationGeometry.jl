

SOBOL.SobolSeq(C::HyperCube, Maxval::Real=1e15; maxval::Real=Maxval, seed::Int=rand(0:Int(1e7)), N::Int=100) = SOBOL.skip(SOBOL.SobolSeq(clamp(C.L, -maxval*ones(length(C)), maxval*ones(length(C))), clamp(C.U, -maxval*ones(length(C)), maxval*ones(length(C)))), seed; exact=true)
SobolGenerator(args...; kwargs...) = (S=SOBOL.SobolSeq(args...; kwargs...);    (SOBOL.next!(S) for i in 1:Int(1e12)))
GenerateSobolPoints(args...; N::Int=100, kwargs...) = (S=SOBOL.SobolSeq(args...; N, kwargs...);    [SOBOL.next!(S) for i in 1:N])

function SobolRejectionSampling(S::SOBOL.AbstractSobolSeq, P::Distributions.Distribution, n::Int=100; N::Int=n)
    @assert N ≥ 1
    i = 1;   y = SOBOL.next!(S);  M = Matrix{eltype(y)}(undef, length(y), N)
    Pusher(x, i::Int) = (rand() ≤ pdf(P, x)  ? (M[:,i] .= x;  true) : false)
 
    while i ≤ N
       Pusher(y, i) && (i += 1)
       SOBOL.next!(S, y)
    end;  [view(M,:,i) for i in axes(M,2)]
end

function MakeMultistartDomain(Pdim::Int, ProspectiveDom::Nothing, maxval::Real=1e5; verbose::Bool=true)
    verbose && @info "No MultistartDomain given, choosing default cube with maxval=$maxval"
    FullDomain(Pdim, maxval)
end
function MakeMultistartDomain(Pdim::Int, ProspectiveDom::HyperCube, maxval::Real=1e5; verbose::Bool=true)
    # clamp ProspectiveDom to finite size
    intersect(ProspectiveDom, FullDomain(length(ProspectiveDom), maxval))
end

MultistartFit(DM::AbstractDataModel; kwargs...) = MultistartFit(DM, Predictor(DM); kwargs...)
MultistartFit(DM::AbstractDataModel, M::ModelMap; MultistartDomain::HyperCube=Domain(M), kwargs...) = MultistartFit(DM, MultistartDomain; MultistartDomain, kwargs...)
function MultistartFit(DM::AbstractDataModel, M::ModelOrFunction; maxval::Real=1e5, MultistartDomain::Union{Nothing,HyperCube}=nothing, verbose::Bool=true, kwargs...)
    Dom = MakeMultistartDomain(pdim(DM), MultistartDomain, maxval; verbose)
    MultistartFit(DM, Dom; MultistartDomain=Dom, maxval, verbose, kwargs...)
end
# Create PointGenerator and drop model again
function MultistartFit(DM::AbstractDataModel, MultistartDom::HyperCube; MultistartDomain::HyperCube=MultistartDom, N::Int=100, seed::Int=rand(0:Int(1e7)), resampling::Bool=true, maxval::Real=1e5, kwargs...)
    MultistartFit(DM, (resampling ? SOBOL.SobolSeq : GenerateSobolPoints)(MultistartDomain, maxval; N, seed); MultistartDomain, N, seed, resampling, kwargs...)
end

## Legacy Method for DS, model, LogPriorFn?
# This is where PerformStepManual! from ProfileLikelihood lands
function MultistartFit(DS::AbstractDataSet, model::ModelOrFunction, startp::AbstractVector{<:Number}=rand(pdim(DS,model)), LogPriorFn::Union{Nothing,Function}=nothing; 
                            MultistartDomain::Union{HyperCube,Nothing}=Domain(model), kwargs...)
    CostFunction = (θ::AbstractVector-> -loglikelihood(DS, model, θ, LogPriorFn))
    # CostFunction generation with GetLogLikelihoodFn apparently very slow (due to specialization for each generated ValInserter model?)
    # CostFunction = GetLogLikelihoodFn(DS, model, LogPriorFn)
    MultistartFit(CostFunction, startp; LogPriorFn, MultistartDomain, kwargs...)
end
# This is where PerformStepGeneral! from ProfileLikelihood lands
function MultistartFit(CostFunction::Function, startp::AbstractVector{<:Number}, MDom::Union{HyperCube,Nothing}=nothing; MultistartDomain::Union{HyperCube,Nothing}=MDom, maxval::Real=1e5, N::Int=100, 
                            seed::Int=rand(1000:15000), resampling::Bool=true, verbose::Bool=true, kwargs...)
    Dom = MakeMultistartDomain(length(startp), MultistartDomain, maxval; verbose)
    InitialPointGen = (resampling ? SOBOL.SobolSeq : GenerateSobolPoints)(Dom, maxval; N, seed)
    # Drop startp
    MultistartFit(CostFunction, InitialPointGen; MultistartDomain=Dom, maxval, N, seed, resampling, verbose, kwargs...)
end

# For PerformStepGeneral!
function MultistartFit(Fs::Tuple, args...; kwargs...)
    @assert 1 ≤ length(Fs) ≤ 3
    Kw = length(Fs) == 1 ? (;) : length(Fs) == 2 ? (;CostGradient=Fs[2]) : (;CostGradient=Fs[2], CostHessian=Fs[3])
    MultistartFit(Fs[1], args...; Kw..., kwargs...)
end


"""
    MultistartFit(DM::AbstractDataModel; maxval::Real=1e5, MultistartDomain::HyperCube=FullDomain(pdim(DM), maxval), kwargs...)
Performs Multistart optimization with `N` starts and timeout of fits after `timeout` seconds.
If `resampling=true`, if likelihood non-finite new initial starts are redrawn until `N` suitable initials are found. 
If `Robust=true`, performs optimization wrt. p-norm according to given kwarg `p`.
For `Full=false`, only the final MLE is returned, otherwise a `MultistartResults` object is returned, which can be further analyzed and plotted.
The keyword `TransformSample` can be used to specify a function which is applied to the sample, allowing e.g. for sampling only a subset of the parameters and then adding on components which should stay at fixed initial values for the multistart.
!!! note
    Any further keyword arguments are passed through to the optimization procedure [`InformationGeometry.minimize`](@ref) such as tolerances, optimization methods, domain constraints, etc.
"""
function MultistartFit(DM::AbstractDataModel, InitialPointGen::Union{AbstractVector{<:AbstractVector{<:Number}}, Distributions.MultivariateDistribution, Base.Generator, SOBOL.AbstractSobolSeq}; 
                                        CostFunction::Function=Negate(loglikelihood(DM)), LogPriorFn::Union{Nothing,Function}=LogPrior(DM), pnames::AbstractVector{<:StringOrSymb}=pnames(DM),
                                        meth=((isnothing(LogPriorFn) && DM isa DataModel && Data(DM) isa AbstractFixedUncertaintyDataSet) ? nothing : Optim.NewtonTrustRegion()), kwargs...)
    MultistartFit(CostFunction, InitialPointGen; LogPriorFn, pnames, meth, DM=DM, kwargs...)
end
function MultistartFit(costfunction::Function, InitialPointGen::Union{AbstractVector{<:AbstractVector{<:Number}}, Distributions.MultivariateDistribution, Base.Generator, SOBOL.AbstractSobolSeq}; showprogress::Bool=true, N::Int=100, maxval::Real=1e5, plot::Bool=false, 
                                        DM::Union{Nothing,AbstractDataModel}=nothing, LogPriorFn::Union{Nothing,Function}=nothing, CostFunction::Function=costfunction, resampling::Bool=!(InitialPointGen isa AbstractVector), pnames::AbstractVector{<:StringOrSymb}=Symbol[], TransformSample::Function=identity,
                                        MultistartDomain::Union{HyperCube,Nothing}=nothing, parallel::Bool=true, Robust::Bool=false, TryCatchOptimizer::Bool=true, TryCatchCostFunc::Bool=false, p::Real=2, timeout::Real=120, verbose::Bool=false, 
                                        meth=((isnothing(LogPriorFn) && DM isa DataModel && Data(DM) isa AbstractFixedUncertaintyDataSet) ? nothing : Optim.NewtonTrustRegion()), Full::Bool=true, SaveFullOptimizationResults::Bool=Full, seed::Union{Int,Nothing}=nothing, kwargs...)
    @assert N ≥ 1
    @assert resampling ? !(InitialPointGen isa AbstractVector) : (InitialPointGen isa AbstractVector)
    
    # +Inf if error during optimization, should rarely happen
    BareOptimFunc = if !isnothing(DM)
        @assert !Robust || p > 0
        # DM given, so information about Boundaries etc. passed on
        #### SHOULD ALSO PASS COSTFUNCTION HERE
        RobustFunc(θ::AbstractVector{<:Number}) = RobustFit(DM, θ; p, timeout, verbose, meth, Full, kwargs...)
        Func(θ::AbstractVector{<:Number}) = InformationGeometry.minimize(DM, θ; timeout, verbose, meth, Full, kwargs...)
        Robust ? RobustFunc : Func
    else
        @assert !Robust "Cannot generate Robust p-norm version if only Cost Function given."
        FuncPureCost(θ::AbstractVector{<:Number}) = InformationGeometry.minimize(CostFunction, θ; timeout, verbose, meth, Full, kwargs...)
    end
    # Allow for disabling try catch;
    # TotalFunc(θ::AbstractVector{<:Number}) = try    InformationGeometry.TotalLeastSquaresV()    catch;  fill(-Inf, length(θ))   end
    
    TryCatchWrapper(F::Function, Default=-Inf) = x -> try F(x) catch;   Default   end
    # Double negation... Use LogLikelihoodFn instead? Make this consistent with GetProfile()
    LogLikeFunc = (TryCatchCostFunc ? TryCatchWrapper : identity)(Negate(CostFunction))

    TakeFromUnclamped(X::Distributions.Distribution) = rand(X)
    TakeFromUnclamped(X::Base.Generator) = iterate(X)[1]
    TakeFromUnclamped(S::SOBOL.AbstractSobolSeq) = SOBOL.next!(S)
    TakeFromClamped(X) = clamp(TakeFromUnclamped(X), MultistartDomain)
    TakeFrom = (!isnothing(MultistartDomain) && !(InitialPointGen isa Union{AbstractVector{<:AbstractVector{<:Number}},SOBOL.AbstractSobolSeq})) ? TransformSample∘TakeFromClamped : TransformSample∘TakeFromUnclamped
    # count total sampling attempts when resampling
    InitialPoints, InitialObjectives = if resampling
        InitPoints = typeof(TakeFrom(InitialPointGen))[]
        InitObjectives = eltype(eltype(InitPoints))[]
        sizehint!(InitPoints, 3N);  sizehint!(InitObjectives, 3N)
        SuccessCount = 0
        while SuccessCount < N
            push!(InitPoints, TakeFrom(InitialPointGen))
            push!(InitObjectives, LogLikeFunc(InitPoints[end]))
            isfinite(InitObjectives[end]) && (SuccessCount += 1)
        end
        InitPoints, InitObjectives
    else
        InitialPointGen, (parallel ? pmap : map)(LogLikeFunc, InitialPointGen)
    end
    OptimFunc = TryCatchOptimizer ? TryCatchWrapper(BareOptimFunc,fill(-Inf, length(InitialPoints[1]))) : BareOptimFunc
    Res = if showprogress
        (parallel ?  progress_pmap : progress_map)(OptimFunc, InitialPoints; progress=Progress(length(InitialPoints), desc="Multistart fitting... "*(parallel ? "(parallel, $(nworkers()) workers) " : ""), showspeed=true))
    else
        (parallel ?  pmap : map)(OptimFunc, InitialPoints)
    end
    FinalPoints = Full ? GetMinimizer.(Res) : Res

    FinalObjectives = (parallel ? pmap : map)(LogLikeFunc, FinalPoints)
    
    if Full
        Iterations = GetIterations.(Res)
        # By internal optimizer criterion:
        Converged = map(x->HasConverged(x; verbose=false), Res)
        PNames = length(pnames) == 0 ? CreateSymbolNames(length(FinalPoints[1])) : pnames
        R = MultistartResults(FinalPoints, InitialPoints, FinalObjectives, InitialObjectives, Iterations, Converged, PNames, meth, seed, MultistartDomain, SaveFullOptimizationResults ? Res : nothing; verbose)
        plot && display(RecipesBase.plot(R))
        R
    else
        MaxVal, MaxInd = findmax(FinalObjectives)
        GetMinimizer(FinalPoints[MaxInd])
    end
end

"""
    LocalMultistartFit(DM::AbstractDataModel, scale::Real=sqrt(InvChisqCDF(DOF(DM), ConfVol(2.0))); kwargs...)
    LocalMultistartFit(DM::AbstractDataModel, mle::AbstractVector{<:Number}, scale::Real=sqrt(InvChisqCDF(DOF(DM), ConfVol(2.0))); kwargs...)
Performs a multistart search locally around a given MLE in a cube constructed from the Fisher information and multiplied with `scale`.
"""
LocalMultistartFit(DM::AbstractDataModel, scale::Real=sqrt(InvChisqCDF(DOF(DM), ConfVol(2.0))); kwargs...) = LocalMultistartFit(DM, MLE(DM), scale; kwargs...)
LocalMultistartFit(DM::AbstractDataModel, mle::AbstractVector{<:Number}, scale::Real=sqrt(InvChisqCDF(DOF(DM), ConfVol(2.0))); kwargs...) = LocalMultistartFit(DM, MLEuncert(DM, mle), scale; kwargs...)
LocalMultistartFit(DM::AbstractDataModel, mleuncert::AbstractVector{<:Measurements.Measurement}, scale::Real=sqrt(InvChisqCDF(DOF(DM), ConfVol(2.0))); kwargs...) = MultistartFit(DM; MultistartDomain=HyperCube(mleuncert, scale), kwargs...)


struct MultistartResults <: AbstractMultistartResults
    FinalPoints::AbstractVector{<:AbstractVector{<:Number}}
    InitialPoints::AbstractVector{<:AbstractVector{<:Number}}
    FinalObjectives::AbstractVector{<:Number}
    InitialObjectives::AbstractVector{<:Number}
    Iterations::AbstractVector{<:Int}
    Converged::AbstractVector{<:Bool}
    pnames::AbstractVector{Symbol}
    OptimMeth
    seed::Union{Int,Nothing}
    MultistartDomain::Union{Nothing,HyperCube}
    FullOptimResults
    Meta
    function MultistartResults(
            FinalPoints::AbstractVector{<:AbstractVector{<:Number}},
            InitialPoints::AbstractVector{<:AbstractVector{<:Number}},
            FinalObjectives::AbstractVector{<:Number},
            InitialObjectives::AbstractVector{<:Number},
            Iterations::AbstractVector{<:Int}=zeros(Int, length(FinalObjectives)),
            Converged::AbstractVector{<:Bool}=map(isfinite, FinalObjectives),
            pnames::AbstractVector{<:StringOrSymb}=CreateSymbolNames(length(FinalPoints[1])),
            meth=missing,
            seed::Union{Int,Nothing}=nothing,
            MultistartDomain::Union{Nothing,HyperCube}=nothing,
            FullOptimResults=nothing,
            Meta=nothing; 
            verbose::Bool=true
        )
        @assert length(FinalPoints) == length(InitialPoints) == length(FinalObjectives) == length(InitialObjectives) == length(Iterations)
        @assert ConsistentElDims(FinalPoints) == length(pnames)
        OptimMeth = isnothing(meth) ? LsqFit.LevenbergMarquardt() : meth
        
        if verbose
            # Convert possible NaNs in FinalObjectives to -Inf to avoid problems in sorting NaNs
            nans = 0
            for i in eachindex(FinalObjectives)
                (!isfinite(FinalObjectives[i]) || !any(isfinite, FinalPoints[i])) && (FinalObjectives[i] = -Inf;    isfinite(InitialObjectives[i]) && (nans += 1))
            end
            if !any(isfinite, FinalObjectives)
                if any(isfinite, InitialObjectives)
                    @warn "ALL multistart optimizations with $(typeof(OptimMeth)) crashed! Most likely the options supplied to the optimizer were wrong. Automatic catching of optimizer errors can be disabled with kwarg TryCatchOptimizer=false."
                else
                    @warn "ALL multistarts failed on initial evaluation! Most likely the specified CostFunction was ill-defined and / or initial points were ill-chosen (Domain too large?) or too few. Automatic catching of CostFunction errors can be disabled with kwarg TryCatchCostFunc=false."
                end
            elseif nans > 0
                @info "$nans runs crashed during multistart optimization with $(typeof(OptimMeth))."
            end
        end
        if !issorted(FinalObjectives; rev=true)
            Perm = sortperm(FinalObjectives; rev=true)
            new(FinalPoints[Perm], InitialPoints[Perm], FinalObjectives[Perm], InitialObjectives[Perm], Iterations[Perm], Converged[Perm], Symbol.(pnames), OptimMeth, seed, MultistartDomain, isnothing(FullOptimResults) ? nothing : FullOptimResults[Perm], Meta)
        else
            new(FinalPoints, InitialPoints, FinalObjectives, InitialObjectives, Iterations, Converged, Symbol.(pnames), OptimMeth, seed, MultistartDomain, isnothing(FullOptimResults) ? nothing : FullOptimResults, Meta)
        end
    end
end
MultistartResults(;
    FinalPoints::AbstractVector{<:AbstractVector{<:Number}}=[Float64[]],
    InitialPoints::AbstractVector{<:AbstractVector{<:Number}}=[Float64[]],
    FinalObjectives::AbstractVector{<:Number}=Float64[],
    InitialObjectives::AbstractVector{<:Number}=Float64[],
    Iterations::AbstractVector{<:Int}=Int[],
    Converged::AbstractVector{<:Bool}=Bool[],
    pnames::AbstractVector{<:StringOrSymb}=Symbol[],
    OptimMeth=nothing,
    seed::Union{Int,Nothing}=nothing,
    MultistartDomain::Union{Nothing,HyperCube}=nothing,
    FullOptimResults=nothing,
    Meta=nothing,
    verbose::Bool=false
) = MultistartResults(FinalPoints, InitialPoints, FinalObjectives, InitialObjectives, Iterations, Converged, pnames, OptimMeth, seed, MultistartDomain, FullOptimResults, Meta; verbose)

function Base.vcat(R1::MultistartResults, R2::MultistartResults)
    @assert length(R1.pnames) == length(R2.pnames)
    R1.pnames != R2.pnames && @warn "Using pnames from first MultistartResults object."
    NewOptimMeth = if (!isnothing(R1.OptimMeth) && !isnothing(R2.OptimMeth)) && (!ismissing(R1.OptimMeth) && !ismissing(R2.OptimMeth)) && R1.OptimMeth != R2.OptimMeth
        @warn "Combining results from different optimizers."
        vcat(R1.OptimMeth, R2.OptimMeth)
    else R1.OptimMeth end
    R1.MultistartDomain != R2.MultistartDomain && @warn "Using MultistartDomain from first MultistartResults object."
    # Adopt larger seed for continuation
    NewSeed = if !(isnothing(R1.seed) && isnothing(R2.seed))
        @info "Using bigger of two seeds in vcat of MultistartResults."
        max(R1.seed, R2.seed)
    else nothing end

    MultistartResults(vcat(R1.FinalPoints, R2.FinalPoints), vcat(R1.InitialPoints, R2.InitialPoints),
        vcat(R1.FinalObjectives, R2.FinalObjectives), vcat(R1.InitialObjectives, R2.InitialObjectives), vcat(R1.Iterations, R2.Iterations), vcat(R1.Converged, R2.Converged),
        R1.pnames, NewOptimMeth, NewSeed, R1.MultistartDomain, (!isnothing(R1.FullOptimResults) && !isnothing(R2.FullOptimResults) ? vcat(R1.FullOptimResults,R2.FullOptimResults) : nothing),
        ((!isnothing(R1.Meta) && !isnothing(R2.Meta) && (@assert R1.Meta == R2.Meta; true)) ? R1.Meta : nothing),
    )
end


Base.length(R::MultistartResults) = length(R.FinalObjectives)
Base.firstindex(R::MultistartResults) = firstindex(R.FinalObjectives)
Base.lastindex(R::MultistartResults) = length(R.FinalObjectives)

MLE(R::MultistartResults) = R.FinalPoints[1]
pnames(R::MultistartResults) = R.pnames .|> string
Pnames(R::MultistartResults) = R.pnames
Domain(R::MultistartResults) = R.MultistartDomain
pdim(R::MultistartResults) = length(MLE(R))


SubsetMultistartResults(R::MultistartResults, LastInd::Int; idxs::AbstractVector{<:Int}=1:LastInd, kwargs...) = SubsetMultistartResults(R, idxs; kwargs...)
function SubsetMultistartResults(R::MultistartResults, idxs::AbstractVector{<:Int}; kwargs...)
    @assert all(1 .≤ idxs .≤ length(R.FinalObjectives)) && allunique(idxs)
    remake(R; FinalPoints=(@view R.FinalPoints[idxs]), 
        InitialPoints=(@view R.InitialPoints[idxs]), 
        FinalObjectives=(@view R.FinalObjectives[idxs]), 
        InitialObjectives=(@view R.InitialObjectives[idxs]), 
        Iterations=(@view R.Iterations[idxs]), Converged=(@view R.Converged[idxs]), 
        (R.OptimMeth isa AbstractVector ? (;meth=(@view R.OptimMeth[idxs])) : (;))..., 
        (R.FullOptimResults isa AbstractVector ? (; FullOptimResults=(@view R.FullOptimResults[idxs])) : (;))..., 
        verbose=false, kwargs...)
end


"""
    WaterfallPlot(R::MultistartResults; BiLog::Bool=true, MaxValue::Real=3000, StepTol::Real=1e-3, kwargs...)
Shows Waterfall plot for the given results of MultistartFit.
`StepTol` is used to decide which difference of two neighbouring values in the Waterfall plot constitutes a step. `StepTol=0` deactivates step marks.
`MaxValue` is used to set threshold for ignoring points whose cost function after optimization is too large compared with best optimum.
`BiLog=false` disables logarithmic scale for cost function. A custom transformation can be specified via keyword `Trafo::Function`.
"""
WaterfallPlot(R::MultistartResults; kwargs...) = RecipesBase.plot(R, Val(:Waterfall); kwargs...)

"""
    ParameterPlot(R::MultistartResults; st=:dotplot, BiLog::Bool=true, Nsteps::Int=5, StepTol::Real=1e-3, MaxValue=3000)
Plots the parameter values of the `MultistartResults` separated by step to show whether the different optima are localized or not.
`st` can be either `:dotplot`, `:boxplot` or `:violin`.
!!! note
    StatsPlots.jl needs to be loaded to use this plotting function.
"""
ParameterPlot(R::MultistartResults; kwargs...) = try
    RecipesBase.plot(R, Val(:StepAnalysis); kwargs...)
catch E;
    (E isa ErrorException ? throw("StatsPlots.jl needs to be loaded before using ParameterPlot.") : rethrow(E))
end

FindLastIndSafe(R::MultistartResults) = (LastFinite=findlast(isfinite, R.FinalObjectives);  isnothing(LastFinite) ? 0 : LastFinite)

# GetStepInds always includes last point of lower step
function GetStepInds(R::MultistartResults, ymaxind::Int=FindLastIndSafe(R); StepTol::Real=1e-3)
    !isnothing(R.Meta) || (@assert 1 ≤ ymaxind ≤ length(R) && StepTol > 0);       F = -R.FinalObjectives
    S = [i for i in 1:ymaxind-1 if isfinite(F[i+1]) && abs(F[i+1]-F[i]) > StepTol]
    length(S) < 1 ? [ymaxind] : S
end
function GetStepRanges(R::MultistartResults, ymaxind::Int=FindLastIndSafe(R), StepInds::AbstractVector{<:Int}=GetStepInds(R,ymaxind))
    Steps = [1:StepInds[1]]
    length(StepInds) > 1 && append!(Steps, [StepInds[i-1]+1:StepInds[i] for i in 2:length(StepInds)])
    StepInds[end] < ymaxind && push!(Steps, StepInds[end]+1:ymaxind)
    Steps
end

function GetFirstStepInd(R::MultistartResults, ymaxind::Int=FindLastIndSafe(R); StepTol::Real=1e-3)
    !isnothing(R.Meta) || (@assert 1 ≤ ymaxind ≤ length(R) && StepTol > 0);  F=-R.FinalObjectives
    FirstStepInd = findfirst(i->isfinite(F[i+1]) && abs(F[i+1]-F[i]) > StepTol, 1:ymaxind-1)
    isnothing(FirstStepInd) ? ymaxind : FirstStepInd
end

"""
    GetStepParameters(R::MultistartResults, n::Int, m::Int=1; StepTol::Real=1e-3, kwargs...)
Returns the `m`-th parameter configuration of the `n`-th step in the `WaterfallPlot`.
"""
GetStepParameters(R::MultistartResults, n::Int, m::Int=1; StepTol::Real=1e-3, ymaxind::Int=FindLastIndSafe(R), StepInds::AbstractVector{<:Int}=GetStepInds(R,ymaxind;StepTol)) = R.FinalPoints[GetStepRanges(R,ymaxind,StepInds)[n][m]]


# RecipesBase.@recipe f(R::MultistartResults, S::Symbol=(isnothing(R.Meta) ? :Waterfall : :SubspaceProjection)) = R, Val(S)
function RecipesBase.plot(R::MultistartResults, S::Symbol=(isnothing(R.Meta) ? :Waterfall : :StochasticProfiles), args...; kwargs...)
    S === :StochasticProfiles && return StochasticProfileLikelihoodPlot(R, args...; kwargs...)
    RecipesBase.plot(R, Val(S), args...; kwargs...)
end
# kwargs BiLog, StepTol, MaxValue, MaxInd, ColorIterations
RecipesBase.@recipe function f(R::MultistartResults, ::Val{:Waterfall})
    DoBiLog = get(plotattributes, :BiLog, true)
    Trafo = get(plotattributes, :Trafo, DoBiLog ? BiLog : identity)
    MaxValue = get(plotattributes, :MaxValue, BiExp(8))
    ColorIterations = get(plotattributes, :ColorIterations, true)
    @assert MaxValue ≥ 0
    Fin = Trafo.(-R.FinalObjectives)
    # Cut off results with difference to lowest optimum greater than MaxValue
    ymaxind = get(plotattributes, :MaxInd, (Q=findlast(x->isfinite(x) && abs(x-Fin[1]) < Trafo(MaxValue), Fin);   isnothing(Q) ? length(Fin) : Q))
    xlabel --> "Run (sorted by cost function result)"
    ylabel --> ApplyTrafoNames("Final Cost Value", Trafo)
    title --> "Waterfall plot $(ymaxind)/$(length(Fin))"
    leg --> nothing
    st --> :scatter
    
    ymin, ymax = Fin[1], Fin[ymaxind]
    ylims --> (ydiff=ymax-ymin; (min(ymin-0.05*ydiff,ymin-0.01), max(ymax+0.05*ydiff, ymin+0.01)))
    @series begin
        label --> "Finals"
        markersize --> 20/sqrt(ymaxind)
        msw --> 0
        markershape --> map(x->x ? :circle : :utriangle, (@view R.Converged[1:ymaxind]))
        if ColorIterations
            zcolor --> map(x->isfinite(x) ? x : 0, R.Iterations)
            color --> cgrad(:plasma; rev=true)
        end
        @view Fin[1:ymaxind]
    end
    @series begin
        label --> "Initials"
        marker --> :cross
        Trafo.(-R.InitialObjectives[1:ymaxind])
    end
    # Mark steps with vline, StepTol on linear scale
    StepTol = get(plotattributes, :StepTol, 1e-3)
    if 0 < StepTol
        Steps = GetStepInds(R, ymaxind; StepTol=StepTol)
        for (i,AfterIter) in enumerate(@view Steps[1:end-1]) # Do not plot line after last point
            @series begin
                st := :vline
                line --> :dash
                label := i > 1 ? string(Steps[i]-Steps[i-1]) : string(Steps[i])
                color --> :grey
                [AfterIter + 0.5]
            end
        end
    end
end

# kwargs st, BiLog, MaxValue, StepTol, pnames, Nsteps, Spacer
RecipesBase.@recipe function f(R::MultistartResults, ::Union{Val{:ParameterPlot}, Val{:StepAnalysis}})
    DoBiLog = get(plotattributes, :BiLog, true)
    Trafo = get(plotattributes, :Trafo, DoBiLog ? BiLog : identity)
    MaxValue = get(plotattributes, :MaxValue, BiExp(8))
    StepTol = get(plotattributes, :StepTol, 1e-3)
    pnames = get(plotattributes, :pnames, string.(Pnames(R)))
    Nsteps = get(plotattributes, :Nsteps, 5)
    Spacer = get(plotattributes, :Spacer, 2)

    Fin = -R.FinalObjectives
    ymaxind = (Q=findlast(x->isfinite(x) && abs(x-Fin[1]) < MaxValue, Fin);   isnothing(Q) ? length(Fin) : Q)
    @assert StepTol > 0
    StepInds = GetStepInds(R, ymaxind; StepTol=StepTol)
    AllSteps = GetStepRanges(R, ymaxind, StepInds)
    Steps = @view AllSteps[1:min(Nsteps, length(AllSteps))]
    dfs = [Trafo.(reduce(vcat, R.FinalPoints[step])) for step in Steps]
    Xvals = 1.0:Spacer*length(dfs):Spacer*length(dfs)*length(pnames) |> collect

    xtick --> (Xvals .+ Spacer*0.5(length(dfs)-1), pnames)
    ylabel --> ApplyTrafoNames("Parameter Value", Trafo)
    xlabel --> "Parameter"
    seriestype --> get(plotattributes, :st, :dotplot)
    color_palette = get(plotattributes, :color_palette, :default)
    for i in eachindex(dfs)
        @series begin
            color --> palette(color_palette)[(((i-1)%15)+1)]
            label := "Step $i"
            repeat(Xvals .+ Spacer*(i-1); outer=length(dfs[i])÷length(pnames)), dfs[i]
        end
    end
    for i in 1:length(Xvals)-1
        @series begin
            seriestype := :vline
            color := :grey
            line := :dash
            label := nothing
            [Xvals[i] .+ Spacer*length(dfs).-0.5Spacer]
        end
    end
end

RecipesBase.@recipe f(DM::AbstractDataModel, R::MultistartResults, args...) = DM, MLE(R), args...

"""
    DistanceMatrixWithinStep(DM::AbstractDataModel, R::MultistartResults, Ind::Int; logarithmic::Bool=true, plot::Bool=isloaded(:Plots), kwargs...)
Returns matrix of mutual distances between optima in step `Ind` with respect to Fisher Metric of first entry in step. 
"""
function DistanceMatrixWithinStep(DM::AbstractDataModel, R::MultistartResults, Ind::Int; logarithmic::Bool=true, plot::Bool=isloaded(:Plots), kwargs...)
    ymaxind = FindLastIndSafe(R);    StepInds = GetStepInds(R, ymaxind);     Steps = GetStepRanges(R, ymaxind, StepInds)
    @assert 1 ≤ Ind ≤ length(Steps)
    F = FisherMetric(DM, R.FinalPoints[Steps[Ind][1]]) # get first point in ProfileDomain
    Dists = [sqrt(InnerProduct(F, R.FinalPoints[i] - R.FinalPoints[j])) for i in Steps[Ind], j in Steps[Ind]]
    plot && display(RecipesBase.plot((logarithmic ? log1p : identity).(Dists + Diagonal(fill(NaN, size(Dists,1)))); st=:heatmap, clims=(0, Inf), kwargs...))
    Dists
end

"""
    DistanceMatrixBetweenSteps(DM::AbstractDataModel, R::MultistartResults; logarithmic::Bool=true, plot::Bool=isloaded(:Plots), kwargs...)
Returns matrix of mutual distances between first optima in steps with respect to Fisher Metric of best optimum. 
"""
function DistanceMatrixBetweenSteps(DM::AbstractDataModel, R::MultistartResults; logarithmic::Bool=true, plot::Bool=isloaded(:Plots), kwargs...)
    ymaxind = FindLastIndSafe(R);    StepInds = GetStepInds(R, ymaxind);     Steps = GetStepRanges(R, ymaxind, StepInds)
    F = FisherMetric(DM, R.FinalPoints[1])
    Dists = [sqrt(InnerProduct(F, R.FinalPoints[Steps[i][1]] - R.FinalPoints[Steps[j][1]])) for i in eachindex(Steps), j in eachindex(Steps)]
    plot && display(RecipesBase.plot((logarithmic ? log1p : identity).(Dists + Diagonal(fill(NaN, size(Dists,1)))); st=:heatmap, clims=(0, Inf), kwargs...))
    Dists
end



ContinuationSeed(R::MultistartResults) = R.seed + length(R.FinalObjectives)


GetNFromTargetTime(DM::AbstractDataModel, args...; kwargs...) = GetNFromTargetTime(loglikelihood(DM), MLE(DM), args...; kwargs...)
function GetNFromTargetTime(L::Function, startp::AbstractVector, TargetTime::Real=60; minval=2, maxval=Inf, verbose::Bool=true)
    L(startp);  Tsingle = @elapsed L(startp)
    # Tsingle = @belapsed $L($startp)
    N = TargetTime / Tsingle |> floor |> Int
    Nsingle = clamp(N^(1/length(startp)), minval, maxval) |> floor |> Int
    verbose && @info "Single evaluation took $(round(Tsingle; sigdigits=3))s, suggesting approximately N=$N samples in total (Nsingle=$Nsingle, $Nsingle^$(length(startp))=$(Nsingle^length(startp))) to fill allotted $(TargetTime)s."
    N
end


HistoBins(X::AbstractVector{<:Number}, Bins::Int=Int(ceil(sqrt(length(X)))); nbins::Int=Bins) = range(Measurements.value.(extrema(X))...; length=nbins+1)
HistoBins(X::AbstractVector{<:AbstractVector}, ind::Int, Bins::Int=Int(ceil(sqrt(length(X)))); nbins::Int=Bins) = range(Measurements.value.(extrema(x->x[ind],X))...; length=nbins+1)

"""
   StochasticProfileLikelihood(DM::AbstractDataModel, C::HyperCube=Domain(DM); TargetTime=30, Nsingle::Int=5, N::Int=Nsingle^length(C), nbins::Int=0.5Nsingle, parallel::Bool=true, maxval::Real=1e5, BiLog::Bool=true, kwargs...)
Samples the likelihood over the parameter space, splits the value range of each parameter into bins and visualizes the best observed likelihood value from all samples with the respective parameter value in this bin.
Therefore, it gives a coarse-grained global approximation to the profile likelihood, approaching the true profile likelihood in the large sample limit as `N` ⟶ ∞, which is however much cheaper to compute as no re-optimization of nuisance parameters is required.
This can already give hints for good candidates for initial parameter values from which to start optimization and conversely also illustrate parts of the parameter ranges which can be excluded from multistart fitting due to their consistently unsuitable likelihood values.

The results of this sampling are saved in the `FinalPoints` and `FinalObjectives` fields of a `MultistartResults` object.

The `TargetTime` kwarg can be used to choose the number of samples such that the sampling is expected to require approximately the allotted time in seconds.
"""
function StochasticProfileLikelihood(DM::AbstractDataModel, C::HyperCube=GetDomainSafe(DM); TargetTime::Real=30, Nsingle::Union{Nothing,Int}=nothing, N::Int=isnothing(Nsingle) ? GetNFromTargetTime(DM, TargetTime) : Nsingle^length(C), 
                                                        nbins::Int=isnothing(Nsingle) ? Int(ceil(clamp(0.5*(N^(1/length(C))),3,100))) : Nsingle, maxval::Real=1e5, Domain::HyperCube=C∩FullDomain(length(C),maxval), TransformSample::Function=identity, seed::Int=rand(0:Int(1e7)), kwargs...)
    Points = GenerateSobolPoints(Domain; seed, N, maxval)
    !(TransformSample === identity) && (Points .= TransformSample.(Points))
    StochasticProfileLikelihood(DM, Points; Domain, nbins, seed, kwargs...)
end
function StochasticProfileLikelihood(DM::AbstractDataModel, Points::AbstractVector{<:AbstractVector}; LogLikelihoodFn::Function=loglikelihood(DM), parallel::Bool=true, pnames::AbstractVector{<:AbstractString}=pnames(DM), kwargs...)
   @info "Starting $(length(Points)) samples."
   Likelihoods = (parallel ? progress_pmap : progress_map)(LogLikelihoodFn, Points; progress=Progress(length(Points); desc="Sampling Parameter Space... "*(parallel ? "(parallel, $(nworkers()) workers) " : ""), dt=0.2, showspeed=true))
   StochasticProfileLikelihood(Points, Likelihoods; pnames, kwargs...)
end
# Construct MultistartResults and call Plot
function StochasticProfileLikelihood(Points::AbstractVector{<:AbstractVector}, Likelihoods::AbstractVector{<:Number}; plot::Bool=isloaded(:Plots), Domain::Union{HyperCube,Nothing}=nothing, pnames::AbstractVector{<:AbstractString}=CreateSymbolNames(length(Points[1])), Meta=:SampledLikelihood, seed::Union{Nothing,Int}=nothing, kwargs...)
   R = MultistartResults(Points, [eltype(Points[1])[] for i in eachindex(Points)], Likelihoods, fill(-Inf, length(Likelihoods)), zeros(Int, length(Likelihoods)), falses(length(Likelihoods)), pnames, missing, seed, Domain, nothing, Meta)
   plot && display(StochasticProfileLikelihoodPlot(R; kwargs...))
   R
end
# Refine given MultistartObject with more samples
function StochasticProfileLikelihood(R::MultistartResults, args...; plot::Bool=false, kwargs...)
    vcat(R, StochasticProfileLikelihood(args...; plot, seed=ContinuationSeed(R), kwargs...))
end



for F in [:GetStochasticProfile, :_GetStochasticProfile, :GetStochastic2DProfile, :_GetStochastic2DProfile]
    @eval $F(R::MultistartResults, args...; pnames=pnames(R), kwargs...) = $F(R.FinalPoints, R.FinalObjectives, args...; pnames, kwargs...)
end
CountInBin(pBins::AbstractVector{<:Number}, Points::AbstractVector{<:AbstractVector}, ind::Int) = [count(LogLikeInd->pBins[i] .≤ Points[LogLikeInd][ind] .< pBins[i+1], 1:length(Points)) for i in 1:length(pBins)-1]
CountInBin(pxBins::AbstractVector{<:Number}, pyBins::AbstractVector{<:Number}, Points::AbstractVector{<:AbstractVector}, idxs::AbstractVector{<:Int}) = [count(LogLikeInd->pxBins[i] ≤ Points[LogLikeInd][idxs[1]] < pxBins[i+1] && pyBins[j] ≤ Points[LogLikeInd][idxs[2]] < pyBins[j+1], 1:length(Points)) for i in 1:length(pxBins)-1, j in 1:length(pyBins)-1]

CenteredVec(X::AbstractVector) = @views (X[1:end-1] .+ X[2:end]) ./ 2

# Already has factor of two
# Offset optional
function GetStochasticProfile(Points::AbstractVector{<:AbstractVector}, Likelihoods::AbstractVector{<:Number}, Idxs::AbstractVector{<:Int}=1:length(Points[1]); dof::Int=length(Points[1]), pnames::AbstractVector=CreateSymbolNames(length(Points[1])), pBins::Union{Nothing,AbstractVector{<:AbstractVector{<:Number}}}=nothing, kwargs...)
    # Allow for Vector{Vector} pBins to be passed for setting the pBins of all profiles manually
    Res = [_GetStochasticProfile(Points, Likelihoods, i; (!isnothing(pBins) ? (; pBins=pBins[i]) : (;))..., kwargs...) for i in Idxs]
    Bins, Vals, Trajs = getindex.(Res,1), getindex.(Res,2), getindex.(Res,3)
    mle = Trajs[1][findmin(Vals[1])[2]]
    ParameterProfiles([VectorOfArray([CenteredVec(Bins[i]), Vals[i], trues(size(Vals[i],1))]) for i in eachindex(Bins)], Trajs, pnames, mle, dof, true, :StochasticProfiles)
end
function _GetStochasticProfile(Points::AbstractVector{<:AbstractVector}, Likelihoods::AbstractVector{<:Number}, ind::Int; nbins::Int=Int(ceil(clamp(0.5*(length(Likelihoods)^(1/length(Points[1]))),3,100))), Extremizer::Function=findmax, 
                                    UseSorted::Bool=true, pval::AbstractVector=getindex.(Points,ind), pBins::AbstractVector=collect(HistoBins(pval,nbins)), OffsetResults::Bool=true, pnames=String[])
    ResInds = if UseSorted && issorted(Likelihoods; rev=true)
        [(X=findfirst(LogLikeInd->pBins[i] .≤ pval[LogLikeInd] .< pBins[i+1], 1:length(Points));  isnothing(X) ? 0 : X) for i in 1:length(pBins)-1]
    else
        keep = falses(length(pval), length(pBins)-1)
        for i in axes(keep,2)
            keep[:, i] .= pBins[i] .≤ pval .< pBins[i+1]
        end
        [(X=(@view keep[:,i]);   any(X) ? ((1:size(keep,1))[X])[(Extremizer(@view Likelihoods[X]))[2]] : 0) for i in axes(keep,2)]
    end
    Res = [i == 0 ? -Inf : Likelihoods[i] for i in ResInds]
    OffsetResults && (Res .-= Extremizer(Res)[1])
    ResPoints = [i == 0 ? fill(-Inf, length(Points[1])) : Points[i] for i in ResInds]
    Res .*= -2
    pBins, Res, ResPoints
end


function GetStochastic2DProfile(Points::AbstractVector{<:AbstractVector}, Likelihoods::AbstractVector{<:Number}, Idxs::AbstractVector{<:AbstractVector{<:Int}}=OrderedIndCombs2D(1:length(Points[1])); dof::Int=length(Points[1]), pnames::AbstractVector=CreateSymbolNames(length(Points[1])), kwargs...)
    Res = [_GetStochastic2DProfile(Points, Likelihoods, idxs; kwargs...) for idxs in Idxs]
    xBins, yBins, Vals, Trajs = getindex.(Res,1), getindex.(Res,2), getindex.(Res,3), getindex.(Res,4)
    # LogLikeMle, mleind = findmax(Vals[1]);  mle = Trajs[1][mleind]
    # ParameterProfiles([[(((@view Bins[i][1:end-1]) .+ (@view Bins[i][2:end]))./2) 2(LogLikeMle .-Vals[i]) trues(size(Vals[i],1))] for i in eachindex(Bins)], Trajs, pnames, mle, dof, true, :StochasticProfiles)
end
function _GetStochastic2DProfile(Points::AbstractVector{<:AbstractVector}, Likelihoods::AbstractVector{<:Number}, idxs::AbstractVector{<:Int}; Extremizer::Function=findmax, OffsetResults::Bool=true, 
                            nxbins::Int=Int(ceil(clamp(0.5*(length(Likelihoods)^(1/length(Points[1]))),3,100))), nybins::Int=Int(ceil(clamp(0.5*(length(Likelihoods)^(1/length(Points[1]))),3,100))), 
                            pxval::AbstractVector=getindex.(Points, idxs[1]), pyval::AbstractVector=getindex.(Points, idxs[2]), pnames=String[],
                            UseSorted::Bool=true, pxBins::AbstractVector=collect(HistoBins(pxval, nxbins)), pyBins::AbstractVector=collect(HistoBins(pyval, nybins)))
    @assert length(idxs) == 2 && allunique(idxs) && all(1 .≤ idxs .≤ length(Points[1]))
    
    ResInds = if UseSorted && issorted(Likelihoods; rev=true)
        [(X=findfirst(LogLikeInd->pxBins[i] ≤ pxval[LogLikeInd] < pxBins[i+1] && pyBins[j] ≤ pyval[LogLikeInd] < pyBins[j+1], 1:length(pxval));  isnothing(X) ? 0 : X) for i in 1:length(pxBins)-1, j in 1:length(pyBins)-1]
    else
        keep = falses(length(pxval), length(pxBins)-1, length(pyBins)-1)
        for i in axes(keep,2), j in axes(keep,3)
            keep[:, i, j] .= pxBins[i] .≤ pxval .< pxBins[i+1] .&& pyBins[j] .≤ pyval .< pyBins[j+1]
        end
        [(X=(@view keep[:,i,j]);   any(X) ? ((1:size(keep,1))[X])[(Extremizer(@view Likelihoods[X]))[2]] : 0) for j in axes(keep,3), i in axes(keep,2)]
    end
    Res = [i == 0 ? -Inf : Likelihoods[i] for i in ResInds]
    OffsetResults && (Res .-= Extremizer(Res)[1])
    ResPoints = [i == 0 ? fill(-Inf, length(Points[1])) : Points[i] for i in ResInds]
    Res .*= -2
    pxBins, pyBins, Res, ResPoints
end


## Plot results:
StochasticProfileLikelihoodPlot(R::MultistartResults, ind::Int; kwargs...) = (@assert R.Meta === :SampledLikelihood;  StochasticProfileLikelihoodPlot(R.FinalPoints, R.FinalObjectives, ind; xlabel=string(pnames(R)[ind]), kwargs...))
StochasticProfileLikelihoodPlot(R::MultistartResults; kwargs...) = (@assert R.Meta === :SampledLikelihood;  StochasticProfileLikelihoodPlot(R.FinalPoints, R.FinalObjectives; pnames=string.(pnames(R)), kwargs...))
# Collective
function StochasticProfileLikelihoodPlot(Points::AbstractVector{<:AbstractVector}, Likelihoods::AbstractVector{<:Number}; CutoffVal::Real=Inf, Inds::AbstractVector{<:Int}=1:findlast(x->isfinite(x) && abs(x)<CutoffVal,Likelihoods), nbins::Int=Int(ceil(clamp(0.5*(length(Inds)^(1/length(Points[1]))),3,100))), BiLog::Bool=true, Trafo::Function=(BiLog ? InformationGeometry.BiLog : identity), Extremizer::Function=findmax,
                                pnames::AbstractVector{<:AbstractString}=CreateSymbolNames(length(Points[1])), legend=false, OffsetResults::Bool=false, Cutoff::Int=400000, kwargs...)
    P = [StochasticProfileLikelihoodPlot((@view Points[Inds]), (@view Likelihoods[Inds]), i; nbins, Trafo, Extremizer, xlabel=string(pnames[i]), legend, OffsetResults) for i in eachindex(pnames)]
    Offset = OffsetResults ? Likelihoods[1] : 0.0
    AddedPlots = []
    push!(AddedPlots, RecipesBase.plot(1:length(Inds), -Trafo.((@view Likelihoods[Inds]) .-Offset); xlabel="Run index (sorted)", ylabel=ApplyTrafoNames("Objective", Trafo), label="Waterfall", legend))
    if length(pnames) ≤ 3
        # Cutoff elements in plot to avoid overloading backend
        ReducedInds = length(Inds) > Cutoff ? (@view Inds[1:Cutoff]) : Inds
        SP = RecipesBase.plot((@view Points[ReducedInds]); st=:scatter, zcolor=Trafo.((@view Likelihoods[ReducedInds]) .- Offset), msw=0, xlabel=pnames[1], ylabel=pnames[2], zlabel=(length(pnames) ≥ 3 ? pnames[3] : ""), c=:viridis, label="log-likelihood", legend, colorbar=true)
        if Extremizer === findmax
            maxind = Extremizer(Likelihoods)[2]
            RecipesBase.plot!(SP, Points[maxind:maxind]; st=:scatter, color=:red, msw=0, label="Best")
        end
        push!(AddedPlots, SP)
    end
    RecipesBase.plot([P; AddedPlots]...; layout=length(P)+length(AddedPlots), kwargs...)
end
# Individual Parameter
function StochasticProfileLikelihoodPlot(Points::AbstractVector{<:AbstractVector}, Likelihoods::AbstractVector{<:Number}, ind::Int; nbins::Int=Int(ceil(clamp(0.5*(length(Likelihoods)^(1/length(Points[1]))),3,100))), Extremizer::Function=findmax, 
                                 BiLog::Bool=true, Trafo::Function=(BiLog ? InformationGeometry.BiLog : identity), pval::AbstractVector=getindex.(Points, ind), pBins::AbstractVector=collect(HistoBins(pval, nbins)), OffsetResults::Bool=false,
                                 xlabel="Parameter $ind", kwargs...)
    _, Res, ResPoints = _GetStochasticProfile(Points, Likelihoods, ind; nbins, Extremizer, pval, pBins, OffsetResults)
    res = Trafo.(Res)
    HitsPerBin = float.(CountInBin(pBins, Points, ind));  HitsPerBin ./= maximum(HitsPerBin)
    Plt = RecipesBase.plot(CenteredVec(pBins), res; st=:bar, alpha=HitsPerBin, bar_width=diff(pBins), lw=0.5, xlabel, ylabel=ApplyTrafoNames("SPLA", Trafo), label="Conditional Objectives", kwargs...)
    Extremizer === findmax && RecipesBase.plot!(Plt, [ResPoints[Extremizer(-res)[2]][ind]]; st=:vline, line=:dash, c=:red, lw=1.5, label="Best Objective")
    Plt
end


function FindGoodStart(DM::AbstractDataModel, args...; plot::Bool=false, kwargs...)
    R = StochasticProfileLikelihood(DM, args...; plot, kwargs...)
    R.FinalPoints[findmax(R.FinalObjectives)[2]]
end



"""
    CleanupStochasticProfile(R::MultistartResults; nbins::Int=8)
Takes `MultistartResults` object and reduces all samples where the objective function is worse than the largest value used in any profile with `nbins` number of bins.

Note that since the bin widths are based on the observed samples `GetStochasticProfile`, the recalculated profiles often look different since the bin structure has changed!
To retain the same bin structure, supply the keyword argument `pBins` to `_GetStochasticProfile` manually.
"""
function CleanupStochasticProfile(R::MultistartResults, Buffer::Real=0; nbins::Int=8, OffsetResults::Bool=true, kwargs...)
    @assert OffsetResults
    # Factor of two already included in stochastic profile but not in MultistartResults
    P = GetStochasticProfile(R; nbins, OffsetResults, kwargs...)
    # Undo profile transform 2(Lmax - L)
    Value = R.FinalObjectives[1] - 0.5*maximum([maximum(@view Prof[:,2]) for Prof in Profiles(P)]) - abs(Buffer)
    SubsetMultistartResults(R, 1:findlast(L->L≥Value, R.FinalObjectives))
end


## OrderedIndCombs is row-first
OrderedIndCombs2D(paridxs::AbstractVector{<:Int}) = [[paridxs[j],paridxs[i]] for j in 1:length(paridxs)-1, i in 2:length(paridxs) if j < i]
OrderedIndCombs3D(paridxs::AbstractVector{<:Int}) = [[paridxs[k],paridxs[j],paridxs[i]] for k in 1:length(paridxs)-2, j in 2:length(paridxs)-1, i in 3:length(paridxs) if k < j < i]

# All dims in layout plot
@recipe function f(R::MultistartResults, V::Val{:SubspaceProjection}, FiniteInds::AbstractVector=(length(R.FinalObjectives) > 10000 ? (1:10000) : reverse(collect(1:length(R.FinalObjectives))[isfinite.(R.FinalObjectives)])))
    R, OrderedIndCombs2D(1:pdim(R)), V, FiniteInds
end
@recipe function f(R::MultistartResults, Combos::AbstractVector{<:AbstractVector{<:Int}}, V::Val{:SubspaceProjection}, FiniteInds::AbstractVector=(length(R.FinalObjectives) > 10000 ? (1:10000) : reverse(collect(1:length(R.FinalObjectives))[isfinite.(R.FinalObjectives)])))
   @assert allunique(Combos) && 2 ≤ ConsistentElDims(Combos) ≤ 3
   pdim(R) == 2 && return R, [1,2], V, FiniteInds
   layout --> length(Combos)
   size --> (1500,1500)
   for (i,inds) in enumerate(Combos)
      @series begin
         subplot := i
         R, inds, V, FiniteInds
      end
   end
end

# kwargs: BiLog
@recipe function f(R::MultistartResults, idxs::AbstractVector{<:Int}, V::Val{:SubspaceProjection}, FiniteInds::AbstractVector=(length(R.FinalObjectives) > 10000 ? (1:10000) : reverse(collect(1:length(R.FinalObjectives))[isfinite.(R.FinalObjectives)]));
                BiLog=true, Trafo=(BiLog ? InformationGeometry.BiLog : identity))
    color --> :viridis
    zcolor --> Trafo.(@view R.FinalObjectives[FiniteInds])
    colorbar --> false
    xlabel --> Pnames(R)[idxs[1]]
    ylabel --> Pnames(R)[idxs[2]]
    zlabel --> (length(idxs) ≥ 3 ? Pnames(R)[idxs[3]] : "")
    label --> ApplyTrafoNames("Log-Likelihood", Trafo)
    legend --> false
    (@view R.FinalPoints[FiniteInds]), idxs, V
end

@recipe function f(X::AbstractVector{<:AbstractVector}, idxs::AbstractVector{<:Int}, ::Val{:SubspaceProjection})
   @assert 1 ≤ length(idxs) ≤ 3 && allunique(idxs) && all(1 .≤ idxs .≤ ConsistentElDims(X))
   msw := 0
   st := :scatter
   map(ViewElements(idxs), X)
end


# Synonyms in plot recipe
const StochasticProfileVal = Union{Val{:StochasticProfile},Val{:StochasticProfiles}}
const StochasticProfile2DVal = Union{Val{:StochasticProfile2D},Val{:StochasticProfiles2D},Val{:Stochastic2DProfile},Val{:Stochastic2DProfiles}}

## Plot recipe passthrough for replacing `StochasticProfileLikelihoodPlot`, not used yet though
for F in [Symbol("plot"), Symbol("plot!")]
    @eval RecipesBase.$F(R::MultistartResults, V::StochasticProfileVal, args...; pnames=pnames(R), kwargs...) = RecipesBase.$F(R.FinalPoints, R.FinalObjectives, V, args...; pnames, kwargs...)
    @eval RecipesBase.$F(R::MultistartResults, ind::Int, V::StochasticProfileVal, args...; pnames=pnames(R), kwargs...) = RecipesBase.$F(R.FinalPoints, R.FinalObjectives, ind, V, args...; pnames, kwargs...)
end
@recipe function StochasticProfileLikelihoodPlot(Points::AbstractVector{<:AbstractVector}, Likelihoods::AbstractVector{<:Number}, V::StochasticProfileVal)
                                # Nbins::Int=clamp(Int(ceil(sqrt(length(Likelihoods)))),3,100), BiLog::Bool=true, Trafo::Function=(BiLog ? InformationGeometry.BiLog : identity))
    Nbins = get(plotattributes, :Nbins, Int(ceil(clamp(0.5*(length(Likelihoods)^(1/length(Points[1]))),3,100))))
    Extremizer = get(plotattributes, :Extremizer, findmax)
    DoBiLog = get(plotattributes, :BiLog, true)
    Trafo = get(plotattributes, :Trafo, (DoBiLog ? BiLog : identity))
    TrafoName, TrafoNameEnd = GetTrafoNames(Trafo)
    pnames = get(plotattributes, :pnames, CreateSymbolNames(length(Points[1])))
    legend --> false
    layout --> length(pnames) + 1 + (length(pnames) ≤ 3 ? 1 : 0)
    for i in eachindex(pnames)
        @series begin
            subplot := i
            xlabel := string(pnames[i])
            ylabel := TrafoName * "Objective" * TrafoNameEnd
            Nbins := Nbins
            BiLog := BiLog
            Trafo := Trafo
            Extremizer := Extremizer
            Points, Likelihoods, i, V
        end
    end
    @series begin
        subplot := length(pnames) + 1
        xlabel --> "Run index (sorted)"
        ylabel --> TrafoName * "-2*Objective" * TrafoNameEnd
        label --> "Waterfall"
        1:length(Likelihoods), Trafo.(-2 .*Likelihoods)
    end
    if length(pnames) ≤ 3
        @series begin
            st := :scatter
            subplot := length(pnames) + 2
            c --> :viridis
            colorbar --> true
            zcolor --> Trafo.(2 .*Likelihoods)
            msw --> 0
            xlabel --> pnames[1]
            ylabel --> pnames[2]
            zlabel --> (length(pnames) ≥ 3 ? pnames[3] : "")
            label --> "log-likelihood"
            Points
        end
        @series begin
            st := :scatter
            subplot := length(pnames) + 2
            mc --> :red
            msw --> 0
            label --> "Best"
            maxind = findmax(Likelihoods)[2]
            Points[maxind:maxind]
        end
    end
    # RecipesBase.plot([P; AddedPlots]...; layout=length(P)+length(AddedPlots), kwargs...)
end
@recipe function f(Points::AbstractVector{<:AbstractVector}, Likelihoods::AbstractVector{<:Number}, ind::Int, ::StochasticProfileVal)
                # Nbins=clamp(Int(ceil(sqrt(length(Likelihoods)))),3,100), Extremizer=maximum, BiLog=true, Trafo=(BiLog ? InformationGeometry.BiLog : identity), OffsetResults=false)
    Nbins = get(plotattributes, :Nbins, Int(ceil(clamp(0.5*(length(Likelihoods)^(1/length(Points[1]))),3,100))))
    Extremizer = get(plotattributes, :Extremizer, findmax)
    DoBiLog = get(plotattributes, :DoBiLog, true)
    Trafo = get(plotattributes, :Trafo, (DoBiLog ? BiLog : identity))
    TrafoName, TrafoNameEnd = GetTrafoNames(Trafo)
    OffsetResults = get(plotattributes, :OffsetResults, false)
    pnames = get(plotattributes, :pnames, CreateSymbolNames(length(Points[1])))
    pval = get(plotattributes, :pval, getindex.(Points, ind))
    pBins = get(plotattributes, :pBins, collect(HistoBins(pval, Nbins)))

    _, Res, ResPoints = _GetStochasticProfile(Points, Likelihoods, ind; nbins=Nbins, Extremizer, pval=getindex.(Points, ind), pBins, OffsetResults)
    res = Trafo.(Res)
    HitsPerBin = float.(CountInBin(pBins, Points, ind));  HitsPerBin ./= maximum(HitsPerBin)
    
    @series begin
        st := :bar
        alpha --> max.(0,HitsPerBin)
        bar_width --> diff(pBins)
        lw --> 0.5
        xlabel := pnames[ind]
        ylabel := TrafoName * "SPLA" * TrafoNameEnd
        label --> "Conditional Objectives"
        CenteredVec(pBins), res
    end
    if Extremizer === findmax   @series begin
        st := :vline
        line --> :dash
        lc --> :red
        lw --> 1.5
        xlabel := pnames[ind]
        ylabel := TrafoName * "SPLA" * TrafoNameEnd
        label --> "Best Objective"
        [ResPoints[Extremizer(-res)[2]][ind]]
    end end
end


## Recipe Passthrough
for F in [Symbol("plot"), Symbol("plot!")]
    @eval RecipesBase.$F(R::MultistartResults, V::StochasticProfile2DVal, args...; pnames=pnames(R), kwargs...) = RecipesBase.$F(R.FinalPoints, R.FinalObjectives, V, args...; pnames, kwargs...)
    @eval RecipesBase.$F(R::MultistartResults, idxs, V::StochasticProfile2DVal, args...; pnames=pnames(R), kwargs...) = RecipesBase.$F(R.FinalPoints, R.FinalObjectives, idxs, V, args...; pnames, kwargs...)
    @eval RecipesBase.$F(Points::AbstractVector{<:AbstractVector}, Likelihoods::AbstractVector{<:Number}, V::StochasticProfile2DVal, args...; kwargs...) = RecipesBase.$F(Points, Likelihoods, OrderedIndCombs2D(1:length(Points[1])), V, args...; kwargs...)
end
## 2D stochastic profiles
@recipe function f(Points::AbstractVector{<:AbstractVector}, Likelihoods::AbstractVector{<:Number}, idxs::AbstractVector{<:Int}, ::StochasticProfile2DVal;
                nxbins = Int(ceil(clamp(0.5*(length(Likelihoods)^(1/length(Points[1]))),3,100))),
                nybins = Int(ceil(clamp(0.5*(length(Likelihoods)^(1/length(Points[1]))),3,100))),
                pnames = CreateSymbolNames(length(Points[1])),
                Extremizer = findmax, BiLog = true, Trafo = (BiLog ? InformationGeometry.BiLog : identity), OffsetResults = false)
    @assert length(idxs) == 2 && allunique(idxs) && all(1 .≤ idxs .≤ length(Points[1]))
    
    pxval = get(plotattributes, :pxval, getindex.(Points, idxs[1]))
    pyval = get(plotattributes, :pyval, getindex.(Points, idxs[2]))
    pxBins = get(plotattributes, :pxBins, collect(HistoBins(pxval, nxbins)))
    pyBins = get(plotattributes, :pyBins, collect(HistoBins(pyval, nybins)))

    _, _, Res, ResPoints = _GetStochastic2DProfile(Points, Likelihoods, idxs; Extremizer, OffsetResults, nxbins, nybins, pxval, pyval, pxBins, pyBins)
    res = Trafo.(Res)
    HitsPerBin = float.(CountInBin(pxBins, pyBins, Points, idxs));  HitsPerBin ./= maximum(HitsPerBin)

    legend --> false
    @series begin
        st := :heatmap
        color --> :viridis
        colorbar := false
        xlabel := pnames[idxs[1]]
        ylabel := pnames[idxs[2]]
        # alpha --> max.(0,HitsPerBin)
        lw --> 0.5
        label --> ApplyTrafoNames("-SPLA", Trafo)
        # heatmap order needs transposed matrix and xy
        CenteredVec(pxBins), CenteredVec(pyBins), -res'
    end
    if Extremizer === findmax   @series begin
        st := :scatter
        c := :red
        mc := :red
        xlabel := pnames[idxs[1]]
        ylabel := pnames[idxs[2]]
        ms --> 1.5
        label --> "Best Objective"
        [ResPoints[Extremizer(-res)[2]][idxs]]
    end end
end
@recipe function f(Points::AbstractVector{<:AbstractVector}, Likelihoods::AbstractVector{<:Number}, Combos::AbstractVector{<:AbstractVector{<:Int}}, V::StochasticProfile2DVal)
    @assert allunique(Combos) && 2 ≤ ConsistentElDims(Combos) ≤ 3
    length(Points[1]) == 2 && return Points, Likelihoods, [1,2], V
    layout --> length(Combos)
    length(Combos) > 1 && (size --> (1500,1500))
    for (i,inds) in enumerate(Combos)
        @series begin
            subplot := i
            Points, Likelihoods, inds, V
        end
    end
end


const StochasticProfile2DValLower = Union{Val{:Stochastic2DProfilesLowerTriangular},Val{:Stochastic2DProfilesLower}}

for F in [Symbol("plot"), Symbol("plot!")]
    @eval RecipesBase.$F(R::MultistartResults, V::StochasticProfile2DValLower, args...; pnames=pnames(R), kwargs...) = RecipesBase.$F(R.FinalPoints, R.FinalObjectives, V, args...; pnames, kwargs...)
    @eval RecipesBase.$F(R::MultistartResults, idxs, V::StochasticProfile2DValLower, args...; pnames=pnames(R), kwargs...) = RecipesBase.$F(R.FinalPoints, R.FinalObjectives, idxs, V, args...; pnames, kwargs...)
    @eval RecipesBase.$F(Points::AbstractVector{<:AbstractVector}, Likelihoods::AbstractVector{<:Number}, V::StochasticProfile2DValLower, args...; kwargs...) = RecipesBase.$F(Points, Likelihoods, 1:length(Points[1]), V, args...; kwargs...)
end
@recipe function f(Points::AbstractVector{<:AbstractVector}, Likelihoods::AbstractVector{<:Number}, paridxs::AbstractVector{<:Int}, V::StochasticProfile2DValLower; 
                IndMat=[[x,y] for y in paridxs, x in paridxs], idxs=vec(IndMat), comparison=Base.isless)
    @assert IndMat isa AbstractMatrix{<:AbstractVector{<:Int}}
    @assert idxs isa AbstractVector{<:AbstractVector{<:Int}}
    @assert allunique(idxs) && ConsistentElDims(idxs) == 2 
    @assert all(1 .≤ getindex.(idxs,1) .≤ length(Points[1])) && all(1 .≤ getindex.(idxs,2) .≤ length(Points[1]))
    
    size --> (1500,1500)
    layout --> (length(paridxs)-1, length(paridxs)-1)

    k = 0
    for i in 2:length(paridxs), j in 1:(length(paridxs)-1)
        inds = IndMat[i,j];     k += 1
        if comparison(j,i)
            @series begin
                subplot := k
                Points, Likelihoods, inds, Val(:Stochastic2DProfiles)
            end
        else
            @series begin
                subplot := k
                framestyle := :none
                label := nothing
                alpha := 0
                Points[1][inds[1:1]], Points[1][inds[2:2]]
            end
        end
    end
end

function LowerTriangular2DSPLA(Points::AbstractVector{<:AbstractVector}, Likelihoods::AbstractVector{<:Number}, paridxs::AbstractVector{<:Int}=1:length(Points[1]); 
                                IndMat::AbstractMatrix{<:AbstractVector{<:Int}}=[[x,y] for y in paridxs, x in paridxs],
                                idxs::AbstractVector{<:AbstractVector{<:Int}}=vec(IndMat), comparison::Function=Base.isless, size=(1500,1500), kwargs...)
    @assert allunique(idxs) && ConsistentElDims(idxs) == 2 
    @assert all(1 .≤ getindex.(idxs,1) .≤ length(Points[1])) && all(1 .≤ getindex.(idxs,2) .≤ length(Points[1]))
    # RecipesBase.plot([RecipesBase.plot(Points, Likelihoods, inds, Val(:Stochastic2DProfiles); kwargs...) for inds in idxs]...; layout=length(idxs), size)
    Plts = []
    for i in 2:length(paridxs), j in 1:(length(paridxs)-1)
        inds = IndMat[i,j]
        if comparison(j,i)
            push!(Plts, RecipesBase.plot(Points, Likelihoods, inds, Val(:Stochastic2DProfiles); kwargs...))
        else
            push!(Plts, RecipesBase.plot(; framestyle = :none))
        end
    end;    RecipesBase.plot(Plts...; layout=(length(paridxs)-1, length(paridxs)-1), size) |> display
end
LowerTriangular2DSPLA(R::MultistartResults, args...; kwargs...) = LowerTriangular2DSPLA(R.FinalPoints, R.FinalObjectives, args...; pnames=pnames(R), kwargs...)
