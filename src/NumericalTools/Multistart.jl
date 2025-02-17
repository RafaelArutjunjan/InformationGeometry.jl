

SOBOL.SobolSeq(C::HyperCube, maxval::Real=1e5; seed::Int=rand(1000:15000), N::Int=100) = SOBOL.skip(SOBOL.SobolSeq(clamp(C.L, -maxval*ones(length(C)), maxval*ones(length(C))), clamp(C.U, -maxval*ones(length(C)), maxval*ones(length(C)))), seed; exact=true)
SobolGenerator(args...; kwargs...) = (S=SOBOL.SobolSeq(args...; kwargs...);    (SOBOL.next!(S) for i in 1:Int(1e10)))
GenerateSobolPoints(args...; N::Int=100, kwargs...) = (S=SOBOL.SobolSeq(args...; N, kwargs...);    [SOBOL.next!(S) for i in 1:N])

function MakeMultistartDomain(Pdim::Int, ProspectiveDom::Nothing, maxval::Real=1e5; verbose::Bool=true)
    verbose && @info "No MultistartDomain given, choosing default cube with maxval=$maxval"
    FullDomain(Pdim, maxval)
end
function MakeMultistartDomain(Pdim::Int, ProspectiveDom::HyperCube, maxval::Real=1e5; verbose::Bool=true)
    # clamp ProspectiveDom to finite size
    intersect(ProspectiveDom, FullDomain(length(ProspectiveDom), maxval))
end

MultistartFit(DM::AbstractDataModel; kwargs...) = MultistartFit(DM, Predictor(DM); kwargs...)
MultistartFit(DM::AbstractDataModel, M::ModelMap; MultistartDomain::HyperCube=Domain(M), kwargs...) = MultistartFit(DS, MultistartDomain; MultistartDomain, kwargs...)
function MultistartFit(DM::AbstractDataModel, M::ModelOrFunction; maxval::Real=1e5, MultistartDomain::Union{Nothing,HyperCube}=nothing, verbose::Bool=true, kwargs...)
    Dom = MakeMultistartDomain(pdim(DM), MultistartDomain, maxval; verbose)
    MultistartFit(DM, Dom; MultistartDomain=Dom, maxval, verbose, kwargs...)
end
# Create PointGenerator and drop model again
function MultistartFit(DM::AbstractDataModel, MultistartDom::HyperCube; MultistartDomain::HyperCube=MultistartDom, N::Int=100, seed::Int=rand(1000:15000), resampling::Bool=true, maxval::Real=1e5, kwargs...)
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
function MultistartFit(CostFunction::Function, startp::AbstractVector{<:Number}; MultistartDomain::Union{HyperCube,Nothing}=nothing, maxval::Real=1e5, N::Int=100, 
                            seed::Int=rand(1000:15000), resampling::Bool=true, verbose::Bool=true, kwargs...)
    Dom = MakeMultistartDomain(length(startp), MultistartDomain, maxval; verbose)
    InitialPointGen = (resampling ? SOBOL.SobolSeq : GenerateSobolPoints)(Dom, maxval; N, seed)
    # Drop startp
    MultistartFit(CostFunction, InitialPointGen; MultistartDomain=Dom, maxval, N, seed, resampling, verbose, kwargs...)
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
function MultistartFit(CostFunction::Function, InitialPointGen::Union{AbstractVector{<:AbstractVector{<:Number}}, Distributions.MultivariateDistribution, Base.Generator, SOBOL.AbstractSobolSeq}; showprogress::Bool=true, N::Int=100, maxval=1e5, 
                                        DM::Union{Nothing,AbstractDataModel}=nothing, LogPriorFn::Union{Nothing,Function}=nothing, resampling::Bool=!(InitialPointGen isa AbstractVector), pnames::AbstractVector{<:StringOrSymb}=Symbol[], TransformSample::Function=identity,
                                        MultistartDomain::Union{HyperCube,Nothing}=nothing, parallel::Bool=true, Robust::Bool=false, TryCatchOptimizer::Bool=true, TryCatchCostFunc::Bool=true, p::Real=2, timeout::Real=120, verbose::Bool=false, 
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
        (parallel ?  progress_pmap : progress_map)(OptimFunc, InitialPoints; progress=Progress(length(InitialPoints), desc="Multistart fitting... ", showspeed=true))
    else
        (parallel ?  pmap : map)(OptimFunc, InitialPoints)
    end
    FinalPoints = Full ? GetMinimizer.(Res) : Res

    FinalObjectives = (parallel ? pmap : map)(LogLikeFunc, FinalPoints)
    
    if Full
        Iterations = GetIterations.(Res)
        # By internal optimizer criterion:
        Converged = HasConverged.(Res; verbose=false)
        PNames = length(pnames) == 0 ? CreateSymbolNames(length(FinalPoints[1])) : pnames
        MultistartResults(FinalPoints, InitialPoints, FinalObjectives, InitialObjectives, Iterations, Converged, PNames, meth, seed, MultistartDomain, SaveFullOptimizationResults ? Res : nothing; verbose)
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


struct MultistartResults <: AbstractMultiStartResults
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
    function MultistartResults(
            FinalPoints::AbstractVector{<:AbstractVector{<:Number}},
            InitialPoints::AbstractVector{<:AbstractVector{<:Number}},
            FinalObjectives::AbstractVector{<:Number},
            InitialObjectives::AbstractVector{<:Number},
            Iterations::AbstractVector{<:Int},
            Converged::AbstractVector{<:Bool},
            pnames::AbstractVector{<:StringOrSymb},
            meth,
            seed::Union{Int, Nothing}=nothing,
            MultistartDomain::Union{Nothing,HyperCube}=nothing,
            FullOptimResults=nothing; 
            verbose::Bool=true
        )
        @assert length(FinalPoints) == length(InitialPoints) == length(FinalObjectives) == length(InitialObjectives) == length(Iterations)
        @assert ConsistentElDims(FinalPoints) == length(pnames)
        OptimMeth = isnothing(meth) ? LsqFit.LevenbergMarquardt() : meth
        
        # Convert possible NaNs in FinalObjectives to -Inf to avoid problems in sorting NaNs
        nans = 0
        for i in eachindex(FinalObjectives)
            (!isfinite(FinalObjectives[i]) || all(isinf, FinalPoints[i])) && (FinalObjectives[i] = -Inf;    isfinite(InitialObjectives[i]) && (nans += 1))
        end
        if verbose
            if all(isinf, FinalObjectives)
                if any(isfinite, InitialObjectives)
                    @warn "ALL multistart optimizations with $(typeof(OptimMeth)) crashed! Most likely the options supplied to the optimizer were wrong. Automatic catching of optimizer errors can be disabled with kwarg TryCatchOptimizer=false."
                else
                    @warn "ALL multistarts failed on initial evaluation! Most likely the specified CostFunction was ill-defined and / or initial points were ill-chosen (Domain too large?) or too few. Automatic catching of CostFunction errors can be disabled with kwarg TryCatchCostFunc=false."
                end
            elseif nans > 0
                @info "$nans runs crashed during multistart optimization with $(typeof(OptimMeth))."
            end
        end

        Perm = sortperm(FinalObjectives; rev=true)
        new(FinalPoints[Perm], InitialPoints[Perm], FinalObjectives[Perm], InitialObjectives[Perm], Iterations[Perm], Converged[Perm], Symbol.(pnames), OptimMeth, seed, MultistartDomain, isnothing(FullOptimResults) ? nothing : FullOptimResults[Perm])
    end
end

function Base.vcat(R1::MultistartResults, R2::MultistartResults)
    @assert length(R1.pnames) == length(R2.pnames)
    R1.pnames != R2.pnames && @warn "Using pnames from first MultistartResults object."
    R1.OptimMeth != R2.OptimMeth && @warn "Combining results from different optimizers."

    MultistartResults(vcat(R1.FinalPoints, R2.FinalPoints), vcat(R1.InitialPoints, R2.InitialPoints),
        vcat(R1.FinalObjectives, R2.FinalObjectives), vcat(R1.InitialObjectives, R2.InitialObjectives), vcat(R1.Iterations, R2.Iterations), vcat(R1.Converged, R2.Converged),
        R1.pnames, R1.OptimMeth != R2.OptimMeth ? [R1.OptimMeth, R2.OptimMeth] : R1.OptimMeth, nothing, R1.MultistartDomain,
        (!isnothing(R1.FullOptimResults) && !isnothing(R2.FullOptimResults) ? vcat(R1.FullOptimResults,R2.FullOptimResults) : nothing)
    )
end


Base.length(R::MultistartResults) = length(R.FinalObjectives)
Base.firstindex(R::MultistartResults) = firstindex(R.FinalObjectives)
Base.lastindex(R::MultistartResults) = length(R.FinalObjectives)

MLE(R::MultistartResults) = R.FinalPoints[1]
pnames(R::MultistartResults) = R.pnames .|> string
Pnames(R::MultistartResults) = R.pnames
Domain(R::MultistartResults) = R.MultistartDomain


"""
    WaterfallPlot(R::MultistartResults; BiLog::Bool=true, MaxValue::Real=3000, StepTol::Real=1e-3, kwargs...)
Shows Waterfall plot for the given results of MultistartFit.
`StepTol` is used to decide which difference of two neighbouring values in the Waterfall plot constitutes a step. `StepTol=0` deactivates step marks.
`MaxValue` is used to set threshold for ignoring points whose cost function after optimization is too large compared with best optimum.
`DoBiLog=false` disables logarithmic scale for cost function.
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
    (@assert 1 ≤ ymaxind ≤ length(R) && StepTol > 0);       F = -R.FinalObjectives
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
    (@assert 1 ≤ ymaxind ≤ length(R) && StepTol > 0);  F=-R.FinalObjectives
    FirstStepInd = findfirst(i->isfinite(F[i+1]) && abs(F[i+1]-F[i]) > StepTol, 1:ymaxind-1)
    isnothing(FirstStepInd) ? ymaxind : FirstStepInd
end


RecipesBase.@recipe f(R::MultistartResults, S::Symbol=:Waterfall) = R, Val(S)
# kwargs BiLog, StepTol, MaxValue, MaxInd, ColorIterations
RecipesBase.@recipe function f(R::MultistartResults, ::Val{:Waterfall})
    DoBiLog = get(plotattributes, :BiLog, true)
    MaxValue = get(plotattributes, :MaxValue, BiExp(8))
    ColorIterations = get(plotattributes, :ColorIterations, true)
    @assert MaxValue ≥ 0
    Fin = (DoBiLog ? BiLog : identity)(-R.FinalObjectives)
    # Cut off results with difference to lowest optimum greater than MaxValue
    ymaxind = get(plotattributes, :MaxInd, (Q=findlast(x->isfinite(x) && abs(x-Fin[1]) < (DoBiLog ? BiLog(MaxValue) : MaxValue), Fin);   isnothing(Q) ? length(Fin) : Q))
    xlabel --> "Run (sorted by cost function result)"
    ylabel --> (DoBiLog ? "BiLog(Final Cost Value)" : "Final Cost Value")
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
        (DoBiLog ? BiLog : identity)(-R.InitialObjectives[1:ymaxind])
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
    dfs = [(DoBiLog ? BiLog : identity)(reduce(vcat, R.FinalPoints[step])) for step in Steps]
    Xvals = 1.0:Spacer*length(dfs):Spacer*length(dfs)*length(pnames) |> collect

    xtick --> (Xvals .+ Spacer*0.5(length(dfs)-1), pnames)
    ylabel --> (DoBiLog ? "BiLog(Parameter Value)" : "Parameter Value")
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