

SOBOL.SobolSeq(C::HyperCube, Maxval::Real=1e15; maxval::Real=Maxval, seed::Int=rand(1000:15000), N::Int=100) = SOBOL.skip(SOBOL.SobolSeq(clamp(C.L, -maxval*ones(length(C)), maxval*ones(length(C))), clamp(C.U, -maxval*ones(length(C)), maxval*ones(length(C)))), seed; exact=true)
SobolGenerator(args...; kwargs...) = (S=SOBOL.SobolSeq(args...; kwargs...);    (SOBOL.next!(S) for i in 1:Int(1e10)))
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
        Converged = HasConverged.(Res; verbose=false)
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
            Iterations::AbstractVector{<:Int},
            Converged::AbstractVector{<:Bool},
            pnames::AbstractVector{<:StringOrSymb},
            meth,
            seed::Union{Int,Nothing}=nothing,
            MultistartDomain::Union{Nothing,HyperCube}=nothing,
            FullOptimResults=nothing,
            Meta=nothing; 
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
        new(FinalPoints[Perm], InitialPoints[Perm], FinalObjectives[Perm], InitialObjectives[Perm], Iterations[Perm], Converged[Perm], Symbol.(pnames), OptimMeth, seed, MultistartDomain, isnothing(FullOptimResults) ? nothing : FullOptimResults[Perm], Meta)
    end
end

function Base.vcat(R1::MultistartResults, R2::MultistartResults)
    @assert length(R1.pnames) == length(R2.pnames)
    R1.pnames != R2.pnames && @warn "Using pnames from first MultistartResults object."
    R1.OptimMeth != R2.OptimMeth && @warn "Combining results from different optimizers."

    MultistartResults(vcat(R1.FinalPoints, R2.FinalPoints), vcat(R1.InitialPoints, R2.InitialPoints),
        vcat(R1.FinalObjectives, R2.FinalObjectives), vcat(R1.InitialObjectives, R2.InitialObjectives), vcat(R1.Iterations, R2.Iterations), vcat(R1.Converged, R2.Converged),
        R1.pnames, R1.OptimMeth != R2.OptimMeth ? [R1.OptimMeth, R2.OptimMeth] : R1.OptimMeth, nothing, R1.MultistartDomain,
        (!isnothing(R1.FullOptimResults) && !isnothing(R2.FullOptimResults) ? vcat(R1.FullOptimResults,R2.FullOptimResults) : nothing),
        (!isnothing(R1.Meta) && !isnothing(R2.Meta) ? vcat(R1.Meta,R2.Meta) : nothing),
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
function RecipesBase.plot(R::MultistartResults, S::Symbol=(isnothing(R.Meta) ? :Waterfall : :StochasticProfile), args...; kwargs...)
    S === :StochasticProfile && return StochasticProfileLikelihoodPlot(R, args...; kwargs...)
    RecipesBase.plot(R, Val(S), args...; kwargs...)
end
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


GetNsingleFromTargetTime(DM::AbstractDataModel, args...; kwargs...) = GetNsingleFromTargetTime(loglikelihood(DM), MLE(DM), args...; kwargs...)
function GetNsingleFromTargetTime(L::Function, startp::AbstractVector, TargetTime::Real=60; minval=2, maxval=1000, verbose::Bool=true)
    L(startp);  Tsingle = @elapsed L(startp)
    N = TargetTime / Tsingle |> floor |> Int
    Nsingle = clamp(N^(1/length(startp)), minval, maxval) |> floor |> Int
    verbose && @info "Single evaluation took $(round(Tsingle; sigdigits=3))s, suggesting approximately N=$N samples in total (Nsingle=$Nsingle, $Nsingle^$(length(startp))=$(Nsingle^length(startp))) to fill allotted $(TargetTime)s."
    Nsingle
end


HistoBins(X::AbstractVector, Bins::Int=Int(ceil(sqrt(length(X)))); nbins::Int=Bins) = range(Measurements.value.(extrema(X))...; length=nbins+1)
"""
   StochasticProfileLikelihood(DM::AbstractDataModel, C::HyperCube=Domain(Predictor(DM)); TargetTime=60, Nsingle::Int=5, N::Int=Nsingle^length(C), nbins::Int=4Nsingle, parallel::Bool=true, maxval::Real=1e5, DoBiLog::Bool=true, kwargs...)
Samples the likelihood over the parameter space, splits the value range of each parameter into bins and visualizes the best observed likelihood value from all samples with the respective parameter value in this bin.
Therefore, it gives a coarse-grained global approximation to the profile likelihood, approaching the true profile likelihood in the large sample limit as `N` ⟶ ∞, which is however much cheaper to compute as no re-optimization of nuisance parameters is required.
This can already give hints for good candidates for initial parameter values from which to start optimization and conversely also illustrate parts of the parameter ranges which can be excluded from multistart fitting due to their consistently unsuitable likelihood values.

The results of this sampling are saved in the `FinalPoints` and `FinalObjectives` fields of a `MultistartResults` object.

The `TargetTime` kwarg can be used to choose the number of samples such that the sampling is expected to require approximately the allotted time in seconds.
"""
function StochasticProfileLikelihood(DM::AbstractDataModel, C::HyperCube=Domain(Predictor(DM)); TargetTime::Real=60, Nsingle::Int=GetNsingleFromTargetTime(DM, TargetTime), N::Int=Nsingle^length(C), 
                                                        nbins::Int=clamp(Nsingle,4,100), maxval::Real=1e5, Domain::HyperCube=C∩FullDomain(pdim(DM),maxval), kwargs...)
   Points = GenerateSobolPoints(Domain; N, maxval)
   StochasticProfileLikelihood(DM, Points; Domain, nbins, kwargs...)
end
function StochasticProfileLikelihood(DM::AbstractDataModel, Points::AbstractVector{<:AbstractVector}; LogLikelihoodFn::Function=loglikelihood(DM), parallel::Bool=true, pnames::AbstractVector{<:AbstractString}=string.(pnames(DM)), kwargs...)
   @info "Starting $(length(Points)) samples of the log-likelihood."
   Likelihoods = (parallel ? progress_pmap : progress_map)(LogLikelihoodFn, Points; progress=Progress(length(Points); desc="Sampling Parameter Space... "*(parallel ? "(parallel, $(nworkers()) workers) " : ""), dt=0.2, showspeed=true))
   StochasticProfileLikelihood(Points, Likelihoods; pnames, kwargs...)
end
# Construct MultistartResults and call Plot
function StochasticProfileLikelihood(Points::AbstractVector{<:AbstractVector}, Likelihoods::AbstractVector{<:Number}; plot::Bool=isloaded(:Plots), Domain::Union{HyperCube,Nothing}=nothing, pnames::AbstractVector{<:AbstractString}=CreateSymbolNames(length(Points[1])), kwargs...)
   R = MultistartResults(Points, [eltype(Points[1])[] for i in eachindex(Points)], Likelihoods, fill(Inf, length(Likelihoods)), zeros(Int, length(Likelihoods)), falses(length(Likelihoods)), pnames, nothing, nothing, Domain, nothing, :SampledLikelihood)
   plot && display(StochasticProfileLikelihoodPlot(R; kwargs...))
   R
end

## Plot results:
StochasticProfileLikelihoodPlot(R::MultistartResults, ind::Int; kwargs...) = (@assert R.Meta === :SampledLikelihood;  StochasticProfileLikelihoodPlot(R.FinalPoints, R.FinalObjectives, ind; xlabel=string(pnames(R)[ind]), kwargs...))
StochasticProfileLikelihoodPlot(R::MultistartResults; kwargs...) = (@assert R.Meta === :SampledLikelihood;  StochasticProfileLikelihoodPlot(R.FinalPoints, R.FinalObjectives; pnames=string.(pnames(R)), kwargs...))
# Collective
function StochasticProfileLikelihoodPlot(Points::AbstractVector{<:AbstractVector}, Likelihoods::AbstractVector{<:Number}; nbins::Int=clamp(Int(ceil(sqrt(length(Likelihoods)))),4,100), DoBiLog::Bool=true, Trafo::Function=(DoBiLog ? BiLog : identity), Extremizer::Function=maximum,
                                pnames::AbstractVector{<:AbstractString}=CreateSymbolNames(length(Points[1])), legend=false, OffsetResults::Bool=false, kwargs...)
    P = [StochasticProfileLikelihoodPlot(Points, Likelihoods, i; nbins, DoBiLog, Trafo, Extremizer, xlabel=string(pnames[i]), legend, OffsetResults) for i in eachindex(pnames)]
    AddedPlots = []
    push!(AddedPlots, RecipesBase.plot(1:length(Likelihoods), -Trafo.(Likelihoods); xlabel="Run index (sorted)", ylabel=(DoBiLog ? "BiLog(" : "")*"Objective value"*(DoBiLog ? ")" : ""), label="Waterfall", legend))
    if length(pnames) ≤ 3
        SP = RecipesBase.plot(Points; st=:scatter, zcolor=Trafo.(Likelihoods), msw=0, xlabel=pnames[1], ylabel=pnames[2], zlabel=(length(pnames) ≥ 3 ? pnames[3] : ""), c=:viridis, label="log-likelihood", legend, colorbar=true)
        if Extremizer === maximum 
            maxind = findmax(Likelihoods)[2]
            RecipesBase.plot!(SP, Points[maxind:maxind]; st=:scatter, color=:red, msw=0, label="Best")
        end
        push!(AddedPlots, SP)
    end
    RecipesBase.plot([P; AddedPlots]...; layout=length(P)+length(AddedPlots), kwargs...)
end
# Individual Parameter
function StochasticProfileLikelihoodPlot(Points::AbstractVector{<:AbstractVector}, Likelihoods::AbstractVector{<:Number}, ind::Int; nbins::Int=clamp(Int(ceil(sqrt(length(Likelihoods)))),4,100), Extremizer::Function=maximum, 
                                 DoBiLog::Bool=true, Trafo::Function=(DoBiLog ? BiLog : identity), xlabel="Parameter $ind", OffsetResults::Bool=false, kwargs...)
   pval = getindex.(Points, ind);   pBins = HistoBins(pval, nbins);   keep = falses(length(pval), length(pBins)-1)
   for i in axes(keep,2)
      keep[:, i] .= pBins[i] .≤ pval .< pBins[i+1]
   end
   MaxLike = OffsetResults ? Extremizer(Trafo.(Likelihoods)) : 0
   res = [(try Extremizer(Trafo.(@view Likelihoods[col]).-MaxLike) catch; -Inf end) for col in eachcol(keep)]
   HitsPerBin = Float64[sum(col) for col in eachcol(keep)];  HitsPerBin ./= maximum(HitsPerBin)
   Plt = RecipesBase.plot((@views (pBins[1:end-1] .+ pBins[2:end]) ./ 2), -res; st=:bar, alpha=max.(0,HitsPerBin), bar_width=diff(pBins), lw=0.5, xlabel, ylabel=(DoBiLog ? "BiLog(" : "")*"Minimal Objective"*(DoBiLog ? ")" : ""), label="Conditional Objectives", kwargs...)
   Extremizer === maximum && RecipesBase.plot!(Plt, [Points[findmax(Trafo.(Likelihoods))[2]][ind]]; st=:vline, line=:dash, c=:red, lw=1.5, label="Best Objective")
   Plt
end


function FindGoodStart(DM::AbstractDataModel, args...; plot::Bool=false, kwargs...)
    R = StochasticProfileLikelihood(DM, args...; plot, kwargs...)
    R.FinalPoints[findmax(R.FinalObjectives)[2]]
end




## SubspaceProjection plots for sampling results in MultistartResults format
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
                DoBiLog=true, Trafo=(DoBiLog ? BiLog : identity))
    # DoBiLog = get(plotattributes, :BiLog, true);    Trafo = DoBiLog ? BiLog : identity
    color --> :viridis
    zcolor --> Trafo.(@view R.FinalObjectives[FiniteInds])
    colorbar --> false
    xlabel --> Pnames(R)[idxs[1]]
    ylabel --> Pnames(R)[idxs[2]]
    zlabel --> (length(idxs) ≥ 3 ? Pnames(R)[idxs[3]] : "")
    label --> (DoBiLog ? "BiLog(" : "")* "Log-Likelihood" * (DoBiLog ? ")" : "")
    legend --> false
    (@view R.FinalPoints[FiniteInds]), idxs, V
end

@recipe function f(X::AbstractVector{<:AbstractVector}, idxs::AbstractVector{<:Int}, ::Val{:SubspaceProjection})
   @assert 1 ≤ length(idxs) ≤ 3 && allunique(idxs) && all(1 .≤ idxs .≤ ConsistentElDims(X))
   msw := 0
   st := :scatter
   map(ViewElements(idxs), X)
end