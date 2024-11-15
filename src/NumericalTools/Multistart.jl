

SOBOL.SobolSeq(C::HyperCube, maxval::Real=1e5; seed::Int=rand(1000:15000), N::Int=100) = SOBOL.skip(SOBOL.SobolSeq(clamp(C.L, -maxval*ones(length(C)), maxval*ones(length(C))), clamp(C.U, -maxval*ones(length(C)), maxval*ones(length(C)))), seed; exact=true)
SobolGenerator(args...; kwargs...) = (S=SOBOL.SobolSeq(args...; kwargs...);    (SOBOL.next!(S) for i in 1:Int(1e10)))
GenerateSobolPoints(args...; N::Int=100, kwargs...) = (S=SOBOL.SobolSeq(args...; N, kwargs...);    [SOBOL.next!(S) for i in 1:N])


MultistartFit(DM::AbstractDataModel, args...; CostFunction::Function=Negloglikelihood(DM), kwargs...) = MultistartFit(Data(DM), Predictor(DM), LogPrior(DM), args...; CostFunction, pnames=pnames(DM), kwargs...)
MultistartFit(DS::AbstractDataSet, M::ModelMap, LogPriorFn::Union{Nothing,Function}=nothing; MultistartDomain::HyperCube=Domain(M), kwargs...) = MultistartFit(DS, M, LogPriorFn; MultistartDomain, kwargs...)
function MultistartFit(DS::AbstractDataSet, M::ModelOrFunction, LogPriorFn::Union{Nothing,Function}; maxval::Real=1e5, MultistartDomain::Union{Nothing, HyperCube}=nothing, verbose::Bool=true, kwargs...)
    if isnothing(MultistartDomain)
        verbose && @info "No MultistartDomain given, choosing default cube with maxval=$maxval"
        MultistartFit(DS, M, LogPriorFn, FullDomain(pdim(DS, M), maxval); maxval, verbose, kwargs...)
    else
        MultistartFit(DS, M, LogPriorFn, MultistartDomain; maxval, verbose, kwargs...)
    end
end
function MultistartFit(DS::AbstractDataSet, model::ModelOrFunction, LogPriorFn::Union{Nothing,Function}, MultistartDomain::HyperCube; N::Int=100, seed::Int=rand(1000:15000), resampling::Bool=true, maxval::Real=1e5, kwargs...)
    @assert N ≥ 1
    MultistartFit(DS, model, (resampling ? SOBOL.SobolSeq : GenerateSobolPoints)(MultistartDomain, maxval; N, seed), LogPriorFn; N, resampling, seed, kwargs...)
end

"""
    MultistartFit(DM::AbstractDataModel; maxval::Real=1e5, MultistartDomain::HyperCube=FullDomain(pdim(DM), maxval), kwargs...)
    MultistartFit(DS::AbstractDataSet, model::ModelOrFunction, LogPriorFn::Union{Nothing,Function}, MultistartDomain::HyperCube; N::Int=100, resampling::Bool=true, timeout::Real=120, Full=true, parallel::Bool=true, Robust::Bool=true, p::Real=2, kwargs...)
Performs Multistart optimization with `N` starts and timeout of fits after `timeout` seconds.
If `resampling=true`, if likelihood non-finite new initial starts are redrawn until `N` suitable initials are found. 
If `Robust=true`, performs optimization wrt. p-norm according to given kwarg `p`.
For `Full=false`, only the final MLE is returned, otherwise a `MultistartResults` object is returned, which can be further analyzed and plotted.
!!! note
    Any further keyword arguments are passed through to the optimization procedure [InformationGeometry.minimize](@ref) such as tolerances, optimization methods, domain constraints, etc.
"""
function MultistartFit(DS::AbstractDataSet, model::ModelOrFunction, InitialPointGen::Union{AbstractVector{<:AbstractVector{<:Number}}, Base.Generator, SOBOL.AbstractSobolSeq}, LogPriorFn::Union{Nothing,Function}; showprogress::Bool=true,
                                        CostFunction::Union{Nothing,Function}=nothing, N::Int=100, resampling::Bool=!(InitialPointGen isa AbstractVector), pnames::AbstractVector{<:AbstractString}=CreateSymbolNames(pdim(DS,model)),
                                        parallel::Bool=true, Robust::Bool=false, TryCatchOptimizer::Bool=true, TryCatchCostFunc::Bool=true, p::Real=2, timeout::Real=120, verbose::Bool=false, 
                                        meth=((isnothing(LogPriorFn) && DS isa AbstractFixedUncertaintyDataSet) ? nothing : Optim.NewtonTrustRegion()), Full::Bool=true, seed::Union{Int,Nothing}=nothing, kwargs...)
    @assert !Robust || (p > 0 && !TotalLeastSquares)
    @assert resampling ? (InitialPointGen isa Union{Base.Generator,SOBOL.AbstractSobolSeq}) : (InitialPointGen isa AbstractVector)
    
    # +Inf if error during optimization, should rarely happen
    # RobustFunc(θ::AbstractVector{<:Number}, InitialVal::Real) = isfinite(InitialVal) ? (try    RobustFit(DS, model, θ, LogPriorFn; p, timeout, Full, kwargs...)    catch;  fill(-Inf, length(θ))   end) : fill(-Inf, length(θ))
    # Func(θ::AbstractVector{<:Number}, InitialVal::Real) = isfinite(InitialVal) ? (try    InformationGeometry.minimize(DS, model, θ, LogPriorFn; timeout, Full, kwargs...)    catch;  fill(-Inf, length(θ))   end) : fill(-Inf, length(θ))
    RobustFunc(θ::AbstractVector{<:Number}) = RobustFit(DS, model, θ, LogPriorFn; p, timeout, verbose, meth, Full, kwargs...)
    Func(θ::AbstractVector{<:Number}) = InformationGeometry.minimize(DS, model, θ, LogPriorFn; timeout, verbose, meth, Full, kwargs...)
    # Allow for disabling try catch;
    # TotalFunc(θ::AbstractVector{<:Number}) = try    InformationGeometry.TotalLeastSquaresV()    catch;  fill(-Inf, length(θ))   end
    
    TryCatchWrapper(F::Function, Default=-Inf) = x -> try F(x) catch;   Default   end
    LogLikeFunc = (TryCatchCostFunc ? TryCatchWrapper : identity)(isnothing(CostFunction) ? (θ->loglikelihood(DS, model, θ, LogPriorFn)) : Negate(CostFunction))

    TakeFrom(X::Base.Generator) = iterate(X)[1]
    TakeFrom(S::SOBOL.AbstractSobolSeq) = SOBOL.next!(S)
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
    OptimFunc = TryCatchOptimizer ? (Robust ? TryCatchWrapper(RobustFunc,fill(-Inf, length(InitialPoints[1]))) : TryCatchWrapper(Func,fill(-Inf, length(InitialPoints[1])))) : (Robust ? RobustFunc : Func)
    Res = if showprogress
        (parallel ?  progress_pmap : progress_map)(OptimFunc, InitialPoints; progress=Progress(length(InitialPoints), desc="Multistart fitting... ", showspeed=true))
    else
        (parallel ?  pmap : map)(OptimFunc, InitialPoints)
    end
    FinalPoints = Full ? GetMinimizer.(Res) : Res

    FinalObjectives = (parallel ? pmap : map)(LogLikeFunc, FinalPoints)
    
    # Some printing?
    if Full
        Iterations = GetIterations.(Res)
        Perm = sortperm(FinalObjectives; rev=true)
        MultistartResults(FinalPoints[Perm], InitialPoints[Perm], FinalObjectives[Perm], InitialObjectives[Perm], Iterations[Perm], pnames, meth, seed)
    else
        MaxVal, MaxInd = findmax(FinalObjectives)
        GetMinimizer(FinalPoints[MaxInd])
    end
end


struct MultistartResults <: AbstractMultiStartResults
    FinalPoints::AbstractVector{<:AbstractVector{<:Number}}
    InitialPoints::AbstractVector{<:AbstractVector{<:Number}}
    FinalObjectives::AbstractVector{<:Number}
    InitialObjectives::AbstractVector{<:Number}
    Iterations::AbstractVector{<:Int}
    pnames::AbstractVector{<:AbstractString}
    OptimMeth
    seed::Union{Int,Nothing}
    function MultistartResults(
            FinalPoints::AbstractVector{<:AbstractVector{<:Number}},
            InitialPoints::AbstractVector{<:AbstractVector{<:Number}},
            FinalObjectives::AbstractVector{<:Number},
            InitialObjectives::AbstractVector{<:Number},
            Iterations::AbstractVector{<:Int},
            pnames::AbstractVector{<:AbstractString},
            meth,
            seed::Union{Int, Nothing}=nothing
        )
        @assert length(FinalPoints) == length(InitialPoints) == length(FinalObjectives) == length(InitialObjectives) == length(Iterations)
        @assert ConsistentElDims(FinalPoints) == length(pnames)
        @assert issorted(FinalObjectives; rev=true)
        OptimMeth = isnothing(meth) ? LsqFit.LevenbergMarquardt() : meth
        !any(isfinite, FinalObjectives) && @warn "No finite Multistart results! It is likely that inputs to optimizer were unsuitable and thus try-catch was triggered on every run."
        new(FinalPoints, InitialPoints, FinalObjectives, InitialObjectives, Iterations, pnames, OptimMeth, seed)
    end
end


Base.length(R::MultistartResults) = length(R.FinalObjectives)
Base.firstindex(R::MultistartResults) = firstindex(R.FinalObjectives)
Base.lastindex(R::MultistartResults) = length(R.FinalObjectives)

MLE(R::MultistartResults) = R.FinalPoints[1]
pnames(R::MultistartResults) = R.pnames


"""
    WaterfallPlot(R::MultistartResults; BiLog::Bool=true, MaxValue::Real=3000, StepTol::Real=0.01, kwargs...)
Shows Waterfall plot for the given results of MultistartFit.
`StepTol` is used to decide which difference of two neighbouring values in the Waterfall plot constitutes a step. `StepTol=0` deactivates step marks.
`MaxValue` is used to set threshold for ignoring points whose cost function after optimization is too large compared with best optimum.
`DoBiLog=false` disables logarithmic scale for cost function.
"""
WaterfallPlot(R::MultistartResults; kwargs...) = RecipesBase.plot(R, Val(:Waterfall); kwargs...)

"""
    ParameterPlot(R::MultistartResults; st=:dotplot, BiLog::Bool=true, Nsteps::Int=5, StepTol::Real=0.01, MaxValue=3000)
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

# Implement findlast in safe manner so "nothing" is never returned
GetStepInds(R::MultistartResults, ymaxind::Int=findlast(isfinite,R.FinalObjectives); StepTol::Real=0.01) = ((@assert 1 ≤ ymaxind ≤ length(R) && StepTol > 0);  F=-R.FinalObjectives;  [i for i in 1:ymaxind-1 if isfinite(F[i+1]) && abs(F[i+1]-F[i]) > StepTol])

function GetFirstStepInd(R::MultistartResults, ymaxind::Int=findlast(isfinite,R.FinalObjectives); StepTol::Real=0.01)
    (@assert 1 ≤ ymaxind ≤ length(R) && StepTol > 0);  F=-R.FinalObjectives
    FirstStepInd = findfirst(i->isfinite(F[i+1]) && abs(F[i+1]-F[i]) > StepTol, 1:ymaxind-1)
    isnothing(FirstStepInd) ? ymaxind : FirstStepInd
end


RecipesBase.@recipe f(R::MultistartResults, S::Symbol=:Waterfall) = R, Val(S)
# kwargs BiLog, StepTol, MaxValue
RecipesBase.@recipe function f(R::MultistartResults, ::Val{:Waterfall})
    DoBiLog = get(plotattributes, :BiLog, true)
    MaxValue = get(plotattributes, :MaxValue, BiExp(8))
    @assert MaxValue ≥ 0
    Fin = (DoBiLog ? BiLog : identity)(-R.FinalObjectives)
    # Cut off results with difference to lowest optimum greater than MaxValue
    ymaxind = (Q=findlast(x->isfinite(x) && abs(x-Fin[1]) < (DoBiLog ? BiLog(MaxValue) : MaxValue), Fin);   isnothing(Q) ? length(Fin) : Q)
    xlabel --> "Run (sorted by cost function result)"
    ylabel --> (DoBiLog ? "BiLog(Cost function)" : "Cost function")
    title --> "Waterfall plot $(ymaxind)/$(length(Fin))"
    leg --> nothing
    st --> :scatter
    
    ymin, ymax = Fin[1], Fin[ymaxind]
    ylims --> (ydiff=ymax-ymin; (ymin-0.05*ydiff, max(ymax+0.05*ydiff, ymin+0.01)))
    @series begin
        label --> "Finals"
        markersize --> 20/sqrt(ymaxind)
        msw --> 0
        zcolor --> map(x->isfinite(x) ? x : 0, R.Iterations)
        color --> cgrad(:plasma; rev=true)
        Fin[1:ymaxind]
    end
    @series begin
        label --> "Initials"
        marker --> :cross
        (DoBiLog ? BiLog : identity)(-R.InitialObjectives[1:ymaxind])
    end
    # Mark steps with vline, StepTol on linear scale
    StepTol = get(plotattributes, :ShowSteps, 0.01)
    if 0 < StepTol
        Steps = GetStepInds(R, ymaxind; StepTol=StepTol)
        for (i,AfterIter) in enumerate(Steps)
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
    StepTol = get(plotattributes, :StepTol, 0.01)
    pnames = get(plotattributes, :pnames, InformationGeometry.pnames(R))
    Nsteps = get(plotattributes, :Nsteps, 5)
    Spacer = get(plotattributes, :Spacer, 2)

    Fin = -R.FinalObjectives
    ymaxind = (Q=findlast(x->isfinite(x) && abs(x-Fin[1]) < MaxValue, Fin);   isnothing(Q) ? length(Fin) : Q)
    @assert StepTol > 0
    StepInds = GetStepInds(R, ymaxind; StepTol=StepTol)
    AllSteps = vcat([1:StepInds[1]], [StepInds[i]:StepInds[i+1] for i in 1:length(StepInds)-1], [StepInds[end]:ymaxind])
    Steps = @view AllSteps[1:min(Nsteps, length(AllSteps))]
    dfs = [(DoBiLog ? BiLog : identity)(reduce(vcat, R.FinalPoints[step])) for step in Steps]
    Xvals = 1.0:Spacer*length(dfs):Spacer*length(dfs)*length(pnames) |> collect

    xtick --> (Xvals .+ Spacer*0.5(length(dfs)-1), pnames)
    ylabel --> (DoBiLog ? "BiLog(Parameter Value)" : "Parameter Value")
    xlabel --> "Parameter"
    seriestype --> get(plotattributes, :st, :dotplot)
    for i in 1:length(dfs)
        @series begin
            color --> palette(:default)[(((i-1)%15)+1)]
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
