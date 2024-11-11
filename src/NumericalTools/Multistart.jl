

SOBOL.SobolSeq(C::HyperCube, maxval::Real=1e5; N::Int=100) = SOBOL.skip(SOBOL.SobolSeq(clamp(C.L, -maxval*ones(length(C)), maxval*ones(length(C))), clamp(C.U, -maxval*ones(length(C)), maxval*ones(length(C)))), N)
SobolGenerator(args...; N::Int=100) = (S=SOBOL.SobolSeq(args...; N);    (SOBOL.next!(S) for i in 1:Int(1e10)))
GenerateSobolPoints(args...; N::Int=100) = (S=SOBOL.SobolSeq(args...; N);    [SOBOL.next!(S) for i in 1:N])


MultistartFit(DM::AbstractDataModel, args...; CostFunction::Function=Negloglikelihood(DM), kwargs...) = MultistartFit(Data(DM), Predictor(DM), LogPrior(DM), args...; CostFunction, kwargs...)
MultistartFit(DS::AbstractDataSet, M::ModelMap, LogPriorFn::Union{Nothing,Function}=nothing; MultistartDomain::HyperCube=Domain(M), kwargs...) = MultistartFit(DS, M, LogPriorFn; MultistartDomain, kwargs...)
function MultistartFit(DS::AbstractDataSet, M::ModelOrFunction, LogPriorFn::Union{Nothing,Function}; maxval::Real=1e5, MultistartDomain::Union{Nothing, HyperCube}=nothing, kwargs...)
    if isnothing(MultistartDomain)
        @info "No MultistartDomain given, choosing default cube with maxval=$maxval"
        MultistartFit(DS, M, LogPriorFn, FullDomain(pdim(DS, M), maxval); maxval, kwargs...)
    else
        MultistartFit(DS, M, LogPriorFn, MultistartDomain; maxval, kwargs...)
    end
end
function MultistartFit(DS::AbstractDataSet, model::ModelOrFunction, LogPriorFn::Union{Nothing,Function}, MultistartDomain::HyperCube; N::Int=100, resampling::Bool=true, maxval::Real=1e5, kwargs...)
    @assert N ≥ 1
    MultistartFit(DS, model, (resampling ? SOBOL.SobolSeq : GenerateSobolPoints)(MultistartDomain, maxval; N), LogPriorFn; N, resampling, kwargs...)
end

"""
    MultistartFit(DM::AbstractDataModel; maxval::Real=1e5, MultistartDomain::HyperCube=FullDomain(pdim(DM), maxval), kwargs...)
    MultistartFit(DS::AbstractDataSet, model::ModelOrFunction, LogPriorFn::Union{Nothing,Function}, MultistartDomain::HyperCube; N::Int=100, timeout::Real=120, Full=true, parallel::Bool=true, Robust::Bool=true, p::Real=2, kwargs...)
Performs Multistart optimization with `N` starts and timeout of fits after `timeout` seconds.
If `Robust=true`, performs optimization wrt. p-norm according to given kwarg `p`.
For `Full=false`, only the final MLE is returned, otherwise a `MultistarResults` object is returned, which can be further analyzed and plotted.
!!! note
    Any further keyword arguments are passed through to the optimization procedure [InformationGeometry.minimize](@ref) such as tolerances, optimization methods, domain constraints, etc.
"""
function MultistartFit(DS::AbstractDataSet, model::ModelOrFunction, InitialPointGen::Union{AbstractVector{<:AbstractVector{<:Number}}, Base.Generator, SOBOL.AbstractSobolSeq}, LogPriorFn::Union{Nothing,Function}; 
                                        CostFunction::Union{Nothing,Function}=nothing, N::Int=100, resampling::Bool=!(InitialPointGen isa AbstractVector), 
                                        parallel::Bool=true, Robust::Bool=false, p::Real=2, TotalLeastSquares::Bool=false, timeout::Real=120, Full::Bool=true, kwargs...)
    @assert !Robust || (p > 0 && !TotalLeastSquares)
    @assert resampling ? (InitialPointGen isa Union{Base.Generator,SOBOL.AbstractSobolSeq}) : (InitialPointGen isa AbstractVector)
    
    # +Inf if error during optimization, should rarely happen
    # RobustFunc(θ::AbstractVector{<:Number}, InitialVal::Real) = isfinite(InitialVal) ? (try    RobustFit(DS, model, θ, LogPriorFn; p, timeout, Full, kwargs...)    catch;  fill(-Inf, length(θ))   end) : fill(-Inf, length(θ))
    # Func(θ::AbstractVector{<:Number}, InitialVal::Real) = isfinite(InitialVal) ? (try    InformationGeometry.minimize(DS, model, θ, LogPriorFn; timeout, Full, kwargs...)    catch;  fill(-Inf, length(θ))   end) : fill(-Inf, length(θ))
    RobustFunc(θ::AbstractVector{<:Number}) = try    RobustFit(DS, model, θ, LogPriorFn; p, timeout, Full, kwargs...)    catch;  fill(-Inf, length(θ))   end
    Func(θ::AbstractVector{<:Number}) = try    InformationGeometry.minimize(DS, model, θ, LogPriorFn; timeout, Full, kwargs...)    catch;  fill(-Inf, length(θ))   end
    # TotalFunc(θ::AbstractVector{<:Number}) = try    InformationGeometry.TotalLeastSquaresV()    catch;  fill(-Inf, length(θ))   end
    
    TryCatchWrapper(F::Function, Default=-Inf) = x -> try F(x) catch;   Default   end
    LogLikeFunc = TryCatchWrapper(isnothing(CostFunction) ? (θ->loglikelihood(DS, model, θ, LogPriorFn)) : Negate(CostFunction))

    TakeFrom(X::Base.Generator) = iterate(X)[1]
    TakeFrom(S::SOBOL.AbstractSobolSeq) = SOBOL.next!(S)
    # count total sampling attempts when resampling
    InitialPoints, InitialObjectives = if resampling
        InitPoints = Vector{Float64}[]
        InitObjectives = Float64[]
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
    Res = (parallel ?  progress_pmap : progress_map)((Robust ? RobustFunc : Func), InitialPoints; progress=Progress(length(InitialPoints), desc="Multistart fitting... ", showspeed=true))
    GetMinimizerSafe(X::AbstractVector) = X;    GetMinimizerSafe(R) = GetMinimizer(R)
    FinalPoints = Full ? GetMinimizerSafe.(Res) : Res

    FinalObjectives = (parallel ? pmap : map)(LogLikeFunc, FinalPoints)
    
    # Some printing?
    if Full
        GetIterationsSafe(X::AbstractVector) = -Inf;    GetIterationsSafe(R) = GetIterations(R)
        Iterations = GetIterationsSafe.(Res)
        Perm = sortperm(FinalObjectives; rev=true)
        MultistartResults(FinalPoints[Perm], InitialPoints[Perm], FinalObjectives[Perm], InitialObjectives[Perm], Iterations[Perm])
    else
        MaxVal, MaxInd = findmax(FinalObjectives)
        FinalPoints[MaxInd]
    end
end


struct MultistartResults
    FinalPoints::AbstractVector{<:AbstractVector{<:Number}}
    InitialPoints::AbstractVector{<:AbstractVector{<:Number}}
    FinalObjectives::AbstractVector{<:Number}
    InitialObjectives::AbstractVector{<:Number}
    Iterations::AbstractVector{<:Number}
    function MultistartResults(
            FinalPoints::AbstractVector{<:AbstractVector{<:Number}},
            InitialPoints::AbstractVector{<:AbstractVector{<:Number}},
            FinalObjectives::AbstractVector{<:Number},
            InitialObjectives::AbstractVector{<:Number},
            Iterations::AbstractVector{<:Number}
        )
        @assert length(FinalPoints) == length(InitialPoints) == length(FinalObjectives) == length(InitialObjectives) == length(Iterations)
        @assert issorted(FinalObjectives; rev=true)
        @assert any(isfinite, FinalObjectives)
        new(FinalPoints, InitialPoints, FinalObjectives, InitialObjectives, Iterations)
    end
end


Base.length(R::MultistartResults) = length(R.FinalObjectives)
Base.firstindex(R::MultistartResults) = firstindex(R.FinalObjectives)
Base.lastindex(R::MultistartResults) = length(R.FinalObjectives)

MLE(R::MultistartResults) = R.FinalPoints[1]

# kwarg BiLog, StepTol
RecipesBase.@recipe function f(R::MultistartResults)
    DoBiLog = get(plotattributes, :BiLog, true)
    Fin = (DoBiLog ? BiLog : identity)(-R.FinalObjectives)
    ymaxind = (Q=findlast(x->isfinite(x) && abs(x-Fin[1]) < (DoBiLog ? 8 : BiExp(8)), Fin);   isnothing(Q) ? length(Fin) : Q)
    xlabel --> "Iteration"
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
    # Mark steps with vline
    StepTol = get(plotattributes, :ShowSteps, 0.01)
    if 0 < StepTol
        Steps = [i for i in 1:ymaxind-1 if isfinite(Fin[i+1]) && abs(Fin[i+1]-Fin[i]) > ((DoBiLog ? BiLog : identity)(StepTol))]
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

# Add tests, plotting, step analysis
