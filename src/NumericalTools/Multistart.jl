


function GenerateSobolPoints(C::HyperCube, maxval::Real=1e10; N::Int=100)
    S = SOBOL.skip(SOBOL.SobolSeq(clamp(C.L, -maxval*ones(length(C)), maxval*ones(length(C))), clamp(C.U, -maxval*ones(length(C)), maxval*ones(length(C)))), rand(1:5000); exact=true)
    [SOBOL.next!(S) for i in 1:N]
end

MultistartFit(DM::AbstractDataModel, args...; CostFunction::Function=Negloglikelihood(DM), kwargs...) = MultistartFit(Data(DM), Predictor(DM), LogPrior(DM), args...; CostFunction, kwargs...)
MultistartFit(DS::AbstractDataSet, M::ModelMap, LogPriorFn::Union{Nothing,Function}=nothing; kwargs...) = MultistartFit(DS, M, LogPriorFn; MultistartDomain=Domain(M), kwargs...)
function MultistartFit(DS::AbstractDataSet, M::Function, LogPriorFn::Union{Nothing,Function}=nothing; maxval::Real=1e5, MultistartDomain::Union{Nothing, HyperCube}=nothing, kwargs...)
    if isnothing(MultistartDomain)
        @info "No MultistartDomain given, choosing default cube with maxval=$maxval"
        MultistartFit(DS, M, LogPriorFn, FullDomain(pdim(DS, M), maxval); kwargs...)
    else
        MultistartFit(DS, M, LogPriorFn, MultistartDomain; kwargs...)
    end
end
function MultistartFit(DS::AbstractDataSet, model::ModelOrFunction, LogPriorFn::Union{Nothing,Function}, MultistartDomain::HyperCube; N::Int=100, kwargs...)
    @assert N ≥ 1
    MultistartFit(DS, model, GenerateSobolPoints(MultistartDomain; N=N), LogPriorFn; kwargs...)
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
function MultistartFit(DS::AbstractDataSet, model::ModelOrFunction, InitialPoints::AbstractVector{<:AbstractVector{<:Number}}, LogPriorFn::Union{Nothing,Function}; CostFunction::Union{Nothing,Function}=nothing,
                                        parallel::Bool=true, Robust::Bool=false, p::Real=2, TotalLeastSquares::Bool=false, timeout::Real=120, Full::Bool=true, kwargs...)
    @assert !Robust || (p > 0 && !TotalLeastSquares)
    RobustFunc(θ::AbstractVector{<:Number}) = try    RobustFit(DS, model, θ, LogPriorFn; p, timeout, kwargs...)    catch;  θ   end
    Func(θ::AbstractVector{<:Number}) = try    InformationGeometry.minimize(DS, model, θ, LogPriorFn; timeout, kwargs...)    catch;  θ   end
    
    FinalPoints = (parallel ?  progress_pmap : progress_map)(Robust ? RobustFunc : Func, InitialPoints; progress=Progress(length(InitialPoints), desc="Multistart fitting... ", showspeed=true))

    TryCatchWrapper(F::Function) = x -> try F(x) catch;   -Inf   end
    FinalObjectives = (parallel ?  pmap : map)(TryCatchWrapper(isnothing(CostFunction) ? (θ->loglikelihood(DS, model, θ, LogPriorFn)) : Negate(CostFunction)), FinalPoints)
    
    # Some printing?
    if Full
        InitialObjectives = (parallel ?  pmap : map)(TryCatchWrapper(isnothing(CostFunction) ? (θ->loglikelihood(DS, model, θ, LogPriorFn)) : Negate(CostFunction)), FinalPoints)
        Perm = sortperm(FinalObjectives; rev=true)
        MaxVal, MaxInd = findmax(FinalObjectives)
        MultistartResults(FinalPoints[Perm], InitialPoints[Perm], FinalObjectives[Perm], InitialObjectives[Perm], MaxVal, MaxInd)
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
    MaxVal::Number
    MaxInd::Int
end

