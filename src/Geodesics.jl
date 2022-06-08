

GetGeodesicODE(Metric::Function, InitialPos::AbstractVector{<:Number}, approx::Bool=false; kwargs...) = GetGeodesicODE(Metric, InitialPos, Val(approx); kwargs...)
function GetGeodesicODE(Metric::Function, InitialPos::AbstractVector{<:Number}, approx::Val{false}; kwargs...)
    function GeodesicODE!(du,u,p,t)
        n = length(u)÷2
        du[1:n] = u[(n+1):end]
        du[(n+1):end] = ChristoffelTerm(ChristoffelSymbol(Metric,u[1:n]; kwargs...),du[1:n])
    end
end
function GetGeodesicODE(Metric::Function, InitialPos::AbstractVector{<:Number}, approx::Val{true}; kwargs...)
    Γ = ChristoffelSymbol(Metric, InitialPos; kwargs...)
    function ApproxGeodesicODE!(du,u,p,t)
        n = length(u)÷2
        du[1:n] = u[(n+1):end]
        du[(n+1):end] = ChristoffelTerm(Γ, du[1:n])
    end
end

# accuracy ≈ 6e-11
function ComputeGeodesic(Metric::Function, InitialPos::AbstractVector, InitialVel::AbstractVector, Endtime::Number=50.;
                        Boundaries::Union{Function,Nothing}=nothing, tol::Real=1e-11, meth::OrdinaryDiffEqAlgorithm=GetMethod(tol), approx::Bool=false, kwargs...)
    @assert length(InitialPos) == length(InitialVel)
    prob = ODEProblem(GetGeodesicODE(Metric, InitialPos, approx), PromoteStatic(vcat(InitialPos,InitialVel), true), (0.0,Endtime))
    if isnothing(Boundaries)
        solve(prob, meth; reltol=tol, abstol=tol, kwargs...)
    else
        solve(prob, meth; reltol=tol, abstol=tol, callback=DiscreteCallback(Boundaries,terminate!), kwargs...)
    end
end

"""
    ComputeGeodesic(DM::DataModel, InitialPos::AbstractVector, InitialVel::AbstractVector, Endtime::Number=50.;
                                    Boundaries::Union{Function,Nothing}=nothing, tol::Real=1e-11, approx::Bool=false, meth=Tsit5())
Constructs geodesic with given initial position and velocity.
It is possible to specify a boolean-valued function `Boundaries(u,t,int)`, which terminates the integration process when it returns `true`.

By setting the keyword `approx=true`, the ChristoffelSymbols are assumed to be constant and only computed once at the initial position. This simplifies the computation immensely but may also constitute an inaccurate approximation depending on the magnitude of the Ricci curvature.
"""
ComputeGeodesic(DM::AbstractDataModel, args...; kwargs...) = ComputeGeodesic(FisherMetric(DM), args...; kwargs...)


"""
    GeodesicLength(DM::DataModel,sol::AbstractODESolution, Endrange::Number=sol.t[end]; FullSol::Bool=false, tol=1e-14)
    GeodesicLength(Metric::Function,sol::AbstractODESolution, Endrange::Number=sol.t[end]; FullSol::Bool=false, tol=1e-14)
Calculates the length of a geodesic `sol` using the `Metric` up to parameter value `Endrange`.
```math
L[\\gamma] \\coloneqq \\int_a^b \\mathrm{d} t \\, \\sqrt{g_{\\gamma(t)} \\big(\\dot{\\gamma}(t), \\dot{\\gamma}(t)\\big)}
```
"""
GeodesicLength(DM::AbstractDataModel, args...; kwargs...) = GeodesicLength(FisherMetric(DM), args...; kwargs...)
function GeodesicLength(Metric::Function, sol::AbstractODESolution, Endrange::Number=sol.t[end]; FullSol::Bool=false, tol::Real=1e-14, kwargs...)
    @assert length(sol.u[1]) % 2 == 0
    n = length(sol.u[1])÷2
    function Integrand(t)
        FullGamma = sol(t)
        InnerProduct(Metric(FullGamma[1:n]), FullGamma[(n+1):end]) |> sqrt
    end
    Integrate1D(Integrand, (sol.t[1],Endrange); FullSol=FullSol, tol=tol, kwargs...)
end

"""
    GeodesicCrossing(DM::DataModel, sol::AbstractODESolution, Conf::Real=ConfVol(1); tol=1e-15)
Gives the parameter value of the geodesic `sol` at which the confidence level `Conf` is crossed.
"""
function GeodesicCrossing(DM::AbstractDataModel, sol::AbstractODESolution, Conf::Real=ConfVol(1); tol::Real=1e-15)
    start = sol.t[end]/2
    if (tol < 1e-15)
        start *= one(BigFloat)
        @warn "GeodesicCrossing: Conf value not programmed for BigFloat yet."
    end
    A = loglikelihood(DM,sol(0.)[1:2]) - (1/2)*quantile(Chisq(Int(length(sol(0.))/2)),Conf)
    f(t) = A - loglikelihood(DM,sol(t)[1:2])
    find_zero(f, start, Order1B(); xatol=tol)
end


"""
    DistanceAlongGeodesic(Metric::Function,sol::AbstractODESolution,L::Number; tol=1e-14)
Calculates at which parameter value of the geodesic `sol` the length `L` is reached.
"""
function DistanceAlongGeodesic(Metric::Function, sol::AbstractODESolution, L::Number; tol::Real=1e-14, ADmode::Union{Val,Symbol}=Val(:ForwardDiff))
    L ≤ 0 && throw(BoundsError("DistanceAlongGeodesic: L=$L"))
    # Use interpolated Solution of integral for improved accuracy
    GeoLength = GeodesicLength(Metric,sol,sol.t[end], FullSol=true, tol=tol)
    Func(x) = L - GeoLength(x)
    find_zero((Func,GetDeriv(ADmode, Func)),sol.t[end]/2*one(typeof(L)),Roots.Newton(),xatol=tol)
end


# Input Array of Geodesics, Output Array of its endpoints
function Endpoints(Geodesics::AbstractVector{<:AbstractODESolution})
    Endpoints = Vector{Vector{Float64}}(undef,0)
    Numb = Int(length(Geodesics[1].u[1])/2)
    for Curve in Geodesics
        T = Curve.t[end]
        push!(Endpoints,Curve(T)[1:Numb])
    end;    Endpoints
end

"""
    EvaluateEach(geos::AbstractVector{<:AbstractODESolution}, Ts::AbstractVector) -> Vector
Evalues a family `geos` of geodesics on a set of parameters `Ts`. `geos[1]` is evaluated at `Ts[1]`, `geos[2]` is evaluated at `Ts[2]` and so on.
The second half of the values respresenting the velocities is automatically truncated.
"""
function EvaluateEach(sols::AbstractVector{<:AbstractODESolution}, Ts::AbstractVector{<:Number})
    length(sols) != length(Ts) && throw(ArgumentError("Dimension Mismatch."))
    n = Int(length(sols[1].u[1])/2)
    Res = Vector{Vector{Float64}}(undef,0)
    for i in 1:length(Ts)
        F = sols[i]
        push!(Res,F(Ts[i])[1:n])
    end;    Res
end


# ADAPT FOR PLANES
function ConstLengthGeodesics(DM::AbstractDataModel, Metric::Function, MLE::AbstractVector, Conf::Real=ConfVol(1), N::Int=100; tol::Real=6e-11)
    angles = [2π*n/N      for n in 1:N]
    Initials = [ [MLE...,cos(alpha),sin(alpha)] for alpha in angles]
    solving = 0
    function Constructor(Initial)
        solving += 1
        println("Computing Geodesic $(solving) / $N\t")
        ConfidenceBoundaryViaGeodesic(DM,Metric,Initial,Conf; tol=tol)
    end
    pmap(Constructor,Initials)
end


function BoundaryViaGeodesic(DM::AbstractDataModel, InitialPos::AbstractVector, InitialVel::AbstractVector,
                                    Confnum::Real=1, Endtime::Real=500.0; dof::Int=length(MLE(DM)), kwargs...)
    WilksCond = (1/2)*quantile(Chisq(dof),ConfVol(Confnum))
    BoundaryFunc(u,t,int) = LogLikeMLE(DM) - loglikelihood(DM, u[1:end÷2]) > WilksCond
    ComputeGeodesic(FisherMetric(DM), InitialPos, InitialVel, Endtime; Boundaries=BoundaryFunc, kwargs...)
end

###############################################################


function pConstParamGeodesics(Metric::Function,MLE::AbstractVector,Endtime::Number=10.,N::Int=100;
    Boundaries::Union{Function,Nothing}=nothing, tol::Real=1e-13, parallel::Bool=true)
    ConstParamGeodesics(Metric,MLE,Endtime,N;Boundaries=Boundaries, tol=tol, parallel=parallel)
end

function ConstParamGeodesics(Metric::Function,MLE::AbstractVector,Endtime::Number=10.,N::Int=100;
    Boundaries::Union{Function,Nothing}=nothing, tol::Real=1e-13, parallel::Bool=false)
    Initials = [ [cos(alpha),sin(alpha)] for alpha in range(0,2π;length=N)];    solving = 0
    Map = parallel ? pmap : map
    function Constructor(Initial)
        solving += 1
        println("Computing Geodesic $(solving) / $N")
        ComputeGeodesic(Metric,MLE,Initial,Endtime;tol=tol,Boundaries=Boundaries)
    end
    Map(Constructor,Initials)
end

GeodesicBoundaryFunction(Cube::HyperCube) = (u,p,t) -> u[1:end÷2] ∉ Cube
function GeodesicBoundaryFunction(M::ModelMap)
    function ModelMapBoundaries(u,p,t)
        S = !IsInDomain(M, u[1:end÷2])
        S && @warn "Geodesic ran into boundaries specified by ModelMap at $(u[1:end÷2])."
        return S
    end
end

# Also add Plane method!
function RadialGeodesics(DM::AbstractDataModel, Cube::HyperCube; N::Int=50, tol::Real=1e-9, Boundaries::Union{Function,Nothing}=nothing, parallel::Bool=false, verbose::Bool=true, kwargs...)
    @assert length(Cube) == 2 && MLE(DM) ∈ Cube
    widths = CubeWidths(Cube);    Metric(x) = FisherMetric(DM, x)
    initialvels = [widths .* [cos(α), sin(α)] for α in range(0, 2π*(1-1/N); length=N)]
    CB = DiscreteCallback(GeodesicBoundaryFunction(Cube),terminate!)
    CB = !isnothing(Boundaries) ? CallbackSet(CB, DiscreteCallback(Boundaries,terminate!)) : CB
    CB = Predictor(DM) isa ModelMap ? CallbackSet(CB, DiscreteCallback(GeodesicBoundaryFunction(Predictor(DM)),terminate!)) : CB
    # Already added Boundaries(u,p,t) function to callbacks if any was passed via kwarg
    Constructor(InitialVel) = ComputeGeodesic(Metric, MLE(DM), InitialVel, 10.0; tol=tol, Boundaries=nothing, callback=CB, kwargs...)
    Prog = Progress(length(initialvels); enabled=verbose, desc="Computing Geodesics... ", dt=1, showspeed=true)
    (parallel ? progress_pmap : progress_map)(Constructor, initialvels; progress=Prog)
end


"""
    GeodesicBetween(DM::DataModel, P::AbstractVector{<:Number}, Q::AbstractVector{<:Number}; tol::Real=1e-10, meth=Tsit5())
    GeodesicBetween(Metric::Function, P::AbstractVector{<:Number}, Q::AbstractVector{<:Number}; tol::Real=1e-10, meth=Tsit5())
Computes a geodesic between two given points on the parameter manifold and an expression for the metric.

By setting the keyword `approx=true`, the ChristoffelSymbols are assumed to be constant and only computed once at the initial position. This simplifies the computation immensely but may also constitute an inaccurate approximation depending on the magnitude of the Ricci curvature.
"""
GeodesicBetween(DM::AbstractDataModel, args...; kwargs...) = GeodesicBetween(FisherMetric(DM), args...; kwargs...)
function GeodesicBetween(Metric::Function, P::AbstractVector{<:Number}, Q::AbstractVector{<:Number}, Endtime::Real=10.0; tol::Real=1e-9, meth::OrdinaryDiffEqAlgorithm=Tsit5(), approx::Bool=false, kwargs...)
    length(P) != length(Q) && throw("GeodesicBetween: Points not of same dim.")
    dim = length(P)
    function bc!(resid, u, p, t)
        resid[1:dim] .= u[1][1:dim] .- P
        resid[(dim+1):2dim] .= u[end][1:dim] .- Q
    end
    # Slightly perturb initial direction:
    initial = vcat(P, ((Q - P) ./ Endtime) .+ 1e-8 .*(rand(dim) .- 0.5))
    BVP = BVProblem(GetGeodesicODE(Metric, P, approx), bc!, PromoteStatic(initial, true), (0.0, Endtime))
    solve(BVP, Shooting(meth); reltol=tol, abstol=tol, kwargs...)
end

"""
    GeodesicDistance(DM::DataModel, P::AbstractVector{<:Number}, Q::AbstractVector{<:Number}; tol::Real=1e-10)
    GeodesicDistance(Metric::Function, P::AbstractVector{<:Number}, Q::AbstractVector{<:Number}; tol::Real=1e-10)
Computes the length of a geodesic connecting the points `P` and `Q`.
"""
GeodesicDistance(DM::AbstractDataModel, args...; kwargs...) = GeodesicDistance(FisherMetric(DM), args...; kwargs...)
function GeodesicDistance(Metric::Function, P::AbstractVector{<:Number}, Q::AbstractVector{<:Number}; kwargs...)
    GeodesicLength(Metric, GeodesicBetween(Metric, P, Q; kwargs...))
end

ParamVol(sol::AbstractODESolution) = sol.t[end] - sol.t[1]
GeodesicEnergy(DM::DataModel,sol::AbstractODESolution,Endrange::Number=sol.t[end];FullSol::Bool=false,tol::Real=1e-14) = GeodesicEnergy(x->FisherMetric(DM,x),sol,Endrange;tol=tol)
function GeodesicEnergy(Metric::Function,sol::AbstractODESolution,Endrange=sol.t[end]; FullSol::Bool=false,tol::Real=1e-14)
    @assert length(sol.u[1]) % 2 == 0
    n = length(sol.u[1])÷2
    function Integrand(t)
        FullGamma = sol(t)
        InnerProduct(Metric(FullGamma[1:n]), FullGamma[(n+1):end])
    end
    Integrate1D(Integrand, [sol.t[1],Endrange]; FullSol=FullSol, tol=tol)
end



"""
    ExponentialMap(Metric::Function, point::AbstractVector{<:Number}, tangent::AbstractVector{<:Number}; tol::Real=1e-9)
Computes the differential-geometric exponential map ``\\mathrm{exp}_p(v)`` which returns the endpoint that is reached by a geodesic ``\\gamma:[0,1] \\longrightarrow \\mathcal{M}`` with initial direction ``v \\in T_p \\mathcal{M}``.
"""
function ExponentialMap(Metric::Function, point::AbstractVector{<:Number}, tangent::AbstractVector{<:Number}; FullSol::Bool=false, kwargs...)
    if FullSol
        ComputeGeodesic(Metric, point, tangent, 1.0; kwargs...)
    else
        ComputeGeodesic(Metric, point, tangent, 1.0; save_everystep=false, save_start=false, save_end=true, kwargs...).u[end][1:end÷2]
    end
end
ExponentialMap(DM::AbstractDataModel, args...; kwargs...) = ExponentialMap(FisherMetric(DM), args...; kwargs...)

"""
    LogarithmicMap(Metric::Function, P::AbstractVector{<:Number}, Q::AbstractVector{<:Number}; tol::Real=1e-9)
Computes the inverse of the differential-geometric exponential map, i.e. ``\\mathrm{ln}_p(q) \\equiv (\\mathrm{exp}^{-1})_p(q)`` which returns a (possibly non-unique!) initial direction ``v \\in T_p \\mathcal{M}`` for a geodesic ``\\gamma:[0,1] \\longrightarrow \\mathcal{M}`` that goes from ``p \\in \\mathcal{M}`` to ``q \\in \\mathcal{M}``.
"""
function LogarithmicMap(Metric::Function, P::AbstractVector{<:Number}, Q::AbstractVector{<:Number}; FullSol::Bool=false, kwargs...)
    if FullSol
        GeodesicBetween(Metric, P, Q, 1.0; kwargs...)
    else
        GeodesicBetween(Metric, P, Q, 1.0; save_everystep=false, save_start=true, save_end=false, kwargs...).u[1][((end÷2)+1):end]
    end
end
LogarithmicMap(DM::AbstractDataModel, args...; kwargs...) = LogarithmicMap(FisherMetric(DM), args...; kwargs...)


function KarcherMeanStep(Metric::Function, points::AbstractVector{<:AbstractVector{<:Number}}, initialmean::AbstractVector{<:Number}=sum(points)/length(points); kwargs...)
    dirs = map(x->LogarithmicMap(Metric, initialmean, x; kwargs...), points)
    ExponentialMap(Metric, initialmean, sum(dirs) / length(dirs))
end
function KarcherMean(Metric::Function, points::AbstractVector{<:AbstractVector{<:Number}}, initialmean::AbstractVector{<:Number}=sum(points)/length(points); verbose::Bool=true, tol::Real=1e-8, meth::OrdinaryDiffEqAlgorithm=GetMethod(tol), maxiter::Int=10, kwargs...)
    @assert ConsistentElDims(points) == length(initialmean)
    oldmean = initialmean
    for iter in 1:maxiter
        verbose && @info "Karcher Mean iteration $iter."
        newmean = KarcherMeanStep(Metric, points, oldmean; tol=tol, meth=meth, kwargs...)
        if norm(newmean-oldmean) < 20tol
            return newmean
        elseif iter < maxiter
            oldmean = newmean
            continue;
        else
            @warn "KarcherMean: Hit maxiter. Returning last result."
            return newmean
        end
    end
end


"""
Return `true` when integration of ODE should be terminated.
"""
function MBAMBoundaries(DM::AbstractDataModel; componentlim::Real=1e4, singularlim::Real=1e-8)
    function MBAMBoundary(u,t,int)
        if !all(x->abs(x) < componentlim, u)
            @warn "Terminated because a position / velocity coordinate > $componentlim at: $u."
            return true
        elseif svdvals(FisherMetric(DM, u[1:end÷2]))[end] < singularlim
            @warn "Terminated because Fisher metric became singular (i.e. < $singularlim) at: $u."
            return true
        else
            return false
        end
    end
end

function MBAM(DM::AbstractDataModel; Boundaries::Union{Function,Nothing}=nothing, tol::Real=1e-5, meth::OrdinaryDiffEqAlgorithm=GetMethod(tol), componentlim::Real=1e4, singularlim::Real=1e-8, kwargs...)
    InitialVel = normalize(LeastInformativeDirection(DM,MLE(DM)))
    CB = DiscreteCallback(MBAMBoundaries(DM; componentlim=componentlim, singularlim=singularlim), terminate!)
    CB = !isnothing(Boundaries) ? CallbackSet(CB, DiscreteCallback(Boundaries, terminate!)) : CB
    CB = Predictor(DM) isa ModelMap ? CallbackSet(CB, DiscreteCallback(GeodesicBoundaryFunction(Predictor(DM)),terminate!)) : CB
    # Already added Boundaries(u,p,t) function to callbacks if any was passed via kwarg
    ComputeGeodesic(DM, MLE(DM), InitialVel, 1e4; Boundaries=nothing, callback=CB, tol=tol, meth=meth, kwargs...)
end
