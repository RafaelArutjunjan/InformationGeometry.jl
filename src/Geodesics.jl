

# tol = 6e-11
function ComputeGeodesic(Metric::Function, InitialPos::AbstractVector, InitialVel::AbstractVector, Endtime::Number=50.;
                        Boundaries::Union{Function,Nothing}=nothing, tol::Real=1e-11, meth::OrdinaryDiffEqAlgorithm=GetMethod(tol), kwargs...)
    function GeodesicODE!(du,u,p,t)
        n = Int(length(u)/2)
        du[1:n] = u[(n+1):2n]
        du[(n+1):2n] = ChristoffelTerm(ChristoffelSymbol(Metric,u[1:n]),du[1:n])
    end
    tspan = (0.0,Endtime);
    Initial = vcat(InitialPos, InitialVel)
    prob = ODEProblem(GeodesicODE!, Initial, tspan)
    if Boundaries isa Nothing
        return solve(prob, meth; reltol=tol, abstol=tol, kwargs...)
    else
        return solve(prob, meth; reltol=tol, abstol=tol, callback=DiscreteCallback(Boundaries,terminate!), kwargs...)
    end
end

"""
    ComputeGeodesic(DM::DataModel, InitialPos::Vector, InitialVel::Vector, Endtime::Number=50.;
                                    Boundaries::Union{Function,Nothing}=nothing, tol::Real=1e-11, meth=Tsit5())
Constructs geodesic with given initial position and velocity.
It is possible to specify a boolean-valued function `Boundaries(u,t,int)`, which terminates the integration process when it returns `true`.
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
function GeodesicLength(Metric::Function, sol::AbstractODESolution, Endrange::Number=sol.t[end]; FullSol::Bool=false, tol::Real=1e-14)
    # GET RID OF ENDRANGE PARAMETER?
    n = length(sol.u[1])/2 |> Int
    function Integrand(t)
        FullGamma = sol(t)
        sqrt(transpose(FullGamma[(n+1):2n]) * Metric(FullGamma[1:n]) * FullGamma[(n+1):2n])
    end
    return Integrate1D(Integrand, (sol.t[1],Endrange); FullSol=FullSol, tol=tol)
end

"""
    GeodesicCrossing(DM::DataModel, sol::AbstractODESolution, Conf::Real=ConfVol(1); tol=1e-15)
Gives the parameter value of the geodesic `sol` at which the confidence level `Conf` is crossed.
"""
function GeodesicCrossing(DM::AbstractDataModel, sol::AbstractODESolution, Conf::Real=ConfVol(1); tol::Real=1e-15)
    start = sol.t[end]/2
    if (tol < 1e-15)
        start *= one(BigFloat)
        println("GeodesicCrossing: Conf value not programmed as BigFloat yet.")
    end
    A = loglikelihood(DM,sol(0.)[1:2]) - (1/2)*quantile(Chisq(Int(length(sol(0.))/2)),Conf)
    f(t) = A - loglikelihood(DM,sol(t)[1:2])
    find_zero(f,start,Order1B(),xatol=tol)
end


"""
    DistanceAlongGeodesic(Metric::Function,sol::AbstractODESolution,L::Number; tol=1e-14)
Calculates at which parameter value of the geodesic `sol` the length `L` is reached.
"""
function DistanceAlongGeodesic(Metric::Function, sol::AbstractODESolution, L::Number; tol::Real=1e-14)
    L < 0 && throw(BoundsError("DistanceAlongGeodesic: L=$L"))
    # Use interpolated Solution of integral for improved accuracy
    GeoLength = GeodesicLength(Metric,sol,sol.t[end], FullSol=true, tol=tol)
    Func(x) = L - GeoLength(x)
    dFunc(x) = ForwardDiff.derivative(Func,x)
    find_zero((Func,dFunc),sol.t[end]/2*one(typeof(L)),Roots.Newton(),xatol=tol)
end


# Input Array of Geodesics, Output Array of its endpoints
function Endpoints(Geodesics::Vector{<:AbstractODESolution})
    Endpoints = Vector{Vector{Float64}}(undef,0)
    Number = Int(length(Geodesics[1].u[1])/2)
    for Curve in Geodesics
        T = Curve.t[end]
        push!(Endpoints,Curve(T)[1:Number])
    end;    Endpoints
end

"""
    EvaluateEach(geos::Vector{<:AbstractODESolution}, Ts::Vector) -> Vector
Evalues a family `geos` of geodesics on a set of parameters `Ts`. `geos[1]` is evaluated at `Ts[1]`, `geos[2]` is evaluated at `Ts[2]` and so on.
The second half of the values respresenting the velocities is automatically truncated.
"""
function EvaluateEach(sols::Vector{<:AbstractODESolution}, Ts::AbstractVector{<:Number})
    length(sols) != length(Ts) && throw(ArgumentError("Dimension Mismatch."))
    n = Int(length(sols[1].u[1])/2)
    Res = Vector{Vector{Float64}}(undef,0)
    for i in 1:length(Ts)
        F = sols[i]
        push!(Res,F(Ts[i])[1:n])
    end
    Res
end


# """
#     Truncated(sol::AbstractODESolution) -> Function
# Given a geodesic `sol`, the second half of the components which represent the velocity are truncated off.
# The result gives the position of the geodesic as a function of the parameter.
# However, since it is no longer of type `ODESolution`, one no longer has access to the fields `sol.t`, `sol.u` and so on.
# """
# Truncated(sol::AbstractODESolution) = (t->sol(t)[1:Int(length(sol.u[1])/2)])


################################################################ MERGE THESE FUNCTION VARIATIONS
# function ConstLengthGeodesics(DM::DataModel,Metric::Function,MLE::Vector,Conf::Float64=ConfVol(1),N::Int=100)
#     angles = [2*pi*n/N      for n in 1:N]
#     sols = Vector{ODESolution}(undef,0)
#     for α in angles
#         InitialVelocity = [cos(α),sin(α)]
#         push!(sols,ConfidenceBoundaryViaGeodesic(DM,Metric,[MLE...,InitialVelocity...],Conf))
#         num = α*N/(2*pi) |> round |> Int
#         println("Calculated Geodesic $num of $N.")
#     end
#     Region = Endpoints(sols);   plot(Unpack(Region))
#     sols
# end

# ADAPT FOR PLANES
function ConstLengthGeodesics(DM::AbstractDataModel, Metric::Function, MLE::Vector, Conf::Real=ConfVol(1), N::Int=100; tol::Real=6e-11)
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


function ConfidenceBoundaryViaGeodesic(DM::AbstractDataModel,Metric::Function,InitialVec::Vector,Conf::Real=ConfVol(1); tol::Real=6e-11, meth::OrdinaryDiffEqAlgorithm=GetMethod(tol))
    function GeodesicODE!(du,u,p,t)
        n = length(u)
        (n%2==1) && throw(ArgumentError("dim(u)=$n, should be even."))
        n = Int(n/2)
        du[1:n] = u[(n+1):2n]
        du[(n+1):2n] = ChristoffelTerm(ChristoffelSymbol(Metric,u[1:n]),du[1:n])
    end
    WilksCond = loglikelihood(DM,MLE) - (1/2)*quantile(Chisq(length(MLE)),Conf)
    InsideRegion(x)::Bool = (WilksCond <= loglikelihood(DM,x))
    terminatecondition(u,t,integrator) = !InsideRegion(u)
    # Use ContinuousCallback
    cb = DiscreteCallback(terminatecondition,terminate!,save_positions=(true,true))
    tspan = (0.0,150.0)
    prob = ODEProblem(GeodesicODE!,InitialVec,tspan)
    @time solve(prob,meth,reltol=tol,abstol=tol,callback=cb)
end

###############################################################


function pConstParamGeodesics(Metric::Function,MLE::Vector,Endtime::Number=10.,N::Int=100;
    Boundaries::Union{Function,Nothing}=nothing, tol::Real=1e-13, parallel::Bool=true)
    ConstParamGeodesics(Metric,MLE,Endtime,N;Boundaries=Boundaries, tol=tol, parallel=parallel)
end

function ConstParamGeodesics(Metric::Function,MLE::Vector,Endtime::Number=10.,N::Int=100;
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


# Also add Plane method!
function RadialGeodesics(DM::AbstractDataModel, Cube::HyperCube; N::Int=50, tol::Real=1e-9, Boundaries::Union{Function,Nothing}=nothing, parallel::Bool=false)
    @assert length(Cube) == 2 && MLE(DM) ∈ Cube
    widths = CubeWidths(Cube);    Map = parallel ? pmap : map;    Metric(x) = FisherMetric(DM, x)
    initialvels = [widths .* [cos(α), sin(α)] for α in range(0, 2π*(1-1/N); length=N)]
    OutsideBoundaries(u,p,t) = Boundaries isa Nothing ? u[1:end÷2] ∉ Cube : Boundaries
    solving = 0
    function Constructor(Initial)
        solving += 1
        println("Computing Geodesic $(solving) / $N")
        ComputeGeodesic(Metric, MLE(DM), Initial, 10.0; tol=tol, Boundaries=OutsideBoundaries)
    end
    Map(Constructor,initialvels)
end


"""
    GeodesicBetween(DM::DataModel, P::AbstractVector{<:Number}, Q::AbstractVector{<:Number}; tol::Real=1e-10, meth=Tsit5())
    GeodesicBetween(Metric::Function, P::AbstractVector{<:Number}, Q::AbstractVector{<:Number}; tol::Real=1e-10, meth=Tsit5())
Computes a geodesic between two given points on the parameter manifold and an expression for the metric.
"""
GeodesicBetween(DM::AbstractDataModel, args...; kwargs...) = GeodesicBetween(FisherMetric(DM), args...; kwargs...)
function GeodesicBetween(Metric::Function, P::AbstractVector{<:Number}, Q::AbstractVector{<:Number}, Endtime::Real=10.0; tol::Real=1e-9, meth::OrdinaryDiffEqAlgorithm=GetMethod(tol), kwargs...)
    length(P) != length(Q) && throw("GeodesicBetween: Points not of same dim.")
    dim = length(P)
    function GeodesicODE!(du,u,p,t)
        du[1:dim] = u[(dim+1):2dim]
        du[(dim+1):2dim] = ChristoffelTerm(ChristoffelSymbol(Metric,u[1:dim]),du[1:dim])
    end
    function bc!(resid, u, p, t)
        resid[1:dim] = u[1][1:dim] .- P
        resid[(dim+1):2dim] = u[end][1:dim] .- Q
    end
    # Slightly perturb initial direction:
    initial = vcat(P, ((Q - P) ./ Endtime) .+ 1e-8 .*(rand(dim) .- 0.5))
    BVP = BVProblem(GeodesicODE!, bc!, initial, (0.0, Endtime))
    solve(BVP, Shooting(meth); reltol=tol, abstol=tol, kwargs...)
end

"""
    GeodesicDistance(DM::DataModel, P::AbstractVector{<:Number}, Q::AbstractVector{<:Number}; tol::Real=1e-10)
    GeodesicDistance(Metric::Function, P::AbstractVector{<:Number}, Q::AbstractVector{<:Number}; tol::Real=1e-10)
Computes the length of a geodesic connecting the points `P` and `Q`.
"""
GeodesicDistance(DM::AbstractDataModel, args...; kwargs...) = GeodesicDistance(FisherMetric(DM), args...; kwargs...)
function GeodesicDistance(Metric::Function, P::AbstractVector{<:Number}, Q::AbstractVector{<:Number}; tol::Real=1e-10, meth::OrdinaryDiffEqAlgorithm=GetMethod(tol), kwargs...)
    GeodesicLength(Metric, GeodesicBetween(Metric, P, Q; tol=tol, meth=meth, kwargs...))
end

ParamVol(sol::AbstractODESolution) = sol.t[end] - sol.t[1]
GeodesicEnergy(DM::DataModel,sol::AbstractODESolution,Endrange::Number=sol.t[end];FullSol::Bool=false,tol::Real=1e-14) = GeodesicEnergy(x->FisherMetric(DM,x),sol,Endrange;tol=tol)
function GeodesicEnergy(Metric::Function,sol::AbstractODESolution,Endrange=sol.t[end]; FullSol::Bool=false,tol::Real=1e-14)
    n = length(sol.u[1])/2 |> Int
    function Integrand(t)
        FullGamma = sol(t)
        transpose(FullGamma[(n+1):2n]) * Metric(FullGamma[1:n]) * FullGamma[(n+1):2n]
    end
    Integrate1D(Integrand,[sol.t[1],Endrange]; FullSol=FullSol,tol=tol)
end



"""
    ExponentialMap(Metric::Function, point::AbstractVector{<:Number}, tangent::AbstractVector{<:Number}; tol::Real=1e-9)
Computes the differential-geometric exponential map ``\\mathrm{exp}_p(v)`` which returns the endpoint that is reached by a geodesic ``\\gamma:[0,1] \\longrightarrow \\mathcal{M}`` with initial direction ``v \\in T_p \\mathcal{M}``.
"""
function ExponentialMap(Metric::Function, point::AbstractVector{<:Number}, tangent::AbstractVector{<:Number}; tol::Real=1e-9, meth::OrdinaryDiffEqAlgorithm=GetMethod(tol), FullSol::Bool=false, kwargs...)
    if FullSol
        ComputeGeodesic(Metric, point, tangent, 1.0; tol=tol, meth=meth, kwargs...)
    else
        ComputeGeodesic(Metric, point, tangent, 1.0; tol=tol, meth=meth, save_everystep=false, save_start=false, save_end=true, kwargs...).u[end][1:end÷2]
    end
end
ExponentialMap(DM::AbstractDataModel, args...; kwargs...) = ExponentialMap(FisherMetric(DM), args...; kwargs...)

"""
    LogarithmicMap(Metric::Function, P::AbstractVector{<:Number}, Q::AbstractVector{<:Number}; tol::Real=1e-9)
Computes the inverse of the differential-geometric exponential map, i.e. ``\\mathrm{ln}_p(q) \\equiv (\\mathrm{exp}^{-1})_p(q)`` which returns a (possibly non-unique!) initial direction ``v \\in T_p \\mathcal{M}`` for a geodesic ``\\gamma:[0,1] \\longrightarrow \\mathcal{M}`` that goes from ``p \\in \\mathcal{M}`` to ``q \\in \\mathcal{M}``.
"""
function LogarithmicMap(Metric::Function, P::AbstractVector{<:Number}, Q::AbstractVector{<:Number}; tol::Real=1e-9, meth::OrdinaryDiffEqAlgorithm=GetMethod(tol), FullSol::Bool=false, kwargs...)
    if FullSol
        GeodesicBetween(Metric, P, Q, 1.0; tol=tol, meth=meth, kwargs...)
    else
        GeodesicBetween(Metric, P, Q, 1.0; tol=tol, meth=meth, save_everystep=false, save_start=true, save_end=false, kwargs...).u[1][((end÷2)+1):end]
    end
end
LogarithmicMap(DM::AbstractDataModel, args...; kwargs...) = LogarithmicMap(FisherMetric(DM), args...; kwargs...)


function KarcherMeanStep(Metric::Function, points::AbstractVector{<:AbstractVector{<:Number}}, initialmean::AbstractVector{<:Number}=sum(points)/length(points); kwargs...)
    dirs = map(x->LogarithmicMap(Metric, initialmean, x; kwargs...), points)
    ExponentialMap(Metric, initialmean, sum(dirs) / length(dirs))
end
function KarcherMean(Metric::Function, points::AbstractVector{<:AbstractVector{<:Number}}, initialmean::AbstractVector{<:Number}=sum(points)/length(points); tol::Real=1e-8, meth::OrdinaryDiffEqAlgorithm=GetMethod(tol), maxiter::Int=10, kwargs...)
    @assert ConsistentElDims(points) == length(initialmean)
    oldmean = initialmean
    for iter in 1:maxiter
        println("Karcher Mean iteration $iter.")
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
function MBAMBoundaries(u, t, int, DM; componentlim = 1e3, singularlim = 1e-8)::Bool
    if !all(x->x < componentlim, u)
        @warn "Terminated because a position / velocity coordinate > $componentlim at: $u."
        return true
    elseif svdvals(FisherMetric(DM,u[1:Int(length(u)/2)]))[end] < singularlim
        @warn "Terminated because Fisher metric became singular (i.e. < $singularlim) at: $u."
        return true
    else
        return false
    end
end

function MBAM(DM::AbstractDataModel; Boundaries::Union{Function,Nothing}=nothing, tol::Real=1e-5, meth::OrdinaryDiffEqAlgorithm=GetMethod(tol))
    InitialVel = normalize(LeastInformativeDirection(DM,MLE(DM)))
    if typeof(Boundaries) == Nothing
        MBAMboundary(u,t,int) = MBAMBoundaries(u,t,int,DM)
        return ComputeGeodesic(DM, MLE(DM), InitialVel, 1e3; Boundaries=MBAMboundary, tol=tol, meth=meth)
    else
        CombinedBoundaries(u,t,int)::Bool = Boundaries(u,t,int) || MBAMBoundaries(u,t,int,DM)
        return ComputeGeodesic(DM, MLE(DM), InitialVel, 1e3; Boundaries=CombinedBoundaries, tol=tol, meth=meth)
    end
end
