
using TensorOperations, Roots
using Combinatorics


# TensorOperations roughly 10 times slower than native Julia Linear Algebra
GetH(x::Type) = (x == BigFloat) ? convert(BigFloat,10 .^(-precision(x)/10)) : 1e-6


function signature(I::Vector,dims::Int)
    rank = length(I)
    (rank < 2 || dims < 2) && throw(BoundsError("Signature error: dims = $dims, rank = $rank"))
    maximum(I) > dims && throw(BoundsError("Signature error: dims = $dims, Index value was $(maximum(I))"))
    minimum(I) < 1 && throw(BoundsError("Sign error: Index value $(minimum(I))"))
    rank > dims && return 0
    !allunique(I) && return 0
    swapped = false;    nswaps = 0;    Rightmost = rank-1
    while Rightmost > 0
        for i in 1:Rightmost
            if I[i] > I[i+1]
                store = I[i+1]; I[i+1] = I[i];  I[i] = store
                nswaps += 1;    swapped = true
            end
        end
        if !swapped  break  end
        Rightmost -= 1;     swapped = false
    end
    if iseven(nswaps)   return 1    else    return -1   end
end

function GenerateEpsilonTensor(dims::Int,rank::Int=3)
    (dims < 2) && throw(ArgumentError("dims = $dims"))
    (rank < 2) && throw(ArgumentError("rank = $rank"))
    if dims < rank
        throw(ArgumentError("GenerateEpsilonTensor Error: dims: $dims, rank: $rank."))
        println("GenerateEpsilonTensor Error: dims: $dims, rank: $rank. Returned zero tensor")
    end
    G = zeros(Int,(dims.*ones(Int,rank))...)
    for indices in permutations(1:dims,rank)
        G[Tuple(indices)...] += signature(indices, dims) |> Int
    end
    G
end

function Cross(A::Vector{<:Real},B::Vector{<:Real})
    length(A) != length(B) && throw(ArgumentError("Cross: Dimension Mismatch: $A, $B."))
    if length(A) > 3
        return @tensor C[a] := GenerateEpsilonTensor(length(A),3)[a,b,c]*A[b]*B[c]
    elseif length(A) == 3
        return cross(A,B)
    elseif length(A) == 2
        println("Using Cross for $A of length 2 right now. Try not to.")
        return cross([A...,0],[B...,0])[1:2]
    else
        throw(ArgumentError("Error: length(A) = $(length(A))"))
    end
end

function ChristoffelTerm(ConnectionCoeff::AbstractArray{<:Real,3}, v::Vector{<:Real})
    (Tuple(Int.(length(v) .*ones(3))) != size(ConnectionCoeff)) && throw(ArgumentError("Connectioncoefficients don't match vector: dim(v) = $(length(v)), size(Connection) = $(size(ConnectionCoeff))"))
    @tensor Res[a] := (-1*ConnectionCoeff)[a,b,c]*v[b]*v[c]
end


# Accuracy ≈ 3e-11
# ROUND TO 1e-10???
# BigCalc for using BigFloat Calculation in finite differencing step but outputting Float64 again.
ChristoffelSymbol(DM::DataModel, point::Vector; BigCalc::Bool=false) = ChristoffelSymbol(z->FisherMetric(DM,z), point, BigCalc=BigCalc)
"""
    ChristoffelSymbol(DM::DataModel, point::Vector; BigCalc::Bool=false)
    ChristoffelSymbol(Metric::Function, point::Vector; BigCalc::Bool=false)
Calculates the Christoffel symbol at a point `p` though finite differencing of the `Metric`. Accurate to ≈ 3e-11.
`BigCalc=true` increases accuracy through BigFloat calculation.
"""
function ChristoffelSymbol(Metric::Function, point::Vector; BigCalc::Bool=false)
    Finv = inv(Metric(point))
    function FPDVs(Metric,point; BigCalc::Bool=false)
        if BigCalc      point = BigFloat.(point)        end
        PDV = zeros(suff(point),length(point),length(point),length(point))
        h = GetH(suff(point))
        for i in 1:length(point)
            PDV[:,:,i] .= (1/(2*h)).*(Metric(point .+ h.*BasisVector(i,length(point))) .- Metric(point .- h.*BasisVector(i,length(point))))
        end
        PDV
    end
    FPDV = FPDVs(Metric,point,BigCalc=BigCalc)
    if (suff(point) != BigFloat) && BigCalc
        FPDV = convert(Array{Float64,3},FPDV)
    end
    @tensor Christoffels[a,i,j] := ((1/2) * Finv)[a,m] * (FPDV[j,m,i] + FPDV[m,i,j] - FPDV[i,j,m])
end

# PROVIDE A FUNCTION WHICH SPECIFIES THE BOUNDARIES OF A MODEL AND TERMINATES GEODESICS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# tol = 6e-11
function ComputeGeodesic(Metric::Function,InitialPos::Vector,InitialVel::Vector, Endtime::Float64=50.;
                                    Boundaries::Union{Function,Nothing}=nothing, tol::Real=1e-11, meth=Tsit5())
    function GeodesicODE!(du,u,p,t)
        n = length(u)
        (n%2==1) && throw(ArgumentError("dim(u)=$n, should be even."))
        n = Int(n/2)
        du[1:n] = u[(n+1):2n]
        du[(n+1):2n] = ChristoffelTerm(ChristoffelSymbol(Metric,u[1:n]),du[1:n])
    end
    tspan = (0.0,Endtime);
    Initial = [InitialPos...,InitialVel...]
    prob = ODEProblem(GeodesicODE!,Initial,tspan)
    if typeof(Boundaries) == Nothing
        return solve(prob,meth,reltol=tol,abstol=tol)
    else
        return solve(prob,meth,reltol=tol,abstol=tol,callback=DiscreteCallback(Boundaries,terminate!))
    end
end
"""
    ComputeGeodesic(DM::DataModel,InitialPos::Vector,InitialVel::Vector, Endtime::Float64=50.;
                                    Boundaries::Union{Function,Nothing}=nothing, tol::Real=1e-11, meth=Tsit5())
Constructs geodesic with given initial position and velocity.
It is possible to specify a boolean-valued function `Boundaries(u,t,int)`, which terminates the integration process it returns `false`.
"""
function ComputeGeodesic(DM::DataModel,InitialPos::Vector,InitialVel::Vector, Endtime::Float64=50.;
                                    Boundaries::Union{Function,Nothing}=nothing, tol::Real=1e-11, meth=Tsit5())
    ComputeGeodesic(x->FisherMetric(DM,x),InitialPos,InitialVel, Endtime, Boundaries=Boundaries,tol=tol,meth=meth)
end

function MetricNorm(G,v::Vector,w::Vector=v) where Q<:Real
    (Tuple(Int.(length(v) .*ones(2))) != size(G)) && throw(ArgumentError("MetricNorm Dimension Mismatch."))
    sqrt(transpose(v)*G*w)
    # @tensor Res = G[a,b]*v[a]*w[b];     sqrt(Res)
end

"""
    GeodesicLength(Metric::Function,sol::ODESolution, Endrange::Real=0.; fullSol::Bool=false, tol=1e-14)
Calculates the length of a geodesic `sol` using the `Metric` up to parameter value `Endrange`.
"""
function GeodesicLength(Metric::Function,sol::ODESolution, Endrange::Real=sol.t[end]; fullSol::Bool=false, tol=1e-14)
    # GET RID OF ENDRANGE PARAMETER?
    n = length(sol.u[1])/2 |> Int
    function Integrand(t)
        FullGamma = sol(t)
        MetricNorm(Metric(FullGamma[1:n]),FullGamma[(n+1):2n])
    end
    return Integrate1D(Integrand,[sol.t[1],Endrange],fullSol=fullSol,tol=tol)
end
GeodesicLength(DM::DataModel,sol::ODESolution, Endrange::Real=sol.t[end]; fullSol::Bool=false, tol=1e-14) = GeodesicLength(x->FisherMetric(DM,x),sol,Endrange; fullSol=fullSol, tol=tol)

"""
    GeodesicCrossing(DM::DataModel,sol::ODESolution,Conf::Real=ConfVol(1); tol=1e-15)
Gives the parameter value of the geodesic `sol` at which the confidence level `Conf` is crossed.
"""
function GeodesicCrossing(DM::DataModel,sol::ODESolution,Conf::Real=ConfVol(1); tol=1e-15)
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
    DistanceAlongGeodesic(Metric::Function,sol::ODESolution,L::Real; tol=1e-14)
Calculates at which parameter value of the geodesic `sol` the length `L` is reached.
"""
function DistanceAlongGeodesic(Metric::Function,sol::ODESolution,L::Real; tol=1e-14)
    L < 0 && throw(BoundsError("DistanceAlongGeodesic: L=$L"))
    #Use interpolated Solution of integral for improved accuracy
    GeoLength = GeodesicLength(Metric,sol,sol.t[end], fullSol=true, tol=tol)
    Func(x) = L - GeoLength(x)
    dFunc(x) = ForwardDiff.derivative(Func,x)
    find_zero((Func,dFunc),sol.t[end]/2*one(typeof(L)),Roots.Newton(),xatol=tol)
end


# Input Array of Geodesics, Output Array of its endpoints
function Endpoints(Geodesics::Vector{ODESolution})
    Endpoints = Vector{Vector{Float64}}(undef,0)
    Number = Int(length(Geodesics[1].u[1])/2)
    for Curve in Geodesics
        T = Curve.t[end]
        push!(Endpoints,Curve(T)[1:Number])
    end;    Endpoints
end

# function GaussNewton(Metric::Function,Cost::Function,point::Vector)
#     Ginv = inv(Metric(point))
#     Cov = ForwardDiff.gradient(Cost,point)
#     @tensor v[a] := Ginv[a,b] * Cov[b]
# end


"""
    Truncated(sol::ODESolution) -> Function
Given a geodesic `sol`, the second half of the components which represent the velocity are truncated off.
The result gives the position of the geodesic as a function of the parameter.
However, since it is no longer of type `ODESolution`, one no longer has access to the fields `sol.t`, `sol.u` and so on.
"""
Truncated(sol::ODESolution) = (t->sol(t)[1:Int(length(sol.u[1])/2)])


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
#     Region = Endpoints(sols);   plot(SensibleOutput(Region))
#     sols
# end

# ADAPT FOR PLANES
function ConstLengthGeodesics(DM::DataModel,Metric::Function,MLE::Vector,Conf::Float64=ConfVol(1),N::Int=100; tol::Float64=6e-11)
    angles = [2*pi*n/N      for n in 1:N]
    Initials = [ [MLE...,cos(alpha),sin(alpha)] for alpha in angles]
    solving = 0
    function Constructor(Initial)
        solving += 1
        println("Computing Geodesic $(solving) / $N\t")
        ConfidenceBoundaryViaGeodesic(DM,Metric,Initial,Conf; tol=tol)
    end
    pmap(Constructor,Initials)
end


function ConfidenceBoundaryViaGeodesic(DM::DataModel,Metric::Function,InitialVec::Vector,Conf::Float64=ConfVol(1); tol::Float64=6e-11)
    function GeodesicODE!(du,u,p,t)
        n = length(u)
        (n%2==1) && throw(ArgumentError("dim(u)=$n, should be even."))
        n = Int(n/2)
        du[1:n] = u[(n+1):2n]
        du[(n+1):2n] = ChristoffelTerm(ChristoffelSymbol(Metric,u[1:n]),du[1:n])
    end
    WilksCond = loglikelihood(DM,MLE) - (1/2)*quantile(Chisq(length(MLE)),Conf)
    Inside(x)::Bool = (WilksCond <= loglikelihood(DM,x))
    terminatecondition(u,t,integrator) = !Inside(u)
    # Use ContinuousCallback
    cb = DiscreteCallback(terminatecondition,terminate!,save_positions=(true,true))
    tspan = (0.0,150.0)
    prob = ODEProblem(GeodesicODE!,InitialVec,tspan)
    @time solve(prob,Vern9(),reltol=tol,abstol=tol,callback=cb)
end

###############################################################


function pConstParamGeodesics(Metric::Function,MLE::Vector,Endtime::Float64=10.,N::Int=100;
    Boundaries::Union{Function,Nothing}=nothing, tol::Float64=1e-13, parallel::Bool=true)
    ConstParamGeodesics(Metric,MLE,Endtime,N;Boundaries=Boundaries, tol=tol, parallel=parallel)
end

function ConstParamGeodesics(Metric::Function,MLE::Vector,Endtime::Float64=10.,N::Int=100;
    Boundaries::Union{Function,Nothing}=nothing, tol::Float64=1e-13, parallel::Bool=false)
    Initials = [ [cos(alpha),sin(alpha)] for alpha in range(0,2pi,length=N)]
    solving = 0
    function Constructor(Initial)
        solving += 1
        println("Computing Geodesic $(solving) / $N")
        ComputeGeodesic(Metric,MLE,Initial,Endtime,tol=tol,Boundaries=Boundaries)
    end
    parallel && return pmap(Constructor,Initials)
    map(Constructor,Initials)
end

"""
    GeodesicBetween(Metric::Function,P::Vector{<:Real},Q::Vector{<:Real}; tol::Real=1e-10, meth=Tsit5())
Computes a geodesic between two given points on the parameter manifold and an expression for the metric.
"""
function GeodesicBetween(Metric::Function,P::Vector{<:Real},Q::Vector{<:Real}; tol::Real=1e-10, meth=Tsit5())
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
    # Add a bit of randomness to initial direction:
    tspan = (0.,10.);    initial = [P..., ((Q .- P)./tspan[2])...] .+ 1e-6 .*(rand(2dim) .-0.5)
    BVP = BVProblem(GeodesicODE!, bc!, initial, tspan)
    solve(BVP, Shooting(meth), reltol=tol,abstol=tol)
end
GeodesicBetween(DM::DataModel,P::Vector{<:Real},Q::Vector{<:Real}; tol::Real=1e-10, meth=Tsit5()) = GeodesicBetween(x->FisherMetric(DM,x),P,Q; tol=tol, meth=meth)

"""
    GeodesicDistance(Metric::Function,P::Vector{<:Real},Q::Vector{<:Real}; tol::Real=1e-10, meth=Tsit5())
Computes the length of a geodesic connecting the points `P` and `Q`.
"""
function GeodesicDistance(Metric::Function,P::Vector{<:Real},Q::Vector{<:Real}; tol::Real=1e-10)
    GeodesicLength(Metric,GeodesicBetween(Metric,P,Q,tol=tol))
end
GeodesicDistance(DM::DataModel,P::Vector{<:Real},Q::Vector{<:Real}; tol::Real=1e-10) = GeodesicDistance(x->FisherMetric(DM,x),P,Q,tol=tol)

ParamVol(sol::ODESolution) = sol.t[end] - sol.t[1]
GeodesicEnergy(DM::DataModel,sol::ODESolution,Endrange=sol.t[end];fullSol::Bool=false,tol=1e-14) = GeodesicEnergy(x->FisherMetric(DM,x),sol,Endrange;tol=tol)
function GeodesicEnergy(Metric::Function,sol::ODESolution,Endrange=sol.t[end]; fullSol::Bool=false,tol=1e-14)
    n = length(sol.u[1])/2 |> Int
    function Integrand(t)
        FullGamma = sol(t)
        transpose(FullGamma[(n+1):2n]) * Metric(FullGamma[1:n]) * FullGamma[(n+1):2n]
    end
    Integrate1D(Integrand,[sol.t[1],Endrange],fullSol=fullSol,tol=tol)
end


function PlotCurves(Curves::Vector; N::Int=100)
    p = [];    A = Array{Float64,2}(undef,N,2)
    for sol in Curves
        ran = range(sol.t[1],sol.t[end],length=N)
        for i in 1:length(ran)    A[i,:] = sol(ran[i])[1:2]  end
        p = Plots.plot!(A[:,1],A[:,2])
        # p = Plots.plot!(sol,vars=(1,2))
    end
    p
end

"""
    EvaluateEach(sols::Vector{Q}, Ts::Vector) where Q <: ODESolution
Evalues a family `sols` of geodesics on a set of parameters `Ts`. `sols[1]` is evaluated at `Ts[1]`, `sols[2]` is evaluated at `Ts[2]` and so on.
The second half of the values respresenting the velocities is automatically truncated.
"""
function EvaluateEach(sols::Vector, Ts::Vector)
    length(sols) != length(Ts) && throw(ArgumentError("Dimension Mismatch."))
    n = Int(length(sols[1].u[1])/2)
    Res = Vector{Vector{Float64}}(undef,0)
    for i in 1:length(Ts)
        F = sols[i]
        push!(Res,F(Ts[i])[1:n])
    end
    Res
end

EvaluateAlongGeodesic(F::Function,sol::ODESolution, Interval::Vector=[sol.t[1],sol.t[end]]; N::Int=1000) = [F(sol(t)[1:Int(length(sol.u[1])/2)]) for t in range(Interval[1],Interval[2],length=N)]
function PlotAlongGeodesic(F::Function,sol::ODESolution, Interval::Vector=[sol.t[1],sol.t[end]]; N::Int=1000, OverWrite::Bool=false)
    Z = EvaluateAlongGeodesic(F,sol,Interval, N=N)
    if length(Z[1]) == 1
        if OverWrite
            Plots.plot(range(Interval[1],Interval[2],length=N),Z) |> display
        else
            Plots.plot!(range(Interval[1],Interval[2],length=N),Z) |> display
        end
    end
    [collect(range(Interval[1],Interval[2],length=N)) Z]
end
EvaluateAlongGeodesicLength(DM::DataModel,F::Function,sol::ODESolution, Interval::Vector=[sol.t[1],sol.t[end]]; N::Int=1000) = EvaluateAlongGeodesic(F,sol,Interval, N=N)
function PlotAlongGeodesicLength(DM::DataModel,F::Function,sol::ODESolution, Interval::Vector=[sol.t[1],sol.t[end]]; N::Int=1000, OverWrite::Bool=false)
    Z = EvaluateAlongGeodesic(F,sol,Interval, N=N)
    Geo = GeodesicLength(x->FisherMetric(DM,x), sol,sol.t[end];fullSol=true, Auto=true, tol=1e-14)
    Ls = map(Geo,range(Interval[1],Interval[2],length=N))
    if length(Z[1]) == 1
        if OverWrite
            Plots.plot(Ls,Z) |> display
        else
            Plots.plot!(Ls,Z) |> display
        end
    end
    [Ls Z]
end
EvaluateAlongCurve(F::Function,sol::ODESolution, Interval::Vector=[sol.t[1],sol.t[end]]; N::Int=1000) = [F(sol(t)) for t in range(Interval[1],Interval[2],length=N)]
function PlotAlongCurve(F::Function,sol::ODESolution, Interval::Vector=[sol.t[1],sol.t[end]]; N::Int=1000, OverWrite::Bool=false)
    Z = EvaluateAlongCurve(F,sol,Interval, N=N)
    if length(Z[1]) == 1
        if OverWrite
            Plots.plot(range(Interval[1],Interval[2],length=N),Z) |> display
        else
            Plots.plot!(range(Interval[1],Interval[2],length=N),Z) |> display
        end
    end
    Z
end

function SaveAdaptive(sol::ODESolution,N::Int=500; curvature = 0.003, Ntol=0.08)
    Tspan = (sol.t[1],sol.t[end]);      maxiter=30
    for _ in 1:maxiter
        T = vcat([refine_grid(x->sol(x)[i],Tspan,max_curvature=curvature)[1] for i in 1:length(sol.u[1])]...) |> unique |> sort
        if length(T) > N
            curvature *= 1.2
        elseif length(T) < (1-Ntol) * N
            curvature *= 0.85
        else
            return Homogenize(T,N)
        end
    end
    throw("SaveAdaptive: DNF in $maxiter iterations.")
end
# function SaveAdaptive(Ts::Vector,N::Int=500)
#     N > length(Ts) && return Homogenize(Ts,N)
#     N < length(Ts) && return Dehomogenize(Ts,N)
#     println("SaveAdaptive: Nothing to do here, returning input as is.")
#     Ts
# end
Homogenize(sol::ODESolution,N::Int=500) = Homogenize(sol.t,N)
function Homogenize(V::Vector,N::Int=500)
    Ts = unique(V)
    for i in 1:(N-length(Ts))
        s = findmax(diff(Ts))[2]
        insert!(Ts,s+1,Ts[s] + (Ts[s+1]-Ts[s])/2)
    end;    Ts
end
Dehomogenize(sol::ODESolution,N::Int=500) = Dehomogenize(sol.t,N)
function Dehomogenize(V::Vector,N::Int=500)
    Ts = unique(V)
    for i in 1:(length(Ts)-N)
        s = findmin(diff(Ts))[2]
        deleteat!(Ts,s+1)
    end;    Ts
end

"""
    SaveConfidence(sols::Vector,N::Int=500; sigdigits::Int=7,adaptive::Bool=true)
Returns `DataFrame` of `N` points of each `ODESolution` in `sols`. Different points correspond to different rows whereas the columns correspond to different components.
"""
function SaveConfidence(sols::Vector,N::Int=500; sigdigits::Int=7,adaptive::Bool=true)
    !isa(sols[1],ODESolution) && throw(ArgumentError("Wrong type."))
    d = length(sols[1].u[1])
    Res = Array{Float64}(undef,N,d*length(sols))
    for i in 1:length(sols)
        T = range((sols[i]).t[1],(sols[i]).t[end],length=N)
        if adaptive
            T = SaveAdaptive(sols[i],N)
        end
        Res[:,((i-1)*d+1):(d*i)] .= sols[i].(T) |> Unpack
    end
    round.(Res,sigdigits=sigdigits) |> DataFrame
end


"""
    SaveGeodesics(sols::Vector,N::Int=500; sigdigits::Int=7,adaptive::Bool=true)
Returns `DataFrame` of `N` points of each `ODESolution` in `sols`. Different points correspond to different rows whereas the columns correspond to different components.
Since the solution objects for geodesics contain the velocity as the second half of the components, only the first half of the components is saved.
"""
function SaveGeodesics(sols::Vector,N::Int=500; sigdigits::Int=7,adaptive::Bool=true)
    !isa(sols[1],ODESolution) && throw(ArgumentError("Wrong type."))
    d = length(sols[1].u[1])/2 |> Int
    Res = Array{Float64}(undef,N,d*length(sols))
    for i in 1:length(sols)
        T = range((sols[i]).t[1],(sols[i]).t[end],length=N)
        if adaptive
            T = SaveAdaptive(sols[i],N)
        end
        Res[:,((i-1)*d+1):(d*i)] .= Unpack(sols[i].(T))[:,1:d]
    end
    round.(Res,sigdigits=sigdigits) |> DataFrame
end

"""
    SaveDataSet(DS::DataSet; sigdigits::Int=0)
Returns a `DataFrame` whose columns respectively constitute the x-values, y-values and standard distributions associated with the data points.
For `sigdigits > 0` the values are rounded to the specified number of significant digits.
"""
function SaveDataSet(DS::DataSet; sigdigits::Int=0)
    !(length(DS.x[1]) == length(DS.y[1]) == length(DS.sigma[1])) && throw("Not programmed yet.")
    if sigdigits < 1
        return DataFrame([xdata(DS) ydata(DS) sigma(DS)])
    else
        return DataFrame(round.([xdata(DS) ydata(DS) sigma(DS)],sigdigits=sigdigits))
    end
end
SaveDataSet(DM::DataModel; sigdigits::Int=0) = SaveDataSet(DM.Data, sigdigits=sigdigits)

############### Curvature ################

"""
    Riemann(Metric::Function, point::Vector; BigCalc::Bool=false)
Calculates the Riemann tensor by finite differencing of the `Metric`. `BigCalc=true` increases accuracy through BigFloat calculation.
"""
function Riemann(Metric::Function,point::Vector; BigCalc::Bool=false)
    function ChristoffelPartials(Metric,point; BigCalc::Bool=false)
        if BigCalc      point = BigFloat.(point)        end
        DownUpDownDown = Array{suff(point)}(undef,length(point),length(point),length(point),length(point))
        h = GetH(suff(point))
        for i in 1:length(point)
            DownUpDownDown[i,:,:,:] .= (ChristoffelSymbol(Metric,point + h*BasisVector(i,length(point))) .- ChristoffelSymbol(Metric,point - h*BasisVector(i,length(point))))
        end
        (1/(2*h))*DownUpDownDown
    end
    DownUpDownDown = ChristoffelPartials(Metric,point,BigCalc=BigCalc)
    if (suff(point) != BigFloat) && BigCalc
        DownUpDownDown = convert(Array{Float64,4},DownUpDownDown)
    end
    Gamma = ChristoffelSymbol(Metric,point,BigCalc=BigCalc)
    # @tensor Riem[m,i,k,p] := DownUpDownDown[k,m,i,p] - DownUpDownDown[p,m,i,k] + Gamma[a,i,p]*Gamma[m,a,k] - Gamma[a,i,k]*Gamma[m,a,p]
    @tensor Riem[i,j,k,l] := DownUpDownDown[k,i,j,l] - DownUpDownDown[l,i,j,k] + Gamma[i,a,k]*Gamma[a,j,l] - Gamma[i,a,l]*Gamma[a,j,k]
end
Riemann(DM::DataModel, point::Vector; BigCalc::Bool=false) = Riem(z->FisherMetric(DM,z), point, BigCalc=BigCalc)

"""
    Ricci(Metric::Function, point::Vector; BigCalc::Bool=false)
Calculates the Ricci tensor by finite differencing of `Metric`. `BigCalc=true` increases accuracy through BigFloat calculation.
"""
function Ricci(Metric::Function, point::Vector; BigCalc::Bool=false)
    Riem = Riemann(Metric,point,BigCalc=BigCalc)
    # For some reason, it is necessary to prefill here.
    RIC = zeros(suff(point),length(point),length(point))
    @tensor RIC[a,b] = Riem[c,a,c,b]
end
Ricci(DM::DataModel, point::Vector; BigCalc::Bool=false) = Ric(z->FisherMetric(DM,z), point, BigCalc=BigCalc)

"""
    RicciScalar(Metric::Function, point::Vector; BigCalc::Bool=false)
Calculates the Ricci Scalar by finite differencing of the `Metric`. `BigCalc=true` increases accuracy through BigFloat calculation.
"""
function RicciScalar(Metric::Function,point::Vector; BigCalc::Bool=false)
    RIC = Ricci(Metric,point,BigCalc=BigCalc)
    tr(transpose(RIC)*inv(Metric(point)))
end
RicciScalar(DM::DataModel,point::Vector; BigCalc::Bool=false) = RicciScalar(z->FisherMetric(DM,z),point,BigCalc=BigCalc)

"""
    GeometricDensity(DM::DataModel,point::Vector)
Computes the square root of the determinant of the Fisher metric at `p`.
"""
GeometricDensity(DM::DataModel,point::Vector) = GeometricDensity(x->FisherMetric(DM,x),point)
GeometricDensity(Metric::Function,point::Vector) = sqrt(det(Metric(point)))


# Adaptation from PlotUtils.jl
function refine_grid(f, minmax::Tuple{Real, Real}; max_recursions = 10, max_curvature = 0.003)
    if minmax[1] > minmax[2]
        throw(ArgumentError("interval must be given as (min, max)"))
    elseif minmax[1] == minmax[2]
        x = minmax[1]
        return [x], [f(x)]
    end

    # Initial number of points
    n_points = 21
    n_intervals = n_points ÷ 2
    @assert isodd(n_points)

    xs = collect(range(minmax[1]; stop=minmax[2], length=n_points))
    # Move the first and last interior points a bit closer to the end points
    xs[2] = xs[1] + (xs[2] - xs[1]) * 0.25
    xs[end-1] = xs[end] - (xs[end] - xs[end-1]) * 0.25

    # Wiggle interior points a bit to prevent aliasing and other degenerate cases
    rng = MersenneTwister(1337)
    rand_factor = 0.05
    for i in 2:length(xs)-1
        xs[i] += rand_factor * 2 * (rand(rng) - 0.5) * (xs[i+1] - xs[i-1])
    end

    n_tot_refinements = zeros(Int, n_intervals)

    # Replace DomainErrors with NaNs
    g = function(x)
        local y
        try
            y = f(x)
        catch err
            if err isa DomainError
                y = NaN
            else
                rethrow(err)
            end
        end
        return y
    end
    # We evaluate the function on the whole interval
    fs = g.(xs)
    while true
        curvatures = zeros(n_intervals)
        active = falses(n_intervals)
        isfinite_f = isfinite.(fs)
        min_f, max_f = any(isfinite_f) ? extrema(fs[isfinite_f]) : (0.0, 0.0)
        f_range = max_f - min_f
        # Guard against division by zero later
        if f_range == 0 || !isfinite(f_range)
            f_range = one(f_range)
        end
        # Skip first and last interval
        for interval in 1:n_intervals
            p = 2 * interval
            if n_tot_refinements[interval] >= max_recursions
                # Skip intervals that have been refined too much
                active[interval] = false
            elseif !all(isfinite.(fs[[p-1,p,p+1]]))
                active[interval] = true
            else
                tot_w = 0.0
                # Do a small convolution
                for (q,w) in ((-1, 0.25), (0, 0.5), (1, 0.25))
                    interval == 1 && q == -1 && continue
                    interval == n_intervals && q == 1 && continue
                    tot_w += w
                    i = p + q
                    # Estimate integral of second derivative over interval, use that as a refinement indicator
                    # https://mathformeremortals.wordpress.com/2013/01/12/a-numerical-second-derivative-from-three-points/
                    curvatures[interval] += abs(2 * ((fs[i+1] - fs[i]) / ((xs[i+1]-xs[i]) * (xs[i+1]-xs[i-1]))
                                                    -(fs[i] - fs[i-1]) / ((xs[i]-xs[i-1]) * (xs[i+1]-xs[i-1])))
                                                    * (xs[i+1] - xs[i-1])^2) / f_range * w
                end
                curvatures[interval] /= tot_w
                # Only consider intervals with a high enough curvature
                active[interval] = curvatures[interval] > max_curvature
            end
        end
        # Approximate end intervals as being the same curvature as those next to it.
        # This avoids computing the function in the end points
        curvatures[1] = curvatures[2]
        active[1] = active[2]
        curvatures[end] = curvatures[end-1]
        active[end] = active[end-1]
        if all(x -> x >= max_recursions, n_tot_refinements[active])
            break
        end
        n_target_refinements = n_intervals ÷ 2
        interval_candidates = collect(1:n_intervals)[active]
        n_refinements = min(n_target_refinements, length(interval_candidates))
        perm = sortperm(curvatures[active])
        intervals_to_refine = sort(interval_candidates[perm[length(perm) - n_refinements + 1:end]])
        n_intervals_to_refine = length(intervals_to_refine)
        n_new_points = 2*length(intervals_to_refine)

        # Do division of the intervals
        new_xs = zeros(eltype(xs), n_points + n_new_points)
        new_fs = zeros(eltype(fs), n_points + n_new_points)
        new_tot_refinements = zeros(Int, n_intervals + n_intervals_to_refine)
        k = 0
        kk = 0
        for i in 1:n_points
            if iseven(i) # This is a point in an interval
                interval = i ÷ 2
                if interval in intervals_to_refine
                    kk += 1
                    new_tot_refinements[interval - 1 + kk] = n_tot_refinements[interval] + 1
                    new_tot_refinements[interval + kk] = n_tot_refinements[interval] + 1

                    k += 1
                    new_xs[i - 1 + k] = (xs[i] + xs[i-1]) / 2
                    new_fs[i - 1 + k] = g(new_xs[i-1 + k])

                    new_xs[i + k] = xs[i]
                    new_fs[i + k] = fs[i]

                    new_xs[i + 1 + k] = (xs[i+1] + xs[i]) / 2
                    new_fs[i + 1 + k] = g(new_xs[i + 1 + k])
                    k += 1
                else
                    new_tot_refinements[interval + kk] = n_tot_refinements[interval]
                    new_xs[i + k] = xs[i]
                    new_fs[i + k] = fs[i]
                end
            else
                new_xs[i + k] = xs[i]
                new_fs[i + k] = fs[i]
            end
        end
        xs = new_xs
        fs = new_fs
        n_tot_refinements = new_tot_refinements
        n_points = n_points + n_new_points
        n_intervals = n_points ÷ 2
    end
    return xs, fs
end
