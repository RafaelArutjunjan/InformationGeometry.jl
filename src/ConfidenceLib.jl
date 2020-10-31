

################### Probability Stuff
"""
    likelihood(DM::DataModel,θ::AbstractVector) -> Real
Calculates the likelihood ``L(\\mathrm{data} \\, | \\, \\theta)`` a `DataModel` and a parameter configuration ``\\theta``.
"""
likelihood(args...) = exp(loglikelihood(args...))

import Distributions.loglikelihood
"""
    loglikelihood(DM::DataModel, θ::AbstractVector) -> Real
Calculates the logarithm of the likelihood ``L``, i.e. ``\\ell(\\mathrm{data} \\, | \\, \\theta) \\coloneqq \\mathrm{ln} \\big( L(\\mathrm{data} \\, | \\, \\theta) \\big)`` given a `DataModel` and a parameter configuration ``\\theta``.
"""
loglikelihood(DM::AbstractDataModel,θ::AbstractVector{<:Number}) = loglikelihood(DM.Data,DM.model,θ)

function loglikelihood(DS::DataSet,model::Function,θ::AbstractVector{<:Number})
    Y = ydata(DS) - EmbeddingMap(DS,model,θ)
    -0.5*(N(DS)*ydim(DS)*log(2pi) - logdetInvCov(DS) + transpose(Y) * InvCov(DS) * Y)
end


"""
    ConfAlpha(n::Real)
Probability volume outside of a confidence interval of level n⋅σ where σ is the standard deviation of a normal distribution.
"""
ConfAlpha(n::Real) = 1.0 - ConfVol(n)

"""
    ConfVol(n::Real)
Probability volume contained in a confidence interval of level n⋅σ where σ is the standard deviation of a normal distribution.
"""
function ConfVol(n::Real)
    if abs(n) < 8
        return erf(n/sqrt(2))
    else
        println("ConfVol: Float64 precision not enough for n = $n. Returning BigFloat instead.")
        return ConfVol(BigFloat(n))
    end
end
ConfVol(n::BigFloat) = erf(n/sqrt(BigFloat(2)))

InvConfVol(q::Real; kwargs...) = sqrt(2) * erfinv(q)
InvConfVol(x::BigFloat; tol::Real=GetH(x)) = find_zero(z->(ConfVol(z)-x),one(BigFloat),Order2(),xatol=tol)

ChisqCDF(k::Int,x::BigFloat) = gamma_inc(BigFloat(k)/2., x/2., 0)[1]
ChisqCDF(k::Int,x::Real) = gamma_inc(k/2., x/2., 0)[1]
InvChisqCDF(k::Int,p::Real) = 2gamma_inc_inv(k/2., p, 1-p)


# Cannot be used with DiffEq since tags conflict. Use Zygote.jl?
AutoScore(DM::AbstractDataModel,θ::AbstractVector{<:Number}) = AutoScore(DM.Data,DM.model,θ)
AutoMetric(DM::AbstractDataModel,θ::AbstractVector{<:Number}) = AutoMetric(DM.Data,DM.model,θ)
AutoScore(DS::AbstractDataSet,model::Function,θ::AbstractVector{<:Number}) = ForwardDiff.gradient(x->loglikelihood(DS,model,x),θ)
AutoMetric(DS::AbstractDataSet,model::Function,θ::AbstractVector{<:Number}) = ForwardDiff.hessian(x->(-loglikelihood(DS,model,x)),θ)


"""
    Score(DM::DataModel, θ::AbstractVector{<:Number}; Auto::Bool=false)
Calculates the gradient of the log-likelihood ``\\ell`` with respect to a set of parameters ``\\theta``. `Auto=true` uses automatic differentiation.
"""
Score(DM::AbstractDataModel, θ::AbstractVector{<:Number}; Auto::Bool=false) = Score(DM.Data,DM.model,DM.dmodel,θ; Auto=Auto)

function Score(DS::AbstractDataSet, model::Function, dmodel::Function, θ::AbstractVector{<:Number}; Auto::Bool=false)
    Auto && return AutoScore(DS,model,θ)
    Score(DS,model,dmodel,θ)
end

function Score(DS::DataSet,model::Function,dmodel::Function,θ::AbstractVector{<:Number})
    transpose(EmbeddingMatrix(DS,dmodel,θ)) * (InvCov(DS) * (ydata(DS) - EmbeddingMap(DS,model,θ)))
end

"""
Point θ lies outside confidence region of level `Confvol` if this function > 0.
"""
WilksCriterion(DM::AbstractDataModel, θ::AbstractVector{BigFloat}, Confvol::BigFloat=ConfVol(BigFloat(1.))) = ChisqCDF(pdim(DM), 2(LogLikeMLE(DM) - loglikelihood(DM,θ))) - Confvol
WilksCriterion(DM::AbstractDataModel, θ::AbstractVector{Float64}, Confvol::Float64=ConfVol(one(suff(θ)))) = cdf(Chisq(pdim(DM)), 2(LogLikeMLE(DM) - loglikelihood(DM,θ))) - Confvol


"""
    WilksTest(DM::DataModel, θ::AbstractVector{<:Number}, Confvol=ConfVol(1)) -> Bool
Checks whether a given parameter configuration `p` is within a confidence interval of level `ConfVol` using Wilks' theorem.
This makes the assumption, that the likelihood has the form of a normal distribution, which is asymptotically correct in the limit that the number of datapoints is infinite.
"""
WilksTest(DM::AbstractDataModel, θ::AbstractVector{<:Number}, Confvol::Real=ConfVol(one(suff(θ))))::Bool = WilksCriterion(DM, θ, Confvol) < 0.



function FtestPrepared(DM::DataModel, θ::Vector, S_MLE::Real, ConfVol=ConfVol(1))::Bool
    n = length(ydata(DM));  p = length(θ);    S(P) = sum(((ydata(DM) .- map(x->DM.model(x,P),xdata(DM)))./sigma(DM)).^2)
    S(θ) <= S_MLE * (1. + p/(n-p)) * quantile(FDist(p, n-p),ConfVol)
end
Ftest(DM::DataModel, θ::Vector, MLE::Vector, Conf=ConfVol(1))::Bool = FtestPrepared(DM,θ,sum((ydata(DM) .- map(x->DM.model(x,MLE),xdata(DM))).^2),Conf)

FDistCDF(x,d1,d2) = beta_inc(d1/2.,d2/2.,d1*x/(d1*x + d2)) #, 1 .-d1*BigFloat(x)/(d1*BigFloat(x) + d2))[1]
function Ftest2(DM::DataModel, point::Vector{T}, MLE::Vector{T}, ConfVol::T=ConfVol(1))::Bool where {T<:BigFloat}
    n = length(ydata(DM));  p = length(point);    S(P) = sum(((ydata(DM) .- map(x->DM.model(x,P),xdata(DM)))./sigma(DM)).^2)
    FDistCDF(S(point) / (S(MLE) * (1 + p/(n-p))),p,n-p) <= ConfVol
end

function WilksBoundary(DM::DataModel,MLE::Vector{<:Real},Confnum::Real=1.;tol=1e-15)
    if tol < 1e-15 || suff(MLE) == BigFloat || typeof(ConfVol(Confnum)) == BigFloat
        return WilksBoundaryBig(DM,MLE,Confnum)
    end
    lMLE = loglikelihood(DM,MLE)
    A = lMLE - (1/2)*quantile(Chisq(length(MLE)),ConfVol(Confnum))
    Func(p::Real) = loglikelihood(DM,MLE .+ p*BasisVector(1,length(MLE))) - A
    D(f) = x->ForwardDiff.derivative(f,x)
    b = find_zero((Func,D(Func)),0.1,Roots.Order0(),xatol=tol)
    MLE + b.*BasisVector(1,length(MLE))
end

function WilksBoundaryBig(DM::DataModel,MLE::Vector{<:Real},Confnum::Real=1.;tol::Real=convert(BigFloat,10 .^(-precision(BigFloat)/30)))
    suff(MLE) != BigFloat && println("WilksBoundaryBig: You should pass the MLE as BigFloat!")
    print("Starting WilksBoundaryBig.   ")
    L = loglikelihood(DM,BigFloat.(MLE));    CF = ConfVol(BigFloat(Confnum))
    f(x::Real) = ChisqCDF(length(MLE),2(L-loglikelihood(DM,MLE + (x .* BasisVector(1,length(MLE)))))) - CF
    df(x) = ForwardDiff.gradient(f,x)
    @time b = find_zero((f,df),BigFloat(1),Roots.Order2(),xatol=tol)
    println("Finished.")
    MLE +  b .* BasisVector(1,length(MLE))
end

function Interval1D(DM::AbstractDataModel, Confnum::Real=1.; tol::Real=1e-14)
    tol < 2e-15 || Confnum > 8 && throw("Interval1D not programmed for BigFloat yet.")
    pdim(DM) != 1 && throw("Interval1D not defined for p != 1.")
    A = LogLikeMLE(DM) - (1/2)*InvChisqCDF(pdim(DM),ConfVol(Confnum))
    Func(p::Real) = loglikelihood(DM,MLE(DM) + p*BasisVector(1,pdim(DM))) - A
    D(f) = x->ForwardDiff.derivative(f,x);  NegFunc(x) = Func(-x)
    B = find_zero((Func,D(Func)),0.1,Roots.Order1(),xatol=tol)
    A = find_zero((Func,D(Func)),-B,Roots.Order1(),xatol=tol)
    rts = [MLE(DM)[1]+A, MLE(DM)[1]+B]
    rts[1] < rts[2] && return rts
    throw("Interval1D errored...")
end


"""
    LineSearch(Test::Function, start::Real=0; tol::Real=8e-15, maxiter::Int=10000) -> Real
Finds real number `x` where the boolean-valued `Test(x::Real)` goes from `true` to `false`.
"""
function LineSearch(Test::Function, start::Real=0; tol::Real=8e-15, maxiter::Int=10000)
    ((suff(start) != BigFloat) && tol < 1e-15) && throw("LineSearch: start not BigFloat but tol=$tol.")
    !Test(start) && throw(ArgumentError("LineSearch: Test not true for starting value."))
    stepsize = one(suff(start))/4.;  value = start
    for i in 1:maxiter
        if Test(value + stepsize) # inside
            value += stepsize
            value - start > 20 && throw("FindConfBoundary: Value larger than 20.")
        else            #outside
            if stepsize < tol
                return value
            end
            stepsize /= 5
        end
    end
    throw("$maxiter iterations over. Value=$value, Stepsize=$stepsize")
end


function FindConfBoundary(DM::DataModel,Confnum::Real; tol::Real=4e-15, maxiter::Int=10000)
    ((suff(MLE(DM)) != BigFloat) && tol < 2e-15) && throw("MLE(DM) must first be promoted to BigFloat via DM = DataModel(DM.Data,DM.model,DM.dmodel,BigFloat.(MLE(DM))).")
    Confvol = ConfVol(Confnum);    Test(x::Real) = WilksTest(DM, MLE(DM) .+ (x .* BasisVector(1,pdim(DM))), Confvol)
    !(Test(0)) && throw(ArgumentError("FindConfBoundary: Given MLE not inside Confidence Interval."))
    stepsize = one(suff(MLE(DM)))/4.;  value = zero(suff(MLE(DM)))
    for i in 1:maxiter
        if Test(value + stepsize) # inside
            value += stepsize
            value > 20 && throw("FindConfBoundary: Value larger than 10.")
        else            #outside
            if stepsize < tol
                return value .* BasisVector(1,pdim(DM)) .+ MLE(DM)
            end
            stepsize /= 10
        end
    end
    throw(Error("$maxiter iterations over. Value=$value, Stepsize=$stepsize"))
end


function FindFBoundary(DM::DataModel,MLE::Vector,Confnum::Real; tol::Real=4e-15, maxiter::Int=10000)
    S_MLE = sum(((ydata(DM) .- map(x->DM.model(x,MLE),xdata(DM)))./sigma(DM)).^2)
    CF = ConfVol(convert(suff(MLE),Confnum))
    Test(x::Real) = FtestPrepared(DM, MLE .+ (x .* BasisVector(1,length(MLE))), S_MLE, CF)
    LineSearch(Test,0,tol=tol,maxiter=maxiter) .* BasisVector(1,length(MLE)) .+ MLE
end


inversefactor(m::Real) = 1. / sqrt((m - 1.) + (m - 1.)^2)
function GetAlpha(n::Int)
    V = Vector{Float64}(undef,n)
    fill!(V,-inversefactor(n))
    V[end] = (n-1) * inversefactor(n)
    V
end

"""
    OrthVF(DM::DataModel, θ::AbstractVector{<:Real}; Auto::Bool=false) -> Vector
Calculates a direction (in parameter space) in which the value of the log-likelihood does not change, given a parameter configuration ``\\theta``.
`Auto=true` uses automatic differentiation to calculate the score.
"""
function OrthVF(DM::AbstractDataModel, θ::AbstractVector{<:Number};
                alpha::AbstractVector=GetAlpha(length(θ)), Auto::Bool=false)
    OrthVF(DM.Data,DM.model,DM.dmodel,θ; alpha=alpha, Auto=Auto)
end

function OrthVF(DS::AbstractDataSet, model::Function, dmodel::Function, θ::AbstractVector{<:Number};
                alpha::AbstractVector=GetAlpha(length(θ)), Auto::Bool=false)
    length(θ) < 2 && throw(ArgumentError("dim(Parameter Space) < 2  --> No orthogonal VF possible."))
    S = -Score(DS,model,dmodel,θ; Auto=Auto);    P = prod(S);    VF = P ./ S
    alpha .* VF |> normalize
end


"""
    OrthVF(DM::DataModel, PL::Plane, θ::Vector{<:Real}; Auto::Bool=false) -> Vector
Calculates a direction (in parameter space) in which the value of the log-likelihood does not change, given a parameter configuration ``\\theta``.
If a `Plane` is specified, the direction will be specified in the planar coordinates using a 2-component vector.
`Auto=true` uses automatic differentiation to calculate the score.
"""
function OrthVF(DM::AbstractDataModel,PL::Plane,θ::AbstractVector{<:Number}; Auto::Bool=false)
    S = transpose([PL.Vx PL.Vy]) * (-Score(DM,PlaneCoordinates(PL,θ);Auto=Auto))
    P = prod(S)
    return SA[-P/S[1],P/S[2]] |> normalize
end


"""
    ConstructCube(M::Matrix{<:Real}; Padding::Real=1/50) -> HyperCube
Returns a `HyperCube` which encloses the extrema of the columns of the input matrix.
"""
function ConstructCube(M::AbstractMatrix{<:Real}; Padding::Real=1/50)
    lowers = [minimum(M[:,i]) for i in 1:size(M,2)]
    uppers = [maximum(M[:,i]) for i in 1:size(M,2)]
    diff = (uppers - lowers) .* Padding
    HyperCube(lowers - diff,uppers + diff)
end
ConstructCube(V::AbstractVector{<:Real}) = HyperCube(extrema(V))
ConstructCube(PL::Plane,sol::ODESolution; Padding::Real=1/50) = ConstructCube(Deplanarize(PL,sol;N=300); Padding=Padding)

function ConstructCube(sol::ODESolution,Npoints::Int=200; Padding::Real=1/50)
    ConstructCube(Unpack(map(sol,range(sol.t[1],sol.t[end],length=Npoints))); Padding=Padding)
end


MonteCarloArea(Test::Function,Cube::HyperCube,N::Int=Int(1e7)) = CubeVol(Cube) * MonteCarloRatio(Test,Cube,N)
function MonteCarloRatio(Test::Function,Cube::HyperCube,N::Int=Int(1e7))
    (1/N)* @distributed (+) for i in 1:N
        Test(rand.(Uniform.(Cube.L,Cube.U)))
    end
end


function MonteCarloRatioWE(Test::Function,LU::HyperCube,N::Int=Int(1e7); chunksize::Int=Int(N/20))
    chunksize > N && error("chunksize > N")
    if N%chunksize != 0
        println("N % chunksize = $(N%chunksize). Rounded N up from $N to $(N + N%chunksize + 1).")
        N += Int(N%chunksize + 1)
    end
    chunks = Int(N/chunksize)
    # Output not normalized by chunksize
    function CarloLoop(Test::Function,LU::HyperCube,chunksize::Int)
        tot = [rand.(Uniform.(LU.L,LU.U)) for i in 1:chunksize] .|> Test
        res = sum(tot)
        [res, sum((tot.-(res/chunksize)).^2)]
    end
    Tot = @distributed (+) for i in 1:chunks
        CarloLoop(Test,LU,chunksize)
    end
    measurement(Tot[1]/N, sqrt(1/((N-1)*N) * Tot[2]))
end
MonteCarloAreaWE(Test::Function,Space::HyperCube,N::Int=Int(1e7)) = CubeVol(Space)*MonteCarloRatioWE(Test,Space,N)


function ConfidenceRegionVolume(DM::DataModel,sol::ODESolution,N::Int=Int(1e5); WE::Bool=false)
    length(sol.u[1]) != 2 && throw("Not Programmed for dim > 2 yet.")
    LogLikeBoundary = likelihood(DM,sol(0))
    Cube = ConstructCube(sol; Padding=1e-5)
    # Indicator function for Integral
    InsideRegion(X::AbstractVector{<:Real})::Bool = loglikelihood(DM,X) < LogLikeBoundary
    Test(X::AbstractVector) = InsideRegion(X) ? sqrt(det(FisherMetric(DM,X))) : 0.
    WE && return MonteCarloAreaWE(Test,Cube,N)
    MonteCarloArea(Test,Cube,N)
end


FindMLEBig(DM::DataModel,start::AbstractVector{<:Number}=MLE(DM)) = FindMLEBig(DM.Data,DM.model,convert(Vector,start))
function FindMLEBig(DS::AbstractDataSet,model::Function,start::Union{Bool,AbstractVector}=false)
    if isa(start,Vector)
        NegEll(p::AbstractVector{<:Number}) = -loglikelihood(DS,model,p)
        return optimize(NegEll, BigFloat.(convert(Vector,start)), BFGS(), Optim.Options(g_tol=convert(BigFloat,10^(-precision(BigFloat)/30))), autodiff = :forward) |> Optim.minimizer
    elseif isa(start,Bool)
        return FindMLEBig(DS,model,FindMLE(DS,model))
    end
end

function GetStartP(DS::AbstractDataSet,model::Function)
    P = pdim(DS,model)
    ones(P) .+ 0.01*(rand(P) .- 0.5)
end

FindMLE(DM::DataModel,args...;kwargs...) = MLE(DM)
function FindMLE(DS::AbstractDataSet,model::Function,start::Union{Bool,AbstractVector}=false; Big::Bool=false, tol::Real=1e-14)
    (Big || tol < 2.3e-15) && return FindMLEBig(DS,model,start)
    # NegEll(p::AbstractVector{<:Number}) = -loglikelihood(DS,model,p)
    if isa(start,Bool)
        return curve_fit(DS,model,GetStartP(DS,model); tol=tol).param
        # return optimize(NegEll, ones(pdim(DS,model)), BFGS(), Optim.Options(g_tol=tol), autodiff = :forward) |> Optim.minimizer
    elseif isa(start,AbstractVector)
        if suff(start) == BigFloat
            return FindMLEBig(DS,model,convert(Vector,start))
        else
            return curve_fit(DS,model,start; tol=tol).param
            # return optimize(NegEll, convert(Vector,start), BFGS(), Optim.Options(g_tol=tol), autodiff = :forward) |> Optim.minimizer
        end
    end
end



"""
    GenerateBoundary(DM::DataModel, u0::AbstractVector{<:Number}; tol::Real=1e-14, meth=Tsit5(), mfd::Bool=true) -> ODESolution
Basic method for constructing a curve lying on the confidence region associated with the initial configuration `u0`.
"""
function GenerateBoundary(DM::AbstractDataModel,u0::AbstractVector{<:Number}; tol::Real=1e-12,
                            meth::OrdinaryDiffEqAlgorithm=Tsit5(), mfd::Bool=true, Auto::Bool=false, kwargs...)
    GenerateBoundary(DM.Data,DM.model,DM.dmodel,u0; tol=tol, meth=meth, mfd=mfd, Auto=Auto, kwargs...)
end

function GenerateBoundary(DS::AbstractDataSet,model::Function,dmodel::Function,u0::AbstractVector{<:Number}; tol::Real=1e-12,
                            meth::OrdinaryDiffEqAlgorithm=Tsit5(), mfd::Bool=true, Auto::Bool=false, kwargs...)
    LogLikeOnBoundary = loglikelihood(DS,model,u0)
    IntCurveODE!(du,u,p,t)  =  du .= 0.1 .* OrthVF(DS,model,dmodel,u; Auto=Auto)
    g!(resid,u,p,t)  =  resid[1] = LogLikeOnBoundary - loglikelihood(DS,model,u)
    terminatecondition(u,t,integrator) = u[2] - u0[2]
    # TerminateCondition only on upwards crossing --> supply two different affect functions, leave second free I
    cb = CallbackSet(ManifoldProjection(g!),ContinuousCallback(terminatecondition,terminate!,nothing))
    tspan = (0.,1e5);    prob = ODEProblem(IntCurveODE!,u0,tspan)
    if mfd
        return solve(prob,meth; reltol=tol,abstol=tol,callback=cb,kwargs...)
    else
        return solve(prob,meth; reltol=tol,abstol=tol,callback=ContinuousCallback(terminatecondition,terminate!,nothing),kwargs...)
    end
end

function GenerateBoundary(DM::AbstractDataModel, PL::Plane, u0::AbstractVector{<:Number};
                    tol::Real=1e-12, meth::OrdinaryDiffEqAlgorithm=Tsit5(), mfd::Bool=false, Auto::Bool=false, kwargs...)
    length(u0) != 2 && throw("length(u0) != 2 although a Plane was specified.")
    LogLikeOnBoundary = loglikelihood(DM,PlaneCoordinates(PL,u0))
    IntCurveODE!(du,u,p,t)  =  du .= 0.1 * OrthVF(DM,PL,u; Auto=Auto)
    g!(resid,u,p,t)  =  resid[1] = LogLikeOnBoundary - loglikelihood(DM,PlaneCoordinates(PL,u))
    terminatecondition(u,t,integrator) = u[2] - u0[2]
    cb = CallbackSet(ManifoldProjection(g!),ContinuousCallback(terminatecondition,terminate!,nothing))
    tspan = (0.,1e5);    prob = ODEProblem(IntCurveODE!,u0,tspan)
    if mfd
        return solve(prob,meth; reltol=tol,abstol=tol,callback=cb, kwargs...)
    else
        return solve(prob,meth; reltol=tol,abstol=tol,callback=ContinuousCallback(terminatecondition,terminate!,nothing), kwargs...)
    end
end

"""
Choose method depending on dimensionality of the parameter space.
"""
function ConfidenceRegion(DM::AbstractDataModel, Confnum::Real=1.; tol::Real=1e-12, meth::OrdinaryDiffEqAlgorithm=Tsit5(), mfd::Bool=true, Auto::Bool=false, kwargs...)
    if pdim(DM) == 1
        return Interval1D(DM, Confnum; tol=tol)
    elseif pdim(DM) == 2
        return GenerateBoundary(DM, FindConfBoundary(DM, Confnum; tol=tol); tol=tol, meth=meth, mfd=mfd, Auto=Auto, kwargs...)
    else
        println("ConfidenceRegion() computes solutions in the θ[1]-θ[2] plane which are separated in the θ[3] direction. For more explicit control, call MincedBoundaries() and set options manually.")
        Cube = LinearCuboid(DM,Confnum)
        Planes = IntersectCube(DM,Cube,Confnum; Dirs=[1,2,3], N=30)
        return Planes, MincedBoundaries(DM,Planes,Confnum; tol=tol, Auto=Auto, meth=meth, mfd=false)
    end
end


IsStructurallyIdentifiable(DM::AbstractDataModel,sol::ODESolution)::Bool = length(StructurallyIdentifiable(DM,sol)) == 0

function StructurallyIdentifiable(DM::AbstractDataModel,sol::ODESolution)
    find_zeros(t->GeometricDensity(x->FisherMetric(DM,x),sol(t)), sol.t[1], sol.t[end])
end
StructurallyIdentifiable(DM::AbstractDataModel,sols::Vector{<:ODESolution}) = map(x->StructurallyIdentifiable(DM,x), sols)



"""
    ConfidenceRegions(DM::DataModel, Range::Union{AbstractRange,AbstractVector})
Computes the boundaries of confidence regions for two-dimensional parameter spaces given a vector or range of confidence levels.
A convenient interface which extends this to higher dimensions is currently still under development.

For example,
```julia
ConfidenceRegions(DM, 1:3; tol=1e-9)
```
computes the ``1\\sigma``, ``2\\sigma`` and ``3\\sigma`` confidence regions associated with a given `DataModel` using a solver tolerance of ``10^{-9}``.

Keyword arguments:
* `IsConfVol = true` can be used to specify the desired confidence level directly in terms of a probability ``p \\in [0,1]`` instead of in units of standard deviations ``\\sigma``,
* `tol` can be used to quantify the tolerance with which the ODE which defines the confidence boundary is solved (default `tol = 1e-12`),
* `meth` can be used to specify the solver algorithm (default `meth = Tsit5()`),
* `Auto` is a boolean which controls whether the derivatives of the likelihood are computed using automatic differentiation or an analytic expression involving `DM.dmodel` (default `Auto = false`).
"""
function ConfidenceRegions(DM::DataModel, Confnums::Union{AbstractRange,AbstractVector}; IsConfVol::Bool=false,
                        tol::Real=1e-12, meth::OrdinaryDiffEqAlgorithm=Tsit5(), mfd::Bool=true, Auto::Bool=false,
                        CheckSols::Bool=true, kwargs...)
    Range = IsConfVol ? InvConfVol.(Confnums) : Confnums
    if pdim(DM) == 1
        return map(x->ConfidenceRegion(DM,x; tol=tol), Range)
    elseif pdim(DM) == 2
        sols = map(x->ConfidenceRegion(DM,x; tol=tol,meth=meth,mfd=mfd,Auto=Auto,kwargs...), Range)
        if CheckSols
            NotTerminated = map(x->(x.retcode != :Terminated), sols)
            sum(NotTerminated) != 0 && println("Solutions $((1:length(sols))[NotTerminated]) did not exit properly.")
            roots = StructurallyIdentifiable(DM,sols)
            Unidentifiables = map(x->(length(x) != 0), roots)
            for i in 1:length(roots)
                length(roots[i]) != 0 && println("Solution $i hits chart boundary at t = $(roots[i]) and is therefore invalid.")
            end
        end
        return sols
    else
        throw("This functionality is still under construction. Use ConfidenceRegion() instead.")
    end
end

MultipleConfidenceRegions(args...;kwargs...) = ConfidenceRegions(args...;kwargs...)
@deprecate MultipleConfidenceRegions(DM,Confnums) ConfidenceRegions(DM,Confnums)


function CurveInsideInterval(Test::Function, sol::ODESolution, N::Int = 1000)
    NoPoints = Vector{Vector{suff(sol.u)}}(undef,0)
    for t in range(sol.t[1],sol.t[end], length=N)
        num = sol(t)
        if !Test(num)
            push!(NoPoints,num)
        end
    end
    println("CurveInsideInterval: Solution has $(length(NoPoints))/$N points outside the desired confidence interval.")
    return NoPoints
end


function Inside(Cube::HyperCube,p::AbstractVector{<:Real})::Bool
    length(Cube) != length(p) && throw("Inside: Dimension mismatch between Cube and point.")
    for i in 1:length(Cube)
        !(Cube.L[i] <= p[i] <= Cube.U[i]) && return false
    end;    true
end

"""
    Rsquared(DM::DataModel) -> Real
Calculates the R² value associated with the maximum likelihood estimate of a `DataModel`. It should be noted that the R² value is only a valid measure for the goodness of a fit for linear relationships.
"""
function Rsquared(DM::DataModel)
    !(xdim(DM) == ydim(DM) == 1) && return -1
    mean = sum(ydata(DM)) / length(ydata(DM))
    Stot = (ydata(DM) .- mean).^2 |> sum
    Sres = (ydata(DM) - EmbeddingMap(DM,MLE(DM))).^2 |> sum
    1 - Sres / Stot
end


"""
    Integrate1D(F::Function, Cube::HyperCube; tol::Real=1e-14, fullSol::Bool=false, meth=nothing)
Integrates `F` over a one-dimensional domain specified via a `HyperCube` by rephrasing the integral as an ODE and using `DifferentialEquations.jl`.
"""
function Integrate1D(F::Function, Cube::HyperCube; tol::Real=1e-14, fullSol::Bool=false, meth=nothing)
    length(Cube) != 1 && throw(ArgumentError("Cube dim = $(length(Cube)) instead of 1"))
    Integrate1D(F,(Cube.L[1],Cube.U[1]); tol=tol,fullSol=fullSol,meth=meth)
end
Integrate1D(F::Function, Interval::AbstractVector{<:Real}; tol::Real=1e-14, fullSol::Bool=false, meth=nothing) = Integrate1D(F, Tuple(Interval); tol=tol, fullSol=fullSol, meth=meth)
function Integrate1D(F::Function, Interval::Tuple{<:Real,<:Real}; tol::Real=1e-14, fullSol::Bool=false, meth=nothing)
    Interval = float.(Interval)
    !(0. < tol < 1.) && throw("Integrate1D: tol unsuitable")
    Interval[1] > Interval[2] && throw(ArgumentError("Interval orientation wrong."))
    f(u,p,t) = F(t)
    if tol < 1e-15
        u0 = BigFloat(0.);        Interval = BigFloat.(Interval)
        meth = (meth == nothing) ? Vern9() : meth
    else
        u0 = 0.
        meth = (meth == nothing) ? Tsit5() : meth
    end
    if fullSol
        return solve(ODEProblem(f,u0,Interval),meth,reltol=tol,abstol=tol)
    else
        return solve(ODEProblem(f,u0,Interval),meth,reltol=tol,abstol=tol,save_everystep=false,save_start=false,save_end=true).u[end]
    end
end

"""
    IntegrateND(F::Function,Cube::HyperCube; tol::Real=1e-12, WE::Bool=false, kwargs...)
Integrates the function `F` over `Cube` with the help of **HCubature.jl** to a tolerance of `tol`.
If `WE=true`, the result is returned as a `Measurement` which also contains the estimated error in the result.
"""
function IntegrateND(F::Function,Cube::HyperCube; tol::Real=1e-12, WE::Bool=false, kwargs...)
    if length(Cube) == 1
        val, uncert = hquadrature(F, Cube.L[1], Cube.U[1]; rtol=tol, atol=tol, kwargs...)
    else
        val, uncert = hcubature(F, Cube.L, Cube.U; rtol=tol, atol=tol, kwargs...)
    end
    if length(val) == 1
        return WE ? measurement(val[1],uncert[1]) : val[1]
    else
        return WE ? measurement.(val,uncert) : val
    end
end
IntegrateND(F::Function, L::AbstractVector{<:Real}, U::AbstractVector{<:Real}; tol::Real=1e-12, WE::Bool=false, kwargs...) = IntegrateND(F,HyperCube(L,U); tol=tol, WE=WE, kwargs...)
IntegrateND(F::Function, Interval::Union{AbstractVector{<:Real},Tuple{<:Real,<:Real}}; tol::Real=1e-12, WE::Bool=false, kwargs...) = IntegrateND(F,HyperCube(Interval); tol=tol, WE=WE, kwargs...)


# Assume that sums from Fisher metric defined with first derivatives of loglikelihood pull out
"""
    FisherMetric(DM::DataModel, θ::AbstractVector{<:Number})
Computes the Fisher metric ``g`` given a `DataModel` and a parameter configuration ``\\theta`` under the assumption that the likelihood ``L(\\mathrm{data} \\, | \\, \\theta)`` is a multivariate normal distribution.
```math
g_{ab}(\\theta) \\coloneqq -\\int_{\\mathcal{D}} \\mathrm{d}^m y_{\\mathrm{data}} \\, L(y_{\\mathrm{data}} \\,|\\, \\theta) \\, \\frac{\\partial^2 \\, \\mathrm{ln}(L)}{\\partial \\theta^a \\, \\partial \\theta^b} = -\\mathbb{E} \\bigg( \\frac{\\partial^2 \\, \\mathrm{ln}(L)}{\\partial \\theta^a \\, \\partial \\theta^b} \\bigg)
```
"""
FisherMetric(DM::AbstractDataModel, θ::AbstractVector{<:Number}) = Pullback(DM, InvCov(DM), θ)

# function FisherMetric(DM::DataModel, θ::Vector{<:Real})
#     F = zeros(suff(θ),length(θ),length(θ))
#     dmod = sigma(DM).^(-1) .* map(z->DM.dmodel(z,θ),xdata(DM))
#     for i in 1:length(xdata(DM))
#         F .+=  transpose(dmod[i]) * dmod[i]
#     end;    F
# end



# Always use Integreat for 1D -> Check Dim of Space to decide?
"""
    KullbackLeibler(p::Function,q::Function,Domain::HyperCube=HyperCube([-15,15]); tol=2e-15, N::Int=Int(3e7), Carlo::Bool=(length(Domain)!=1))
Computes the Kullback-Leibler divergence between two probability distributions `p` and `q` over the `Domain`.
If `Carlo=true`, this is done using a Monte Carlo Simulation with `N` samples.
If the `Domain` is one-dimensional, the calculation is performed without Monte Carlo to a tolerance of ≈ `tol`.
```math
D_{\\text{KL}}[p,q] \\coloneqq \\int \\mathrm{d}^m y \\, p(y) \\, \\mathrm{ln} \\bigg( \\frac{p(y)}{q(y)} \\bigg)
```
"""
function KullbackLeibler(p::Function,q::Function,Domain::HyperCube=HyperCube([-15,15]); tol::Real=1e-14, N::Int=Int(3e7), Carlo::Bool=(length(Domain)!=1))
    function Integrand(x)
        P = p(x)[1];   Q = q(x)[1];   Rat = P/Q
        (Rat <= 0. || !isfinite(Rat)) && throw(ArgumentError("Ratio p(x)/q(x) = $Rat in log(p/q) for x=$x."))
        P*log(Rat)
    end
    if Carlo
        length(Domain) == 1 && return MonteCarloArea(Integrand,Domain,N)[1]
        return MonteCarloArea(Integrand,Domain,N)
    elseif length(Domain) == 1
        return Integrate1D(Integrand,Domain; tol=tol)
    else
        throw("KL: Carlo=false and dim(Domain) != 1. Aborting.")
    end
end

function KullbackLeibler(p::Distribution,q::Distribution,Domain::HyperCube=HyperCube([[-15,15] for i in 1:length(p)]);
    tol::Real=1e-14, N::Int=Int(3e7), Carlo::Bool=(length(Domain)!=1))
    !(length(p) == length(q) == length(Domain)) && throw("KL: Sampling dimension mismatch: dim(p) = $(length(p)), dim(q) = $(length(q)), Domain = $(length(Domain))")
    function Integrand(x)
        P = logpdf(p,x);   Q = logpdf(q,x)
        exp(P)*(P - Q)
    end
    function Integrand1D(x)
        # Apparently, logpdf() without broadcast is deprecated for univariate distributions.
        P = logpdf.(p,x);   Q = logpdf.(q,x)
        exp.(P)*(P - Q)[1]
    end
    if Carlo
        length(p) == 1 && return MonteCarloArea(Integrand1D,Domain,N)[1]
        return MonteCarloArea(Integrand,Domain,N)
    elseif length(Domain) == 1
        return Integrate1D(Integrand1D,Domain; tol=tol)
    else
        throw("KL: Carlo=false and dim(Domain) != 1. Aborting.")
    end
end

# function KullbackLeibler(p::Product, q::Product, Domain::HyperCube=HyperCube([[-15,15] for i in 1:length(p)]); tol::Real=1e-14, kwargs...)
#     !(length(p) == length(q) == length(Domain)) && throw("KL: Sampling dimension mismatch: dim(p) = $(length(p)), dim(q) = $(length(q)), Domain = $(length(Domain))")
#     prod(KullbackLeibler(p.v[i], q.v[i], HyperCube([Domain.L[i], Domain.U[i]]); tol=tol) for i in 1:length(p))
# end


# Analytic expressions
function KullbackLeibler(P::MvNormal,Q::MvNormal,args...;kwargs...)
    length(P.μ) != length(Q.μ) && throw("Normals not of same dim.")
    (1/2) * (log(det(Q.Σ.mat)/det(P.Σ.mat)) - length(P.μ) + tr(inv(Q.Σ.mat) * P.Σ.mat) + transpose(Q.μ-P.μ) * inv(Q.Σ.mat) * (Q.μ-P.μ))
end
KullbackLeibler(P::Normal,Q::Normal,args...;kwargs...) = log(Q.σ/P.σ) + (1/2) * ((P.σ/Q.σ)^2 + (P.μ-Q.μ)^2 * Q.σ^(-2) -1.)
KullbackLeibler(P::Cauchy,Q::Cauchy,args...;kwargs...) = log(((P.σ+Q.σ)^2 + (P.μ-Q.μ)^2) / (4P.σ*Q.σ))
# Note the sign difference between the conventions (1/θ)exp(-x/θ) and λexp(-λx). Distributions.jl uses the first.
KullbackLeibler(P::Exponential,Q::Exponential,args...;kwargs...) = log(Q.θ/P.θ) + P.θ/Q.θ -1
function KullbackLeibler(P::Weibull,Q::Weibull,args...;kwargs...)
    log(P.α/(P.θ^P.α)) - log(Q.α/(Q.θ^Q.α)) + (P.α - Q.α)*(log(P.θ) - Base.MathConstants.γ/P.α) + (P.θ/Q.θ)^Q.α * SpecialFunctions.gamma(1 + Q.α/P.α) -1
end
function KullbackLeibler(P::Distributions.Gamma,Q::Distributions.Gamma,args...;kwargs...)
    (P.α - Q.α) * digamma(P.α) - loggamma(P.α) + loggamma(Q.α) + Q.α*log(Q.θ/P.θ) + P.α*(P.θ/Q.θ - 1)
end

# Add Wishart, Beta, Gompertz, generalized gamma


# """
#     NormalDist(DM::DataModel,p::Vector) -> Distribution
# Constructs either `Normal` or `MvNormal` type from `Distributions.jl` using data and a parameter configuration.
# This makes the assumption, that the errors associated with the data are normal.
# """
# function NormalDist(DM::DataModel,p::Vector)::Distribution
#     if length(ydata(DM)[1]) == 1
#         length(ydata(DM)) == 1 && return Normal(ydata(DM)[1] .- DM.model(xdata(DM)[1],p),sigma(DM)[1])
#         return MvNormal(ydata(DM) .- map(x->DM.model(x,p),xdata(DM)),diagm(float.(sigma(DM).^2)))
#     else
#         throw("Not programmed yet.")
#     end
# end

"""
    KullbackLeibler(DM::DataModel,p::Vector,q::Vector)
Calculates Kullback-Leibler divergence under the assumption of a normal likelihood.
"""
KullbackLeibler(DM::DataModel,p::AbstractVector,q::AbstractVector) = KullbackLeibler(NormalDist(DM,p),NormalDist(DM,q))

KullbackLeibler(DM::DataModel,p::AbstractVector) = KullbackLeibler(MvNormal(zeros(length(ydata(DM))),inv(InvCov(DM))),NormalDist(DM,p))


# h(θ) ∈ Dataspace
"""
    EmbeddingMap(DM::DataModel,θ::AbstractVector{<:Number})
Returns a vector of the collective predictions of the `model` as evaluated at the x-values and the parameter configuration ``\\theta``.
```
h(\\theta) \\coloneqq \\big(y_\\mathrm{model}(x_1;\\theta),...,y_\\mathrm{model}(x_N;\\theta)\\big) \\in \\mathcal{D}
```
"""
EmbeddingMap(DM::AbstractDataModel,θ::AbstractVector{<:Number}) = EmbeddingMap(DM.Data,DM.model,θ)

EmbeddingMap(DS::AbstractDataSet,model::Function,θ::AbstractVector{<:Number}) = performMap(DS,model,θ,WoundX(DS))

function performMap(DS::AbstractDataSet,model::Function,θ::AbstractVector{<:Number},woundX::AbstractVector)
    if ydim(DS) > 1
        return reduce(vcat,map(x->model(x,θ),woundX))
    else
        return map(x->model(x,θ),woundX)
    end
end

function performMap2(DS::AbstractDataSet,model::Function,θ::AbstractVector{<:Number},woundX::AbstractVector)
    if ydim(DS) > 1
        Res = Vector{suff(θ)}(undef,N(DS)*ydim(DS))
        for i in 1:N(DS)
            Res[1+(i-1)*ydim(DS):(i*ydim(DS))] = model(woundX[i],θ)
        end;    return Res
    else
        return map(x->model(x,θ),woundX)
    end
end

function performMap3(DS::AbstractDataSet,model::Function,θ::AbstractVector{<:Number},woundX::AbstractVector{<:SArray})
    # if model outputs StaticArrays
    reinterpret(suff(θ),map(x->model(x,θ),woundX))
end


EmbeddingMatrix(DM::AbstractDataModel,θ::AbstractVector{<:Number}) = EmbeddingMatrix(DM.Data,DM.dmodel,θ)

EmbeddingMatrix(DS::AbstractDataSet,dmodel::Function,θ::AbstractVector{<:Number}) = performDMap(DS,dmodel,float.(θ),WoundX(DS))

performDMap(DS::AbstractDataSet,dmodel::Function,θ::AbstractVector{<:Number},woundX::AbstractVector) = reduce(vcat,map(x->dmodel(x,θ),woundX))

# very slightly faster apparently
function performDMap2(DS::AbstractDataSet,dmodel::Function,θ::AbstractVector{<:Number},woundX::AbstractVector)
    Res = Array{suff(θ)}(undef,N(DS)*ydim(DS),length(θ))
    for i in 1:N(DS)
        Res[1+(i-1)*ydim(DS):(i*ydim(DS)),:] = dmodel(woundX[i],θ)
    end;    Res
end



# M ⟵ D
Pullback(DM::DataModel,F::Function,θ::AbstractVector{<:Number}) = F(EmbeddingMap(DM,θ))
"""
    Pullback(DM::DataModel, ω::AbstractVector{<:Real}, θ::Vector) -> Vector
Pull-back of a covector to the parameter manifold.
"""
Pullback(DM::DataModel, ω::AbstractVector{<:Real}, θ::AbstractVector{<:Number}) = transpose(EmbeddingMatrix(DM,θ)) * ω


"""
    Pullback(DM::DataModel, G::AbstractArray{<:Real,2}, θ::Vector)
Pull-back of a (0,2)-tensor `G` to the parameter manifold.
"""
function Pullback(DM::AbstractDataModel, G::AbstractMatrix, θ::AbstractVector{<:Number})
    J = EmbeddingMatrix(DM,θ)
    return transpose(J) * G * J
end

# M ⟶ D
"""
    Pushforward(DM::DataModel, X::AbstractVector, θ::AbstractVector)
Calculates the push-forward of a vector `X` from the parameter manifold to the data space.
"""
Pushforward(DM::DataModel, X::Vector, θ::AbstractVector{<:Number}) = EmbeddingMatrix(DM,θ) * X



# Compute all major axes of Fisher Ellipsoid from eigensystem of Fisher metric
FisherEllipsoid(DM::DataModel, θ::AbstractVector{<:Number}) = FisherEllipsoid(p->FisherMetric(DM,p), θ)
FisherEllipsoid(Metric::Function, θ::AbstractVector{<:Number}) = eigvecs(Metric(θ))


"""
    AIC(DM::DataModel, θ::AbstractVector) -> Real
Calculates the Akaike Information Criterion given a parameter configuration ``\\theta`` defined by ``\\mathrm{AIC} = 2 \\, \\mathrm{length}(\\theta) -2 \\, \\ell(\\mathrm{data} \\, | \\, \\theta)``.
Lower values for the AIC indicate that the associated model function is more likely to be correct. For linearly parametrized models and small sample sizes, it is advisable to instead use the AICc which is more accurate.
"""
AIC(DM::AbstractDataModel, θ::AbstractVector{<:Number}) = 2length(θ) - 2loglikelihood(DM,θ)
AIC(DM::DataModel) = AIC(DM,MLE(DM))

"""
    AICc(DM::DataModel, θ::AbstractVector) -> Real
Computes Akaike Information Criterion with an added correction term that prevents the AIC from selecting models with too many parameters (i.e. overfitting) in the case of small sample sizes.
``\\mathrm{AICc} = \\mathrm{AIC} + \\frac{2\\mathrm{length}(\\theta)^2 + 2 \\mathrm{length}(\\theta)}{N - \\mathrm{length}(\\theta) - 1}`` where ``N`` is the number of data points.
Whereas AIC constitutes a first order estimate of the information loss, the AICc constitutes a second order estimate. However, this particular correction term assumes that the model is **linearly parametrized**.
"""
AICc(DM::AbstractDataModel, θ::AbstractVector{<:Number}) = AIC(DM,θ) + (2length(θ)^2 + 2length(θ)) / (N(DM.Data) - length(θ) - 1)
AICc(DM::DataModel) = AICc(DM,MLE(DM))

"""
    BIC(DM::DataModel, θ::AbstractVector) -> Real
Calculates the Bayesian Information Criterion given a parameter configuration ``\\theta`` defined by ``\\mathrm{BIC} = \\mathrm{ln}(N) \\cdot \\mathrm{length}(\\theta) -2 \\, \\ell(\\mathrm{data} \\, | \\, \\theta)`` where ``N`` is the number of data points.
"""
BIC(DM::AbstractDataModel, θ::AbstractVector{<:Number}) = length(θ)*log(N(DM.Data)) - 2loglikelihood(DM,θ)
BIC(DM::DataModel) = BIC(DM,MLE(DM))


"""
    ModelComparison(DM1::AbstractDataModel, DM2::AbstractDataModel) -> Tuple{Int,Real}
Compares the AICc values of both models at best fit and estimates probability that one model is more likely than the other.
First entry of tuple returns which model is more likely to be correct (1 or 2) whereas the second entry returns the ratio of probabilities.
"""
function ModelComparison(DM1::AbstractDataModel, DM2::AbstractDataModel)
    !(ydata(DM1) == ydata(DM2) && xdata(DM1) == xdata(DM2)) && throw("Not comparing against same data!")
    Mod1 = AICc(DM1,MLE(DM1));      Mod2 = AICc(DM2,MLE(DM2))
    res = (Int((Mod1 > Mod2) + 1), round(exp(0.5*abs(Mod2-Mod1)),sigdigits=5))
    println("Model $(res[1]) is estimated to be $(res[2]) times as likely to be correct from difference in AICc values.")
    res
end


"""
    IsLinearParameter(DM::DataModel) -> Vector{Bool}
Checks with respect to which parameters the model function `model(x,θ)` is linear and returns vector of booleans where `true` indicates linearity.
This test is performed by comparing the Jacobians of the model for two random configurations ``\\theta_1, \\theta_2 \\in \\mathcal{M}`` column by column.
"""
function IsLinearParameter(DM::AbstractDataModel)::Vector{Bool}
    P = pdim(DM);    J1 = EmbeddingMatrix(DM,rand(P));    J2 = EmbeddingMatrix(DM,rand(P))
    [J1[:,i] == J2[:,i]  for i in 1:size(J1,2)]
end

"""
    IsLinear(DM::DataModel) -> Bool
Checks whether the `model(x,θ)` function is linear with respect to all of its parameters ``\\theta \\in \\mathcal{M}``.
A componentwise check can be attained via the method `IsLinearParameter(DM)`.
"""
function IsLinear(DM::AbstractDataModel)::Bool
    res = IsLinearParameter(DM)
    sum(res) == length(res)
end

"""
    LeastInformativeDirection(DM::DataModel,θ::AbstractVector{<:Number}=MLE(DM)) -> Vector{Float64}
Returns a vector which points in the direction in which the likelihood decreases most slowly.
"""
function LeastInformativeDirection(DM::AbstractDataModel,θ::AbstractVector{<:Number}=MLE(DM))
    M = eigen(FisherMetric(DM,θ));  i = findmin(M.values)[2]
    M.vectors[:,i] / sqrt(M.values[i])
end


"""
    FindConfBoundaryOnPlane(DM::AbstractDataModel,PL::Plane,Confnum::Real=1.; tol::Real=1e-12) -> Union{Vector{Real},Bool}
Computes point inside the plane `PL` which lies on the boundary of a confidence region of level `Confnum`.
If such a point cannot be found (i.e. does not seem to exist), the method returns `false`.
"""
function FindConfBoundaryOnPlane(DM::AbstractDataModel,PL::Plane,Confnum::Real=1.; tol::Real=1e-12, maxiter::Int=10000)
    CF = ConfVol(Confnum);      mle = MLEinPlane(DM,PL; tol=1e-8)
    planarmod(x,p::AbstractVector{<:Number}) = DM.model(x,PlaneCoordinates(PL,p))
    Test(x::Real) = ChisqCDF(pdim(DM), abs(2(LogLikeMLE(DM) - loglikelihood(DM.Data,planarmod, mle + [x,0.])))) - CF < 0.
    !Test(0.) && return false
    [LineSearch(Test,0.;tol=tol,maxiter=maxiter), 0.] + mle
end


Shift(PlaneBegin::Plane,PlaneEnd::Plane) = TranslatePlane(PlaneEnd,PlaneEnd.stütz - PlaneBegin.stütz)
function Prune(DM::AbstractDataModel,Planes::Vector{<:Plane},Confnum::Real=1.)
    CF = ConfVol(Confnum)
    while length(Planes) > 0
        !WilksTest(DM,PlaneCoordinates(Planes[1],MLEinPlane(DM,Planes[1])),CF) ? popfirst!(Planes) : break
    end
    while length(Planes) > 0
        !WilksTest(DM,PlaneCoordinates(Planes[end],MLEinPlane(DM,Planes[end])),CF) ? pop!(Planes) : break
    end;    Planes
end

function AntiPrune(DM::AbstractDataModel,Planes::Vector{<:Plane},Confnum::Real=1.)
    length(Planes) < 2 && throw("Not enough Planes to infer translation direction.")
    CF = ConfVol(Confnum)
    while true
        TestPlane = Shift(Planes[2],Planes[1])
        WilksTest(DM,PlaneCoordinates(TestPlane,MLEinPlane(DM,TestPlane)),CF) ? pushfirst!(Planes,TestPlane) : break
    end
    while true
        TestPlane = Shift(Planes[end-1],Planes[end])
        WilksTest(DM,PlaneCoordinates(TestPlane,MLEinPlane(DM,TestPlane)),CF) ? push!(Planes,TestPlane) : break
    end;    Planes
end


"""
    IntersectCube(DM::AbstractDataModel,Cube::HyperCube,Confnum::Real=1.; Dirs::Vector=[1,2,3], N::Int=31) -> Vector{Plane}
Returns a set of parallel 2D planes which intersect `Cube`. The planes span the directions corresponding to the basis vectors corresponding to the first two components of `Dirs`.
They are separated in the direction of the basis vector associated with the third component of `Dirs`.
The keyword `N` can be used to approximately control the number of planes which are returned.
This depends on whether more (or fewer) planes than `N` are necessary to cover the whole confidence region of level `Confnum`.
"""
function IntersectCube(DM::AbstractDataModel,Cube::HyperCube,Confnum::Real=1.; N::Int=31, Dirs::Vector{<:Int}=[1,2,3])
    length(Dirs) != 3 || !allunique(Dirs) || !all(x->(1 <= x <= pdim(DM)),Dirs) && throw("Invalid choice of Dirs: $Dirs.")
    PL = Plane(Center(Cube),BasisVector(Dirs[1],pdim(DM)),BasisVector(Dirs[2],pdim(DM)))
    width = CubeWidths(Cube)[Dirs[3]]
    IntersectRegion(DM,PL,width * BasisVector(Dirs[3],pdim(DM)), Confnum; N=N)
end

"""
    IntersectRegion(DM::AbstractDataModel,PL::Plane,v::Vector{<:Real},Confnum::Real=1.; N::Int=31) -> Vector{Plane}
Translates family of `N` planes which are translated approximately from `-v` to `+v` and intersect the confidence region of level `Confnum`.
If necessary, planes are removed or more planes added such that the maximal family of planes is found.
"""
function IntersectRegion(DM::AbstractDataModel,PL::Plane,v::Vector{<:Real},Confnum::Real=1.; N::Int=31)
    IsOnPlane(Plane(zeros(length(v)), PL.Vx, PL.Vy),v) && throw("Translation vector v = $v lies in given Plane $PL.")
    Planes = ParallelPlanes(PL, v, range(-0.5,0.5,length=N))
    AntiPrune(DM,Prune(DM,Planes,Confnum),Confnum)
end

# """
#     MincedBoundaries(DM::DataModel, Dirs::Vector{<:Int}=[1,2,3], Confnum::Real=1.; N::Int=31, tol::Real=1e-8, meth=Tsit5(), mfd::Bool=false, Auto::Bool=false)
# Intersects the confidence region of level `Confnum` with roughly `N` many Planes which are parallel to `PL` and computes integral curves.
# """
# function MincedBoundaries(DM::DataModel, Dirs::Vector{<:Int}=[1,2,3], Confnum::Real=1.; Padding::Real=1/2, N::Int=31, tol::Real=1e-8,
#                         meth::OrdinaryDiffEqAlgorithm=Tsit5(), mfd::Bool=false, Auto::Bool=false, parallel::Bool=false)
#     Cube = LinearCuboid(DM,Confnum;Padding=Padding)
#     Planes = IntersectCube(DM,Cube,Confnum; N=N, Dirs=Dirs)
#     MincedBoundaries(DM, Planes, Confnum; tol=tol, meth=meth, mfd=mfd, Auto=Auto, parallel=parallel)
# end

"""
    MincedBoundaries(DM::AbstractDataModel, Planes::Vector{<:Plane}, Confnum::Real=1.; tol::Real=1e-9, Auto::Bool=false, meth=Tsit5(), mfd::Bool=false)
Intersects the confidence boundary of level `Confnum` with `Planes` and computes `ODESolution`s which parametrize this intersection.
"""
function MincedBoundaries(DM::AbstractDataModel, Planes::Vector{<:Plane}, Confnum::Real=1.; tol::Real=1e-8, Auto::Bool=false,
                                meth::OrdinaryDiffEqAlgorithm=Tsit5(), mfd::Bool=false, parallel::Bool=false)
    if parallel
        return pmap(X->GenerateBoundary(DM,X,FindConfBoundaryOnPlane(DM,X,Confnum;tol=tol);tol=tol, meth=meth, mfd=mfd, Auto=Auto), Planes)
    else
        return map(X->GenerateBoundary(DM,X,FindConfBoundaryOnPlane(DM,X,Confnum;tol=tol);tol=tol, meth=meth, mfd=mfd, Auto=Auto), Planes)
        # [GenerateBoundary(DM,Planes[i],FindConfBoundaryOnPlane(DM,Planes[i],Confnum;tol=tol);
        #                 tol=tol, meth=meth, mfd=mfd, Auto=Auto) for i in 1:length(Planes)]
    end
end

"""
Returns `HyperCube` which bounds the linearized confidence region of level `Confnum` for a `DataModel`.
"""
function LinearCuboid(DM::DataModel, Confnum::Real=1.; Padding::Real=1/30, N::Int=200)
    L = sqrt(quantile(Chisq(pdim(DM)),ConfVol(Confnum))) .* cholesky(inv(Symmetric(FisherMetric(DM,MLE(DM))))).L
    C = [ConstructCube(Unpack([L * RotatedVector(α,dims[1],dims[2],pdim(DM)) for α in range(0,2pi,length=N)]);Padding=Padding) for dims in permutations(1:pdim(DM),2)]
    TranslateCube(CoverCubes(C...),MLE(DM))
end


# LSQFIT
import LsqFit.curve_fit
curve_fit(DM::AbstractDataModel,initial::AbstractVector{<:Number}=MLE(DM);tol::Real=6e-15,kwargs...) = curve_fit(DM.Data,DM.model,initial;tol=tol,kwargs...)
function curve_fit(DS::AbstractDataSet,model::Function,initial::AbstractVector{<:Number}=GetStartP(DS,model); tol::Real=6e-15,kwargs...)
    X = xdata(DS);  Y = ydata(DS)
    LsqFit.check_data_health(X, Y)
    u = cholesky(InvCov(DS)).U
    f(p) = u * (EmbeddingMap(DS, model, p) - Y)
    p0 = convert(Vector,initial);    r = f(p0)
    R = OnceDifferentiable(f, p0, copy(r); inplace = false, autodiff = :finite)
    LsqFit.lmfit(R, p0, InvCov(DS); x_tol=tol,g_tol=tol,kwargs...)
end

# function curve_fit(DS::AbstractDataSet,model::Function,initial::AbstractVector{<:Number}=rand(pdim(DS,model));tol::Real=6e-15,kwargs...)
#     X = xdata(DS);  Y = ydata(DS)
#     LsqFit.check_data_health(X, Y)
#     u = cholesky(InvCov(DS)).U
#     f(p) = u * (EmbeddingMap(DS, model, p) - Y)
#     # Using Jacobian apparently slower
#     df(p) = ForwardDiff.jacobian(f,p)
#     p0 = convert(Vector,initial);    r = f(p0)
#     R = OnceDifferentiable(f, df, p0, copy(r); inplace=false)
#     LsqFit.lmfit(R, p0, InvCov(DS); x_tol=tol,g_tol=tol,kwargs...)
# end
