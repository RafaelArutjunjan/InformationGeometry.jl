

################### Probability Stuff
import Distributions.loglikelihood
"""
    likelihood(DM::DataModel,p::Vector)
Calculates the Likelihood given a DataModel and a parameter configuration `p`.
"""
likelihood(args...) = exp(loglikelihood(args...))

"""
    loglikelihood(DM::DataModel, p::Vector)
Calculates the logarithm of the Likelihood given a DataModel and a parameter configuration `p`.
"""
function loglikelihood(DM::DataModel, params::Vector)
    R = zero(suff(params))
    if length(ydata(DM)[1]) == 1    # For some reason this is faster.
        for i in 1:length(xdata(DM))
            R += ((ydata(DM)[i]-DM.model(xdata(DM)[i],params))/sigma(DM)[i])^2
        end
    else
        term(i) = dot((ydata(DM)[i] .- DM.model(xdata(DM)[i],params))/sigma(DM)[i])
        R = sum( term(i) for i in 1:length(xdata(DM)) )
    end
    -0.5*(length(xdata(DM))*log(2pi) + 2*sum(log.(sigma(DM))) + R)
end


function loglikelihood2(DM::DataModel, p::Vector)
    logpdf(product_distribution([Normal(ydata(DM)[i],sigma(DM)[i]) for i in 1:length(ydata(DM))]),EmbeddingMap(DM,p))
end

function ploglikelihood(DM::DataModel, params::Vector)
    R = @distributed (+) for i in 1:length(xdata(DM))
        (2*log(sigma(DM)[i]) + ((ydata(DM)[i]-DM.model(xdata(DM)[i],params))/sigma(DM)[i])^2)
    end
    -0.5*(length(xdata(DM))*log(2pi) + R)
end

# Naive Way:
LLRnaive(DM::DataModel,params1::Vector{<:Real},params2::Vector{<:Real}) = loglikelihood(DM,params1) - loglikelihood(DM,params2)
function LLR(DM::DataModel,params1::Vector{<:Real},params2::Vector{<:Real})
    mod1 = DM.model(xdata(DM),params1);     mod2 = DM.model(xdata(DM),params2)
    R = 0.0
    for i in 1:length(xdata(DM))
        R += 0.5*sigma(DM)[i]^-2 * (mod2[i]^2 - mod1[i]^2 + 2*ydata(DM)[i]*(mod1[i]-mod2[i]))
    end
    R
end

"""
    ConfAlpha(n::Real)
Probability volume outside of a confidence interval of level n⋅σ where σ is the standard deviation of a normal distribution.
"""
ConfAlpha(n::Real) = 1.0 .- ConfVol(n)

"""
    ConfVol(n::Real)
Probability volume contained in a confidence interval of level n⋅σ where σ is the standard deviation of a normal distribution.
"""
function ConfVol(n::Real)
    num = erf(BigFloat(n)/sqrt(BigFloat(2)))
    num == 1.0 && throw("ConfVol: BigFloat precision of $(precision(BigFloat)) not enough for n = $n.")
    if suff(n) == BigFloat
        return num
    elseif n > 8
        println("ConfVol: Float64 precision not enough for n = $n. Returning BigFloat instead.")
        return num
    else
        return convert(Float64,num)
    end
end
InvConfVol(x::Real; tol=1e-15) = find_zero((z->(ConfVol(z)-x)),one(suff(x)),Order8(),xatol=tol)
ChisqCDF(k,x) = gamma_inc(BigFloat(k)/2., BigFloat(x)/2.,0)[1] ./ gamma(BigFloat(k)/2.)
InvChisqCDF(k::Real,x::Real; tol=1e-15) = find_zero((z->(ChisqCDF(k,z)-x)),(1e-5*one(suff(x)),300),Bisection(),xatol=tol)

ChiQuant(sigma::Real=1.,k::Int=2) = (1/2)*quantile(Chisq(k),ConfVol(sigma))
ChiQuantToSigma(ChiQuant::Real,k::Int=2) = cdf.(Chisq(k),2*ChiQuant) |> InvConfVol

# Cannot be used with DiffEq since tags conflict. Use Zygote.jl?
AutoScore(DM::DataModel,y) = ForwardDiff.gradient(x->loglikelihood(DM,x),y)
AutoMetric(DM::DataModel,y) = ForwardDiff.hessian(x->(-loglikelihood(DM,x)),y)



"""
Calculates the score of models with y vales of dim > 1.
"""
function ScoreDimN(DM::DataModel,p::Vector{<:Real})
    Res = zeros(suff(p),length(p))
    moddif = (sigma(DM)).^(-2) .* (ydata(DM) .- map(z->DM.model(z,p),xdata(DM)))
    dmod = map(z->DM.dmodel(z,p),xdata(DM))
    for i in 1:length(xdata(DM))
        Res[:] .+= (transpose(moddif[i]) * DM.dmodel(xdata(DM)[i],p))[:]
    end
    Res
end

"""
    Score(DM::DataModel,p::Vector{<:Real}; Auto::Bool=false)
Calculates the gradient of the log-likelihood with respect to a set of parameters `p`. `Auto=true` uses automatic differentiation.
"""
function Score(DM::DataModel,p::Vector{<:Real}; Auto::Bool=false)
    Auto && return AutoScore(DM,p)
    length(ydata(DM)[1]) != 1 && return ScoreDimN(DM,p)
    Res = zeros(suff(p),length(p))
    mod = map(z->DM.model(z,p),xdata(DM))
    dmod = DM.dmodel(xdata(DM),p)
    for j in 1:length(p)
        Res[j] += sum((sigma(DM)[i])^(-2) *(ydata(DM)[i]-mod[i])*dmod[i,j]   for i in 1:length(xdata(DM)))
    end
    Res
end
function ScoreOrth(DM::DataModel,PL::Plane,p::Vector{<:Real})
    length(p) == 2 && return [0 -1; 1 0]*Score(DM,p)
    -RotateVector(PL,ProjectOntoPlane(PL,Score(DM,p)),pi/2)
end
"""
    WilksTest(DM::DataModel, p::Vector{<:Real},MLE::Vector{<:Real},ConfVol=ConfVol(1)) -> Bool
Checks whether a given parameter configuration `p` is within a confidence interval of level `ConfVol` using Wilks' theorem.
This makes the assumption, that the likelihood has the form of a normal distribution, which is asymptotically correct in the limit that the number of datapoints is infinite.
"""
function WilksTest(DM::DataModel, p::Vector{<:Real},MLE::Vector{<:Real},ConfVol=ConfVol(1))::Bool
    # return (loglikelihood(DM,MLE) - loglikelihood(DM,p) <= (1/2)*quantile(Chisq(length(MLE)),Conf))
    return ChisqCDF(length(MLE), 2(loglikelihood(DM,MLE) - loglikelihood(DM,p))) - ConfVol < 0
end

"""
    WilksTestPrepared(DM::DataModel, p::Vector{<:Real}, LogLikeMLE::Real, ConfVol=ConfVol(1)) -> Bool
Checks whether a given parameter configuration `p` is within a confidence interval of level `ConfVol` using Wilks' theorem.
This makes the assumption, that the likelihood has the form of a normal distribution, which is asymptotically correct in the limit that the number of datapoints is infinite.
To simplify the calculation, `LogLikeMLE` accepts the value of the log-likelihood evaluated at the MLE.
"""
function WilksTestPrepared(DM::DataModel, p::Vector{<:Real}, LogLikeMLE::Real, ConfVol=ConfVol(1))::Bool
    # return (LogLikeMLE - loglikelihood(DM,p) <= (1/2)*quantile(Chisq(length(MLE)),Conf))
    return ChisqCDF(length(p), 2(LogLikeMLE - loglikelihood(DM,p))) - ConfVol < 0
end



function FtestPrepared(DM::DataModel, point::Vector, S_MLE::Real, ConfVol=ConfVol(1))::Bool
    n = length(ydata(DM));  p = length(point);    S(P) = sum(((ydata(DM) .- map(x->DM.model(x,P),xdata(DM)))./sigma(DM)).^2)
    S(point) <= S_MLE * (1. + p/(n-p)) * quantile(FDist(p, n-p),ConfVol)
end
Ftest(DM::DataModel, point::Vector, MLE::Vector, Conf=ConfVol(1))::Bool = FtestPrepared(DM,point,sum((ydata(DM) .- map(x->DM.model(x,MLE),xdata(DM))).^2),Conf)

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
    typeof(MLE[1]) != BigFloat && println("WilksBoundaryBig: You should pass the MLE as BigFloat!")
    print("Starting WilksBoundaryBig.   ")
    L = loglikelihood(DM,BigFloat.(MLE));    CF = ConfVol(BigFloat(Confnum))
    f(x::Real) = ChisqCDF(length(MLE),2(L-loglikelihood(DM,MLE .+ (x .* BasisVector(1,length(MLE)))))) - CF
    df(x) = ForwardDiff.gradient(f,x)
    @time b = find_zero((f,df),BigFloat(1),Roots.Order2(),xatol=tol)
    println("Finished.")
    MLE +  b .* BasisVector(1,length(MLE))
end

function Interval1D(DM::DataModel,MLE::Vector,Confnum::Real=1.;tol=1e-14)
    if tol < 1e-15 || suff(MLE) == BigFloat || typeof(ConfVol(Confnum)) == BigFloat
        throw("Interval1D not programmed for BigFloat yet.")
    end
    (length(MLE) != 1) && throw("Interval1D not defined for p != 1.")
    lMLE = loglikelihood(DM,MLE)
    A = lMLE - (1/2)*quantile(Chisq(length(MLE)),ConfVol(Confnum))
    Func(p::Real) = loglikelihood(DM,MLE .+ p*BasisVector(1,length(MLE))) - A
    D(f) = x->ForwardDiff.derivative(f,x);  NegFunc(x) = Func(-x)
    B = find_zero((Func,D(Func)),0.1,Roots.Order1(),xatol=tol)
    A = find_zero((Func,D(Func)),-B,Roots.Order1(),xatol=tol)
    rts = [MLE[1]+A, MLE[1]+B]
    rts[1] < rts[2] && return rts
    throw("Interval1D errored...")
end

# function FBoundary(DM::DataModel,MLE::Vector,Confnum::Real=1.; tol=1e-12)
#     n = length(ydata(DM));  p = length(MLE)
#     S(P) = dot(ydata(DM),map(x->DM.model(x,P),xdata(DM)))
#     A = S(MLE) * (1. + p/(n-p)) * quantile(FDist(p, n-p),ConfVol(Confnum))
#     Func(p::Q) where Q<:Real =  S(MLE +p*BasisVector(1,length(MLE))) - A
#     b = find_zero(Func,1,Roots.Order8(),xatol=tol)
#     b*BasisVector(1,length(MLE))
# end


# function FindConfBoundaryOld(DM::DataModel,MLE::Vector,Confnum::Real;tol::Real=1e-15,maxiter::Int=10000,Interval = [0., 5.])
#     lMLE = loglikelihood(DM,MLE);   A = lMLE - (1/2)*quantile(Chisq(length(MLE)),ConfVol(Confnum))
#     Test(x::Real)::Bool = (A <= loglikelihood(DM, MLE .+ x.*BasisVector(1,length(MLE)) ))
#     (!(Test(Interval[1])) || Test(Interval[2])) && throw(ArgumentError("Step not inside Interval."))
#     stepsize=(Interval[2]-Interval[1])/4.
#     value=Interval[1]
#     for i in 1:maxiter
#         if Test(value) # inside
#             value += stepsize
#         else            #outside
#             value -= stepsize
#             if stepsize < tol
#                 return value .* BasisVector(1,length(MLE)) .+ MLE
#             end
#             stepsize *= 1/10
#         end
#     end
#     throw(Error("$maxiter iterations over. Value=$value, Stepsize=$stepsize"))
# end

# Flips when the Boolean-valued test goes from true to false
# Tests in positive direction
function LineSearch(Test::Function, start::Real=0; tol::Real=8e-15, maxiter::Int=10000)
    ((suff(start) != BigFloat) && tol < 1e-15) && throw("LineSearch: start not BigFloat but tol=$tol.")
    !(Test(start)) && throw(ArgumentError("LineSearch: Test not true for starting value."))
    stepsize = one(suff(start))/4.;  value = start
    for i in 1:maxiter
        if Test(value + stepsize) # inside
            value += stepsize
            value-start > 20 && throw("FindConfBoundary: Value larger than 20.")
        else            #outside
            if stepsize < tol
                return value
            end
            stepsize /= 5
        end
    end
    throw(Error("$maxiter iterations over. Value=$value, Stepsize=$stepsize"))
end

# VERY SLOW for some reason...
# function FindConfBoundary(DM::DataModel,MLE::Vector,Confnum::Real; tol::Real=8e-15, maxiter::Int=10000)
#     LogLikeMLE = loglikelihood(DM,MLE);    CF = ConfVol(convert(suff(MLE),Confnum))
#     Test(x::Real) = WilksTestPrepared(DM, MLE .+ (x .* BasisVector(1,length(MLE))), LogLikeMLE, CF)
#     LineSearch(Test,0,tol=tol,maxiter=maxiter) .* BasisVector(1,length(MLE)) .+ MLE
# end

function FindConfBoundary(DM::DataModel,MLE::Vector,Confnum::Real; tol::Real=4e-15, maxiter::Int=10000)
    ((suff(MLE) != BigFloat) && tol < 1e-15) && throw("FindConfBoundary: MLE not BigFloat but tol=$tol.")
    LogLikeMLE = loglikelihood(DM,MLE);    Confvol = ConfVol(convert(suff(MLE),Confnum))
    Test(x::Real) = WilksTestPrepared(DM, MLE .+ (x .* BasisVector(1,length(MLE))), LogLikeMLE, Confvol)
    !(Test(0)) && throw(ArgumentError("FindConfBoundary: Given MLE not inside Confidence Interval."))
    stepsize = one(suff(MLE))/4.;  value = zero(suff(MLE))
    for i in 1:maxiter
        if Test(value + stepsize) # inside
            value += stepsize
            value > 20 && throw("FindConfBoundary: Value larger than 10.")
        else            #outside
            if stepsize < tol
                return value .* BasisVector(1,length(MLE)) .+ MLE
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


"""
    OrthVF(DM::DataModel, p::Vector{<:Real}; Auto::Bool=false) -> Vector
Calculates a direction (in parameter space) in which the value of the log-likelihood does not change, given a parameter configuration `p`.
`Auto=true` uses automatic differentiation to calculate the score.
"""
function OrthVF(DM::DataModel, p::Vector{<:Real}; Auto::Bool=false)
    length(p) < 2 && throw(ArgumentError("dim(Parameter Space) < 2  --> No orthogonal VF possible"))
    S = Score(DM,p;Auto=Auto);    P = prod(S);    VF = P ./ S
    VF[length(p)] = -(length(p)-1)*VF[length(p)]
    VF |> normalize
end


"""
    OrthVF(DM::DataModel, PL::Plane, p::Vector{<:Real}; Auto::Bool=false) -> Vector
Calculates a direction (in parameter space) in which the value of the log-likelihood does not change, given a parameter configuration `p`.
If a `Plane` is specified, the direction will be projected onto it.
`Auto=true` uses automatic differentiation to calculate the score.
"""
function OrthVF(DM::DataModel, PL::Plane, p::Vector{<:Real}; Auto::Bool=false)
    length(p) < 2 && throw(ArgumentError("dim(Parameter Space) < 2  --> No orthogonal VF possible"))
    planeorth = Cross(PL.Vx,PL.Vy)
    if length(p) != 2
        !IsOnPlane(PL,p) && throw(ArgumentError("Parameter Configuration not on specified Plane."))
        länge(planeorth .- ones(length(p))) < 1e-14 && throw(ArgumentError("Visualization plane unsuitable: $planeorth"))
    end

    S = Score(DM,p; Auto=Auto);    P = prod(S);    VF = P ./ S

    alpha = []
    if length(p) > 2
        alpha = Cross(ones(length(p)),planeorth)
    else
        alpha = Cross(ones(3),planeorth)[1:2]
    end
    # ProjectOntoPlane(PL,alpha .* VF) |> normalize
    normalize(alpha .* VF)
end

ConstructCube(sol::ODESolution,Npoints::Int=200;Padding::Float64=1/50) = HyperCube(ConstructLowerUpper(sol,Npoints,Padding=Padding))
function ConstructLowerUpper(sol::ODESolution,Npoints::Int=200;Padding::Float64=1/50)
    Padding < 0. && throw("Cube Padding negative.")
    nparams = length(sol.u[1]);  lowers = copy(sol.u[1]);  uppers = copy(sol.u[1])
    for t in range(sol.t[1],sol.t[end], length=Npoints)
        Point = sol(t)
        for i in 1:nparams
            lowers[i] = minimum([lowers[i],Point[i]])
            uppers[i] = maximum([uppers[i],Point[i]])
        end
    end
    diff = (uppers .- lowers) .* Padding
    LowerUpper(lowers .- diff,uppers .+ diff)
end

########################## Monte Carlo


MonteCarloArea(Test::Function,Space::HyperCube,N::Int=Int(1e7)) = CubeVol(Space)*MonteCarloRatio(Test,Space,N)
MonteCarloRatio(Test::Function,Space::HyperCube,N::Int=Int(1e7)) = MonteCarloRatio(Test,LowerUpper(Space),N)
function MonteCarloRatio(Test::Function,LU::LowerUpper,N::Int=Int(1e7))
    (1/N)* @distributed (+) for i in 1:N
        Test(rand.(Uniform.(LU.L,LU.U)))
    end
end


function MonteCarloRatioWE(Test::Function,Space::HyperCube,N::Int=Int(1e7); chunksize::Int=Int(N/20))
    chunksize > N && error("chunksize > N")
    if N%chunksize != 0
        println("N % chunksize = $(N%chunksize). Rounded N up from $N to $(N + N%chunksize + 1).")
        N += Int(N%chunksize + 1)
    end
    chunks = Int(N/chunksize);   LU = LowerUpper(Space)
    # Output not normalized by chunksize
    function CarloLoop(Test::Function,LU::LowerUpper,chunksize::Int)
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
    # sendto(workers(),LogLikeBoundary=LogLikeBoundary)
    Cube = ConstructCube(sol,Padding=1e-5)
    # Indicator function for Integral
    InsideRegion(X::Vector{<:Real})::Bool = loglikelihood(DM,X) < LogLikeBoundary
    Test(X::Vector) = InsideRegion(X) ? sqrt(det(FisherMetric(DM,X))) : 0.
    WE && return MonteCarloAreaWE(Test,Cube,N)
    MonteCarloArea(Test,Cube,N)
end



# Find MLE to BigFLoat precision.
function FindMLEBig(DM::DataModel,start::Union{Bool,Vector}=false)
    if isa(start,Bool)
        return FindMLEBig(DM,BigFloat.(FindMLE(DM)))
    elseif isa(start,Vector)
        println("Warning: Passed $start to FindMLEBig.")
        NegEll(x) = -loglikelihood(DM,x)
        return optimize(NegEll, BigFloat.(start), LBFGS(), Optim.Options(g_tol=convert(BigFloat,10 .^(-precision(BigFloat)/30))), autodiff = :forward) |> Optim.minimizer
    end
end

"""
    FindMLE(DM::DataModel,start::Union{Bool,Vector}=false; Big::Bool=false)
Finds the maximum likelihood parameter configuration given a DataModel and optionally a starting configuration. `Big=true` will return the value as a `BigFloat`.
"""
function FindMLE(DM::DataModel,start::Union{Bool,Vector}=false; Big::Bool=false)
    Big && return FindMLEBig(DM,start)
    max = 50
    NegEll(x) = -loglikelihood(DM,x)
    if isa(start,Bool)
        A = Vector{Real}
        for i in 1:max
            try
                A = Optim.minimizer(optimize(NegEll, ones(i), BFGS(), Optim.Options(g_tol=1e-14), autodiff = :forward))
            catch y
                if isa(y, BoundsError) || isa(y, ArgumentError)
                    println("FindMLE automatically deduced that model must have >$i parameter(s).")
                    continue
                else    throw(y)    end
            end
            if i != max  return A    else
                throw(ArgumentError("dim(ParamSpace) appears to be larger than $max. FindMLE unsuccessful."))
            end
        end
    elseif isa(start,Vector)
        if suff(start) == BigFloat
            return FindMLEBig(DM,start)
        else
            println("Warning: Passed $start to FindMLE.")
            return optimize(NegEll, start, BFGS(), Optim.Options(g_tol=1e-14), autodiff = :forward) |> Optim.minimizer
        end
    end
end

function GenerateBoundary(DM::DataModel, u0::Vector{<:Real}; tol::Real=1e-14, meth=Tsit5(), mfd::Bool=true)
    L(params) = loglikelihood(DM,params);    V(params) = OrthVF(DM,params)
    LogLikeOnBoundary = L(u0)
    function IntCurveODE(du,u,p,t)
        du .= 0.1 .* V(u)
    end
    function g(resid,u,p,t)
      resid[1] = LogLikeOnBoundary - L(u)
      resid[2] = 0.
    end
    # Need to change terminatecondition for more than 2 parameters!
    terminatecondition(u,t,integrator) = u[2] - u0[2]
    # function terminatecondition(u,t,integrator)
    #     s = u[2] - u0[2]
    #     if abs(sum(u .- u0)) > 1e-1
    #         s += 1000
    #     end;    s
    # end
    # TerminateCondition only on upwards crossing --> supply two different affect functions, leave second free I
    cb = CallbackSet(ManifoldProjection(g),ContinuousCallback(terminatecondition,terminate!,nothing))
    # tspan = (zero(suff(u0)),convert(suff(u0),1000.))
    tspan = (0.,1e4)
    prob = ODEProblem(IntCurveODE,u0,tspan)
    if mfd
        return solve(prob,meth,reltol=tol,abstol=tol,callback=cb,save_everystep=false)
    else
        return solve(prob,meth,reltol=tol,abstol=tol,callback=ContinuousCallback(terminatecondition,terminate!,nothing))
    end
end

GenerateConfidenceInterval(DM::DataModel,Confnum=1; tol::Real=1e-14, meth=Tsit5(), mfd::Bool=true) = GenerateConfidenceInterval(DM,FindMLE(DM),Confnum, tol=tol, meth=meth, mfd=mfd)

function GenerateConfidenceInterval(DM::DataModel,MLE::Vector{<:Real},Confnum=1; tol::Real=1e-14, meth=Tsit5(), mfd::Bool=true)
    if (suff(MLE) != BigFloat) && tol < 2e-15
        MLE = FindMLEBig(DM,MLE)
        println("GenerateConfidenceInterval: Promoting MLE to BigFloat because tol=$tol.")
    end
    if length(MLE) == 1
        return Interval1D(DM,MLE,Confnum,tol=tol)
    else
        return GenerateBoundary(DM, FindConfBoundary(DM,MLE,Confnum; tol=tol), tol=tol, meth=meth, mfd=mfd)
    end
end


function GenerateConfidenceInterval(DM::DataModel,PL::Plane,MLE::Vector{<:Real},Confnum=1; tol=1e-14, meth=Tsit5())
    throw("GenerateConfidenceInterval: Not Programmed properly yet for Plane.")
    L(params) = loglikelihood(DM,params);    V(params) = OrthVF(DM,PL,params)
    LogLikeMLE = L(MLE);    Confvol = ConfVol(Confnum)
    Tester(x) = WilksTestPrepared(DM,MLE+x.*PL.Vx,LogLikeMLE,Confvol)
    u0 = MLE .+ LineSearch(Tester; tol=tol).*PL.Vx
    Vectorscaling = 0.5/sqrt(sum(V(u0).^2))
    function IntCurveODE(du,u,p,t)
        du .= Vectorscaling.*V(u)
    end
    function g(resid,u,p,t)
      resid[1] = (L(u) - L(u0))
      resid[2] = 0.
    end
    terminatecondition(u,t,integrator) = u[2]-u0[2]
    # TerminateCondition only on upwards crossing --> supply two different affect functions, leave second free I
    # ADD ManifoldProjection TO THE PLOT PLANE? Not Necessary if ORTHVF already projects there.
    cb = CallbackSet(ManifoldProjection(g),ContinuousCallback(terminatecondition,terminate!,nothing))
    tspan = (0.0,200.0)
    prob = ODEProblem(IntCurveODE,u0,tspan)
    solve(prob,meth,reltol=tol,abstol=tol,callback=cb)
end

function StructurallyIdentifiable(DM::DataModel,sol::ODESolution)
    roots = find_zeros(t->GeometricDensity(DM,sol(t)),sol.t[1],sol.t[end])
    length(roots)==0, roots
end

function GenerateMultipleIntervals(DM::DataModel, Range, MLE=[Inf,Inf]; IsConfVol::Bool=false, tol=1e-14, meth=Tsit5(), mfd::Bool=true)
    if MLE == [Inf,Inf]     MLE = FindMLE(DM)    end
    length(MLE) == 1 && return map(x->GenerateConfidenceInterval(DM,MLE,x,tol=tol),Range)
    LogLikeMLE = loglikelihood(DM,MLE);     sols = Vector{ODESolution}(undef,0)
    for CONF in Range
        if IsConfVol
            @time push!(sols, GenerateConfidenceInterval(DM,MLE,InvConfVol(CONF),tol=tol,meth=meth,mfd=mfd))
        else
            @time push!(sols, GenerateConfidenceInterval(DM,MLE,CONF,tol=tol,meth=meth,mfd=mfd))
        end
        if sols[end].retcode == :Terminated
            # Be more quantitative in how large the discrepancy is?
            # ConfVolume = 0
            # if IsConfVol
            #     ConfVolume = CONF
            # else
            #     ConfVolume = ConfVol(CONF)
            # end
            # Tester(X) = WilksTestPrepared(DM,X,LogLikeMLE,ConfVolume)
            # CurveInsideInterval(Tester,sols[end],1000);
            _ , rts = StructurallyIdentifiable(DM,sols[end])
            if length(rts) != 0
                println("Solution $(length(sols)) corresponding to $(Range[length(sols)]) hits chart boundary at t=$rts and is therefore invalid.")
            end
        else
            println("solution $(length(sols)) did not exit properly: retcode=$(sols[end].retcode).")
        end
    end
    sols
end


function CurveInsideInterval(Test::Function, sol::ODESolution, N::Int = 1000)
    NoPoints = Vector{Vector{suff(sol.u[1])}}(undef,0)
    for t in range(sol.t[1],sol.t[end], length=N)
        num = sol(t)
        if !Test(num)
            push!(NoPoints,num)
        end
    end
    println("CurveInsideInterval: Solution has $(length(NoPoints))/$N points outside the desired confidence interval.")
    return NoPoints
end


Inside(C::HyperCube,p::Vector) = Inside(LowerUpper(C),p)
function Inside(LU::LowerUpper,p::Vector)::Bool
    length(LU.L) != length(p) && throw("Inside: Dimension mismatch between Cube and point.")
    for i in 1:length(LU.L)
        !(LU.L[i] <= p[i] <= LU.U[i]) && return false
    end;    true
end

"""
    Rsquared(DM::DataModel,Fit::LsqFit.LsqFitResult) -> Real
Calculates the R² value of the fit result `Fit`.
"""
function Rsquared(DM::DataModel,Fit::LsqFit.LsqFitResult)
    length(ydata(DM)[1]) != 1  && return -1
    mean = sum(ydata(DM))/length(ydata(DM))
    Stot = sum((ydata(DM) .- mean).^2)
    Sres = sum((ydata(DM) .- DM.model(xdata(DM),Fit.param)).^2)
    1 - Sres/Stot
end


function Integrate1D(F::Function,Cube::HyperCube; tol::Real=1e-14, fullSol::Bool=false, meth=nothing)
    Cube.dim != 1 && throw(ArgumentError("Cube dim = $(Cube.dim) instead of 1"))
    Integrate1D(F,Cube.vals[1][:],tol=tol,fullSol=fullSol,meth=meth)
end
function Integrate1D(F::Function,Interval::Vector;tol::Real=1e-14, fullSol::Bool=false, meth=nothing)
    length(Interval) != 2 && throw(ArgumentError("Interval not suitable for integration."))
    !(0. < tol < 1.) && throw("Integrate1D: tol unsuitable")
    Interval[1] > Interval[2] && throw(ArgumentError("Interval orientation wrong."))
    f(u,p,t) = F(t)
    if tol < 1e-15
        u0 = BigFloat(0.);        tspan = Tuple(BigFloat.(Interval))
        if meth == nothing
            meth = Feagin10()
        end
    else
        u0 = 0.;        tspan = Tuple(Interval)
        if meth == nothing
            meth = Tsit5()
        end
    end
    if fullSol
        return solve(ODEProblem(f,u0,tspan),meth,reltol=tol,abstol=tol)
    else
        return solve(ODEProblem(f,u0,tspan),meth,reltol=tol,abstol=tol,save_everystep=false,save_start=false,save_end=true).u[end]
    end
end


# Trapezoidal
function Intemeh(F::Function,Interval::Vector{Q}, N::Int=500) where Q<:Real
    Interval[1] > Interval[2] && throw(ArgumentError("Interval Unsuitable in Integreat."))
    if Interval[1] == Interval[2] return 0. end
    dx = (Interval[2]-Interval[1])/N;   range = (Interval[1]+dx):dx:(Interval[2]-dx)
    Res = 0.5*(F(Interval[1]) + F(Interval[2]))
    Res += @distributed (+) for x in range  F(x)    end
    dx*Res
end

# Simpson rule (quadratic)
function Integreat(F::Function,Interval::Vector{Q}, N::Int=300) where Q<:Real
    Interval[1] > Interval[2] && throw(ArgumentError("Interval Unsuitable in Integreat."))
    if Interval[1] == Interval[2] return 0. end
    if N%2 == 0 N+=1 end
    dx = (Interval[2]-Interval[1])/N;    Res = F(Interval[1]) + F(Interval[2]) + 4F(Interval[1]+dx)
    range = (Interval[1]+dx):dx:(Interval[2]-dx)
    Res += @distributed (+) for i in (2:2:length(range)-1)
        2F(range[i]) + 4F(range[i+1])
    end;    Res*dx/3
end

function Intebest(F::Function,Interval::Vector{Q}, N::Int=300) where Q<:Real
    Interval[1] > Interval[2] && throw(ArgumentError("Interval Unsuitable in Integreat."))
    if Interval[1] == Interval[2] return 0. end
    while (N%3 != 0) N+=1 end
    dx = (Interval[2]-Interval[1])/(N+1)
    range = (Interval[1]+dx):dx:(Interval[2])
    Res = F(Interval[1]) + F(Interval[2])
    Res += @distributed (+) for i in (1:3:length(range)-1)
        3F(range[i]) + 3F(range[i+1]) + 2F(range[i+2])
    end
    3*dx/8 * Res
end


# Assume that sums from Fisher metric defined with first derivatives of loglikelihood pull out
"""
    FisherMetric(DM::DataModel, p::Vector{<:Real})
Computes the Fisher metric given a `DataModel` and a parameter configuration `p` under the assumption that the likelihood ``L`` is a multivariate normal distribution.
```math
g_{ab} = -\\int_{\\mathcal{D}} \\mathrm{d}^N y \\, L(y,\\theta) \\, \\frac{\\partial^2 \\, \\mathrm{ln}(L)}{\\partial \\theta^a \\, \\partial \\theta^b}
```
where ``\\theta`` corresponds to the parameters `p`.
"""
function FisherMetric(DM::DataModel, p::Vector{<:Real})
    F = zeros(suff(p),length(p),length(p))
    dmod = sigma(DM).^(-1) .* map(z->DM.dmodel(z,p),xdata(DM))
    for i in 1:length(xdata(DM))
        F .+=  transpose(dmod[i]) * dmod[i]
    end;    F
end


meshgrid(x, y) = (repeat(x, outer=length(y)), repeat(y, inner=length(x)))

# Always use Integreat for 1D -> Check Dim of Space to decide?
"""
    KullbackLeibler(p::Function,q::Function,Domain::HyperCube=HyperCube([[-15,15]]); tol=1e-15, N::Int=Int(3e7), Carlo::Bool=(Domain.dim!=1))
Computes the Kullback-Leibler divergence between two probability distributions `p` and `q` over the `Domain`.
If `Carlo=true`, this is done using a Monte Carlo Simulation with `N` samples.
If the `Domain` is one-dimensional, the calculation is performed without Monte Carlo to a tolerance of ≈ `tol`.
```math
D_{\\text{KL}} = \\int \\mathrm{d}^m y \\, p(y) \\, \\mathrm{ln} \\bigg( \\frac{p(y)}{q(y)} \\bigg)
```
"""
function KullbackLeibler(p::Function,q::Function,Domain::HyperCube=HyperCube([[-15,15]]); tol=1e-15, N::Int=Int(3e7), Carlo::Bool=(Domain.dim!=1))
    function Integrand(x)
        P = p(x)[1];   Q = q(x)[1];   Rat = P/Q
        Rat <= 0 && throw(ArgumentError("Ratio p(x)/q(x) = $Rat in log(p/q) for x=$x."))
        P*log(Rat)
    end
    if Carlo
        Domain.dim==1 && return MonteCarloArea(Integrand,Domain,N)[1]
        return MonteCarloArea(Integrand,Domain,N)
    elseif Domain.dim == 1
        return Integrate1D(Integrand,Domain,tol=tol)
    else
        throw("KL: Carlo=false and Domain.dim != 1. Aborting.")
    end
end

function KullbackLeibler(p::Distribution,q::Distribution,Domain::HyperCube=HyperCube([[-15,15]]);
    tol=1e-15, N::Int=Int(3e7), Carlo::Bool=(Domain.dim!=1))
    !(length(p) == length(q) == Domain.dim) && throw("KL: Sampling dimension mismatch: dim(p) = $(length(p)), dim(q) = $(length(q)), Domain = $(Domain.dim)")
    if length(p) == 1
        !(sum(insupport(p,Domain.vals[1])) == 2) && throw("KL: p=$p not supported on Domain $(Domain.vals[1]).")
        !(sum(insupport(q,Domain.vals[1])) == 2)  && throw("KL: q=$q not supported on Domain $(Domain.vals[1]).")
    end
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
    elseif Domain.dim == 1
        return Integrate1D(Integrand1D,Domain,tol=tol)
    else
        throw("KL: Carlo=false and Domain.dim != 1. Aborting.")
    end
end

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

"""
    function NormalDist(DM::DataModel,p::Vector) -> Distribution
Constructs either `Normal` or `MvNormal` type from `Distributions.jl` using data and a parameter configuration.
This makes the assumption, that the errors associated with the data are normal.
"""
function NormalDist(DM::DataModel,p::Vector)::Distribution
    if length(ydata(DM)[1]) == 1
        length(ydata(DM)) == 1 && return Normal(ydata(DM)[1] .- DM.model(xdata(DM)[1],p),sigma(DM)[1])
        return MvNormal(ydata(DM) .- map(x->DM.model(x,p),xdata(DM)),diagm(float.(sigma(DM).^2)))
    else
        throw("Not programmed yet.")
    end
end

"""
    KullbackLeibler(DM::DataModel,p::Vector,q::Vector)
Calculates KL divergence under assumption of normal error distribution around data.
"""
KullbackLeibler(DM::DataModel,p::Vector,q::Vector) = KullbackLeibler(NormalDist(DM,p),NormalDist(DM,q))

KullbackLeibler(DM::DataModel,p::Vector) = KullbackLeibler(MvNormal(zeros(length(sigma(DM))),diagm(float.(sigma(DM).^2))),NormalDist(DM,p))


# h(p) ∈ Dataspace
EmbeddingMap(DM::DataModel,p::Vector{<:Real}) = map(x->DM.model(x,p),xdata(DM))

# NOT TO BE CONFUSED WITH JACOBIAN FROM TRANSTRUM PAPER -- NO SIGMAS HERE
# Correct this to ensure that the shape of the matrix is correct, irrespective of the model definition.
EmbeddingMatrix(DM::DataModel,point::Vector{<:Real}) = DM.dmodel(xdata(DM),float.(point))

# From D to M
Pullback(DM::DataModel,F::Function,point::Vector) = F(EmbeddingMap(DM,point))
"""
    Pullback(DM::DataModel, omega::Vector{<:Real}, point::Vector) -> Vector
Pull-back of a covector to the parameter manifold.
"""
Pullback(DM::DataModel, omega::Vector{<:Real}, point::Vector) = transpose(EmbeddingMatrix(DM,point))*omega


"""
    Pullback(DM::DataModel, G::AbstractArray{<:Real,2}, point::Vector)
Pull-back of a (0,2)-tensor `G` to the parameter manifold.
"""
function Pullback(DM::DataModel, G::AbstractArray{<:Real,2}, point::Vector)
    J = EmbeddingMatrix(DM,point)
    # return transpose(J) * G * J
    @tensor R[a,b] := J[i,a]*J[j,b]*G[i,j]
end

# M to D
"""
    Pushforward(DM::DataModel, X::Vector, point::Vector)
Calculates the push-forward of a vector `X` from the parameter manifold to the data space.
"""
Pushforward(DM::DataModel, X::Vector, point::Vector) = EmbeddingMatrix(DM,point) * X

"""
    DataSpaceDist(DM::DataModel,v::Vector) -> Real
Calculates the euclidean distance between a point `v` in the data space and the data.
"""
function DataSpaceDist(DM::DataModel,v::Vector)
    length(ydata(DM)) != length(v) && error("DataSpaceDist: Dimensional Mismatch")
    return MetricNorm(Diagonal(sigma(DM).^(-2)),(ydata(DM) .- v))
end



# Compute all major axes of Fisher Ellipsoid from eigensystem of Fisher metric
FisherEllipsoid(DM::DataModel, point::Vector{<:Real}) = FisherEllipsoid(p->FisherMetric(DM,p), point)
FisherEllipsoid(Metric::Function, point::Vector{<:Real}) = eigvecs(Metric(point))


"""
    AIC(DM::DataModel,p::Vector)
Calculates the Akaike Information Criterion given a parameter configuration `p`.
"""
AIC(DM::DataModel,p::Vector) = -2loglikelihood(DM,p) + 2length(p)
