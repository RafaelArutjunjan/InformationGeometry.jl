

################### Probability Stuff
"""
    likelihood(DM::DataModel,θ::Vector)
Calculates the likelihood ``L(\\mathrm{data} \\, | \\, \\theta)`` a `DataModel` and a parameter configuration ``\\theta``.
"""
likelihood(args...) = exp(loglikelihood(args...))

import Distributions.loglikelihood
"""
    loglikelihood(DM::DataModel, θ::Vector)
Calculates the logarithm of the likelihood ``\\ell(\\mathrm{data} \\, | \\, \\theta) \\coloneqq \\mathrm{ln} \\big( L(\\mathrm{data} \\, | \\, \\theta) \\big)`` given a `DataModel` and a parameter configuration ``\\theta``.
"""
loglikelihood(DM::AbstractDataModel,θ::Vector{<:Number}) = loglikelihood(DM.Data,DM.model,θ)

# function loglikelihood(DS::DataSet,model::Function,θ::Vector{<:Number})
#     R = zero(suff(θ))
#     Dot(x) = dot(x,x)
#     if length(ydata(DS)[1]) == 1    # For some reason this is faster.
#         for i in 1:length(xdata(DS))
#             R += ((ydata(DS)[i]-model(xdata(DS)[i],θ))/sigma(DS)[i])^2
#         end
#     else
#         term(i) = Dot((ydata(DS)[i] .- model(xdata(DS)[i],θ))/sigma(DS)[i])
#         R = sum( term(i) for i in 1:length(xdata(DS)) )
#     end
#     -0.5*(length(xdata(DS))*log(2pi) + 2*sum(log.(sigma(DS))) + R)
# end

# function logdet(C::AbstractMatrix)
#     L = log(det(C))
#     if abs(L) < Inf
#         return L
#     else
#         return sum(log.(eigvals(C)))
#     end
# end
# logdet(C::Diagonal) = tr(log(C))

function loglikelihood(DS::DataSet,model::Function,θ::Vector{<:Number})
    Y = ydata(DS) - EmbeddingMap(DS,model,θ)
    # -0.5*(length(xdata(DS))*log(2pi) - log(det(InvCov(DS))) + transpose(Y) * InvCov(DS) * Y)
    -0.5*(N(DS)*log(2pi) - logdet(InvCov(DS)) + transpose(Y) * InvCov(DS) * Y)
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
    n < 0 && throw(ArgumentError("Input must be positive."))
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
InvConfVol(x::Real; tol::Real=1e-15) = find_zero((z->(ConfVol(z)-x)),one(suff(x)),Order8(),xatol=tol)
ChisqCDF(k,x) = gamma_inc(BigFloat(k)/2., BigFloat(x)/2.,0)[1] ./ gamma(BigFloat(k)/2.)
InvChisqCDF(k::Real,x::Real; tol::Real=1e-15) = find_zero((z->(ChisqCDF(k,z)-x)),(1e-5*one(suff(x)),300),Bisection(),xatol=tol)

ChiQuant(sig::Real=1.,k::Int=2) = (1/2)*quantile(Chisq(k),ConfVol(sig))
ChiQuantToSigma(ChiQuant::Real,k::Int=2) = cdf.(Chisq(k),2*ChiQuant) |> InvConfVol

# Cannot be used with DiffEq since tags conflict. Use Zygote.jl?
AutoScore(DM::AbstractDataModel,θ::Vector{<:Number}) = AutoScore(DM.Data,DM.model,θ)
AutoMetric(DM::AbstractDataModel,θ::Vector{<:Number}) = AutoMetric(DM.Data,DM.model,θ)
AutoScore(DS::AbstractDataSet,model::Function,θ::Vector{<:Number}) = ForwardDiff.gradient(x->loglikelihood(DS,model,x),θ)
AutoMetric(DS::AbstractDataSet,model::Function,θ::Vector{<:Number}) = ForwardDiff.hessian(x->(-loglikelihood(DS,model,x)),θ)

# """
# Calculates the score of models with y vales of dim > 1.
# """
# function ScoreDimN(DM::DataModel,p::Vector{<:Number})
#     Res = zeros(suff(p),length(p))
#     moddif = (sigma(DM)).^(-2) .* (ydata(DM) .- map(z->DM.model(z,p),xdata(DM)))
#     dmod = map(z->DM.dmodel(z,p),xdata(DM))
#     for i in 1:length(xdata(DM))
#         Res[:] .+= (transpose(moddif[i]) * DM.dmodel(xdata(DM)[i],p))[:]
#     end
#     Res
# end
#
# # function Score1D(DS::DataSet,model::Function,dmodel::Function,θ::Vector{<:Number})
# #     Res = zeros(suff(θ),length(θ))
# #     mod = EmbeddingMap(DS,model,θ);    dmod = EmbeddingMatrix(DS,dmodel,θ)
# #     for j in 1:length(θ)
# #         Res[j] += sum((sigma(DS)[i])^(-2) *(ydata(DS)[i]-mod[i])*dmod[i,j]   for i in 1:length(xdata(DS)))
# #     end;    Res
# # end
#
# Score1D(DM::AbstractDataModel,θ::Vector{<:Number}) = Score1D(DM.Data,DM.model,DM.dmodel,θ)

function Score(DS::DataSet,model::Function,dmodel::Function,θ::Vector{<:Number})
    transpose(EmbeddingMatrix(DS,dmodel,θ)) * InvCov(DS) * (ydata(DS) - EmbeddingMap(DS,model,θ))
end



"""
    Score(DM::DataModel, θ::Vector{<:Number}; Auto::Bool=false)
Calculates the gradient of the log-likelihood with respect to a set of parameters `p`. `Auto=true` uses automatic differentiation.
"""
function Score(DM::AbstractDataModel, θ::Vector{<:Number}; Auto::Bool=false)
    Auto && return AutoScore(DM,θ)
    return Score(DM.Data,DM.model,DM.dmodel,θ)
    # length(ydata(DM)[1]) != 1 && return ScoreDimN(DM,θ)
    # return Score1D(DM,θ)
end


"""
    WilksTest(DM::DataModel, θ::Vector{<:Real}, ConfVol=ConfVol(1)) -> Bool
Checks whether a given parameter configuration `p` is within a confidence interval of level `ConfVol` using Wilks' theorem.
This makes the assumption, that the likelihood has the form of a normal distribution, which is asymptotically correct in the limit that the number of datapoints is infinite.
"""
WilksTest(DM::DataModel, θ::Vector{<:Real}, ConfVol=ConfVol(1))::Bool = ChisqCDF(length(MLE(DM)), 2(LogLikeMLE(DM) - loglikelihood(DM,θ))) - ConfVol < 0.

# function WilksTest(DM::DataModel, θ::Vector{<:Real}, MLE::Vector{<:Real},ConfVol=ConfVol(1))::Bool
#     # return (loglikelihood(DM,MLE) - loglikelihood(DM,p) <= (1/2)*quantile(Chisq(length(MLE)),Conf))
#     return ChisqCDF(length(MLE), 2(loglikelihood(DM,MLE) - loglikelihood(DM,θ))) - ConfVol < 0
# end


# """
#     WilksTestPrepared(DM::DataModel, θ::Vector{<:Real}, LogLikeMLE::Real, ConfVol=ConfVol(1)) -> Bool
# Checks whether a given parameter configuration ``\\theta`` is within a confidence interval of level `ConfVol` using Wilks' theorem.
# This makes the assumption, that the likelihood has the form of a normal distribution, which is asymptotically correct in the limit that the number of datapoints is infinite.
# To simplify the calculation, `LogLikeMLE` accepts the value of the log-likelihood evaluated at the MLE.
# """
# function WilksTestPrepared(DM::DataModel, θ::Vector{<:Real}, LogLikeMLE::Real, ConfVol=ConfVol(1))::Bool
#     # return (LogLikeMLE - loglikelihood(DM,p) <= (1/2)*quantile(Chisq(length(MLE)),Conf))
#     return ChisqCDF(length(θ), 2(LogLikeMLE - loglikelihood(DM,θ))) - ConfVol < 0.
# end
#
# @deprecate WilksTestPrepared(DM,θ,LogLikeMLE,ConfVol) WilksTest(DM,θ,ConfVol)
# @deprecate WilksTest(DM,θ,MLE,ConfVol) WilksTest(DM,θ,ConfVol)


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
    f(x::Real) = ChisqCDF(length(MLE),2(L-loglikelihood(DM,MLE .+ (x .* BasisVector(1,length(MLE)))))) - CF
    df(x) = ForwardDiff.gradient(f,x)
    @time b = find_zero((f,df),BigFloat(1),Roots.Order2(),xatol=tol)
    println("Finished.")
    MLE +  b .* BasisVector(1,length(MLE))
end


# function Interval1D(DM::DataModel,MLE::Vector,Confnum::Real=1.;tol::Real=1e-14)
#     if tol < 1e-15 || suff(MLE) == BigFloat || typeof(ConfVol(Confnum)) == BigFloat
#         throw("Interval1D not programmed for BigFloat yet.")
#     end
#     (length(MLE) != 1) && throw("Interval1D not defined for p != 1.")
#     lMLE = loglikelihood(DM,MLE)
#     A = lMLE - (1/2)*quantile(Chisq(length(MLE)),ConfVol(Confnum))
#     Func(p::Real) = loglikelihood(DM,MLE .+ p*BasisVector(1,length(MLE))) - A
#     D(f) = x->ForwardDiff.derivative(f,x);  NegFunc(x) = Func(-x)
#     B = find_zero((Func,D(Func)),0.1,Roots.Order1(),xatol=tol)
#     A = find_zero((Func,D(Func)),-B,Roots.Order1(),xatol=tol)
#     rts = [MLE[1]+A, MLE[1]+B]
#     rts[1] < rts[2] && return rts
#     throw("Interval1D errored...")
# end
# @deprecate Interval1D(DM,MLE,Confnum) Interval1D(DM,Confnum)

function Interval1D(DM::DataModel, Confnum::Real=1.; tol::Real=1e-14)
    if tol < 2e-15 || typeof(ConfVol(Confnum)) == BigFloat
        throw("Interval1D not programmed for BigFloat yet.")
    end
    (length(MLE(DM)) != 1) && throw("Interval1D not defined for p != 1.")
    A = LogLikeMLE(DM) - (1/2)*quantile(Chisq(pdim(DM)),ConfVol(Confnum))
    Func(p::Real) = loglikelihood(DM,MLE(DM) .+ p*BasisVector(1,pdim(DM))) - A
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
            value - start > 20 && throw("FindConfBoundary: Value larger than 20.")
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

# function FindConfBoundary(DM::DataModel,MLE::Vector,Confnum::Real; tol::Real=4e-15, maxiter::Int=10000)
#     ((suff(MLE) != BigFloat) && tol < 1e-15) && throw("FindConfBoundary: MLE not BigFloat but tol=$tol.")
#     LogLikeMLE = loglikelihood(DM,MLE);    Confvol = ConfVol(convert(suff(MLE),Confnum))
#     Test(x::Real) = WilksTestPrepared(DM, MLE .+ (x .* BasisVector(1,length(MLE))), LogLikeMLE, Confvol)
#     !(Test(0)) && throw(ArgumentError("FindConfBoundary: Given MLE not inside Confidence Interval."))
#     stepsize = one(suff(MLE))/4.;  value = zero(suff(MLE))
#     for i in 1:maxiter
#         if Test(value + stepsize) # inside
#             value += stepsize
#             value > 20 && throw("FindConfBoundary: Value larger than 10.")
#         else            #outside
#             if stepsize < tol
#                 return value .* BasisVector(1,length(MLE)) .+ MLE
#             end
#             stepsize /= 10
#         end
#     end
#     throw(Error("$maxiter iterations over. Value=$value, Stepsize=$stepsize"))
# end
# @deprecate FindConfBoundary(DM,MLE,Confnum) FindConfBoundary(DM,Confnum)

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


"""
    OrthVF(DM::DataModel, θ::Vector{<:Real}; Auto::Bool=false) -> Vector
Calculates a direction (in parameter space) in which the value of the log-likelihood does not change, given a parameter configuration ``\\theta``.
`Auto=true` uses automatic differentiation to calculate the score.
"""
function OrthVF(DM::DataModel, θ::Vector{<:Number}; alpha::Vector=normalize(length(θ)*BasisVector(length(θ),length(θ)) - ones(length(θ))), Auto::Bool=false)
    length(θ) < 2 && throw(ArgumentError("dim(Parameter Space) < 2  --> No orthogonal VF possible."))
    S = -Score(DM,θ; Auto=Auto);    P = prod(S);    VF = P ./ S
    alpha .* VF |> normalize
end


"""
    OrthVF(DM::DataModel, PL::Plane, θ::Vector{<:Real}; Auto::Bool=false) -> Vector
Calculates a direction (in parameter space) in which the value of the log-likelihood does not change, given a parameter configuration ``\\theta``.
If a `Plane` is specified, the direction will be projected onto it.
`Auto=true` uses automatic differentiation to calculate the score.
"""
function OrthVF(DM::DataModel, PL::Plane, θ::Vector{<:Real}; Auto::Bool=false)
    throw("Needs Reprogramming.")
    # Use PlanarDataModel to ensure proper projection which retains orthogonality to score.
    length(θ) < 2 && throw(ArgumentError("dim(Parameter Space) < 2  --> No orthogonal VF possible"))
    planeorth = Cross(PL.Vx,PL.Vy)
    if length(θ) != 2
        !IsOnPlane(PL,θ) && throw(ArgumentError("Parameter Configuration not on specified Plane."))
        norm(planeorth .- ones(length(θ))) < 1e-14 && throw(ArgumentError("Visualization plane unsuitable: $planeorth"))
    end

    S = Score(DM,θ; Auto=Auto);    P = prod(S);    VF = P ./ S

    alpha = []
    if length(θ) > 2
        alpha = Cross(ones(length(θ)),planeorth)
    else
        alpha = Cross(ones(3),planeorth)[1:2]
    end
    # ProjectOntoPlane(PL,alpha .* VF) |> normalize
    normalize(alpha .* VF)
end


ConstructCube(M::Matrix{<:Real}; Padding::Real=1/50) = HyperCube(ConstructLowerUpper(M; Padding=Padding))
function ConstructLowerUpper(M::Matrix{<:Real}; Padding::Real=1/50)
    lowers = [minimum(M[:,i]) for i in 1:size(M,2)]
    uppers = [maximum(M[:,i]) for i in 1:size(M,2)]
    diff = (uppers - lowers) .* Padding
    LowerUpper(lowers - diff,uppers + diff)
end
ConstructCube(PL::Plane,sol::ODESolution; Padding::Real=1/50) = ConstructCube(Deplanarize(PL,sol;N=300),Padding=Padding)

ConstructCube(sol::ODESolution,Npoints::Int=200; Padding::Real=1/50) = HyperCube(ConstructLowerUpper(sol,Npoints,Padding=Padding))
function ConstructLowerUpper(sol::ODESolution,Npoints::Int=200; Padding::Real=1/50)
    ConstructLowerUpper(Unpack(map(sol,range(sol.t[1],sol.t[end],length=Npoints))),Padding=Padding)
end



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
    Cube = ConstructCube(sol,Padding=1e-5)
    # Indicator function for Integral
    InsideRegion(X::Vector{<:Real})::Bool = loglikelihood(DM,X) < LogLikeBoundary
    Test(X::Vector) = InsideRegion(X) ? sqrt(det(FisherMetric(DM,X))) : 0.
    WE && return MonteCarloAreaWE(Test,Cube,N)
    MonteCarloArea(Test,Cube,N)
end



FindMLEBig(DM::DataModel,start::Vector=MLE(DM)) = FindMLEBig(DM.Data,DM.model,start)
function FindMLEBig(DS::AbstractDataSet,model::Function,start::Union{Bool,Vector}=false)
    if isa(start,Vector)
        NegEll(p::Vector{<:Number}) = -loglikelihood(DS,model,p)
        return optimize(NegEll, BigFloat.(start), BFGS(), Optim.Options(g_tol=convert(BigFloat,10 .^(-precision(BigFloat)/30))), autodiff = :forward) |> Optim.minimizer
    elseif isa(start,Bool)
        return FindMLEBig(DS,model,FindMLE(DS,model))
    end
end

# function FindMLE(DM::DataModel,start::Union{Bool,Vector}=false; Big::Bool=false, tol::Real=1e-14, max::Int=50)
#     Big && return FindMLEBig(DM,start)
#     NegEll(x) = -loglikelihood(DM,x)
#     if isa(start,Bool)
#         return optimize(NegEll, ones(pdim(DM; max=max)), BFGS(), Optim.Options(g_tol=tol), autodiff = :forward) |> Optim.minimizer
#     elseif isa(start,Vector)
#         if suff(start) == BigFloat
#             return FindMLEBig(DM,start)
#         else
#             # println("Warning: Passed $start to FindMLE as starting value.")
#             return optimize(NegEll, start, BFGS(), Optim.Options(g_tol=tol), autodiff = :forward) |> Optim.minimizer
#         end
#     end
# end
# """
#     FindMLE(DM::DataModel,start::Union{Bool,Vector}=false; Big::Bool=false, max::Int=50) -> Vector
# Finds the maximum likelihood parameter configuration given a `DataModel` and optionally a starting configuration. `Big=true` will return the value as a `BigFloat`.
# If no starting value is provided (i.e. `start=false`) the dimension of the parameter space is inferred automatically and the initial configuration is chosen as `start=ones(dim)`.
# """

FindMLE(DM::DataModel,args...;kwargs...) = MLE(DM)
function FindMLE(DS::AbstractDataSet,model::Function,start::Union{Bool,Vector}=false; Big::Bool=false, tol::Real=1e-14)
    (Big || tol < 2.3e-15) && return FindMLEBig(DS,model,start)
    NegEll(p::Vector{<:Number}) = -loglikelihood(DS,model,p)
    if isa(start,Bool)
        # return curve_fit(DS,model,ones(pdim(model,xdata(DS)[1])); tol=tol).param
        return optimize(NegEll, ones(pdim(model,xdata(DS)[1])), BFGS(), Optim.Options(g_tol=tol), autodiff = :forward) |> Optim.minimizer
    elseif isa(start,Vector)
        if suff(start) == BigFloat
            return FindMLEBig(DS,model,start)
        else
            # return curve_fit(DS,model,start; tol=tol).param
            return optimize(NegEll, start, BFGS(), Optim.Options(g_tol=tol), autodiff = :forward) |> Optim.minimizer
        end
    end
end


"""
    GenerateBoundary(DM::DataModel, u0::Vector{<:Real}; tol::Real=1e-14, meth=Tsit5(), mfd::Bool=true) -> ODESolution
Basic method for constructing a curve lying on the confidence region associated with the initial configuration `u0`.
"""
function GenerateBoundary(DM::AbstractDataModel,u0::Vector{<:Number}; tol::Real=1e-14,
    meth::OrdinaryDiffEqAlgorithm=Tsit5(), mfd::Bool=true, Auto::Bool=true)
    LogLikeOnBoundary = loglikelihood(DM,u0)
    IntCurveODE(du,u,p,t) = du .= 0.1 .* OrthVF(DM,u;Auto=Auto)
    g(resid,u,p,t) = resid[1] = LogLikeOnBoundary - loglikelihood(DM,u)
    terminatecondition(u,t,integrator) = u[2] - u0[2]
    # TerminateCondition only on upwards crossing --> supply two different affect functions, leave second free I
    cb = CallbackSet(ManifoldProjection(g),ContinuousCallback(terminatecondition,terminate!,nothing))
    tspan = (0.,1e5);    prob = ODEProblem(IntCurveODE,u0,tspan)
    if mfd
        return solve(prob,meth,reltol=tol,abstol=tol,callback=cb,save_everystep=false)
    else
        return solve(prob,meth,reltol=tol,abstol=tol,callback=ContinuousCallback(terminatecondition,terminate!,nothing))
    end
end

# function GenerateBoundary(DM::DataModel, u0::Vector{<:Number}; tol::Real=1e-14, meth::OrdinaryDiffEqAlgorithm=Tsit5(), mfd::Bool=true)
#     L(p) = loglikelihood(DM,p);    V(p) = OrthVF(DM,p)
#     LogLikeOnBoundary = L(u0)
#     function IntCurveODE(du,u,p,t)
#         du .= 0.1 .* V(u)
#     end
#     function g(resid,u,p,t)
#       resid[1] = LogLikeOnBoundary - L(u)
#     end
#     terminatecondition(u,t,integrator) = u[2] - u0[2]
#     # TerminateCondition only on upwards crossing --> supply two different affect functions, leave second free I
#     cb = CallbackSet(ManifoldProjection(g),ContinuousCallback(terminatecondition,terminate!,nothing))
#     tspan = (0.,1e5)
#     prob = ODEProblem(IntCurveODE,u0,tspan)
#     if mfd
#         return solve(prob,meth,reltol=tol,abstol=tol,callback=cb,save_everystep=false)
#     else
#         return solve(prob,meth,reltol=tol,abstol=tol,callback=ContinuousCallback(terminatecondition,terminate!,nothing))
#     end
# end



# GenerateConfidenceInterval(DM::DataModel,Confnum=1; tol::Real=1e-14, meth::OrdinaryDiffEqAlgorithm=Tsit5(), mfd::Bool=true) = GenerateConfidenceInterval(DM,FindMLE(DM),Confnum, tol=tol, meth=meth, mfd=mfd)
# function GenerateConfidenceInterval(DM::DataModel,MLE::Vector{<:Real},Confnum=1; tol::Real=1e-14, meth::OrdinaryDiffEqAlgorithm=Tsit5(), mfd::Bool=true)
#     if (suff(MLE) != BigFloat) && tol < 2e-15
#         MLE = FindMLEBig(DM,MLE)
#         println("GenerateConfidenceInterval: Promoting MLE to BigFloat because tol=$tol.")
#     end
#     if length(MLE) == 1
#         return Interval1D(DM,MLE,Confnum,tol=tol)
#     else
#         return GenerateBoundary(DM, FindConfBoundary(DM,MLE,Confnum; tol=tol), tol=tol, meth=meth, mfd=mfd)
#     end
# end
# @deprecate GenerateConfidenceInterval(DM,Confnum) GenerateConfidenceRegion(DM,Confnum)
# @deprecate GenerateConfidenceInterval(DM,MLE,Confnum) GenerateConfidenceRegion(DM,Confnum)

function GenerateConfidenceRegion(DM::DataModel, Confnum::Real=1.; tol::Real=1e-14, meth::OrdinaryDiffEqAlgorithm=Tsit5(), mfd::Bool=true)
    if pdim(DM) == 1
        return Interval1D(DM, Confnum; tol=tol)
    else
        return GenerateBoundary(DM, FindConfBoundary(DM, Confnum; tol=tol); tol=tol, meth=meth, mfd=mfd)
    end
end


function StructurallyIdentifiable(DM::DataModel,sol::ODESolution)
    roots = find_zeros(t->GeometricDensity(DM,sol(t)),sol.t[1],sol.t[end])
    length(roots)==0, roots
end

# function GenerateMultipleIntervals(DM::DataModel, Range, MLE=[Inf,Inf]; IsConfVol::Bool=false, tol=1e-14, meth::OrdinaryDiffEqAlgorithm=Tsit5(), mfd::Bool=true)
#     if MLE == [Inf,Inf]     MLE = FindMLE(DM)    end
#     length(MLE) == 1 && return map(x->GenerateConfidenceInterval(DM,MLE,x,tol=tol),Range)
#     LogLikeMLE = loglikelihood(DM,MLE);     sols = Vector{ODESolution}(undef,0)
#     for CONF in Range
#         if IsConfVol
#             @time push!(sols, GenerateConfidenceInterval(DM,MLE,InvConfVol(CONF),tol=tol,meth=meth,mfd=mfd))
#         else
#             @time push!(sols, GenerateConfidenceInterval(DM,MLE,CONF,tol=tol,meth=meth,mfd=mfd))
#         end
#         if sols[end].retcode == :Terminated
#             _ , rts = StructurallyIdentifiable(DM,sols[end])
#             if length(rts) != 0
#                 println("Solution $(length(sols)) corresponding to $(Range[length(sols)]) hits chart boundary at t=$rts and is therefore invalid.")
#             end
#         else
#             println("solution $(length(sols)) did not exit properly: retcode=$(sols[end].retcode).")
#         end
#     end
#     sols
# end
# @deprecate GenerateMultipleIntervals(DM,Range,MLE) MultipleConfidenceRegions(DM,Range)
# @deprecate GenerateMultipleIntervals(DM,Range) MultipleConfidenceRegions(DM,Range)

function MultipleConfidenceRegions(DM::DataModel, Range::Union{AbstractRange,Vector}; IsConfVol::Bool=false, tol::Real=1e-14, meth::OrdinaryDiffEqAlgorithm=Tsit5(), mfd::Bool=true)
    pdim(DM) == 1 && return map(x->GenerateConfidenceRegion(DM,x;tol=tol),Range)
    sols = Vector{ODESolution}(undef,0)
    for CONF in Range
        if IsConfVol
            @time push!(sols, GenerateConfidenceRegion(DM,InvConfVol(CONF);tol=tol,meth=meth,mfd=mfd))
        else
            @time push!(sols, GenerateConfidenceRegion(DM,CONF;tol=tol,meth=meth,mfd=mfd))
        end
        if sols[end].retcode == :Terminated
            _ , rts = StructurallyIdentifiable(DM,sols[end])
            if length(rts) != 0
                println("Solution $(length(sols)) hits chart boundary at t=$rts and is therefore invalid.")
            end
        else
            println("solution $(length(sols)) did not exit properly: retcode=$(sols[end].retcode).")
        end
    end;    sols
end

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

Inside(C::HyperCube,p::Vector) = Inside(LowerUpper(C),p)
function Inside(LU::LowerUpper,p::Vector)::Bool
    length(LU.L) != length(p) && throw("Inside: Dimension mismatch between Cube and point.")
    for i in 1:length(LU.L)
        !(LU.L[i] <= p[i] <= LU.U[i]) && return false
    end;    true
end

"""
    Rsquared(DM::DataModel,Fit::LsqFit.LsqFitResult) -> Real
Calculates the R² value of the fit result `Fit`. It should be noted that the R² value is only a valid measure for the goodness of a fit for linear relationships.
"""
function Rsquared(DM::DataModel,Fit::LsqFit.LsqFitResult)
    length(ydata(DM)[1]) != 1  && return -1
    mean = sum(ydata(DM))/length(ydata(DM))
    Stot = sum((ydata(DM) .- mean).^2)
    Sres = sum((ydata(DM) .- DM.model(xdata(DM),Fit.param)).^2)
    1 - Sres/Stot
end


"""
    Integrate1D(F::Function, Cube::HyperCube; tol::Real=1e-14, fullSol::Bool=false, meth=nothing)
Integrates `F` over a one-dimensional domain specified via a `HyperCube` by rephrasing the integral as an ODE and using `DifferentialEquations.jl`.
"""
function Integrate1D(F::Function, Cube::HyperCube; tol::Real=1e-14, fullSol::Bool=false, meth=nothing)
    Cube.dim != 1 && throw(ArgumentError("Cube dim = $(Cube.dim) instead of 1"))
    Integrate1D(F,Cube.vals[1][:],tol=tol,fullSol=fullSol,meth=meth)
end
function Integrate1D(F::Function, Interval::Vector; tol::Real=1e-14, fullSol::Bool=false, meth=nothing)
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



# Assume that sums from Fisher metric defined with first derivatives of loglikelihood pull out
"""
    FisherMetric(DM::DataModel, θ::Vector{<:Number})
Computes the Fisher metric ``g`` given a `DataModel` and a parameter configuration ``\\theta`` under the assumption that the likelihood ``L(\\mathrm{data} \\, | \\, \\theta)`` is a multivariate normal distribution.
```math
g_{ab}(\\theta) \\coloneqq -\\int_{\\mathcal{D}} \\mathrm{d}^m y_{\\mathrm{data}} \\, L(y_{\\mathrm{data}} \\,|\\, \\theta) \\, \\frac{\\partial^2 \\, \\mathrm{ln}(L)}{\\partial \\theta^a \\, \\partial \\theta^b} = -\\mathbb{E} \\bigg( \\frac{\\partial^2 \\, \\mathrm{ln}(L)}{\\partial \\theta^a \\, \\partial \\theta^b} \\bigg)
```
"""
FisherMetric(DM::AbstractDataModel, θ::Vector{<:Number}) = Pullback(DM,InvCov(DM), θ)

# function FisherMetric(DM::DataModel, θ::Vector{<:Real})
#     F = zeros(suff(θ),length(θ),length(θ))
#     dmod = sigma(DM).^(-1) .* map(z->DM.dmodel(z,θ),xdata(DM))
#     for i in 1:length(xdata(DM))
#         F .+=  transpose(dmod[i]) * dmod[i]
#     end;    F
# end


meshgrid(x, y) = (repeat(x, outer=length(y)), repeat(y, inner=length(x)))

# Always use Integreat for 1D -> Check Dim of Space to decide?
"""
    KullbackLeibler(p::Function,q::Function,Domain::HyperCube=HyperCube([[-15,15]]); tol=2e-15, N::Int=Int(3e7), Carlo::Bool=(Domain.dim!=1))
Computes the Kullback-Leibler divergence between two probability distributions `p` and `q` over the `Domain`.
If `Carlo=true`, this is done using a Monte Carlo Simulation with `N` samples.
If the `Domain` is one-dimensional, the calculation is performed without Monte Carlo to a tolerance of ≈ `tol`.
```math
D_{\\text{KL}}[p,q] \\coloneqq \\int \\mathrm{d}^m y \\, p(y) \\, \\mathrm{ln} \\bigg( \\frac{p(y)}{q(y)} \\bigg)
```
"""
function KullbackLeibler(p::Function,q::Function,Domain::HyperCube=HyperCube([[-15,15]]); tol::Real=2e-15, N::Int=Int(3e7), Carlo::Bool=(Domain.dim!=1))
    function Integrand(x)
        P = p(x)[1];   Q = q(x)[1];   Rat = P/Q
        (Rat <= 0. || abs(Rat) == Inf) && throw(ArgumentError("Ratio p(x)/q(x) = $Rat in log(p/q) for x=$x."))
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
Calculates Kullback-Leibler divergence under the assumption of a normal likelihood.
"""
KullbackLeibler(DM::DataModel,p::Vector,q::Vector) = KullbackLeibler(NormalDist(DM,p),NormalDist(DM,q))

KullbackLeibler(DM::DataModel,p::Vector) = KullbackLeibler(MvNormal(zeros(length(ydata(DM))),inv(InvCov(DM))),NormalDist(DM,p))


# h(p) ∈ Dataspace
"""
    EmbeddingMap(DM::DataModel,θ::Vector{<:Number})
Returns a vector of the collective predictions of the `model` as evaluated at the x-values and the parameter configuration ``\\theta``.
```
h(\\theta) \\coloneqq \\big(y_\\mathrm{model}(x_1;\\theta),...,y_\\mathrm{model}(x_N,\\theta)\\big) \\in \\mathcal{D}
```
"""
EmbeddingMap(DM::AbstractDataModel,θ::Vector{<:Number}) = EmbeddingMap(DM.Data,DM.model,θ)

EmbeddingMap(DS::AbstractDataSet,model::Function,θ::Vector{<:Number}) = model(xdata(DS),θ)

# EmbeddingMap(DM::AbstractDataModel,θ::Vector{<:Number}) = map(x->DM.model(x,θ),xdata(DM))



EmbeddingMatrix(DM::AbstractDataModel,θ::Vector{<:Number}) = EmbeddingMatrix(DM.Data,DM.dmodel,θ)

EmbeddingMatrix(DS::AbstractDataSet,dmodel::Function,θ::Vector{<:Number}) = dmodel(xdata(DS),float.(θ))


# From D to M
Pullback(DM::DataModel,model::Function,θ::Vector) = model(EmbeddingMap(DM,θ))
"""
    Pullback(DM::DataModel, ω::Vector{<:Real}, θ::Vector) -> Vector
Pull-back of a covector to the parameter manifold.
"""
Pullback(DM::DataModel, ω::Vector{<:Real}, θ::Vector) = transpose(EmbeddingMatrix(DM,θ)) * ω


"""
    Pullback(DM::DataModel, G::AbstractArray{<:Real,2}, θ::Vector)
Pull-back of a (0,2)-tensor `G` to the parameter manifold.
"""
function Pullback(DM::AbstractDataModel, G::AbstractMatrix, θ::Vector)
    J = EmbeddingMatrix(DM,θ)
    return transpose(J) * G * J
end

# M to D
"""
    Pushforward(DM::DataModel, X::Vector, point::Vector)
Calculates the push-forward of a vector `X` from the parameter manifold to the data space.
"""
Pushforward(DM::DataModel, X::Vector, θ::Vector) = EmbeddingMatrix(DM,θ) * X

# """
#     DataSpaceDist(DM::DataModel,v::Vector) -> Real
# Calculates the euclidean distance between a point `v` in the data space and the data.
# """
# function DataSpaceDist(DM::DataModel,v::Vector)
#     length(ydata(DM)) != length(v) && error("DataSpaceDist: Dimensional Mismatch")
#     return MetricNorm(Diagonal(sigma(DM).^(-2)),(ydata(DM) .- v))
# end



# Compute all major axes of Fisher Ellipsoid from eigensystem of Fisher metric
FisherEllipsoid(DM::DataModel, θ::Vector{<:Number}) = FisherEllipsoid(p->FisherMetric(DM,p), θ)
FisherEllipsoid(Metric::Function, θ::Vector{<:Number}) = eigvecs(Metric(θ))


"""
    AIC(DM::DataModel, θ::Vector) -> Real
Calculates the Akaike Information Criterion given a parameter configuration ``\\theta`` defined by ``\\mathrm{AIC} = 2 \\, \\mathrm{length}(\\theta) -2 \\, \\ell(\\mathrm{data} \\, | \\, \\theta)``.
"""
AIC(DM::AbstractDataModel, θ::Vector{<:Number}) = 2length(θ) - 2loglikelihood(DM,θ)

"""
    BIC(DM::DataModel, θ::Vector) -> Real
Calculates the Bayesian Information Criterion given a parameter configuration ``\\theta`` defined by ``\\mathrm{AIC} = \\mathrm{ln}(N) \\cdot \\mathrm{length}(\\theta) -2 \\, \\ell(\\mathrm{data} \\, | \\, \\theta)`` where ``N`` is the number of data points.
"""
BIC(DM::AbstractDataModel, θ::Vector{<:Number}) = length(θ)*log(length(DM.Data)) - 2loglikelihood(DM,θ)

"""
    IsLinearParameter(DM::DataModel) -> Vector{Bool}
Checks with respect to which parameters the model function `model(x,θ)` is linear and returns vector of booleans where `true` indicates linearity.
This test is performed by comparing the Jacobians of the model for two random configurations ``\\theta_1, \\theta_2 \\in \\mathcal{M}`` column by column.
"""
function IsLinearParameter(DM::AbstractDataModel)::Vector{Bool}
    J1 = EmbeddingMatrix(DM,rand(pdim(DM)));        J2 = EmbeddingMatrix(DM,rand(pdim(DM)))
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
    CorrectedCovariance(DM::DataModel; tol::Real=1e-14, disc::Bool=false)
Experimental function which attempts to compute the exact covariance matrix for linear models.
"""
function CorrectedCovariance(DM::DataModel; tol::Real=1e-14, disc::Bool=false)
    if !IsLinear(DM)
        println("CorrectedCovariance: model not linear, thus ∄ linear covariance matrix.")
        return false
    end
    C = Symmetric(inv(FisherMetric(DM,MLE(DM))));    L = cholesky(C).L
    lenp = length(MLE(DM));    res = 0.
    v = L*normalize(ones(lenp));    CF = ConfVol(1.)
    TestCont(x::Real) = ChisqCDF(lenp,2(LogLikeMLE(DM)-loglikelihood(DM,MLE(DM) + x.*v))) - CF
    TestDisc(x::Real)::Bool = WilksTest(DM, MLE(DM) + x.*v, CF)
    if disc
        res = LineSearch(TestDisc,1.;tol=tol)
    else
        res = find_zero(TestCont,1.;xatol=tol)
    end
    return res^2 .* C
end
