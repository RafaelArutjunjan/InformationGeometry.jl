

"""
    likelihood(DM::DataModel,θ::AbstractVector) -> Real
Calculates the likelihood ``L(\\mathrm{data} \\, | \\, \\theta)`` a `DataModel` and a parameter configuration ``\\theta``.
"""
likelihood(args...) = exp(loglikelihood(args...))

# import Distributions.loglikelihood
"""
    loglikelihood(DM::DataModel, θ::AbstractVector) -> Real
Calculates the logarithm of the likelihood ``L``, i.e. ``\\ell(\\mathrm{data} \\, | \\, \\theta) \\coloneqq \\mathrm{ln} \\big( L(\\mathrm{data} \\, | \\, \\theta) \\big)`` given a `DataModel` and a parameter configuration ``\\theta``.
"""
loglikelihood(DM::AbstractDataModel, θ::AbstractVector{<:Number}) = loglikelihood(Data(DM), Predictor(DM), θ)

@inline function loglikelihood(DS::DataSet, model::ModelOrFunction, θ::AbstractVector{<:Number})
    Y = ydata(DS) - EmbeddingMap(DS, model, θ)
    -0.5*(Npoints(DS)*ydim(DS)*log(2pi) - logdetInvCov(DS) + transpose(Y) * InvCov(DS) * Y)
end


AutoScore(DM::AbstractDataModel, θ::AbstractVector{<:Number}) = AutoScore(Data(DM), Predictor(DM), θ)
AutoMetric(DM::AbstractDataModel, θ::AbstractVector{<:Number}) = AutoMetric(Data(DM), Predictor(DM), θ)
AutoScore(DS::AbstractDataSet,model::ModelOrFunction,θ::AbstractVector{<:Number}) = ForwardDiff.gradient(x->loglikelihood(DS,model,x),θ)
AutoMetric(DS::AbstractDataSet,model::ModelOrFunction,θ::AbstractVector{<:Number}) = ForwardDiff.hessian(x->(-loglikelihood(DS,model,x)),θ)


"""
    Score(DM::DataModel, θ::AbstractVector{<:Number}; Auto::Bool=false)
Calculates the gradient of the log-likelihood ``\\ell`` with respect to a set of parameters ``\\theta``. `Auto=true` uses automatic differentiation.
"""
Score(DM::AbstractDataModel, θ::AbstractVector{<:Number}; Auto::Bool=false) = Score(Data(DM),Predictor(DM),dPredictor(DM),θ; Auto=Auto)

function Score(DS::AbstractDataSet, model::ModelOrFunction, dmodel::ModelOrFunction, θ::AbstractVector{<:Number}; Auto::Bool=false)
    Auto && return AutoScore(DS,model,θ)
    Score(DS,model,dmodel,θ)
end

@inline function Score(DS::DataSet,model::ModelOrFunction,dmodel::ModelOrFunction,θ::AbstractVector{<:Number})
    transpose(EmbeddingMatrix(DS,dmodel,θ)) * (InvCov(DS) * (ydata(DS) - EmbeddingMap(DS,model,θ)))
end

"""
Point θ lies outside confidence region of level `Confvol` if this function > 0.
"""
@inline WilksCriterion(DM::AbstractDataModel, θ::AbstractVector{<:BigFloat}, Confvol::BigFloat=ConfVol(BigFloat(1.))) = ChisqCDF(pdim(DM), 2(LogLikeMLE(DM) - loglikelihood(DM,θ))) - Confvol
@inline WilksCriterion(DM::AbstractDataModel, θ::AbstractVector{<:Number}, Confvol::Real=ConfVol(1.)) = cdf(Chisq(pdim(DM)), 2(LogLikeMLE(DM) - loglikelihood(DM,θ))) - Confvol

# Do not give default to third argument here such as to not overrule the defaults from above
WilksCriterion(DM::AbstractDataModel, θ::AbstractVector{<:Float64}, Confvol::BigFloat) = WilksCriterion(DM, BigFloat.(θ), Confvol)
WilksCriterion(DM::AbstractDataModel, θ::AbstractVector{<:BigFloat}, Confvol::Float64) = WilksCriterion(DM, θ, BigFloat(Confvol))

"""
    WilksTest(DM::DataModel, θ::AbstractVector{<:Number}, Confvol=ConfVol(1)) -> Bool
Checks whether a given parameter configuration `θ` is within a confidence interval of level `Confvol` using Wilks' theorem.
This makes the assumption, that the likelihood has the form of a normal distribution, which is asymptotically correct in the limit that the number of datapoints is infinite.
"""
WilksTest(DM::AbstractDataModel, θ::AbstractVector{<:Number}, Confvol::Real=ConfVol(one(suff(θ))))::Bool = WilksCriterion(DM, θ, Confvol) < 0.


# function FindConfBoundary(DM::DataModel, Confnum::Real; tol::Real=4e-15, maxiter::Int=10000)
#     ((suff(MLE(DM)) != BigFloat) && tol < 2e-15) && throw("MLE(DM) must first be promoted to BigFloat via DM = DataModel(Data(DM),DM.model,DM.dmodel,BigFloat.(MLE(DM))).")
#     Confvol = ConfVol(Confnum);    Test(x::Real) = WilksTest(DM, MLE(DM) .+ (x .* BasisVector(1,pdim(DM))), Confvol)
#     !(Test(0)) && throw(ArgumentError("FindConfBoundary: Given MLE not inside Confidence Interval."))
#     stepsize = one(suff(MLE(DM)))/4.;  value = zero(suff(MLE(DM)))
#     for i in 1:maxiter
#         if Test(value + stepsize) # inside
#             value += stepsize
#             value > 20 && throw("FindConfBoundary: Value larger than 10.")
#         else            #outside
#             if stepsize < tol
#                 return value .* BasisVector(1,pdim(DM)) .+ MLE(DM)
#             end
#             stepsize /= 10
#         end
#     end
#     throw(Error("$maxiter iterations over. Value=$value, Stepsize=$stepsize"))
# end

function FindConfBoundary(DM::AbstractDataModel, Confnum::Real; tol::Real=4e-15, maxiter::Int=10000)
    ((suff(MLE(DM)) != BigFloat) && tol < 2e-15) && throw("For tol < 2e-15, MLE(DM) must first be promoted to BigFloat via DM = DataModel(Data(DM),DM.model,DM.dmodel,BigFloat.(MLE(DM))).")
    CF = ConfVol(Confnum)
    Test(x) = WilksTest(DM,x .* BasisVector(1,pdim(DM)) + MLE(DM), CF)
    MLE(DM) .+ LineSearch(Test; tol=tol, maxiter=maxiter) .* BasisVector(1,pdim(DM))
end


# function FtestPrepared(DM::DataModel, θ::Vector, S_MLE::Real, ConfVol=ConfVol(1))::Bool
#     n = length(ydata(DM));  p = length(θ);    S(P) = sum(((ydata(DM) .- map(x->DM.model(x,P),xdata(DM)))./sigma(DM)).^2)
#     S(θ) ≤ S_MLE * (1. + p/(n-p)) * quantile(FDist(p, n-p),ConfVol)
# end
# Ftest(DM::DataModel, θ::Vector, MLE::Vector, Conf=ConfVol(1))::Bool = FtestPrepared(DM,θ,sum((ydata(DM) .- map(x->DM.model(x,MLE),xdata(DM))).^2),Conf)

# equivalent to ResidualSquares(DM,MLE(DM))
RS_MLE(DM::AbstractDataModel) = InformationGeometry.logdetInvCov(DM) - Npoints(DM)*ydim(DM)*log(2pi) - 2LogLikeMLE(DM)
ResidualSquares(DM::AbstractDataModel, θ::AbstractVector{<:Real}) = ResidualSquares(Data(DM), Predictor(DM), θ)
function ResidualSquares(DS::AbstractDataSet, model::ModelOrFunction, θ::AbstractVector{<:Real})
    Y = ydata(DS) - EmbeddingMap(DS,model,θ)
    transpose(Y) * InvCov(DS) * Y
end
function FCriterion(DM::AbstractDataModel, θ::AbstractVector{<:Real}, Confvol::Real=ConfVol(one(suff(θ))))
    n = length(ydata(DM));  p = length(θ)
    ResidualSquares(DM,θ) - RS_MLE(DM) * (1. + length(θ)/(n - p)) * quantile(FDist(p, n-p),Confvol)
end
function FTest(DM::AbstractDataModel, θ::AbstractVector{<:Real}, Confvol::Real=ConfVol(one(suff(θ))))::Bool
    FCriterion(DM,θ,Confvol) < 0
end
function FindFBoundary(DM::AbstractDataModel, Confnum::Real; tol::Real=4e-15, maxiter::Int=10000)
    ((suff(MLE(DM)) != BigFloat) && tol < 2e-15) && throw("For tol < 2e-15, MLE(DM) must first be promoted to BigFloat via DM = DataModel(Data(DM),DM.model,DM.dmodel,BigFloat.(MLE(DM))).")
    CF = ConfVol(Confnum)
    Test(x) = FTest(DM,x .* BasisVector(1,pdim(DM)) + MLE(DM), CF)
    MLE(DM) .+ LineSearch(Test; tol=tol, maxiter=maxiter) .* BasisVector(1,pdim(DM))
end


FDistCDF(x,d1,d2) = beta_inc(d1/2.,d2/2.,d1*x/(d1*x + d2)) #, 1 .-d1*BigFloat(x)/(d1*BigFloat(x) + d2))[1]
# function Ftest2(DM::DataModel, point::Vector{T}, MLE::Vector{T}, ConfVol::T=ConfVol(1))::Bool where {T<:BigFloat}
#     n = length(ydata(DM));  p = length(point);    S(P) = sum(((ydata(DM) .- map(x->DM.model(x,P),xdata(DM)))./sigma(DM)).^2)
#     FDistCDF(S(point) / (S(MLE) * (1 + p/(n-p))),p,n-p) ≤ ConfVol
# end


inversefactor(m::Real) = 1. / sqrt((m - 1.) + (m - 1.)^2)
@inline function GetAlpha(n::Int)
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
function OrthVF(DM::AbstractDataModel, θ::AbstractVector{<:Number}; alpha::AbstractVector=GetAlpha(length(θ)), Auto::Bool=false)
    OrthVF(Data(DM), Predictor(DM), dPredictor(DM), θ; alpha=alpha, Auto=Auto)
end

function OrthVF(DS::AbstractDataSet, model::ModelOrFunction, dmodel::ModelOrFunction, θ::AbstractVector{<:Number};
                alpha::AbstractVector=GetAlpha(length(θ)), Auto::Bool=false)
    length(θ) < 2 && throw(ArgumentError("dim(Parameter Space) < 2  --> No orthogonal VF possible."))
    S = -Score(DS, model, dmodel, θ; Auto=Auto);    P = prod(S);    VF = P ./ S
    alpha .* VF |> normalize
end


"""
    OrthVF(DM::DataModel, PL::Plane, θ::Vector{<:Real}; Auto::Bool=false) -> Vector
Calculates a direction (in parameter space) in which the value of the log-likelihood does not change, given a parameter configuration ``\\theta``.
If a `Plane` is specified, the direction will be specified in the planar coordinates using a 2-component vector.
`Auto=true` uses automatic differentiation to calculate the score.
"""
function OrthVF(DM::AbstractDataModel,PL::Plane,θ::AbstractVector{<:Number}; Auto::Bool=false)
    S = transpose([PL.Vx PL.Vy]) * (-Score(DM, PlaneCoordinates(PL,θ); Auto=Auto))
    P = prod(S)
    return SA[-P/S[1],P/S[2]] |> normalize
end


FindMLEBig(DM::DataModel,start::AbstractVector{<:Number}=MLE(DM)) = FindMLEBig(Data(DM), Predictor(DM), convert(Vector,start))
function FindMLEBig(DS::AbstractDataSet,model::ModelOrFunction,start::Union{Bool,AbstractVector}=false)
    if isa(start,Vector)
        NegEll(p::AbstractVector{<:Number}) = -loglikelihood(DS,model,p)
        return optimize(NegEll, BigFloat.(convert(Vector,start)), BFGS(), Optim.Options(g_tol=convert(BigFloat,10^(-precision(BigFloat)/30))), autodiff = :forward) |> Optim.minimizer
    elseif isa(start,Bool)
        return FindMLEBig(DS,model,FindMLE(DS,model))
    end
end

GetStartP(DM::AbstractDataModel) = GetStartP(Data(DM), Predictor(DM))
function GetStartP(DS::AbstractDataSet, model::ModelOrFunction)
    P = pdim(DS,model)
    ones(P) .+ 0.01*(rand(P) .- 0.5)
end

function FindMLE(DM::AbstractDataModel, start::Union{Bool,AbstractVector}=false; Big::Bool=false, tol::Real=1e-14)
    FindMLE(Data(DM), Predictor(DM), start; Big=Big, tol=tol)
end
function FindMLE(DS::AbstractDataSet, model::ModelOrFunction, start::Union{Bool,AbstractVector}=false; Big::Bool=false, tol::Real=1e-14)
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
    ConfidenceInterval1D(DM::AbstractDataModel, Confnum::Real=1.; tol::Real=1e-14) -> Tuple{Real,Real}
Returns the confidence interval associated with confidence level `Confnum` in the case of one-dimensional parameter spaces.
"""
function ConfidenceInterval1D(DM::AbstractDataModel, Confnum::Real=1.; tol::Real=1e-13)
    (tol < 2e-15 || Confnum > 8) && throw("ConfidenceInterval1D not programmed for BigFloat yet.")
    pdim(DM) != 1 && throw("ConfidenceInterval1D not defined for p != 1.")
    A = LogLikeMLE(DM) - (1/2)*InvChisqCDF(pdim(DM),ConfVol(Confnum))
    Func(p::Real) = loglikelihood(DM,MLE(DM) + p*BasisVector(1,pdim(DM))) - A
    D(f) = x->ForwardDiff.derivative(f,x);  NegFunc(x) = Func(-x)
    B = find_zero((Func,D(Func)),0.1,Roots.Order1(); xatol=tol)
    A = find_zero((Func,D(Func)),-B,Roots.Order1(); xatol=tol)
    rts = (MLE(DM)[1]+A, MLE(DM)[1]+B)
    rts[1] ≥ rts[2] ? throw("ConfidenceInterval1D errored...") : return rts
end


"""
    GenerateBoundary(DM::DataModel, u0::AbstractVector{<:Number}; tol::Real=1e-14, meth=Tsit5(), mfd::Bool=true) -> ODESolution
Basic method for constructing a curve lying on the confidence region associated with the initial configuration `u0`.
"""
function GenerateBoundary(DM::AbstractDataModel,u0::AbstractVector{<:Number}; tol::Real=1e-9, Boundaries::Union{Function,Nothing}=nothing,
                            meth::OrdinaryDiffEqAlgorithm=Tsit5(), mfd::Bool=false, Auto::Bool=false, kwargs...)
    GenerateBoundary(Data(DM),Predictor(DM),dPredictor(DM),u0; tol=tol, Boundaries=Boundaries, meth=meth, mfd=mfd, Auto=Auto, kwargs...)
end

function GenerateBoundary(DS::AbstractDataSet, model::ModelOrFunction, dmodel::ModelOrFunction, u0::AbstractVector{<:Number}; tol::Real=1e-9,
                            Boundaries::Union{Function,Nothing}=nothing, meth::OrdinaryDiffEqAlgorithm=Tsit5(), mfd::Bool=false, Auto::Bool=false, kwargs...)
    LogLikeOnBoundary = loglikelihood(DS, model, u0)
    IntCurveODE!(du,u,p,t)  =  du .= 0.1 .* OrthVF(DS,model,dmodel,u; Auto=Auto)
    g!(resid,u,p,t)  =  resid[1] = LogLikeOnBoundary - loglikelihood(DS,model,u)
    terminatecondition(u,t,integrator) = u[2] - u0[2]
    # TerminateCondition only on upwards crossing --> supply two different affect functions, leave second free I
    CB = ContinuousCallback(terminatecondition,terminate!,nothing)
    CB = Boundaries != nothing ? CallbackSet(CB, DiscreteCallback(Boundaries,terminate!)) : CB
    tspan = (0.,1e5);    prob = ODEProblem(IntCurveODE!,u0,tspan)
    if mfd
        return solve(prob, meth; reltol=tol, abstol=tol, callback=CallbackSet(CB,ManifoldProjection(g!)), kwargs...)
    else
        return solve(prob, meth; reltol=tol, abstol=tol, callback=CB, kwargs...)
    end
end

function GenerateBoundary(DM::AbstractDataModel, PL::Plane, u0::AbstractVector{<:Number}; tol::Real=1e-9, mfd::Bool=false,
                            Boundaries::Union{Function,Nothing}=nothing, meth::OrdinaryDiffEqAlgorithm=Tsit5(), Auto::Bool=false, kwargs...)
    length(u0) != 2 && throw("length(u0) != 2 although a Plane was specified.")
    LogLikeOnBoundary = loglikelihood(DM,PlaneCoordinates(PL,u0))
    IntCurveODE!(du,u,p,t)  =  du .= 0.1 * OrthVF(DM,PL,u; Auto=Auto)
    g!(resid,u,p,t)  =  resid[1] = LogLikeOnBoundary - loglikelihood(DM,PlaneCoordinates(PL,u))
    terminatecondition(u,t,integrator) = u[2] - u0[2]
    CB = ContinuousCallback(terminatecondition,terminate!,nothing)
    CB = Boundaries != nothing ? CallbackSet(CB, DiscreteCallback(Boundaries, terminate!)) : CB
    tspan = (0.,1e5);    prob = ODEProblem(IntCurveODE!,u0,tspan)
    if mfd
        return solve(prob,meth; reltol=tol,abstol=tol,callback=CallbackSet(CB, ManifoldProjection(g!)), kwargs...)
    else
        return solve(prob,meth; reltol=tol,abstol=tol,callback=CB, kwargs...)
    end
end

"""
Choose method depending on dimensionality of the parameter space.
"""
function ConfidenceRegion(DM::AbstractDataModel, Confnum::Real=1.; tol::Real=1e-9, meth::OrdinaryDiffEqAlgorithm=Tsit5(), mfd::Bool=false,
                            Boundaries::Union{Function,Nothing}=nothing, Auto::Bool=false, parallel::Bool=false, kwargs...)
    if pdim(DM) == 1
        return ConfidenceInterval1D(DM, Confnum; tol=tol)
    elseif pdim(DM) == 2
        return GenerateBoundary(DM, FindConfBoundary(DM, Confnum; tol=tol); tol=tol, Boundaries=Boundaries, meth=meth, mfd=mfd, Auto=Auto, kwargs...)
    else
        println("ConfidenceRegion() computes solutions in the θ[1]-θ[2] plane which are separated in the θ[3] direction. For more explicit control, call MincedBoundaries() and set options manually.")
        Cube = LinearCuboid(DM,Confnum)
        Planes = IntersectCube(DM,Cube,Confnum; Dirs=[1,2,3], N=30)
        return Planes, MincedBoundaries(DM,Planes,Confnum; tol=tol, Boundaries=Boundaries, Auto=Auto, meth=meth, mfd=mfd, parallel=parallel)
    end
end


IsStructurallyIdentifiable(DM::AbstractDataModel, sol::ODESolution)::Bool = length(StructurallyIdentifiable(DM,sol)) == 0

function StructurallyIdentifiable(DM::AbstractDataModel, sol::ODESolution)
    find_zeros(t->GeometricDensity(x->FisherMetric(DM,x),sol(t)), sol.t[1], sol.t[end])
end
function StructurallyIdentifiable(DM::AbstractDataModel, sols::Vector{<:ODESolution}; parallel::Bool=false)
    parallel ? pmap(x->StructurallyIdentifiable(DM,x), sols) : map(x->StructurallyIdentifiable(DM,x), sols)
end



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
function ConfidenceRegions(DM::DataModel, Confnums::Union{AbstractRange,AbstractVector}=1:1; IsConfVol::Bool=false,
                        tol::Real=1e-9, meth::OrdinaryDiffEqAlgorithm=Tsit5(), mfd::Bool=false, Auto::Bool=false,
                        Boundaries::Union{Function,Nothing}=nothing, CheckSols::Bool=true, parallel::Bool=false, kwargs...)
    Range = IsConfVol ? InvConfVol.(Confnums) : Confnums
    Map = parallel ? pmap : map
    if pdim(DM) == 1
        return Map(x->ConfidenceRegion(DM,x; tol=tol), Range)
    elseif pdim(DM) == 2
        sols = Map(x->ConfidenceRegion(DM,x; tol=tol,Boundaries=Boundaries,meth=meth,mfd=mfd,Auto=Auto,kwargs...), Range)
        if CheckSols
            NotTerminated = map(x->(x.retcode != :Terminated), sols)
            sum(NotTerminated) != 0 && println("Solutions $((1:length(sols))[NotTerminated]) did not exit properly.")
            roots = StructurallyIdentifiable(DM, sols; parallel=parallel)
            Unidentifiables = map(x->(length(x) != 0), roots)
            for i in 1:length(roots)
                length(roots[i]) != 0 && println("Solution $i hits chart boundary at t = $(roots[i]) and should therefore be considered invalid.")
            end
        end
        return sols
    else
        throw("This functionality is still under construction. Use ConfidenceRegion() instead.")
    end
end

"""
    GenerateInterruptedBoundary(DM::AbstractDataModel, u0::AbstractVector{<:Number}; Boundaries::Union{Function,Nothing}=nothing, tol::Real=1e-12,
                                redo::Bool=true, meth::OrdinaryDiffEqAlgorithm=Tsit5(), mfd::Bool=true, Auto::Bool=false, kwargs...) -> ODESolution
Integrates along the level lines of the log-likelihood in the counter-clockwise direction until the model becomes either
1. structurally identifiable via `det(g) < tol`
2. the given `Boundaries(u,t,int)` method evaluates to `true`.
It then integrates from where this obstruction was met in the clockwise direction until said obstruction is hit again, resulting in a half-open confidence region.
"""
function GenerateInterruptedBoundary(DM::AbstractDataModel, u0::AbstractVector{<:Number}; Boundaries::Union{Function,Nothing}=nothing, tol::Real=1e-12,
                                redo::Bool=true, meth::OrdinaryDiffEqAlgorithm=Tsit5(), mfd::Bool=true, Auto::Bool=false, kwargs...)
    GenerateInterruptedBoundary(Data(DM), Predictor(DM), dPredictor(DM), u0;
                    Boundaries=Boundaries, tol=tol, meth=meth, mfd=mfd, Auto=Auto, kwargs...)
end

function GenerateInterruptedBoundary(DS::AbstractDataSet, model::ModelOrFunction, dmodel::ModelOrFunction, u0::AbstractVector{<:Number}; tol::Real=1e-12,
                                redo::Bool=true, Boundaries::Union{Function,Nothing}=nothing, meth::OrdinaryDiffEqAlgorithm=Tsit5(), mfd::Bool=true, Auto::Bool=false, kwargs...)
    LogLikeOnBoundary = loglikelihood(DS,model,u0)
    IntCurveODE!(du,u,p,t)  =  du .= 0.1 .* OrthVF(DS,model,dmodel,u; Auto=Auto)
    BackwardsIntCurveODE!(du,u,p,t)  =  du .= -0.1 .* OrthVF(DS,model,dmodel,u; Auto=Auto)
    g!(resid,u,p,t)  =  resid[1] = LogLikeOnBoundary - loglikelihood(DS, model, u)

    terminatecondition(u,t,integrator) = u[2] - u0[2]
    Singularity(u,t,integrator) = det(FisherMetric(DS, dmodel, u)) - tol

    ForwardsTerminate = ContinuousCallback(terminatecondition,terminate!,nothing)
    nonmfdCB = CallbackSet(ForwardsTerminate, ContinuousCallback(Singularity,terminate!))
    nonmfdCB = Boundaries != nothing ? CallbackSet(nonmfdCB, DiscreteCallback(Boundaries,terminate!)) : nonmfdCB
    mfdCB = CallbackSet(ManifoldProjection(g!), nonmfdCB)

    tspan = (0., 1e5);    Forwardprob = ODEProblem(IntCurveODE!,u0,tspan)
    sol1 = mfd ? solve(Forwardprob,meth; reltol=tol,abstol=tol,callback=mfdCB,kwargs...) : solve(Forwardprob,meth; reltol=tol,abstol=tol,callback=nonmfdCB,kwargs...)
    if norm(sol1.u[end] - sol1.u[1]) < 10tol
        return sol1
    else
        nonmfdCB = ContinuousCallback(Singularity, terminate!)
        nonmfdCB = Boundaries != nothing ? CallbackSet(nonmfdCB, DiscreteCallback(Boundaries,terminate!)) : nonmfdCB
        mfdCB = CallbackSet(ManifoldProjection(g!), nonmfdCB)
        Backprob = redo ? ODEProblem(BackwardsIntCurveODE!,sol1.u[end],tspan) : ODEProblem(BackwardsIntCurveODE!,u0,tspan)
        sol2 = mfd ? solve(Backprob,meth; reltol=tol,abstol=tol,callback=mfdCB,kwargs...) : solve(Backprob,meth; reltol=tol,abstol=tol,callback=nonmfdCB,kwargs...)
    end
    return redo ? sol2 : [sol1, sol2]
end


# Assume that sums from Fisher metric defined with first derivatives of loglikelihood pull out
"""
    FisherMetric(DM::DataModel, θ::AbstractVector{<:Number})
Computes the Fisher metric ``g`` given a `DataModel` and a parameter configuration ``\\theta`` under the assumption that the likelihood ``L(\\mathrm{data} \\, | \\, \\theta)`` is a multivariate normal distribution.
```math
g_{ab}(\\theta) \\coloneqq -\\int_{\\mathcal{D}} \\mathrm{d}^m y_{\\mathrm{data}} \\, L(y_{\\mathrm{data}} \\,|\\, \\theta) \\, \\frac{\\partial^2 \\, \\mathrm{ln}(L)}{\\partial \\theta^a \\, \\partial \\theta^b} = -\\mathbb{E} \\bigg( \\frac{\\partial^2 \\, \\mathrm{ln}(L)}{\\partial \\theta^a \\, \\partial \\theta^b} \\bigg)
```
"""
FisherMetric(DM::AbstractDataModel, θ::AbstractVector{<:Number}) = FisherMetric(Data(DM), dPredictor(DM), θ)
FisherMetric(DS::AbstractDataSet, dmodel::ModelOrFunction, θ::AbstractVector{<:Number}) = Pullback(DS, dmodel, InvCov(DS), θ)

"""
    GeometricDensity(DM::DataModel, θ::AbstractVector) -> Real
Computes the square root of the determinant of the Fisher metric ``\\sqrt{\\mathrm{det}\\big(g(\\theta)\\big)}`` at the point ``\\theta``.
"""
GeometricDensity(DM::AbstractDataModel, θ::AbstractVector{<:Number}) = GeometricDensity(Data(DM), dPredictor(DM), θ)
GeometricDensity(DS::AbstractDataSet, dmodel::ModelOrFunction, θ::AbstractVector{<:Number}) = FisherMetric(DS, dmodel, θ) |> det |> sqrt
GeometricDensity(Metric::Function, θ::AbstractVector{<:Number}) = sqrt(det(Metric(θ)))



function ConfidenceRegionVolume(DM::DataModel, sol::ODESolution, N::Int=Int(1e5); WE::Bool=false)
    length(sol.u[1]) != 2 && throw("Not Programmed for dim > 2 yet.")
    LogLikeBoundary = likelihood(DM,sol(0))
    Cube = ConstructCube(sol; Padding=1e-5)
    # Indicator function for Integral, USE HCubature INSTEAD OF MONTECARLO
    InsideRegion(X::AbstractVector{<:Real})::Bool = loglikelihood(DM,X) < LogLikeBoundary
    Test(X::AbstractVector) = InsideRegion(X) ? GeometricDensity(DM,X) : zero(suff(X))
    MonteCarloArea(Test,Cube,N; WE=WE)
end


# h(θ) ∈ Dataspace
"""
    EmbeddingMap(DM::DataModel,θ::AbstractVector{<:Number})
Returns a vector of the collective predictions of the `model` as evaluated at the x-values and the parameter configuration ``\\theta``.
```
h(\\theta) \\coloneqq \\big(y_\\mathrm{model}(x_1;\\theta),...,y_\\mathrm{model}(x_N;\\theta)\\big) \\in \\mathcal{D}
```
"""
EmbeddingMap(DM::AbstractDataModel, θ::AbstractVector{<:Number}) = EmbeddingMap(Data(DM), Predictor(DM), θ)

EmbeddingMap(DS::AbstractDataSet, model::ModelOrFunction, θ::AbstractVector{<:Number}) = performMap(DS, model, θ, WoundX(DS))

function performMap(DS::AbstractDataSet, model::ModelOrFunction, θ::AbstractVector{<:Number}, woundX::AbstractVector)
    if ydim(DS) > 1
        return reduce(vcat, map(x->model(x,θ),woundX))
    else
        return map(x->model(x,θ), woundX)
    end
end

# Typically slower than reduce(vcat,...)
function performMap2(DS::AbstractDataSet, model::ModelOrFunction, θ::AbstractVector{<:Number}, woundX::AbstractVector)
    if ydim(DS) > 1
        Res = Vector{suff(θ)}(undef, Npoints(DS)*ydim(DS))
        for i in 1:Npoints(DS)
            Res[1+(i-1)*ydim(DS):(i*ydim(DS))] = model(woundX[i],θ)
        end;    return Res
    else
        return map(x->model(x,θ), woundX)
    end
end

function performMap3(DS::AbstractDataSet, model::ModelOrFunction, θ::AbstractVector{<:Number}, woundX::AbstractVector{<:SArray})
    # if model outputs StaticArrays
    reinterpret(suff(θ), map(x->model(x,θ),woundX))
end


EmbeddingMatrix(DM::AbstractDataModel, θ::AbstractVector{<:Number}) = EmbeddingMatrix(Data(DM), dPredictor(DM), θ)

EmbeddingMatrix(DS::AbstractDataSet, dmodel::ModelOrFunction, θ::AbstractVector{<:Number}) = performDMap(DS, dmodel, float.(θ), WoundX(DS))

performDMap(DS::AbstractDataSet,dmodel::ModelOrFunction,θ::AbstractVector{<:Number},woundX::AbstractVector) = reduce(vcat,map(x->dmodel(x,θ),woundX))

# very slightly faster apparently
function performDMap2(DS::AbstractDataSet,dmodel::ModelOrFunction,θ::AbstractVector{<:Number},woundX::AbstractVector)
    Res = Array{suff(θ)}(undef,Npoints(DS)*ydim(DS),length(θ))
    for i in 1:Npoints(DS)
        Res[1+(i-1)*ydim(DS):(i*ydim(DS)),:] = dmodel(woundX[i],θ)
    end;    Res
end



# M ⟵ D
Pullback(DM::AbstractDataModel,F::Function,θ::AbstractVector{<:Number}) = F(EmbeddingMap(DM,θ))
"""
    Pullback(DM::DataModel, ω::AbstractVector{<:Real}, θ::Vector) -> Vector
Pull-back of a covector to the parameter manifold.
"""
Pullback(DM::AbstractDataModel, ω::AbstractVector{<:Real}, θ::AbstractVector{<:Number}) = transpose(EmbeddingMatrix(DM,θ)) * ω


"""
    Pullback(DM::DataModel, G::AbstractArray{<:Real,2}, θ::Vector)
Pull-back of a (0,2)-tensor `G` to the parameter manifold.
"""
Pullback(DM::AbstractDataModel, G::AbstractMatrix, θ::AbstractVector{<:Number}) = Pullback(Data(DM), dPredictor(DM), G, θ)
@inline function Pullback(DS::AbstractDataSet, dmodel::ModelOrFunction, G::AbstractMatrix, θ::AbstractVector{<:Number})
    J = EmbeddingMatrix(DS,dmodel,θ)
    transpose(J) * G * J
end

# M ⟶ D
"""
    Pushforward(DM::DataModel, X::AbstractVector, θ::AbstractVector)
Calculates the push-forward of a vector `X` from the parameter manifold to the data space.
"""
Pushforward(DM::DataModel, X::AbstractVector, θ::AbstractVector{<:Number}) = EmbeddingMatrix(DM,θ) * X


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
function AICc(DM::AbstractDataModel, θ::AbstractVector{<:Number})
    (Npoints(DM) - length(θ) - 1) == 0 && throw("DataSet too small to appy AIC correction. Use AIC instead.")
    AIC(DM,θ) + (2length(θ)^2 + 2length(θ)) / (Npoints(DM) - length(θ) - 1)
end
AICc(DM::DataModel) = AICc(DM,MLE(DM))

"""
    BIC(DM::DataModel, θ::AbstractVector) -> Real
Calculates the Bayesian Information Criterion given a parameter configuration ``\\theta`` defined by ``\\mathrm{BIC} = \\mathrm{ln}(N) \\cdot \\mathrm{length}(\\theta) -2 \\, \\ell(\\mathrm{data} \\, | \\, \\theta)`` where ``N`` is the number of data points.
"""
BIC(DM::AbstractDataModel, θ::AbstractVector{<:Number}) = length(θ)*log(Npoints(DM)) - 2loglikelihood(DM,θ)
BIC(DM::DataModel) = BIC(DM,MLE(DM))


"""
    ModelComparison(DM1::AbstractDataModel, DM2::AbstractDataModel) -> Tuple{Int,Real}
Compares the AICc values of both models at best fit and estimates probability that one model is more likely than the other.
First entry of tuple returns which model is more likely to be correct (1 or 2) whereas the second entry returns the ratio of probabilities.
"""
function ModelComparison(DM1::AbstractDataModel, DM2::AbstractDataModel)
    !(ydata(DM1) == ydata(DM2) && xdata(DM1) == xdata(DM2) && InvCov(DM1) == InvCov(DM2)) && throw("Not comparing against same data!")
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
    BitArray(J1[:,i] == J2[:,i]  for i in 1:size(J1,2))
end

"""
    IsLinear(DM::DataModel) -> Bool
Checks whether the `model(x,θ)` function is linear with respect to all of its parameters ``\\theta \\in \\mathcal{M}``.
A componentwise check can be attained via the method `IsLinearParameter(DM)`.
"""
IsLinear(DM::AbstractDataModel) = all(IsLinearParameter(DM))

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
    CF = ConfVol(Confnum);      mle = MLEinPlane(DM,PL; tol=1e-8);      model = Predictor(DM)
    planarmod(x,p::AbstractVector{<:Number}) = model(x, PlaneCoordinates(PL,p))
    Test(x::Real) = ChisqCDF(pdim(DM), abs(2(LogLikeMLE(DM) - loglikelihood(Data(DM),planarmod, mle + [x,0.])))) - CF < 0.
    !Test(0.) && return false
    [LineSearch(Test,0.;tol=tol,maxiter=maxiter), 0.] + mle
end


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
    LinearCuboid(DM::AbstractDataModel, Confnum::Real=1.; Padding::Real=1/30, N::Int=200) -> HyperCube
Returns `HyperCube` which bounds the linearized confidence region of level `Confnum` for a `DataModel`.
"""
function LinearCuboid(DM::AbstractDataModel, Confnum::Real=1.; Padding::Real=1/30, N::Int=200)
    L = sqrt(InvChisqCDF(pdim(DM),ConfVol(Confnum))) .* cholesky(inv(Symmetric(FisherMetric(DM,MLE(DM))))).L
    C = [ConstructCube(Unpack([L * RotatedVector(α,dims[1],dims[2],pdim(DM)) for α in range(0,2pi,length=N)]);Padding=Padding) for dims in permutations(1:pdim(DM),2)]
    TranslateCube(Union(C), MLE(DM))
end

"""
    IntersectCube(DM::AbstractDataModel,Cube::HyperCube,Confnum::Real=1.; Dirs::Vector=[1,2,3], N::Int=31) -> Vector{Plane}
Returns a set of parallel 2D planes which intersect `Cube`. The planes span the directions corresponding to the basis vectors corresponding to the first two components of `Dirs`.
They are separated in the direction of the basis vector associated with the third component of `Dirs`.
The keyword `N` can be used to approximately control the number of planes which are returned.
This depends on whether more (or fewer) planes than `N` are necessary to cover the whole confidence region of level `Confnum`.
"""
function IntersectCube(DM::AbstractDataModel, Cube::HyperCube, Confnum::Real=1.; N::Int=31, Dirs::Vector{<:Int}=[1,2,3])
    (length(Dirs) != 3 || !allunique(Dirs) || !all(x->(1 ≤ x ≤ pdim(DM)),Dirs)) && throw("Invalid choice of Dirs: $Dirs.")
    PL = Plane(Center(Cube),BasisVector(Dirs[1],pdim(DM)),BasisVector(Dirs[2],pdim(DM)))
    width = CubeWidths(Cube)[Dirs[3]]
    IntersectRegion(DM,PL,width * BasisVector(Dirs[3],pdim(DM)), Confnum; N=N)
end

"""
    IntersectRegion(DM::AbstractDataModel,PL::Plane,v::Vector{<:Real},Confnum::Real=1.; N::Int=31) -> Vector{Plane}
Translates family of `N` planes which are translated approximately from `-v` to `+v` and intersect the confidence region of level `Confnum`.
If necessary, planes are removed or more planes added such that the maximal family of planes is found.
"""
function IntersectRegion(DM::AbstractDataModel, PL::Plane, v::Vector{<:Real},Confnum::Real=1.; N::Int=31)
    IsOnPlane(Plane(zeros(length(v)), PL.Vx, PL.Vy),v) && throw("Translation vector v = $v lies in given Plane $PL.")
    Planes = ParallelPlanes(PL, v, range(-0.5,0.5,length=N))
    AntiPrune(DM,Prune(DM,Planes,Confnum),Confnum)
end


function GenerateEmbeddedBoundary(DM::AbstractDataModel, PL::Plane, Confnum::Real=1.; tol::Real=1e-8, Auto::Bool=false,
                                Boundaries::Union{Function,Nothing}=nothing, meth::OrdinaryDiffEqAlgorithm=Tsit5(), mfd::Bool=false)
    GenerateBoundary(DM, PL, FindConfBoundaryOnPlane(DM, PL, Confnum; tol=tol); tol=tol, Boundaries=Boundaries, meth=meth, mfd=mfd, Auto=Auto)
end

"""
    MincedBoundaries(DM::AbstractDataModel, Planes::Vector{<:Plane}, Confnum::Real=1.; tol::Real=1e-9, Auto::Bool=false, meth=Tsit5(), mfd::Bool=false)
Intersects the confidence boundary of level `Confnum` with `Planes` and computes `ODESolution`s which parametrize this intersection.
"""
function MincedBoundaries(DM::AbstractDataModel, Planes::Vector{<:Plane}, Confnum::Real=1.; tol::Real=1e-8, Auto::Bool=false,
                        Boundaries::Union{Function,Nothing}=nothing, meth::OrdinaryDiffEqAlgorithm=Tsit5(), mfd::Bool=false, parallel::Bool=false)
    Map = parallel ? pmap : map
    Map(X->GenerateEmbeddedBoundary(DM, X, Confnum; tol=tol, Boundaries=Boundaries, meth=meth, mfd=mfd, Auto=Auto), Planes)
end
