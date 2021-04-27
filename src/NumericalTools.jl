

GetH(x) = (suff(x) == BigFloat) ? convert(BigFloat,10^(-precision(BigFloat)/10)) : 1e-6

"""
    suff(x) -> Type
If `x` stores BigFloats, `suff` returns BigFloat, else `suff` returns `Float64`.
"""
suff(x::BigFloat) = BigFloat
suff(x::Float32) = Float32
# suff(x::Float16) = Float16
suff(x::Real) = Float64
suff(x::Num) = Num
suff(x::Complex) = real(x)
suff(x::Union{AbstractArray,AbstractRange}) = suff(x[1])
suff(x::DataFrame) = suff(x[1,1])
suff(x::Tuple) = suff(x...)
suff(args...) = suff(promote(args...)[1])


"""
    MaximalNumberOfArguments(F::Function) -> Int
Infers argument structure of given function, i.e. whether it is of the form `F(x)` or `F(x,y)` or `F(x,y,z)` etc. and returns maximal number of accepted arguments of all overloads of `F` as integer.
"""
MaximalNumberOfArguments(F::Function) = maximum([length(Base.unwrap_unionall(m.sig).parameters)-1 for m in methods(F)])


"""
    Unpack(Z::Vector{S}) where S <: Union{Vector,Tuple} -> Matrix
Converts vector of vectors to a matrix whose n-th column corresponds to the n-th component of the inner vectors.
"""
@inline function Unpack(Z::AbstractVector{S}) where S <: Union{AbstractVector{<:Number},Tuple,AbstractRange}
    N = length(Z);      M = length(Z[1])
    A = Array{suff(Z)}(undef,N,M)
    for i in 1:N
        for j in 1:M
            A[i,j] = Z[i][j]
        end
    end;    A
end
Unpack(Z::AbstractVector{<:Number}) = Z

Unwind(M::AbstractMatrix{<:Number}) = Unwind(collect(eachrow(M)))
Unwind(X::AbstractVector{<:AbstractVector{<:Number}}) = reduce(vcat, X)
Unwind(X::AbstractVector{<:Number}) = X


Windup(X::AbstractVector{<:Number}, n::Int) = n < 2 ? X : [X[(1+(i-1)*n):(i*n)] for i in 1:Int(length(X)/n)]

ToCols(M::Matrix) = Tuple(M[:,i] for i in 1:size(M,2))


ValToBool(x::Val{true}) = true
ValToBool(x::Val{false}) = false


DomainSamples(Domain::Union{Tuple{Real,Real}, HyperCube}; N::Int=500) = DomainSamples(Domain, N)
DomainSamples(Cube::HyperCube, N::Int) = length(Cube) == 1 ? DomainSamples((Cube.L[1],Cube.U[1]), N) : throw("Domain not suitable.")
function DomainSamples(Domain::Tuple{Real,Real}, N::Int)
    @assert N > 2 && Domain[1] < Domain[2]
    range(Domain[1], Domain[2]; length=N) |> collect
end


function GetMethod(tol::Real)
    if tol > 1e-8
        Tsit5()
    elseif tol < 1e-11
        Vern9()
    else
        Vern7()
    end
end

# Check for length
PromoteStatic(X::AbstractArray, inplace::Bool=true) = length(X) > 90 ? X : PromoteStatic(X, Val(inplace))

# No checking for length
PromoteStatic(X::AbstractArray, mutable::Val{true}) = _PromoteMutable(X)
PromoteStatic(X::AbstractArray, mutable::Val{false}) = _PromoteStatic(X)

_PromoteMutable(X::AbstractVector, Length=length(X)) = MVector{Length}(X)
_PromoteMutable(X::AbstractArray, Size=size(X)) = MArray{Tuple{Size...}}(X)
_PromoteStatic(X::AbstractVector, Length=length(X)) = SVector{Length}(X)
_PromoteStatic(X::AbstractArray, Size=size(X)) = SArray{Tuple{Size...}}(X)


# Surely, this can be made more efficient?
SplitAfter(n::Int) = X->(X[1:n], X[n+1:end])


"""
    invert(F::Function, x::Number; tol::Real=GetH(x)) -> Real
Finds ``z`` such that ``F(z) = x`` to a tolerance of `tol` for continuous ``F`` using Roots.jl. Ideally, `F` should be monotone and there should only be one correct result.
"""
function invert(F::Function, x::Number, Domain::Tuple{<:Number,<:Number}=(zero(suff(x)), 1e4*one(suff(x)));
                    tol::Real=GetH(x), meth::Roots.AbstractUnivariateZeroMethod=Roots.Order1())
    @assert Domain[1] < Domain[2]
    try
        if meth isa Roots.AbstractNonBracketing
            find_zero(z-> F(z) - x, 0.5one(suff(x)), meth; xatol=tol)
        else
            find_zero(z-> F(z) - x, Domain, meth; xatol=tol)
        end
    catch err
        @warn "invert() errored: $(nameof(typeof(err))). Assuming result is bracketed by $Domain and falling back to Bisection-like method."
        find_zero(z-> F(z) - x, Domain, Roots.AlefeldPotraShi(); xatol=tol)
    end
end
# function invert(F::Function, x::Number; tol::Real=GetH(x)*100, meth::Roots.AbstractUnivariateZeroMethod=Order1())
#     find_zero(z-> F(z) - x, 0.8one(suff(x)), meth; xatol=tol)
# end


"""
    ConfAlpha(n::Real)
Probability volume outside of a confidence interval of level n⋅σ where σ is the standard deviation of a normal distribution.
"""
ConfAlpha(n::Real) = 1 - ConfVol(n)

"""
    ConfVol(n::Real)
Probability volume contained in a confidence interval of level n⋅σ where σ is the standard deviation of a normal distribution.
"""
function ConfVol(n::Real)
    if abs(n) ≤ 8
        return erf(n / sqrt(2))
    else
        println("ConfVol: Float64 precision not enough for n = $n. Returning BigFloat instead.")
        return ConfVol(BigFloat(n))
    end
end
ConfVol(n::BigFloat) = erf(n / sqrt(BigFloat(2)))

InvConfVol(q::Real; kwargs...) = sqrt(2) * erfinv(q)
InvConfVol(x::BigFloat; tol::Real=GetH(x)) = invert(ConfVol, x; tol=tol)

ChisqCDF(k::Int, x::Real) = gamma_inc(k/2., x/2., 0)[1]
# ChisqCDF(k::Int, x::Real) = cdf(Chisq(k), x)
ChisqCDF(k::Int, x::BigFloat) = gamma_inc(BigFloat(k)/2., x/2., 0)[1]

InvChisqCDF(k::Int, p::Real; kwargs...) = 2gamma_inc_inv(k/2., p, 1-p)
InvChisqCDF(k::Int, p::BigFloat; tol::Real=GetH(p)) = invert(x->ChisqCDF(k, x), p; tol=tol)




import Base.==
==(DS1::DataSet, DS2::DataSet) = xdata(DS1) == xdata(DS2) && ydata(DS1) == ydata(DS2) && ysigma(DS1) == ysigma(DS2)
==(DS2::DataSetExact, DS1::DataSet) = DS1 == DS2
function ==(DS1::DataSet, DS2::DataSetExact)
    if !(xdist(DS2) isa InformationGeometry.Dirac)
        return false
    elseif xdata(DS1) == xdata(DS2) && ydata(DS1) == ydata(DS2) && ysigma(DS1) == ysigma(DS2)
        return true
    else
        false
    end
end
==(DS1::AbstractDataSet, DS2::AbstractDataSet) = xdist(DS1) == xdist(DS2) && ydist(DS1) == ydist(DS2)



"""
    Integrate1D(F::Function, Cube::HyperCube; tol::Real=1e-14, FullSol::Bool=false, meth=nothing)
Integrates `F` over a one-dimensional domain specified via a `HyperCube` by rephrasing the integral as an ODE and using `DifferentialEquations.jl`.
"""
function Integrate1D(F::Function, Cube::HyperCube; tol::Real=1e-14, FullSol::Bool=false, meth=nothing)
    length(Cube) != 1 && throw(ArgumentError("Cube dim = $(length(Cube)) instead of 1"))
    Integrate1D(F,(Cube.L[1],Cube.U[1]); tol=tol,FullSol=FullSol,meth=meth)
end
Integrate1D(F::Function, Interval::AbstractVector{<:Number}; tol::Real=1e-14, FullSol::Bool=false, meth=nothing) = Integrate1D(F, Tuple(Interval); tol=tol, FullSol=FullSol, meth=meth)
function Integrate1D(F::Function, Interval::Tuple{<:Number,<:Number}; tol::Real=1e-14, FullSol::Bool=false, meth=nothing)
    Interval = float.(Interval)
    !(0. < tol < 1.) && throw("Integrate1D: tol unsuitable")
    Interval[1] > Interval[2] && throw(ArgumentError("Interval orientation wrong."))
    f(u,p,t) = F(t);    u0 = 0.
    if tol < 1e-15
        u0 = BigFloat(u0);        Interval = BigFloat.(Interval)
        meth = (meth == nothing) ? Vern9() : meth
    else
        meth = (meth == nothing) ? Tsit5() : meth
    end
    if FullSol
        return solve(ODEProblem(f,u0,Interval),meth; reltol=tol,abstol=tol)
    else
        return solve(ODEProblem(f,u0,Interval),meth; reltol=tol,abstol=tol,save_everystep=false,save_start=false,save_end=true).u[end]
    end
end

"""
    IntegrateND(F::Function,Cube::HyperCube; tol::Real=1e-12, WE::Bool=false, kwargs...)
Integrates the function `F` over `Cube` with the help of **HCubature.jl** to a tolerance of `tol`.
If `WE=true`, the result is returned as a `Measurement` which also contains the estimated error in the result.
"""
function IntegrateND(F::Function, Cube::HyperCube; tol::Real=1e-12, WE::Bool=false, kwargs...)
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
IntegrateND(F::Function, L::AbstractVector{<:Number}, U::AbstractVector{<:Number}; tol::Real=1e-12, WE::Bool=false, kwargs...) = IntegrateND(F,HyperCube(L,U); tol=tol, WE=WE, kwargs...)
IntegrateND(F::Function, Interval::Union{AbstractVector{<:Number},Tuple{<:Number,<:Number}}; tol::Real=1e-12, WE::Bool=false, kwargs...) = IntegrateND(F,HyperCube(Interval); tol=tol, WE=WE, kwargs...)


"""
    IntegrateOverConfidenceRegion(DM::AbstractDataModel, Domain::HyperCube, Confnum::Real, F::Function; N::Int=Int(1e5), WE::Bool=true, kwargs...)
Integrates a function `F` over the intersection of `Domain` and the confidence region of level `Confnum`.
"""
function IntegrateOverConfidenceRegion(DM::AbstractDataModel, Domain::HyperCube, Confnum::Real, F::Function; N::Int=Int(1e5), WE::Bool=true, kwargs...)
    @assert length(Domain) == pdim(DM)
    # Multiply F with characteristic function for confidence region
    Threshold = LogLikeMLE(DM) - 0.5InvChisqCDF(pdim(DM), ConfVol(Confnum))
    InsideRegion(X::AbstractVector{<:Number}) = loglikelihood(DM, X; kwargs...) > Threshold
    Integrand(X::AbstractVector{<:Number}) = InsideRegion(X) ? F(X) : zero(suff(X))
    # Use HCubature instead of MonteCarlo
    MonteCarloArea(Integrand, Domain, N; WE=WE)
end

"""
    IntegrateOverApproxConfidenceRegion(DM::AbstractDataModel, Domain::HyperCube, sol::AbstractODESolution, F::Function; N::Int=Int(1e5), WE::Bool=true, kwargs...)
    IntegrateOverApproxConfidenceRegion(DM::AbstractDataModel, Domain::HyperCube, Planes::Vector{<:Plane}, sols::Vector{<:AbstractODESolution}, F::Function; N::Int=Int(1e5), WE::Bool=true, kwargs...)
Integrates a function `F` over the intersection of `Domain` and the polygon defined by `sol`.
"""
function IntegrateOverApproxConfidenceRegion(DM::AbstractDataModel, Domain::HyperCube, sol::AbstractODESolution, F::Function; N::Int=Int(1e5), WE::Bool=true, kwargs...)
    @assert length(Domain) == pdim(DM) == length(sol.u[1]) == 2
    Integrand(X::AbstractVector{<:Number}) = ApproxInRegion(sol, X) ? F(X) : zero(suff(X))
    # Use HCubature instead of MonteCarlo
    MonteCarloArea(Integrand, Domain, N; WE=WE)
end

function IntegrateOverApproxConfidenceRegion(DM::AbstractDataModel, Domain::HyperCube, Tup::Tuple{<:Vector{<:Plane},<:Vector{<:AbstractODESolution}}, F::Function; N::Int=Int(1e5), WE::Bool=true, kwargs...)
    IntegrateOverApproxConfidenceRegion(DM, Domain, Tup[1], Tup[2], F; N=N, WE=WE, kwargs...)
end
function IntegrateOverApproxConfidenceRegion(DM::AbstractDataModel, Domain::HyperCube, Planes::Vector{<:Plane}, sols::Vector{<:AbstractODESolution}, F::Function; N::Int=Int(1e5), WE::Bool=true, kwargs...)
    @assert length(Domain) == pdim(DM) == ConsistentElDims(Planes)
    @assert length(Planes) == length(sols)
    Integrand(X::AbstractVector{<:Number}) = ApproxInRegion(Planes, sols, X) ? F(X) : zero(suff(X))
    # Use HCubature instead of MonteCarlo
    MonteCarloArea(Integrand, Domain, N; WE=WE)
end


"""
    LineSearch(Test::Function, start::Number=0.; tol::Real=8e-15, maxiter::Int=10000) -> Number
Finds real number `x` where the boolean-valued `Test(x::Number)` goes from `true` to `false`.
"""
function LineSearch(Test::Function, start::Number=0.; tol::Real=8e-15, maxiter::Int=10000)
    if ((suff(start) != BigFloat) && tol < 1e-15)
        println("LineSearch: start not BigFloat but tol=$tol. Promoting and continuing.")
        start = BigFloat(start)
    end
    if !Test(start)
        start += 1e-10
        println("LineSearch: Test(start) did not work, trying Test(start + 1e-10).")
        !Test(start) && throw(ArgumentError("LineSearch: Test not true for starting value."))
    end
    # For some weird reason, if the division by 4 is removed, the loop never terminates for BigFloat-valued "start"s - maybe the compiler erroneously tries to optimize the variable "stepsize" away or something?! (Julia version ≤ 1.6.0)
    stepsize = one(suff(start)) / 4.;       value = start
    for i in 1:maxiter
        if Test(value + stepsize) # inside
            value += stepsize
            # value - start > 2000. && throw("FindConfBoundary: Value larger than 2000.")
        else            #outside
            if stepsize < tol
                return value + stepsize
            end
            stepsize /= 5.
        end
    end
    throw("$maxiter iterations over. Value=$value, Stepsize=$stepsize")
end


function MonteCarloArea(Test::Function,Cube::HyperCube,N::Int=Int(1e7); WE::Bool=false)
    if WE
        return CubeVol(Cube) * MonteCarloRatioWE(Test,Cube,N)
    else
        return CubeVol(Cube) * MonteCarloRatio(Test,Cube,N)
    end
end
function MonteCarloRatio(Test::Function,Cube::HyperCube,N::Int=Int(1e7))
    (1/N)* @distributed (+) for i in 1:N
        Test(rand.(Uniform.(Cube.L,Cube.U)))
    end
end

function MonteCarloRatioWE(Test::Function,LU::HyperCube,N::Int=Int(1e7); chunksize::Int=Int(N/20))
    chunksize > N && error("chunksize > N")
    if N%chunksize != 0
        N += Int(N%chunksize + 1)
    end
    chunks = Int(N/chunksize)
    # Output not normalized by chunksize
    function CarloLoop(Test::Function,LU::HyperCube,chunksize::Int)
        tot = [rand.(Uniform.(LU.L,LU.U)) for i in 1:chunksize] .|> Test
        res = sum(tot)
        [res, sum(abs2, (tot .- (res/chunksize)))]
    end
    Tot = @distributed (+) for i in 1:chunks
        CarloLoop(Test,LU,chunksize)
    end
    measurement(Tot[1]/N, sqrt(1/((N-1)*N) * Tot[2]))
end


# # From Cuba.jl docs
# function CompactDomainTransform(F::ModelOrFunction, Cube::HyperCube)
#     (!all(x->isfinite(x),Cube.L) || !all(x->isfinite(x),Cube.U)) && throw("Not applicable.")
#     if length(Cube) == 1
#         W = Cube.U[1] - Cube.L[1]
#         return x -> W * F(Cube.L[1] + W * x)
#         # Use mul! or something like that?
#     else
#         W = CubeWidths(Cube);   V = prod(CubeWidths)
#         return x -> V * F(Cube.L + W * x)
#     end
# end
#
# function HalfInfiniteTransform(F::ModelOrFunction, Cube::HyperCube)
#     # integral over [a,∞]
#     if Cube.U[1] == Inf && isfinite(Cube.L[1])
#         return x -> (1-x)^-2 * F(Cube.L[1] + x/(1-x))
#     end
# end
#
# function InfiniteDomainTransform(F::ModelOrFunction, Cube::HyperCube)
#     if Cube.L[1] == -Inf && Cube.L[1] == Inf
#         return x -> F((2x - 1.)/((1 - x)*x)) * (2x^2 - 2y + 1) / ((1-x^2)*x^2)
#     end
# end


import LsqFit.curve_fit
function curve_fit(DM::AbstractDataModel, initial::AbstractVector{<:Number}=MLE(DM); tol::Real=1e-14, kwargs...)
    curve_fit(Data(DM), Predictor(DM), dPredictor(DM), initial; tol=tol, kwargs...)
end

function curve_fit(DS::AbstractDataSet, M::ModelMap, initial::AbstractVector{<:Number}=GetStartP(DS,M); tol::Real=1e-14, kwargs...)
    curve_fit(DS, M.Map, initial; tol=tol, lower=convert(Vector,M.Domain.L), upper=convert(Vector,M.Domain.U), kwargs...)
end

function curve_fit(DS::AbstractDataSet, M::ModelMap, dM::ModelOrFunction, initial::AbstractVector{<:Number}=GetStartP(DS,M); tol::Real=1e-14, kwargs...)
    curve_fit(DS, M.Map, dM, initial; tol=tol, lower=convert(Vector,M.Domain.L), upper=convert(Vector,M.Domain.U), kwargs...)
end

function curve_fit(DS::AbstractDataSet, model::Function, initial::AbstractVector{<:Number}=GetStartP(DS,model); tol::Real=1e-14, kwargs...)
    X = xdata(DS);  Y = ydata(DS);    LsqFit.check_data_health(X, Y)
    u = cholesky(InvCov(DS)).U
    f(p) = u * (EmbeddingMap(DS, model, p) - Y)
    p0 = convert(Vector, initial)
    R = LsqFit.OnceDifferentiable(f, p0, copy(f(p0)); inplace = false, autodiff = :forward)
    LsqFit.lmfit(R, p0, InvCov(DS); x_tol=tol, g_tol=tol, kwargs...)
end

function curve_fit(DS::AbstractDataSet, model::Function, dmodel::ModelOrFunction, initial::AbstractVector{<:Number}=GetStartP(DS,model); tol::Real=1e-14, kwargs...)
    X = xdata(DS);  Y = ydata(DS);    LsqFit.check_data_health(X, Y)
    u = cholesky(InvCov(DS)).U
    f(p) = u * (EmbeddingMap(DS, model, p) - Y)
    df(p) = u * EmbeddingMatrix(DS, dmodel, p)
    p0 = convert(Vector, initial)
    R = LsqFit.OnceDifferentiable(f, df, p0, copy(f(p0)); inplace = false)
    LsqFit.lmfit(R, p0, InvCov(DS); x_tol=tol, g_tol=tol, kwargs...)
end

function normalizedjac(M::AbstractMatrix{<:Number}, xlen::Int)
    M[:,1:xlen] .*= sqrt(size(M,1)/xlen -1.);    return M
end


TotalLeastSquares(DM::AbstractDataModel, args...; kwargs...) = TotalLeastSquares(Data(DM), Predictor(DM), args...; kwargs...)
"""
    TotalLeastSquares(DSE::DataSetExact, model::ModelOrFunction, initial::AbstractVector{<:Number}; tol::Real=1e-13, kwargs...) -> Vector
Experimental feature which takes into account uncertainties in x-values to improve the accuracy of the fit.
Returns concatenated vector of x-values and parameters. Assumes that the uncertainties in the x-values and y-values are normal, i.e. Gaussian!
"""
function TotalLeastSquares(DSE::DataSetExact, model::ModelOrFunction, initial::Union{Nothing,AbstractVector{<:Number}}=nothing; tol::Real=1e-13, rescale::Bool=true, kwargs...)
    # Improve starting values by fitting normally with simple least squares
    initial = curve_fit(DataSet(WoundX(DSE),Windup(ydata(DSE),ydim(DSE)),ysigma(DSE)), model, (initial == nothing ? GetStartP(DSE, model) : initial); tol=tol, kwargs...).param
    if xdist(DSE) isa InformationGeometry.Dirac
        println("xdist of given data is Dirac, can only use ordinary least squares.")
        return xdata(DSE), initial
    end

    plen = pdim(DSE,model);  xlen = Npoints(DSE) * xdim(DSE)
    function predictY(ξ)
        x = view(ξ, 1:xlen);        p = view(ξ, (xlen+1):length(ξ))
        # INPLACE EmbeddingMap!() would be great here!
        vcat(x, EmbeddingMap(DSE, model, p, Windup(x,xdim(DSE))))
    end
    u = cholesky(BlockMatrix(InvCov(xdist(DSE)),InvCov(ydist(DSE)))).U;    Ydata = vcat(xdata(DSE), ydata(DSE))
    f(p) = u * (predictY(p) - Ydata)
    dfnormalized(p) = u * normalizedjac(ForwardDiff.jacobian(predictY,p), xlen)
    df(p) = u * ForwardDiff.jacobian(predictY,p)
    p0 = vcat(xdata(DSE), initial)
    R = rescale ? LsqFit.OnceDifferentiable(f, dfnormalized, p0, copy(f(p0)); inplace = false) : LsqFit.OnceDifferentiable(f, df, p0, copy(f(p0)); inplace = false)
    fit = LsqFit.lmfit(R, p0, BlockMatrix(InvCov(xdist(DSE)), InvCov(ydist(DSE))); x_tol=tol, g_tol=tol, kwargs...)
    Windup(fit.param[1:xlen],xdim(DSE)), fit.param[xlen+1:end]
end


"""
    minimize(F::Function, start::Vector{<:Number}; tol::Real=1e-10, meth=BFGS(), autodiff::Bool=true, Full::Bool=false, kwargs...) -> Vector
Minimizes the input function using the given `start` using algorithms from `Optim.jl` specified via the keyword `meth`.
`autodiff=false` uses finite differencing and `Full=true` returns the full solution object instead of only the minimizing result.
Optionally, the search domain can be bounded by passing a suitable `HyperCube` object as the third argument.
"""
function minimize(F::Function, start::AbstractVector{<:Number}, Domain::Union{Nothing,HyperCube}=nothing; tol::Real=1e-10, meth::Optim.AbstractOptimizer=BFGS(), timeout::Real=200, autodiff::Bool=true, Full::Bool=false, kwargs...)
    !(F(start) isa Number) && throw("Given function must return scalar values, got $(typeof(F(start))) instead.")
    diffval = autodiff ? :forward : :finite
    Res = if Domain === nothing
        optimize(F, float.(start), meth, Optim.Options(g_tol=tol, time_limit=float(timeout)); autodiff=diffval, kwargs...)
    else
        start ∉ Domain && throw("Given starting value not in specified domain.")
        optimize(F, convert(Vector{Float64},Domain.L), convert(Vector{Float64},Domain.U), float.(start), meth, Optim.Options(g_tol=tol, time_limit=float(timeout)); autodiff=diffval, kwargs...)
    end
    Full ? Res : Optim.minimizer(Res)
end

"""
    RobustFit(DM::AbstractDataModel, start::Vector{<:Number}; tol::Real=1e-10, p::Real=1, kwargs...)
Uses `p`-Norm to judge distance on Dataspace as specified by the keyword.
"""
RobustFit(DM::AbstractDataModel, args...; kwargs...) = RobustFit(Data(DM), Predictor(DM), args...; kwargs...)
function RobustFit(DS::AbstractDataSet, M::ModelOrFunction, start::AbstractVector{<:Number}=GetStartP(DS,M), Domain::Union{Nothing,HyperCube}=(M isa ModelMap ? M.Domain : nothing); tol::Real=1e-10, p::Real=1, kwargs...)
    HalfSig = cholesky(InvCov(DS)).U
    F(x::AbstractVector) = norm(HalfSig * (ydata(DS) - EmbeddingMap(DS, M, x)), p)
    minimize(F, start, Domain; tol=tol, kwargs...)
end


"""
    GetArgSize(model::ModelOrFunction; max::Int=100)::Tuple{Int,Int}
Returns tuple `(xdim,pdim)` associated with the method `model(x,p)`.
"""
function GetArgSize(model::Function; max::Int=100)::Tuple{Int,Int}
    try         return (1, GetArgLength(p->model(1.,p); max=max))       catch; end
    for i in 2:(max + 1)
        plen = try      GetArgLength(p->model(ones(i),p); max=max)      catch; continue end
        i == (max + 1) ? throw("Wasn't able to find config.") : return (i, plen)
    end
end
GetArgSize(model::ModelMap; max::Int=100) = (model.xyp[1], model.xyp[3])


function GetArgLength(F::Function; max::Int=100)::Int
    max < 1 && throw("pdim: max = $max too small.")
    try     F(1.);  return 1    catch; end
    for i in 1:(max+1)
        try
            F(ones(i))
        catch y
            (isa(y, BoundsError) || isa(y, MethodError) || isa(y, DimensionMismatch) || isa(y, ArgumentError) || isa(y, AssertionError)) && continue
            println("pdim: Encountered error in specification of model function.");     rethrow()
        end
        i == (max + 1) ? throw(ArgumentError("pdim: Parameter space appears to have >$max dims. Aborting. Maybe wrong type of x was inserted?")) : return i
    end
end



normalize(x::AbstractVector{<:Number}, scaling::Float64=1.0) = (scaling / norm(x)) * x
function normalizeVF(u::AbstractVector{<:Number}, v::AbstractVector{<:Number}, scaling::Float64=1.0)
    newu = u;    newv = v
    for i in 1:length(u)
        factor = sqrt(u[i]^2 + v[i]^2)
        newu[i] = (scaling/factor)*u[i]
        newv[i] = (scaling/factor)*v[i]
    end
    newu, newv
end
function normalizeVF(u::AbstractVector{<:Number},v::AbstractVector{<:Number},PlanarCube::HyperCube,scaling::Float64=1.0)
    length(PlanarCube) != 2 && throw("normalizeVF: Cube not planar.")
    newu = u;    newv = v
    Widths = CubeWidths(PlanarCube) |> normalize
    for i in 1:length(u)
        factor = sqrt(u[i]^2 + v[i]^2)
        newu[i] = (scaling/factor)*u[i] * Widths[1]
        newv[i] = (scaling/factor)*v[i] * Widths[2]
    end
    newu, newv
end


"""
    BlockMatrix(M::AbstractMatrix, N::Int)
Returns matrix which contains `N` many blocks of the matrix `M` along its diagonal.
"""
function BlockMatrix(M::AbstractMatrix, N::Int)
    Res = zeros(size(M,1)*N,size(M,2)*N)
    for i in 1:N
        Res[((i-1)*size(M,1) + 1):(i*size(M,1)),((i-1)*size(M,1) + 1):(i*size(M,1))] = M
    end;    Res
end
BlockMatrix(M::Diagonal, N::Int) = Diagonal(repeat(M.diag, N))

"""
    BlockMatrix(A::AbstractMatrix, B::AbstractMatrix)
Constructs blockdiagonal matrix from `A` and `B`.
"""
function BlockMatrix(A::AbstractMatrix, B::AbstractMatrix)
    Res = zeros(suff(A), size(A,1)+size(B,1), size(A,2)+size(B,2))
    Res[1:size(A,1), 1:size(A,1)] = A
    Res[size(A,1)+1:end, size(A,1)+1:end] = B
    Res
end
BlockMatrix(A::Diagonal, B::Diagonal) = Diagonal(vcat(A.diag, B.diag))


BlockMatrix(A::AbstractMatrix, B::AbstractMatrix, args...) = BlockMatrix(BlockMatrix(A,B), args...)
