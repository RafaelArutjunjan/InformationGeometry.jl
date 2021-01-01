

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

# Same as Reduction()
Unwind(X::AbstractVector{<:AbstractVector{<:Number}}) = reduce(vcat, X)
Unwind(X::AbstractVector{<:Number}) = X

Windup(X::AbstractVector{<:Number}, n::Int) = n < 2 ? X : [X[(1+(i-1)*n):(i*n)] for i in 1:Int(length(X)/n)]
# Windup(X::AbstractMatrix{<:Number}) = reduce(vcat, collect(eachrow(X)))

ToCols(M::Matrix) = Tuple(M[:,i] for i in 1:size(M,2))


"""
    invert(F::Function, x::Real; tol::Real=GetH(x)) -> Real
Finds ``z`` such that ``F(z) = x`` to a tolerance of `tol`. Ideally, F should be monotone and there should only be one correct result.
"""
function invert(F::Function, x::Real; tol::Real=GetH(x), meth::Roots.AbstractUnivariateZeroMethod=Order1())
    find_zero(z-> F(z) - x, one(suff(x)), meth; xatol=tol)
end


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
ChisqCDF(k::Int, x::BigFloat) = gamma_inc(BigFloat(k)/2., x/2., 0)[1]

InvChisqCDF(k::Int, p::Real; kwargs...) = 2gamma_inc_inv(k/2., p, 1-p)
InvChisqCDF(k::Int, p::BigFloat; tol::Real=GetH(p)) = invert(x->ChisqCDF(k, x), p; tol=tol)

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
    f(u,p,t) = F(t);    u0 = 0.
    if tol < 1e-15
        u0 = BigFloat(u0);        Interval = BigFloat.(Interval)
        meth = (meth == nothing) ? Vern9() : meth
    else
        meth = (meth == nothing) ? Tsit5() : meth
    end
    if fullSol
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
IntegrateND(F::Function, L::AbstractVector{<:Real}, U::AbstractVector{<:Real}; tol::Real=1e-12, WE::Bool=false, kwargs...) = IntegrateND(F,HyperCube(L,U); tol=tol, WE=WE, kwargs...)
IntegrateND(F::Function, Interval::Union{AbstractVector{<:Real},Tuple{<:Real,<:Real}}; tol::Real=1e-12, WE::Bool=false, kwargs...) = IntegrateND(F,HyperCube(Interval); tol=tol, WE=WE, kwargs...)



"""
    LineSearch(Test::Function, start::Real=0; tol::Real=8e-15, maxiter::Int=10000) -> Real
Finds real number `x` where the boolean-valued `Test(x::Real)` goes from `true` to `false`.
"""
function LineSearch(Test::Function, start::Real=0.; tol::Real=8e-15, maxiter::Int=10000)
    if ((suff(start) != BigFloat) && tol < 1e-15)
        println("LineSearch: start not BigFloat but tol=$tol. Promoting and continuing.")
        start = BigFloat(start)
    end
    !Test(start) && throw(ArgumentError("LineSearch: Test not true for starting value."))
    stepsize = one(suff(start))/4.;  value = start
    for i in 1:maxiter
        if Test(value + stepsize) # inside
            value += stepsize
            value - start > 200 && throw("FindConfBoundary: Value larger than 200.")
        else            #outside
            if stepsize < tol
                return value + stepsize
            end
            stepsize /= 5
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
        [res, sum((tot .- (res/chunksize)).^2)]
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
curve_fit(DM::AbstractDataModel,initial::AbstractVector{<:Number}=MLE(DM);tol::Real=6e-15,kwargs...) = curve_fit(Data(DM),Predictor(DM),initial;tol=tol,kwargs...)

function curve_fit(DS::AbstractDataSet, M::ModelMap, initial::AbstractVector{<:Number}=GetStartP(DS,M); tol::Real=6e-15, kwargs...)
    # Use bounds from HyperCube for curve_fit
    curve_fit(DS, M.Map, initial; tol=tol, lower=M.Domain.L, upper=M.Domain.U, kwargs...)
end

function curve_fit(DS::AbstractDataSet, model::Function, initial::AbstractVector{<:Number}=GetStartP(DS,model); tol::Real=6e-15, kwargs...)
    X = xdata(DS);  Y = ydata(DS)
    LsqFit.check_data_health(X, Y)
    u = cholesky(InvCov(DS)).U
    f(p) = u * (EmbeddingMap(DS, model, p) - Y)
    p0 = convert(Vector,initial);    r = f(p0)
    # R = OnceDifferentiable(f, p0, copy(r); inplace = false, autodiff = :finite)
    R = OnceDifferentiable(f, p0, copy(r); inplace = false, autodiff = :forward)
    LsqFit.lmfit(R, p0, InvCov(DS); x_tol=tol,g_tol=tol,kwargs...)
end


function normalizedjac(M::AbstractMatrix{<:Number}, xlen::Int=length(x))
    M[:,1:xlen] .*= sqrt(size(M,1)/xlen -1.);    return M
end
function curve_fit2(DSE::DataSetExact, model::ModelOrFunction, initial::AbstractVector{<:Number}=GetStartP(DSE,model); tol::Real=1e-13, kwargs...)
    isa(xdist(DSE),InformationGeometry.Dirac) && return curve_fit(DataSet(xdata(DSE),ydata(DSE),ysigma(DSE)), model, initial; tol=tol, kwargs...)
    plen = pdim(DSE,model);  xlen = Npoints(DSE) * xdim(DSE)
    function predictY(ξ)
        x = ξ[1:xlen];        p = ξ[xlen+1:end]
        # use performMap here instead
        vcat(x,reduce(vcat,map(z->model(z,p),x)))
    end
    # Get total inverse covariance matrix on the cartesian product space X^N \times Y^N = domain(P)
    u = cholesky(BlockMatrix(InvCov(xdist(DSE)),InvCov(ydist(DSE)))).U;    Ydata = vcat(xdata(DSE),ydata(DSE))
    f(p) = u * (predictY(p) - Ydata)
    df(p) = u * normalizedjac(ForwardDiff.jacobian(predictY,p),xlen)
    p0 = vcat(xdata(DSE),initial);    r = f(p0)
    R = OnceDifferentiable(f, df, p0, copy(r); inplace = false)
    LsqFit.lmfit(R, p0, invcov(P); x_tol=tol,g_tol=tol,kwargs...)
end

# function curve_fit(DS::AbstractDataSet,model::ModelOrFunction,initial::AbstractVector{<:Number}=rand(pdim(DS,model));tol::Real=6e-15,kwargs...)
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



normalize(x::AbstractVector{<:Real},scaling::Float64=1.0) = (scaling / norm(x)) * x
function normalizeVF(u::AbstractVector{<:Real},v::AbstractVector{<:Real},scaling::Float64=1.0)
    newu = u;    newv = v
    for i in 1:length(u)
        factor = sqrt(u[i]^2 + v[i]^2)
        newu[i] = (scaling/factor)*u[i]
        newv[i] = (scaling/factor)*v[i]
    end
    newu, newv
end
function normalizeVF(u::Vector{<:Real},v::Vector{<:Real},PlanarCube::HyperCube,scaling::Float64=1.0)
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
