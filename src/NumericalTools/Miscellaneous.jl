


GetH(x) = (suff(x) == BigFloat) ? convert(BigFloat,exp10(-precision(BigFloat)/10)) : 1e-6

# import DerivableFunctions: suff
suff(x::DataFrame) = suff(x[1,1])
suff(x::T) where T<:Measurement = T

floatify(x::AbstractArray{<:AbstractFloat}) = x;   floatify(x::AbstractArray) = float.(x)
floatify(x::AbstractFloat) = x;                     floatify(x::Number) = float(x)
floatify(x) = float.(x)


"""
    Unpack(Z::AbstractVector{S}) where S <: Union{AbstractVector,Tuple} -> Matrix
Converts vector of vectors to a matrix whose n-th column corresponds to the n-th component of the inner vectors.
"""
@inline function Unpack(Z::AbstractVector{S}) where S <: Union{AbstractVector{<:Number},Tuple}
    N = length(Z);      M = length(Z[1])
    A = Array{suff(Z)}(undef, N, M)
    @inbounds for i in Base.OneTo(N)
        for j in Base.OneTo(M)
            A[i,j] = Z[i][j]
        end
    end;    A
end
Unpack(Z::AbstractVector{<:Number}) = Z

Unwind(M::AbstractMatrix{<:Number}) = Unwind(collect(eachrow(M)))
Unwind(X::AbstractVector{<:AbstractVector{<:Number}}) = reduce(vcat, X)
Unwind(X::AbstractVector{<:Number}) = X


function Windup(X::AbstractVector{<:Union{Number,Missing}}, n::Int)
    @boundscheck @assert length(X) % n == 0 "Got length(X)=$(length(X)) and n=$n"
    n < 2 ? X : collect(Iterators.partition(X, n))
end

UnpackWindup(X::AbstractVector{<:Union{Number,Missing}}, dim::Int) = (@assert length(X)%dim==0;  permutedims(reshape(X, (dim,:))))


ToCols(M::AbstractMatrix) = Tuple(view(M,:,i) for i in 1:size(M,2))


ValToBool(x::Val{true}) = true
ValToBool(x::Val{false}) = false


function GetMethod(tol::Real)
    if tol > 1e-8
        Tsit5()
    elseif tol < 1e-11
        Vern9()
    else
        Vern7()
    end
end

# For ODE-based models, lower order methods in integral curve generation perform better.
function GetBoundaryMethod(tol::Real, DM::AbstractDataModel)
    if IsDEbased(DM)
        tol > 1e-10 ? BS3() : Tsit5()
    else
        GetMethod(tol)
    end
end
GetBoundaryMethod(tol::Real) = GetMethod(tol)


# Check for length
# PromoteStatic(X::SArray, inplace::Bool=true) = inplace ? _PromoteMutable(X) : X
PromoteStatic(X::AbstractArray{<:BigFloat}, inplace::Bool=true) = X
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
ConfAlpha(n::Real; kwargs...) = 1 - ConfVol(n; kwargs...)

"""
    ConfVol(n::Real)
Probability volume contained in a confidence interval of level n⋅σ where σ is the standard deviation of a normal distribution.
"""
function ConfVol(n::Real; verbose::Bool=true, kwargs...)
    if abs(n) ≤ 8
        erf(n / sqrt(2))
    else
        verbose && @info "ConfVol: Float64 precision insufficient for n=$n. Returning BigFloat instead."
        ConfVol(BigFloat(n))
    end
end
ConfVol(n::BigFloat; kwargs...) = erf(n / sqrt(BigFloat(2)))

InvConfVol(q::Real; kwargs...) = sqrt(2) * erfinv(q)
InvConfVol(x::BigFloat; tol::Real=GetH(x)) = invert(ConfVol, x; tol=tol)

ChisqCDF(k::Int, x::Real) = gamma_inc(k/2., x/2., 0)[1]
# ChisqCDF(k::Int, x::Real) = cdf(Chisq(k), x)
ChisqCDF(k::Int, x::BigFloat) = gamma_inc(BigFloat(k)/2., x/2., 0)[1]

InvChisqCDF(k::Int, p::Real; kwargs...) = 2gamma_inc_inv(k/2., p, 1-p)
InvChisqCDF(k::Int, p::BigFloat; tol::Real=GetH(p)) = invert(x->ChisqCDF(k, x), p; tol=tol)


InnerProduct(Mat::AbstractMatrix, Y::AbstractVector) = transpose(Y) * Mat * Y
# InnerProduct(Mat::PDMats.PDMat, Y::AbstractVector) = (R = Mat.chol.U * Y;  dot(R,R))

InnerProductV(Mat::AbstractMatrix, Y::AbstractVector) = @tullio Res := Y[i] * Mat[i,j] * Y[j]
InnerProductV(Mat::Diagonal, Y::AbstractVector) = @tullio Res := Mat.diag[j] * Y[j]^2


import Base.==
==(DS1::DataSet, DS2::DataSet) = xdata(DS1) ≈ xdata(DS2) && ydata(DS1) ≈ ydata(DS2) && yInvCov(DS1) ≈ yInvCov(DS2)
==(DS1::DataSetExact, DS2::DataSet) = DS2 ≈ DS1
function ==(DS1::DataSet, DS2::DataSetExact)
    if !(xdist(DS2) isa InformationGeometry.Dirac && ydist(DS2) isa MvNormal)
        return false
    elseif xdata(DS1) ≈ xdata(DS2) && ydata(DS1) ≈ ydata(DS2) && yInvCov(DS1) ≈ yInvCov(DS2)
        return true
    else
        false
    end
end
==(DS1::AbstractDataSet, DS2::AbstractDataSet) = xdist(DS1) == xdist(DS2) && ydist(DS1) == ydist(DS2)


function ==(DM1::AbstractDataModel, DM2::AbstractDataModel)
    Data(DM1) != Data(DM2) && return false
    pdim(DM1) != pdim(DM2) && return false
    z1, z2 = MLE(DM1) + rand(length(MLE(DM1))), MLE(DM1) + rand(length(MLE(DM1)))
    !(EmbeddingMap(DM1, z1) ≈ EmbeddingMap(DM2, z1) && EmbeddingMap(DM1, z2) ≈ EmbeddingMap(DM2, z2)) && return false
    !(EmbeddingMatrix(DM1, z1) ≈ EmbeddingMatrix(DM2, z1) && EmbeddingMatrix(DM1, z2) ≈ EmbeddingMatrix(DM2, z2)) && return false
    return true
end


"""
    GetArgSize(model::ModelOrFunction; max::Int=100)
Returns tuple `(xdim,pdim)` associated with the method `model(x,p)`.
"""
GetArgSize(model::Function; max::Int=100) = GetArgSize(model, Val(MaximalNumberOfArguments(model) > 2); max=max)
function GetArgSize(model::Function, inplace::Val{false}; max::Int=100)
    try return (1, GetArgLength(p->model(rand(),p); max=max)) catch; end
    for i in 2:max
        try return (i,GetArgLength(p->model(rand(i),p); max=max)) catch; end
    end;    throw("Wasn't able to find config for max=$max.")
end
function GetArgSize(model!::Function, inplace::Val{true}; max::Int=100)
    try return (1, GetArgLength((Res,p)->model!(Res,rand(),p); max=max)) catch; end
    for i in 2:max
        try return (i,GetArgLength((Res,p)->model!(Res,rand(i),p); max=max)) catch; end
    end;    throw("Wasn't able to find config for max=$max.")
end
GetArgSize(model::ModelMap; max::Int=100) = (model.xyp[1], model.xyp[3])


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

BlockMatrix(As::AbstractVector{<:AbstractMatrix}) = BlockMatrix(As...)
BlockMatrix(A::AbstractMatrix, B::AbstractMatrix, args...) = BlockMatrix(BlockMatrix(A,B), args...)
