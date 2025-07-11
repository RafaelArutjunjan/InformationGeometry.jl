


GetH(x) = (suff(x) == BigFloat) ? convert(BigFloat,exp10(-precision(BigFloat)/10)) : 1e-6

# import DerivableFunctions: suff
suff(x::DataFrame) = suff(x[1,1])
suff(x::T) where T<:Measurement = T

floatify(x::AbstractArray{<:AbstractFloat}) = x;   floatify(x::AbstractArray) = float.(x)
floatify(x::AbstractFloat) = x;                     floatify(x::Number) = float(x)
floatify(x) = float.(x)

"""
    isloaded(Module::Symbol) -> Bool
Checks if module with name `Module` was already loaded at runtime.
"""
isloaded(Module::Symbol) = any(S->isequal(Module, Symbol(S)), values(Base.loaded_modules))

"""
    Unpack(Z::AbstractVector{S}) where S <: Union{AbstractVector,Tuple} -> Matrix
Converts vector of vectors to a matrix whose n-th column corresponds to the n-th component of the inner vectors.
"""
@inline function Unpack(Z::AbstractVector{S}) where S <: Union{AbstractVector{<:Number},Tuple}
    N = length(Z);      M = length(Z[1])
    A = Matrix{suff(Z)}(undef, N, M)
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


ToCols(M::AbstractMatrix) = Tuple(view(M,:,i) for i in axes(M,2))


ValToBool(x::Val{true}) = true
ValToBool(x::Val{false}) = false

negate!(x::Union{T, AbstractArray{T}}) where T<:Number = (x .*= -one(T))
negate(x::Union{T, AbstractArray{T}}) where T<:Number = -x
Negate(F::Function) = negate∘F
Negate!!(F!::Function) = (x, args...; kwargs...) -> (F!(x, args...; kwargs...);     negate!(x))
NegateBoth(F::Function) = MergeOneArgMethods(Negate(F), Negate!!(F))

function GetMethod(tol::Real)
    if tol > 1e-8
        AutoTsit5(Rosenbrock23())
    elseif tol < 1e-11
        AutoVern9(Rodas5())
    else
        AutoVern7(Rodas5())
    end
end

# For ODE-based models, lower order methods in integral curve generation perform better.
function GetBoundaryMethod(tol::Real, DM::Union{AbstractDataModel,Nothing}=nothing)
    # if IsDEbased(DM)
        tol > 1e-9 ? BS3() : Tsit5()
    # else
    #     GetMethod(tol)
    # end
end


# Check for length
# PromoteStatic(X::SArray, inplace::Bool=true) = inplace ? _PromoteMutable(X) : X
PromoteStatic(X::AbstractArray{<:BigFloat}, inplace::Bool=true) = X
PromoteStatic(X::Number, inplace::Bool=true) = X
PromoteStatic(X::AbstractArray, inplace::Bool=true) = length(X) > 90 ? X : PromoteStatic(X, Val(inplace))

# No checking for length
PromoteStatic(X::AbstractArray, mutable::Val{true}) = _PromoteMutable(X)
PromoteStatic(X::AbstractArray, mutable::Val{false}) = _PromoteStatic(X)

_PromoteMutable(X::AbstractVector, Length=length(X)) = MVector{Length}(X)
_PromoteMutable(X::AbstractArray, Size=size(X)) = MArray{Tuple{Size...}}(X)
_PromoteStatic(X::AbstractVector, Length=length(X)) = SVector{Length}(X)
_PromoteStatic(X::AbstractArray, Size=size(X)) = SArray{Tuple{Size...}}(X)

"""
    IndVec(X::AbstractVector{<:Bool})
Turn `Bool` vector into vector of integer indices.
"""
IndVec(X::AbstractVector{<:Bool}) = [i for i in eachindex(X) if X[i]]
IndVec(X::AbstractVector{<:Int}) = X


DeStatic(X::AbstractArray{T,N}) where T <: Number where N = convert(Array{T, N}, X)
DeStatic(X::Array{T,N}) where T <: Number where N = X


# Return type of DataSet without specialized parametrization
DataSetType(DS::AbstractDataSet) = typeof(DS)
DataSetType(DS::DataSet) = DataSet
DataSetType(DS::DataSetExact) = DataSetExact
DataSetType(DS::CompositeDataSet) = CompositeDataSet
DataSetType(DS::GeneralizedDataSet) = GeneralizedDataSet



# Surely, this can be made more efficient?
SplitAfter(n::Int) = X->(view(X,1:n), view(X,n+1:length(X)))


const DiffArray{N,D} = Union{AbstractArray{<:N, D}, DiffCache{<:AbstractArray{<:N, D}}}
const DiffVector{N<:Number} = DiffArray{N, 1}
const DiffMatrix{N<:Number} = DiffArray{N, 2}

## Extend get_tmp to three arguments and determine wether to get tmp based on second or third argument
UnrollCache(dc, u, v::Union{T,AbstractArray{T},AbstractArray{<:AbstractArray{T}}}) where T<:ForwardDiff.Dual = UnrollCache(dc, v)
UnrollCache(dc, u, v) = UnrollCache(dc, u)
UnrollCache(dc, u) = PreallocationTools.get_tmp(dc, u)
# Overload to do nothing if first arg not DiffCache
UnrollCache(dc::AbstractArray, u) = dc


"""
    invert(F::Function, x::Number; tol::Real=GetH(x)) -> Real
Finds ``z`` such that ``F(z) = x`` to a tolerance of `tol` for continuous ``F`` using Roots.jl. Ideally, `F` should be monotone and there should only be one correct result.
"""
function invert(F::Function, x::T, Domain::Tuple{<:Number,<:Number}=(zero(T), 1e4*one(T));
                    tol::Real=GetH(x), meth::Roots.AbstractUnivariateZeroMethod=Roots.Order1()) where T<:Number
    @assert Domain[1] < Domain[2] && isfinite(Domain[2])
    try
        if meth isa Roots.AbstractNonBracketing
            find_zero(z-> F(z) - x, 0.5one(T), meth; xatol=tol)
        else
            find_zero(z-> F(z) - x, Domain, meth; xatol=tol)
        end
    catch err
        @warn "invert() errored: $(nameof(typeof(err))). Assuming result is bracketed by $Domain and falling back to Bisection-like method."
        find_zero(z-> F(z) - x, Domain, Roots.AlefeldPotraShi(); xatol=tol)
    end
end
# function invert(F::Function, x::T; tol::Real=GetH(x)*100, meth::Roots.AbstractUnivariateZeroMethod=Order1()) where T<:Number
#     find_zero(z-> F(z) - x, 0.8one(T), meth; xatol=tol)
# end


"""
    ConfAlpha(n::Real)
Probability volume outside of a confidence interval of level n⋅σ where σ is the standard deviation of a normal distribution.
"""
ConfAlpha(n::Number; kwargs...) = 1 - ConfVol(n; kwargs...)

"""
    ConfVol(n::Real)
Probability volume contained in a confidence interval of level n⋅σ where σ is the standard deviation of a normal distribution.
"""
function ConfVol(n::Number; verbose::Bool=true, kwargs...)
    if abs(n) ≤ 8
        erf(n / sqrt(2))
    else
        verbose && @info "ConfVol: Float64 precision insufficient for n=$n. Returning BigFloat instead."
        ConfVol(BigFloat(n))
    end
end
ConfVol(n::BigFloat; kwargs...) = erf(n / sqrt(BigFloat(2)))
ConfVol(n::Int; kwargs...) = ConfVol(float(n); kwargs...)

InvConfVol(q::Number; kwargs...) = sqrt(2) * erfinv(q)
InvConfVol(x::BigFloat; tol::Real=GetH(x)) = invert(ConfVol, x; tol=tol)

ChisqCDF(k::Int, x::Int) = ChisqCDF(k, float(x))
ChisqCDF(k::Int, x::T) where T<:Number = gamma_inc(T(k)/2, x/2, 0)[1]
# ChisqCDF(k::Int, x::Real) = cdf(Chisq(k), x)
# ChisqCDF(k::Int, x::BigFloat) = gamma_inc(BigFloat(k)/2., x/2., 0)[1]

InvChisqCDF(k::Int, p::Int; kwargs...) = InvChisqCDF(k, floatify(p))
InvChisqCDF(k::Int, p::T; kwargs...) where T<:Number = 2gamma_inc_inv(T(k)/2, p, 1-p)
InvChisqCDF(k::Int, p::T; maxval::Real=1e4, Domain::Tuple{Real,Real}=(zero(T), maxval*one(T)), kwargs...) where T<:BigFloat = invert(x->ChisqCDF(k, x), p, Domain; kwargs...)



"""
    Sgn(x::Number) -> Number
Sign function with the right-continuous convention that `Sgn(0) = 1` which means that the derivative is formally defined as zero everywhere.
Most importantly, this helps avoid automatic differentiation issues for functions involving `Sgn` of the argument, where 
derivatives of functions involving `sgn` are discontinuously forced to zero at the origin.
"""
Sgn(x::T) where T<:Number = ifelse(x < 0, -one(T), one(T))
"""
    BiLog(x::Union{T, AbstractVector{T}}; C::Real=one(T)) where T<:Number
Computes bi-symmetric logarithm, which can also be applied to negative numbers
```math
BiLog(x) = \\sgn(x) \\sgn(C) \\cdot \\log(1 + |C| \\cdot |x|)
```
similar to the definition in https://doi.org/10.1088/0957-0233/24/2/027001
The constant `C` controls the slope of the bi-logarithm at zero.
The inverse transformation is given by [`BiExp`](@ref).
"""
BiLog(x::Union{T, AbstractArray{T}}; C::Real=one(T)) where T<:Number = @. Sgn(x) * Sgn(C) * log1p(abs(C)*abs(x))
"""
    BiExp(y::Union{T, AbstractVector{T}}; C::Real=one(T)) where T<:Number
Computes bi-symmetric exponential, which is the inverse transformation to [`BiLog`](@ref)
```math
BiExp(y) = \\sgn(y) \\cdot 1/C \\cdot (\\exp(|y|) - 1)
```
similar to the definition in https://doi.org/10.1088/0957-0233/24/2/027001
The constant `C` controls the slope of the bi-logarithm at zero, i.e. the bi-exponential has slope `1/C`.
"""
BiExp(y::Union{T, AbstractArray{T}}; C::Real=one(T)) where T<:Number = @. Sgn(y) * inv(C) * (exp(abs(y)) - one(T))
"""
    BiLog10(x::Union{T, AbstractVector{T}}; C::Real=one(T)) where T<:Number
Computes bi-symmetric logarithm of base 10, which can also be applied to negative numbers
```math
BiLog10(x) = \\sgn(x) \\sgn(C) \\cdot \\log_{10}(1 + |C| \\cdot |x|)
```
similar to the definition in https://doi.org/10.1088/0957-0233/24/2/027001
The constant `C` controls the slope of the bi-logarithm at zero.
The inverse transformation is given by [`BiExp10`](@ref).
"""
BiLog10(x::Union{T, AbstractArray{T}}; C::Real=one(T)) where T<:Number = @. Sgn(x) * Sgn(C) * log10(1 + abs(C)*abs(x))
"""
    BiExp10(y::Union{T, AbstractVector{T}}; C::Real=one(T)) where T<:Number
Computes bi-symmetric exponential of base 10, which is the inverse transformation to [`BiLog10`](@ref)
```math
BiExp10(y) = \\sgn(y) \\cdot 1/C \\cdot (\\exp10(|y|) - 1)
```
similar to the definition in https://doi.org/10.1088/0957-0233/24/2/027001
The constant `C` controls the slope of the bi-logarithm at zero, i.e. the bi-exponential has slope `1/C`.
"""
BiExp10(y::Union{T, AbstractArray{T}}; C::Real=one(T)) where T<:Number = @. Sgn(y) * inv(C) * (exp10(abs(y)) - one(T))

"""
    BiRoot(x::Union{T, AbstractVector{T}}, m::Real=2.0; C::Real=one(T)) where T<:Number
Computes bi-symmetric `m`-th root, which is the inverse transformation to [`BiPower`](@ref) in such a way that the slope is `C` at the origin:
```math
BiRoot_m(x) = \\sgn(x) \\sgn(C) \\cdot ((1 + m \\cdot |C| \\cdot |x|)^{1/m} -1)
```
The bi-symmetric root and power are inspired by the bi-symmetric logarithm in https://doi.org/10.1088/0957-0233/24/2/027001
"""
BiRoot(x::Union{T, AbstractArray{T}}, m::Real=2.0; C::Real=one(T)) where T<:Number = @. Sgn(x) * Sgn(C) * (exp(log1p(m * abs(C) * abs(x))/m) - one(T))
"""
    BiPower(y::Union{T, AbstractVector{T}}, m::Real=2.0; C::Real=one(T)) where T<:Number
Computes bi-symmetric `m`-th power, which is the inverse transformation to [`BiRoot`](@ref) in such a way that the slope of [`BiPower`](@ref) is `1/C` at the origin (i.e. the slope of the original [`BiRoot`](@ref) is `C`):
```math
BiPower_m(y) = \\sgn(x) \\cdot (m C)^{-1} \\cdot ((1 + |y|)^m - 1)
```
The bi-symmetric root and power are inspired by the bi-symmetric logarithm in https://doi.org/10.1088/0957-0233/24/2/027001
"""
BiPower(y::Union{T, AbstractArray{T}}, m::Real=2.0; C::Real=one(T)) where T<:Number = @. Sgn(y) * inv(m * C) * (exp(m * log1p(abs(y))) - one(T))


"""
    SoftAbs(x::Union{T, AbstractVector{T}}; eps::Real=1e-100) where T<:Number
Computes differentiable approximation of absolute value function `abs` as `sqrt(abs2(x) + eps)`.
"""
SoftAbs(x::Union{T, AbstractArray{T}}; eps::Real=1e-100) where T<:Number = @. sqrt(abs2(x) + eps)
"""
    SoftLog(x::Union{T, AbstractVector{T}}; eps::Real=1e-20) where T<:Number
Computes `log(x + eps)` to avoid `NaN` errors in automatic differentiation.
"""
SoftLog(x::Union{T, AbstractArray{T}}; eps::Real=1e-20) where T<:Number = @. log(x + eps)



import Base.==
==(DS1::DataSet, DS2::DataSet) = xdata(DS1) ≈ xdata(DS2) && ydata(DS1) ≈ ydata(DS2) && yInvCov(DS1) ≈ yInvCov(DS2)
==(DS1::DataSetExact, DS2::DataSet) = DS2 == DS1
function ==(DS1::DataSet, DS2::DataSetExact)
    if !(xdist(DS2) isa InformationGeometry.Dirac && ydist(DS2) isa MvNormal)
        return false
    elseif xdata(DS1) ≈ xdata(DS2) && ydata(DS1) ≈ ydata(DS2) && yInvCov(DS1) ≈ yInvCov(DS2)
        return true
    else
        false
    end
end
==(DS1::AbstractDataSet, DS2::AbstractDataSet) = xdist(DS1) ≈ xdist(DS2) && ydist(DS1) ≈ ydist(DS2)


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
GetArgSize(model::Function; max::Int=100) = GetArgSize(model, Val(isinplacemodel(model)); max=max)
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


# overload non-static out-of-place method
normalize(x::AbstractVector{<:Number}) = x ./ norm(x)
function normalizeVF(u::AbstractVector{<:Number}, v::AbstractVector{<:Number}, scaling::Float64=1.0)
    newu = u;    newv = v
    for i in eachindex(u)
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
    for i in eachindex(u)
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
function BlockMatrix(A::AbstractMatrix{T}, B::AbstractMatrix{S}) where {T<:Number, S<:Number}
    # Adopt eltype of first matrix instead of union
    Res = zeros(T, size(A,1)+size(B,1), size(A,2)+size(B,2))
    Res[1:size(A,1), 1:size(A,1)] .= A
    Res[size(A,1)+1:end, size(A,1)+1:end] .= B
    Res
end
BlockMatrix(A::Diagonal, B::Diagonal) = Diagonal(vcat(A.diag, B.diag))

BlockMatrix(As::AbstractVector{<:AbstractMatrix}) = reduce(BlockMatrix, As)
BlockMatrix(A::AbstractMatrix, B::AbstractMatrix, args...) = BlockMatrix(BlockMatrix(A,B), args...)



"""
    FindExtremalNeighboring(X::AbstractVector; Diff::Function=(x,y)->(y-x), Comparison=Base.:>)
Finds extremal changes between neighboring elements of array.
`Diff` is used to measure the change between two neighboring elements `x` and `y`.
`Comparison(NewDiff, OldDiff)` is used to establish an ordering and should return `true` if `NewDiff` is preferential to `OldDiff`.

# Examples
findmax(diff(X))[2] == FindExtremalNeighboring(X::AbstractVector; Comparison=Base.:>)[2]
findmin(diff(X))[2] == FindExtremalNeighboring(X::AbstractVector; Comparison=Base.:<)[2]
"""
function FindExtremalNeighboring(X::AbstractArray{T}; Diff::Function=(x::T,y::T)->(y-x), Comparison=Base.:>) where T<:Number
    maxchange = Diff(X[1],X[2])
    testchange = zero(T)
    ind = 1
    @inbounds for i in 2:length(X)-1
        testchange = Diff(X[i], X[i+1])
        if Comparison(testchange, maxchange)
            ind = i
            maxchange = testchange
        end
    end;    (maxchange, ind)
end

FindMinDiff(X::AbstractArray{<:Number}) = FindExtremalNeighboring(X; Comparison=Base.:<)
FindMaxDiff(X::AbstractArray{<:Number}) = FindExtremalNeighboring(X; Comparison=Base.:>)



"""
    ViewElements(Inds::Union{AbstractArray, Tuple, Colon, Int}) -> Function
Produces a function `X::AbstractArray{<:Number}->view(X, Inds)`` for viewing indices of an array wrapped inside struct for more informative stack traces.
"""
struct ViewElements{Inds} <: Function
    F::Function
    Inds::Union{AbstractArray, Tuple, Colon, Int}
    ViewElements(Inds::AbstractRange) = new{Inds}(X::AbstractArray{<:Number}->view(X, Inds), Inds)
    ViewElements(Inds::Int) = new{Inds}(X::AbstractArray{<:Number}->view(X, Inds), Inds)
    ViewElements(Inds::AbstractArray{<:Int}) = new{Tuple(Inds)}(X::AbstractArray{<:Number}->view(X, Inds), Inds)
    ViewElements(Inds::AbstractArray{<:Bool}) = ViewElements((1:length(Inds))[Inds])
    ViewElements(Inds::NTuple{N,<:Int}) where N = ViewElements(collect(Inds))
    ViewElements(Inds::Colon) = new{Inds}(identity, Inds)
    # could extend this to 2D views later by simply defining further constructors

    # Do not wrap any other and simply use anonymous view
    ViewElements(Inds) = X::AbstractArray{<:Number}->view(X, Inds)
end
(V::ViewElements)(args...; kwargs...) = V.F(args...; kwargs...)



function MergeOneArgMethods(::Nothing, Inplace!::Function, args...; OutIn::Tuple=DerivableFunctionsBase._GetArgLengthInPlace(Inplace!), J=nothing, kwargs...)
    FakeOutOfPlace(x::AbstractVector{T}) where T<:Number = (X = !isnothing(J) ? J : (rand(T, OutIn[1]...));   Inplace!(X,x);   X)
    MergeOneArgMethods(FakeOutOfPlace, Inplace!, args...; kwargs...)
 end
 function MergeOneArgMethods(OutOfPlace::Function, ::Nothing, args...; kwargs...)
    FakeInplace!(J::AbstractArray, x::AbstractVector) = copyto!(J, OutOfPlace(x))
    MergeOneArgMethods(OutOfPlace, FakeInplace!, args...; kwargs...)
 end
 """
    MergeOneArgMethods(OutOfPlace::Union{Nothing,Function}, Inplace!::Union{Nothing,Function}; J=nothing)
 Merges the `OutOfPlace` and `Inplace!` methods under a single `Function` so that `OutOfPlace(x)` is called whenever only a single arg is given and `Inplace!(J,x)`
 is used whenever the function is called with two arguments.
 
 If `nothing` is given for either function, an appropriate method is generated.
 """
 function MergeOneArgMethods(OutOfPlace::Function, Inplace!::Function)
    OverloadedMethod(x::AbstractVector; kwargs...) = OutOfPlace(x; kwargs...)
    OverloadedMethod(J::AbstractArray, x::AbstractVector; kwargs...) = Inplace!(J, x; kwargs...)
    # Allow for splitting tuples generated in composition with ∘
    OverloadedMethod(Tup::Tuple{<:AbstractArray, <:AbstractArray}; kwargs...) = OverloadedMethod(Tup[1], Tup[2]; kwargs...)
 end