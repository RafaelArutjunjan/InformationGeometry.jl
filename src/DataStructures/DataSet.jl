

"""
The `DataSet` type is a versatile container for storing data. Typically, it is constructed by passing it three vectors `x`, `y`, `sigma` where the components of `sigma` quantify the standard deviation associated with each y-value.
Alternatively, a full covariance matrix can be supplied for the `ydata` instead of a vector of standard deviations. The contents of a `DataSet` `DS` can later be accessed via `xdata(DS)`, `ydata(DS)`, `ysigma(DS)`.

Examples:

In the simplest case, where all data points are mutually independent and have a single ``x``-component and a single ``y``-component each, a `DataSet` consisting of four points can be constructed via
```julia
DataSet([1,2,3,4], [4,5,6.5,7.8], [0.5,0.45,0.6,0.8])
```
or alternatively by
```julia
using LinearAlgebra
DataSet([1,2,3,4], [4,5,6.5,7.8], Diagonal([0.5,0.45,0.6,0.8].^2))
```
where the diagonal covariance matrix in the second line is equivalent to the vector of standard deviations supplied in the first line.

For measurements with multiple components, it is also possible to enter them as a `Matrix` where the columns correspond to the respective components.
```julia
DataSet([0, 0.5, 1], [1 100; 2 103; 3 108], [0.5 8; 0.4 5; 0.6 10])
```
Note that if the uncertainty matrix is square, it may be falsely interpreted as a covariance matrix instead of as the columnwise specification of standard deviations.

More generally, if a dataset consists of ``N`` points where each ``x``-value has ``n`` many components and each ``y``-value has ``m`` many components, this can be specified to the `DataSet` constructor via a tuple ``(N,n,m)`` in addition to the vectors `x`, `y` and the covariance matrix.
For example:
```julia
X = [0.9, 1.0, 1.1, 1.9, 2.0, 2.1, 2.9, 3.0, 3.1, 3.9, 4.0, 4.1]
Y = [1.0, 5.0, 4.0, 8.0, 9.0, 13.0, 16.0, 20.0]
Cov = Diagonal([2.0, 4.0, 2.0, 4.0, 2.0, 4.0, 2.0, 4.0])
dims = (4, 3, 2)
DS = DataSet(X, Y, Cov, dims)
```
In this case, `X` is a vector consisting of the concatenated x-values (with 3 components each) for 4 different data points.
The values of `Y` are the corresponding concatenated y-values (with 2 components each) of said 4 data points. Clearly, the covariance matrix must therefore be a positive-definite ``(m \\cdot N) \\times (m \\cdot N)`` matrix.
"""
struct DataSet <: AbstractFixedUncertaintyDataSet
    x::AbstractVector
    y::AbstractVector
    InvCov::AbstractMatrix
    dims::Tuple{Int,Int,Int}
    logdetInvCov::Real
    WoundX::Union{AbstractVector,Nothing}
    xnames::AbstractVector{Symbol}
    ynames::AbstractVector{Symbol}
    name::Symbol
    DataSet(df::DataFrame; kwargs...) = DataSet(Matrix(df); xnames=[names(df)[1]], ynames=[names(df)[1]], kwargs...)
    DataSet(df::AbstractMatrix; kwargs...) = size(df, 2) ≤ 3 ? DataSet(ToCols(df)...; kwargs...) : throw("Unclear dimensions of input $df.")
    function DataSet(Xdf::DataFrame, Ydf::DataFrame, sigma::Union{Real,DataFrame}=1.0, args...; kwargs...)
        DataSet(Matrix(Xdf), Matrix(Ydf), (sigma isa Real ? sigma : Matrix(sigma)); xnames=names(Xdf), ynames=names(Ydf), kwargs...)
    end
    function DataSet(x::AbstractArray, y::AbstractArray; kwargs...)
        @info "No uncertainties in the y-values were specified for given DataSet, assuming σ=1 for all y's."
        DataSet(x, y, 1.0; kwargs...)
    end
    DataSet(x::AbstractArray, y::AbstractArray, allsigmas::Real; kwargs...) = DataSet(x, y, allsigmas*ones(length(y)*length(y[1])); kwargs...)
    # Also make a fancy version for DataFrames that infers the variable names?
    function DataSet(X::AbstractArray, Y::AbstractArray, Σ_y::AbstractArray; kwargs...)
        size(X,1) != size(Y,1) && throw("Inconsistent number of x-values and y-values given: $(size(X,1)) != $(size(Y,1)). Specify a tuple (Npoints, xdim, ydim) in the DataSet constructor.")
        # If Σ_y not a square matrix, assume each column is vector of standard deviations associated with y:
        Σ_y = (size(Σ_y,1) == size(Σ_y,2) && size(Y,1) != size(Y,2)) ? Σ_y : Unwind(Σ_y)
        #Σ_y = size(Σ_y,1) != size(Σ_y,2) ? Unwind(Σ_y) : Σ_y
        DataSet(Unwind(X), Unwind(Y), Σ_y, (size(X,1), ConsistentElDims(X), ConsistentElDims(Y)); kwargs...)
    end
    DataSet(x::AbstractArray{<:Number}, y::AbstractArray{<:Measurement}, args...; kwargs...) = DataSet(x, Measurements.value.(y), Measurements.uncertainty.(y), args...; kwargs...)
    ####### Only looking at sigma from here on out
    function DataSet(x::AbstractVector, y::AbstractVector, sigma::AbstractVector, dims::Tuple{Int,Int,Int}; kwargs...)
        Sigma = Unwind(sigma)
        DataSet(Unwind(x), Unwind(y), Sigma, Diagonal([Sigma[i]^(-2) for i in eachindex(Sigma)]), dims; kwargs...)
    end
    DataSet(x::AbstractVector, y::AbstractVector, Σ::AbstractMatrix, dims::Tuple{Int,Int,Int}; kwargs...) = DataSet(Unwind(x), Unwind(y), Σ, inv(Σ), dims; kwargs...)
    function DataSet(x::AbstractVector{<:Number},y::AbstractVector{<:Number},sigma::AbstractArray{<:Number},InvCov::AbstractMatrix{<:Number},dims::Tuple{Int,Int,Int}; kwargs...)
        !all(x->(x > 0), dims) && throw("Not all dims > 0: $dims.")
        !(Npoints(dims) == Int(length(x)/xdim(dims)) == Int(length(y)/ydim(dims)) == Int(size(InvCov,1)/ydim(dims))) && throw("Inconsistent input dimensions.")
        x = floatify(x);  y = floatify(y);  InvCov = HealthyCovariance(InvCov)
        if xdim(dims) == 1
            DataSet(x, y, InvCov, dims, logdet(InvCov), nothing; kwargs...)
        else
            # return new(x,y,InvCov,dims,logdet(InvCov),collect(Iterators.partition(x,xdim(dims))))
            DataSet(x, y, InvCov, dims, logdet(InvCov), [SVector{xdim(dims)}(Z) for Z in Windup(x,xdim(dims))]; kwargs...)
        end
    end
    function DataSet(x::AbstractVector, y::AbstractVector, InvCov::AbstractMatrix, dims::Tuple{Int,Int,Int}, logdetInvCov::Real, WoundX::Union{AbstractVector,Nothing};
                        xnames::AbstractVector{<:StringOrSymb}=CreateSymbolNames(xdim(dims),"x"), ynames::AbstractVector{<:StringOrSymb}=CreateSymbolNames(ydim(dims),"y"), name::StringOrSymb=Symbol(), kwargs...)
        @assert length(xnames) == xdim(dims) && length(ynames) == ydim(dims)
        DataSet(x, y, InvCov, dims, logdetInvCov, WoundX, xnames, ynames, name; kwargs...)
    end
    function DataSet(x::AbstractVector, y::AbstractVector, InvCov::AbstractMatrix, dims::Tuple{Int,Int,Int},
                            logdetInvCov::Real, WoundX::Union{AbstractVector,Nothing}, xnames::AbstractVector{<:StringOrSymb}, ynames::AbstractVector{<:StringOrSymb}, name::StringOrSymb=Symbol())
        new(x, y, InvCov, dims, logdetInvCov, WoundX, Symbol.(xnames), Symbol.(ynames), Symbol(name))
    end
end

function (::Type{T})(DS::DataSet; kwargs...) where T<:Number
	NewInvCov = T.(yInvCov(DS))
    DataSet(T.(xdata(DS)), T.(ydata(DS)), NewInvCov, dims(DS), logdet(NewInvCov), [SVector{xdim(dims(DS))}(Z) for Z in Windup(T.(xdata(DS)),xdim(dims(DS)))]; xnames=Xnames(DS), ynames=Ynames(DS), name=name(DS), kwargs...)
end


# For SciMLBase.remake
DataSet(;
x::AbstractVector=[0.],
y::AbstractVector=[0.],
InvCov::AbstractMatrix=Diagonal([1.]),
dims::Tuple{Int,Int,Int}=(1,1,1),
logdetInvCov::Real=-Inf,
WoundX::Union{AbstractVector,Nothing}=nothing,
xnames::AbstractVector{<:StringOrSymb}=[:x],
ynames::AbstractVector{<:StringOrSymb}=[:y],
name::StringOrSymb=Symbol()) = DataSet(x, y, InvCov, dims, logdetInvCov, WoundX, xnames, ynames, name)

# Specialized methods for DataSet
dims(DS::DataSet) = DS.dims
xdata(DS::DataSet) = DS.x
ydata(DS::DataSet) = DS.y

ysigma(DS::DataSet) = _ysigma_fromInvCov(DS, yInvCov(DS))
_ysigma_fromInvCov(DS::AbstractDataSet, M::AbstractSparseMatrix) = inv(convert(Matrix, M)) |> _TryVectorize
_ysigma_fromInvCov(DS::AbstractDataSet, M::AbstractMatrix) = inv(M) |> _TryVectorize
_TryVectorize(M::AbstractMatrix) = isdiag(M) ? sqrt.(Diagonal(M).diag) : M
_TryVectorize(D::Diagonal) = sqrt.(D.diag)

xsigma(DS::DataSet) = zeros(Npoints(DS)*xdim(DS))
HasXerror(DS::DataSet) = false

yInvCov(DS::DataSet) = DS.InvCov


WoundX(DS::DataSet) = _WoundX(DS, DS.WoundX)

logdetInvCov(DS::DataSet) = DS.logdetInvCov

xnames(DS::DataSet) = Xnames(DS) .|> string
ynames(DS::DataSet) = Ynames(DS) .|> string

Xnames(DS::DataSet) = DS.xnames
Ynames(DS::DataSet) = DS.ynames

name(DS::DataSet) = DS.name

