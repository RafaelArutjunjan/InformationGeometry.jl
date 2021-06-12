

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
struct DataSet <: AbstractDataSet
    x::AbstractVector
    y::AbstractVector
    InvCov::AbstractMatrix
    dims::Tuple{Int,Int,Int}
    logdetInvCov::Real
    WoundX::Union{AbstractVector,Nothing}
    xnames::Vector{String}
    ynames::Vector{String}
    function DataSet(DF::Union{DataFrame,AbstractMatrix})
        size(DF,2) > 3 && throw("Unclear dimensions of input $DF.")
        DataSet(ToCols(Matrix(DF))...)
    end
    function DataSet(x::AbstractArray, y::AbstractArray)
        println("No uncertainties in the y-values were specified for given DataSet, assuming σ=1 for all y's.")
        DataSet(x, y, 1.0)
    end
    DataSet(x::AbstractArray, y::AbstractArray, allsigmas::Real) = DataSet(x, y, allsigmas*ones(length(y)*length(y[1])))
    # Also make a fancy version for DataFrames that infers the variable names?
    function DataSet(X::AbstractArray, Y::AbstractArray, Σ_y::AbstractArray)
        size(X,1) != size(Y,1) && throw("Inconsistent number of x-values and y-values given: $(size(X,1)) != $(size(Y,1)).")
        # If Σ_y not a square matrix, assume each column is vector of standard deviations associated with y:
        Σ_y = size(Σ_y,1) != size(Σ_y,2) ? Unwind(Σ_y) : Σ_y
        DataSet(Unwind(X), Unwind(Y), Σ_y, (size(X,1), ConsistentElDims(X), ConsistentElDims(Y)))
    end
    DataSet(x::AbstractVector{<:Number}, y::AbstractVector{<:Measurement}, args...) = DataSet(x,[y[i].val for i in 1:length(y)],[y[i].err for i in 1:length(y)], args...)
    ####### Only looking at sigma from here on out
    function DataSet(x::AbstractVector, y::AbstractVector, sigma::AbstractVector, dims::Tuple{Int,Int,Int})
        Sigma = Unwind(sigma)
        DataSet(Unwind(x), Unwind(y), Sigma, Diagonal([Sigma[i]^(-2) for i in 1:length(Sigma)]), dims)
    end
    DataSet(x::AbstractVector, y::AbstractVector, Σ::AbstractMatrix, dims::Tuple{Int,Int,Int}) = DataSet(Unwind(x), Unwind(y), Σ, inv(Σ), dims)
    function DataSet(x::AbstractVector{<:Number},y::AbstractVector{<:Number},sigma::AbstractArray{<:Number},InvCov::AbstractMatrix{<:Number},dims::Tuple{Int,Int,Int})
        !all(x->(x > 0), dims) && throw("Not all dims > 0: $dims.")
        !(Npoints(dims) == Int(length(x)/xdim(dims)) == Int(length(y)/ydim(dims)) == Int(size(InvCov,1)/ydim(dims))) && throw("Inconsistent input dimensions.")
        x = float.(x);  y = float.(y);  InvCov = HealthyCovariance(InvCov)
        if xdim(dims) == 1
            return DataSet(x, y, InvCov, dims, logdet(InvCov), nothing)
        else
            # return new(x,y,InvCov,dims,logdet(InvCov),collect(Iterators.partition(x,xdim(dims))))
            return DataSet(x, y, InvCov, dims, logdet(InvCov), [SVector{xdim(dims)}(Z) for Z in Windup(x,xdim(dims))])
        end
    end
    function DataSet(x::AbstractVector, y::AbstractVector, InvCov::AbstractMatrix, dims::Tuple{Int,Int,Int}, logdetInvCov::Real, WoundX::Union{AbstractVector,Nothing})
        DataSet(x, y, InvCov, dims, logdetInvCov, WoundX, CreateSymbolNames(xdim(dims),"x"), CreateSymbolNames(ydim(dims),"y"))
    end
    function DataSet(x::AbstractVector, y::AbstractVector, InvCov::AbstractMatrix, dims::Tuple{Int,Int,Int},
                            logdetInvCov::Real, WoundX::Union{AbstractVector,Nothing}, xnames::Vector{String}, ynames::Vector{String})
        new(x, y, InvCov, dims, logdetInvCov, WoundX, xnames, ynames)
    end
end

# For SciMLBase.remake
DataSet(;
x::AbstractVector=[0.],
y::AbstractVector=[0.],
InvCov::AbstractMatrix=Diagonal([1.]),
dims::Tuple{Int,Int,Int}=(1,1,1),
logdetInvCov::Real=-Inf,
WoundX::Union{AbstractVector,Nothing}=nothing,
xnames::Vector{String}=["x"],
ynames::Vector{String}=["y"]) = DataSet(x, y, InvCov, dims, logdetInvCov, WoundX, xnames, ynames)

# Specialized methods for DataSet
dims(DS::DataSet) = DS.dims
xdata(DS::DataSet) = DS.x
ydata(DS::DataSet) = DS.y
function ysigma(DS::DataSet)
    sig = !issparse(InvCov(DS)) ? inv(InvCov(DS)) : inv(convert(Matrix,InvCov(DS)))
    return isdiag(sig) ? sqrt.(Diagonal(sig).diag) : sig
end
xsigma(DS::DataSet) = zeros(Npoints(DS)*xdim(DS))

InvCov(DS::DataSet) = DS.InvCov
# WoundX(DS::DataSet) = xdim(DS) < 2 ? xdata(DS) : DS.WoundX
WoundX(DS::DataSet) = _WoundX(DS, DS.WoundX)
_WoundX(DS::DataSet, WoundX::Nothing) = xdata(DS)
_WoundX(DS::DataSet, WoundX::AbstractVector) = WoundX

logdetInvCov(DS::DataSet) = DS.logdetInvCov

xnames(DS::DataSet) = DS.xnames
ynames(DS::DataSet) = DS.ynames

function InformNames(DS::DataSet, xnames::Vector{String}, ynames::Vector{String})
    @assert length(xnames) == xdim(DS) && length(ynames) == ydim(DS)
    DataSet(xdata(DS), ydata(DS), InvCov(DS), (Npoints(DS),xdim(DS),ydim(DS)), logdetInvCov(DS), WoundX(DS), xnames, ynames)
end
