

"""
    DataSetExact(x::AbstractArray, y::AbstractArray, Σ_y::AbstractArray)
    DataSetExact(x::AbstractArray, Σ_x::AbstractArray, y::AbstractArray, Σ_y::AbstractArray)
    DataSetExact(xd::Distribution, yd::Distribution, dims::Tuple{Int,Int,Int}=(length(xd),1,1))
A data container which allows for uncertainties in the independent variables, i.e. ``x``-variables.
Moreover, the observed data is stored in terms of two probability distributions over the spaces ``\\mathcal{X}^N`` and ``\\mathcal{Y}^N`` respectively, which also allows for uncertainties in the observations that are non-Gaussian.
For instance, the uncertainties associated with a given observation might follow a Cauchy, t-student, log-normal or some other smooth distribution.

Examples:
```julia
using InformationGeometry, Distributions
X = product_distribution([Normal(0, 1), Cauchy(2, 0.5)])
Y = MvTDist(2, [3, 8.], [1 0.5; 0.5 3])
DataSetExact(X, Y, (2,1,1))
```

!!! note
    Uncertainties in the independent ``x``-variables are optional for `DataSetExact`, and can be set to zero by wrapping the `x`-data in a `InformationGeometry.Dirac` "distribution".
    The following illustrates numerically equivalent ways of encoding a dataset whose uncertainties in the ``x``-variables is zero:
    ```julia
    using InformationGeometry, Distributions, LinearAlgebra
    DS1 = DataSetExact(InformationGeometry.Dirac([1,2]), MvNormal([5,6], Diagonal([0.1, 0.2].^2)))
    DS2 = DataSetExact([1,2], [5,6], [0.1, 0.2])
    DS3 = DataSet([1,2], [5,6], [0.1, 0.2])
    ```
    where `DS1 == DS2 == DS3` will evaluate to `true`.
"""
struct DataSetExact <: AbstractDataSet
    xdist::Distribution
    ydist::Distribution
    dims::Tuple{Int,Int,Int}
    InvCov::AbstractMatrix{<:Number}
    WoundX::Union{AbstractVector,Nothing}
    xnames::AbstractVector{String}
    ynames::AbstractVector{String}
    name::Union{String,Symbol}
    DataSetExact(DM::AbstractDataModel; kwargs...) = DataSetExact(Data(DM); kwargs...)
    DataSetExact(DS::DataSet; kwargs...) = DataSetExact(xDataDist(DS), yDataDist(DS), dims(DS); xnames=xnames(DS), ynames=ynames(DS), name=name(DS), kwargs...)
    DataSetExact(x::AbstractArray, y::AbstractArray, allsigmas::Real=1.0; kwargs...) = DataSetExact(x, y, allsigmas*ones(length(y)*length(y[1])); kwargs...)
    DataSetExact(x::AbstractArray, allxsigmas::Real=1.0, args...; kwargs...) = DataSetExact(x, allxsigmas*ones(length(x)*length(x[1])), args...; kwargs...)
    DataSetExact(x::AbstractArray, y::AbstractArray, yerr::AbstractArray; kwargs...) = DataSetExact(x, zeros(size(x,1)*length(x[1])), y, yerr; kwargs...)
    DataSetExact(x::AbstractVector{<:Number},y::AbstractVector{<:Measurement}; kwargs...) = DataSetExact(x,[y[i].val for i in 1:length(y)],[y[i].err for i in 1:length(y)]; kwargs...)
    function DataSetExact(x::AbstractVector{<:Measurement}, y::AbstractVector{<:Measurement}; kwargs...)
        DataSetExact([x[i].val for i in 1:length(x)],[x[i].err for i in 1:length(x)],[y[i].val for i in 1:length(y)],[y[i].err for i in 1:length(y)]; kwargs...)
    end
    ###### No Unwinding above here.
    # Offload most of the checking to DataSet
    function DataSetExact(x::AbstractArray, Σ_x::AbstractArray, y::AbstractArray, Σ_y::AbstractArray, Dims::Union{Tuple{Int,Int,Int},Nothing}=nothing; kwargs...)
        DS = Dims isa Nothing ? DataSet(x, y, Σ_y; kwargs...) : DataSet(x, y, Σ_y, Dims; kwargs...)
        Dims = Dims isa Nothing ? dims(DS) : Dims
        dims(DS) != Dims && throw("DataSetExact: Given dims Tuple inconsistent: $Dims.")
        DataSetExact(DS, Σ_x; kwargs...)
    end
    DataSetExact(DS::DataSet, σ_x::Real; kwargs...) = DataSetExact(DS, σ_x*ones(length(xdata)); kwargs...)
    function DataSetExact(DS::DataSet, Σ_x::AbstractArray; kwargs...)
        Σ_x = size(Σ_x,1) != size(Σ_x,2) ? Unwind(Σ_x) : Σ_x
        if (Σ_x == zeros(size(Σ_x,1))) || (Σ_x == Diagonal(zeros(size(Σ_x, 1))))
            DataSetExact(InformationGeometry.Dirac(xdata(DS)), yDataDist(DS), dims(DS); xnames=xnames(DS), ynames=ynames(DS), name=name(DS), kwargs...)
        else
            DataSetExact(DataDist(xdata(DS),HealthyCovariance(Σ_x)), yDataDist(DS), dims(DS); xnames=xnames(DS), ynames=ynames(DS), name=name(DS), kwargs...)
        end
    end
    function DataSetExact(xd::Distribution, yd::Distribution; kwargs...)
        @info "No information about dimensionality of x-values or y-values given. Assuming that each x and y value has a single component from here on out."
        DataSetExact(xd, yd, (length(xd),1,1); kwargs...)
    end
    function DataSetExact(xd::Distribution, yd::Distribution, dims::Tuple{Int,Int,Int}; kwargs...)
        !(Int(length(xd)/xdim(dims)) == Int(length(yd)/ydim(dims)) == Npoints(dims)) && throw("Dimensions of given distributions are inconsistent with dimensions $dims.")
        Σinv = HealthyCovariance(InvCov(yd))
        if xdim(dims) == 1
            DataSetExact(xd, yd, dims, Σinv, nothing; kwargs...)
        else
            # return new(xd,yd,dims,InvCov(yd),collect(Iterators.partition(GetMean(xd),xdim(dims))))
            DataSetExact(xd, yd, dims, Σinv, [SVector{xdim(dims)}(Z) for Z in Windup(GetMean(xd),xdim(dims))]; kwargs...)
        end
    end
    function DataSetExact(xd::Distribution, yd::Distribution, dims::Tuple{Int,Int,Int}, InvCov::AbstractMatrix{<:Number}, WoundX::Union{AbstractVector,Nothing};
                            xnames::AbstractVector{String}=CreateSymbolNames(xdim(dims),"x"), ynames::AbstractVector{String}=CreateSymbolNames(xdim(dims),"y"), name::Union{String,Symbol}=Symbol(), kwargs...)
        @assert length(xnames) == xdim(dims) && length(ynames) == ydim(dims)
        DataSetExact(xd, yd, dims, InvCov, WoundX, xnames, ynames, name; kwargs...)
    end
    function DataSetExact(xd::Distribution, yd::Distribution, dims::Tuple{Int,Int,Int}, InvCov::AbstractMatrix{<:Number}, WoundX::Union{AbstractVector,Nothing},
                            xnames::AbstractVector{String}, ynames::AbstractVector{String}, Name::Union{String,Symbol}=Symbol())
        new(xd, yd, dims, InvCov, WoundX, xnames, ynames, Name)
    end
end


# For SciMLBase.remake
DataSetExact(;
xdist::Distribution=Normal(0,1),
ydist::Distribution=Normal(0,1),
dims::Tuple{Int,Int,Int}=(1,1,1),
InvCov::AbstractMatrix{<:Number}=Diagonal([1.]),
WoundX::Union{AbstractVector,Nothing}=nothing,
xnames::AbstractVector{String}=["x"],
ynames::AbstractVector{String}=["y"],
name::Union{String,Symbol}=Symbol()) = DataSetExact(xdist, ydist, dims, InvCov, WoundX, xnames, ynames, name)


# Conversion to DataSet
function DataSet(DSE::DataSetExact)
    sum(abs,xsigma(DSE)) > 0 && @warn "Dropping x-uncertainties in conversion to DataSet."
    DataSet(xdata(DSE), ydata(DSE), ysigma(DSE), dims(DSE); xnames=xnames(DSE), ynames=ynames(DSE), name=name(DSE))
end


dims(DSE::DataSetExact) = DSE.dims
yInvCov(DSE::DataSetExact) = DSE.InvCov
xInvCov(DSE::DataSetExact) = InvCov(xdist(DSE))

WoundX(DS::DataSetExact) = _WoundX(DS, DS.WoundX)

xdist(DSE::DataSetExact) = DSE.xdist
ydist(DSE::DataSetExact) = DSE.ydist


GetMean(P::Product) = [location(P.v[i]) for i in 1:length(P)]
GetMean(P::Distribution) = mean(P)
# GetMean(P::Distribution) = P.μ

xdata(DSE::DataSetExact) = GetMean(xdist(DSE))
ydata(DSE::DataSetExact) = GetMean(ydist(DSE))

Sigma(P::Product) = [P.v[i].σ^2 for i in 1:length(P)] |> Diagonal
# I thought this was faster but it occasionally causes type problems with PDDiagMat etc.
# Sigma(P::Distribution) = P.Σ
Sigma(P::Distribution) = cov(P)
# Sigma(P::Distribution) = try P.Σ catch; cov(P) end
xsigma(DSE::DataSetExact) = Sigma(xdist(DSE)) |> _TryVectorize
ysigma(DSE::DataSetExact) = Sigma(ydist(DSE)) |> _TryVectorize

xnames(DSE::DataSetExact) = DSE.xnames
ynames(DSE::DataSetExact) = DSE.ynames

name(DSE::DataSetExact) = DSE.name |> name

# function InformNames(DS::DataSetExact, xnames::AbstractVector{String}, ynames::AbstractVector{String})
#     @assert length(xnames) == xdim(DS) && length(ynames) == ydim(DS)
#     DataSetExact(xdist(DS), ydist(DS), (Npoints(DS),xdim(DS),ydim(DS)), yInvCov(DS), WoundX(DS), xnames, ynames)
# end


InvCov(P::Product) = [P.v[i].σ^(-2) for i in 1:length(P)] |> Diagonal
function InvCov(P::Distributions.GenericMvTDist)
    if P.df < 3
        return inv(P.Σ).mat
    else
        return invcov(P)
    end
end
InvCov(P::Distribution) = invcov(P)


DataMetric(P::Distribution) = InvCov(P)
function DataMetric(P::Distributions.GenericMvTDist)
    if P.df == 1
        return 0.5InvCov(P)
    else
        @warn "DataMetric: Don't know what to do for t-distribution with dof=$(P.df), just returning usual inverse covariance matrix."
        return InvCov(P)
    end
end

# Needs testing!!!!
isCauchy(P::Distribution{Univariate,Continuous}) = false;    isCauchy(P::Cauchy) = true
function DataMetric(P::Product)
    icov = InvCov(P).diag
    [isCauchy(P.v[i]) ? 0.5 : 1. for i in 1:length(P)] .* icov |> Diagonal
end

LogLike(DM::AbstractDataSet, x::AbstractVector{<:Number}, y::AbstractVector{<:Number}) = LogLike(Data(DM), x, y)
LogLike(DSE::DataSetExact, x::AbstractVector{<:Number}, y::AbstractVector{<:Number}) = logpdf(xdist(DSE), x) + logpdf(ydist(DSE), y)

import Distributions: loglikelihood
_loglikelihood(DSE::DataSetExact, model::ModelOrFunction, θ::AbstractVector{<:Number}; kwargs...) = LogLike(DSE, xdata(DSE), EmbeddingMap(DSE,model,θ; kwargs...))

function _Score(DSE::DataSetExact, model::ModelOrFunction, dmodel::ModelOrFunction, θ::AbstractVector{<:Number}, ADmode::Val{false}; kwargs...)
    transpose(EmbeddingMatrix(DSE,dmodel,θ; kwargs...)) * gradlogpdf(ydist(DSE), EmbeddingMap(DSE,model,θ; kwargs...))
end

# _FisherMetric(DS::DataSetExact, dmodel::ModelOrFunction, θ::AbstractVector{<:Number}; kwargs...) = Pullback(DS,dmodel,DataMetric(DS),θ; kwargs...)
