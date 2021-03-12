
# Define as subtype of continuous distribution to get accepted by methods more seamlessly
# although it is actually discontinuous.
struct Dirac <: ContinuousMultivariateDistribution
    μ::AbstractVector{<:Number}
    Dirac(μ) = new(float.(Unwind(μ)))
end

import Base.length
length(d::InformationGeometry.Dirac) = length(d.μ)

import Distributions: insupport, mean, cov, invcov, pdf, logpdf
insupport(d::InformationGeometry.Dirac, x::AbstractVector) = length(d) == length(x) && all(isfinite, x)
mean(d::InformationGeometry.Dirac) = d.μ
cov(d::InformationGeometry.Dirac) = Diagonal(zeros(length(d)))
invcov(d::InformationGeometry.Dirac) = Diagonal([Inf for i in 1:length(d)])
pdf(d::InformationGeometry.Dirac, x::AbstractVector{<:Number}) = x == mean(d) ? 1. : 0.
logpdf(d::InformationGeometry.Dirac, x::AbstractVector{<:Number}) = log(pdf(d, x))


# Fix gradlogpdf for Cauchy distribution and product distributions in general
import Distributions: gradlogpdf
gradlogpdf(P::Cauchy,x::Real) = gradlogpdf(TDist(1), (x - P.μ) / P.σ) / P.σ
gradlogpdf(P::Product,x::AbstractVector) = [gradlogpdf(P.v[i],x[i]) for i in 1:length(x)]


struct DataSetExact <: AbstractDataSet
    xdist::Distribution
    ydist::Distribution
    dims::Tuple{Int,Int,Int}
    InvCov::AbstractMatrix{<:Number}
    WoundX::Union{AbstractVector,Nothing}
    xnames::Vector{String}
    ynames::Vector{String}
    DataSetExact(DM::AbstractDataModel) = DataSetExact(Data(DM))
    DataSetExact(DS::DataSet) = InformNames(DataSetExact(xDataDist(DS), yDataDist(DS), dims(DS)), xnames(DS), ynames(DS))
    DataSetExact(x::AbstractArray, y::AbstractArray, allsigmas::Real=1.0) = DataSetExact(x, y, allsigmas*ones(length(y)*length(y[1])))
    DataSetExact(x::AbstractArray, allxsigmas::Real=1.0, args...) = DataSetExact(x, allxsigmas*ones(length(x)*length(x[1])), args...)
    DataSetExact(x::AbstractArray, y::AbstractArray, yerr::AbstractArray) = DataSetExact(x, zeros(size(x,1)*length(x[1])), y, yerr)
    DataSetExact(x::AbstractVector{<:Number},y::AbstractVector{<:Measurement}) = DataSetExact(x,[y[i].val for i in 1:length(y)],[y[i].err for i in 1:length(y)])
    function DataSetExact(x::AbstractVector{<:Measurement}, y::AbstractVector{<:Measurement})
        DataSetExact([x[i].val for i in 1:length(x)],[x[i].err for i in 1:length(x)],[y[i].val for i in 1:length(y)],[y[i].err for i in 1:length(y)])
    end
    ###### No Unwinding above here.
    # Offload most of the checking to DataSet
    function DataSetExact(x::AbstractArray, Σ_x::AbstractArray, y::AbstractArray, Σ_y::AbstractArray, Dims::Union{Tuple{Int,Int,Int},Nothing}=nothing)
        DS = Dims isa Nothing ? DataSet(x, y, Σ_y) : DataSet(x, y, Σ_y, Dims)
        Dims = Dims isa Nothing ? dims(DS) : Dims
        dims(DS) != Dims && throw("DataSetExact: Given dims Tuple inconsistent: $Dims.")
        DataSetExact(DS, Σ_x)
    end
    function DataSetExact(DS::DataSet, Σ_x::AbstractArray)
        Σ_x = size(Σ_x,1) > size(Σ_x,2) ? Unwind(Σ_x) : Σ_x
        if (Σ_x == zeros(size(Σ_x,1))) || (Σ_x == Diagonal(zeros(size(Σ_x, 1))))
            return InformNames(DataSetExact(InformationGeometry.Dirac(xdata(DS)), yDataDist(DS), dims(DS)), xnames(DS), ynames(DS))
        else
            return InformNames(DataSetExact(DataDist(xdata(DS),HealthyCovariance(Σ_x)), yDataDist(DS), dims(DS)), xnames(DS), ynames(DS))
        end
    end
    function DataSetExact(xd::Distribution, yd::Distribution)
        println("No information about dimensionality of x-values or y-values given. Assuming that each x and y value has a single component from here on out.")
        DataSetExact(xd, yd, (length(xd),1,1))
    end
    function DataSetExact(xd::Distribution, yd::Distribution, dims::Tuple{Int,Int,Int})
        !(Int(length(xd)/xdim(dims)) == Int(length(yd)/ydim(dims)) == Npoints(dims)) && throw("Dimensions of given distributions are inconsistent with dimensions $dims.")
        Σinv = HealthyCovariance(InvCov(yd))
        if xdim(dims) == 1
            return DataSetExact(xd, yd, dims, Σinv, nothing)
        else
            # return new(xd,yd,dims,InvCov(yd),collect(Iterators.partition(GetMean(xd),xdim(dims))))
            return DataSetExact(xd, yd, dims, Σinv, [SVector{xdim(dims)}(Z) for Z in Windup(GetMean(xd),xdim(dims))])
        end
    end
    function DataSetExact(xd::Distribution, yd::Distribution, dims::Tuple{Int,Int,Int}, InvCov::AbstractMatrix{<:Number}, WoundX::Union{AbstractVector,Nothing},
                    xnames::Vector{String}=CreateSymbolNames(xdim(dims),"x"), ynames::Vector{String}=CreateSymbolNames(xdim(dims),"y"))
        new(xd, yd, dims, InvCov, WoundX, xnames, ynames)
    end
end


dims(DSE::DataSetExact) = DSE.dims
InvCov(DSE::DataSetExact) = DSE.InvCov
# WoundX(DS::DataSetExact) = xdim(DS) < 2 ? xdata(DS) : DS.WoundX
WoundX(DS::DataSetExact) = _WoundX(DS, DS.WoundX)
_WoundX(DS::DataSetExact, WoundX::Nothing) = xdata(DS)
_WoundX(DS::DataSetExact, WoundX::AbstractVector) = WoundX

xdist(DSE::DataSetExact) = DSE.xdist
ydist(DSE::DataSetExact) = DSE.ydist


GetMean(P::Product) = [location(P.v[i]) for i in 1:length(P)]
GetMean(P::Distribution) = mean(P)
# GetMean(P::Distribution) = P.μ

data(DSE::DataSetExact,F::Function) = GetMean(F(DSE))
xdata(DSE::DataSetExact) = data(DSE,xdist)
ydata(DSE::DataSetExact) = data(DSE,ydist)

Sigma(P::Product) = [P.v[i].σ^2 for i in 1:length(P)] |> Diagonal
Sigma(P::Distribution) = P.Σ
Sigma(P::InformationGeometry.Dirac) = cov(P)
# Sigma(P::Distribution) = try P.Σ catch; cov(P) end
xsigma(DSE::DataSetExact) = Sigma(xdist(DSE))
ysigma(DSE::DataSetExact) = Sigma(ydist(DSE))

xnames(DSE::DataSetExact) = DSE.xnames
ynames(DSE::DataSetExact) = DSE.ynames

function InformNames(DS::DataSetExact, xnames::Vector{String}, ynames::Vector{String})
    @assert length(xnames) == xdim(DS) && length(ynames) == ydim(DS)
    DataSetExact(xdist(DS), ydist(DS), (Npoints(DS),xdim(DS),ydim(DS)), InvCov(DS), WoundX(DS), xnames, ynames)
end

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
        return 0.5 .* InvCov(P)
    else
        println("DataMetric: Don't know what to do for t-distribution with dof=$(P.df), just returning usual inverse covariance matrix.")
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
LogLike(DSE::DataSetExact, x::AbstractVector{<:Number}, y::AbstractVector{<:Number}) = logpdf(xdist(DSE),x) + logpdf(ydist(DSE),y)

import Distributions: loglikelihood
loglikelihood(DSE::DataSetExact, model::ModelOrFunction, θ::AbstractVector{<:Number}; kwargs...) = LogLike(DSE, xdata(DSE), EmbeddingMap(DSE,model,θ; kwargs...))

function Score(DSE::DataSetExact, model::ModelOrFunction, dmodel::ModelOrFunction, θ::AbstractVector{<:Number}, Auto::Val{false}; kwargs...)
    transpose(EmbeddingMatrix(DSE,dmodel,θ; kwargs...)) * gradlogpdf(ydist(DSE), EmbeddingMap(DSE,model,θ; kwargs...))
end

# FisherMetric(DS::DataSetExact, dmodel::ModelOrFunction, θ::AbstractVector{<:Number}; kwargs...) = Pullback(DS,dmodel,DataMetric(DS),θ; kwargs...)
