
# Define as subtype of continuous distribution to get accepted by methods more seamlessly
# although it is actually discontinuous.
struct Dirac <: ContinuousMultivariateDistribution
    μ::AbstractVector{<:Real}
    Dirac(μ) = new(float.(Unwind(μ)))
end

import Base.length
length(d::InformationGeometry.Dirac) = length(d.μ)

import Distributions: insupport, mean, cov, invcov, pdf, logpdf
insupport(d::InformationGeometry.Dirac, x::AbstractVector) = length(d) == length(x) && all(isfinite, x)
mean(d::InformationGeometry.Dirac) = d.μ
cov(d::InformationGeometry.Dirac) = Diagonal(zeros(length(d)))
invcov(d::InformationGeometry.Dirac) = Diagonal([Inf for i in 1:length(d)])
pdf(d::InformationGeometry.Dirac, x::AbstractVector{<:Real}) = x == mean(d) ? 1. : 0.
logpdf(d::InformationGeometry.Dirac, x::AbstractVector{<:Real}) = log(pdf(d, x))


# Fix gradlogpdf for Cauchy distribution and product distributions in general
import Distributions: gradlogpdf
gradlogpdf(P::Cauchy,x::Real) = gradlogpdf(TDist(1), (x - P.μ) / P.σ) / P.σ
gradlogpdf(P::Product,x::AbstractVector) = [gradlogpdf(P.v[i],x[i]) for i in 1:length(x)]


struct DataSetExact <: AbstractDataSet
    xdist::Distribution
    ydist::Distribution
    dims::Tuple{Int,Int,Int}
    InvCov::AbstractMatrix{<:Real}
    WoundX::Union{AbstractVector,Bool}
    xnames::Vector{String}
    ynames::Vector{String}
    DataSetExact(DM::AbstractDataModel) = DataSetExact(Data(DM))
    DataSetExact(DS::DataSet) = InformNames(DataSetExact(xDataDist(DS), yDataDist(DS), (Npoints(DS),xdim(DS),ydim(DS))), xnames(DS), ynames(DS))
    DataSetExact(x::AbstractVector,y::AbstractVector) = DataSetExact(x,zeros(length(x)),y,ones(length(y)))
    DataSetExact(x::AbstractVector{<:Real},y::AbstractVector{<:Measurement}) = DataSetExact(x,[y[i].val for i in 1:length(y)],[y[i].err for i in 1:length(y)])
    DataSetExact(x::AbstractVector,y::AbstractVector,yerr::AbstractVector) = DataSetExact(x,zeros(length(x)*length(x[1])),y,yerr)
    function DataSetExact(x::AbstractVector{<:Measurement},y::AbstractVector{<:Measurement})
        DataSetExact([x[i].val for i in 1:length(x)],[x[i].err for i in 1:length(x)],[y[i].val for i in 1:length(y)],[y[i].err for i in 1:length(y)])
    end
    function DataSetExact(x::AbstractVector, xSig::AbstractVector, y::AbstractVector, ySig::AbstractVector)
        dims = HealthyData(x,y)
        length(Unwind(xSig)) != xdim(dims)*Npoints(dims) && throw("Problem with x errors.")
        length(Unwind(ySig)) != ydim(dims)*Npoints(dims) && throw("Problem with y errors.")
        if (xSig == zeros(length(xSig))) || (xSig == Diagonal([Inf for i in 1:length(xSig)]))
            return DataSetExact(InformationGeometry.Dirac(x), DataDist(y,ySig), dims)
        else
            return DataSetExact(DataDist(x,xSig), DataDist(y,ySig), dims)
        end
    end
    function DataSetExact(x::AbstractVector, xCov::AbstractMatrix, y::AbstractVector, yCov::AbstractMatrix)
        dims = HealthyData(x,y)
        !(length(x) == length(y) == size(xCov,1) == size(yCov,1)) && throw("Vectors must have same length.")
        (!isposdef(Symmetric(xCov)) || !isposdef(Symmetric(yCov))) && throw("Covariance matrices not positive-definite.")
        DataSetExact(MvNormal(x,xCov), MvNormal(y,yCov), dims)
    end
    function DataSetExact(xd::Distribution, yd::Distribution)
        println("No information about dimensionality of x-values or y-values given. Assuming that each x and y value has a single component from here on out.")
        DataSetExact(xd, yd, (length(xd),1,1))
    end
    function DataSetExact(xd::Distribution, yd::Distribution, dims::Tuple{Int,Int,Int})
        !(Int(length(xd)/xdim(dims)) == Int(length(yd)/ydim(dims)) == Npoints(dims)) && throw("Dimensions of given distributions are inconsistent with dimensions $dims.")
        if xdim(dims) == 1
            return DataSetExact(xd, yd, dims, InvCov(yd), false)
        else
            # return new(xd,yd,dims,InvCov(yd),collect(Iterators.partition(GetMean(xd),xdim(dims))))
            return DataSetExact(xd, yd, dims, InvCov(yd), [SVector{xdim(dims)}(Z) for Z in Windup(GetMean(xd),xdim(dims))])
        end
    end
    function DataSetExact(xd::Distribution, yd::Distribution, dims::Tuple{Int,Int,Int}, InvCov::AbstractMatrix{<:Real}, WoundX::Union{AbstractVector,Bool})
        new(xd, yd, dims, InvCov, WoundX, CreateSymbolNames(xdim(dims),"x"), CreateSymbolNames(ydim(dims),"y"))
    end
end


dims(DSE::DataSetExact) = DSE.dims
InvCov(DSE::DataSetExact) = DSE.InvCov
WoundX(DS::DataSetExact) = xdim(DS) < 2 ? xdata(DS) : DS.WoundX

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
    (length(xnames) != xdim(DS) || length(ynames) != ydim(DS)) && throw("Error.")
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


LogLike(DSE::DataSetExact, x::AbstractVector{<:Real}, y::AbstractVector{<:Real}) = logpdf(xdist(DSE),x) + logpdf(ydist(DSE),y)

import Distributions: loglikelihood
loglikelihood(DSE::DataSetExact, model::ModelOrFunction, θ::AbstractVector{<:Number}; kwargs...) = LogLike(DSE, xdata(DSE), EmbeddingMap(DSE,model,θ; kwargs...))

# function _Score(DSE::DataSetExact, model::ModelOrFunction, dmodel::ModelOrFunction, θ::AbstractVector{<:Number}; kwargs...)
#     transpose(EmbeddingMatrix(DSE,dmodel,θ; kwargs...)) * gradlogpdf(ydist(DSE), EmbeddingMap(DSE,model,θ; kwargs...))
# end
function Score(DSE::DataSetExact, model::ModelOrFunction, dmodel::ModelOrFunction, θ::AbstractVector{<:Number}, Auto::Val{false}; kwargs...)
    transpose(EmbeddingMatrix(DSE,dmodel,θ; kwargs...)) * gradlogpdf(ydist(DSE), EmbeddingMap(DSE,model,θ; kwargs...))
end

# FisherMetric(DS::DataSetExact, dmodel::ModelOrFunction, θ::AbstractVector{<:Number}; kwargs...) = Pullback(DS,dmodel,DataMetric(DS),θ; kwargs...)
