
# Define as subtype of continuous distribution to get accepted by methods more seamlessly
# although it is actually discontinuous.
struct Dirac <: ContinuousMultivariateDistribution
    μ::AbstractVector{<:Real}
    Dirac(μ) = new(float.(Unwind(μ)))
end

import Base.length
length(d::Dirac) = length(d.μ)

import Distributions: insupport, mean, cov, invcov, pdf, logpdf
insupport(d::Dirac, x::AbstractVector) = length(d) == length(x) && all(isfinite, x)
mean(d::Dirac) = d.μ
cov(d::Dirac) = Diagonal(zeros(length(d)))
invcov(d::Dirac) = Diagonal([Inf for i in 1:length(d)])
pdf(d::Dirac, x::AbstractVector{<:Real}) = x == mean(d) ? 1. : 0.
logpdf(d::Dirac, x::AbstractVector{<:Real}) = log(pdf(d, x))


# Fix gradlogpdf for Cauchy distribution and product distributions in general
import Distributions: gradlogpdf
gradlogpdf(P::Cauchy,x::Real) = gradlogpdf(TDist(1), (x - P.μ) / P.σ) / P.σ
gradlogpdf(P::Product,x::AbstractVector) = [gradlogpdf(P.v[i],x[i]) for i in 1:length(x)]


struct DataSetExact <: AbstractDataSet
    xdist::Distribution
    ydist::Distribution
    dims::Tuple{Int,Int,Int}
    InvCov::AbstractMatrix
    WoundX::Union{AbstractVector,Bool}
    DataSetExact(DS::DataSet) = DataSetExact(xdata(DS),zeros(length(xdata(DS))*length(xdata(DS)[1])),ydata(DS),sigma(DS))
    DataSetExact(DM::AbstractDataModel) = DataSetExact(DM.Data)
    DataSetExact(x::AbstractVector,y::AbstractVector) = DataSetExact(x,zeros(length(x)),y,ones(length(y)))
    DataSetExact(x::AbstractVector,y::AbstractVector,yerr::AbstractVector) = DataSetExact(x,zeros(length(x)*length(x[1])),y,yerr)
    function DataSetExact(x::AbstractVector,xSig::AbstractVector,y::AbstractVector,ySig::AbstractVector)
        dims = HealthyData(x,y)
        length(Unwind(xSig)) != xdim(dims)*Npoints(dims) && throw("Problem with x errors.")
        length(Unwind(ySig)) != ydim(dims)*Npoints(dims) && throw("Problem with y errors.")
        if xSig == zeros(length(xSig))
            return DataSetExact(Dirac(x),DataDist(y,ySig),dims)
        else
            return DataSetExact(DataDist(x,xSig),DataDist(y,ySig),dims)
        end
    end
    function DataSetExact(x::AbstractVector,xCov::AbstractMatrix,y::AbstractVector,yCov::AbstractMatrix)
        dims = HealthyData(x,y)
        !(length(x) == length(y) == size(xCov,1) == size(yCov,1)) && throw("Vectors must have same length.")
        (!isposdef(Symmetric(xCov)) || !isposdef(Symmetric(yCov))) && throw("Covariance matrices not positive-definite.")
        DataSetExact(MvNormal(x,xCov),MvNormal(y,yCov),dims)
    end
    function DataSetExact(xd::Distribution,yd::Distribution)
        println("No information about dimensionality of x-values or y-values given. Assuming that each x and y value has a single component from here on out.")
        DataSetExact(xd,yd,(length(xd),1,1))
    end
    function DataSetExact(xd::Distribution,yd::Distribution,dims::Tuple{Int,Int,Int})
        !(Int(length(xd)/xdim(dims)) == Int(length(yd)/ydim(dims)) == Npoints(dims)) && throw("Dimensions of given distributions are inconsistent with dimensions $dims.")
        if xdim(dims) == 1
            return new(xd,yd,dims,InvCov(yd),false)
        else
            return new(xd,yd,dims,InvCov(yd),[SVector{xdim(dims)}(Z) for Z in Windup(GetMean(xd),xdim(dims))])
        end
    end
end


Npoints(DSE::DataSetExact) = Npoints(DSE.dims)
xdim(DSE::DataSetExact) = xdim(DSE.dims)
ydim(DSE::DataSetExact) = ydim(DSE.dims)
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
# Sigma(P::Distribution) = try P.Σ catch; cov(P) end
xsigma(DSE::DataSetExact) = Sigma(xdist(DSE))
ysigma(DSE::DataSetExact) = Sigma(ydist(DSE))


xdist(DME::AbstractDataModel) = xdist(DME.Data)
ydist(DME::AbstractDataModel) = ydist(DME.Data)
xsigma(DME::AbstractDataModel) = xsigma(DME.Data)
ysigma(DME::AbstractDataModel) = ysigma(DME.Data)


InvCov(P::Product) = [P.v[i].σ^(-2) for i in 1:length(P)] |> Diagonal
function InvCov(P::Distributions.GenericMvTDist)
    if P.df < 3
        return inv(P.Σ).mat
    else
        return Diagonal([Inf for i in 1:length(P)])
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


LogLike(DSE::DataSetExact,x::AbstractVector{<:Real},y::AbstractVector{<:Real}) = logpdf(xdist(DSE),x) + logpdf(ydist(DSE),y)

import Distributions: loglikelihood
loglikelihood(DSE::DataSetExact,model::ModelOrFunction,θ::AbstractVector{<:Number}) = LogLike(DSE,xdata(DSE),EmbeddingMap(DSE,model,θ))

function Score(DSE::DataSetExact,model::ModelOrFunction,dmodel::ModelOrFunction,θ::AbstractVector{<:Number})
    transpose(EmbeddingMatrix(DSE,dmodel,θ)) * gradlogpdf(ydist(DSE),EmbeddingMap(DSE,model,θ))
end

# FisherMetric(DS::AbstractDataSet, dmodel::ModelOrFunction, θ::AbstractVector{<:Number}) = Pullback(DS,dmodel,DataMetric(DS),θ)
