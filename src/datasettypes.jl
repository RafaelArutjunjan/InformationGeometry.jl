

abstract type TemperedDistributions <: ContinuousMultivariateDistribution end
struct Dirac <: TemperedDistributions
    μ::AbstractVector{<:Real}
    Dirac(μ) = new(float.(Unwind(μ)))
end

import Base.length
length(d::TemperedDistributions) = length(d.μ)

import Distributions: insupport, mean, cov, invcov, pdf, logpdf
insupport(d::TemperedDistributions,x::AbstractVector) = length(d) == length(x) && all(isfinite, x)
mean(d::TemperedDistributions) = d.μ
cov(d::TemperedDistributions) = Diagonal(zeros(length(d)))
invcov(d::TemperedDistributions) = Diagonal([Inf for i in 1:length(d)])
pdf(d::TemperedDistributions,x::AbstractVector)::Float64 = x == mean(d) ? 1. : 0.
logpdf(d::TemperedDistributions,x::AbstractVector) = log(pdf(d,x))



struct DataSetExact <: AbstractDataSet
    xdist::Distribution
    ydist::Distribution
    dims::Tuple{Int,Int,Int}
    InvCov::AbstractMatrix
    # X::AbstractVector
    DataSetExact(DS::DataSet) = DataSetExact(xdata(DS),zeros(length(xdata(DS))*length(xdata(DS)[1])),ydata(DS),sigma(DS))
    DataSetExact(DM::AbstractDataModel) = DataSetExact(DM.Data)
    DataSetExact(x::AbstractVector,y::AbstractVector) = DataSetExact(x,zeros(length(x)),y,ones(length(y)))
    DataSetExact(x::AbstractVector,y::AbstractVector,yerr::AbstractVector) = DataSetExact(x,zeros(length(x)*length(x[1])),y,yerr)
    function DataSetExact(x::AbstractVector,xSig::AbstractVector,y::AbstractVector,ySig::AbstractVector)
        dims = HealthyData(x,y)
        length(Unwind(xSig)) != xdim(dims)*N(dims) && throw("Problem with x errors.")
        length(Unwind(ySig)) != ydim(dims)*N(dims) && throw("Problem with y errors.")
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
        DataSetExact(xd,yd,Tuple([length(xd),1,1]))
    end
    function DataSetExact(xd::Distribution,yd::Distribution,dims::Tuple{Int,Int,Int})
        Int(length(xd)/xdim(dims)) == Int(length(yd)/ydim(dims)) == N(dims) && return new(xd,yd,dims,InvCov(yd))
        throw("Dimensions of distributions are inconsistent with $dims: $xd and $yd.")
    end
end


N(DSE::DataSetExact) = N(DSE.dims)
xdim(DSE::DataSetExact) = xdim(DSE.dims)
ydim(DSE::DataSetExact) = ydim(DSE.dims)
InvCov(DSE::DataSetExact) = DSE.InvCov
xdist(DSE::DataSetExact) = DSE.xdist
ydist(DSE::DataSetExact) = DSE.ydist


GetMean(P::Product) = [location(P.v[i]) for i in 1:length(P)]
GetMean(P::Distribution) = P.μ
data(DSE::DataSetExact,F::Function) = GetMean(F(DSE))
xdata(DSE::DataSetExact) = data(DSE,xdist)
ydata(DSE::DataSetExact) = data(DSE,ydist)

sigma(P::Product) = [P.v[i].σ^2 for i in 1:length(P)] |> Diagonal
sigma(P::Distribution) = P.Σ
xsigma(DSE::DataSetExact) = Sigma(xdist(DSE))
ysigma(DSE::DataSetExact) = Sigma(ydist(DSE))

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
        return 0.5 .*InvCov(P)
    else
        println("DataMetric: Don't know what to do for t-distribution with dof=$(P.df), just returning usual inverse covariance matrix.")
        return InvCov(P)
    end
end

LogLike(DSE::DataSetExact,x::AbstractVector{<:Real},y::AbstractVector{<:Real}) = logpdf(xdist(DSE),x) + logpdf(ydist(DSE),y)

loglikelihood(DSE::DataSetExact,model::Function,θ::AbstractVector{<:Number}) = LogLike(DSE,xdata(DSE),EmbeddingMap(DSE,model,θ))

function Score(DSE::DataSetExact,model::Function,dmodel::Function,θ::AbstractVector{<:Number})
    transpose(EmbeddingMatrix(DSE,dmodel,θ)) * gradlogpdf(ydist(DSE),EmbeddingMap(DSE,model,θ))
end
