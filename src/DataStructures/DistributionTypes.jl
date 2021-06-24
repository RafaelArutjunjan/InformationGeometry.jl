


"""
Extension of product_distribution from Distributions.jl to allow for taking products between arbitrary distributions, not just univariate.
"""
struct GeneralProduct <: ContinuousMultivariateDistribution
    v::AbstractVector{<:Union{ContinuousMultivariateDistribution, ContinuousUnivariateDistribution}}
    function GeneralProduct(v::AbstractVector{<:Union{ContinuousMultivariateDistribution, ContinuousUnivariateDistribution}})
        if all(x->x isa AbstractMvNormal, v)
            MvNormal(mapreduce(mean, vcat, v), HealthyCovariance(reduce(BlockMatrix, map(HealthyCovariance∘cov, v))))
        else
            new(v)
        end
    end
end
GeneralProduct(DM::Union{AbstractDataModel,AbstractDataSet}) = GeneralProduct([xdist(DM),ydist(DM)])


import Base.length
length(P::GeneralProduct) = sum(length(P.v[i]) for i in 1:length(P.v))

import Distributions: insupport, mean, cov, invcov, pdf, logpdf, gradlogpdf, product_distribution
# insupport(P::GeneralProduct, X::AbstractVector)::Bool = all([insupport(P.v[i], X[P.lengths[i],]) for i in 1:length(P.lengths)])
# sum(!insupport(P.v[i],X[i]) for i in 1:length(P.v)) == 0
mean(P::GeneralProduct) = reduce(vcat, map(mean, P.v))
cov(P::GeneralProduct) = BlockMatrix(map(cov, P.v)...)


for F = (:logpdf, :gradlogpdf)
    eval(quote
        # Base.$F(a::MyNumber) = MyNumber($F(a.x))
        $F(P::GeneralProduct, X::AbstractVector) = sum($F(P.v[i], X[i]) for i in 1:length(P.v))
        $F(P::GeneralProduct, X::AbstractVector{<:Real}) = $F(P, X, Val(length(P.v)))
        $F(P::GeneralProduct, X::AbstractVector{<:Real}, ::Val{1}) = $F(P.v[1], X)
        $F(P::GeneralProduct, X::AbstractVector{<:Real}, ::Val{2}) = $F(P.v[1], view(X,1:length(P.v[1]))) + $F(P.v[2], view(X,length(P.v[1])+1:lastindex(X)))
        # There has to be a more performant way than this!
        function $F(P::GeneralProduct, X::AbstractVector{<:Real}, ::Val)
            C = vcat([0],cumsum(length.(P.v)))
            sum($F(P.v[i], X[C[i]+1:C[i+1]]) for i in 1:length(P.v))
        end
    end)
end


pdf(P::GeneralProduct, X::AbstractVector) = exp(logpdf(P,X))
invcov(P::GeneralProduct) = BlockMatrix(map(invcov,P.v)...)
product_distribution(X::AbstractVector{<:ContinuousMultivariateDistribution}) = GeneralProduct(X)


Sigma(P::GeneralProduct) = cov(P)


# LogLike(P::GeneralProduct, args...) = logpdf(P,[args...])





# Define as subtype of continuous distribution to get accepted by methods more seamlessly
# although it is actually discontinuous.
"""
pdf at μ takes value 1 and 0 everywhere else.
"""
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
pdf(d::InformationGeometry.Dirac, x::AbstractVector{<:Number}) = x == mean(d) ? 1.0 : 0.0
logpdf(d::InformationGeometry.Dirac, x::AbstractVector{<:Number}) = log(pdf(d, x))


# Fix gradlogpdf for Cauchy distribution and product distributions in general
import Distributions: gradlogpdf
gradlogpdf(P::Cauchy,x::Real) = gradlogpdf(TDist(1), (x - P.μ) / P.σ) / P.σ
gradlogpdf(P::Product,x::AbstractVector) = [gradlogpdf(P.v[i],x[i]) for i in 1:length(x)]
