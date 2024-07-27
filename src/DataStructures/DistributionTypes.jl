


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



Base.length(P::InformationGeometry.GeneralProduct) = sum(length(P.v[i]) for i in eachindex(P.v))


# insupport(P::InformationGeometry.GeneralProduct, X::AbstractVector)::Bool = all([insupport(P.v[i], X[P.lengths[i],]) for i in eachindex(P.lengths)])
# sum(!insupport(P.v[i],X[i]) for i in eachindex(P.v)) == 0
Distributions.mean(P::InformationGeometry.GeneralProduct) = reduce(vcat, map(GetMean, P.v))
Distributions.cov(P::InformationGeometry.GeneralProduct) = reduce(BlockMatrix, map(Sigma, P.v))

# For MetaProgramming using the module prefix "Distributions." doesn't appear to work and functions must be imported explicitly
import Distributions: logpdf, gradlogpdf
for F = (Symbol("logpdf"), Symbol("gradlogpdf"))
    eval(quote
        # Base.$F(a::MyNumber) = MyNumber($F(a.x))
        $F(P::InformationGeometry.GeneralProduct, X::AbstractVector) = sum($F(P.v[i], X[i]) for i in eachindex(P.v))
        $F(P::InformationGeometry.GeneralProduct, X::AbstractVector{<:Real}) = $F(P, X, Val(length(P.v)))
        $F(P::InformationGeometry.GeneralProduct, X::AbstractVector{<:Real}, ::Val{1}) = $F(P.v[1], X)
        $F(P::InformationGeometry.GeneralProduct, X::AbstractVector{<:Real}, ::Val{2}) = $F(P.v[1], view(X,1:length(P.v[1]))) + $F(P.v[2], view(X,length(P.v[1])+1:lastindex(X)))
        # There has to be a more performant way than this!
        function $F(P::InformationGeometry.GeneralProduct, X::AbstractVector{<:Real}, ::Val)
            C = vcat([0],cumsum(length.(P.v)))
            sum($F(P.v[i], X[C[i]+1:C[i+1]]) for i in eachindex(P.v))
        end
    end)
end


Distributions.pdf(P::InformationGeometry.GeneralProduct, X::AbstractVector) = exp(logpdf(P,X))
Distributions.invcov(P::InformationGeometry.GeneralProduct) = reduce(BlockMatrix, map(InvCov,P.v))
Distributions.product_distribution(X::AbstractVector{<:ContinuousMultivariateDistribution}) = GeneralProduct(X)


Sigma(P::InformationGeometry.GeneralProduct) = cov(P)


# LogLike(P::InformationGeometry.GeneralProduct, args...) = logpdf(P,[args...])





# Define as subtype of continuous distribution to get accepted by methods more seamlessly
# although it is actually discontinuous.
"""
pdf at μ takes value 1 and 0 everywhere else.
"""
struct Dirac <: ContinuousMultivariateDistribution
    μ::AbstractVector{<:Number}
    Dirac(μ) = new(floatify(Unwind(μ)))
end


Base.length(d::InformationGeometry.Dirac) = length(d.μ)


Distributions.insupport(d::InformationGeometry.Dirac, x::AbstractVector) = length(d) == length(x) && all(isfinite, x)
Distributions.mean(d::InformationGeometry.Dirac) = d.μ
Distributions.cov(d::InformationGeometry.Dirac) = Diagonal(zeros(eltype(d.μ),length(d)))
Distributions.invcov(d::InformationGeometry.Dirac) = Diagonal(eltype(d.μ)[Inf for i in 1:length(d)])
Distributions.pdf(d::InformationGeometry.Dirac, x::AbstractVector{<:Number}) = x == mean(d) ? 1.0 : 0.0
Distributions.logpdf(d::InformationGeometry.Dirac, x::AbstractVector{<:Number}) = log(pdf(d, x))
Distributions.params(d::InformationGeometry.Dirac) = (d.μ,)


# Fix gradlogpdf for Cauchy distribution and product distributions in general
Distributions.gradlogpdf(P::Cauchy,x::Real) = gradlogpdf(TDist(1), (x - P.μ) / P.σ) / P.σ
Distributions.gradlogpdf(P::Product,x::AbstractVector) = [gradlogpdf(P.v[i],x[i]) for i in eachindex(x)]



# Get Symbol for everything before {} in type.
UnparametrizeType(D) = (S=string(typeof(D)); ind=findfirst('{',S); isnothing(ind) ? Symbol(S) : Symbol(S[1:(ind-1)]))

## Change Number Type of distributions
ConvertDist(D::Distributions.Distribution, ::Type{T}) where T<:Number = eval(quote $(UnparametrizeType(D))(broadcast(x->broadcast($T,x), $(Distributions.params(D)))...) end)
# Dirac not exported so MetaProgramming solution does not work
ConvertDist(D::InformationGeometry.Dirac, ::Type{T}) where T<:Number = InformationGeometry.Dirac(T.(D.μ))
# Has type specializations depending on structure of covariance which should be ignored due to missing constructors
ConvertDist(D::MvNormal, ::Type{T}) where T<:Number = MvNormal(T.(mean(D)), T.(cov(D)))

function ConvertDist(D::Union{Distributions.Product, InformationGeometry.GeneralProduct}, ::Type{T}) where T<:Number
    product_distribution(broadcast(x->ConvertDist(x,T), D.v))
end