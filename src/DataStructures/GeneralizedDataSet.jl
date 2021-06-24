


"""
Data structure which can take general x-y-covariance into account.
Also 0-dim x-values possible, i.e. only ydata.
"""
struct GeneralizedDataSet <: AbstractDataSet
    dist::ContinuousMultivariateDistribution
    dims::Tuple{Int,Int,Int}
    WoundX::Union{AbstractVector,Nothing}
    xnames::AbstractVector{String}
    ynames::AbstractVector{String}
    GeneralizedDataSet(DM::AbstractDataModel) = GeneralizedDataSet(Data(DM))
    function GeneralizedDataSet(DS::DataSetExact)
        xdist(DS) isa InformationGeometry.Dirac && @warn "xdist passed to GeneralizedDataSet is Dirac, continuing anyway."
        GeneralizedDataSet(GeneralProduct([xdist(DS),ydist(DS)]), dims(DS), WoundX(DS), xnames(DS), ynames(DS))
    end
    GeneralizedDataSet(args...) = DataSetExact(args...) |> GeneralizedDataSet
    function GeneralizedDataSet(dist::ContinuousMultivariateDistribution)
        @assert length(dist) % 2 == 0
        @warn "No dims given for distribution, assuming xdim = ydim = 1 and continuing."
        GeneralizedDataSet(dist, (length(dist)÷2, 1, 1))
    end
    function GeneralizedDataSet(dist::ContinuousMultivariateDistribution, dims::Tuple{Int,Int,Int})
        WoundX = xdim(dims) == 1 ? nothing : [SVector{xdim(dims)}(Z) for Z in Windup(mean(dist)[1:Npoints(dims)*xdim(dims)],xdim(dims))]
        GeneralizedDataSet(dist, dims, WoundX)
    end
    function GeneralizedDataSet(dist::ContinuousMultivariateDistribution, dims::Tuple{Int,Int,Int}, WoundX::Union{AbstractVector,Nothing}, xnames::AbstractVector{String}=CreateSymbolNames(xdim(dims),"x"), ynames::AbstractVector{String}=CreateSymbolNames(ydim(dims),"y"))
        @assert Npoints(dims) > 0 && xdim(dims) ≥ 0 && ydim(dims) > 0
        @assert xdim(dims) == length(xnames) && ydim(dims) == length(ynames)
        @assert WoundX isa AbstractVector ? (ConsistentElDims(WoundX) == xdim(dims)) : (xdim(dims) == 0)

        new(dist, dims, WoundX, xnames, ynames)
    end
end

# For SciMLBase.remake
begin
    GeneralizedDataSet(;
    dist::Distribution=Normal(0,1),
    dims::Tuple{Int,Int,Int}=(1,0,1),
    WoundX::Union{AbstractVector,Nothing}=nothing,
    xnames::Vector{String}=[],
    ynames::Vector{String}=["y"]) = GeneralizedDataSet(dist, dims, WoundX, xnames, ynames)
end


dims(GDS::GeneralizedDataSet) = GDS.dims
InvCov(GDS::GeneralizedDataSet) = GDS |> dist |> invcov

WoundX(GDS::GeneralizedDataSet) = _WoundX(GDS, GDS.WoundX)
xnames(GDS::GeneralizedDataSet) = GDS.xnames
ynames(GDS::GeneralizedDataSet) = GDS.ynames


dist(GDS::GeneralizedDataSet) = GDS.dist


"""
Does the data have mixed covariance, i.e. offdiagonal blocks in total covariance matrix nonzero?
"""
isseparable(GDS::GeneralizedDataSet) = _isseparable(dist(GDS))
_isseparable(P::Distribution) = false
_isseparable(P::GeneralProduct) = length(dist(GDS).v) == 2 ? true : false


function xsigma(GDS::GeneralizedDataSet)
    if isseparable(GDS)
        Sigma(dist(GDS).v[1])
    else
        Sigma(dist(GDS))[1:(Npoints(GDS)*xdim(GDS)),1:(Npoints(GDS)*xdim(GDS))]
    end
end
function ysigma(GDS::GeneralizedDataSet)
    if isseparable(GDS)
        Sigma(dist(GDS).v[2])
    else
        Sigma(dist(GDS))[(Npoints(GDS)*xdim(GDS) +1):end,(Npoints(GDS)*xdim(GDS) +1):end]
    end
end


xdata(GDS::GeneralizedDataSet) = isseparable(GDS) ? mean(dist(GDS).v[1]) : mean(dist(GDS))[1:(Npoints(GDS)*xdim(GDS))]
ydata(GDS::GeneralizedDataSet) = isseparable(GDS) ? mean(dist(GDS).v[2]) : mean(dist(GDS))[(Npoints(GDS)*xdim(GDS) +1):end]
sigma(GDS::GeneralizedDataSet) = Sigma(dist(GDS))

xdist(GDS::GeneralizedDataSet) = isseparable(GDS) ? dist(GDS).v[1] : typeof(dist(GDS))(xdata(GDS), xsigma(GDS))
ydist(GDS::GeneralizedDataSet) = isseparable(GDS) ? dist(GDS).v[2] : typeof(dist(GDS))(ydata(GDS), ysigma(GDS))

# not used yet
fullsigma(DS::AbstractDataSet) = cov(GeneralProduct([xdist(DS), ydist(DS)]))
fullsigma(GDS::GeneralizedDataSet) = cov(dist(GDS))



# Specialize to check if dist separable?
LogLike(GDS::GeneralizedDataSet, x::AbstractVector{<:Number}, y::AbstractVector{<:Number}) = logpdf(dist(GDS), vcat(x,y))


_loglikelihood(GDS::GeneralizedDataSet, model::ModelOrFunction, θ::AbstractVector{<:Number}; kwargs...) = LogLike(GDS, xdata(GDS), EmbeddingMap(GDS,model,θ; kwargs...))

function _Score(GDS::GeneralizedDataSet, model::ModelOrFunction, dmodel::ModelOrFunction, θ::AbstractVector{<:Number}, Auto::Val{false}; kwargs...)
    transpose(EmbeddingMatrix(GDS,dmodel,θ; kwargs...)) * gradlogpdf(ydist(GDS), EmbeddingMap(GDS,model,θ; kwargs...))
end

# _FisherMetric(DS::GeneralizedDataSet, dmodel::ModelOrFunction, θ::AbstractVector{<:Number}; kwargs...) = Pullback(DS,dmodel,DataMetric(DS),θ; kwargs...)
