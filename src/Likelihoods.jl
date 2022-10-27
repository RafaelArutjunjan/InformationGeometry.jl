

### Likelihoods

"""
    likelihood(DM::DataModel, θ::AbstractVector) -> Real
Calculates the likelihood ``L(\\mathrm{data} \\, | \\, \\theta)`` a `DataModel` and a parameter configuration ``\\theta``.
"""
likelihood(args...; kwargs...) = exp(loglikelihood(args...; kwargs...))



## Prefix underscore for likelihood, Score and FisherMetric indicates that Prior has already been accounted for upstream
loglikelihood(DM::AbstractDataModel; kwargs...) = LogLikelihood(θ::AbstractVector{<:Number}; Kwargs...) = loglikelihood(DM, θ; kwargs..., Kwargs...)
Negloglikelihood(DM::AbstractDataModel; kwargs...) = NegativeLogLikelihood(θ::AbstractVector{<:Number}; Kwargs...) = -loglikelihood(DM, θ; kwargs..., Kwargs...)

# import Distributions.loglikelihood
"""
    loglikelihood(DM::DataModel, θ::AbstractVector) -> Real
Calculates the logarithm of the likelihood ``L``, i.e. ``\\ell(\\mathrm{data} \\, | \\, \\theta) \\coloneqq \\mathrm{ln} \\big( L(\\mathrm{data} \\, | \\, \\theta) \\big)`` given a `DataModel` and a parameter configuration ``\\theta``.
"""
loglikelihood(DM::AbstractDataModel, θ::AbstractVector{<:Number}, LogPriorFn::Union{Nothing,Function}=LogPrior(DM); kwargs...) = loglikelihood(Data(DM), Predictor(DM), θ, LogPriorFn; kwargs...)

loglikelihood(DS::AbstractDataSet, model::ModelOrFunction, θ::AbstractVector{<:Number}, LogPriorFn::Nothing; kwargs...) = _loglikelihood(DS, model, θ; kwargs...)
loglikelihood(DS::AbstractDataSet, model::ModelOrFunction, θ::AbstractVector{<:Number}, LogPriorFn::Function; kwargs...) = _loglikelihood(DS, model, θ; kwargs...) + EvalLogPrior(LogPriorFn, θ)


# Specialize this for different DataSet types
@inline function _loglikelihood(DS::AbstractDataSet, model::ModelOrFunction, θ::AbstractVector{<:Number}; kwargs...)
    -0.5*(DataspaceDim(DS)*log(2π) - logdetInvCov(DS) + InnerProduct(yInvCov(DS), ydata(DS)-EmbeddingMap(DS, model, θ; kwargs...)))
end



InnerProduct(Mat::AbstractMatrix, Y::AbstractVector) = transpose(Y) * Mat * Y
# InnerProduct(Mat::PDMats.PDMat, Y::AbstractVector) = (R = Mat.chol.U * Y;  dot(R,R))

InnerProductV(Mat::AbstractMatrix, Y::AbstractVector) = @tullio Res := Y[i] * Mat[i,j] * Y[j]
InnerProductV(Mat::Diagonal, Y::AbstractVector) = @tullio Res := Mat.diag[j] * Y[j]^2

# Does not hit BLAS, sadly
"""
    InnerProductChol(Mat::AbstractMatrix, Y::AbstractVector{T}) -> T
Computes ``|| Mat * Y ||^2``, i.e. ``Y^t \\, * (Mat^t * Mat) * Y``.
"""
function InnerProductChol(Mat::UpperTriangular, Y::AbstractVector{T})::T where T <: Number
    @assert size(Mat,2) == length(Y)
    Res = zero(T);    temp = zero(T);    n = size(Mat,2)
    @inbounds for i in 1:n
        temp = dot(view(Mat,i,i:n), view(Y,i:n))
        Res += temp^2
    end;    Res
end
function InnerProductChol(Mat::Diagonal, Y::AbstractVector{T})::T where T <: Number
    @assert length(Mat.diag) == length(Y)
    # sum(abs2, Mat.diag .* Y)
    sum((Mat.diag .* Y).^2) # faster for differentiation
end
function InnerProductChol(Mat::AbstractMatrix, Y::AbstractVector{T})::T where T <: Number
    @assert size(Mat,1) == size(Mat,2) == length(Y)
    sum(abs2, Mat*Y)
end




AutoScore(DM::AbstractDataModel, θ::AbstractVector{<:Number}, LogPriorFn::Union{Nothing, Function}=LogPrior(DM); kwargs...) = AutoScore(Data(DM), Predictor(DM), θ, LogPriorFn; kwargs...)
AutoScore(DS::AbstractDataSet, model::ModelOrFunction, θ::AbstractVector{<:Number}, LogPriorFn::Nothing; kwargs...) = _AutoScore(DS, model, θ; kwargs...)
function AutoScore(DS::AbstractDataSet, model::ModelOrFunction, θ::AbstractVector{<:Number}, LogPriorFn::Function; ADmode::Union{Symbol,Val}=Val(:ForwardDiff), kwargs...)
    _AutoScore(DS, model, θ; ADmode=ADmode, kwargs...) + EvalLogPriorGrad(LogPriorFn, θ; ADmode=ADmode)
end
_AutoScore(DS::AbstractDataSet, model::ModelOrFunction, θ::AbstractVector{<:Number}; ADmode::Union{Symbol,Val}=Val(:ForwardDiff), kwargs...) = GetGrad(ADmode, x->_loglikelihood(DS, model, x; kwargs...))(θ)


AutoMetric(DM::AbstractDataModel, θ::AbstractVector{<:Number}, LogPriorFn::Union{Nothing, Function}=LogPrior(DM); kwargs...) = AutoMetric(Data(DM), Predictor(DM), θ, LogPriorFn; kwargs...)
AutoMetric(DS::AbstractDataSet, model::ModelOrFunction, θ::AbstractVector{<:Number}, LogPriorFn::Nothing; kwargs...) = _AutoMetric(DS, model, θ; kwargs...)
function AutoMetric(DS::AbstractDataSet, model::ModelOrFunction, θ::AbstractVector{<:Number}, LogPriorFn::Function; ADmode::Union{Symbol,Val}=Val(:ForwardDiff), kwargs...)
    _AutoMetric(DS, model, θ; ADmode=ADmode, kwargs...) - EvalLogPriorHess(LogPriorFn, θ; ADmode=ADmode)
end
_AutoMetric(DS::AbstractDataSet, model::ModelOrFunction, θ::AbstractVector{<:Number}; ADmode::Union{Symbol,Val}=Val(:ForwardDiff), kwargs...) = GetHess(ADmode, x->-_loglikelihood(DS, model, x; kwargs...))(θ)



Score(DM::AbstractDataModel; kwargs...) = LogLikelihoodGradient(θ::AbstractVector{<:Number}; Kwargs...) = Score(DM, θ; kwargs..., Kwargs...)
NegScore(DM::AbstractDataModel; kwargs...) = NegativeLogLikelihoodGradient(θ::AbstractVector{<:Number}; Kwargs...) = -Score(DM, θ; kwargs..., Kwargs...)

"""
    Score(DM::DataModel, θ::AbstractVector{<:Number}; ADmode::Val=Val(:ForwardDiff))
Calculates the gradient of the log-likelihood ``\\ell`` with respect to a set of parameters ``\\theta``.
`ADmode=Val(false)` computes the Score by separately evaluating the `model` as well as the Jacobian `dmodel` provided in `DM`.
Other choices of `ADmode` directly compute the Score by differentiating the formula the log-likelihood, i.e. only one evaluation on a dual variable is performed.
"""
Score(DM::AbstractDataModel, θ::AbstractVector{<:Number}, LogPriorFn::Union{Nothing,Function}=LogPrior(DM); kwargs...) = Score(Data(DM), Predictor(DM), dPredictor(DM), θ, LogPriorFn; kwargs...)

Score(DS::AbstractDataSet, model::ModelOrFunction, dmodel::ModelOrFunction, θ::AbstractVector{<:Number}, LogPriorFn::Nothing; kwargs...) = _Score(DS, model, dmodel, θ; kwargs...)
Score(DS::AbstractDataSet, model::ModelOrFunction, dmodel::ModelOrFunction, θ::AbstractVector{<:Number}, LogPriorFn::Function; kwargs...) = _Score(DS, model, dmodel, θ; kwargs...) + EvalLogPriorGrad(LogPriorFn, θ)



_Score(DS::AbstractDataSet, model::ModelOrFunction, dmodel::ModelOrFunction, θ::AbstractVector{<:Number}; ADmode::Val=Val(:ForwardDiff), kwargs...) = _Score(DS, model, dmodel, θ, ADmode; kwargs...)
# Delegate to AutoScore
_Score(DS::AbstractDataSet, model::ModelOrFunction, dmodel::ModelOrFunction, θ::AbstractVector{<:Number}, ADmode::Val{true}; kwargs...) = _AutoScore(DS, model, θ; ADmode=ADmode, kwargs...)
_Score(DS::AbstractDataSet, model::ModelOrFunction, dmodel::ModelOrFunction, θ::AbstractVector{<:Number}, ADmode::Union{Symbol,Val}; kwargs...) = _AutoScore(DS, model, θ; ADmode=ADmode, kwargs...)

# Specialize this for different DataSet types
@inline function _Score(DS::AbstractDataSet, model::ModelOrFunction, dmodel::ModelOrFunction, θ::AbstractVector{<:Number}, ADmode::Val{false}; kwargs...)
    transpose(EmbeddingMatrix(DS,dmodel,θ; kwargs...)) * (yInvCov(DS) * (ydata(DS) - EmbeddingMap(DS,model,θ; kwargs...)))
end





"""
    LiftedLogLikelihood(DM::AbstractDataModel) -> Function
Computes log-likelihood on the extended data space ``\\hat{\\ell} : \\mathcal{X}^N \\times\\mathcal{Y}^N \\longrightarrow \\mathbb{R}``.
Should be maximized.
!!! note
    CANNOT ACCOUNT FOR PRIORS.
"""
LiftedLogLikelihood(DM::Union{AbstractDataModel,AbstractDataSet}) = (G = dist(DM);  ℓ(Z::AbstractVector{<:Number}) = logpdf(G, Z))
"""
    LiftedCost(DM::AbstractDataModel) -> Function
Computes negative log-likelihood as cost function on the extended data space ``C : \\mathcal{X}^N \\times\\mathcal{Y}^N \\longrightarrow \\mathbb{R}``.
Should be minimized.
!!! note
    CANNOT ACCOUNT FOR PRIORS.
"""
LiftedCost(DM::Union{AbstractDataModel,AbstractDataSet}) = (G = dist(DM);  Negativeℓ(Z::AbstractVector{<:Number}) = -logpdf(G, Z))

"""
    LiftedEmbedding(DM::AbstractDataModel) -> Function
Constructs lifted embedding map from initial space into extended dataspace ``\\hat{h} : \\mathcal{X}^N \\times \\mathcal{M} \\longrightarrow \\mathcal{X}^N \\times\\mathcal{Y}^N`` effecting
``\\xi = (x_\\text{opt}, \\theta) \\longmapsto \\hat{h}(\\xi) = (x_\\text{opt}, h(x_\\text{opt}, \\theta))``.
"""
LiftedEmbedding(DM::AbstractDataModel) = LiftedEmbedding(Data(DM), Predictor(DM), pdim(DM))
function LiftedEmbedding(DS::AbstractDataSet, Model::ModelOrFunction, pd::Int)
    ĥ(ξ::AbstractVector; kwargs...) = ĥ(view(ξ,1:length(ξ)-pd), view(ξ,length(ξ)-pd+1:length(ξ)); kwargs...)
    ĥ(xdat::AbstractVector, θ::AbstractVector{<:Number}; kwargs...) = [xdat; EmbeddingMap(DS, Model, θ, Windup(xdat, xdim(DS)); kwargs...)]
end

"""
    FullLiftedLogLikelihood(DM::AbstractDataModel) -> Function
Computes the full likelihood ``\\hat{\\ell} : \\mathcal{X}^N \\times\\mathcal{M}^N \\longrightarrow \\mathbb{R}`` given `Xθ` from initial space INCLUDING PRIOR.
"""
FullLiftedLogLikelihood(DM::AbstractDataModel) = FullLiftedLogLikelihood(Data(DM), Predictor(DM), LogPrior(DM), pdim(DM))
FullLiftedLogLikelihood(DS::AbstractDataSet, model::ModelOrFunction, LogPriorFn::Nothing, pd::Int) = LiftedLogLikelihood(DS)∘LiftedEmbedding(DS, model, pd)
function FullLiftedLogLikelihood(DS::AbstractDataSet, model::ModelOrFunction, LogPriorFn::Function, pd::Int)
    L = LiftedLogLikelihood(DS)∘LiftedEmbedding(DS, model, pd)
    ℓ(Xθ::AbstractVector{<:Number}; kwargs...) = L(Xθ; kwargs...) + EvalLogPrior(LogPriorFn, view(Xθ, length(Xθ)-pd+1:length(Xθ)))
end
FullLiftedNegLogLikelihood(args...; kwargs...) = (L=FullLiftedLogLikelihood(args...; kwargs...); Xθ::AbstractVector{<:Number}->-L(Xθ))
