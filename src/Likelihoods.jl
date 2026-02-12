

### Likelihoods

"""
    likelihood(DM::DataModel, θ::AbstractVector) -> Real
Calculates the likelihood ``L(\\mathrm{data} \\, | \\, \\theta)`` a `DataModel` and a parameter configuration ``\\theta``.
"""
likelihood(args...; kwargs...) = exp(loglikelihood(args...; kwargs...))



## Prefix underscore for likelihood, Score and FisherMetric indicates that Prior has already been accounted for upstream
loglikelihood(DM::AbstractDataModel; kwargs...) = LogLikelihood(θ::AbstractVector{<:Number}; Kwargs...) = loglikelihood(DM, θ; kwargs..., Kwargs...)
Negloglikelihood(DM::AbstractDataModel; kwargs...) = Negate(loglikelihood(DM; kwargs...))

# import Distributions.loglikelihood
"""
    loglikelihood(DM::DataModel, θ::AbstractVector) -> Real
Calculates the logarithm of the likelihood ``L``, i.e. ``\\ell(\\mathrm{data} \\, | \\, \\theta) \\coloneqq \\mathrm{ln} \\big( L(\\mathrm{data} \\, | \\, \\theta) \\big)`` given a `DataModel` and a parameter configuration ``\\theta``.
"""
loglikelihood(DM::AbstractDataModel, θ::AbstractVector{<:Number}; kwargs...) = loglikelihood(DM)(θ; kwargs...)
loglikelihood(DM::AbstractDataModel, θ::AbstractVector{<:Number}, LogPriorFn::Union{Nothing,Function}; kwargs...) = (@warn "Will deprecate this loglikelihood method soon!";   loglikelihood(Data(DM), Predictor(DM), θ, LogPriorFn; kwargs...))
# loglikelihood(DM::AbstractDataModel, θ::AbstractVector{<:Number}, LogPriorFn::Union{Nothing,Function}; kwargs...) = throw("Here's a stackstrace for you!")

loglikelihood(DS::AbstractDataSet, model::ModelOrFunction, θ::AbstractVector{<:Number}, LogPriorFn::Nothing; kwargs...) = _loglikelihood(DS, model, θ; kwargs...)
loglikelihood(DS::AbstractDataSet, model::ModelOrFunction, θ::AbstractVector{<:Number}, LogPriorFn::Function; kwargs...) = _loglikelihood(DS, model, θ; kwargs...) + EvalLogPrior(LogPriorFn, θ)


# Specialize this for different DataSet types
function _loglikelihood(DS::AbstractDataSet, model::ModelOrFunction, θ::AbstractVector{T}; kwargs...)::T where T<:Number
    (DataspaceDim(DS)*log(2T(π)) - logdetInvCov(DS) + InnerProduct(yInvCov(DS, θ), ydata(DS).-EmbeddingMap(DS, model, θ; kwargs...))) / T(-2)
end

# function _loglikelihood(DS::AbstractDataSet, model::ModelOrFunction, θ::AbstractVector{<:Number}; kwargs...)
#     __loglikelihood(DS, model, θ, ydata(DS), yInvCov(DS, θ); kwargs...)
# end

# function __loglikelihood(DS::AbstractDataSet, model::ModelOrFunction, θ::AbstractVector{<:Number}, Ydata::AbstractVector, YinvCov::AbstractMatrix; kwargs...)
#     Y = EmbeddingMap(DS, model, θ; kwargs...)
#     @inline (DataspaceDim(DS)*log(2π) - logdetInvCov(DS) + InnerProduct(YinvCov, Ydata .- Y)) / (-2)
# end
# function __loglikelihood(DS::AbstractDataSet, model::ModelOrFunction, θ::AbstractVector{<:Number}; kwargs...)
#     (DataspaceDim(DS)*log(2π) - logdetInvCov(DS) + InnerProduct(yInvCov(DS, θ), ydata(DS).-EmbeddingMap(DS, model, θ; kwargs...))) / (-2)
# end


## Fully symbolic representation
# function _loglikelihood(DS::AbstractDataSet, model::ModelOrFunction, θ::AbstractVector{<:Num}; kwargs...)
#     -(DataspaceDim(DS)*log(2π) - logdetInvCov(DS) + InnerProduct(yInvCov(DS, θ), MakeMTKVariables(Symbol.(CreateSymbolNames(length(ydata(DS)), "y_data"))) .- EmbeddingMap(DS, model, θ; kwargs...))) // (2)
# end

function GetLogLikelihoodFn(DS::AbstractDataSet, model::ModelOrFunction, LogPriorFn::Union{Nothing,Function}; ADmode::Union{Symbol,Val}=Val(:ForwardDiff), inplace::Bool=isinplacemodel(model), levels::Int=3, Kwargs...)
    # Pre-Computations or buffers here
    BareLikelihood = if inplace
        GetInplaceLikelihood(DS, model; levels)
    else
        OutOfPlaceLikelihood(θ::AbstractVector{<:Number}; kwargs...) = _loglikelihood(DS, model, θ; kwargs...)
    end
    if isnothing(LogPriorFn)
        """
            LogLikelihoodWithoutPrior(θ::AbstractVector; kwargs...) -> Real
        Calculates the logarithm of the likelihood ``L``, i.e. ``\\ell(\\mathrm{data} \\, | \\, \\theta) \\coloneqq \\mathrm{ln} \\big( L(\\mathrm{data} \\, | \\, \\theta) \\big)`` given a `DataModel` and a parameter configuration ``\\theta``.
        No prior was given for the parameters.
        """
        function LogLikelihoodWithoutPrior(θ::AbstractVector; kwargs...)
            BareLikelihood(θ; Kwargs..., kwargs...)
        end
    else
        """
            LogLikelihoodWithPrior(θ::AbstractVector; kwargs...) -> Real
        Calculates the logarithm of the likelihood ``L``, i.e. ``\\ell(\\mathrm{data} \\, | \\, \\theta) \\coloneqq \\mathrm{ln} \\big( L(\\mathrm{data} \\, | \\, \\theta) \\big)`` given a `DataModel` and a parameter configuration ``\\theta``.
        The given prior information is already incorporated.
        """
        function LogLikelihoodWithPrior(θ::AbstractVector{T}; kwargs...)::T where T<:Number
            BareLikelihood(θ; Kwargs..., kwargs...) + EvalLogPrior(LogPriorFn, θ)
        end
    end
end
GetNeglogLikelihoodFn(args...; kwargs...) = Negate(GetLogLikelihoodFn(args...; kwargs...))


# dot(Y, Mat, Y) faster for differentiation than transpose(Y) * Mat * Y due to fewer allocations
InnerProduct(Mat::AbstractMatrix, Y::AbstractVector) = _InnerProduct(Mat, Y)
# For sparse arrays, direct multiplication also faster:
_InnerProduct(Mat::AbstractMatrix, Y::AbstractVector{<:Number}) = transpose(Y) * Mat * Y
_InnerProduct(Mat::AbstractMatrix, Y::AbstractVector{<:ForwardDiff.Dual}) = dot(Y, Mat, Y)

function InnerProduct(Mat::DiagonalType, Y::AbstractVector{T}) where T <: Number
    d = Mat.diag
    @boundscheck length(d) == length(Y)
    Res = zero(T)
    @inbounds @simd for i in eachindex(Y)
        Res += abs2(Y[i]) * d[i]
    end;    Res
end
# InnerProduct(Mat::PDMats.PDMat, Y::AbstractVector) = (R = Mat.chol.U * Y;  dot(R,R))

## Only marginally faster than simd loop for Diagonal
function InnerProduct(D::LinearAlgebra.UniformScaling, Y::AbstractVector{T}) where T<:Number
    d = D.λ;    Res = zero(T)
    @inbounds @simd for y in Y
        Res += abs2(y)
    end;    Res *= d;    Res
end

InnerProductV(Mat::AbstractMatrix, Y::AbstractVector) = @tullio Res := Y[i] * Mat[i,j] * Y[j]
InnerProductV(Mat::DiagonalType, Y::AbstractVector) = (d = Mat.diag;  @tullio Res := d[j] * abs2(Y[j]))

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
function InnerProductChol(Mat::DiagonalType, Y::AbstractVector{T})::T where T <: Number
    @assert length(Mat.diag) == length(Y)
    # sum(abs2, Mat.diag .* Y)
    sum((Mat.diag .* Y).^2) # faster for differentiation
end
function InnerProductChol(Mat::AbstractMatrix, Y::AbstractVector{T})::T where T <: Number
    @assert size(Mat,1) == size(Mat,2) == length(Y)
    sum(abs2, Mat*Y)
end

GetInplaceLikelihood(DM::AbstractDataModel; kwargs...) = GetInplaceLikelihood(Data(DM), Predictor(DM); kwargs...)
function GetInplaceLikelihood(DS::AbstractFixedUncertaintyDataSet, model::ModelOrFunction; levels::Int=3)
    @assert !HasXerror(DS)
    ydat = ydata(DS);    invCov = yInvCov(DS);      woundX = WoundX(DS)
    Ycache = DiffCache(similar(ydat); levels)
    NormalizationConst = DataspaceDim(DS)*log(2π) - logdetInvCov(DS)
    function _Like!(Z::Union{AbstractVector,DiffCache{<:AbstractVector}}, invCov::AbstractMatrix, NormalizationConst::Number)
        -0.5.*(NormalizationConst .+ InnerProduct(invCov, Z))
    end
    function InplaceLikelihood(θ::AbstractVector{<:Number}; kwargs...)
        Y = UnrollCache(Ycache, θ, woundX)
        fill!(Y, zero(eltype(Y)))
        EmbeddingMap!(Y, DS, model, θ, woundX; kwargs...)
        ## Fuse with LogPrior execution here for priors which re-use Ypred?
        Y .-= ydat
        _Like!(Y, invCov, NormalizationConst)
    end
end



AutoScore(DM::AbstractDataModel; kwargs...) = θ::AbstractVector{<:Number}->AutoScore(DM, θ; kwargs...)
AutoScore(DM::AbstractDataModel, θ::AbstractVector{<:Number}, LogPriorFn::Union{Nothing, Function}=LogPrior(DM); kwargs...) = AutoScore(Data(DM), Predictor(DM), θ, LogPriorFn; kwargs...)
AutoScore(DS::AbstractDataSet, model::ModelOrFunction, θ::AbstractVector{<:Number}, LogPriorFn::Nothing; kwargs...) = _AutoScore(DS, model, θ; kwargs...)
function AutoScore(DS::AbstractDataSet, model::ModelOrFunction, θ::AbstractVector{<:Number}, LogPriorFn::Function; ADmode::Union{Symbol,Val}=Val(:ForwardDiff), kwargs...)
    _AutoScore(DS, model, θ; ADmode=ADmode, kwargs...) + EvalLogPriorGrad(LogPriorFn, θ; ADmode=ADmode)
end
_AutoScore(DS::AbstractDataSet, model::ModelOrFunction, θ::AbstractVector{<:Number}; ADmode::Union{Symbol,Val}=Val(:ForwardDiff), kwargs...) = GetGrad(ADmode, x->_loglikelihood(DS, model, x; kwargs...))(θ)

AutoScore!(S::AbstractVector, args...; kwargs...) = copyto!(S, AutoScore(args...; kwargs...))
AutoScore!(DM::AbstractDataModel; kwargs...) = (S::AbstractVector,θ::AbstractVector)->AutoScore!(S, DM, θ; kwargs...)


AutoMetric(DM::AbstractDataModel; kwargs...) = θ::AbstractVector{<:Number}->AutoMetric(DM, θ; kwargs...)
AutoMetric(DM::AbstractDataModel, θ::AbstractVector{<:Number}, LogPriorFn::Union{Nothing, Function}; kwargs...) = AutoMetric(Data(DM), Predictor(DM), θ, LogPriorFn; kwargs...)
AutoMetric(DS::AbstractDataSet, model::ModelOrFunction, θ::AbstractVector{<:Number}, LogPriorFn::Nothing; kwargs...) = _AutoMetric(DS, model, θ; kwargs...)
function AutoMetric(DS::AbstractDataSet, model::ModelOrFunction, θ::AbstractVector{<:Number}, LogPriorFn::Function; ADmode::Union{Symbol,Val}=Val(:ForwardDiff), kwargs...)
    _AutoMetric(DS, model, θ; ADmode=ADmode, kwargs...) - EvalLogPriorHess(LogPriorFn, θ; ADmode=ADmode)
end
_AutoMetric(DS::AbstractDataSet, model::ModelOrFunction, θ::AbstractVector{<:Number}; ADmode::Union{Symbol,Val}=Val(:ForwardDiff), kwargs...) = GetHess(ADmode, x->-_loglikelihood(DS, model, x; kwargs...))(θ)

AutoMetric!(H::AbstractMatrix, args...; kwargs...) = copyto!(H, AutoMetric(args...; kwargs...))
AutoMetric!(DM::AbstractDataModel; kwargs...) = (H::AbstractMatrix,θ::AbstractVector)->AutoMetric!(H, DM, θ; kwargs...)

AutoMetric(DM::AbstractDataModel, θ::AbstractVector{<:Number}; kwargs...) = CostHessian(DM)(θ; kwargs...)
CostHessian(DM::AbstractDataModel, θ::AbstractVector{<:Number}; kwargs...) = CostHessian(DM)(θ; kwargs...)



## Generate Hessian from given Gradient with ADmode
## E.g. when cost not AutoDiffble but Score manually provided
## For FiniteDifferences and similar, much faster than using GetHess on original scalar function
AutoMetricFromScore(DM::AbstractDataModel, S::Function=Score(DM); kwargs...) = AutoMetricFromScore(S; kwargs...)
AutoMetricFromScore(S::Function; ADmode::Val=Val(:ForwardDiff)) = MergeOneArgMethods(GetJac(ADmode, Negate(S)), GetJac!(ADmode, Negate!!(S)))
# No negation
AutoMetricFromNegScore(N::Function; ADmode::Val=Val(:ForwardDiff)) = MergeOneArgMethods(GetJac(ADmode, N), GetJac!(ADmode, N))



Score(DM::AbstractDataModel; kwargs...) = LogLikelihoodGradient(θ::AbstractVector{<:Number}; Kwargs...) = Score(DM, θ; kwargs..., Kwargs...)
NegScore(DM::AbstractDataModel; kwargs...) = NegateBoth(Score(DM; kwargs...))

"""
    Score(DM::DataModel, θ::AbstractVector{<:Number}; ADmode::Val=Val(:ForwardDiff))
Calculates the gradient of the log-likelihood ``\\ell`` with respect to a set of parameters ``\\theta``.
`ADmode=Val(false)` computes the Score using the Jacobian `dmodel` provided in `DM`, i.e. by having to separately evaluate both the `model` as well as `dmodel`.
Other choices of `ADmode` directly compute the Score by differentiating the formula the log-likelihood, i.e. only one evaluation on a dual variable is performed.
"""
Score(DM::AbstractDataModel, θ::AbstractVector{<:Number}, LogPriorFn::Union{Nothing,Function}; kwargs...) = (@warn "Will deprecate this Score method soon!";   Score(Data(DM), Predictor(DM), dPredictor(DM), θ, LogPriorFn; kwargs...))
Score(DM::AbstractDataModel, θ::AbstractVector{<:Number}; kwargs...) = Score(DM)(θ; kwargs...)

Score(DS::AbstractDataSet, model::ModelOrFunction, dmodel::ModelOrFunction, θ::AbstractVector{<:Number}, LogPriorFn::Nothing; kwargs...) = _Score(DS, model, dmodel, θ; kwargs...)
Score(DS::AbstractDataSet, model::ModelOrFunction, dmodel::ModelOrFunction, θ::AbstractVector{<:Number}, LogPriorFn::Function; kwargs...) = _Score(DS, model, dmodel, θ; kwargs...) + EvalLogPriorGrad(LogPriorFn, θ)



_Score(DS::AbstractDataSet, model::ModelOrFunction, dmodel::ModelOrFunction, θ::AbstractVector{<:Number}; ADmode::Val=Val(:ForwardDiff), kwargs...) = _Score(DS, model, dmodel, θ, ADmode; kwargs...)
# Delegate to AutoScore
_Score(DS::AbstractDataSet, model::ModelOrFunction, dmodel::ModelOrFunction, θ::AbstractVector{<:Number}, ADmode::Val{true}; kwargs...) = _AutoScore(DS, model, θ; ADmode=ADmode, kwargs...)
_Score(DS::AbstractDataSet, model::ModelOrFunction, dmodel::ModelOrFunction, θ::AbstractVector{<:Number}, ADmode::Union{Symbol,Val}; kwargs...) = _AutoScore(DS, model, θ; ADmode=ADmode, kwargs...)

# Specialize this for different DataSet types
function _Score(DS::AbstractDataSet, model::ModelOrFunction, dmodel::ModelOrFunction, θ::AbstractVector{<:Number}, ADmode::Val{false}; kwargs...)
    mul3(A,B,C) = A * (B * C)
    mul3(transpose(EmbeddingMatrix(DS,dmodel,θ; kwargs...)), yInvCov(DS), (ydata(DS) - EmbeddingMap(DS,model,θ; kwargs...)))
    # transpose(EmbeddingMatrix(DS,dmodel,θ; kwargs...)) * (yInvCov(DS) * (ydata(DS) - EmbeddingMap(DS,model,θ; kwargs...)))
end

# Checks whether Val(false) should be used for score, in order to be safe.
# Slower than direct AD but safe since it reuses the given dmodel
const SafeScores = [:ForwardDiff, :FiniteDiff, :FiniteDifferences, :ReverseDiff, :Zygote]
UnsafeScore(S::Symbol) = S ∉ SafeScores
UnsafeScore(B::Bool) = !B
UnsafeScore(V::Val{T}) where T = UnsafeScore(T)

function GetScoreFn(DS::AbstractDataSet, model::ModelOrFunction, dmodel::ModelOrFunction, LogPriorFn::Union{Nothing,Function}, LogLikelihoodFn::Function; ADmode::Union{Symbol,Val}=Val(:ForwardDiff), SafeScore::Bool=UnsafeScore(ADmode), Kwargs...)
    if !SafeScore
        # In this case, performance gain from in-place score is usually marginal since it uses out-of-place EmbeddingMap
        ADS, ADS! = GetGrad(ADmode, LogLikelihoodFn), GetGrad!(ADmode, LogLikelihoodFn)
        """
            LogLikelihoodGradient(θ::AbstractVector; ADmode::Val=Val(:ForwardDiff), kwargs...) -> Vector
            LogLikelihoodGradient(dl::AbstractVector, θ::AbstractVector; ADmode::Val=Val(:ForwardDiff), kwargs...) -> Vector
        Calculates the gradient of the log-likelihood ``\\ell`` with respect to a set of parameters ``\\theta``.
        Both an out-of-place as well as an in-place method is provided.
        !!! note
            The `ADmode` kwarg can be used to switch backends. For more information on the currently loaded backends, see `diff_backends()`.
        """
        LogLikelihoodGradient(θ::AbstractVector{<:Number}; kwargs...) = ADS(θ; Kwargs..., kwargs...)
        LogLikelihoodGradient(dl::AbstractVector{<:Number}, θ::AbstractVector{<:Number}; kwargs...) = ADS!(dl, θ; Kwargs..., kwargs...)
    else
        @warn "The currently provided in-place method for Score is fake since fallback SafeScore=true was chosen."
        V = Val(false)
        S = ((θ::AbstractVector; ADmode=V, kwargs...) -> _Score(DS, model, dmodel, θ, V; Kwargs..., kwargs...))
        # This is where the buffer magic should happen for inplace EmbeddingMap! and EmbeddingMatrix!
        S! = ((dl::AbstractVector, θ::AbstractVector; ADmode=V, kwargs...) -> copyto!(dl,_Score(DS, model, dmodel, θ, V; Kwargs..., kwargs...)))

        if isnothing(LogPriorFn)
            """
                ScoreWithoutPrior(θ::AbstractVector; ADmode::Val=Val(:ForwardDiff), kwargs...) -> Vector
                ScoreWithoutPrior(dl::AbstractVector, θ::AbstractVector; ADmode::Val=Val(:ForwardDiff), kwargs...) -> Vector
            Calculates the gradient of the log-likelihood ``\\ell`` with respect to a set of parameters ``\\theta``. No prior was given for the parameters.
            Both an out-of-place as well as an in-place method is provided.
            !!! note
                The `ADmode` kwarg can be used to switch backends. For more information on the currently loaded backends, see `diff_backends()`.
            """
            ScoreWithoutPrior(θ::AbstractVector{<:Number}; kwargs...) = S(θ; kwargs...)
            ScoreWithoutPrior(dl::AbstractVector{<:Number}, θ::AbstractVector{<:Number}; kwargs...) = S!(dl, θ; kwargs...)
        else
            """
                ScoreWithPrior(θ::AbstractVector; ADmode::Val=Val(:ForwardDiff), kwargs...) -> Vector
                ScoreWithPrior(dl::AbstractVector, θ::AbstractVector; ADmode::Val=Val(:ForwardDiff), kwargs...) -> Vector
            Calculates the gradient of the log-likelihood ``\\ell`` with respect to a set of parameters ``\\theta``. No prior was given for the parameters.
            Both an out-of-place as well as an in-place method is provided.
            !!! note
                The `ADmode` kwarg can be used to switch backends. For more information on the currently loaded backends, see `diff_backends()`.
            """
            ScoreWithPrior(θ::AbstractVector{<:Number}; kwargs...) = S(θ; kwargs...) .+ EvalLogPriorGrad(LogPriorFn, θ; ADmode)
            function ScoreWithPrior(dl::AbstractVector{<:Number}, θ::AbstractVector{<:Number}; kwargs...)
                S!(dl, θ; kwargs...);    dl += EvalLogPriorGrad(LogPriorFn, θ; ADmode);    dl
            end
        end
    end
end

function GetFisherInfoFn(DS::AbstractDataSet, model::ModelOrFunction, dmodel::ModelOrFunction, LogPriorFn::Union{Nothing,Function}, LogLikelihoodFn::Function; ADmode::Union{Symbol,Val}=Val(:ForwardDiff), 
                                    UseHess::Bool=DS isa AbstractUnknownUncertaintyDataSet, Kwargs...)
    ## Pure autodiff typically slow since must be recompiled!
    if UseHess
        NL = Negate(LogLikelihoodFn);    F, F! = GetHess(ADmode, NL), GetHess!(ADmode, NL)
        FisherInformation(θ::AbstractVector{<:Number}; kwargs...) = F(θ; Kwargs..., kwargs...)
        FisherInformation(M::AbstractMatrix{<:Number}, θ::AbstractVector{<:Number}; kwargs...) = F!(M, θ; Kwargs..., kwargs...)
    else
        FisherMetricFn(θ::AbstractVector{<:Number}; kwargs...) = FisherMetric(DS, model, dmodel, θ, LogPriorFn; kwargs...)
        FisherMetricFn(M::AbstractMatrix{<:Number}, θ::AbstractVector{<:Number}; kwargs...) = copyto!(M, FisherMetricFn(θ; kwargs...))
    end
end

function GetFisherInfoFn(DMs::AbstractVector{<:AbstractDataModel}, Trafos::AbstractParameterTransformations, LogPriorFn::Union{Function,Nothing}; ADmode::Union{Symbol,Val}=Val(:ForwardDiff), Kwargs...)
    @assert length(DMs) == length(Trafos)
    Jac = DerivableFunctionsBase._GetJac(ADmode)
    ## For Fisher information, second derivatives of Parameter Trafos (multiplied with Score) drop out since expectation of score is zero
    ## For Hessians, have additional term H_paramtrafo * Score_i if parameter trafo non-linear
    # IsLinear = IsLinear(Trafos)
    function CompositeFisherMetricFn(θ::AbstractVector{<:Number}; kwargs...)
        sum((if Trafos[i] == identity
                FisherMetric(DM, θ; Kwargs..., kwargs...)
            else
                J = Jac(Trafos[i], θ)
                transpose(J) * FisherMetric(DM, Trafos[i](θ); Kwargs..., kwargs...) * J
            end) for (i,DM) in enumerate(DMs))
    end
    function CompositeFisherMetricFn(M::AbstractMatrix{<:Number}, θ::AbstractVector{T}; kwargs...) where T<:Number
        fill!(M, zero(T))
        for (i,DM) in enumerate(DMs)
            if Trafos[i] == identity
                M .+= FisherMetric(DM, θ; Kwargs..., kwargs...)
            else
                J = Jac(Trafos[i], θ)
                M .+= transpose(J) * FisherMetric(DM, Trafos[i](θ); Kwargs..., kwargs...) * J
            end
        end;    M
    end
    if isnothing(LogPriorFn)
        CompositeFisherMetricFn
    else
        CompositeFisherMetricFnWithPrior(θ::AbstractVector{<:Number}; kwargs...) = CompositeFisherMetricFn(θ; Kwargs..., kwargs...) .- EvalLogPriorHess(LogPriorFn, θ; ADmode)
        CompositeFisherMetricFnWithPrior(M::AbstractMatrix{<:Number}, θ::AbstractVector{<:Number}; kwargs...) = (CompositeFisherMetricFn(M, θ; Kwargs..., kwargs...);   M .-= EvalLogPriorHess(LogPriorFn, θ; ADmode);  M)
    end
end


"""
    GetRemainderFunction(DM::AbstractDataModel)
Returns remainder function ``R(θ) = ℓ(θ) - QuadraticApprox(θ)``.
"""
function GetRemainderFunction(DM::AbstractDataModel, mle::AbstractVector{<:Number}=MLE(DM); kwargs...)
    F = FisherMetric(DM, mle; kwargs...)
    QuadraticApprox(θ::AbstractVector{<:Number}) = LogLikeMLE(DM) - 0.5 * InformationGeometry.InnerProduct(F, θ - mle)
    Remainder(θ::AbstractVector{<:Number}) = loglikelihood(DM, θ) - QuadraticApprox(θ)
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
    ĥ(ξ::AbstractVector; kwargs...) = ĥ2(view(ξ,1:length(ξ)-pd), view(ξ,length(ξ)-pd+1:length(ξ)); kwargs...)
    ĥ2(xdat::AbstractVector, θ::AbstractVector{<:Number}; kwargs...) = [xdat; EmbeddingMap(DS, Model, θ, Windup(xdat, xdim(DS)); kwargs...)]
    ĥ
end
"""
    LiftedEmbeddingInplace(DM::AbstractDataModel) -> Function
Constructs lifted embedding map from initial space into extended dataspace ``\\hat{h} : \\mathcal{X}^N \\times \\mathcal{M} \\longrightarrow \\mathcal{X}^N \\times\\mathcal{Y}^N`` effecting
``\\xi = (x_\\text{opt}, \\theta) \\longmapsto \\hat{h}(\\xi) = (x_\\text{opt}, h(x_\\text{opt}, \\theta))``.
"""
LiftedEmbeddingInplace(DM::AbstractDataModel) = LiftedEmbeddingInplace(Data(DM), Predictor(DM), pdim(DM))
function LiftedEmbeddingInplace(DS::AbstractDataSet, Model::ModelOrFunction, pd::Int)
    function ĥ!(XY::AbstractVector, XP::AbstractVector; kwargs...)
        xlen = length(XP) - pd;     Xinds = 1:xlen;    Xs = view(XP, Xinds)
        copyto!(view(XY,Xinds), Xs)
        EmbeddingMap!(view(XY,xlen+1:length(XY)), DS, Model, view(XP, xlen+1:length(XP)), Windup(Xs, xdim(DS)); kwargs...)
    end
end

"""
    FullLiftedLogLikelihood(DM::AbstractDataModel) -> Function
Computes the full likelihood ``\\hat{\\ell} : \\mathcal{X}^N \\times\\mathcal{M}^N \\longrightarrow \\mathbb{R}`` given `Xθ` from initial space INCLUDING PRIOR.
"""
FullLiftedLogLikelihood(DM::AbstractDataModel; kwargs...) = FullLiftedLogLikelihood(Data(DM), Predictor(DM), LogPrior(DM), pdim(DM); kwargs...)
FullLiftedLogLikelihood(DS::AbstractDataSet, model::ModelOrFunction, LogPriorFn::Nothing, pd::Int; kwargs...) = LiftedLogLikelihood(DS)∘LiftedEmbedding(DS, model, pd; kwargs...)
function FullLiftedLogLikelihood(DS::AbstractDataSet, model::ModelOrFunction, LogPriorFn::Function, pd::Int; Kwargs...)
    L = LiftedLogLikelihood(DS)∘LiftedEmbedding(DS, model, pd; Kwargs...)
    ℓ(Xθ::AbstractVector{<:Number}; kwargs...) = L(Xθ; kwargs...) + EvalLogPrior(LogPriorFn, view(Xθ, length(Xθ)-pd+1:length(Xθ)))
end
FullLiftedNegLogLikelihood(args...; kwargs...) = Negate(FullLiftedLogLikelihood(args...; kwargs...))


"""
    FullFisherMetric(DM::AbstractDataModel, XP::AbstractVector; ADmode::Union{Val,Symbol}=Val(:ForwardDiff), kwargs...)
Computes Fisher metric, taking into account x-uncertainties.
!!! note
    Ignores error parameters and does not take into account priors.
"""
function FullFisherMetric(DM::AbstractDataModel, XP::AbstractVector; ADmode::Union{Val,Symbol}=Val(:ForwardDiff), kwargs...)
    J = GetJac(ADmode, LiftedEmbedding(DM))(XP);      transpose(J) * InvCov(dist(DM)) * J
end


"""
    GeneralizedDOF(DM::AbstractDataModel; meth=Newton(;linesearch=LineSearches.BackTracking()), MLE::AbstractVector{<:Number}=MLE(DM), kwargs...)
Computes the generalized degrees of freedom for a given `DataModel` which can take non-integer values and is defined in:
https://doi.org/10.1093/biomet/asv019
https://doi.org/10.2307/2669609
Its definition is based on the fact that the expected prediction error (EPE) is overestimated by the residual squared sum (RSS) by an amount `2σ^2 * dof`.
It can be computed as ``\\sum_{i=1}^{n} \\partial \\hat{y}_i / \\partial {y_\\text{data}}_i``.
"""
function GeneralizedDOF(DM::AbstractDataModel; meth=Newton(;linesearch=LineSearches.BackTracking()), ADmode::Union{Symbol,Val}=Val(:FiniteDifferences), MLE::AbstractVector{<:Number}=MLE(DM), kwargs...)
    NewData(DS::DataSet, y_data) = remake(DS; y=y_data)
    NewData(DS::DataSetUncertain{B}, y_data) where B = remake(DS; y=y_data, BesselCorrection=B)
    # NewData(DS::DataSetExact, y_data) = remake(DS; ydist=typeof(ydist(DS))(y_data, ))
    NewData(DS::AbstractDataSet, y_data) = throw("GDF not programmed for $(typeof(DS)) yet.")
    ChangeData(DM::AbstractDataModel, y_data; kwargs...) = (DS=NewData(Data(DM), y_data);   remake(DM; Data=DS, LogLikelihoodFn=GetLogLikelihoodFn(DS,Predictor(DM),LogPrior(DM)), SkipTests=true, SkipOptim=true, kwargs...))
    MLEgivenData(y_data) = InformationGeometry.minimize(Negloglikelihood(ChangeData(DM, y_data; MLE=MLE)), MLE, meth; kwargs...)
    MLEjac = GetJac(ADmode, MLEgivenData, length(ydata(DM)))
    LinearAlgebra.tr(EmbeddingMatrix(DM, MLE) * MLEjac(ydata(DM)))
end


function GeneralizedDOF(DM::AbstractConditionGrid; meth=Newton(;linesearch=LineSearches.BackTracking()), ADmode::Union{Symbol,Val}=Val(:FiniteDifferences), MLE::AbstractVector{<:Number}=MLE(DM), kwargs...)
    Inds = IndsVecFromLengths(length.(ydata.(Conditions(DM))));    mles = Trafos(DM)(MLE);    ydatas = ydata.(Conditions(DM))
    NewData(DS::DataSet, y_data) = remake(DS; y=y_data)
    NewData(DS::DataSetUncertain{B}, y_data) where B = remake(DS; y=y_data, BesselCorrection=B)
    NewData(DS::AbstractDataSet, y_data) = throw("GDF not programmed for $(typeof(DS)) yet.")
    ChangeData(DM::AbstractDataModel, y_data; kwargs...) = (DS=NewData(Data(DM), y_data);   remake(DM; Data=DS, LogLikelihoodFn=GetLogLikelihoodFn(DS,Predictor(DM),LogPrior(DM)), SkipTests=true, SkipOptim=true, kwargs...))
    function MLEgivenDataCG(y_data)
        C = Negloglikelihood(ConditionGrid([ChangeData(dm, y_data[Inds[i]]; MLE=collect(mles[i])) for (i,dm) in enumerate(Conditions(DM))], Trafos(DM), LogPrior(DM), MLE; SkipTests=true, SkipOptim=true, verbose=false))
        InformationGeometry.minimize(C, MLE, meth; kwargs...)
    end
    LinearAlgebra.tr(EmbeddingMatrix(DM, MLE) * (GetJac(ADmode, MLEgivenDataCG, sum(Inds))(reduce(vcat,ydatas))))
end