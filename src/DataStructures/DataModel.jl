


"""
In addition to storing a `DataSet`, a `DataModel` also contains a function `model(x,θ)` and its derivative `dmodel(x,θ)` where `x` denotes the x-value of the data and `θ` is a vector of parameters on which the model depends.
Crucially, `dmodel` contains the derivatives of the model with respect to the parameters `θ`, not the x-values.
For example
```julia
DS = DataSet([1,2,3,4], [4,5,6.5,7.8], [0.5,0.45,0.6,0.8])
model(x::Number, θ::AbstractVector{<:Number}) = θ[1] * x + θ[2]
DM = DataModel(DS, model)
```
In cases where the output of the model has more than one component (i.e. `ydim > 1`), it is advisable to define the model function in such a way that it outputs static vectors using **StaticArrays.jl** for increased performance.
For `ydim = 1`, **InformationGeometry.jl** expects the model to output a number instead of a vector with one component. In contrast, the parameter configuration `θ` must always be supplied as a vector (even if it only has a single component).

An initial guess for the maximum likelihood parameters can optionally be passed to the `DataModel` as a vector via
```julia
DM = DataModel(DS, model, [1.0,2.5])
```
During the construction of a `DataModel` process which includes the search for the maximum likelihood estimate ``\\theta_\\text{MLE}``, multiple tests are run. If necessary, these tests can be skipped by appending `true` as the last argument in the constructor:
```julia
DM = DataModel(DS, model, [-Inf,π,1], true)
```

If a `DataModel` is constructed as shown in the above examples, the gradient of the model with respect to the parameters `θ` (i.e. its "Jacobian") will be calculated using automatic differentiation. Alternatively, an explicit analytic expression for the Jacobian can be specified by hand:
```julia
using StaticArrays
function dmodel(x::Number, θ::AbstractVector{<:Number})
   @SMatrix [x  1.]     # ∂(model)/∂θ₁ and ∂(model)/∂θ₂
end
DM = DataModel(DS, model, dmodel)
```
The output of the Jacobian must be a matrix whose columns correspond to the partial derivatives with respect to different components of `θ` and whose rows correspond to evaluations at different components of `x`.
Again, although it is not strictly required, outputting the Jacobian in form of a static matrix is typically beneficial for the overall performance.

It is also possible to specify a (logarithmized) prior distribution on the parameter space to the `DataModel` constructor after the initial guess for the MLE. For example:
```julia
using Distributions
Dist = MvNormal(ones(2), [1 0; 0 3.])
LogPriorFn(θ) = logpdf(Dist, θ)
DM = DataModel(DS, model, [1.0,2.5], LogPriorFn)
```

The `DataSet` contained in a `DataModel` named `DM` can be accessed via `Data(DM)`, whereas the model and its Jacobian can be used via `Predictor(DM)` and `dPredictor(DM)` respectively. The MLE and the value of the log-likelihood at the MLE are accessible via `MLE(DM)` and `LogLikeMLE(DM)`. The logarithmized prior can be accessed via `LogPrior(DM)`.
"""
struct DataModel <: AbstractDataModel
    Data::AbstractDataSet
    model::ModelOrFunction
    dmodel::ModelOrFunction
    MLE::AbstractVector{<:Number}
    LogLikeMLE::Real
    LogPrior::Union{Function,Nothing}
    DataModel(DF::DataFrame, args...; kwargs...) = DataModel(DataSet(DF), args...; kwargs...)
    DataModel(DS::AbstractDataSet,model::ModelOrFunction,SkipTests::Bool=false; kwargs...) = DataModel(DS,model,DetermineDmodel(DS,model; kwargs...), SkipTests)
    DataModel(DS::AbstractDataSet,model::ModelOrFunction,mle::AbstractVector,SkipTests::Bool=false; kwargs...) = DataModel(DS,model,DetermineDmodel(DS,model; kwargs...),mle,SkipTests)
    function DataModel(DS::AbstractDataSet,model::ModelOrFunction,mle::AbstractVector,LogPriorFn::Union{Function,Nothing},SkipTests::Bool=false; kwargs...)
        DataModel(DS, model, DetermineDmodel(DS,model; kwargs...), mle, LogPriorFn, SkipTests)
    end
    function DataModel(DS::AbstractDataSet, model::ModelOrFunction, dmodel::ModelOrFunction, SkipTests::Bool=false)
        SkipTests ? DataModel(DS, model, dmodel, [-Inf,-Inf], true) : DataModel(DS, model, dmodel, FindMLE(DS,model,dmodel))
    end
    function DataModel(DS::AbstractDataSet,model::ModelOrFunction,dmodel::ModelOrFunction,mle::AbstractVector{<:Number},SkipTests::Bool=false)
        DataModel(DS, model, dmodel, mle, nothing, SkipTests)
    end
    function DataModel(DS::AbstractDataSet,model::ModelOrFunction,dmodel::ModelOrFunction,mle::AbstractVector{<:Number},LogPriorFn::Union{Function,Nothing},SkipTests::Bool=false)
        SkipTests && return DataModel(DS, model, dmodel, mle, (try loglikelihood(DS, model, mle, LogPriorFn) catch; -Inf end), true)
        MLE = FindMLE(DS, model, dmodel, mle, LogPriorFn);        LogLikeMLE = loglikelihood(DS, model, MLE, LogPriorFn)
        DataModel(DS, model, dmodel, MLE, LogLikeMLE, LogPriorFn, SkipTests)
    end
    function DataModel(DS::AbstractDataSet,model::ModelOrFunction,dmodel::ModelOrFunction,MLE::AbstractVector{<:Number},LogLikeMLE::Real,SkipTests::Bool=false)
        DataModel(DS, model, dmodel, MLE, LogLikeMLE, nothing, SkipTests)
    end
    function DataModel(DS::AbstractDataSet,model::ModelOrFunction,dmodel::ModelOrFunction,MLE::AbstractVector{<:Number},LogLikeMLE::Real,LogPriorFn::Union{Function,Nothing},SkipTests::Bool=false)
        !SkipTests && TestDataModel(DS, model, dmodel, MLE, LogLikeMLE, LogPriorFn)
        new(DS, model, dmodel, MLE, LogLikeMLE, Prior(LogPriorFn))
    end
end

function TestDataModel(DS::AbstractDataSet,model::ModelOrFunction,dmodel::ModelOrFunction,MLE::AbstractVector{<:Number},LogLikeMLE::Real,LogPriorFn::Union{Function,Nothing}=nothing)
    if LogPriorFn isa Function
        @assert LogPriorFn(MLE) isa Real && LogPriorFn(MLE) ≤ 0.0
    end
    CheckModelHealth(DS, model)
    S = Score(DS, model, dmodel, MLE, LogPriorFn)
    norm(S) > sqrt(length(MLE))*1e-3 && @warn "Norm of gradient of log-likelihood at supposed MLE=$MLE comparatively large: $(norm(S))."
    g = FisherMetric(DS, dmodel, MLE, LogPriorFn)
    det(g) == 0. && @warn "Model appears to contain superfluous parameters since it is not structurally identifiable at supposed MLE=$MLE."
    !isposdef(Symmetric(g)) && throw("Hessian of likelihood at supposed MLE=$MLE not negative-definite: Consider passing an appropriate initial parameter configuration 'init' for the estimation of the MLE to DataModel e.g. via DataModel(DS,model,init).")
end

# For SciMLBase.remake
DataModel(;
Data::AbstractDataSet=DataSet([0.], [0.], [1.]),
model::ModelOrFunction=(x,p)->-Inf,
dmodel::ModelOrFunction=(x,p)->[-Inf],
MLE::AbstractVector{<:Number}=[-Inf],
LogLikeMLE::Real=-Inf,
LogPrior::Union{Function,Nothing}=nothing) = DataModel(Data, model, dmodel, MLE, LogLikeMLE, LogPrior)


# Specialized methods for DataModel

Data(DM::DataModel) = DM.Data
Predictor(DM::DataModel) = DM.model
dPredictor(DM::DataModel) = DM.dmodel
LogPrior(DM::DataModel) = DM.LogPrior

"""
    MLE(DM::DataModel) -> Vector
Returns the parameter configuration ``\\theta_\\text{MLE} \\in \\mathcal{M}`` which is estimated to have the highest likelihood of producing the observed data (under the assumption that the specified model captures the true relationship present in the data).
For performance reasons, the maximum likelihood estimate is stored as a part of the `DataModel` type.
"""
MLE(DM::DataModel) = DM.MLE

"""
    LogLikeMLE(DM::DataModel) -> Real
Returns the value of the log-likelihood ``\\ell`` when evaluated at the maximum likelihood estimate, i.e. ``\\ell(\\mathrm{data} \\, | \\, \\theta_\\text{MLE})``.
For performance reasons, this value is stored as a part of the `DataModel` type.
"""
LogLikeMLE(DM::DataModel) = DM.LogLikeMLE

pdim(DM::DataModel) = length(MLE(DM))


import Base: BigFloat
BigFloat(DM::DataModel) = DataModel(Data(DM), Predictor(DM), dPredictor(DM), BigFloat.(MLE(DM)))



InformNames(DM::AbstractDataModel, xnames::Vector{String}, ynames::Vector{String}) = DataModel(InformNames(Data(DM), xnames, ynames), Predictor(DM), dPredictor(DM), MLE(DM), LogLikeMLE(DM), LogPrior(DM), true)

struct Prior <: Function
    LogPriorFunc::Function
    LogPriorGrad::Function
    LogPriorHess::Function
end
Prior(Func::Nothing; ADmode::Union{Symbol,Val}=Val(:ForwardDiff)) = nothing
Prior(Func::Function; ADmode::Union{Symbol,Val}=Val(:ForwardDiff)) = Prior(Func, θ->GetGrad(ADmode)(Func,θ); ADmode=ADmode)
Prior(Func::Function, GradFunc::Function; ADmode::Union{Symbol,Val}=Val(:ForwardDiff)) = Prior(Func, GradFunc, θ->GetHess(ADmode)(Func,θ))
(P::Prior)(θ::AbstractVector{<:Number}) = P.LogPriorFunc(θ)

LogPrior(P::Prior) = P.LogPriorFunc
LogPriorGrad(P::Prior) = P.LogPriorGrad
LogPriorHess(P::Prior) = P.LogPriorHess

EvalLogPrior(P::Prior, θ::AbstractVector{<:Number}; kwargs...) = P.LogPriorFunc(θ)
EvalLogPriorGrad(P::Prior, θ::AbstractVector{<:Number}; kwargs...) = P.LogPriorGrad(θ)
EvalLogPriorHess(P::Prior, θ::AbstractVector{<:Number}; kwargs...) = P.LogPriorHess(θ)
