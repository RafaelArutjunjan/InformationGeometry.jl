


"""
In addition to storing a `DataSet`, a `DataModel` also contains a function `model(x,θ)` and its derivative `dmodel(x,θ)` where `x` denotes the x-value of the data and `θ` is a vector of parameters on which the model depends.
Crucially, `dmodel` contains the derivatives of the model with respect to the parameters `θ`, not the x-values.
For example
```julia
DS = DataSet([1,2,3,4], [4,5,6.5,7.8], [0.5,0.45,0.6,0.8])
model(x::Real, θ::AbstractVector{<:Real}) = θ[1] * x + θ[2]
DM = DataModel(DS, model)
```
In cases where the output of the model has more than one component (i.e. `ydim > 1`), it is advisable to define the model function in such a way that it outputs static vectors using **StaticArrays.jl** for increased performance.
For `ydim = 1`, **InformationGeometry.jl** expects the model to output a number instead of a vector with one component. In contrast, the parameter configuration `θ` must always be supplied as a vector.

A starting value for the maximum likelihood estimation can be passed to the `DataModel` constructor by appending an appropriate vector, e.g.
```julia
DM = DataModel(DS, model, [1.0,2.5])
```
During the construction of a `DataModel` process which includes the search for the maximum likelihood estimate ``\\theta_\\text{MLE}``, multiple tests are run. If necessary, these tests can be skipped by appending `true` as the last argument in the constructor:
```julia
DM = DataModel(DS, model, [-Inf,π,1+im], true)
```

If a `DataModel` is constructed as shown in the above examples, the gradient of the model with respect to the parameters `θ` (i.e. its "Jacobian") will be calculated using automatic differentiation. Alternatively, an explicit analytic expression for the Jacobian can be specified by hand:
```julia
using StaticArrays
function dmodel(x::Real, θ::AbstractVector{<:Real})
   @SMatrix [x  1.]     # ∂(model)/∂θ₁ and ∂(model)/∂θ₂
end
DM = DataModel(DS, model, dmodel)
```
The output of the Jacobian must be a matrix whose columns correspond to the partial derivatives with respect to different components of `θ` and whose rows correspond to evaluations at different components of `x`.
Again, although it is not strictly required, outputting the Jacobian in form of a static matrix is typically beneficial for the overall performance.

The `DataSet` contained in a `DataModel` named `DM` can be accessed via `Data(DM)`, whereas the model and its Jacobian can be used via `DM.model` and `DM.dmodel` respectively.
"""
struct DataModel <: AbstractDataModel
    Data::AbstractDataSet
    model::ModelOrFunction
    dmodel::ModelOrFunction
    MLE::AbstractVector{<:Number}
    LogLikeMLE::Real
    DataModel(DF::DataFrame, args...) = DataModel(DataSet(DF),args...)
    DataModel(DS::AbstractDataSet,model::ModelOrFunction,sneak::Bool=false) = DataModel(DS,model,DetermineDmodel(DS,model),sneak)
    DataModel(DS::AbstractDataSet,model::ModelOrFunction,mle::AbstractVector,sneak::Bool=false) = DataModel(DS,model,DetermineDmodel(DS,model),mle,sneak)
    function DataModel(DS::AbstractDataSet, model::ModelOrFunction, dmodel::ModelOrFunction, sneak::Bool=false)
        sneak ? DataModel(DS, model, dmodel, [-Inf,-Inf], true) : DataModel(DS, model, dmodel, FindMLE(DS,model,dmodel))
    end
    function DataModel(DS::AbstractDataSet,model::ModelOrFunction,dmodel::ModelOrFunction,mle::AbstractVector{<:Number},sneak::Bool=false)
        sneak && return DataModel(DS, model, dmodel, mle, (try loglikelihood(DS, model, mle) catch; -Inf end), true)
        MLE = FindMLE(DS, model, dmodel, mle);        LogLikeMLE = loglikelihood(DS, model, MLE)
        DataModel(DS, model, dmodel, MLE, LogLikeMLE)
    end
    # Check whether the determined MLE corresponds to a maximum of the likelihood unless sneak==true.
    function DataModel(DS::AbstractDataSet,model::ModelOrFunction,dmodel::ModelOrFunction,MLE::AbstractVector{<:Number},LogLikeMLE::Real,sneak::Bool=false)
        sneak && return new(DS, model, dmodel, MLE, LogLikeMLE)
        CheckModelHealth(DS, model)
        S = Score(DS, model, dmodel, MLE)
        norm(S) > 1e-5 && @warn "Norm of gradient of log-likelihood at supposed MLE=$MLE comparatively large: $(norm(S))."
        g = FisherMetric(DS, dmodel, MLE)
        det(g) == 0. && throw("Model appears to contain superfluous parameters since it is not structurally identifiable at supposed MLE=$MLE.")
        !isposdef(Symmetric(g)) && throw("Hessian of likelihood at supposed MLE=$MLE not negative-definite: Consider passing an appropriate initial parameter configuration 'init' for the estimation of the MLE to DataModel e.g. via DataModel(DS,model,init).")
        new(DS, model, dmodel, MLE, LogLikeMLE)
    end
end

# Specialized methods for DataModel

Data(DM::DataModel) = DM.Data
Predictor(DM::DataModel) = DM.model
dPredictor(DM::DataModel) = DM.dmodel

"""
    MLE(DM::DataModel) -> Vector
Returns the parameter configuration ``\\theta_\\text{MLE} \\in \\mathcal{M}`` which is estimated to have the highest likelihood of producing the observed data (under the assumption that the specified model captures the true relationship present in the data).
For performance reasons, the maximum likelihood estimate is stored as a part of the `DataModel` type.
"""
MLE(DM::DataModel) = DM.MLE
FindMLE(DM::DataModel, args...; kwargs...) = MLE(DM)

"""
    LogLikeMLE(DM::DataModel) -> Real
Returns the value of the log-likelihood ``\\ell`` when evaluated at the maximum likelihood estimate, i.e. ``\\ell(\\mathrm{data} \\, | \\, \\theta_\\text{MLE})``.
For performance reasons, this value is stored as a part of the `DataModel` type.
"""
LogLikeMLE(DM::DataModel) = DM.LogLikeMLE

pdim(DM::DataModel) = length(MLE(DM))


import Base: BigFloat
BigFloat(DM::DataModel) = DataModel(Data(DM), Predictor(DM), dPredictor(DM), BigFloat.(MLE(DM)))



InformNames(DM::AbstractDataModel, xnames::Vector{String}, ynames::Vector{String}) = DataModel(InformNames(Data(DM), xnames, ynames), Predictor(DM), dPredictor(DM), MLE(DM), LogLikeMLE(DM), true)
