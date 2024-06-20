


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
    # LogLikelihoodFn::Function
    # ScoreFn::Function
    DataModel(DF::DataFrame, args...; kwargs...) = DataModel(DataSet(DF), args...; kwargs...)
    function DataModel(DS::AbstractDataSet,model::ModelOrFunction,SkipTests::Bool=false; custom::Bool=iscustom(model), ADmode::Union{Symbol,Val}=Val(:ForwardDiff),kwargs...)
        DataModel(DS,model,DetermineDmodel(DS,model; custom=custom, ADmode=ADmode), SkipTests; ADmode=ADmode, kwargs...)
    end
    function DataModel(DS::AbstractDataSet,model::ModelOrFunction,mle::AbstractVector,SkipTests::Bool=false; custom::Bool=iscustom(model), ADmode::Union{Symbol,Val}=Val(:ForwardDiff), kwargs...)
        DataModel(DS,model,DetermineDmodel(DS,model; custom=custom, ADmode=ADmode),mle,SkipTests; ADmode=ADmode, kwargs...)
    end
    function DataModel(DS::AbstractDataSet,model::ModelOrFunction,mle::AbstractVector,LogPriorFn::Union{Function,Nothing},SkipTests::Bool=false; custom::Bool=iscustom(model), ADmode::Union{Symbol,Val}=Val(:ForwardDiff), kwargs...)
        DataModel(DS, model, DetermineDmodel(DS, model; custom=custom, ADmode=ADmode), mle, LogPriorFn, SkipTests; ADmode=ADmode, kwargs...)
    end
    function DataModel(DS::AbstractDataSet, model::ModelOrFunction, dmodel::ModelOrFunction, SkipTests::Bool=false; tol::Real=1e-12, meth=LBFGS(), kwargs...)
        SkipTests ? DataModel(DS, model, dmodel, [-Inf,-Inf], true; kwargs...) : DataModel(DS, model, dmodel, FindMLE(DS,model,dmodel; tol=tol, meth=meth); kwargs...)
    end
    function DataModel(DS::AbstractDataSet,model::ModelOrFunction,dmodel::ModelOrFunction,mle::AbstractVector{<:Number},SkipTests::Bool=false; kwargs...)
        DataModel(DS, model, dmodel, mle, nothing, SkipTests; kwargs...)
    end
    function DataModel(DS::AbstractDataSet,model::ModelOrFunction,dmodel::ModelOrFunction,mle::AbstractVector{<:Number},logPriorFn::Union{Function,Nothing},SkipTests::Bool=false; tol::Real=1e-12, meth=LBFGS(), kwargs...)
        LogPriorFn = Prior(logPriorFn, mle)
        SkipTests && return DataModel(DS, model, dmodel, mle, (try loglikelihood(DS, model, mle, LogPriorFn) catch; -Inf end), true; kwargs...)
        MLE = FindMLE(DS, model, dmodel, mle, LogPriorFn; tol=tol, meth=meth);        LogLikeMLE = loglikelihood(DS, model, MLE, LogPriorFn)
        DataModel(DS, model, dmodel, MLE, LogLikeMLE, LogPriorFn, SkipTests; kwargs...)
    end
    function DataModel(DS::AbstractDataSet,model::ModelOrFunction,dmodel::ModelOrFunction,MLE::AbstractVector{<:Number},LogLikeMLE::Real,SkipTests::Bool=false; kwargs...)
        DataModel(DS, model, dmodel, MLE, LogLikeMLE, nothing, SkipTests; kwargs...)
    end
    # Block kwargs here.
    function DataModel(DS::AbstractDataSet,model::ModelOrFunction,dmodel::ModelOrFunction,MLE::AbstractVector{<:Number},LogLikeMLE::Real,LogPriorFn::Union{Function,Nothing},SkipTests::Bool=false;
                                            name::Union{Symbol,String}=Symbol(), ADmode::Union{Symbol,Val}=Val(:ForwardDiff))
        length(string(name)) > 0 && (@warn "DataModel does not have own 'name' field, forwarding to model.";    model=Christen(model, name))
        # length(MLE) < 20 && (MLE = SVector{length(MLE)}(MLE))
        !SkipTests && TestDataModel(DS, model, dmodel, MLE, LogLikeMLE, LogPriorFn)
        NewLogPriorFn = Prior(LogPriorFn,MLE)
        # LogLikelihoodFn = GetLogLikelihoodFn(DS,model,NewLogPriorFn)
        # ScoreFn = GetScoreFn(DS,model,dmodel,NewLogPriorFn,LogLikelihoodFn; ADmode=ADmode)
        new(DS, model, dmodel, MLE, LogLikeMLE, NewLogPriorFn) #, LogLikelihoodFn, ScoreFn)
    end
end

function TestDataModel(DS::AbstractDataSet,model::ModelOrFunction,dmodel::ModelOrFunction,MLE::AbstractVector{<:Number},LogLikeMLE::Real,LogPriorFn::Union{Function,Nothing}=nothing)
    CheckModelHealth(DS, model, MLE)
    if model isa ModelMap
        !IsInDomain(model, MLE) && @warn "Supposed MLE $MLE not inside valid parameter domain specified for ModelMap. Consider specifying an appropriate intial parameter configuration."
    end
    if LogPriorFn isa Function
        @assert LogPriorFn(MLE) isa Real
        !all(x->x ≤ 0.0, eigvals(EvalLogPriorHess(LogPriorFn, MLE))) && @warn "Hessian of specified LogPrior does not appear to be negative-semidefinite at MLE."
    end
    S = Score(DS, model, dmodel, MLE, LogPriorFn)
    norm(S) > sqrt(length(MLE)*1e-5) && @warn "Norm of gradient of log-likelihood at supposed MLE $MLE comparatively large: $(norm(S))."
    g = FisherMetric(DS, model, dmodel, MLE, LogPriorFn)
    logdet(g) == -Inf && @warn "Model appears to contain superfluous parameters since it is not structurally identifiable at supposed MLE $MLE."
    !isposdef(Symmetric(g)) && @warn "Hessian of likelihood at supposed MLE $MLE not negative-definite: Consider passing an appropriate initial parameter configuration 'init' for the estimation of the MLE to DataModel e.g. via DataModel(DS,model,init)."
end

# For SciMLBase.remake
DataModel(;
Data::AbstractDataSet=DataSet([0.], [0.], [1.]),
model::ModelOrFunction=(x,p)->-Inf,
dmodel::ModelOrFunction=(x,p)->[-Inf],
MLE::AbstractVector{<:Number}=[-Inf],
LogLikeMLE::Real=-Inf,
LogPrior::Union{Function,Nothing}=nothing,
# LogLikelihoodFn::Function=p->0.0,
# ScoreFn::Function=p->ones(length(p))
) = DataModel(Data, model, dmodel, MLE, LogLikeMLE, LogPrior) #, LogLikelihoodFn, ScoreFn)


# Specialized methods for DataModel

Data(DM::DataModel) = DM.Data
Predictor(DM::DataModel) = DM.model
dPredictor(DM::DataModel) = DM.dmodel
LogPrior(DM::DataModel) = DM.LogPrior
# loglikelihood(DM::DataModel) = DM.LogLikelihoodFn
# Score(DM::DataModel) = DM.ScoreFn

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


Base.BigFloat(DM::DataModel) = DataModel((try BigFloat(Data(DM)) catch; Data(DM) end), Predictor(DM), dPredictor(DM), BigFloat.(MLE(DM)))
Base.Float64(DM::DataModel) = DataModel(Data(DM), Predictor(DM), dPredictor(DM), Float64.(MLE(DM)))


InformNames(DM::AbstractDataModel, xnames::AbstractVector{String}, ynames::AbstractVector{String}) = DataModel(InformNames(Data(DM), xnames, ynames), Predictor(DM), dPredictor(DM), MLE(DM), LogLikeMLE(DM), LogPrior(DM), true)


# Dot not create Prior object when there is no prior.
Prior(Func::Nothing, args...; kwargs...) = nothing
# DFunction uses :Symbolic by default which can lead to problems
Prior(args...; ADmode::Union{Symbol,Val}=Val(:ForwardDiff), kwargs...) = DFunction(args...; ADmode=ADmode, kwargs...)
Prior(D::DFunction, args...; kwargs...) = D

EvalLogPrior(P, θ::AbstractVector{<:Number}; kwargs...) = EvalF(P, θ; kwargs...)
# EvalLogPriorGrad(P, θ::AbstractVector{<:Number}; kwargs...) = EvaldF(P, θ; kwargs...)
# EvalLogPriorHess(P, θ::AbstractVector{<:Number}; kwargs...) = EvalddF(P, θ; kwargs...)
EvalLogPriorGrad(P, θ::AbstractVector{<:Number}; kwargs...) = GetGrad(P; kwargs...)(θ)
EvalLogPriorHess(P, θ::AbstractVector{<:Number}; kwargs...) = GetHess(P; kwargs...)(θ)

EvalLogPrior(D::Nothing, x::AbstractVector{T}; kwargs...) where T<:Number = zero(T)
EvalLogPriorGrad(D::Nothing, x::AbstractVector{T}; kwargs...) where T<:Number = zeros(T, length(x))
EvalLogPriorHess(D::Nothing, x::AbstractVector{T}; kwargs...) where T<:Number = zeros(T, length(x), length(x))
