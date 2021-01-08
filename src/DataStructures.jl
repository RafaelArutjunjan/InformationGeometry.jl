

function HealthyData(x::AbstractVector,y::AbstractVector)::Tuple{Int,Int,Int}
    length(x) != length(y) && throw(ArgumentError("Dimension mismatch. length(x) = $(length(x)), length(y) = $(length(y))."))
    # Check that dimensions of x-values and y-values are consistent
    xdim = length(x[1]);    ydim = length(y[1])
    sum(length(x[i]) != xdim   for i in 1:length(x)) > 0 && throw("Inconsistent length of x-values.")
    sum(length(y[i]) != ydim   for i in 1:length(y)) > 0 && throw("Inconsistent length of y-values.")
    return (length(x), xdim, ydim)
end

HealthyCovariance(sigma::AbstractVector{<:Real}) = !all(x->(0. < x),sigma) && throw("Some uncertainties not positive.")
HealthyCovariance(sigma::AbstractMatrix{<:Real}) = !isposdef(sigma) && throw("Covariance matrix not positive-definite.")


abstract type AbstractDataSet end
abstract type AbstractDataModel end
abstract type Cuboid end

"""
The `DataSet` type is a versatile container for storing data. Typically, it is constructed by passing it three vectors `x`, `y`, `sigma` where the components of `sigma` quantify the standard deviation associated with each y-value.
Alternatively, a full covariance matrix can be supplied for the `ydata` instead of a vector of standard deviations. The contents of a `DataSet` `DS` can later be accessed via `xdata(DS)`, `ydata(DS)`, `sigma(DS)`.

Examples:

In the simplest case, where all data points are mutually independent and have a single ``x``-component and a single ``y``-component each, a `DataSet` consisting of four points can be constructed via
```julia
DataSet([1,2,3,4], [4,5,6.5,7.8], [0.5,0.45,0.6,0.8])
```
or alternatively by
```julia
using LinearAlgebra
DataSet([1,2,3,4], [4,5,6.5,7.8], Diagonal([0.5,0.45,0.6,0.8].^2))
```
where the diagonal covariance matrix in the second line is equivalent to the vector of standard deviations supplied in the first line.

More generally, if a dataset consists of ``N`` points where each ``x``-value has ``n`` many components and each ``y``-value has ``m`` many components, this can be specified to the `DataSet` constructor via a tuple ``(N,n,m)`` in addition to the vectors `x`, `y` and the covariance matrix.
For example:
```julia
X = [0.9, 1.0, 1.1, 1.9, 2.0, 2.1, 2.9, 3.0, 3.1, 3.9, 4.0, 4.1]
Y = [1.0, 5.0, 4.0, 8.0, 9.0, 13.0, 16.0, 20.0]
Cov = Diagonal([2.0, 4.0, 2.0, 4.0, 2.0, 4.0, 2.0, 4.0])
dims = (4, 3, 2)
DS = DataSet(X, Y, Cov, dims)
```
In this case, `X` is a vector consisting of the concatenated x-values (with 3 components each) for 4 different data points.
The values of `Y` are the corresponding concatenated y-values (with 2 components each) of said 4 data points. Clearly, the covariance matrix must therefore be a positive-definite ``(m \\cdot N) \\times (m \\cdot N)`` matrix.
"""
struct DataSet <: AbstractDataSet
    x::AbstractVector
    y::AbstractVector
    InvCov::AbstractMatrix
    dims::Tuple{Int,Int,Int}
    logdetInvCov::Real
    WoundX::Union{AbstractVector,Bool}
    xnames::Vector{String}
    ynames::Vector{String}
    function DataSet(DF::Union{DataFrame,AbstractMatrix})
        size(DF,2) > 3 && throw("Unclear dimensions of input $DF.")
        DataSet(ToCols(convert(Matrix,DF))...)
    end
    function DataSet(x::AbstractVector,y::AbstractVector)
        println("No uncertainties in the y-values were specified for this DataSet, assuming σ=1 for all y's.")
        DataSet(x,y,ones(length(y)*length(y[1])))
    end
    DataSet(x::AbstractVector{<:Real}, y::AbstractVector{<:Measurement}) = DataSet(x,[y[i].val for i in 1:length(y)],[y[i].err for i in 1:length(y)])
    DataSet(x::AbstractVector, y::AbstractVector, sigma::AbstractArray) = DataSet(x,y,sigma,HealthyData(x,y))
    function DataSet(x::AbstractVector, y::AbstractVector, sigma::AbstractVector, dims::Tuple{Int,Int,Int})
        Sigma = Unwind(sigma)
        DataSet(Unwind(x), Unwind(y), Sigma, Diagonal([Sigma[i]^(-2) for i in 1:length(Sigma)]), dims)
    end
    DataSet(x::AbstractVector,y::AbstractVector,sigma::AbstractMatrix,dims::Tuple{Int,Int,Int}) = DataSet(Unwind(x),Unwind(y),sigma,inv(sigma),dims)
    function DataSet(x::AbstractVector{<:Real},y::AbstractVector{<:Real},sigma::AbstractArray{<:Real},InvCov::AbstractMatrix{<:Real},dims::Tuple{Int,Int,Int})
        !all(x->(x > 0), dims) && throw("Not all dims > 0: $dims.")
        !(Npoints(dims) == Int(length(x)/xdim(dims)) == Int(length(y)/ydim(dims)) == Int(size(InvCov,1)/ydim(dims))) && throw("Inconsistent input dimensions.")
        x = float.(x);  y = float.(y);  InvCov = float.(InvCov)
        InvCov = isdiag(InvCov) ? Diagonal(InvCov) : InvCov
        if !isposdef(InvCov)
            println("Inverse covariance matrix not perfectly positive-definite. Using only upper half and symmetrizing.")
            !isposdef(Symmetric(InvCov)) && throw("Inverse covariance matrix still not positive-definite after symmetrization.")
            InvCov = convert(Matrix, Symmetric(InvCov))
        end
        if xdim(dims) == 1
            return DataSet(x, y, InvCov, dims, logdet(InvCov), false)
        else
            # return new(x,y,InvCov,dims,logdet(InvCov),collect(Iterators.partition(x,xdim(dims))))
            return DataSet(x, y, InvCov, dims, logdet(InvCov), [SVector{xdim(dims)}(Z) for Z in Windup(x,xdim(dims))])
        end
    end
    function DataSet(x::AbstractVector, y::AbstractVector, InvCov::AbstractMatrix, dims::Tuple{Int,Int,Int}, logdetInvCov::Real, WoundX::Union{AbstractVector,Bool})
        DataSet(x, y, InvCov, dims, logdetInvCov, WoundX, CreateSymbolNames(xdim(dims),"x"), CreateSymbolNames(ydim(dims),"y"))
    end
    function DataSet(x::AbstractVector, y::AbstractVector, InvCov::AbstractMatrix, dims::Tuple{Int,Int,Int},
                            logdetInvCov::Real, WoundX::Union{AbstractVector,Bool}, xnames::Vector{String}, ynames::Vector{String})
        new(x, y, InvCov, dims, logdetInvCov, WoundX, xnames, ynames)
    end
end


# Specialized methods for DataSet
dims(DS::DataSet) = DS.dims
xdata(DS::DataSet) = DS.x
ydata(DS::DataSet) = DS.y
function sigma(DS::DataSet)
    sig = !issparse(InvCov(DS)) ? inv(InvCov(DS)) : inv(convert(Matrix,InvCov(DS)))
    return isdiag(sig) ? sqrt.(Diagonal(sig).diag) : sig
end
xsigma(DS::DataSet) = zeros(length(DS))
ysigma(DS::DataSet) = sigma(DS)

InvCov(DS::DataSet) = DS.InvCov
WoundX(DS::DataSet) = xdim(DS) < 2 ? xdata(DS) : DS.WoundX
logdetInvCov(DS::DataSet) = DS.logdetInvCov

xnames(DS::DataSet) = DS.xnames
ynames(DS::DataSet) = DS.ynames

function InformNames(DS::DataSet, xnames::Vector{String}, ynames::Vector{String})
    (length(xnames) != xdim(DS) || length(ynames) != ydim(DS)) && throw("Error.")
    DataSet(xdata(DS), ydata(DS), InvCov(DS), (Npoints(DS),xdim(DS),ydim(DS)), logdetInvCov(DS), WoundX(DS), xnames, ynames)
end


# Callback triggers when Boundaries is `true`.
"""
Container for model functions which carries additional information, e.g. about the parameter domain on which it is valid.
"""
struct ModelMap
    Map::Function
    InDomain::Function
    Domain::Union{Cuboid,Bool}
    xyp::Tuple{Int,Int,Int}
    ParamNames::Vector{String}
    StaticOutput::Val
    inplace::Val
    CustomEmbedding::Val
    # Given: Bool-valued domain function
    function ModelMap(model::Function, InDomain::Function, xyp::Tuple{Int,Int,Int}; ParamNames::Union{Vector{String},Bool}=false)
        ModelMap(model, InDomain, false, xyp; ParamNames=ParamNames)
    end
    # Given: HyperCube
    function ModelMap(model::Function, Domain::Cuboid, xyp::Union{Tuple{Int,Int,Int},Bool}=false; ParamNames::Union{Vector{String},Bool}=false)
        # Change this to θ -> true to avoid double checking cuboid. Obviously make sure Boundaries() is constructed using both the function test
        # and the Cuboid test first before changing this.
        InDomain(θ::AbstractVector{<:Number})::Bool = θ ∈ Domain
        typeof(xyp) == Bool ? ModelMap(model, InDomain, Domain; ParamNames=ParamNames) : ModelMap(model, InDomain, Domain, xyp; ParamNames=ParamNames)
    end
    # Given: Function only (potentially) -> Find xyp
    function ModelMap(model::Function, InDomain::Function=θ::AbstractVector{<:Number}->true, Domain::Union{Cuboid,Bool}=false; ParamNames::Union{Vector{String},Bool}=false)
        xyp = if typeof(Domain) == Bool
            xlen, plen = GetArgSize(model);     testout = model((xlen < 2 ? 1. : ones(xlen)),ones(plen))
            (xlen, size(testout,1), plen)
        else
            plen = length(Domain);      xlen = GetArgLength(x->model(x,ones(plen)));    testout = model((xlen < 2 ? 1. : ones(xlen)),ones(plen))
            (xlen, size(testout,1), plen)
        end
        ModelMap(model, InDomain, Domain, xyp; ParamNames=ParamNames)
    end
    function ModelMap(model::Function, InDomain::Function, Domain::Union{Cuboid,Bool}, xyp::Tuple{Int,Int,Int}; ParamNames::Union{Vector{String},Bool}=false)
        Domain = typeof(Domain) == Bool ? FullDomain(xyp[3]) : Domain
        ParamNames = typeof(ParamNames) == Bool ? CreateSymbolNames(xyp[3],"θ") : ParamNames
        StaticOutput = typeof(model((xyp[1] < 2 ? 1. : ones(xyp[1])), ones(xyp[3]))) <: SVector
        ModelMap(model, InDomain, Domain, xyp, ParamNames, Val(StaticOutput), Val(false), Val(false))
    end
    "Construct new ModelMap from function `F` with data from `M`."
    ModelMap(F::Function, M::ModelMap) = ModelMap(F, M.InDomain, M.Domain, M.xyp, M.ParamNames, M.StaticOutput, M.inplace, M.CustomEmbedding)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Careful with inheriting CustomEmbedding to the Jacobian! For automatically generated dmodels (symbolic or autodiff) it should be OFF!
    function ModelMap(Map::Function, InDomain::Function, Domain::Union{Cuboid,Bool}, xyp::Tuple{Int,Int,Int},
                        ParamNames::Vector{String}, StaticOutput::Val, inplace::Val=Val(false), CustomEmbedding::Val=Val(false))
        new(Map, InDomain, Domain, xyp, ParamNames, StaticOutput, inplace, CustomEmbedding)
    end
end
(M::ModelMap)(x, θ::AbstractVector{<:Number}; kwargs...) = M.Map(x, θ; kwargs...)
ModelOrFunction = Union{Function,ModelMap}

pnames(M::ModelMap) = M.ParamNames
Domain(M::ModelMap) = M.Domain

function ModelMap(F::Nothing, M::ModelMap)
    println("ModelMap: Got nothing instead of function to build new ModelMap")
    nothing
end
function CreateSymbolNames(n::Int, base::String="θ")
    n == 1 && return [base]
    D = Dict(string.(0:9) .=> ["₀","₁","₂","₃","₄","₅","₆","₇","₈","₉"])
    base .* [prod(get(D,"$x","Q") for x in string(digit)) for digit in 1:n]
end

pdim(DS::AbstractDataSet, model::ModelMap)::Int = model.xyp[3]
ModelMappize(DM::AbstractDataModel) = DataModel(Data(DM), ModelMap(Predictor(DM)), ModelMap(dPredictor(DM)), MLE(DM))
ModelMap(M::ModelMap) = M

function OutsideBoundariesFunction(M::ModelMap)
    OutsideBoundaries(u,t,int)::Bool = !((Res ∈ M.Domain) && M.InDomain(Res))
end

ValToBool(x::Val{true}) = true
ValToBool(x::Val{false}) = false
isinplace(M::ModelMap) = ValToBool(M.inplace)


Apply(x::AbstractVector{<:Number}, Componentwise::Function, indxs::BitVector) = [(indxs[i] ? Componentwise(x[i]) : x[i]) for i in eachindex(indxs)]
ApplyFull(x::AbstractVector{<:Number}, Vectorial::Function) = Vectorial(x)

MonotoneIncreasing(F::Function, Interval::Tuple{Real,Real})::Bool = Monotonicity(F, Interval) == :increasing
MonotoneDecreasing(F::Function, Interval::Tuple{Real,Real})::Bool = Monotonicity(F, Interval) == :decreasing
function Monotonicity(F::Function, Interval::Tuple{Real,Real})
    derivs = map(x->ForwardDiff.derivative(F, x), range(Interval[1], Interval[2]; length=200))
    all(x-> x≥0., derivs) && return :increasing
    all(x-> x≤0., derivs) && return :decreasing
    :neither
end

Transform(F::Function, indxs::BitVector, Transform::Function, InverseTransform::Function=x->invert(Transform,x)) = _Transform(F, indxs, Transform, InverseTransform)

# Try to do a bit of inference for the new domain here!
function Transform(M::ModelMap, indxs::BitVector, Transform::Function, InverseTransform::Function=x->invert(Transform,x))
    TranslatedDomain(θ::AbstractVector{<:Number}) = M.InDomain(Apply(θ, Transform, indxs))
    mono = Monotonicity(Transform, (1e-12,50.))
    NewCube = if mono == :increasing
        HyperCube(Apply(M.Domain.L, InverseTransform, indxs), Apply(M.Domain.U, InverseTransform, indxs))
    elseif mono == :decreasing
        println("Detected monotone decreasing transformation.")
        HyperCube(Apply(M.Domain.U, InverseTransform, indxs), Apply(M.Domain.L, InverseTransform, indxs))
    else
        FullDomain(length(indxs))
        @warn "Transformation does not appear to be monotone."
    end
    ModelMap(_Transform(M.Map, indxs, Transform, InverseTransform), TranslatedDomain, NewCube,
                        M.xyp, M.ParamNames, M.StaticOutput, M.inplace, M.CustomEmbedding)
end

function _Transform(F::Function, indxs::BitVector, Transform::Function, InverseTransform::Function)
    function TransformedModel(x::Union{Number, AbstractVector{<:Number}}, θ::AbstractVector{<:Number}; kwargs...)
        F(x, Apply(θ, Transform, indxs); kwargs...)
    end
end

LogTransform(F::ModelOrFunction, indxs::BitVector) = Transform(F, indxs, log, exp)
Log10Transform(F::ModelOrFunction, indxs::BitVector) = Transform(F, indxs, log10, x->10^x)
ReflectionTransform(F::ModelOrFunction, indxs::BitVector) = Transform(F, indxs, x-> -x, x-> -x)
ScaleTransform(F::ModelOrFunction, indxs::BitVector, factor::Real) = Transform(F, indxs, x->factor*x, x->x/factor)

function TranslationTransform(F::Function, v::AbstractVector{<:Number})
    TranslatedModel(x, θ::AbstractVector{<:Number}; kwargs...) = F(x, θ + v; kwargs...)
end
function TranslationTransform(M::ModelMap, v::AbstractVector{<:Number})
    ModelMap(TranslationTransform(M.Map, v), θ->M.InDomain(θ + v), TranslateCube(M.Domain, -v), M.xyp, M.ParamNames, M.StaticOutput,
                                    M.inplace, M.CustomEmbedding)
end

function LinearTransform(F::Function, A::AbstractMatrix{<:Number})
    TranslatedModel(x, θ::AbstractVector{<:Number}; kwargs...) = F(x, A*θ; kwargs...)
end

function LinearTransform(M::ModelMap, A::AbstractMatrix{<:Number})
    !isposdef(A) && println("Matrix in linear transform not positive definite.")
    Ainv = inv(A)
    ModelMap(LinearTransform(M.Map, A), θ->M.InDomain(A*θ), HyperCube(Ainv * M.Domain.L, Ainv * M.Domain.U),
                    M.xyp, M.ParamNames, M.StaticOutput, M.inplace, M.CustomEmbedding)
end

function AffineTransform(F::Function, A::AbstractMatrix{<:Number}, v::AbstractVector{<:Number})
    TranslatedModel(x, θ::AbstractVector{<:Number}; kwargs...) = F(x, A*θ + v; kwargs...)
end

function AffineTransform(M::ModelMap, A::AbstractMatrix{<:Number}, v::AbstractVector{<:Number})
    !isposdef(A) && println("Matrix in linear transform not positive definite.")
    Ainv = inv(A)
    ModelMap(AffineTransform(M.Map, A, v), θ->M.InDomain(A*θ+v), HyperCube(Ainv * (M.Domain.L-v), Ainv * (M.Domain.U-v)),
                    M.xyp, M.ParamNames, M.StaticOutput, M.inplace, M.CustomEmbedding)
end

# For dmodels, the output dim is ydim × pdim, i.e. xyp[2] × xyp[3].

"""
    GetArgSize(model::ModelOrFunction; max::Int=100)::Tuple{Int,Int}
Returns tuple `(xdim,pdim)` associated with the method `model(x,p)`.
"""
function GetArgSize(model::Function; max::Int=100)::Tuple{Int,Int}
    try         return (1, GetArgLength(p->model(1.,p); max=max))       catch; end
    for i in 2:(max + 1)
        plen = try      GetArgLength(p->model(ones(i),p); max=max)      catch; continue end
        i == (max + 1) ? throw("Wasn't able to find config.") : return (i, plen)
    end
end
GetArgSize(model::ModelMap; max::Int=100) = (model.xyp[1], model.xyp[3])



function AutoDiffDmodel(DS::AbstractDataSet, model::Function; custom::Bool=false)
    Autodmodel(x::Number,θ::AbstractVector{<:Number}; kwargs...) = transpose(ForwardDiff.gradient(z->model(x,z; kwargs...),θ))
    NAutodmodel(x::AbstractVector{<:Number},θ::AbstractVector{<:Number}; kwargs...) = transpose(ForwardDiff.gradient(z->model(x,z; kwargs...),θ))
    AutodmodelN(x::Number,θ::AbstractVector{<:Number}; kwargs...) = ForwardDiff.jacobian(p->model(x,p; kwargs...),θ)
    NAutodmodelN(x::AbstractVector{<:Number},θ::AbstractVector{<:Number}; kwargs...) = ForwardDiff.jacobian(p->model(x,p; kwargs...),θ)
    # Getting extract_gradient! error from ForwardDiff when using gradient method with observables
    # CustomAutodmodel(x::Union{Number,AbstractVector{<:Number}},θ::AbstractVector{<:Number}) = transpose(ForwardDiff.gradient(p->model(x,p),θ))
    CustomAutodmodelN(x::Union{Number,AbstractVector{<:Number}},θ::AbstractVector{<:Number}; kwargs...) = ForwardDiff.jacobian(p->model(x,p; kwargs...),θ)
    if ydim(DS) == 1
        custom && return CustomAutodmodelN
        return xdim(DS) == 1 ? Autodmodel : NAutodmodel
    else
        custom && return CustomAutodmodelN
        return xdim(DS) == 1 ? AutodmodelN : NAutodmodelN
    end
end


"""
    DetermineDmodel(DS::AbstractDataSet, model::Function)::Function
Returns appropriate function which constitutes the automatic derivative of the `model(x,θ)` with respect to the parameters `θ` depending on the format of the x-values and y-values of the DataSet.
"""
function DetermineDmodel(DS::AbstractDataSet, model::Function, TryOptimize::Bool=false; custom::Bool=false)
    # Try to use symbolic dmodel:
    if TryOptimize
        Symbolic_dmodel = Optimize(DS, model; inplace=false)[2]
        Symbolic_dmodel != nothing && return Symbolic_dmodel
    end
    AutoDiffDmodel(DS, model; custom=custom)
end
function DetermineDmodel(DS::AbstractDataSet, M::ModelMap, TryOptimize::Bool=false; custom::Bool=ValToBool(M.CustomEmbedding))
    ModelMap(DetermineDmodel(DS, M.Map, TryOptimize; custom=custom), M)
end


function CheckModelHealth(DS::AbstractDataSet, model::ModelOrFunction)
    P = ones(pdim(DS,model));   X = xdim(DS) < 2 ? xdata(DS)[1] : xdata(DS)[1:xdim(DS)]
    try  model(X,P)   catch Err
        throw("Got xdim=$(xdim(DS)) but model appears to not accept x-values of this size.")
    end
    !(size(model(X,P),1) == ydim(DS)) && println("Got ydim=$(ydim(DS)) but output of model does not have this size.")
    !(model(X,P) isa SVector) && ydim(DS) > 1 && @warn "It may be beneficial for the overall performance to define the model function such that it outputs static vectors, i.e. SVectors."
    return
end

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
        sneak ? DataModel(DS, model, dmodel, [-Inf,-Inf], true) : DataModel(DS, model, dmodel, FindMLE(DS,model))
    end
    function DataModel(DS::AbstractDataSet,model::ModelOrFunction,dmodel::ModelOrFunction,mle::AbstractVector{<:Number},sneak::Bool=false)
        sneak && return DataModel(DS, model, dmodel, mle, (try loglikelihood(DS, model, mle) catch; -Inf end), true)
        MLE = FindMLE(DS, model, mle);        LogLikeMLE = loglikelihood(DS, model, MLE)
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



# Need to implement for each DataSet:   xdata, ydata, sigma, xsigma, ysigma, InvCov, Npoints, xdim, ydim,
#                                       WoundX (already generic), logdetInvCov (already generic), length (already generic)
# Need to implement for each DataModel: the above and: Data, model (Predictor), dmodel (dPredictor),
#                                       pdim, MLE, LogLikeMLE,
#                                       EmbeddingMap (already (kind of) generic), EmbeddingMatrix (already (kind of) generic)
#                                       Score (already (kind of) generic), FisherMetric (already (kind of) generic)



# Generic Methods for AbstractDataSets      -----       May be superceded by more specialized functions!
import Base.length
length(DS::AbstractDataSet) = Npoints(DS)
WoundX(DS::AbstractDataSet) = Windup(xdata(DS),xdim(DS))
logdetInvCov(DS::AbstractDataSet) = logdet(InvCov(DS))
DataspaceDim(DS::AbstractDataSet) = Npoints(DS) * ydim(DS)
# Data(DS::AbstractDataSet) = DS

xdist(DS::AbstractDataSet) = xDataDist(DS)
ydist(DS::AbstractDataSet) = yDataDist(DS)

Npoints(DS::AbstractDataSet) = Npoints(dims(DS))
xdim(DS::AbstractDataSet) = xdim(dims(DS))
ydim(DS::AbstractDataSet) = ydim(dims(DS))
Npoints(dims::Tuple{Int,Int,Int}) = dims[1]
xdim(dims::Tuple{Int,Int,Int}) = dims[2]
ydim(dims::Tuple{Int,Int,Int}) = dims[3]


# Generic Methods for AbstractDataModels      -----       May be superceded by more specialized functions!
pdim(DM::AbstractDataModel) = pdim(Data(DM), Predictor(DM))
MLE(DM::AbstractDataModel) = FindMLE(DM)
LogLikeMLE(DM::AbstractDataModel) = loglikelihood(DM, MLE(DM))


# Generic passthrough of queries from AbstractDataModel to AbstractDataSet for following functions:
# for F in [xdata, ydata, sigma, xsigma, ysigma, InvCov, dims, Npoints, length, xdim, ydim,
#                     logdetInvCov, WoundX, DataspaceDim, xnames, ynames, xdist, ydist]
#     F(DM::AbstractDataModel) = F(Data(DM))
# end
xdata(DM::AbstractDataModel) = xdata(Data(DM))
ydata(DM::AbstractDataModel) = ydata(Data(DM))
sigma(DM::AbstractDataModel) = sigma(Data(DM))
xsigma(DM::AbstractDataModel) = xsigma(Data(DM))
ysigma(DM::AbstractDataModel) = ysigma(Data(DM))
InvCov(DM::AbstractDataModel) = InvCov(Data(DM))

Npoints(DM::AbstractDataModel) = Npoints(Data(DM))
length(DM::AbstractDataModel) = length(Data(DM))
xdim(DM::AbstractDataModel) = xdim(Data(DM))
ydim(DM::AbstractDataModel) = ydim(Data(DM))
dims(DM::AbstractDataModel) = dims(Data(DM))

logdetInvCov(DM::AbstractDataModel) = logdetInvCov(Data(DM))
WoundX(DM::AbstractDataModel) = WoundX(Data(DM))
DataspaceDim(DM::AbstractDataModel) = DataspaceDim(Data(DM))

xnames(DM::AbstractDataModel) = xnames(Data(DM))
ynames(DM::AbstractDataModel) = ynames(Data(DM))

xdist(DM::AbstractDataModel) = xdist(Data(DM))
ydist(DM::AbstractDataModel) = ydist(Data(DM))


# Generic Methods which are not simply passed through
pnames(DM::AbstractDataModel) = pnames(DM, Predictor(DM))
pnames(DM::AbstractDataModel, M::ModelMap) = pnames(M)
pnames(DM::AbstractDataModel, F::Function) = CreateSymbolNames(pdim(DM),"θ")


DataDist(Y::AbstractVector, Sig::AbstractVector, dist=Normal) = product_distribution([dist(Y[i],Sig[i]) for i in eachindex(Y)])
DataDist(Y::AbstractVector, Sig::AbstractMatrix, dist=MvNormal) = dist(Y, Symmetric(Sig))
yDataDist(DS::DataSet) = DataDist(ydata(DS), sigma(DS))
xDataDist(DS::DataSet) = InformationGeometry.Dirac(xdata(DS))
yDataDist(DM::DataModel) = yDataDist(Data(DM));    xDataDist(DM::DataModel) = xDataDist(Data(DM))



"""
    pdim(DS::AbstractDataSet, model::ModelOrFunction) -> Int
Infers the (minimal) number of components that the given function `F` accepts as input by successively testing it on vectors of increasing length.
"""
pdim(DS::AbstractDataSet, model::ModelOrFunction) = xdim(DS) < 2 ? GetArgLength(p->model(xdata(DS)[1],p)) : GetArgLength(p->model(xdata(DS)[1:xdim(DS)],p))

function GetArgLength(F::Function; max::Int=100)::Int
    max < 1 && throw("pdim: max = $max too small.")
    try     F(1.);  return 1    catch; end
    for i in 1:(max+1)
        try
            F(ones(i))
        catch y
            (isa(y, BoundsError) || isa(y, MethodError) || isa(y, DimensionMismatch) || isa(y, ArgumentError)) && continue
            println("pdim: Encountered error in specification of model function.");     rethrow()
        end
        i == (max + 1) ? throw(ArgumentError("pdim: Parameter space appears to have >$max dims. Aborting. Maybe wrong type of x was inserted?")) : return i
    end
end

LinearModel(x::Union{Real,AbstractVector{<:Real}}, θ::AbstractVector{<:Real}) = dot(θ[1:end-1], x) + θ[end]
QuadraticModel(x::Union{Real,AbstractVector{<:Real}}, θ::AbstractVector{<:Real}) = dot(θ[1:Int((end-1)/2)], x.^2) + dot(θ[Int((end-1)/2)+1:end-1], x) + θ[end]
ExponentialModel(x::Union{Real,AbstractVector{<:Real}}, θ::AbstractVector{<:Real}) = exp(LinearModel(x,θ))
SumExponentialsModel(x::Union{Real,AbstractVector{<:Real}},θ::AbstractVector{<:Real}) = sum(exp.(θ .* x))

import DataFrames.DataFrame
DataFrame(DM::DataModel) = DataFrame(Data(DM))
DataFrame(DS::DataSet) = SaveDataSet(DS)

import Base.join
function join(DS1::DataSet, DS2::DataSet)
    !(xdim(DS1) == xdim(DS2) && ydim(DS1) == ydim(DS2)) && throw("DataSets incompatible.")
    NewΣ = if typeof(sigma(DS1)) <: AbstractVector && typeof(sigma(DS2)) <: AbstractVector
        vcat(sigma(DS1), sigma(DS2))
    else
        BlockMatrix(sigma(DS1), sigma(DS2))
    end
    DataSet(vcat(xdata(DS1), xdata(DS2)), vcat(ydata(DS1), ydata(DS2)), NewΣ, (Npoints(DS1)+Npoints(DS2), xdim(DS1), ydim(DS1)))
end
join(DM1::DataModel, DM2::DataModel) = DataModel(join(Data(DM1),Data(DM2)), Predictor(DM1), dPredictor(DM1))
join(DS1::T, DS2::T, args...) where T <: Union{DataSet,DataModel} = join(join(DS1,DS2), args...)
join(DSVec::Vector{T}) where T <: Union{DataSet,DataModel} = join(DSVec...)

SortDataSet(DS::DataSet) = DS |> DataFrame |> sort |> DataSet
SortDataModel(DM::DataModel) = DataModel(SortDataSet(Data(DM)), Predictor(DM), dPredictor(DM))
function SubDataSet(DS::DataSet, range::Union{AbstractRange,AbstractVector})
    Npoints(DS) < length(range) && throw("Length of given range unsuitable for DataSet.")
    X = WoundX(DS)[range] |> Unwind
    Y = Windup(ydata(DS),ydim(DS))[range] |> Unwind
    Σ = sigma(DS)
    if typeof(Σ) <: AbstractVector
        Σ = Windup(Σ,ydim(DS))[range] |> Unwind
    elseif ydim(DS) == 1
        Σ = Σ[range,range]
    else
        throw("Under construction.")
    end
    DataSet(X,Y,Σ,(Int(length(X)/xdim(DS)),xdim(DS),ydim(DS)))
end
SubDataModel(DM::DataModel, range::Union{AbstractRange,AbstractVector}) = DataModel(SubDataSet(Data(DM),range), Predictor(DM), dPredictor(DM))

Sparsify(DS::DataSet) = SubDataSet(DS, rand(Bool,Npoints(DS)))
Sparsify(DM::DataModel) = SubDataSet(DS, rand(Bool,Npoints(DS)))


"""
Specifies a 2D plane in the so-called parameter form using 3 vectors.
"""
struct Plane
    stütz::AbstractVector
    Vx::AbstractVector
    Vy::AbstractVector
    function Plane(stütz::AbstractVector{<:Real}, Vx::AbstractVector{<:Real}, Vy::AbstractVector{<:Real})
        if length(stütz) == 2 stütz = [stütz[1],stütz[2],0] end
        if !(length(stütz) == length(Vx) == length(Vy))
            throw(ArgumentError("Dimension mismatch. length(stütz) = $(length(stütz)), length(Vx) = $(length(Vx)), length(Vy) = $(length(Vy))"))
        elseif dot(Vx,Vy) != 0
            println("Plane: Making Vy orthogonal to Vx.")
            return Plane(stütz, Vx, Make2ndOrthogonal(Vx,Vy))
        else
            stütz = SVector{length(Vx)}(float.(stütz));     Vx = SVector{length(Vx)}(float.(Vx))
            Vy = SVector{length(Vx)}(float.(Vy))
            return new(stütz, Vx, Vy)
        end
    end
end

length(PL::Plane) = length(PL.stütz)

function MLEinPlane(DM::AbstractDataModel, PL::Plane, start::AbstractVector{<:Number}=0.0001rand(2); tol::Real=1e-8)
    length(start) != 2 && throw("Dimensional Mismatch.");      model = Predictor(DM)
    planarmod(x, θ::AbstractVector{<:Number}; kwargs...) = model(x, PlaneCoordinates(PL,θ); kwargs...)
    curve_fit(Data(DM), planarmod, start; tol=tol).param
end

function PlanarDataModel(DM::DataModel, PL::Plane)
    model = Predictor(DM);      dmodel = dPredictor(DM)
    newmod = (x,θ::AbstractVector{<:Number}; kwargs...) -> model(x, PlaneCoordinates(PL,θ); kwargs...)
    dnewmod = (x,θ::AbstractVector{<:Number}; kwargs...) -> dmodel(x, PlaneCoordinates(PL,θ); kwargs...) * [PL.Vx PL.Vy]
    mle = MLEinPlane(DM, PL)
    DataModel(Data(DM), newmod, dnewmod, mle, loglikelihood(DM,PlaneCoordinates(mle)), true)
end

# Performance gains of using static vectors is lost if their length exceeds 32
BasisVectorSV(Slot::Int, dims::Int) = dims < 33 ? BasisVectorSVdo(Slot,dims) : BasisVector(Slot,dims)
BasisVectorSVdo(Slot::Int, dims::Int) = Slot > dims ? throw("Dimensional Mismatch.") : SVector{dims}(Float64(i == Slot) for i in 1:dims)
function BasisVector(Slot::Int, dims::Int)
    Res = zeros(dims);    Res[Slot] = 1.;    Res
end


"""
    PlaneCoordinates(PL::Plane, v::AbstractVector{<:Real})
Returns an n-dimensional vector from a tuple of two real numbers which correspond to the coordinates in the 2D `Plane`.
"""
PlaneCoordinates(PL::Plane, v::AbstractVector) = PL.stütz + [PL.Vx PL.Vy]*v

Shift(PlaneBegin::Plane, PlaneEnd::Plane) = TranslatePlane(PlaneEnd, PlaneEnd.stütz - PlaneBegin.stütz)

IsOnPlane(PL::Plane,x::AbstractVector)::Bool = (DistanceToPlane(PL,x) == 0)
TranslatePlane(PL::Plane, v::AbstractVector) = Plane(PL.stütz + v, PL.Vx, PL.Vy)
RotatePlane(PL::Plane, rads::Real=pi/2) = Plane(PL.stütz,cos(rads)*PL.Vx + sin(rads)*PL.Vy, cos(rads)*PL.Vy - sin(rads)*PL.Vx)
function RotationMatrix(PL::Plane,rads::Real)
    V = PL.Vx*transpose(PL.Vx) + PL.Vy*transpose(PL.Vy)
    W = PL.Vx*transpose(PL.Vy) - PL.Vy*transpose(PL.Vx)
    Diagonal(ones(length(PL.stütz))) + (cos(rads)-1.)*V -sin(rads)*W
end
RotateVector(PL::Plane,v::AbstractVector,rads::Real) = RotationMatrix(PL,rads)*v

function RotatedVector(α::Real,n1::Int,n2::Int,tot::Int)
    !(n1 <= tot && n2 <= tot && n1 != n2 && all(x->(x>0),[n1,n2,tot])) && throw("Error")
    res = zeros(tot);   res[n1] = cos(α);   res[n2] = sin(α);   res
end


function DecomposeWRTPlane(PL::Plane,x::AbstractVector)
    !IsOnPlane(PL,x) && throw(ArgumentError("Decompose Error: Vector not on Plane."))
    V = x - PL.stütz
    [ProjectOnto(V,PL.Vx), ProjectOnto(V,PL.Vy)]
end

DistanceToPlane(PL::Plane,x::AbstractVector) = (diagm(ones(Float64,length(x))) - ProjectionOperator(PL)) * (x - PL.stütz) |> norm
ProjectOntoPlane(PL::Plane,x::AbstractVector) = ProjectionOperator(PL) * (x - PL.stütz) + PL.stütz

function ProjectionOperator(A::AbstractMatrix)
    size(A,2) != 2 && println("ProjectionOperator: Matrix size $(size(A)) not as expected.")
    A * inv(transpose(A) * A) * transpose(A)
end
ProjectionOperator(PL::Plane) = ProjectionOperator([PL.Vx PL.Vy])

IsNormalToPlane(PL::Plane,v::AbstractVector)::Bool = (dot(PL.Vx,v) == dot(PL.Vy,v) == 0.)

function Make2ndOrthogonal(X::AbstractVector,Y::AbstractVector)
    Basis = GramSchmidt(float.([X,Y]))
    # Maybe add check for orientation?
    return Basis[2]
end

"""
    MinimizeOnPlane(PL::Plane,F::Function,initial::AbstractVector=[1,-1.]; tol::Real=1e-5)
Minimizes given function in Plane and returns the optimal point in the ambient space in which the plane lies.
"""
function MinimizeOnPlane(PL::Plane,F::Function,initial::AbstractVector=[1,-1.]; tol::Real=1e-5)
    G(x) = F(PlaneCoordinates(PL,x))
    X = Optim.minimizer(optimize(G,initial, BFGS(), Optim.Options(g_tol=tol), autodiff = :forward))
    PlaneCoordinates(PL,X)
end

"""
    ProjectOnto(v::Vector,u::Vector)
Project `v` onto `u`.
"""
ProjectOnto(v::AbstractVector,u::AbstractVector) = (dot(v,u) / dot(u,u)) * u

"""
    ParallelPlanes(PL::Plane,v::AbstractVector,range) -> Vector{Plane}
Returns Vector of Planes which have been translated by `a .* v` for all `a` in `range`.
"""
function ParallelPlanes(PL::Plane,v::AbstractVector,range::Union{AbstractRange,AbstractVector})
    norm(v) == 0. && throw("Vector has length zero.")
    # PL.Projector * v == v && throw("Plane and vector linearly dependent.")
    ProjectOntoPlane(PL,v) == v && throw("Plane and vector linearly dependent.")
    [TranslatePlane(PL, ran .* v) for ran in range]
end


function GramSchmidt(v::AbstractVector,dim::Int=length(v))
    Basis = Vector{suff(v)}(undef,0)
    push!(Basis,v)
    for i in 2:length(v)
        push!(Basis,BasisVector(i,length(v)))
    end
    GramSchmidt([normalize(Basis[i]) for i in 1:length(Basis)])
end
function GramSchmidt(Basis::AbstractVector{<:AbstractVector})
    ONBasis = float.(Basis)
    for j in 1:length(Basis)
        for i in 2:j
            ONBasis[j] .-= ProjectOnto(Basis[j],ONBasis[i-1])
        end
    end
    [normalize(ONBasis[i]) for i in 1:length(ONBasis)]
end



"""
The `HyperCube` type is used to specify a cuboid region in the form of a cartesian product of ``N`` real intervals, thereby offering a convenient way of passing domains for integration or plotting between functions.
A `HyperCube` object `cube` type has two fields: `cube.L` and `cube.U` which are two vectors which respectively store the lower and upper boundaries of the real intervals in order.
Examples for constructing `HyperCube`s:
```julia
HyperCube([[1,3],[pi,2π],[-500,100]])
HyperCube([1,π,-500],[3,2π,100])
HyperCube([[-1,1]])
HyperCube([-1,1])
HyperCube(collect([-7,7.] for i in 1:3))
```
Examples of quantities that can be computed from and operations involving a `HyperCube` object `X`:
```julia
CubeVol(X)
TranslateCube(X,v::Vector)
CubeWidths(X)
```
"""
struct HyperCube{Q<:Real} <: Cuboid
    L::AbstractVector{Q}
    U::AbstractVector{Q}
    function HyperCube(lowers::AbstractVector{<:Real},uppers::AbstractVector{<:Real}; Padding::Real=0.)
        length(lowers) != length(uppers) && throw("Dimensional Mismatch.")
        if Padding != 0.
            diff = (uppers - lowers) .* Padding
            lowers -= diff;     uppers += diff
        end
        !all(lowers .≤ uppers) && throw("First argument of HyperCube must be larger than second.")
        new{suff(lowers)}(float.(lowers),float.(uppers))
    end
    function HyperCube(H::AbstractVector{<:AbstractVector{<:Real}}; Padding::Real=0.)
        len = length(H[1]);        !all(x->(length(x) == len),H) && throw("Inconsistent lengths.")
        M = Unpack(H);        HyperCube(M[:,1],M[:,2]; Padding=Padding)
    end
    function HyperCube(vals::AbstractVector{<:Real}; Padding::Real=0.)
        length(vals) != 2 && throw("Input has too many components.")
        HyperCube([vals[1]],[vals[2]]; Padding=Padding)
    end
    HyperCube(vals::Tuple{<:Real,<:Real}; Padding::Real=0.) = HyperCube([vals[1]],[vals[2]]; Padding=Padding)
end

length(Cube::HyperCube) = length(Cube.L)

"""
    Inside(Cube::HyperCube, p::AbstractVector{<:Real}) -> Bool
Checks whether a point `p` lies inside `Cube`.
"""
Inside(Cube::HyperCube, p::AbstractVector{<:Real}) = all(Cube.L .≤ p) && all(p .≤ Cube.U)


import Base.in
"""
    in(Cube::HyperCube, p::AbstractVector{<:Real}) -> Bool
Checks whether a point `p` lies inside `Cube`.
"""
in(p::AbstractVector{<:Real}, Cube::HyperCube) = Inside(Cube, p)

"""
    ConstructCube(M::Matrix{<:Real}; Padding::Real=1/50) -> HyperCube
Returns a `HyperCube` which encloses the extrema of the columns of the input matrix.
"""
ConstructCube(M::AbstractMatrix{<:Real}; Padding::Real=0.) = HyperCube([minimum(M[:,i]) for i in 1:size(M,2)], [maximum(M[:,i]) for i in 1:size(M,2)]; Padding=Padding)
ConstructCube(V::AbstractVector{<:Real}; Padding::Real=0.) = HyperCube(extrema(V); Padding=Padding)
ConstructCube(PL::Plane,sol::ODESolution; Padding::Real=0.) = ConstructCube(Deplanarize(PL,sol; N=300); Padding=Padding)

function ConstructCube(sol::ODESolution, Npoints::Int=200; Padding::Real=0.)
    ConstructCube(Unpack(map(sol,range(sol.t[1],sol.t[end],length=Npoints))); Padding=Padding)
end

"""
    CubeWidths(H::HyperCube) -> Vector
Returns vector of widths of the `HyperCube`.
"""
CubeWidths(Cube::HyperCube) = Cube.U - Cube.L

"""
    CubeVol(Cube::HyperCube) -> Real
Computes volume of a `HyperCube` as the product of its sidelengths.
"""
CubeVol(Cube::HyperCube) = prod(CubeWidths(Cube))

"""
    Center(Cube::HyperCube) |> Vector
Returns center of mass of `Cube`.
"""
Center(Cube::HyperCube) = 0.5 * (Cube.L + Cube.U)

"""
    TranslateCube(Cube::HyperCube,x::Vector{<:Real}) -> HyperCube
Returns a `HyperCube` object which has been translated by `x`.
"""
TranslateCube(Cube::HyperCube, x::AbstractVector{<:Real}) = HyperCube(Cube.L + x, Cube.U + x)


### Slower than union
# """
#     CoverCubes(A::HyperCube, B::HyperCube)
# Return a new HyperCube which covers two other given HyperCubes.
# """
# function CoverCubes(A::HyperCube, B::HyperCube)
#     length(A) != length(B) && throw("CoverCubes: Cubes have different dims.")
#     lower = A.L; upper = A.U
#     for i in 1:length(A)
#         if A.L[i] > B.L[i]
#             lower[i] = B.L[i]
#         end
#         if A.U[i] < B.U[i]
#             upper[i] = B.U[i]
#         end
#     end
#     HyperCube(lower,upper)
# end
# CoverCubes(A::HyperCube, B::HyperCube) = Union(A, B)
# CoverCubes(args...) = CoverCubes([args...])
# CoverCubes(V::Vector{<:HyperCube}) = Union(V)

"""
    Intersect(A::HyperCube, B::HyperCube) -> HyperCube
    Intersect(Cubes::Vector{<:HyperCube}) -> HyperCube
Returns new `HyperCube` which is the intersection of the given `HyperCube`s.
"""
Intersect(A::HyperCube, B::HyperCube) = Intersect([A, B])
function Intersect(Cubes::Vector{<:HyperCube})
    LowerMatrix = [Cubes[i].L for i in 1:length(Cubes)] |> Unpack
    UpperMatrix = [Cubes[i].U for i in 1:length(Cubes)] |> Unpack
    HyperCube([maximum(col) for col in eachcol(LowerMatrix)], [minimum(col) for col in eachcol(UpperMatrix)])
end

"""
    Union(A::HyperCube, B::HyperCube) -> HyperCube
    Union(Cubes::Vector{<:HyperCube}) -> HyperCube
Returns new `HyperCube` which contains both given `HyperCube`s.
That is, the returned cube is strictly speaking not the union, but a cover (which contains the union).
"""
Union(A::HyperCube, B::HyperCube) = Union([A, B])
function Union(Cubes::Vector{<:HyperCube})
    LowerMatrix = [Cubes[i].L for i in 1:length(Cubes)] |> Unpack
    UpperMatrix = [Cubes[i].U for i in 1:length(Cubes)] |> Unpack
    HyperCube([minimum(col) for col in eachcol(LowerMatrix)], [maximum(col) for col in eachcol(UpperMatrix)])
end

import Base: union, intersect
union(A::HyperCube, B::HyperCube) = Union(A, B)
intersect(A::HyperCube, B::HyperCube) = Intersect(A, B)

import Base.==
==(A::HyperCube, B::HyperCube) = A.L == B.L && A.U == B.U
==(A::Plane, B::Plane) = A.stütz == B.stütz && A.Vx == B.Vx && A.Vy == B.Vy

PositiveDomain(n::Int) = HyperCube(zeros(n), fill(Inf,n))
PositiveDomain(indxs::BitVector) = HyperCube([(indxs[i] ? 0. : -Inf) for i in eachindex(indxs)], fill(Inf,length(indxs)))
NegativeDomain(n::Int) = HyperCube(fill(-Inf,n), zeros(n))
NegativeDomain(indxs::BitVector) = HyperCube(fill(-Inf,length(indxs)), [(indxs[i] ? 0. : Inf) for i in eachindex(indxs)])
FullDomain(n::Int) = HyperCube(fill(-Inf,n), fill(Inf,n))

import Base.rand
rand(Cube::HyperCube) = Cube.L + (Cube.U - Cube.L) .* rand(length(Cube.L))
