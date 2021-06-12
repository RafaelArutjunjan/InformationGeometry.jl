

function ConsistentElDims(X::AbstractVector)
    elDim = length(X[1])
    all(x -> length(x) == elDim, X) ? elDim : throw("Inconsistent element lengths for given Vector.")
end
ConsistentElDims(X::AbstractVector{<:Number}) = 1
ConsistentElDims(M::AbstractMatrix{<:Number}) = size(M,2)
ConsistentElDims(T::Tuple) = ConsistentElDims(collect(T))



function HealthyCovariance(M::AbstractMatrix{<:Number})
    M = size(M,1) > size(M,2) ? Unwind(M) : M
    M = if isdiag(M)
        Diagonal(float.(M))
    elseif sum(abs, M - Diagonal(float.(M))) < 1e-20
        Diagonal(float.(M))
    else
        float.(M)
    end
    if !isposdef(M)
        println("Given Matrix not perfectly positive-definite. Using only upper half and symmetrizing.")
        M = Symmetric(M)
        !isposdef(M) && throw("Matrix still not positive-definite after symmetrization.")
        M = convert(Matrix, M)
    end
    return M
end
HealthyCovariance(D::Diagonal) = all(x->x>0, D.diag) ? D : throw("Given covariance Matrix has non-positive values on diagonal: $(D.diag)")
# Interpret vector as uncertainties, therefore square before converting to Matrix
HealthyCovariance(X::AbstractVector{<:Number}) = all(x->x>0, X) ? Diagonal(float.(X).^2) : throw("Not all given uncertainties positive: $(X)")
HealthyCovariance(X::AbstractVector{<:AbstractVector{<:Number}}) = Unwind(X)



# Need to implement for each DataSet:   xdata, ydata, sigma, xsigma, ysigma, InvCov, Npoints, xdim, ydim, dims
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
#sigma(DS::AbstractDataSet) = ysigma(DS)
@deprecate sigma(x) ysigma(x) true

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
LogPrior(DM::AbstractDataModel) = x->0.0


# Generic passthrough of queries from AbstractDataModel to AbstractDataSet for following functions:
for F in [  :xdata, :ydata, :xsigma, :ysigma, :InvCov,
            :dims, :length, :Npoints, :xdim, :ydim, :DataspaceDim,
            :logdetInvCov, :WoundX, :xnames, :ynames, :xdist, :ydist]
    @eval $F(DM::AbstractDataModel) = $F(Data(DM))
end


# Generic Methods which are not simply passed through
pnames(DM::AbstractDataModel) = pnames(DM, Predictor(DM))
pnames(DM::AbstractDataModel, M::ModelMap) = pnames(M)
pnames(DM::AbstractDataModel, F::Function) = CreateSymbolNames(pdim(DM),"θ")

Domain(DM::AbstractDataModel) = Predictor(DM) isa ModelMap ? Domain(Predictor(DM)) : FullDomain(pdim(DM))


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
    # For the symbolically generated jacobians to work with MArrays, it requires ≥ v0.11.3 of SymbolicUtils.jl:  https://github.com/JuliaSymbolics/SymbolicUtils.jl/pull/286
    if TryOptimize
        Symbolic_dmodel = Optimize(DS, model; inplace=false)[2]
        Symbolic_dmodel != nothing && return Symbolic_dmodel
    end
    AutoDiffDmodel(DS, model; custom=custom)
end
function DetermineDmodel(DS::AbstractDataSet, M::ModelMap, TryOptimize::Bool=false; custom::Bool=iscustom(M))
    ModelMap(DetermineDmodel(DS, M.Map, TryOptimize; custom=custom), M)
end


function CheckModelHealth(DS::AbstractDataSet, model::ModelOrFunction)
    P = ones(pdim(DS,model));   X = xdim(DS) < 2 ? xdata(DS)[1] : xdata(DS)[1:xdim(DS)]
    try  model(X,P)   catch Err
        throw("Got xdim=$(xdim(DS)) but model appears to not accept x-values of this size.")
    end
    out = model(X,P)
    !(size(out,1) == ydim(DS)) && println("Got ydim=$(ydim(DS)) but output of model does not have this size.")
    !(out isa SVector || out isa MVector) && (1 < ydim(DS) < 90) && @info "It may be beneficial for the overall performance to define the model function such that it outputs static vectors, i.e. SVectors."
    return
end



DataDist(Y::AbstractVector, Sig::AbstractVector, dist=Normal) = product_distribution([dist(Y[i],Sig[i]) for i in eachindex(Y)])
DataDist(Y::AbstractVector, Sig::AbstractMatrix, dist=MvNormal) = dist(Y, HealthyCovariance(Sig))
yDataDist(DS::AbstractDataSet) = DataDist(ydata(DS), ysigma(DS))
xDataDist(DS::AbstractDataSet) = xsigma(DS) == zeros(Npoints(DS)*xdim(DS)) ? InformationGeometry.Dirac(xdata(DS)) : DataDist(xdata(DS), HealthyCovariance(xsigma(DS)))
yDataDist(DM::AbstractDataModel) = yDataDist(Data(DM))
xDataDist(DM::AbstractDataModel) = xDataDist(Data(DM))



"""
    pdim(DS::AbstractDataSet, model::ModelOrFunction) -> Int
Infers the (minimal) number of components that the given function `F` accepts as input by successively testing it on vectors of increasing length.
"""
pdim(DS::AbstractDataSet, model::ModelOrFunction) = xdim(DS) < 2 ? GetArgLength(p->model(xdata(DS)[1],p)) : GetArgLength(p->model(xdata(DS)[1:xdim(DS)],p))


# DataFrame(DM::DataModel) = DataFrame(Data(DM))
DataFrame(DS::AbstractDataSet; kwargs...) = SaveDataSet(DS; kwargs...)


import Base.join
function join(DS1::T, DS2::T) where T <: AbstractDataSet
    !(xdim(DS1) == xdim(DS2) && ydim(DS1) == ydim(DS2)) && throw("DataSets incompatible.")
    NewΣ = if typeof(ysigma(DS1)) <: AbstractVector && typeof(ysigma(DS2)) <: AbstractVector
        vcat(ysigma(DS1), ysigma(DS2))
    else
        BlockMatrix(ysigma(DS1), ysigma(DS2))
    end
    DataSet(vcat(xdata(DS1), xdata(DS2)), vcat(ydata(DS1), ydata(DS2)), NewΣ, (Npoints(DS1)+Npoints(DS2), xdim(DS1), ydim(DS1)))
end
join(DM1::AbstractDataModel, DM2::AbstractDataModel) = DataModel(join(Data(DM1),Data(DM2)), Predictor(DM1), dPredictor(DM1))
join(DS1::T, DS2::T, args...) where T <: Union{AbstractDataSet,AbstractDataModel} = join(join(DS1,DS2), args...)
join(DSVec::Vector{T}) where T <: Union{AbstractDataSet,AbstractDataModel} = join(DSVec...)

SortDataSet(DS::AbstractDataSet) = DS |> DataFrame |> sort |> DataSet
SortDataModel(DM::AbstractDataModel) = DataModel(SortDataSet(Data(DM)), Predictor(DM), dPredictor(DM), MLE(DM))
function SubDataSet(DS::AbstractDataSet, range::Union{AbstractVector{<:Int},BoolVector})
    @assert DS isa DataSet || xdist(DS) isa InformationGeometry.Dirac
    Npoints(DS) < length(range) && throw("Length of given range unsuitable for DataSet.")
    X = WoundX(DS)[range] |> Unwind
    Y = Windup(ydata(DS),ydim(DS))[range] |> Unwind
    Σ = ysigma(DS)
    if typeof(Σ) <: AbstractVector
        Σ = Windup(Σ,ydim(DS))[range] |> Unwind
    elseif ydim(DS) == 1
        Σ = Σ[range,range]
    else
        throw("Under construction.")
    end
    InformNames(DataSet(X,Y,Σ,(Int(length(X)/xdim(DS)),xdim(DS),ydim(DS))), xnames(DS), ynames(DS))
end
SubDataModel(DM::AbstractDataModel, range::Union{AbstractVector{<:Int},BoolVector}) = DataModel(SubDataSet(Data(DM),range), Predictor(DM), dPredictor(DM), MLE(DM))

import Base: getindex, lastindex
getindex(DS::AbstractDataSet, x) = SubDataSet(DS, x)
lastindex(DS::AbstractDataSet) = Npoints(DS)

Sparsify(DS::AbstractDataSet) = SubDataSet(DS, rand(Bool,Npoints(DS)))
Sparsify(DM::AbstractDataModel) = SubDataSet(DS, rand(Bool,Npoints(DS)))
