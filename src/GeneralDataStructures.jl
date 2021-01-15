

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
sigma(DS::AbstractDataSet) = ysigma(DS)

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



DataDist(Y::AbstractVector, Sig::AbstractVector, dist=Normal) = product_distribution([dist(Y[i],Sig[i]) for i in eachindex(Y)])
DataDist(Y::AbstractVector, Sig::AbstractMatrix, dist=MvNormal) = dist(Y, Symmetric(Sig))
yDataDist(DS::AbstractDataSet) = DataDist(ydata(DS), ysigma(DS))
xDataDist(DS::AbstractDataSet) = xsigma(DS) == zeros(Npoints(DS)*xdim(DS)) ? InformationGeometry.Dirac(xdata(DS)) : DataDist(xdata(DS), xsigma(DS))
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
    NewΣ = if typeof(sigma(DS1)) <: AbstractVector && typeof(sigma(DS2)) <: AbstractVector
        vcat(sigma(DS1), sigma(DS2))
    else
        BlockMatrix(sigma(DS1), sigma(DS2))
    end
    DataSet(vcat(xdata(DS1), xdata(DS2)), vcat(ydata(DS1), ydata(DS2)), NewΣ, (Npoints(DS1)+Npoints(DS2), xdim(DS1), ydim(DS1)))
end
join(DM1::AbstractDataModel, DM2::AbstractDataModel) = DataModel(join(Data(DM1),Data(DM2)), Predictor(DM1), dPredictor(DM1))
join(DS1::T, DS2::T, args...) where T <: Union{AbstractDataSet,AbstractDataModel} = join(join(DS1,DS2), args...)
join(DSVec::Vector{T}) where T <: Union{AbstractDataSet,AbstractDataModel} = join(DSVec...)

SortDataSet(DS::AbstractDataSet) = DS |> DataFrame |> sort |> DataSet
SortDataModel(DM::AbstractDataModel) = DataModel(SortDataSet(Data(DM)), Predictor(DM), dPredictor(DM), MLE(DM))
function SubDataSet(DS::AbstractDataSet, range::Union{AbstractRange,AbstractVector})
    @assert DS isa DataSet
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
SubDataModel(DM::AbstractDataModel, range::Union{AbstractRange,AbstractVector}) = DataModel(SubDataSet(Data(DM),range), Predictor(DM), dPredictor(DM), MLE(DM))

Sparsify(DS::AbstractDataSet) = SubDataSet(DS, rand(Bool,Npoints(DS)))
Sparsify(DM::AbstractDataModel) = SubDataSet(DS, rand(Bool,Npoints(DS)))
