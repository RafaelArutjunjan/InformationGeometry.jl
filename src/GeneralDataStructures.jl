

function ConsistentElDims(X::AbstractVector)
    elDim = length(X[1])
    all(x -> length(x) == elDim, X) ? elDim : throw("Inconsistent element lengths for given Vector.")
end
ConsistentElDims(X::AbstractVector{<:Number}) = 1
ConsistentElDims(M::AbstractMatrix{<:Number}) = size(M,2)
ConsistentElDims(T::Tuple) = ConsistentElDims(collect(T))



function HealthyCovariance(M::AbstractMatrix{<:Number}; verbose::Bool=true, kwargs...)
    M = size(M,1) > size(M,2) ? Unwind(M) : M
    M = if isdiag(M)
        Diagonal(floatify(M))
    elseif sum(abs, M - Diagonal(floatify(M))) < 1e-20
        Diagonal(floatify(M))
    else
        floatify(M)
    end
    if !isposdef(M)
        verbose && @warn "Given Matrix not perfectly positive-definite. Using only upper half and symmetrizing."
        M = Symmetric(M)
        !isposdef(M) && throw("Matrix still not positive-definite after symmetrization.")
        M = convert(Matrix, M)
    end
    return M
end
HealthyCovariance(D::Diagonal; kwargs...) = all(x->x>0, D.diag) ? D : throw("Given covariance Matrix has non-positive values on diagonal: $(D.diag)")
# Interpret vector as uncertainties, therefore square before converting to Matrix
HealthyCovariance(X::AbstractVector{<:Number}; kwargs...) = all(x->x>0, X) ? Diagonal(floatify(X).^2) : throw("Not all given uncertainties positive: $(X)")
HealthyCovariance(X::AbstractVector{<:AbstractVector{<:Number}}; kwargs...) = Unwind(X)



# Need to implement for each DataSet:   xdata, ydata, sigma, xsigma, ysigma, InvCov, Npoints, xdim, ydim, dims
#                                       WoundX (already generic), logdetInvCov (already generic), length (already generic)
# Need to implement for each DataModel: the above and: Data, model (Predictor), dmodel (dPredictor),
#                                       pdim, MLE, LogLikeMLE,
#                                       EmbeddingMap (already (kind of) generic), EmbeddingMatrix (already (kind of) generic)
#                                       Score (already (kind of) generic), FisherMetric (already (kind of) generic)


# Generic Methods for AbstractDataSets      -----       May be superceded by more specialized functions!
Base.length(DS::AbstractDataSet) = Npoints(DS)
WoundX(DS::AbstractDataSet) = Windup(xdata(DS),xdim(DS))
WoundY(DS::AbstractDataSet) = Windup(ydata(DS),ydim(DS))

WoundInvCov(DS::AbstractDataSet) = _WoundInvCov(DS, InvCov(DS))
_WoundInvCov(DS::AbstractDataSet, D::Diagonal) = Windup(D.diag, ydim(DS))
_WoundInvCov(DS::AbstractDataSet, M::AbstractMatrix) = throw("WoundInvCov can only be used with diagonal covariance matrices.")

# Becomes dangerous now that there is a distinction between yInvCov and xInvCov
# logdetInvCov(DS::AbstractDataSet) = logdet(InvCov(DS))
DataspaceDim(DS::AbstractDataSet) = Npoints(DS) * ydim(DS)
#sigma(DS::AbstractDataSet) = ysigma(DS)
@deprecate sigma(x) ysigma(x) true

xdist(DS::AbstractDataSet) = xDataDist(DS)
ydist(DS::AbstractDataSet) = yDataDist(DS)

dist(DS::AbstractDataSet) = GeneralProduct([xdist(DS), ydist(DS)])

Npoints(DS::AbstractDataSet) = Npoints(dims(DS))
xdim(DS::AbstractDataSet) = xdim(dims(DS))
ydim(DS::AbstractDataSet) = ydim(dims(DS))
Npoints(dims::Tuple{Int,Int,Int}) = dims[1]
xdim(dims::Tuple{Int,Int,Int}) = dims[2]
ydim(dims::Tuple{Int,Int,Int}) = dims[3]


_WoundX(DS::AbstractDataSet, WoundX::Nothing) = xdata(DS)
_WoundX(DS::AbstractDataSet, WoundX::AbstractVector) = WoundX

# Can eliminate specialized methods
function InformNames(DS::T, xnames::AbstractVector{String}, ynames::AbstractVector{String}) where T <: AbstractDataSet
    @assert length(xnames) == xdim(DS) && length(ynames) == ydim(DS)
    remake(DS; xnames=xnames, ynames=ynames)
end


# Generic Methods for AbstractDataModels      -----       May be superceded by more specialized functions!
pdim(DM::AbstractDataModel) = pdim(Data(DM), Predictor(DM))
MLE(DM::AbstractDataModel) = FindMLE(DM)
LogLikeMLE(DM::AbstractDataModel) = loglikelihood(DM, MLE(DM))
LogPrior(DM::AbstractDataModel) = x->0.0


xpdim(DM::AbstractDataModel) = Npoints(DM) * xdim(DM) + pdim(DM)


# Generic passthrough of queries from AbstractDataModel to AbstractDataSet for following functions:
for F in [  :xdata, :ydata, :xsigma, :ysigma, :xInvCov, :yInvCov,
            :dims, :length, :Npoints, :xdim, :ydim, :DataspaceDim,
            :logdetInvCov, :WoundX, :WoundY, :WoundInvCov,
            :xnames, :ynames, :xdist, :ydist, :dist]
    @eval $F(DM::AbstractDataModel) = $F(Data(DM))
end


# Generic Methods which are not simply passed through
pnames(DM::AbstractDataModel) = pnames(DM, Predictor(DM))
pnames(DM::AbstractDataModel, M::ModelMap) = pnames(M)
pnames(DM::AbstractDataModel, F::Function) = CreateSymbolNames(pdim(DM),"θ")

Domain(DM::AbstractDataModel) = Predictor(DM) isa ModelMap ? Domain(Predictor(DM)) : FullDomain(pdim(DM))


function AutoDiffDmodel(DS::AbstractDataSet, model::Function; custom::Bool=false, ADmode::Union{Symbol,Val}=Val(:ForwardDiff), Kwargs...)
    Grad, Jac = DerivableFunctionsBase._GetGrad(ADmode; Kwargs...), DerivableFunctionsBase._GetJac(ADmode; Kwargs...)
    GradPass, JacPass = DerivableFunctionsBase._GetGradPass, DerivableFunctionsBase._GetJacPass
    ## Allow for symbolic passthrough here
    Autodmodel(x::Number,θ::AbstractVector{<:Number}; kwargs...) = transpose(Grad(z->model(x,z; kwargs...),θ))
    Autodmodel(x::Number,θ::AbstractVector{<:Num}; kwargs...) = transpose(GradPass(z->model(x,z; kwargs...),θ))
    NAutodmodel(x::AbstractVector{<:Number},θ::AbstractVector{<:Number}; kwargs...) = transpose(Grad(z->model(x,z; kwargs...),θ))
    NAutodmodel(x::AbstractVector{<:Number},θ::AbstractVector{<:Num}; kwargs...) = transpose(GradPass(z->model(x,z; kwargs...),θ))
    AutodmodelN(x::Number,θ::AbstractVector{<:Number}; kwargs...) = Jac(p->model(x,p; kwargs...), θ)
    AutodmodelN(x::Number,θ::AbstractVector{<:Num}; kwargs...) = JacPass(p->model(x,p; kwargs...), θ)
    NAutodmodelN(x::AbstractVector{<:Number},θ::AbstractVector{<:Number}; kwargs...) = Jac(p->model(x,p; kwargs...),θ)
    NAutodmodelN(x::AbstractVector{<:Number},θ::AbstractVector{<:Num}; kwargs...) = JacPass(p->model(x,p; kwargs...),θ)
    # Getting extract_gradient! error from ForwardDiff when using gradient method with observables
    # CustomAutodmodel(x::Union{Number,AbstractVector{<:Number}},θ::AbstractVector{<:Number}) = transpose(Grad(p->model(x,p),θ))
    CustomAutodmodelN(x::Union{Number,AbstractVector{<:Number}},θ::AbstractVector{<:Number}; kwargs...) = Jac(p->model(x,p; kwargs...),θ)
    CustomAutodmodelN(x::Union{Number,AbstractVector{<:Number}},θ::AbstractVector{<:Num}; kwargs...) = JacPass(p->model(x,p; kwargs...),θ)
    if ydim(DS) == 1
        custom && return CustomAutodmodelN
        return xdim(DS) == 1 ? Autodmodel : NAutodmodel
    else
        custom && return CustomAutodmodelN
        return xdim(DS) == 1 ? AutodmodelN : NAutodmodelN
    end
end


"""
    DetermineDmodel(DS::AbstractDataSet, model::Function; ADmode::Union{Symbol,Val}=:ForwardDiff)::Function
Returns appropriate function which constitutes the automatic derivative of the `model(x,θ)` with respect to the parameters `θ` depending on the format of the x-values and y-values of the DataSet.
"""
function DetermineDmodel(DS::AbstractDataSet, model::Function; custom::Bool=false, ADmode::Union{Symbol,Val}=Val(:ForwardDiff), kwargs...)
    # For the symbolically generated jacobians to work with MArrays, it requires ≥ v0.11.3 of SymbolicUtils.jl:  https://github.com/JuliaSymbolics/SymbolicUtils.jl/pull/286
    if ADmode === :Symbolic || ADmode isa Val{:Symbolic}
        Symbolic_dmodel = Optimize(DS, model; inplace=false)[2]
        !isnothing(Symbolic_dmodel) && return Symbolic_dmodel
        # Fall back to ForwarDiff if Symbolic differentiation did not work
        @info "Falling back to ForwardDiff for model jacobian."
        ADmode = Val(:ForwardDiff)
    end
    AutoDiffDmodel(DS, model; custom=custom, ADmode=ADmode, kwargs...)
end
function DetermineDmodel(DS::AbstractDataSet, M::ModelMap; custom::Bool=iscustom(M), kwargs...)
    ModelMap(DetermineDmodel(DS, M.Map; custom=custom, kwargs...), M)
end

"""
    MeasureAutoDiffPerformance(DM::DataModel; modes=diff_backends())
Tests the performance of various AD backends for performance one on the given DataModel.
See `diff_backends()` for the available backends currently loaded.
"""
MeasureAutoDiffPerformance(DM::AbstractDataModel; kwargs...) = MeasureAutoDiffPerformance(Data(DM), Predictor(DM), MLE(DM); kwargs...)
function MeasureAutoDiffPerformance(DS::AbstractDataSet, model::ModelOrFunction, mle::AbstractVector; modes::AbstractVector{<:Union{Val,Symbol}}=diff_backends(), kwargs...)
    perfs = Vector{Float64}(undef, length(modes))
    for (i,mode) in enumerate(modes)
        perfs[i] = try
            dmodel = DetermineDmodel(DS, model; ADmode=mode, kwargs...)
            @assert EmbeddingMatrix(DS, dmodel, mle) isa AbstractMatrix
            @belapsed EmbeddingMatrix($DS, $dmodel, $mle)
        catch;
            Inf
        end
    end
    M = sortslices([round.(perfs;sigdigits=4) modes]; dims=1)
    Res = [M[:,2] M[:,1]]

    println("$(Res)")
    println("$(Res[1,1]) performed best at $(Res[1,2])s. Second best was $(Res[2,1]) at $(Res[2,2])s.")
    Res
end

function CheckModelHealth(DS::AbstractDataSet, model::ModelOrFunction; verbose::Bool=true)
    P = GetStartP(DS, model)
    out = try  model(WoundX(DS)[1],P)   catch Err
        throw("Model evaluation failed for x=$(WoundX(DS)[1]) and θ=$P.")
    end
    size(out,1) != ydim(DS) && @warn "Got ydim=$(ydim(DS)) but output of model does not have this size."
    verbose && !(out isa SVector || out isa MVector) && (1 < ydim(DS) < 90) && @info "It may be beneficial for the overall performance to define the model function such that it outputs static vectors, i.e. SVectors."
    return nothing
end



DataDist(Y::AbstractVector, Sig::AbstractVector, Dist=Normal) = product_distribution([Dist(Y[i],Sig[i]) for i in eachindex(Y)])
DataDist(Y::AbstractVector, Sig::AbstractMatrix, Dist=MvNormal) = Dist(Y, HealthyCovariance(Sig))
yDataDist(DS::AbstractDataSet) = DataDist(ydata(DS), ysigma(DS))
xDataDist(DS::AbstractDataSet) = xsigma(DS) == zeros(Npoints(DS)*xdim(DS)) ? InformationGeometry.Dirac(xdata(DS)) : DataDist(xdata(DS), HealthyCovariance(xsigma(DS)))
yDataDist(DM::AbstractDataModel) = yDataDist(Data(DM))
xDataDist(DM::AbstractDataModel) = xDataDist(Data(DM))



"""
    pdim(DS::AbstractDataSet, model::ModelOrFunction) -> Int
Infers the (minimal) number of components that the given function `F` accepts as input by successively testing it on vectors of increasing length.
"""
function pdim(DS::AbstractDataSet, model::ModelOrFunction)
    if MaximalNumberOfArguments(model) == 2
        GetArgLength(p->model(WoundX(DS)[1],p))
    else #inplace model
        GetArgLength((Res,p)->model(Res,WoundX(DS)[1],p))
    end
end


# DataFrame(DM::DataModel) = DataFrame(Data(DM))
DataFrame(DS::AbstractDataSet; kwargs...) = SaveDataSet(DS; kwargs...)


function Base.join(DS1::T, DS2::T) where T <: AbstractDataSet
    !(xdim(DS1) == xdim(DS2) && ydim(DS1) == ydim(DS2)) && throw("DataSets incompatible.")
    NewΣ = if typeof(ysigma(DS1)) <: AbstractVector && typeof(ysigma(DS2)) <: AbstractVector
        vcat(ysigma(DS1), ysigma(DS2))
    else
        BlockMatrix(ysigma(DS1), ysigma(DS2))
    end
    DataSet(vcat(xdata(DS1), xdata(DS2)), vcat(ydata(DS1), ydata(DS2)), NewΣ, (Npoints(DS1)+Npoints(DS2), xdim(DS1), ydim(DS1)))
end
Base.join(DM1::AbstractDataModel, DM2::AbstractDataModel) = DataModel(join(Data(DM1),Data(DM2)), Predictor(DM1), dPredictor(DM1))
Base.join(DS1::T, DS2::T, args...) where T <: Union{AbstractDataSet,AbstractDataModel} = join(join(DS1,DS2), args...)
Base.join(DSVec::AbstractVector{T}) where T <: Union{AbstractDataSet,AbstractDataModel} = join(DSVec...)

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

Base.getindex(DS::AbstractDataSet, x) = SubDataSet(DS, x)
Base.firstindex(DS::AbstractDataSet) = 1
Base.lastindex(DS::AbstractDataSet) = Npoints(DS)

Sparsify(DS::AbstractDataSet) = SubDataSet(DS, rand(Bool,Npoints(DS)))
Sparsify(DM::AbstractDataModel) = SubDataSet(DS, rand(Bool,Npoints(DS)))
