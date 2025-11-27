

function ConsistentElDims(X::AbstractVector)
    elDim = length(X[1])
    all(x -> length(x) == elDim, X) ? elDim : throw("Inconsistent element lengths for given Vector.")
end
ConsistentElDims(X::AbstractVector{<:Number}) = 1
ConsistentElDims(M::AbstractMatrix{<:Number}) = size(M,2)
ConsistentElDims(T::Tuple) = ConsistentElDims(collect(T))



function HealthyCovariance(M::AbstractMatrix{<:Number}; verbose::Bool=true, tol::Real=1e-20, kwargs...)
    M = size(M,1) > size(M,2) ? Unwind(M) : M
    M = if isdiag(M)
        Diagonal(floatify(M))
    elseif sum(abs, M - Diagonal(floatify(M))) < tol
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
HealthyCovariance(D::DiagonalType; kwargs...) = all(x->x>0, D.diag) ? D : throw("Given covariance Matrix has non-positive values on diagonal: $(D.diag)")
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

WoundInvCov(DS::AbstractDataSet) = _WoundMatrix(yInvCov(DS), ydim(DS))
_WoundMatrix(D::DiagonalType, Yd::Int) = Windup(D.diag, Yd)
_WoundMatrix(M::AbstractMatrix, Yd::Int) = throw("WoundInvCov can only be used with diagonal covariance matrices.")

# Becomes dangerous now that there is a distinction between yInvCov and xInvCov
# logdetInvCov(DS::AbstractDataSet) = logdet(InvCov(DS))
DataspaceDim(DS::AbstractDataSet) = Npoints(DS) * ydim(DS)
#sigma(DS::AbstractDataSet) = ysigma(DS)
@deprecate sigma ysigma

xdist(DS::AbstractDataSet) = xDataDist(DS)
ydist(DS::AbstractDataSet) = yDataDist(DS)

HasXerror(DS::AbstractDataSet; verbose::Bool=false, kwargs...) = any(x->x>0.0, xsigma(DS; verbose, kwargs...))

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
function InformNames(DS::AbstractDataSet, XNames::AbstractVector{<:StringOrSymb}, YNames::AbstractVector{<:StringOrSymb})
    @assert length(XNames) == xdim(DS) && length(YNames) == ydim(DS)
    remake(DS; xnames=XNames, ynames=YNames)
end


SplitErrorParams(DS::AbstractFixedUncertaintyDataSet) = X::AbstractVector{<:Number} -> (X, 0:-1)

SkipXs(DS::AbstractDataModel) = SkipXs(Data(DS))
SkipXs(DS::Union{AbstractConditionGrid,AbstractDataSet}) = identity

## DataModel not defined yet
GetOnlyModelParams(DS::AbstractDataModel) = GetOnlyModelParams(Data(DS))
## For ConditionGrid or fixed uncertainty dataset, MLE has no error params or xpars
GetOnlyModelParams(DS::Union{AbstractConditionGrid,AbstractFixedUncertaintyDataSet}) = identity
function GetOnlyModelParams(DS::AbstractUnknownUncertaintyDataSet)
    Splitter = SplitErrorParams(DS);    Skipper = SkipXs(DS)
    mle::AbstractVector->Skipper(Splitter(mle)[1])
end


# Generic Methods for AbstractDataModels      -----       May be superceded by more specialized functions!
pdim(DM::AbstractDataModel) = pdim(Data(DM), Predictor(DM))
MLE(DM::AbstractDataModel) = FindMLE(DM)
LogLikeMLE(DM::AbstractDataModel) = loglikelihood(DM, MLE(DM))
LogPrior(DM::AbstractDataModel) = x::AbstractVector->zero(eltype(x))


xpdim(DM::AbstractDataModel) = Npoints(DM) * xdim(DM) + pdim(DM)

"""
    MLEuncert(DM::AbstractDataModel, mle::AbstractVector=MLE(DM), F::AbstractMatrix=FisherMetric(DM, mle))
Returns vector of type `Measurements.Measurement` where the parameter uncertainties are approximated via the diagonal of the inverse Fisher metric.
That is, the stated uncertainties are a linearized symmetric approximation of the true parameter uncertainties around the MLE.
"""
function MLEuncert(DM::AbstractDataModel, mle::AbstractVector=MLE(DM), F::AbstractMatrix=FisherMetric(DM, mle); verbose::Bool=true)
    @assert size(F,1) == size(F,2)
    # Use AutoMetric instead of FisherMetric since significantly more performant for large datasets due to reduced allocations
    # Also, diagonal basically unaffected in terms of precision
    try
        # sqrt∘Diagonal Larger than Diagonal∘cholesky entries
        mle .± sqrt.(Diagonal(inv(F)).diag)
    catch y;
        if y isa SingularException
            verbose && @warn "MLEuncert: FisherMetric singular, trying to estimate conservative uncertainties for non-degenerate eigendirections."
        elseif y isa DomainError
            verbose && @warn "MLEuncert: inverse Fisher metric not positive-definite, trying to estimate conservative uncertainties for non-degenerate eigendirections."
        else
            rethrow(y)
        end
        # Larger than Diagonal∘pinv
        mle .± sqrt.(Diagonal(ConservativeInverse(F)).diag)
    end
end

xdataMat(DS::AbstractDataSet) = UnpackWindup(xdata(DS), xdim(DS))
ydataMat(DS::AbstractDataSet) = UnpackWindup(ydata(DS), ydim(DS))

Base.keys(DS::AbstractDataSet) = 1:Npoints(DS)


# Generic passthrough of queries from AbstractDataModel to AbstractDataSet for following functions:
for F in [  :xdata, :ydata,
            :dims, :length, :Npoints, :xdim, :ydim, :DataspaceDim,
            :logdetInvCov, :WoundX, :WoundY, :WoundInvCov,
            :xnames, :ynames, :Xnames, :Ynames, :xdist, :ydist, :dist, :HasXerror,
            :xdataMat, :ydataMat,
            :SplitErrorParams]
    @eval $F(DM::AbstractDataModel) = $F(Data(DM))
end


# Generic Methods which are not simply passed through
pnames(DM::AbstractDataModel) = pnames(DM, Predictor(DM))
pnames(DM::AbstractDataModel, M::ModelMap) = pnames(M)
pnames(DM::AbstractDataModel, F::Function) = CreateSymbolNames(pdim(DM),"θ")

Pnames(DM::AbstractDataModel) = Pnames(DM, Predictor(DM))
Pnames(DM::AbstractDataModel, M::ModelMap) = Pnames(M)
Pnames(DM::AbstractDataModel, F::Function) = CreateSymbolNames(pdim(DM),"θ") .|> Symbol


xerrorparams(DM::AbstractDataModel, mle::AbstractVector=MLE(DM)) = xerrorparams(Data(DM), mle)
yerrorparams(DM::AbstractDataModel, mle::AbstractVector=MLE(DM)) = yerrorparams(Data(DM), mle)

xerrorparams(DS::AbstractFixedUncertaintyDataSet, mle::AbstractVector) = nothing
yerrorparams(DS::AbstractFixedUncertaintyDataSet, mle::AbstractVector) = nothing


# For first arg DM, expect full MLE, for first arg DS, expect error params only
xsigma(DM::AbstractDataModel, mle::AbstractVector=MLE(DM); kwargs...) = xsigmadecide(Data(DM), mle; kwargs...)
ysigma(DM::AbstractDataModel, mle::AbstractVector=MLE(DM); kwargs...) = ysigmadecide(Data(DM), mle; kwargs...)
xInvCov(DM::AbstractDataModel, mle::AbstractVector=MLE(DM); kwargs...) = xInvCovdecide(Data(DM), mle; kwargs...)
yInvCov(DM::AbstractDataModel, mle::AbstractVector=MLE(DM); kwargs...) = yInvCovdecide(Data(DM), mle; kwargs...)

# Reduce from full MLE to error params
xsigmadecide(DS::AbstractUnknownUncertaintyDataSet, mle::AbstractVector; kwargs...) = xsigma(DS, xerrorparams(DS, mle); kwargs...)
ysigmadecide(DS::AbstractUnknownUncertaintyDataSet, mle::AbstractVector; kwargs...) = ysigma(DS, yerrorparams(DS, mle); kwargs...)
xInvCovdecide(DS::AbstractUnknownUncertaintyDataSet, mle::AbstractVector; kwargs...) = xInvCov(DS, xerrorparams(DS, mle); kwargs...)
yInvCovdecide(DS::AbstractUnknownUncertaintyDataSet, mle::AbstractVector; kwargs...) = yInvCov(DS, yerrorparams(DS, mle); kwargs...)

# Pass params unaffected since dropped in next step
xsigmadecide(DS::AbstractFixedUncertaintyDataSet, mle::AbstractVector; kwargs...) = xsigma(DS, mle; kwargs...)
ysigmadecide(DS::AbstractFixedUncertaintyDataSet, mle::AbstractVector; kwargs...) = ysigma(DS, mle; kwargs...)
xInvCovdecide(DS::AbstractFixedUncertaintyDataSet, mle::AbstractVector; kwargs...) = xInvCov(DS, mle; kwargs...)
yInvCovdecide(DS::AbstractFixedUncertaintyDataSet, mle::AbstractVector; kwargs...) = yInvCov(DS, mle; kwargs...)

# Drop params since uncertainty fixed
xsigma(DS::AbstractFixedUncertaintyDataSet, mle::AbstractVector; verbose::Bool=true) = xsigma(DS)
ysigma(DS::AbstractFixedUncertaintyDataSet, mle::AbstractVector; verbose::Bool=true) = ysigma(DS)
xInvCov(DS::AbstractFixedUncertaintyDataSet, mle::AbstractVector; verbose::Bool=true) = xInvCov(DS)
yInvCov(DS::AbstractFixedUncertaintyDataSet, mle::AbstractVector; verbose::Bool=true) = yInvCov(DS)



# How many error parameters do the containing datasets have?
NumberOfErrorParameters(DM::AbstractDataModel, mle::AbstractVector=MLE(DM)) = NumberOfErrorParameters(Data(DM), mle)
NumberOfErrorParameters(DS::AbstractUnknownUncertaintyDataSet, mle::AbstractVector) = sum(length, (SplitErrorParams(DS)(mle))[2:end])
NumberOfErrorParameters(DS::AbstractFixedUncertaintyDataSet, mle::AbstractVector) = 0
## Currently not possible for CDS to have error parameters yet.
# NumberOfErrorParameters(CDS::CompositeDataSet, mle::AbstractVector) = sum(sum(length, (SplitErrorParams(DS)(mle))[2:end]) for DS in Data(CDS))

errormoddim(DS::AbstractFixedUncertaintyDataSet; kwargs...) = 0
xpars(DS::AbstractDataSet) = 0

# How many degrees of freedom does the model have?
# Error parameters should not be counted
"""
    DOF(DM::AbstractDataModel)
Parameter degrees of freedom of given model not counting error parameters.
"""
DOF(DM::AbstractDataModel, mle::AbstractVector=MLE(DM)) = DOF(Data(DM), mle)
DOF(DS::AbstractDataSet, mle::AbstractVector) = length(mle) - NumberOfErrorParameters(DS, mle)


name(DS::Union{AbstractDataSet, AbstractVector{<:Union{<:AbstractDataSet,<:AbstractDataModel}}}) = ""


Christen(DS::Union{ModelMap,AbstractDataSet}, name::StringOrSymb) = remake(DS; name=Symbol(name))
Christen(F::Function, name::StringOrSymb) = (@warn "Cannot add name to function, needs to be wrapped in ModelMap first.";   ModelMap(F; name=Symbol(name)))
Christen(DM::AbstractDataModel, name::StringOrSymb) = remake(DM; name=Symbol(name))


function AutoDiffDmodel(DS::AbstractDataSet, model::Function; custom::Bool=false, ADmode::Union{Symbol,Val}=Val(:ForwardDiff), inplace::Bool=isinplacemodel(model), makeinplace::Bool=inplace, Kwargs...)
    ADmode isa Symbol && (ADmode = Val(ADmode))
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
    CustomAutodmodelN(x,θ::AbstractVector{<:Number}; kwargs...) = try Jac(p->model(x,p; kwargs...),θ) catch; transpose(Grad(p->model(x,p; kwargs...),θ)) end
    CustomAutodmodelN(x,θ::AbstractVector{<:Num}; kwargs...) = JacPass(p->model(x,p; kwargs...),θ)
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
        Symbolic_dmodel = OptimizeModel(DS, model)[2]
        !isnothing(Symbolic_dmodel) && return Symbolic_dmodel
        # Fall back to ForwardDiff if Symbolic differentiation did not work
        @info "Falling back to ForwardDiff for model jacobian."
        ADmode = Val(:ForwardDiff)
    end
    AutoDiffDmodel(DS, model; custom, ADmode, kwargs...)
end
function DetermineDmodel(DS::AbstractDataSet, M::ModelMap; custom::Bool=iscustommodel(M), inplace::Bool=isinplacemodel(M), makeinplace::Bool=inplace, kwargs...)
    ModelMap(DetermineDmodel(DS, M.Map; custom, inplace, makeinplace, kwargs...), M; inplace=makeinplace)
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

function CheckModelHealth(DS::AbstractDataSet, model::ModelOrFunction, P::AbstractVector=GetStartP(DS, model); verbose::Bool=true)
    out = try  model(WoundX(DS)[1],GetOnlyModelParams(DS)(P))   catch Err
        throw("Model evaluation failed for x=$(WoundX(DS)[1]) and θ=$P.")
    end
    size(out,1) != ydim(DS) && @warn "Got ydim=$(ydim(DS)) but output of model does not have this size."
    # if verbose && !isinplacemodel(model) && !(out isa SVector || out isa MVector) && (1 < ydim(DS) < 90)
    #     @info "It may be beneficial for the overall performance to define the model function such that it outputs static vectors, i.e. SVectors."
    # end
    return nothing
end



DataDist(Y::AbstractVector, Sig::AbstractVector, Dist=Normal) = product_distribution([Dist(Y[i],Sig[i]) for i in eachindex(Y)])
DataDist(Y::AbstractVector, Sig::AbstractMatrix, Dist=MvNormal) = Dist(Y, HealthyCovariance(Sig))
yDataDist(DS::AbstractDataSet) = DataDist(ydata(DS), ysigma(DS))
xDataDist(DS::AbstractDataSet) = xsigma(DS) == Zeros(Npoints(DS)*xdim(DS)) ? InformationGeometry.Dirac(xdata(DS)) : DataDist(xdata(DS), HealthyCovariance(xsigma(DS)))
yDataDist(DM::AbstractDataModel) = yDataDist(Data(DM))
xDataDist(DM::AbstractDataModel) = xDataDist(Data(DM))

"""
    ScaledResiduals(DM::AbstractDataModel, mle::AbstractVector=MLE(DM))
Computes residuals with additional scaling by the cholesky decomposition of the inverse ``y``-covariance matrix.
"""
function ScaledResiduals(DM::AbstractDataModel, mle::AbstractVector=MLE(DM); xerrors::Bool=false, verbose::Bool=true, kwargs...)
    Ypred = if xerrors && HasXerror(DM)
        try
            Xsig = xsigma(DM, mle) isa AbstractVector ? xsigma(DM, mle) : sqrt.(Diagonal(xsigma(DM, mle)).diag)
            EmbeddingMap(DM, mle, Measurement.(collect(xdata(DM)), Xsig))
        catch;
            verbose && @warn "Ignoring x-uncertainties in computatation of Residuals."
            EmbeddingMap(DM, mle)
        end
    else
        EmbeddingMap(DM, mle)
    end
    cholesky(yInvCov(DM, mle)).U * (ydata(DM) - Ypred)
end


# Called by GetStartP
"""
    pdim(DS::AbstractDataSet, model::ModelOrFunction; max::Int=200) -> Int
Infers the (minimal) number of components that the given function `F` accepts as input by successively testing it on vectors of increasing length.
"""
function pdim(DS::AbstractDataSet, model::Function; max::Int=MaxArgLen)
    pModel = if !isinplacemodel(model)
        GetArgLength(p->model(WoundX(DS)[1],p); max)
    else #inplace model
        GetArgLength((Res,p)->model(Res,WoundX(DS)[1],p); max)
    end
    # Error parameter definitely not accounted for in pModel yet
    pModel + errormoddim(DS)
end
pdim(DS::AbstractFixedUncertaintyDataSet, M::ModelMap) = pdim(M)
function pdim(DS::AbstractUnknownUncertaintyDataSet, M::ModelMap)
    # Check if given ModelMap Domain already includes error parameters
    try
        # Should already compensate for in-place models
        M(WoundX(DS)[1],GetOnlyModelParams(DS)(ElaborateGetStartP(M)))
        pdim(M)
    catch E;
        @warn "pdim(DS,M): It appears that error parameters are not included in given ModelMap Domain $(Domain(M)) yet? Got error $E. Appending $(errormoddim(DS)) component(s) for error parameters and $(xpars(DS)) xpars to initial parameter guess and trying to continue."
        xpars(DS) + pdim(M) + errormoddim(DS)
    end
end


# DataSet types not defined at point of loading this
function Base.join(DS1::T, DS2::T) where T <: AbstractDataSet
    @assert !HasXerror(DS1) && !HasXerror(DS2)
    !(xdim(DS1) == xdim(DS2) && ydim(DS1) == ydim(DS2)) && throw("DataSets incompatible.")
    NewΣ = if typeof(ysigma(DS1)) <: AbstractVector && typeof(ysigma(DS2)) <: AbstractVector
        vcat(ysigma(DS1), ysigma(DS2))
    else
        BlockMatrix(ysigma(DS1), ysigma(DS2))
    end
    DataSet(vcat(xdata(DS1), xdata(DS2)), vcat(ydata(DS1), ydata(DS2)), NewΣ, (Npoints(DS1)+Npoints(DS2), xdim(DS1), ydim(DS1));
        xnames=Xnames(DS1), ynames=Ynames(DS1), name=string(name(DS1)) * " + " * string(name(DS2)))
end
"""
    join(DM1::AbstractDataModel, DM2::AbstractDataModel) -> DataModel
Joins the data from two different models while keeping the model and prior of the first.
"""
Base.join(DM1::AbstractDataModel, DM2::AbstractDataModel) = DataModel(join(Data(DM1),Data(DM2)), Predictor(DM1), dPredictor(DM1), MLE(DM1), LogPrior(DM1))
Base.join(DS1::T, DS2::T, args...) where T <: Union{AbstractDataSet,AbstractDataModel} = join(join(DS1,DS2), args...)
Base.join(DSVec::AbstractVector{T}) where T <: Union{AbstractDataSet,AbstractDataModel} = join(DSVec...)

SortDataSet(DS::AbstractDataSet) = DS |> DataFrame |> sort |> DataSet
SortDataModel(DM::AbstractDataModel) = remake(DM; Data=SortDataSet(Data(DM)))


"""
    AddDataPoint(DS::AbstractDataSet, Tup::Tuple) -> AbstractDataSet
Given a tuple `Tup = (x,y,σ)`, with `σ` the standard deviation(s), this data point is added to `DS`.
"""
function AddDataPoint(DS::AbstractDataSet, Tup::Tuple{<:Union{Number,AbstractVector{<:Number}}, <:Union{Number,AbstractVector{<:Number}}, <:Union{Number,AbstractVector{<:Number}}}; kwargs...)
    @assert !(DS isa CompositeDataSet) "Cannot handle CompositeDataSets."
    @assert length(Tup[1]) == xdim(DS) && length(Tup[2]) == ydim(DS) && length(Tup[3]) == ydim(DS)
    remake(DS; x=[xdata(DS);Tup[1]], y=[ydata(DS);Tup[2]], InvCov=BlockMatrix(yInvCov(DS), Diagonal([Tup[3][i]^(-2) for i in eachindex(Tup[3])])),
                dims=(Npoints(DS)+1, xdim(DS), ydim(DS)), kwargs...)
end

"""
    AddDataPoint(DS::AbstractDataSet, Tup::Tuple) -> AbstractDataSet
Given a tuple `Tup = (x,xσ,y,yσ)`, with `σ` the standard deviation(s), this data point is added to `DS`.
"""
function AddDataPoint(DS::AbstractDataSet, Tup::Tuple{<:Union{Number,AbstractVector{<:Number}}, <:Union{Number,AbstractVector{<:Number}}, <:Union{Number,AbstractVector{<:Number}}, <:Union{Number,AbstractVector{<:Number}}}; kwargs...)
    @assert !(DS isa CompositeDataSet) "Cannot handle CompositeDataSets."
    @assert length(Tup[1]) == xdim(DS) && length(Tup[2]) == xdim(DS)
    @assert length(Tup[3]) == ydim(DS) && length(Tup[4]) == ydim(DS)
    throw("Not implemented yet.")
end


"""
    SubDataSetComponent(DS::AbstractDataSet, i::Int)
Get a dataset containing only the `i`-th y-components as observations.
"""
function SubDataSetComponent(DS::AbstractFixedUncertaintyDataSet, i::Union{Int,AbstractVector{<:Int}})
    idxs = i isa AbstractVector ? i : [i]
    DS isa CompositeDataSet && return (length(idxs) == 1 ? Data(DS)[idxs[1]] : CompositeDataSet(Data(DS)[idxs])) # CompositeDataSet defined downstream
    @assert all(1 .≤ idxs .≤ ydim(DS)) && allunique(idxs)
    keep = repeat([j ∈ idxs for j in 1:ydim(DS)], Npoints(DS))
    if !HasXerror(DS)
        DataSet(xdata(DS), ydata(DS)[keep], (ysigma(DS) isa AbstractVector ? ysigma(DS)[keep] : ysigma(DS)[keep,keep]), (Npoints(DS), xdim(DS), length(idxs));
                xnames=Xnames(DS), ynames=Ynames(DS)[idxs], name=name(DS))
    else
        DataSetExact(xdata(DS), xsigma(DS), ydata(DS)[keep], (ysigma(DS) isa AbstractVector ? ysigma(DS)[keep] : ysigma(DS)[keep,keep]), (Npoints(DS), xdim(DS), length(idxs));
                xnames=Xnames(DS), ynames=Ynames(DS)[idxs], name=name(DS))
    end
end


"""
    SubDataSet(DS::AbstractDataSet, range::Union{AbstractVector{<:Int},BoolVector,Int})
Shorten the dataset by restricting to the data points specified by `range`.
"""
function SubDataSet(DS::AbstractDataSet, range::Union{AbstractVector{<:Int},BoolVector,Int}; verbose::Bool=true, kwargs...)
    @assert DS isa DataSet || xdist(DS) isa InformationGeometry.Dirac
    Npoints(DS) < length(range) && throw("Length of given range unsuitable for DataSet.")
    verbose && !allunique(range) && @warn "Not all given indices unique!"
    X = WoundX(DS)[range] |> Unwind
    Y = Windup(ydata(DS),ydim(DS))[range] |> Unwind
    Σ = ysigma(DS)
    if Σ isa AbstractVector
        Σ = Windup(Σ,ydim(DS))[range] |> Unwind
    elseif ydim(DS) == 1
        Σ = Σ[range,range]
    else
        Σ = _WoundMatrix(Σ, ydim(DS))[range, range] |> BlockMatrix
    end
    DataSet(X,Y,Σ, (Int(length(X)/xdim(DS)),xdim(DS),ydim(DS)); xnames=Xnames(DS), ynames=Ynames(DS), name=name(DS), kwargs...)
end
SubDataModel(DM::AbstractDataModel, range::Union{AbstractVector{<:Int},BoolVector}; kwargs...) = DataModel(SubDataSet(Data(DM), range; kwargs...), Predictor(DM), dPredictor(DM), MLE(DM), LogPrior(DM))

Base.getindex(DS::AbstractDataSet, x) = SubDataSet(DS, x)
Base.firstindex(DS::AbstractDataSet) = 1
Base.lastindex(DS::AbstractDataSet) = Npoints(DS)

# Check if independent variables sorted
Base.issorted(DS::AbstractDataSet) = (@assert xdim(DS) == 1;    issorted(xdata(DS)))
Base.sort(DS::AbstractDataSet; rev::Bool=false, kwargs...) = (@assert xdim(DS)==1;    P=sortperm(xdata(DS); rev);    SubDataSet(DS, P; kwargs...))


Sparsify(DS::AbstractDataSet) = SubDataSet(DS, rand(Bool,Npoints(DS)))
Sparsify(DM::AbstractDataModel) = SubDataSet(DM, rand(Bool,Npoints(DM)))


"""
    Minimize(DM::AbstractDataModel, startp::AbstractVector=MLE(DM); Full::Bool=false, Multistart::Int=0, kwargs...)
Performs multistart optimization with `MultistartFit` or `InformationGeometry.minimize` depending on whether `Multistart > 0` under one interface.
"""
function Minimize(DM, startp::AbstractVector=(DM isa AbstractDataModel ? MLE(DM) : Float64[]), args...; Full::Bool=false, Multistart::Int=0, kwargs...)
    (Full ? identity : GetMinimizer)(Multistart > 0 ? MultistartFit(DM; N=Multistart, kwargs...) : InformationGeometry.minimize(DM, startp; kwargs...))
end

"""
    Refit(DM::AbstractDataModel, startp::AbstractVector=MLE(DM); Multistart::Int=0, kwargs...)
Refits `DM` and returns result as new `DataModel`.
If `Multistart > 0`, then `MultistartFit` is used for the optimization and `startp` is dropped.
Otherwise `InformationGeometry.minimize` is called with the given `startp`.
"""
function Refit(DM::AbstractDataModel, startp::AbstractVector=MLE(DM); SkipTests::Bool=false, Multistart::Int=0, kwargs...)
    X = InformationGeometry.Minimize(DM, startp; Full=false, Multistart, kwargs...)
    remake(DM; MLE=X, LogLikeMLE=loglikelihood(DM, X), SkipTests)
end

