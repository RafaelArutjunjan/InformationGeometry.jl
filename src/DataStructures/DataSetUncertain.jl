
## Todo: also check via ForwardDiff whether error model as x / y dependence.
## Add optimized inplace methods and full matrix builder
function ErrorModelTester(inverrormodelraw::Function, testoutput)
    Inverrormodelraw, Inverrormodel = if testoutput isa AbstractVector
        inverrormodelraw, (x,y,c::AbstractVector)->Diagonal(inverrormodelraw(x,y,c)) # Wrap vector in Diagonal
    elseif testoutput isa Diagonal
        (x,y,c::AbstractVector)->inverrormodelraw(x,y,c).diag, inverrormodelraw # Unwrap Diagonal in raw model
    elseif testoutput isa Number || testoutput isa AbstractMatrix
        inverrormodelraw, inverrormodelraw # Do nothing
    else
        throw("Not implemented for error model output $testoutput of type $(typeof(testoutput)) yet.")
    end
    ## First output as vector, second as matrix
    Inverrormodelraw, Inverrormodel
end


"""
    GetPredKeep(T::AbstractVector{<:Number}, Ydata::AbstractMatrix{<:Number})
Returns minimal `Tstar = sort!(unique(T))` for evaluating predictions on, as well as vector `predkeep` such that
`Pred(Tstar)[predkeep] = Pred(T)`
By further masking `predkeep` with `datakeep`, missing values can be accounted for.
"""
function GetPredKeep(T::AbstractVector, Ydata::AbstractMatrix)
    @boundscheck @assert length(T) == size(Ydata,1)
    @assert !any(z->!any(x->!ismissing(x) && isfinite(x),z), eachrow(Ydata)) "Whole rows in ydata matrix missing, remove empty rows and corresponding times first."
    Ydim = size(Ydata,2)
    RowIndices(row::Int) = (row-1)*Ydim+1:row*Ydim
    
    Tstar = sort!(unique(T))
    predkeep = reduce(vcat, [RowIndices(row) for row in Int.(indexin(T, Tstar))])
    Tstar, predkeep
end

# Use general bitvector mask to implement missing values

"""
    DataSetUncertain(x::AbstractVector, y::AbstractVector, σ⁻¹::Function, c::AbstractVector; BesselCorrection::Bool=false)
    DataSetUncertain(x::AbstractVector, y::AbstractVector, σ⁻¹::Function, errorparamsplitter::Function, c::AbstractVector, dims::Tuple{Int,Int,Int}; BesselCorrection::Bool=false)
The `DataSetUncertain` type encodes data for which the size of the variance is unknown a-priori but whose error is specified via an error model of the form `σ(x, y_pred, c)` where `c` is a vector of error parameters.
This parametrized error model is subsequently used to estimate the standard deviations in the observations `y`.
!!! note
    To enhance performance, the implementation actually requires the specification of a *reciprocal* error model, i.e. a function `σ⁻¹(x, y_pred, c)`.
    If `ydim` is larger than one, the reciprocal error model should output a matrix, i.e. the cholesky decomposition `S` of the covariance `Σ` such that `Σ == S' * S`.

To construct a `DataSetUncertain`, one has to specify a vector of independent variables `x`, a vector of dependent variables `y`, a reciprocal error model `σ⁻¹(x, y_pred, c)` and an initial guess for the vector of error parameters `c`.
Optionally, an explicit `errorparamsplitter` function of the form `θ -> (modelparams, errorparams)` may be specified, which splits the parameters into a tuple of model parameters, which are subsequently forwarded into the model, and error parameters `c`, which are only passed to the reciprocal error model `σ⁻¹`.
!!! warning
    The parameters which are visible to the outside are processed by `errorparamsplitter` FIRST, before forwarding into the model, where `modelparams` might be further modified by embedding transformations.

# Examples:

In the simplest case, where all data points are mutually independent and have a single ``x``-component and a single ``y``-component each, a `DataSet` consisting of four points can be constructed via
```julia
DS = DataSetUncertain([1,2,3,4], [4,5,6.5,7.8], (x,y,c)->1/exp10(c[1]), [0.5])
```
!!! note
    It is generally advisable to exponentiate error parameters, since they are penalized poportional to `log(c)` in the normalization term of Gaussian likelihoods.
    A Bessel correction `sqrt((length(ydata(DS))-length(params))/length(ydata(DS)))` can be applied to the reciprocal error to account for the fact that the maximum likelihood estimator for the variance is biased via kwarg `BesselCorrection`.
"""
struct DataSetUncertain{BesselCorrection, Keep} <: AbstractUnknownUncertaintyDataSet
    x::AbstractVector{<:Number}
    y::AbstractVector{<:Number}
    dims::Tuple{Int,Int,Int} # Nxy
    inverrormodelraw::Function # 1/errormodel as Number, Vector or Matrix
    testout::Union{Number,<:AbstractVector,<:AbstractMatrix}
    inverrormodel::Function # 1./errormodel wrapped as AbstractMatrix, e.g. Diagonal
    testpy::AbstractVector{<:Number}
    errorparamsplitter::Function # θ -> (view(θ, MODEL), view(θ, ERRORMODEL))
    datakeep::Union{Nothing, AbstractVector{<:Bool}} ## falses correspond to locations of missing values
    predkeep::Union{Nothing, AbstractVector{<:Int}} ## Which ys to keep from EmbeddingMap evaluated at sparsified woundXpred to reconstruct ydata
    woundXpred::Union{Nothing, AbstractVector} # sorted woundX with duplicates removed
    xnames::AbstractVector{Symbol}
    ynames::AbstractVector{Symbol}
    name::Symbol

    DataSetUncertain(DM::AbstractDataModel, args...; kwargs...) = DataSetUncertain(Data(DM), args...; kwargs...)
    DataSetUncertain(DS::AbstractDataSet, args...; kwargs...) = DataSetUncertain(WoundX(DS), WoundY(DS), args...; xnames=Xnames(DS), ynames=Ynames(DS), name=name(DS), kwargs...)
    function InformationGeometry.DataSetUncertain(DS::CompositeDataSet, args...; kwargs...)
        X, Y = ReconstructDataMatrices(DS)
        DataSetUncertain(X, Y, (size(Y,1), size(X,2), size(Y,2)), args...; xnames=Xnames(DS), ynames=Ynames(DS), name=name(DS), kwargs...)
    end
    function DataSetUncertain(X::AbstractArray, Y::AbstractArray, dims::Tuple{Int,Int,Int}=(size(X,1), ConsistentElDims(X), ConsistentElDims(Y)); verbose::Bool=true, kwargs...)
        verbose && @info "Assuming error model σ(x,y,c) = exp10.(c)"
        errmod = ydim(dims) == 1 ? ((x,y,c::AbstractVector)->exp10(-c[1])) : ((x,y,c::AbstractVector)->exp10.(-c))
        DataSetUncertain(Unwind(X), Unwind(Y), errmod, Fill(0.1,ydim(dims)), dims; verbose, kwargs...)
    end
    function DataSetUncertain(X::AbstractArray{<:Number}, Y::AbstractArray{<:Number}, inverrormodel::Function, testpy::AbstractVector; kwargs...)
        size(X,1) != size(Y,1) && throw("Inconsistent number of x-values and y-values given: $(size(X,1)) != $(size(Y,1)). Specify a tuple (Npoints, xdim, ydim) in the constructor.")
        DataSetUncertain(collect(eachrow(X)), collect(eachrow(Y)), inverrormodel, testpy; kwargs...)
    end
    function DataSetUncertain(X::AbstractDataFrame, Y::AbstractDataFrame, args...; xnames=names(X), ynames=names(Y), kwargs...)
        DataSetUncertain(collect.(eachrow(X)), collect.(eachrow(Y)), args...; xnames, ynames, kwargs...)
    end
    function DataSetUncertain(X::AbstractArray, Y::AbstractArray, inverrormodel::Function, testpy::AbstractVector; kwargs...)
        size(X,1) != size(Y,1) && throw("Inconsistent number of x-values and y-values given: $(size(X,1)) != $(size(Y,1)). Specify a tuple (Npoints, xdim, ydim) in the constructor.")
        DataSetUncertain(Unwind(X), Unwind(Y), inverrormodel, testpy, (size(X,1), ConsistentElDims(X), ConsistentElDims(Y)); kwargs...)
    end
    function DataSetUncertain(x::AbstractVector, y::AbstractVector, inverrormodel::Function, testpy::AbstractVector, dims::Tuple{Int,Int,Int}; verbose::Bool=true, kwargs...)
        verbose && @info "Assuming error parameters always given by last $(length(testpy)) parameters."
        DataSetUncertain(x, y, inverrormodel, DefaultErrorModelSplitter(length(testpy)), testpy, dims; verbose, kwargs...)
    end
    function DataSetUncertain(x::AbstractVector, y::AbstractVector, inverrormodel::Function, errorparamsplitter::Function, testpy::AbstractVector, dims::Tuple{Int,Int,Int}=(size(x,1), ConsistentElDims(x), ConsistentElDims(y));
                xnames::AbstractVector{<:StringOrSymb}=CreateSymbolNames(xdim(dims),"x"), ynames::AbstractVector{<:StringOrSymb}=CreateSymbolNames(ydim(dims),"y"), name::StringOrSymb=Symbol(), kwargs...)
        DataSetUncertain(Unwind(x), Unwind(y), dims, inverrormodel, errorparamsplitter, testpy, xnames, ynames, name; kwargs...)
    end
    function DataSetUncertain(x::AbstractVector, y::AbstractVector, dims::Tuple{Int,Int,Int}, inverrormodelraw::Function, errorparamsplitter::Function, testpy::AbstractVector,
                xnames::AbstractVector{<:StringOrSymb}=CreateSymbolNames(xdim(dims),"x"), ynames::AbstractVector{<:StringOrSymb}=CreateSymbolNames(ydim(dims),"y"), name::StringOrSymb=Symbol(); 
                datakeep::Union{Nothing,AbstractVector{<:Bool}}=map(z->!ismissing(z) && isfinite(z), y), predkeep::Union{Nothing,AbstractVector{<:Int}}=nothing, woundXpred::Union{Nothing,AbstractVector}=nothing, kwargs...)
        testout = inverrormodelraw(Windup(x, xdim(dims))[1], Windup(y, ydim(dims))[1], testpy)
        Inverrormodelraw, Inverrormodel = ErrorModelTester(inverrormodelraw, testout)

        WoundXpred, Keep = if !all(datakeep)
            if !isnothing(predkeep) && !isnothing(woundXpred)
                woundXpred, predkeep
            else
                sparseX, predkeepraw = GetPredKeep(x, transpose(reshape(y, ydim(dims), :)))
                sparseX, predkeepraw[datakeep]
            end
        else  woundXpred, predkeep  end
        DataSetUncertain(x, (!all(datakeep) ? y[datakeep] : y), dims, Inverrormodelraw, testout, Inverrormodel, testpy, errorparamsplitter, (!all(datakeep) ? datakeep : nothing), Keep, WoundXpred, xnames, ynames, name; kwargs...)
    end
    ## Assume missings already removed and accounted for in keep and WoundX
    function DataSetUncertain(x::AbstractVector, y::AbstractVector, dims::Tuple{Int,Int,Int}, inverrormodelraw::Function, testout::Union{Number,<:AbstractVector,<:AbstractMatrix}, inverrormodel::Function, testpy::AbstractVector, 
                errorparamsplitter::Function, datakeep::Union{Nothing,AbstractVector{<:Bool}}, predkeep::Union{Nothing,AbstractVector{<:Int}}, woundXpred::Union{Nothing,AbstractVector}, 
                xnames::AbstractVector{<:StringOrSymb}, ynames::AbstractVector{<:StringOrSymb}, name::StringOrSymb=Symbol(); BesselCorrection::Bool=false, verbose::Bool=true)
        @assert all(x->(x > 0), dims) "Not all dims > 0: $dims."
        @assert Npoints(dims) == Int(length(x)/xdim(dims)) "Inconsistent input dimensions. Specify a tuple (Npoints, xdim, ydim) in the constructor."
        @assert length(xnames) == xdim(dims) && length(ynames) == ydim(dims)
        @assert !any(z->ismissing(z) || !isfinite(z), y)
        @assert (isnothing(datakeep) && isnothing(predkeep) && Npoints(dims) == Int(length(y)/ydim(dims))) || (!isnothing(datakeep) && !isnothing(predkeep) && length(y) == length(predkeep) && length(predkeep) ≤ ydim(dims)*Npoints(dims))

        # Check that inverrormodel either outputs Matrix for ydim > 1
        ydim(dims) == 1 && (@assert testout isa Number && testout > 0)
        ydim(dims) > 1 && @assert (testout isa AbstractVector && length(testout) == ydim(dims) && all(testout .> 0)) || (testout isa AbstractMatrix && size(testout,1) == size(testout,2) == ydim(dims) && det(testout) > 0)
        
        new{BesselCorrection, typeof(predkeep)}(x, y, dims, inverrormodelraw, testout, inverrormodel, testpy, errorparamsplitter, datakeep, predkeep, woundXpred, Symbol.(xnames), Symbol.(ynames), Symbol(name))
    end
end

function (::Type{T})(DS::DataSetUncertain{B}; kwargs...) where T<:Number where B
    remake(DS; x=T.(xdata(DS)), y=T.(ydata(DS)), testpy=T.(DS.testpy), testout=T.(DS.testout), BesselCorrection=B, (!isnothing(DS.woundXpred) ? (;woundXpred=map(T, DS.woundXpred)) : (;))...,kwargs...)
end

# For SciMLBase.remake
DataSetUncertain(;
x::AbstractVector=[0.],
y::AbstractVector=[0.],
dims::Tuple{Int,Int,Int}=(1,1,1),
inverrormodelraw::Function=identity,
testout::Union{Number,<:AbstractVector,<:AbstractMatrix}=5,
inverrormodel::Function=identity, 
testpy::AbstractVector{<:Number}=[0.],
errorparamsplitter::Function=x->(x[1], x[2]),
xnames::AbstractVector{<:StringOrSymb}=[:x],
ynames::AbstractVector{<:StringOrSymb}=[:y],
BesselCorrection::Bool=false,
verbose::Bool=true,
datakeep::Union{Nothing,AbstractVector{<:Bool}}=nothing, 
predkeep::Union{Nothing,AbstractVector{<:Int}}=nothing,
woundXpred::Union{Nothing,AbstractVector}=nothing,
name::StringOrSymb=Symbol()) = DataSetUncertain(x, y, dims, inverrormodelraw, testout, inverrormodel, testpy, errorparamsplitter, datakeep, predkeep, woundXpred, xnames, ynames, name; BesselCorrection, verbose)

DefaultErrorModelSplitter(n::Int) = ((θ::AbstractVector{<:Number}; kwargs...) -> @views (θ[1:end-n], θ[end-n+1:end]))
Identity2Splitter = ((θ::AbstractVector{<:Number}; kwargs...) -> (θ, θ))

xdata(DS::DataSetUncertain) = DS.x
ydata(DS::DataSetUncertain) = DS.y
dims(DS::DataSetUncertain) = DS.dims
xnames(DS::DataSetUncertain) = Xnames(DS) .|> string
ynames(DS::DataSetUncertain) = Ynames(DS) .|> string
Xnames(DS::DataSetUncertain) = DS.xnames
Ynames(DS::DataSetUncertain) = DS.ynames
name(DS::DataSetUncertain) = DS.name

xsigma(DS::DataSetUncertain, mle::AbstractVector=Float64[]) = Zeros(length(xdata(DS)))

HasXerror(DS::DataSetUncertain) = false

xerrormoddim(DS::DataSetUncertain) = 0
yerrormoddim(DS::DataSetUncertain) = length(DS.testpy)
errormoddim(DS::DataSetUncertain; kwargs...) = length(DS.testpy)


SplitErrorParams(DS::DataSetUncertain) = DS.errorparamsplitter

yinverrormodel(DS::DataSetUncertain) = DS.inverrormodel
yinverrormodelraw(DS::DataSetUncertain) = DS.inverrormodelraw

xerrorparams(DS::DataSetUncertain, mle::AbstractVector) = nothing
yerrorparams(DS::DataSetUncertain, mle::AbstractVector) = (SplitErrorParams(DS)(mle))[2]

HasBessel(DS::DataSetUncertain{T}) where T = T

HasMissingValues(CDS::DataSetUncertain{<:Any, <:AbstractVector}) = true

## Map WoundY to WoundYmasked for DataSetUncertain, since the only reason to want to wind up is for predictions without missings
WoundY(DS::DataSetUncertain) = WoundYmasked(DS)

WoundYSigmaMasked(DS::DataSetUncertain{<:Any,<:Nothing}, testpy::AbstractVector=DS.testpy) = Windup(ysigma(DS, testpy), ydim(DS))
WoundYSigmaMasked(DS::DataSetUncertain{<:Any,<:AbstractVector}, testpy::AbstractVector=DS.testpy) = (YsigmaNan = ReconstructYdataSigmaMatrix(DS, testpy);    [view(YsigmaNan, i, :) for i in axes(YsigmaNan,1)])

WoundYmasked(DS::DataSetUncertain{<:Any,<:Nothing}) = Windup(ydata(DS), ydim(DS))
WoundYmasked(DS::DataSetUncertain{<:Any,<:AbstractVector}) = (Ynan = ReconstructDataMatrices(DS)[2];    [view(Ynan, i, :) for i in axes(Ynan,1)])


_TryVectorizeNoSqrt(X::AbstractVector{<:Number}) = X
_TryVectorizeNoSqrt(X::AbstractVector{<:AbstractArray}) = BlockMatrix(X) |> _TryVectorizeNoSqrt
_TryVectorizeNoSqrt(M::AbstractMatrix) = isdiag(M) ? Diagonal(M).diag : M
_TryVectorizeNoSqrt(D::DiagonalType) = D.diag

## Get submatrix specified via BitVector
MaskedSymmetricMatrix(M::AbstractMatrix, ::Nothing) = M
MaskedSymmetricMatrix(M::Diagonal, keep::AbstractVector) = Diagonal(view(M.diag,keep))
MaskedSymmetricMatrix(M::AbstractMatrix, keep::AbstractVector) = view(M, keep, keep)

## Subset which rows are kept
MaskedJacobian(J::AbstractMatrix, ::Nothing) = J
MaskedJacobian(J::AbstractMatrix, keep::AbstractVector) = view(J, keep, :)


function _ReconstructDataMatrix(Ydata::AbstractVector, datakeep::AbstractVector{<:Bool}, Ydim::Int)
    @assert sum(datakeep) == length(Ydata)
    Res = fill(NaN, length(datakeep));    k = 1
    for i in eachindex(Res)
        datakeep[i] && (Res[i] = Ydata[k];    k += 1)
    end;    _ReconstructDataMatrix(Res, nothing, Ydim)
end
_ReconstructDataMatrix(Ydata::AbstractVector, ::Nothing, Ydim::Int) = transpose(reshape(Ydata, Ydim, :))

## Returns Xmatrix, Ymatrix
ReconstructDataMatrices(DSU::DataSetUncertain) = _ReconstructDataMatrix(xdata(DSU), nothing, xdim(DSU)), _ReconstructDataMatrix(ydata(DSU), DSU.datakeep, ydim(DSU))
ReconstructDataMatrices(DS::AbstractDataSet, args...) = (@assert !HasMissingValues(DS);    (_ReconstructDataMatrix(xdata(DS), nothing, xdim(DS)), _ReconstructDataMatrix(ydata(DS), nothing, ydim(DS))))


ReconstructYdataSigmaMatrix(DSU::DataSetUncertain, testpy::AbstractVector=DSU.testpy) = _ReconstructDataMatrix(ysigma(DSU, testpy), DSU.datakeep, ydim(DSU))


# using non-raw out-of-place version of error model here
function _InvCov(inverrmod::Function, errorparams::AbstractVector, woundX::AbstractVector, woundY::AbstractVector, testout=nothing)
    ErrorPrediction = (x,y) -> (S=inverrmod(x,y,errorparams);      S' * S) ## inplace-ify this step
    map(ErrorPrediction, woundX, woundY) |> BlockMatrix
end



## Bessel correction should only be applied in likelihood for correct weighting, not in ysigma and YInvCov

# Uncertainty must be constructed around prediction!
function ysigma(DS::DataSetUncertain{BesselCorrection,Nothing}, c::AbstractVector{<:Number}=DS.testpy; verbose::Bool=true) where BesselCorrection
    C = if length(c) != length(DS.testpy)
        verbose && @warn "ysigma: Given parameters not of expected length - expected $(length(DS.testpy)) got $(length(c)). Only pass error params!"
        (SplitErrorParams(DS)(c))[end]
    else
        verbose && c === DS.testpy && @warn "ysigma: Cheating by not constructing uncertainty around given prediction."
        c
    end;    errmod = yinverrormodel(DS)
    map((x,y)->inv(errmod(x,y,C)), WoundX(DS), WoundY(DS)) |> _TryVectorizeNoSqrt
end

function yInvCov(DS::DataSetUncertain{BesselCorrection,Nothing}, c::AbstractVector{<:Number}=DS.testpy; verbose::Bool=true) where BesselCorrection
    C = if length(c) != length(DS.testpy)
        verbose && @warn "yInvCov: Given parameters not of expected length - expected $(length(DS.testpy)) got $(length(c)). Only pass error params."
        (SplitErrorParams(DS)(c))[end]
    else
        verbose && c === DS.testpy && @warn "yInvCov: Cheating by not constructing uncertainty around given prediction."
        c
    end;    _InvCov(yinverrormodel(DS), C, WoundX(DS), WoundY(DS), DS.testout)
end


## Masked versions for missing data
function ysigma(DS::DataSetUncertain{BesselCorrection,Keep}, c::AbstractVector{<:Number}=DS.testpy; verbose::Bool=true) where {BesselCorrection,Keep}
    C = if length(c) != length(DS.testpy)
        verbose && @warn "ysigma: Given parameters not of expected length - expected $(length(DS.testpy)) got $(length(c)). Only pass error params!"
        (SplitErrorParams(DS)(c))[end]
    else
        verbose && c === DS.testpy && @warn "ysigma: Cheating by not constructing uncertainty around given prediction."
        c
    end;    errmod = yinverrormodel(DS)
    try
        map((x,y)->inv(errmod(x,y,C)), WoundX(DS), WoundYmasked(DS)) |> _TryVectorizeNoSqrt |> x->view(x, DS.datakeep)
    catch E;
        verbose && println(E)
        Fill(NaN, length(ydata(DS)))
    end
end

function yInvCov(DS::DataSetUncertain{BesselCorrection,Keep}, c::AbstractVector{<:Number}=DS.testpy; verbose::Bool=true) where {BesselCorrection,Keep}
    C = if length(c) != length(DS.testpy)
        verbose && @warn "yInvCov: Given parameters not of expected length - expected $(length(DS.testpy)) got $(length(c)). Only pass error params."
        (SplitErrorParams(DS)(c))[end]
    else
        verbose && c === DS.testpy && @warn "yInvCov: Cheating by not constructing uncertainty around given prediction."
        c
    end
    try
        MaskedSymmetricMatrix(_InvCov(yinverrormodel(DS), C, WoundX(DS), WoundYmasked(DS), DS.testout), DS.datakeep)
    catch E;
        verbose && println(E)
        Diagonal(Fill(NaN, length(ydata(DS))))
    end
end


function EmbeddingMap(DSU::DataSetUncertain{<:Any,<:AbstractVector}, model::ModelOrFunction, θ::AbstractVector{<:Number}, woundX::AbstractVector=WoundX(DSU); kwargs...)
    EmbeddingMap(Val(:Masked), model, θ, woundX, DSU.datakeep; kwargs...)
end
function EmbeddingMatrix(DSU::DataSetUncertain{<:Any,<:AbstractVector}, dmodel::ModelOrFunction, θ::AbstractVector{<:Number}, woundX::AbstractVector=WoundX(DSU); kwargs...)
    EmbeddingMatrix(Val(:Masked), dmodel, θ, woundX, DSU.datakeep; kwargs...)
end



function _loglikelihood(DS::DataSetUncertain{BesselCorrection,Nothing}, model::ModelOrFunction, θ::AbstractVector{T}; kwargs...)::T where T<:Number where BesselCorrection
    Splitter = SplitErrorParams(DS);    normalparams, errorparams = Splitter(θ);    yinverrmod = yinverrormodel(DS)
    woundYpred = Windup(EmbeddingMap(DS, model, normalparams; kwargs...), ydim(DS))
    Bessel = BesselCorrection ? sqrt((length(ydata(DS))-DOF(DS, θ))/(length(ydata(DS)))) : one(T)
    woundY = WoundY(DS);    woundX = WoundX(DS)
    woundInvσ = map((x,y)->Bessel .* yinverrmod(x,y,errorparams), woundX, woundYpred)
    function _Eval(woundYpred, woundInvσ, woundY)
        Res::T = -(length(woundY)*length(woundY[1]))*log(2T(π))
        @inbounds for i in eachindex(woundY)
            Res += 2logdet(woundInvσ[i])
            Res -= sum(abs2, woundInvσ[i] * (woundY[i] - woundYpred[i]))
        end
        Res *= 0.5;    Res
    end;    _Eval(woundYpred, woundInvσ, woundY)
end

### Missing data
function _loglikelihood(DS::DataSetUncertain{BesselCorrection,Keep}, model::ModelOrFunction, θ::AbstractVector{T}; kwargs...)::T where T<:Number where {BesselCorrection,Keep}
    Splitter = SplitErrorParams(DS);    normalparams, errorparams = Splitter(θ);    yinverrmod = yinverrormodelraw(DS) ### raw here
    yPredSparse = EmbeddingMap(Val(true), model, normalparams, DS.woundXpred; kwargs...)
    woundYpredSparse = Windup(yPredSparse, ydim(DS))
    Bessel = BesselCorrection ? sqrt((length(ydata(DS))-DOF(DS, θ))/(length(ydata(DS)))) : one(T)
    ## Currently only works for diagonal error models
    InvσSparse = map((x,y)->Bessel .* yinverrmod(x,y,errorparams), DS.woundXpred, woundYpredSparse) |> Reduction
    
    function _Eval(Ypred, Invσ, Ydat)
        @assert length(Ydat) == length(Ypred) == length(Invσ)
        Res::T = -length(Ydat)*log(2T(π))
        @inbounds for i in eachindex(Ydat)
            Res += 2log(Invσ[i]) - sum(abs2, Invσ[i] * (Ydat[i] - Ypred[i]))
        end
        Res *= 0.5;    Res
    end;    _Eval((@view yPredSparse[DS.predkeep]), (@view InvσSparse[DS.predkeep]), ydata(DS))
end

# Requires _Score !!!

# Build dedicated method for constructing full cholesky matrix from given error model and add to struct.

# Potential for optimization by specializing on Type of invcov
# AutoMetric SIGNIFICANTLY more performant for large datasets since orders of magnitude less allocations
function _FisherMetric(DS::DataSetUncertain{BesselCorrection,Nothing}, model::ModelOrFunction, dmodel::ModelOrFunction, θ::AbstractVector{T}; ADmode::Val=Val(:ForwardDiff), kwargs...) where T<:Number where BesselCorrection
    Splitter = SplitErrorParams(DS);    normalparams, errorparams = Splitter(θ);    yinverrmod = yinverrormodel(DS)
    woundX = WoundX(DS)
    woundYpred = Windup(EmbeddingMap(Val(true), model, normalparams, woundX), ydim(DS))
    Bessel = BesselCorrection ? sqrt((length(ydata(DS))-DOF(DS, θ))/(length(ydata(DS)))) : one(T)
    woundInvσ = map((x,y)->Bessel .* yinverrmod(x,y,errorparams), woundX, woundYpred)
    Σneghalf = BlockMatrix(woundInvσ)

    NormalParamJac = SplitterJacNormalParams(DS)(θ)
    SJ = (MaskedSymmetricMatrix(Σneghalf, DS.datakeep) * MaskedJacobian(EmbeddingMatrix(Val(true), dmodel, normalparams, woundX), DS.datakeep)) * NormalParamJac
    F_m = transpose(SJ) * SJ

    Σposhalf = BlockMatrix(map(inv, woundInvσ))
    function InvSqrtCovFromFull(θ::AbstractVector)
        normalparams, errorparams = Splitter(θ)
        BlockMatrix(map((x,y)->Bessel .* yinverrmod(x,y,errorparams), woundX, Windup(EmbeddingMap(Val(true), model, normalparams, woundX), ydim(DS))))
    end
    ΣneghalfJac = GetMatrixJac(ADmode, InvSqrtCovFromFull, length(θ), size(Σposhalf))(θ)

    # @tullio F_e[i,j] := 2 * Σposhalf[a,b] * ΣneghalfJac[b,c,i] * Σposhalf[c,d] * ΣneghalfJac[d,a,j]
    @inline AddCovarianceContribution!(F_m::AbstractMatrix, Σposhalf::AbstractMatrix, ΣneghalfJac::AbstractArray{<:Number,3}) = @tullio F_m[i,j] += 2 * Σposhalf[a,b] * ΣneghalfJac[b,c,i] * Σposhalf[c,d] * ΣneghalfJac[d,a,j]
    @inline function AddCovarianceContribution!(F_m::AbstractMatrix, Σposhalf::Diagonal, ΣneghalfJac::AbstractMatrix) # (N × θ) in diagonal case
        s = Σposhalf.diag;  @tullio F_m[i,j] += 2 * s[a]^2 * ΣneghalfJac[a,i] * ΣneghalfJac[a,j]    # F_m .+= 2 .* (ΣneghalfJac' * Diagonal(s.^2) * ΣneghalfJac)
    end

    AddCovarianceContribution!(F_m, Σposhalf, ΣneghalfJac)
    # F_m .+ F_e
    F_m
end

# function _Score(DSE::DataSetExact, model::ModelOrFunction, dmodel::ModelOrFunction, θ::AbstractVector{<:Number}, ADmode::Val{false}; kwargs...)
#     transpose(EmbeddingMatrix(DSE,dmodel,θ; kwargs...)) * gradlogpdf(ydist(DSE), EmbeddingMap(DSE,model,θ; kwargs...))
# end
