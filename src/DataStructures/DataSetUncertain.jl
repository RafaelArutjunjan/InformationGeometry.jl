

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
!!! warn
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
struct DataSetUncertain{BesselCorrection} <: AbstractUnknownUncertaintyDataSet
    x::AbstractVector{<:Number}
    y::AbstractVector{<:Number}
    dims::Tuple{Int,Int,Int} # Nxy
    inverrormodel::Function # 1./errormodel
    testp::AbstractVector{<:Number}
    errorparamsplitter::Function # θ -> (view(θ, MODEL), view(θ, ERRORMODEL))
    keep::Union{Nothing, AbstractVector{<:Bool}}
    xnames::AbstractVector{Symbol}
    ynames::AbstractVector{Symbol}
    name::Symbol

    DataSetUncertain(DS::AbstractDataSet; kwargs...) = DataSetUncertain(xdata(DS), ydata(DS), dims(DS); xnames=Xnames(DS), ynames=Ynames(DS), kwargs...)
    function DataSetUncertain(X::AbstractArray, Y::AbstractArray, dims::Tuple{Int,Int,Int}=(size(X,1), ConsistentElDims(X), ConsistentElDims(Y)); verbose::Bool=true, kwargs...)
        verbose && @info "Assuming error model σ(x,y,c) = exp10.(c)"
        errmod = ydim(dims) == 1 ? ((x,y,c::AbstractVector)->inv(exp10(c[1]))) : (x,y,c::AbstractVector)->Diagonal(inv.(exp10.(c)))
        DataSetUncertain(Unwind(X), Unwind(Y), errmod, 0.1ones(ydim(dims)), dims; verbose, kwargs...)
    end
    function DataSetUncertain(X::AbstractArray{<:Number}, Y::AbstractArray{<:Number}, inverrormodel::Function, testp::AbstractVector; kwargs...)
        size(X,1) != size(Y,1) && throw("Inconsistent number of x-values and y-values given: $(size(X,1)) != $(size(Y,1)). Specify a tuple (Npoints, xdim, ydim) in the constructor.")
        DataSetUncertain(collect(eachrow(X)), collect(eachrow(Y)), inverrormodel, testp; kwargs...)
    end
    function DataSetUncertain(X::AbstractArray, Y::AbstractArray, inverrormodel::Function, testp::AbstractVector; kwargs...)
        size(X,1) != size(Y,1) && throw("Inconsistent number of x-values and y-values given: $(size(X,1)) != $(size(Y,1)). Specify a tuple (Npoints, xdim, ydim) in the constructor.")
        DataSetUncertain(Unwind(X), Unwind(Y), inverrormodel, testp, (size(X,1), ConsistentElDims(X), ConsistentElDims(Y)); kwargs...)
    end
    DataSetUncertain(DS::AbstractDataSet, inverrormodel::Function, testp::AbstractVector=0.1ones(ydim(DS)); kwargs...) = DataSetUncertain(xdata(DS), ydata(DS), inverrormodel, testp, dims(DS); xnames=Xnames(DS), ynames=Ynames(DS), kwargs...)
    function DataSetUncertain(x::AbstractVector, y::AbstractVector, inverrormodel::Function, testp::AbstractVector, dims::Tuple{Int,Int,Int}; verbose::Bool=true, kwargs...)
        verbose && @info "Assuming error parameters always given by last $(length(testp)) parameters."
        DataSetUncertain(x, y, inverrormodel, DefaultErrorModelSplitter(length(testp)), testp, dims; verbose, kwargs...)
    end
    function DataSetUncertain(x::AbstractVector, y::AbstractVector, inverrormodel::Function, errorparamsplitter::Function, testp::AbstractVector, dims::Tuple{Int,Int,Int};
            xnames::AbstractVector{<:StringOrSymb}=CreateSymbolNames(xdim(dims),"x"), ynames::AbstractVector{<:StringOrSymb}=CreateSymbolNames(ydim(dims),"y"),
            name::StringOrSymb=Symbol(), kwargs...)
        DataSetUncertain(x, y, dims, inverrormodel, errorparamsplitter, testp, xnames, ynames, name; kwargs...)
    end
    function DataSetUncertain(x::AbstractVector, y::AbstractVector, dims::Tuple{Int,Int,Int}, inverrormodel::Function, errorparamsplitter::Function, testp::AbstractVector,
            xnames::AbstractVector{<:StringOrSymb}, ynames::AbstractVector{<:StringOrSymb}, name::StringOrSymb=Symbol(); keep::Union{Nothing, AbstractVector{<:Bool}}=nothing, BesselCorrection::Bool=false, verbose::Bool=true)
        @assert all(x->(x > 0), dims) "Not all dims > 0: $dims."
        @assert Npoints(dims) == Int(length(x)/xdim(dims)) == Int(length(y)/ydim(dims)) "Inconsistent input dimensions. Specify a tuple (Npoints, xdim, ydim) in the constructor."
        @assert length(xnames) == xdim(dims) && length(ynames) == ydim(dims)
        @assert isnothing(keep) || length(keep) == length(y)
        # Check that inverrormodel either outputs Matrix for ydim > 1
        M = inverrormodel(Windup(x, xdim(dims))[1], Windup(y, ydim(dims))[1], testp)
        ydim(dims) == 1 ? (@assert M isa Number && M > 0) : (@assert M isa AbstractMatrix && size(M,1) == size(M,2) == ydim(dims) && det(M) > 0)
        
        new{BesselCorrection}(x, y, dims, inverrormodel, testp, errorparamsplitter, keep, Symbol.(xnames), Symbol.(ynames), Symbol(name))
    end
end

function (::Type{T})(DS::DataSetUncertain{B}; kwargs...) where T<:Number where B
	DataSetUncertain(T.(xdata(DS)), T.(ydata(DS)), dims(DS), yinverrormodel(DS), SplitErrorParams(DS), T.(DS.testp), Xnames(DS), Ynames(DS), name(DS); keep=DS.keep, BesselCorrection=B, kwargs...)
end

# For SciMLBase.remake
DataSetUncertain(;
x::AbstractVector=[0.],
y::AbstractVector=[0.],
dims::Tuple{Int,Int,Int}=(1,1,1),
inverrormodel::Function=identity,
testp::AbstractVector{<:Number}=[0.],
errorparamsplitter::Function=x->(x[1], x[2]),
xnames::AbstractVector{<:StringOrSymb}=[:x],
ynames::AbstractVector{<:StringOrSymb}=[:y],
BesselCorrection::Bool=false,
verbose::Bool=true,
keep::Union{Nothing, AbstractVector{<:Bool}}=nothing,
name::StringOrSymb=Symbol()) = DataSetUncertain(x, y, dims, inverrormodel, errorparamsplitter, testp, xnames, ynames, name; keep, BesselCorrection, verbose)

DefaultErrorModelSplitter(n::Int) = ((θ::AbstractVector{<:Number}; kwargs...) -> @views (θ[1:end-n], θ[end-n+1:end]))


xdata(DS::DataSetUncertain) = DS.x
ydata(DS::DataSetUncertain) = DS.y
dims(DS::DataSetUncertain) = DS.dims
xnames(DS::DataSetUncertain) = Xnames(DS) .|> string
ynames(DS::DataSetUncertain) = Ynames(DS) .|> string
Xnames(DS::DataSetUncertain) = DS.xnames
Ynames(DS::DataSetUncertain) = DS.ynames
name(DS::DataSetUncertain) = DS.name

xsigma(DS::DataSetUncertain, mle::AbstractVector=Float64[]) = zeros(length(xdata(DS)))

HasXerror(DS::DataSetUncertain) = false

xerrormoddim(DS::DataSetUncertain) = 0
yerrormoddim(DS::DataSetUncertain) = length(DS.testp)
errormoddim(DS::DataSetUncertain) = length(DS.testp)


SplitErrorParams(DS::DataSetUncertain) = DS.errorparamsplitter

yinverrormodel(DS::DataSetUncertain) = DS.inverrormodel


xerrorparams(DS::DataSetUncertain, mle::AbstractVector) = nothing
yerrorparams(DS::DataSetUncertain, mle::AbstractVector) = (SplitErrorParams(DS)(mle))[2]

HasBessel(DS::DataSetUncertain{T}) where T = T

_TryVectorizeNoSqrt(X::AbstractVector{<:Number}) = X
_TryVectorizeNoSqrt(X::AbstractVector{<:AbstractArray}) = InformationGeometry.BlockReduce(X) |> _TryVectorizeNoSqrt
_TryVectorizeNoSqrt(M::AbstractMatrix) = isdiag(M) ? Diagonal(M).diag : M
_TryVectorizeNoSqrt(D::Diagonal) = D.diag

BlockReduce(X::AbstractVector{<:Number}) = Diagonal(X)

## Bessel correction should only be applied in likelihood for correct weighting, not in ysigma and YInvCov

# Uncertainty must be constructed around prediction!
function ysigma(DS::DataSetUncertain, c::AbstractVector{<:Number}=DS.testp; verbose::Bool=true)
    C = if length(c) != length(DS.testp) 
        verbose && @warn "ysigma: Given parameters not of expected length - expected $(length(DS.testp)) got $(length(c)). Only pass error params!"
        (SplitErrorParams(DS)(c))[end]
    else
        verbose && c === DS.testp && @warn "Cheating by not constructing uncertainty around given prediction."
        c
    end;    errmod = yinverrormodel(DS)
    map((x,y)->inv(errmod(x,y,C)), WoundX(DS), WoundY(DS)) |> _TryVectorizeNoSqrt
end

function yInvCov(DS::DataSetUncertain, c::AbstractVector{<:Number}=DS.testp; verbose::Bool=true)
    C = if length(c) != length(DS.testp) 
        verbose && @warn "yInvCov: Given parameters not of expected length - expected $(length(DS.testp)) got $(length(c)). Only pass error params."
        (SplitErrorParams(DS)(c))[end]
    else
        verbose && c === DS.testp && @warn "Cheating by not constructing uncertainty around given prediction."
        c
    end;    errmod = yinverrormodel(DS)
    map(((x,y)->(S=errmod(x,y,C); S' * S)), WoundX(DS), WoundY(DS)) |> BlockReduce
end


function _loglikelihood(DS::DataSetUncertain{BesselCorrection}, model::ModelOrFunction, θ::AbstractVector{T}; kwargs...) where T<:Number where BesselCorrection
    Splitter = SplitErrorParams(DS);    normalparams, errorparams = Splitter(θ);    yinverrmod = yinverrormodel(DS)
    woundYpred = Windup(EmbeddingMap(DS, model, normalparams; kwargs...), ydim(DS))
    Bessel = BesselCorrection ? sqrt((length(ydata(DS))-DOF(DS, θ))/(length(ydata(DS)))) : one(T)
    woundY = WoundY(DS);    woundX = WoundX(DS)
    woundInvσ = map((x,y)->Bessel .* yinverrmod(x,y,errorparams), woundX, woundYpred)
    function _Eval(DS, woundYpred, woundInvσ, woundY)
        Res::T = -DataspaceDim(DS)*log(2π)
        @inbounds for i in eachindex(woundY)
            Res += 2logdet(woundInvσ[i])
            Res -= sum(abs2, woundInvσ[i] * (woundY[i] - woundYpred[i]))
        end
        Res *= 0.5;    Res
    end;    _Eval(DS, woundYpred, woundInvσ, woundY)
end


# Potential for optimization by specializing on Type of invcov
# AutoMetric SIGNIFICANTLY more performant for large datasets since orders of magnitude less allocations
function _FisherMetric(DS::DataSetUncertain{BesselCorrection}, model::ModelOrFunction, dmodel::ModelOrFunction, θ::AbstractVector{T}; ADmode::Val=Val(:ForwardDiff), kwargs...) where T<:Number where BesselCorrection
    Splitter = SplitErrorParams(DS);    normalparams, errorparams = Splitter(θ);    yinverrmod = yinverrormodel(DS)
    woundYpred = Windup(EmbeddingMap(DS, model, normalparams), ydim(DS))
    Bessel = BesselCorrection ? sqrt((length(ydata(DS))-DOF(DS, θ))/(length(ydata(DS)))) : one(T)
    woundX = WoundX(DS)
    woundInvσ = map((x,y)->Bessel .* yinverrmod(x,y,errorparams), woundX, woundYpred)

    SJ = BlockReduce(woundInvσ) * EmbeddingMatrix(DS, dmodel, θ) # Using θ for correct F size
    F_m = transpose(SJ) * SJ

    Σposhalf = BlockReduce(map(inv, woundInvσ))
    function InvSqrtCovFromFull(θ)
        normalparams, errorparams = Splitter(θ)
        BlockReduce(map((x,y)->yinverrmod(x,y,errorparams), woundX, Windup(EmbeddingMap(DS, model, normalparams), ydim(DS))))
    end
    ΣneghalfJac = GetMatrixJac(ADmode, InvSqrtCovFromFull, length(θ), size(Σposhalf))(θ)

    @tullio F_e[i,j] := 2 * Σposhalf[a,b] * ΣneghalfJac[b,c,i] * Σposhalf[c,d] * ΣneghalfJac[d,a,j]
    F_m + F_e
end

# function _Score(DSE::DataSetExact, model::ModelOrFunction, dmodel::ModelOrFunction, θ::AbstractVector{<:Number}, ADmode::Val{false}; kwargs...)
#     transpose(EmbeddingMatrix(DSE,dmodel,θ; kwargs...)) * gradlogpdf(ydist(DSE), EmbeddingMap(DSE,model,θ; kwargs...))
# end
