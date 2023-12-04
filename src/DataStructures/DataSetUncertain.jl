

# Use general bitvector mask to implement missing values

"""
    DataSetUncertain(x::AbstractVector, y::AbstractVector, σ⁻¹::Function, c::AbstractVector)
The `DataSetUncertain` type encodes data for which the size of the variance is unknown a-priori but whose error is specified via an error model of the form `σ(x, y_pred, c)` where `c` is a vector of error parameters.
This parametrized error model is subsequently used to estimate the standard deviations in the observations `y`.
!!! note
    To enhance performance, the implementation actually requires the specification of a *reciprocal* error model, i.e. a function `σ⁻¹(x, y_pred, c)`.

To construct a `DataSetUncertain`, one has to specify a vector of independent variables `x`, a vector of dependent variables `y`, a reciprocal error model `σ⁻¹(x, y_pred, c)` and an initial guess for the vector of error parameters `c`.

Examples:

In the simplest case, where all data points are mutually independent and have a single ``x``-component and a single ``y``-component each, a `DataSet` consisting of four points can be constructed via
```julia
DS = DataSetUncertain([1,2,3,4], [4,5,6.5,7.8], (x,y,c)->1/abs(c[1]), [0.5])
```
"""
struct DataSetUncertain <: AbstractUnknownUncertaintyDataSet
    x::AbstractVector{<:Number}
    y::AbstractVector{<:Number}
    dims::Tuple{Int,Int,Int}
    inverrormodel::Function # 1./errormodel
    testp::AbstractVector{<:Number}
    errorparamsplitter::Function # θ -> (view(θ, MODEL), view(θ, ERRORMODEL))
    xnames::AbstractVector{<:AbstractString}
    ynames::AbstractVector{<:AbstractString}
    name::Union{<:AbstractString,<:Symbol}

    function DataSetUncertain(X::AbstractArray{<:Number}, Y::AbstractArray{<:Number}, inverrormodel::Function, testp::AbstractVector; kwargs...)
        size(X,1) != size(Y,1) && throw("Inconsistent number of x-values and y-values given: $(size(X,1)) != $(size(Y,1)). Specify a tuple (Npoints, xdim, ydim) in the constructor.")
        DataSetUncertain(collect(eachrow(X)), collect(eachrow(Y)), inverrormodel, testp; kwargs...)
    end
    function DataSetUncertain(X::AbstractVector, Y::AbstractVector, inverrormodel::Function, testp::AbstractVector; kwargs...)
        size(X,1) != size(Y,1) && throw("Inconsistent number of x-values and y-values given: $(size(X,1)) != $(size(Y,1)). Specify a tuple (Npoints, xdim, ydim) in the constructor.")
        DataSetUncertain(Unwind(X), Unwind(Y), inverrormodel, testp, (size(X,1), ConsistentElDims(X), ConsistentElDims(Y)); kwargs...)
    end
    function DataSetUncertain(x::AbstractVector, y::AbstractVector, inverrormodel::Function, testp::AbstractVector, dims::Tuple{Int,Int,Int}; kwargs...)
        @info "Assuming error parameters always given by last $(length(testp)) parameters."
        DataSetUncertain(x, y, dims, inverrormodel, DefaultErrorModel(length(testp)), testp; kwargs...)
    end
    function DataSetUncertain(x::AbstractVector, y::AbstractVector, dims::Tuple{Int,Int,Int}, inverrormodel::Function, errorparamsplitter::Function, testp::AbstractVector;
            xnames::AbstractVector{<:String}=CreateSymbolNames(xdim(dims),"x"), ynames::AbstractVector{<:String}=CreateSymbolNames(ydim(dims),"y"),
            name::Union{String,Symbol}=Symbol(), kwargs...)
        @assert all(x->(x > 0), dims) "Not all dims > 0: $dims."
        @assert Npoints(dims) == Int(length(x)/xdim(dims)) == Int(length(y)/ydim(dims)) "Inconsistent input dimensions."
        @assert length(xnames) == xdim(dims) && length(ynames) == ydim(dims)
        # Check that inverrormodel either outputs Matrix for ydim > 1
        if ydim(dims) == 1
            M = _TestErrorModel(inverrormodel, Windup(x, xdim(dims))[1], Windup(y, ydim(dims))[1], testp)
            @assert M isa Number && M > 0
        else
            M = _TestErrorModel(inverrormodel, Windup(x, xdim(dims))[1], Windup(y, ydim(dims))[1], testp)
            @assert M isa AbstractMatrix && size(M,1) == size(M,2) == ydim(dims) && det(M) > 0
        end
        new(x, y, dims, inverrormodel, testp, errorparamsplitter, xnames, ynames, name)
    end
end

DefaultErrorModel(n::Int) = ((θ::AbstractVector{<:Number}; kwargs...) -> @views (θ[1:end-n], θ[end-n+1:end]))


xdata(DS::DataSetUncertain) = DS.x
ydata(DS::DataSetUncertain) = DS.y
dims(DS::DataSetUncertain) = DS.dims
xnames(DS::DataSetUncertain) = DS.xnames
ynames(DS::DataSetUncertain) = DS.ynames
name(DS::DataSetUncertain) = DS.name |> string

xsigma(DS::DataSetUncertain) = zeros(length(xdata(DS)))

errormoddim(DS::DataSetUncertain) = length(DS.testp)

SplitErrorParams(DS::DataSetUncertain) = DS.errorparamsplitter


_TestErrorModel(DS::DataSetUncertain) = _TestErrorModel(DS.inverrormodel, WoundX(DS)[1], WoundY(DS)[1], DS.testp)
_TestErrorModel(inverrormodel::Function, x, y, c::AbstractVector) = inverrormodel(x,y,c)

_TryVectorizeNoSqrt(X::AbstractVector{<:Number}) = X
_TryVectorizeNoSqrt(X::AbstractVector{<:AbstractArray}) = InformationGeometry.BlockReduce(X) |> _TryVectorizeNoSqrt
_TryVectorizeNoSqrt(M::AbstractMatrix) = isdiag(M) ? Diagonal(M).diag : M
_TryVectorizeNoSqrt(D::Diagonal) = D.diag

# Uncertainty must be constructed around prediction!
function ysigma(DS::DataSetUncertain, c::AbstractVector{<:Number}=DS.testp)
    c === DS.testp && @warn "Cheating by not constructing uncertainty around given prediction."
    map((x,y)->inv(DS.inverrormodel(x,y,c)), WoundX(DS), WoundY(DS)) |> _TryVectorizeNoSqrt
end


BlockReduce(X::AbstractVector{<:Number}) = Diagonal(X)
function yInvCov(DS::DataSetUncertain, c::AbstractVector=DS.testp)
    c === DS.testp && @warn "Cheating by not constructing uncertainty around given prediction."
    map(((x,y)->(S=DS.inverrormodel(x,y,c); S' * S)), WoundX(DS), WoundY(DS)) |> BlockReduce
end


function _loglikelihood(DS::DataSetUncertain, model::ModelOrFunction, θ::AbstractVector{<:Number}; kwargs...)
    normalparams, errorparams = DS.errorparamsplitter(θ)
    woundYpred = Windup(EmbeddingMap(DS, model, θ; kwargs...), ydim(DS))
    woundInvσ = map((x,y)->DS.inverrormodel(x,y,errorparams), WoundX(DS), woundYpred)
    woundY = WoundY(DS)
    function _Eval(DS, woundYpred, woundInvσ, woundY)
        Res = -DataspaceDim(DS)*log(2π)
        @inbounds for i in 1:length(woundY)
            Res += 2logdet(woundInvσ[i])
            Res -= sum(abs2, woundInvσ[i] * (woundY[i] - woundYpred[i]))
        end
        Res *= 0.5;    Res
    end;    _Eval(DS, woundYpred, woundInvσ, woundY)
end


# Potential for optimization by specializing on Type of invcov
function _FisherMetric(DS::DataSetUncertain, model::ModelOrFunction, dmodel::ModelOrFunction, θ::AbstractVector{<:Number}; ADmode::Val=Val(:ForwardDiff), kwargs...)
    normalparams, errorparams = DS.errorparamsplitter(θ)
    woundYpred = Windup(EmbeddingMap(DS, model, θ), ydim(DS))
    woundInvσ = BlockReduce(map((x,y)->DS.inverrormodel(x,y,errorparams), WoundX(DS), woundYpred))
    J = EmbeddingMatrix(DS, dmodel, θ)
    F_m = transpose(J) * transpose(woundInvσ) * woundInvσ * J

    Σposhalf = inv(woundInvσ)
    InvSqrtCovFromFull(θ) = ((normalparams, errorparams) = DS.errorparamsplitter(θ);  BlockReduce(map((x,y)->DS.inverrormodel(x,y,errorparams), WoundX(DS), Windup(EmbeddingMap(DS, model, θ), ydim(DS)))))
    ΣneghalfJac = GetMatrixJac(ADmode, InvSqrtCovFromFull, length(θ), size(Σposhalf))(θ)

    @tullio F_e[i,j] := 2 * Σposhalf[a,b] * ΣneghalfJac[b,c,i] * Σposhalf[c,d] * ΣneghalfJac[d,a,j]
    F_m + F_e
end

# function _Score(DSE::DataSetExact, model::ModelOrFunction, dmodel::ModelOrFunction, θ::AbstractVector{<:Number}, ADmode::Val{false}; kwargs...)
#     transpose(EmbeddingMatrix(DSE,dmodel,θ; kwargs...)) * gradlogpdf(ydist(DSE), EmbeddingMap(DSE,model,θ; kwargs...))
# end
