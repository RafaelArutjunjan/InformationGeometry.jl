

# Use general bitvector mask to implement missing values

"""
    UnknownVarianceDataSet(x::AbstractVector, y::AbstractVector, σ⁻¹::Function, c::AbstractVector)
The `UnknownVarianceDataSet` type encodes data for which the size of the variance is unknown a-priori but whose error is specified via an error model of the form `σ(x, y_pred, c)` where `c` is a vector of error parameters.
This parametrized error model is subsequently used to estimate the standard deviations in the observations `y`.
!!! note
    To enhance performance, the implementation actually requires the specification of a *reciprocal* error model, i.e. a function `σ⁻¹(x, y_pred, c)`.

To construct a `UnknownVarianceDataSet`, one has to specify a vector of independent variables `x`, a vector of dependent variables `y`, a reciprocal error model `σ⁻¹(x, y_pred, c)` and an initial guess for the vector of error parameters `c`.

Examples:

In the simplest case, where all data points are mutually independent and have a single ``x``-component and a single ``y``-component each, a `DataSet` consisting of four points can be constructed via
```julia
DS = UnknownVarianceDataSet([1,2,3,4], [4,5,6.5,7.8], (x,y,c)->1/abs(c[1]), [0.5])
```
"""
struct UnknownVarianceDataSet <: AbstractUnknownUncertaintyDataSet
    x::AbstractVector{<:Number}
    y::AbstractVector{<:Number}
    dims::Tuple{Int,Int,Int}
    invXvariancemodel::Function # σₓ⁻²
    invYvariancemodel::Function # σ_y⁻²
    testpx::AbstractVector{<:Number}
    testpy::AbstractVector{<:Number}
    errorparamsplitter::Function # θ -> (view(θ, MODEL), view(θ, xERRORMODEL), view(θ, yERRORMODEL))
    xnames::AbstractVector{<:AbstractString}
    ynames::AbstractVector{<:AbstractString}
    name::Union{<:AbstractString,<:Symbol}

    
    function UnknownVarianceDataSet(x::AbstractArray, y::AbstractArray, Testpx::AbstractVector, Testpy::AbstractVector; kwargs...)
        UnknownVarianceDataSet(x, y; testpx=Testpx, testpy=Testpy, kwargs...)
    end
    function UnknownVarianceDataSet(X::AbstractArray, Y::AbstractArray, dims::Tuple{Int,Int,Int}=(size(X,1), ConsistentElDims(X), ConsistentElDims(Y)); 
                        testpx::AbstractVector=zeros(xdim(dims)), testpy::AbstractVector=zeros(ydim(dims)), kwargs...)
        size(X,1) != size(Y,1) && throw("Inconsistent number of x-values and y-values given: $(size(X,1)) != $(size(Y,1)). Specify a tuple (Npoints, xdim, ydim) in the constructor.")
        @info "Assuming error models σ(x,y,c) = exp10.(c)"
        xerrmod = xdim(dims) == 1 ? ((x,y,c)->inv(exp10(c[1]))) : (x,y,c)->inv.(exp10.(c))
        yerrmod = ydim(dims) == 1 ? ((x,y,c)->inv(exp10(c[1]))) : (x,y,c)->inv.(exp10.(c))
        UnknownVarianceDataSet(Unwind(X), Unwind(Y), xerrmod, yerrmod, testpx, testpy, dims; kwargs...)
    end
    function UnknownVarianceDataSet(x::AbstractVector, y::AbstractVector, invXvariancemodel::Function, invYvariancemodel::Function,
        testpx::AbstractVector, testpy::AbstractVector, dims::Tuple{Int,Int,Int}=(size(X,1), ConsistentElDims(X), ConsistentElDims(Y)); kwargs...)
        @info "Assuming error parameters always given by last ($(length(testpx)),$(length(testpy))) parameters."
        # Error param splitter
        UnknownVarianceDataSet(Unwind(x), Unwind(y), dims, invXvariancemodel, invYvariancemodel, testpx, testpy, DefaultErrorModel(length(testpx),length(testpy)); kwargs...)
    end
    function UnknownVarianceDataSet(x::AbstractVector, y::AbstractVector, dims::Tuple{Int,Int,Int}, 
            invXvariancemodel::Function, invYvariancemodel::Function, testpx::AbstractVector, testpy::AbstractVector, errorparamsplitter::Function;
            xnames::AbstractVector{<:String}=CreateSymbolNames(xdim(dims),"x"), ynames::AbstractVector{<:String}=CreateSymbolNames(ydim(dims),"y"),
            name::Union{String,Symbol}=Symbol(), kwargs...)
            UnknownVarianceDataSet(x, y, dims, invXvariancemodel, invYvariancemodel, testpx, testpy, errorparamsplitter, xnames, ynames, name; kwargs...)
    end
    function UnknownVarianceDataSet(x::AbstractVector, y::AbstractVector, dims::Tuple{Int,Int,Int}, 
        invXvariancemodel::Function, invYvariancemodel::Function, testpx::AbstractVector, testpy::AbstractVector, errorparamsplitter::Function,
        xnames::AbstractVector{<:String}, ynames::AbstractVector{<:String}, name::Union{String,Symbol}=Symbol())
        @assert all(x->(x > 0), dims) "Not all dims > 0: $dims."
        @assert Npoints(dims) == Int(length(x)/xdim(dims)) == Int(length(y)/ydim(dims)) "Inconsistent input dimensions."
        @assert length(xnames) == xdim(dims) && length(ynames) == ydim(dims)
        @warn "Missing error model tests"
        ## Check that inverrormodel either outputs Matrix for ydim > 1
        Q = invXvariancemodel(Windup(x, xdim(dims))[1], Windup(y, ydim(dims))[1], testpx)
        M = invYvariancemodel(Windup(x, xdim(dims))[1], Windup(y, ydim(dims))[1], testpy)
        xdim(dims) == 1 ? (@assert Q isa Number && Q > 0) : (@assert Q isa AbstractMatrix && size(Q,1) == size(Q,2) == xdim(dims) && det(Q) > 0)
        ydim(dims) == 1 ? (@assert M isa Number && M > 0) : (@assert M isa AbstractMatrix && size(M,1) == size(M,2) == ydim(dims) && det(M) > 0)
        new(x, y, dims, invXvariancemodel, invYvariancemodel, testpx, testpy, errorparamsplitter, xnames, ynames, name)
    end
end


function (::Type{T})(DS::UnknownVarianceDataSet; kwargs...) where T<:Number
	UnknownVarianceDataSet(T.(xdata(DS)), T.(ydata(DS)), dims(DS), xinverrormodel(DS), yinverrormodel(DS), 
                T.(DS.testpx), T.(DS.testpy), SplitErrorParams(DS); xnames=xnames(DS), ynames=ynames(DS), name=name(DS), kwargs...)
end

# For SciMLBase.remake
UnknownVarianceDataSet(;
x::AbstractVector=[0.],
y::AbstractVector=[0.],
dims::Tuple{Int,Int,Int}=(1,1,1),
invXvariancemodel::Function=identity,
invYvariancemodel::Function=identity,
testpx::AbstractVector{<:Number}=[0.],
testpy::AbstractVector{<:Number}=[0.],
errorparamsplitter::Function=x->(x[1], x[2]),
xnames::AbstractVector{String}=["x"],
ynames::AbstractVector{String}=["y"],
name::Union{String,Symbol}=Symbol()) = UnknownVarianceDataSet(x, y, dims, invXvariancemodel, invYvariancemodel, testpx, testpy, errorparamsplitter, xnames, ynames, name)


DefaultErrorModel(n::Int, m::Int) = ((θ::AbstractVector{<:Number}; kwargs...) -> @views (θ[1:end-n-m], θ[end-n-m+1:end-m], θ[end-m+1:end]))


xdata(DS::UnknownVarianceDataSet) = DS.x
ydata(DS::UnknownVarianceDataSet) = DS.y
dims(DS::UnknownVarianceDataSet) = DS.dims
xnames(DS::UnknownVarianceDataSet) = DS.xnames
ynames(DS::UnknownVarianceDataSet) = DS.ynames
name(DS::UnknownVarianceDataSet) = DS.name |> string


xerrormoddim(DS::UnknownVarianceDataSet) = length(DS.testpx)
yerrormoddim(DS::UnknownVarianceDataSet) = length(DS.testpy)
errormoddim(DS::UnknownVarianceDataSet) = xerrormoddim(DS) + yerrormoddim(DS)

SplitErrorParams(DS::UnknownVarianceDataSet) = DS.errorparamsplitter

xinverrormodel(DS::UnknownVarianceDataSet) = DS.invXvariancemodel
yinverrormodel(DS::UnknownVarianceDataSet) = DS.invYvariancemodel


xerrorparams(DS::UnknownVarianceDataSet, mle::AbstractVector) = (SplitErrorParams(DS)(mle))[2]
yerrorparams(DS::UnknownVarianceDataSet, mle::AbstractVector) = (SplitErrorParams(DS)(mle))[3]


# Uncertainty must be constructed around prediction!
function xsigma(DS::UnknownVarianceDataSet, c::AbstractVector{<:Number}=DS.testpx)
    c === DS.testpx && @warn "Cheating by not constructing uncertainty around given prediction."
    map((x,y)->inv(xinverrormodel(DS)(x,y,c)), WoundX(DS), WoundY(DS)) |> _TryVectorizeNoSqrt
end

function xInvCov(DS::UnknownVarianceDataSet, c::AbstractVector=DS.testpx)
    c === DS.testpx && @warn "Cheating by not constructing uncertainty around given prediction."
    map(((x,y)->(S=xinverrormodel(DS)(x,y,c); S' * S)), WoundX(DS), WoundY(DS)) |> BlockReduce
end

# Uncertainty must be constructed around prediction!
function ysigma(DS::UnknownVarianceDataSet, c::AbstractVector{<:Number}=DS.testpy)
    c === DS.testpy && @warn "Cheating by not constructing uncertainty around given prediction."
    map((x,y)->inv(yinverrormodel(DS)(x,y,c)), WoundX(DS), WoundY(DS)) |> _TryVectorizeNoSqrt
end

function yInvCov(DS::UnknownVarianceDataSet, c::AbstractVector=DS.testpy)
    c === DS.testpy && @warn "Cheating by not constructing uncertainty around given prediction."
    map(((x,y)->(S=yinverrormodel(DS)(x,y,c); S' * S)), WoundX(DS), WoundY(DS)) |> BlockReduce
end


function _loglikelihood(DS::UnknownVarianceDataSet, model::ModelOrFunction, θ::AbstractVector{<:Number}; kwargs...)
    normalparams, xerrorparams, yerrorparams = SplitErrorParams(DS)(θ)
    # Emb = LiftedEmbedding(DS, model, length(normalparams)-length(xdata(DS)))

    # Lets xp pass through to model UNLIKE LiftedEmbedding currently!
    function LiftedEmb(ξ::AbstractVector; kwargs...)
        xdat = view(ξ, 1:length(xdata(DS)))
        [xdat; EmbeddingMap(DS, model, ξ, Windup(xdat, xdim(DS)); kwargs...)]
    end
    XY = LiftedEmb(normalparams)
    woundXpred = Windup(view(XY, 1:length(xdata(DS))), xdim(DS))
    woundYpred = Windup(view(XY, length(xdata(DS))+1:length(XY)), ydim(DS))
    woundInvXσ = map((x,y)->xinverrormodel(DS)(x,y,xerrorparams), woundXpred, woundYpred)
    woundInvYσ = map((x,y)->yinverrormodel(DS)(x,y,yerrorparams), woundXpred, woundYpred)
    woundX = WoundX(DS);    woundY = WoundY(DS)
    function _Eval(DS, woundYpred, woundInvYσ, woundY, woundXpred, woundInvXσ, woundX)
        Res = -(length(ydata(DS)) + length(xdata(DS)))*log(2π)
        for i in eachindex(woundY)
            Res += 2logdet(woundInvYσ[i])
            Res -= sum(abs2, woundInvYσ[i] * (woundY[i] - woundYpred[i]))
        end
        for j in eachindex(woundX)
            Res += 2logdet(woundInvXσ[j])
            Res -= sum(abs2, woundInvXσ[j] * (woundX[j] - woundXpred[j]))
        end
        Res *= 0.5;    Res
    end;    _Eval(DS, woundYpred, woundInvYσ, woundY, woundXpred, woundInvXσ, woundX)
end


# Can get parameter indices by SplitErrorParams(DS)(1:length(θ))

# Potential for optimization by specializing on Type of invcov
function _FisherMetric(DS::UnknownVarianceDataSet, model::ModelOrFunction, dmodel::ModelOrFunction, θ::AbstractVector{<:Number}; ADmode::Val=Val(:ForwardDiff), kwargs...)
    normalparams, xerrorparams, yerrorparams = SplitErrorParams(DS)(θ)
    # normalinds, xerrorinds, yerrorinds = SplitErrorParams(DS)(1:length(θ))

    # Lets xp pass through to model UNLIKE LiftedEmbedding currently!
    function LiftedEmb(ξ::AbstractVector; kwargs...)
        xdat = view(ξ, 1:length(xdata(DS)))
        [xdat; EmbeddingMap(DS, model, ξ, Windup(xdat, xdim(DS)); kwargs...)]
    end
    XY = LiftedEmb(normalparams)
    woundXpred = Windup(view(XY, 1:length(xdata(DS))), xdim(DS))
    woundYpred = Windup(view(XY, length(xdata(DS))+1:length(XY)), ydim(DS))
    woundInvXσ = map((x,y)->xinverrormodel(DS)(x,y,xerrorparams), woundXpred, woundYpred)
    woundInvYσ = map((x,y)->yinverrormodel(DS)(x,y,yerrorparams), woundXpred, woundYpred)

    J = BlockMatrix(BlockReduce(woundInvXσ), BlockReduce(woundInvYσ)) * GetJac(ADmode, LiftedEmb, length(θ))(θ)
    F_m = transpose(J) * J

    yΣposhalf = map(inv, woundInvYσ) |> BlockReduce
    function InvSqrtyCovFromFull(θ)
        normalparams, xerrorparams, yerrorparams = SplitErrorParams(DS)(θ)
        XY = LiftedEmb(normalparams)
        woundXpred = Windup(view(XY, 1:length(xdata(DS))), xdim(DS))
        woundYpred = Windup(view(XY, length(xdata(DS))+1:length(XY)), ydim(DS))
        BlockReduce(map((x,y)->yinverrormodel(DS)(x,y,yerrorparams), woundXpred, woundYpred))
    end
    yΣneghalfJac = GetMatrixJac(ADmode, InvSqrtyCovFromFull, length(θ), size(yΣposhalf))(θ)
    @tullio F_ey[i,j] := 2 * yΣposhalf[a,b] * yΣneghalfJac[b,c,i] * yΣposhalf[c,d] * yΣneghalfJac[d,a,j]
    
    xΣposhalf = map(inv, woundInvXσ) |> BlockReduce
    function InvSqrtxCovFromFull(θ)
        normalparams, xerrorparams, yerrorparams = SplitErrorParams(DS)(θ)
        XY = LiftedEmb(normalparams)
        woundXpred = Windup(view(XY, 1:length(xdata(DS))), xdim(DS))
        woundYpred = Windup(view(XY, length(xdata(DS))+1:length(XY)), ydim(DS))
        BlockReduce(map((x,y)->xinverrormodel(DS)(x,y,xerrorparams), woundXpred, woundYpred))
    end
    xΣneghalfJac = GetMatrixJac(ADmode, InvSqrtxCovFromFull, length(θ), size(xΣposhalf))(θ)
    @tullio F_ex[i,j] := 2 * xΣposhalf[a,b] * xΣneghalfJac[b,c,i] * xΣposhalf[c,d] * xΣneghalfJac[d,a,j]
    
    F_m + F_ey + F_ex
end

# function _Score(DSE::DataSetExact, model::ModelOrFunction, dmodel::ModelOrFunction, θ::AbstractVector{<:Number}, ADmode::Val{false}; kwargs...)
#     transpose(EmbeddingMatrix(DSE,dmodel,θ; kwargs...)) * gradlogpdf(ydist(DSE), EmbeddingMap(DSE,model,θ; kwargs...))
# end
