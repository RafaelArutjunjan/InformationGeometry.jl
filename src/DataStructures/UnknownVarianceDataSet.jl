

# Use general bitvector mask to implement missing values

"""
    UnknownVarianceDataSet(x::AbstractVector, y::AbstractVector, σ_x⁻¹::Function, σ_y⁻¹::Function, cx::AbstractVector, cy::AbstractVector; BesselCorrection::Bool=false)
    UnknownVarianceDataSet(x::AbstractVector, y::AbstractVector, dims::Tuple{Int,Int,Int}, σ_x⁻¹::Function, σ_y⁻¹::Function, cx::AbstractVector, cy::AbstractVector, errorparamsplitter::Function; BesselCorrection::Bool=false)
The `UnknownVarianceDataSet` type encodes data for which the size of the variance is unknown a-priori but whose error is specified via an error model of the form `σ(x, y_pred, c)` where `c` is a vector of error parameters.
This parametrized error model is subsequently used to estimate the standard deviations in the observations `y`.
!!! note
    To enhance performance, the implementation actually requires the specification of a *reciprocal* error model, i.e. a function `σ⁻¹(x, y_pred, c)`.
    If `ydim` is larger than one, the reciprocal error model should output a matrix, i.e. the cholesky decomposition `S` of the covariance `Σ` such that `Σ == S' * S`.

To construct a `UnknownVarianceDataSet`, one has to specify a vector of independent variables `x`, a vector of dependent variables `y`, reciprocal error models `σ⁻¹(x, y_pred, c)` and an initial guess for the vector of error parameters `c`.
Optionally, an explicit `errorparamsplitter` function of the form `θ -> (modelparams, xerrorparams, yerrorparams)` may be specified, which splits the parameters into a tuple of model parameters, which are subsequently forwarded into the model, and error parameters, which are only passed to the reciprocal error models `σ⁻¹`.
!!! warn
    The parameters which are visible to the outside are processed by `errorparamsplitter` FIRST, before forwarding into the model, where `modelparams` might be further modified by embedding transformations.

# Examples:

In the simplest case, where all data points are mutually independent and have a single ``x``-component and a single ``y``-component each, a `DataSet` consisting of four points can be constructed via
```julia
DS = UnknownVarianceDataSet([1,2,3,4], [4,5,6.5,7.8], (x,y,cx)->1/exp10(cx[1]), (x,y,cy)->1/exp10(cy[1]), [0.25], [0.8])
```
!!! note
    It is generally advisable to exponentiate error parameters, since they are penalized poportional to `log(c)` in the normalization term of Gaussian likelihoods.
    A Bessel correction `sqrt((length(xdata(DS))+length(ydata(DS))-length(params))/(length(xdata(DS))+length(ydata(DS))))` can be applied to the reciprocal error to account for the fact that the maximum likelihood estimator for the variance is biased via kwarg `BesselCorrection`.
"""
struct UnknownVarianceDataSet{BesselCorrection} <: AbstractUnknownUncertaintyDataSet
    x::AbstractVector{<:Number}
    y::AbstractVector{<:Number}
    dims::Tuple{Int,Int,Int}
    invxerrormodelraw::Function # σ_x⁻¹ as Number, Vector or Matrix
    invyerrormodelraw::Function # σ_y⁻¹ as Number, Vector or Matrix
    testoutx::Union{Number,<:AbstractVector,<:AbstractMatrix}
    testouty::Union{Number,<:AbstractVector,<:AbstractMatrix}
    invxerrormodel::Function # σ_x⁻¹ wrapped as AbstractMatrix, e.g. Diagonal
    invyerrormodel::Function # σ_y⁻¹ wrapped as AbstractMatrix, e.g. Diagonal
    testpx::AbstractVector{<:Number}
    testpy::AbstractVector{<:Number}
    errorparamsplitter::Function # θ -> (view(θ, MODEL), view(θ, xERRORMODEL), view(θ, yERRORMODEL))
    SkipXs::Function
    xnames::AbstractVector{Symbol}
    ynames::AbstractVector{Symbol}
    name::Symbol

    UnknownVarianceDataSet(DS::AbstractDataSet; kwargs...) = UnknownVarianceDataSet(xdata(DS), ydata(DS), dims(DS); xnames=Xnames(DS), ynames=Ynames(DS), kwargs...)
    function UnknownVarianceDataSet(X::AbstractArray, Y::AbstractArray, dims::Tuple{Int,Int,Int}=(size(X,1), ConsistentElDims(X), ConsistentElDims(Y)); 
                        testpx::AbstractVector=0.1ones(xdim(dims)), testpy::AbstractVector=0.1ones(ydim(dims)), kwargs...)
        @info "Assuming error models σ(x,y,c) = exp10.(c)"
        xerrmod = xdim(dims) == 1 ? ((x,y,c::AbstractVector)->exp10(-c[1])) : ((x,y,c::AbstractVector)->exp10.(-c))
        yerrmod = ydim(dims) == 1 ? ((x,y,c::AbstractVector)->exp10(-c[1])) : ((x,y,c::AbstractVector)->exp10.(-c))
        UnknownVarianceDataSet(Unwind(X), Unwind(Y), xerrmod, yerrmod, testpx, testpy, dims; kwargs...)
    end
    function UnknownVarianceDataSet(DS::AbstractDataSet, invxerrormodel::Function, invyerrormodel::Function, testpx::AbstractVector=0.1ones(xdim(DS)), testpy::AbstractVector=0.1ones(ydim(DS)); kwargs...)
        UnknownVarianceDataSet(xdata(DS), ydata(DS), invxerrormodel, invyerrormodel, testpx, testpy, dims(DS); xnames=Xnames(DS), ynames=Ynames(DS), kwargs...)
    end
    function UnknownVarianceDataSet(x::AbstractVector, y::AbstractVector, invxerrormodel::Function, invyerrormodel::Function,
        testpx::AbstractVector, testpy::AbstractVector, dims::Tuple{Int,Int,Int}=(size(X,1), ConsistentElDims(X), ConsistentElDims(Y)); kwargs...)
        @info "Assuming error parameters always given by last ($(length(testpx)),$(length(testpy))) parameters respectively."
        # Error param splitter
        UnknownVarianceDataSet(Unwind(x), Unwind(y), dims, invxerrormodel, invyerrormodel, testpx, testpy, DefaultErrorModelSplitter(length(testpx),length(testpy)); kwargs...)
    end
    function UnknownVarianceDataSet(x::AbstractVector, y::AbstractVector, dims::Tuple{Int,Int,Int}, 
            invxerrormodel::Function, invyerrormodel::Function, testpx::AbstractVector, testpy::AbstractVector, errorparamsplitter::Function;
            xnames::AbstractVector{<:StringOrSymb}=CreateSymbolNames(xdim(dims),"x"), ynames::AbstractVector{<:StringOrSymb}=CreateSymbolNames(ydim(dims),"y"),
            name::StringOrSymb=Symbol(), kwargs...)
            UnknownVarianceDataSet(x, y, dims, invxerrormodel, invyerrormodel, testpx, testpy, errorparamsplitter, xnames, ynames, name; kwargs...)
    end
    function UnknownVarianceDataSet(x::AbstractVector, y::AbstractVector, dims::Tuple{Int,Int,Int}, 
            invxerrormodelraw::Function, invyerrormodelraw::Function, testpx::AbstractVector, testpy::AbstractVector, errorparamsplitter::Function,
            xnames::AbstractVector{<:StringOrSymb}, ynames::AbstractVector{<:StringOrSymb}, name::StringOrSymb=Symbol(); BesselCorrection::Bool=false, 
            SkipXs::Function=(n = length(x); p::AbstractVector{<:Number}->@view p[n+1:end]))
        @assert all(x->(x > 0), dims) "Not all dims > 0: $dims."
        @assert Npoints(dims) == Int(length(x)/xdim(dims)) == Int(length(y)/ydim(dims)) "Inconsistent input dimensions."
        @assert length(xnames) == xdim(dims) && length(ynames) == ydim(dims)
        
        ## Check that inverrormodel either outputs Matrix for ydim > 1
        Q = invxerrormodelraw(Windup(x, xdim(dims))[1], Windup(y, ydim(dims))[1], testpx)
        Invxerrormodelraw, Invxerrormodel = ErrorModelTester(invxerrormodelraw, Q)
        
        M = invyerrormodelraw(Windup(x, xdim(dims))[1], Windup(y, ydim(dims))[1], testpy)
        Invyerrormodelraw, Invyerrormodel = ErrorModelTester(invyerrormodelraw, M)

        xdim(dims) == 1 && (@assert Q isa Number && Q > 0)
        xdim(dims) > 1 && @assert (Q isa AbstractVector && length(Q) == xdim(dims) && all(Q .> 0)) || (Q isa AbstractMatrix && size(Q,1) == size(Q,2) == xdim(dims) && det(Q) > 0)
        ydim(dims) == 1 && (@assert M isa Number && M > 0)
        ydim(dims) > 1 && @assert (M isa AbstractVector && length(M) == ydim(dims) && all(M .> 0)) || (M isa AbstractMatrix && size(M,1) == size(M,2) == ydim(dims) && det(M) > 0)
        
        new{BesselCorrection}(x, y, dims, Invxerrormodelraw, Invyerrormodelraw, Q, M, Invxerrormodel, Invyerrormodel, testpx, testpy, errorparamsplitter, SkipXs, Symbol.(xnames), Symbol.(ynames), Symbol(name))
    end
end


function (::Type{T})(DS::UnknownVarianceDataSet{B}; kwargs...) where T<:Number where B
	UnknownVarianceDataSet(T.(xdata(DS)), T.(ydata(DS)), dims(DS), xinverrormodelraw(DS), yinverrormodelraw(DS), 
                T.(DS.testpx), T.(DS.testpy), SplitErrorParams(DS); xnames=Xnames(DS), ynames=Ynames(DS), name=name(DS), BesselCorrection=B, kwargs...)
end

# For SciMLBase.remake
UnknownVarianceDataSet(;
x::AbstractVector=[0.],
y::AbstractVector=[0.],
dims::Tuple{Int,Int,Int}=(1,1,1),
invxerrormodel::Function=identity,
invyerrormodel::Function=identity,
testpx::AbstractVector{<:Number}=[0.],
testpy::AbstractVector{<:Number}=[0.],
errorparamsplitter::Function=x->(x[1], x[2]),
xnames::AbstractVector{<:StringOrSymb}=[:x],
ynames::AbstractVector{<:StringOrSymb}=[:y],
name::StringOrSymb=Symbol()) = UnknownVarianceDataSet(x, y, dims, invxerrormodel, invyerrormodel, testpx, testpy, errorparamsplitter, xnames, ynames, name)


DefaultErrorModelSplitter(n::Int, m::Int) = ((θ::AbstractVector{<:Number}; kwargs...) -> @views (θ[1:end-n-m], θ[end-n-m+1:end-m], θ[end-m+1:end]))


xdata(DS::UnknownVarianceDataSet) = DS.x
ydata(DS::UnknownVarianceDataSet) = DS.y
dims(DS::UnknownVarianceDataSet) = DS.dims
xnames(DS::UnknownVarianceDataSet) = Xnames(DS) .|> string
ynames(DS::UnknownVarianceDataSet) = Ynames(DS) .|> string
Xnames(DS::UnknownVarianceDataSet) = DS.xnames
Ynames(DS::UnknownVarianceDataSet) = DS.ynames
name(DS::UnknownVarianceDataSet) = DS.name


xerrormoddim(DS::UnknownVarianceDataSet) = length(DS.testpx)
yerrormoddim(DS::UnknownVarianceDataSet) = length(DS.testpy)
errormoddim(DS::UnknownVarianceDataSet) = xerrormoddim(DS) + yerrormoddim(DS)

SplitErrorParams(DS::UnknownVarianceDataSet) = DS.errorparamsplitter

xinverrormodel(DS::UnknownVarianceDataSet) = DS.invxerrormodel
yinverrormodel(DS::UnknownVarianceDataSet) = DS.invyerrormodel

xinverrormodelraw(DS::UnknownVarianceDataSet) = DS.invxerrormodelraw
yinverrormodelraw(DS::UnknownVarianceDataSet) = DS.invyerrormodelraw

xerrorparams(DS::UnknownVarianceDataSet, mle::AbstractVector) = (SplitErrorParams(DS)(mle))[2]
yerrorparams(DS::UnknownVarianceDataSet, mle::AbstractVector) = (SplitErrorParams(DS)(mle))[3]

HasBessel(DS::UnknownVarianceDataSet{T}) where T = T

# Uncertainty must be constructed around prediction!
function xsigma(DS::UnknownVarianceDataSet, c::AbstractVector{<:Number}=DS.testpx; verbose::Bool=true)
    C = if length(c) != length(DS.testpx) 
        verbose && @warn "yInvCov: Given parameters not of expected length - expected $(length(DS.testpx)) got $(length(c)). Only pass error params."
        (SplitErrorParams(DS)(c))[2]
    else
        verbose && c === DS.testpx && @warn "Cheating by not constructing uncertainty around given prediction."
        c
    end;    errmod = xinverrormodel(DS)
    map((x,y)->inv(errmod(x,y,C)), WoundX(DS), WoundY(DS)) |> _TryVectorizeNoSqrt
end

function xInvCov(DS::UnknownVarianceDataSet, c::AbstractVector{<:Number}=DS.testpx; verbose::Bool=true)
    C = if length(c) != length(DS.testpx) 
        verbose && @warn "yInvCov: Given parameters not of expected length - expected $(length(DS.testpx)) got $(length(c)). Only pass error params."
        (SplitErrorParams(DS)(c))[2]
    else
        verbose && c === DS.testpx && @warn "Cheating by not constructing uncertainty around given prediction."
        c
    end;    errmod = xinverrormodel(DS)
    map(((x,y)->(S=errmod(x,y,C); S' * S)), WoundX(DS), WoundY(DS)) |> BlockReduce
end

# Uncertainty must be constructed around prediction!
function ysigma(DS::UnknownVarianceDataSet, c::AbstractVector{<:Number}=DS.testpy; verbose::Bool=true)
    C = if length(c) != length(DS.testpy) 
        verbose && @warn "yInvCov: Given parameters not of expected length - expected $(length(DS.testpy)) got $(length(c)). Only pass error params."
        (SplitErrorParams(DS)(c))[end]
    else
        verbose && c === DS.testpy && @warn "Cheating by not constructing uncertainty around given prediction."
        c
    end;    errmod = yinverrormodel(DS)
    map((x,y)->inv(errmod(x,y,C)), WoundX(DS), WoundY(DS)) |> _TryVectorizeNoSqrt
end

function yInvCov(DS::UnknownVarianceDataSet, c::AbstractVector{<:Number}=DS.testpy; verbose::Bool=true)
    C = if length(c) != length(DS.testpy) 
        verbose && @warn "yInvCov: Given parameters not of expected length - expected $(length(DS.testpy)) got $(length(c)). Only pass error params."
        (SplitErrorParams(DS)(c))[end]
    else
        verbose && c === DS.testpy && @warn "Cheating by not constructing uncertainty around given prediction."
        c
    end;    errmod = yinverrormodel(DS)
    map(((x,y)->(S=errmod(x,y,C); S' * S)), WoundX(DS), WoundY(DS)) |> BlockReduce
end


SkipXs(DS::UnknownVarianceDataSet) = DS.SkipXs

function GetOnlyModelParams(DS::UnknownVarianceDataSet)
    Splitter = SplitErrorParams(DS);    Skipper = SkipXs(DS)
    mle::AbstractVector->Skipper(Splitter(mle)[1])
end

xpars(DS::UnknownVarianceDataSet) = length(xdata(DS))


function _loglikelihood(DS::UnknownVarianceDataSet{BesselCorrection}, model::ModelOrFunction, θ::AbstractVector{T}; kwargs...) where T<:Number where BesselCorrection
    Splitter = SplitErrorParams(DS);    xinverrmod = xinverrormodel(DS);    yinverrmod = yinverrormodel(DS)
    normalparams, xerrorparams, yerrorparams = Splitter(θ)  # normalparams also contains estimated x-values
    LiftedEmb = LiftedEmbedding(DS, model, length(normalparams)-length(xdata(DS))) # Picks out last length(normalparams)-length(xdata(DS)) as model parameters
    # LiftedEmb = LiftedEmbedding(DS, model, length(normalparams))
    XY = LiftedEmb(normalparams);   Xinds = 1:length(xdata(DS));    NonXinds = length(xdata(DS))+1:length(XY)
    woundXpred = Windup(view(XY, Xinds), xdim(DS));    woundYpred = Windup(view(XY, NonXinds), ydim(DS))
    Bessel = BesselCorrection ? sqrt((length(xdata(DS))+length(ydata(DS))-DOF(DS, θ))/(length(xdata(DS))+length(ydata(DS)))) : one(T)
    woundInvXσ = map((x,y)->Bessel .* xinverrmod(x,y,xerrorparams), woundXpred, woundYpred)
    woundInvYσ = map((x,y)->Bessel .* yinverrmod(x,y,yerrorparams), woundXpred, woundYpred)
    woundX = WoundX(DS);    woundY = WoundY(DS)
    function _Eval(DS, woundYpred, woundInvYσ, woundY, woundXpred, woundInvXσ, woundX)
        Res::T = -(length(ydata(DS)) + length(xdata(DS)))*log(2π)
        @inbounds for i in eachindex(woundY)
            Res += 2logdet(woundInvYσ[i])
            Res -= sum(abs2, woundInvYσ[i] * (woundY[i] - woundYpred[i]))
        end
        @inbounds for j in eachindex(woundX)
            Res += 2logdet(woundInvXσ[j])
            Res -= sum(abs2, woundInvXσ[j] * (woundX[j] - woundXpred[j]))
        end
        Res *= 0.5;    Res
    end;    _Eval(DS, woundYpred, woundInvYσ, woundY, woundXpred, woundInvXσ, woundX)
end


# Can get parameter indices by SplitErrorParams(DS)(1:length(θ))

# Potential for optimization by specializing on Type of invcov
function _FisherMetric(DS::UnknownVarianceDataSet{BesselCorrection}, model::ModelOrFunction, dmodel::ModelOrFunction, θ::AbstractVector{T}; 
                        ADmode::Val=Val(:ForwardDiff), kwargs...) where T<:Number where BesselCorrection
    Splitter = SplitErrorParams(DS);    xinverrmod = xinverrormodel(DS);    yinverrmod = yinverrormodel(DS)
    normalparams, xerrorparams, yerrorparams = Splitter(θ)  # normalparams also contains estimated x-values
    LiftedEmb = LiftedEmbedding(DS, model, length(normalparams)-length(xdata(DS)))
    # LiftedEmb = LiftedEmbedding(DS, model, length(normalparams))
    XY = LiftedEmb(normalparams);   Xinds = 1:length(xdata(DS));    NonXinds = length(xdata(DS))+1:length(XY)
    woundXpred = Windup(view(XY, Xinds), xdim(DS));    woundYpred = Windup(view(XY, NonXinds), ydim(DS))
    Bessel = BesselCorrection ? sqrt((length(xdata(DS))+length(ydata(DS))-DOF(DS, θ))/(length(xdata(DS))+length(ydata(DS)))) : one(T)
    woundInvXσ = map((x,y)->Bessel .* xinverrmod(x,y,xerrorparams), woundXpred, woundYpred)
    woundInvYσ = map((x,y)->Bessel .* yinverrmod(x,y,yerrorparams), woundXpred, woundYpred)

    J = BlockMatrix(BlockReduce(woundInvXσ), BlockReduce(woundInvYσ)) * GetJac(ADmode, LiftedEmb, length(θ))(θ)
    F_m = transpose(J) * J

    yΣposhalf = map(inv, woundInvYσ) |> BlockReduce
    function InvSqrtyCovFromFull(θ)
        normalparams, xerrorparams, yerrorparams = Splitter(θ)
        XY = LiftedEmb(normalparams)
        woundXpred = Windup(view(XY, Xinds), xdim(DS))
        woundYpred = Windup(view(XY, NonXinds), ydim(DS))
        BlockReduce(map((x,y)->Bessel .* yinverrmod(x,y,yerrorparams), woundXpred, woundYpred))
    end
    yΣneghalfJac = GetMatrixJac(ADmode, InvSqrtyCovFromFull, length(θ), size(yΣposhalf))(θ)
    # @tullio F_ey[i,j] := 2 * yΣposhalf[a,b] * yΣneghalfJac[b,c,i] * yΣposhalf[c,d] * yΣneghalfJac[d,a,j]
    @tullio F_m[i,j] += 2 * yΣposhalf[a,b] * yΣneghalfJac[b,c,i] * yΣposhalf[c,d] * yΣneghalfJac[d,a,j]
    
    xΣposhalf = map(inv, woundInvXσ) |> BlockReduce
    function InvSqrtxCovFromFull(θ)
        normalparams, xerrorparams, yerrorparams = Splitter(θ)
        XY = LiftedEmb(normalparams)
        woundXpred = Windup(view(XY, Xinds), xdim(DS))
        woundYpred = Windup(view(XY, NonXinds), ydim(DS))
        BlockReduce(map((x,y)->Bessel .* xinverrmod(x,y,xerrorparams), woundXpred, woundYpred))
    end
    xΣneghalfJac = GetMatrixJac(ADmode, InvSqrtxCovFromFull, length(θ), size(xΣposhalf))(θ)
    # @tullio F_ex[i,j] := 2 * xΣposhalf[a,b] * xΣneghalfJac[b,c,i] * xΣposhalf[c,d] * xΣneghalfJac[d,a,j]
    @tullio F_m[i,j] += 2 * xΣposhalf[a,b] * xΣneghalfJac[b,c,i] * xΣposhalf[c,d] * xΣneghalfJac[d,a,j]
    # F_m .+ F_ey .+ F_ex
    F_m
end

# function _Score(DSE::DataSetExact, model::ModelOrFunction, dmodel::ModelOrFunction, θ::AbstractVector{<:Number}, ADmode::Val{false}; kwargs...)
#     transpose(EmbeddingMatrix(DSE,dmodel,θ; kwargs...)) * gradlogpdf(ydist(DSE), EmbeddingMap(DSE,model,θ; kwargs...))
# end
