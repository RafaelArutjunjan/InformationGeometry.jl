

# Missing observations use compact observed-space representations while the
# latent x block remains complete for model evaluation.

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
!!! warning
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
struct UnknownVarianceDataSet{BesselCorrection, X<:AbstractVector, Y<:AbstractVector, XER<:Function, YER<:Function, SP<:Function, SX<:Function, XK<:Union{Nothing,<:AbstractVector}, YK<:Union{Nothing,<:AbstractVector}} <: AbstractUnknownUncertaintyDataSet
    x::X
    y::Y
    dims::Tuple{Int,Int,Int}
    invxerrormodelraw::XER # σ_x⁻¹ as Number, Vector or Matrix
    invyerrormodelraw::YER # σ_y⁻¹ as Number, Vector or Matrix
    testoutx::Union{Number,<:AbstractVector,<:AbstractMatrix}
    testouty::Union{Number,<:AbstractVector,<:AbstractMatrix}
    invxerrormodel::Function # σ_x⁻¹ wrapped as AbstractMatrix, e.g. Diagonal
    invyerrormodel::Function # σ_y⁻¹ wrapped as AbstractMatrix, e.g. Diagonal
    testpx::AbstractVector{<:Number}
    testpy::AbstractVector{<:Number}
    errorparamsplitter::SP # θ -> (view(θ, MODEL), view(θ, xERRORMODEL), view(θ, yERRORMODEL))
    SkipXs::SX
    xnames::AbstractVector{Symbol}
    ynames::AbstractVector{Symbol}
    name::Symbol
    xdatakeep::XK
    datakeep::YK
    predkeep::Union{Nothing,AbstractVector{<:Int}}
    woundXpred::Union{Nothing,AbstractVector}

    UnknownVarianceDataSet(DM::AbstractDataModel, args...; kwargs...) = UnknownVarianceDataSet(Data(DM), args...; kwargs...)
    UnknownVarianceDataSet(DS::AbstractDataSet, args...; kwargs...) = UnknownVarianceDataSet(WoundX(DS), WoundY(DS), args...; xnames=Xnames(DS), ynames=Ynames(DS), name=name(DS), kwargs...)
    function UnknownVarianceDataSet(X::AbstractArray, Y::AbstractArray, dims::Tuple{Int,Int,Int}=(size(X,1), ConsistentElDims(X), ConsistentElDims(Y)); 
                        testpx::AbstractVector=Fill(0.1,xdim(dims)), testpy::AbstractVector=Fill(0.1,ydim(dims)), verbose::Bool=true, kwargs...)
        verbose && @info "Assuming error models σ(x,y,c) = exp10.(c)"
        xerrmod = xdim(dims) == 1 ? ((x,y,c::AbstractVector)->exp10(-c[1])) : ((x,y,c::AbstractVector)->exp10.(-c))
        yerrmod = ydim(dims) == 1 ? ((x,y,c::AbstractVector)->exp10(-c[1])) : ((x,y,c::AbstractVector)->exp10.(-c))
        UnknownVarianceDataSet(collect(Iterators.flatten(eachrow(X))), Unwind(Y), xerrmod, yerrmod, testpx, testpy, dims; verbose, kwargs...)
    end
    function UnknownVarianceDataSet(x::AbstractVector, y::AbstractVector, invxerrormodel::Function, invyerrormodel::Function,
        testpx::AbstractVector, testpy::AbstractVector, dims::Tuple{Int,Int,Int}=(size(x,1), (all(z -> z isa Number || ismissing(z), x) ? 1 : ConsistentElDims(x)), (all(z -> z isa Number || ismissing(z), y) ? 1 : ConsistentElDims(y))); verbose::Bool=true, kwargs...)
        verbose && @info "Assuming error parameters always given by last ($(length(testpx)),$(length(testpy))) parameters respectively."
        # Error param splitter
        xflat = all(z -> z isa Number || ismissing(z), x) ? collect(x) : reduce(vcat, x)
        yflat = all(z -> z isa Number || ismissing(z), y) ? collect(y) : Unwind(y)
        UnknownVarianceDataSet(xflat, yflat, dims, invxerrormodel, invyerrormodel, testpx, testpy, DefaultErrorModelSplitter(length(testpx),length(testpy)); verbose, kwargs...)
    end
    function UnknownVarianceDataSet(x::AbstractVector, y::AbstractVector, dims::Tuple{Int,Int,Int}, 
            invxerrormodel::Function, invyerrormodel::Function, testpx::AbstractVector, testpy::AbstractVector, errorparamsplitter::Function;
            xnames::AbstractVector{<:StringOrSymb}=CreateSymbolNames(xdim(dims),"x"), ynames::AbstractVector{<:StringOrSymb}=CreateSymbolNames(ydim(dims),"y"),
            name::StringOrSymb=Symbol(), kwargs...)
            UnknownVarianceDataSet(x, y, dims, invxerrormodel, invyerrormodel, testpx, testpy, errorparamsplitter, xnames, ynames, name; kwargs...)
    end
    function UnknownVarianceDataSet(x::AbstractVector, y::AbstractVector, dims::Tuple{Int,Int,Int}, 
            invxerrormodelraw::Function, invyerrormodelraw::Function, testpx::AbstractVector, testpy::AbstractVector, errorparamsplitter::Function,
            xnames::AbstractVector{<:StringOrSymb}, ynames::AbstractVector{<:StringOrSymb}, name::StringOrSymb=Symbol(); SkipXs::Function=(n = length(x); p::AbstractVector{<:Number}->SafeView(p,n+1:length(p))), xdatakeep::Union{Nothing,AbstractVector{<:Bool}}=map(z->!ismissing(z) && isfinite(z), x), datakeep::Union{Nothing,AbstractVector{<:Bool}}=map(z->!ismissing(z) && isfinite(z), y), predkeep::Union{Nothing,AbstractVector{<:Int}}=nothing, woundXpred::Union{Nothing,AbstractVector}=nothing, kwargs...)
         @assert length(x) == Npoints(dims) * xdim(dims)
         @assert length(y) == Npoints(dims) * ydim(dims)
         @assert all(any(view(xdatakeep, (i-1)*xdim(dims)+1:i*xdim(dims))) for i in 1:Npoints(dims)) "Each row must contain at least one observed x component."
         xinit = _UVDFillMissingX(x, xdatakeep, dims)
         yfull = [_finite_or_zero(y[i]) for i in eachindex(y)]
         sparseX, sparsePredKeep = if all(xdatakeep)
             if all(datakeep)
                 (nothing, nothing)
             elseif !isnothing(predkeep) && !isnothing(woundXpred)
                 (woundXpred, predkeep)
             else
                 GetPredKeep(xinit, transpose(reshape(yfull, ydim(dims), :)))
             end
         else
             (nothing, nothing)
         end
        Q = invxerrormodelraw(Windup(xinit, xdim(dims))[1], Windup(yfull, ydim(dims))[1], testpx)
         Invxerrormodelraw, Invxerrormodel = ErrorModelTester(invxerrormodelraw, Q)
         M = invyerrormodelraw(Windup(xinit, xdim(dims))[1], Windup(yfull, ydim(dims))[1], testpy)
         Invyerrormodelraw, Invyerrormodel = ErrorModelTester(invyerrormodelraw, M)
         
         UnknownVarianceDataSet(xinit, y[datakeep], dims, Invxerrormodelraw, Invyerrormodelraw, Q, M, Invxerrormodel, Invyerrormodel, testpx, testpy, errorparamsplitter, SkipXs, xnames, ynames, name; xdatakeep=all(xdatakeep) ? nothing : xdatakeep, datakeep=all(datakeep) ? nothing : datakeep, predkeep=all(datakeep) && all(xdatakeep) ? nothing : sparsePredKeep, woundXpred=sparseX, kwargs...)
    end
    function UnknownVarianceDataSet(x::AbstractVector{<:Number}, y::AbstractVector{<:Number}, dims::Tuple{Int,Int,Int}, 
            invxerrormodelraw::Function, invyerrormodelraw::Function, testoutx::Union{Number,<:AbstractVector,<:AbstractMatrix}, testouty::Union{Number,<:AbstractVector,<:AbstractMatrix},
            invxerrormodel::Function, invyerrormodel::Function, testpx::AbstractVector, testpy::AbstractVector, errorparamsplitter::Function, SkipXs::Function,
            xnames::AbstractVector{<:StringOrSymb}, ynames::AbstractVector{<:StringOrSymb}, name::StringOrSymb=Symbol(); BesselCorrection::Bool=false, verbose::Bool=true, xdatakeep::Union{Nothing,AbstractVector{<:Bool}}=nothing, datakeep::Union{Nothing,AbstractVector{<:Bool}}=nothing, predkeep::Union{Nothing,AbstractVector{<:Int}}=nothing, woundXpred::Union{Nothing,AbstractVector}=nothing)
        @assert all(x->(x > 0), dims) "Not all dims > 0: $dims."
        @assert Npoints(dims) == Int(length(x)/xdim(dims)) "Inconsistent x input dimensions."
        @assert isnothing(datakeep) ? Npoints(dims) == Int(length(y)/ydim(dims)) : length(y) == sum(datakeep) "Inconsistent y input dimensions."
        @assert length(xnames) == xdim(dims) && length(ynames) == ydim(dims)
        ## Check that inverrormodel either outputs Matrix for ydim > 1
        xdim(dims) == 1 && (@assert testoutx isa Number && testoutx > 0)
        xdim(dims) > 1 && @assert (testoutx isa AbstractVector && length(testoutx) == xdim(dims) && all(testoutx .> 0)) || (testoutx isa AbstractMatrix && size(testoutx,1) == size(testoutx,2) == xdim(dims) && det(testoutx) > 0)
        ydim(dims) == 1 && (@assert testouty isa Number && testouty > 0)
        ydim(dims) > 1 && @assert (testouty isa AbstractVector && length(testouty) == ydim(dims) && all(testouty .> 0)) || (testouty isa AbstractMatrix && size(testouty,1) == size(testouty,2) == ydim(dims) && det(testouty) > 0)
        
        @assert isnothing(datakeep) || length(datakeep) == Npoints(dims) * ydim(dims)
        @assert isnothing(datakeep) || length(y) == sum(datakeep)
        @assert isnothing(xdatakeep) || length(xdatakeep) == Npoints(dims) * xdim(dims)
        new{BesselCorrection, typeof(x), typeof(y), typeof(invxerrormodelraw), typeof(invyerrormodelraw), typeof(errorparamsplitter), typeof(SkipXs), typeof(xdatakeep), typeof(datakeep)}(x, y, dims,
                    invxerrormodelraw, invyerrormodelraw, testoutx, testouty, invxerrormodel, invyerrormodel, testpx, testpy, errorparamsplitter, SkipXs, Symbol.(xnames), Symbol.(ynames), Symbol(name), xdatakeep, datakeep, predkeep, woundXpred)
    end
end


function (::Type{T})(DS::UnknownVarianceDataSet{B}; kwargs...) where T<:Number where B
    remake(DS; x=T.(xdata(DS)), y=T.(ydata(DS)), testpx=T.(DS.testpx), testoutx=T.(DS.testoutx), testpy=T.(DS.testpy), testouty=T.(DS.testouty), BesselCorrection=B, kwargs...)
end

# For SciMLBase.remake
UnknownVarianceDataSet(;
x::AbstractVector=[0.],
y::AbstractVector=[0.],
dims::Tuple{Int,Int,Int}=(1,1,1),
invxerrormodelraw::Function=identity,
invyerrormodelraw::Function=identity,
testoutx::Union{Number,<:AbstractVector,<:AbstractMatrix}=5,
testouty::Union{Number,<:AbstractVector,<:AbstractMatrix}=5,
invxerrormodel::Function=identity,
invyerrormodel::Function=identity,
testpx::AbstractVector{<:Number}=[0.],
testpy::AbstractVector{<:Number}=[0.],
errorparamsplitter::Function=x->(x[1], x[2]),
    SkipXs::Function=identity,
    xdatakeep::Union{Nothing,AbstractVector{<:Bool}}=nothing,
    datakeep::Union{Nothing,AbstractVector{<:Bool}}=nothing,
xnames::AbstractVector{<:StringOrSymb}=[:x],
ynames::AbstractVector{<:StringOrSymb}=[:y],
name::StringOrSymb=Symbol(),
BesselCorrection::Bool=false,
verbose::Bool=true
) = UnknownVarianceDataSet(x, y, dims, invxerrormodelraw, invyerrormodelraw, testoutx, testouty, invxerrormodel, invyerrormodel, testpx, testpy, errorparamsplitter, SkipXs, xnames, ynames, name; BesselCorrection, verbose, xdatakeep, datakeep)


function DefaultErrorModelSplitter(n::Int, m::Int)
    Splitter(θ::AbstractVector{<:Number}; kwargs...) = @views (θ[1:end-n-m], θ[end-n-m+1:end-m], θ[end-m+1:end])
    Splitter(θ::ComponentVector{<:Number}; kwargs...) = (L=length(θ);    (θ[KeepIndex(1:L-n-m)], θ[KeepIndex(L-n-m+1:L-m)], θ[KeepIndex(L-m+1:L)]))
end
Identity3Splitter = ((θ::AbstractVector{<:Number}; kwargs...) -> (θ, θ, θ))


xdata(DS::UnknownVarianceDataSet) = DS.x
ydata(DS::UnknownVarianceDataSet) = DS.y
dims(DS::UnknownVarianceDataSet) = DS.dims
xnames(DS::UnknownVarianceDataSet) = Xnames(DS) .|> string
ynames(DS::UnknownVarianceDataSet) = Ynames(DS) .|> string
Xnames(DS::UnknownVarianceDataSet) = DS.xnames
Ynames(DS::UnknownVarianceDataSet) = DS.ynames
name(DS::UnknownVarianceDataSet) = DS.name

HasXerror(DS::UnknownVarianceDataSet; kwargs...) = true
HasMissingValues(DS::UnknownVarianceDataSet) = !isnothing(DS.xdatakeep) || !isnothing(DS.datakeep)
DataspaceDim(DS::UnknownVarianceDataSet) = _UVDDataspaceDim(DS.datakeep, ydata(DS))
_UVDDataspaceDim(::Nothing, y) = length(y)
_UVDDataspaceDim(keep::AbstractVector, y) = sum(keep)

WoundY(DS::UnknownVarianceDataSet) = _WoundYUVD(DS.datakeep, ydata(DS), ydim(DS), eltype(DS.testpy))
_WoundYForPrediction(DS::UnknownVarianceDataSet) = _WoundYForPrediction(DS.datakeep, ydata(DS), ydim(DS))
_WoundYForPrediction(::Nothing, y, ydim) = Windup(y, ydim)
function _WoundYForPrediction(keep::AbstractVector, y, ydim)
    Y = fill(zero(eltype(y)), length(keep)); Y[keep] .= y; Windup(Y, ydim)
end
_WoundYUVD(::Nothing, y, ydim, T) = Windup(y, ydim)
function _WoundYUVD(keep::AbstractVector, y, ydim, T)
    Y = fill(convert(T, NaN), length(keep))
    Y[keep] .= y
    Windup(Y, ydim)
end
WoundYmasked(DS::UnknownVarianceDataSet) = WoundY(DS)
function _WoundXReconstructed(DS::UnknownVarianceDataSet)
    X = copy(xdata(DS))
    !isnothing(DS.xdatakeep) && (X[.!DS.xdatakeep] .= convert(eltype(X), NaN))
    Windup(X, xdim(DS))
end
ReconstructDataMatrices(DS::UnknownVarianceDataSet) = (_WoundXReconstructed(DS), WoundY(DS))

_WoundYFinite(DS::UnknownVarianceDataSet) = _WoundYForPrediction(DS)

_finite_or_zero(z) = ismissing(z) || !isfinite(z) ? 0.0 : z
function _UVDFillMissingX(x::AbstractVector, keep::AbstractVector{Bool}, dims)
    xinit = Float64.(_finite_or_zero.(x))
    xd = xdim(dims)
    for j in 1:xd
        inds = j:xd:length(x)
        observed = [x[i] for i in inds if keep[i]]
        value = isempty(observed) ? 0.0 : sum(observed) / length(observed)
        for i in inds
            keep[i] || (xinit[i] = value)
        end
    end
    xinit
end

_MaskedUVD(M::AbstractMatrix, ::Nothing) = M
_MaskedUVD(M::Diagonal, ::Nothing) = M
_MaskedUVD(M::Diagonal, keep::AbstractVector) = Diagonal(view(M.diag, keep))
_MaskedUVD(M::AbstractMatrix, keep::AbstractVector) = view(M, keep, keep)
_MaskedUVD(v::AbstractVector, ::Nothing) = v
_MaskedUVD(v::AbstractVector, keep::AbstractVector) = view(v, keep)


xerrormoddim(DS::UnknownVarianceDataSet) = length(DS.testpx)
yerrormoddim(DS::UnknownVarianceDataSet) = length(DS.testpy)

function errormoddim(DS::UnknownVarianceDataSet; max::Int=500)
    try
        # Allow overlapping parameters for error models
        ((_,x,y) = SplitErrorParams(DS)(1:max);     length(x ∪ y))
    catch E;
        @warn "Got error $E, returning upper bound for errormoddim."
        xerrormoddim(DS) + yerrormoddim(DS)
    end
end
errormoddim(DM::AbstractDataModel; max::Int=pdim(DM)) = errormoddim(Data(DS); max)

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
        verbose && c === DS.testpx && @warn "xsigma: Cheating by not constructing uncertainty around given prediction."
        c
    end;    errmod = xinverrormodel(DS)
    _MaskedUVD(map((x,y)->inv(errmod(x,y,C)), WoundX(DS), _WoundYFinite(DS)) |> _TryVectorizeNoSqrt, DS.xdatakeep)
end

function xInvCov(DS::UnknownVarianceDataSet, c::AbstractVector{<:Number}=DS.testpx; verbose::Bool=true)
    C = if length(c) != length(DS.testpx)
        verbose && @warn "yInvCov: Given parameters not of expected length - expected $(length(DS.testpx)) got $(length(c)). Only pass error params."
        (SplitErrorParams(DS)(c))[2]
    else
        verbose && c === DS.testpx && @warn "xInvCov: Cheating by not constructing uncertainty around given prediction."
        c
    end;    _MaskedUVD(_InvCov(xinverrormodel(DS), C, WoundX(DS), _WoundYFinite(DS), DS.testoutx), DS.xdatakeep)
end

# Uncertainty must be constructed around prediction!
function ysigma(DS::UnknownVarianceDataSet, c::AbstractVector{<:Number}=DS.testpy; verbose::Bool=true)
    C = if length(c) != length(DS.testpy)
        verbose && @warn "yInvCov: Given parameters not of expected length - expected $(length(DS.testpy)) got $(length(c)). Only pass error params."
        (SplitErrorParams(DS)(c))[end]
    else
        verbose && c === DS.testpy && @warn "ysigma: Cheating by not constructing uncertainty around given prediction."
        c
    end;    errmod = yinverrormodel(DS)
    _MaskedUVD(map((x,y)->inv(errmod(x,y,C)), WoundX(DS), _WoundYFinite(DS)) |> _TryVectorizeNoSqrt, DS.datakeep)
end

function yInvCov(DS::UnknownVarianceDataSet, c::AbstractVector{<:Number}=DS.testpy; verbose::Bool=true)
    C = if length(c) != length(DS.testpy)
        verbose && @warn "yInvCov: Given parameters not of expected length - expected $(length(DS.testpy)) got $(length(c)). Only pass error params."
        (SplitErrorParams(DS)(c))[end]
    else
        verbose && c === DS.testpy && @warn "yInvCov: Cheating by not constructing uncertainty around given prediction."
        c
    end;    _MaskedUVD(_InvCov(yinverrormodel(DS), C, WoundX(DS), _WoundYFinite(DS), DS.testouty), DS.datakeep)
end


SkipXs(DS::UnknownVarianceDataSet) = DS.SkipXs

xpars(DS::UnknownVarianceDataSet) = length(xdata(DS))


function _loglikelihood(DS::UnknownVarianceDataSet{BesselCorrection}, model::ModelOrFunction, θ::AbstractVector{T}; kwargs...)::T where T<:Number where BesselCorrection
    Splitter = SplitErrorParams(DS);    xinverrmodraw = xinverrormodelraw(DS);    yinverrmodraw = yinverrormodelraw(DS)
    normalparams, xerrorparams, yerrorparams = Splitter(θ)  # normalparams also contains estimated x-values
    LiftedEmb = LiftedEmbedding(DS, model, length(normalparams)-length(xdata(DS))) # Picks out last length(normalparams)-length(xdata(DS)) as model parameters
    XY = LiftedEmb(normalparams);   Xinds = 1:length(xdata(DS));    NonXinds = length(xdata(DS))+1:length(XY)
    woundXpred = Windup(view(XY, Xinds), xdim(DS));    woundYpred = Windup(view(XY, NonXinds), ydim(DS))
    _EvaluateUVDFullLikelihood(DS, DS.datakeep, woundXpred, woundYpred, xinverrmodraw, yinverrmodraw, xerrorparams, yerrorparams, θ, T)
end

function _EvaluateUVDFullLikelihood(DS::UnknownVarianceDataSet{BesselCorrection}, ::Nothing, woundXpred, woundYpred, xinverrmodraw, yinverrmodraw, xerrorparams, yerrorparams, θ, ::Type{T}) where {BesselCorrection,T}
    if isnothing(DS.datakeep)
        Bessel = BesselCorrection ? sqrt((length(xdata(DS)) + length(ydata(DS)) - DOF(DS, θ)) / (length(xdata(DS)) + length(ydata(DS)))) : one(T)
        xinv = map((x,y)->Bessel .* xinverrmodraw(x,y,xerrorparams), woundXpred, woundYpred)
        yinv = map((x,y)->Bessel .* yinverrmodraw(x,y,yerrorparams), woundXpred, woundYpred)
        return _EvalWoundsLogLikelihood(woundYpred, yinv, WoundY(DS)) + _EvalWoundsLogLikelihood(woundXpred, xinv, WoundX(DS))
    end
    _EvaluateUVDFullLikelihood(DS, DS.xdatakeep, DS.datakeep, woundXpred, woundYpred, xinverrmodraw, yinverrmodraw, xerrorparams, yerrorparams, θ, T)
end

function _EvaluateUVDFullLikelihood(DS::UnknownVarianceDataSet{BesselCorrection}, keep::AbstractVector, woundXpred, woundYpred, xinverrmodraw, yinverrmodraw, xerrorparams, yerrorparams, θ, ::Type{T}) where {BesselCorrection,T}
    _EvaluateUVDFullLikelihood(DS, DS.xdatakeep, keep, woundXpred, woundYpred, xinverrmodraw, yinverrmodraw, xerrorparams, yerrorparams, θ, T)
end

function _EvaluateUVDFullLikelihood(DS::UnknownVarianceDataSet{BesselCorrection}, xkeep, ykeep, woundXpred, woundYpred, xinverrmodraw, yinverrmodraw, xerrorparams, yerrorparams, θ, ::Type{T}) where {BesselCorrection,T}
    nx = isnothing(xkeep) ? length(xdata(DS)) : sum(xkeep)
    ny = isnothing(ykeep) ? length(ydata(DS)) : sum(ykeep)
    nobs = nx + ny
    Bessel = BesselCorrection ? sqrt((nobs - DOF(DS, θ)) / nobs) : one(T)
    xinv = _MaskedUVD(map((x,y)->Bessel .* xinverrmodraw(x,y,xerrorparams), woundXpred, woundYpred) |> BlockMatrix, xkeep)
    yinv = _MaskedUVD(map((x,y)->Bessel .* yinverrmodraw(x,y,yerrorparams), woundXpred, woundYpred) |> BlockMatrix, ykeep)
    xpred = view(Unwind(woundXpred), isnothing(xkeep) ? trues(length(xdata(DS))) : xkeep)
    ypred = view(Unwind(woundYpred), isnothing(ykeep) ? trues(length(woundYpred) * ydim(DS)) : ykeep)
    xobs = isnothing(xkeep) ? xdata(DS) : view(xdata(DS), xkeep)
    yobs = isnothing(ykeep) ? Unwind(WoundY(DS)) : ydata(DS)
    _EvalObservedUVD(xpred, xinv, xobs, T) + _EvalObservedUVD(ypred, yinv, yobs, T)
end

function _EvalObservedUVD(pred, invσ::AbstractVector, data, ::Type{T}) where T
    Res::T = -length(data) * log(2T(π))
    @inbounds for i in eachindex(data)
        Res += 2log(invσ[i]) - abs2(invσ[i] * (data[i] - pred[i]))
    end;    Res / 2
end
function _EvalObservedUVD(pred, invσ::AbstractMatrix, data, ::Type{T}) where T
    (-length(data) * log(2T(π)) + 2logdet(invσ) - sum(abs2, invσ * (data - pred))) / 2
end


# Can get parameter indices by SplitErrorParams(DS)(1:length(θ))

# Potential for optimization by specializing on Type of invcov
function _FisherMetric(DS::UnknownVarianceDataSet{BesselCorrection}, model::ModelOrFunction, dmodel::ModelOrFunction, θ::AbstractVector{T}; 
                         ADmode::Val=Val(:ForwardDiff), kwargs...) where T<:Number where BesselCorrection
    if !isnothing(DS.xdatakeep)
        return GetHess(ADmode, z -> -_loglikelihood(DS, model, z; kwargs...))(θ)
    end
    Splitter = SplitErrorParams(DS);    xinverrmod = xinverrormodel(DS);    yinverrmod = yinverrormodel(DS)
    xinverrmodraw = xinverrormodelraw(DS);    yinverrmodraw = yinverrormodelraw(DS)
    normalparams, xerrorparams, yerrorparams = Splitter(θ)  # normalparams also contains estimated x-values
    LiftedEmb = LiftedEmbedding(DS, model, length(normalparams)-length(xdata(DS)))
    # LiftedEmb = LiftedEmbedding(DS, model, length(normalparams))
    XY = LiftedEmb(normalparams);   Xinds = 1:length(xdata(DS));    NonXinds = length(xdata(DS))+1:length(XY)
    woundXpred = Windup(view(XY, Xinds), xdim(DS));    woundYpred = Windup(view(XY, NonXinds), ydim(DS))
    Bessel = BesselCorrection ? sqrt((length(xdata(DS))+length(ydata(DS))-DOF(DS, θ))/(length(xdata(DS))+length(ydata(DS)))) : one(T)
    woundInvXσ = map((x,y)->Bessel .* xinverrmod(x,y,xerrorparams), woundXpred, woundYpred)
    woundInvYσ = map((x,y)->Bessel .* yinverrmod(x,y,yerrorparams), woundXpred, woundYpred)

    NormalParamJac = SplitterJacNormalParams(DS; SkipXs=identity)(θ)
    Jfull = BlockMatrix(BlockMatrix(woundInvXσ), BlockMatrix(woundInvYσ)) * (GetJac(ADmode, LiftedEmb, length(normalparams))(normalparams)) * NormalParamJac

    @inline _UVDFisherMeanJacobian(J, ::Nothing, xlen) = J
    @inline _UVDFisherMeanJacobian(J, keep::AbstractVector, xlen) = (yrows = xlen .+ (1:length(keep));     vcat(view(J, 1:xlen, :), view(J, yrows[keep], :)))
    totalxlen = length(xdata(DS))

    J = _UVDFisherMeanJacobian(Jfull, DS.datakeep, totalxlen)
    F_m = transpose(J) * J

    # Check if small invs faster than single inversion of full matrix
    yΣposhalf = map(inv, woundInvYσ) |> BlockMatrix
    yΣneghalfJac = if yΣposhalf isa Diagonal
        function InvSqrtyCovFromFullDiagonal(θ)
            normalparams, xerrorparams, yerrorparams = Splitter(θ);    XY = LiftedEmb(normalparams)
            woundXpred = Windup(view(XY, Xinds), xdim(DS));    woundYpred = Windup(view(XY, NonXinds), ydim(DS))
            reduce(vcat, map((x,y)->Bessel .* yinverrmodraw(x,y,yerrorparams), woundXpred, woundYpred))
        end
        GetJac(ADmode, InvSqrtyCovFromFullDiagonal, length(θ))(θ)
    else
        function InvSqrtyCovFromFull(θ)
            normalparams, xerrorparams, yerrorparams = Splitter(θ);    XY = LiftedEmb(normalparams)
            woundXpred = Windup(view(XY, Xinds), xdim(DS));    woundYpred = Windup(view(XY, NonXinds), ydim(DS))
            BlockMatrix(map((x,y)->Bessel .* yinverrmod(x,y,yerrorparams), woundXpred, woundYpred))
        end
        GetMatrixJac(ADmode, InvSqrtyCovFromFull, length(θ), size(yΣposhalf))(θ)
    end

    xΣposhalf = map(inv, woundInvXσ) |> BlockMatrix
    xΣneghalfJac = if xΣposhalf isa Diagonal
        function InvSqrtxCovFromFullDiagonal(θ)
            normalparams, xerrorparams, yerrorparams = Splitter(θ);    XY = LiftedEmb(normalparams)
            woundXpred = Windup(view(XY, Xinds), xdim(DS));    woundYpred = Windup(view(XY, NonXinds), ydim(DS))
            reduce(vcat, map((x,y)->Bessel .* xinverrmodraw(x,y,xerrorparams), woundXpred, woundYpred))
        end
        GetJac(ADmode, InvSqrtxCovFromFullDiagonal, length(θ))(θ)
    else
        function InvSqrtxCovFromFull(θ)
            normalparams, xerrorparams, yerrorparams = Splitter(θ);    XY = LiftedEmb(normalparams)
            woundXpred = Windup(view(XY, Xinds), xdim(DS));    woundYpred = Windup(view(XY, NonXinds), ydim(DS))
            BlockMatrix(map((x,y)->Bessel .* xinverrmod(x,y,xerrorparams), woundXpred, woundYpred))
        end
        GetMatrixJac(ADmode, InvSqrtxCovFromFull, length(θ), size(xΣposhalf))(θ)
    end

    @inline _UVDFisherYCovariance(dS::AbstractArray, ::Nothing) = dS
    @inline _UVDFisherYCovariance(dS::AbstractMatrix, keep::AbstractVector) = view(dS, keep, :)
    @inline _UVDFisherYCovariance(dS::AbstractArray, keep::AbstractVector) = view(dS, keep, keep, :)
    yΣposhalf = _MaskedUVD(yΣposhalf, DS.datakeep)
    yΣneghalfJac = _UVDFisherYCovariance(yΣneghalfJac, DS.datakeep)

    @inline AddCovarianceContribution!(F_m::AbstractMatrix, Σposhalf::AbstractMatrix, ΣneghalfJac::AbstractArray{<:Number,3}) = @tullio F_m[i,j] += 2 * Σposhalf[a,b] * ΣneghalfJac[b,c,i] * Σposhalf[c,d] * ΣneghalfJac[d,a,j]
    @inline function AddCovarianceContribution!(F_m::AbstractMatrix, Σposhalf::Diagonal, ΣneghalfJac::AbstractMatrix) # (N × θ) in diagonal case
        s = Σposhalf.diag;  @tullio F_m[i,j] += 2 * s[a]^2 * ΣneghalfJac[a,i] * ΣneghalfJac[a,j]    # F_m .+= 2 .* (ΣneghalfJac' * Diagonal(s.^2) * ΣneghalfJac)
    end
    
    AddCovarianceContribution!(F_m, yΣposhalf, yΣneghalfJac)
    AddCovarianceContribution!(F_m, xΣposhalf, xΣneghalfJac)
    F_m
end


# function _Score(DSE::DataSetExact, model::ModelOrFunction, dmodel::ModelOrFunction, θ::AbstractVector{<:Number}, ADmode::Val{false}; kwargs...)
#     transpose(EmbeddingMatrix(DSE,dmodel,θ; kwargs...)) * gradlogpdf(ydist(DSE), EmbeddingMap(DSE,model,θ; kwargs...))
# end

## Cannot work with missing values in UVD yet.
function UnknownVarianceDataSet(DS::DataSetUncertain{<:Any,<:Nothing}, invxerrormodel::Function=xdim(dims) == 1 ? ((x,y,c::AbstractVector)->exp10(-c[1])) : ((x,y,c::AbstractVector)->exp10.(-c)), testpx::AbstractVector=Fill(0.1,xdim(dims)); kwargs...)
    UnknownVarianceDataSet(xdata(DS), ydata(DS), dims(DS), invxerrormodel, yinverrormodelraw(DS), testpx, DS.testpy; xnames=Xnames(DS), ynames=Ynames(DS), name=name(DS), kwargs...)
end
function DataSetUncertain(DS::UnknownVarianceDataSet; kwargs...)
    DataSetUncertain(xdata(DS), ydata(DS), yinverrormodelraw(DS), DS.testpy, dims(DS); xnames=Xnames(DS), ynames=Ynames(DS), name=name(DS), kwargs...)
end
