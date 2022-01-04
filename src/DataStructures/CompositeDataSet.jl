

ToDataVec(M::AbstractVector) = floatify(M)
ToDataVec(M::AbstractMatrix) = reduce(vcat, collect(eachrow(ToArray(M))))
ToDataVec(M::DataFrame) = M |> ToArray |> ToDataVec

ToArray(df::AbstractVector{<:Real}) = df |> floatify
ToArray(df::AbstractMatrix) = size(df,2) == 1 ? floatify(df[:,1]) : floatify(df)
ToArray(df::DataFrame) = size(df,2) == 1 ? convert(Vector{suff(df)}, floatify(df[:,1])) : convert(Matrix{suff(df)}, floatify(df))
function ToArray(df::AbstractVector{<:Union{Missing, AbstractFloat}})
    any(ismissing, df) && throw("Input contains missing values.")
    convert(Vector{suff(df)},floatify(df))
end


function ReadIn(df::DataFrame, xdims::Int=1, ydims::Int=Int((size(df,2)-1)/2); xerrs::Bool=false, stripedXs::Bool=true, stripedYs::Bool=true, verbose::Bool=true)
    if xerrs
        (size(df,2) != 2xdims + 2ydims) && throw("Inconsistent no. of columns on DataFrame: got $(size(df,2))")
        Xcols = stripedXs ? (1:2:2xdims) : (1:xdims)
        Xerrs = stripedXs ? (2:2:2xdims) : (xdims+1:2xdims)
        Ycols = stripedYs ? ((2xdims+1):2:(2xdims + 2ydims)) : ((2xdims+1):(2xdims + ydims))
        Yerrs = stripedYs ? ((2xdims+2):2:(2xdims + 2ydims)) : ((2xdims+ydims+1):(2xdims + 2ydims))
    else
        (size(df,2) != xdims + 2ydims) && throw("Inconsistent no. of columns on DataFrame: got $(size(df,2))")
        Xcols = 1:xdims;        Xerrs = Val(false)
        Ycols = stripedYs ? ((xdims+1):2:(xdims + 2ydims)) : ((xdims+1):(xdims + ydims))
        Yerrs = stripedYs ? ((xdims+2):2:(xdims + 2ydims)) : ((xdims+ydims+1):(xdims + 2ydims))
    end
    xnames = names(df[:,Xcols]);    ynames = names(df[:,Ycols])
    verbose && println("Variable names inferred from DataFrame: xnames=$xnames, ynames=$ynames.")
    DSs = _ReadIn(df, Xcols, Xerrs, Ycols, Yerrs)
    InformNames(DSs, xnames, ynames)
end

function _ReadIn(df::DataFrame, xcols::AbstractVector{<:Int}, xerrs::AbstractVector{<:Int}, ycols::AbstractVector{<:Int}, yerrs::AbstractVector{<:Int})
    X = df[:, xcols];    Xerr = df[:, xerrs];   Y = df[:, ycols];    Yerr = df[:, yerrs]
    DSs = Array{AbstractDataSet}(undef, size(Y,2))
    for (i,Col) in enumerate(eachcol(Y))
        inds = DitchMissingRows(X, Col)
        DSs[i] = DataSetExact(ToDataVec(X[inds,:]), ToArray(Xerr[inds,:]), ToArray(Y[inds,:]), ToArray(Yerr[inds,:]), (sum(inds), size(X,2), 1))
    end;    DSs
end

function _ReadIn(df::DataFrame, xcols::AbstractVector{<:Int}, xerrs::Val{false}, ycols::AbstractVector{<:Int}, yerrs::AbstractVector{<:Int})
    X = df[:, xcols];    Y = df[:, ycols];    Yerr = df[:, yerrs]
    DSs = Array{AbstractDataSet}(undef, size(Y,2))
    for (i,Col) in enumerate(eachcol(Y))
        inds = DitchMissingRows(X, Col)
        DSs[i] = DataSet(ToDataVec(X[inds,:]), ToArray(Y[inds,i]), ToArray(Yerr[inds,i]), (sum(inds), size(X,2), 1))
    end;    DSs
end


DitchMissingRows(df1, df2) = hcat(df1, df2) |> DitchMissingRows
DitchMissingRows(df) = DitchMissingRows(DataFrame(df, :auto))
function DitchMissingRows(df::Union{DataFrame, AbstractArray{<:Union{Missing,AbstractFloat}}})
    inds = falses(size(df,1))
    for i in eachindex(inds)
        if !any(ismissing, df[i,:])
            inds[i] = true
        end
    end;    inds
end


function SplitDS(DS::DataSet)
    if typeof(ysigma(DS)) <: AbstractVector
        return [InformNames(DataSet(xdata(DS), ydata(DS)[i:ydim(DS):end], ysigma(DS)[i:ydim(DS):end], (Npoints(DS), xdim(DS), 1)), xnames(DS), ynames(DS)[i:ydim(DS):end]) for i in 1:ydim(DS)]
    else
        return [InformNames(DataSet(xdata(DS), ydata(DS)[i:ydim(DS):end], ysigma(DS)[i:ydim(DS):end,i:ydim(DS):end], (Npoints(DS), xdim(DS), 1)), xnames(DS), ynames(DS)[i:ydim(DS):end]) for i in 1:ydim(DS)]
    end
end
function SplitDS(DS::DataSetExact)
    if typeof(ysigma(DS)) <: AbstractVector
        return [InformNames(DataSetExact(xdata(DS), xsigma(DS), ydata(DS)[i:ydim(DS):end], ysigma(DS)[i:ydim(DS):end], (Npoints(DS), xdim(DS), 1)), xnames(DS), ynames(DS)[i:ydim(DS):end]) for i in 1:ydim(DS)]
    else

        return [InformNames(DataSetExact(xdata(DS), xsigma(DS), ydata(DS)[i:ydim(DS):end], ysigma(DS)[i:ydim(DS):end,i:ydim(DS):end], (Npoints(DS), xdim(DS), 1)), xnames(DS), ynames(DS)[i:ydim(DS):end]) for i in 1:ydim(DS)]
    end
end


# Add Namelist of y-components to be able to look up order of columns
"""
The `CompositeDataSet` type is a more elaborate (and typically less performant) container for storing data.
Essentially, it splits observed data which has multiple `y`-components into separate data containers (e.g. of type `DataSet`), each of which corresponds to one of the components of the `y`-data.
Crucially, each of the smaller data containers still shares the same "kind" of `x`-data, that is, the same `xdim`, units and so on, although they do **not** need to share the exact same particular `x`-data.

The main advantage of this approach is that it can be applied when there are `missing` `y`-components in some observations.
A typical use case for `CompositeDataSet`s are time series where multiple quantities are tracked but not every quantity is necessarily recorded at each time step.
Example:
```julia
using DataFrames
t = [1,2,3,4]
y₁ = [2.5, 6, missing, 9];      y₂ = [missing, 5, 3.1, 1.4]
σ₁ = 0.3*ones(4);               σ₂ = [missing, 0.2, 0.1, 0.5]
df = DataFrame([t y₁ σ₁ y₂ σ])

xdim = 1;   ydim = 2
CompositeDataSet(df, xdim, ydim; xerrs=false, stripedYs=true)
```
The boolean-valued keywords `stripedXs` and `stripedYs` can be used to indicate to the constructor whether the values and corresponding ``1\\sigma`` uncertainties are given in alternating order, or whether the initial block of `ydim` many columns are the values and the second `ydim` many columns are the corresponding uncertainties.
Also, `xerrs=true` can be used to indicate that the `x`-values also carry uncertainties.
Basically all functions which can be called on other data containers such as `DataSet` have been specialized to also work with `CompositeDataSet`s.
"""
struct CompositeDataSet <: AbstractDataSet
    DSs::AbstractVector{<:AbstractDataSet}
    InvCov::AbstractMatrix{<:Number}
    logdetInvCov::Real
    WoundX::AbstractVector
    SharedYdim::Val
    function CompositeDataSet(pDSs::AbstractVector{<:AbstractDataSet})
        !all(DS->xdim(DS)==xdim(pDSs[1]), pDSs) && throw("Inconsistent dimensionality of x-data between data containers.")
        DSs = reduce(vcat, map(SplitDS, pDSs))
        InvCov = mapreduce(yInvCov, BlockMatrix, DSs) |> HealthyCovariance
        CompositeDataSet(DSs, InvCov, logdet(InvCov), unique(mapreduce(WoundX, vcat, DSs)), Val(all(DS->ydim(DS)==ydim(DSs[1]), DSs)))
    end
    function CompositeDataSet(DSs::AbstractVector{<:AbstractDataSet}, InvCov::AbstractMatrix, logdetInvCov::Real, WoundX::AbstractVector, SharedYdim::Val)
        new(DSs, InvCov, logdetInvCov, WoundX, SharedYdim)
    end
end
CompositeDataSet(DS::AbstractDataSet) = CompositeDataSet([DS])
function CompositeDataSet(df::DataFrame, xdims::Int=1, ydims::Int=Int((size(df,2)-1)/2); xerrs::Bool=false, stripedXs::Bool=true, stripedYs::Bool=true)
    CompositeDataSet(ReadIn(floatify(df), xdims, ydims; xerrs=xerrs, stripedXs=stripedXs, stripedYs=stripedYs))
end


# For SciMLBase.remake
CompositeDataSet(;
DSs::AbstractVector{<:AbstractDataSet}=[DataSet([0.],[0.],[1.])],
InvCov::AbstractMatrix=Diagonal([1,2.]),
logdetInvCov::Real=-Inf,
WoundX::AbstractVector=[0.],
SharedYdim::Val=Val(true)) = CompositeDataSet(DSs, logdetInvCov, WoundX, SharedYdim)


Data(CDS::CompositeDataSet) = CDS.DSs
xdata(CDS::CompositeDataSet) = mapreduce(xdata, vcat, Data(CDS))
ydata(CDS::CompositeDataSet) = mapreduce(ydata, vcat, Data(CDS))

# BlockReduce(X::AbstractVector{<:AbstractVector{<:Number}}) = reduce(vcat, X)
# BlockReduce(X::AbstractVector{<:AbstractMatrix{<:Number}}) = reduce(BlockMatrix, X)
BlockReduce(X::AbstractVector{<:AbstractArray{<:Number}}) = reduce(BlockMatrix, [(typeof(x) <: AbstractVector ? Diagonal(x.^2) : x) for x in X])

ysigma(CDS::CompositeDataSet) = map(ysigma, Data(CDS)) |> BlockReduce
xsigma(CDS::CompositeDataSet) = map(xsigma, Data(CDS)) |> BlockReduce
yInvCov(CDS::CompositeDataSet) = mapreduce(yInvCov, BlockMatrix, Data(CDS))

Npoints(CDS::CompositeDataSet) = mapreduce(Npoints, +, Data(CDS))
ydim(CDS::CompositeDataSet) = mapreduce(ydim, +, Data(CDS))
xdim(CDS::CompositeDataSet) = xdim(Data(CDS)[1])

WoundX(CDS::CompositeDataSet) = CDS.WoundX
logdetInvCov(CDS::CompositeDataSet) = CDS.logdetInvCov

DataspaceDim(CDS::CompositeDataSet) = mapreduce(DS->Npoints(DS)*ydim(DS), +, Data(CDS))

xnames(CDS::CompositeDataSet) = xnames(Data(CDS)[1])
ynames(CDS::CompositeDataSet) = mapreduce(ynames, vcat, Data(CDS))



function InformNames(CDS::CompositeDataSet, xnames::AbstractVector{String}, ynames::AbstractVector{String})
    CompositeDataSet(InformNames(Data(CDS), xnames, ynames))
end
function InformNames(DSs::AbstractVector{<:AbstractDataSet}, xnames::AbstractVector{String}, ynames::AbstractVector{String})
    # Use InformNames for single DataSet recursively
    @assert length(ynames) == sum(ydim.(DSs)) && all(x->xdim(x)==length(xnames), DSs)
    Res = Vector{AbstractDataSet}(undef, length(DSs))
    j = 1   # Use this to infer how many elements have been popped from ynames
    for i in 1:length(DSs)
        Res[i] = InformNames(DSs[i], xnames, ynames[j:j-1+ydim(DSs[i])])
        j += ydim(DSs[i])
    end;    Res
end


function _CustomOrNot(CDS::CompositeDataSet, model::Function, θ::AbstractVector{<:Number}, woundX::AbstractVector, custom::Val{false}, inplace::Val{false}; kwargs...)
    @assert CDS.SharedYdim isa Val{true} && ydim(Data(CDS)[1]) == 1
    # reduce(vcat, transpose) faster than Unpack?
    X = unique(woundX)
    _FillResVector(CDS, X, reduce(vcat, map(z->transpose(model(z, θ; kwargs...)), X)))
end

# Apparently reduce(vcat, map(z->transpose(G(z)), X))  just as fast as   transpose(reshape(reduce(vcat, map(z->transpose(G(z)), X)), ydim, :))
function _CustomOrNot(CDS::CompositeDataSet, model::Function, θ::AbstractVector{<:Number}, woundX::AbstractVector, custom::Val{true}, inplace::Val{false}; kwargs...)
    @assert CDS.SharedYdim isa Val{true} && ydim(Data(CDS)[1]) == 1
    # reduce(vcat, transpose) faster than Unpack?
    X = unique(woundX)
    _FillResVector(CDS, X, (ydim(CDS) == 1 ? reshape(model(X, θ; kwargs...), :, 1) : transpose(reshape(model(X, θ; kwargs...), ydim(CDS), :))))
end

function _FillResVector(CDS::CompositeDataSet, X::AbstractVector, Mapped::AbstractMatrix{<:Number})
    Res = Vector{suff(Mapped)}(undef, DataspaceDim(CDS));      i = 1
    for SetInd in 1:length(Data(CDS))
        for xval in WoundX(Data(CDS)[SetInd])
            # Res[i] = view(Mapped, findfirst(isequal(xval),X), SetInd]
            Res[i] = Mapped[findfirst(isequal(xval),X), SetInd]
            i += 1
        end
    end;    return Res
end


function _CustomOrNotdM(CDS::CompositeDataSet, dmodel::Function, θ::AbstractVector{<:Number}, woundX::AbstractVector, custom::Val{false}, inplace::Val{false}; kwargs...)
    @assert CDS.SharedYdim isa Val{true} && ydim(Data(CDS)[1]) == 1
    X = unique(woundX)
    _FillResMatrix(CDS, X, map(z->dmodel(z,θ; kwargs...), X))
end

function _CustomOrNotdM(CDS::CompositeDataSet, dmodel::Function, θ::AbstractVector{<:Number}, woundX::AbstractVector, custom::Val{true}, inplace::Val{false}; kwargs...)
    @assert CDS.SharedYdim isa Val{true} && ydim(Data(CDS)[1]) == 1
    X = unique(woundX);    Mapped = dmodel(X, θ; kwargs...)
    [view(Mapped, (1 + (i-1)*ydim(CDS)):(i*ydim(CDS)) , :) for i in 1:length(X)]
    _FillResMatrix(CDS, X, map(z->dmodel(z,θ; kwargs...), X))
end

_FillResMatrix(CDS::CompositeDataSet, X::AbstractVector, Mapped::AbstractVector{<:AbstractMatrix{<:Number}}) = reduce(vcat, map(i->_getViewDmod(CDS,i,X,Mapped), 1:length(Data(CDS))))

@inline function _getViewDmod(CDS::CompositeDataSet, SetInd::Int, X::AbstractVector, Mapped::AbstractVector{<:AbstractMatrix{<:Number}})
    subXs = WoundX(Data(CDS)[SetInd])
    Res = Mapped[findfirst(isequal(subXs[1]), X)][SetInd,:] |> transpose
    for i in 2:length(subXs)
        Res = vcat(Res, transpose(Mapped[findfirst(isequal(subXs[i]), X)][SetInd,:]))
    end;    Res
end

# Ignore xpositions
RecipesBase.@recipe function f(CDS::CompositeDataSet, xpositions::AbstractVector{<:Number}=xdata(CDS))
    xdim(CDS) != 1 && throw("Not programmed for plotting xdim != 1 yet.")
    !all(x->ydim(x)==1, Data(CDS)) && throw("Not programmed for plotting ydim > 1 yet.")
    xguide -->  xnames(CDS)[1]
    yguide -->  "Observations"
    for (i,DS) in enumerate(Data(CDS))
        @series begin
            label --> "Data: " * ynames(DS)[1]
            markercolor --> [:red,:blue,:green,:orange,:grey][i]
            DS
        end
    end
end


function ResidualStandardError(CDS::CompositeDataSet, model::ModelOrFunction, MLE::AbstractVector{<:Number})
    any(x->DataspaceDim(x) ≤ length(MLE), Data(CDS)) && (println("Too few data points to compute RSE"); return nothing)
    ydiff = ydata(CDS) - EmbeddingMap(CDS, model, MLE);    Res = zeros(length(Data(CDS)));    startind = 1
    for i in eachindex(Res)
        ypred = view(ydiff, startind:startind+Npoints(Data(CDS)[i])*ydim(Data(CDS)[i])-1)
        startind += DataspaceDim(Data(CDS)[i])
        Res[i] = sqrt(sum(abs2, ypred) / (DataspaceDim(Data(CDS)[i]) - length(MLE)))
    end
    @assert (startind - 1) == DataspaceDim(CDS)
    ydim(CDS) == 1 ? Res[1] : Res
end
