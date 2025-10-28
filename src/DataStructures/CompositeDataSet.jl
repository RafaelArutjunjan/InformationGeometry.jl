

ToDataVec(M::AbstractVector) = floatify(M)
ToDataVec(M::AbstractMatrix) = reduce(vcat, collect(eachrow(ToArray(M))))
ToDataVec(M::DataFrame) = M |> ToArray |> ToDataVec

ToArray(df::AbstractVector{<:Real}) = df |> floatify
ToArray(df::AbstractMatrix) = size(df,2) == 1 ? floatify(df[:,1]) : floatify(df)
ToArray(df::DataFrame) = size(df,2) == 1 ? Vector(floatify(df[:,1])) : Matrix(floatify(df))
function ToArray(df::AbstractVector{<:Union{Missing, AbstractFloat}})
    any(ismissing, df) && throw("Input contains missing values.")
    # Ensure Type is not union with Missing anymore
    MissingToNan(x::Number) = x
    MissingToNan(x::Missing) = NaN
    MissingToNan.(floatify(df))
end


function ReadIn(df::DataFrame, xdims::Int=1, ydims::Int=Int((size(df,2)-xdims)/2); xerrs::Bool=false, stripedXs::Bool=true, stripedYs::Bool=true, verbose::Bool=true,
                        xnames::Union{Nothing,AbstractVector{<:StringOrSymb}}=nothing, ynames::Union{Nothing,AbstractVector{<:StringOrSymb}}=nothing)
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
    Xnames = isnothing(xnames) ? names(df[:,Xcols]) : xnames;    Ynames = isnothing(ynames) ? names(df[:,Ycols]) : ynames
    verbose && (isnothing(xnames) || isnothing(ynames)) && @info "Variable names inferred from DataFrame: xnames=$Xnames, ynames=$Ynames."
    DSs = _ReadIn(df, Xcols, Xerrs, Ycols, Yerrs)
    InformNames(DSs, Xnames, Ynames)
end

function _ReadIn(df::DataFrame, xcols::AbstractVector{<:Int}, xerrs::AbstractVector{<:Int}, ycols::AbstractVector{<:Int}, yerrs::AbstractVector{<:Int})
    X = df[:, xcols];    Xerr = df[:, xerrs];   Y = df[:, ycols];    Yerr = df[:, yerrs]
    [(inds = DitchMissingRows(X, Col);    DataSetExact(ToDataVec(X[inds,:]), ToArray(Xerr[inds,:]), ToArray(Y[inds,:]), ToArray(Yerr[inds,:]), (sum(inds), size(X,2), 1))) for (i,Col) in enumerate(eachcol(Y))]
end

function _ReadIn(df::DataFrame, xcols::AbstractVector{<:Int}, xerrs::Val{false}, ycols::AbstractVector{<:Int}, yerrs::AbstractVector{<:Int})
    X = df[:, xcols];    Y = df[:, ycols];    Yerr = df[:, yerrs]
    [(inds = DitchMissingRows(X, Col);      DataSet(ToDataVec(X[inds,:]), ToArray(Y[inds,i]), ToArray(Yerr[inds,i]), (sum(inds), size(X,2), 1))) for (i,Col) in enumerate(eachcol(Y))]
end


DitchMissingRows(B::AbstractArray, df::DataFrame) = DitchMissingRows(df, B)
DitchMissingRows(df::DataFrame, B::AbstractMatrix) = DitchMissingRows(df, DataFrame(B, :auto))
DitchMissingRows(df::DataFrame, B::AbstractVector) = DitchMissingRows(df, DataFrame(reshape(B, :, 1), :auto))
DitchMissingRows(df1, df2) = hcat(df1, df2) |> DitchMissingRows
DitchMissingRows(df) = DitchMissingRows(DataFrame(df, :auto))
DitchMissingRows(df::Union{DataFrame, AbstractArray{<:Union{Missing,AbstractFloat}}})::BitVector = map(row->!any(ismissing, row), eachrow(df))


function SplitDS(DS::DataSet)
    if ysigma(DS) isa AbstractVector
        [InformNames(DataSet(xdata(DS), ydata(DS)[i:ydim(DS):end], ysigma(DS)[i:ydim(DS):end], (Npoints(DS), xdim(DS), 1); name=name(DS)), Xnames(DS), Ynames(DS)[i:ydim(DS):end]) for i in 1:ydim(DS)]
    else
        [InformNames(DataSet(xdata(DS), ydata(DS)[i:ydim(DS):end], ysigma(DS)[i:ydim(DS):end,i:ydim(DS):end], (Npoints(DS), xdim(DS), 1); name=name(DS)), Xnames(DS), Ynames(DS)[i:ydim(DS):end]) for i in 1:ydim(DS)]
    end
end
function SplitDS(DS::DataSetExact)
    if ysigma(DS) isa AbstractVector
        [InformNames(DataSetExact(xdata(DS), xsigma(DS), ydata(DS)[i:ydim(DS):end], ysigma(DS)[i:ydim(DS):end], (Npoints(DS), xdim(DS), 1); name=name(DS)), Xnames(DS), Ynames(DS)[i:ydim(DS):end]) for i in 1:ydim(DS)]
    else
        [InformNames(DataSetExact(xdata(DS), xsigma(DS), ydata(DS)[i:ydim(DS):end], ysigma(DS)[i:ydim(DS):end,i:ydim(DS):end], (Npoints(DS), xdim(DS), 1); name=name(DS)), Xnames(DS), Ynames(DS)[i:ydim(DS):end]) for i in 1:ydim(DS)]
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
df = DataFrame([t y₁ σ₁ y₂ σ₂], :auto)

xdim = 1;   ydim = 2
CompositeDataSet(df, xdim, ydim; xerrs=false, stripedYs=true)
```
The boolean-valued keywords `stripedXs` and `stripedYs` can be used to indicate to the constructor whether the values and corresponding ``1\\sigma`` uncertainties are given in alternating order, or whether the initial block of `ydim` many columns are the values and the second `ydim` many columns are the corresponding uncertainties.
Also, `xerrs=true` can be used to indicate that the `x`-values also carry uncertainties.
Basically all functions which can be called on other data containers such as `DataSet` have been specialized to also work with `CompositeDataSet`s.
"""
struct CompositeDataSet <: AbstractFixedUncertaintyDataSet
    DSs::AbstractVector{<:AbstractDataSet}
    InvCov::AbstractMatrix{<:Number}
    logdetInvCov::Real
    WoundX::AbstractVector
    SharedYdim::Val
    name::Symbol
    function CompositeDataSet(pDSs::AbstractVector{<:AbstractDataSet}; kwargs...)
        !all(DS->xdim(DS)==xdim(pDSs[1]), pDSs) && throw("Inconsistent dimensionality of x-data between data containers.")
        DSs = reduce(vcat, map(SplitDS, pDSs))
        InvCov = mapreduce(yInvCov, BlockMatrix, DSs) |> HealthyCovariance
        CompositeDataSet(DSs, InvCov, logdet(InvCov), unique(mapreduce(WoundX, vcat, DSs)), Val(all(DS->ydim(DS)==ydim(DSs[1]), DSs)); kwargs...)
    end
    function CompositeDataSet(DSs::AbstractVector{<:AbstractDataSet}, InvCov::AbstractMatrix, logdetInvCov::Real, WoundX::AbstractVector, SharedYdim::Val; name::StringOrSymb=Symbol(), kwargs...)
        CompositeDataSet(DSs, InvCov, logdetInvCov, WoundX, SharedYdim, name; kwargs...)
    end
    # What about CompositeDataSets with estimated errors in the future?
    function CompositeDataSet(DSs::AbstractVector{<:AbstractFixedUncertaintyDataSet}, InvCov::AbstractMatrix, logdetInvCov::Real, WoundX::AbstractVector, SharedYdim::Val, name::StringOrSymb)
        new(DSs, InvCov, logdetInvCov, WoundX, SharedYdim, Symbol(name))
    end
end

function (::Type{T})(DS::CompositeDataSet; kwargs...) where T<:Number
    DSs = T.(Data(DS))
    InvCov = mapreduce(yInvCov, BlockMatrix, DSs) |> HealthyCovariance
    CompositeDataSet(DSs, InvCov, logdet(InvCov), unique(mapreduce(WoundX, vcat, DSs)), Val(all(DS->ydim(DS)==ydim(DSs[1]), DSs)); name=name(DS), kwargs...)
end


CompositeDataSet(DS::AbstractDataSet; kwargs...) = CompositeDataSet([DS]; kwargs...)
function CompositeDataSet(df::DataFrame, xdims::Int=1, ydims::Int=Int((size(df,2)-1)/2); xerrs::Bool=false, stripedXs::Bool=true, stripedYs::Bool=true, 
                            xnames::Union{Nothing,AbstractVector{<:StringOrSymb}}=nothing, ynames::Union{Nothing,AbstractVector{<:StringOrSymb}}=nothing, kwargs...)
    CompositeDataSet(ReadIn(floatify(df), xdims, ydims; xerrs, stripedXs, stripedYs, xnames, ynames); kwargs...)
end
function CompositeDataSet(xdf::DataFrame, ydf::DataFrame, sig::Real=1.0; kwargs...)
    CompositeDataSet(xdf, ydf, DataFrame(sig*ones(size(ydf)...), names(ydf).*"_σ"); kwargs...)
end
function CompositeDataSet(xdf::DataFrame, ydf::DataFrame, sigdf::DataFrame; xerrs::Bool=false, stripedYs::Bool=false, kwargs...)
    # Enforce stripedYs=false
    @assert !stripedYs
    CompositeDataSet(hcat(xdf, ydf, sigdf), (xerrs ? Int(size(xdf,2)/2) : size(xdf,2)), size(ydf,2); xerrs, stripedYs, kwargs...)
end

# For SciMLBase.remake
CompositeDataSet(;
DSs::AbstractVector{<:AbstractDataSet}=[DataSet([0.],[0.],[1.])],
InvCov::AbstractMatrix=Diagonal([1,2.]),
logdetInvCov::Real=-Inf,
WoundX::AbstractVector=[0.],
SharedYdim::Val=Val(true),
name::StringOrSymb=Symbol()) = CompositeDataSet(DSs, InvCov, logdetInvCov, WoundX, SharedYdim, name)


Data(CDS::CompositeDataSet) = CDS.DSs
xdata(CDS::CompositeDataSet) = mapreduce(xdata, vcat, Data(CDS))
ydata(CDS::CompositeDataSet) = mapreduce(ydata, vcat, Data(CDS))

# BlockReduce(X::AbstractVector{<:AbstractVector{<:Number}}) = reduce(vcat, X)
# BlockReduce(X::AbstractVector{<:AbstractMatrix{<:Number}}) = reduce(BlockMatrix, X)
BlockReduce(X::AbstractVector{<:AbstractArray{<:Number}}) = reduce(BlockMatrix, [(x isa AbstractVector ? Diagonal(x.^2) : x) for x in X])

ysigma(CDS::CompositeDataSet) = map(ysigma, Data(CDS)) |> BlockReduce |> _TryVectorize
xsigma(CDS::CompositeDataSet) = map(xsigma, Data(CDS)) |> BlockReduce |> _TryVectorize
yInvCov(CDS::CompositeDataSet) = mapreduce(yInvCov, BlockMatrix, Data(CDS))

Npoints(CDS::CompositeDataSet) = mapreduce(Npoints, +, Data(CDS))
ydim(CDS::CompositeDataSet) = mapreduce(ydim, +, Data(CDS))
xdim(CDS::CompositeDataSet) = xdim(Data(CDS)[1])

WoundX(CDS::CompositeDataSet) = CDS.WoundX
logdetInvCov(CDS::CompositeDataSet) = CDS.logdetInvCov

DataspaceDim(CDS::CompositeDataSet) = mapreduce(DS->Npoints(DS)*ydim(DS), +, Data(CDS))

xnames(CDS::CompositeDataSet) = Xnames(CDS) .|> string
ynames(CDS::CompositeDataSet) = Ynames(CDS) .|> string

Xnames(CDS::CompositeDataSet) = Xnames(Data(CDS)[1])
Ynames(CDS::CompositeDataSet) = mapreduce(Ynames, vcat, Data(CDS))

name(CDS::CompositeDataSet) = CDS.name


function InformNames(CDS::CompositeDataSet, xnames::AbstractVector{<:StringOrSymb}, ynames::AbstractVector{<:StringOrSymb})
    CompositeDataSet(InformNames(Data(CDS), xnames, ynames))
end
function InformNames(DSs::AbstractVector{<:AbstractDataSet}, xnames::AbstractVector{<:StringOrSymb}, ynames::AbstractVector{<:StringOrSymb})
    # Use InformNames for single DataSet recursively
    @assert length(ynames) == sum(ydim.(DSs)) && all(x->xdim(x)==length(xnames), DSs)
    Res = Vector{AbstractDataSet}(undef, length(DSs))
    j = 1   # Use this to infer how many elements have been popped from ynames
    for i in eachindex(DSs)
        Res[i] = InformNames(DSs[i], xnames, ynames[j:j-1+ydim(DSs[i])])
        j += ydim(DSs[i])
    end;    Res
end


function _CustomOrNot(CDS::CompositeDataSet, model::Union{Function, ModelMap{false,false}}, θ::AbstractVector{<:Number}, woundX::AbstractVector; kwargs...)
    @assert CDS.SharedYdim isa Val{true} && ydim(Data(CDS)[1]) == 1
    # reduce(vcat, transpose) faster than Unpack?
    X = unique(woundX)
    _FillResVector(CDS, X, reduce(vcat, map(z->transpose(model(z, θ; kwargs...)), X)))
end

# Apparently reduce(vcat, map(z->transpose(G(z)), X))  just as fast as   transpose(reshape(reduce(vcat, map(z->transpose(G(z)), X)), ydim, :))
function _CustomOrNot(CDS::CompositeDataSet, model::ModelMap{false,true}, θ::AbstractVector{<:Number}, woundX::AbstractVector; kwargs...)
    @assert CDS.SharedYdim isa Val{true} && ydim(Data(CDS)[1]) == 1
    # reduce(vcat, transpose) faster than Unpack?
    X = unique(woundX)
    _FillResVector(CDS, X, (ydim(CDS) == 1 ? reshape(model(X, θ; kwargs...), :, 1) : transpose(reshape(model(X, θ; kwargs...), ydim(CDS), :))))
end

# function _CustomOrNot(CDS::CompositeDataSet, model::ModelMap{true}, θ::AbstractVector{<:Number}, woundX::AbstractVector, custom::Val{true}, inplace::Val{false}; kwargs...)
#     throw("Not programmed CompositeDataSet for in-place ModelMaps yet.")
# end

# Also add test for in-place CompositeDataSet with missing values


function _FillResVector(CDS::CompositeDataSet, X::AbstractVector, Mapped::AbstractMatrix{T}) where T<:Number
    Res = Vector{T}(undef, DataspaceDim(CDS));      i = 1
    for SetInd in eachindex(Data(CDS))
        for xval in WoundX(Data(CDS)[SetInd])
            # Res[i] = view(Mapped, findfirst(isequal(xval),X), SetInd]
            Res[i] = Mapped[findfirst(isequal(xval),X), SetInd]
            i += 1
        end
    end;    return Res
end


function _CustomOrNotdM(CDS::CompositeDataSet, dmodel::Union{Function, ModelMap{false,false}}, θ::AbstractVector{<:Number}, woundX::AbstractVector; kwargs...)
    @assert CDS.SharedYdim isa Val{true} && ydim(Data(CDS)[1]) == 1
    X = unique(woundX)
    _FillResMatrix(CDS, X, map(z->dmodel(z,θ; kwargs...), X))
end

function _CustomOrNotdM(CDS::CompositeDataSet, dmodel::ModelMap{false, true}, θ::AbstractVector{<:Number}, woundX::AbstractVector; kwargs...)
    @assert CDS.SharedYdim isa Val{true} && ydim(Data(CDS)[1]) == 1
    X = unique(woundX);    Mapped = dmodel(X, θ; kwargs...)
    [view(Mapped, (1 + (i-1)*ydim(CDS)):(i*ydim(CDS)) , :) for i in eachindex(X)]
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
    if ydim(CDS) ≤ 16
        !all(x->ydim(x)==1, Data(CDS)) && throw("Not programmed for plotting ydim > 1 yet.")
        plot_title --> string(name(CDS))
        xguide -->  string(Xnames(CDS)[1])
        yguide -->  "Observations"
        color_palette = get(plotattributes, :color_palette, :default)
        for (i,DS) in enumerate(Data(CDS))
            @series begin
                label --> "Data: " * string(Ynames(DS)[1])
                # requires ydim(CDS) ≤ 16
                color --> palette(color_palette)[(((i)%15)+1)]
                DS
            end
        end
    else
        Data(CDS)
    end
end


function ResidualStandardError(CDS::CompositeDataSet, model::ModelOrFunction, MLE::AbstractVector{<:Number}; verbose::Bool=true)
    any(x->DataspaceDim(x) ≤ length(MLE), Data(CDS)) && ((verbose && @warn "Too few data points to compute RSE"); return nothing)
    ydiff = ydata(CDS) - EmbeddingMap(CDS, model, MLE);    Res = zeros(length(Data(CDS)));    startind = 1
    for i in eachindex(Res)
        ypred = view(ydiff, startind:startind+Npoints(Data(CDS)[i])*ydim(Data(CDS)[i])-1)
        startind += DataspaceDim(Data(CDS)[i])
        Res[i] = sqrt(sum(abs2, ypred) / (DataspaceDim(Data(CDS)[i]) - length(MLE)))
    end
    @assert (startind - 1) == DataspaceDim(CDS)
    ydim(CDS) == 1 ? Res[1] : Res
end



function ReadLongTableSingleCondition(Df::AbstractDataFrame; Time::Symbol=:time, Measurement::Symbol=:measurement, Noise::Symbol=:noiseParameters, SimulationId::Symbol=:simulationConditionId, ObservableId::Symbol=:observableId, kwargs...)
    df = copy(Df);  df[!, ObservableId] .= Symbol.(df[!, ObservableId]);    df[!, SimulationId] .= Symbol.(df[!, SimulationId])
    @assert length(unique(df[!,SimulationId])) == 1 "Only single value for $SimulationId allowed, got $df."
    L = [(keep = df[!, ObservableId] .== Name;    perm=sortperm(df[!, Time][keep]);   (df[!,SimulationId][1], Name, df[!, Time][keep][perm], df[!, Measurement][keep][perm], df[!, Noise][keep][perm])) for Name in sort(Symbol.(unique(df[!, ObservableId])))]
    if eltype(df[!,Noise]) <: Number
        CompositeDataSet([DataSet(T[3], T[4], T[5], (length(T[4]), 1, 1); xnames=["time"], ynames=[T[2]], name=string(T[1])*"_"*string(T[2])) for T in L]; name=df[!,SimulationId][1], kwargs...)
    else
        throw("Only implemented for fixed known uncertainties, got eltype(df.$Noise)=$(eltype(eltype(df[!,Noise]))).")
    end
end

"""
    ReadLongTable(df::AbstractDataFrame; Time::Symbol=:time, Measurement::Symbol=:measurement, Noise::Symbol=:noiseParameters, SimulationId::Symbol=:simulationConditionId, ObservableId::Symbol=:observableId, kwargs...)
Reads a `DataFrame` in so-called "long" format into (Vector of) `CompositeDataSet`.
Currently only works for fixed known uncertainties, i.e. observation noise.
"""
function ReadLongTable(df::AbstractDataFrame; SimulationId::Symbol=:simulationConditionId, kwargs...)
    Gs = groupby(df, SimulationId)
    map(i->ReadLongTableSingleCondition(Gs[i]; SimulationId, kwargs...), collect(eachindex(Gs)))
end
