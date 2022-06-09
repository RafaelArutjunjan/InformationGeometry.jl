

macro CSI_str(str)
    return :(string("\x1b[", $(esc(str)), "m"))
end
const TYPE_COLOR = CSI"36"
# const RED_COLOR = CSI"35"
# const BLUE_COLOR = CSI"34;1"
# const YELLOW_COLOR = CSI"33"
const ORANGE_COLOR = CSI"38;5;208"
const NO_COLOR = CSI"0"

import Base: summary
Base.summary(DS::AbstractDataSet) = string(TYPE_COLOR, nameof(typeof(DS)), NO_COLOR, " with N=$(Npoints(DS)), xdim=$(xdim(DS)) and ydim=$(ydim(DS))")

GeneratedFromAutoDiff(F::Function) = occursin("Autodmodel", string(nameof(typeof(F))))
GeneratedFromAutoDiff(F::ModelMap) = GeneratedFromAutoDiff(F.Map)
GeneratedFromSymbolic(F::Function) = occursin("SymbolicModel", string(nameof(typeof(F))))
GeneratedFromSymbolic(F::ModelMap) = GeneratedFromSymbolic(F.Map)

###### Useful info: Autodmodel? Symbolic? StaticArray output? In-place?
function Base.summary(DM::AbstractDataModel)
    # Also use "RuntimeGeneratedFunction" string from build_function in ModelingToolkit.jl
    string(TYPE_COLOR, nameof(typeof(DM)),
    NO_COLOR, " containing ",
    TYPE_COLOR, nameof(typeof(Data(DM))),
    NO_COLOR, ". Model jacobian: ",
    ORANGE_COLOR, (GeneratedFromAutoDiff(dPredictor(DM)) ? "AutoDiff" : (GeneratedFromSymbolic(dPredictor(DM)) ? "Symbolic" : "manually provided")),
    NO_COLOR)
end

function Base.summary(M::ModelMap)
    string(TYPE_COLOR, "ModelMap ",
        ORANGE_COLOR, (isinplacemodel(M) ? "in-place" : "out-of-place"),
        NO_COLOR, " with xdim=$(xdim(M)), ydim=$(ydim(M)), pdim=$(pdim(M))")
end


# http://docs.junolab.org/stable/man/info_developer/#
# hastreeview, numberofnodes, treelabel, treenode
TreeViews.hastreeview(x::Union{AbstractDataSet,AbstractDataModel,ModelMap}) = true
TreeViews.numberofnodes(x::AbstractDataSet) = 4
TreeViews.numberofnodes(x::AbstractDataModel) = 4
TreeViews.numberofnodes(x::ModelMap) = 4
TreeViews.numberofnodes(x::CompositeDataSet) = 1
TreeViews.numberofnodes(x::GeneralizedDataSet) = 1

function TreeViews.treelabel(io::IO, DS::Union{AbstractDataSet,AbstractDataModel,ModelMap}, mime::MIME"text/plain"=MIME"text/plain"())
    show(io, mime, Text(Base.summary(DS)))
end
# To hide the treenode display, simply return missing:
# treenode(x::Foo, i::Int) = missing


function ParamSummary(DM::AbstractDataModel)
    IsLin = try IsLinearParameter(DM) catch; nothing end
    L, U = if Predictor(DM) isa ModelMap
        round.(Predictor(DM).Domain.L; sigdigits=2), round.(Predictor(DM).Domain.U; sigdigits=2)
    else
        fill(-Inf, pdim(DM)), fill(Inf, pdim(DM))
    end
    mle = MLE(DM)
    OnLowerBoundary = @. (mle-L) / (U-L) < 1/200
    OnUpperBoundary = @. (U-mle) / (U-L) < 1/200
    if !isnothing(IsLin) && any(IsLin)
        H = Highlighter((data,zeile,spalte) -> spalte < 5 && (OnLowerBoundary[zeile] || OnUpperBoundary[zeile]); bold=true, foreground=:red)
        pretty_table([pnames(DM) L MLEuncert(DM;verbose=false) U IsLin]; crop=:none, header=["Parameter", "Lower Bound", "MLE", "Upper Bound", "Linear Dependence"], alignment=[:l, :c, :c, :c, :c], highlighters=H)
    else
        H = Highlighter((data,zeile,spalte) -> OnLowerBoundary[zeile] || OnUpperBoundary[zeile]; bold=true, foreground=:red)
        pretty_table([pnames(DM) L MLEuncert(DM;verbose=false) U]; crop=:none, header=["Parameter", "Lower Bound", "MLE", "Upper Bound"], alignment=[:l, :c, :c, :c], highlighters=H)
    end
end
function ParamSummary(io::IO, DM::AbstractDataModel)
    IsLin = try IsLinearParameter(DM) catch; nothing end
    L, U = if Predictor(DM) isa ModelMap
        round.(Predictor(DM).Domain.L; sigdigits=2), round.(Predictor(DM).Domain.U; sigdigits=2)
    else
        fill(-Inf, pdim(DM)), fill(Inf, pdim(DM))
    end
    mle = MLE(DM)
    OnLowerBoundary = @. (mle-L) / (U-L) < 1/200
    OnUpperBoundary = @. (U-mle) / (U-L) < 1/200
    if !isnothing(IsLin) && any(IsLin)
        H = Highlighter((data,zeile,spalte) -> spalte < 5 && (OnLowerBoundary[zeile] || OnUpperBoundary[zeile]); bold=true, foreground=:red)
        pretty_table(io, [pnames(DM) L MLEuncert(DM;verbose=false) U IsLin]; crop=:none, header=["Parameter", "Lower Bound", "MLE", "Upper Bound", "Linear Dependence"], alignment=[:l, :c, :c, :c, :c], highlighters=H)
    else
        H = Highlighter((data,zeile,spalte) -> OnLowerBoundary[zeile] || OnUpperBoundary[zeile]; bold=true, foreground=:red)
        pretty_table(io, [pnames(DM) L MLEuncert(DM;verbose=false) U]; crop=:none, header=["Parameter", "Lower Bound", "MLE", "Upper Bound"], alignment=[:l, :c, :c, :c], highlighters=H)
    end
end


import Base: show
#### Need proper show() methods for DataSet, DataModel, ModelMap
#### Show Distribution types for DataSetExact
function Base.show(io::IO, DS::AbstractDataSet)
    if length(name(DS)) > 0
        println(io, "$(nameof(typeof(DS))) '$(name(DS))' with N=$(Npoints(DS)), xdim=$(xdim(DS)) and ydim=$(ydim(DS)):")
    else
        println(io, "$(nameof(typeof(DS))) with N=$(Npoints(DS)), xdim=$(xdim(DS)) and ydim=$(ydim(DS)):")
    end
    xnameinsert = any(xnames(DS) .!= CreateSymbolNames(xdim(DS),"x")) ? " ["*prod(xnames(DS) .* ", ")[1:end-2]*"] " : ""
    print(io, "x-data" * xnameinsert * ": ")
    show(io, xdata(DS));    print(io, "\n")
    if DS isa DataSetExact
        if xsigma(DS) isa AbstractVector
            println(io, "Standard deviation associated with x-data:")
            show(io, xsigma(DS))
        else
            println(io, "Covariance matrix associated with x-data:")
            show(io, xsigma(DS))
        end
        print(io, "\n")
    end
    ynameinsert = any(ynames(DS) .!= CreateSymbolNames(ydim(DS),"y")) ? " ["*prod(ynames(DS) .* ", ")[1:end-2]*"] " : ""
    print(io, "y-data" * ynameinsert * ": ")
    show(io, ydata(DS));    print(io, "\n")
    if ysigma(DS) isa AbstractVector
        println(io, "Standard deviation associated with y-data:")
        show(io, ysigma(DS))
    else
        println(io, "Covariance matrix associated with y-data:")
        show(io, ysigma(DS))
    end
end

function Base.show(io::IO, GDS::GeneralizedDataSet)
    println(io, "$(nameof(typeof(GDS))) with N=$(Npoints(GDS)), xdim=$(xdim(GDS)) and ydim=$(ydim(GDS)):")
    print(io, "Combined x-y data: ");    show(io, GetMean(dist(GDS)));    print(io, "\n")
    print(io, "Combined x-y covariance: ");    show(io, Sigma(dist(GDS)));    print(io, "\n")
end

function Base.show(io::IO, DM::AbstractDataModel)
    Expr = SymbolicModel(DM)
    if length(name(Data(DM))) > 0
        println(io, "$(nameof(typeof(DM))) containing a $(nameof(typeof(Data(DM)))) '$(name(Data(DM)))'")
    else
        println(io, "$(nameof(typeof(DM))) containing a $(nameof(typeof(Data(DM))))")
    end
    Jac = if GeneratedFromAutoDiff(dPredictor(DM))
        "generated via automatic differentiation"
    elseif GeneratedFromSymbolic(dPredictor(DM))
        "symbolically provided"
    else
        "manually provided by user"
    end
    println(io, "Model jacobian " * Jac)
    println(io, "Maximal value of log-likelihood: $(LogLikeMLE(DM))")
    Expr[1] == 'y' && println(io, "Model Expr:  $Expr")
    ParamSummary(io, DM)
end

function Base.show(io::IO, M::ModelMap)
    Expr = SymbolicModel(M)
    if length(name(M)) > 0
        println(io, (isinplacemodel(M) ? "In-place" : "Out-of-place") * " model '$(name(M))' with with xdim=$(xdim(M)), ydim=$(ydim(M)), pdim=$(pdim(M))")
    else
        println(io, (isinplacemodel(M) ? "In-place" : "Out-of-place") * " Model with with xdim=$(xdim(M)), ydim=$(ydim(M)), pdim=$(pdim(M))")
    end
    Expr[1] == 'y' && println(io, "Model Expr:  $Expr")
    pnames(M) != CreateSymbolNames(pdim(M)) && println(io, "Parameters: $(pnames(M))")
end
