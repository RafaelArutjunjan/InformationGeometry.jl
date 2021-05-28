

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

# http://docs.junolab.org/stable/man/info_developer/#
# hastreeview, numberofnodes, treelabel, treenode
TreeViews.hastreeview(x::Union{AbstractDataSet,AbstractDataModel,ModelMap}) = true
TreeViews.numberofnodes(x::AbstractDataSet) = 4
TreeViews.numberofnodes(x::AbstractDataModel) = 4
TreeViews.numberofnodes(x::ModelMap) = 4
function TreeViews.treelabel(io::IO, DS::Union{AbstractDataSet,AbstractDataModel}, mime::MIME"text/plain" = MIME"text/plain"())
    show(io, mime, Text(Base.summary(DS)))
end
# To hide the treenode display, simply return missing:
# treenode(x::Foo, i::Int) = missing


import Base: show
#### Need proper show() methods for DataSet, DataModel, ModelMap
#### Show Distribution types for DataSetExact
function Base.show(io::IO, mime::MIME"text/plain", DS::AbstractDataSet)
    println(io, "$(nameof(typeof(DS))) with N=$(Npoints(DS)), xdim=$(xdim(DS)) and ydim=$(ydim(DS)):")
    print(io, "x-data: ");    show(io, mime, xdata(DS));    print(io, "\n")
    if typeof(DS) == DataSetExact
        if typeof(xsigma(DS)) <: AbstractVector
            println(io, "Standard deviation associated with x-data:")
            show(io, mime, xsigma(DS))
        else
            println(io, "Covariance matrix associated with x-data:")
            show(io, mime, xsigma(DS))
        end
        print(io, "\n")
    end
    print(io, "y-data: ");    show(io, mime, ydata(DS));    print(io, "\n")
    if typeof(ysigma(DS)) <: AbstractVector
        println(io, "Standard deviation associated with y-data:")
        show(io, mime, ysigma(DS))
    else
        println(io, "Covariance matrix associated with y-data:")
        show(io, mime, ysigma(DS))
    end
end

function Base.show(io::IO, DS::AbstractDataSet)
    println(io, "$(nameof(typeof(DS))) with N=$(Npoints(DS)), xdim=$(xdim(DS)) and ydim=$(ydim(DS)):")
    print(io, "x-data: ");    show(io, xdata(DS));    print(io, "\n")
    if typeof(DS) == DataSetExact
        if typeof(xsigma(DS)) <: AbstractVector
            println(io, "Standard deviation associated with x-data:")
            show(io, xsigma(DS))
        else
            println(io, "Covariance matrix associated with x-data:")
            show(io, xsigma(DS))
        end
        print(io, "\n")
    end
    print(io, "y-data: ");    show(io, ydata(DS));    print(io, "\n")
    if typeof(ysigma(DS)) <: AbstractVector
        println(io, "Standard deviation associated with y-data:")
        show(io, ysigma(DS))
    else
        println(io, "Covariance matrix associated with y-data:")
        show(io, ysigma(DS))
    end
end

##### StaticOutput?
function Base.show(io::IO, mime::MIME"text/plain", DM::AbstractDataModel)
    Expr = ToExpr(Data(DM), Predictor(DM))
    IsLin = try IsLinearParameter(DM) catch; nothing end
    println(io, mime, "$(nameof(typeof(DM))) containing a $(nameof(typeof(Data(DM))))")
    Jac = if GeneratedFromAutoDiff(dPredictor(DM))
        "generated via automatic differentiation"
    elseif GeneratedFromSymbolic(dPredictor(DM))
        "symbolically provided"
    else
        "manually provided by user"
    end
    println(io, mime, "Model jacobian " * Jac)
    if DM isa DataModel
        println(io, mime, "Maximum Likelihood Estimate: $(MLE(DM))")
        println(io, mime, "Maximal value of log-likelihood: $(LogLikeMLE(DM))")
    end
    !isa(Expr, Nothing) && println(io, mime, "Model:  y(x,θ) = $Expr")
    (IsLin === nothing) && println(io, mime, "Model parametrization linear in n-th parameter: $(IsLin)")
end


function Base.show(io::IO, DM::AbstractDataModel)
    Expr = ToExpr(Data(DM), Predictor(DM))
    IsLin = try IsLinearParameter(DM) catch; nothing end
    println(io, "$(nameof(typeof(DM))) containing a $(nameof(typeof(Data(DM))))")
    Jac = if GeneratedFromAutoDiff(dPredictor(DM))
        "generated via automatic differentiation"
    elseif GeneratedFromSymbolic(dPredictor(DM))
        "symbolically provided"
    else
        "manually provided by user"
    end
    println(io, "Model jacobian " * Jac)
    if DM isa DataModel
        println(io, "Maximum Likelihood Estimate: $(MLE(DM))")
        println(io, "Maximal value of log-likelihood: $(LogLikeMLE(DM))")
    end
    !isa(Expr, Nothing) && println(io, "Model:  y(x,θ) = $Expr")
    (IsLin === nothing) && println(io, "Model parametrization linear in n-th parameter: $(IsLin)")
end
