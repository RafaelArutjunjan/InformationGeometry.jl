

macro CSI_str(str)
    return :(string("\x1b[", $(esc(str)), "m"))
end


const ORANGE_COLOR = CSI"38;5;208"
# ANSI Format: 38;2;R;G;B
# Color codes from Juno.syntaxcolors()
const TYPE_COLOR = CSI"38;2;86;182;194" # correct hex color: 0x0056b6c2
const NUMBER_COLOR = CSI"38;2;209;154;102" # correct hex color: 0x00d19a66
const STRING_COLOR = CSI"38;2;152;195;121" # correct hex color: 0x0098c379
const OPERATOR_COLOR = CSI"38;2;189;120;220" # correct hex color: 0x00c678dd
const NO_COLOR = CSI"0"


GeneratedFromAutoDiff(F::Function) = occursin("Autodmodel", string(nameof(typeof(F))))
GeneratedFromAutoDiff(F::ModelMap) = GeneratedFromAutoDiff(F.Map)
GeneratedFromSymbolic(F::Function) = occursin("SymbolicModel", string(nameof(typeof(F))))
GeneratedFromSymbolic(F::ModelMap) = GeneratedFromSymbolic(F.Map)


Base.summary(DS::AbstractDataSet) = string(TYPE_COLOR, nameof(typeof(DS)),
                                        NO_COLOR, (length(name(DS)) > 0 ? " '" *STRING_COLOR*string(name(DS)) *NO_COLOR*"'" : ""),
                                        " with N="*string(Npoints(DS))*", xdim=" * string(xdim(DS))*" and ydim="*string(ydim(DS)))

Base.summary(DS::AbstractUnknownUncertaintyDataSet) = string(TYPE_COLOR, nameof(typeof(DS)),
                                        ORANGE_COLOR, HasBessel(DS) ? " Bessel-corrected" : " not Bessel-corrected",
                                        NO_COLOR, (length(name(DS)) > 0 ? " '" *STRING_COLOR*string(name(DS)) *NO_COLOR*"'" : ""),
                                        " with N="*string(Npoints(DS))*", xdim=" * string(xdim(DS))*" and ydim="*string(ydim(DS)))

###### Useful info: Autodmodel? Symbolic? StaticArray output? In-place?
function Base.summary(DM::AbstractDataModel)
    # Also use "RuntimeGeneratedFunction" string from build_function in ModelingToolkit.jl
    string(TYPE_COLOR, nameof(typeof(DM)),
    NO_COLOR, (length(name(DM)) > 0 ? " '"*STRING_COLOR* string(name(DM)) *NO_COLOR*"'" : ""),
    " containing ",
    TYPE_COLOR, nameof(typeof(Data(DM))),
    NO_COLOR, (length(name(Data(DM))) > 0 ? " '"*STRING_COLOR* string(name(Data(DM)))*NO_COLOR*"'" : ""),
    ". Model jacobian: ",
    ORANGE_COLOR, (GeneratedFromAutoDiff(dPredictor(DM)) ? "AutoDiff" : (GeneratedFromSymbolic(dPredictor(DM)) ? "Symbolic" : "manually provided")),
    NO_COLOR)
end

function Base.summary(M::ModelMap)
    string(TYPE_COLOR, "ModelMap",
        NO_COLOR, (length(name(M)) > 0 ? " '"*STRING_COLOR*string(name(M))*NO_COLOR*"' " : " "),
        ORANGE_COLOR, (isinplacemodel(M) ? "in-place" : "out-of-place"),
        NO_COLOR, " with xdim="*string(xdim(M))*", ydim="*string(ydim(M))*", pdim="*string(pdim(M)))
end

Base.summary(P::ParameterProfiles) = string(TYPE_COLOR, "ParameterProfiles", NO_COLOR, " with pdim="* string(pdim(P)), !HasTrajectories(P) ? string(", ", ORANGE_COLOR, "no trajectories", NO_COLOR) : "")
Base.summary(PV::ParameterProfilesView) = string(TYPE_COLOR, "ParameterProfile", NO_COLOR, " for index "* string(PV.i) *"/"* string(pdim(PV)), !HasTrajectories(PV) ? string(", ", ORANGE_COLOR, "no trajectories", NO_COLOR) : "")

Base.summary(R::MultistartResults) = string(TYPE_COLOR, "MultistartResults", NO_COLOR, " with "* string(FindLastIndSafe(R)) *"/"* string(length(R)) *" successful starts")


# http://docs.junolab.org/stable/man/info_developer/#
# hastreeview, numberofnodes, treelabel, treenode
TreeViews.hastreeview(x::Union{AbstractDataSet,AbstractDataModel,ModelMap}) = true
TreeViews.numberofnodes(x::AbstractDataSet) = 4
TreeViews.numberofnodes(x::AbstractDataModel) = 4
TreeViews.numberofnodes(x::ModelMap) = 4
TreeViews.numberofnodes(x::CompositeDataSet) = 1
TreeViews.numberofnodes(x::GeneralizedDataSet) = 1
TreeViews.numberofnodes(x::DataSetUncertain) = 5


# function TreeViews.treelabel(io::IO, DS::Union{AbstractDataSet,AbstractDataModel,ModelMap}, mime::MIME"text/plain"=MIME"text/plain"())
#     show(io, mime, Text(Base.summary(DS)))
# end

TreeViews.treelabel(io::IO, DS::Union{AbstractDataSet,AbstractDataModel,ModelMap}, ::MIME"text/plain") = print(io, Base.summary(DS))


# To hide the treenode display, simply return missing:
# treenode(x::Foo, i::Int) = missing


function ParamSummary(DM::AbstractDataModel, mle::AbstractVector{<:Number}=MLE(DM); FisherFn::Function=FisherMetric(DM))
    IsLin = try IsLinearParameter(DM) catch; nothing end
    L, U = if Predictor(DM) isa ModelMap
        round.(Predictor(DM).Domain.L; sigdigits=2), round.(Predictor(DM).Domain.U; sigdigits=2)
    else
        fill(-Inf, pdim(DM)), fill(Inf, pdim(DM))
    end
    OnLowerBoundary = @. (mle-L) / (U-L) < 1/200
    OnUpperBoundary = @. (U-mle) / (U-L) < 1/200
    if !isnothing(IsLin) && any(IsLin)
        H = Highlighter((data,zeile,spalte) -> ((OnLowerBoundary[zeile] && spalte ∈ (2,3,4)) || (OnUpperBoundary[zeile] && spalte ∈ (2,4,5))); bold=true, foreground=:red)
        pretty_table([1:pdim(DM) pnames(DM) L MLEuncert(DM, mle, FisherFn(mle);verbose=false) U IsLin]; crop=:none, header=["i", "Parameter", "Lower Bound", "MLE", "Upper Bound", "Linear Dependence"], alignment=[:c, :l, :c, :c, :c, :c], highlighters=H)
    else
        H = Highlighter((data,zeile,spalte) -> ((OnLowerBoundary[zeile] && spalte ∈ (2,3,4)) || (OnUpperBoundary[zeile] && spalte ∈ (2,4,5))); bold=true, foreground=:red)
        pretty_table([1:pdim(DM) pnames(DM) L MLEuncert(DM, mle, FisherFn(mle);verbose=false) U]; crop=:none, header=["i", "Parameter", "Lower Bound", "MLE", "Upper Bound"], alignment=[:c, :l, :c, :c, :c], highlighters=H)
    end
end
function ParamSummary(io::IO, DM::AbstractDataModel, mle::AbstractVector{<:Number}=MLE(DM); FisherFn::Function=FisherMetric(DM))
    IsLin = try IsLinearParameter(DM) catch; nothing end
    L, U = if Predictor(DM) isa ModelMap
        round.(Predictor(DM).Domain.L; sigdigits=2), round.(Predictor(DM).Domain.U; sigdigits=2)
    else
        fill(-Inf, pdim(DM)), fill(Inf, pdim(DM))
    end
    OnLowerBoundary = @. (mle-L) / (U-L) < 1/200
    OnUpperBoundary = @. (U-mle) / (U-L) < 1/200
    if !isnothing(IsLin) && any(IsLin)
        H = Highlighter((data,zeile,spalte) -> ((OnLowerBoundary[zeile] && spalte ∈ (2,3,4)) || (OnUpperBoundary[zeile] && spalte ∈ (2,4,5))); bold=true, foreground=:red)
        pretty_table(io, [1:pdim(DM) pnames(DM) L MLEuncert(DM, mle, FisherFn(mle);verbose=false) U IsLin]; crop=:none, header=["i", "Parameter", "Lower Bound", "MLE", "Upper Bound", "Linear Dependence"], alignment=[:c, :l, :c, :c, :c, :c], highlighters=H)
    else
        H = Highlighter((data,zeile,spalte) -> ((OnLowerBoundary[zeile] && spalte ∈ (2,3,4)) || (OnUpperBoundary[zeile] && spalte ∈ (2,4,5))); bold=true, foreground=:red)
        pretty_table(io, [1:pdim(DM) pnames(DM) L MLEuncert(DM, mle, FisherFn(mle);verbose=false) U]; crop=:none, header=["i", "Parameter", "Lower Bound", "MLE", "Upper Bound"], alignment=[:c, :l, :c, :c, :c], highlighters=H)
    end
end


#### Need proper show() methods for DataSet, DataModel, ModelMap
#### Show Distribution types for DataSetExact
function Base.show(io::IO, DS::AbstractDataSet)
    println(io, Base.summary(DS) * ":")
    xnameinsert = any(xnames(DS) .!= CreateSymbolNames(xdim(DS),"x")) ? (" [" * join(xnames(DS), ", ") * "] ") : ""
    print(io, "x-data" * xnameinsert * ": ")
    show(io, xdata(DS));    print(io, "\n")
    if HasXerror(DS)
        xsig = xsigma(DS)
        println(io, (xsig isa AbstractVector ? "Standard deviation " : "Covariance matrix ") * "associated with x-data:")
        show(io, xsig)
        print(io, "\n")
    end
    ynameinsert = any(ynames(DS) .!= CreateSymbolNames(ydim(DS),"y")) ? (" [" * join(ynames(DS), ", ") * "] ") : ""
    print(io, "y-data" * ynameinsert * ": ")
    show(io, ydata(DS));    print(io, "\n")
    ysig = ysigma(DS)
    println(io, (ysig isa AbstractVector ? "Standard deviation " : "Covariance matrix ") * "associated with y-data:")
    show(io, ysig)
end

function Base.show(io::IO, GDS::GeneralizedDataSet)
    println(io, Base.summary(GDS) * ":")
    print(io, "Combined x-y data: ");    show(io, GetMean(dist(GDS)));    print(io, "\n")
    print(io, "Combined x-y covariance: ");    show(io, Sigma(dist(GDS)));    print(io, "\n")
end

# Multi-line display when used on its own in REPL
function Base.show(io::IO, ::MIME"text/plain", DM::AbstractDataModel)
    Expr = string(SymbolicModel(DM))
    LogPr = !isnothing(LogPrior(DM)) ? LogPrior(DM)(MLE(DM)) : nothing
    println(io, Base.summary(DM))
    println(io, "Maximal value of log-likelihood: "*string(round(LogLikeMLE(DM); sigdigits=5)))
    isnothing(LogPr) || println(io, "Log prior at MLE: "*string(round(LogPr; sigdigits=5)))
    Expr[1] == 'y' && println(io, "Model Expr:  " * Expr)
    try ParamSummary(io, DM) catch; end
end

# Single line display
function Base.show(io::IO, DM::AbstractDataModel)
    # Expr = SymbolicModel(DM)
    println(io, Base.summary(DM))
    print(io, "Maximal value of log-likelihood: "*string(round(LogLikeMLE(DM); sigdigits=5)))
    # Expr[1] == 'y' && println(io, "Model Expr:  $Expr")
end


# Multi-line display when used on its own in REPL
function Base.show(io::IO, ::MIME"text/plain", M::ModelMap)
    Expr = string(SymbolicModel(M))
    print(io, Base.summary(M))
    Expr[1] == 'y' && print(io, "\nModel Expr:  " * Expr)
    pnames(M) != CreateSymbolNames(pdim(M)) && print(io, "\nParameters: θ = [" * join(pnames(M), ", ") * "]")
end

# Single line display
function Base.show(io::IO, M::ModelMap)
    # Expr = SymbolicModel(M)
    print(io, Base.summary(M))
    # Expr[1] == 'y' && println(io, "Model Expr:  $Expr")
    pnames(M) != CreateSymbolNames(pdim(M)) && print(io, "\nParameters: θ = [" * join(pnames(M), ", ") * "]")
end



# Which parameters ran into boundary?
# Which indices have profiles if not all?
# Has no trajectories?
# Has priors?

# Multi-line display when used on its own in REPL
function Base.show(io::IO, ::MIME"text/plain", P::ParameterProfiles)
    HasProf = [i for i in eachindex(P) if HasProfiles(P[i])]
    print(io, Base.summary(P))
    1:pdim(P) != HasProf && print(io, "\nComputed Profiles: ", string(HasProf))
end

# Single line display
Base.show(io::IO, P::Union{ParameterProfiles,ParameterProfilesView}) = print(io, Base.summary(P))


# Multi-line display when used on its own in REPL
function Base.show(io::IO, ::MIME"text/plain", R::MultistartResults)
    LastFinite = FindLastIndSafe(R)
    FirstStepInd = GetFirstStepInd(R, LastFinite)
    println(io, Base.summary(R))
    println(io, "Median number of iterations: "*string(Int(round(median(@view R.Iterations[1:LastFinite])))))
    print(io, string(FirstStepInd) *"/"* string(LastFinite) * " ("*string(round(100*FirstStepInd/LastFinite; sigdigits=2))*" %) convergent fits had same final objective value")
    print(io, ": "*string(round(R.FinalObjectives[1]; sigdigits=5)))
end

# Single line display
Base.show(io::IO, R::MultistartResults) = print(io, Base.summary(R))