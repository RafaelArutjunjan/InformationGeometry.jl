module InformationGeometry

# using Reexport
# @reexport
using LinearAlgebra, Random, Distributions, DataFrames

using Distributed, StaticArrays, SparseArrays
using OrdinaryDiffEq, DiffEqCallbacks, BoundaryValueDiffEq, ModelingToolkit
using ForwardDiff, BenchmarkTools, LsqFit, Measurements, HCubature
using SpecialFunctions, TensorOperations, DataFrames, Roots, Combinatorics
using RecipesBase, Plots, Optim
using TreeViews


######### General Todos:
# General Parameter Space Transformations: return new model where `log` has been applied to some / all θ compoents
# Try to recognize which components of θ are inside an exponential such that `log` leads to linear parametrization -> i.e. compare EmbeddingMatrices
# Fix FindConfBoundary() for Confnum > 8, i.e. BigFloat
# Employ MultistartOptimization for obtaining the MLE.
# Fix GenerateBoundary for MLE and LogLikeMLE of type BigFloat.
# Extend GenerateBoundary() to employ a Boundaries function. -> Allow for specification
# Use information contained in ModelMap type to build Boundaries function
# Implement formula to obtain bounding box of hyperellipse exactly.
# Use try catch; to recognize automatically when solution runs into chart boundary.
# Use IntegrateND() instead of MonteCarloArea()
# Improve ConfidenceBands
# Compute EigenFlow of Fisher Metric -> Should be transformed to straight lines (in coordinates) by Decorrelation Transformation
# Implement ODR for estimating maximum likelihood with x-errors.
# Normal distributions around data
# Use ModelingToolkit for dmodel, preferably inplace
# Allow for inplace models
# Create method which integrates regions up to structural unidentifiability and then backwards for half-open confidence regions
# Make user-facing keywords more uniform: tol, meth, Big, ...
# Allow for specification of a domain for model function in DataModel
# Add descriptions of how to save stuff to docs
# Remove Plots.jl dependency and instead write recipes for FittedPlot and ResidualPlot
# Change EvaluateAlongGeodesicLength using affine reparametrization to save computations
# Use Cuba.jl for Monte Carlo integration
# Redo F-test and ConstParamGeodesics
# Generalize FisherMetric to other error distributions
# Custom data type for confidence boundaries with suitable IO functions



include("DataStructures.jl")
export AbstractDataSet, AbstractDataModel, DataSet, DataModel
export ModelMap, ModelOrFunction
export Plane, HyperCube, Cuboid

# export HealthyData, HealthyCovariance, CheckModelHealth
export xdata, ydata, sigma, InvCov, Npoints, xdim, ydim, pdim, length, Data, MLE, LogLikeMLE, WoundX
export Predictor, dPredictor, LinearModel, QuadraticModel, ExponentialModel, SumExponentialsModel
export DataDist, SortDataSet, SortDataModel, SubDataSet, SubDataModel, join, DataFrame
export MLEinPlane, PlanarDataModel, DetermineDmodel, Transform

# Planes
export BasisVector, PlaneCoordinates, Shift, IsOnPlane, TranslatePlane, RotatePlane, DecomposeWRTPlane
export DistanceToPlane, ProjectOntoPlane, IsNormalToPlane, MinimizeOnPlane, ParallelPlanes
# HyperCubes
export Inside, in, ConstructCube, CubeWidths, CubeVol, Center, TranslateCube, Intersect, Union, PositiveDomain, FullDomain, rand


include("DataSetExact.jl")
export Dirac, DataSetExact, Cov, Sigma, xSigma, ySigma, LogLike, xdist, ydist, xsigma, ysigma


include("ConfidenceRegions.jl")
export likelihood, loglikelihood, Score, WilksCriterion, WilksTest, OrthVF, FindMLE
export AutoScore, AutoMetric
export FindConfBoundary, FCriterion, FTest, FindFBoundary
export GenerateBoundary, GenerateInterruptedBoundary, ConfidenceRegion, ConfidenceRegions
export IsStructurallyIdentifiable, StructurallyIdentifiable
export FisherMetric, GeometricDensity, ConfidenceRegionVolume

export EmbeddingMap, EmbeddingMatrix, Pullback, Pushforward
export AIC, AICc, BIC, ModelComparison, IsLinearParameter, IsLinear, LeastInformativeDirection

export FindConfBoundaryOnPlane, LinearCuboid, IntersectCube, IntersectRegion, MincedBoundaries


include("NumericalTools.jl")
export GetH, suff, Unpack, Unwind, Windup, ToCols
export ConfAlpha, ConfVol, InvConfVol, ChisqCDF, InvChisqCDF
export Integrate1D, IntegrateND, LineSearch, MonteCarloArea
export curve_fit, BlockMatrix


include("InformationDivergences.jl")
export KullbackLeibler


include("Geodesics.jl")
export ComputeGeodesic, GeodesicLength, GeodesicCrossing, DistanceAlongGeodesic, Endpoints, EvaluateEach
export GeodesicBetween, GeodesicDistance, GeodesicEnergy, MBAM
# Needs redo:
# export ConstLengthGeodesics, ConstParamGeodesics, ConfidenceBoundaryViaGeodesic


include("Curvature.jl")
export ChristoffelSymbol, ChristoffelTerm, Riemann, Ricci, RicciScalar
# export Weyl
# Also add Kretschmann, Schouten?


include("SymbolicComputations.jl")
export GetModel, GetDModel, Optimize, OptimizedDM, EvaluateSol


include("Plotting.jl")
export FittedPlot, ResidualPlot, PlotScalar, PlotLoglikelihood, Plot2DVF
export Deplanarize, VisualizeSols, VisualizeGeos, VisualizeSolPoints
export ConfidenceBands, PlotMatrix
export EvaluateAlongGeodesic, PlotAlongGeodesic, EvaluateAlongCurve, PlotAlongCurve
# export RectangularFacetIndices, RectToTriangFacets, CreateMesh, ToObj, WriteObj


include("Exporting.jl")
export SaveAdaptive, SaveConfidence, SaveGeodesics, SaveDataSet
# export Homogenize, Dehomogenize



####### IO stuff

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
    NO_COLOR, ". Model jacobian via AutoDiff: ",
    ORANGE_COLOR, (GeneratedFromAutoDiff(dPredictor(DM)) ? "true" : "false"),
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
            println(io, "Covariance Matrix associated with x-data:")
            show(io, mime, xsigma(DS))
        end
        print(io, "\n")
    end
    print(io, "y-data: ");    show(io, mime, ydata(DS));    print(io, "\n")
    if typeof(ysigma(DS)) <: AbstractVector
        println(io, "Standard deviation associated with y-data:")
        show(io, mime, ysigma(DS))
    else
        println(io, "Covariance Matrix associated with y-data:")
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
            println(io, "Covariance Matrix associated with x-data:")
            show(io, xsigma(DS))
        end
        print(io, "\n")
    end
    print(io, "y-data: ");    show(io, ydata(DS));    print(io, "\n")
    if typeof(ysigma(DS)) <: AbstractVector
        println(io, "Standard deviation associated with y-data:")
        show(io, ysigma(DS))
    else
        println(io, "Covariance Matrix associated with y-data:")
        show(io, ysigma(DS))
    end
end

##### StaticOutput?
function Base.show(io::IO, mime::MIME"text/plain", DM::AbstractDataModel)
    auto = GeneratedFromAutoDiff(Predictor(DM))
    println(io, "$(nameof(typeof(DM))) containing a $(nameof(typeof(Data(DM))))")
    println(io, "Model jacobian ", auto ? "obtained via automatic differentiation" : "symbolically provided")
    if typeof(DM) == DataModel
        println(io, "Maximum Likelihood Estimate: $(MLE(DM))")
        println(io, "Maximal value of log-likelihood: $(LogLikeMLE(DM))")
    end
    println(io, "Model parametrization linear in n-th parameter: $(IsLinearParameter(DM))")
end

end # module
