
module InformationGeometry

# using Reexport
# @reexport
using LinearAlgebra, Random, Distributions, DataFrames

using Distributed, StaticArrays, SparseArrays
using OrdinaryDiffEq, DiffEqCallbacks, BoundaryValueDiffEq
using ModelingToolkit, Symbolics, DataInterpolations
using ForwardDiff, DerivableFunctions
using BenchmarkTools, LsqFit, Optim, Measurements, HCubature
using SpecialFunctions, Tullio, DataFrames, Roots, Combinatorics
using LibGEOS, Sobol, ProgressMeter, Suppressor
using RecipesBase, Plots
using TreeViews

import DataInterpolations: AbstractInterpolation
import SciMLBase: AbstractODESolution, AbstractODEFunction

import SciMLBase: remake
export remake

import DerivableFunctions: suff, MaximalNumberOfArguments, KillAfter, GetArgLength, Builder
import DerivableFunctions: DerivableFunction, DFunction, EvalF, EvaldF, EvalddF
import DerivableFunctions: GetDeriv, GetGrad, GetJac, GetHess, GetMatrixJac
import DerivableFunctions: GetGrad!, GetJac!, GetHess!, GetMatrixJac!, diff_backends

# include("NumericalTools/Differentiation.jl")
export diff_backends
export GetDeriv, GetGrad, GetJac, GetMatrixJac, GetHess, GetDoubleJac
export GetGrad!, GetHess!, GetJac!, GetMatrixJac!


############## TODOs ##############

# How can kwargs be consistently added to FindConfBoundary(), FindMLE(), ... such that tols are not messed up?

#### Functionality

# Employ ShadowTheatre to project simultaneous x-θ-confidence regions to M.
# Geodesic coordinates - use geodesic radius and angles to specify parameter configurations.
# ADD TESTS: General Parameter Space / Model Transformations: return new model where `log` has been applied to some / all θ compoents
# Allow for inplace models and dmodels -- potentially very large performance benefits
# ADD TESTS: Add CompositeDataSet to master
# Test F-Boundaries in more detail
# Integrate along structural unidentifiability -- Switching ODE functions mid-integration: https://diffeq.sciml.ai/stable/basics/faq/
# Use InDomain function specified for ModelMaps for constrained optimization.
# Symbolic Fisher? Christoffel (from 2nd derivatives)?
# Use vector-callback with domain hypercube for continuous boundary detection

# Compute EigenFlow of Fisher Metric -> Should be transformed to straight lines (in coordinates) by Decorrelation Transformation
# Employ MultistartOptimization for obtaining the MLE.
# Try to recognize which components of θ are inside an exponential such that `log` leads to linear parametrization -> i.e. compare EmbeddingMatrices

#### Cosmetic / Convenience

# Custom type for confidence boundaries?
# Show distribution types for DataSetExact
# Infer variable names from DataFrames
# IO methods for ModelMaps in general
# Add measures of practical unidentifiability (eigenvalues of Fisher?)


#### Performance / Optimization

# Make OrthVF in-place everywhere
# Optimize GetModel() for ODEs for performance by only executing ODEFunction once
# Use IntegrateND() instead of MonteCarloArea()
# Remove Plots.jl dependency and instead write recipes for VisualizeSols and others
# Change EvaluateAlongGeodesicLength using affine reparametrization to save computations
# Use Cuba.jl for Monte Carlo integration
# Generalize FisherMetric to other error distributions


#### Tests to add

# GenerateInterruptedBoundary
# F-Boundary
# Constructors of ModelMap, DataSet, DataModel, etc
# ..... make full list


#### Miscellaneous

# Add exporting methods and detailed examples to docs
# Improve ConfidenceBands
# Fix FindConfBoundary() and GenerateBoundary() for BigFloat    -- does it work now? Add a test
# Make user-facing keywords (even) more uniform: tol, meth, Big, ...


abstract type AbstractDataSet end
abstract type AbstractDataModel end
abstract type Cuboid end
export AbstractDataSet, AbstractDataModel, Cuboid



import Base: length, rand, BigFloat, in, union, intersect, join, ==
import DataFrames.DataFrame
import Distributions.loglikelihood

BoolArray = Union{BitArray,AbstractArray{<:Bool}}
BoolVector = Union{BitVector,AbstractVector{<:Bool}}


include("Subspaces.jl")
export Plane
export BasisVector, PlaneCoordinates, Shift, IsOnPlane, TranslatePlane, RotatePlane, DecomposeWRTPlane
export DistanceToPlane, ProjectOntoPlane, IsNormalToPlane, MinimizeOnPlane, ParallelPlanes
export HyperPlane, HyperCube
export Inside, in, ConstructCube, CubeWidths, CubeVol, Center, TranslateCube, ResizeCube
export DropCubeDims, FaceCenters, Corners
export PositiveDomain, NegativeDomain, FullDomain, rand, EmbeddedODESolution


include("ModelMaps.jl")
export ModelMap, ModelOrFunction, InformNames, IsInDomain
export Transform, LogTransform, Log10Transform, ExpTransform, Power10Transform, ScaleTransform
export TranslationTransform, LinearTransform, AffineTransform, LinearDecorrelation
export EmbedModelVia, Embedding
export LinearModel, QuadraticModel, ExponentialModel, SumExponentialsModel, PolynomialModel


include("DataStructures/DistributionTypes.jl")
export Dirac, GeneralProduct


include("GeneralDataStructures.jl")
# export HealthyData, HealthyCovariance, CheckModelHealth
export xdata, ydata, xsigma, ysigma, xInvCov, yInvCov, Npoints, xdim, ydim, pdim, DataspaceDim, Data, MLE, LogLikeMLE, WoundX
export xdist, ydist, dist
export Predictor, dPredictor, LogPrior, ConsistentElDims
export MeasureAutoDiffPerformance
export DataDist, SortDataSet, SortDataModel, SubDataSet, SubDataModel, DataFrame, join, length
export MLEinPlane, PlanarDataModel, DetermineDmodel


include("DataStructures/DataSet.jl")
export DataSet


include("DataStructures/DataSetExact.jl")
export DataSetExact, LogLike


include("DataStructures/CompositeDataSet.jl")
export CompositeDataSet


include("DataStructures/GeneralizedDataSet.jl")
export GeneralizedDataSet


include("DataStructures/DataModel.jl")
export DataModel, Prior


include("NumericalTools/Miscellaneous.jl")

include("NumericalTools/Integration.jl")
include("NumericalTools/Optimization.jl")
export GetH, suff, Unpack, Unwind, Windup, ToCols, PromoteStatic, SplitAfter
export ConfAlpha, ConfVol, InvConfVol, ChisqCDF, InvChisqCDF
export KillAfter
export Integrate1D, IntegrateND, IntegrateOverConfidenceRegion, IntegrateOverApproxConfidenceRegion
export LineSearch, MonteCarloArea
export curve_fit, RobustFit, TotalLeastSquares, BlockMatrix


include("ConfidenceRegions.jl")
export likelihood, loglikelihood, Score, WilksCriterion, WilksTest, OrthVF, FindMLE
export AutoScore, AutoMetric
export FindConfBoundary, FCriterion, FTest, FindFBoundary
export GenerateBoundary, ConfidenceRegion, ConfidenceRegions
export GenerateInterruptedBoundary, InterruptedConfidenceRegion
export IsStructurallyIdentifiable, StructurallyIdentifiable
export FisherMetric, GeometricDensity
export ConfidenceRegionVolume, CoordinateVolume
export ExpectedInvariantVolume, GeodesicRadius, CoordinateDistortion, Sensitivity

export EmbeddingMap, EmbeddingMatrix, EmbeddingMap!, EmbeddingMatrix!, Pullback, Pushforward
export AIC, AICc, BIC, ModelComparison, IsLinearParameter, IsLinear, LeastInformativeDirection

export FindConfBoundaryOnPlane, LinearCuboid, IntersectCube, IntersectRegion, MincedBoundaries, ConfidenceBoundary

export ApproxInRegion, ShadowTheatre, CastShadow



include("ProfileLikelihood.jl")
export ProfileLikelihood, PlotProfileTrajectories, InterpolatedProfiles, ProfileBox, PracticallyIdentifiable
export ValInserter, InsertIntoFirst, InsertIntoLast

include("Divergences.jl")
export KullbackLeibler


include("Geodesics.jl")
export ComputeGeodesic, GeodesicLength, GeodesicCrossing, DistanceAlongGeodesic, Endpoints, EvaluateEach
export RadialGeodesics, BoundaryViaGeodesic, GeodesicBetween, GeodesicDistance, GeodesicEnergy, ExponentialMap, LogarithmicMap, KarcherMean, MBAM
# Needs redo:
# export ConstLengthGeodesics, ConstParamGeodesics


include("Curvature.jl")
export ChristoffelSymbol, Riemann, Ricci, RicciScalar
# export Weyl
# Also add Kretschmann, Schouten?


include("DiffEqModels.jl")
export GetModel, ModifyODEmodel


include("SymbolicComputations.jl")
export Optimize, OptimizedDM, InplaceDM, SymbolicModel, SymbolicdModel


include("Plotting.jl")
export FittedPlot, ResidualPlot, PlotScalar, PlotLoglikelihood, Plot2DVF, ResidualSquaredError, PlotEllipses
export Deplanarize, VisualizeSols, VisualizeGeos, VisualizeSolPoints, ConstructAmbientSolution
export ConfidenceBands, ApproxConfidenceBands, PlotConfidenceBands, ConfidenceBandWidth, PredictionEnsemble, PlotMatrix
export EvaluateAlongGeodesic, PlotAlongGeodesic, EvaluateAlongCurve, PlotAlongCurve, PhaseSpacePlot
# export RectangularFacetIndices, RectToTriangFacets, CreateMesh, ToObj, WriteObj


include("Exporting.jl")
export SaveAdaptive, SaveConfidence, SaveGeodesics, SaveDataSet
# export Homogenize, Dehomogenize


include("CustomIO.jl")
# export GeneratedFromAutoDiff, GeneratedFromSymbolic


end # module
