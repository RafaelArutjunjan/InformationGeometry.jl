
module InformationGeometry

# using Reexport
# @reexport
using LinearAlgebra, Random, Distributions, DataFrames

using Distributed, StaticArrays, SparseArrays
using OrdinaryDiffEq, DiffEqCallbacks, BoundaryValueDiffEq
using ModelingToolkit, DataInterpolations
using ForwardDiff, BenchmarkTools, LsqFit, Measurements, HCubature
using SpecialFunctions, TensorOperations, DataFrames, Roots, Combinatorics
using LibGEOS
using RecipesBase, Plots, Optim
using TreeViews

import SciMLBase: AbstractODESolution


############## TODOs ##############

# How can kwargs be consistently added to FindConfBoundary(), FindMLE(), ... such that tols are not messed up?

#### Functionality

# Geodesic coordinates - use geodesic radius and angles to specify parameter configurations.
# General Parameter Space / Model Transformations: return new model where `log` has been applied to some / all θ compoents
# Allow for inplace models and dmodels -- potentially very large performance benefits
# Finish Boundaries() specification methods in confidence boundary and geodesic construction
# Use information contained in ModelMap type to build Boundaries function
# Add CompositeDataSet to master
# Test F-Boundaries in more detail
# Reactivate TryOptimize
# Integrate along structural unidentifiability -- Switching ODE functions mid-integration: https://diffeq.sciml.ai/stable/basics/faq/
# Symbolic Fisher? Christoffel (from 2nd derivatives)?
# Use OrdinaryDiffEq.build_solution() to translate planar 2D solution to ambient space as new solution object
# Use vector-callback with domain hypercube for continuous boundary detection

# Compute EigenFlow of Fisher Metric -> Should be transformed to straight lines (in coordinates) by Decorrelation Transformation
# Employ MultistartOptimization for obtaining the MLE.
# Try to recognize which components of θ are inside an exponential such that `log` leads to linear parametrization -> i.e. compare EmbeddingMatrices
# Add GeneralProductDistribution to master
# Implement Karcher mean

#### Cosmetic / Convenience

# Custom type for confidence boundaries?
# Use try catch; to recognize automatically when solution runs into chart boundary.
# Show distribution types for DataSetExact
# Infer variable names from DataFrames
# Save Symbolic Expressions for models (if available) and display them
# IO methods for ModelMaps in general
# Add measures of practical unidentifiability (eigenvalues of Fisher?)


#### Performance / Optimization

# Optimize GetModel() for ODEs for performance by only executing ODEFunction once
# Use IntegrateND() instead of MonteCarloArea()
# Implement formula to obtain bounding box of hyperellipse exactly.
# Remove Plots.jl dependency and instead write recipes for VisualizeSols and others
# Change EvaluateAlongGeodesicLength using affine reparametrization to save computations
# Use Cuba.jl for Monte Carlo integration
# Generalize FisherMetric to other error distributions
# Redo Geodesic methods


#### Tests to add

# GenerateInterruptedBoundary
# F-Boundary
# Constructors of ModelMap, DataSet, DataModel, etc
# ..... make full list


#### Miscellaneous

# Get Travis working again
# Redo README.md
# Add exporting methods and detailed examples to docs
# Improve ConfidenceBands
# Fix FindConfBoundary() and GenerateBoundary() for BigFloat    -- does it work now? Add a test
# Has PlotUtils.jl released a version with adapted_grid() yet? Restrict compat to that version and delete duplicated source code
# Make user-facing keywords (even) more uniform: tol, meth, Big, ...


abstract type AbstractDataSet end
abstract type AbstractDataModel end
abstract type Cuboid end

export AbstractDataSet, AbstractDataModel, Cuboid



import Base: length, rand, BigFloat, in, union, intersect, join, ==
import DataFrames.DataFrame
import Distributions.loglikelihood


include("Subspaces.jl")
export Plane
export BasisVector, PlaneCoordinates, Shift, IsOnPlane, TranslatePlane, RotatePlane, DecomposeWRTPlane
export DistanceToPlane, ProjectOntoPlane, IsNormalToPlane, MinimizeOnPlane, ParallelPlanes
export HyperPlane, HyperCube
export Inside, in, ConstructCube, CubeWidths, CubeVol, Center, TranslateCube, ResizeCube
export DropCubeDim, DropCubeDims, FaceCenters, Intersect, Union
export PositiveDomain, NegativeDomain, FullDomain, rand, EmbeddedODESolution


include("ModelMaps.jl")
export ModelMap, ModelOrFunction, InformNames
export Transform, LogTransform, TranslationTransform, LinearTransform, AffineTransform
export LinearModel, QuadraticModel, ExponentialModel, SumExponentialsModel, PolynomialModel


include("GeneralDataStructures.jl")
# export HealthyData, HealthyCovariance, CheckModelHealth
export xdata, ydata, sigma, InvCov, Npoints, xdim, ydim, pdim, DataspaceDim, Data, MLE, LogLikeMLE, WoundX
export Predictor, dPredictor, ConsistentElDims
export DataDist, SortDataSet, SortDataModel, SubDataSet, SubDataModel, DataFrame, join, length
export MLEinPlane, PlanarDataModel, DetermineDmodel


include("DataStructures/DataSet.jl")
export DataSet


include("DataStructures/DataModel.jl")
export DataModel


include("DataStructures/DataSetExact.jl")
export DataSetExact, Dirac
export Cov, LogLike, xdist, ydist, xsigma, ysigma


include("DataStructures/CompositeDataSet.jl")
export CompositeDataSet


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

export EmbeddingMap, EmbeddingMatrix, Pullback, Pushforward
export AIC, AICc, BIC, ModelComparison, IsLinearParameter, IsLinear, LeastInformativeDirection

export FindConfBoundaryOnPlane, LinearCuboid, IntersectCube, IntersectRegion, MincedBoundaries, ConfidenceBoundary

export ApproxInRegion, ShadowTheatre



include("ProfileLikelihood.jl")
export EmbedModelVia, ProfileLikelihood, InterpolatedProfiles, ProfileBox


include("NumericalTools.jl")
export GetH, suff, Unpack, Unwind, Windup, ToCols, PromoteStatic, SplitAfter
export ConfAlpha, ConfVol, InvConfVol, ChisqCDF, InvChisqCDF
export Integrate1D, IntegrateND, IntegrateOverConfidenceRegion, IntegrateOverApproxConfidenceRegion
export LineSearch, MonteCarloArea
export curve_fit, TotalLeastSquares, BlockMatrix


include("Divergences.jl")
export KullbackLeibler


include("Geodesics.jl")
export ComputeGeodesic, GeodesicLength, GeodesicCrossing, DistanceAlongGeodesic, Endpoints, EvaluateEach
export RadialGeodesics, GeodesicBetween, GeodesicDistance, GeodesicEnergy, MBAM
# Needs redo:
# export ConstLengthGeodesics, ConstParamGeodesics, ConfidenceBoundaryViaGeodesic


include("Curvature.jl")
export ChristoffelSymbol, ChristoffelTerm, Riemann, Ricci, RicciScalar
# export Weyl
# Also add Kretschmann, Schouten?


include("SymbolicComputations.jl")
export GetModel, Optimize, OptimizedDM, PrintModel, PrintdModel


include("Plotting.jl")
export FittedPlot, ResidualPlot, PlotScalar, PlotLoglikelihood, Plot2DVF, ResidualSquaredError
export Deplanarize, VisualizeSols, VisualizeGeos, VisualizeSolPoints, ConstructAmbientSolution
export ConfidenceBands, ApproxConfidenceBands, PlotConfidenceBands, ConfidenceBandWidth, PlotMatrix
export EvaluateAlongGeodesic, PlotAlongGeodesic, EvaluateAlongCurve, PlotAlongCurve
# export RectangularFacetIndices, RectToTriangFacets, CreateMesh, ToObj, WriteObj


include("Exporting.jl")
export SaveAdaptive, SaveConfidence, SaveGeodesics, SaveDataSet
# export Homogenize, Dehomogenize


include("CustomIO.jl")
# export GeneratedFromAutoDiff, GeneratedFromSymbolic


end # module
