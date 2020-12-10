module InformationGeometry

using Distributed, LinearAlgebra, StaticArrays, SparseArrays
using OrdinaryDiffEq, DiffEqCallbacks, BoundaryValueDiffEq, ModelingToolkit
using ForwardDiff, BenchmarkTools, LsqFit, Random, Measurements, HCubature
using Distributions, SpecialFunctions, TensorOperations, DataFrames, Roots, Combinatorics
using RecipesBase, Plots, Optim


# General Todos:
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



include("DataStructures.jl")
export AbstractDataSet, AbstractDataModel, ModelOrFunction, DataSet, DataModel, Plane, HyperCube

# export HealthyData, HealthyCovariance, CheckModelHealth
export xdata, ydata, sigma, InvCov, Npoints, xdim, ydim, pdim, length, Data, MLE, LogLikeMLE, WoundX
export LinearModel, QuadraticModel
export DataDist, SortDataSet, SortDataModel, SubDataSet, SubDataModel, join, DataFrame
export MLEinPlane, PlanarDataModel, DetermineDmodel

# Planes
export BasisVector, PlaneCoordinates, Shift, IsOnPlane, TranslatePlane, RotatePlane, DecomposeWRTPlane
export DistanceToPlane, ProjectOntoPlane, IsNormalToPlane, MinimizeOnPlane, ParallelPlanes
# HyperCubes
export Inside, in, ConstructCube, CubeWidths, CubeVol, Center, TranslateCube, CoverCubes


include("DataSetExact.jl")
export Dirac, DataSetExact, Cov, Sigma, xSigma, ySigma, LogLike, xdist, ydist, xsigma, ysigma


include("ConfidenceRegions.jl")
export likelihood, loglikelihood, Score, WilksCriterion, WilksTest, OrthVF, FindMLE
export AutoScore, AutoMetric
export FindConfBoundary, FCriterion, FTest, FindFBoundary
export GenerateBoundary, ConfidenceRegion, ConfidenceRegions
export IsStructurallyIdentifiable, StructurallyIdentifiable
export FisherMetric, GeometricDensity, ConfidenceRegionVolume

export EmbeddingMap, EmbeddingMatrix, Pullback, Pushforward
export AIC, AICc, BIC, ModelComparison, IsLinearParameter, IsLinear, LeastInformativeDirection

export FindConfBoundaryOnPlane, LinearCuboid, IntersectCube, IntersectRegion, MincedBoundaries


include("NumericalTools.jl")
export GetH, suff, Unpack, Unwind, Windup, ToCols
export ConfAlpha, ConfVol, InvConfVol, ChisqCDF, InvChisqCDF
export Integrate1D, IntegrateND, LineSearch, MonteCarloArea
export curve_fit, BlockDiagonal, BlockMatrix


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
export GetModel, Optimize, EvaluateSol


include("Plotting.jl")
export FittedPlot, ResidualPlot, PlotScalar, PlotLoglikelihood, Plot2DVF
export Deplanarize, VisualizeSols, VisualizeGeos, VisualizeSolPoints
export ConfidenceBands, PlotMatrix
export EvaluateAlongGeodesic, PlotAlongGeodesic, EvaluateAlongCurve, PlotAlongCurve
# export RectangularFacetIndices, RectToTriangFacets, CreateMesh, ToObj, WriteObj


include("Exporting.jl")
export SaveAdaptive, SaveConfidence, SaveGeodesics, SaveDataSet
# export Homogenize, Dehomogenize


end # module
