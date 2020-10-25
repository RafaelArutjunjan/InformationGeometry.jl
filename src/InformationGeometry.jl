module InformationGeometry

using Distributed, LinearAlgebra, StaticArrays, SparseArrays
using OrdinaryDiffEq, DiffEqCallbacks, BoundaryValueDiffEq
using ForwardDiff, BenchmarkTools, Optim, LsqFit, Random, Measurements
using Distributions, SpecialFunctions, TensorOperations, DataFrames
using RecipesBase, Plots


# Make interface more uniform: tolerance, method, Big, ...
# Generalize accepted models (covariance, non-gaussian Fisher metrics,...)
# Improve Plotting for higher-dim models

# ALLOW FOR SPECIFICATION OF A DOMAIN IN DATAMODEL?
# WOULD BE BEST TO MAKE THIS A BOOLEAN-VALUED FUNCTION WHICH CAN ALSO BE INDUCED BY A HYPERCUBE.

# Remove Plots.jl dependency and instead write recipes for FittedPlot and ResidualPlot

# SaveDataSet is basically equivalent to DataFrame(DS)?
# Create method which boils DataSet down to, say length(θ)+1 points for visualization of h(M) ⊆ D

# Add tests for model functions: test on single and multiple datapoints

# Add descriptions of how to save stuff to docs


include("Structures.jl")
# Types
export AbstractDataSet, AbstractDataModel, DataSet, DataModel, Plane, HyperCube, LowerUpper


# Functions
export plot, BasisVector, RotatePlane, RotatedVector, TranslatePlane, IsOnPlane, DecomposeWRTPlane, DistanceToPlane, normalizeVF
export CubeWidths, CubeVol, Center, TranslateCube, CoverCubes, ConstructCube, PlaneCoordinates, MinimizeOnPlane, ProjectOntoPlane, ParallelPlanes
export xdata, ydata, sigma, InvCov, N, xdim, ydim, pdim, MLE, LogLikeMLE, PlanarDataModel, MLEinPlane
export length, DataFrame, join, SortDataSet, SortDataModel, SubDataSet, SubDataModel, Unpack, ToCols, Unwind, Windup
export HealthyData, HealthyCovariance, DataDist, BlockDiagonal, IsNormalToPlane


include("ConfidenceLib.jl")
export likelihood, loglikelihood, ConfVol, ConfAlpha, InvConfVol, ChisqCDF, InvChisqCDF, Score, FisherMetric, Rsquared, AIC, AICc, BIC, ModelComparison
export WilksTest, FindConfBoundary, OrthVF, FindMLE, FindMLEBig, GenerateBoundary, Inside, KullbackLeibler, NormalDist, LineSearch
export EmbeddingMap, EmbeddingMatrix, Pullback, StructurallyIdentifiable, Integrate1D
export MultipleConfidenceRegions, GenerateConfidenceRegion, Interval1D, LeastInformativeDirection
export IsLinearParameter, IsLinear


# Change EvaluateAlongGeodesicLength using affine reparametrization to save computation
include("Geodesics.jl")

export Cross, ChristoffelTerm, ChristoffelSymbol, ComputeGeodesic, GeodesicLength, GeodesicCrossing
export DistanceAlongGeodesic, Endpoints, Truncated, ConstLengthGeodesics, ConfidenceBoundaryViaGeodesic
export ConstParamGeodesics, GeodesicBetween, GeodesicDistance
export PlotCurves, EvaluateEach, EvaluateAlongGeodesic, PlotAlongGeodesic
export EvaluateAlongGeodesicLength, PlotAlongGeodesicLength, EvaluateAlongCurve, PlotAlongCurve, SaveConfidence, SaveGeodesics, SaveAdaptive
export Riemann, Ricci, RicciScalar, GeometricDensity
export AutoScore, AutoMetric


include("Plotting.jl")
export curve_fit, FittedPlot, ResidualPlot, PlotLoglikelihood, PlotScalar, VisualizeSol, VisualizeSols, PlotMatrix
export VisualizeGeos, VisualizeGeo, PointwiseConfidenceBand, Deplanarize
export suff, ConfidenceRegionVolume, SaveDataSet


include("datasettypes.jl")
export Dirac, DataSetExact, Cov, Sigma, xSigma, ySigma, LogLike


end # module
