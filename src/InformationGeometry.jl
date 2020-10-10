module InformationGeometry

using Distributed, LinearAlgebra
using OrdinaryDiffEq, DiffEqCallbacks, BoundaryValueDiffEq
using ForwardDiff, BenchmarkTools, Optim, LsqFit, Random, Measurements
using Distributions, SpecialFunctions, TensorOperations, DataFrames
using Plots

# todo: Add testing
# Remove old unused code
# Make interface more uniform: tolerance, method, Big, ...
# Write documentation
# Generalize accepted models (covariance, non-gaussian Fisher metrics,...)
# Improve Plotting for higher-dim models

# ALLOW FOR SPECIFICATION OF A DOMAIN IN DATAMODEL?
# WOULD BE BEST TO MAKE THIS A BOOLEAN-VALUED FUNCTION WHICH CAN ALSO BE INDUCED BY A HYPERCUBE.
# THEN TEST FOR det(g) ≠ 0 FOR ~10 RANDOM PARAMETER CONFIGURATIONS IN DOMAIN ON INITIALISATION.
# ALSO, TEST FOR det(g) ≠ 0 ALONG CONFIDENCE BOUNDARY AND PRINT IF THIS HAPPENS BUT CONTINUE CURVE.

# Immediately determine MLE and LogLikeMLE when building DataModel
# Remove Plots.jl dependency and instead write recipes for FittedPlot and ResidualPlot

# SaveDataSet is basically equivalent to DataFrame(DS)?
# Create method which boils DataSet down to, say length(θ)+1 points for visualization of h(M) ⊆ D

# ADD TESTS FOR CorrectedCovariance


include("Structures.jl")
# Types
export DataSet, DataModel, Plane, HyperCube, LowerUpper


# Functions
export Unpack, ToCols, plot, BasisVector, RotatePlane, TranslatePlane, IsOnPlane, DecomposeWRTPlane, DistanceToPlane, normalizeVF
export CubeWidths, CubeVol, TranslateCube, CoverCubes, ConstructCube, PlaneCoordinates, MinimizeOnPlane, ProjectOntoPlane
export xdata, ydata, sigma, xdim, ydim, pdim, MLE, LogLikeMLE, PlanarDataModel
export length, DataFrame, join, SortDataSet, SortDataModel, SubDataSet, SubDataModel


include("ConfidenceLib.jl")
export likelihood, loglikelihood, ConfVol, ConfAlpha, InvConfVol, ChisqCDF, InvChisqCDF, Score, FisherMetric, Rsquared, AIC
export WilksTest, WilksTestPrepared, FindConfBoundary, OrthVF, FindMLE, FindMLEBig, GenerateBoundary, Inside, KullbackLeibler, NormalDist, LineSearch
export EmbeddingMap, EmbeddingMatrix, Pullback, StructurallyIdentifiable, Integrate1D
export MultipleConfidenceRegions, GenerateConfidenceRegion
export GenerateMultipleIntervals, GenerateConfidenceInterval, Interval1D
export IsLinearParameter, IsLinear, CorrectedCovariance


# Change EvaluateAlongGeodesicLength using affine reparametrization to save computation
include("Geodesics.jl")

export Cross, ChristoffelTerm, ChristoffelSymbol, ComputeGeodesic, MetricNorm, GeodesicLength, GeodesicCrossing
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

end # module
