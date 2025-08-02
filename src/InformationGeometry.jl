
module InformationGeometry

# using Reexport
# @reexport
using LinearAlgebra, Random, Distributions, DistributionsAD, DataFrames, ComponentArrays

using Distributed, StaticArrays, ForwardDiff, PreallocationTools, RecursiveArrayTools
using OrdinaryDiffEqCore, OrdinaryDiffEqVerner, OrdinaryDiffEqTsit5, OrdinaryDiffEqLowOrderRK, OrdinaryDiffEqRosenbrock
using DiffEqCallbacks
using Symbolics, DataInterpolations
using DerivableFunctionsBase
using LsqFit, Optim, LineSearches, ADTypes, Optimization
using BenchmarkTools, Measurements, HCubature
using SpecialFunctions, Tullio, Roots, Combinatorics
using LibGEOS, ProgressMeter, Suppressor
using RecipesBase, Requires, PlotUtils
using TreeViews, PrettyTables

import Sobol as SOBOL


import DataInterpolations: AbstractInterpolation
import SciMLBase: AbstractODESolution, AbstractODEFunction, AbstractODEAlgorithm, AbstractADType

import SciMLBase: remake
export remake


import DerivableFunctionsBase: suff, MaximalNumberOfArguments, KillAfter, GetArgLength, Builder
import DerivableFunctionsBase: DerivableFunction, DFunction, EvalF, EvaldF, EvalddF
import DerivableFunctionsBase: GetDeriv, GetGrad, GetJac, GetHess, GetMatrixJac
import DerivableFunctionsBase: GetGrad!, GetJac!, GetHess!, GetMatrixJac!, diff_backends

# include("NumericalTools/Differentiation.jl")
export diff_backends
export GetDeriv, GetGrad, GetJac, GetMatrixJac, GetHess, GetDoubleJac
export GetGrad!, GetHess!, GetJac!, GetMatrixJac!


############## TODOs ##############


# Make GetModelRobust (vs. GetModelFast) default
# Give a warning message for GetModelFast that time gradients are not possible



#### Functionality

# More in-place overloads
# Use InDomain function specified for ModelMaps for constrained optimization.
# Employ ShadowTheatre to project simultaneous x-θ-confidence regions to M.
# Geodesic coordinates - use geodesic radius and angles to specify parameter configurations.
# Allow for inplace models and dmodels -- potentially very large performance benefits
# Integrate along structural unidentifiability -- Switching ODE functions mid-integration: https://diffeq.sciml.ai/stable/basics/faq/
# Symbolic Fisher? Christoffel (from 2nd derivatives)?
# Use vector-callback with domain hypercube for continuous boundary detection

# Employ MultistartOptimization for obtaining the MLE.
# Propose transformations which make parametrization "less non-linear", i.e. try to recognize if components of θ are inside exp or log via EmbeddingMatrices


#### Needs Tests

# CompositeDataSet
# Parameter Space and X-Y data transformations
# F-Boundaries
# GenerateInterruptedBoundary
# ExponentialMap and LogarithmicMap


#### Cosmetic / Convenience

# Custom type for confidence boundaries?
# Custom type for confidence Profiles?
# Show distribution types for DataSetExact
# Show dataset and model names
# IO methods for ModelMaps in general
# Rewrite Plot methods, particularly for datasets


#### Performance / Optimization

# Make OrthVF in-place everywhere
# Optimize GetModel() for ODEs for performance by only executing ODEFunction once
# Use IntegrateND() instead of MonteCarloArea()
# Change EvaluateAlongGeodesicLength using affine reparametrization to save computations
# Use Cuba.jl for Monte Carlo integration
# Generalize FisherMetric to other error distributions


#### Miscellaneous

# Add data transforms to docs
# Add parameter transform docstrings
# Add exporting methods and detailed examples to docs
# Fix FindConfBoundary() and GenerateBoundary() for BigFloat
# Make user-facing keywords (even) more uniform: tol, meth, Big, OptimMeth...


abstract type AbstractDataSet end
abstract type AbstractDataModel end
abstract type AbstractConditionGrid <: AbstractDataModel end
abstract type Cuboid end
abstract type AbstractProfiles end
abstract type AbstractMultistartResults end
export AbstractDataSet, AbstractDataModel, Cuboid

abstract type AbstractUnknownUncertaintyDataSet <: AbstractDataSet end
abstract type AbstractFixedUncertaintyDataSet <: AbstractDataSet end


import Base: length, rand, BigFloat, in, union, intersect, join, ==
import DataFrames.DataFrame
import Distributions.loglikelihood

const BoolArray = Union{BitArray,AbstractArray{<:Bool}}
const BoolVector = Union{BitVector,AbstractVector{<:Bool}}
const StringOrSymb = Union{Symbol, <:AbstractString}

function BiLog end
function BiExp end
function BiRoot end
function BiPower end

include("Subspaces.jl")
export Plane
export BasisVector, PlaneCoordinates, IsOnPlane, TranslatePlane, RotatePlane, DecomposeWRTPlane
export DistanceToPlane, ProjectOntoPlane, IsNormalToPlane, MinimizeOnPlane, ParallelPlanes
export HyperPlane, HyperCube
export ConstructCube, CubeWidths, CubeVol, Center, TranslateCube, ResizeCube
export DropCubeDims, FaceCenters, Corners
export PositiveDomain, NegativeDomain, FullDomain, rand, EmbeddedODESolution


include("ModelMaps.jl")
export ModelMap, ModelOrFunction, InformNames, IsInDomain
# Parameter Space transforms
export ComponentwiseModelTransform
export ScaleTransform, TranslationTransform, LinearTransform, AffineTransform, LinearDecorrelation
export EmbedModelVia, ModelEmbedding
# Input and Output Transforms
export TransformXdata, TransformYdata


include("PredefinedModels.jl")
# Predefined Models
export LinearModel, QuadraticModel, ExponentialModel, SumExponentialsModel, PolynomialModel, GetLinearModel, GetGeneralLinearModel


include("DataStructures/DistributionTypes.jl")
export GeneralProduct


include("GeneralDataStructures.jl")
# export HealthyData, HealthyCovariance, CheckModelHealth
export xdata, ydata, xsigma, ysigma, xInvCov, yInvCov, Npoints, xdim, ydim, pdim, xpdim, DataspaceDim, Data, MLE, MLEuncert, LogLikeMLE, WoundX
export xdist, ydist, dist, ScaledResiduals
export Predictor, dPredictor, LogPrior, ConsistentElDims
export MeasureAutoDiffPerformance
export DataDist, SortDataSet, SortDataModel, SubDataSet, SubDataModel, DataFrame, join, length, AddDataPoint
export MLEinPlane, PlanarDataModel, DetermineDmodel
export Refit

include("DataStructures/DataSet.jl")
export DataSet


include("DataStructures/DataSetExact.jl")
export DataSetExact, LogLike


include("DataStructures/CompositeDataSet.jl")
export CompositeDataSet


include("DataStructures/GeneralizedDataSet.jl")
export GeneralizedDataSet


include("DataStructures/DataSetUncertain.jl")
export DataSetUncertain


include("DataStructures/UnknownVarianceDataSet.jl")
export UnknownVarianceDataSet


include("DataStructures/DataModel.jl")
export DataModel, AddLogPrior


include("DataStructures/ConditionGrid.jl")
export ConditionGrid, Conditions, ConditionSpecificProfiles, ConditionSpecificWaterFalls


include("NumericalTools/Miscellaneous.jl")
export BiLog, BiExp, BiLog10, BiExp10, BiRoot, BiPower, SoftAbs, SoftLog, ViewElements

include("NumericalTools/Integration.jl")
include("NumericalTools/Optimization.jl")
export Unpack, Unwind, Windup, PromoteStatic, SplitAfter
export GetH, suff, ToCols
export ConfAlpha, ConfVol, InvConfVol, ChisqCDF, InvChisqCDF
export KillAfter
export Integrate1D, IntegrateND, IntegrateOverConfidenceRegion, IntegrateOverApproxConfidenceRegion
export LineSearch, MonteCarloArea
export curve_fit, RobustFit, TotalLeastSquares, BlockMatrix
export ParameterSavingCallback, IncrementalTimeSeriesFit


include("Likelihoods.jl")
export likelihood, loglikelihood, Score, GetRemainderFunction, GeneralizedDOF


include("ModelPredictions.jl")
export EmbeddingMap, EmbeddingMatrix, EmbeddingMap!, EmbeddingMatrix!


include("ConfidenceRegions.jl")
export WilksCriterion, WilksTest, OrthVF, OrthVF!, FindMLE
export AutoScore, AutoMetric
export FindConfBoundary, FCriterion, FTest, FindFBoundary, ChisquaredReduced
export GenerateBoundary, ConfidenceRegion, ConfidenceRegions
export GenerateInterruptedBoundary, InterruptedConfidenceRegion
export IsStructurallyIdentifiable, StructurallyIdentifiable
export FisherMetric, GeometricDensity, VariancePropagation
export ConfidenceRegionVolume, CoordinateVolume
export ExpectedInvariantVolume, GeodesicRadius, CoordinateDistortion, Sensitivity

export Pullback, Pushforward
export AIC, AICc, BIC, ModelComparison, IsLinearParameter, IsLinear, LeastInformativeDirection

export FindConfBoundaryOnPlane, LinearCuboid, IntersectCube, IntersectRegion, MincedBoundaries, ConfidenceBoundary
export ContourDiagram, ContourDiagramLowerTriangular

export ApproxInRegion, ShadowTheatre, CastShadow, CrossValidation



include("ProfileLikelihood.jl")
# export ProfileLikelihood
export PlotProfileTrajectories, InterpolatedProfiles, ProfileBox, PracticallyIdentifiable
export ValInserter, InsertIntoFirst, InsertIntoLast, PinParameters, LinkParameters
export ParameterProfiles
export PlotProfilePaths, PlotProfilePathDiffs, PlotProfilePathNormDiffs
export ValidationProfiles, PredictionProfiles, ConvertValidationToPredictionProfiles


include("NumericalTools/Multistart.jl")
export MultistartFit, LocalMultistartFit, MultistartResults, WaterfallPlot, ParameterPlot, GetStepParameters
export DistanceMatrixWithinStep, DistanceMatrixBetweenSteps
export StochasticProfileLikelihood, FindGoodStart


include("Divergences.jl")
export KullbackLeibler


include("Geodesics.jl")
export ComputeGeodesic, GeodesicLength, GeodesicCrossing, DistanceAlongGeodesic, Endpoints, EvaluateEach
export RadialGeodesics, BoundaryViaGeodesic, GeodesicBetween, GeodesicDistance, GeodesicEnergy, ExponentialMap, LogarithmicMap, KarcherMean, MBAM
# Needs redo:
# export ConstLengthGeodesics, ConstParamGeodesics


include("Curvature.jl")
export ChristoffelSymbol, Riemann, Ricci, RicciScalar
# Also add Kretschmann, Schouten, Weyl?


include("DiffEqModels.jl")
export GetModel, ModifyODEmodel


include("SymbolicComputations.jl")
export OptimizeModel, OptimizedDM, InplaceDM, SymbolicModel, SymbolicdModel


include("Plotting.jl")
export PlotFit, ResidualPlot, ResidualVsFittedPlot, PlotQuantiles
export PlotScalar, PlotLogLikelihood, Plot2DVF, ResidualStandardError, PlotEllipses
export ParameterPlot, PlotErrorModel, TracePlot
export Deplanarize, VisualizeSols, VisualizeGeos, VisualizeSolPoints, ConstructAmbientSolution
export ConfidenceBands, ApproxConfidenceBands, PlotConfidenceBands, ConfidenceBandWidth, PredictionEnsemble, PlotMatrix
export XCube, PropagateUncertainty
export EvaluateAlongGeodesic, PlotAlongGeodesic, EvaluateAlongCurve, PlotAlongCurve, PhaseSpacePlot
# export RectangularFacetIndices, RectToTriangFacets, CreateMesh, ToObj, WriteObj


include("Exporting.jl")
export SaveAdaptive, SaveConfidence, SaveGeodesics, SaveDataSet
# export Homogenize, Dehomogenize


include("CustomIO.jl")
# export GeneratedFromAutoDiff, GeneratedFromSymbolic


# For debugging of PEtabExt
function GetNllh end
function GetNllhGrads end
function GetNllhHesses end
function GetDataUncertainty end
function GetConditionData end
function GetDataSets end
function GetModelFunction end
function SplitParamsIntoCategories end
function NicifyPEtabNames end


using SnoopPrecompile

SnoopPrecompile.@precompile_all_calls begin
    ParameterProfiles(DataModel(DataSet([1,2,3,4], [4,5,6.5,9], [0.5,0.45,0.6,1]), LinearModel); plot=false)
    ParameterProfiles(DataModel(DataSetUncertain([1,2,3,4], [4,5,6.5,9]; verbose=false), (x,p)->p[1].*x .+ p[2], [1.5,2,-0.4]); plot=false, verbose=false)
    MultistartFit(DataModel(DataSetExact([1,2,3,4], 0.5*[0.5,0.45,0.6,1], [4,5,6.5,9], [0.5,0.45,0.6,1]), (x,p)->p[1]*x + p[2], [1.48,2.27], true); N=3, maxval=5, verbose=false, showprogress=false)

    function SIR!(du,u,p,t)
        S, I, R = u
        β, γ = p
        du[1] = -β * I * S
        du[2] = +β * I * S - γ * I
        du[3] = +γ * I
        nothing
    end
    DataModel(DataSet(collect(1:14), [3, 8, 28, 75, 221, 291, 255, 235, 190, 126, 70, 28, 12, 5], 15ones(14); xnames= ["Days"], ynames=["Infected"]),
            ODEFunction(SIR!), X->([763.0-X[1], X[1], 0.0], @view X[2:3]), x->x[2], [0.614,0.00231,0.458], true; tol=1e-4)

    SIR! = nothing

    io = IOBuffer()
    ParamSummary(io, DataModel(DataSet([0.33, 1, 3], [0.88,0.5,0.35], [0.1,0.3,0.2]),
                ModelMap((x::Real,p::AbstractVector)->exp(-p[1]*x) + exp(-p[2]*x), θ::AbstractVector -> θ[1]-θ[2], PositiveDomain(2,1e2), (1,1,2)), [16, 0.41]))
    close(io)

    TotalLeastSquaresV(
        DataModel(DataSetExact([0.33, 1, 3], 0.5*[0.1,0.3,0.2], [0.88,0.5,0.35], [0.1,0.3,0.2]),
                ModelMap((x::Real,p::AbstractVector)->exp(-p[1]*x) + exp(-p[2]*x), θ::AbstractVector -> θ[1]-θ[2], PositiveDomain(2,1e2), (1,1,2)), [16, 0.41], true)
    )
    nothing
end


end # module
