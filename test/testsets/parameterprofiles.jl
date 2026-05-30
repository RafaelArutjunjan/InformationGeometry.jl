using InformationGeometry, Test, Distributions, LinearAlgebra, Optim


## Test value insertion embeddings
using ComponentArrays, BenchmarkTools, ForwardDiff
N = 40
F(x::AbstractVector) = sum(abs2, x)
vi! = ValInserter!([2, 5], [π, 1.0], zeros(N+2))
vi = ValInserter([2, 5], [π, 1.0], zeros(N+2); cached=false, nonmutating=false)
f2 = F∘vi!
f3 = F∘vi
X = randn(40)

@test vi!(X) == vi(X)
@test f2(X) == f3(X)
@test ForwardDiff.gradient(f2,X) == ForwardDiff.gradient(f3,X)
# @test (@belapsed ForwardDiff.gradient(f2,$X)) < (@belapsed ForwardDiff.gradient(f3,$X))

G(x::AbstractVector) = sum(abs2, x.P1) + sum([1,2,3.] .* x.P2)
vi2! = ValInserter!([2, 5], [π, 1.0], ComponentVector(P1=rand(N-1), P2=rand(3)))
vi2 = ValInserter([2, 5], [π, 1.0], ComponentVector(P1=rand(N-1), P2=rand(3)); cached=false, nonmutating=false)
g2 = G∘vi2!
g3 = G∘vi2

@test vi2!(X) == vi2(X)
@test g2(X) == g3(X)
@test ForwardDiff.gradient(g2,X) == ForwardDiff.gradient(g3,X)
# @test (@belapsed ForwardDiff.gradient(g2,$X)) < (@belapsed ForwardDiff.gradient(g3,$X))

using Zygote
@test_broken Zygote.gradient(g2,X)
@test_broken Zygote.gradient(g3,X)
vi3 = ValInserter([2, 5], [π, 1.0], ComponentVector(P1=rand(N-1), P2=rand(3)); cached=false, nonmutating=true)
g4 = G∘vi3
@test Zygote.gradient(g4,X)[1] == ForwardDiff.gradient(g3,X)


DM = DataModel(DataSet(1:4, [4,5,6.5,9], [0.5,0.45,0.6,1]), (x,p)->(p[1]+p[2])*x + exp(p[1]-p[2]), [1.3, 0.2]; name=:DM)
DMp = DataModel(DataSet(1:4, [4,5,6.5,9], [0.5,0.45,0.6,1]), (x,p)->(p[1]+p[2])*x + exp(p[1]-p[2]), [1.3, 0.2], x->logpdf(Laplace(0,0.5),x[1]); name=:DMp)

P1 = ParameterProfiles(DM, 2; IsCost=false, N=50, maxval=10, plot=false)
B1 = ProfileBox(P1, 2)

P1p = ParameterProfiles(DMp, 2; IsCost=false, N=50, maxval=10, plot=false)
B1p = ProfileBox(P1p, 2)

P2 = ParameterProfiles(DM, 2; IsCost=false, N=50, Multistart=20, maxval=10, plot=false, verbose=false)
B2 = ProfileBox(P2, 2)

P2p = ParameterProfiles(DMp, 2; IsCost=false, N=50, Multistart=20, maxval=10, plot=false, verbose=false)
B2p = ProfileBox(P2p, 2)

# Try Multistart reoptimization with general=true for pure cost function and general=false for manual step
P3 = ParameterProfiles(DM, 2; general=true, N=30, maxval=15, Multistart=15, plot=false, verbose=false)
P4 = ParameterProfiles(DM, 2; general=false, N=30, maxval=15, Multistart=15, plot=false, verbose=false)
# With Prior
P3p = ParameterProfiles(DMp, 2; general=true, N=30, maxval=15, Multistart=15, plot=false, verbose=false)
P4p = ParameterProfiles(DMp, 2; general=false, N=30, maxval=15, Multistart=15, plot=false, verbose=false)

B3 = ProfileBox(P3, 2)
B4 = ProfileBox(P4, 2)
B3p = ProfileBox(P3p, 2)
B4p = ProfileBox(P4p, 2)

# Check that all go above threshold
@test all(isfinite∘sum, Tuple(B1))
@test all(isfinite∘sum, Tuple(B2))
@test all(isfinite∘sum, Tuple(B3))
@test all(isfinite∘sum, Tuple(B4))
@test all(isfinite∘sum, Tuple(B1p))
@test all(isfinite∘sum, Tuple(B2p))
@test all(isfinite∘sum, Tuple(B3p))
@test all(isfinite∘sum, Tuple(B4p))
@test all(isfinite, ProfileBox(P3[1],2)[1])


DMU = DataModel(DataSetUncertain(1:4, [4,5,6.5,9]; verbose=false), (x,p)->(p[1]+p[2])*x + exp(p[1]-p[2]), [1.3, 0.2, -0.5])

PU = ParameterProfiles(DMU, 2; N=30, plot=false, verbose=false)
# Problem here: For some reason, this overflows if verbose = false is used but only with default NewtonTrustRegion or LBFGS!
# SOME Trajectories missing for these
# PU2 = ParameterProfiles(DMU, 2; N=30, maxval=15, Multistart=15, plot=false, verbose=false)


PU2 = ParameterProfiles(DMU, 2; N=30, maxval=15, Multistart=15, plot=false)
# PU2 = ParameterProfiles(DMU, 2; N=30, meth=NewtonTrustRegion(), maxval=15, Multistart=15, plot=false, TryCatchCostFunc=false, verbose=true)

BU = ProfileBox(PU, 2)
BU2 = ProfileBox(PU2, 2)

# Check that all go above threshold
@test all(isfinite∘sum, Tuple(BU))
@test all(isfinite∘sum, Tuple(BU2))


### Test other ParameterProfile computation methods
APU = ParameterProfiles(DMU, 3; ApproximatePaths=true, N=31, plot=false, verbose=false)
@test all(isfinite∘sum, Tuple(ProfileBox(APU, 2)))

PbPU = InformationGeometry.PreapproximatedParameterProfiles(DMU, 2; N=31, plot=false, verbose=false)
@test all(isfinite∘sum, Tuple(ProfileBox(PbPU, 1)))

IPU = IntegrationParameterProfiles(DMU, 2.2; N=31, plot=false, verbose=false)
@test all(isfinite∘sum, Tuple(ProfileBox(IPU, 2)))
IPU2 = IntegrationParameterProfiles(DMU, 2.2; N=31, γ=0.5, plot=false, verbose=false)
@test all(isfinite∘sum, Tuple(ProfileBox(IPU2, 2)))

S = StochasticProfileLikelihood(DMU; maxval=10, Nsingle=2, plot=false)
@test S isa InformationGeometry.MultistartResults


## Test FullParameterProfiles
DME = DataModel(DataSetExact(Data(DM), 0.25), (x,p)->(p[1]+p[2])*x + exp(p[1]-p[2]), [1.3, 0.2]; name=:DME)
FP = FullParameterProfiles(DME,1; pDomain=FullDomain(2,10), plot=false)
@test all(isfinite∘sum, Tuple(ProfileBox(FP, 1))[end-1:end])


# Test PredictionProfiles
PP = PredictionProfiles(DM, 1)
@test all(isfinite∘sum, Tuple(ProfileBox(PP,1)))


CG = ConditionGrid([DM, DMp])
PCG1 = ParameterProfiles(CG; general=true, plot=false, verbose=false)
PCG2 = ParameterProfiles(CG; maxval=20, Multistart=10, plot=false, verbose=false)
@test all(isfinite∘sum, Tuple(ProfileBox(PCG1)))
@test all(isfinite∘sum, Tuple(ProfileBox(PCG2)))

using FiniteDifferences
@test abs(pdim(DM) - GeneralizedDOF(DM)) < 1e-5
@test pdim(CG) > GeneralizedDOF(ConditionGrid([DM, DMp], [ViewElements([1,2]), ViewElements([1,3])], rand(3)))


using ComponentArrays, InformationGeometry
Model(x, p::ComponentVector) = p.A .* x .+ p.B
cpdm = DataModel(DataSet(1:4, [4,5,6.5,9], [0.5,0.45,0.6,1]), Model, ComponentVector(A=5.0, B=3.0))
@test all(isfinite∘sum, Tuple(ProfileBox(ParameterProfiles(cpdm; plot=false, Confnum=2), 2)))
@test all(isfinite∘sum, Tuple(ProfileBox(ParameterProfiles(cpdm, 1, 1:1; plot=false, Confnum=2), 2))[1])
@test all(isfinite∘sum, Tuple(ProfileBox(PredictionProfiles(cpdm, 1; Confnum=2), 2)))

# LinearModel ensures correct use of GetOnlyModelParams in PredictionProfiles
cpdmu = DataModel(DataSetUncertain(1:4, [4,5,6.5,9]), LinearModel, ComponentVector(A=5.0, B=3.0, σ=0.1))
@test all(isfinite∘sum, Tuple(ProfileBox(ParameterProfiles(cpdmu; plot=false, Confnum=2), 2)))
@test all(isfinite∘sum, Tuple(ProfileBox(ParameterProfiles(cpdmu, 1, 1:1; plot=false, Confnum=2), 2))[1])
@test all(isfinite∘sum, Tuple(ProfileBox(PredictionProfiles(cpdmu, 1; Confnum=2), 2)))

@test all(isfinite∘sum, Tuple(ProfileBox(ReoptimizeProfile(cpdmu, ParameterProfiles(cpdmu; maxiters=100, plot=false, Confnum=2)), 2)))

using Plots
@test Plots.plot(PU) isa Plots.Plot
@test PlotProfilePaths(PU) isa Plots.Plot
@test PlotProfileTrajectories(PU) isa Plots.Plot
@test PlotAlongProfilePaths(PU, norm) isa Plots.Plot
@test Plots.plot(S) isa Plots.Plot