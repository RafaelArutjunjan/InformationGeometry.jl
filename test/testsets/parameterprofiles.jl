using InformationGeometry, Test, Distributions, LinearAlgebra, Optim

DM = DataModel(DataSet(1:3, [4,5,6.5], [0.5,0.45,0.6]), (x,p)->(p[1]+p[2])*x + exp(p[1]-p[2]), [1.3, 0.2]; name=:DM)
DMp = DataModel(DataSet(1:3, [4,5,6.5], [0.5,0.45,0.6]), (x,p)->(p[1]+p[2])*x + exp(p[1]-p[2]), [1.3, 0.2], x->logpdf(Laplace(0,0.5),x[1]); name=:DMp)

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


DMU = DataModel(DataSetUncertain(1:3, [4,5,6.5]; verbose=false), (x,p)->(p[1]+p[2])*x + exp(p[1]-p[2]), [1.3, 0.2, -0.5])

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
PbPU = PreburnedParameterProfiles(DMU, 2; N=31, plot=false, verbose=false)
IPU = IntegrationParameterProfiles(DMU, 2; N=31, plot=false, verbose=false)

@test all(isfinite∘sum, Tuple(ProfileBox(PbPU, 2)))
@test all(isfinite∘sum, Tuple(ProfileBox(IPU, 2)))


APU = ParameterProfiles(DMU, 3; ApproximatePaths=true, N=31, plot=false, verbose=false)
@test all(isfinite∘sum, Tuple(ProfileBox(APU, 2)))



CG = ConditionGrid([DM, DMp])
PCG1 = ParameterProfiles(CG; general=true, plot=false, verbose=false)
PCG2 = ParameterProfiles(CG; maxval=20, Multistart=10, plot=false, verbose=false)
@test all(isfinite∘sum, Tuple(ProfileBox(PCG1)))
@test all(isfinite∘sum, Tuple(ProfileBox(PCG2)))

using FiniteDifferences
@test abs(pdim(DM) - GeneralizedDOF(DM)) < 1e-5
@test pdim(CG) > GeneralizedDOF(ConditionGrid([DM, DMp], [ViewElements([1,2]), ViewElements([1,3])], rand(3)))


using ComponentArrays, InformationGeometry
X = ComponentVector(A=5.0, B=3.0)
Model(x, p::ComponentVector) = p.A .* x .+ p.B
@test all(isfinite∘sum, Tuple(ProfileBox(ParameterProfiles(DataModel(DataSet(1:3, [4,5,6.5], [0.5,0.45,0.6]), Model, X); plot=false))))
