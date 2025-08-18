using InformationGeometry, Test, OrdinaryDiffEq, ModelingToolkit, LinearAlgebra

function SIR!(du,u,p,t)
    S, I, R = u
    β, γ = p
    du[1] = -β * I * S
    du[2] = +β * I * S - γ * I
    du[3] = +γ * I
    nothing
end
SIRsys = ODEFunction(SIR!)
infected = [3, 8, 28, 75, 221, 291, 255, 235, 190, 126, 70, 28, 12, 5]
SIRDS = InformNames(DataSet(collect(1:14), infected, 5ones(14)), ["Days"], ["Infected"])

SIRinitial = X->([763.0-X[1], X[1], 0.0], X[2:3])

# Use SplitterFunction SIRinitial to infer initial condition I₀ as first parameter
SIRDM = DataModel(SIRDS, SIRsys, SIRinitial, x->x[2], [0.6,0.0023,0.46]; tol=1e-6)
@test SIRDM isa DataModel
@test DataModel(SIRDS, SIRsys, [762, 1, 0.], [2], [0.0022,0.45], true; tol=1e-6) isa DataModel
@test norm(2*EmbeddingMap(Data(SIRDM), Predictor(SIRDM), MLE(SIRDM)) - EmbeddingMap(Data(SIRDM), ModifyODEmodel(SIRDM, x->2*x[2]), MLE(SIRDM))) < 2e-4

# Use this to avoid the macros @parameters and @variables inside @safetestset
eval(ModelingToolkit._parse_vars(:parameters, Real, (:t, :k), ModelingToolkit.toparam))
eval(ModelingToolkit._parse_vars(:variables, Real, (:(A(t)),)))
sys = ODESystem([Differential(t)(A) ~ -k*A], t, [A], [k]; name=Symbol("Decay System"))
Split(θ) = (θ[1:1], θ[2:2])
Observe(u) = 2u[1]
dm = DataModel(DataSet([0.2,1,2], [0.9,0.4,0.25], [0.1,0.1,0.1]), sys, Split, Observe)
@test dm isa DataModel
# ObservationFunction is extended to 3 arguments, does not equal initially given PreObservationFunction
@test length(Predictor(dm).Meta) == 4 && Predictor(dm).Meta[1:2] == (sys, Split)
@test string(Predictor(dm).name) == "Decay System"

# Backwards in time integration (GetModelRobust)
@test all(EmbeddingMap(SIRDM, MLE(SIRDM), [-10, 3, -0.5, 15]) .> 0)

# Check AutoDiff in time
using ForwardDiff
@assert 0 < ForwardDiff.derivative(t->Predictor(SIRDM)(t, MLE(SIRDM))[1], 1.)
@assert isposdef(ForwardDiff.jacobian(x->EmbeddingMap(SIRDM, MLE(SIRDM), x), [1,2,3.]))