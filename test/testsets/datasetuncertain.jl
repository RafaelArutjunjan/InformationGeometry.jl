using InformationGeometry, Test, LinearAlgebra, Distributions, Optim

using Random;   rng=Random.seed!(1234)

# Detect if error parameters already accounted for in given ModelMap Domain and attempt to fix automatically
X = 1:5;    Y = rand(rng, 5)
@test DataModel(DataSetUncertain(X, Y), ModelMap(LinearModel; startp=rand(3))) isa DataModel
@test DataModel(DataSetUncertain(X, Y), ModelMap(LinearModel; startp=rand(2))) isa DataModel
@test DataModel(DataSetUncertain(X, Y), ModelMap(LinearModel)) isa DataModel
@test DataModel(DataSetUncertain(X, Y), LinearModel) isa DataModel

function tester(DMU::AbstractDataModel, DM::AbstractDataModel, DME::AbstractDataModel, mle::AbstractVector; atol::Real=1e-6, atol2::Real=0.5)
    @assert Data(DMU) isa InformationGeometry.AbstractUnknownUncertaintyDataSet
    @assert Data(DM) isa InformationGeometry.AbstractFixedUncertaintyDataSet
    @assert Data(DME) isa InformationGeometry.AbstractFixedUncertaintyDataSet

    @test sum(abs, loglikelihood(DMU)(mle) - loglikelihood(DM)(mle)) < atol
    @test sum(abs, Score(DMU)(mle)[1:pdim(DM)] - Score(DM)(mle)[1:pdim(DM)]) < atol
    @test sum(abs, Score(DM)(mle)[1:pdim(DM)] - Score(DME)(mle)[1:pdim(DM)]) < atol

    # Average deviation between AutoMetric and hardcoded Fisher
    @test sum(abs, inv(InformationGeometry._FisherMetric(Data(DMU), Predictor(DMU), dPredictor(DMU), mle)) .- inv(AutoMetric(DMU, mle))) / (pdim(DMU)^2) < atol2
    @test sum(abs, inv(InformationGeometry._FisherMetric(Data(DM), Predictor(DM), dPredictor(DM), mle[1:end-1])) .- inv(AutoMetric(DM, mle[1:end-1]))) / (pdim(DM)^2) < atol2

    # Depending on whether Fisher metric is AutoMetric or custom, there may be slight deviations. Use inverse instead since less sensitive
    @test sum(abs, inv(FisherMetric(DM)(mle)[1:pdim(DM),1:pdim(DM)]) .- inv(FisherMetric(DMU)(mle)[1:pdim(DM),1:pdim(DM)])) < atol
    @test sum(abs, inv(FisherMetric(DM)(mle)[1:pdim(DM),1:pdim(DM)]) .- inv(FisherMetric(DME)(mle)[1:pdim(DM),1:pdim(DM)])) < atol
    @test sum(abs, -GetHess(loglikelihood(DMU))(mle)[1:pdim(DM),1:pdim(DM)] .+ GetHess(loglikelihood(DM))(mle)[1:pdim(DM),1:pdim(DM)]) < atol
    @test sum(abs, -GetHess(loglikelihood(DM))(mle)[1:pdim(DM),1:pdim(DM)] .+ GetHess(loglikelihood(DME))(mle)[1:pdim(DM),1:pdim(DM)]) < atol
end

import InformationGeometry: GetOnlyModelParams
# Agreement between variance propagation and confidence bands for linearly parametrised models
function TestAgreement(DM::AbstractDataModel, Value::Real=2e-3; N::Int=51, Confnum::Real=3)
    Xs = range(XCube(DM); length=N);   Ys = EmbeddingMap(DM, GetOnlyModelParams(DM)(MLE(DM)), Xs)
    F = VariancePropagation(DM; Confnum)
    S = ConfidenceRegion(DM, Confnum)
    M = (@view ConfidenceBands(DM, S, Xs; plot=false)[:,end]) .- Ys;
    @test norm((M .- F.(Xs)) ./ length(Xs)) < Value
end


Mlee = [1,1,0.4]

DSU = DataSetUncertain(1:5, (1:5) + [rand(Normal(0,0.4)) for i in 1:5], (x,y,p)->1/abs(p[1]), [0.4]; BesselCorrection=false)
dsu = DataSetUncertain(xdata(DSU), ydata(DSU), (x,y,p)->exp10(-p[1]), [0]; BesselCorrection=false)
DS = DataSet(xdata(DSU), ydata(DSU), 0.4)
DSE = DataSetExact(xdata(DSU), ydata(DSU), 0.4)

DMU = DataModel(DSU, (x,p)->p[1]*x + p[2], [1, 1, 1.]; meth=Optim.Newton())
dmu = DataModel(dsu, (x,p)->p[1]*x + p[2], [1, 1, 1.]; meth=Optim.Newton())
DM = DataModel(DS, (x,p)->p[1].*x .+ p[2], [1, 1.]; meth=Optim.Newton())
DME = DataModel(DSE, (x,p)->p[1]*x + p[2], [1, 1.]; meth=Optim.Newton())
tester(DMU, DM, DME, Mlee)

Mlee = [1,1,log10(0.4)]
tester(dmu, DM, DME, Mlee)

## Agrees exactly for AbstractFixedUncertaintyDataSet and linearly parametrized model
TestAgreement(DM)
## Does not agree exactly for AbstractUnknownUncertaintyDataSet
TestAgreement(DMU, 0.2)



using ModelingToolkit, OrdinaryDiffEq, LinearAlgebra

eval(ModelingToolkit._parse_vars(:parameters, Real, (:t, :β, :γ), ModelingToolkit.toparam))
eval(ModelingToolkit._parse_vars(:variables, Real, (:(S(t)), :(I(t)), :(R(t)))))
Dt = Differential(t)
Eqs = Equation[Dt(S) ~ -β*S*I/(S+I+R), Dt(I) ~ β*S*I/(S+I+R) -γ*I, Dt(R) ~ γ*I]

SIRsys = ODESystem(Eqs, t, [S, I, R], [β, γ]; name=:SIR)
SIRinitial = p->([763.0-exp10(p[1]), exp10(p[1]), 0], p[2:3])
SIRobservables = [2]

times = 1:14
mle = [-0.21161, 0.24626, -0.33957]
Mod = GetModel(Exp10Transform(SIRsys), SIRinitial, SIRobservables; tol=1e-8, Domain=FullDomain(3,3), meth=Tsit5())
infected = EmbeddingMap(Val(true), Mod, mle, times) .+ rand(rng, Normal(0,15), length(times))


# infected = [3, 8, 28, 75, 221, 291, 255, 235, 190, 126, 70, 28, 12, 5]
SIRDS = DataSet(times, infected, 15)
SIRDSE = DataSetExact(times, 0.2ones(length(times)), infected, 15ones(length(times)))
SIRDSU = DataSetUncertain(times, infected, (x,y,p)->1/abs(p[1]), [15.]; BesselCorrection=false)
SIRdsu = DataSetUncertain(times, infected, (x,y,p)->exp10(-p[1]), [0]; BesselCorrection=false)

SIRDM = DataModel(SIRDS, Mod, [0, 0, 0]; tol=1e-10)
SIRDME = DataModel(SIRDSE, Mod, [0, 0, 0]; tol=1e-10)
SIRDMU = DataModel(SIRDSU, remake(Mod; Domain=vcat(FullDomain(3,3), PositiveDomain(1,30)), xyp=(1,1,4), pnames=InformationGeometry.CreateSymbolNames(4)), [0, 0, 0, 15.0]; tol=1e-10)
SIRdmu = DataModel(SIRdsu, remake(Mod; Domain=vcat(FullDomain(3,3), PositiveDomain(1,30)), xyp=(1,1,4), pnames=InformationGeometry.CreateSymbolNames(4)), [0, 0, 0, 1.0]; tol=1e-10)

Mlee = [mle; 15]
tester(SIRDMU, SIRDM, SIRDME, Mlee; atol=1e-2, atol2=1.2)

Mlee = [mle; log10(15)]
tester(SIRdmu, SIRDM, SIRDME, Mlee; atol=1e-2, atol2=1.2)


## Does not agree exactly for non-linear model
TestAgreement(SIRDM, 0.5)
TestAgreement(SIRdmu, 0.5)



##### Missing values in DataSetUncertain
T = Float64[1,4,3,3,2,5]
SingleModel(t,p) = p[1]*t + p[2]t^2 + p[3]
CopiedModel(t,p) = [SingleModel(t,p); SingleModel(t,p)]
ptrue = rand(3)
Y = map(t->SingleModel(t,ptrue), T) .+ randn(length(T))
Mask = trues(length(T));    Mask[[1,3,5]] .= false
Yd = [[(i ∈ [1,3,5] ? NaN : y) for (i,y) in enumerate(Y)] [(i ∉ [1,3,5] ? NaN : y) for (i,y) in enumerate(Y)]]

DSU = DataSetUncertain(T, Y)
DMU = DataModel(DSU, SingleModel)

DSUm = DataSetUncertain(T, Unwind(Yd), (6,1,2), (x,y,c)->[exp10(-c[1]), exp10(-c[1])], InformationGeometry.DefaultErrorModelSplitter(1), [0.1])
DMUm = DataModel(DSUm, CopiedModel)


@test loglikelihood(DMU, MLE(DMU)) ≈ loglikelihood(DMUm, MLE(DMU))
@test Score(DMU, MLE(DMU)) ≈ Score(DMUm, MLE(DMU))
# Currently using AutoMetric
@test FisherMetric(DMU, MLE(DMU)) ≈ FisherMetric(DMUm, MLE(DMU))
# @test InformationGeometry._FisherMetric(Data(DMU), Predictor(DMU), dPredictor(DMU), MLE(DMU)) ≈ InformationGeometry._FisherMetric(Data(DMUm), Predictor(DMUm), dPredictor(DMUm), MLE(DMU))

@test ysigma(DMU, MLE(DMU)) ≈ ysigma(DMUm, MLE(DMU))
@test yInvCov(DMU, MLE(DMU)) ≈ yInvCov(DMUm, MLE(DMU))


# DMU = SIRDMU
# DM = SIRDM

# FisherMetric(DMU)(Mlee)[1:pdim(DM),1:pdim(DM)]
# FisherMetric(DM)(Mlee[1:end-1])[1:pdim(DM),1:pdim(DM)]

# AutoMetric(DMU, Mlee)[1:pdim(DM),1:pdim(DM)]
# AutoMetric(DM, Mlee[1:end-1])[1:pdim(DM),1:pdim(DM)]

# Score(DMU)(Mlee)
# Score(DM)(Mlee[1:end-1])

# AutoScore(DMU)(Mlee)
# AutoScore(DM)(Mlee[1:end-1])

# inv(FisherMetric(DMU)(Mlee))[1:pdim(DM),1:pdim(DM)]
# inv(FisherMetric(DM)(Mlee[1:end-1]))[1:pdim(DM),1:pdim(DM)]

# inv(AutoMetric(DMU)(Mlee))[1:pdim(DM),1:pdim(DM)]
# inv(AutoMetric(DM)(Mlee[1:end-1]))[1:pdim(DM),1:pdim(DM)]


# inv(FisherMetric(DM)(mle)[1:pdim(DM),1:pdim(DM)]) .- inv(FisherMetric(DMU)(Mlee)[1:pdim(DM),1:pdim(DM)])