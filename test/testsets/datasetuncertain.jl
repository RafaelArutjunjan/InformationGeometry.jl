using InformationGeometry, Test, LinearAlgebra, Distributions, Optim

# Detect if error parameters already accounted for in given ModelMap Domain and attempt to fix automatically
X = 1:5;    Y = rand(5)
@test DataModel(DataSetUncertain(X, Y), ModelMap(LinearModel; startp=rand(3))) isa DataModel
@test DataModel(DataSetUncertain(X, Y), ModelMap(LinearModel; startp=rand(2))) isa DataModel
@test DataModel(DataSetUncertain(X, Y), ModelMap(LinearModel)) isa DataModel
@test DataModel(DataSetUncertain(X, Y), LinearModel) isa DataModel

function tester(DM::AbstractDataModel, DM2::AbstractDataModel, DM3::AbstractDataModel, mle::AbstractVector; atol::Real=1e-6)
    @assert Data(DM) isa InformationGeometry.AbstractUnknownUncertaintyDataSet
    @assert Data(DM2) isa InformationGeometry.AbstractFixedUncertaintyDataSet
    @assert Data(DM3) isa InformationGeometry.AbstractFixedUncertaintyDataSet

    @test isapprox(loglikelihood(DM)(mle), loglikelihood(DM2)(mle); atol)
    @test isapprox(Score(DM)(mle)[1:pdim(DM2)], Score(DM2)(mle)[1:pdim(DM2)]; atol)
    @test isapprox(Score(DM2)(mle)[1:pdim(DM2)], Score(DM3)(mle)[1:pdim(DM2)]; atol)
    @test isapprox(FisherMetric(DM)(mle)[1:pdim(DM2),1:pdim(DM2)], FisherMetric(DM2)(mle)[1:pdim(DM2),1:pdim(DM2)]; atol)
    @test isapprox(FisherMetric(DM2)(mle)[1:pdim(DM2),1:pdim(DM2)], FisherMetric(DM3)(mle)[1:pdim(DM2),1:pdim(DM2)]; atol)
    @test isapprox(-GetHess(loglikelihood(DM))(mle)[1:pdim(DM2),1:pdim(DM2)], -GetHess(loglikelihood(DM2))(mle)[1:pdim(DM2),1:pdim(DM2)]; atol)
    @test isapprox(-GetHess(loglikelihood(DM2))(mle)[1:pdim(DM2),1:pdim(DM2)], -GetHess(loglikelihood(DM3))(mle)[1:pdim(DM2),1:pdim(DM2)]; atol)
end

DS = DataSetUncertain(1:5, (1:5) + [rand(Normal(0,0.4)) for i in 1:5], (x,y,p)->1/abs(p[1]), [0.4]; xnames=["Time"], ynames=["Signal"])
tester(
    DataModel(DS, (x,p)->p[1]*x + p[2], [1, 1, 1.]; meth=Optim.Newton()),
    DataModel(DataSet(xdata(DS), ydata(DS), 0.4; xnames=["Time"], ynames=["Signal"]), (x,p)->p[1].*x .+ p[2], [1, 1.]; meth=Optim.Newton()),
    DataModel(DataSetExact(xdata(DS), ydata(DS), 0.4), (x,p)->p[1]*x + p[2], [1, 1.]; meth=Optim.Newton()),
    [1,1,0.4]
)

import InformationGeometry: SplitErrorParams
# Agreement between variance propagation and confidence bands for linearly parametrised models
function TestAgreement(DM::AbstractDataModel, Value::Real=2e-3; N::Int=51, Confnum::Real=3)
    Xs = range(XCube(DM); length=N);   Ys = EmbeddingMap(DM, SplitErrorParams(DM)(MLE(DM))[1], Xs)
    F = VariancePropagation(DM; Confnum)
    S = ConfidenceRegion(DM, Confnum)
    M = (@view ConfidenceBands(DM, S, Xs; plot=false)[:,end]) .- Ys;
    @test norm((M .- F.(Xs)) ./ length(Xs)) < Value
end
## Agrees for AbstractFixedUncertaintyDataSet
TestAgreement(DataModel(DataSet(X, Y), LinearModel))
## Does not agree for exactly AbstractUnknownUncertaintyDataSet
# TestAgreement(DataModel(DataSetUncertain(X, Y), LinearModel))

using ModelingToolkit, OrdinaryDiffEq, LinearAlgebra

ToNuc = 0.45/1.4
FromNuc = 1/ToNuc

eval(ModelingToolkit._parse_vars(:parameters, Real, (:t, :Epo_degradation_BaF3,
                        :k_phos, :k_imp_homo, :k_imp_hetero, :k_exp_homo, :k_exp_hetero), ModelingToolkit.toparam))
eval(ModelingToolkit._parse_vars(:variables, Real, (:(BaF3_Epo(t)),
                        :(STAT5A(t)), :(STAT5B(t)), :(pApA(t)), :(pApB(t)),
                        :(pBpB(t)), :(nucpApA(t)), :(nucpApB(t)), :(nucpBpB(t)))))
Dt = Differential(t)
Eqs = Equation[Dt(BaF3_Epo) ~ -Epo_degradation_BaF3*BaF3_Epo,
    Dt(STAT5A)  ~ k_exp_hetero*(nucpApB^2) + 2k_exp_homo*(nucpApA^2) - k_phos*(STAT5A^2)*BaF3_Epo - k_phos*BaF3_Epo*STAT5A*STAT5B,
    Dt(STAT5B)  ~ k_exp_hetero*(nucpApB^2) + 2k_exp_homo*(nucpBpB^2) - k_phos*(STAT5B^2)*BaF3_Epo - k_phos*BaF3_Epo*STAT5A*STAT5B,
    Dt(pApA)    ~ (1/2)*k_phos*(STAT5A^2)*BaF3_Epo - k_imp_homo*(pApA^2),
    Dt(pApB)    ~ k_phos*BaF3_Epo*STAT5A*STAT5B - k_imp_hetero*(pApB^2),
    Dt(pBpB)    ~ (1/2)*k_phos*(STAT5B^2)*BaF3_Epo - k_imp_homo*(pBpB^2),
    Dt(nucpApA) ~ k_imp_homo*(pApA^2) - k_exp_homo*(nucpApA^2),
    Dt(nucpApB) ~ k_imp_hetero*(pApB^2) - k_exp_hetero*(nucpApB^2),
    Dt(nucpBpB) ~ k_imp_homo*(pBpB^2) - k_exp_homo*(nucpBpB^2)]

BöhmSys = ODESystem(Eqs, t, [BaF3_Epo, STAT5A, STAT5B, pApA, pApB, pBpB, nucpApA, nucpApB, nucpBpB], [Epo_degradation_BaF3, k_phos, k_imp_homo, k_imp_hetero, k_exp_homo, k_exp_hetero]; name=:Boehm)

function BöhmObservation(u::AbstractVector, t::Real, θ::AbstractVector)
    specC17 = 0.107 # θ[end]
    BaF3_Epo, STAT5A, STAT5B, pApA, pApB, pBpB, nucpApA, nucpApB, nucpBpB = u
    [100 * (2 * specC17 * pApA + pApB)/(2 * specC17 * pApA + specC17 * STAT5A + pApB),
        100 * (2 * (1-specC17) * pBpB + pApB)/(2 * (1-specC17) * pBpB + (1-specC17) * STAT5B + pApB),
        100 * (2 * specC17 * pApA + specC17 * STAT5A + pApB)/(2 * specC17 * pApA + 2 * (1-specC17) * pBpB + 2 * pApB + specC17 * STAT5A + (1-specC17) * STAT5B)]
end

ratio = 0.693
Viewer = InformationGeometry.ViewElements(1:6)
BöhmInitial = p-> ([1.25e-7, 207.6 * ratio, 207.6 * (1-ratio), zeros(6)...], Viewer(p))

BöhmT = [0,2.5,5,10,15,20,30,40,50,60,80,100,120,160,200,240]
pSTAT5A_rel = [7.901072999,66.36349397,81.17132392,94.73030806,95.11648305,91.44171655,91.25709923,93.67229784,88.75423282,85.26970322,81.13239534,76.13592848,65.24805913,42.59965871,25.15779754,15.4301824]
pSTAT5B_rel = [4.596533343,29.63454599,46.04380647,81.97473362,80.5716093,79.03571964,75.67238037,71.62471986,69.06286328,67.14738432,60.89947629,54.80925777,43.98128998,29.77145816,20.08901656,10.96184517]
rSTAT5A_rel = [14.72316822,33.76234229,36.79985129,49.71760229,46.9281201,47.83657456,46.92872725,40.59775294,43.78366389,44.45738765,41.32715926,41.06273321,39.23583003,36.61946054,34.8937144,32.21107716]

BöhmDS = DataSet(BöhmT, [pSTAT5A_rel pSTAT5B_rel rSTAT5A_rel], [[4.12, 7.04, 3.37] for i in eachindex(BöhmT)]; xnames=["Time"], ynames=["pSTAT5A_rel", "pSTAT5B_rel", "rSTAT5A_rel"])
BöhmModel = GetModel(Exp10Transform(BöhmSys), BöhmInitial, BöhmObservation; tol=1e-4, Domain=HyperCube(-6ones(6), 6ones(6)))
BöhmDM = DataModel(BöhmDS, BöhmModel, [-1.5690, 4.1978, 4.99, -1.7859, -2.2, -5.]; tol=1e-6)

Böhmds = DataSetUncertain(BöhmT, [pSTAT5A_rel pSTAT5B_rel rSTAT5A_rel], (x,y,c)->Diagonal(1 ./ abs.(c)), [4.12, 7.04, 3.37]; xnames=["Time"], ynames=["pSTAT5A_rel", "pSTAT5B_rel", "rSTAT5A_rel"])
Böhmdm = DataModel(Böhmds, GetModel(Exp10Transform(BöhmSys), BöhmInitial, BöhmObservation; tol=1e-4, Domain=vcat(HyperCube(-6ones(6), 6ones(6)), HyperCube(1e-2ones(3), 20ones(3)))),
                        vcat([-1.5690, 4.1978, 4.99, -1.7859, -2.2, -5.], [4.12, 7.04, 3.37]); tol=1e-6)
tester(
    Böhmdm, BöhmDM,
    DataModel(DataSetExact(BöhmDS), Predictor(BöhmDM), MLE(BöhmDM); tol=1e-6),
    vcat(MLE(BöhmDM), [4.12, 7.04, 3.37]); atol=1e-4
)