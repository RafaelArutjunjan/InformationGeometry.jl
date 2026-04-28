
using InformationGeometry, PEtab, FiniteDifferences, Plots, Test

import InformationGeometry: Trafos
function TestConversion(petab_prob::PEtabODEProblem, DM::AbstractDataModel=ConditionGrid(petab_prob); atol=1e-10, atol2=0.1)
    
    @test sum(abs, -InformationGeometry.loglikelihood(DM,MLE(DM)) - petab_prob.nllh(MLE(DM))) < atol
    @test sum(abs, -Score(DM, MLE(DM)) - petab_prob.grad(MLE(DM))) < atol
    @test sum(abs, FisherMetric(DM, MLE(DM)) - petab_prob.FIM(MLE(DM))) < atol
    @test sum(abs, InformationGeometry.CostHessian(DM)(MLE(DM)) - petab_prob.hess(MLE(DM))) < atol
    
    GradRes1 = rand(length(MLE(DM)));    GradRes2 = rand(length(MLE(DM)));    HessRes1 = rand(length(MLE(DM)), length(MLE(DM)));    HessRes2 = rand(length(MLE(DM)), length(MLE(DM)))
    Score(DM)(GradRes1, MLE(DM));   petab_prob.grad!(GradRes2, MLE(DM));    @test sum(abs, -GradRes1 -GradRes2) < atol
    FisherMetric(DM)(HessRes1, MLE(DM));   petab_prob.FIM!(HessRes2, MLE(DM));    @test sum(abs, HessRes1 -HessRes2) < atol
    InformationGeometry.CostHessian(DM)(HessRes1, MLE(DM));   petab_prob.hess!(HessRes2, MLE(DM));    @test sum(abs, HessRes1 -HessRes2) < atol

    ###### Score seems to be slightly dissimilar

    # Consistency of likelihoods of individual conditions with total
    @test sum(loglikelihood(DM[i], Trafos(DM)[i](MLE(DM))) for i in eachindex(DM)) == loglikelihood(DM, MLE(DM))
    @test sum(abs, sum(Score(DM[i], Trafos(DM)[i](MLE(DM))) for i in eachindex(DM)) .- Score(DM, MLE(DM))) < atol2
    @test sum(abs, sum(FisherMetric(DM[i], Trafos(DM)[i](MLE(DM))) for i in eachindex(DM)) .- FisherMetric(DM, MLE(DM))) < atol

    cids = InformationGeometry.ConditionNames(DM);  j = 1
    # Compute reduced chi^2
    @test sum(abs2, (EmbeddingMap(DM, MLE(DM), cids[j]) - ydata(Data(Conditions(DM)[j]))) ./ ysigma(Conditions(DM)[j], Trafos(DM)[j](MLE(DM)))) / InformationGeometry.DataspaceDim(Conditions(DM)[j]) < 5
    # Test derivative of prediction (currently FiniteDifferences)
    @test !all(iszero, EmbeddingMatrix(DM, MLE(DM), cids[j]))

    ## Check that reconstructed objective function and Score corresponds to PEtab.jl for simple model
    ## Assuming normal data!
    dmj = DataModel(DataSet(xdata(Data(Conditions(DM)[j])), ydata(Data(Conditions(DM)[j])), ysigma(Conditions(DM)[j], Trafos(DM)[j](MLE(DM)))), Predictor(Conditions(DM)[j]), Trafos(DM)[j](MLE(DM)), true)
    @test sum(InformationGeometry.LogLikeMLE(dmj) - loglikelihood(DM[j], Trafos(DM)[j](MLE(DM)))) < atol

    ###### Check data transfer for DataSetUncertain
end

using ModelingToolkitBase, Distributions, OrdinaryDiffEq, DataFrames
using ModelingToolkitBase: t_nounits as t, D_nounits as D
begin
    ps = @parameters S0 c1 c2 c3=3.0
    sps = @variables S(t) = S0 E(t) = 50.0 SE(t) = 0.0 P(t) = 0.0 obs1(t) obs2(t)
    eqs = [
        # Dynamics
        D(S) ~ -c1 * S * E + c2 * SE
        D(E) ~ -c1 * S * E + c2 * SE + c3 * SE
        D(SE) ~ c1 * S * E - c2 * SE - c3 * SE
        D(P) ~ c3 * SE
        # Observables
        obs1 ~ S + E
        obs2 ~ P
    ]
    @named sys_model = System(eqs, t, sps, ps)
    sys = mtkcompile(sys_model)

    TryParse(x::AbstractString) = try parse(Float64, x) catch; x end

    petab_obs1 = PEtabObservable(:petab_obs1, :obs1, 3.0)
    @parameters sigma
    petab_obs2 = PEtabObservable(:petab_obs2, :obs2, sigma)

    observables = [petab_obs1, petab_obs2]

    p_c1 = PEtabParameter(:c1)
    p_c2 = PEtabParameter(:c2; prior = LogNormal(1.0, 0.3))

    p_S0 = PEtabParameter(:S0)
    p_sigma = PEtabParameter(:sigma)
    pest = [p_c1, p_c2, p_S0, p_sigma]

    # Simulate with 'true' parameters
    ps = [:c1 => 1.0, :c2 => 10.0, :c3 => 3.0, :S0 => 100.0]
    u0 = [:E => 50.0, :SE => 0.0, :P => 0.0]
    tspan = (0.0, 5.0)
    oprob = ODEProblem(sys, u0, tspan, ps)
    sol = solve(oprob, Rodas5P(); saveat = 0:0.5:5.0)

    obs1 = (sol[:S] + sol[:E]) .+ randn(length(sol[:E]))
    obs2 = sol[:P] .+ randn(length(sol[:P]))

    df1 = DataFrame(obs_id = "petab_obs1", time = sol.t, measurement = obs1)
    df2 = DataFrame(obs_id = "petab_obs2", time = sol.t[1:end-1], measurement = obs2[1:end-1])
    measurements = vcat(df1, df2)

    model_sys = PEtabModel(sys, observables, measurements, pest)
    petab_prob = PEtabODEProblem(model_sys; gradient_method=:ForwardDiff, hessian_method=:ForwardDiff)

    measurements2 = vcat(
        DataFrame(obs_id = "petab_obs1", time = sol.t, measurement = obs1, noiseParameters = TryParse(petab_obs1.noise_formula)),
        DataFrame(obs_id = "petab_obs2", time = sol.t, measurement = obs2, noiseParameters = TryParse(petab_obs2.noise_formula))
    )
    petab_prob2 = PEtabODEProblem(PEtabModel(sys, observables, measurements2, pest))
end

start = [-0.4, 0.5, 2, 0.08]
# For DOF, see https://github.com/sebapersson/PEtab.jl/issues/362
DM1 = ConditionGrid(petab_prob, start; FixedError=true, SkipOptim=false);       @test Plots.plot(DM1) isa Plots.Plot;     @test_broken InformationGeometry.DOF(DM1) == 3
DM2 = ConditionGrid(petab_prob, start; FixedError=false, SkipOptim=false);      @test Plots.plot(DM2) isa Plots.Plot;     @test_broken InformationGeometry.DOF(DM2) == 3;      @test Data(DM2).nerrorparameters == 1
DM3 = ConditionGrid(petab_prob2, start; FixedError=true, SkipOptim=false);      @test Plots.plot(DM3) isa Plots.Plot;     @test InformationGeometry.DOF(DM3) == 3
DM4 = ConditionGrid(petab_prob2, start; FixedError=false, SkipOptim=false);     @test Plots.plot(DM4) isa Plots.Plot;     @test InformationGeometry.DOF(DM4) == 3;      @test Data(DM4).nerrorparameters == 1

TestConversion(petab_prob, DM1)
