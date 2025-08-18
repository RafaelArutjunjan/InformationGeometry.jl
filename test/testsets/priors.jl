using InformationGeometry, Test, LinearAlgebra, Distributions

DS1 = DataSet([0,0.5],[1.,3.],[1.2,2.]);    DS2 = DataSet([1,1.5],[7.,8.1],[0.6,1.])
DS = join(DS1, DS2);    model(x,θ) = θ[1] * x + θ[2];
DM1 = DataModel(DS1,model);     DM = DataModel(DS,model);

logprior(X) = loglikelihood(DM1, X)

DM12 = DataModel(DS2, model, MLE(DM1), logprior)
@test norm(MLE(DM) - MLE(DM12)) < 1e-6
@test loglikelihood(DM, MLE(DM)) ≈ loglikelihood(DM12, MLE(DM))
@test norm(Score(DM, MLE(DM)) - Score(DM12, MLE(DM))) < 1e-12
@test FisherMetric(DM, MLE(DM)) ≈ FisherMetric(DM12, MLE(DM))

# Intentionally not giving information about LogPrior input here
dm = DataModel(DS, model, MLE(DM), x->0.0)
@test loglikelihood(DM, MLE(DM)) ≈ loglikelihood(dm, MLE(DM))
@test Score(DM, MLE(DM)) ≈ Score(dm, MLE(DM))
@test FisherMetric(DM, MLE(DM)) ≈ FisherMetric(dm, MLE(DM))