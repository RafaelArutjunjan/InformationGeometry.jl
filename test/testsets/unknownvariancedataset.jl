
using InformationGeometry, Test, LinearAlgebra, Distributions, Optim

using Random;   rng=Random.seed!(1234)
X = 1:10;    Y = rand(rng, length(X))

DSE = DataSetExact(X, 0.3ones(length(X)), Y, 0.2ones(length(Y)))
DME = DataModel(DSE, LinearModel)

DSU = UnknownVarianceDataSet(X, Y, (x,y,c)->exp10(-c[1]), (x,y,c)->exp10(-c[1]), [log10(0.3)], [log10(0.2)])
DMU = DataModel(DSU, ModelMap(LinearModel))

mle = MLE(DME)
mleu = [xdata(DME); mle; [log10(0.3), log10(0.2)]]
keep = [falses(length(xdata(DME))); trues(length(mle)); falses(2)]

@test loglikelihood(DME, mle) ≈ loglikelihood(DMU, mleu)
@test sum(abs, Score(DME, mle) - Score(DMU, mleu)[keep]) < 1e-9
@test sum(abs, FisherMetric(DME, mle) - FisherMetric(DMU, mleu)[keep,keep]) < 1e-9
@test sum(abs, AutoMetric(DME, mle) - AutoMetric(DMU, mleu)[keep,keep]) < 1e-9
@test sum(abs, tr(inv(FisherMetric(DMU, mleu)) * AutoMetric(DMU, mleu)) - 14) < 1


### Add a prior on the ratio of sigma_x/sigma_y
dmu = DataModel(UnknownVarianceDataSet(1:7, [2.7, 4.4, 5.3, 6.6, 6.5, 6.3, 7.7]), ModelMap((x,p)->p[2]*x/(x+p[1])), rand(11), p->logpdf(Normal(0,0.3), p[10]-p[11]))
@test all(isfinite∘sum, Tuple(ProfileBox(ParameterProfiles(dmu, 2; N=31, plot=false),2)))



using Plots
@test Plots.plot(DMU, mleu, Confnum=1) isa Plots.Plot
