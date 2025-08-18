using InformationGeometry, Test, LinearAlgebra, Distributions

DS = DataSet([0,0.5,1,1.5],[1.,3.,7.,8.1],[1.2,2.,0.6,1.])
DM = DataModel(DS,LinearModel);    p = rand(2)

@test IsLinear(DM)
Dist = DataDist(ydata(DM),ysigma(DM))
@test abs(loglikelihood(DM,p) - logpdf(Dist,EmbeddingMap(DM,p))) < 1e-13
@test Score(DM,p) ≈ transpose(EmbeddingMatrix(DM,p)) * gradlogpdf(Dist,EmbeddingMap(DM,p))
@test FisherMetric(DM,p) ≈ transpose(EmbeddingMatrix(DM,p)) * inv(cov(Dist)) * EmbeddingMatrix(DM,p)

# Test AD vs manual derivative
@test norm(AutoScore(DM,p) - Score(DM,p)) < 2e-13
@test norm(AutoMetric(DM,p) .- FisherMetric(DM,p), 1) < 2e-9

# Do these tests in higher dimensions, check that OrthVF(PL) IsOnPlane....
# @test OrthVF(DM,XYPlane,p) == OrthVF(DM,p)

@test dot(OrthVF(DM,p),Score(DM,p)) < 1e-14
@test norm(FindMLE(DM) - [5.01511545953636, 1.4629658803705]) < 5e-10