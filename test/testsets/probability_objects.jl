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


## Test ADmode control
using ForwardDiff, Optim, Zygote
DisallowAD(F::Function) = (x,p; kwargs...)->(@assert !isa(p, AbstractVector{<:ForwardDiff.Dual}) "Got Dual Number in Model";    F(x,p; kwargs...))
NoADmodel = DisallowAD(LinearModel)
# Check that model indeed errors on Dual
@test_broken ForwardDiff.jacobian(x->NoADmodel(2, x), rand(2))

# Works because using dmodel directly via EmbeddingMatrix in LsqFit.jl
@test DataModel(DS, NoADmodel; ADmode=Val(:Symbolic), meth=nothing) isa DataModel

# Use ADmodeOptim kwarg if FiniteDifferences not loaded
@test DataModel(DS, NoADmodel; ADmode=Val(:Symbolic), ADmodeOptim=Val(:Zygote), meth=Newton()) isa DataModel
## Default ForwardDiff should not work here!
@test_broken DataModel(DS, NoADmodel; ADmode=Val(:Symbolic), meth=Newton()) isa DataModel
using FiniteDifferences
# Should work now since default ADmodeOptim switched to FiniteDifferences now
@test DataModel(DS, NoADmodel; ADmode=Val(:Symbolic), meth=Newton()) isa DataModel

@test DataModel(DS, NoADmodel; ADmode=Val(:FiniteDifferences)) isa DataModel
@test DataModel(DS, NoADmodel; ADmode=Val(:Zygote), meth=Newton()) isa DataModel

@test Score(DS, NoADmodel, DetermineDmodel(DS, NoADmodel; ADmode=Val(:Zygote)), [1,2.], nothing; ADmode=Val(:Zygote)) isa AbstractVector
@test Score(DS, NoADmodel, DetermineDmodel(DS, NoADmodel; ADmode=Val(:Zygote)), [1,2.], nothing; ADmode=Val(false)) isa AbstractVector