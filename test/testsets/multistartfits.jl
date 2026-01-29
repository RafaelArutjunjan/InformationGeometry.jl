using InformationGeometry, Test, Distributions, LinearAlgebra, Optim, LsqFit

DM = DataModel(DataSet(1:4, [4,5,6.5,9], [0.5,0.45,0.6,1]), (x,p)->(p[1]+p[2])*x + exp(p[1]-p[2]), [1.3, 0.2]; SkipOptim=true)

R = MultistartFit(DM; N=20, maxval=10)
R2 = MultistartFit(DM, MvNormal([0,0], Diagonal(ones(2))); MultistartDomain=HyperCube([-1,-1],[3,4]), N=10)
R3 = MultistartFit(InformationGeometry.Negloglikelihood(DM), MvNormal([0.], Diagonal(ones(1))); MultistartDomain=HyperCube([-1],[3]), N=10,
                    TransformSample=X->[X; 0.])
@test sum(abs, MLE(R)-MLE(R2)) < 1e-8
@test sum(abs, MLE(R)-MLE(R3)) < 1e-8
@test sum(abs, MLE(LocalMultistartFit(DM, MLE(R); N=5)) -MLE(R)) < 1e-8

@test sum(abs, IncrementalTimeSeriesFit(DM, MLE(R))-MLE(R)) < 1e-8

@assert SubsetMultistartResults(R, 10) isa InformationGeometry.MultistartResults


DMU = DataModel(DataSetUncertain(1:3, [4,5,6.5]; verbose=false), (x,p)->(p[1]+p[2])*x + exp(p[1]-p[2]), [1.3, 0.2, -0.5]; SkipOptim=true)

RU = MultistartFit(DMU; N=20, maxval=10)
RU2 = MultistartFit(DMU, MvNormal([0,0,0], Diagonal(ones(3))); MultistartDomain=HyperCube([-1,-1,-5],[3,4,1]), N=10)
RU3 = MultistartFit(InformationGeometry.Negloglikelihood(DMU), MvNormal([0.,0], Diagonal(ones(2))); MultistartDomain=HyperCube([-1,-1],[3,4]), N=10,
                    TransformSample=X->[X; -0.5])
@test sum(abs, MLE(RU)-MLE(RU2)) < 1e-8
@test sum(abs, MLE(RU)-MLE(RU3)) < 1e-8
@test sum(abs, MLE(LocalMultistartFit(DMU, MLE(RU); N=5)) -MLE(RU)) < 1e-8

@test sum(abs, IncrementalTimeSeriesFit(DMU, MLE(RU))-MLE(RU)) < 1e-8