using InformationGeometry, Test, Random, Distributions, SciMLBase, OrdinaryDiffEq, LinearAlgebra

Random.seed!(31415);    normerr(sig::Number) = rand(Normal(0,sig));     quarticlin(x,p) = p[1]*x.^4 .+ p[2]
X = collect(0:0.2:3);   err = 2. .+ 2sqrt.(X);      Y = quarticlin(X,[1,8.]) + normerr.(err)
ToyDME = DataModel(DataSetExact(X,0.1ones(length(X)),Y,err), (x,p) -> 15p[1]^3 * x.^4 .+ p[2]^5)

@test InterruptedConfidenceRegion(BigFloat(ToyDME), 8; tol=1e-5) isa SciMLBase.AbstractODESolution
@test InterruptedConfidenceRegion(BigFloat(ToyDME), 8.5; tol=1e-5) isa SciMLBase.AbstractODESolution

NewX, NewP = TotalLeastSquares(ToyDME)
@test LogLike(Data(ToyDME), NewX, EmbeddingMap(Data(ToyDME),Predictor(ToyDME),NewP,NewX)) > loglikelihood(ToyDME, MLE(ToyDME))

@test ModelMap(Predictor(ToyDME), PositiveDomain(2)) isa ModelMap

sol = ConfidenceRegion(ToyDME, 1; tol=1e-6)
@test ApproxInRegion(sol, MLE(ToyDME)) && !ApproxInRegion(sol, sol.u[1] + 1e-5BasisVector(1,2))

#Check that bounding box from ProfileLikelihood coincides roughly with exact box.
Mats = ParameterProfiles(ToyDME, 2; IsCost=false, N=60, plot=false)
Mats2 = ParameterProfiles(ToyDME, 2; IsCost=true, N=60, plot=false)
Mats3 = ParameterProfiles(ToyDME, 2; adaptive=false, IsCost=true, N=60, plot=false)
ProfBox = ProfileBox(Mats, 1)
ProfBox2 = ProfileBox(Mats2, 1)
ProfBox3 = ProfileBox(Mats3, 1)
ExactBox = ConstructCube(sol)
@test norm(Center(ProfBox) - Center(ExactBox)) < 3e-5 && norm(CubeWidths(ProfBox) - CubeWidths(ExactBox)) < 3e-4
@test norm(Center(ProfBox2) - Center(ExactBox)) < 3e-5 && norm(CubeWidths(ProfBox2) - CubeWidths(ExactBox)) < 3e-4
@test norm(Center(ProfBox3) - Center(ExactBox)) < 3e-5 && norm(CubeWidths(ProfBox3) - CubeWidths(ExactBox)) < 3e-4
@test 0 < PracticallyIdentifiable(Mats) < PracticallyIdentifiable(Mats2) < 10

# Test rescaling method for confidence boundary generation
sol2b = InformationGeometry.GenerateBoundary2(ToyDME, sol.u[1]; tol=1e-4, Embedded=false)
sol2e = InformationGeometry.GenerateBoundary2(ToyDME, sol.u[1]; tol=1e-4, Embedded=true)
Cb, Ce = ConstructCube(sol2b), ConstructCube(sol2e)
@test norm(Center(Ce) - Center(ExactBox)) < 2e-4 && norm(CubeWidths(Ce) - CubeWidths(ExactBox)) < 2e-4
@test norm(Center(Cb)) < 0.3 && norm(CubeWidths(Cb)) < 5

# Method for general cost functions on 2D domains
sol = GenerateBoundary(x->-norm(x,1.5), [1., 0])
@test 0.23 ≤ length(GenerateBoundary(x->-norm(x,1.5), [1., 0]; Boundaries=(u,t,int)->u[1]<0.).u) / length(sol.u) ≤ 0.27