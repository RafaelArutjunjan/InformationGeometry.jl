using InformationGeometry, Test

@test abs(InformationGeometry.MonteCarloArea(x->((x[1]^2 + x[2]^2) < 1), HyperCube([[-1,1],[-1,1]])) - π) < 2e-3
@test abs(Integrate1D(cos, (0,π/2); tol=1e-12) - IntegrateND(cos, (0,π/2); tol=1e-12)) < 1e-10
z = 3rand()
@test abs(Integrate1D(x->2/sqrt(π) * exp(-x^2), [0,z/sqrt(2)]) - ConfVol(z)) < 1e-12
@test abs(LineSearch(x->(x < BigFloat(π))) - π) < 1e-14
@test abs(LineSearch(x->(x < BigFloat(π)), BigFloat(1e-14); tol=1e-30) - BigFloat(π)) < 1e-25
@test abs(CubeVol(TranslateCube(HyperCube([[0,1],[0,π],[-sqrt(2),0]]),rand(3))) - sqrt(2)*π) < 3e-15

k = rand(1:20);     r = 10rand()
@test InvChisqCDF(k,Float64(ChisqCDF(k,r))) ≈ r
@test abs(InvChisqCDF(k,ChisqCDF(k,BigFloat(r)); tol=1e-20) - r) < 1e-18

F = InformationGeometry.MergeOneArgMethods(nothing, (A::AbstractMatrix, x::AbstractVector)->(A[1:3,1] = x))
G = InformationGeometry.MergeOneArgMethods(x->2.0*x, nothing)
X = rand(3);   J = rand(3)
@test all(F(X) .== X)
@test (G(J,X);    J == 2X)