using InformationGeometry, Test, BenchmarkTools, LinearAlgebra, Optim
import InformationGeometry.minimize

# Test optimizers:
F(x) = x[1]^2 + 0.5x[2]^4;    initial = 10ones(2) + rand(2);    Cube = HyperCube(-2ones(2), 18ones(2))

@test norm(minimize(F, initial, Cube; tol=1e-5, meth=Optim.NelderMead())) < 5e-1
@test norm(minimize(F, initial, Cube; tol=1e-5, meth=Optim.LBFGS())) < 5e-2
@test norm(minimize(F, initial, Cube; tol=1e-5, meth=Optim.Newton())) < 5e-2

@test norm(minimize(F, initial; tol=1e-5, meth=Optim.NelderMead())) < 5e-1
@test norm(minimize(F, initial; tol=1e-5, meth=Optim.LBFGS())) < 5e-2
@test norm(minimize(F, initial; tol=1e-5, meth=Optim.Newton())) < 5e-2



using Optimization, OptimizationNLopt
@test norm(minimize(F, initial, Cube; tol=1e-5, meth=NLopt.LD_LBFGS())) < 5e-1
@test norm(minimize(F, initial; tol=1e-5, meth=NLopt.LD_LBFGS())) < 5e-1


# Check optimization with non-linear constraints and box constraints

# Check in-place and out-of-place optimization

# Test RobustFit

# Check type stability of optimization
using ComponentArrays
@test minimize(X->X.A[1]^2 + 0.5X.B[1]^4, ComponentVector(A=[initial[1]], B=[initial[1]]); tol=1e-5, meth=Optim.Newton()) isa ComponentVector