using InformationGeometry, Test, BenchmarkTools, LinearAlgebra, Optim
import InformationGeometry: minimize, Minimize

# Test optimizers:
F(x) = x[1]^2 + 0.5x[2]^4;    initial = 10ones(2) + rand(2);    Cube = HyperCube(-2ones(2), 18ones(2))

@test norm(minimize(F, initial, Cube; tol=1e-5, meth=Optim.NelderMead())) < 5e-1
@test norm(minimize(F, initial, Cube; tol=1e-5, meth=Optim.LBFGS())) < 5e-2
@test norm(minimize(F, initial, Cube; tol=1e-5, meth=Optim.Newton())) < 5e-2

@test norm(minimize(F, initial; tol=1e-5, meth=Optim.NelderMead())) < 5e-1
@test norm(minimize(F, initial; tol=1e-5, meth=Optim.LBFGS())) < 5e-2
@test norm(minimize(F, initial; tol=1e-5, meth=Optim.Newton())) < 5e-2

@test norm(Minimize(F, initial, Cube; tol=1e-5, meth=Optim.LBFGS(), Multistart=5)) < 5e-2
@test norm(Minimize(F, initial; tol=1e-5, meth=Optim.LBFGS(), Multistart=5)) < 5e-2


using Optimization, OptimizationNLopt
@test norm(minimize(F, initial, Cube; tol=1e-5, meth=NLopt.LD_LBFGS())) < 5e-1
@test norm(minimize(F, initial; tol=1e-5, meth=NLopt.LD_LBFGS())) < 5e-1


using OptimizationOptimisers
@test norm(Prefit(F, initial, Cube; tol=1e-5, meth=[OptimizationOptimisers.OAdam(), NLopt.LD_LBFGS()], maxiters=[500,500])) < 5e-1
@test norm(Minimize(F, initial, Cube; MultistartFit=5, MinimizeFunc=Prefit, tol=1e-5, meth=[OptimizationOptimisers.OAdam(), NLopt.LD_LBFGS()], maxiters=[500,500])) < 5e-1


# Check optimization with non-linear constraints and box constraints

# Check in-place and out-of-place optimization

# Test RobustFit

# Check type stability of optimization
using ComponentArrays
@test minimize(X->X.A[1]^2 + 0.5X.B[1]^4, ComponentVector(A=[initial[1]], B=[initial[1]]); tol=1e-5, meth=Optim.Newton()) isa ComponentVector


Res = Vector{Float64}[]
@test AlternatingMinimization(x->10sum(sqrt∘abs, [1,2,3] .*x), [3,2,1.], (1:1, 2:3, 3:3); meth=Optim.GradientDescent(), tol=1e-6, maxiters=3, SavedParams=Res, verbose=false) isa AbstractVector
@test length(Res) > 0
@test AlternatingMinimization(x::ComponentVector->10sum(sqrt∘abs, [1,2,3] .*x), ComponentVector(x=[3,2,1.]), (1:2, 2:3, [1,3]); meth=Optim.GradientDescent(), tol=1e-6, maxiters=3, verbose=false) isa AbstractVector

@test PartialMinimization(x::ComponentVector->10sum(sqrt∘abs, [1,2,3.] .*x), ComponentVector(x=[3,2,1.]), [1,3]; meth=Optim.GradientDescent(), tol=1e-6, maxiters=3, verbose=false) isa AbstractVector



## Test constrained optimisation for method errors

using NonlinearSolveFirstOrder
DM = DataModel(DataSet([1,2,3,4], [4,5,6.5,9], [0.5,0.45,0.6,1]), PolynomialModel(2))

Sols, finalidxs = InformationGeometry.GenericLowerTriangular(DM; N=50, meth=TrustRegion(), Confnum=1, Refine=false)
Sols2, finalidxs2 = InformationGeometry.GenericLowerTriangularWithDecorrelation(DM; N=60, meth=TrustRegion(), Confnum=1, Refine=false)

Sols3, finalidxs3 = InformationGeometry.GenericLowerTriangular(DM; N=50, meth=IPNewton(), Confnum=1, Refine=false)
Sols4, finalidxs4 = InformationGeometry.GenericLowerTriangularWithDecorrelation(DM; N=50, meth=IPNewton(), Confnum=1, Refine=false)

@test all(x->all(isfinite, x), Sols[1])
@test all(x->all(isfinite, x), Sols2[1])
@test all(x->all(isfinite, x), Sols3[1])
@test all(x->all(isfinite, x), Sols4[1])

Sols, finalidxs = InformationGeometry.GenericLowerTriangular(DM; N=50, meth=TrustRegion(), Confnum=1, Refine=false, Multistart=10, maxval=5)
Sols2, finalidxs2 = InformationGeometry.GenericLowerTriangularWithDecorrelation(DM; N=50, meth=TrustRegion(), Confnum=1, Refine=false, Multistart=10, maxval=5)

Sols3, finalidxs3 = InformationGeometry.GenericLowerTriangular(DM; N=50, meth=IPNewton(), Confnum=1, Refine=false, Multistart=10, maxval=5)
Sols4, finalidxs4 = InformationGeometry.GenericLowerTriangularWithDecorrelation(DM; N=50, meth=IPNewton(), Confnum=1, Refine=false, Multistart=10, maxval=5)

@test all(x->all(isfinite, x), Sols[1])
@test all(x->all(isfinite, x), Sols2[1])
@test all(x->all(isfinite, x), Sols3[1])
@test all(x->all(isfinite, x), Sols4[1])