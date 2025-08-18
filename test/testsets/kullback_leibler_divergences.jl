using InformationGeometry, Test, LinearAlgebra, Distributions

# Analytical divergences via types defined in Distributions.jl
@test KullbackLeibler(MvNormal([1,2,3.],diagm([2,4,5.])),MvNormal([1,-2,-3.],diagm([1,5,2.]))) ≈ 11.056852819440055
@test KullbackLeibler(Normal(1,2),Normal(5,1)) ≈ 8.806852819440055
@test KullbackLeibler(Cauchy(1,3),Cauchy(5,2)) ≈ 0.5355182363563621
@test KullbackLeibler(Exponential(5),Exponential(1)) ≈ 2.3905620875658995
@test KullbackLeibler(Weibull(12,2),Weibull(3,4)) ≈ 2.146124463755512
@test KullbackLeibler(Gamma(5,15),Gamma(20,1)) ≈ 29.409061308330323


# Numerically calculated for via arbitrary types defined in Distributions.jl
# ALSO ADD TESTS FOR DISCRETE DISTRIBUTIONS, DISTRIBUTIONS WITH LIMITED DOMAIN
@test abs(KullbackLeibler(Cauchy(1,2),Normal(5,1),HyperCube([-20,20]); tol=1e-8) - 16.77645704773449) < 1e-5
@test abs(KullbackLeibler(Cauchy(1,2),Normal(5,1),HyperCube([-20,20]); Carlo=true, N=Int(3e6)) - 16.7764) < 5e-2
@test abs(KullbackLeibler(MvTDist(1,[3,2,1.],diagm([1.,2.,3.])),MvNormal([1,2,3.],diagm([2,4,5.])),HyperCube([[-10,10.] for i in 1:3]); Carlo=true, N=Int(3e6)) - 1.6559288) < 3e-1

# Product distributions, particularly Normal and Cauchy
P = [Normal(0,1), Cauchy(1,2)] |> product_distribution
Q = [Cauchy(1,1), Cauchy(2,4)] |> product_distribution
R = [Normal(2,4), Normal(-1,0.5)] |> product_distribution
@test abs(KullbackLeibler(P, Q, HyperCube([[-20,20] for i in 1:2]); tol=1e-7) - 0.719771180) < 1e-8
@test abs(KullbackLeibler(R, P, HyperCube([[-20,20] for i in 1:2]); tol=1e-7) - 9.920379769) < 1e-8
@test abs(KullbackLeibler(P, R, HyperCube([[-20,20] for i in 1:2]); tol=1e-7) - 48.99179438) < 1e-8

# Via any positive (hopefully normalized) functions
@test abs(KullbackLeibler(x->pdf(Normal(1,3),x),y->pdf(Normal(5,2),y),HyperCube([-20,20]); Carlo=true, N=Int(3e6)) - KullbackLeibler(Normal(1,3),Normal(5,2))) < 2e-2
@test abs(KullbackLeibler(x->pdf(Normal(1,3),x),y->pdf(Normal(5,2),y),HyperCube([-20,20]); tol=1e-8) - KullbackLeibler(Normal(1,3),Normal(5,2))) < 1e-5
P = MvNormal([1,2,3.],diagm([1,2,1.5]));    Q = MvNormal([1,-2,-3.],diagm([2,1.5,1.]));     Cube = HyperCube([[-15,15] for i in 1:3])
@test abs(KullbackLeibler(x->pdf(P,x),y->pdf(Q,y),Cube; Carlo=true, N=Int(3e6)) - KullbackLeibler(x->pdf(P,x),y->pdf(Q,y),Cube; tol=1e-8)) < 0.8