
using SafeTestsets



@safetestset "Probability Objects" begin
    using InformationGeometry, Test, LinearAlgebra, Distributions

    DS = DataSet([0,0.5,1,1.5],[1.,3.,7.,8.1],[1.2,2.,0.6,1.])
    model(x,θ) = θ[1] * x + θ[2]
    DM = DataModel(DS,model)
    p = rand(2)

    @test IsLinear(DM)
    Dist = DataDist(ydata(DM),sigma(DM))
    @test abs(loglikelihood(DM,p) - logpdf(Dist,EmbeddingMap(DM,p))) < 1e-13
    @test Score(DM,p) ≈ transpose(EmbeddingMatrix(DM,p)) * gradlogpdf(Dist,EmbeddingMap(DM,p))
    @test FisherMetric(DM,p) ≈ transpose(EmbeddingMatrix(DM,p)) * inv(cov(Dist)) * EmbeddingMatrix(DM,p)

    # Test AD vs manual derivative
    @test sum(abs.(AutoScore(DM,p) - Score(DM,p))) < 2e-13
    @test sum(abs.(AutoMetric(DM,p) - FisherMetric(DM,p))) < 2e-9

    # Do these tests in higher dimensions, check that OrthVF(PL) IsOnPlane....
    # @test OrthVF(DM,XYPlane,p) == OrthVF(DM,p)

    @test dot(OrthVF(DM,p),Score(DM,p)) < 6e-15
    @test sum(abs.(FindMLE(DM) - [5.01511545953636, 1.4629658803705])) < 5e-10
    # ALSO DO NONLINEAR MODEL!
end

@safetestset "Confidence Regions" begin
    using InformationGeometry, Test, Plots

    DS = DataSet([0,0.5,1,1.5],[1.,3.,7.,8.1],[1.2,2.,0.6,1.])
    model(x,θ) = θ[1] * x + θ[2]
    DM = DataModel(DS,model)
    DME = DataModel(DataSetExact([0,0.5,1,1.5],0.1ones(4),[1.,3.,7.,8.1],[1.2,2.,0.6,1.]), model)

    sols = ConfidenceRegions(DM,1:2; tol=1e-6)
    @test IsStructurallyIdentifiable(DM,sols[1]) == true
    @test size(SaveConfidence(sols,50)) == (50,4)
    @test size(SaveGeodesics(sols,50)) == (50,2)
    @test size(SaveDataSet(DM)) == (4,3)
    @test ConfidenceRegionVolume(DM,sols[1];N=5000) < ConfidenceRegionVolume(DM,sols[2];N=5000,WE=true)

    @test size(ConfidenceBands(DM,sols[1]; N=50, plot=false)) == (50,3)
    @test size(PlotMatrix(inv(FisherMetric(DM,MLE(DM))),MLE(DM); N=50,plot=false)) == (50,2)
    @test typeof(FittedPlot(DM)) <: Plots.Plot
    @test typeof(FittedPlot(DME)) <: Plots.Plot
    @test typeof(ResidualPlot(DM)) <: Plots.Plot

    @test typeof(VisualizeGeos([MBAM(DM)])) <: Plots.Plot
    simplermodel(x,p) = p[1]*x;    DMSimp = DataModel(DS,simplermodel)
    @test length(ConfidenceRegion(DMSimp,1.)) == 2
    @test ModelComparison(DM,DMSimp)[2] > 0.

    @test FindFBoundary(DM,1)[1] - FindConfBoundary(DM,1)[1] > 0

    # @test ConfidenceRegions(BigFloat(DM), 1:2; tol=1e-6)
end

@safetestset "More Boundary tests" begin
    using InformationGeometry, Test, Random, Distributions, OrdinaryDiffEq

    Random.seed!(31415);    normerr(sig::Number) = rand(Normal(0,sig));     quarticlin(x,p) = p[1]*x.^4 .+ p[2]
    X = collect(0:0.2:3);   err = 2. .+ 2sqrt.(X);      Y = quarticlin(X,[1,8.]) + normerr.(err)
    ToyDME = DataModel(DataSetExact(X,0.1ones(length(X)),Y,err), (x,p) -> 15p[1]^3 * x.^4 .+ p[2]^5)

    @test InterruptedConfidenceRegion(ToyDME, 8.5; tol=1e-9) isa ODESolution

    NewX, NewP = TotalLeastSquares(ToyDME)
    @test LogLike(Data(ToyDME), NewX, EmbeddingMap(Data(ToyDME),Predictor(ToyDME),NewP,NewX)) > loglikelihood(ToyDME, MLE(ToyDME))

    @test ModelMap(Predictor(ToyDME), PositiveDomain(2)) isa ModelMap

    sol = ConfidenceRegion(ToyDME,1; tol=1e-6)
    @test ApproxInRegion(sol, MLE(ToyDME)) && !ApproxInRegion(sol, sol.u[1] + 1e-5BasisVector(1,2))
end

@safetestset "Model Transformations" begin
    using InformationGeometry, Test

    PiDM = DataModel(DataSet([0,1], [0.5pi,1.5pi], [0.5,0.5]), ModelMap((x,p)->p[1], θ->θ[1]>1, HyperCube([[0,5]])))
    @test !PiDM.model.InDomain([0.9]) && PiDM.model.InDomain([1.1])

    # Translation
    PiDM2 = DataModel(Data(PiDM), TranslationTransform(Predictor(PiDM),[1.]))
    @test !PiDM2.model.InDomain([-0.1]) && PiDM2.model.InDomain([0.1])

    # LogTransform
    PiDM3 = DataModel(Data(PiDM), LogTransform(Predictor(PiDM),trues(1)))
    @test !PiDM3.model.InDomain(exp.([1])-[0.1]) && PiDM3.model.InDomain(exp.([1])+[0.1])

    # Does Score / FisherMetric and AutoDiff still work?
end


@safetestset "Inputting Datasets of various shapes" begin
    using InformationGeometry, Test, LinearAlgebra, Random, Distributions, StaticArrays, Plots

    ycovtrue = [1.0 0.1 -0.5; 0.1 2.0 0.0; -0.5 0.0 3.0]
    ptrue = [1.,pi,-5.];        ErrorDistTrue = MvNormal(zeros(3),ycovtrue)

    model(x::AbstractVector{<:Number},p::AbstractVector{<:Number}) = SA[p[1] * x[1]^2 + p[3]^3 * x[2],
                                                        sinh(p[2]) * (x[1] + x[2]), exp(p[1]*x[1] + p[1]*x[2])]
    Gen(t) = float.([t,0.5t^2]);    Xdata = Gen.(0.5:0.1:3)
    Ydata = [model(x,ptrue) + rand(ErrorDistTrue) for x in Xdata]
    Sig = BlockMatrix(ycovtrue,length(Ydata));    DS = DataSet(Xdata,Ydata,Sig)
    DM = DataModel(DS,model)
    @test norm(MLE(DM) - ptrue) < 5e-2
    DME = DataModel(DataSetExact(DS), model)
    P = MLE(DM) + rand(length(MLE(DM)))
    @test loglikelihood(DM,P) ≈ loglikelihood(DME,P)
    @test Score(DM,P) ≈ Score(DME,P)

    Planes, sols = ConfidenceRegion(DM,1)
    @test typeof(VisualizeSols(Planes,sols)) <: Plots.Plot

    ODM = OptimizedDM(DME)
    @test sum(abs.(EmbeddingMatrix(DME,MLE(DME)) - EmbeddingMatrix(ODM,MLE(DME)))) < 1e-9
end


# Test DataSets of different ydim and xdim and non-linear models.


@safetestset "Kullback-Leibler Divergences" begin
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
    @test abs(KullbackLeibler(x->pdf(P,x),y->pdf(Q,y),Cube; Carlo=true, N=Int(3e6)) - KullbackLeibler(x->pdf(P,x),y->pdf(Q,y),Cube; tol=1e-8)) < 7e-1
end


@safetestset "Differential Geometry" begin
    using InformationGeometry, Test

    function S2metric(θ,ϕ)
        metric = zeros(suff(ϕ),2,2);    metric[1,1] = 1.;    metric[2,2] = sin(θ)^2
        metric
    end
    S2metric(p::Vector) = S2metric(p...)

    function S2Christoffel(θ,ϕ)
        Symbol = zeros(suff(ϕ),2,2,2);    Symbol[1,2,2] = -sin(θ)*cos(θ)
        Symbol[2,1,2] = cot(θ);    Symbol[2,2,1] = cot(θ)
        Symbol
    end
    S2Christoffel(p::Vector) = S2Christoffel(p...)
    ConstMetric(x) = [1. 0.; 0. 1.]

    # Test Numeric Christoffel Symbols, Riemann and Ricci tensors, Ricci Scalar
    # Test WITH AND WITHOUT BIGFLOAT
    x = rand(2)
    @test sum(abs.(ChristoffelSymbol(S2metric,x) .- S2Christoffel(x))) < 5e-9
    @test sum(abs.(ChristoffelSymbol(S2metric,BigFloat.(x)) .- S2Christoffel(BigFloat.(x)))) < 1e-40

    @test abs(RicciScalar(S2metric,rand(2)) - 2) < 5e-4
    @test abs(RicciScalar(S2metric,rand(BigFloat,2)) - 2) < 2e-22

    @test abs(GeodesicDistance(ConstMetric,[0,0],[1,1]) - sqrt(2)) < 1e-13
    @test abs(GeodesicDistance(S2metric,[pi/4,1],[3pi/4,1]) - pi/2) < 1e-11
    @test abs(GeodesicDistance(S2metric,[pi/2,0],[pi/2,pi/2]) - pi/2) < 3e-10

    DS = DataSet([0,0.5,1],[1.,3.,7.],[1.2,2.,0.6])
    model(x,p) = p[1]^3 *x + p[2]^3;        DM = DataModel(DS,model)
    geo = GeodesicBetween(DM,MLE(DM),MLE(DM) + rand(2); tol=1e-11)
    @test sum(abs.(MLE(DM) .- [1.829289173660125,0.942865200406147])) < 1e-7
    @test abs(InformationGeometry.ParamVol(geo) * InformationGeometry.GeodesicEnergy(DM,geo) - GeodesicLength(DM,geo)^2) < 1e-8
end


@safetestset "Numerical Helper Functions" begin
    using InformationGeometry, Test

    # Compare Integrate1D and IntegrateND

    # Test integration, differentiation, Monte Carlo, GeodesicLength
    # TEST WITH AND WITHOUT BIGFLOAT
    @test abs(InformationGeometry.MonteCarloArea(x->((x[1]^2 + x[2]^2) < 1), HyperCube([[-1,1],[-1,1]])) - pi) < 1.5e-3
    @test abs(Integrate1D(cos, (0,pi/2); tol=1e-12) - IntegrateND(cos, (0,pi/2); tol=1e-12)) < 1e-10
    z = 3rand()
    @test abs(Integrate1D(x->2/sqrt(pi) * exp(-x^2), [0,z/sqrt(2)]) - ConfVol(z)) < 1e-12
    @test abs(LineSearch(x->(x < BigFloat(pi))) - pi) < 1e-14
    @test abs(CubeVol(TranslateCube(HyperCube([[0,1],[0,pi],[-sqrt(2),0]]),rand(3))) - sqrt(2)*pi) < 3e-15

    k = rand(1:20);     r = 10rand()
    @test InvChisqCDF(k,Float64(ChisqCDF(k,r))) ≈ r

    # Test invert() method for Float64 and BigFloat
end
