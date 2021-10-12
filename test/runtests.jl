
using SafeTestsets



@safetestset "Probability Objects" begin
    using InformationGeometry, Test, LinearAlgebra, Distributions

    DS = DataSet([0,0.5,1,1.5],[1.,3.,7.,8.1],[1.2,2.,0.6,1.])
    model(x,θ) = θ[1] * x + θ[2]
    DM = DataModel(DS,model)
    p = rand(2)

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
end


@safetestset "Confidence Regions" begin
    using InformationGeometry, Test, Plots

    DS = DataSet([0,0.5,1,1.5],[1.,3.,7.,8.1],[1.2,2.,0.6,1.])
    model(x,θ) = θ[1] * x + θ[2];    DM = DataModel(DS,model)
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
end


@safetestset "More Boundary tests" begin
    using InformationGeometry, Test, Random, Distributions, OrdinaryDiffEq, LinearAlgebra

    Random.seed!(31415);    normerr(sig::Number) = rand(Normal(0,sig));     quarticlin(x,p) = p[1]*x.^4 .+ p[2]
    X = collect(0:0.2:3);   err = 2. .+ 2sqrt.(X);      Y = quarticlin(X,[1,8.]) + normerr.(err)
    ToyDME = DataModel(DataSetExact(X,0.1ones(length(X)),Y,err), (x,p) -> 15p[1]^3 * x.^4 .+ p[2]^5)

    @test InterruptedConfidenceRegion(BigFloat(ToyDME), 8.5; tol=1e-5) isa ODESolution

    NewX, NewP = TotalLeastSquares(ToyDME)
    @test LogLike(Data(ToyDME), NewX, EmbeddingMap(Data(ToyDME),Predictor(ToyDME),NewP,NewX)) > loglikelihood(ToyDME, MLE(ToyDME))

    @test ModelMap(Predictor(ToyDME), PositiveDomain(2)) isa ModelMap

    sol = ConfidenceRegion(ToyDME,1; tol=1e-6)
    @test ApproxInRegion(sol, MLE(ToyDME)) && !ApproxInRegion(sol, sol.u[1] + 1e-5BasisVector(1,2))

    #Check that bounding box from ProfileLikelihood coincides roughly with exact box.
    Mats = ProfileLikelihood(ToyDME,2; plot=false)
    ProfBox = ProfileBox(ToyDME, InterpolatedProfiles(Mats),1)
    ExactBox = ConstructCube(ConfidenceRegion(ToyDME,1; tol=1e-6))
    @test norm(Center(ProfBox) - Center(ExactBox)) < 1e-5
    @test norm(CubeWidths(ProfBox) - CubeWidths(ExactBox)) < 3e-4
    @test 0 < PracticallyIdentifiable(Mats) < 2

    # Method for general cost functions on 2D domains
    sol = GenerateBoundary(x->-norm(x,1.5), [1., 0])
    @test 0.23 ≤ length(GenerateBoundary(x->-norm(x,1.5), [1., 0]; Boundaries=(u,t,int)->u[1]<0.).u) / length(sol.u) ≤ 0.27
end


@safetestset "ODE-based models" begin
    using InformationGeometry, Test, OrdinaryDiffEq, LinearAlgebra

    function SIR!(du,u,p,t)
        S, I, R = u
        β, γ = p
        du[1] = -β * I * S
        du[2] = +β * I * S - γ * I
        du[3] = +γ * I
        nothing
    end
    SIRsys = ODEFunction(SIR!)
    infected = [3, 8, 28, 75, 221, 291, 255, 235, 190, 126, 70, 28, 12, 5]
    SIRDS = InformNames(DataSet(collect(1:14), infected, 5ones(14)), ["Days"], ["Infected"])

    SIRinitial = X->([763.0-X[1], X[1], 0.0], X[2:3])

    # Use SplitterFunction SIRinitial to infer initial condition I₀ as first parameter
    SIRDM = DataModel(SIRDS, SIRsys, SIRinitial, x->x[2], [0.6,0.0023,0.46]; tol=1e-6)
    @test SIRDM isa DataModel
    @test DataModel(SIRDS, SIRsys, [762, 1, 0.], [2], [0.0022,0.45], true; meth=Tsit5(), tol=1e-6) isa DataModel
    @test norm(2*EmbeddingMap(Data(SIRDM), Predictor(SIRDM), MLE(SIRDM)) - EmbeddingMap(Data(SIRDM), ModifyODEmodel(SIRDM, x->2*x[2]), MLE(SIRDM))) < 2e-4
end


@safetestset "Model Transformations" begin
    using InformationGeometry, Test

    PiDM = DataModel(DataSet([0,1], [0.5π,1.5π], [0.5,0.5]), ModelMap((x,p)->p[1], θ->θ[1]-1, HyperCube([[0,5]])))
    @test !IsInDomain(Predictor(PiDM), [0.9]) && IsInDomain(Predictor(PiDM), [1.1])

    # Translation
    PiDM2 = DataModel(Data(PiDM), TranslationTransform(Predictor(PiDM),[1.]))
    @test !IsInDomain(Predictor(PiDM2), [-0.1]) && IsInDomain(Predictor(PiDM2), [0.1])

    # LogTransform
    PiDM3 = DataModel(Data(PiDM), LogTransform(Predictor(PiDM),trues(1)))
    @test !IsInDomain(Predictor(PiDM3), exp.([1])-[0.1]) && IsInDomain(Predictor(PiDM3), exp.([1])+[0.1])

    DS = DataSet([0,0.5,1,1.5],[1.,3.,7.,8.1],[1.2,2.,0.6,1.])
    @test FisherMetric(LinearDecorrelation(DataModel(DS, (x,θ)->θ[1] * x + θ[2])), zeros(2)) ≈ [1 0; 0 1]

    # TranstrumModel = ModelMap((x::Real,p::AbstractVector)->exp(-p[1]*x) + exp(-p[2]*x), θ::AbstractVector -> θ[1]>θ[2], PositiveDomain(2, 1e2), (1,1,2))
    # TranstrumDM = DataModel(DataSet([0.33, 1, 3], [0.88,0.5,0.35], [0.1,0.3,0.2]), TranstrumModel)
    # linTranstrum = LogTransform(TranstrumDM)
    # RicciScalar(TranstrumDM, MLE(TranstrumDM)), RicciScalar(linTranstrum, MLE(linTranstrum))
    # loglikelihood(TranstrumDM, MLE(TranstrumDM)), loglikelihood(linTranstrum, MLE(linTranstrum))

    # Try with normal functions too, not only ModelMaps.
    # Try Ricci in particular, maybe as BigFloat.

    # Does Score / FisherMetric and AutoDiff still work?
end


@safetestset "Inputting Datasets of various shapes" begin
    using InformationGeometry, Test, LinearAlgebra, Random, Distributions, StaticArrays, Plots

    ycovtrue = Diagonal([1,2,3]) |> x->convert(Matrix,x)
    ptrue = [1.,π,-5.];        ErrorDistTrue = MvNormal(zeros(3),ycovtrue)

    model(x::AbstractVector{<:Number},p::AbstractVector{<:Number}) = SA[p[1] * x[1]^2 + p[3]^3 * x[2],
                                                        sinh(p[2]) * (x[1] + x[2]), exp(p[1]*x[1] + p[1]*x[2])]
    Gen(t) = float.([t,0.5t^2]);    Xdata = Gen.(0.5:0.1:3)
    Ydata = [model(x,ptrue) + rand(ErrorDistTrue) for x in Xdata]
    Sig = BlockMatrix(ycovtrue,length(Ydata));    DS = DataSet(Xdata,Ydata,Sig)
    DM = DataModel(DS,model)
    @test norm(MLE(DM) - ptrue) < 5e-2
    DME = DataModel(DataSetExact(DS), model)
    P = MLE(DM) + 0.5rand(length(MLE(DM)))
    @test loglikelihood(DM,P) ≈ loglikelihood(DME,P)
    @test Score(DM,P) ≈ Score(DME,P)

    Planes, sols = ConfidenceRegion(DM,1)
    @test typeof(VisualizeSols(Planes,sols)) <: Plots.Plot

    ODM = OptimizedDM(DME)
    @test norm(EmbeddingMatrix(DME,MLE(DME)) .- EmbeddingMatrix(ODM,MLE(DME)), 1) < 1e-9

    CDM = DataModel(CompositeDataSet(Data(ODM)), Predictor(ODM), dPredictor(ODM), MLE(ODM))
    @test abs(loglikelihood(ODM, P) - loglikelihood(CDM, P)) < 5e-6
    @test norm(Score(ODM, P) - Score(CDM, P)) < 1e-6
    @test norm(FisherMetric(ODM, P) - FisherMetric(CDM, P)) < 1e-6
    @test norm(InformationGeometry.ResidualStandardError(ODM) - InformationGeometry.ResidualStandardError(CDM)) < 1e-10

    lastDS = Data(Data(CDM))[3]
    newCDS = vcat(Data(Data(CDM))[1:end-1], [SubDataSet(lastDS, 1:2:Npoints(lastDS))], [SubDataSet(lastDS, 2:2:Npoints(lastDS))]) |> CompositeDataSet
    # repeat last component
    newmodel(x::AbstractVector{<:Number},p::AbstractVector{<:Number}) = SA[p[1] * x[1]^2 + p[3]^3 * x[2], sinh(p[2]) * (x[1] + x[2]),
                                                        exp(p[1]*x[1] + p[1]*x[2]), exp(p[1]*x[1] + p[1]*x[2])]
    splitCDM = DataModel(newCDS, newmodel, MLE(CDM))
    @test abs(loglikelihood(splitCDM, P) - loglikelihood(CDM, P)) < 1e-5
    @test norm(Score(splitCDM, P) - Score(CDM, P)) < 2e-4
    @test norm(FisherMetric(splitCDM, P) - FisherMetric(CDM, P)) < 2e-3
end


@safetestset "Priors" begin
    using InformationGeometry, Test, LinearAlgebra, Distributions

    DS1 = DataSet([0,0.5],[1.,3.],[1.2,2.]);    DS2 = DataSet([1,1.5],[7.,8.1],[0.6,1.])
    DS = join(DS1, DS2);    model(x,θ) = θ[1] * x + θ[2];
    DM1 = DataModel(DS1,model);     DM = DataModel(DS,model);

    logprior(X) = loglikelihood(DM1, X)

    DM12 = DataModel(DS2, model, MLE(DM1), logprior)
    @test norm(MLE(DM) - MLE(DM12)) < 1e-6
    @test loglikelihood(DM, MLE(DM)) ≈ loglikelihood(DM12, MLE(DM))
    @test norm(Score(DM, MLE(DM)) - Score(DM12, MLE(DM))) < 1e-12
    @test FisherMetric(DM, MLE(DM)) ≈ FisherMetric(DM12, MLE(DM))

    dm = DataModel(DS, model, MLE(DM), x->0.0)
    @test loglikelihood(DM, MLE(DM)) ≈ loglikelihood(dm, MLE(DM))
    @test Score(DM, MLE(DM)) ≈ Score(dm, MLE(DM))
    @test FisherMetric(DM, MLE(DM)) ≈ FisherMetric(dm, MLE(DM))
end

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
    @test abs(KullbackLeibler(x->pdf(P,x),y->pdf(Q,y),Cube; Carlo=true, N=Int(3e6)) - KullbackLeibler(x->pdf(P,x),y->pdf(Q,y),Cube; tol=1e-8)) < 0.8
end


@safetestset "Differential Geometry" begin
    using InformationGeometry, Test, LinearAlgebra, StaticArrays

    S2metric((θ,ϕ)) = Diagonal([1.0, sin(θ)^2])
    function S2Christoffel((θ,ϕ))
        Symbol = zeros(suff(ϕ),2,2,2);    Symbol[1,2,2] = -sin(θ)*cos(θ)
        Symbol[2,1,2] = cot(θ);    Symbol[2,2,1] = cot(θ)
        Symbol
    end
    # Calculation by hand works out such that in this special case:
    S2Ricci(x) = S2metric(x)
    ConstMetric(x) = Diagonal(ones(2))

    # Test Numeric Christoffel Symbols, Riemann and Ricci tensors, Ricci Scalar
    # Test WITH AND WITHOUT BIGFLOAT
    x = rand(2)
    @test norm(ChristoffelSymbol(S2metric,x) .- S2Christoffel(x), 1) < 5e-9
    @test norm(ChristoffelSymbol(S2metric,BigFloat.(x)) .- S2Christoffel(BigFloat.(x)), 1) < 1e-40

    @test abs(RicciScalar(S2metric,x) - 2) < 5e-4
    @test abs(RicciScalar(S2metric,BigFloat.(x)) - 2) < 2e-22

    # Use wilder metric and test AutoDiff vs Finite
    import InformationGeometry: MetricPartials, ChristoffelPartials
    Y = rand(3)
    Metric3(x) = [sinh(x[3]) exp(x[1])*sin(x[2]) 0; 0 cosh(x[2]) cos(x[2])*x[3]*x[2]; exp(x[2]) cos(x[3])*x[1]*x[2] 0.]
    @test MetricPartials(Metric3, Y; ADmode=Val(true)) ≈ MetricPartials(Metric3, Y; ADmode=Val(false))
    @test ChristoffelSymbol(Metric3, Y; ADmode=Val(true)) ≈ ChristoffelSymbol(Metric3, Y; ADmode=Val(false))
    @test maximum(abs.(ChristoffelPartials(Metric3, Y; ADmode=Val(true)) - ChristoffelPartials(Metric3, Y; ADmode=Val(false), BigCalc=true))) < 1e-11
    @test maximum(abs.(Riemann(Metric3, Y; ADmode=Val(true)) - Riemann(Metric3, Y; ADmode=Val(false), BigCalc=true))) < 1e-11
    # Test with static arrays
    Metric3SA(x) = SA[sinh(x[3]) exp(x[1])*sin(x[2]) 0; 0 cosh(x[2]) cos(x[2])*x[3]*x[2]; exp(x[2]) cos(x[3])*x[1]*x[2] 0.]
    @test MetricPartials(Metric3SA, Y; ADmode=Val(true)) ≈ MetricPartials(Metric3SA, Y; ADmode=Val(false))
    @test ChristoffelSymbol(Metric3SA, Y; ADmode=Val(true)) ≈ ChristoffelSymbol(Metric3SA, Y; ADmode=Val(false))
    @test maximum(abs.(ChristoffelPartials(Metric3SA, Y; ADmode=Val(true)) - ChristoffelPartials(Metric3SA, Y; ADmode=Val(false), BigCalc=true))) < 1e-11
    @test maximum(abs.(Riemann(Metric3SA, Y; ADmode=Val(true)) - Riemann(Metric3SA, Y; ADmode=Val(false), BigCalc=true))) < 1e-11
    # Test with BigFloat
    @test -45 > MetricPartials(Metric3SA, BigFloat.(Y); ADmode=Val(true)) - MetricPartials(Metric3SA, BigFloat.(Y); ADmode=Val(false)) |> maximum |> log10 |> Float64
    @test -45 > ChristoffelSymbol(Metric3SA, BigFloat.(Y); ADmode=Val(true)) - ChristoffelSymbol(Metric3SA, BigFloat.(Y); ADmode=Val(false)) |> maximum |> log10 |> Float64
    @test -20 > ChristoffelPartials(Metric3SA, BigFloat.(Y); ADmode=Val(true)) - ChristoffelPartials(Metric3SA, BigFloat.(Y); ADmode=Val(false)) |> maximum |> log10 |> Float64
    @test -20 > Riemann(Metric3SA, BigFloat.(Y); ADmode=Val(true)) - Riemann(Metric3SA, BigFloat.(Y); ADmode=Val(false)) |> maximum |> log10 |> Float64

    @test abs(GeodesicDistance(ConstMetric,[0,0],[1,1]) - sqrt(2)) < 2e-8
    @test abs(GeodesicDistance(S2metric,[π/4,1],[3π/4,1]) - π/2) < 1e-9
    @test abs(GeodesicDistance(S2metric,[π/2,0],[π/2,π/2]) - π/2) < 1e-8

    DS = DataSet([0,0.5,1],[1.,3.,7.],[1.2,2.,0.6]);    DM = DataModel(DS, (x,p) -> p[1]^3 *x + p[2]^3)
    y = MLE(DM) + 0.2(rand(2) .- 0.5)
    geo = GeodesicBetween(DM, MLE(DM), y; tol=1e-11)
    @test norm(MLE(DM) - [1.829289173660125,0.942865200406147]) < 1e-7

    Len = GeodesicLength(DM,geo)
    @test abs(InformationGeometry.ParamVol(geo) * InformationGeometry.GeodesicEnergy(DM,geo) - Len^2) < 1e-8
    Confnum = InvConfVol(ChisqCDF(pdim(DM), 2*(LogLikeMLE(DM) - loglikelihood(DM, y))))
    @test InformationGeometry.GeodesicRadius(DM, Confnum) - Len < 1e-5

    # Apply logarithmic map first since it is typically multi-valued for positively curved manifolds.
    @test norm(ExponentialMap(FisherMetric(DM), MLE(DM), LogarithmicMap(FisherMetric(DM), MLE(DM), y)) - y) < 1

end


@safetestset "Numerical Helper Functions" begin
    using InformationGeometry, Test, BenchmarkTools, ForwardDiff

    # Compare Integrate1D and IntegrateND

    # Test integration, differentiation, Monte Carlo, GeodesicLength
    # TEST WITH AND WITHOUT BIGFLOAT
    @test abs(InformationGeometry.MonteCarloArea(x->((x[1]^2 + x[2]^2) < 1), HyperCube([[-1,1],[-1,1]])) - π) < 1.5e-3
    @test abs(Integrate1D(cos, (0,π/2); tol=1e-12) - IntegrateND(cos, (0,π/2); tol=1e-12)) < 1e-10
    z = 3rand()
    @test abs(Integrate1D(x->2/sqrt(π) * exp(-x^2), [0,z/sqrt(2)]) - ConfVol(z)) < 1e-12
    @test abs(LineSearch(x->(x < BigFloat(π))) - π) < 1e-14
    @test abs(LineSearch(x->(x < BigFloat(π)), BigFloat(1e-14); tol=1e-30) - BigFloat(π)) < 1e-25
    @test abs(CubeVol(TranslateCube(HyperCube([[0,1],[0,π],[-sqrt(2),0]]),rand(3))) - sqrt(2)*π) < 3e-15

    k = rand(1:20);     r = 10rand()
    @test InvChisqCDF(k,Float64(ChisqCDF(k,r))) ≈ r
    @test abs(InvChisqCDF(k,ChisqCDF(k,BigFloat(r)); tol=1e-20) - r) < 1e-18

    # Differentiation
    X = ForwardDiff.gradient(x->x[1]^2 + exp(x[2]), [5,10.])
    Y = ForwardDiff.jacobian(x->[x[1]^2 + exp(x[2])], [5,10.])
    Z = ForwardDiff.hessian(x->x[1]^2 + exp(x[2]) + x[1]*x[2], [5,10.])

    function MyTest(ADmode::Symbol; kwargs...)
        Grad, Jac, Hess = GetGrad(ADmode; kwargs...), GetJac(ADmode; kwargs...), GetHess(ADmode; kwargs...)
        @test Grad(x->x[1]^2 + exp(x[2]), [5,10.]) ≈ X
        @test Jac(x->[x[1]^2 + exp(x[2])], [5,10.]) ≈ Y
        @test Hess(x->x[1]^2 + exp(x[2]) + x[1]*x[2], [5,10.]) ≈ Z
    end

    for ADmode ∈ [:ForwardDiff, :Zygote, :ReverseDiff, :FiniteDiff]
        MyTest(ADmode)
    end
end
