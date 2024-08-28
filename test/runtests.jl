
using SafeTestsets



@safetestset "Probability Objects" begin
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
end


@safetestset "Confidence Regions" begin
    using InformationGeometry, Test, Plots

    DS = DataSet([0,0.5,1,1.5],[1.,3.,7.,8.1],[1.2,2.,0.6,1.]);    DM = DataModel(DS,LinearModel)
    DME = DataModel(DataSetExact([0,0.5,1,1.5],0.1ones(4),[1.,3.,7.,8.1],[1.2,2.,0.6,1.]), LinearModel)

    sols = ConfidenceRegions(DM,1:2; tol=1e-6)
    @test InformationGeometry.IsStructurallyIdentifiableAlong(DM, sols[1]) == true
    @test size(SaveConfidence(sols,50)) == (50,4)
    @test size(SaveGeodesics(sols,50)) == (50,2)
    @test size(SaveDataSet(DM)) == (4,3)
    @test ConfidenceRegionVolume(DM,sols[1];N=5000) < ConfidenceRegionVolume(DM,sols[2];N=5000,WE=true)

    @test size(ConfidenceBands(DM,sols[1]; N=50, plot=false)) == (50,3)
    @test size(PlotMatrix(inv(FisherMetric(DM,MLE(DM))),MLE(DM); N=50,plot=false)) == (50,2)

    @test Plots.plot(DM) isa Plots.Plot
    @test Plots.plot(DME) isa Plots.Plot
    @test ResidualPlot(DM) isa Plots.Plot
    @test VisualizeGeos([MBAM(DM)]) isa Plots.Plot

    simplermodel(x,p) = p[1]*x;    DMSimp = DataModel(DS,simplermodel)
    @test length(ConfidenceRegion(DMSimp,1.)) == 2
    @test ModelComparison(DM,DMSimp)[2] > 0.

    @test FindFBoundary(DM,1)[1] - FindConfBoundary(DM,1)[1] > 0
    z = 8rand()
    @test FindConfBoundary(DM, z; BoolTest=true)[1] - FindConfBoundary(DM, z; BoolTest=false)[1] < 1e-9
end


@safetestset "More Boundary tests" begin
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
    Mats = ParameterProfiles(ToyDME, 2; plot=false)
    ProfBox = ProfileBox(ToyDME, InterpolatedProfiles(Mats), 1)
    ExactBox = ConstructCube(sol)
    @test norm(Center(ProfBox) - Center(ExactBox)) < 3e-5 && norm(CubeWidths(ProfBox) - CubeWidths(ExactBox)) < 3e-4
    @test 0 < PracticallyIdentifiable(Mats) < 2

    # Test rescaling method for confidence boundary generation
    sol2b = InformationGeometry.GenerateBoundary2(ToyDME, sol.u[1]; tol=1e-4, Embedded=false)
    sol2e = InformationGeometry.GenerateBoundary2(ToyDME, sol.u[1]; tol=1e-4, Embedded=true)
    Cb, Ce = ConstructCube(sol2b), ConstructCube(sol2e)
    @test norm(Center(Ce) - Center(ExactBox)) < 2e-4 && norm(CubeWidths(Ce) - CubeWidths(ExactBox)) < 2e-4
    @test norm(Center(Cb)) < 0.3 && norm(CubeWidths(Cb)) < 5

    # Method for general cost functions on 2D domains
    sol = GenerateBoundary(x->-norm(x,1.5), [1., 0])
    @test 0.23 ≤ length(GenerateBoundary(x->-norm(x,1.5), [1., 0]; Boundaries=(u,t,int)->u[1]<0.).u) / length(sol.u) ≤ 0.27
end


@safetestset "ODE-based models" begin
    using InformationGeometry, Test, OrdinaryDiffEq, ModelingToolkit, LinearAlgebra

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
    @test DataModel(SIRDS, SIRsys, [762, 1, 0.], [2], [0.0022,0.45], true; tol=1e-6) isa DataModel
    @test norm(2*EmbeddingMap(Data(SIRDM), Predictor(SIRDM), MLE(SIRDM)) - EmbeddingMap(Data(SIRDM), ModifyODEmodel(SIRDM, x->2*x[2]), MLE(SIRDM))) < 2e-4

    # Use this to avoid the macros @parameters and @variables inside @safetestset
    eval(ModelingToolkit._parse_vars(:parameters, Real, (:t, :k), ModelingToolkit.toparam))
    eval(ModelingToolkit._parse_vars(:variables, Real, (:(A(t)),)))
    sys = ODESystem([Differential(t)(A) ~ -k*A], t, [A], [k]; name=Symbol("Decay System"))
    Split(θ) = (θ[1:1], θ[2:2])
    Observe(u) = 2u[1]
    dm = DataModel(DataSet([0.2,1,2], [0.9,0.4,0.25], [0.1,0.1,0.1]), sys, Split, Observe)
    @test dm isa DataModel
    # ObservationFunction is extended to 3 arguments, does not equal initially given PreObservationFunction
    @test length(Predictor(dm).Meta) == 4 && Predictor(dm).Meta[1:2] == (sys, Split)
    @test string(Predictor(dm).name) == "Decay System"

    # Backwards in time integration (GetModelRobust)
    @test all(EmbeddingMap(SIRDM, MLE(SIRDM), [-10, 3, -0.5, 15]) .> 0)

    # Check AutoDiff in time
    using ForwardDiff
    @assert 0 < ForwardDiff.derivative(t->Predictor(SIRDM)(t, MLE(SIRDM))[1], 1.)
    @assert isposdef(ForwardDiff.jacobian(x->EmbeddingMap(SIRDM, MLE(SIRDM), x), [1,2,3.]))
end


@safetestset "Model and Data Transformations" begin
    using InformationGeometry, Test, LinearAlgebra, StaticArrays

    ## Parameter Transforms
    PiDM = DataModel(DataSet([0,1], [0.5π,1.5π], [0.5,0.5]), ModelMap((x,p)->p[1], θ->θ[1]-1, HyperCube([[0,5]])))
    @test !IsInDomain(Predictor(PiDM), [0.9]) && IsInDomain(Predictor(PiDM), [1.1])

    # Translation
    PiDM2 = DataModel(Data(PiDM), TranslationTransform(Predictor(PiDM),[1.]))
    @test !IsInDomain(Predictor(PiDM2), [-0.1]) && IsInDomain(Predictor(PiDM2), [0.1])

    # LogTransform
    PiDM3 = DataModel(Data(PiDM), LogTransform(Predictor(PiDM),trues(1)))
    @test !IsInDomain(Predictor(PiDM3), exp.([1])-[0.1]) && IsInDomain(Predictor(PiDM3), exp.([1])+[0.1])

    DS = DataSet([0.1,0.5,1,1.5],[1.,3.,7.,8.1],[1.2,2.,0.6,1.])
    DM = DataModel(DS, LinearModel)
    @test FisherMetric(LinearDecorrelation(DM), zeros(2)) ≈ [1 0; 0 1]

    ## Input-Output Transforms
    DME = DataModel(DataSetExact(DS, 0.3ones(length(xdata(DS)))), LinearModel)

    @test ysigma(LogYdata(DM)) ≈ inv.(ydata(DM)) .* ysigma(DM)
    @test xsigma(LogXdata(DME)) ≈ inv.(xdata(DME)) .* xsigma(DME)

    @test MLE(LogXdata(DM)) ≈ MLE(DM)

    @test (Log10Xdata∘Exp10Xdata)(DME) == DME
    @test (LogYdata∘ExpYdata)(DME) == DME

    @test Log10Xdata(DS) == DataSet(log10.([0.1,0.5,1,1.5]),[1.,3.,7.,8.1],[1.2,2.,0.6,1.])

    @test PinParameters(DataModel(DS, Predictor(DM), dPredictor(DM), MLE(DM), θ->-norm(θ)), 2, 0.5) == DataModel(DS, (x,p)->p[1]*x + 0.5, (x,p)->SMatrix{1,1}([x]), [1.5], x->-norm(SA[x,0.5]))

    # TranstrumModel = ModelMap((x::Real,p::AbstractVector)->exp(-p[1]*x) + exp(-p[2]*x), θ::AbstractVector -> θ[1]>θ[2], PositiveDomain(2, 1e2), (1,1,2))
    # TranstrumDM = DataModel(DataSet([0.33, 1, 3], [0.88,0.5,0.35], [0.1,0.3,0.2]), TranstrumModel)
    # linTranstrum = LogTransform(TranstrumDM)
    # RicciScalar(TranstrumDM, MLE(TranstrumDM)), RicciScalar(linTranstrum, MLE(linTranstrum))
    # loglikelihood(TranstrumDM, MLE(TranstrumDM)), loglikelihood(linTranstrum, MLE(linTranstrum))

    # Try with normal functions too, not only ModelMaps.
    # Try Ricci in particular, maybe as BigFloat.

    # Does Score / FisherMetric and AutoDiff still work?
end


@safetestset "In-place ModelMaps" begin
    using InformationGeometry, Test, LinearAlgebra

    DM = DataModel(DataSet([1,2,3],[4,1,5,2,6.5,3.5],[0.5,0.5,0.45,0.45,0.6,0.6], (3,1,2)), (x,p)-> [p[1]^3*x, p[2]^2*x])
    dm = InplaceDM(DM)

    @test EmbeddingMap(DM, MLE(DM)) ≈ EmbeddingMap(dm, MLE(dm))
    @test EmbeddingMatrix(DM, MLE(DM)) ≈ EmbeddingMatrix(dm, MLE(dm))
    @test Score(DM, MLE(DM)) ≈ Score(dm, MLE(dm))
    @test FisherMetric(DM, MLE(DM)) ≈ FisherMetric(dm, MLE(dm))

    @test OrthVF(DM, MLE(DM)) ≈ OrthVF(dm, MLE(dm))

    # (y,x,p)->(y .= [p[1]^3*x, p[2]^2*x])
    @test DataModel(Data(DM), OptimizeModel(Predictor(dm); inplace=false)...) isa AbstractDataModel

    @test curve_fit(Data(DM), Predictor(DM), rand(pdim(DM))).param ≈ curve_fit(Data(DM), Predictor(DM), dPredictor(DM), rand(pdim(DM))).param
    # import InformationGeometry.minimize
    # @test minimize(Data(DM), Predictor(DM), rand(pdim(DM))) ≈ minimize(Data(DM), Predictor(DM), dPredictor(DM), rand(pdim(DM)))
end

@safetestset "Inputting Datasets of various shapes" begin
    using InformationGeometry, Test, LinearAlgebra, Random, Distributions, StaticArrays, Plots

    ycovtrue = Diagonal([1,2,3]) |> Matrix
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
    @test abs(loglikelihood(ODM, P) - loglikelihood(CDM, P)) < 1e-5
    @test norm(Score(ODM, P) - Score(CDM, P)) < 2e-4
    @test norm(FisherMetric(ODM, P) - FisherMetric(CDM, P)) < 2e-4
    @test norm(InformationGeometry.ResidualStandardError(ODM) - InformationGeometry.ResidualStandardError(CDM)) < 1e-10

    lastDS = Data(Data(CDM))[3]
    newCDS = vcat(Data(Data(CDM))[1:end-1], [SubDataSet(lastDS, 1:2:Npoints(lastDS))], [SubDataSet(lastDS, 2:2:Npoints(lastDS))]) |> CompositeDataSet
    # repeat last component
    newmodel(x::AbstractVector{<:Number},p::AbstractVector{<:Number}) = SA[p[1] * x[1]^2 + p[3]^3 * x[2], sinh(p[2]) * (x[1] + x[2]),
                                                        exp(p[1]*x[1] + p[1]*x[2]), exp(p[1]*x[1] + p[1]*x[2])]
    splitCDM = DataModel(newCDS, newmodel, MLE(CDM))
    @test abs(loglikelihood(splitCDM, P) - loglikelihood(CDM, P)) < 1e-5
    @test norm(Score(splitCDM, P) - Score(CDM, P)) < 2e-4
    @test norm(FisherMetric(splitCDM, P) - FisherMetric(CDM, P)) < 2e-4


    GDM = DataModel(GeneralizedDataSet(DME), Predictor(DME), MLE(DME))
    @test abs(loglikelihood(GDM, P) - loglikelihood(CDM, P)) < 1e-5
    @test norm(Score(GDM, P) - Score(CDM, P)) < 2e-4
    @test norm(FisherMetric(GDM, P) - FisherMetric(CDM, P)) < 2e-4

    UDS = DataSetUncertain(1:5, (1:5) + [rand(Normal(0,0.4)) for i in 1:5], (x,y,p)->1/exp(p[1]), [log(0.4)])
    UDM = DataModel(UDS, (x,p)->p[1]*x + p[2], [1, 1, 1.])
    
    # Test Type conversions for Datasets
    function TypeTester(DM::AbstractDataModel, ::Type{T}) where T<:Number
        dm = T(DM)
        @test eltype(xdata(dm)) <: T
        @test eltype(ydata(dm)) <: T
        @test eltype(yInvCov(dm)) <: T
        @test eltype(eltype(WoundX(dm))) <: T
        @test eltype(MLE(dm)) <: T
    end
    TypeTester(DM,  Float16)
    TypeTester(DME, Float16)
    TypeTester(CDM, Float16)
    # TypeTester(splitCDM, Float16)
    TypeTester(GDM, Float16)
    # TypeTester(UDM, Float16)

    TypeTester(DM,  BigFloat)
    TypeTester(DME, BigFloat)
    TypeTester(CDM, BigFloat)
    # TypeTester(splitCDM, BigFloat)
    TypeTester(GDM, BigFloat)
    # TypeTester(UDM, BigFloat)
end


@safetestset "DataSetUncertain" begin
    using InformationGeometry, Test, Distributions, Optim

    function tester(DM::AbstractDataModel, DM2::AbstractDataModel, DM3::AbstractDataModel, mle::AbstractVector; atol::Real=1e-6)
        @assert Data(DM) isa InformationGeometry.AbstractUnknownUncertaintyDataSet
        @assert Data(DM2) isa InformationGeometry.AbstractFixedUncertaintyDataSet
        @assert Data(DM3) isa InformationGeometry.AbstractFixedUncertaintyDataSet

        @test isapprox(loglikelihood(DM)(mle), loglikelihood(DM2)(mle); atol)
        @test Score(DM)(mle)[1:pdim(DM2)] ≈ Score(DM2)(mle)[1:pdim(DM2)] ≈ Score(DM3)(mle)[1:pdim(DM2)]
        @test FisherMetric(DM)(mle)[1:pdim(DM2),1:pdim(DM2)] ≈ FisherMetric(DM2)(mle)[1:pdim(DM2),1:pdim(DM2)] ≈ FisherMetric(DM3)(mle)[1:pdim(DM2),1:pdim(DM2)]
        @test -GetHess(loglikelihood(DM))(mle)[1:pdim(DM2),1:pdim(DM2)] ≈ -GetHess(loglikelihood(DM2))(mle)[1:pdim(DM2),1:pdim(DM2)] ≈ -GetHess(loglikelihood(DM3))(mle)[1:pdim(DM2),1:pdim(DM2)]
    end

    DS = DataSetUncertain(1:5, (1:5) + [rand(Normal(0,0.4)) for i in 1:5], (x,y,p)->1/abs(p[1]), [0.4]; xnames=["Time"], ynames=["Signal"])
    tester(
        DataModel(DS, (x,p)->p[1]*x + p[2], [1, 1, 1.]; meth=Optim.Newton()),
        DataModel(DataSet(xdata(DS), ydata(DS), 0.4; xnames=["Time"], ynames=["Signal"]), (x,p)->p[1].*x .+ p[2], [1, 1.]; meth=Optim.Newton()),
        DataModel(DataSetExact(xdata(DS), ydata(DS), 0.4), (x,p)->p[1]*x + p[2], [1, 1.]; meth=Optim.Newton()),
        [1,1,0.4]
    )


    using ModelingToolkit, StaticArrays, OrdinaryDiffEq, LinearAlgebra

    ToNuc = 0.45/1.4
    FromNuc = 1/ToNuc

    eval(ModelingToolkit._parse_vars(:parameters, Real, (:t, :Epo_degradation_BaF3,
                            :k_phos, :k_imp_homo, :k_imp_hetero, :k_exp_homo, :k_exp_hetero), ModelingToolkit.toparam))
    eval(ModelingToolkit._parse_vars(:variables, Real, (:(BaF3_Epo(t)),
                            :(STAT5A(t)), :(STAT5B(t)), :(pApA(t)), :(pApB(t)),
                            :(pBpB(t)), :(nucpApA(t)), :(nucpApB(t)), :(nucpBpB(t)))))
    Dt = Differential(t)
    Eqs = Equation[Dt(BaF3_Epo) ~ -Epo_degradation_BaF3*BaF3_Epo,
        Dt(STAT5A)  ~ k_exp_hetero*(nucpApB^2) + 2k_exp_homo*(nucpApA^2) - k_phos*(STAT5A^2)*BaF3_Epo - k_phos*BaF3_Epo*STAT5A*STAT5B,
        Dt(STAT5B)  ~ k_exp_hetero*(nucpApB^2) + 2k_exp_homo*(nucpBpB^2) - k_phos*(STAT5B^2)*BaF3_Epo - k_phos*BaF3_Epo*STAT5A*STAT5B,
        Dt(pApA)    ~ (1/2)*k_phos*(STAT5A^2)*BaF3_Epo - k_imp_homo*(pApA^2),
        Dt(pApB)    ~ k_phos*BaF3_Epo*STAT5A*STAT5B - k_imp_hetero*(pApB^2),
        Dt(pBpB)    ~ (1/2)*k_phos*(STAT5B^2)*BaF3_Epo - k_imp_homo*(pBpB^2),
        Dt(nucpApA) ~ k_imp_homo*(pApA^2) - k_exp_homo*(nucpApA^2),
        Dt(nucpApB) ~ k_imp_hetero*(pApB^2) - k_exp_hetero*(nucpApB^2),
        Dt(nucpBpB) ~ k_imp_homo*(pBpB^2) - k_exp_homo*(nucpBpB^2)]

    BöhmSys = ODESystem(Eqs, t, [BaF3_Epo, STAT5A, STAT5B, pApA, pApB, pBpB, nucpApA, nucpApB, nucpBpB], [Epo_degradation_BaF3, k_phos, k_imp_homo, k_imp_hetero, k_exp_homo, k_exp_hetero]; name=:Boehm)

    function BöhmObservation(u::AbstractVector, t::Real, θ::AbstractVector)
        specC17 = 0.107 # θ[end]
        BaF3_Epo, STAT5A, STAT5B, pApA, pApB, pBpB, nucpApA, nucpApB, nucpBpB = u
        SA[ 100 * (2 * specC17 * pApA + pApB)/(2 * specC17 * pApA + specC17 * STAT5A + pApB),
            100 * (2 * (1-specC17) * pBpB + pApB)/(2 * (1-specC17) * pBpB + (1-specC17) * STAT5B + pApB),
            100 * (2 * specC17 * pApA + specC17 * STAT5A + pApB)/(2 * specC17 * pApA + 2 * (1-specC17) * pBpB + 2 * pApB + specC17 * STAT5A + (1-specC17) * STAT5B)]
    end

    ratio = 0.693
    BöhmInitial = p-> ([1.25e-7, 207.6 * ratio, 207.6 * (1-ratio), zeros(6)...], view(p, 1:6))

    BöhmT = [0,2.5,5,10,15,20,30,40,50,60,80,100,120,160,200,240]
    pSTAT5A_rel = [7.901072999,66.36349397,81.17132392,94.73030806,95.11648305,91.44171655,91.25709923,93.67229784,88.75423282,85.26970322,81.13239534,76.13592848,65.24805913,42.59965871,25.15779754,15.4301824]
    pSTAT5B_rel = [4.596533343,29.63454599,46.04380647,81.97473362,80.5716093,79.03571964,75.67238037,71.62471986,69.06286328,67.14738432,60.89947629,54.80925777,43.98128998,29.77145816,20.08901656,10.96184517]
    rSTAT5A_rel = [14.72316822,33.76234229,36.79985129,49.71760229,46.9281201,47.83657456,46.92872725,40.59775294,43.78366389,44.45738765,41.32715926,41.06273321,39.23583003,36.61946054,34.8937144,32.21107716]

    BöhmDS = DataSet(BöhmT, [pSTAT5A_rel pSTAT5B_rel rSTAT5A_rel], [[4.12, 7.04, 3.37] for i in eachindex(BöhmT)]; xnames=["Time"], ynames=["pSTAT5A_rel", "pSTAT5B_rel", "rSTAT5A_rel"])
    BöhmModel = GetModel(Exp10Transform(BöhmSys), BöhmInitial, BöhmObservation; tol=1e-4, Domain=HyperCube(-6ones(6), 6ones(6)))
    BöhmDM = DataModel(BöhmDS, BöhmModel, [-1.5690, 4.1978, 4.99, -1.7859, -2.2, -5.]; tol=1e-6)

    Böhmds = DataSetUncertain(BöhmT, [pSTAT5A_rel pSTAT5B_rel rSTAT5A_rel], (x,y,c)->Diagonal(1 ./ abs.(c)), [4.12, 7.04, 3.37]; xnames=["Time"], ynames=["pSTAT5A_rel", "pSTAT5B_rel", "rSTAT5A_rel"])
    Böhmdm = DataModel(Böhmds, GetModel(Exp10Transform(BöhmSys), BöhmInitial, BöhmObservation; tol=1e-4, Domain=vcat(HyperCube(-6ones(6), 6ones(6)), HyperCube(1e-2ones(3), 20ones(3)))),
                            vcat([-1.5690, 4.1978, 4.99, -1.7859, -2.2, -5.], [4.12, 7.04, 3.37]); tol=1e-6)
    tester(
        Böhmdm, BöhmDM,
        DataModel(DataSetExact(BöhmDS), Predictor(BöhmDM), MLE(BöhmDM); tol=1e-6),
        vcat(MLE(BöhmDM), [4.12, 7.04, 3.37])
    )
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

    # Intentionally not giving information about LogPrior input here
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

    S2metric((θ,ϕ)) = [1.0 0; 0 sin(θ)^2]
    function S2Christoffel((θ,ϕ))
        Symbol = zeros(typeof(ϕ),2,2,2);    Symbol[1,2,2] = -sin(θ)*cos(θ)
        Symbol[2,1,2] = Symbol[2,2,1] = cos(θ)/sin(θ);  Symbol
    end
    # Calculation by hand works out such that in this special case:
    S2Ricci(x) = S2metric(x)
    ConstMetric(x) = Diagonal(ones(2))

    # Test Numeric Christoffel Symbols, Riemann and Ricci tensors, Ricci Scalar
    # Test WITH AND WITHOUT BIGFLOAT
    x = rand(2)
    @test norm(ChristoffelSymbol(S2metric,x) .- S2Christoffel(x), 1) < 5e-9
    @test norm(ChristoffelSymbol(S2metric,BigFloat.(x)) .- S2Christoffel(BigFloat.(x)), 1) < 1e-39

    @test abs(RicciScalar(S2metric,x) - 2) < 5e-4
    @test abs(RicciScalar(S2metric,BigFloat.(x)) - 2) < 2e-21

    # Use wilder metric and test AutoDiff vs Finite
    import InformationGeometry: MetricPartials, ChristoffelPartials
    Y = rand(3)
    Metric3(x) = [sinh(x[3]) exp(x[1])*sin(x[2]) 0; 0 cosh(x[2]) cos(x[2])*x[3]*x[2]; exp(x[2]) cos(x[3])*x[1]*x[2] 0.]
    @test MetricPartials(Metric3, Y; ADmode=Val(true)) ≈ MetricPartials(Metric3, Y; ADmode=Val(false))
    @test ChristoffelSymbol(Metric3, Y; ADmode=Val(true)) ≈ ChristoffelSymbol(Metric3, Y; ADmode=Val(false))
    @test maximum(abs.(ChristoffelPartials(Metric3, Y; ADmode=Val(true)) - ChristoffelPartials(Metric3, Y; ADmode=Val(false), BigCalc=true))) < 3e-10
    @test maximum(abs.(Riemann(Metric3, Y; ADmode=Val(true)) - Riemann(Metric3, Y; ADmode=Val(false), BigCalc=true))) < 3e-10
    # Test with static arrays
    Metric3SA(x) = SA[sinh(x[3]) exp(x[1])*sin(x[2]) 0; 0 cosh(x[2]) cos(x[2])*x[3]*x[2]; exp(x[2]) cos(x[3])*x[1]*x[2] 0.]
    @test MetricPartials(Metric3SA, Y; ADmode=Val(true)) ≈ MetricPartials(Metric3SA, Y; ADmode=Val(false))
    @test ChristoffelSymbol(Metric3SA, Y; ADmode=Val(true)) ≈ ChristoffelSymbol(Metric3SA, Y; ADmode=Val(false))
    @test maximum(abs.(ChristoffelPartials(Metric3SA, Y; ADmode=Val(true)) - ChristoffelPartials(Metric3SA, Y; ADmode=Val(false), BigCalc=true))) < 3e-10
    @test maximum(abs.(Riemann(Metric3SA, Y; ADmode=Val(true)) - Riemann(Metric3SA, Y; ADmode=Val(false), BigCalc=true))) < 3e-10
    # Test with BigFloat
    @test -45 > MetricPartials(Metric3SA, BigFloat.(Y); ADmode=Val(true)) - MetricPartials(Metric3SA, BigFloat.(Y); ADmode=Val(false)) |> maximum |> log10 |> Float64
    @test -45 > ChristoffelSymbol(Metric3SA, BigFloat.(Y); ADmode=Val(true)) - ChristoffelSymbol(Metric3SA, BigFloat.(Y); ADmode=Val(false)) |> maximum |> log10 |> Float64
    @test -20 > ChristoffelPartials(Metric3SA, BigFloat.(Y); ADmode=Val(true)) - ChristoffelPartials(Metric3SA, BigFloat.(Y); ADmode=Val(false)) |> maximum |> log10 |> Float64
    @test -20 > Riemann(Metric3SA, BigFloat.(Y); ADmode=Val(true)) - Riemann(Metric3SA, BigFloat.(Y); ADmode=Val(false)) |> maximum |> log10 |> Float64

    @test abs(GeodesicDistance(ConstMetric,[0,0],[1,1]) - sqrt(2)) < 2e-8
    @test abs(GeodesicDistance(S2metric,[π/4,1],[3π/4,1]) - π/2) < 1e-8
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


@safetestset "Optimization Functions" begin
    using InformationGeometry, Test, BenchmarkTools, LinearAlgebra, Optim
    import InformationGeometry.minimize

    # Test optimizers:
    F(x) = x[1]^2 + 0.5x[2]^4;    initial = 10ones(2) + rand(2);    Cube = HyperCube(-2ones(2), 18ones(2))

    @test norm(minimize(F, initial, Cube; tol=1e-5, meth=NelderMead())) < 5e-1
    @test norm(minimize(F, initial, Cube; tol=1e-5, meth=LBFGS())) < 5e-2
    @test norm(minimize(F, initial, Cube; tol=1e-5, meth=Optim.Newton())) < 5e-2

    @test norm(minimize(F, initial; tol=1e-5, meth=NelderMead())) < 5e-1
    @test norm(minimize(F, initial; tol=1e-5, meth=LBFGS())) < 5e-2
    @test norm(minimize(F, initial; tol=1e-5, meth=Optim.Newton())) < 5e-2

    # Check optimization with non-linear constraints and box constraints

    # Check in-place and out-of-place optimization

    # Check type stability of optimization
    using ComponentArrays
    @test minimize(X->X.A[1]^2 + 0.5X.B[1]^4, ComponentVector(A=[initial[1]], B=[initial[1]]); tol=1e-5, meth=Optim.Newton()) isa ComponentVector
end


@safetestset "Numerical Helper Functions" begin
    using InformationGeometry, Test
    
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
end
