using InformationGeometry, Test, LinearAlgebra, BenchmarkTools, LsqFit

DM = DataModel(DataSet([1,2,3],[4,1,5,2,6.5,3.5],[0.5,0.5,0.45,0.45,0.6,0.6], (3,1,2)), (x,p)-> [p[1]^3*x, p[2]^2*x])
dm = InplaceDM(DM)

@test EmbeddingMap(DM, MLE(DM)) ≈ EmbeddingMap(dm, MLE(dm))
@test EmbeddingMatrix(DM, MLE(DM)) ≈ EmbeddingMatrix(dm, MLE(dm))
@test Score(DM, MLE(DM)) ≈ Score(dm, MLE(dm))
@test FisherMetric(DM, MLE(DM)) ≈ FisherMetric(dm, MLE(dm))

@test OrthVF(DM, MLE(DM)) ≈ OrthVF(dm, MLE(dm))

# (y,x,p)->(y .= [p[1]^3*x, p[2]^2*x])
@test DataModel(Data(DM), OptimizeModel(Predictor(dm); inplace=false)...) isa AbstractDataModel

@test InformationGeometry.Curve_fit(Data(DM), Predictor(DM), rand(pdim(DM))).param ≈ InformationGeometry.Curve_fit(Data(DM), Predictor(DM), dPredictor(DM), rand(pdim(DM))).param
# import InformationGeometry.minimize
# @test minimize(Data(DM), Predictor(DM), rand(pdim(DM))) ≈ minimize(Data(DM), Predictor(DM), dPredictor(DM), rand(pdim(DM)))


## Test inplace likelihood, score and Fisher against out-of-place
function TestDataModelQuantities(dm::AbstractDataModel)
    Sres = rand(pdim(dm));  Fres = rand(pdim(dm), pdim(dm))

    @test loglikelihood(dm, MLE(dm)) isa Real
    @test Score(dm)(MLE(dm)) isa AbstractVector
    @test Score(dm)(Sres, MLE(dm)) isa AbstractVector
    @test Score(dm)(MLE(dm)) ≈ Score(dm)(Sres, MLE(dm))
    @test FisherMetric(dm)(MLE(dm)) isa AbstractMatrix
    @test FisherMetric(dm)(Fres, MLE(dm)) isa AbstractMatrix
    @test FisherMetric(dm)(MLE(dm)) ≈ FisherMetric(dm)(Fres, MLE(dm))

    Yres = rand(length(ydata(dm)));     J = rand(length(Yres), pdim(dm))

    @test EmbeddingMap(dm, MLE(dm)) isa AbstractVector
    @test (EmbeddingMap!(Yres, dm, MLE(dm)); Yres) isa AbstractVector
    @test EmbeddingMap(dm, MLE(dm)) ≈ Yres
    @test EmbeddingMatrix(dm, MLE(dm)) isa AbstractMatrix
    @test (EmbeddingMatrix!(J, dm, MLE(dm));    J) isa AbstractMatrix
    @test EmbeddingMatrix(dm, MLE(dm)) ≈ J

    yres = rand(length(ydim(dm)));     j = rand(length(yres), pdim(dm))

    @test length(Predictor(dm)(WoundX(dm)[1], MLE(dm))) == ydim(dm)
    @test length((Predictor(dm)(yres, WoundX(dm)[1], MLE(dm));  yres)) == ydim(dm)
    @test all(Predictor(dm)(WoundX(dm)[1], MLE(dm)) .≈ yres)
    @test size(dPredictor(dm)(WoundX(dm)[1], MLE(dm))) == (ydim(dm), pdim(dm))
    @test size((dPredictor(dm)(j, WoundX(dm)[1], MLE(dm));  j)) == (ydim(dm), pdim(dm))
    @test dPredictor(dm)(WoundX(dm)[1], MLE(dm)) ≈ j
end

function CompareTimings(dm::AbstractDataModel, idm::AbstractDataModel; Ratio=0.8)
    @test dm == idm
    S1, S2 = rand(pdim(dm)), rand(pdim(dm))
    F1, F2 = rand(pdim(dm),pdim(dm)), rand(pdim(dm),pdim(dm))
    @test (@belapsed loglikelihood($dm, $(MLE(dm)))) / (@belapsed loglikelihood($idm, $(MLE(dm)))) > Ratio
    @test (@belapsed Score($dm, $(MLE(dm)))) / (@belapsed Score($idm, $(MLE(dm)))) > Ratio
    @test (@belapsed $(Score(dm))($S1, $(MLE(dm)))) / (@belapsed $(Score(idm))($S2, $(MLE(dm)))) > Ratio
    @test (@belapsed FisherMetric($dm, $(MLE(dm)))) / (@belapsed FisherMetric($idm, $(MLE(dm)))) > Ratio
    @test (@belapsed $(FisherMetric(dm))($F1, $(MLE(dm)))) / (@belapsed $(FisherMetric(idm))($F2, $(MLE(dm)))) > Ratio
end



## Do inplace and out-of-place, ydim=1 and ydim > 1, iscustom and not custom

DS = DataSet([1,2,3,4], [4,5,6.5,9], [0.5,0.45,0.6,1])

M = ModelMap((x, p)->p[1] .*x .+ p[2], (1,1,2); IsCustom=true)
dM = DetermineDmodel(DS, M)
dm = DataModel(DS, M, dM)

TestDataModelQuantities(dm)

## inplace model
iM = ModelMap((y, x, p)->y .= p[1] .*x .+ p[2], (1,1,2); IsCustom=true)
diM = DetermineDmodel(DS, iM)
idm = DataModel(DS, iM, diM, ones(2))

Jtest = EmbeddingMatrix(DS, diM, ones(2))
@test Jtest ≈ [1.0 1.0; 2.0 1.0; 3.0 1.0; 4.0 1.0]

TestDataModelQuantities(idm)
CompareTimings(dm, idm)

