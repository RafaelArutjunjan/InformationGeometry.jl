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