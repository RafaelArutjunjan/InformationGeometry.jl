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

DS2 = DataSet([-1, 0, 1], [-1 1; 0 0; 1.1 -1.1], transpose(hcat(0.75ones(2), 0.5ones(2), 0.85ones(2))))
DM2 = DataModel(DS2, ModelMap((x,p)->[p[1]*x + p[2]-1, p[3]*x + p[4]-1]; pnames=["a", "b", "c", "d"]))

@test LinkParameters(DM2, [2,4]) == DataModel(DS2, (x,p)->[p[1]*x + p[2]-1, p[3]*x + p[2]-1], MLE(DM2)[1:end-1])

# TranstrumModel = ModelMap((x::Real,p::AbstractVector)->exp(-p[1]*x) + exp(-p[2]*x), θ::AbstractVector -> θ[1]>θ[2], PositiveDomain(2, 1e2), (1,1,2))
# TranstrumDM = DataModel(DataSet([0.33, 1, 3], [0.88,0.5,0.35], [0.1,0.3,0.2]), TranstrumModel)
# linTranstrum = LogTransform(TranstrumDM)
# RicciScalar(TranstrumDM, MLE(TranstrumDM)), RicciScalar(linTranstrum, MLE(linTranstrum))
# loglikelihood(TranstrumDM, MLE(TranstrumDM)), loglikelihood(linTranstrum, MLE(linTranstrum))

# Try with normal functions too, not only ModelMaps.
# Try Ricci in particular, maybe as BigFloat.

# Does Score / FisherMetric and AutoDiff still work?