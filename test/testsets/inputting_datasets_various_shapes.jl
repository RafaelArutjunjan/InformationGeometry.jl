using InformationGeometry, Test, LinearAlgebra, Random, Distributions, StaticArrays, Plots

ycovtrue = Diagonal([1,2,3]) |> Matrix
ptrue = [1.,π,-5.];        ErrorDistTrue = MvNormal(zeros(3),ycovtrue)

model(x::AbstractVector{<:Number},p::AbstractVector{<:Number}) = SA[p[1] * x[1]^2 + p[3]^3 * x[2],
                                                    sinh(p[2]) * (x[1] + x[2]), exp(p[1]*x[1] + p[1]*x[2])]
Gen(t) = float.([t,0.5t^2]);    Xdata = Gen.(0.5:0.1:3)
Ydata = [model(x,ptrue) + rand(ErrorDistTrue) for x in Xdata]
Sig = BlockMatrix(ycovtrue,length(Ydata));    DS = DataSet(Xdata,Ydata,Sig)
DM = DataModel(DS, model, [1,3,-5.])
@test norm(MLE(DM) - ptrue) < 5e-2
DME = DataModel(DataSetExact(DS), model)
P = MLE(DM) + 0.5rand(length(MLE(DM)))
@test loglikelihood(DM,P) ≈ loglikelihood(DME,P)
@test Score(DM,P) ≈ Score(DME,P)

Planes, sols = ConfidenceRegion(DM,1)
@test typeof(VisualizeSols(Planes,sols)) <: Plots.Plot

# Planetests:
PL = Plane(rand(3), rand(3), rand(3))
x = rand(2)
@test PlaneCoordinates(PL, x) ≈ x[1] .* PL.Vx .+ x[2] .* PL.Vy .+ PL.stütz
@test DecomposeWRTPlane(PL, PlaneCoordinates(PL, x)) ≈ x


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