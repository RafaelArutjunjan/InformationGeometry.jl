using InformationGeometry, Test, LinearAlgebra, StaticArrays, Random

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



import InformationGeometry: EmbeddingMapExtended, SFFNormSquared, AmbientMetric
## Test extrinsic curvature and Bartlett correction
a = 1+rand();   b = 2+rand()
DM = DataModel(DataSet(0:0, rand(2), Diagonal([inv(a), inv(b)]), (1,1,2)), (x,p)->[p[1], p[1]^2])
θ = MLE(DM)[1]
@test yInvCov(DM) ≈ Diagonal([a,b])
@test FisherMetric(DM, MLE(DM))[1] ≈ a + 4b*θ^2
@test SecondFundamentalForm(DM, MLE(DM))[1] ≈ 2/(a + 4b*θ^2) .* [-2b*θ, a]
@test EfronMeanCurvature(DM, MLE(DM)) ≈ 2/(a + 4b*θ^2)^2 .* [-2b*θ, a]
@test EfronScalarCurvature(DM, MLE(DM)) ≈ 2a*b/(a + 4b*θ^2)^3

DM2 = DataModel(DataSet(0:0, rand(3), Diagonal([inv(a), inv(a), inv(b)]), (1,1,3)), (x,p)->[p[1], p[2], p[1]^2 + p[2]^2])
@test yInvCov(DM2) ≈ Diagonal([a,a,b])
@test FisherMetric(DM2, [0,0.]) ≈ Diagonal([a,a])
@test SecondFundamentalForm(DM2, [0,0.])[1] ≈ [0,0,2.]
@test EfronScalarCurvature(DM2, [0,0.]) ≈ 4b/a^2


## Second fundamental form for fixed known uncertainties: fixed-variance regression anchor
Random.seed!(1234)
DMlin = DataModel(DataSet([1,2,3,4], [4,5,6.5,9], [0.5,0.45,0.6,1]), (x,p)->p[1]*x + p[2])
@test SFFNormSquared(DMlin, MLE(DMlin)) ≈ 2EfronScalarCurvature(DMlin, MLE(DMlin))
IIlin = SecondFundamentalForm(DMlin, MLE(DMlin))
@test IIlin isa Matrix{<:AbstractVector}    ## fixed-variance format unchanged by the merge
@test length(IIlin) == pdim(DMlin)^2

## Extrinsic curvature with simultaneously estimated uncertainties (DataSetUncertain)
DSU = DataSetUncertain(1:2, rand(2), (x,y,c)->c[1], x->(x,x), [1.0]; nerrorparameters=0)
DMU = DataModel(DSU, (x,p)->p[1], [0.1])
θU = MLE(DMU)[1];    NU = DataspaceDim(DMU)
@test FisherMetric(DMU, MLE(DMU))[1] ≈ NU * (θU^4 +2) / θU^2
@test sum(InformationGeometry.yInvCovProper(DMU, MLE(DMU))) ≈ NU * θU^2
expectedIIμ = fill((θU^4 + 6) / (θU * (θU^4 + 2)), NU)
expectedIIΛ = -4θU^4 / (θU^4 + 2) * Matrix{Float64}(I,NU,NU) - θU^4 * ones(NU,NU)
@test SFFNormSquared(DMU, MLE(DMU)) ≈ AmbientMetric(
    InformationGeometry.yInvCovProper(DMU), (expectedIIμ, expectedIIΛ)) / FisherMetric(DMU, MLE(DMU))[1]^2
IIμ, IIΛ = SecondFundamentalForm(DMU, MLE(DMU))
@test IIμ[1,1] ≈ expectedIIμ
@test IIΛ[1,1] ≈ expectedIIΛ
# @test size(IIμ) == (NU, 1, 1) && size(IIΛ) == (NU, NU, 1, 1)
@test EfronScalarCurvature(DMU, MLE(DMU)) ≈ SFFNormSquared(DMU, MLE(DMU))/2
Hμ, HΛ = EfronMeanCurvature(DMU, MLE(DMU))
# @test length(Hμ) == NU && size(HΛ) == (NU, NU)
@test all(isfinite, InformationGeometry.EfronRicciCurvature(DMU, MLE(DMU)))
@test all(isfinite, InformationGeometry.EfronRiemannCurvature(DMU, MLE(DMU)))
Embed, Prec = EmbeddingMapExtended(DMU, MLE(DMU))
@test Embed ≈ EmbeddingMap(DMU, MLE(DMU))
@test Prec ≈ InformationGeometry.yInvCovProper(DMU, MLE(DMU))

## SFFNormSquared is invariant under reparametrisations of the model and data space
DSUlog = DataSetUncertain(1:2, ydata(DMU), (x,y,c)->exp(c[1]), x->(x,x), [1.0]; nerrorparameters=0)
DMUlog = DataModel(DSUlog, (x,p)->exp(p[1]), [log(MLE(DMU)[1])])
@test SFFNormSquared(DMUlog, MLE(DMUlog)) ≈ SFFNormSquared(DMU, MLE(DMU))

## Diagonal vs dense error models yield consistent results
DMU1 = DataModel(DataSetUncertain(Windup([1,2,3,4],2), Windup([4,5,6.5,9],2), (x,y,c)->Diagonal([exp10(-c[1]),exp10(-c[1])]), [0.0]), (x,p)->[p[1]*x[1] + p[2], p[1]*x[2] + p[2]], rand(3))
DMU2 = DataModel(DataSetUncertain(Windup([1,2,3,4],2), Windup([4,5,6.5,9],2), (x,y,c)->diagm([exp10(-c[1]),exp10(-c[1])]), [0.0]), (x,p)->[p[1]*x[1] + p[2], p[1]*x[2] + p[2]], rand(3))
@test MLE(DMU1) ≈ MLE(DMU2)
IIμ1, IIΛ1 = SecondFundamentalForm(DMU1, MLE(DMU1))
IIμ2, IIΛ2 = SecondFundamentalForm(DMU2, MLE(DMU2))
@test isapprox(IIμ1, IIμ2; atol=1e-11)
@test isapprox(IIΛ1, IIΛ2; atol=1e-10, rtol=1e-8)
@test isapprox(EfronScalarCurvature(DMU1, MLE(DMU1)), EfronScalarCurvature(DMU2, MLE(DMU2)); atol=1e-12, rtol=1e-10)

## UnknownVarianceDataSet is not supported yet (x-values are fitted parameters, i.e. the ambient
## space also contains an x-block), but yInvCovProper and ysigmaProper work for it
XU = 1:5;    YU = 2 .* XU .+ 1 .+ 0.1 .* randn(5)
DSUV = UnknownVarianceDataSet(XU, YU, (x,y,c)->exp10(-c[1]), (x,y,c)->exp10(-c[1]), [log10(0.3)], [log10(0.2)])
θV = [Float64.(XU); 0.4; 0.2; log10(0.3); log10(0.2)]
DMUV = DataModel(DSUV, ModelMap(LinearModel), θV; SkipOptim=true, SkipTests=true)
SplitV = InformationGeometry.SplitErrorParams(DSUV)(θV)
LiftV = InformationGeometry.LiftedEmbedding(DSUV, Predictor(DMUV), length(SplitV[1]) - length(XU))(SplitV[1])
PredV = view(LiftV, length(XU)+1:length(LiftV))
@test InformationGeometry.yInvCovProper(DMUV, θV) ≈ InformationGeometry._InvCov(InformationGeometry.yinverrormodel(DSUV), SplitV[end], InformationGeometry.WoundX(DSUV), InformationGeometry.Windup(PredV, ydim(DSUV))) |> x->InformationGeometry._MaskedUVD(x, DSUV.datakeep)
@test all(isfinite, InformationGeometry.ysigmaProper(DMUV, θV))
@test_throws String SecondFundamentalForm(DMUV, θV)
@test_throws String SFFNormSquared(DMUV, θV)
