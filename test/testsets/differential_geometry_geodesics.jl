using InformationGeometry, Test, LinearAlgebra, StaticArrays, BoundaryValueDiffEq

S2metric((θ,ϕ)) = [1.0 0; 0 sin(θ)^2]
function S2Christoffel((θ,ϕ))
    Symbol = zeros(typeof(ϕ),2,2,2);    Symbol[1,2,2] = -sin(θ)*cos(θ)
    Symbol[2,1,2] = Symbol[2,2,1] = cos(θ)/sin(θ);  Symbol
end
# Calculation by hand works out such that in this special case:
S2Ricci(x) = S2metric(x)
ConstMetric(x) = Diagonal(ones(2))

@test abs(GeodesicDistance(ConstMetric,[0,0],[1,1]) - sqrt(2)) < 2e-8
# Errors on Shooting methods:
@test abs(GeodesicDistance(S2metric,[π/4,1],[3π/4,1]; BVPmeth=MIRK2(), dt=0.02) - π/2) < 1e-8
@test abs(GeodesicDistance(S2metric,[π/2,0],[π/2,π/2]; BVPmeth=MIRK2(), dt=0.02) - π/2) < 1e-8

DS = DataSet([0,0.5,1],[1.,3.,7.],[1.2,2.,0.6]);    DM = DataModel(DS, (x,p) -> p[1]^3 *x + p[2]^3)
y = MLE(DM) + 0.2(rand(2) .- 0.5)
geo = GeodesicBetween(DM, MLE(DM), y; BVPmeth=MIRK2(), dt=0.02, tol=1e-11)
@test norm(MLE(DM) - [1.829289173660125,0.942865200406147]) < 1e-7

Len = GeodesicLength(DM,geo)
@test abs(InformationGeometry.ParamVol(geo) * InformationGeometry.GeodesicEnergy(DM,geo) - Len^2) < 1e-8
Confnum = InvConfVol(ChisqCDF(pdim(DM), 2*(LogLikeMLE(DM) - loglikelihood(DM, y))))
@test InformationGeometry.GeodesicRadius(DM, Confnum) - Len < 1e-5

# Apply logarithmic map first since it is typically multi-valued for positively curved manifolds.
@test norm(ExponentialMap(FisherMetric(DM), MLE(DM), LogarithmicMap(FisherMetric(DM), MLE(DM), y; BVPmeth=MIRK2(), dt=0.02)) - y) < 1