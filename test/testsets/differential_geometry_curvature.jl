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