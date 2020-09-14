# InformationGeometry.jl

*A Julia package for geometric analyses of statistical problems.*

| **Build Status**                                                                                |
|:-----------------------------------------------------------------------------------------------:|
| [![][travis-img]][travis-url] [![][appveyor-img]][appveyor-url] [![][codecov-img]][codecov-url] |

This package offers a set of basic tools to compute quantities of interest in information geometry and statistical analysis.
Among these are (log-)likelihoods, Fisher Metrics, Kullback-Leibler divergences, Geodesics, Riemann and Ricci curvature tensors and so on.
In particular, this package allows the user to efficiently compute the exact boundary of confidence intervals of a non-linear statistical model given a dataset.

An explanation of how these methods work can be found in my [Master's Thesis](https://github.com/RafaelArutjunjan/Master-Thesis).


DataModels
----------
The `DataSet` and `DataModel` types represent immutable containers to store datasets and models and conveniently pass them to functions.
Some elementary examples:
```julia
using InformationGeometry
DS = DataSet([1,2,3.],[4,5,6.5],[0.5,0.45,0.6])
model(x,p) = p[1] .* x .+ p[2]
DM = DataModel(DS,model)
```
If provided like this, the gradient of the model with respect to the parameters `p` (i.e. its "Jacobian") will be calculated using automatic differentiation. Alternatively, an explicit analytic expression for the Jacobian can be specified by hand:
```julia
function dmodel(x,p::Vector)
   J = Array{Float64}(undef, length(x), length(p))
   @. J[:,1] = x        # ∂(model)/∂p₁
   @. J[:,2] = 1.       # ∂(model)/∂p₂
   return J
end
DM = DataModel(DS,model,dmodel)
```
The output of the Jacobian must be a matrix whose columns correspond to the partial derivatives with respect to different components of `p` and whose rows correspond to evaluations at different values of `x`.

Given such a `DataModel`, further quantities of interest can be calculated, e.g. via
```julia
loglikelihood(DM::DataModel,p::Vector)
FindMLE(DM::DataModel)
Score(DM::DataModel,p::Vector)
FisherMetric(DM::DataModel,p::Vector)
GeometricDensity(DM::DataModel,p::Vector)
ChristoffelSymbol(DM::DataModel,p::Vector)
Riemann(DM::DataModel,p::Vector)
Ricci(DM::DataModel,p::Vector)
RicciScalar(DM::DataModel,p::Vector)
AIC(DM::DataModel,p::Vector)
GeodesicDistance(DM::DataModel,p::Vector,q::Vector)
```
where `Riemann` returns the components of the (1,3)-Riemann tensor and `ChristoffelSymbol` returns the (1,2) form of the Christoffel symbols, that is, the Christoffel symbols "of the second kind".


Calculating Kullback-Leibler divergences
----------------------------------------
Using the distribution types defined in [Distributions.jl](https://github.com/JuliaStats/Distributions.jl), this package implements a variety of schemes for the numerical evaluation of the [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence) between distributions over a specified domain:
* For univariate distributions, the integral is rephrased in the form of an ordinary differential equation and solved with the sophisticated methods provided by [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl).
* For multivariate distributions, Monte Carlo simulation is used to estimate the integral.
* In many cases, closed-form expressions for the Kullback-Leibler divergence are known. These include: Normal, Cauchy, Exponential, Weibull, Gamma

Examples of use:
```julia
KullbackLeibler(Cauchy(1.,2.4),Normal(-4,0.5),HyperCube([-100,100]),Carlo=false,tol=1e-12)
KullbackLeibler(MvNormal([0,2.5],diagm([1,4.])),MvTDist(1,[3,2],diagm([2.,3.])),HyperCube([[-50,50],[-50,50]]),N=Int(1e8))
```

Installation
------------
As with any Julia package, `InformationGeometry.jl` can be added from the Julia terminal via
```julia
julia> ] add InformationGeometry
```
or alternatively by
```julia
julia> using Pkg; Pkg.add("InformationGeometry")
```

Future Plans for this package
-----------------------------
`InformationGeometry.jl` is a loose collection of code which I wrote especially to perform calculations in the specific context of my [Master's Thesis](https://github.com/RafaelArutjunjan/Master-Thesis). As such, it was not originally written with its publication as a module in mind but instead mostly optimized to fit my personal needs. Now that my thesis is finished, I plan on revising the internals of this package to improve performance and significantly extend its functionality. In addition, I will do my best to provide detailed documentation and examples of how to use this package over the coming weeks.


Todo:
-----
* Fisher Metrics: Allow for non-normal uncertainties in measurements e.g. by interpolating and deriving the Kullback-Leibler divergence over a domain
* Parallelism: Improve support for parallel computations of geodesics, curvature tensors and so on
* Divergences: Discrete distributions, more user control over parallelization for Monte Carlo, Importance Sampling
* Plotting: Improve visualization capabilities for high-dimensional models
* Documentation: More docstrings, show concrete examples / write guide
* Interface: Standardize the user-facing keyword arguments
* Benchmarks: Provide benchmarks for how `InformationGeometry.jl` compares to conventional inference methods
* Optimisation: Use IntervalArithmetic









[travis-img]: https://travis-ci.com/RafaelArutjunjan/InformationGeometry.jl.svg?branch=master
[travis-url]: https://travis-ci.com/RafaelArutjunjan/InformationGeometry.jl

[appveyor-img]: https://ci.appveyor.com/api/projects/status/github/RafaelArutjunjan/InformationGeometry.jl?svg=true
[appveyor-url]: https://ci.appveyor.com/project/RafaelArutjunjan/InformationGeometry-jl

[codecov-img]: https://codecov.io/gh/RafaelArutjunjan/InformationGeometry.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/RafaelArutjunjan/InformationGeometry.jl
