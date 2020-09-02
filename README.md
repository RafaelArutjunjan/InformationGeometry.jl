# InformationGeometry.jl
[![Build Status](https://travis-ci.com/RafaelArutjunjan/InformationGeometry.jl.svg?branch=master)](https://travis-ci.com/RafaelArutjunjan/InformationGeometry.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/RafaelArutjunjan/InformationGeometry.jl?svg=true)](https://ci.appveyor.com/project/RafaelArutjunjan/InformationGeometry-jl)
[![Codecov](https://codecov.io/gh/RafaelArutjunjan/InformationGeometry.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/RafaelArutjunjan/InformationGeometry.jl)
[![Coveralls](https://coveralls.io/repos/github/RafaelArutjunjan/InformationGeometry.jl/badge.svg?branch=master)](https://coveralls.io/github/RafaelArutjunjan/InformationGeometry.jl?branch=master)

This package offers a set of basic tools to compute quantities of interest in information geometry and statistical analysis.
Among these are (log-)likelihoods, Fisher Metrics, Kullback-Leibler divergences, Geodesics, Riemann and Ricci curvature tensors and so on.
In particular, this package allows the user to efficiently compute the exact boundary of confidence intervals of a non-linear statistical model given a dataset.

An explanation of how these methods work can be found in my [Master's Thesis](https://github.com/RafaelArutjunjan/Master-Thesis).


DataModels
----------
The `DataSet` and `DataModel` types represent containers to store datasets and models and conveniently pass them to functions.
Some elementary examples:
```
DS = DataSet([1,2,3.],[4,5,6.5],[0.5,0.45,0.6])
model(x,p) = p[1] .* x .+ p[2]
DM = DataModel(DS,model)
loglikelihood(DM::DataModel,p::Vector)
MLE = FindMLE(DM)
```



Calculating Kullback-Leibler divergences
----------------------------------------

Using the distribution types defined in `Distributions.jl`, this package implements a variety of schemes for the numerical evaluation of the [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence) between distributions over a specified domain:
* For univariate distributions, the integral is rephrased in the form of an ordinary differential equation and solved with the sophisticated solvers provided by `DifferentialEquations.jl`.
* For multivariate distributions, Monte Carlo simulation is used to estimate the integral.
* In many cases where closed-form expressions for the Kullback-Leibler are known. These include: Normal, Cauchy, Exponential, Weibull, Gamma

Examples of use:
```
KullbackLeibler(Cauchy(1.,2.4),Normal(-4,0.5),HyperCube([-100,100]),Carlo=false,tol=1e-12)
KullbackLeibler(MvNormal([0,2.5],diagm([1,4.])),MvTDist(1,[3,2],diagm([2.,3.])),HyperCube([[-50,50],[-50,50]]),N=Int(1e8))
```


Future Plans for this package
-----------------------------
`InformationGeometry.jl` is a loose collection of code which I wrote especially to perform calculations in the specific context of my [Master's Thesis](https://github.com/RafaelArutjunjan/Master-Thesis). As such, it was not originally written as a module but instead mostly optimized to fit my personal needs. Now that my thesis is finished, I plan on revising the internals of this package to improve performance and significantly extend its functionality. In addition, I will do my best to provide detailed documentation and examples of how to use this package over the coming weeks.


Todo:
-----
* Fisher Metrics: Allow for non-normal error distributions by interpolating and deriving the Kullback-Leibler divergence over a domain
* Parallelism: Improve support for parallel computations of geodesics, curvature tensors and so on
* Divergences: Discrete distributions, more user control over parallelization for Monte Carlo, Importance Sampling
* Plotting: Improve capabilities for high-dimensional models
* Documentation: More docstrings, show concrete examples / write guide
* Interface: Standardize the user-facing keyword arguments
* Benchmarks: Provide benchmarks for how `InformationGeometry.jl` compares to conventional inference methods
* Optimisation: Use IntervalArithmetic
