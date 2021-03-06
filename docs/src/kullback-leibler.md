


### Kullback-Leibler Divergences

Using the `Distributions` type provided by [**Distributions.jl**](https://github.com/JuliaStats/Distributions.jl), the `KullbackLeibler` method offers a convenient way of computing the [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence) between distributions. In several cases an analytical expression for the Kullback-Leibler divergence is known. These include: (univariate and multivariate) Normal, Cauchy, Exponential, Weibull and Gamma distributions.

Furthermore, for distributions over a one-dimensional domain where no analytic result is known, `KullbackLeibler` rephrases the integral in terms of an ODE and employs an efficient integration scheme from the [**DifferentialEquations.jl**](https://github.com/SciML/DifferentialEquations.jl) suite. For multivariate distributions, Monte Carlo integration is used.

Examples of use:
```julia
KullbackLeibler(Cauchy(1.,2.4), Normal(-4,0.5), HyperCube([-100,100]); tol=1e-12)
KullbackLeibler(MvNormal([0,2.5],diagm([1,4.])), MvTDist(1,[3,2],diagm([2.,3.])), HyperCube([[-50,50],[-50,50]]); Carlo=true, N=Int(1e8))
```

In addition, it is of course also possible to input generic functions, whose positivity and normalization should be ensured by the user.
```@docs
KullbackLeibler(::Function,::Function,::HyperCube)
```

For example, the Kullback-Leibler divergence between a Cauchy distribution with ``\mu=1`` and ``s=2`` and a normal (i.e. Gaussian) distribution with ``\mu=-4`` and ``\sigma=1/2`` can be calculated via:
```@example 2
using InformationGeometry # hide
using LinearAlgebra, Distributions
KullbackLeibler(Cauchy(1.,2.), Normal(-4.,0.5), HyperCube([-100,100]); tol=1e-12)
```
Specifically, the keyword arguments used here numerically compute the divergence over the domain ``[-100,100]`` to a tolerance of ``\approx 10^{-12}``.

The domain of the integral involved in the computation of the divergence is specified using the `HyperCube` datatype, which stores a cuboid region in ``N`` dimensions as a vector of intervals.
```@docs
HyperCube
```

Furthermore, the Kullback-Leibler divergence between multivariate distributions can be computed for example by
```@example 2
KullbackLeibler(MvNormal([0,2.5],diagm([1,4.])), MvTDist(1,[3,2],diagm([2.,3.])), HyperCube([[-20,50],[-20,50]]); tol=1e-8)
```
using adaptive integration methods from [**HCubature.jl**](https://github.com/JuliaMath/HCubature.jl). Alternatively, Monte Carlo integration can be employed by specifying `Carlo=true`:
```@example 2
KullbackLeibler(MvNormal([0,2.5],diagm([1,4.])), MvTDist(1,[3,2],diagm([2.,3.])), HyperCube([[-50,50],[-50,50]]); Carlo=true, N=Int(5e6))
```
In addition, the keyword argument `N` now determines the number of points where the integrand is evaluated over the given domain ``[-50,50] \times [-50,50]``.

So far, importance sampling has not been implemented for the Monte Carlo integration. Instead, the domain is sampled uniformly.
