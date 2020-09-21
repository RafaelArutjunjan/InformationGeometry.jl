


### Kullback-Leibler Divergences

Using the `Distribution` types established in **Distributions.jl**, the `KullbackLeibler` method offers a convenient way of computing Kullback-Leibler divergences between distributions. In several cases an analytical expression for the Kullback-Leibler divergence is known. These include: (multivariate) Normal, Cauchy, Exponential, Weibull and Gamma distributions.

Furthermore, for distributions over a one-dimensional domain where no analytic result is known, `KullbackLeibler` rephrases the integral in terms of an ODE and employs an efficient integration scheme from the **DifferentialEquations.jl** suite. For multivariate distributions, Monte Carlo integration is used.


In addition, it is of course also possible to input generic functions, whose positivity and normalization should be ensured by the user.
```@docs
KullbackLeibler(::Function,::Function)
```

```@example 2
using InformationGeometry # hide
using LinearAlgebra, Distributions
KullbackLeibler(Cauchy(1.,2.4),Normal(-4,0.5),HyperCube([-100,100]),Carlo=false,tol=1e-12)
```
The `HyperCube` datatype specifies a cuboid region in ``N`` dimensions by storing a vector of intervals.
```@docs
HyperCube
```
```@example 2
KullbackLeibler(MvNormal([0,2.5],diagm([1,4.])),MvTDist(1,[3,2],diagm([2.,3.])),HyperCube([[-50,50],[-50,50]]),N=Int(1e8))
```


**To be continued...**
