
### Maximum Likelihood Estimation
```@setup Multistart
using InformationGeometry, Plots; gr()
```

Unless the kwarg `SkipOptim=true` is passed to the `DataModel` constructor, it automatically tries to optimize the model parameters in an attempt to find the MLE. If no initial guess for the values parameters is provided, the constructor will try to infer the correct number of parameters and make a random initial guess.
However, more control can be exerted over the optimization process by choosing an appropriate optimization method and corresponding options. These options may either be be passed to the `DataModel` constructor, or the optimization method `InformationGeometry.minimize` may be invoked explicitly after the `DataModel` was constructed with either unsuccessful optimization or skipping the automatic initial optimization altogether.

Most importantly, `InformationGeometry.minimize` is compatible with the [**Optimization.jl**](https://github.com/SciML/Optimization.jl) ecosystem of optimizer methods via the `meth` keyword. Best support is available for optimizers from the [**Optim.jl**](https://github.com/JuliaNLSolvers/Optim.jl) via a custom wrapper. Choosing `meth=nothing` also allows for using a Levenberg-Marquardt method from the [**LsqFit.jl**](https://github.com/JuliaNLSolvers/LsqFit.jl) optimizer, which is however only possible if data with fixed Gaussian uncertainties and no priors are used.
```@example Multistart
DM = DataModel(DataSet(1:3, [4,5,6.5], [0.5,0.45,0.6]), (x,p)->(p[1]+p[2])*x + exp(p[1]-p[2]))
plot(DM)
```



### Multistart Optimization

When a reasonable estimate for the initial parameter configuration is not available and optimizations starting from the approximate center of the parameter domain do not converge to a suitable optimum, a systematic search of the parameter space is needed. This can be achieved with the `MultistartFit` method, which will sample the parameter space ``\mathcal{M}`` either according to a given probability distribution (e.g. uniform / normal) on ``\mathcal{M}`` or by drawing the parameter values from a [low-discrepancy sequence](https://en.wikipedia.org/wiki/Low-discrepancy_sequence) such as the [Sobol](https://github.com/JuliaMath/Sobol.jl) sequence. Compared with uniformly drawn values, values drawn from low-discrepancy sequences achieve a more even and "equidistant" coverage, thereby slightly increasing the chances of discovering a larger number of distinct local optima overall.

Just like `InformationGeometry.minimize`, the `MultistartFit` method is also compatible with the [**Optimization.jl**](https://github.com/SciML/Optimization.jl) ecosystem of optimizer methods via the `meth` keyword.
```@example Multistart
using Optim
R = MultistartFit(DM; N=200, maxval=500, meth=LBFGS())
```
The most relevant keywords are:
* `MultistartDomain::HyperCube` for defining the domain from which the initial guesses are drawn
* `N::Int` for choosing the number of starts
* `meth` for choosing the optimizer
* `resampling=false` disables the drawing of new guesses when the objective function cannot be evaluated in some regions of the parameter space
* `maxval::Real=1e5` if no `MultistartDomain` is specified, a cube of size `(-maxval, maxval)^(×n)` is constructed

This returns a `MultistartResults` object, which saves some additional information such as the number of iterations taken per run, the initial guesses, etc. The parameter configuration is obtained with `MLE(R)`.
Alternatively, one could for example specify a distribution for the initials of the multistart optimization via
```julia
MultistartFit(DDM, MvNormal([0,0], Diagonal(ones(2))), ProfileDomain=HyperCube([-1,-1],[3,4]), N=200, meth=Newton(), plot=false)
```



The results of a multistart optimization can be visualized with a plot of the sorted final objective values, ideally showing prominent step-like features where the optimization reliably converged to the same local optima multiple times.
```@example Multistart
WaterfallPlot(R; BiLog=true)
```
It can also be useful to investigate whether the parameter configurations within one "step" of the plot, where the final objective function value was the same, are actually "close" to each other, or whether there are distinct (or spread out) local optima, which nevertheless produce a fit of the same quality. This can be plotted via the `ParameterPlot` method, which can display the data either in a `:dotplot`, `:boxplot` or `:violin` plot. 
```julia
using StatsPlots
ParameterPlot(R; st=:dotplot)
```
Seeing a spread in the `ParameterPlot` for the parameter configurations of the lowest "step" is already an indication that some parameter may be non-identifiable.


```@docs
InformationGeometry.minimize
MultistartFit
LocalMultistartFit
WaterfallPlot
ParameterPlot(::MultistartResults)
IncrementalTimeSeriesFit(::DataModel)
RobustFit
```