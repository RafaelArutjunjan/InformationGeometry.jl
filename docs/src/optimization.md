
### Maximum Likelihood Estimation




### Multistart Optimization

When a reasonable estimate for the initial parameter configuration is not available and optimizations starting from the approximate center of the parameter domain do not converge to a suitable optimum, a systematic search of the parameter space is needed. This can be achieved with the `MultistartFit` method, which will sample the parameter space ``\mathcal{M}`` either according to a given probability distribution (e.g. uniform / normal) on ``\mathcal{M}`` or by drawing the parameter values from a [low-discrepancy sequence](https://en.wikipedia.org/wiki/Low-discrepancy_sequence) such as the [Sobol](https://github.com/JuliaMath/Sobol.jl) sequence. Compared with uniformly drawn values, values drawn from low-discrepancy sequences achieve a more even and "equidistant" coverage, thereby slightly increasing the chances of discovering a larger number of distinct local optima overall.

Just like `InformationGeometry.minimize`, the `MultistartFit` method is also compatible with the [**Optimization.jl**](https://github.com/SciML/Optimization.jl) ecosystem of optimizer methods via the `meth` keyword.
```julia
R = MultistartFit(DM)
```
The most relevant keywords are:
* `MultistartDomain::HyperCube` for defining the domain from which the initial guesses are drawn
* `N::Int` for choosing the number of starts
* `meth` for choosing the optimizer
* `resampling=false` disables the drawing of new guesses when the objective function cannot be evaluated in some regions of the parameter space
* `maxval::Real=1e5` if no `MultistartDomain` is specified, a cube of size `(-maxval, maxval)^(×n)` is constructed

Returns a `MultistartResults` object, which saves some additional information such as the number of iterations taken per run, the initial guesses, etc. The parameter configuration is obtained with `MLE(R)`.


The results of a multistart optimization can be visualized with a plot of the sorted final objective values, ideally showing prominent step-like features where the optimization reliably converged to the same local optima multiple times.
```julia
WaterfallPlot(R)
```
It can also be useful to investigate whether the parameter configurations within one "step" of the plot, where the final objective function value was the same, are actually "close" to each other, or whether there are distinct (or spread out) local optima, which nevertheless produce a fit of the same quality. This can be plotted via the `ParameterPlot` method, which can display the data either in a `:dotplot`, `:boxplot` or `:violin` plot. 
```julia
using StatsPlots
ParameterPlot(R)
```
Seeing a spread in the `ParameterPlot` for the parameter configurations of the lowest "step" is already an indication that some parameter may be non-identifiable.


```@docs
MultistartFit
LocalMultistartFit
WaterfallPlot
ParameterPlot(::MultistartResults)
IncrementalTimeSeriesFit(::DataModel)
RobustFit
```