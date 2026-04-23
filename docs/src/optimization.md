



### Maximum Likelihood Estimation
```@setup Multistart
using InformationGeometry, Plots; gr()
```

Unless the kwarg `SkipOptim=true` is passed to the `DataModel` constructor, it automatically tries to optimize the model parameters in an attempt to find the MLE. If no initial guess for the values parameters is provided, the constructor will try to infer the correct number of parameters and make a random initial guess.
However, more control can be exerted over the optimization process by choosing an appropriate optimization method and corresponding options. These options may either be be passed to the `DataModel` constructor, or the optimization method `InformationGeometry.minimize` may be invoked explicitly after the `DataModel` was constructed with either unsuccessful optimization or skipping the automatic initial optimization altogether.

Most importantly, `InformationGeometry.minimize` is compatible with the [**Optimization.jl**](https://github.com/SciML/Optimization.jl) ecosystem of optimizer methods via the `meth` keyword. Best support is available for optimizers from the [**Optim.jl**](https://github.com/JuliaNLSolvers/Optim.jl) via a custom wrapper. Choosing `meth=nothing` also allows for using a Levenberg-Marquardt method from the [**LsqFit.jl**](https://github.com/JuliaNLSolvers/LsqFit.jl) optimizer, which is however only possible if data with fixed Gaussian uncertainties and no priors are used.

```@example Multistart
DM = DataModel(DataSet(1:4, [4,5,6.5,9], [0.5,0.45,0.6,1]), (x,p)->(p[1]+p[2])*x + exp(p[1]-p[2]), [1.3, 0.2]; SkipOptim=true)
plot(DM; Confnum=0)
```

The optimization can then be performed with:
```@example Multistart
using Optim
mle = InformationGeometry.minimize(DM; meth=Optim.NewtonTrustRegion(), tol=1e-12, maxtime=60.0, Domain=HyperCube(zeros(2), 10ones(2)), verbose=true)
```
The full solution object is returned if the keyword argument `Full=true` is additionally provided.

Alternatively, one might use [optimizers](https://docs.sciml.ai/Optimization) from the [**Optimization.jl**](https://github.com/SciML/Optimization.jl) ecosystem e.g. via
```julia
using Optimization, OptimizationOptimisers
mle = InformationGeometry.minimize(DM, rand(2); meth=OptimizationOptimisers.AdamW())
```
or
```julia
using Optimization, OptimizationNLopt
mle = InformationGeometry.minimize(DM, rand(2); meth=OptimizationNLopt.GN_DIRECT())
```

Finally, the newly found optimum can be visually inspected with
```@example Multistart
plot(DM, mle)
```
and added to the original unoptimized `DataModel` object via
```julia
DM = DataModel(Data(DM), Predictor(DM), mle; SkipOptim=true)
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
MultistartFit(DM, MvNormal([0,0], Diagonal(ones(2))); MultistartDomain=HyperCube([-1,-1],[3,4]), N=200, meth=Newton())
```



The results of a multistart optimization can be visualized with a plot of the sorted final objective values, ideally showing prominent step-like features where the optimization reliably converged to the same local optima multiple times.
```@example Multistart
WaterfallPlot(R; BiLog=true)
```
It can also be useful to investigate whether the parameter configurations within one "step" of the plot, where the final objective function value was the same, are actually "close" to each other, or whether there are distinct (or spread out) local optima, which nevertheless produce a fit of the same quality. This can be plotted via the `ParameterPlot` method, which can display the data either in a `:dotplot`, `:boxplot` or `:violin` plot. 
```@example Multistart
using StatsPlots
ParameterPlot(R; st=:dotplot)
```
Seeing a spread in the `ParameterPlot` for the parameter configurations of the lowest "step" is already an indication that some parameter may be non-identifiable.

Exemplary parameter configurations from the `n`-th step in a Waterfall plot can be conviently retrieved via `GetStepParameters(R, n)` for plotting and other analyses.
For instance, to compare the fit corresponding to the local optimum constituted by the second step:
```@example Multistart
plot(DM, GetStepParameters(R, 2))
```


### Robust Prefitting

While the above described methods `InformationGeometry.minimize` and `MultistartFit` allow for transparent and explicit control over individual optimization processes, the methods `Prefit` and `Minimize` are provided as convenient wrappers for combining these different functionalities recursively in powerful ways.

```@docs
Minimize(::DataModel)
Prefit(::DataModel)
```

The `Minimize` method merges the `MultistartFit` interface with the `InformationGeometry.minimize` interface, allowing the user to simply specify the number of multistarts via the `Multistart::Int` kwarg, where `Minimize` will only return the best parameter configuration observed during the multistart optimization and the default `Multistart=0` falls back to using `InformationGeometry.minimize`.

```julia
Minimize(DM; MultistartDomain=HyperCube([-1,-1],[3,4]), Multistart=200, meth=Newton())
```

The `Prefit` method allows for chaining multiple optimizations with different optimizers together in series and returning only the best observed parameter configuration.
Moreover, it allows for sharing settings between the runs or choosing different settings per run by supplying the choices as a vector to the corresponding kwargs.
For example:
```julia
using OptimizationOptimisers, OptimizationOptimJL
Prefit(DM; meth=[OAdam(), LBFGS(), Newton()], maxiters=[3000, 500, 50], tol=[0, 1e-8, 1e-12])
```

Almost all "higher order" wrapping methods for optimization (e.g. `Prefit`, `IncrementalTimeSeriesFit`, `RobustFit`, `ParameterProfiles`, etc.) accept the `MinimizeFunc` kwarg, which can be used to specify which concrete function should be used for the optimization. Since many of these methods share a similar interface, they can be combined and nested in complex ways.
For instance, multistart optimization can be performed, where each of the individual runs in the multistart is carried out with a sequence of different optimizers with different settings via `Prefit`:
```julia
Minimize(DM; Multistart=100, MultistartDomain=HyperCube([-1,-1],[3,4]), MinimizeFunc=Prefit, meth=[OAdam(), LBFGS(), Newton()], maxiters=[3000, 500, 50], tol=[0, 1e-8, 1e-12])
```
This results in 100 runs, where the random initial parameter configurations are first improved via robust stochastic optimizers to guide the optimization into roughly the correct region of parameter space and then refined with deterministic derivative-based optimizers.
For complex models which easily become stiff for ill-chosen parameter configurations, this can significantly improve both convergence and overall computational effort of the multistart.
This is because the deterministic optimisers get stuck far less often in parameter regions where the model is stiff and derivatives are more both more time-intensive to compute and less reliable.
Therefore, less computational time is wasted on runs which eventually do not converge to a local optimum.


!!! note
    By default, the `Prefit` method uses `Domain=nothing` to turn of boundaries for parameters in order to improve compatibility with different optimization methods, which might not support specification of boundaries.
    However, if the chosen optimization method(s) support(s) boundaries, this can be turned back by manually supplying the kwarg `Domain`.


```@docs
InformationGeometry.minimize
MultistartFit
LocalMultistartFit
WaterfallPlot
ParameterPlot(::MultistartResults)
IncrementalTimeSeriesFit(::DataModel)
RobustFit
```