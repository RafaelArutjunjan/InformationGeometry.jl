
### Plotting Fit Results

```@setup Plotting
using InformationGeometry, Plots;    gr()
```
Given a dataset type or `DataModel`, there are pre-defined recipes for the [**Plots.jl**](https://github.com/JuliaPlots/Plots.jl) package so that they can be visualized via:
```@example Plotting
DM = DataModel(DataSet(1:3, [4,5,6.5], [0.5,0.45,0.6]), (x,p)->(p[1]+p[2])*x + exp(p[1]-p[2]), [1.3, 0.2])
plot(DM, MLE(DM))
```
where the best fit corresponding to the MLE parameters `MLE(DM)` is drawn in by default. Alternatively, other parameter values may optionally be specified in the second argument.

By default, a linearized (i.e. *approximate*) confidence band is computed around the prediction via [Gaussian error propagation](https://en.wikipedia.org/wiki/Propagation_of_uncertainty) of the inverse Fisher information, which provides a *lower* bound on the parameter uncertainty by the [Cramér-Rao theorem](https://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93Rao_bound#Multivariate_case).

A desired (vector of) confidence level(s) in units of ``\sigma`` can be specified via the `Confnum` keyword for more control over the bands, with `Confnum=0` disabling the bands.
```@example Plotting
plot(DM; Confnum=[1,2])
```
These linearized pointwise confidence bands only take into account the uncertainties in the fit parameters and therefore quantify pointwise at every input `x` (again, *approximately* for non-linearly parametrized models!) and quantify the interval in which the model prediction of the _true_ parameters is estimated to lie with the specified confidence, given the observed data.

However, depending on the uncertainty associated with the measurement process, future recorded data points can lie far outside the confidence bands.
In contrast, to assess the range in which future validation measurements are likely to land, one can draw bands where the data uncertainty is added on top of the prediction uncertainty due to the parameters. This can be achieved via the `Validation=true` keyword in the `plot` method, where the `Confnum` can be controlled as before:
```@example Plotting
plot(DM; Confnum=[1,2], Validation=true)
```
In the literature, these are often instead referred to as "prediction bands" instead of validation bands. However, this terminology is somewhat prone to confusion in my humble opinion which is why I personally prefer to avoid it.

The linearized confidence and validation bands are computed via the [`VariancePropagation`](@ref) and [`ValidationPropagation`](@ref) methods respectively.

For `DataModel`s or simply datasets which have multiple different components (i.e. `ydim > 1`), the `Symbol` `:Individual` may be appended as the last argument to split the components into separate plots:
```@example Plotting
DS122 = DataSet([1,2,3],[2,1,4,2,6.8,3.5],[0.5,0.5,0.45,0.45,0.55,0.6], (3,1,2))
DM122 = DataModel(DS122, (x,p)-> [p[1]*x, p[2]*x])
plot(DM122, MLE(DM122), :Individual; Confnum=1, Validation=false)
```
as compared with plotting both in one:
```@example Plotting
plot(DM122, MLE(DM122); Confnum=1, Validation=false)
```

### Residuals


```docs
PlotQuantiles
ResidualVsFitted
ResidualPlot
```

### Plotting Parameters


```docs
ParameterPlot
ParameterSavingCallback
TracePlot
```
