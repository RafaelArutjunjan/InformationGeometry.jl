

### Model Transformations

Occasionally, one might wish to perform coordinate transformations on the parameter space of a model without having to redefine the entire model as this can be a cumbersome process for complex models. For example, this might be useful in the fitting process when the allowable parameter range spans several orders of magnitude or when trying to enforce the positivity of parameters.

A few methods are provided to make this process more convenient. These include: `LogTransform`, `ExpTransform`, `Log10Transform`, `Exp10Transform`, `ReflectionTransform` and `ScaleTransform`.
These methods accept a vector of booleans as an optional second argument to restrict the application of the transformation to specific parameter components if desired.
For first argument, one can either provide just a model function to obtain its transformed counterpart or alternatively supply an entire `DataModel` to be transformed.

```@example 5
using InformationGeometry # hide
DM = DataModel(DataSet([1,2,3,4], [4,5,6.5,9], [0.5,0.45,0.6,1]), LinearModel)
logDM = LogTransform(DM)
ExpDM = ExpTransform(Exp10Transform(DM, [false, true]), [true, false])
SymbolicModel(logDM), SymbolicModel(ExpDM)
```

It is also possible to provide other differentiable functions for parameter transformations by hand using the following method:
```@docs
ComponentwiseModelTransform
```
The provided scalar function `F` should be strictly monotonic to avoid problems when differentiating the model.

In addition to componentwise application of scalar functions to the parameters, there are also higher-dimensional transformations such as `TranslationTransform`, `LinearTransform` and their combination `AffineTransform` which allow for mixing between the components.

Lastly, the method `LinearDecorrelation` is a special case of `AffineTransform` which subtracts the MLE from the parameters and applies the cholesky decomposition (i.e. "square root") of the inverse Fisher metric at the best fit. This centers the confidence regions on the origin and will result in confidence boundaries which constitute concentric circles / spheres for linearly parametrized models. For models which are non-linear with respect to their parameters, the confidence boundaries of the "linearly decorrelated" model showcase the deviations of the confidence boundaries of the original model from ellipsoidal shape, therefore nicely illustrating the magnitude of the coordinate distortion present on the parameter space.

For general (differentiable) multivariable transformations on the parameter space, one can use:
```@docs
ModelEmbedding
```
