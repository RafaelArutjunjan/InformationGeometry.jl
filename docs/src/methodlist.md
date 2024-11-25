
### List of useful methods


The following lists docstrings for various important functions.




Once a `DataModel` object has been defined, it can subsequently be used to compute various quantities as follows:

```@docs
loglikelihood(::DataModel,::Vector{Float64})
MLE(::DataModel)
MLEuncert(::DataModel)
LogLikeMLE(::DataModel)
```

Various geometric quantities which are intrinsic to the parameter manifold ``\mathcal{M}`` can be computed as a result of the Fisher metric ``g`` (and subsequent choice of the Levi-Civita connection) such as the Riemann and Ricci tensors and the Ricci scalar ``R``.
```@docs
Score(::DataModel,::Vector{Float64})
FisherMetric(::DataModel,::Vector{Float64})
GeometricDensity(::DataModel,::Vector{Float64})
ChristoffelSymbol(::Function,::Vector{Float64})
Riemann(::Function,::Vector{Float64})
Ricci(::Function,::Vector{Float64})
RicciScalar(::Function,::Vector{Float64})
```

Further, studying the geodesics associated with a metric manifold can yield insights into its geometry.
```@docs
GeodesicDistance(::DataModel,::Vector{Float64},::Vector{Float64})
AIC(::DataModel,::Vector{Float64})
AICc(::DataModel,::Vector{Float64})
BIC(::DataModel,::Vector{Float64})
IsLinearParameter
ConfidenceRegionVolume
Pullback
Pushforward
```

In many applied settings, one often does not have a dataset of sufficient size for all parameters in the model to be "practically identifiable", which means that bounded confidence regions may only exist for very low confidence levels (e.g. up to ``0.1\sigma``). In such cases, it is still possible to compute radial geodesics emanating from the MLE to study the geometry of the parameter space.

A slightly more robust alternative to using geodesics is given by the so-called profile likelihood method. Essentially, it consists of pinning one of the parameters at particular values on a grid, while optimizing the remaining parameters to maximize the likelihood function at every step. Ultimately, one ends up with one-dimensional slices of the parameter manifold along which the likelihood decays most slowly.

```@docs
ParameterProfiles(::DataModel,::Float64)
ProfileBox
```
