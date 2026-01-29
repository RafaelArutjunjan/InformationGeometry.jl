
### List of useful methods


The following lists docstrings for various important functions.




Once a `DataModel` object has been defined, it can subsequently be used to compute various quantities as follows:

```@docs
loglikelihood(::DataModel,::Vector{Float64})
MLE(::DataModel)
MLEuncert(::DataModel)
LogLikeMLE(::DataModel)
```

Properties related to structural identifiability can be computed via:
```@docs
IsStructurallyIdentifiable
StructurallyIdentifiable
PracticallyIdentifiable
IsLinearParameter
```


Various geometric quantities which are intrinsic to the parameter manifold ``\mathcal{M}`` can be computed as a result of the Fisher metric ``g`` (and subsequent choice of the Levi-Civita connection) such as the Riemann and Ricci tensors and the Ricci scalar ``R``.
```@docs
Score(::DataModel,::Vector{Float64},::Nothing)
FisherMetric(::DataModel,::Vector{Float64})
GeometricDensity(::DataModel,::Vector{Float64})
ChristoffelSymbol(::Function,::Vector{Float64})
Riemann(::Function,::Vector{Float64})
Ricci(::Function,::Vector{Float64})
RicciScalar(::Function,::Vector{Float64})
GeodesicDistance(::DataModel,::Vector{Float64},::Vector{Float64})
```

Different model comparison and model selection criteria are available via:
```@docs
AIC(::DataModel,::Vector{Float64})
AICc(::DataModel,::Vector{Float64})
BIC(::DataModel,::Vector{Float64})
CAIC(::DataModel,::Vector{Float64})
CAICF(::DataModel,::Vector{Float64})
ModelComparison
```
