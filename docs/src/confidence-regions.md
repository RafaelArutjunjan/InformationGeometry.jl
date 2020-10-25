
### Confidence Regions

Once a `DataModel` object has been defined, it can subsequently be used to compute various quantities as follows:

```@docs
loglikelihood(::DataModel,::Vector{Float64})
MLE(::DataModel)
LogLikeMLE(::DataModel)
```

```@example 1
using InformationGeometry, Plots; gr() # hide
DS = DataSet([1,2,3.],[4,5,6.5],[0.5,0.45,0.6])
model(x::Real,θ::AbstractVector{<:Real}) = θ[1] * x + θ[2]
DM = DataModel(DS,model)
MLE(DM), LogLikeMLE(DM)
```

One of the primary goals of **InformationGeometry.jl** is to enable the user to investigate the relationships between different parameters in a model in detail by determining and visualizing the **exact** confidence regions associated with the best fit parameters. In this context, *exact* refers to the fact that no simplifying assumptions are made about the shape of the confidence regions.

Depending on how the parameters ``\theta`` enter into the model, the shapes of confidence regions associated with the model may be distorted. For the linearly parametrized model shown above, the ``1 \sigma`` and ``2 \sigma`` confidence regions form perfect ellipses around the maximum likelihood estimate as expected:
```@example 1
GenerateConfidenceRegion(DM,1.) # hide
sols = MultipleConfidenceRegions(DM,1:2;tol=1e-9)
VisualizeSols(sols)
#plot(sols[1],vars=(1,2),label="1σ CR",title="Confidence Regions for linearly parametrized model", xlabel="θ[1]", ylabel="θ[2]") # hide
#plot!(sols[2],vars=(1,2),label="2σ CR") # hide
#scatter!([MLE(DM)[1]],[MLE(DM)[2]],marker=:c,label="MLE") # hide
#savefig("../assets/sols.svg"); nothing # hide
```
![](https://raw.githubusercontent.com/RafaelArutjunjan/InformationGeometry.jl/master/docs/assets/sols.svg)


For a non-linearly parametrized model, the confidence regions are no longer ellipsoidal:
```@example 1
model2(x::Real,θ::AbstractVector{<:Real}) = θ[1]^3 * x + exp(θ[1] + θ[2])
DM2 = DataModel(DS,model2)
GenerateConfidenceRegion(DM2,1.) # hide
sols2 = MultipleConfidenceRegions(DM2,1:2;tol=1e-9)
VisualizeSols(sols2)
#plot(sols2[1],vars=(1,2),label="1σ CR",title="Confidence Regions for non-linearly parametrized model", xlabel="θ[1]", ylabel="θ[2]") # hide
#plot!(sols2[2],vars=(1,2),label="2σ CR") # hide
#scatter!([MLE(DM2)[1]],[MLE(DM2)[2]],marker=:c,label="MLE") # hide
#savefig("../assets/sols2.svg"); nothing # hide
```
![](https://raw.githubusercontent.com/RafaelArutjunjan/InformationGeometry.jl/master/docs/assets/sols2.svg)

Specifically in the case of two-dimensional parameter spaces as shown here, the problem of finding the exact boundaries of the confidence regions is turned into a system of ordinary differential equations and subsequently solved using the [**DifferentialEquations.jl**](https://github.com/SciML/DifferentialEquations.jl) suite. As a result, the boundaries of the confidence regions are obtained in the form of `ODESolution` objects, which come equipped with elaborate interpolation methods.


```@docs
MultipleConfidenceRegions(::DataModel,::Vector{Float64})
```

Since both finding and visualizing exact confidence regions for models depending on more than two parameters (i.e. ``\mathrm{dim} \, \mathcal{M} > 2``) is more challenging from a technical perspective, the above methods only work for ``\mathrm{dim} \, \mathcal{M} = 2`` at this point in time. However, methods which allow for visualizations of confidence regions in arbitrary three-dimensional subspaces of parameter manifolds of any dimension are close to being finished and will follow soon.


Various geometric quantities which are intrinsic to the parameter manifold ``\mathcal{M}`` can be computed as a result of the Fisher metric ``g`` (and subsequent choice of the Levi-Civita connection) such as the Riemann and Ricci tensors and the Ricci scalar ``R``.
```@docs
FisherMetric(::DataModel,::Vector{Float64})
GeometricDensity(::DataModel,::Vector{Float64})
ChristoffelSymbol(::DataModel,::Vector{Float64})
Riemann(::DataModel,::Vector{Float64})
Ricci(::DataModel,::Vector{Float64})
RicciScalar(::DataModel,::Vector{Float64})
```

Further, studying the geodesics associated with a metric manifold can yield valuable insights into its geometry.
```@docs
Score(::DataModel,::Vector{Float64})
AIC(::DataModel,::Vector{Float64})
AICc(::DataModel,::Vector{Float64})
BIC(::DataModel,::Vector{Float64})
GeodesicDistance(::DataModel,::Vector{Float64},::Vector{Float64})
```
