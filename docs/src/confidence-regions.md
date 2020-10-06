
### Confidence Regions

```@docs
DataSet
DataModel
InformationGeometry.loglikelihood(::DataModel,::Vector{Float64})
FindMLE(::DataModel)
```
```@example 1
using InformationGeometry, Plots; gr() # hide
DS = DataSet([1,2,3.],[4,5,6.5],[0.5,0.45,0.6])
model(x,θ) = θ[1] .* x .+ θ[2]
DM = DataModel(DS,model)
MLE = FindMLE(DM)
```
Depending on how the parameters ``\theta`` enter into the model, the shapes of confidence regions associated with the model may be distorted. For the linearly parametrized model shown above, the ``1 \sigma`` and ``2 \sigma`` confidence regions form perfect hyperellipses as expected:
```@example 1
sols = GenerateMultipleIntervals(DM,1:2)
VisualizeSols(sols)
# plot(sols[1],vars=(1,2),label="1σ CR",title="Confidence Regions for linearly parametrized model", xlabel="θ[1]", ylabel="θ[2]") # hide
# plot!(sols[2],vars=(1,2),label="2σ CR") # hide
# scatter!([MLE[1]],[MLE[2]],marker=:c,label="MLE") # hide
# savefig("../assets/sols.svg"); nothing # hide
```
![](https://raw.githubusercontent.com/RafaelArutjunjan/InformationGeometry.jl/master/docs/assets/sols.svg)



For a non-linearly parametrized model, the confidence regions are found to be non-ellipsoidal:
```@example 1
model2(x,θ) = θ[1]^3 .* x .+ exp(θ[1] + θ[2])
DM2 = DataModel(DS,model2)
sols2 = GenerateMultipleIntervals(DM2,1:2)
VisualizeSols(sols2)
#plot(sols2[1],vars=(1,2),label="1σ CR",title="Confidence Regions for non-linearly parametrized model", xlabel="θ[1]", ylabel="θ[2]") # hide
#plot!(sols2[2],vars=(1,2),label="2σ CR") # hide
#MLE2 = FindMLE(DM2);  scatter!([MLE2[1]],[MLE2[2]],marker=:c,label="MLE") # hide
#savefig("../assets/sols2.svg"); nothing # hide
```
![](https://raw.githubusercontent.com/RafaelArutjunjan/InformationGeometry.jl/master/docs/assets/sols2.svg)

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

Further, studying the geodesics / autoparallels on a manifold can yield enlightening insights about its geometry.
```@docs
Score(::DataModel,::Vector{Float64})
AIC(::DataModel,::Vector{Float64})
GeodesicDistance(::DataModel,::Vector{Float64},::Vector{Float64})
```

**To be continued...**
