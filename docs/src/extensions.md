


### ProfileLikelihood.jl


Although profile likelihood functionality is also provided by [**InformationGeometry.jl**](https://github.com/RafaelArutjunjan/InformationGeometry.jl), constructors for the types used by [**ProfileLikelihood.jl**](https://github.com/DanielVandH/ProfileLikelihood.jl) are also provided via an extension.
This allows for straightforward comparisons of results between the methods provided by both packages. In particular, this also allows for computing two-dimensional profiles with `bivariate_profile`, where parameters are fixed in pairs at various values and the remaining parameters are re-optimized, as this functionality is not implemented by [**InformationGeometry.jl**](https://github.com/RafaelArutjunjan/InformationGeometry.jl) (yet?).

For computing the `1σ` profiles, two possible methods are given by:
```julia
using InformationGeometry, ProfileLikelihood, Optimization, OptimizationOptimJL
DM = DataModel(DataSet([1,2,3,4], [4,5,6.5,9], [0.5,0.45,0.6,1]), (x::Real, θ::AbstractVector{<:Real}) -> θ[1]^3 * x + exp(θ[1] + θ[2]))

prob = LikelihoodProblem(DM)
sol = mle(prob, Optim.LBFGS())
P1 = ProfileLikelihood.profile(prob, sol, 1:pdim(DM); alg=Optim.LBFGS(), parallel=true, conf_level=ConfVol(1),
    threshold=-0.5InformationGeometry.InvChisqCDF(InformationGeometry.DOF(DM), ConfVol(1)), resolution=51)

## Alternatively:
P2 = ProfileLikelihood.profile(DM, 1; alg=Optim.LBFGS(), parallel=true, resolution=51)
```
Note that the [`ProfileLikelihood.profile`](@ref) function is sensitive to the boundaries of the provided parameter domains. A `HyperCube` may be passed via the `Domain` kwarg, 


For more examples of how to use the [**ProfileLikelihood.jl**](https://github.com/DanielVandH/ProfileLikelihood.jl) package, see [the documentation](https://danielvandh.github.io/ProfileLikelihood.jl/dev).