
### Profile Likelihoods

Whenever the parameter space of a non-linearly parametrized model is high-dimensional, computing exact confidence regions via integral curves becomes increasingly difficult and time intensive. Moreover, it becomes more likely that not all the parameters are sufficiently informed by the available data for the confidence regions of levels ``1 \sigma`` or ``2 \sigma`` to be bounded.
Therefore, a popular alternative for the individual uncertainty assessments of the model parameters ``\theta \in \mathcal{M}`` is given by the profile likelihood.

In essence, the profile likelihood simply constitutes a projection of the high-dimensional simultaneous confidence region onto the individual parameter directions. As a result, although there may be directions in which the confidence region is _not_ bounded since the associated parameter component is insufficiently informed by the data, the remaining directions in which the confidence region _is_ bounded are unaffected.
Thus, the one-dimensional profile likelihoods provide a convenient and exact summary of the individual uncertainties associated with the model parameters.

For an in-depth discussion of the theory underlying the profile likelihood, see e.g. [this article](https://febs.onlinelibrary.wiley.com/doi/10.1111/febs.12276).


Returning again to the example from the previous [section on Confidence Regions](https://rafaelarutjunjan.github.io/InformationGeometry.jl/stable/confidence-regions), we define:
```@example Profiles
using InformationGeometry, Plots
DS = DataSet([1,2,3,4], [4,5,6.5,9], [0.5,0.45,0.6,1])
model(x::Real, θ::AbstractVector{<:Real}) = θ[1] * x + θ[2]
DM = DataModel(DS, model)
model2(x::Real, θ::AbstractVector{<:Real}) = θ[1]^3 * x + exp(θ[1] + θ[2])
DM2 = DataModel(DS, model2)
```

Instead of computing the associated confidence regions as before, we can compute the profile likelihoods via
```@example Profiles
P1 = ParameterProfiles(DM, 2; N=100, plot=true, IsCost=false, adaptive=true, SaveTrajectories=true)
plot(P1, false) #hide
```
where the second argument `2` determines the confidence level in units of ``\sigma`` to which the profile is computed and `N` determines the number of points in the profile. The keyword `IsCost=false` automatically applies a rescaling to the vertical axis so that it already displays the confidence level up to which the parameter value on the horizontal axis is still compatible with the data. In contrast, the default `IsCost=true` does not apply this rescaling and provides the cost function value ``2\,\big(\ell(\theta_{MLE}) - \ell(\theta)\big)`` with ``\ell`` the log-likelihood.
 
Since the model function of `DM` is linear with respect to all its parameters all its profiles are not only symmetric around the MLE, but also bounded up to arbitrary confidence levels. In comparison, the non-linearly parametrized model map from `DM2` results in asymmetric profiles:
```@example Profiles
P2 = ParameterProfiles(DM2, 2; N=100, plot=true, IsCost=false, adaptive=true, SaveTrajectories=true)
plot(P2, false) #hide
```
Moreover, when computing the profile likelihood for even higher confidence levels of up to ``3\sigma``, we finally see that the profile of the second parameter ``\theta_2`` does not reach above the ``3\sigma`` threshold, and is "open" to one side:
```@example Profiles
P3 = ParameterProfiles(DM2, 3; N=100, plot=true, IsCost=false, adaptive=true, SaveTrajectories=true)
plot(P3, false) #hide
```
This means that arbitrarily small values of ``\theta_2`` are still compatible with the data up to a confidence of ``3\sigma \approx 99.73\%``.


Once a profile `P3` has been computed, it can be plotted via `plot(P3)`. It is also possible to just plot any of the individual profiles via `plot(P3[i])` where the integer `i` denotes the profile of the ``i``-th parameter. It is also possible to investigate the optimization paths of individual profile trajectories via `PlotRelativeParameterTrajectories(P3[i])` or simply
```@example Profiles
plot(P3[2], true; RelChange=true, BiLog=false)
```

```@docs
ParameterProfiles
```