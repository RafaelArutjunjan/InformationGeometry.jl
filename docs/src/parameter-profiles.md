
### Profile Likelihoods

Whenever the parameter space of a non-linearly parametrized model is high-dimensional, computing exact confidence regions via integral curves becomes increasingly difficult and time intensive. Moreover, it becomes more likely that not all the parameters are sufficiently informed by the available data for the confidence regions of levels ``1 \sigma`` or ``2 \sigma`` to be bounded.
Therefore, a popular alternative for the individual uncertainty assessments of the model parameters ``\theta \in \mathcal{M}`` is given by the profile likelihood.

In essence, the profile likelihood simply constitutes a projection of the high-dimensional simultaneous confidence region onto the individual parameter directions. As a result, although there may be directions in which a given confidence region is _not_ bounded since the associated parameter component is insufficiently informed by the data, the remaining directions in which the confidence region _is_ bounded are unaffected.
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
where the second argument `2` determines the confidence level in units of ``\sigma`` to which the profile is computed and keyword `N` determines the number of points in the profile (approximately). 
The keyword `IsCost=false` automatically applies a rescaling to the vertical axis via the inverse ``\chi^2`` distribution with ``\mathrm{dim}\,\mathcal{M}`` degrees of freedom so that it already displays the confidence level up to which the parameter value on the horizontal axis is still compatible with the data. 
In contrast, the default `IsCost=true` does not apply this rescaling and provides the cost function value ``2\,\big(\ell(\theta_{MLE}) - \mathrm{PL}_i(\theta_i)\big)`` with ``\ell`` the log-likelihood and ``\mathrm{PL}_i(\theta_i)`` the profile likelihood of the ``i``-th parameter.
 
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

Once a profile `P3` has been computed, it can be plotted via `plot(P3)`. It is also possible to just plot any of the individual profiles via `plot(P3[i])` where the integer `i` denotes the profile of the ``i``-th parameter. 

```@docs
ParameterProfiles
```

Instead of computing the profile likelihood by explicitly reoptimizing the nuisance parameters at every step along the profile, it is possible to instead compute the path of the profile likelihood of a parameter via an ODE in the parameter space based on the Hessian of the log-likelihood. While the Hessian can be extremely expensive to compute for models with many parameters due to its quadratic size, the trade-off of not having to perform optimization at every step is often still worth it, particularly for small to moderately sized models.
```@example Profiles
P4 = IntegrationParameterProfiles(DM2, 3; reltol=1e-3, N=101, γ=nothing, plot=true)
plot(P4, false) #hide
```

!!! note
    For `IntegrationParameterProfiles`, the computational effort and accuracy of the result is almost purely controlled by the integration tolerances.
    The parameter `N` only specifies the number of points at which the parameter trajectory is subsequently interpolated to compute the log-likelihood, which is however much faster than the Hessian evaluations during the integration. `N=nothing` does not interpolate the trajectory and only evaluates the log-likelihood at the steps taken by the integrator.

In the original derivation of the ODE for this profile parameter path by [Chen and Jennrich](https://doi.org/10.1198/106186002493), an extra stabilization term controlled by a factor ``\\gamma`` was added, to avoid the trajectory moving off the constraint submanifold satisfying nuisance parameter optimality when using inaccurate Hessian approximations.
While non-zero values of ``\\gamma`` essentially correspond to adding a Newton-like contribution towards the constraint submanifold of nuisance parameter optimality to the direction of the trajectory at every step, this results in an asymptotically *biased* trajectory.
In other words, the computed trajectory strictly speaking no longer tends towards the *true* trajectory for non-zero ``\\gamma``, even if the integration tolerances are chosen arbitrarily small.
Therefore, whenever a model allows for computing the Hessian via automatic differentiation, meaning that the Hessian is essentially accurate to machine precision, the ``\\gamma`` stabilization term should not be used.

```@docs
IntegrationParameterProfiles
```


### Extracting Confidence Intervals

Finally, the precise values where the profile likelihood of a given parameter intersects the threshold corresponding to some confidence level ``1-\alpha = q \in (0,1)`` can be extracted from a `ParameterProfiles` object via
```@example Profiles
ProfileBox(P3, 1)
```
which computes the ``1\sigma`` confidence intervals as a `HyperCube` where the confidence level is again provided in units of ``\sigma``.
The desired confidence level can also be provided in ``\%`` via `InvConfVol`. For instance, to compute precisely the intervals associated with the ``95\% \approx 1.959\sigma`` thresholds:
```@example Profiles
Tuple(ProfileBox(P3, InvConfVol(0.95)))
```
and the confidence intervals can be extracted from the returned `HyperCube` by indexing into the result or applying `Tuple`.
When the confidence interval of associated with a given level is not bounded to one or both sides, `-Inf` or `+Inf` respectively is returned. For instance, the half-open ``3\sigma`` interval for ``\theta_2``:
```@example Profiles
ProfileBox(P3[2], 3)[1]
```

The final confidence level up to which a parameter was identified by the available data, i.e. where the confidence interval is still bounded in both directions, can be computed more precisely from a given profile by
```@example Profiles
PracticallyIdentifiable(P3[2])
```
which can either be computed for all profiles together or a single profile of interest.

```@docs
ProfileBox(::ParameterProfiles)
PracticallyIdentifiable(::ParameterProfiles)
```

### Investigating Profile Paths

The paths traced out by the individual profiles in the parameter space encode useful information about the mutual relationships between different parameters.
For instance, if during the profile of a given parameter ``a``, the nuisance parameter ``b`` remains at a constant value throughout, this indicates that the effects of ``a`` are independent of the mechanism encoded by ``b``.
In this sense, the influences of the two respective mechanisms on the final model predictions can be considered to be orthogonal.
Conversely, if the nuisance parameter `b` changes at a somewhat linear or even superlinear rate along the profile of ``a``, this means that the effects of the two mechanisms are strongly linked in the sense that changes in the value of the nuisance parameter ``b`` are able to compensate at least partially for changes in the parameter ``a``.
Moreover, the directionality of the changes in ``b`` reveals whether the mutual effects of the respective mechanisms are synergistic or antagonistic.
This kind of nuanced information is crucial in the process of model reduction.


For example, in ODE-based mechanistic models describing biological systems, this can be exploited to perform systematic model reduction in a straightforward, iterative process, where practically non-identifiable parameters are successively removed, by fixing them to their limiting values of zero or infinity, until all remaining parameters of the model are fully identified by the given data.
A paper outlining this systematic reduction strategy in detail is given by [Driving the model to its limits](https://ideas.repec.org/a/plo/pone00/0162366.html).

The saved paths of a `ParameterProfiles` object `P` can be accessed via `InformationGeometry.Trajectories(P)` for computations.
They can be plotted via several methods, such as `PlotProfilePaths`:
```@example Profiles
PlotProfilePaths(P3; RelChange=false, TrafoPath=identity, idxs=1:length(P3))
```
where the `RelChange` kwarg can be used to specify whether the relative change `p_i / p_mle` or the difference `p_i - p_mle` is plotted.
Also, the `TrafoPath` allows for specifying a transformation which is broadcasted over the results to account for non-linear distortions on the parameter space, making relevant features more easily visible, for instance `TrafoPath=BiLog`.

In the above example, one can see from the trajectories of the profile for ``\theta_2`` that the first parameter ``\theta_1`` must decrease if the second parameter ``\theta_2`` increases in order to compensate, such that the model still describes the data as well as possible.

It is also possible to plot the finite differences of the profile paths (i.e. their discrete first derivative) via `PlotProfilePathDiffs` and `PlotProfilePathNormDiffs`, which can make it easier to visually identify discontinuous jumps in the parameter paths.

