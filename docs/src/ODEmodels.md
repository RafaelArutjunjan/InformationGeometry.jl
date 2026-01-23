
### ODE-based models

Often, an explicit analytical expression for a given mathematical model is not known. Instead, the model might be defined implicitly, e.g. as the solution to a system of ordinary differential equations. Especially in fields such as systems biology, modeling in terms of (various kinds of) differential equations is standard practice.

As a toy example, we will consider the well-known "SIR model" in the following, which groups a population into susceptible, infected and recovered subpopulations and describes the growths and decays of the respective populations via simple rate laws.

While the [**DifferentialEquations.jl**](https://github.com/SciML/DifferentialEquations.jl) ecosystem offers many different ways of specifying such systems, we will use the syntax introduced by [**ModelingToolkit.jl**](https://github.com/SciML/ModelingToolkit.jl) since it is particularly convenient in this case.
```@example ODE
using InformationGeometry, ModelingToolkit, Plots
using ModelingToolkit: t_nounits as t, D_nounits as D
@parameters β γ
@variables S(t) I(t) R(t)

SIReqs = [ D(S) ~ -β * I * S,
        D(I) ~ +β * I * S - γ * I,
        D(R) ~ +γ * I]

SIRstates = [S, I, R];    SIRparams = [β, γ]
@named SIRsys = System(SIReqs, t, SIRstates, SIRparams)
```
Here, the parameter `β` denotes the transmission rate of the disease and `γ` is the recovery rate. Note that in the symbolic scheme of [**ModelingToolkit.jl**](https://github.com/SciML/ModelingToolkit.jl), the equal sign is represented via `~`.

!!! warning
    Using the `@mtkcompile` macro in the `ModelingToolkit.System` construction typically causes the states and equations to be reordered in the final model! If you want to ensure that the original order is retained, use `@named` or supply the `ODEFunction` directly instead of a `ModelingToolkit.System`.

An infection dataset which is well-known in the literature is taken from an influenza outbreak at a English boarding school in 1978. Its numerical values can be found e.g. in [table 1 of this paper](https://www.researchgate.net/publication/336701551_On_parameter_estimation_approaches_for_predicting_disease_transmission_through_optimization_deep_learning_and_statistical_inference_methods). As no uncertainties associated with the number of infections is given, we will assume the ``1\sigma`` uncertainties to be ``\pm 15`` as a reasonable value. Further, it is known that the total number of students at said boarding school was ``763`` and we will therefore assume the initial conditions to be
```@setup ODE
SIRinitial = [762, 1, 0.]
```
```julia
SIRinitial = [762, 1, 0.]
```
for the respective susceptible, infected and recovered subpopulations on day zero. Next, the `DataSet` object is constructed as:
```@example ODE
days = collect(1:14)
infected = [3, 8, 28, 75, 221, 291, 255, 235, 190, 126, 70, 28, 12, 5]
SIRDS = DataSet(days, infected, 15ones(14); xnames=["Days"], ynames=["Infected"])
```

Finally, the `DataModel` associated with the SIR model and the given data is constructed by
```@example ODE
SIRobservables = [2]
SIRDM = DataModel(SIRDS, SIRsys, SIRinitial, SIRobservables, [0.002, 0.5]; tol=1e-11)
```
where `SIRobservables` denotes the components of the `ODESystem` that have actually been observed in the given dataset (i.e. the second component which are the infected in this case). The optional vector `[0.001, 0.1]` is our initial guess for the parameters `[β, γ]` for the maximum likelihood estimation and the keyword `tol` specifies the desired accuracy of the ODE solver for all model predictions.

!!! tip
    Instead of specifying the observable components of an ODE system as an array, it is also possible to provide an arbitrary observation function with argument signature `f(u)`, `f(u,t)` or `f(u,t,θ)`.
    Similarly, (parts of) the initial conditions for the ODE system can be included as parameters of the problem and estimated from data by providing a splitter function of the form `θ -> (u0, p)`. The first entry of the returned tuple will be used as the initial condition for the ODE system and the second argument enters into the `ODEFunction` itself.

    In this particular example, one might include the initial number of infections as a dynamical parameter via the splitter function `θ -> ([763.0 - θ[1], θ[1], 0.0], θ[2:3])`.


It is now possible to compute properties of this `DataModel` such as confidence regions, confidence bands, geodesics, profile likelihoods, curvature tensors and so on as with any other model.
```@example ODE
sols = ConfidenceRegions(SIRDM, 1:2)
VisualizeSols(SIRDM, sols)
```

While it visually appears as though the confidence regions are perfectly ellipsoidal and the model would therefore be linearly dependent on its parameters `β` and `γ`, this is of course not the case. The non-linearity with respect to the parameters becomes much more apparent further away from the MLE, as one can confirm e.g. via radial geodesics emanating from the MLE or the profile likelihood.

Finally the exact confidence bands associated with the computed confidence regions are obtained as:
```@example ODE
plot(SIRDM; Confnum=0)
ConfidenceBands(SIRDM, sols[2])
plot(SIRDM; Confnum=0) # hide
M = ConfidenceBands(SIRDM, sols[2]; plot=false) # hide
InformationGeometry.PlotConfidenceBands(SIRDM, M; label=["$(round(InformationGeometry.GetConfnum(SIRDM,sols[2])))σ Conf. Band" nothing]) # hide
```