
### ODE based models

Often, an explicit analytical expression for a given mathematical model is not known. Instead, the model might be defined implicitly, e.g. as the solution to a system of ordinary differential equations. Especially in fields such as systems biology, modeling in terms of (various kinds of) differential equations appears to be the norm.

As a toy example, we will consider the well-known "SIR model" in the following, which groups a population into susceptible, infected and recovered subpopulations and assumes mass action kinetics with constant transmission and recovery rates to describe the growths and decays of the respective populations.

While the [**DifferentialEquations.jl**](https://github.com/SciML/DifferentialEquations.jl) ecosystem offers many different ways of specifying such systems, we will use [**ModelingToolkit.jl**](https://github.com/SciML/ModelingToolkit.jl) since it is particularly convenient in this case.
```@example 2
using InformationGeometry, ModelingToolkit
@parameters t β γ
@variables S(t) I(t) R(t)
Dt = Differential(t)

SIReqs = [ Dt(S) ~ -β * I * S,
        Dt(I) ~ +β * I * S - γ * I,
        Dt(R) ~ +γ * I]

SIRstates = [S, I, R];    SIRparams = [β, γ]
SIRsys = ODESystem(SIReqs, t, SIRstates, SIRparams)
nothing # hide
```
Here, the parameter `β` denotes the transmission rate of the disease and `γ` is the recovery rate. Note that in the symbolic scheme of [**ModelingToolkit.jl**](https://github.com/SciML/ModelingToolkit.jl), the equal sign is represented via `~`.

An infection dataset which is well-known in the literature is taken from an influenza outbreak at a English boarding school in .... Its numerical values can be found e.g. in [table 1 of this paper](https://www.researchgate.net/publication/336701551_On_parameter_estimation_approaches_for_predicting_disease_transmission_through_optimization_deep_learning_and_statistical_inference_methods). As no uncertainties associated with the number of infections is given, we will assume the ``1\sigma`` uncertainties to be ``\pm 5`` as a reasonable value. Further, it is known that the total number of students at said boarding school was ``763`` and we will therefore assume the initial conditions to be
```@setup 2
SIRinitial = [762, 1, 0.]
```
```julia
SIRinitial = [762, 1, 0.]
```
for the respective susceptible, infected and recovered subpopulations on day zero. Next, the `DataSet` object is constructed as:
```@example 2
days = collect(1:14)
infected = [3, 8, 28, 75, 221, 291, 255, 235, 190, 126, 70, 28, 12, 5]
SIRDS = DataSet(days, infected, 5ones(14))
SIRDS = InformationGeometry.InformNames(DataSet(days, infected, 5ones(14)), ["Days"], ["Infected"]) # hide
nothing # hide
```

Finally, the `DataModel` associated with the SIR model and the given data is constructed by
```@example 2
SIRobservables = [2]
SIRDM = DataModel(SIRDS, SIRsys, SIRinitial, SIRobservables, [0.001, 0.1]; tol=1e-7)
```
where `SIRobservables` denotes the components of the `ODESystem` that have actually been observed in the given dataset (i.e. the second component which are the infected in this case). The optional vector `[0.001, 0.1]` is our initial guess for the parameters `[β, γ]` for the maximum likelihood estimation and the keyword `tol` specifies the desired accuracy of the ODE solver for all model predictions.

It is now possible to compute properties of this `DataModel` such as confidence regions, confidence bands, geodesics, profile likelihoods, curvature tensors and so on as with any other model.
```@example 2
sols = ConfidenceRegions(SIRDM, 1:2)
VisualizeSols(SIRDM, sols)
```
```julia
FittedPlot(SIRDM)
ConfidenceBands(SIRDM, sols[2])
```
```@setup 2
B = ConfidenceBands(SIRDM, sols[2]) # hide
using Plots # hide
FittedPlot(SIRDM) # hide
plot!(B[:,1], B[:,3], label="2σ Conf. Band", color=:orange) # hide
plot!(B[:,1], B[:,2], label="", color=:orange) # hide
# savefig("../assets/SIRBands.svg"); nothing # hide
```
![](https://raw.githubusercontent.com/RafaelArutjunjan/InformationGeometry.jl/master/docs/assets/SIRBands.svg)

While it visually appears as though the confidence regions are perfectly ellipsoidal and the model would therefore be linearly dependent on its parameters `β` and `γ`, this is of course not the case. The non-linearity with respect to the parameters becomes much more apparent further away from the MLE, as one can confirm e.g. via radial geodesics emanating from the MLE or the profile likelihood.
