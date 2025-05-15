

### PEtab.jl

The [**PEtab.jl**](https://github.com/sebapersson/PEtab.jl) package allows for loading models defined in the [**PEtab**](https://petab.readthedocs.io/en/latest) standard for parameter estimation into Julia.
In addition to datasets of recorded measurements, the PEtab format allows for the specification of a mathematical model for describing said data (which are usually based on Ordinary Differential Equations and provided via [**SBML**](https://sbml.org/documents/specifications/) files), as well as the final parameter values resulting from the estimation process.
Most prominently, the [**PEtab**](https://petab.readthedocs.io/en/latest) standard has been used to publish modelling results in Systems Biology so far, see e.g. the [Benchmark Collection](https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab).

Currently, it is possible to convert a `PEtabODEProblem` from the [**PEtab.jl**](https://github.com/sebapersson/PEtab.jl) package into a `DataModel`, (or a `ConditionGrid` if it consists of more than one condition) via by applying the `DataModel` constructor. For instance, for the Böhm model with .yaml saved under `BöhmYamlPath`:
```julia
using InformationGeometry, PEtab, Plots
Böhm = PEtabODEProblem(PEtabModel(BöhmYamlPath); gradient_method=:ForwardEquations, hessian_method=:ForwardDiff)
DM = DataModel(Böhm; FixedError=true)
```
This will automatically extract a simplified representation of the dataset. 
If error models are used in the `PEtabODEProblem` to estimate the data uncertainties, they are currently dropped and the uncertainties are fixed to the values dictated by the error model at the best fit values of the error parameters. However, since the likelihood function and its gradient are directly copied from the given `PEtabODEProblem`, this does not affect optimisation (such as during profile likelihood computation or multistart optimisation), where changes in the given error parameters are properly accounted for.

Therefore, this mainly affects plots of the datasets from the `DM`, if further changes to the values of the error parameters have been made after the import of the `PEtabODEProblem` to a `DataModel` or `ConditionGrid`, as these will currently not be visible in said plots of the dataset.

Most other functionality of `InformationGeometry.jl` should be unaffected by this. In particular, it should be possible to compute `ParameterProfiles`, `MultistartFit`s and so on without issue.
```julia
plot(DM; Confnum=0)
R = MultistartFit(DM; N=5000)
P = ParameterProfiles(DM; N=20, meth=NelderMead())
```

!!! note
    For `PEtabODEProblem`s consisting of multiple conditions, only the gradient method `:ForwardEquations` is currently supported.  
    For the Hessian method, the three options `:ForwardDiff`, `:BlockForwardDiff` and `:GaussNewton` are supported.