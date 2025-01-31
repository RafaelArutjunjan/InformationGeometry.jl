# InformationGeometry.jl

*A Julia package for differential-geometric analyses of parameter inference problems.*

| **Documentation** | **Build Status** | **DOI** |
|:-----------------:|:----------------:|:----------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] | [![][appveyor-img]][appveyor-url] [![][codecov-img]][codecov-url] | [![][doi-img]][doi-url] |


In essence, the functionality provided by **InformationGeometry.jl** revolves around analyzing the parameter space associated with mathematical models, given observed data.
In particular, it employs novel methods to quantify and visualize the *exact* uncertainties in the best fit parameter configuration.
That is, the confidence regions around the best fit parameters can be constructed without resorting to any approximations of their shape (e.g. as ellipsoids by assuming linear covariance between the parameters).
Moreover, the utilized schemes are highly efficient since they do *not* require one to sample large domains of the parameter space either on a grid or stochastically in order to find said confidence regions, which constitutes a significant improvement over previously established methods.

For example, given two different parametrizations of the same linear relationship between observed *x* and *y*-data, one finds the following confidence regions:

`y(x, θ) = θ[1] * x + θ[2]` | `y(x, θ) = θ[1]^3 * x + exp(θ[1] + θ[2])`
:------|------:
<img src="https://github.com/RafaelArutjunjan/InformationGeometry.jl/blob/master/docs/assets/sols.svg" width="410"/> | <img src="https://github.com/RafaelArutjunjan/InformationGeometry.jl/blob/master/docs/assets/sols2.svg" width="410"/>
<img src="https://github.com/RafaelArutjunjan/InformationGeometry.jl/blob/master/docs/assets/Profiles1.svg" width="410"/> | <img src="https://github.com/RafaelArutjunjan/InformationGeometry.jl/blob/master/docs/assets/Profiles2.svg" width="410"/>

In addition, this package also allows for computations of

* confidence bands around the prediction,
* profile likelihoods,
* multistart optimization with optimizers from the [**Optimization.jl**](https://github.com/SciML/Optimization.jl) ecosystem,
* the Fisher metric, geodesics,
* Riemann and Ricci curvature tensors

and more. With its unique features, the toolkit of **InformationGeometry.jl** offers valuable insights into complicated modeling problems from various areas of science and engineering.
Examples detailing how to use this package can be found in the [**documentation**](https://RafaelArutjunjan.github.io/InformationGeometry.jl/dev).




## Further reading
A preprint discussing the mathematical ideas underlying the methods employed by **InformationGeometry.jl** can be found in [**2211.03421**](https://arxiv.org/abs/2211.03421).

If **InformationGeometry.jl** was helpful in your own work, please consider citing [https://doi.org/10.48550/arXiv.2211.03421](https://doi.org/10.48550/arXiv.2211.03421) and [https://doi.org/10.5281/zenodo.5530660](https://doi.org/10.5281/zenodo.5530660).




[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://RafaelArutjunjan.github.io/InformationGeometry.jl/stable

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://RafaelArutjunjan.github.io/InformationGeometry.jl/dev

[appveyor-img]: https://ci.appveyor.com/api/projects/status/github/RafaelArutjunjan/InformationGeometry.jl?svg=true
[appveyor-url]: https://ci.appveyor.com/project/RafaelArutjunjan/InformationGeometry-jl

[codecov-img]: https://codecov.io/gh/RafaelArutjunjan/InformationGeometry.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/RafaelArutjunjan/InformationGeometry.jl

[doi-img]: https://zenodo.org/badge/291016637.svg
[doi-url]: https://zenodo.org/badge/latestdoi/291016637
