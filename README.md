# InformationGeometry.jl

*A Julia package for differential-geometric analyses of parameter inference problems.*

| **Documentation** | **Build Status** |
|:-----------------:|:----------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] | [![][appveyor-img]][appveyor-url] [![][codecov-img]][codecov-url] |


In essence, the functionality provided by **InformationGeometry.jl** revolves around analyzing the parameter space associated with mathematical models, given observed data.
In particular, it employs novel methods to quantify and visualize the *exact* uncertainties in the best fit parameter configuration.
That is, the confidence regions around the best fit parameters can be constructed without resorting to any approximations of their shape (e.g. as ellipsoids by assuming linear covariance between the parameters).
Moreover, the utilized schemes are highly efficient since they do *not* require one to sample large domains of the parameter space either on a grid or stochastically in order to find said confidence regions, which constitutes a significant improvement over previously established methods.

In addition, this package also allows for computations of

* likelihoods,
* confidence bands,
* Kullback-Leibler divergences,
* the Fisher metric, geodesics,
* Riemann and Ricci curvature tensors

and more. With its rich set of unique features and great performance, the toolkit of **InformationGeometry.jl** offers valuable insights into complicated modeling problems from various areas of science and engineering.

Resources detailing how to use this package can be found in the [**documentation**](https://RafaelArutjunjan.github.io/InformationGeometry.jl/dev). A discussion of the mathematical ideas underlying the methods employed by **InformationGeometry.jl** can be found in my [Master's Thesis](https://github.com/RafaelArutjunjan/Master-Thesis). A citeable publication discussing these methods is already under way.



[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://RafaelArutjunjan.github.io/InformationGeometry.jl/stable

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://RafaelArutjunjan.github.io/InformationGeometry.jl/dev

[appveyor-img]: https://ci.appveyor.com/api/projects/status/github/RafaelArutjunjan/InformationGeometry.jl?svg=true
[appveyor-url]: https://ci.appveyor.com/project/RafaelArutjunjan/InformationGeometry-jl

[codecov-img]: https://codecov.io/gh/RafaelArutjunjan/InformationGeometry.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/RafaelArutjunjan/InformationGeometry.jl
