# InformationGeometry.jl

*A Julia package for differential-geometric analyses of statistical problems.*

| **Documentation** | **Build Status** |
|:-----------------:|:----------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] | [![][travis-img]][travis-url] [![][appveyor-img]][appveyor-url] [![][codecov-img]][codecov-url] |

Upon closer inspection, one finds that the parameter space of a model (i.e. the set of all admissible parameter configurations) typically does not constitute a vector space as one might naively assume. Instead, the parameter space must be considered more generally as a (smooth) manifold.

Given a dataset and a prescription of a model function, which is assumed to accurately reflect the relationship present in the data, this package offers a set of basic tools to compute quantities of interest in applied information geometry and study the parameter space in detail. Said quantities of interest include

* likelihoods,
* confidence regions,
* Kullback-Leibler divergences,
* the Fisher metric,
* geodesics,
* Riemann and Ricci curvature tensors

and so on.

Most importantly, this package provides methods for efficient computations of the *exact* boundaries of confidence regions for statistical models. Given that it is common practice in experimental sciences to rely on linear approximations of the uncertainty / covariance associated with the best fit parameters of a model, the methods provided by **InformationGeometry.jl** constitute a significant improvement, particularly when working with models which depend non-linearly on their parameters.



Resources detailing how to use this package can be found in the [**documentation**](https://RafaelArutjunjan.github.io/InformationGeometry.jl/dev).

A discussion of the mathematical ideas underlying the methods employed by **InformationGeometry.jl** can be found in my [Master's Thesis](https://github.com/RafaelArutjunjan/Master-Thesis).


Future Plans for this package
-----------------------------
**InformationGeometry.jl** is a loose collection of code which I wrote especially to perform calculations in the context of my [Master's Thesis](https://github.com/RafaelArutjunjan/Master-Thesis). As such, it was not originally written with its publication as a module in mind but instead mostly optimized to fit my personal needs. Now that my thesis is finished, I plan on revising the internals of this package to improve performance and significantly extend its functionality. In addition, I will do my best to provide detailed documentation and examples of how to use this package over the coming weeks.


[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://RafaelArutjunjan.github.io/InformationGeometry.jl/stable

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://RafaelArutjunjan.github.io/InformationGeometry.jl/dev

[travis-img]: https://travis-ci.com/RafaelArutjunjan/InformationGeometry.jl.svg?branch=master
[travis-url]: https://travis-ci.com/RafaelArutjunjan/InformationGeometry.jl

[appveyor-img]: https://ci.appveyor.com/api/projects/status/github/RafaelArutjunjan/InformationGeometry.jl?svg=true
[appveyor-url]: https://ci.appveyor.com/project/RafaelArutjunjan/InformationGeometry-jl

[codecov-img]: https://codecov.io/gh/RafaelArutjunjan/InformationGeometry.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/RafaelArutjunjan/InformationGeometry.jl
