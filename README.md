# InformationGeometry.jl

*A Julia package for differential-geometric analyses of parameter inference problems.*

| **Documentation** | **Build Status** |
|:-----------------:|:----------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] | [![][appveyor-img]][appveyor-url] [![][codecov-img]][codecov-url] |


In essence, the functionality provided by **InformationGeometry.jl** revolves around analyzing the parameter space associated with mathematical models, given observed data.
In particular, it employs novel methods to quantify and visualize the *exact* uncertainties in the best fit parameter configuration.
That is, the confidence regions around the best fit parameters can be constructed without resorting to any approximations of their shape (e.g. as ellipsoids by assuming linear covariance between the parameters).
Moreover, the utilized schemes are highly efficient since they do *not* require one to sample large domains of the parameter space either on a grid or stochastically in order to find said confidence regions, which constitutes a significant improvement over previously established methods.

For example, given two different ways to parametrize the same linear relationship between recorded *x* and *y*-values, one obtains the following confidence regions:

![equation](https://latex.codecogs.com/svg.latex?y_%5Ctext%7Blinear%7D%28x%3B%5Ctheta%29%20%3D%20%5Ctheta_1%20%5Ccdot%20x%20&plus;%20%5Ctheta_2) | ![equation](https://latex.codecogs.com/svg.latex?y_%5Ctext%7Bnon-linear%7D%28x%3B%5Ctheta%29%20%3D%20%7B%5Ctheta_1%7D%5E%7B%5C%213%7D%20%5Ccdot%20x%20&plus;%20%5Cmathrm%7Bexp%7D%28%5Ctheta_1%20&plus;%20%5Ctheta_2%29)
:------|------:
<img src="https://github.com/RafaelArutjunjan/InformationGeometry.jl/blob/master/docs/assets/sols.svg" width="400"/> | <img src="https://github.com/RafaelArutjunjan/InformationGeometry.jl/blob/master/docs/assets/sols2.svg" width="400"/>

In addition, this package also allows for computations of

* likelihoods,
* confidence bands,
* Kullback-Leibler divergences,
* the Fisher metric, geodesics,
* Riemann and Ricci curvature tensors

and more. With its rich set of unique features and great performance, the toolkit of **InformationGeometry.jl** offers valuable insights into complicated modeling problems from various areas of science and engineering.

Examples and other resources detailing how to use this package can be found in the [**documentation**](https://RafaelArutjunjan.github.io/InformationGeometry.jl/dev). A discussion of the mathematical ideas underlying the methods employed by **InformationGeometry.jl** can be found in my [Master's Thesis](https://github.com/RafaelArutjunjan/Master-Thesis/blob/master/Master's%20Thesis%20Rafael%20Arutjunjan%20-%20Corrected.pdf). A citable publication discussing these methods is already under way.




[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://RafaelArutjunjan.github.io/InformationGeometry.jl/stable

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://RafaelArutjunjan.github.io/InformationGeometry.jl/dev

[appveyor-img]: https://ci.appveyor.com/api/projects/status/github/RafaelArutjunjan/InformationGeometry.jl?svg=true
[appveyor-url]: https://ci.appveyor.com/project/RafaelArutjunjan/InformationGeometry-jl

[codecov-img]: https://codecov.io/gh/RafaelArutjunjan/InformationGeometry.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/RafaelArutjunjan/InformationGeometry.jl
