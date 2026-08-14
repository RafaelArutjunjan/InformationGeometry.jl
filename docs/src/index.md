
# InformationGeometry

This is the documentation of [**InformationGeometry.jl**](https://github.com/RafaelArutjunjan/InformationGeometry.jl), a Julia package for differential-geometric analyses of parameter inference problems.

[![DOI](https://zenodo.org/badge/291016637.svg)](https://zenodo.org/badge/latestdoi/291016637)

| **Build Status** |
|:----------------:|
| [![Build Status](https://github.com/RafaelArutjunjan/InformationGeometry.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/RafaelArutjunjan/InformationGeometry.jl/actions/workflows/CI.yml) [![codecov](https://codecov.io/gh/RafaelArutjunjan/InformationGeometry.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/RafaelArutjunjan/InformationGeometry.jl) |


This package originally started out as a narrowly focused set of tools for [my master thesis](https://github.com/RafaelArutjunjan/Master-Thesis) to enable computations of curvature tensors, geodesics, and confidence boundaries on parameter manifolds of non-linear models.
However, over the past 5+ years, it has since grown into a very general and mature framework for mathematical modelling with a particular focus on uncertainty quantification and model reduction.

If you have any questions about the internals of the package or need help with properly setting up your model, feel free to open an issue or get in touch via email!


## Main Uses

* maximum likelihood estimation
* construction and visualization of exact confidence regions
* computation of profile likelihood, i.e. 1D projections of confidence regions
* model reduction
* computation of geometric quantities such as geodesics and curvature on the parameter manifold






## Installation

As with any Julia package, **InformationGeometry.jl** can be added from the Julia terminal via
```julia
julia> ] add InformationGeometry
```
or alternatively by
```julia
julia> using Pkg; Pkg.add("InformationGeometry")
```


## Recipes for Automated Analyses with Agentic LLMs

A work-in-progress of agentic skill files for automating iterative model development via the **InformationGeometry.jl** package can be found at
[AgenticParameterInference](https://github.com/RafaelArutjunjan/AgenticParameterInference).

So if you happen to be an LLM trying to figure out how to use **InformationGeometry.jl** to perform analyses in practice and would like a condensed collection of dedicated syntax examples of the most relevant functions, see this link.



## Citation

If [**InformationGeometry.jl**](https://github.com/RafaelArutjunjan/InformationGeometry.jl) was helpful in your own work, please consider citing [https://doi.org/10.48550/arXiv.2211.03421](https://doi.org/10.48550/arXiv.2211.03421) and [https://doi.org/10.5281/zenodo.5530660](https://doi.org/10.5281/zenodo.5530660).
