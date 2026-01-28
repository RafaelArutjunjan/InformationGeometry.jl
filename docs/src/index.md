
# InformationGeometry

This is the documentation of [**InformationGeometry.jl**](https://github.com/RafaelArutjunjan/InformationGeometry.jl), a Julia package for differential-geometric analyses of parameter inference problems.

[![DOI](https://zenodo.org/badge/291016637.svg)](https://zenodo.org/badge/latestdoi/291016637)

| **Build Status** |
|:----------------:|
| [![Build Status](https://github.com/RafaelArutjunjan/InformationGeometry.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/RafaelArutjunjan/InformationGeometry.jl/actions/workflows/CI.yml) [![codecov](https://codecov.io/gh/RafaelArutjunjan/InformationGeometry.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/RafaelArutjunjan/InformationGeometry.jl) |


## Main Uses

* maximum likelihood estimation
* construction and visualization of exact confidence regions
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

## Citation

If [**InformationGeometry.jl**](https://github.com/RafaelArutjunjan/InformationGeometry.jl) was helpful in your own work, please consider citing [https://doi.org/10.48550/arXiv.2211.03421](https://doi.org/10.48550/arXiv.2211.03421) and [https://doi.org/10.5281/zenodo.5530660](https://doi.org/10.5281/zenodo.5530660).
