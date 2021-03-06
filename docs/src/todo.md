

## Contributing

* If you encounter a bug, feel free to file an issue detailing the problem in contrast to the behaviour you were expecting. Please provide a minimal working example and make sure to specify the particular version of **InformationGeometry.jl** that was used.
* While pull requests are very much welcome, please try to provide detailed docstrings for all non-trivial methods.



### TODO

* Allow for non-normal uncertainties in measurements e.g. by interpolating and deriving the Kullback-Leibler divergence over a domain
* Parallelism: Improve support for parallel computations of geodesics, curvature tensors and so on
* Employ importance sampling for Monte Carlo computations
* Improve visualization capabilities for high-dimensional models
* Standardize the user-facing keyword arguments
* Provide performance benchmarks for **InformationGeometry.jl**
* Use **IntervalArithmetic.jl** and **IntervalOptimisation.jl** for rigorous guarantees on inference results?
