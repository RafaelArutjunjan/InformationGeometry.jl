
### Parallization

Especially for cases where every single evaluation of the likelihood is computationally expensive (e.g. because the model function is highly complex and / or has to be evaluated for a very large number of data points) a lot of performance can be gained by distributing the workload between multiple threads.

Early on in the development of this package, the design choice was made that computations of quantities such as the likelihood should be kept local to avoid unnecessary overhead for likelihoods which are cheap. However, computations of multiple trajectories on the confidence boundary can be evaluated in parallel.

A prerequisite for parallel computation is that every process has access to the necessary `DataModel` objects. For example, this can be achieved using the `@everywhere` macro from `Distributed.jl`. Note that in this case every step involved in the definition of the `DataModel`, its `DataSet` and model function must be performed on each worker simultaneously, e.g. by wrapping all loading and construction steps in an `@everywhere begin ... end` environment.

Alternatively, it is also possible to share data between processes using packages such as [ParallelDataTransfer.jl](https://github.com/ChrisRackauckas/ParallelDataTransfer.jl). Here, only the final `DataModel` needs to be sent to other workers instead of having to perform all intermediate steps (such as the maximum likelihood estimation involved in the `DataModel` construction) on each worker.


Both the functions `ConfidenceRegion()` and `ConfidenceRegions()` accept the optional keyword `parallel=true` to enable parallel computations of confidence boundaries. Other methods which also accept the keyword `parallel=true` include `PlotScalar()`, `ProfileLikelihood()` and `RadialGeodesics()`.

Example:
```julia
using Distributed;  addprocs(4)
@everywhere using InformationGeometry, ParallelDataTransfer
using Distributions, Random, BenchmarkTools

Random.seed!(123)
X = collect(1:300);     Y = 0.02*X.^2 - 5*X .+ 10 + rand(Normal(0,5),300)

DS = DataSet(X, Y, 5 .* ones(300) + 2rand(300))
DM = DataModel(DS, (x,θ)->sinh(θ[1])*x^2 + (θ[2]+θ[3])*x + (θ[2]-θ[3]))

sendto(workers(); DM=DM)

@btime ConfidenceRegion(DM, 1; parallel=true, tests=false)
@btime ConfidenceRegion(DM, 1; parallel=false, tests=false)
```
