
### Advanced Datasets


Depending on the nature of the dataset that is analyzed, there are multiple data types implemented by [**InformationGeometry.jl**](https://github.com/RafaelArutjunjan/InformationGeometry.jl) to store them in.
Mainly, these data types provide a trade-off in speed / simplicity versus flexibility / generality as illustrated by the following table:


| Container                      | allows non-Gaussian `y`-uncertainty | allows missing values | allows `x`-uncertainty | allows mixed `x`-`y` uncertainty | allows `y`-uncertainty estimation | allows `x`-uncertainty estimation |
|:------------------------------:|:-----------------------------------:|:---------------------:|:----------------------:|:--------------------------------:|:---------------------------------:|:---------------------------------:|
[`DataSet`](@ref)                |                 ❌                  |          ❌           |           ❌           |                ❌                |          ❌                       |          ❌                       |
[`DataSetExact`](@ref)           |                 ✅                  |          ❌           |           ✅           |                ❌                |          ❌                       |          ❌                       |
[`CompositeDataSet`](@ref)       |                 ✅                  |          ✅           |           ❌           |                ❌                |          ❌                       |          ❌                       |
[`GeneralizedDataSet`](@ref)     |                 ✅                  |          ❌           |           ✅           |                ✅                |          ❌                       |          ❌                       |
[`DataSetUncertain`](@ref)       |                 ❌                  |          ✅           |           ❌           |                ❌                |          ✅                       |          ❌                       |
[`UnknownVarianceDataSet`](@ref) |                 ❌                  |          ❌           |           ❌           |                ❌                |          ✅                       |          ✅                       |


```@docs
DataSetExact
CompositeDataSet
GeneralizedDataSet
DataSetUncertain
```
