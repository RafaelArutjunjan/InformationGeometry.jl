
### Advanced Datasets


The following table illustrates the capabilities of the various data types implemented by [**InformationGeometry.jl**](https://github.com/RafaelArutjunjan/InformationGeometry.jl):

| Container                    | allows non-Gaussian `y`-uncertainty | allows `x`-uncertainty | allows mixed `x`-`y` uncertainty | allows missing values |
|:----------------------------:|:-----------------------------------:|:----------------------:|:--------------------------------:|:---------------------:|
[`DataSet`](@ref)              |                 ✖                   |           ✖            |                ✖                 |          ✖            |
[`DataSetExact`](@ref)         |                 ✔                   |           ✔            |                ✖                 |          ✖            |
[`GeneralizedDataSet`](@ref)   |                 ✔                   |           ✔            |                ✔                 |          ✖            |
[`CompositeDataSet`](@ref)     |                 ✖                   |           ✔            |                ✖                 |          ✔            |


```@docs
DataSetExact
CompositeDataSet
GeneralizedDataSet
```
