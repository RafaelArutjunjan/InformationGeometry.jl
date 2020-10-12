
### Providing Datasets

Typically, one of the most difficult parts of any data science problem is to bring the data into a form which lends itself to the subsequent analysis.
Thus, this section aims to describe the containers used by **InformationGeometry.jl** to store data and models in detail.

The data itself is stored using the `DataSet` container.
```@docs
DataSet
```
Depending on the dimensionality of the dataset, that is, the number of components of the respective x-values and y-values, there are multiple ways the `DataSet` can be constructed.

To complete the specification of the inference problem, a model function which takes an x-value and a parameter configuration ``\theta`` must be added.

```@docs
DataModel
```

**To be continued...**
