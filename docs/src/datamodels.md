
### Providing Datasets

Typically, one of the most difficult parts of any data science problem is to bring the data into a form which lends itself to the subsequent analysis.
This section aims to describe the containers used by **InformationGeometry.jl** to store datasets and models in detail.

The data itself is stored using the `DataSet` container.
```@docs
DataSet
```

To complete the specification of an inference problem, a model function which is assumed to be able to capture the relationship which is inherent in the data must be added.

```@docs
DataModel
```

"Simple" `DataSet`s and `DataModel`s can be visualized directly via `plot(DM)` using pre-written recipes for the **Plots.jl** package.
