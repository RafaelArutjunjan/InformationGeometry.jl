
### Exporting

For added convenience, [**InformationGeometry.jl**](https://github.com/RafaelArutjunjan/InformationGeometry.jl) already provides several methods, which can be used to export results like confidence regions or geodesics.
```@docs
SaveConfidence
```
In particular, choosing the keyword `adaptive=true` samples the `ODESolution` objects roughly proportional to their curvature (instead of equidistant), which means that more samples are provided from tight bends than from segments that are straight, leading to more faithful representations of the confidence boundary when plotting.

```@docs
SaveDataSet
```
