

### Lux.jl

The [**Lux.jl**](https://github.com/LuxDL/Lux.jl) package allows for defining neural network architectures for scientific machine learning (SciML) and other deep learning applications.
The InformationGeometry Lux extension provides simplified constructors for creating dense neural network components with different input-output sizes.
In particular, `NormalizedNeuralModel` conveniently wraps the resulting neural network model in a `ModelMap`, including the options for pre- and post-transforms of the respective inputs and outputs to the neural net.


```@docs
NeuralNet
NormalizedNeuralModel
```

