module InformationGeometryLuxExt


using InformationGeometry, Lux
using ComponentArrays, Random, StaticArrays, Distributions

import InformationGeometry: WoundX, WoundY, NeuralNet, NormalizedNeuralModel
GetWidths(X::AbstractVector{<:AbstractVector}) = [GetWidths(getindex.(X, i)) for i in eachindex(X[1])]
GetWidths(X::AbstractVector{<:Number}) = (E=extrema(X);    2(E[2]-E[1]))
# Should normalize inputs and outputs around model for best performance!


NeuralNet(DS::AbstractDataSet, args...; kwargs...) = NeuralNet(xdim(DS), ydim(DS), args...; kwargs...)
"""
    NeuralNet(DS::AbstractDataSet, N::Int=2, hidden::Int=1; kwargs...)
    NeuralNet(In::Int, Out::Int, N::Int=2, hidden::Int=1; positive::Bool=false, HiddenActivation::Function=tanh, 
                            FinalActivation::Function=(positive ? softplus : identity), gain::Real=1, kwargs...)
`N` is number of neurons in intermediate layers, `hidden` is number of hidden layers, returns `Lux.Chain`.
If `hidden` is `-1`, the dedicated output layer is also dropped, returning a single `Lux.Dense` layer.
"""
function NeuralNet(In::Int, Out::Int, N::Int=2, hidden::Int=1; positive::Bool=false, activation::Function=tanh, HiddenActivation::Function=activation, 
                            FinalActivation::Function=(positive ? softplus : identity), gain::Real=1, 
                            init_weight=Lux.kaiming_uniform(; gain), kwargs...)
    hidden == -1 && return Lux.Dense(In, Out, HiddenActivation; init_weight, kwargs...)

    Lux.Chain(Lux.Dense(In, N, HiddenActivation; init_weight, kwargs...), 
            [Lux.Dense(N, N, HiddenActivation; init_weight, kwargs...) for i in 1:hidden]..., 
            Lux.Dense(N, Out, FinalActivation; init_weight, kwargs...))
end

function NormalizedNeuralModel(DS::AbstractDataSet, args...; kwargs...)
    NormalizedNeuralModel(xdim(DS), ydim(DS), args...; Xmean=mean(WoundX(DS)), Xdiv=GetWidths(WoundX(DS)), Ymean=mean(WoundY(DS)), Ydiv=GetWidths(WoundY(DS)), kwargs...)
end
"""
    NormalizedNeuralModel(xd::Int, yd::Int, N::Int=2, hidden::Int=1; kwargs...)
    NormalizedNeuralModel(DS::AbstractDataSet, N::Int=2, hidden::Int=1; rng=Random.default_rng(), PreTransform::Function=x->(x .- Xmean) ./ Xdiv, 
                    PostTransform::Function=y->(Ydiv .* y) .+ Ymean, kwargs...)
Returns Tuple `(M, P, U)` where `M` is a `ModelMap` of the neural net with given input dimensions including a normalization of inputs and outputs.
`P` is a random initial `ComponentVector` parameter configuration and `U` is the `Lux.Chain` object of the underlying neural net.
"""
function NormalizedNeuralModel(xd::Int, yd::Int, args...; rng=Random.default_rng(), 
                    Xmean::Union{AbstractVector{<:Number},<:Number}=(xd == 1 ? 0.0 : zeros(xd)), 
                    Xdiv::Union{AbstractVector{<:Number},<:Number}=(xd == 1 ? 1.0 : ones(xd)),
                    Ymean::Union{AbstractVector{<:Number},<:Number}=(yd == 1 ? 0.0 : zeros(yd)), 
                    Ydiv::Union{AbstractVector{<:Number},<:Number}=(yd == 1 ? 1.0 : ones(yd)), 
                    PreTransform::Function=x->(x .- Xmean) ./ Xdiv, 
                    PostTransform::Function=y->(Ydiv .* y) .+ Ymean, Domain::Union{Nothing,HyperCube}=nothing, kwargs...)
    U = NeuralNet(xd, yd, args...; kwargs...)
    NormalizedNeuralModel(U, xd, yd; rng, Xmean, Xdiv, Ymean, Ydiv, PreTransform, PostTransform, Domain)
end
function NormalizedNeuralModel(U::Lux.AbstractLuxLayer, xd::Int, yd::Int; rng=Random.default_rng(), 
                    Xmean::Union{AbstractVector{<:Number},<:Number}=(xd == 1 ? 0.0 : zeros(xd)), 
                    Xdiv::Union{AbstractVector{<:Number},<:Number}=(xd == 1 ? 1.0 : ones(xd)),
                    Ymean::Union{AbstractVector{<:Number},<:Number}=(yd == 1 ? 0.0 : zeros(yd)), 
                    Ydiv::Union{AbstractVector{<:Number},<:Number}=(yd == 1 ? 1.0 : ones(yd)), 
                    PreTransform::Function=x->(x .- Xmean) ./ Xdiv, 
                    PostTransform::Function=y->(Ydiv .* y) .+ Ymean, Domain::Union{Nothing,HyperCube}=nothing)
    p_NN_32, _st = Lux.setup(rng, U);    p_NN = ComponentVector{Float64}(p_NN_32)
    Unet(x::AbstractVector{<:Number}, p_NN::AbstractVector{<:Number}) = U(x, p_NN, _st)[1]
    model(x::AbstractVector, p_NN::ComponentVector) = PostTransform(Unet(PreTransform(x), p_NN))
    model(x::AbstractVector, p::AbstractVector) = model(x, convert(typeof(p_NN), p))
    M = xd == 1 ? (x::Real,p::AbstractVector)->model(SA[x],p) : model
    ModelMap((yd == 1 ? (x,p)->M(x,p)[1] : M), Domain, nothing, (xd, yd, length(p_NN)); startp=p_NN, IsCustom=false), p_NN, U
end


end # module