


### for-loop typically slower than reduce(vcat, ...)
### Apparently curve_fit() throws an error in conjuction with ForwardDiff when reinterpret() is used
# Reduction(X::AbstractVector{<:SVector{Len,T}}) where Len where T = reinterpret(T, X)
Reduction(X::AbstractVector{<:AbstractVector}) = reduce(vcat, X)
Reduction(X::AbstractVector{<:Number}) = X
Reduction(X::AbstractVector{<:SubArray{<:Number, 0}}) = [@inbounds X[i][1] for i in eachindex(X)]


# h(θ) ∈ Dataspace
"""
    EmbeddingMap(DM::AbstractDataModel, θ::AbstractVector{<:Number}) -> Vector
Returns a vector of the collective predictions of the `model` as evaluated at the x-values and the parameter configuration ``\\theta``.
```
h(\\theta) \\coloneqq \\big(y_\\mathrm{model}(x_1;\\theta),...,y_\\mathrm{model}(x_N;\\theta)\\big) \\in \\mathcal{D}
```
"""
EmbeddingMap(DM::AbstractDataModel, θ::AbstractVector{<:Number}, woundX::AbstractVector=WoundX(DM); kwargs...) = EmbeddingMap(Data(DM), Predictor(DM), θ, woundX; kwargs...)
EmbeddingMap(DS::AbstractDataSet, model::ModelOrFunction, θ::AbstractVector{<:Number}, woundX::AbstractVector=WoundX(DS); kwargs...) = _CustomOrNot(DS, model, θ, woundX; kwargs...)
EmbeddingMap(DS::Val, model::ModelOrFunction, θ::AbstractVector{<:Number}, woundX::AbstractVector; kwargs...) = _CustomOrNot(DS, model, θ, woundX; kwargs...)


#### Specialize this for different Dataset types
# Shortcut without needing to create Vals
_CustomOrNot(DS::Union{Val,AbstractDataSet}, model::Union{Function,ModelMap{false, false}}, θ::AbstractVector{<:Number}, woundX::AbstractVector; kwargs...) = Reduction(map(x->model(x,θ; kwargs...), woundX))
_CustomOrNot(DS::Union{Val,AbstractDataSet}, model::ModelMap{false, true}, θ::AbstractVector{<:Number}, woundX::AbstractVector; kwargs...) = model(woundX, θ; kwargs...)
function _CustomOrNot(DS::AbstractDataSet, model!::ModelMap{true, true}, θ::AbstractVector{<:Number}, woundX::AbstractVector; Ycache=Vector{suff(θ)}(undef, length(woundX)*ydim(DS)), kwargs...)
    model!(Ycache, woundX, θ; kwargs...);    Ycache
end
function _CustomOrNot(DS::AbstractDataSet, model!::ModelMap{true, false}, θ::AbstractVector{<:Number}, woundX::AbstractVector; Ycache=Vector{suff(θ)}(undef, length(woundX)*ydim(DS)), kwargs...)
    EmbeddingMap!(Ycache, DS, model!, θ, woundX; kwargs...);     Ycache
end

## Backwards compatible manual indication of custom and inplace
# _CustomOrNot(DS::Union{Val,AbstractDataSet}, model::ModelOrFunction, θ::AbstractVector{<:Number}, woundX::AbstractVector; kwargs...) = _CustomOrNot(DS, model, θ, woundX, Val(false), Val(false); kwargs...)
# _CustomOrNot(DS::Union{Val,AbstractDataSet}, M::ModelMap, θ::AbstractVector{<:Number}, woundX::AbstractVector; kwargs...) = _CustomOrNot(DS, M.Map, θ, woundX, M.CustomEmbedding, M.inplace; kwargs...)
# _CustomOrNot(::Union{Val,AbstractDataSet}, model::ModelOrFunction, θ::AbstractVector{<:Number}, woundX::AbstractVector, custom::Val{false}, inplace::Val{false}; kwargs...) = Reduction(map(x->model(x,θ; kwargs...), woundX))
# _CustomOrNot(::Union{Val,AbstractDataSet}, model::ModelOrFunction, θ::AbstractVector{<:Number}, woundX::AbstractVector, custom::Val{true}, inplace::Val{false}; kwargs...) = model(woundX, θ; kwargs...)
#
# function _CustomOrNot(DS::AbstractDataSet, model!::ModelOrFunction, θ::AbstractVector{<:Number}, woundX::AbstractVector, custom::Val{true}, inplace::Val{true}; Ycache=Vector{suff(θ)}(undef, length(woundX)*ydim(DS)), kwargs...)
#     model!(Ycache, woundX, θ; kwargs...);    Ycache
# end
# function _CustomOrNot(DS::AbstractDataSet, model!::ModelOrFunction, θ::AbstractVector{<:Number}, woundX::AbstractVector, custom::Val{false}, inplace::Val{true}; Ycache=Vector{suff(θ)}(undef, length(woundX)*ydim(DS)), kwargs...)
#     EmbeddingMap!(Ycache, DS, model!, θ, woundX; kwargs...);     Ycache
# end


function EmbeddingMap!(Y::AbstractVector{<:Number}, DM::AbstractDataModel, θ::AbstractVector{<:Number}, woundX::AbstractVector=WoundX(DM); kwargs...)
    EmbeddingMap!(Y, Data(DM), Predictor(DM), θ, woundX; kwargs...)
end
function EmbeddingMap!(Y::AbstractVector{<:Number}, DS::AbstractDataSet, model!::ModelOrFunction, θ::AbstractVector{<:Number}, woundX::AbstractVector=WoundX(DS); kwargs...)
    _EmbeddingMap!(Y, model!, θ, woundX, Val(ydim(DS)); kwargs...)
end
function EmbeddingMap!(Y::AbstractVector{<:Number}, DS::AbstractDataSet, model!::ModelMap{true, false}, θ::AbstractVector{<:Number}, woundX::AbstractVector=WoundX(DS); kwargs...)
    _EmbeddingMap!(Y, model!, θ, woundX, Val(ydim(DS)); kwargs...)
end
function EmbeddingMap!(Y::AbstractVector{<:Number}, DS::AbstractDataSet, model!::ModelMap{true, true}, θ::AbstractVector{<:Number}, woundX::AbstractVector=WoundX(DS); kwargs...)
    model!(Y, woundX, θ; kwargs...)
end
# Fallback for out-of-place models
function EmbeddingMap!(Y::AbstractVector{<:Number}, DS::AbstractDataSet, model!::Union{Function,ModelMap{false}}, θ::AbstractVector{<:Number}, woundX::AbstractVector=WoundX(DS); kwargs...)
    copyto!(Y, EmbeddingMap(DS, model!, θ, woundX; kwargs...))
end

# in-place does not really make sense for 1D output
function _EmbeddingMap!(Y::AbstractVector{<:Number}, model!::ModelOrFunction, θ::AbstractVector{<:Number}, woundX::AbstractVector, Ydim::Val{1}; kwargs...)
    @inbounds for i in Base.OneTo(length(Y))
        model!(Y[i], woundX[i], θ; kwargs...)
    end
end
function _EmbeddingMap!(Y::AbstractVector{<:Number}, model!::ModelOrFunction, θ::AbstractVector{<:Number}, woundX::AbstractVector, Ydim::Val{T}; kwargs...) where T
    @inbounds for (i, row) in enumerate(Iterators.partition(1:length(Y), T))
        model!(view(Y,row), woundX[i], θ; kwargs...)
    end
end



"""
    EmbeddingMatrix(DM::AbstractDataModel, θ::AbstractVector{<:Number}) -> Matrix
Returns the jacobian of the embedding map as evaluated at the x-values and the parameter configuration ``\\theta``.
"""
EmbeddingMatrix(DM::AbstractDataModel, θ::AbstractVector{<:Number}, woundX::AbstractVector=WoundX(DM); kwargs...) = EmbeddingMatrix(Data(DM), dPredictor(DM), θ, woundX; kwargs...)
EmbeddingMatrix(DS::AbstractDataSet, dmodel::ModelOrFunction, θ::AbstractVector{<:Number}, woundX::AbstractVector=WoundX(DS); kwargs...) = _CustomOrNotdM(DS, dmodel, θ, woundX; kwargs...)
EmbeddingMatrix(DS::Val, dmodel::ModelOrFunction, θ::AbstractVector{<:Number}, woundX::AbstractVector; kwargs...) = _CustomOrNotdM(DS, dmodel, θ, woundX; kwargs...)

#### Specialize this for different Dataset types
# Shortcut without needing to create Vals
_CustomOrNotdM(::Union{Val,AbstractDataSet}, dmodel::Union{Function,ModelMap{false, false}}, θ::AbstractVector{<:Number}, woundX::AbstractVector; kwargs...) = reduce(vcat, map(x->dmodel(x,θ; kwargs...), woundX))
_CustomOrNotdM(::Union{Val,AbstractDataSet}, dmodel::ModelMap{false, true}, θ::AbstractVector{<:Number}, woundX::AbstractVector; kwargs...) = dmodel(woundX, θ; kwargs...)
function _CustomOrNotdM(DS::AbstractDataSet, dmodel!::ModelMap{true, true}, θ::AbstractVector{<:Number}, woundX::AbstractVector; Jcache=Matrix{suff(θ)}(undef, length(woundX)*ydim(DS), length(θ)), kwargs...)
    dmodel!(Jcache, woundX, θ; kwargs...);   Jcache
end
function _CustomOrNotdM(DS::AbstractDataSet, dmodel!::ModelMap{true, false}, θ::AbstractVector{<:Number}, woundX::AbstractVector; Jcache=Matrix{suff(θ)}(undef, length(woundX)*ydim(DS), length(θ)), kwargs...)
    EmbeddingMatrix!(Jcache, DS, dmodel!, θ, woundX; kwargs...);     Jcache
end


## Backwards compatible manual indication of custom and inplace
# _CustomOrNotdM(DS::Union{Val,AbstractDataSet}, dmodel::ModelOrFunction, θ::AbstractVector{<:Number}, woundX::AbstractVector; kwargs...) = _CustomOrNotdM(DS, dmodel, floatify(θ), woundX, Val(false), Val(false); kwargs...)
# _CustomOrNotdM(DS::Union{Val,AbstractDataSet}, dM::ModelMap, θ::AbstractVector{<:Number}, woundX::AbstractVector; kwargs...) = _CustomOrNotdM(DS, dM.Map, floatify(θ), woundX, dM.CustomEmbedding, dM.inplace; kwargs...)
# _CustomOrNotdM(::Union{Val,AbstractDataSet}, dmodel::ModelOrFunction, θ::AbstractVector{<:Number}, woundX::AbstractVector, custom::Val{false}, inplace::Val{false}; kwargs...) = reduce(vcat, map(x->dmodel(x,θ; kwargs...), woundX))
# _CustomOrNotdM(::Union{Val,AbstractDataSet}, dmodel::ModelOrFunction, θ::AbstractVector{<:Number}, woundX::AbstractVector, custom::Val{true}, inplace::Val{false}; kwargs...) = dmodel(woundX, θ; kwargs...)
#
#
# function _CustomOrNotdM(DS::AbstractDataSet, dmodel!::ModelOrFunction, θ::AbstractVector{<:Number}, woundX::AbstractVector, custom::Val{true}, inplace::Val{true}; Jcache=Matrix{suff(θ)}(undef, length(woundX)*ydim(DS), length(θ)), kwargs...)
#     dmodel!(Jcache, woundX, θ; kwargs...);   Jcache
# end
# function _CustomOrNotdM(DS::AbstractDataSet, dmodel!::ModelOrFunction, θ::AbstractVector{<:Number}, woundX::AbstractVector, custom::Val{false}, inplace::Val{true}; Jcache=Matrix{suff(θ)}(undef, length(woundX)*ydim(DS), length(θ)), kwargs...)
#     EmbeddingMatrix!(Jcache, DS, dmodel!, θ, woundX; kwargs...);     Jcache
# end


function EmbeddingMatrix!(J::AbstractMatrix{<:Number}, DM::AbstractDataModel, θ::AbstractVector{<:Number}, woundX::AbstractVector=WoundX(DM); kwargs...)
    EmbeddingMatrix!(J, Data(DM), dPredictor(DM), θ, woundX; kwargs...)
end
# function EmbeddingMatrix!(J::AbstractMatrix{<:Number}, DS::AbstractDataSet, dmodel!::ModelOrFunction, θ::AbstractVector{<:Number}, woundX::AbstractVector=WoundX(DS); kwargs...)
#     _EmbeddingMatrix!(J, dmodel!, θ, woundX, Val(ydim(DS)); kwargs...)
# end
function EmbeddingMatrix!(J::AbstractMatrix{<:Number}, DS::AbstractDataSet, dmodel!::ModelMap{true, false}, θ::AbstractVector{<:Number}, woundX::AbstractVector=WoundX(DS); kwargs...)
    _EmbeddingMatrix!(J, dmodel!, θ, woundX, Val(ydim(DS)); kwargs...)
end
function EmbeddingMatrix!(J::AbstractMatrix{<:Number}, DS::AbstractDataSet, dmodel!::ModelMap{true, true}, θ::AbstractVector{<:Number}, woundX::AbstractVector=WoundX(DS); kwargs...)
    dmodel!(J, woundX, θ; kwargs...)
end
# Fallback for out-of-place models
function EmbeddingMatrix!(J::AbstractMatrix{<:Number}, DS::AbstractDataSet, dmodel!::Union{Function,ModelMap{false}}, θ::AbstractVector{<:Number}, woundX::AbstractVector=WoundX(DS); kwargs...)
    copyto!(J, EmbeddingMatrix(DS, dmodel!, θ, woundX; kwargs...))
end


function _EmbeddingMatrix!(J::AbstractMatrix{<:Number}, dmodel!::ModelOrFunction, θ::AbstractVector{<:Number}, woundX::AbstractVector, Ydim::Val{1}; kwargs...)
    @inbounds for row in Base.OneTo(size(J,1))
        dmodel!(view(J,row:row,:), woundX[row], θ; kwargs...)
    end
end
function _EmbeddingMatrix!(J::AbstractMatrix{<:Number}, dmodel!::ModelOrFunction, θ::AbstractVector{<:Number}, woundX::AbstractVector, Ydim::Val{T}; kwargs...) where T
    @inbounds for (i, row) in enumerate(Iterators.partition(1:size(J,1), T))
        dmodel!(view(J,row,:), woundX[i], θ; kwargs...)
    end
end
