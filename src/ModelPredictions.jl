


### for-loop typically slower than reduce(vcat, ...)
### Apparently curve_fit() throws an error in conjuction with ForwardDiff when reinterpret() is used
# Reduction(X::AbstractVector{<:SVector{Len,T}}) where Len where T = reinterpret(T, X)
Reduction(X::AbstractVector{<:AbstractVector}) = reduce(vcat, X)
Reduction(X::AbstractVector{<:Number}) = X
Reduction(X::AbstractVector{<:SubArray{<:Number, 0}}) = [@inbounds X[i][1] for i in 1:length(X)]


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

_CustomOrNot(DS::Union{Val,AbstractDataSet}, model::Function, θ::AbstractVector{<:Number}, woundX::AbstractVector; kwargs...) = _CustomOrNot(DS, model, θ, woundX, Val(false), Val(false); kwargs...)
_CustomOrNot(DS::Union{Val,AbstractDataSet}, M::ModelMap, θ::AbstractVector{<:Number}, woundX::AbstractVector; kwargs...) = _CustomOrNot(DS, M.Map, θ, woundX, M.CustomEmbedding, M.inplace; kwargs...)

# Specialize this for different Dataset types
_CustomOrNot(::Union{Val,AbstractDataSet}, model::ModelOrFunction, θ::AbstractVector{<:Number}, woundX::AbstractVector, custom::Val{false}, inplace::Val{false}; kwargs...) = Reduction(map(x->model(x,θ; kwargs...), woundX))
_CustomOrNot(::Union{Val,AbstractDataSet}, model::ModelOrFunction, θ::AbstractVector{<:Number}, woundX::AbstractVector, custom::Val{true}, inplace::Val{false}; kwargs...) = model(woundX, θ; kwargs...)


function _CustomOrNot(DS::AbstractDataSet, model!::ModelOrFunction, θ::AbstractVector{<:Number}, woundX::AbstractVector, custom::Val{true}, inplace::Val{true}; kwargs...)
    Y = Vector{suff(θ)}(undef, length(woundX)*ydim(DS))
    model!(Y, woundX, θ; kwargs...);    Y
end
function _CustomOrNot(DS::AbstractDataSet, model!::ModelOrFunction, θ::AbstractVector{<:Number}, woundX::AbstractVector, custom::Val{false}, inplace::Val{true}; kwargs...)
    Y = Vector{suff(θ)}(undef, length(woundX)*ydim(DS))
    EmbeddingMap!(Y, DS, model!, θ, woundX; kwargs...);     Y
end


function EmbeddingMap!(Y::AbstractVector{<:Number}, DM::AbstractDataModel, θ::AbstractVector{<:Number}, woundX::AbstractVector=WoundX(DM); kwargs...)
    EmbeddingMap!(Y, Data(DM), Predictor(DM), θ, woundX; kwargs...)
end
function EmbeddingMap!(Y::AbstractVector{<:Number}, DS::AbstractDataSet, model!::ModelOrFunction, θ::AbstractVector{<:Number}, woundX::AbstractVector=WoundX(DS); kwargs...)
    _EmbeddingMap!(Y, model!, θ, woundX, Val(ydim(DS)), Val(false); kwargs...)
end
function EmbeddingMap!(Y::AbstractVector{<:Number}, DS::AbstractDataSet, model!::ModelMap{true}, θ::AbstractVector{<:Number}, woundX::AbstractVector=WoundX(DS); kwargs...)
    _EmbeddingMap!(Y, model!, θ, woundX, Val(ydim(DS)), model!.CustomEmbedding; kwargs...)
end
# Fallback for out-of-place models
function EmbeddingMap!(Y::AbstractVector{<:Number}, DS::AbstractDataSet, model!::ModelMap{false}, θ::AbstractVector{<:Number}, woundX::AbstractVector=WoundX(DS); kwargs...)
    copyto!(Y, EmbeddingMap(DS, model!, θ, woundX; kwargs...))
end


# in-place does not really make sense for 1D output
function _EmbeddingMap!(Y::AbstractVector{<:Number}, model!::ModelOrFunction, θ::AbstractVector{<:Number}, woundX::AbstractVector, Ydim::Val{1}, custom::Val{false}; kwargs...)
    @inbounds for i in Base.OneTo(length(Y))
        model!(Y[i], woundX[i], θ; kwargs...)
    end
end
function _EmbeddingMap!(Y::AbstractVector{<:Number}, model!::ModelOrFunction, θ::AbstractVector{<:Number}, woundX::AbstractVector, Ydim::Val{T}, custom::Val{false}; kwargs...) where T
    @inbounds for (i, row) in enumerate(Iterators.partition(1:length(Y), T))
        model!(view(Y,row), woundX[i], θ; kwargs...)
    end
end
function _EmbeddingMap!(Y::AbstractVector{<:Number}, model!::ModelOrFunction, θ::AbstractVector{<:Number}, woundX::AbstractVector, Ydim::Val, custom::Val{true}; kwargs...)
    model!(Y, woundX, θ; kwargs...)
end



"""
    EmbeddingMatrix(DM::AbstractDataModel, θ::AbstractVector{<:Number}) -> Matrix
Returns the jacobian of the embedding map as evaluated at the x-values and the parameter configuration ``\\theta``.
"""
EmbeddingMatrix(DM::AbstractDataModel, θ::AbstractVector{<:Number}, woundX::AbstractVector=WoundX(DM); kwargs...) = EmbeddingMatrix(Data(DM), dPredictor(DM), θ, woundX; kwargs...)
EmbeddingMatrix(DS::AbstractDataSet, dmodel::ModelOrFunction, θ::AbstractVector{<:Number}, woundX::AbstractVector=WoundX(DS); kwargs...) = _CustomOrNotdM(DS, dmodel, θ, woundX; kwargs...)
EmbeddingMatrix(DS::Val, dmodel::ModelOrFunction, θ::AbstractVector{<:Number}, woundX::AbstractVector; kwargs...) = _CustomOrNotdM(DS, dmodel, θ, woundX; kwargs...)

_CustomOrNotdM(DS::Union{Val,AbstractDataSet}, dmodel::Function, θ::AbstractVector{<:Number}, woundX::AbstractVector; kwargs...) = _CustomOrNotdM(DS, dmodel, floatify(θ), woundX, Val(false), Val(false); kwargs...)
_CustomOrNotdM(DS::Union{Val,AbstractDataSet}, dM::ModelMap, θ::AbstractVector{<:Number}, woundX::AbstractVector; kwargs...) = _CustomOrNotdM(DS, dM.Map, floatify(θ), woundX, dM.CustomEmbedding, dM.inplace; kwargs...)

# Specialize this for different Dataset types
_CustomOrNotdM(::Union{Val,AbstractDataSet}, dmodel::ModelOrFunction, θ::AbstractVector{<:Number}, woundX::AbstractVector, custom::Val{false}, inplace::Val{false}; kwargs...) = reduce(vcat, map(x->dmodel(x,θ; kwargs...), woundX))
_CustomOrNotdM(::Union{Val,AbstractDataSet}, dmodel::ModelOrFunction, θ::AbstractVector{<:Number}, woundX::AbstractVector, custom::Val{true}, inplace::Val{false}; kwargs...) = dmodel(woundX, θ; kwargs...)


function _CustomOrNotdM(DS::AbstractDataSet, dmodel!::ModelOrFunction, θ::AbstractVector{<:Number}, woundX::AbstractVector, custom::Val{true}, inplace::Val{true}; kwargs...)
    J = Matrix{suff(θ)}(undef, length(woundX)*ydim(DS), length(θ))
    dmodel!(J, woundX, θ; kwargs...);   J
end
function _CustomOrNotdM(DS::AbstractDataSet, dmodel!::ModelOrFunction, θ::AbstractVector{<:Number}, woundX::AbstractVector, custom::Val{false}, inplace::Val{true}; kwargs...)
    J = Matrix{suff(θ)}(undef, length(woundX)*ydim(DS), length(θ))
    EmbeddingMatrix!(J, DS, dmodel!, θ, woundX; kwargs...);     J
end


function EmbeddingMatrix!(J::AbstractMatrix{<:Number}, DM::AbstractDataModel, θ::AbstractVector{<:Number}, woundX::AbstractVector=WoundX(DM); kwargs...)
    EmbeddingMatrix!(J, Data(DM), dPredictor(DM), θ, woundX; kwargs...)
end
function EmbeddingMatrix!(J::AbstractMatrix{<:Number}, DS::AbstractDataSet, dmodel!::ModelOrFunction, θ::AbstractVector{<:Number}, woundX::AbstractVector=WoundX(DS); kwargs...)
    _EmbeddingMatrix!(J, dmodel!, θ, woundX, Val(ydim(DS)), Val(false); kwargs...)
end
function EmbeddingMatrix!(J::AbstractMatrix{<:Number}, DS::AbstractDataSet, dmodel!::ModelMap{true}, θ::AbstractVector{<:Number}, woundX::AbstractVector=WoundX(DS); kwargs...)
    _EmbeddingMatrix!(J, dmodel!, θ, woundX, Val(ydim(DS)), dmodel!.CustomEmbedding; kwargs...)
end
# Fallback for out-of-place models
function EmbeddingMatrix!(J::AbstractMatrix{<:Number}, DS::AbstractDataSet, dmodel!::ModelMap{false}, θ::AbstractVector{<:Number}, woundX::AbstractVector=WoundX(DS); kwargs...)
    copyto!(J, EmbeddingMatrix(DS, dmodel!, θ, woundX; kwargs...))
end


function _EmbeddingMatrix!(J::AbstractMatrix{<:Number}, dmodel!::ModelOrFunction, θ::AbstractVector{<:Number}, woundX::AbstractVector, Ydim::Val{1}, custom::Val{false}; kwargs...)
    @inbounds for row in Base.OneTo(size(J,1))
        dmodel!(view(J,row:row,:), woundX[row], θ; kwargs...)
    end
end
function _EmbeddingMatrix!(J::AbstractMatrix{<:Number}, dmodel!::ModelOrFunction, θ::AbstractVector{<:Number}, woundX::AbstractVector, Ydim::Val{T}, custom::Val{false}; kwargs...) where T
    @inbounds for (i, row) in enumerate(Iterators.partition(1:size(J,1), T))
        dmodel!(view(J,row,:), woundX[i], θ; kwargs...)
    end
end
function _EmbeddingMatrix!(J::AbstractMatrix{<:Number}, dmodel!::ModelOrFunction, θ::AbstractVector{<:Number}, woundX::AbstractVector, Ydim::Val, custom::Val{true}; kwargs...)
    dmodel!(J, woundX, θ; kwargs...)
end
