

"""
    LinearModel(x::Union{Number,AbstractVector{<:Number}}, θ::AbstractVector{<:Number})
```math
y(x,θ) = θ_{n+1} + x_1 * θ_1 + x_2 * θ_2 + ... + x_n * θ_n
```
"""
LinearModel(x::Union{Number,AbstractVector{<:Number}}, θ::AbstractVector{<:Number}) = dot(view(θ, 1:(length(θ)-1)), x) + θ[end]
"""
    QuadraticModel(x::Union{Number,AbstractVector{<:Number}}, θ::AbstractVector{<:Number})
```math
y(x,θ) = θ_1 * x^2 + θ_2 * x + θ_3
```
"""
QuadraticModel(x::Union{Number,AbstractVector{<:Number}}, θ::AbstractVector{<:Number}) = (n=length(θ);  dot(view(θ,1:((n-1)÷2)), x.^2) + dot(view(θ,(n-1)÷2+1:n-1), x) + θ[end])
"""
    ExponentialModel(x::Union{Number,AbstractVector{<:Number}}, θ::AbstractVector{<:Number})
```math
y(x,θ) = exp(θ_{n+1} + x_1 * θ_1 + x_2 * θ_2 + ... + x_n * θ_n)
```
"""
ExponentialModel = exp∘LinearModel
"""
    SumExponentialsModel(x::Union{Number,AbstractVector{<:Number}}, θ::AbstractVector{<:Number})    
```math
y(x,θ) = θ_{n+1} + exp(x_1 * θ_1) + exp(x_2 * θ_2) + ... + exp(x_n * θ_n)
```
"""
SumExponentialsModel(x::Union{Number,AbstractVector{<:Number}}, θ::AbstractVector{<:Number}) = sum(exp.(view(θ,1:length(θ)-1) .* x)) + θ[end]
"""
    PolynomialModel(n::Int)
Creates a polynomial of degree `n`:
```math
y(x,θ) = θ_1 * x^n + θ_2 * x^{n-1} + ... θ_{n} * x + θ_{n+1}
```
"""
PolynomialModel(degree::Int) = Polynomial(x::Number, θ::AbstractVector{<:Number}) = sum(θ[i] * x^(i-1) for i in 1:(degree+1))


function GetLinearModel(DS::AbstractDataSet)
    ydim(DS) != 1 && return GetGeneralLinearModel(DS)
    Names = "p_(" .* ynames(DS) .* " × " .* xnames(DS) .*")"
    push!(Names, "p_(" * ynames(DS)[1] * " × Offset)")
    ModelMap(LinearModel, (xdim(DS), ydim(DS), xdim(DS)+1); pnames=Names)
end

function GetGeneralLinearModel(DS::AbstractDataSet)
    ydim(DS) == 1 && return GetLinearModel(DS)
    Xdim, Ydim = xdim(DS), ydim(DS)
    NaiveGeneralLinearModel(x::AbstractVector{<:Number}, θ::AbstractVector{T}) where T <: Number = SVector{Ydim, T}(LinearModel(x, p) for p in Iterators.partition(θ, Xdim+1))
    Names = ["p_(" .* ynames(DS)[i] .* " × " .* xnames(DS) .*")" for i in 1:ydim(DS)]
    for (i,series) in enumerate(Names)
        push!(series, "p_(" * ynames(DS)[i] * " × Offset)")
    end
    OptimizeModel(ModelMap(NaiveGeneralLinearModel, nothing, nothing, (xdim(DS), ydim(DS), ydim(DS)*(xdim(DS)+1)), reduce(vcat, Names), Val(true), Val(false), Val(false)); inplace=false)[1]
end



