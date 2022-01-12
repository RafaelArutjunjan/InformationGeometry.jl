

# Forwarding to Metric methods ## Does not seem to work for some reason?
# for Func in [:MetricPartials, :AutoMetricPartials, :ChristoffelSymbol, :ChristoffelPartials, :Riemann, :Ricci, :RicciScalar, :Weyl]
#     @eval ($Func(DM::AbstractDataModel, θ::AbstractVector{<:Number}; BigCalc::Bool=false, kwargs...) = $Func(FisherMetric(DM; kwargs...), θ; BicCalc=BigCalc))
# end

MetricPartials(DM::AbstractDataModel, θ::AbstractVector{<:Number}; ADmode::Val=Val(:ForwardDiff), BigCalc::Bool=false, kwargs...) = MetricPartials(FisherMetric(DM; kwargs...), θ; ADmode=ADmode, BigCalc=BigCalc)
AutoMetricPartials(DM::AbstractDataModel, θ::AbstractVector{<:Number}; ADmode::Val=Val(:ForwardDiff), BigCalc::Bool=false, kwargs...) = AutoMetricPartials(FisherMetric(DM; kwargs...), θ; ADmode=ADmode, BigCalc=BigCalc)
ChristoffelSymbol(DM::AbstractDataModel, θ::AbstractVector{<:Number}; ADmode::Val=Val(:ForwardDiff), BigCalc::Bool=false, kwargs...) = ChristoffelSymbol(FisherMetric(DM; kwargs...), θ; ADmode=ADmode, BigCalc=BigCalc)
ChristoffelPartials(DM::AbstractDataModel, θ::AbstractVector{<:Number}; ADmode::Val=Val(:ForwardDiff), BigCalc::Bool=false, kwargs...) = ChristoffelPartials(FisherMetric(DM; kwargs...), θ; ADmode=ADmode, BigCalc=BigCalc)
Riemann(DM::AbstractDataModel, θ::AbstractVector{<:Number}; ADmode::Val=Val(:ForwardDiff), BigCalc::Bool=false, kwargs...) = Riemann(FisherMetric(DM; kwargs...), θ; ADmode=ADmode, BigCalc=BigCalc)
Ricci(DM::AbstractDataModel, θ::AbstractVector{<:Number}; ADmode::Val=Val(:ForwardDiff), BigCalc::Bool=false, kwargs...) = Ricci(FisherMetric(DM; kwargs...), θ; ADmode=ADmode, BigCalc=BigCalc)
RicciScalar(DM::AbstractDataModel, θ::AbstractVector{<:Number}; ADmode::Val=Val(:ForwardDiff), BigCalc::Bool=false, kwargs...) = RicciScalar(FisherMetric(DM; kwargs...), θ; ADmode=ADmode, BigCalc=BigCalc)
Weyl(DM::AbstractDataModel, θ::AbstractVector{<:Number}; ADmode::Val=Val(:ForwardDiff), BigCalc::Bool=false, kwargs...) = Weyl(FisherMetric(DM; kwargs...), θ; ADmode=ADmode, BigCalc=BigCalc)

"""
Computes
```math
PDV[i,j,m] = (\\partial g)_{ij,m}
```
where the last index is the direction of the derivative.
"""
function MetricPartials(Metric::Function, θ::AbstractVector{<:Number}; ADmode::Val=Val(:ForwardDiff), kwargs...)
    PDV = Array{suff(θ), 3}(undef, length(θ), length(θ), length(θ))
    MetricPartials!(PDV, Metric, θ, ADmode; kwargs...);   PDV
end
function MetricPartials!(PDV::AbstractArray{<:Number,3}, Metric::Function, θ::AbstractVector{<:Number}, ADmode::Val{false}; BigCalc::Bool=false)
    BigCalc && (θ = BigFloat.(θ))
    h = GetH(θ)
    for i in Base.OneTo(length(θ))
        PDV[:,:,i] = (1/(2*h))*(Metric(θ + h*BasisVector(i,length(θ))) - Metric(θ - h*BasisVector(i,length(θ))))
    end
    nothing
end

# For Float64, use AD to compute partial derivatives of metric, else use finite difference with BigFloat.
function AutoMetricPartials(Metric::Function, θ::AbstractVector{<:Number}; ADmode::Val=Val(:ForwardDiff), kwargs...)
    PDV = Array{suff(θ), 3}(undef, length(θ), length(θ), length(θ))
    MetricPartials!(PDV, Metric, θ, ADmode; kwargs...); PDV
end
function MetricPartials!(PDV::AbstractArray{<:Number,3}, Metric::Function, θ::AbstractVector{<:Number}, ADmode::Val; BigCalc::Bool=false)
    BigCalc && (θ = BigFloat.(θ))
    # PDV[:] = vec(GetJac(ADmode,Metric)(θ))
    # GetJac!(ADmode)(PDV, Metric, θ) ## no passthrough
    GetMatrixJac!(ADmode,Metric)(PDV, θ)
    nothing
end


# function ChristoffelSymbol(Metric::Function, θ::AbstractVector{<:Number}; BigCalc::Bool=false)
#     Finv = inv(Metric(θ));    FPDV = MetricPartials(Metric, θ; BigCalc=BigCalc)
#     if (suff(θ) == Float64) && BigCalc     FPDV = convert(Array{Float64,3}, FPDV)    end
#     @tullio Christoffels[a,i,j] := ((1/2) * Finv)[a,m] * (FPDV[j,m,i] + FPDV[m,i,j] - FPDV[i,j,m])
# end

# Accuracy ≈ 3e-11
# BigCalc for using BigFloat Calculation in finite differencing step but outputting Float64 again.
"""
    ChristoffelSymbol(DM::DataModel, θ::AbstractVector; BigCalc::Bool=false)
    ChristoffelSymbol(Metric::Function, θ::AbstractVector; BigCalc::Bool=false)
Calculates the components of the ``(1,2)`` Christoffel symbol ``\\Gamma`` at a point ``\\theta`` (i.e. the Christoffel symbol "of the second kind") through finite differencing of the `Metric`. Accurate to ≈ 3e-11.
`BigCalc=true` increases accuracy through `BigFloat` calculation.
"""
function ChristoffelSymbol(Metric::Function, θ::AbstractVector{<:Number}; ADmode::Val=Val(:ForwardDiff), kwargs...)
    # Need to initialize with zeros() instead of Array{}() due to tensor macro
    Γ = zeros(suff(θ), length(θ), length(θ), length(θ))
    ChristoffelSymbol!(Γ, Metric, θ, ADmode; kwargs...);  Γ
end
function ChristoffelSymbol!(Γ::AbstractArray{<:Number,3}, Metric::Function, θ::AbstractVector{<:Number}, ADmode::Val; kwargs...)
    PDV = zeros(suff(θ), length(θ), length(θ), length(θ))
    ChristoffelSymbol!!(PDV, Γ, Metric, θ, ADmode; kwargs...)
end
function ChristoffelSymbol!!(PDV::AbstractArray{<:Number,3}, Γ::AbstractArray{<:Number,3}, Metric::Function, θ::AbstractVector{<:Number}, ADmode::Val; kwargs...)
    MetricPartials!(PDV, Metric, θ, ADmode; kwargs...)
    # ((suff(θ) == Float64) && BigCalc) && (FPDV = convert(Array{Float64,3}, FPDV))
    PDV2Christoffel!(Γ, inv(Metric(θ)), PDV)
    nothing
end


function PDV2Christoffel!(Γ::AbstractArray{<:Number,3}, InvMetric::AbstractMatrix{<:Number}, PDV::AbstractArray{<:Number,3})
    @tullio Γ[a,i,j] = ((1/2) * InvMetric)[a,m] * (PDV[j,m,i] + PDV[m,i,j] - PDV[i,j,m])
    nothing
end
function PDV2Christoffel(InvMetric::AbstractMatrix{<:Number}, PDV::AbstractArray{<:Number,3})
    n = size(InvMetric)[1];    Γ = Array{suff(PDV), 3}(undef, n, n, n)
    PDV2Christoffel!(Γ, InvMetric, PDV);    Γ
end

function PDV2Christoffel2!(Γ::AbstractArray{<:Number,3}, InvMetric::AbstractMatrix{<:Number}, PDV::AbstractArray{<:Number,3})
    # Use symmetry in lower indices of Christoffel symbols due to Levi--Civita connection to eliminate one term in contraction
    @tullio Γ[a,i,j] = InvMetric[a,m] * (PDV[m,j,i] - (0.5*PDV)[i,j,m])
    nothing
end

ChristoffelTerm(Γ::AbstractArray{<:Number,3}, v::AbstractVector{<:Number}) = @tullio Res[a] := (-1*Γ)[a,b,c] * v[b] * v[c]


# function ChristoffelPartials(Metric::Function, θ::AbstractVector{<:Number}; BigCalc::Bool=false)
#     BigCalc && (θ = BigFloat.(θ))
#     DownUpDownDown = Array{suff(θ)}(undef,length(θ),length(θ),length(θ),length(θ))
#     h = GetH(θ)
#     for i in 1:length(θ)
#         DownUpDownDown[i,:,:,:] .= (ChristoffelSymbol(Metric,θ + h*BasisVector(i,length(θ))) .- ChristoffelSymbol(Metric,θ - h*BasisVector(i,length(θ))))
#     end;        (1/(2*h))*DownUpDownDown
# end

function ChristoffelPartials(Metric::Function, θ::AbstractVector{<:Number}; ADmode::Val=Val(:ForwardDiff), kwargs...)
    ∂Γ = Array{suff(θ), 4}(undef, length(θ), length(θ), length(θ), length(θ))
    ChristoffelPartials!(∂Γ, Metric, θ, ADmode; kwargs...);   ∂Γ
end

"""
Returns partial derivatives of Christoffel Symbols Γ with index structure up-down-down-down, i.e. (∂Γ)ᵇₑⱼₐ = Γᵇₑⱼ,ₐ where the last index denotes the direction of the derivative.
"""
function ChristoffelPartials!(∂Γ::AbstractArray{<:Number,4}, Metric::Function, θ::AbstractVector{<:Number}, ADmode::Val{false}; BigCalc::Bool=false)
    BigCalc && (θ = BigFloat.(θ))
    h = GetH(θ)
    FPDV = Array{suff(θ),3}(undef, length(θ), length(θ), length(θ))
    Γ₁ = Array{suff(θ),3}(undef, length(θ), length(θ), length(θ));  Γ₂ = Array{suff(θ),3}(undef, length(θ), length(θ), length(θ))
    for i in Base.OneTo(length(θ))
        # ∂Γ[i,:,:,:] .= (1/(2*h))*(ChristoffelSymbol(Metric,θ + h*BasisVector(i,length(θ))) .- ChristoffelSymbol(Metric,θ - h*BasisVector(i,length(θ))))
        ChristoffelSymbol!!(FPDV, Γ₂, Metric, θ + h*BasisVector(i,length(θ)), ADmode; BigCalc=BigCalc)
        ChristoffelSymbol!!(FPDV, Γ₁, Metric, θ - h*BasisVector(i,length(θ)), ADmode; BigCalc=BigCalc)
        ∂Γ[:,:,:,i] = (1/(2*h))*(Γ₂ - Γ₁)
    end;    nothing
end
function ChristoffelPartials!(∂Γ::AbstractArray{<:Number,4}, Metric::Function, θ::AbstractVector{<:Number}, ADmode::Val; BigCalc::Bool=false)
    BigCalc && (θ = BigFloat.(θ))
    GetMatrixJac!(ADmode, x->ChristoffelSymbol(Metric, x; ADmode=ADmode))(∂Γ, θ)
    nothing
end

"""
    Riemann(DM::DataModel, θ::AbstractVector; BigCalc::Bool=false)
    Riemann(Metric::Function, θ::AbstractVector; BigCalc::Bool=false)
Calculates the components of the ``(1,3)`` Riemann tensor by finite differencing of the `Metric`. `BigCalc=true` increases accuracy through BigFloat calculation.
"""
function Riemann(Metric::Function, θ::AbstractVector{<:Number}; ADmode::Val=Val(:ForwardDiff), BigCalc::Bool=false)
    ∂Γ = ChristoffelPartials(Metric, θ; ADmode=ADmode, BigCalc=BigCalc)
    if (suff(θ) == Float64) && BigCalc
        ∂Γ = convert(Array{Float64,4}, ∂Γ)
    end
    Γ = ChristoffelSymbol(Metric, θ; ADmode=ADmode, BigCalc=BigCalc)
    # @tullio Riem[m,i,k,p] := DownUpDownDown[k,m,i,p] - DownUpDownDown[p,m,i,k] + Gamma[a,i,p]*Gamma[m,a,k] - Gamma[a,i,k]*Gamma[m,a,p]
    Riem = Array{suff(θ), 4}(undef, length(θ), length(θ), length(θ), length(θ))
    RiemannLastInd!(Riem, ∂Γ, Γ);  Riem
end

# """
# Assumes index structure (∂Γ)ₐᵇₑⱼ where the FIRST index denotes direction of derivative.
# ```math
# \\tensor{\\mathrm{Riem}}{^i _j _k _l} = \\pdv{x^k}\\, \\tensor{\\Gamma}{^i _j _l} - \\pdv{x^l}\\, \\tensor{\\Gamma}{^i _j _k} + \\tensor{\\Gamma}{^i _a _k} \\, \\tensor{\\Gamma}{^a _j _l} - \\tensor{\\Gamma}{^i _a _l} \\, \\tensor{\\Gamma}{^a _j _k}
# ```
# """
# function RiemannFirstInd!(Riem::AbstractArray{<:Number,4}, ∂Γ::AbstractArray{<:Number,4}, Γ::AbstractArray{<:Number,3})
#     # @tensor Riem[i,j,k,l] = ∂Γ[k,i,j,l] - ∂Γ[l,i,j,k] + Γ[i,a,k]*Γ[a,j,l] - Γ[i,a,l]*Γ[a,j,k]
#     # With Tullio, this needs to be split over two lines!!! (Why?!)
#     @tullio Riem[i,j,k,l] = ∂Γ[k,i,j,l] - ∂Γ[l,i,j,k]
#     @tullio Riem[i,j,k,l] += Γ[i,a,k]*Γ[a,j,l] - Γ[i,a,l]*Γ[a,j,k]
#     nothing
# end
"""
Assumes index structure (∂Γ)ᵇₑⱼₐ = Γᵇₑⱼ,ₐ where the LAST index denotes direction of derivative.
```math
\\tensor{\\mathrm{Riem}}{^i _j _k _l} = \\pdv{x^k}\\, \\tensor{\\Gamma}{^i _j _l} - \\pdv{x^l}\\, \\tensor{\\Gamma}{^i _j _k} + \\tensor{\\Gamma}{^i _a _k} \\, \\tensor{\\Gamma}{^a _j _l} - \\tensor{\\Gamma}{^i _a _l} \\, \\tensor{\\Gamma}{^a _j _k}
```
"""
function RiemannLastInd!(Riem::AbstractArray{<:Number,4}, ∂Γ::AbstractArray{<:Number,4}, Γ::AbstractArray{<:Number,3})
    # @tensor Riem[i,j,k,l] = ∂Γ[i,j,l,k] - ∂Γ[i,j,k,l] + Γ[i,a,k]*Γ[a,j,l] - Γ[i,a,l]*Γ[a,j,k]
    # With Tullio, this needs to be split over two lines!!! (Why?!)
    @tullio Riem[i,j,k,l] = ∂Γ[i,j,l,k] - ∂Γ[i,j,k,l]
    @tullio Riem[i,j,k,l] += Γ[i,a,k]*Γ[a,j,l] - Γ[i,a,l]*Γ[a,j,k]
    nothing
end

"""
    Ricci(DM::DataModel, θ::AbstractVector; BigCalc::Bool=false)
    Ricci(Metric::Function, θ::AbstractVector; BigCalc::Bool=false)
Calculates the components of the ``(0,2)`` Ricci tensor by finite differencing of the `Metric`. `BigCalc=true` increases accuracy through `BigFloat` calculation.
"""
function Ricci(Metric::Function, θ::AbstractVector{<:Number}; kwargs...)
    Riem = Riemann(Metric, θ; kwargs...)
    # For some reason, it is necessary to prefill here.
    RIC = Array{suff(θ),2}(undef, length(θ), length(θ))
    @tullio RIC[a,b] = Riem[c,a,c,b]
end

"""
    RicciScalar(DM::DataModel, θ::AbstractVector; BigCalc::Bool=false) -> Real
    RicciScalar(Metric::Function, θ::AbstractVector; BigCalc::Bool=false) -> Real
Calculates the Ricci scalar by finite differencing of the `Metric`. `BigCalc=true` increases accuracy through `BigFloat` calculation.
"""
RicciScalar(Metric::Function, θ::AbstractVector{<:Number}; kwargs...) = tr(transpose(Ricci(Metric, θ; kwargs...)) * inv(Metric(θ)))
