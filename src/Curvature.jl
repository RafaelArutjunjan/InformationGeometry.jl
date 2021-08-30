

# Forwarding to Metric methods ## Does not seem to work for some reason?
# for Func in [:MetricPartials, :AutoMetricPartials, :ChristoffelSymbol, :ChristoffelPartials, :Riemann, :Ricci, :RicciScalar, :Weyl]
#     @eval ($Func(DM::AbstractDataModel, θ::AbstractVector{<:Number}; BigCalc::Bool=false, kwargs...) = $Func(FisherMetric(DM; kwargs...), θ; BicCalc=BigCalc))
# end

MetricPartials(DM::AbstractDataModel, θ::AbstractVector{<:Number}; BigCalc::Bool=false, kwargs...) = MetricPartials(FisherMetric(DM; kwargs...), θ; BigCalc=BigCalc)
AutoMetricPartials(DM::AbstractDataModel, θ::AbstractVector{<:Number}; BigCalc::Bool=false, kwargs...) = AutoMetricPartials(FisherMetric(DM; kwargs...), θ; BigCalc=BigCalc)
ChristoffelSymbol(DM::AbstractDataModel, θ::AbstractVector{<:Number}; BigCalc::Bool=false, kwargs...) = ChristoffelSymbol(FisherMetric(DM; kwargs...), θ; BigCalc=BigCalc)
ChristoffelPartials(DM::AbstractDataModel, θ::AbstractVector{<:Number}; BigCalc::Bool=false, kwargs...) = ChristoffelPartials(FisherMetric(DM; kwargs...), θ; BigCalc=BigCalc)
Riemann(DM::AbstractDataModel, θ::AbstractVector{<:Number}; BigCalc::Bool=false, kwargs...) = Riemann(FisherMetric(DM; kwargs...), θ; BigCalc=BigCalc)
Ricci(DM::AbstractDataModel, θ::AbstractVector{<:Number}; BigCalc::Bool=false, kwargs...) = Ricci(FisherMetric(DM; kwargs...), θ; BigCalc=BigCalc)
RicciScalar(DM::AbstractDataModel, θ::AbstractVector{<:Number}; BigCalc::Bool=false, kwargs...) = RicciScalar(FisherMetric(DM; kwargs...), θ; BigCalc=BigCalc)
Weyl(DM::AbstractDataModel, θ::AbstractVector{<:Number}; BigCalc::Bool=false, kwargs...) = Weyl(FisherMetric(DM; kwargs...), θ; BigCalc=BigCalc)

# function MetricPartials(Metric::Function, θ::AbstractVector{<:Number}; BigCalc::Bool=false)
#     BigCalc && (θ = BigFloat.(θ))
#     PDV = zeros(suff(θ), length(θ), length(θ), length(θ));    h = GetH(θ)
#     for i in 1:length(θ)
#         PDV[:,:,i] = (1/(2*h))*(Metric(θ + h*BasisVector(i,length(θ))) - Metric(θ - h*BasisVector(i,length(θ))))
#     end;        PDV
# end

function MetricPartials(Metric::Function, θ::AbstractVector{<:Number}; kwargs...)
    PDV = Array{suff(θ), 3}(undef, length(θ), length(θ), length(θ))
    MetricPartials!(PDV, Metric, θ; kwargs...);   PDV
end
function MetricPartials!(PDV::AbstractArray{<:Number,3}, Metric::Function, θ::AbstractVector{<:Number}; BigCalc::Bool=false)
    BigCalc && (θ = BigFloat.(θ))
    h = GetH(θ)
    for i in 1:length(θ)
        PDV[:,:,i] = (1/(2*h))*(Metric(θ + h*BasisVector(i,length(θ))) - Metric(θ - h*BasisVector(i,length(θ))))
    end
end

# For Float64, use AD to compute partial derivatives of metric, else use finite difference with BigFloat.
function AutoMetricPartials(Metric::Function, θ::AbstractVector{<:Number}; kwargs...)
    PDV = Array{suff(θ)}(undef, length(θ), length(θ), length(θ))
    AutoMetricPartials!(PDV, Metric, θ; kwargs...); PDV
end
function AutoMetricPartials!(PDV::AbstractArray{<:Float64,3}, Metric::Function, θ::AbstractVector{<:Float64}; BigCalc::Bool=false, ADmode::Union{Val,Symbol}=:ForwardDiff)
    J = GetJac(ADmode,Metric)(θ)
    for i in 1:length(θ)
        PDV[:,:,i] = reshape(J[:,i], (length(θ),length(θ)))
    end
end
function AutoMetricPartials!(Metric::Function, θ::AbstractVector{<:Number}; ADmode::Union{Val,Symbol}=:ForwardDiff, kwargs...)
    PDV = Array{Float64}(undef, length(θ), length(θ), length(θ))
    MetricPartials!(PDV, Metric, θ; kwargs...); PDV
end

# function ChristoffelSymbol(Metric::Function, θ::AbstractVector{<:Number}; BigCalc::Bool=false)
#     Finv = inv(Metric(θ));    FPDV = MetricPartials(Metric, θ; BigCalc=BigCalc)
#     if (suff(θ) == Float64) && BigCalc     FPDV = convert(Array{Float64,3}, FPDV)    end
#     @tensor Christoffels[a,i,j] := ((1/2) * Finv)[a,m] * (FPDV[j,m,i] + FPDV[m,i,j] - FPDV[i,j,m])
# end

# Accuracy ≈ 3e-11
# BigCalc for using BigFloat Calculation in finite differencing step but outputting Float64 again.
"""
    ChristoffelSymbol(DM::DataModel, θ::AbstractVector; BigCalc::Bool=false)
    ChristoffelSymbol(Metric::Function, θ::AbstractVector; BigCalc::Bool=false)
Calculates the components of the ``(1,2)`` Christoffel symbol ``\\Gamma`` at a point ``\\theta`` (i.e. the Christoffel symbol "of the second kind") through finite differencing of the `Metric`. Accurate to ≈ 3e-11.
`BigCalc=true` increases accuracy through `BigFloat` calculation.
"""
function ChristoffelSymbol(Metric::Function, θ::AbstractVector{<:Number}; BigCalc::Bool=false)
    # Need to initialize with zeros() instead of Array{}() due to tensor macro
    Γ = zeros(suff(θ), length(θ), length(θ), length(θ))
    ChristoffelSymbol!(Γ, Metric, θ; BigCalc=BigCalc);  Γ
end
function ChristoffelSymbol!(Γ::AbstractArray{<:Number,3}, Metric::Function, θ::AbstractVector{<:Number}; BigCalc::Bool=false)
    PDV = Array{suff(θ)}(undef, length(θ), length(θ), length(θ))
    ChristoffelSymbol!!(PDV, Γ, Metric, θ; BigCalc=BigCalc)
end
function ChristoffelSymbol!!(PDV::AbstractArray{<:Number,3}, Γ::AbstractArray{<:Number,3}, Metric::Function, θ::AbstractVector{<:Number}; BigCalc::Bool=false)
    MetricPartials!(PDV, Metric, θ; BigCalc=BigCalc)
    # ((suff(θ) == Float64) && BigCalc) && (FPDV = convert(Array{Float64,3}, FPDV))
    PDV2Christoffel!(Γ, inv(Metric(θ)), PDV)
end

PDV2Christoffel!(Γ::AbstractArray{<:Number,3}, InvMetric::AbstractMatrix{<:Number}, PDV::AbstractArray{<:Number,3}) = @tensor Γ[a,i,j] = ((1/2) * InvMetric)[a,m] * (PDV[j,m,i] + PDV[m,i,j] - PDV[i,j,m])
function PDV2Christoffel(InvMetric::AbstractMatrix{<:Number}, PDV::AbstractArray{<:Number,3})
    n = size(InvMetric)[1];    Γ = zeros(suff(PDV), n, n, n)
    PDV2Christoffel!(Γ, InvMetric, PDV);    Γ
end

ChristoffelTerm(Γ::AbstractArray{<:Number,3}, v::AbstractVector{<:Number}) = @tensor Res[a] := (-1*Γ)[a,b,c] * v[b] * v[c]


# function ChristoffelPartials(Metric::Function, θ::AbstractVector{<:Number}; BigCalc::Bool=false)
#     BigCalc && (θ = BigFloat.(θ))
#     DownUpDownDown = Array{suff(θ)}(undef,length(θ),length(θ),length(θ),length(θ))
#     h = GetH(θ)
#     for i in 1:length(θ)
#         DownUpDownDown[i,:,:,:] .= (ChristoffelSymbol(Metric,θ + h*BasisVector(i,length(θ))) .- ChristoffelSymbol(Metric,θ - h*BasisVector(i,length(θ))))
#     end;        (1/(2*h))*DownUpDownDown
# end

function ChristoffelPartials(Metric::Function, θ::AbstractVector{<:Number}; BigCalc::Bool=false)
    ∂Γ = Array{suff(θ),4}(undef,length(θ),length(θ),length(θ),length(θ))
    ChristoffelPartials!(∂Γ, Metric, θ; BigCalc=BigCalc);   ∂Γ
end

"""
    Returns partial derivatives of Christoffel Symbols Γ with index structure down-up-down-down, i.e. (∂Γ)ₐᵇₑⱼ.
"""
function ChristoffelPartials!(∂Γ::AbstractArray{<:Number,4}, Metric::Function, θ::AbstractVector{<:Number}; BigCalc::Bool=false)
    BigCalc && (θ = BigFloat.(θ))
    h = GetH(θ)
    FPDV = Array{suff(θ),3}(undef, length(θ), length(θ), length(θ))
    Γ₁ = Array{suff(θ),3}(undef, length(θ), length(θ), length(θ));  Γ₂ = Array{suff(θ),3}(undef, length(θ), length(θ), length(θ))
    for i in 1:length(θ)
        # ∂Γ[i,:,:,:] .= (1/(2*h))*(ChristoffelSymbol(Metric,θ + h*BasisVector(i,length(θ))) .- ChristoffelSymbol(Metric,θ - h*BasisVector(i,length(θ))))
        ChristoffelSymbol!!(FPDV, Γ₁, Metric, θ + h*BasisVector(i,length(θ)); BigCalc=BigCalc)
        ChristoffelSymbol!!(FPDV, Γ₂, Metric, θ - h*BasisVector(i,length(θ)); BigCalc=BigCalc)
        ∂Γ[i,:,:,:] = (1/(2*h))*(Γ₁ - Γ₂)
    end
end

"""
    Riemann(DM::DataModel, θ::AbstractVector; BigCalc::Bool=false)
    Riemann(Metric::Function, θ::AbstractVector; BigCalc::Bool=false)
Calculates the components of the ``(1,3)`` Riemann tensor by finite differencing of the `Metric`. `BigCalc=true` increases accuracy through BigFloat calculation.
"""
function Riemann(Metric::Function, θ::AbstractVector{<:Number}; BigCalc::Bool=false)
    ∂Γ = ChristoffelPartials(Metric, θ; BigCalc=BigCalc)
    if (suff(θ) == Float64) && BigCalc
        ∂Γ = convert(Array{Float64,4}, ∂Γ)
    end
    Γ = ChristoffelSymbol(Metric, θ; BigCalc=BigCalc)
    # @tensor Riem[m,i,k,p] := DownUpDownDown[k,m,i,p] - DownUpDownDown[p,m,i,k] + Gamma[a,i,p]*Gamma[m,a,k] - Gamma[a,i,k]*Gamma[m,a,p]
    @tensor Riem[i,j,k,l] := ∂Γ[k,i,j,l] - ∂Γ[l,i,j,k] + Γ[i,a,k]*Γ[a,j,l] - Γ[i,a,l]*Γ[a,j,k]
end

"""
    Ricci(DM::DataModel, θ::AbstractVector; BigCalc::Bool=false)
    Ricci(Metric::Function, θ::AbstractVector; BigCalc::Bool=false)
Calculates the components of the ``(0,2)`` Ricci tensor by finite differencing of the `Metric`. `BigCalc=true` increases accuracy through `BigFloat` calculation.
"""
function Ricci(Metric::Function, θ::AbstractVector{<:Number}; BigCalc::Bool=false)
    Riem = Riemann(Metric, θ; BigCalc=BigCalc)
    # For some reason, it is necessary to prefill here.
    RIC = zeros(suff(θ), length(θ), length(θ))
    @tensor RIC[a,b] = Riem[c,a,c,b]
end

"""
    RicciScalar(DM::DataModel, θ::AbstractVector; BigCalc::Bool=false) -> Real
    RicciScalar(Metric::Function, θ::AbstractVector; BigCalc::Bool=false) -> Real
Calculates the Ricci scalar by finite differencing of the `Metric`. `BigCalc=true` increases accuracy through `BigFloat` calculation.
"""
RicciScalar(Metric::Function, θ::AbstractVector{<:Number}; BigCalc::Bool=false) = tr(transpose(Ricci(Metric, θ; BigCalc=BigCalc)) * inv(Metric(θ)))


"""
(0,4) Weyl curvature tensor. NEEDS TESTING.
"""
function Weyl(Metric::Function, θ::AbstractVector{<:Number}; BigCalc::Bool=false)
    length(θ) < 4 && return zeros(length(θ),length(θ),length(θ),length(θ))
    Riem = Riemann(Metric,θ; BigCalc=BigCalc)
    g = BigCalc ? Metric(BigFloat.(θ)) : Metric(θ)
    @tensor Ric[a,b] := Riem[m,a,m,b]
    @tensor PartA[i,k,l,m] := Ric[i,m]*g[k,l] - Ric[i,l] * g[k,m] + Ric[k,l] * g[i,m] - Ric[k,m] * g[i,l]
    @tensor PartB[i,k,l,m] := g[i,l] * g[k,m] - g[i,m] * g[k,l]
    @tensor Rlow[a,b,c,d] := g[a,e] * Riem[e,b,c,d]
    Rlow .+ (length(θ) - 2)^(-1) .* PartA .+ ((length(θ) - 1)^(-1) * (length(θ) - 2)^(-1) * tr(inv(g) * Ric)) .* PartB
end
