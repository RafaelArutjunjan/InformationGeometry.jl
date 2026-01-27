

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
ChristoffelTerm!(Res::AbstractVector{T}, Γ::AbstractArray{<:Number,3}, v::AbstractVector{<:Number}) where T<:Number = (@tullio Res[a] = Γ[a,b,c] * v[b] * v[c];   Res .*= -one(T))


# function ChristoffelPartials(Metric::Function, θ::AbstractVector{<:Number}; BigCalc::Bool=false)
#     BigCalc && (θ = BigFloat.(θ))
#     DownUpDownDown = Array{suff(θ)}(undef,length(θ),length(θ),length(θ),length(θ))
#     h = GetH(θ)
#     for i in eachindex(θ)
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



## For computing extrinsic curvature
"""
    _SecondFundamentalForm(J::AbstractMatrix{T}, H::AbstractArray, P_N::AbstractMatrix)
Computes the second fundamental form where J is the Jacobian and H the Hessian of the embedding map (h : M -> D).
P_N is the projection onto the direction normal to the model manifold in the "tangent space of the data space".
"""
function _SecondFundamentalForm(J::AbstractMatrix{T}, H::AbstractArray, P_N::AbstractMatrix) where T<:Number
    N, k = size(J)
    @boundscheck @assert N == size(H,1) == size(P_N,1) == size(P_N,2)
    @boundscheck @assert k == size(H,2) == size(H,3)
    
    II = Array{Vector{T}}(undef, k, k)
    @inbounds for i in 1:k, j in 1:k
        II[i,j] = P_N * view(H, :, i, j)
    end;    II
end
"""
    SecondFundamentalForm(DM::AbstractDataModel, mle::AbstractVector; ADmode::Val=Val(:ForwardDiff), g::AbstractMatrix=FisherMetric(DM, mle))
Computes the second fundamental form at the given parameters under the assumption of constant known data variance.
Measures how strongly tangent vectors to the embedded model manifold fail to stay tangent under parallel transport with respect to the ambient connection, i.e. extrinsic curvature.
Normal vector valued (0,2) tensor.
"""
function SecondFundamentalForm(DM::AbstractDataModel, mle::AbstractVector; ADmode::Val=Val(:ForwardDiff), EmbeddingFn::Function=p -> EmbeddingMap(DM,p),
                        g::AbstractMatrix=FisherMetric(DM, mle), g⁻¹::AbstractMatrix=inv(g), Σ⁻¹::AbstractMatrix{<:Number}=yInvCov(DM, mle))
    @boundscheck @assert length(mle) == size(g,1) == size(g,2) == size(g⁻¹,1) == size(g⁻¹,2)
    J = GetJac(ADmode, EmbeddingFn)(mle);    H = GetDoubleJac(ADmode, EmbeddingFn)(mle)
    P_N = Eye(size(J,1)) .- J * g⁻¹ * (J' * Σ⁻¹) # Normal projector of model manifold tangent space in ambient space
    @inbounds _SecondFundamentalForm(J, H, P_N)
end

for F in [:EfronScalarCurvature, :EfronMeanCurvature,
    :EfronRicciCurvature, :EfronRicciCurvature2,
    :EfronRiemannCurvature, :EfronRiemannCurvature2,
    :EfronSectionalCurvature, :EfronSectionalCurvatureMap,
    :EfronShapeOperator, :EfronCurvatureIsotropy]
    @eval function $F(DM::AbstractDataModel, mle::AbstractVector; g::AbstractMatrix=FisherMetric(DM, mle), MakePosDef::Bool=false, verbose::Bool=true,
            g⁻¹::AbstractMatrix=try inv(g) catch E; E isa SingularException && MakePosDef ? (verbose && @warn "$($F): Adding 1e-14 to diagonal before since FisherMetric singular.";   inv(Symmetric(g + 1e-10Eye(length(mle))))) : rethrow(E) end, 
            Σ⁻¹::AbstractMatrix{<:Number}=yInvCov(DM, mle), kwargs...)
        @assert !HasEstimatedUncertainties(DM) "Not implemented for parameter-dependent data variance yet."
        $F(SecondFundamentalForm(DM, mle; g, g⁻¹, Σ⁻¹, kwargs...), g, g⁻¹, Σ⁻¹)
    end
    @eval $F(DM::AbstractDataModel; kwargs...) = X::AbstractVector->$F(DM, X; kwargs...)
end
## Gives proportionality to 2nd order MLE risk
function EfronScalarCurvature(II::AbstractMatrix{<:AbstractVector{<:Number}}, g::AbstractMatrix{<:Number}, g⁻¹::AbstractMatrix{<:Number}, Σ⁻¹::AbstractMatrix{<:Number})
    @tullio IInorm = g⁻¹[i,a] * g⁻¹[j,b] * dot(II[i,j], Σ⁻¹, II[a,b])
    γ² = IInorm / 2
end
## Gives direction of mean curvature, i.e. externally induced acceleration H = tr_g(II), no 1/k in definition!
function EfronMeanCurvature(II::AbstractMatrix{<:AbstractVector{T}}, g::AbstractMatrix{<:Number}, g⁻¹::AbstractMatrix{<:Number}, Σ⁻¹::AbstractMatrix{<:Number}=Eye(1)) where T<:Number
    # 1/size(g⁻¹,1) .* sum(g⁻¹[i,j]*II[i,j] for i in axes(g⁻¹,1), j in axes(g⁻¹,2))
    Res = zeros(T,length(II[1]))
    for i in axes(g⁻¹,1), j in axes(g⁻¹,2)
        Res .+= g⁻¹[i,j] .* II[i,j]
    end;    Res
end
## Sectional curvature with respect to coordinate basis
function EfronSectionalCurvature(II::AbstractMatrix{<:AbstractVector{T}}, g::AbstractMatrix{<:Number}, g⁻¹::AbstractMatrix{<:Number}, Σ⁻¹::AbstractMatrix{<:Number}) where T<:Number
    K = fill(T(NaN), size(II,1), size(II,1))
    for a in axes(II,1), b in a+1:size(II,1)
        num = dot(II[a,a], Σ⁻¹, II[b,b]) - dot(II[a,b], Σ⁻¹, II[a,b])
        denom = g[a,a] * g[b,b] - g[a,b]^2
        K[a,b] = K[b,a] = num / denom
    end;    K
end
## Sectional curvature with respect to arbitrary given intrinsic tangent vectors
function EfronSectionalCurvatureMap(II::AbstractMatrix{<:AbstractVector{T}}, g::AbstractMatrix{<:Number}, g⁻¹::AbstractMatrix{<:Number}, Σ⁻¹::AbstractMatrix{<:Number}) where T<:Number
    ## Precompute Q
    k = size(II,1);    Q = zeros(T, k, k, k, k)
    for i in 1:k, j in i:k, a in 1:k, b in a:k
        val = dot(II[i,j], Σ⁻¹, II[a,b])
        Q[i,j,a,b] = Q[j,i,a,b] = Q[i,j,b,a] = Q[j,i,b,a] = Q[a,b,i,j] = Q[b,a,i,j] = Q[a,b,j,i] = Q[b,a,j,i] = val
    end
    function ExtrinsicSectionalCurvatureTensor(u::AbstractVector{S}, v::AbstractVector{S}) where S<:Number
        @boundscheck @assert length(u) == length(v) == k
        num1 = zero(S);    num2 = zero(S)
        @inbounds for i in 1:k, j in 1:k, a in 1:k, b in 1:k
            num1 += u[i]*u[j]*v[a]*v[b] * Q[i,j,a,b]
            num2 += u[i]*v[j]*u[a]*v[b] * Q[i,j,a,b]
        end
        uu = dot(u, g, u);    vv = dot(v, g, v);    uv = dot(u, g, v)
        (num1 - num2) / (uu*vv - uv^2)
    end
end

function EfronRicciCurvature(II::AbstractMatrix{<:AbstractVector{<:Number}}, g::AbstractMatrix{<:Number}, g⁻¹::AbstractMatrix{<:Number}, Σ⁻¹::AbstractMatrix{<:Number})
    @tullio Ric[i,j] := g⁻¹[a,b] * dot(II[i,j], Σ⁻¹, II[a,b])
end
function EfronRiemannCurvature(II::AbstractMatrix{<:AbstractVector{<:Number}}, g::AbstractMatrix{<:Number}, g⁻¹::AbstractMatrix{<:Number}, Σ⁻¹::AbstractMatrix{<:Number})
    @tullio Riem[i,j,k,l] := dot(II[i,k], Σ⁻¹, II[j,l]) - dot(II[i,l], Σ⁻¹, II[j,k])
end
## Pre-whitening of II, more performant for large Σ⁻¹ with non-zero off-diagonal
function EfronRicciCurvature2(II::AbstractMatrix{<:AbstractVector{<:Number}}, g::AbstractMatrix{<:Number}, g⁻¹::AbstractMatrix{<:Number}, Σ⁻¹::AbstractMatrix{<:Number})
    L = cholesky(Σ⁻¹).L;    IIw = [L * II[i,j] for i in axes(II,1), j in axes(II,2)]
    @tullio Ric[i,j] := g⁻¹[a,b] * dot(IIw[i,a], IIw[j,b])
end
## Pre-whitening of II, more performant for large Σ⁻¹ with non-zero off-diagonal
function EfronRiemannCurvature2(II::AbstractMatrix{<:AbstractVector{<:Number}}, g::AbstractMatrix{<:Number}, g⁻¹::AbstractMatrix{<:Number}, Σ⁻¹::AbstractMatrix{<:Number})
    L = cholesky(Σ⁻¹).L;    IIw = [L * II[i,j] for i in axes(II,1), j in axes(II,2)]
    @tullio Riem[i,j,k,l] := dot(IIw[i,k], IIw[j,l]) - dot(IIw[i,l], IIw[j,k])
end

# Returns shape operator matrix (S_n)^i_j for a given normal vector n
function EfronShapeOperator(II::AbstractMatrix{<:AbstractVector{<:Number}}, g::AbstractMatrix{<:Number}, g⁻¹::AbstractMatrix{<:Number}, Σ⁻¹::AbstractMatrix{<:Number})
    k = size(II,1)
    function ShapeOperator(n::AbstractVector{T}) where T<:Number
        @boundscheck @assert length(n) == length(II[1])
        S = zeros(T, k, k)
        @inbounds for i in 1:k, j in 1:k
            for a in 1:k
                S[i,j] += g⁻¹[i,a] * dot(II[a,j], Σ⁻¹, n)
            end
        end;    S
    end
end

## norm(H)^2 / norm(II)^2
## If -> 1, then curvature mostly mean, if -> 0 then curvature anisotropic, i.e. due to few distinct parameter directions. 
function EfronCurvatureIsotropy(II::AbstractMatrix{<:AbstractVector{<:Number}}, g::AbstractMatrix{<:Number}, g⁻¹::AbstractMatrix{<:Number}, Σ⁻¹::AbstractMatrix{<:Number})
    H = EfronMeanCurvature(II, g, g⁻¹, Σ⁻¹)
    Hnorm² = InnerProduct(Σ⁻¹, H)
    IInorm² = 2*EfronScalarCurvature(II, g, g⁻¹, Σ⁻¹)
    Hnorm² / IInorm²
end



CencovTensor(DM::AbstractDataModel; kwargs...) = X::AbstractVector -> CencovTensor(DM, X; kwargs...)
"""
    CencovTensor(DM::AbstractDataModel, mle::AbstractVector; ADmode::Val=Val(:ForwardDiff), kwargs...)
Computes the (0,3) Amari-Čencov tensor ``C_{ijk}(\\theta) = \\mathrm{E}(\\pdv{\\ell}{\\theta^i} \\pdv{\\ell}{\\theta^j} \\pdv{\\ell}{\\theta^k})`` for data with Gaussian noise of known or unknown variance.
"""
function CencovTensor(DM::AbstractDataModel, mle::AbstractVector; ADmode::Val=Val(:ForwardDiff), EmbeddingFn::Function=p -> EmbeddingMap(DM,p), Σ⁻¹::AbstractMatrix{<:Number}=yInvCov(DM, mle), 
                        ∂Σ⁻¹::Union{Nothing,AbstractArray{<:Number,3}}=HasEstimatedUncertainties(DM) ? GetMatrixJac(ADmode,p->yInvCov(DM,p))(mle) : nothing, kwargs...)
    J = GetJac(ADmode, EmbeddingFn)(mle);    H = GetDoubleJac(ADmode, EmbeddingFn)(mle)
    _CencovTensor(J, H, Σ⁻¹, ∂Σ⁻¹)
end
# No prescaled Jacobian, apparently slightly slower
function _CencovTensor2(J::AbstractMatrix, H::AbstractArray{<:Number,3}, Σ⁻¹::AbstractMatrix, N::Nothing=nothing)
    @boundscheck @assert size(J,1) == size(H,1) == size(Σ⁻¹,1) == size(Σ⁻¹,2) && size(J,2) == size(H,2) == size(H,3)
    # C = 1/3 * Symmetric permutation of H[a,i,j] * J̃[a, k]
    @tullio C[i,j,k] := (H[a,i,j] * Σ⁻¹[a,b] * J[b,k] + H[a,i,k] * Σ⁻¹[a,b] * J[b,j] + H[a,j,k] * Σ⁻¹[a,b] * J[b,i]) / 3
end
function _CencovTensor(J::AbstractMatrix, H::AbstractArray{<:Number,3}, Σ⁻¹::AbstractMatrix, N::Nothing=nothing)
    @boundscheck @assert size(J,1) == size(H,1) == size(Σ⁻¹,1) == size(Σ⁻¹,2) && size(J,2) == size(H,2) == size(H,3)
    # Prescale Jacobian
    J̃ = Σ⁻¹ * J
    # C = 1/3 * Symmetric permutation of H[a,i,j] * J̃[a, k]
    @tullio C[i,j,k] := (H[a,i,j] * J̃[a, k] + H[a,j,k] * J̃[a, i] + H[a,k,i] * J̃[a, j]) / 3
end
function _CencovTensor(J::AbstractMatrix, H::AbstractArray{<:Number,3}, Σ⁻¹::AbstractMatrix, ∂Σ⁻¹::AbstractArray{<:Number,3})
    @boundscheck @assert size(J,1) == size(H,1) == size(Σ⁻¹,1) == size(Σ⁻¹,2) == size(∂Σ⁻¹,1) == size(∂Σ⁻¹,2)
    @boundscheck @assert size(J,2) == size(H,2) == size(H,3) == size(∂Σ⁻¹,3)
    # Parameter-dependent mean part
    @tullio T[i,j,k] := H[a,i,j] * Σ⁻¹[a,b] * J[b,k]
    # Mean-covariance interaction
    @tullio T[i,j,k] += -0.5* J[a,k] * ∂Σ⁻¹[a,b,i] * J[b,j]
    # Pure covariance contribution
    @tullio T[i,j,k] += -0.25 * ∂Σ⁻¹[a,b,i] * ∂Σ⁻¹[b,c,j] * ∂Σ⁻¹[c,a,k]
    # Symmetrisation
    @tullio C[i,j,k] := (T[i,j,k] + T[j,k,i] + T[k,i,j]) / 3
end


"""
    EChristoffelSymbol(DM::AbstractDataModel, MLE::AbstractVector=MLE(DM); kwargs...)
Computes the (1,2) Christoffel symbol of the e-connection, i.e. the α-connection with α=-1.
"""
function EChristoffelSymbol(DM::AbstractDataModel, MLE::AbstractVector=MLE(DM); g::AbstractMatrix=FisherMetric(DM, MLE), g⁻¹::AbstractMatrix{<:Number}=inv(g), kwargs...)
    Γ = ChristoffelSymbol(DM, MLE; kwargs...)
    C = CencovTensor(DM, MLE; kwargs...)
    @tullio Γ[i,j,k] += -0.5 * g⁻¹[i,a] * C[a,j,k]
    Γ
end
"""
    MChristoffelSymbol(DM::AbstractDataModel, MLE::AbstractVector=MLE(DM); kwargs...)
Computes the (1,2) Christoffel symbol of the m-connection, i.e. the α-connection with α=+1.
"""
function MChristoffelSymbol(DM::AbstractDataModel, MLE::AbstractVector=MLE(DM); g::AbstractMatrix=FisherMetric(DM, MLE), g⁻¹::AbstractMatrix{<:Number}=inv(g), kwargs...)
    Γ = ChristoffelSymbol(DM, MLE; kwargs...)
    C = CencovTensor(DM, MLE; kwargs...)
    @tullio Γ[i,j,k] += +0.5 * g⁻¹[i,a] * C[a,j,k]
    Γ
end