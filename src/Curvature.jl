
MetricPartials(DM::AbstractDataModel, θ::AbstractVector{<:Number}; BigCalc::Bool=false) = MetricPartials(z->FisherMetric(DM,z), θ; BigCalc=BigCalc)
function MetricPartials(Metric::Function, θ::AbstractVector{<:Number}; BigCalc::Bool=false)
    if BigCalc      θ = BigFloat.(θ)        end
    PDV = zeros(suff(θ), length(θ), length(θ), length(θ));    h = GetH(θ)
    for i in 1:length(θ)
        PDV[:,:,i] .= (1/(2*h)).*(Metric(θ .+ h.*BasisVector(i,length(θ))) .- Metric(θ .- h.*BasisVector(i,length(θ))))
    end;        PDV
end

MetricAutoPartials(DM::AbstractDataModel, θ::AbstractVector{<:Number}; BigCalc::Bool=false) = MetricAutoPartials(p->FisherMetric(DM,p), θ)
function MetricAutoPartials(Metric::Function, θ::AbstractVector{<:Number}; BigCalc::Bool=false)
    J = ForwardDiff.jacobian(Metric, θ)
    Res = Array{suff(J)}(undef, length(θ), length(θ), length(θ))
    for i in 1:length(θ)
        Res[:,:,i] = reshape(J[:,i], (length(θ),length(θ)))
    end;    Res
end

# Accuracy ≈ 3e-11
# BigCalc for using BigFloat Calculation in finite differencing step but outputting Float64 again.
"""
    ChristoffelSymbol(DM::DataModel, θ::AbstractVector; BigCalc::Bool=false)
    ChristoffelSymbol(Metric::Function, θ::AbstractVector; BigCalc::Bool=false)
Calculates the components of the ``(1,2)`` Christoffel symbol ``\\Gamma`` at a point ``\\theta`` (i.e. the Christoffel symbol "of the second kind") through finite differencing of the `Metric`. Accurate to ≈ 3e-11.
`BigCalc=true` increases accuracy through `BigFloat` calculation.
"""
ChristoffelSymbol(DM::AbstractDataModel, θ::AbstractVector{<:Number}; BigCalc::Bool=false) = ChristoffelSymbol(z->FisherMetric(DM,z), θ; BigCalc=BigCalc)
function ChristoffelSymbol(Metric::Function, θ::AbstractVector{<:Number}; BigCalc::Bool=false)
    Finv = inv(Metric(θ));    FPDV = MetricAutoPartials(Metric, θ; BigCalc=BigCalc)
    if (suff(θ) != BigFloat) && BigCalc     FPDV = convert(Array{Float64,3}, FPDV)    end
    @tensor Christoffels[a,i,j] := ((1/2) * Finv)[a,m] * (FPDV[j,m,i] + FPDV[m,i,j] - FPDV[i,j,m])
end

function ChristoffelTerm(ConnectionCoeff::AbstractArray{<:Real,3}, v::AbstractVector{<:Real})
    (Tuple(length(v) .* ones(Int,3)) != size(ConnectionCoeff)) && throw(ArgumentError("Connectioncoefficients don't match vector: dim(v) = $(length(v)), size(Connection) = $(size(ConnectionCoeff))"))
    @tensor Res[a] := (-1*ConnectionCoeff)[a,b,c] * v[b] * v[c]
end


ChristoffelPartials(DM::AbstractDataModel, θ::AbstractVector{<:Number}; BigCalc::Bool=false) = ChristoffelPartials(z->FisherMetric(DM,z), θ; BigCalc=BigCalc)
function ChristoffelPartials(Metric::Function, θ::AbstractVector{<:Number}; BigCalc::Bool=false)
    if BigCalc      θ = BigFloat.(θ)        end
    DownUpDownDown = Array{suff(θ)}(undef,length(θ),length(θ),length(θ),length(θ))
    h = GetH(θ)
    for i in 1:length(θ)
        DownUpDownDown[i,:,:,:] .= (ChristoffelSymbol(Metric,θ + h*BasisVector(i,length(θ))) .- ChristoffelSymbol(Metric,θ - h*BasisVector(i,length(θ))))
    end;        (1/(2*h))*DownUpDownDown
end

"""
    Riemann(DM::DataModel, θ::AbstractVector; BigCalc::Bool=false)
    Riemann(Metric::Function, θ::AbstractVector; BigCalc::Bool=false)
Calculates the components of the ``(1,3)`` Riemann tensor by finite differencing of the `Metric`. `BigCalc=true` increases accuracy through BigFloat calculation.
"""
Riemann(DM::AbstractDataModel, θ::AbstractVector{<:Number}; BigCalc::Bool=false) = Riemann(z->FisherMetric(DM,z), θ; BigCalc=BigCalc)
function Riemann(Metric::Function, θ::AbstractVector{<:Number}; BigCalc::Bool=false)
    DownUpDownDown = ChristoffelPartials(Metric, θ; BigCalc=BigCalc)
    if (suff(θ) != BigFloat) && BigCalc
        DownUpDownDown = convert(Array{Float64,4},DownUpDownDown)
    end
    Gamma = ChristoffelSymbol(Metric, θ; BigCalc=BigCalc)
    # @tensor Riem[m,i,k,p] := DownUpDownDown[k,m,i,p] - DownUpDownDown[p,m,i,k] + Gamma[a,i,p]*Gamma[m,a,k] - Gamma[a,i,k]*Gamma[m,a,p]
    @tensor Riem[i,j,k,l] := DownUpDownDown[k,i,j,l] - DownUpDownDown[l,i,j,k] + Gamma[i,a,k]*Gamma[a,j,l] - Gamma[i,a,l]*Gamma[a,j,k]
end

"""
    Ricci(DM::DataModel, θ::AbstractVector; BigCalc::Bool=false)
    Ricci(Metric::Function, θ::AbstractVector; BigCalc::Bool=false)
Calculates the components of the ``(0,2)`` Ricci tensor by finite differencing of the `Metric`. `BigCalc=true` increases accuracy through `BigFloat` calculation.
"""
Ricci(DM::AbstractDataModel, θ::AbstractVector{<:Number}; BigCalc::Bool=false) = Ricci(z->FisherMetric(DM,z), θ; BigCalc=BigCalc)
function Ricci(Metric::Function, θ::AbstractVector{<:Number}; BigCalc::Bool=false)
    Riem = Riemann(Metric, θ; BigCalc=BigCalc)
    # For some reason, it is necessary to prefill here.
    RIC = zeros(suff(θ),length(θ),length(θ))
    @tensor RIC[a,b] = Riem[c,a,c,b]
end

"""
    RicciScalar(DM::DataModel, θ::AbstractVector; BigCalc::Bool=false) -> Real
    RicciScalar(Metric::Function, θ::AbstractVector; BigCalc::Bool=false) -<> Real
Calculates the Ricci scalar by finite differencing of the `Metric`. `BigCalc=true` increases accuracy through `BigFloat` calculation.
"""
RicciScalar(DM::AbstractDataModel, θ::AbstractVector{<:Number}; BigCalc::Bool=false) = RicciScalar(z->FisherMetric(DM,z),θ; BigCalc=BigCalc)
RicciScalar(Metric::Function, θ::AbstractVector{<:Number}; BigCalc::Bool=false) = tr(transpose(Ricci(Metric, θ; BigCalc=BigCalc)) * inv(Metric(θ)))


"""
(0,4) Weyl curvature tensor. NEEDS TESTING.
"""
Weyl(DM::AbstractDataModel,θ::AbstractVector{<:Number}; BigCalc::Bool=false) = Weyl(z->FisherMetric(DM,z),θ; BigCalc=BigCalc)
function Weyl(Metric::Function,θ::AbstractVector{<:Number}; BigCalc::Bool=false)
    length(θ) < 4 && return zeros(length(θ),length(θ),length(θ),length(θ))
    Riem = Riemann(Metric,θ; BigCalc=BigCalc)
    g = BigCalc ? Metric(BigFloat.(θ)) : Metric(θ)
    @tensor Ric[a,b] := Riem[m,a,m,b]
    @tensor PartA[i,k,l,m] := Ric[i,m]*g[k,l] - Ric[i,l] * g[k,m] + Ric[k,l] * g[i,m] - Ric[k,m] * g[i,l]
    @tensor PartB[i,k,l,m] := g[i,l] * g[k,m] - g[i,m] * g[k,l]
    @tensor Rlow[a,b,c,d] := g[a,e] * Riem[e,b,c,d]
    Rlow .+ (length(θ) - 2)^(-1) .* PartA .+ ((length(θ) - 1)^(-1) * (length(θ) - 2)^(-1) * tr(inv(g) * Ric)) .* PartB
end
