

# Todo:
#
# Abstract integrands and allow for other f-Divergences to work with the implemented methods
# Create more elaborate method for default domain of integration: Check that both means are inside cube and then choose widths of HyperCube proportional to Σ?
# Add other distributions with known analytical expressions for KL-divergence: Wishart, Beta, Gompertz, generalized Gamma

"""
    KullbackLeibler(p::Function,q::Function,Domain::HyperCube=HyperCube([-15,15]); tol=2e-15, N::Int=Int(3e7), Carlo::Bool=(length(Domain)!=1))
Computes the Kullback-Leibler divergence between two probability distributions `p` and `q` over the `Domain`.
If `Carlo=true`, this is done using a Monte Carlo Simulation with `N` samples.
If the `Domain` is one-dimensional, the calculation is performed without Monte Carlo to a tolerance of ≈ `tol`.
```math
D_{\\text{KL}}[p,q] \\coloneqq \\int \\mathrm{d}^m y \\, p(y) \\, \\mathrm{ln} \\bigg( \\frac{p(y)}{q(y)} \\bigg)
```
"""
function KullbackLeibler(p::Function, q::Function, Domain::HyperCube; tol::Real=1e-9, N::Int=Int(3e7), Carlo::Bool=false)
    function Integrand(x)
        P = p(x)[1];   Q = q(x)[1];   Rat = P / Q
        (Rat ≤ 0. || !isfinite(Rat)) && throw(ArgumentError("Ratio p(x)/q(x) = $Rat in log(p/q) for x=$x."))
        P * log(Rat)
    end
    if !Carlo
        return IntegrateND(Integrand, Domain; tol=tol)
    else
        return length(Domain) == 1 ? MonteCarloArea(x->Integrand(x[1]),Domain,N)[1] : MonteCarloArea(Integrand,Domain,N)
    end
end


function KullbackLeibler(p::Product{Continuous}, q::DiagNormal, Domain::HyperCube=HyperCube([[-20,20] for i in 1:length(p)]); tol::Real=1e-12, kwargs...)
    !(length(p) == length(q) == length(Domain)) && throw("KL: Sampling dimension mismatch: dim(p) = $(length(p)), dim(q) = $(length(q)), Domain = $(length(Domain))")
    sum(KullbackLeibler(p.v[i], Normal(q.μ[i],sqrt(q.Σ.diag[i])), HyperCube([Domain.L[i], Domain.U[i]]); tol=tol) for i in 1:length(p))
end
function KullbackLeibler(p::DiagNormal, q::Product{Continuous}, Domain::HyperCube=HyperCube([[-20,20] for i in 1:length(p)]); tol::Real=1e-12, kwargs...)
    !(length(p) == length(q) == length(Domain)) && throw("KL: Sampling dimension mismatch: dim(p) = $(length(p)), dim(q) = $(length(q)), Domain = $(length(Domain))")
    sum(KullbackLeibler(Normal(p.μ[i],sqrt(p.Σ.diag[i])), q.v[i], HyperCube([Domain.L[i], Domain.U[i]]); tol=tol) for i in 1:length(p))
end
function KullbackLeibler(p::Product{Continuous}, q::Product{Continuous}, Domain::HyperCube=HyperCube([[-20,20] for i in 1:length(p)]); tol::Real=1e-12, kwargs...)
    !(length(p) == length(q) == length(Domain)) && throw("KL: Sampling dimension mismatch: dim(p) = $(length(p)), dim(q) = $(length(q)), Domain = $(length(Domain))")
    sum(KullbackLeibler(p.v[i], q.v[i], HyperCube([Domain.L[i], Domain.U[i]]); tol=tol) for i in 1:length(p))
end

function KullbackLeibler(p::Distribution, q::Distribution, Domain::HyperCube=HyperCube([[-20,20] for i in 1:length(p)]); tol::Real=1e-9, N::Int=Int(3e7), Carlo::Bool=false)
    !(length(p) == length(q) == length(Domain)) && throw("KL: Sampling dimension mismatch: dim(p) = $(length(p)), dim(q) = $(length(q)), Domain = $(length(Domain))")
    function Integrand(x)
        P = logpdf(p,x);   Q = logpdf(q,x)
        exp(P) * (P - Q)
    end
    if !Carlo
        return IntegrateND(Integrand, Domain; tol=tol)
    else
        return length(p) == 1 ? MonteCarloArea(x->Integrand(x[1]),Domain,N)[1] : MonteCarloArea(Integrand,Domain,N)
    end
end


# Known analytic expressions
function KullbackLeibler(P::MvNormal, Q::MvNormal, Domain::HyperCube=HyperCube([-Inf,Inf]); kwargs...)
    (1/2) * (logdet(Q.Σ) - logdet(P.Σ) - length(P.μ) + tr(inv(Q.Σ) * P.Σ) + InnerProduct(inv(Q.Σ), Q.μ-P.μ))
end
KullbackLeibler(P::Normal, Q::Normal, Domain::HyperCube=HyperCube([-Inf,Inf]); kwargs...) = log(Q.σ / P.σ) + (1/2) * ((P.σ / Q.σ)^2 + (P.μ - Q.μ)^2 * Q.σ^(-2) -1.)
KullbackLeibler(P::Cauchy, Q::Cauchy, Domain::HyperCube=HyperCube([-Inf,Inf]); kwargs...) = log(((P.σ + Q.σ)^2 + (P.μ - Q.μ)^2) / (4P.σ * Q.σ))

# Note the sign difference between the conventions (1/θ)*exp(-x/θ) and λ*exp(-λx). Distributions.jl uses the former.
KullbackLeibler(P::Exponential, Q::Exponential, Domain::HyperCube=HyperCube([-Inf,Inf]); kwargs...) = log(Q.θ / P.θ) + P.θ / Q.θ - 1.
function KullbackLeibler(P::Weibull, Q::Weibull, Domain::HyperCube=HyperCube([-Inf,Inf]); kwargs...)
    log(P.α / (P.θ^P.α)) - log(Q.α / (Q.θ^Q.α)) + (P.α - Q.α)*(log(P.θ) - Base.MathConstants.γ / P.α) + (P.θ / Q.θ)^Q.α * SpecialFunctions.gamma(1 + Q.α / P.α) - 1.
end
function KullbackLeibler(P::Distributions.Gamma, Q::Distributions.Gamma, Domain::HyperCube=HyperCube([-Inf,Inf]); kwargs...)
    (P.α - Q.α) * digamma(P.α) - loggamma(P.α) + loggamma(Q.α) + Q.α*log(Q.θ / P.θ) + P.α*(P.θ / Q.θ - 1.)
end




# """
#     NormalDist(DM::DataModel,p::Vector) -> Distribution
# Constructs either `Normal` or `MvNormal` type from `Distributions.jl` using data and a parameter configuration.
# This makes the assumption, that the errors associated with the data are normal.
# """
# function NormalDist(DM::DataModel,p::Vector)::Distribution
#     if length(ydata(DM)[1]) == 1
#         length(ydata(DM)) == 1 && return Normal(ydata(DM)[1] .- DM.model(xdata(DM)[1],p),sigma(DM)[1])
#         return MvNormal(ydata(DM) .- map(x->DM.model(x,p),xdata(DM)),diagm(float.(sigma(DM).^2)))
#     else
#         throw("Not programmed yet.")
#     end
# end

# """
#     KullbackLeibler(DM::DataModel,p::Vector,q::Vector)
# Calculates Kullback-Leibler divergence under the assumption of a normal likelihood.
# """
# KullbackLeibler(DM::DataModel,p::AbstractVector,q::AbstractVector) = KullbackLeibler(NormalDist(DM,p),NormalDist(DM,q))
#
# KullbackLeibler(DM::DataModel,p::AbstractVector) = KullbackLeibler(MvNormal(zeros(length(ydata(DM))),inv(InvCov(DM))),NormalDist(DM,p))
