


"""
    ProjectToLevel!(θ::AbstractVector{<:Number}, loglike::Function, C::Number; ADmode::Val=Val(:ForwardDiff), maxiters::Int=50, tol::Real=1e-10)
Project `θ` onto the level set `loglike(θ) = C` using damped gradient steps.
Returns `(θ, residual)`.
"""
function ProjectToLevel!(θ::AbstractVector{<:Number}, loglike::Function, C::Number; ADmode::Val=Val(:ForwardDiff), maxiters::Int=50, tol::Real=1e-10, reg::Real=1e-13)
    g = similar(θ);     GradGetter! = GetGrad!(ADmode, loglike)
    for _ in 1:maxiters
        r = loglike(θ) - C
        abs(r) <= tol && return θ, abs(r)
        GradGetter!(g, θ)
        # Scalar Newton step along the gradient direction
        α = r / (dot(g, g) + reg)
        θ .-= α .* g
    end;    θ, abs(loglike(θ) - C)
end

function RefineInitialCondition(DM::AbstractDataModel, t0::Real, ConfNum::Real=2; Confnum::Real=ConfNum, kwargs...)
    RefineInitialCondition(DM, MLE(DM), t0, ConfNum; Confnum, kwargs...)
end

function RefineInitialCondition(DM::AbstractDataModel, θguess::AbstractVector{<:Real}, t0::Real, ConfNum::Real=2; Confnum::Real=ConfNum, constraint::Function=loglikelihood(DM), 
            dof::Real=DOF(DM), loglikeMLE::Real=LogLikeMLE(DM), objective::Function=only∘Predictor(DM), C::Real=loglikeMLE-0.5*icdfThreshold(dof, Confnum), meth=nothing, kwargs...)
    SolveConstrainedOptimisationProblem(θ->objective(t0, θ), constraint, θguess, C, nothing; kwargs...)
end


### Overload in extension
function SolveConstrainedOptimisationProblem(objective_fixedt0::Function, constraint::Function, θguess::AbstractVector{<:Real}, C::Real, meth; kwargs...)
    throw("Need to load NonlinearSolve.jl first or pass explicit solve method as final positional argument or kwarg `meth`.")
end

function SolveConstrainedOptimisationProblem(objective_fixedt0::Function, constraint::Function, θguess::AbstractVector{<:Real}, C::Real, meth::SciMLBase.AbstractNonlinearAlgorithm;
                            sense::Int=1, reg::Real=1e-14, maxiters::Int=100, tol::Real=1e-10, # objective::Function=(MaximalNumberOfArguments(objective)==1 ? objective : θ->objective(t0, θ)), 
                            ADmode::Val=Val(:ForwardDiff), GradientGetter::Function=DerivableFunctionsBase._GetGrad(ADmode), HessianGetter::Function=DerivableFunctionsBase._GetHess(ADmode),
                            TransformGuess::Bool=true, ProjectFirst::Bool=true, ProjectIters::Int=1, ProjectTol::Real=1e-4, InteriorTol::Real=1e-2, kwargs...)
    @assert abs(sense) == 1
    n = length(θguess);    T = promote_type(eltype(θguess), typeof(C))
    if TransformGuess
        gobj = GradientGetter(objective_fixedt0, θguess)
        gcon = GradientGetter(constraint, θguess)
        Δ = constraint(θguess) - C

        if norm(gcon) <= InteriorTol
            # θguess is near a stationary point of the constraint (e.g. the MLE):
            # quadratic branch-selecting seed, signed by `sense`.
            Hcon = HessianGetter(constraint, θguess)
            Hpos = copy(-Hcon)
            @inbounds for i in 1:n    Hpos[i, i] += reg    end
            d = Hpos \ gobj
            q = dot(gobj, d)

            if isfinite(Δ) && isfinite(q) && Δ > 0 && q > 0
                α0 = sqrt(2 * Δ / q)
                θseed = θguess .+ sense .* α0 .* d
                λseed = -sense * sqrt(q / (2 * Δ))
            else
                # Conservative fallback: a small step along ±∇(objective)
                gnorm = norm(gobj)
                if gnorm > reg
                    θseed = θguess .+ sense .* 0.1 .* (gobj ./ gnorm)
                else
                    θseed = copy(θguess)
                end
                λseed = T(sense)
            end
        else
            # θguess already near the constraint contour: refine in place.
            # Estimate λ from the stationarity condition sense·gobj + λ·gcon = 0
            θseed = copy(θguess)
            λseed = -sense * dot(gcon, gobj) / (dot(gcon, gcon) + reg)
        end
        ProjectFirst && (θseed, _ = ProjectToLevel!(θseed, constraint, C; ADmode, maxiters=ProjectIters, tol=ProjectTol, reg))
    else
        θseed = θguess;     λseed = T(sense)
    end
    x0 = vcat(copy(θseed), λseed)

    function residual!(F, x, p)
        θ = @view x[1:n];      λ = x[end]
        gobj = GradientGetter(objective_fixedt0, θ)
        gcon = GradientGetter(constraint, θ)

        F[1:n] .= sense .* gobj .+ λ .* gcon
        F[n + 1] = constraint(θ) - C
        nothing
    end
    function jacobian!(J, x, p)
        θ = @view x[1:n];      λ = x[end]
        Hobj = HessianGetter(objective_fixedt0, θ)
        Hcon = HessianGetter(constraint, θ)
        gcon = GradientGetter(constraint, θ)

        fill!(J, zero(eltype(J)))
        J[1:n, 1:n] .= sense .* Hobj .+ λ .* Hcon
        @inbounds for i in 1:n    J[i, i] += reg     end
        J[1:n, n + 1] .= gcon
        J[n + 1, 1:n] .= gcon
        nothing
    end
    nlf = NonlinearFunction(residual!; jac = jacobian!)
    prob = NonlinearProblem(nlf, x0, nothing)
    sol = solve(prob, meth; abstol=tol, reltol=tol, maxiters, kwargs...)
    θ0 = copy(sol.u[1:n]);    λ0 = sol.u[end]
    θ0, λ0
end
