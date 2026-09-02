


"""
    ProjectToLevel!(θ::AbstractVector{<:Number}, loglike::Function, C::Number; ADmode::Val=Val(:ForwardDiff), maxiters::Int=50, tol::Real=1e-10)
Projects `θ` onto the level set `loglike(θ) = C` using damped gradient steps and returns `(θ, residual)`.
"""
function ProjectToLevel!(θ::AbstractVector{<:Number}, loglike::Function, C::Number; ADmode::Val=Val(:ForwardDiff), Gradient!::Function=GetGrad!(ADmode, loglike), maxiters::Int=50, tol::Real=1e-10, reg::Real=1e-13)
    g = similar(θ)
    for _ in 1:maxiters
        r = loglike(θ) - C
        abs(r) <= tol && return θ, abs(r)
        Gradient!(g, θ)
        # Scalar Newton step along the gradient direction
        α = r / (dot(g, g) + reg)
        θ .-= α .* g
    end;    θ, abs(loglike(θ) - C)
end


## Computes extremal parameter configuration on confidence boundary, which produces the most extreme prediction at a fixed confidence level
function RefineInitialCondition(DM::AbstractDataModel, t0::Real, ConfNum::Real=2; Confnum::Real=ConfNum, kwargs...)
    RefineInitialCondition(DM, MLE(DM), t0, ConfNum; Confnum, kwargs...)
end

function RefineInitialCondition(DM::AbstractDataModel, θguess::AbstractVector{<:Number}, t0::Real, ConfNum::Real=2; Confnum::Real=ConfNum, constraint::Function=loglikelihood(DM), 
            dof::Real=DOF(DM), loglikeMLE::Real=LogLikeMLE(DM), objective::Function=only∘Predictor(DM), C::Real=loglikeMLE-0.5*icdfThreshold(dof, Confnum), meth=nothing, kwargs...)
    SolveConstrainedOptimisationProblem(θ->objective(t0, θ), constraint, θguess, C, nothing; kwargs...)
end



## Consolidate function with version that does not take C anymore but already absorbed into constraint
function SolveConstrainedOptimisationProblem(objective_fixedt0::Function, constraint::Function, θguess::AbstractVector{<:Number}, C::Real, meth; kwargs...)
    ZerodConstraint(θ) = constraint(θ) - C
    SolveConstrainedOptimisationProblem(objective_fixedt0, ZerodConstraint, θguess, meth; kwargs...)
end


### Overload in extension
function SolveConstrainedOptimisationProblem(objective_fixedt0::Function, constraint::Function, θguess::AbstractVector{<:Number}, meth; kwargs...)
    throw("Need to load NonlinearSolve.jl first or pass explicit solve method as final positional argument or kwarg `meth`.")
end

function _ConstrainedOptimisationInitialGuess(ZerodConstraint::Function, θguess::AbstractVector{T}, gobjcache::Union{AbstractArray,DiffCache}, gconcache::Union{AbstractVector,DiffCache}, Hconcache::Union{AbstractMatrix,DiffCache};
                        sense::Int=1, reg::Real=1e-14, ADmode::Val=Val(:ForwardDiff), ObjectiveGradient!::Function, ConstraintGradient!::Function, ConstraintHessian!::Function,
                        ProjectFirst::Bool=true, ProjectIters::Int=1, ProjectTol::Real=1e-4, InteriorTol::Real=1e-2) where T<:Number
    n = length(θguess)
    gobj = UnrollCache(gobjcache, θguess);    gcon = UnrollCache(gconcache, θguess)
    ObjectiveGradient!(gobj, θguess);    ConstraintGradient!(gcon, θguess)
    Δ = ZerodConstraint(θguess)

    if norm(gcon) <= InteriorTol
        # θguess is near a stationary point of the constraint (e.g. the MLE):
        # quadratic branch-selecting seed, signed by `sense`.
        Hcon = UnrollCache(Hconcache, θguess);    ConstraintHessian!(Hcon, θguess)
        Hcon .*= -1
        @inbounds for i in 1:n    Hcon[i, i] += reg    end
        d = Hcon \ gobj
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
    ProjectFirst && (θseed, _ = ProjectToLevel!(θseed, ZerodConstraint, 0; ADmode, Gradient! =ConstraintGradient!, maxiters=ProjectIters, tol=ProjectTol, reg))
    θseed, λseed
end

"""
    SolveConstrainedOptimisationProblem(objective_fixedt0::Function, ZerodConstraint::Function, θguess::AbstractVector{<:Real}, meth::SciMLBase.AbstractNonlinearAlgorithm;
                            sense::Int=1, reg::Real=1e-14, maxiters::Int=100, tol::Real=1e-10, ADmode::Val=Val(:ForwardDiff), 
                            TransformGuess::Bool=true, ProjectFirst::Bool=true, ProjectIters::Int=1, ProjectTol::Real=1e-4, InteriorTol::Real=1e-2, kwargs...)
Solves constrained optimisation problem for objective `objective_fixedt0(θ)` under the constraint that `ZerodConstraint(θ) == 0`.
For `sense == +1`, the given objective is maximized, for `sense == -1`, the objective is minimized.
If the initial guess is already known to be reasonably accurate, this projection step can be avoided by setting `TransformGuess = false`.
"""
function SolveConstrainedOptimisationProblem(objective_fixedt0::Function, ZerodConstraint::Function, θguess::AbstractVector{T}, meth::SciMLBase.AbstractNonlinearAlgorithm;
                            sense::Int=1, reg::Real=1e-14, maxiters::Int=100, tol::Real=1e-10, Full::Bool=false, ADmode::Val=Val(:ForwardDiff), levels::Int=1, 
                            ObjectiveGradient!::Function=GetGrad!(ADmode, objective_fixedt0), ObjectiveHessian!::Function=GetHess!(ADmode, objective_fixedt0),
                            ConstraintGradient!::Function=GetGrad!(ADmode, ZerodConstraint), ConstraintHessian!::Function=GetHess!(ADmode, ZerodConstraint), 
                            TransformGuess::Bool=false, ProjectFirst::Bool=true, ProjectIters::Int=1, ProjectTol::Real=1e-4, InteriorTol::Real=1e-2,
                            lower::AbstractVector=Fill(-Inf, length(θguess)), upper::AbstractVector=Fill(Inf, length(θguess)), kwargs...) where T<:Number
    @assert abs(sense) == 1;    n = length(θguess)
    @assert length(lower) == n == length(upper)
    ## Use DiffCache and make derivative getters inplace? Use Score and CostHessian
    gobj = DiffCache(copy(θguess); levels);    gcon = DiffCache(copy(θguess); levels)
    Hobj = DiffCache(rand(eltype(θguess), length(θguess), length(θguess)); levels);    Hcon = DiffCache(rand(eltype(θguess), length(θguess), length(θguess)); levels)
    θseed, λseed = if TransformGuess
        _ConstrainedOptimisationInitialGuess(ZerodConstraint, θguess, gobj, gcon, Hcon; sense, reg, ADmode, ObjectiveGradient!, ConstraintGradient!, ConstraintHessian!, ProjectFirst, ProjectIters, ProjectTol, InteriorTol)
    else
        copy(θguess), T(sense)
    end
    x0 = vcat(θseed, λseed)

    function residual!(F, x, p)
        θ = @view x[1:n];      λ = x[end]
        gobj = UnrollCache(gobj, x)
        gcon = UnrollCache(gcon, x)
        ObjectiveGradient!(gobj, θ)
        ConstraintGradient!(gcon, θ)

        F[1:n] .= sense .* gobj .+ λ .* gcon
        F[n + 1] = ZerodConstraint(θ)
        nothing
    end
    function jacobian!(J, x, p)
        θ = @view x[1:n];      λ = x[end]
        gcon = UnrollCache(gcon, x)
        Hobj = UnrollCache(Hobj, x)
        Hcon = UnrollCache(Hcon, x)
        ConstraintGradient!(gcon, θ)
        ObjectiveHessian!(Hobj, θ)
        ConstraintHessian!(Hcon, θ)

        J .= 0
        J[1:n, 1:n] .= sense .* Hobj .+ λ .* Hcon
        @inbounds for i in 1:n    J[i, i] += reg     end
        J[1:n, n + 1] .= gcon
        J[n + 1, 1:n] .= gcon
        nothing
    end
    nlf = NonlinearFunction(residual!; jac = jacobian!)
    prob = NonlinearProblem(nlf, x0, nothing)
    sol = solve(prob, meth; abstol=tol, reltol=tol, maxiters, kwargs...)
    Full && return sol
    θ0 = sol.u[1:n];    λ0 = sol.u[end];    (θ0, λ0)
end

function SolveConstrainedOptimisationProblem(objective_fixedt0::Function, ZerodConstraint::Function, θguess::AbstractVector{T}, meth::Optim.AbstractConstrainedOptimizer;
                            sense::Int=1, reg::Real=1e-14, maxiters::Int=100, tol::Real=1e-10, Full::Bool=false, ADmode::Val=Val(:ForwardDiff), levels::Int=1, 
                            ObjectiveGradient!::Function=GetGrad!(ADmode, objective_fixedt0), ObjectiveHessian!::Function=GetHess!(ADmode, objective_fixedt0),
                            ConstraintGradient!::Function=GetGrad!(ADmode, ZerodConstraint), ConstraintHessian!::Function=GetHess!(ADmode, ZerodConstraint), 
                            TransformGuess::Bool=false, ProjectFirst::Bool=true, ProjectIters::Int=1, ProjectTol::Real=1e-4, InteriorTol::Real=1e-2, 
                            lower::AbstractVector=Fill(-Inf, length(θguess)), upper::AbstractVector=Fill(Inf, length(θguess)), kwargs...) where T<:Number
    @assert abs(sense) == 1
    @assert length(lower) == length(θguess) == length(upper)
    gcon = similar(θguess)
    startz = if TransformGuess
        gobj = similar(θguess);      Hcon = similar(θguess, length(θguess), length(θguess))
        _ConstrainedOptimisationInitialGuess(ZerodConstraint, θguess, gobj, gcon, Hcon; sense, reg, ADmode, ObjectiveGradient!, ConstraintGradient!, ConstraintHessian!, ProjectFirst, ProjectIters, ProjectTol, InteriorTol)[1]
    else
        copy(θguess)
    end
    df = Optim.TwiceDifferentiable(objective_fixedt0, ObjectiveGradient!, ObjectiveHessian!, startz)
    
    con_c!(c, z) = (c[1] = ZerodConstraint(z))
    con_jacobian!(J, z) = (ConstraintGradient!(gcon, z);   J[1, :] .= gcon;     nothing)
    con_hessian!(H, z, λ) = (ConstraintHessian!(H, z);     H .*= λ[1];     nothing)

    dfc = Optim.TwiceDifferentiableConstraints(con_c!, con_jacobian!, con_hessian!, lower, upper, [zero(T)], [zero(T)])
    result = Optim.optimize(df, dfc, startz, meth, Optim.Options(; iterations=maxiters, g_tol=tol, g_abstol=tol, kwargs...))
    Full ? result : (GetMinimizer(result), -Inf)
end


function SolvePointSphereOptimisationProblem(DM::AbstractDataModel, FixedInds::AbstractVector{<:Int}, FullInitial::AbstractVector{<:Number}, meth::Optim.AbstractConstrainedOptimizer;
                    factor::Real=1, XP::AbstractVector=zeros(length(FullInitial)), constraint::Function=Negloglikelihood(DM),
                    Confnum::Real=2, dof::Real=DOF(DM), IC::Real=icdfThreshold(dof, Confnum), loglikeMLE::Real=LogLikeMLE(DM), C::Real=-(loglikeMLE-0.5*IC),
                    ADmode::Val=Val(:ForwardDiff), levels::Int=1,
                    GenerateNewScore::Bool=true, GenerateNewCostHessian::Bool=false,
                    Multistart::Int=0, MultistartDomain::Union{Nothing,HyperCube}=(Multistart > 0 ? GetDomainSafe(DM) : nothing), Full::Bool=false, maxval::Real=1, ValInserter::Function=InformationGeometry.ValInserter,
                    TransformGuess::Bool=false, radiuslower::Real=1e-7, radiusupper::Real=1e3, Domain::Union{Nothing,HyperCube}=GetDomainSafe(DM), kwargs...)
    @assert all(1 .≤ FixedInds .≤ length(FullInitial)) && allunique(FixedInds)
    @assert 0 ≤ radiuslower < radiusupper
    ParameterDirection = view(FullInitial, FixedInds) .- view(XP, FixedInds)
    XPsub = view(XP, FixedInds)
    NuisanceInds = setdiff(1:length(FullInitial), FixedInds)
    startz = vcat(view(FullInitial, NuisanceInds), 1.0);    gbuf = similar(startz)
    V = ValInserter(FixedInds, ParameterDirection, FullInitial) # Inserted values will be completely replaced anyway
    function ReconstructModelParams(z::AbstractVector)
        Res = V(@view z[1:end-1])
        Res[FixedInds] .= XPsub .+ abs(z[end]) .* ParameterDirection
        Res
    end
    Jac! = GetJac!(ADmode, ReconstructModelParams)
    ZerodConstraintFunction(z::AbstractVector) = constraint(ReconstructModelParams(z))-C
    ConstraintGradient! = GenerateNewScore ? GetGrad!(ADmode, ZerodConstraintFunction) : EmbedScore(NegScore(DM), ReconstructModelParams, startz, FullInitial; ADmode, Jac!, levels)
    ConstraintHessian! = GenerateNewCostHessian ? GetHess!(ADmode, ZerodConstraintFunction) : EmbedFisher(CostHessian(DM), ReconstructModelParams, startz, FullInitial; ADmode, Jac!, levels)
    # ObjectiveFunction(z::AbstractVector) = -factor * exp(z[end])
    # ObjectiveGradient!(g, z::AbstractVector) = (g .= 0;  g[end] = -factor * exp(z[end]))
    # ObjectiveHessian!(H, z::AbstractVector) = (H .= 0;  H[end, end] = -factor * exp(z[end]))
    ObjectiveFunction(z::AbstractVector) = -factor * abs(z[end])
    ObjectiveGradient!(J, z::AbstractVector) = (J .= 0;  J[end] = -factor * Sgn(z[end]))
    ObjectiveHessian!(H, z::AbstractVector) = (H .= 0)
    df = Optim.TwiceDifferentiable(ObjectiveFunction, ObjectiveGradient!, ObjectiveHessian!, startz)
    
    con_c!(c, z) = (c[1] = ZerodConstraintFunction(z))
    function con_jacobian!(J, z)
        ConstraintGradient!(gbuf, z)
        J[1, :] .= gbuf
        nothing
    end
    function con_hessian!(H, z, λ)
        ConstraintHessian!(H, z)
        H .*= λ[1]
        nothing
    end
    if isnothing(Domain)
        lower = fill(-Inf, length(startz))
        upper = fill(Inf, length(startz))
    else
        @assert length(Domain) == length(FullInitial)
        lower = collect(Domain.L[NuisanceInds]) .- XP[NuisanceInds]
        upper = collect(Domain.U[NuisanceInds]) .- XP[NuisanceInds]
    end
    push!(lower, log(radiuslower))
    push!(upper, log(radiusupper))
    if Multistart > 0
        @assert length(MultistartDomain) == length(FullInitial)
        Points = GenerateSobolPoints(MultistartDomain; maxval, N=Multistart)
        for i in eachindex(Points)
            Points[i] = vcat(view(Points[i], NuisanceInds), 1.0)
        end
        MinimizeFunc = (F, x0; Kwargs...)->begin
            dfc = Optim.TwiceDifferentiableConstraints(con_c!, con_jacobian!, con_hessian!, lower, upper, [0.0], [0.0])
            Optim.optimize(df, dfc, x0, meth, Optim.Options(; Kwargs...))
        end
        Res = MultistartFit(ObjectiveFunction, Points; MinimizeFunc=MinimizeFunc, DM=nothing, showprogress=false, kwargs...)
        Full ? Res : ReconstructModelParams(GetMinimizer(Res)[1:end-1])
    else
        dfc = Optim.TwiceDifferentiableConstraints(con_c!, con_jacobian!, con_hessian!, lower, upper, [0.0], [0.0])
        result = Optim.optimize(df, dfc, startz, meth, Optim.Options(; kwargs...))
        Full ? result : ReconstructModelParams(GetMinimizer(result))
    end
end

### Projects FullInitial onto given confidence boundary strictly radially in the subspace defined by `FixedInds`.
function SolvePointSphereOptimisationProblem(DM::AbstractDataModel, FixedInds::AbstractVector{<:Int}, FullInitial::AbstractVector{<:Number}, meth=nothing; factor::Real=1, XP::AbstractVector=zeros(length(FullInitial)), 
                    constraint::Function=Negloglikelihood(DM), Confnum::Real=2, dof::Real=DOF(DM), IC::Real=icdfThreshold(dof, Confnum), loglikeMLE::Real=LogLikeMLE(DM), C::Real=-(loglikeMLE-0.5*IC), ADmode::Val=Val(:ForwardDiff), levels::Int=1, 
                    GenerateNewScore::Bool=true, GenerateNewCostHessian::Bool=false, 
                    Multistart::Int=0, MultistartDomain::Union{Nothing,HyperCube}=(Multistart > 0 ? GetDomainSafe(DM) : nothing), Full::Bool=false, maxval::Real=1, ValInserter::Function=InformationGeometry.ValInserter!, kwargs...)
    @assert all(1 .≤ FixedInds .≤ length(FullInitial)) && allunique(FixedInds)
    ## Fix direction in which parameters are to be changed and put its log-radius in the last component
    ParameterDirection = view(FullInitial, FixedInds) .- view(XP, FixedInds)
    XPsub = view(XP, FixedInds)
    @assert !iszero(norm(ParameterDirection)) "FullInitial - XP must have a nonzero component in span(Directions)."
    NuisanceInds = setdiff(1:length(FullInitial), FixedInds)
    startz = vcat(view(FullInitial, NuisanceInds), 1.0)
    V = ValInserter(FixedInds, ParameterDirection, FullInitial) # Inserted values will be completely replaced anyway
    function ReconstructModelParams(z::AbstractVector)
        Res = V(@view z[1:end-1])
        Res[FixedInds] .= XPsub .+ abs(z[end]) .* ParameterDirection
        Res
    end
    Jac! = GetJac!(ADmode, ReconstructModelParams)
    ### Maybe use Negloglikelihood as constraint, use Negate(Score(DM)) and leave CostHessian as-is?
    ZerodConstraint(z::AbstractVector) = constraint(ReconstructModelParams(z))-C
    ConstraintGradient! = GenerateNewScore ? GetGrad!(ADmode, ZerodConstraint) : EmbedScore(NegScore(DM), ReconstructModelParams, startz, FullInitial; ADmode, Jac!, levels)
    ConstraintHessian! = GenerateNewCostHessian ? GetHess!(ADmode, ZerodConstraint) : EmbedFisher(CostHessian(DM), ReconstructModelParams, startz, FullInitial; ADmode, Jac!, levels)
    # Objective same for all, could define upstream and reuse?
    # Optim minimizes directly; negate the radius to match the KKT solver's maximization.
    ObjectiveFunction(z::AbstractVector) = -factor * abs(z[end])
    ObjectiveGradient!(J, z::AbstractVector) = (J .= 0;  J[end] = -factor * Sgn(z[end]))
    ObjectiveHessian!(H, z::AbstractVector) = (H .= 0)
    if Multistart > 0
        FixedMeth = meth
        MinimizeFunc = (ObjectiveFunction, startz; meth=nothing, timeout=nothing, Kwargs...)->SolveConstrainedOptimisationProblem(ObjectiveFunction, ZerodConstraint, startz, FixedMeth; ADmode, levels, sense=1, Full=true, 
                    ConstraintGradient!, ConstraintHessian!, ObjectiveGradient!, ObjectiveHessian!, Kwargs...)
        Dom = isnothing(MultistartDomain) ? FullDomain(length(FullInitial), maxval) : MultistartDomain
        @assert length(Dom) == length(FullInitial)
        Points = GenerateSobolPoints(Dom; maxval, N=Multistart)
        for i in eachindex(Points)
            Points[i] = vcat(view(Points[i], NuisanceInds), 1.0)
        end
        Res = MultistartFit(ObjectiveFunction, Points; MinimizeFunc=MinimizeFunc, DM=nothing, showprogress=false, kwargs...)
        Full ? Res : ReconstructModelParams(MLE(Res)[1:end-1])
    else
        Res = SolveConstrainedOptimisationProblem(ObjectiveFunction, ZerodConstraint, startz, meth; ADmode, levels, sense=1, 
                    ConstraintGradient!, ConstraintHessian!, ObjectiveGradient!, ObjectiveHessian!, kwargs...)
        Full ? Res : ReconstructModelParams(Res[1])
    end
end


### Projects `FullInitial` onto a confidence boundary strictly radially in the subspace spanned by the columns of `Directions`.
function SolvePointSphereOptimisationProblem(DM::AbstractDataModel, Directions::AbstractMatrix{<:Number}, FullInitial::AbstractVector{<:Number}, meth=nothing; factor::Real=1, XP::AbstractVector=zeros(length(FullInitial)),
                    constraint::Function=Negloglikelihood(DM), Confnum::Real=2, dof::Real=DOF(DM), IC::Real=icdfThreshold(dof, Confnum), loglikeMLE::Real=LogLikeMLE(DM), C::Real=-(loglikeMLE-0.5*IC), ADmode::Val=Val(:ForwardDiff), levels::Int=1,
                    GenerateNewScore::Bool=true, GenerateNewCostHessian::Bool=false,
                    Multistart::Int=0, MultistartDomain::Union{Nothing,HyperCube}=(Multistart > 0 ? GetDomainSafe(DM) : nothing), Full::Bool=false, maxval::Real=1, NuisanceBasis::Union{Nothing,AbstractMatrix}=nothing, kwargs...)
    n = length(FullInitial);    subdim = size(Directions, 2)
    @assert size(Directions, 1) == n && 0 < subdim ≤ n
    @assert rank(Directions) == subdim "Columns of Directions must be linearly independent."
    @assert length(XP) == n
    if isnothing(NuisanceBasis)
        Q = Matrix(qr(Directions).Q[:, 1:n])
        NuisanceBasis = Q[:, subdim+1:end]
    end
    Basis = hcat(Directions, NuisanceBasis)
    @assert size(Basis) == (n, n) && rank(Basis) == n "Directions and NuisanceBasis must form a basis of parameter space."
    InitialCoordinates = Basis \ (FullInitial - XP)
    ParameterCoordinates = view(InitialCoordinates, 1:subdim)
    ParameterDirection = Directions * ParameterCoordinates
    @assert !iszero(norm(ParameterDirection)) "FullInitial - XP must have a nonzero component in span(Directions)."
    NuisanceCoordinates = view(InitialCoordinates, subdim+1:n)
    startz = vcat(NuisanceCoordinates, 1.0)
    function ReconstructModelParams(z::AbstractVector)
        XP .+ ParameterDirection .* abs(z[end]) .+ NuisanceBasis * (@view z[1:end-1])
    end
    Jac! = GetJac!(ADmode, ReconstructModelParams)
    ZerodConstraintFunction(z::AbstractVector) = constraint(ReconstructModelParams(z))-C
    ConstraintGradient! = GenerateNewScore ? GetGrad!(ADmode, ZerodConstraintFunction) : EmbedScore(NegScore(DM), ReconstructModelParams, startz, FullInitial; ADmode, Jac!, levels)
    ConstraintHessian! = GenerateNewCostHessian ? GetHess!(ADmode, ZerodConstraintFunction) : EmbedFisher(CostHessian(DM), ReconstructModelParams, startz, FullInitial; ADmode, Jac!, levels)
    # ObjectiveFunction(z::AbstractVector) = factor * exp(z[end])
    # ObjectiveGradient!(J, z::AbstractVector) = (J .= 0;  J[end] = factor * exp(z[end]))
    # ObjectiveHessian!(H, z::AbstractVector) = (H .= 0;  H[end, end] = factor * exp(z[end]))
    ObjectiveFunction(z::AbstractVector) = factor * abs(z[end])
    ObjectiveGradient!(J, z::AbstractVector) = (J .= 0;  J[end] = factor * Sgn(z[end]))
    ObjectiveHessian!(H, z::AbstractVector) = (H .= 0)
    if Multistart > 0
        FixedMeth = meth
        MinimizeFunc = (ObjectiveFunction, startz; meth=nothing, timeout=nothing, Kwargs...)->SolveConstrainedOptimisationProblem(ObjectiveFunction, ZerodConstraintFunction, startz, FixedMeth; ADmode, levels, sense=1, Full=true,
                    ConstraintGradient!, ConstraintHessian!, ObjectiveGradient!, ObjectiveHessian!, Kwargs...)
        Dom = isnothing(MultistartDomain) ? FullDomain(n, maxval) : MultistartDomain
        @assert length(Dom) == n
        Points = GenerateSobolPoints(Dom; maxval, N=Multistart)
        for i in eachindex(Points)
            Coordinates = Basis \ (Points[i] - XP)
            Points[i] = vcat(view(Coordinates, subdim+1:n), 1.0)
        end
        Res = MultistartFit(ObjectiveFunction, Points; MinimizeFunc=MinimizeFunc, DM=nothing, showprogress=false, kwargs...)
        Full ? Res : ReconstructModelParams(MLE(Res)[1:end-1])
    else
        Res = SolveConstrainedOptimisationProblem(ObjectiveFunction, ZerodConstraintFunction, startz, meth; ADmode, levels, sense=1,
                    ConstraintGradient!, ConstraintHessian!, ObjectiveGradient!, ObjectiveHessian!, kwargs...)
        Full ? Res : ReconstructModelParams(Res[1])
    end
end


## Returns new points to add
function BisectInds(FullPs::AbstractVector{T}; factor::Real=1.5, SubSetter::Function=identity, Norm::Function=norm, Median::Function=median) where T<:AbstractVector{<:Number}
    Ps = map(SubSetter, FullPs);    Ds = map(Norm, diff(Ps));    Med = Median(Ds);    Thresh = factor * Med
    Res = [(FullPs[i] .+ FullPs[i+1])./2 for i in eachindex(Ds) if Ds[i] > Thresh]
    Norm(Ps[1]-Ps[end]) > Thresh && push!(Res, (FullPs[1] .+ FullPs[end])./2)
    length(Res) == 0 ? T[] : Res
end

## Project points to two dimensions and order original points counter-clockwise based on projections
function ReorderPointsCCW(FullPs::AbstractVector{<:AbstractVector}; SubSetter::Function=identity, rev::Bool=false,
                    Ps::AbstractVector=map(SubSetter,FullPs), SubSettedMeanPoint::AbstractVector{<:Number}=mean(Ps))
    @assert ConsistentElDims(Ps) == 2
    order = map(p->mod2pi(atan(p[2]-SubSettedMeanPoint[2], p[1]-SubSettedMeanPoint[1])+2π), Ps)
    if !issorted(order; rev)
        @view FullPs[sortperm(order; rev)]
    else
        FullPs
    end
end

function IterativeBisectInds(Ps::AbstractVector{<:AbstractVector}; maxiters::Int=2, ProcessPoints::Function=identity, SubSetter::Function=identity, parallel::Bool=false, 
            XP::AbstractVector{<:Number}=Float64[], SubSettedMeanPoint::Union{Nothing,AbstractVector{<:Number}}=(length(XP) > 0 ? SubSetter(XP) : nothing), kwargs...)
    SubSettedMeanPointTuple = (!isnothing(SubSettedMeanPoint) ? (;SubSettedMeanPoint=SubSettedMeanPoint) : (;))
    Pts = collect(ReorderPointsCCW(Ps; SubSetter, SubSettedMeanPointTuple...))
    for _ in 1:maxiters
        ExtraPts = BisectInds(Pts; SubSetter, kwargs...)
        length(ExtraPts) == 0 && break
        Pts = ReorderPointsCCW([Pts; (parallel ? pmap : map)(ProcessPoints,ExtraPts)]; SubSetter, SubSettedMeanPointTuple...)
    end;    Pts
end

## Inner parallelism with parallel=true here should be avoided if GenericLowerTriangular already parallelises over different plots
## Outer parallelism faster since it keeps work more local
function GenerateProjectiveBoundaryPoints(DM::AbstractDataModel, FixedInds::AbstractVector{<:Int}, XP::AbstractVector=MLE(DM); N::Int=50, 
                    parallel::Bool=false, Refine::Bool=true, maxiters::Int=3, factor::Real=1.5, TransformGuess::Bool=false,
                    UnitSpherePointGenerator::Function=subdim->(@assert subdim == 2;  N::Int->[[cos(α), sin(α)] for α in range(0, 2π; length=N+1)[1:end-1]]),
                    Confnum::Real=2, dof::Real=DOF(DM), IC::Real=icdfThreshold(dof, Confnum), sqrtIC::Real=sqrt(IC), reducefactor::Real=0.9, 
                    ## Unit sphere generates starting values for FixedInds subspace, ScalePoints also of dim FixedInds
                    # L::AbstractMatrix=Eye(length(XP)), ScaleMatrix::AbstractMatrix=(@view L[FixedInds,FixedInds]), 
                    L::AbstractMatrix=Eye(length(XP)), CostHessian::Function=CostHessian(DM), H::AbstractMatrix=CostHessian(XP), 
                    Hschur::AbstractMatrix=SchurComplement(H, FixedInds, setdiff(1:length(XP), FixedInds)),
                    ScaleMatrix::AbstractMatrix=(E=eigen(Symmetric(Hschur)); E.vectors * Diagonal(inv.(sqrt.(E.values))) * E.vectors'),
                    ScalePoints::Function=Pt->(@view XP[FixedInds]) .+ reducefactor .* sqrtIC .* (ScaleMatrix*Pt), meth=Optim.IPNewton(), kwargs...)
    @assert all(1 .≤ FixedInds .≤ length(XP)) && allunique(FixedInds)
    subdim = length(FixedInds);    Points = UnitSpherePointGenerator(subdim)(N)
    SeedFixedInds(Pt::AbstractVector; Kwargs...) = SolvePointSphereOptimisationProblem(DM, FixedInds, (Z=copy(XP);   Z[FixedInds] .= Pt;   Z), meth; XP=XP, Confnum, dof, IC, TransformGuess, CostHessian, kwargs..., Kwargs...)
    Res = (parallel ? pmap : map)(SeedFixedInds∘ScalePoints, Points)
    !Refine && return Res
    IterativeBisectInds(Res; ProcessPoints=SeedFixedInds∘ViewElements(FixedInds), SubSetter=ViewElements(FixedInds), parallel, maxiters, factor, XP)
end


## Generates boundary points for the projection onto the subspace spanned by the columns of `Directions`.
function GenerateProjectiveBoundaryPoints(DM::AbstractDataModel, Directions::AbstractMatrix{<:Number}, XP::AbstractVector=MLE(DM); N::Int=50,
                    parallel::Bool=false, Refine::Bool=true, maxiters::Int=3, factor::Real=1.5, TransformGuess::Bool=false,
                    UnitSpherePointGenerator::Function=subdim->(@assert subdim == 2; N::Int->[[cos(α), sin(α)] for α in range(0, 2π; length=N+1)[1:end-1]]),
                    Confnum::Real=2, dof::Real=DOF(DM), IC::Real=icdfThreshold(dof, Confnum), sqrtIC::Real=sqrt(IC), reducefactor::Real=0.9,
                    CostHessian::Function=CostHessian(DM), H::AbstractMatrix=CostHessian(XP), SeedNuisanceBasis::Union{Nothing,AbstractMatrix}=nothing,
                    ScaleMatrix::AbstractMatrix=let
                        k = size(Directions, 2)
                        Q = isnothing(SeedNuisanceBasis) ? Matrix(qr(Directions).Q[:, 1:length(XP)])[:, k+1:end] : SeedNuisanceBasis
                        B = hcat(Directions, Q)
                        HB = B' * H * B
                        Hprofile = k == length(XP) ? HB : SchurComplement(HB, collect(1:k), collect(k+1:length(XP)))
                        E = eigen(Symmetric(Hprofile))
                        E.vectors * Diagonal(inv.(sqrt.(E.values))) * E.vectors'
                    end,
                    ScalePoints::Function=Pt->XP .+ reducefactor .* sqrtIC .* (Directions * (ScaleMatrix * Pt)), meth=Optim.IPNewton(), kwargs...)
    @assert size(Directions, 1) == length(XP) && size(Directions, 2) > 0
    @assert rank(Directions) == size(Directions, 2) "Columns of Directions must be linearly independent."
    @assert size(ScaleMatrix) == (size(Directions, 2), size(Directions, 2))
    subdim = size(Directions, 2);    Points = UnitSpherePointGenerator(subdim)(N)
    SeedDirections(Pt::AbstractVector; Kwargs...) = SolvePointSphereOptimisationProblem(DM, Directions, Pt, meth; XP, Confnum, dof, IC, TransformGuess, CostHessian, kwargs..., Kwargs...)
    Res = (parallel ? pmap : map)(SeedDirections∘ScalePoints, Points)
    !Refine && return Res
    ProjectionCoordinates = (Directions' * Directions) \ Directions'
    IterativeBisectInds(Res; ProcessPoints=SeedDirections, SubSetter=x->ProjectionCoordinates*x, parallel, maxiters, factor, XP)
end

## N corresponds to nuisance inds, F is indices of interest
SchurComplement(H::AbstractMatrix, F::AbstractVector{<:Int}, N::AbstractVector{<:Int}) = @views H[F, F] - H[F, N] * (H[N, N] \ H[N, F])


"""
GenericLowerTriangular(DM::AbstractDataModel, paridxs::AbstractVector{<:Int}=1:pdim(DM); MLE::AbstractVector=MLE(DM), 
            ProcessInds::Function=(inds; Kwargs...)->collect(GenerateProjectiveBoundaryPoints(DM, inds, MLE; Kwargs...)),
            parallel::Bool=true, parallelinner::Bool=!parallel, plot::Bool=isloaded(:Plots), kwargs...)
Plots projections of confidence region onto planes spanned by all pairs of parameters in `paridxs` to show non-linearity of parameter interdependence.

`parallel=true` parallelizes over parameter pairs and is the recommended default. 
Set `parallelinner=true` only when outer parallelism is disabled since enabling both creates significant scheduling overhead.
"""
function GenericLowerTriangular(DM::AbstractDataModel, paridxs::AbstractVector{<:Int}=1:pdim(DM); MLE::AbstractVector=MLE(DM), 
                ProcessInds::Function=(inds; Kwargs...)->collect(GenerateProjectiveBoundaryPoints(DM, inds, MLE; Kwargs...)),
                PrePlot::Function=inds->RecipesBase.plot([MLE[inds]]; ms=3, marker=:hex, label="MLE$(inds)", seriestype=:scatter), 
                ProcessSol::Function=(sol, inds)->map(ViewElements(inds), sol), parallel::Bool=true, parallelinner::Bool=!parallel,
                plot::Bool=isloaded(:Plots), pnames::AbstractVector{<:StringOrSymb}=pnames(DM), PlotMethod::Function=RecipesBase.plot!, SkipTests::Bool=true, 
                IndMat::AbstractMatrix{<:AbstractVector{<:Int}}=[[x,y] for y in paridxs, x in paridxs], PlotKwargs=(;),
                comparison::Function=Base.isless, size=PlotSizer(prod(Base.size(IndMat))), kwargs...)
    @assert length(MLE) ≥ 2
    @assert allunique(IndMat) && ConsistentElDims(@view IndMat[:]) == 2 && all(1 .≤ getindex.(IndMat,1) .≤ length(MLE)) && all(1 .≤ getindex.(IndMat,2) .≤ length(MLE))
    parallel && parallelinner && @warn "Enabling both `parallel` and `parallelinner` creates nested process parallelism and is usually slower due to scheduling overhead!"

    !SkipTests && !IsStructurallyIdentifiable(DM) && @warn "Model does not appear to be structurally identifiable. Continuing anyway."
    n = length(paridxs)
    finalidxs = [IndMat[i,j] for i in 2:n for j in 1:(n-1) if comparison(j,i)]
    Sols = (parallel ? progress_pmap : progress_map)(inds->ProcessInds(inds; parallel=parallelinner, kwargs...), finalidxs)
    plot && PlotLowerTriangular(Sols, IndMat; pnames, comparison, size, PrePlot, ProcessSol, PlotMethod, PlotKwargs...)
    Sols, finalidxs
end

function GenericLowerTriangularWithDecorrelation(DM::AbstractDataModel, paridxs::AbstractVector{<:Int}=1:pdim(DM); MLE::AbstractVector=MLE(DM), 
                PrePlot::Function=inds->RecipesBase.plot([MLE[inds]]; ms=3, marker=:hex, label="MLE$(inds)", seriestype=:scatter), 
                ProcessSol::Function=(sol, inds)->map(ViewElements(inds), sol), 
                plot::Bool=isloaded(:Plots), pnames::AbstractVector{<:StringOrSymb}=pnames(DM),
                IndMat::AbstractMatrix{<:AbstractVector{<:Int}}=[[x,y] for y in paridxs, x in paridxs], PlotKwargs=(;),
                comparison::Function=Base.isless, size=PlotSizer(prod(Base.size(IndMat))), Diagonal::Bool=false, kwargs...)
    Emb, invEmb, Jac!, M = DecorrelationTransformsWithJac(DM, MLE; Diagonal)
    dm = ModelEmbedding(DM, Emb, invEmb; Jac!);     WhitenedDirections = M \ Eye(Base.size(M,1))
    ProcessInds = (inds; Kwargs...)->(NuisanceInds = setdiff(1:Base.size(M,1), inds);  NB = Matrix(qr((M \ Eye(Base.size(M,1)))[:, NuisanceInds]).Q);  collect(GenerateProjectiveBoundaryPoints(dm, view(WhitenedDirections, :, inds), InformationGeometry.MLE(dm); NuisanceBasis=NB, SeedNuisanceBasis=NB, Kwargs..., parallel=false)))
    UntransformedSols, finalidxs = GenericLowerTriangular(dm, paridxs; ProcessInds, plot=false, IndMat, comparison, kwargs...)
    Sols = [map(Emb, sol) for sol in UntransformedSols]

    plot && PlotLowerTriangular(Sols, IndMat; pnames, comparison, size, PrePlot, ProcessSol, PlotKwargs...)
    Sols, finalidxs
end