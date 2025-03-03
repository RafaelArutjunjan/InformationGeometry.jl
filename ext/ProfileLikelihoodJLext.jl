module ProfileLikelihoodJLext


using InformationGeometry, ProfileLikelihood, OptimizationBase, Optim
using RecipesBase


using InformationGeometry: Pnames, Domain, DOF, InvChisqCDF
using Optimization: AbstractADType


"""
    ProfileLikelihood.LikelihoodProblem(DM::AbstractDataModel, mle::AbstractVector=MLE(DM); Domain::Union{HyperCube,Nothing}=Domain(DM),
                adtype::AbstractADType=OptimizationBase.AutoForwardDiff(), syms::AbstractVector{<:Symbol}=Pnames(DM), kwargs...)
Constructs `LikelihoodProblem` struct from given `DataModel`.
"""
function ProfileLikelihood.LikelihoodProblem(DM::AbstractDataModel, mle::AbstractVector=MLE(DM);
                adtype::AbstractADType=OptimizationBase.AutoForwardDiff(), Domain::Union{HyperCube,Nothing}=Domain(DM),
                syms::AbstractVector{<:Symbol}=Pnames(DM),
                #lb=(!isnothing(Domain) ? Domain.L : fill(-Inf,length(mle))), ub=(!isnothing(Domain) ? Domain.U : fill(Inf,length(mle))),
                cons=nothing, lcons=nothing, ucons=nothing, f_kwargs=NamedTuple(), prob_kwargs=NamedTuple(), kwargs...)
    
    L = InformationGeometry.loglikelihood(DM)

    f_kwargs = (; adtype=adtype, cons=cons, f_kwargs...)
    # any(isfinite, lb) && any(isfinite, lb) && (prob_kwargs=(; lb=lb, ub=ub, prob_kwargs...))

    ProfileLikelihood.LikelihoodProblem((p,x)->L(p), mle; syms, f_kwargs=f_kwargs, prob_kwargs=prob_kwargs, kwargs...)
end


"""
    ProfileLikelihood.profile(DM::AbstractDataModel, Confnum::Real=1; idxs=1:pdim(DM), N::Int=31, OptimMeth=Optim.LBFGS(), parallel::Bool=true, kwargs...)
Computes profiles for given `idxs` up to given confidence threshold `Confnum` in units of `σ` via the `ProfileLikelihood.jl` package.
"""
function ProfileLikelihood.profile(DM::AbstractDataModel, Confnum::Real=1; idxs=1:pdim(DM), N::Int=31, meth=Optim.LBFGS(), OptimMeth=meth, alg=OptimMeth, 
                                    parallel::Bool=true, resolution=N, kwargs...)
    prob = ProfileLikelihood.LikelihoodProblem(DM)
    sol = ProfileLikelihood.mle(prob, alg)
    ProfileLikelihood.profile(prob, sol, idxs; alg, parallel, conf_level=ConfVol(Confnum),
        threshold=-0.5*InvChisqCDF(DOF(DM), ConfVol(Confnum)), resolution, kwargs...)
end


"""
    ProfileLikelihood.bivariate_profile(DM::AbstractDataModel, Confnum::Real=1; idxs=nothing, N::Int=31, OptimMeth=Optim.LBFGS(), parallel::Bool=true, kwargs...)
Computes bivariate profiles for given `idxs` up to given confidence threshold `Confnum` in units of `σ`, where pairs of parameters are fixed at different values and the remaining parameters are re-optimized.
"""
function ProfileLikelihood.bivariate_profile(DM::AbstractDataModel, Confnum::Real=1; idxs=nothing, N::Int=31, meth=Optim.LBFGS(), OptimMeth=meth, alg=OptimMeth, 
                                        parallel::Bool=true, resolution=N, kwargs...)
    prob = ProfileLikelihood.LikelihoodProblem(DM)
    sol = ProfileLikelihood.mle(prob, alg)
    if isnothing(idxs)
        ProfileLikelihood.bivariate_profile(prob, sol; alg, parallel, conf_level=ConfVol(Confnum),
            threshold=-0.5InvChisqCDF(DOF(DM), ConfVol(Confnum)), resolution, kwargs...)
    else
        ProfileLikelihood.bivariate_profile(prob, sol, idxs; alg, parallel, conf_level=ConfVol(Confnum),
            threshold=-0.5InvChisqCDF(DOF(DM), ConfVol(Confnum)), resolution, kwargs...)
    end
end


RecipesBase.@recipe function f(P::ProfileLikelihood.ProfileLikelihoodSolutionView)
    ylabel --> "Cost Function"
    xlabel --> string(ProfileLikelihood.variable_symbols(P)[ProfileLikelihood.get_index(P)])
    label --> nothing
    lw --> 1.5
    @series begin
        get_parameter_values(P), -get_profile_values(P)
    end
    @series begin
        label --> nothing
        color --> :red
        st --> :scatter
        markersize --> 2.5
        markerstrokewidth --> 0
        [ProfileLikelihood.get_mle(ProfileLikelihood.get_likelihood_solution(P))[ProfileLikelihood.get_index(P)]], [0.0]
    end
end

RecipesBase.@recipe function f(P::ProfileLikelihood.ProfileLikelihoodSolution)
    layout --> length(ProfileLikelihood.variable_symbols(P))
    for i in eachindex(ProfileLikelihood.variable_symbols(P))
        @series begin
            subplot := i
            P[i]
        end
    end
end


end # module
