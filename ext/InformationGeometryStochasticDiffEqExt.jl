module InformationGeometryStochasticDiffEqExt


using InformationGeometry, StochasticDiffEq, SciMLSensitivity, SciMLBase

import InformationGeometry: GetModel, GetMethod, MaximalNumberOfArguments, CompleteObservationFunction, MakeCustom, Reduction, ConditionalConvert, PromoteStatic
import SciMLBase: AbstractSDEFunction, AbstractNoiseProcess
const SenseAlg = SciMLBase.AbstractSensitivityAlgorithm


# Filter according to func with output Bool
function Base.filter(func::Function, ensemblesol::SciMLBase.AbstractEnsembleSolution)
    EnsembleSolution([f for f in ensemblesol[(1:length(ensemblesol))[func.(ensemblesol.u)]]], ensemblesol.elapsedTime, ensemblesol.converged)
end


function InformationGeometry._GetModelFast(func::SciMLBase.AbstractSDEFunction{T}, SplitterFunction::Function, PreObservationFunction::Function; trajectories::Union{Int,Nothing}=1000, filter::Function=x->true,
                    noise::Union{AbstractNoiseProcess,Nothing}=nothing, noise_rate_prototype::Union{AbstractArray,Nothing}=nothing, dt::Union{Real,Nothing}=nothing, mean::Bool=true,
                    sensealg::SenseAlg=SciMLSensitivity.ForwardDiffSensitivity(), callback=nothing, EnsembleAlg::SciMLBase.EnsembleAlgorithm=SciMLBase.EnsembleThreads(),
                    meth::SciMLBase.AbstractSDEAlgorithm=(noise===noise_rate_prototype===nothing ? SOSRI() : SOSRA()), Domain::Union{HyperCube,Nothing}=nothing, inplace::Bool=true, Kwargs...) where T
    # @assert T == inplace
    CB = callback
    ObservationFunction = CompleteObservationFunction(PreObservationFunction)

    function GetSol(θ::AbstractVector{<:Number}, SplitterFunction::Function; trajectories::Union{Int,Nothing}=trajectories, sensealg::SenseAlg=sensealg, noise::Union{AbstractNoiseProcess,Nothing}=noise,
                                    noise_rate_prototype::Union{AbstractArray,Nothing}=noise_rate_prototype, dt::Union{Real,Nothing}=dt, max_t::Ttype=10., meth::SciMLBase.AbstractSDEAlgorithm=meth, callback=nothing, 
                                    EnsembleAlg::SciMLBase.EnsembleAlgorithm=EnsembleAlg, kwargs...) where Ttype <:Number
        u0, p = SplitterFunction(θ)
        sdeprob = SDEProblem(func, func.g, ConditionalConvert(Ttype,u0), (zero(max_t), max_t), ConditionalConvert(Ttype,p); noise_rate_prototype=noise_rate_prototype, noise=noise)
        if isnothing(trajectories)
            isnothing(dt) ? solve(sdeprob, meth; sensealg=sensealg, callback=CallbackSet(callback, CB), Kwargs..., kwargs...) : solve(sdeprob, meth; sensealg=sensealg, dt=dt, adaptive=false, callback=CallbackSet(callback, CB), Kwargs..., kwargs...)
        else
            isnothing(dt) ? solve(EnsembleProblem(sdeprob), meth, EnsembleAlg; trajectories=trajectories, sensealg=sensealg, callback=CallbackSet(callback, CB), Kwargs..., kwargs...) : solve(EnsembleProblem(sdeprob), meth, EnsembleAlg; trajectories=trajectories, dt=dt, adaptive=false, sensealg=sensealg, callback=CallbackSet(callback, CB), Kwargs..., kwargs...)
        end
    end
    function SDEModel(t::Number, θ::AbstractVector{<:Number}; max_t::Number=t, kwargs...)
        res = SDEModel([t], θ; max_t=max_t, kwargs...)
        (length(res) == 1 && res isa AbstractVector{<:Number}) ? res[1] : res
    end
    function SDEModel(ts::AbstractVector{<:Number}, θ::AbstractVector{<:Number}; filter::Function=filter, ObservationFunction::Function=ObservationFunction, SplitterFunction::Function=SplitterFunction,
                                            noise::Union{AbstractNoiseProcess,Nothing}=noise, noise_rate_prototype::Union{AbstractArray,Nothing}=noise_rate_prototype, dt::Union{Real,Nothing}=dt, sensealg::SenseAlg=sensealg,
                                            trajectories::Union{Int,Nothing}=trajectories, mean::Bool=mean, max_t::Number=maximum(ts), meth::SciMLBase.AbstractSDEAlgorithm=meth, FullSol::Bool=false, kwargs...)
        FullSol && return GetSol(θ, SplitterFunction; trajectories=trajectories, noise_rate_prototype=noise_rate_prototype, noise=noise, dt=dt, sensealg=sensealg, max_t=max_t, meth=meth, tstops=ts, kwargs...)
        sol = GetSol(θ, SplitterFunction; trajectories=trajectories, noise_rate_prototype=noise_rate_prototype, noise=noise, dt=dt, sensealg=sensealg, max_t=max_t, meth=meth, saveat=ts, kwargs...)
        points = if isnothing(trajectories)
            sol.u
        else
            keep = map(filter, sol.u)
            if all(keep)
                mean ? SciMLBase.EnsembleAnalysis.timeseries_point_mean(sol, ts) : SciMLBase.EnsembleAnalysis.timeseries_point_median(sol, ts)
            elseif any(keep)
                mean ? SciMLBase.EnsembleAnalysis.timeseries_point_mean(sol[(1:trajectories)[keep]], ts) : SciMLBase.EnsembleAnalysis.timeseries_point_median(sol[(1:trajectories)[keep]], ts)
            else
                @warn "All solutions in SDEModel filtered out. Aborting filtering process and return all solutions instead."
                mean ? SciMLBase.EnsembleAnalysis.timeseries_point_mean(sol, ts) : SciMLBase.EnsembleAnalysis.timeseries_point_median(sol, ts)
            end
        end
        length(points) != length(ts) && throw("SDE integration failed, maybe try using a lower tolerance value.")
        [ObservationFunction(points[i], ts[i], θ) for i in eachindex(ts)] |> Reduction
    end
    # MakeCustom(SDEModel, Domain; Meta=(func, SplitterFunction, ObservationFunction, callback), verbose=false)
    Meta = (func, SplitterFunction, ObservationFunction, callback)
    SDEModel, Meta
end




end # module
