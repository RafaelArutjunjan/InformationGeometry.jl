


function DataModel(DS::AbstractDataSet, odesys::ODESystem,
    observables::Union{AbstractVector{<:Int},AbstractRange{<:Int}}, args...)
    # DataModel(DS, GetModel(odesys,observables), GetDModel(odesys,observables), args...)
    DataModel(DS, GetModel(odesys,observables), args...)
end

function GetModel(odesys::ODESystem, observables::Union{AbstractVector{<:Int},AbstractRange{<:Int}}=Base.OneTo(length(odesys.states)); inplace::Bool=false)
    function Model(ts::AbstractVector{<:Real}, θ::AbstractVector{<:Number}; tol::Real=1e-6)
        sol = EvaluateSol(odesys, u₀map, tspan, odesys.ps .=> θ; tol=tol)
        mapreduce(t->sol(t)[observables], vcat, ts)
    end
end


# GetDModel here!!!


Optimize(DM::AbstractDataModel; inplace::Bool=false, timeout::Real=5) = Optimize(DM.model, (xdim(DS),ydim(DS),pdim(DM)); inplace=inplace, timeout=timeout)
function Optimize(DS::AbstractDataSet, model::ModelOrFunction; inplace::Bool=false, timeout::Real=5)
    Optimize(model, (xdim(DS),ydim(DS),pdim(DS,model)); inplace=inplace, timeout=timeout)
end
function Optimize(model::ModelOrFunction, xyp::Tuple{Int,Int,Int}; inplace::Bool=false, timeout::Real=5)
    @variables x₁ x[1:xyp[1]] y₁ y[1:xyp[2]] θ[1:xyp[3]]
    X = xyp[1] == 1 ? x₁ : x;         Y = xyp[2] == 1 ? y₁ : y

    # Add option for models which are already inplace
    function TryOptim()
        try
            model(X,θ)
        catch;
            @warn "Automated symbolic optimization of given model failed. Continuing without optimization."
        end
     end

    modelexpr = nothing
    task = @async(TryOptim())
    if timedwait(()->istaskdone(task), timeout) == :timed_out
        @async(Base.throwto(task, DivideError())) # kill task
    else
        modelexpr = fetch(task)
    end
    modelexpr == nothing && return nothing, nothing
    modelexpr = simplify(modelexpr)
    derivative = xyp[2] == 1 ? ModelingToolkit.gradient(modelexpr,θ; simplify=true) : ModelingToolkit.jacobian(modelexpr,θ; simplify=true)

    # Add option for parallel=ModelingToolkit.MultithreadedForm()
    OptimizedModel = inplace ? eval(build_function(modelexpr,X,θ)[2]) : eval(build_function(modelexpr,X,θ)[1])
    OptimizedDModel = inplace ? eval(build_function(derivative,X,θ)[2]) : eval(build_function(derivative,X,θ)[1])

    OptimizedModel, OptimizedDModel
end

"""
Convert Vector{Number} to Vector{Pair{Num,Number}} for u0s and ps.
"""
function EvaluateSol(odesys::ODESystem, u0::AbstractVector{<:Number}, ts::Union{Number,AbstractVector{<:Number}},
    θ::AbstractVector{<:Number}; tol::Real=1e-6, meth::OrdinaryDiffEqAlgorithm=Tsit5())
    EvaluateSol(odesys, odesys.states .=> u0, ts, odesys.ps .=> θ; meth=meth, tol=tol)
end
"""
Convert ts from Vector{Number} to Tuple{Real,Real}. KEEP INITIAL TIME AT ZERO.
"""
function EvaluateSol(odesys::ODESystem, u₀map::AbstractVector{<:Pair{<:Union{Num,Sym},<:Number}}, ts::Union{Number,AbstractVector{<:Number}},
    parammap::AbstractVector{<:Pair{<:Union{Num,Sym},<:Number}}; tol::Real=1e-6, meth::OrdinaryDiffEqAlgorithm=Tsit5())
    EvaluateSol(odesys, u₀map, (0.,maximum(ts)), parammap; meth=meth, tol=tol)
end
"""
Actually return solution object.
"""
function EvaluateSol(odesys::ODESystem, u₀map::AbstractVector{<:Pair{<:Union{Num,Sym},<:Number}}, tspan::Tuple{Real,Real},
    parammap::AbstractVector{<:Pair{<:Union{Num,Sym},<:Number}}; tol::Real=1e-6, meth::OrdinaryDiffEqAlgorithm=Tsit5())
    solve(ODEProblem(odesys, u₀map, tspan, parammap), meth; reltol=tol, abstol=tol)
end
