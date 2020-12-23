


function DataModel(DS::AbstractDataSet, odesys::ODESystem, u₀map::AbstractVector{<:Pair{<:Union{Num,Sym},<:Number}},
                    tspan::Tuple{Real,Real}, observables::Union{AbstractVector{<:Int},AbstractRange{<:Int}}, args...)
    DataModel(DS, GetModel(odesys,u₀map,tspan,observables), args...)
end

function GetModel(odesys::ODESystem, u₀map::AbstractVector{<:Pair{<:Union{Num,Sym},<:Number}}, tspan::Tuple{Real,Real},
                    observables::Union{AbstractVector{<:Int},AbstractRange{<:Int}}=Base.OneTo(length(odesys.states)); inplace::Bool=false)
    function Model(ts::AbstractVector{<:Real}, θ::AbstractVector{<:Number}; tol::Real=1e-6)
        sol = EvaluateSol(odesys, u₀map, tspan, odesys.ps .=> θ; tol=tol)
        mapreduce(t->sol(t)[observables], vcat, ts)
    end
end

function GetDModel(odesys::ODESystem, u₀map::AbstractVector{<:Pair{<:Union{Num,Sym},<:Number}}, tspan::Tuple{Real,Real},
                    observables::Union{AbstractVector{<:Int},AbstractRange{<:Int}}=Base.OneTo(length(odesys.states)); inplace::Bool=false)
    DetermineDmodel(GetModel(odesys, u₀map, tspan, observables; inplace=inplace))
end


function Optimize(DM::AbstractDataModel; inplace::Bool=false, timeout::Real=5, parallel::Bool=false)
    Optimize(Predictor(DM), (xdim(DM),ydim(DM),pdim(DM)); inplace=inplace, timeout=timeout, parallel=parallel)
end
function Optimize(DS::AbstractDataSet, model::ModelOrFunction; inplace::Bool=false, timeout::Real=5, parallel::Bool=false)
    Optimize(model, (xdim(DS),ydim(DS),pdim(DS,model)); inplace=inplace, timeout=timeout, parallel=parallel)
end
function Optimize(M::ModelMap, xyp::Tuple{Int,Int,Int}; inplace::Bool=false, timeout::Real=5, parallel::Bool=false)
    xyp != M.xyp && throw("xyp inconsistent.")
    model, dmodel = Optimize(M.Map, xyp; inplace=inplace, timeout=timeout, parallel=parallel)
    ModelMap(model, M), ModelMap(dmodel, M)
end
function Optimize(model::Function, xyp::Tuple{Int,Int,Int}; inplace::Bool=false, timeout::Real=5, parallel::Bool=false)
    @variables x[1:xyp[1]] y[1:xyp[2]] θ[1:xyp[3]]
    X = xyp[1] == 1 ? x[1] : x;         Y = xyp[2] == 1 ? y[1] : y

    parallel && println("Parallel functionality not implemented yet.")

    # Add option for models which are already inplace
    function TryOptim(model,X,θ)
        try
            model(X,θ)
        catch;
            @warn "Automated symbolic optimization of given model failed. Continuing without optimization."
        end
     end

    modelexpr = nothing
    task = @async(TryOptim(model,X,θ))
    if timedwait(()->istaskdone(task), timeout) == :timed_out
        @async(Base.throwto(task, DivideError())) # kill task
    else
        modelexpr = fetch(task)
    end
    modelexpr == nothing && return nothing, nothing
    # Need to make sure that modelexpr is of type Vector{Num}, not just Num
    modelexpr = xyp[2] == 1 ? [simplify(modelexpr)] : simplify(modelexpr)
    derivative = ModelingToolkit.jacobian(modelexpr,θ; simplify=true)

    # Pass through keyword "parallel" later
    ExprToModelMap(X,θ, modelexpr; inplace=inplace, IsJacobian=false), ExprToModelMap(X,θ, derivative; inplace=inplace, IsJacobian=true)
end

function ExprToModelMap(X::Union{Num,AbstractVector{<:Num}}, P::AbstractVector{Num}, modelexpr::Union{Num,AbstractArray{<:Num}}; inplace::Bool=false, parallel::Bool=false, IsJacobian::Bool=false)
    # ParamNames = P .|> z->string(z.val.name)
    # xyp = (length(X), length(modelexpr), length(P))
    if parallel
        throw("Not programmed yet")
        ## Add option for parallel=ModelingToolkit.MultithreadedForm()
        # OptimizedModel = build_function(modelexpr, X, θ; expression=Val{false})[inplace ? 2 : 1]
        # OptimizedDModel = build_function(derivative, X, θ; expression=Val{false})[inplace ? 2 : 1]
    else
        OptimizedModel = try eval(build_function(modelexpr, X, P)[inplace ? 2 : 1]) catch; eval(build_function(modelexpr, X, P)) end

        ### Pretty Function names
        SymbolicModel(x, θ) = OptimizedModel(x, θ);        SymbolicModelJacobian(x, θ) = OptimizedModel(x, θ)
        return IsJacobian ? SymbolicModelJacobian : SymbolicModel
    end
end



function OptimizedDM(DM::AbstractDataModel)
    model, dmodel = Optimize(DM)
    # Very simple models (ydim=1) typically slower after simplification using ModelingToolkit.jl
    if dmodel != nothing
        return DataModel(Data(DM), Predictor(DM), dmodel, MLE(DM), LogLikeMLE(DM))
    else
        # Get warning from Optimize() that symbolic optimization was unsuccessful
        return DM
    end
end



"""
Convert Vector{Number} to Vector{Pair{Num,Number}} for u0s and ps.
"""
function EvaluateSol(odesys::ODESystem, u0::AbstractVector{<:Number}, ts::Union{Number,AbstractVector{<:Number}},
    θ::AbstractVector{<:Number}; tol::Real=1e-6, meth::OrdinaryDiffEqAlgorithm=Tsit5(), kwargs...)
    EvaluateSol(odesys, odesys.states .=> u0, ts, odesys.ps .=> θ; meth=meth, tol=tol, kwargs...)
end
"""
Convert ts from Vector{Number} to Tuple{Real,Real}. KEEP INITIAL TIME AT ZERO.
"""
function EvaluateSol(odesys::ODESystem, u₀map::AbstractVector{<:Pair{<:Union{Num,Sym},<:Number}}, ts::Union{Number,AbstractVector{<:Number}},
    parammap::AbstractVector{<:Pair{<:Union{Num,Sym},<:Number}}; tol::Real=1e-6, meth::OrdinaryDiffEqAlgorithm=Tsit5(), kwargs...)
    EvaluateSol(odesys, u₀map, (0.,maximum(ts)), parammap; meth=meth, tol=tol, kwargs...)
end
"""
Actually return solution object.
"""
function EvaluateSol(odesys::ODESystem, u₀map::AbstractVector{<:Pair{<:Union{Num,Sym},<:Number}}, tspan::Tuple{Real,Real},
    parammap::AbstractVector{<:Pair{<:Union{Num,Sym},<:Number}}; tol::Real=1e-6, meth::OrdinaryDiffEqAlgorithm=Tsit5(), kwargs...)
    solve(ODEProblem(odesys, u₀map, tspan, parammap), meth; reltol=tol, abstol=tol, kwargs...)
end
