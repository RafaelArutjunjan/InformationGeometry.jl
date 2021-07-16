


Getxyp(DM::AbstractDataModel) = (xdim(DM), ydim(DM), pdim(DM))
Getxyp(DS::AbstractDataSet, model::Function) = (xdim(DS),ydim(DS),pdim(DS,model))
Getxyp(DS::AbstractDataSet, M::ModelMap) = M.xyp

SymbolicArguments(args...) = SymbolicArguments(Getxyp(args...))
function SymbolicArguments(xyp::Tuple{Int,Int,Int})
    @variables X[1:xyp[1]] Y[1:xyp[2]] θ[1:xyp[3]] x y
    X, Y, θ
    (xyp[1] == 1 ? x : X), (xyp[2] == 1 ? y : Y), θ
end


ToExpr(DM::AbstractDataModel; timeout::Real=5) = ToExpr(Predictor(DM), (xdim(DM), ydim(DM), pdim(DM)); timeout=timeout)
ToExpr(DS::AbstractDataSet, model::Function; timeout::Real=5) = ToExpr(model, (xdim(DS),ydim(DS),pdim(DS,model)); timeout=timeout)
ToExpr(DS::AbstractDataSet, M::ModelMap; timeout::Real=5) = ToExpr(M.Map, M.xyp; timeout=timeout)
ToExpr(M::ModelMap; timeout::Real=5) = ToExpr(M.Map, M.xyp; timeout=timeout)
ToExpr(M::ModelMap, xyp::Tuple{Int,Int,Int}; timeout::Real=5) = xyp == M.xyp ? ToExpr(M.Map, M.xyp; timeout=timeout) : throw("Inconsistent xyp information.")

function ToExpr(model::Function, xyp::Tuple{Int,Int,Int}; timeout::Real=5)
    X, Y, θ = SymbolicArguments(xyp)
    KillAfter(model, X, θ; timeout=timeout)
    # Add option for models which are already inplace
end

function SymbolicModel(DM::AbstractDataModel)
    expr = ToExpr(DM)
    expr === nothing ? "Unable to represent given model symbolically." : "y(x,θ) = $expr"
end

function SymbolicdModel(DM::AbstractDataModel)
    if !GeneratedFromSymbolic(dPredictor(DM))
        println("Given Model jacobian not symbolic. Trying to apply OptimizedDM() first.")
        odm = OptimizedDM(DM)
        if ToExpr(odm) === nothing
            return "Unable to represent given jacobian symbolically."
        else
            X, Y, θ = SymbolicArguments(odm)
            return "(∂y/∂θ)(x,θ) = $(dPredictor(odm)(X, θ))"
        end
    else
        X, Y, θ = SymbolicArguments(DM)
        return "(∂y/∂θ)(x,θ) = $(dPredictor(DM)(X, θ))"
    end
end


function Optimize(DM::AbstractDataModel; inplace::Bool=false, timeout::Real=5, parallel::Bool=false)
    Optimize(Data(DM), Predictor(DM); inplace=inplace, timeout=timeout, parallel=parallel)
end
function Optimize(DS::AbstractDataSet, model::ModelOrFunction; inplace::Bool=false, timeout::Real=5, parallel::Bool=false)
    Optimize(model, Getxyp(DS, model); inplace=inplace, timeout=timeout, parallel=parallel)
end
function Optimize(M::ModelMap, xyp::Tuple{Int,Int,Int}; inplace::Bool=false, timeout::Real=5, parallel::Bool=false)
    xyp != M.xyp && throw("xyp inconsistent.")
    model, dmodel = Optimize(M.Map, xyp; inplace=inplace, timeout=timeout, parallel=parallel)
    ModelMap(model, M), ModelMap(dmodel, M)
end
function Optimize(model::Function, xyp::Tuple{Int,Int,Int}; inplace::Bool=false, timeout::Real=5, parallel::Bool=false)
    modelexpr = ToExpr(model, xyp; timeout=timeout) |> Symbolics.simplify
    modelexpr == nothing && return nothing, nothing

    X, Y, θ = SymbolicArguments(xyp)

    # Need to make sure that modelexpr is of type Vector{Num}, not just Num
    !(modelexpr isa AbstractVector{<:Num}) && (modelexpr = [modelexpr])
    derivative = Symbolics.jacobian(modelexpr, θ; simplify=true)

    ExprToModelMap(X, θ, modelexpr; inplace=inplace, parallel=parallel, IsJacobian=false), ExprToModelMap(X, θ, derivative; inplace=inplace, parallel=parallel, IsJacobian=true)
end

function ExprToModelMap(X::Union{Num,AbstractVector{<:Num}}, P::AbstractVector{Num}, modelexpr::Union{Num,AbstractArray{<:Num}};
                                                        inplace::Bool=false, parallel::Bool=false, IsJacobian::Bool=false)
    parallelization = parallel ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()
    OptimizedModel = try
        Symbolics.build_function(modelexpr, X, P; expression=Val{false}, parallel=parallelization)[inplace ? 2 : 1]
    catch;
        Symbolics.build_function(modelexpr, X, P; expression=Val{false}, parallel=parallelization)
    end
    ### Pretty Function names
    if IsJacobian
        # THROWING AWAY KWARGS HERE!
        SymbolicModelJacobian(x::Union{Number,AbstractVector{<:Number}}, θ::AbstractVector{<:Number}; kwargs...) = OptimizedModel(x, θ)
        function SymbolicModelJacobian!(y::Union{Number,AbstractMatrix{<:Number}}, x::Union{Number,AbstractVector{<:Number}}, θ::AbstractVector{<:Number}; kwargs...)
            OptimizedModel(y, x, θ)
        end
        return inplace ? SymbolicModelJacobian! : SymbolicModelJacobian
    else
        # THROWING AWAY KWARGS HERE!
        SymbolicModel(x::Union{Number,AbstractVector{<:Number}}, θ::AbstractVector{<:Number}; kwargs...) = OptimizedModel(x, θ)
        SymbolicModel!(y::Union{Number,AbstractVector{<:Number}}, x::Union{Number,AbstractVector{<:Number}}, θ::AbstractVector{<:Number}; kwargs...) = OptimizedModel(y, x, θ)
        return inplace ? SymbolicModel! : SymbolicModel
    end
end


function OptimizedDM(DM::AbstractDataModel)
    model, dmodel = Optimize(DM)
    # Very simple models (ydim=1) typically slower after simplification using ModelingToolkit.jl / Symbolics.jl
    if dmodel != nothing
        return DataModel(Data(DM), Predictor(DM), dmodel, MLE(DM), LogLikeMLE(DM))
    else
        # Get warning from Optimize() that symbolic optimization was unsuccessful
        return DM
    end
end
