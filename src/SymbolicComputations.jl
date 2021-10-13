


Getxyp(DM::AbstractDataModel) = (xdim(DM), ydim(DM), pdim(DM))
Getxyp(DS::AbstractDataSet, model::Function) = (xdim(DS),ydim(DS),pdim(DS,model))
Getxyp(DS::AbstractDataSet, M::ModelMap) = M.xyp

SymbolicArguments(args...) = SymbolicArguments(Getxyp(args...))
function SymbolicArguments(xyp::Tuple{Int,Int,Int})
    (xyp[1] == 1 ? (@variables x)[1] : (@variables x[1:xyp[1]])[1]),
    (xyp[2] == 1 ? (@variables y)[1] : (@variables y[1:xyp[2]])[1]),
    (@variables θ[1:xyp[3]])[1]
end
# function SymbolicArguments(xyp::Tuple{Int,Int,Int})
#     @variables X[1:xyp[1]] Y[1:xyp[2]] θ[1:xyp[3]] x y
#     X, Y, θ
#     (xyp[1] == 1 ? x : X), (xyp[2] == 1 ? y : Y), θ
# end


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
    expr = @suppress_err ToExpr(DM)
    isnothing(expr) ? "Unable to represent given model symbolically." : "y(x,θ) = $expr"
end

function SymbolicdModel(DM::AbstractDataModel)
    if !GeneratedFromSymbolic(dPredictor(DM))
        println("Given Model jacobian not symbolic. Trying to apply OptimizedDM() first.")
        odm = OptimizedDM(DM)
        if isnothing(ToExpr(odm))
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


function Optimize(DM::AbstractDataModel; inplace::Bool=false, timeout::Real=5, parallel::Bool=false, kwargs...)
    Optimize(Data(DM), Predictor(DM); inplace=inplace, timeout=timeout, parallel=parallel, kwargs...)
end
function Optimize(DS::AbstractDataSet, model::ModelOrFunction; inplace::Bool=false, timeout::Real=5, parallel::Bool=false, kwargs...)
    Optimize(model, Getxyp(DS, model); inplace=inplace, timeout=timeout, parallel=parallel, kwargs...)
end
function Optimize(M::ModelMap, xyp::Tuple{Int,Int,Int}; inplace::Bool=false, timeout::Real=5, parallel::Bool=false, kwargs...)
    xyp != M.xyp && throw("xyp inconsistent.")
    model, dmodel = Optimize(M.Map, xyp; inplace=inplace, timeout=timeout, parallel=parallel, kwargs...)
    ModelMap(model, M), ModelMap(dmodel, M)
end
function Optimize(model::Function, xyp::Tuple{Int,Int,Int}; inplace::Bool=false, timeout::Real=5, parallel::Bool=false, kwargs...)
    modelexpr = ToExpr(model, xyp; timeout=timeout) |> Symbolics.simplify
    isnothing(modelexpr) && return nothing, nothing

    X, Y, θ = SymbolicArguments(xyp)

    # Need to make sure that modelexpr is of type Vector{Num}, not just Num
    !(modelexpr isa AbstractVector{<:Num}) && (modelexpr = [modelexpr])
    derivative = Symbolics.jacobian(modelexpr, θ; simplify=true)

    ExprToModelMap(X, θ, modelexpr; inplace=inplace, parallel=parallel, IsJacobian=false, kwargs...), ExprToModelMap(X, θ, derivative; inplace=inplace, parallel=parallel, IsJacobian=true, kwargs...)
end

function ExprToModelMap(X::Union{Num,AbstractVector{<:Num}}, P::AbstractVector{Num}, modelexpr::Union{Num,AbstractArray{<:Num}};
                                                        inplace::Bool=false, parallel::Bool=false, IsJacobian::Bool=false, force_SA::Bool=IsJacobian, kwargs...)
    OptimizedModel = Builder(modelexpr, X, P; inplace=inplace, parallel=parallel, force_SA=force_SA, kwargs...)

    ### Pretty Function names
    if IsJacobian
        # THROWING AWAY KWARGS HERE!
        SymbolicModelJacobian(x::Union{Number,AbstractVector{<:Number}}, θ::AbstractVector{<:Number}; Kwargs...) = OptimizedModel(x, θ)
        function SymbolicModelJacobian!(y::Union{Number,AbstractMatrix{<:Number}}, x::Union{Number,AbstractVector{<:Number}}, θ::AbstractVector{<:Number}; Kwargs...)
            OptimizedModel(y, x, θ)
        end
        return inplace ? SymbolicModelJacobian! : SymbolicModelJacobian
    else
        # THROWING AWAY KWARGS HERE!
        SymbolicModel(x::Union{Number,AbstractVector{<:Number}}, θ::AbstractVector{<:Number}; Kwargs...) = OptimizedModel(x, θ)
        SymbolicModel!(y::Union{Number,AbstractVector{<:Number}}, x::Union{Number,AbstractVector{<:Number}}, θ::AbstractVector{<:Number}; Kwargs...) = OptimizedModel(y, x, θ)
        return inplace ? SymbolicModel! : SymbolicModel
    end
end


function OptimizedDM(DM::AbstractDataModel; kwargs...)
    model, dmodel = Optimize(DM; kwargs...)
    # Very simple models (ydim=1) typically slower after simplification using ModelingToolkit.jl / Symbolics.jl
    !isnothing(dmodel) ? DataModel(Data(DM), Predictor(DM), dmodel, MLE(DM), LogLikeMLE(DM)) : DM
end
