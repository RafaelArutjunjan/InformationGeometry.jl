


Getxyp(DM::AbstractDataModel) = (xdim(DM), ydim(DM), pdim(DM))
Getxyp(DS::AbstractDataSet, model::Function) = (xdim(DS),ydim(DS),pdim(DS,model))
Getxyp(DS::AbstractDataSet, M::ModelMap) = M.xyp

SymbolicArguments(args...) = SymbolicArguments(Getxyp(args...))
function SymbolicArguments(xyp::Tuple{Int,Int,Int})
    (xyp[1] == 1 ? (@variables x)[1] : (@variables x[1:xyp[1]])[1]),
    (xyp[2] == 1 ? (@variables y)[1] : (@variables y[1:xyp[2]])[1]),
    (@variables θ[1:xyp[3]])[1]
end


ToExpr(DM::AbstractDataModel; timeout::Real=5) = ToExpr(Predictor(DM), (xdim(DM), ydim(DM), pdim(DM)); timeout=timeout)
ToExpr(DS::AbstractDataSet, model::Function; timeout::Real=5) = ToExpr(model, (xdim(DS),ydim(DS),pdim(DS,model)); timeout=timeout)
ToExpr(DS::AbstractDataSet, M::ModelMap; timeout::Real=5) = ToExpr(M, M.xyp; timeout=timeout)

function ToExpr(model::Function, xyp::Tuple{Int,Int,Int}; timeout::Real=5)
    X, Y, θ = SymbolicArguments(xyp)
    n = MaximalNumberOfArguments(model)
    if n == 2
        KillAfter(model, X, θ; timeout=timeout)
    elseif n == 3
        Res=Vector{Num}(undef,length(Y));   KillAfter(model, Res, X, θ; timeout=timeout);   Res
    else
        throw("Got MaximalNumberOfArguments == $n.")
    end
end
function ToExpr(M::ModelMap, xyp::Tuple{Int,Int,Int}=M.xyp; timeout::Real=5)
    @assert xyp == M.xyp "Inconsistent xyp information."
    X, Y, θ = SymbolicArguments(xyp)
    isinplacemodel(M) ? (Res=Vector{Num}(undef,length(Y));  KillAfter(M.Map, Res, X, θ; timeout=timeout); Res) : KillAfter(M.Map, X, θ; timeout=timeout)
end

function SymbolicModel(DM::AbstractDataModel)
    expr = @suppress_err ToExpr(DM)
    isnothing(expr) ? "Unable to represent given model symbolically." : "y(x,θ) = $expr"
end

function SymbolicdModel(DM::AbstractDataModel)
    if !GeneratedFromSymbolic(dPredictor(DM))
        @warn "Given Model jacobian not symbolic. Trying to apply OptimizedDM() first."
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

# mixing inplace (outcome) with isinplace (input) here. Disambiguate!

Optimize(DM::AbstractDataModel; kwargs...) = Optimize(Data(DM), Predictor(DM); kwargs...)
Optimize(DS::AbstractDataSet, model::ModelOrFunction; kwargs...) = Optimize(model, Getxyp(DS, model); kwargs...)

Optimize(model::Function, xyp::Tuple{Int,Int,Int}; kwargs...) = _Optimize(model, xyp; kwargs...)
function Optimize(M::ModelMap, xyp::Tuple{Int,Int,Int}=M.xyp; inplace::Bool=(xyp[2] > 1), kwargs...)
    model, dmodel = _Optimize(M, xyp; inplace=inplace, kwargs...)
    (!isnothing(model) && !isnothing(dmodel)) ? (ModelMap(model, M; inplace=inplace), ModelMap(dmodel, M; inplace=inplace)) : (nothing, nothing)
end

function _Optimize(model::ModelOrFunction, xyp::Tuple{Int,Int,Int}; inplace::Bool=false, timeout::Real=5, parallel::Bool=false, kwargs...)
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
    ### Pretty Function names --- THROWING AWAY KWARGS HERE!
    if IsJacobian
        SymbolicModelJacobian(x::Union{Number,AbstractVector{<:Number}}, θ::AbstractVector{<:Number}; Kwargs...) = OptimizedModel(x, θ)
        SymbolicModelJacobian!(y::Union{Number,AbstractMatrix{<:Number}}, x::Union{Number,AbstractVector{<:Number}}, θ::AbstractVector{<:Number}; Kwargs...) = OptimizedModel(y, x, θ)
        inplace ? SymbolicModelJacobian! : SymbolicModelJacobian
    else
        SymbolicModel(x::Union{Number,AbstractVector{<:Number}}, θ::AbstractVector{<:Number}; Kwargs...) = OptimizedModel(x, θ)
        SymbolicModel!(y::Union{Number,AbstractVector{<:Number}}, x::Union{Number,AbstractVector{<:Number}}, θ::AbstractVector{<:Number}; Kwargs...) = OptimizedModel(y, x, θ)
        inplace ? SymbolicModel! : SymbolicModel
    end
end


function OptimizedDM(DM::AbstractDataModel; kwargs...)
    model, dmodel = Optimize(DM; kwargs...)
    # Very simple models (ydim=1) typically slower after simplification using ModelingToolkit.jl / Symbolics.jl
    !isnothing(dmodel) ? DataModel(Data(DM), Predictor(DM), dmodel, MLE(DM), LogLikeMLE(DM)) : DM
end

function InplaceDM(DM::AbstractDataModel; inplace::Bool=true, kwargs...)
    model!, dmodel! = Optimize(ModelMappize(DM); inplace=inplace, kwargs...)
    @assert !isnothing(model!) && !isnothing(dmodel!) "Could not create inplace version of given DataModel."
    DataModel(Data(DM), model!, dmodel!, MLE(DM), LogLikeMLE(DM), true)
end
