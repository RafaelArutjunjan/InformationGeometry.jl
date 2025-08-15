


Getxyp(DM::AbstractDataModel) = (xdim(DM), ydim(DM), pdim(DM))
Getxyp(DS::AbstractDataSet, model::Function) = (xdim(DS),ydim(DS),pdim(DS,model))
Getxyp(DS::AbstractDataSet, M::ModelMap) = M.xyp
Getxyp(M::ModelMap) = M.xyp

SymbolicArguments(args...) = SymbolicArguments(Getxyp(args...))
function SymbolicArguments(xyp::Tuple{Int,Int,Int})
    (xyp[1] == 1 ? (@variables x)[1] : (@variables x[1:xyp[1]])[1]),
    (xyp[2] == 1 ? (@variables y)[1] : (@variables y[1:xyp[2]])[1]),
    (@variables θ[1:xyp[3]])[1]
end

"""
    SpeedifyTransform(F::Function, X::AbstractArray{<:Num}; inplace::Bool=false) -> Function
Takes function and runs it through Symbolics toolbox for performance optimization.
"""
function SpeedifyTransform(F::Function, X::AbstractArray{<:DerivableFunctionsBase.SymbolicScalar}=(Symbolics.@variables x[1:GetArgLength(F; max=MaxArgLen)])[1], args...; inplace::Bool=false, kwargs...)
    Symbolics.build_function(F(X, args...), X, args...; expression=Val{false}, kwargs...)[inplace ? 2 : 1]
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
    xyp != M.xyp && @warn "Inconsistent xyp information: given $xyp, ModelMap contains $(M.xyp) - using given xyp."
    X, Y, θ = SymbolicArguments(xyp)
    isinplacemodel(M) ? (Res=Vector{Num}(undef,length(Y));  KillAfter(M.Map, Res, X, θ; timeout=timeout); Res) : KillAfter(M.Map, X, θ; timeout=timeout)
end

## Alternatively: (need to catch LaTeXStrings before this to avoid $ though)
MakeMTKVariables(S::Symbol) = eval(Meta.parse("@variables "*string(S)))[1];      MakeMTKVariables(S::AbstractArray{<:Symbol}) = [MakeMTKVariables(s) for s in S]
# MakeMTKParameters(S::Symbol) = eval(Meta.parse("@parameters "*string(S)))[1];      MakeMTKParameters(S::AbstractArray{<:Symbol}) = [MakeMTKParameters(s) for s in S]
MakeSymbolicPars(X::AbstractVector) = MakeMTKVariables(X) # eval(ModelingToolkit._parse_vars(:parameters, Real, X, ModelingToolkit.toparam))

# Apply String to make LaTeXStrings into strings as well
SafeNames(X::AbstractVector) = all(SafeNames, X)
# const ProblematicChars = ["(", "[", "{", "=", "\$"]
const ProblematicCharsRegex = r"[()\[\]\{\}=\$]"
SafeNames(X::AbstractString) = !occursin(ProblematicCharsRegex, String(X))
SafeNames(X::Symbol) = SafeNames(string(X))


SymbolicModelExpr(DM::Union{AbstractDataModel,ModelMap}) = @suppress_err ToExpr(DM)
"""
    SymbolicModel(DM::Union{AbstractDataModel,ModelMap}; sub::Bool=true)
Produces `String` of symbolic expression for the model map if possible.

Kwarg `sub` controls whether the given variable names are taken from `DM` (`sub=true`)
or whether generic variable names, e.g. `θ[1]` are used (`sub=false`) for better copyability.
"""
function SymbolicModel(DM::Union{AbstractDataModel,ModelMap}; sub::Bool=!(DM isa ModelMap) && SafeNames(pnames(DM)) && SafeNames(xnames(DM)) && SafeNames(ynames(DM)), verbose::Bool=true)
    expr = SymbolicModelExpr(DM)
    isnothing(expr) && return "Unable to represent given model symbolically."
    # substitute symbol names with xnames, ynames and pnames?
    if sub
        try
            xold, yold, pold = SymbolicArguments(DM)

            xnew, ynew = if DM isa AbstractDataModel
                MakeSymbolicPars(Xnames(DM)), MakeSymbolicPars(Ynames(DM))
            else
                xold, yold
            end
            pnew = MakeSymbolicPars(Pnames(DM))

            Viewer(X::Symbolics.Arr{<:Num, 1}) = view(X,:);    Viewer(X::Num) = X
            expr = substitute(expr, Dict([Viewer(xold) .=> xnew; Viewer(yold) .=> ynew; Viewer(pold) .=> pnew]); fold=false)
        catch E;
            verbose && @warn "Could not use variable given names due to: $E"
        end
    end
    "y(x,θ) = " * string(expr)
end

function SymbolicdModelExpr(DM::AbstractDataModel)
    if !GeneratedFromSymbolic(dPredictor(DM))
        @warn "Given Model jacobian not symbolic. Trying to apply OptimizedDM() first."
        odm = OptimizedDM(DM)
        X, Y, θ = SymbolicArguments(DM)
        return isnothing(ToExpr(odm)) ? nothing : dPredictor(odm)(X, θ)
    end
    SymbolicdModelExpr(dPredictor(DM))
end
function SymbolicdModelExpr(M::ModelMap)
    X, Y, θ = SymbolicArguments(M)
    isinplacemodel(M) ? (Res=Matrix{Num}(undef,length(Y),length(θ));  M(Res,X,θ)) : M(X, θ)
end

"""
    SymbolicdModel(DM::Union{AbstractDataModel,ModelMap}; sub::Bool=true)
Produces `String` of symbolic expression for the model map if possible.

Kwarg `sub` controls whether the given variable names are taken from `DM` (`sub=true`)
or whether generic variable names, e.g. `θ[1]` are used (`sub=false`) for better copyability.
"""
function SymbolicdModel(DM::Union{AbstractDataModel,ModelMap}; sub::Bool=!(DM isa ModelMap) && SafeNames(pnames(DM)) && SafeNames(xnames(DM)) && SafeNames(ynames(DM)), verbose::Bool=true)
    expr = SymbolicdModelExpr(DM)
    isnothing(expr) && return "Unable to represent given model symbolically."
    # substitute symbol names with xnames, ynames and pnames?
    if sub
        try
            xold, yold, pold = SymbolicArguments(DM)

            xnew, ynew = if DM isa AbstractDataModel
                MakeSymbolicPars(Xnames(DM)), MakeSymbolicPars(Ynames(DM))
            else
                xold, yold
            end
            pnew = MakeSymbolicPars(Pnames(DM))

            Viewer(X::Symbolics.Arr{<:Num, 1}) = view(X,:);    Viewer(X::Num) = X
            expr = substitute(expr, Dict([Viewer(xold) .=> xnew; Viewer(yold) .=> ynew; Viewer(pold) .=> pnew]); fold=false)
        catch E;
            verbose && @warn "Could not use variable given names due to: $E"
        end
    end
    "(∂y/∂θ)(x,θ) = " * string(expr)
end

# mixing inplace (outcome) with isinplace (input) here. Disambiguate!

OptimizeModel(DM::AbstractDataModel; kwargs...) = OptimizeModel(Data(DM), Predictor(DM); kwargs...)
OptimizeModel(DS::AbstractDataSet, model::ModelOrFunction; kwargs...) = OptimizeModel(model, Getxyp(DS, model); kwargs...)

OptimizeModel(model::Function, xyp::Tuple{Int,Int,Int}; kwargs...) = _OptimizeModel(model, xyp; kwargs...)
function OptimizeModel(M::ModelMap, xyp::Tuple{Int,Int,Int}=M.xyp; inplace::Bool=(xyp[2] > 1), kwargs...)
    model, dmodel = _OptimizeModel(M, xyp; inplace=inplace, kwargs...)
    (!isnothing(model) && !isnothing(dmodel)) ? (ModelMap(model, M; inplace=(inplace && (xyp[2] > 1))), ModelMap(dmodel, M; inplace=inplace)) : (nothing, nothing)
end

function _OptimizeModel(model::ModelOrFunction, xyp::Tuple{Int,Int,Int}; inplace::Bool=false, timeout::Real=5, parallel::Bool=false, kwargs...)
    modelexpr = ToExpr(model, xyp; timeout=timeout) |> Symbolics.simplify
    isnothing(modelexpr) && return nothing, nothing

    X, Y, θ = SymbolicArguments(xyp)
    # Need to make sure that modelexpr is of type Vector{Num}, not just Num
    !(modelexpr isa AbstractVector{<:Num}) && (modelexpr = [modelexpr])
    derivative = Symbolics.jacobian(modelexpr, θ; simplify=true)

    inplace && (xyp[2] == 1) && @warn "Although given inplace=true, will nevertheless create out-of-place version for model because ydim=1. Model jacobian will be in-place."

    ExprToModelMap(X, θ, modelexpr; inplace=(inplace && (xyp[2] > 1)), parallel=parallel, IsJacobian=false, kwargs...), ExprToModelMap(X, θ, derivative; inplace=inplace, parallel=parallel, IsJacobian=true, kwargs...)
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
    model, dmodel = OptimizeModel(DM; kwargs...)
    # Very simple models (ydim=1) typically slower after simplification using ModelingToolkit.jl / Symbolics.jl
    !isnothing(dmodel) ? DataModel(Data(DM), Predictor(DM), dmodel, MLE(DM), LogLikeMLE(DM), LogPrior(DM)) : DM
end

function InplaceDM(DM::AbstractDataModel; inplace::Bool=true, kwargs...)
    model!, dmodel! = OptimizeModel(ModelMappize(DM); inplace=inplace, kwargs...)
    @assert !isnothing(model!) && !isnothing(dmodel!) "Could not create inplace version of given DataModel."
    DataModel(Data(DM), model!, dmodel!, MLE(DM), LogLikeMLE(DM), LogPrior(DM), true)
end
