


diff_backends() = [:Symbolic, :ForwardDiff, :ReverseDiff, :Zygote, :FiniteDiff]

abstract type DFunction <: Function end

## Differentiation
"""
    GetDeriv(ADmode::Symbol; kwargs...) -> Function
Returns a function which generates gradients via the method specified by `ADmode`.
`GetDeriv()` currently only available via ForwardDiff.
This outputted function has argument structure `(F::Function, x::Number) -> Number`.
"""
GetDeriv(ADmode::Symbol, args...; kwargs...) = GetDeriv(Val(ADmode), args...; kwargs...)
"""
    GetGrad(ADmode::Symbol; kwargs...) -> Function
Returns a function which generates gradients via the method specified by `ADmode`.
For available backends, see `InformationGeometry.diff_backends()`.
This outputted function has argument structure `(F::Function, x::AbstractVector) -> AbstractVector`.
"""
GetGrad(ADmode::Symbol, args...; kwargs...) = GetGrad(Val(ADmode), args...; kwargs...)
"""
    GetJac(ADmode::Symbol; kwargs...) -> Function
Returns a function which generates jacobians via the method specified by `ADmode`.
For available backends, see `InformationGeometry.diff_backends()`.
This outputted function has argument structure `(F::Function, x::AbstractVector) -> AbstractMatrix`.
"""
GetJac(ADmode::Symbol, args...; kwargs...) = GetJac(Val(ADmode), args...; kwargs...)
"""
    GetHess(ADmode::Symbol; kwargs...) -> Function
Returns a function which generates hessians via the method specified by `ADmode`.
For available backends, see `InformationGeometry.diff_backends()`.
This outputted function has argument structure `(F::Function, x::Number) -> AbstractMatrix`.
"""
GetHess(ADmode::Symbol, args...; kwargs...) = GetHess(Val(ADmode), args...; kwargs...)
"""
    GetDoubleJac(ADmode::Symbol; kwargs...) -> Function
Returns second derivatives of a vector-valued function via the method specified by `ADmode`.
For available backends, see `InformationGeometry.diff_backends()`.
This outputted function has argument structure `(F::Function, x::AbstractVector) -> AbstractArray{3}`.

THIS FEATURE IS STILL EXPERIMENTAL.
"""
GetDoubleJac(ADmode::Symbol, args...; kwargs...) = GetDoubleJac(Val(ADmode), args...; kwargs...)



GetDeriv(ADmode::Val; Kwargs...) = EvaluateDerivative(F::Function, X; kwargs...) = _GetDeriv(ADmode; Kwargs...)(F, X; kwargs...)
GetDeriv(ADmode::Val, F::Function; Kwargs...) = EvaluateDeriv(X) = _GetDeriv(ADmode::Val; Kwargs...)(F, X)
GetDeriv(ADmode::Val, F::DFunction; Kwargs...) = EvaldF(F)
function GetDeriv(ADmode::Val{:Symbolic}, F::Function; kwargs...)
    M = try GetSymbolicDerivative(F,1,:derivative; kwargs...)   catch;  nothing  end
    if isnothing(M)
        @warn "Unable to compute symbolic derivative of $F, falling back to ForwardDiff."
        GetDeriv(Val(:ForwardDiff), F)
    else M end
end

GetGrad(ADmode::Val; Kwargs...) = EvaluateGradient(F::Function, X; kwargs...) = _GetGrad(ADmode; Kwargs...)(F, X; kwargs...)
GetGrad(ADmode::Val, F::Function; Kwargs...) = EvaluateGradient(X) = _GetGrad(ADmode::Val; Kwargs...)(F, X)
GetGrad(ADmode::Val, F::DFunction; Kwargs...) = EvaldF(F)
function GetGrad(ADmode::Val{:Symbolic}, F::Function; kwargs...)
    M = try GetSymbolicDerivative(F,:gradient; kwargs...)   catch;  nothing  end
    if isnothing(M)
        @warn "Unable to compute symbolic derivative of $F, falling back to ForwardDiff."
        GetGrad(Val(:ForwardDiff), F)
    else M end
end

GetJac(ADmode::Val; Kwargs...) = EvaluateJacobian(F::Function, X; kwargs...) = _GetJac(ADmode; Kwargs...)(F, X; kwargs...)
GetJac(ADmode::Val, F::Function; Kwargs...) = EvaluateJacobian(X) = _GetJac(ADmode::Val; Kwargs...)(F, X)
GetJac(ADmode::Val, F::DFunction; Kwargs...) = EvaldF(F)
function GetJac(ADmode::Val{:Symbolic}, F::Function; kwargs...)
    M = try GetSymbolicDerivative(F,:jacobian; kwargs...)   catch;  nothing  end
    if isnothing(M)
        @warn "Unable to compute symbolic derivative of $F, falling back to ForwardDiff."
        GetJac(Val(:ForwardDiff), F)
    else M end
end

GetHess(ADmode::Val; Kwargs...) = EvaluateHessian(F::Function, X; kwargs...) = _GetHess(ADmode; Kwargs...)(F, X; kwargs...)
GetHess(ADmode::Val, F::Function; Kwargs...) = EvaluateHess(X) = _GetHess(ADmode::Val; Kwargs...)(F, X)
GetHess(ADmode::Val, F::DFunction; Kwargs...) = EvalddF(F)
function GetHess(ADmode::Val{:Symbolic}, F::Function; kwargs...)
    M = try GetSymbolicDerivative(F,:hessian; kwargs...)   catch;  nothing  end
    if isnothing(M)
        @warn "Unable to compute symbolic derivative of $F, falling back to ForwardDiff."
        GetHess(Val(:ForwardDiff), F)
    else M end
end

GetDoubleJac(ADmode::Val; Kwargs...) = EvaluateDoubleJacobian(F::Function, X; kwargs...) = _GetDoubleJac(ADmode; Kwargs...)(F, X; kwargs...)
GetDoubleJac(ADmode::Val, F::DFunction; Kwargs...) = EvalddF(F)
function GetDoubleJac(ADmode::Val, F::Function; Kwargs...)
    m = GetArgLength(F);    f = length(F(ones(m)))
    if f == 1
        EvaluateDoubleJac(p) = reshape(_GetJac(ADmode)(vec∘(z->_GetJac(ADmode)(F,z)), p), m, m)
    else
        EvaluateDoubleJacobian(p) = reshape(_GetJac(ADmode)(vec∘(z->_GetJac(ADmode)(F,z)), p), f, m, m)
    end
end
function GetDoubleJac(ADmode::Val{:Symbolic}, F::Function; kwargs...)
    M = try GetSymbolicDerivative(F,:doublejacobian; kwargs...)   catch;  nothing  end
    if isnothing(M)
        @warn "Unable to compute symbolic derivative of $F, falling back to ForwardDiff."
        GetDoubleJac(Val(:ForwardDiff), F)
    else M end
end

function _GetDoubleJac(ADmode::Val; kwargs...)
    Functor(Func::Function, p) = reshape(_GetJac(ADmode)(vec∘(z->_GetJac(ADmode)(Func,z)), p), length(Func(p)), length(p), length(p))
end



# Fall back to ForwardDiff as standard
_GetDeriv(ADmode::Val{true}; kwargs...) = _GetDeriv(Val(:ForwardDiff); kwargs...)
_GetGrad(ADmode::Val{true}; kwargs...) = _GetGrad(Val(:ForwardDiff); kwargs...)
_GetJac(ADmode::Val{true}; kwargs...) = _GetJac(Val(:ForwardDiff); kwargs...)
_GetHess(ADmode::Val{true}; kwargs...) = _GetHess(Val(:ForwardDiff); kwargs...)
_GetDoubleJac(ADmode::Val{true}; kwargs...) = _GetDoubleJac(Val(:ForwardDiff); kwargs...)


_GetDeriv(ADmode::Val{:ForwardDiff}; kwargs...) = ForwardDiff.derivative
_GetGrad(ADmode::Val{:ForwardDiff}; kwargs...) = ForwardDiff.gradient
_GetJac(ADmode::Val{:ForwardDiff}; kwargs...) = ForwardDiff.jacobian
_GetHess(ADmode::Val{:ForwardDiff}; kwargs...) = ForwardDiff.hessian

# Deriv not available for ReverseDiff
_GetGrad(ADmode::Val{:ReverseDiff}; kwargs...) = ReverseDiff.gradient
_GetJac(ADmode::Val{:ReverseDiff}; kwargs...) = ReverseDiff.jacobian
_GetHess(ADmode::Val{:ReverseDiff}; kwargs...) = ReverseDiff.hessian
# Deriv not available for Zygote
_GetGrad(ADmode::Val{:Zygote}; order::Int=-1, kwargs...) = (Func::Function,p;Kwargs...) -> Zygote.gradient(Func, p; kwargs...)[1]
_GetJac(ADmode::Val{:Zygote}; order::Int=-1, kwargs...) = (Func::Function,p;Kwargs...) -> Zygote.jacobian(Func, p; kwargs...)[1]
_GetHess(ADmode::Val{:Zygote}; order::Int=-1, kwargs...) = (Func::Function,p;Kwargs...) -> Zygote.hessian(Func, p; kwargs...)
# Deriv not available for FiniteDifferences
_GetGrad(ADmode::Val{:FiniteDiff}; order::Int=2, kwargs...) = (Func::Function,p;Kwargs...) -> FiniteDifferences.grad(central_fdm(order,1), Func, p; kwargs...)[1]
_GetJac(ADmode::Val{:FiniteDiff}; order::Int=2, kwargs...) = (Func::Function,p;Kwargs...) -> FiniteDifferences.jacobian(central_fdm(order,1), Func, p; kwargs...)[1]
_GetHess(ADmode::Val{:FiniteDiff}; order::Int=5, kwargs...) = (Func::Function,p;Kwargs...) -> FiniteDifferences.jacobian(central_fdm(order,1), z->FiniteDifferences.grad(central_fdm(order,1), Func, z)[1], p)[1]



GetGrad!(ADmode::Symbol; kwargs...) = GetGrad!(Val(ADmode); kwargs...)
GetJac!(ADmode::Symbol; kwargs...) = GetJac!(Val(ADmode); kwargs...)
GetHess!(ADmode::Symbol; kwargs...) = GetHess!(Val(ADmode); kwargs...)
# Fall back to ForwarDiff as standard
GetGrad!(ADmode::Val{true}; kwargs...) = GetGrad!(Val(:ForwardDiff); kwargs...)
GetJac!(ADmode::Val{true}; kwargs...) = GetJac!(Val(:ForwardDiff); kwargs...)
GetHess!(ADmode::Val{true}; kwargs...) = GetHess!(Val(:ForwardDiff); kwargs...)

GetGrad!(ADmode::Val{:ForwardDiff}; kwargs...) = ForwardDiff.gradient!
GetJac!(ADmode::Val{:ForwardDiff}; kwargs...) = ForwardDiff.jacobian!
GetHess!(ADmode::Val{:ForwardDiff}; kwargs...) = ForwardDiff.hessian!
GetGrad!(ADmode::Val{:ReverseDiff}; kwargs...) = ReverseDiff.gradient!
GetJac!(ADmode::Val{:ReverseDiff}; kwargs...) = ReverseDiff.jacobian!
GetHess!(ADmode::Val{:ReverseDiff}; kwargs...) = ReverseDiff.hessian!


"""
    GetSymbolicDerivative(F::Function, inputdim::Int=GetArgLength(F), deriv::Symbol=:jacobian; timeout::Real=5, inplace::Bool=false, parallel::Bool=false)
Computes symbolic derivatives, including `:jacobian`, `:gradient`, `:hessian` and `:derivative` which are specified via `deriv`.
Special care has to be taken that the correct `inputdim` is specified! Silent errors may occur otherwise.
"""
function GetSymbolicDerivative(F::Function, inputdim::Int=GetArgLength(F), deriv::Symbol=:jacobian; timeout::Real=5, kwargs...)
    @assert deriv ∈ [:derivative, :gradient, :jacobian, :hessian, :doublejacobian]
    @variables x X[1:inputdim]
    var = inputdim > 1 ? X : (try F(1.0); x catch; X end)
    Fexpr = KillAfter(F, var; timeout=timeout)
    # Warning already thrown in KillAfter
    isnothing(Fexpr) && return nothing
    GetSymbolicDerivative(Fexpr, var, deriv; kwargs...)
end
GetSymbolicDerivative(F::Function, deriv::Symbol; kwargs...) = GetSymbolicDerivative(F, GetArgLength(F), deriv; kwargs...)
function GetSymbolicDerivative(Fexpr::Union{<:AbstractVector{<:Num},<:Num}, var::Union{<:AbstractVector{<:Num},<:Num}, deriv::Symbol=:jacobian; inplace::Bool=false, parallel::Bool=false, kwargs...)
    if deriv == :jacobian || deriv == :doublejacobian
        @assert (Fexpr isa AbstractVector{<:Num}) && (var isa AbstractVector{<:Num}) "Got $deriv but Fexpr=$(typeof(Fexpr)) and argument=$(typeof(var))."
    elseif deriv == :gradient || deriv == :hessian
        @assert (Fexpr isa Num) && (var isa AbstractVector{<:Num}) "Got $deriv but Fexpr=$(typeof(Fexpr)) and argument=$(typeof(var))."
    elseif deriv == :derivative
        @assert (Fexpr isa Num) && (var isa Num) "Got $deriv but Fexpr=$(typeof(Fexpr)) and argument=$(typeof(var))."
    else
        throw("Invalid deriv type: $deriv.")
    end
    SymbolicDoubleJacobian(V::AbstractVector{<:Num}, z::AbstractVector{<:Num}) = SecondJacobian(Symbolics.jacobian(V,z),z)
    function SecondJacobian(M::AbstractMatrix{<:Num}, z::AbstractVector{<:Num})
        reshape(Symbolics.jacobian(vec(M), z), size(M,1), size(M,2), size(M,2))
    end

    derivative = if deriv == :doublejacobian
        SymbolicDoubleJacobian(Fexpr, var)
    else
        (@eval Symbolics.$deriv)(Fexpr, var; simplify=true)
    end
    Builder(derivative, var; parallel=parallel, inplace=inplace, kwargs...)
end
