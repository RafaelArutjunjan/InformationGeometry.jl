


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
For available backends, see `diff_backends()`.
This outputted function has argument structure `(F::Function, x::AbstractVector) -> AbstractVector`.
"""
GetGrad(ADmode::Symbol, args...; kwargs...) = GetGrad(Val(ADmode), args...; kwargs...)
"""
    GetJac(ADmode::Symbol; kwargs...) -> Function
Returns a function which generates jacobians via the method specified by `ADmode`.
For available backends, see `diff_backends()`.
This outputted function has argument structure `(F::Function, x::AbstractVector) -> AbstractMatrix`.
"""
GetJac(ADmode::Symbol, args...; kwargs...) = GetJac(Val(ADmode), args...; kwargs...)
"""
    GetHess(ADmode::Symbol; kwargs...) -> Function
Returns a function which generates hessians via the method specified by `ADmode`.
For available backends, see `diff_backends()`.
This outputted function has argument structure `(F::Function, x::Number) -> AbstractMatrix`.
"""
GetHess(ADmode::Symbol, args...; kwargs...) = GetHess(Val(ADmode), args...; kwargs...)
"""
    GetDoubleJac(ADmode::Symbol; kwargs...) -> Function
Returns second derivatives of a vector-valued function via the method specified by `ADmode`.
For available backends, see `diff_backends()`.
This outputted function has argument structure `(F::Function, x::AbstractVector) -> AbstractArray{3}`.

THIS FEATURE IS STILL EXPERIMENTAL.
"""
GetDoubleJac(ADmode::Symbol, args...; kwargs...) = GetDoubleJac(Val(ADmode), args...; kwargs...)
"""
    GetMatrixJac(ADmode::Symbol; kwargs...) -> Function
Returns second derivatives of an array-valued function via the method specified by `ADmode`.
For available backends, see `diff_backends()`.
This outputted function has argument structure `(F::Function, x::AbstractVector) -> AbstractArray{n+1}` if `F` outputs `AbstractArray{n}`.

THIS FEATURE IS STILL EXPERIMENTAL.
"""
GetMatrixJac(ADmode::Symbol, args...; kwargs...) = GetMatrixJac(Val(ADmode), args...; kwargs...)


# ForwarDiff as standard
GetDeriv(F::Function, args...; kwargs...) = GetDeriv(Val(:ForwardDiff), F, args...; kwargs...)
GetGrad(F::Function, args...; kwargs...) = GetGrad(Val(:ForwardDiff), F, args...; kwargs...)
GetJac(F::Function, args...; kwargs...) = GetJac(Val(:ForwardDiff), F, args...; kwargs...)
GetHess(F::Function, args...; kwargs...) = GetHess(Val(:ForwardDiff), F, args...; kwargs...)
GetDoubleJac(F::Function, args...; kwargs...) = GetDoubleJac(Val(:ForwardDiff), F, args...; kwargs...)
GetMatrixJac(F::Function, args...; kwargs...) = GetMatrixJac(Val(:ForwardDiff), F, args...; kwargs...)



GetDeriv(ADmode::Val; Kwargs...) = EvaluateDerivative(F::Function, X; kwargs...) = _GetDeriv(ADmode; Kwargs...)(F, X; kwargs...)
GetDeriv(ADmode::Val, F::DFunction, args...; Kwargs...) = EvaldF(F)
function GetDeriv(ADmode::Val, F::Function, args...; kwargs...)
    EvaluateDeriv(X::Number) = _GetDeriv(ADmode; kwargs...)(F, X)
    EvaluateDeriv(X::Num) = _GetDerivPass(F, X)
end
function GetDeriv(ADmode::Val{:Symbolic}, F::Function, args...; verbose::Bool=true, kwargs...)
    M = try GetSymbolicDerivative(F, 1, :derivative; kwargs...)   catch;  nothing  end
    if isnothing(M)
        verbose && (@warn "Unable to compute symbolic derivative of $F, falling back to ForwardDiff.")
        GetDeriv(Val(:ForwardDiff), F)
    else M end
end

GetGrad(ADmode::Val; Kwargs...) = EvaluateGradient(F::Function, X; kwargs...) = _GetGrad(ADmode; Kwargs...)(F, X; kwargs...)
GetGrad(ADmode::Val, F::DFunction, args...; Kwargs...) = EvaldF(F)
function GetGrad(ADmode::Val, F::Function, args...; kwargs...)
    EvaluateGradient(X::AbstractVector{<:Number}) = _GetGrad(ADmode; kwargs...)(F, X)
    EvaluateGradient(X::AbstractVector{<:Num}) = _GetGradPass(F, X)
end
function GetGrad(ADmode::Val{:Symbolic}, F::Function, m::Int=GetArgLength(F), args...; verbose::Bool=true, kwargs...)
    M = try GetSymbolicDerivative(F, m, :gradient; kwargs...)   catch;  nothing  end
    if isnothing(M)
        verbose && (@warn "Unable to compute symbolic derivative of $F, falling back to ForwardDiff.")
        GetGrad(Val(:ForwardDiff), F, m, args...)
    else M end
end

GetJac(ADmode::Val; Kwargs...) = EvaluateJacobian(F::Function, X; kwargs...) = _GetJac(ADmode; Kwargs...)(F, X; kwargs...)
GetJac(ADmode::Val, F::DFunction, args...; Kwargs...) = EvaldF(F)
function GetJac(ADmode::Val, F::Function, args...; kwargs...)
    EvaluateJacobian(X::AbstractVector{<:Number}) = _GetJac(ADmode; kwargs...)(F, X)
    EvaluateJacobian(X::AbstractVector{<:Num}) = _GetJacPass(F, X)
end
function GetJac(ADmode::Val{:Symbolic}, F::Function, m::Int=GetArgLength(F), args...; verbose::Bool=true, kwargs...)
    M = try GetSymbolicDerivative(F, m, :jacobian; kwargs...)   catch;  nothing  end
    if isnothing(M)
        verbose && (@warn "Unable to compute symbolic derivative of $F, falling back to ForwardDiff.")
        GetJac(Val(:ForwardDiff), F, m, args...)
    else M end
end

GetHess(ADmode::Val; Kwargs...) = EvaluateHessian(F::Function, X; kwargs...) = _GetHess(ADmode; Kwargs...)(F, X; kwargs...)
GetHess(ADmode::Val, F::DFunction, args...; Kwargs...) = EvalddF(F)
function GetHess(ADmode::Val, F::Function, args...; kwargs...)
    EvaluateHess(X::AbstractVector{<:Number}) = _GetHess(ADmode; kwargs...)(F, X)
    EvaluateHess(X::AbstractVector{<:Num}) = _GetHessPass(F, X)
end
function GetHess(ADmode::Val{:Symbolic}, F::Function, m::Int=GetArgLength(F), args...; verbose::Bool=true, kwargs...)
    M = try GetSymbolicDerivative(F, m, :hessian; kwargs...)   catch;  nothing  end
    if isnothing(M)
        verbose && (@warn "Unable to compute symbolic derivative of $F, falling back to ForwardDiff.")
        GetHess(Val(:ForwardDiff), F, m, args...)
    else M end
end

GetMatrixJac(ADmode::Val; Kwargs...) = EvaluateMatrixJacobian(F::Function, X; kwargs...) = _GetMatrixJac(ADmode; Kwargs...)(F, X; kwargs...)
GetMatrixJac(ADmode::Val, F::DFunction, args...; Kwargs...) = EvaldF(F)

_MakeTuple(Tup::Int) = (Tup,);    _MakeTuple(Tup::Tuple) = Tup
function _SizeTuple(F::Function, m::Int)
    T = try size(F(rand(m))) catch; size(F(rand())) end
    _MakeTuple(T)
end
function GetMatrixJac(ADmode::Val, F::Function, m::Int=GetArgLength(F), f::Tuple=_SizeTuple(F,m), args...; kwargs...)
    EvaluateMatrixJacobian(X::AbstractVector{<:Number}) = reshape(_GetJac(ADmode; kwargs...)(vec∘F, X), f..., m)
    EvaluateMatrixJacobian(X::Number) = reshape(_GetJac(ADmode; kwargs...)(vec∘F∘(z::AbstractVector->z[1]), [X]), f..., m)
    EvaluateMatrixJacobian(X::Union{<:Num,<:AbstractVector{<:Num}}) = _GetMatrixJacPass(F, X)
end
function GetMatrixJac(ADmode::Val{:Symbolic}, F::Function, m::Int=GetArgLength(F), f::Tuple=_SizeTuple(F,m), args...; verbose::Bool=true, kwargs...)
    M = try GetSymbolicDerivative(F, m, :matrixjacobian; kwargs...)   catch;  nothing  end
    if isnothing(M)
        verbose && (@warn "Unable to compute symbolic derivative of $F, falling back to ForwardDiff.")
        GetMatrixJac(Val(:ForwardDiff), F, m, f, args...)
    else M end
end
# For emergencies: needs an extra evaluation of function to determine length(Func(p))
function _GetMatrixJac(ADmode::Val; kwargs...)
    Functor(Func::Function, X::AbstractVector{<:Number}) = reshape(_GetJac(ADmode; kwargs...)(vec∘Func, X), size(Func(X))..., length(X))
    Functor(Func::Function, X::Number) = reshape(_GetJac(ADmode; kwargs...)(vec∘Func∘(z::AbstractVector->z[1]), [X]), size(Func(X))..., 1)
end

GetDoubleJac(ADmode::Val; Kwargs...) = EvaluateDoubleJacobian(F::Function, X; kwargs...) = _GetDoubleJac(ADmode; Kwargs...)(F, X; kwargs...)
GetDoubleJac(ADmode::Val, F::DFunction, args...; Kwargs...) = EvalddF(F)
function GetDoubleJac(ADmode::Val, F::Function, m::Int=GetArgLength(F), f::Int=length(F(rand(m))), args...; kwargs...)
    if f == 1
        EvaluateDoubleJac(X::AbstractVector{<:Number}) = reshape(_GetJac(ADmode; kwargs...)(vec∘(z->_GetJac(ADmode; kwargs...)(F,z)), X), m, m)
        EvaluateDoubleJac(X::AbstractVector{<:Num}) = _GetDoubleJacPass(F, X)
    else
        EvaluateDoubleJacobian(X::AbstractVector{<:Number}) = reshape(_GetJac(ADmode; kwargs...)(vec∘(z->_GetJac(ADmode; kwargs...)(F,z)), X), f, m, m)
        EvaluateDoubleJacobian(X::AbstractVector{<:Num}) = _GetDoubleJacPass(F, X)
    end
end
function GetDoubleJac(ADmode::Val{:Symbolic}, F::Function, m::Int=GetArgLength(F), f::Int=length(F(rand(m))), args...; verbose::Bool=true, kwargs...)
    M = try GetSymbolicDerivative(F, m, :doublejacobian; kwargs...)   catch;  nothing  end
    if isnothing(M)
        verbose && (@warn "Unable to compute symbolic derivative of $F, falling back to ForwardDiff.")
        GetDoubleJac(Val(:ForwardDiff), F, m, f, args...)
    else M end
end
# For emergencies: needs an extra evaluation of function to determine length(Func(p))
function _GetDoubleJac(ADmode::Val; kwargs...)
    Functor(Func::Function, X) = reshape(_GetJac(ADmode; kwargs...)(vec∘(z->_GetJac(ADmode; kwargs...)(Func,z)), X), length(Func(X)), length(X), length(X))
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
_GetGrad(ADmode::Val{:FiniteDiff}; order::Int=3, kwargs...) = (Func::Function,p;Kwargs...) -> FiniteDifferences.grad(central_fdm(order,1), Func, p; kwargs...)[1]
_GetJac(ADmode::Val{:FiniteDiff}; order::Int=3, kwargs...) = (Func::Function,p;Kwargs...) -> FiniteDifferences.jacobian(central_fdm(order,1), Func, p; kwargs...)[1]
_GetHess(ADmode::Val{:FiniteDiff}; order::Int=5, kwargs...) = (Func::Function,p;Kwargs...) -> FiniteDifferences.jacobian(central_fdm(order,1), z->FiniteDifferences.grad(central_fdm(order,1), Func, z)[1], p)[1]


## User has passed either Num or Vector{Num} to function, try to perfom symbolic passthrough
# Still need to extend this to functions F which are in-place
_GetDerivPass(F::Function, X) = SymbolicPassthrough(F(X), X, :derivative)
_GetGradPass(F::Function, X) = SymbolicPassthrough(F(X), X, :gradient)
_GetJacPass(F::Function, X) = SymbolicPassthrough(F(X), X, :jacobian)
_GetHessPass(F::Function, X) = SymbolicPassthrough(F(X), X, :hessian)

_GetDoubleJacPass(F::Function, X) = SymbolicPassthrough(F(X), X, :doublejacobian)
_GetMatrixJacPass(F::Function, X) = SymbolicPassthrough(F(X), X, :matrixjacobian)



GetGrad!(ADmode::Symbol, args...; kwargs...) = GetGrad!(Val(ADmode), args...; kwargs...)
GetJac!(ADmode::Symbol, args...; kwargs...) = GetJac!(Val(ADmode), args...; kwargs...)
GetHess!(ADmode::Symbol, args...; kwargs...) = GetHess!(Val(ADmode), args...; kwargs...)
GetMatrixJac!(ADmode::Symbol, args...; kwargs...) = GetMatrixJac!(Val(ADmode), args...; kwargs...)

# No Passthrough for these
GetGrad!(ADmode::Val, args...; kwargs...) = _GetGrad!(ADmode, args...; kwargs...)
GetJac!(ADmode::Val, args...; kwargs...) = _GetJac!(ADmode, args...; kwargs...)
GetHess!(ADmode::Val, args...; kwargs...) = _GetHess!(ADmode, args...; kwargs...)
GetMatrixJac!(ADmode::Val, args...; kwargs...) = _GetMatrixJac!(ADmode, args...; kwargs...)

# Evaluation of differentation operations into pre-specified arrays for functions which are themselves out-of-place
function GetGrad!(ADmode::Val, F::Function; kwargs...)
    EvaluateGradient!(Y::AbstractVector{<:Number}, X::AbstractVector{<:Number}) = _GetGrad!(ADmode; kwargs...)(Y, F, X)
    EvaluateGradient!(Y::AbstractVector{<:Num}, X::AbstractVector{<:Num}) = _GetGradPass!(Y, F, X)
end
function GetJac!(ADmode::Val, F::Function; kwargs...)
    EvaluateJacobian!(Y::AbstractMatrix{<:Number}, X::AbstractVector{<:Number}) = _GetJac!(ADmode; kwargs...)(Y, F, X)
    EvaluateJacobian!(Y::AbstractMatrix{<:Num}, X::AbstractVector{<:Num}) = _GetJacPass!(Y, F, X)
end
function GetHess!(ADmode::Val, F::Function; kwargs...)
    EvaluateHess!(Y::AbstractMatrix{<:Number}, X::AbstractVector{<:Number}) = _GetHess!(ADmode; kwargs...)(Y, F, X)
    EvaluateHess!(Y::AbstractMatrix{<:Num}, X::AbstractVector{<:Num}) = _GetHessPass!(Y, F, X)
end

# Looks like for in-place functor jacobian! there is no difference for any array shape
function GetMatrixJac!(ADmode::Val, F::Function; kwargs...)
    EvaluateMatrixJacobian(Y::AbstractArray{<:Number}, X::AbstractVector{<:Number}) = _GetMatrixJac!(ADmode; kwargs...)(Y, F, X)
    EvaluateMatrixJacobian(Y::AbstractArray{<:Num}, X::AbstractVector{<:Num}) = _GetMatrixJacPass!(Y, F, X)
end

# Fall back to ForwardDiff as standard
_GetGrad!(ADmode::Val{true}; kwargs...) = _GetGrad!(Val(:ForwardDiff); kwargs...)
_GetJac!(ADmode::Val{true}; kwargs...) = _GetJac!(Val(:ForwardDiff); kwargs...)
_GetHess!(ADmode::Val{true}; kwargs...) = _GetHess!(Val(:ForwardDiff); kwargs...)
_GetMatrixJac!(ADmode::Val{true}; kwargs...) = _GetMatrixJac!(Val(:ForwardDiff); kwargs...)

_GetGrad!(ADmode::Val{:ForwardDiff}; kwargs...) = ForwardDiff.gradient!
_GetJac!(ADmode::Val{:ForwardDiff}; kwargs...) = ForwardDiff.jacobian!
_GetHess!(ADmode::Val{:ForwardDiff}; kwargs...) = ForwardDiff.hessian!
_GetMatrixJac!(ADmode::Val{:ForwardDiff}; kwargs...) = _GetJac!(ADmode; kwargs...) # DELIBERATE!!!! _GetJac!() recognizes output format from given Array

_GetGrad!(ADmode::Val{:ReverseDiff}; kwargs...) = ReverseDiff.gradient!
_GetJac!(ADmode::Val{:ReverseDiff}; kwargs...) = ReverseDiff.jacobian!
_GetHess!(ADmode::Val{:ReverseDiff}; kwargs...) = ReverseDiff.hessian!
_GetMatrixJac!(ADmode::Val{:ReverseDiff}; kwargs...) = _GetJac!(ADmode; kwargs...) # DELIBERATE!!!! _GetJac!() recognizes output format from given Array

# Fake in-place
function _GetGrad!(ADmode::Union{<:Val{:Zygote},<:Val{:FiniteDiff}}; verbose::Bool=true, kwargs...)
    verbose && (@warn "Using fake in-place differentiation operator for ADmode=$ADmode because backend does not supply appropriate method.")
    FakeInPlaceGrad!(Y::AbstractVector,F::Function,X::AbstractVector) = (Y .= _GetGrad(ADmode; kwargs...)(F, X))
end
function _GetJac!(ADmode::Union{Val{:Zygote},Val{:FiniteDiff}}; verbose::Bool=true, kwargs...)
    verbose && (@warn "Using fake in-place differentiation operator for ADmode=$ADmode because backend does not supply appropriate method.")
    FakeInPlaceJac!(Y::AbstractMatrix,F::Function,X::AbstractVector) = (Y .= _GetJac(ADmode; kwargs...)(F, X))
end
function _GetHess!(ADmode::Union{Val{:Zygote},Val{:FiniteDiff}}; verbose::Bool=true, kwargs...)
    verbose && (@warn "Using fake in-place differentiation operator for ADmode=$ADmode because backend does not supply appropriate method.")
    FakeInPlaceHess!(Y::AbstractMatrix,F::Function,X::AbstractVector) = (Y .= _GetHess(ADmode; kwargs...)(F, X))
end
function _GetMatrixJac!(ADmode::Union{Val{:Zygote},Val{:FiniteDiff}}; verbose::Bool=true, kwargs...)
    verbose && (@warn "Using fake in-place differentiation operator for ADmode=$ADmode because backend does not supply appropriate method.")
    FakeInPlaceMatrixJac!(Y::AbstractArray,F::Function,X::AbstractVector) = (Y[:] .= vec(_GetJac(ADmode; kwargs...)(F, X)))
end

# Need to extend this to functions F which are themselves also in-place
_GetGradPass!(Y, F::Function, X) = (Y.=SymbolicPassthrough(F(X), X, :gradient))
_GetJacPass!(Y, F::Function, X) = (Y.=SymbolicPassthrough(F(X), X, :jacobian))
_GetHessPass!(Y, F::Function, X) = (Y.=SymbolicPassthrough(F(X), X, :hessian))
_GetMatrixJacPass!(Y, F::Function, X) = (Y.=SymbolicPassthrough(F(X), X, :matrixjacobian))




_ConsistencyCheck(Fexpr, var, deriv::Symbol) = _ConsistencyCheck(Fexpr, var, Val(deriv))
_ConsistencyCheck(Fexpr::AbstractVector{<:Num}, var::AbstractVector{<:Num}, ::Union{Val{:jacobian},Val{:doublejacobian}}) = nothing
_ConsistencyCheck(Fexpr::Num, var::AbstractVector{<:Num}, ::Union{Val{:gradient},Val{:hessian}}) = nothing
_ConsistencyCheck(Fexpr::Num, var::Num, ::Val{:derivative}) = nothing
_ConsistencyCheck(Fexpr::AbstractArray{<:Num}, var::Union{<:Num,<:AbstractVector{<:Num}}, ::Val{:matrixjacobian}) = nothing
function _ConsistencyCheck(Fexpr, var, deriv::Val{T}) where T
    if T ∉ [:derivative, :gradient, :jacobian, :hessian, :doublejacobian, :matrixjacobian]
        throw("Invalid deriv type: $T.")
    else
        throw("Got :$T but Fexpr=$(typeof(Fexpr)) and arg=$(typeof(var)).")
    end
end

"""
Executes symbolic derivative as specified by `deriv::Symbol`.
"""
function SymbolicPassthrough(Fexpr::Union{<:AbstractArray{<:Num},<:Num}, var::Union{<:AbstractVector{<:Num},<:Num}, deriv::Symbol=:jacobian; simplify::Bool=true)
    _ConsistencyCheck(Fexpr, var, deriv)

    SymbolicDoubleJacobian(V::AbstractVector{<:Num}, z::Num; simplify::Bool=true) = SymbolicDoubleJacobian(V, [z]; simplify=simplify)
    SymbolicDoubleJacobian(V::AbstractVector{<:Num}, z::AbstractVector{<:Num}; simplify::Bool=true) = SymbolicMatrixJacobian(Symbolics.jacobian(V,z),z; simplify=simplify)
    SymbolicMatrixJacobian(M::AbstractArray{<:Num}, z::Num; simplify::Bool=true) = SymbolicMatrixJacobian(M, [z]; simplify=simplify)
    function SymbolicMatrixJacobian(M::AbstractArray{<:Num}, z::AbstractVector{<:Num}; simplify::Bool=true)
        reshape(Symbolics.jacobian(vec(M), z; simplify=simplify), size(M)..., length(z))
    end

    if deriv == :doublejacobian
        SymbolicDoubleJacobian(Fexpr, var; simplify=simplify)
    elseif deriv == :matrixjacobian
        SymbolicMatrixJacobian(Fexpr, var; simplify=simplify)
    else
        (@eval Symbolics.$deriv)(Fexpr, var; simplify=simplify)
    end
end
"""
    GetSymbolicDerivative(F::Function, inputdim::Int=GetArgLength(F), deriv::Symbol=:jacobian; timeout::Real=5, inplace::Bool=false, parallel::Bool=false)
Computes symbolic derivatives, including `:jacobian`, `:gradient`, `:hessian` and `:derivative` which are specified via `deriv`.
Special care has to be taken that the correct `inputdim` is specified! Silent errors may occur otherwise.
"""
function GetSymbolicDerivative(F::Function, inputdim::Int=GetArgLength(F), deriv::Symbol=:jacobian; timeout::Real=5, kwargs...)
    @assert deriv ∈ [:derivative, :gradient, :jacobian, :hessian, :doublejacobian, :matrixjacobian]
    @variables x X[1:inputdim]
    var = inputdim > 1 ? X : (try F(rand()); x catch; X end)
    Fexpr = KillAfter(F, var; timeout=timeout)
    # Warning already thrown in KillAfter
    isnothing(Fexpr) && return nothing
    GetSymbolicDerivative(Fexpr, var, deriv; kwargs...)
end
GetSymbolicDerivative(F::Function, deriv::Symbol; kwargs...) = GetSymbolicDerivative(F, GetArgLength(F), deriv; kwargs...)

function GetSymbolicDerivative(Fexpr::Union{<:AbstractArray{<:Num},<:Num}, var::Union{<:AbstractVector{<:Num},<:Num}, deriv::Symbol=:jacobian; simplify::Bool=true, inplace::Bool=false, parallel::Bool=false, kwargs...)
    derivative = SymbolicPassthrough(Fexpr, var, deriv; simplify=simplify)
    Builder(derivative, var; parallel=parallel, inplace=inplace, kwargs...)
end
