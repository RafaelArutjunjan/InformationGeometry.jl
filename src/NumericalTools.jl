

GetH(x) = (suff(x) == BigFloat) ? convert(BigFloat,10^(-precision(BigFloat)/10)) : 1e-6

"""
    suff(x) -> Type
If `x` stores BigFloats, `suff` returns BigFloat, else `suff` returns `Float64`.
"""
suff(x::BigFloat) = BigFloat
suff(x::Float32) = Float32
suff(x::Float16) = Float16
suff(x::Real) = Float64
suff(x::Num) = Num
suff(x::Complex) = real(x)
suff(x::AbstractArray) = suff(x[1])
suff(x::DataFrame) = suff(x[1,1])
suff(x::Tuple) = suff(x...)
suff(args...) = try suff(promote(args...)[1]) catch;  suff(args[1]) end
# Allow for differentiation through suff arrays. NEEDS TESTING.
suff(x::ForwardDiff.Dual) = typeof(x)

floatify(x::AbstractArray{<:AbstractFloat}) = x;   floatify(x::AbstractArray) = float.(x)
floatify(x::AbstractFloat) = x;                     floatify(x::Number) = float(x)
floatify(x) = float.(x)

"""
    MaximalNumberOfArguments(F::Function) -> Int
Infers argument structure of given function, i.e. whether it is of the form `F(x)` or `F(x,y)` or `F(x,y,z)` etc. and returns maximal number of accepted arguments of all overloads of `F` as integer.
"""
MaximalNumberOfArguments(F::Function) = maximum([length(Base.unwrap_unionall(m.sig).parameters)-1 for m in methods(F)])


"""
    Unpack(Z::AbstractVector{S}) where S <: Union{AbstractVector,Tuple} -> Matrix
Converts vector of vectors to a matrix whose n-th column corresponds to the n-th component of the inner vectors.
"""
@inline function Unpack(Z::AbstractVector{S}) where S <: Union{AbstractVector{<:Number},Tuple}
    N = length(Z);      M = length(Z[1])
    A = Array{suff(Z)}(undef,N,M)
    @inbounds for i in Base.OneTo(N)
        for j in Base.OneTo(M)
            A[i,j] = Z[i][j]
        end
    end;    A
end
Unpack(Z::AbstractVector{<:Number}) = Z

Unwind(M::AbstractMatrix{<:Number}) = Unwind(collect(eachrow(M)))
Unwind(X::AbstractVector{<:AbstractVector{<:Number}}) = reduce(vcat, X)
Unwind(X::AbstractVector{<:Number}) = X


Windup(X::AbstractVector{<:Number}, n::Int) = n < 2 ? X : [X[(1+(i-1)*n):(i*n)] for i in 1:Int(length(X)/n)]

ToCols(M::Matrix) = Tuple(M[:,i] for i in 1:size(M,2))


ValToBool(x::Val{true}) = true
ValToBool(x::Val{false}) = false


function GetMethod(tol::Real)
    if tol > 1e-8
        Tsit5()
    elseif tol < 1e-11
        Vern9()
    else
        Vern7()
    end
end

# Check for length
PromoteStatic(X::AbstractArray, inplace::Bool=true) = length(X) > 90 ? X : PromoteStatic(X, Val(inplace))

# No checking for length
PromoteStatic(X::AbstractArray, mutable::Val{true}) = _PromoteMutable(X)
PromoteStatic(X::AbstractArray, mutable::Val{false}) = _PromoteStatic(X)

_PromoteMutable(X::AbstractVector, Length=length(X)) = MVector{Length}(X)
_PromoteMutable(X::AbstractArray, Size=size(X)) = MArray{Tuple{Size...}}(X)
_PromoteStatic(X::AbstractVector, Length=length(X)) = SVector{Length}(X)
_PromoteStatic(X::AbstractArray, Size=size(X)) = SArray{Tuple{Size...}}(X)


# Surely, this can be made more efficient?
SplitAfter(n::Int) = X->(X[1:n], X[n+1:end])


"""
    invert(F::Function, x::Number; tol::Real=GetH(x)) -> Real
Finds ``z`` such that ``F(z) = x`` to a tolerance of `tol` for continuous ``F`` using Roots.jl. Ideally, `F` should be monotone and there should only be one correct result.
"""
function invert(F::Function, x::Number, Domain::Tuple{<:Number,<:Number}=(zero(suff(x)), 1e4*one(suff(x)));
                    tol::Real=GetH(x), meth::Roots.AbstractUnivariateZeroMethod=Roots.Order1())
    @assert Domain[1] < Domain[2]
    try
        if meth isa Roots.AbstractNonBracketing
            find_zero(z-> F(z) - x, 0.5one(suff(x)), meth; xatol=tol)
        else
            find_zero(z-> F(z) - x, Domain, meth; xatol=tol)
        end
    catch err
        @warn "invert() errored: $(nameof(typeof(err))). Assuming result is bracketed by $Domain and falling back to Bisection-like method."
        find_zero(z-> F(z) - x, Domain, Roots.AlefeldPotraShi(); xatol=tol)
    end
end
# function invert(F::Function, x::Number; tol::Real=GetH(x)*100, meth::Roots.AbstractUnivariateZeroMethod=Order1())
#     find_zero(z-> F(z) - x, 0.8one(suff(x)), meth; xatol=tol)
# end


"""
    ConfAlpha(n::Real)
Probability volume outside of a confidence interval of level n⋅σ where σ is the standard deviation of a normal distribution.
"""
ConfAlpha(n::Real) = 1 - ConfVol(n)

"""
    ConfVol(n::Real)
Probability volume contained in a confidence interval of level n⋅σ where σ is the standard deviation of a normal distribution.
"""
function ConfVol(n::Real)
    if abs(n) ≤ 8
        return erf(n / sqrt(2))
    else
        println("ConfVol: Float64 precision not enough for n = $n. Returning BigFloat instead.")
        return ConfVol(BigFloat(n))
    end
end
ConfVol(n::BigFloat) = erf(n / sqrt(BigFloat(2)))

InvConfVol(q::Real; kwargs...) = sqrt(2) * erfinv(q)
InvConfVol(x::BigFloat; tol::Real=GetH(x)) = invert(ConfVol, x; tol=tol)

ChisqCDF(k::Int, x::Real) = gamma_inc(k/2., x/2., 0)[1]
# ChisqCDF(k::Int, x::Real) = cdf(Chisq(k), x)
ChisqCDF(k::Int, x::BigFloat) = gamma_inc(BigFloat(k)/2., x/2., 0)[1]

InvChisqCDF(k::Int, p::Real; kwargs...) = 2gamma_inc_inv(k/2., p, 1-p)
InvChisqCDF(k::Int, p::BigFloat; tol::Real=GetH(p)) = invert(x->ChisqCDF(k, x), p; tol=tol)


InnerProduct(Mat::AbstractMatrix, Y::AbstractVector) = transpose(Y) * Mat * Y
# InnerProduct(Mat::PDMats.PDMat, Y::AbstractVector) = (R = Mat.chol.U * Y;  dot(R,R))


import Base.==
==(DS1::DataSet, DS2::DataSet) = xdata(DS1) == xdata(DS2) && ydata(DS1) == ydata(DS2) && ysigma(DS1) == ysigma(DS2)
==(DS1::DataSetExact, DS2::DataSet) = DS2 == DS1
function ==(DS1::DataSet, DS2::DataSetExact)
    if !(xdist(DS2) isa InformationGeometry.Dirac && ydist(DS) isa MvNormal)
        return false
    elseif xdata(DS1) == xdata(DS2) && ydata(DS1) == ydata(DS2) && ysigma(DS1) == ysigma(DS2)
        return true
    else
        false
    end
end
==(DS1::AbstractDataSet, DS2::AbstractDataSet) = xdist(DS1) == xdist(DS2) && ydist(DS1) == ydist(DS2)



"""
    KillAfter(F::Function, args...; timeout::Real=5, kwargs...)
Tries to evaluate a given function `F` before a set `timeout` limit is reached and interrupts the evaluation and returns `nothing` if necessary.
NOTE: The given function is evaluated via F(args...; kwargs...).
"""
function KillAfter(F::Function, args...; timeout::Real=5, kwargs...)
    Res = nothing
    G() = try F(args...; kwargs...) catch Err
        if Err isa DivideError
            @warn "KillAfter: Could not evaluate given Function $(nameof(F)) before timeout limit of $timeout seconds was reached."
        else
            @warn "KillAfter: Could not evaluate given Function $(nameof(F)) because error was thrown: $Err."
        end
    end
    task = @async(G())
    if timedwait(()->istaskdone(task), timeout) == :timed_out
        @async(Base.throwto(task, DivideError())) # kill task
    else
        Res = fetch(task)
    end;    Res
end



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

"""
    Builder(Fexpr::Union{<:AbstractVector{<:Num},<:Num}, args...; inplace::Bool=false, parallel::Bool=false, kwargs...)
Builds `RuntimeGeneratedFunctions` from expressions via build_function().
"""
function Builder(Fexpr::Union{<:AbstractArray{<:Num},<:Num}, args...; inplace::Bool=false, parallel::Bool=false, kwargs...)
    parallelization = parallel ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()
    Res = if (Fexpr isa Num && args[1] isa Num)
        # build_function throws error when using parallel keyword for R⟶R functions
        Symbolics.build_function(Fexpr, args...; expression=Val{false}, kwargs...)
    else
        Symbolics.build_function(Fexpr, args...; expression=Val{false}, parallel=parallelization, kwargs...)
    end
    try
        Res[inplace ? 2 : 1]
    catch;
        Res
    end
end

"""
    Integrate1D(F::Function, Cube::HyperCube; tol::Real=1e-14, FullSol::Bool=false, meth=nothing)
Integrates `F` over a one-dimensional domain specified via a `HyperCube` by rephrasing the integral as an ODE and using `DifferentialEquations.jl`.
"""
function Integrate1D(F::Function, Cube::HyperCube; tol::Real=1e-14, FullSol::Bool=false, meth=nothing)
    length(Cube) != 1 && throw(ArgumentError("Cube dim = $(length(Cube)) instead of 1"))
    Integrate1D(F,(Cube.L[1],Cube.U[1]); tol=tol,FullSol=FullSol,meth=meth)
end
Integrate1D(F::Function, Interval::AbstractVector{<:Number}; tol::Real=1e-14, FullSol::Bool=false, meth=nothing) = Integrate1D(F, Tuple(Interval); tol=tol, FullSol=FullSol, meth=meth)
function Integrate1D(F::Function, Interval::Tuple{<:Number,<:Number}; tol::Real=1e-14, FullSol::Bool=false, meth=nothing)
    Interval = floatify(Interval)
    !(0. < tol < 1.) && throw("Integrate1D: tol unsuitable")
    Interval[1] > Interval[2] && throw(ArgumentError("Interval orientation wrong."))
    f(u,p,t) = F(t);    u0 = 0.
    if tol < 1e-15
        u0 = BigFloat(u0);        Interval = BigFloat.(Interval)
        meth = isnothing(meth) ? Vern9() : meth
    else
        meth = isnothing(meth) ? Tsit5() : meth
    end
    if FullSol
        return solve(ODEProblem(f,u0,Interval),meth; reltol=tol,abstol=tol)
    else
        return solve(ODEProblem(f,u0,Interval),meth; reltol=tol,abstol=tol,save_everystep=false,save_start=false,save_end=true).u[end]
    end
end

"""
    IntegrateND(F::Function,Cube::HyperCube; tol::Real=1e-12, WE::Bool=false, kwargs...)
Integrates the function `F` over `Cube` with the help of **HCubature.jl** to a tolerance of `tol`.
If `WE=true`, the result is returned as a `Measurement` which also contains the estimated error in the result.
"""
function IntegrateND(F::Function, Cube::HyperCube; tol::Real=1e-12, WE::Bool=false, kwargs...)
    if length(Cube) == 1
        val, uncert = hquadrature(F, Cube.L[1], Cube.U[1]; rtol=tol, atol=tol, kwargs...)
    else
        val, uncert = hcubature(F, Cube.L, Cube.U; rtol=tol, atol=tol, kwargs...)
    end
    if length(val) == 1
        return WE ? measurement(val[1],uncert[1]) : val[1]
    else
        return WE ? measurement.(val,uncert) : val
    end
end
IntegrateND(F::Function, L::AbstractVector{<:Number}, U::AbstractVector{<:Number}; tol::Real=1e-12, WE::Bool=false, kwargs...) = IntegrateND(F,HyperCube(L,U); tol=tol, WE=WE, kwargs...)
IntegrateND(F::Function, Interval::Union{AbstractVector{<:Number},Tuple{<:Number,<:Number}}; tol::Real=1e-12, WE::Bool=false, kwargs...) = IntegrateND(F,HyperCube(Interval); tol=tol, WE=WE, kwargs...)


"""
    IntegrateOverConfidenceRegion(DM::AbstractDataModel, Domain::HyperCube, Confnum::Real, F::Function; N::Int=Int(1e5), WE::Bool=true, kwargs...)
Integrates a function `F` over the intersection of `Domain` and the confidence region of level `Confnum`.
"""
function IntegrateOverConfidenceRegion(DM::AbstractDataModel, Domain::HyperCube, Confnum::Real, F::Function; N::Int=Int(1e5), WE::Bool=true, kwargs...)
    @assert length(Domain) == pdim(DM)
    # Multiply F with characteristic function for confidence region
    Threshold = LogLikeMLE(DM) - 0.5InvChisqCDF(pdim(DM), ConfVol(Confnum))
    InsideRegion(X::AbstractVector{<:Number}) = loglikelihood(DM, X; kwargs...) > Threshold
    Integrand(X::AbstractVector{<:Number}) = InsideRegion(X) ? F(X) : zero(suff(X))
    # Use HCubature instead of MonteCarlo
    MonteCarloArea(Integrand, Domain, N; WE=WE)
end

"""
    IntegrateOverApproxConfidenceRegion(DM::AbstractDataModel, Domain::HyperCube, sol::AbstractODESolution, F::Function; N::Int=Int(1e5), WE::Bool=true, kwargs...)
    IntegrateOverApproxConfidenceRegion(DM::AbstractDataModel, Domain::HyperCube, Planes::AbstractVector{<:Plane}, sols::AbstractVector{<:AbstractODESolution}, F::Function; N::Int=Int(1e5), WE::Bool=true, kwargs...)
Integrates a function `F` over the intersection of `Domain` and the polygon defined by `sol`.
"""
function IntegrateOverApproxConfidenceRegion(DM::AbstractDataModel, Domain::HyperCube, sol::AbstractODESolution, F::Function; N::Int=Int(1e5), WE::Bool=true, kwargs...)
    @assert length(Domain) == pdim(DM) == length(sol.u[1]) == 2
    Integrand(X::AbstractVector{<:Number}) = ApproxInRegion(sol, X) ? F(X) : zero(suff(X))
    # Use HCubature instead of MonteCarlo
    MonteCarloArea(Integrand, Domain, N; WE=WE)
end

function IntegrateOverApproxConfidenceRegion(DM::AbstractDataModel, Domain::HyperCube, Tup::Tuple{<:AbstractVector{<:Plane},<:AbstractVector{<:AbstractODESolution}}, F::Function; N::Int=Int(1e5), WE::Bool=true, kwargs...)
    IntegrateOverApproxConfidenceRegion(DM, Domain, Tup[1], Tup[2], F; N=N, WE=WE, kwargs...)
end
function IntegrateOverApproxConfidenceRegion(DM::AbstractDataModel, Domain::HyperCube, Planes::AbstractVector{<:Plane}, sols::AbstractVector{<:AbstractODESolution}, F::Function; N::Int=Int(1e5), WE::Bool=true, kwargs...)
    @assert length(Domain) == pdim(DM) == ConsistentElDims(Planes)
    @assert length(Planes) == length(sols)
    Integrand(X::AbstractVector{<:Number}) = ApproxInRegion(Planes, sols, X) ? F(X) : zero(suff(X))
    # Use HCubature instead of MonteCarlo
    MonteCarloArea(Integrand, Domain, N; WE=WE)
end


"""
    LineSearch(Test::Function, start::Number=0.; tol::Real=8e-15, maxiter::Int=10000) -> Number
Finds real number `x` where the boolean-valued `Test(x::Number)` goes from `true` to `false`.
"""
function LineSearch(Test::Function, start::Number=0.; tol::Real=8e-15, maxiter::Int=10000)
    if ((suff(start) != BigFloat) && tol < 1e-15)
        println("LineSearch: start not BigFloat but tol=$tol. Promoting and continuing.")
        start = BigFloat(start)
    end
    if !Test(start)
        start += 1e-10
        println("LineSearch: Test(start) did not work, trying Test(start + 1e-10).")
        !Test(start) && throw(ArgumentError("LineSearch: Test not true for starting value."))
    end
    # For some weird reason, if the division by 4 is removed, the loop never terminates for BigFloat-valued "start"s - maybe the compiler erroneously tries to optimize the variable "stepsize" away or something?! (Julia version ≤ 1.6.0)
    stepsize = one(suff(start)) / 4.;       value = start
    for i in 1:maxiter
        if Test(value + stepsize) # inside
            value += stepsize
            # value - start > 2000. && throw("FindConfBoundary: Value larger than 2000.")
        else            #outside
            if stepsize < tol
                return value + stepsize
            end
            stepsize /= 5.
        end
    end
    throw("$maxiter iterations over. Value=$value, Stepsize=$stepsize")
end

# function AltLineSearch(Test::Function, x::Number, Domain::Tuple{<:Number,<:Number}=(zero(suff(x)), 1e4*one(suff(x)));
#                     tol::Real=GetH(x), meth::Roots.AbstractUnivariateZeroMethod=Roots.Order1())
#     @assert Domain[1] < Domain[2]
#     try
#         if meth isa Roots.AbstractNonBracketing
#             find_zero(z-> F(z) - x, 0.5one(suff(x)), meth; xatol=tol)
#         else
#             find_zero(z-> F(z) - x, Domain, meth; xatol=tol)
#         end
#     catch err
#         @warn "invert() errored: $(nameof(typeof(err))). Assuming result is bracketed by $Domain and falling back to Bisection-like method."
#         find_zero(z-> F(z) - x, Domain, Roots.AlefeldPotraShi(); xatol=tol)
#     end
# end

function MonteCarloArea(Test::Function,Cube::HyperCube,N::Int=Int(1e7); WE::Bool=false)
    if WE
        return CubeVol(Cube) * MonteCarloRatioWE(Test,Cube,N)
    else
        return CubeVol(Cube) * MonteCarloRatio(Test,Cube,N)
    end
end
function MonteCarloRatio(Test::Function,Cube::HyperCube,N::Int=Int(1e7))
    (1/N)* @distributed (+) for i in 1:N
        Test(rand.(Uniform.(Cube.L,Cube.U)))
    end
end

function MonteCarloRatioWE(Test::Function,LU::HyperCube,N::Int=Int(1e7); chunksize::Int=Int(N/20))
    chunksize > N && error("chunksize > N")
    if N%chunksize != 0
        N += Int(N%chunksize + 1)
    end
    chunks = Int(N/chunksize)
    # Output not normalized by chunksize
    function CarloLoop(Test::Function,LU::HyperCube,chunksize::Int)
        tot = [rand.(Uniform.(LU.L,LU.U)) for i in 1:chunksize] .|> Test
        res = sum(tot)
        [res, sum(abs2, (tot .- (res/chunksize)))]
    end
    Tot = @distributed (+) for i in 1:chunks
        CarloLoop(Test,LU,chunksize)
    end
    measurement(Tot[1]/N, sqrt(1/((N-1)*N) * Tot[2]))
end


# # From Cuba.jl docs
# function CompactDomainTransform(F::ModelOrFunction, Cube::HyperCube)
#     (!all(x->isfinite(x),Cube.L) || !all(x->isfinite(x),Cube.U)) && throw("Not applicable.")
#     if length(Cube) == 1
#         W = Cube.U[1] - Cube.L[1]
#         return x -> W * F(Cube.L[1] + W * x)
#         # Use mul! or something like that?
#     else
#         W = CubeWidths(Cube);   V = prod(CubeWidths)
#         return x -> V * F(Cube.L + W * x)
#     end
# end
#
# function HalfInfiniteTransform(F::ModelOrFunction, Cube::HyperCube)
#     # integral over [a,∞]
#     if Cube.U[1] == Inf && isfinite(Cube.L[1])
#         return x -> (1-x)^-2 * F(Cube.L[1] + x/(1-x))
#     end
# end
#
# function InfiniteDomainTransform(F::ModelOrFunction, Cube::HyperCube)
#     if Cube.L[1] == -Inf && Cube.L[1] == Inf
#         return x -> F((2x - 1.)/((1 - x)*x)) * (2x^2 - 2y + 1) / ((1-x^2)*x^2)
#     end
# end


import LsqFit.curve_fit
function curve_fit(DM::AbstractDataModel, initial::AbstractVector{<:Number}=MLE(DM), LogPriorFn::Union{Nothing,Function}=LogPrior(DM); tol::Real=1e-14, kwargs...)
    curve_fit(Data(DM), Predictor(DM), dPredictor(DM), initial, LogPriorFn; tol=tol, kwargs...)
end

function curve_fit(DS::AbstractDataSet, M::ModelMap, initial::AbstractVector{<:Number}=GetStartP(DS,M), LogPriorFn::Union{Nothing,Function}=nothing; tol::Real=1e-14, kwargs...)
    curve_fit(DS, M.Map, initial, LogPriorFn; tol=tol, lower=convert(Vector,M.Domain.L), upper=convert(Vector,M.Domain.U), kwargs...)
end

function curve_fit(DS::AbstractDataSet, M::ModelMap, dM::ModelOrFunction, initial::AbstractVector{<:Number}=GetStartP(DS,M), LogPriorFn::Union{Nothing,Function}=nothing; tol::Real=1e-14, kwargs...)
    curve_fit(DS, M.Map, dM, initial, LogPriorFn; tol=tol, lower=convert(Vector,M.Domain.L), upper=convert(Vector,M.Domain.U), kwargs...)
end

function curve_fit(DS::AbstractDataSet, model::Function, initial::AbstractVector{<:Number}=GetStartP(DS,model), LogPriorFn::Union{Nothing,Function}=nothing; tol::Real=1e-14, kwargs...)
    X = xdata(DS);  Y = ydata(DS);    LsqFit.check_data_health(X, Y)
    u = cholesky(yInvCov(DS)).U
    !isnothing(LogPriorFn) && @warn "curve_fit() cannot account for priors. Throwing away given prior and continuing anyway."
    f(p) = u * (EmbeddingMap(DS, model, p) - Y)
    p0 = convert(Vector, initial)
    R = LsqFit.OnceDifferentiable(f, p0, copy(f(p0)); inplace = false, autodiff = :forward)
    LsqFit.lmfit(R, p0, yInvCov(DS); x_tol=tol, g_tol=tol, kwargs...)
end

function curve_fit(DS::AbstractDataSet, model::Function, dmodel::ModelOrFunction, initial::AbstractVector{<:Number}=GetStartP(DS,model), LogPriorFn::Union{Nothing,Function}=nothing; tol::Real=1e-14, kwargs...)
    X = xdata(DS);  Y = ydata(DS);    LsqFit.check_data_health(X, Y)
    u = cholesky(yInvCov(DS)).U
    !isnothing(LogPriorFn) && @warn "curve_fit() cannot account for priors. Throwing away given prior and continuing anyway."
    f(p) = u * (EmbeddingMap(DS, model, p) - Y)
    df(p) = u * EmbeddingMatrix(DS, dmodel, p)
    p0 = convert(Vector, initial)
    R = LsqFit.OnceDifferentiable(f, df, p0, copy(f(p0)); inplace = false)
    LsqFit.lmfit(R, p0, yInvCov(DS); x_tol=tol, g_tol=tol, kwargs...)
end

function normalizedjac(M::AbstractMatrix{<:Number}, xlen::Int)
    M[:,1:xlen] .*= sqrt(size(M,1)/xlen -1.);    return M
end


TotalLeastSquares(DM::AbstractDataModel, args...; kwargs...) = TotalLeastSquares(Data(DM), Predictor(DM), args...; kwargs...)
"""
    TotalLeastSquares(DSE::DataSetExact, model::ModelOrFunction, initial::AbstractVector{<:Number}; tol::Real=1e-13, kwargs...) -> Vector
Experimental feature which takes into account uncertainties in x-values to improve the accuracy of the fit.
Returns concatenated vector of x-values and parameters. Assumes that the uncertainties in the x-values and y-values are normal, i.e. Gaussian!
"""
function TotalLeastSquares(DSE::DataSetExact, model::ModelOrFunction, initialp::AbstractVector{<:Number}=GetStartP(DSE, model); ADmode::Union{Symbol,Val}=Val(:ForwardDiff), tol::Real=1e-13, rescale::Bool=true, kwargs...)
    # Improve starting values by fitting with ordinary least squares first
    initialp = curve_fit(DataSet(WoundX(DSE),Windup(ydata(DSE),ydim(DSE)),ysigma(DSE)), model, initialp; tol=tol, kwargs...).param
    if xdist(DSE) isa InformationGeometry.Dirac
        println("xdist of given data is Dirac, can only use ordinary least squares.")
        return xdata(DSE), initialp
    end

    plen = pdim(DSE,model);  xlen = Npoints(DSE) * xdim(DSE)
    function predictY(ξ::AbstractVector)
        x = view(ξ, 1:xlen);        p = view(ξ, (xlen+1):length(ξ))
        # INPLACE EmbeddingMap!() would be great here!
        vcat(x, EmbeddingMap(DSE, model, p, Windup(x,xdim(DSE))))
    end
    u = cholesky(BlockMatrix(InvCov(xdist(DSE)),InvCov(ydist(DSE)))).U;    Ydata = vcat(xdata(DSE), ydata(DSE))
    f(p) = u * (predictY(p) - Ydata)
    Jac = GetJac(ADmode, predictY)
    dfnormalized(p) = u * normalizedjac(Jac(p), xlen)
    df(p) = u * Jac(p)
    p0 = vcat(xdata(DSE), initialp)
    R = rescale ? LsqFit.OnceDifferentiable(f, dfnormalized, p0, copy(f(p0)); inplace = false) : LsqFit.OnceDifferentiable(f, df, p0, copy(f(p0)); inplace = false)
    fit = LsqFit.lmfit(R, p0, BlockMatrix(InvCov(xdist(DSE)), InvCov(ydist(DSE))); x_tol=tol, g_tol=tol, kwargs...)
    Windup(fit.param[1:xlen],xdim(DSE)), fit.param[xlen+1:end]
end

function TotalLeastSquares(DS::AbstractDataSet, model::ModelOrFunction, initialp::AbstractVector{<:Number}=GetStartP(DS, model); tol::Real=1e-13, kwargs...)
    sum(abs, xsigma(DS)) == 0.0 && throw("Cannot perform Total Least Squares Fitting for DataSets without x-uncertainties.")
    xlen = Npoints(DS)*xdim(DS);    Cost(x::AbstractVector) = -logpdf(dist(DS), x)
    function predictY(ξ::AbstractVector)
        x = view(ξ, 1:xlen);        p = view(ξ, (xlen+1):length(ξ))
        vcat(x, EmbeddingMap(DS, model, p, Windup(x,xdim(DS))))
    end
    InformationGeometry.minimize(Cost∘predictY, [xdata(DS); initialp]; tol=tol, kwargs...)
end


"""
    minimize(F::Function, start::AbstractVector{<:Number}; tol::Real=1e-10, meth=NelderMead(), Full::Bool=false, timeout::Real=200, kwargs...) -> Vector
Minimizes the scalar input function using the given `start` using algorithms from `Optim.jl` specified via the keyword `meth`.
`Full=true` returns the full solution object instead of only the minimizing result.
Optionally, the search domain can be bounded by passing a suitable `HyperCube` object as the third argument.
"""
function minimize(F::Function, start::AbstractVector{<:Number}, Domain::Union{HyperCube,Nothing}=nothing; Fthresh::Union{Nothing,Real}=nothing, tol::Real=1e-10, meth::Optim.AbstractOptimizer=NelderMead(), timeout::Real=200, Full::Bool=false, kwargs...)
    !(F(start) isa Number) && throw("Given function must return scalar values, got $(typeof(F(start))) instead.")
    options = if isnothing(Fthresh)
        Optim.Options(g_tol=tol, x_tol=tol, time_limit=floatify(timeout))
    else  # stopping criterion via callback kwarg
        Optim.Options(callback=(z->z.value<Fthresh), g_tol=tol, x_tol=tol, time_limit=floatify(timeout))
    end
    Res = if isnothing(Domain)
        optimize(F, floatify(start), meth, options; kwargs...)
    else
        start ∉ Domain && throw("Given starting value not in specified domain.")
        optimize(F, convert(Vector{Float64},Domain.L), convert(Vector{Float64},Domain.U), floatify(start), meth, options; kwargs...)
    end
    Full ? Res : Optim.minimizer(Res)
end
minimize(FdF::Tuple{Function,Function}, args...; kwargs...) = minimize(FdF[1], FdF[2], args...; kwargs...)
function minimize(F::Function, dF::Function, start::AbstractVector{<:Number}, Domain::Union{HyperCube,Nothing}=nothing; Fthresh::Union{Nothing,Real}=nothing, tol::Real=1e-10, meth::Optim.AbstractOptimizer=BFGS(), timeout::Real=200, Full::Bool=false, kwargs...)
    !(F(start) isa Number) && throw("Given function must return scalar values, got $(typeof(F(start))) instead.")
    # Wrap dF to make it inplace
    newdF = MaximalNumberOfArguments(dF) < 2 ? ((G,x)->(G .= dF(x))) : dF
    options = if isnothing(Fthresh)
        Optim.Options(g_tol=tol, x_tol=tol, time_limit=floatify(timeout))
    else  # stopping criterion via callback kwarg
        Optim.Options(callback=(z->z.value<Fthresh), g_tol=tol, x_tol=tol, time_limit=floatify(timeout))
    end
    Res = if isnothing(Domain)
        optimize(F, newdF, floatify(start), meth, options; kwargs...)
    else
        start ∉ Domain && throw("Given starting value not in specified domain.")
        optimize(F, newdF, convert(Vector{Float64},Domain.L), convert(Vector{Float64},Domain.U), floatify(start), meth, options; kwargs...)
    end
    Full ? Res : Optim.minimizer(Res)
end

"""
    RobustFit(DM::AbstractDataModel, start::AbstractVector{<:Number}; tol::Real=1e-10, p::Real=1, kwargs...)
Uses `p`-Norm to judge distance on Dataspace as specified by the keyword.
"""
RobustFit(DM::AbstractDataModel, args...; kwargs...) = RobustFit(Data(DM), Predictor(DM), args...; kwargs...)
function RobustFit(DS::AbstractDataSet, M::ModelOrFunction, start::AbstractVector{<:Number}=GetStartP(DS,M), Domain::Union{HyperCube,Nothing}=(M isa ModelMap ? M.Domain : nothing); tol::Real=1e-10, p::Real=1, kwargs...)
    HalfSig = cholesky(yInvCov(DS)).U
    F(x::AbstractVector) = norm(HalfSig * (ydata(DS) - EmbeddingMap(DS, M, x)), p)
    InformationGeometry.minimize(F, start, Domain; tol=tol, kwargs...)
end
function RobustFit(DS::AbstractDataSet, M::ModelOrFunction, dM::ModelOrFunction, start::AbstractVector{<:Number}=GetStartP(DS,M), Domain::Union{HyperCube,Nothing}=(M isa ModelMap ? M.Domain : nothing); tol::Real=1e-10, p::Real=1, kwargs...)
    HalfSig = cholesky(yInvCov(DS)).U
    F(x::AbstractVector) = norm(HalfSig * (EmbeddingMap(DS, M, x) - ydata(DS)), p)
    function dFp(x::AbstractVector)
        z = HalfSig * (EmbeddingMap(DS, M, x) - ydata(DS))
        n = sum(z.^p)^(1/p - 1) * z.^(p-1)
        transpose(HalfSig * EmbeddingMatrix(DS, dM, x)) * n
    end
    dF1(x::AbstractVector) = transpose(HalfSig * EmbeddingMatrix(DS, dM, x)) *  sign.(HalfSig * (EmbeddingMap(DS, M, x) - ydata(DS)))
    InformationGeometry.minimize(F, (p == 1 ? dF1 : dFp), start, Domain; tol=tol, kwargs...)
end


"""
    GetArgSize(model::ModelOrFunction; max::Int=100)::Tuple{Int,Int}
Returns tuple `(xdim,pdim)` associated with the method `model(x,p)`.
"""
function GetArgSize(model::Function; max::Int=100)::Tuple{Int,Int}
    try         return (1, GetArgLength(p->model(1.,p); max=max))       catch; end
    for i in 2:(max + 1)
        plen = try      GetArgLength(p->model(ones(i),p); max=max)      catch; continue end
        i == (max + 1) ? throw("Wasn't able to find config.") : return (i, plen)
    end
end
GetArgSize(model::ModelMap; max::Int=100) = (model.xyp[1], model.xyp[3])


function GetArgLength(F::Function; max::Int=100)::Int
    max < 1 && throw("pdim: max = $max too small.")
    try     F(1.);  return 1    catch; end
    for i in 1:(max+1)
        try
            res = F(ones(i))
            isnothing(res) ? throw("pdim: Function returned Nothing for i=$i.") : res
        catch y
            (isa(y, BoundsError) || isa(y, MethodError) || isa(y, DimensionMismatch) || isa(y, ArgumentError) || isa(y, AssertionError)) && continue
            println("pdim: Encountered error in specification of model function.");     rethrow()
        end
        i == (max + 1) ? throw(ArgumentError("pdim: Parameter space appears to have >$max dims. Aborting. Maybe wrong type of x was inserted?")) : return i
    end
end



normalize(x::AbstractVector{<:Number}, scaling::Float64=1.0) = (scaling / norm(x)) * x
function normalizeVF(u::AbstractVector{<:Number}, v::AbstractVector{<:Number}, scaling::Float64=1.0)
    newu = u;    newv = v
    for i in 1:length(u)
        factor = sqrt(u[i]^2 + v[i]^2)
        newu[i] = (scaling/factor)*u[i]
        newv[i] = (scaling/factor)*v[i]
    end
    newu, newv
end
function normalizeVF(u::AbstractVector{<:Number},v::AbstractVector{<:Number},PlanarCube::HyperCube,scaling::Float64=1.0)
    length(PlanarCube) != 2 && throw("normalizeVF: Cube not planar.")
    newu = u;    newv = v
    Widths = CubeWidths(PlanarCube) |> normalize
    for i in 1:length(u)
        factor = sqrt(u[i]^2 + v[i]^2)
        newu[i] = (scaling/factor)*u[i] * Widths[1]
        newv[i] = (scaling/factor)*v[i] * Widths[2]
    end
    newu, newv
end


"""
    BlockMatrix(M::AbstractMatrix, N::Int)
Returns matrix which contains `N` many blocks of the matrix `M` along its diagonal.
"""
function BlockMatrix(M::AbstractMatrix, N::Int)
    Res = zeros(size(M,1)*N,size(M,2)*N)
    for i in 1:N
        Res[((i-1)*size(M,1) + 1):(i*size(M,1)),((i-1)*size(M,1) + 1):(i*size(M,1))] = M
    end;    Res
end
BlockMatrix(M::Diagonal, N::Int) = Diagonal(repeat(M.diag, N))

"""
    BlockMatrix(A::AbstractMatrix, B::AbstractMatrix)
Constructs blockdiagonal matrix from `A` and `B`.
"""
function BlockMatrix(A::AbstractMatrix, B::AbstractMatrix)
    Res = zeros(suff(A), size(A,1)+size(B,1), size(A,2)+size(B,2))
    Res[1:size(A,1), 1:size(A,1)] = A
    Res[size(A,1)+1:end, size(A,1)+1:end] = B
    Res
end
BlockMatrix(A::Diagonal, B::Diagonal) = Diagonal(vcat(A.diag, B.diag))


BlockMatrix(A::AbstractMatrix, B::AbstractMatrix, args...) = BlockMatrix(BlockMatrix(A,B), args...)
