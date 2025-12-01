


"""
    Integrate1D(F::Function, Cube::HyperCube; tol::Real=1e-14, FullSol::Bool=false, meth=Tsit5())
Integrates `F` over a one-dimensional domain specified via a `HyperCube` by rephrasing the integral as an ODE and using `DifferentialEquations.jl`.
"""
function Integrate1D(F::Function, Cube::HyperCube; kwargs...)
    length(Cube) != 1 && throw(ArgumentError("Cube dim = $(length(Cube)) instead of 1"))
    Integrate1D(F, (Cube.L[1],Cube.U[1]); kwargs...)
end
Integrate1D(F::Function, Interval::AbstractVector{<:Number}; kwargs...) = Integrate1D(F, Tuple(Interval); kwargs...)
function Integrate1D(F::Function, Interval::Tuple{<:Number,<:Number}; tol::Real=1e-14, FullSol::Bool=false, meth=nothing, kwargs...)
    Interval = floatify(Interval)
    !(0. < tol < 1.) && throw("Integrate1D: tol unsuitable")
    @assert Interval[1] ≤ Interval[2]
    f(u,p,t) = F(t);    u0 = zero(typeof(F((Interval[1]+Interval[2])/2.0)))
    if tol < 1e-15
        u0 = BigFloat(u0);        Interval = BigFloat.(Interval)
        meth = isnothing(meth) ? Vern9() : meth
    else
        meth = isnothing(meth) ? Tsit5() : meth
    end
    if FullSol
        return solve(ODEProblem(f,u0,Interval),meth; reltol=tol,abstol=tol, kwargs...)
    else
        return solve(ODEProblem(f,u0,Interval),meth; reltol=tol,abstol=tol,save_everystep=false,save_start=false,save_end=true, kwargs...).u[end]
    end
end
Integrate1D(I::DataInterpolations.AbstractInterpolation, dom::Tuple{<:Number,<:Number}=extrema(I.t); kwargs...) = Integrate1D(x->I(x), dom; kwargs...)

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
    Integrand(X::AbstractVector{T}) where T<:Number = InsideRegion(X) ? F(X) : zero(T)
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
    Integrand(X::AbstractVector{T}) where T<:Number = ApproxInRegion(sol, X) ? F(X) : zero(T)
    # Use HCubature instead of MonteCarlo
    MonteCarloArea(Integrand, Domain, N; WE=WE)
end

function IntegrateOverApproxConfidenceRegion(DM::AbstractDataModel, Domain::HyperCube, Tup::Tuple{<:AbstractVector{<:Plane},<:AbstractVector{<:AbstractODESolution}}, F::Function; N::Int=Int(1e5), WE::Bool=true, kwargs...)
    IntegrateOverApproxConfidenceRegion(DM, Domain, Tup[1], Tup[2], F; N=N, WE=WE, kwargs...)
end
function IntegrateOverApproxConfidenceRegion(DM::AbstractDataModel, Domain::HyperCube, Planes::AbstractVector{<:Plane}, sols::AbstractVector{<:AbstractODESolution}, F::Function; N::Int=Int(1e5), WE::Bool=true, kwargs...)
    @assert length(Domain) == pdim(DM) == ConsistentElDims(Planes)
    @assert length(Planes) == length(sols)
    Integrand(X::AbstractVector{T}) where T<:Number  = ApproxInRegion(Planes, sols, X) ? F(X) : zero(T)
    # Use HCubature instead of MonteCarlo
    MonteCarloArea(Integrand, Domain, N; WE=WE)
end





function MonteCarloArea(Test::Function,Cube::HyperCube,n::Int=Int(1e7); N::Int=n, WE::Bool=false)
    if WE
        return CubeVol(Cube) * MonteCarloRatioWE(Test,Cube,N)
    else
        return CubeVol(Cube) * MonteCarloRatio(Test,Cube,N)
    end
end
function MonteCarloRatio(Test::Function,Cube::HyperCube,n::Int=Int(1e7); N::Int=n)
    (1/N)* @distributed (+) for i in 1:N
        Test(rand.(Uniform.(Cube.L,Cube.U)))
    end
end

function MonteCarloRatioWE(Test::Function,LU::HyperCube,n::Int=Int(1e7); N::Int=n, chunksize::Int=Int(N/20))
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
