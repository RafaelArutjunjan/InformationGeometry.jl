

"""
    Plane(P::AbstractVector, Vx::AbstractVector, Vy::AbstractVector)
Specifies a 2D plane in the so-called parameter form using 3 vectors.
Here the first argument `P` is a vector on the plane, the two vectors `Vx` and `Vy` are two other vectors, which span the plane and should ideally be orthogonal.
"""
struct Plane
    stütz::AbstractVector
    Vx::AbstractVector
    Vy::AbstractVector
    function Plane(stütz::AbstractVector{<:Number}, Vx::AbstractVector{<:Number}, Vy::AbstractVector{<:Number}; Make2ndOrthogonal::Bool=true)
        if length(stütz) == 2 stütz = [stütz[1],stütz[2],0] end
        !(length(stütz) == length(Vx) == length(Vy)) && throw("Dimension mismatch. length(stütz) = $(length(stütz)), length(Vx) = $(length(Vx)), length(Vy) = $(length(Vy))")

        (Make2ndOrthogonal && abs(dot(Vx,Vy)) > 4e-15) && return Plane(stütz, Vx, Make2ndOrthogonal(Vx,Vy))

        if length(stütz) < 20
            stütz = SVector{length(Vx)}(floatify(stütz));     Vx = SVector{length(Vx)}(floatify(Vx))
            Vy = SVector{length(Vx)}(floatify(Vy))
            return new(stütz, Vx, Vy)
        else
            return new(floatify(stütz), floatify(Vx), floatify(Vy))
        end
    end
end

length(PL::Plane) = length(PL.stütz)

function MLEinPlane(DM::AbstractDataModel, PL::Plane, start::AbstractVector{<:Number}=0.0001rand(2); tol::Real=1e-8)
    length(start) != 2 && throw("Dimensional Mismatch.")
    PlanarLogPrior = isnothing(LogPrior(DM)) ? nothing : LogPrior(DM)∘PlaneCoordinates(PL)
    planarmod(x, θ::AbstractVector{<:Number}; kwargs...) = Predictor(DM)(x, PlaneCoordinates(PL,θ); kwargs...)
    return try
        # faster but sometimes problems with ForwarDiff-generated gradients in LsqFit
        curve_fit(Data(DM), planarmod, start, PlanarLogPrior; tol=tol).param
    catch;
        planardmod(x, θ::AbstractVector{<:Number}; kwargs...) = dPredictor(DM)(x, PlaneCoordinates(PL,θ); kwargs...) * [PL.Vx PL.Vy]
        curve_fit(Data(DM), planarmod, planardmod, start, PlanarLogPrior; tol=tol).param
    end
end

function PlanarDataModel(DM::AbstractDataModel, PL::Plane)
    @assert DM isa DataModel
    model = Predictor(DM);      dmodel = dPredictor(DM)
    newmod = (x,θ::AbstractVector{<:Number}; kwargs...) -> model(x, PlaneCoordinates(PL,θ); kwargs...)
    dnewmod = (x,θ::AbstractVector{<:Number}; kwargs...) -> dmodel(x, PlaneCoordinates(PL,θ); kwargs...) * [PL.Vx PL.Vy]
    PlanarLogPrior = isnothing(LogPrior(DM)) ? nothing : LogPrior(DM)∘PlaneCoordinates(PL)
    mle = MLEinPlane(DM, PL)
    DataModel(Data(DM), newmod, dnewmod, mle, loglikelihood(DM, PlaneCoordinates(PL, mle), PlanarLogPrior), PlanarLogPrior, true)
end

# Performance gains of using static vectors is lost if their length exceeds 32
BasisVectorSV(Slot::Int, dims::Int) = SVector{dims}(Float64(i == Slot) for i in 1:dims)

"""
    BasisVector(Slot::Int, dims::Int) -> Vector{Float64}
Computes a standard basis vector of length `dims`, i.e. whose components are all zero except for the component `Slot`, which has a value of one.
"""
function BasisVector(Slot::Int, dims::Int)
    Res = zeros(dims);    Res[Slot] = 1.;    Res
end


"""
    PlaneCoordinates(PL::Plane, v::AbstractVector{<:Number})
Returns an n-dimensional vector from a tuple of two real numbers which correspond to the coordinates in the 2D `Plane`.
"""
PlaneCoordinates(PL::Plane, v::AbstractVector) = PL.stütz + [PL.Vx PL.Vy] * v
PlaneCoordinates(PL::Plane) = x->PlaneCoordinates(PL,x)

Shift(PlaneBegin::Plane, PlaneEnd::Plane) = TranslatePlane(PlaneEnd, PlaneEnd.stütz - PlaneBegin.stütz)

IsOnPlane(PL::Plane, x::AbstractVector, ProjectionOp::AbstractMatrix=ProjectionOperator(PL))::Bool = DistanceToPlane(PL, x, ProjectionOp) < 4e-15
TranslatePlane(PL::Plane, v::AbstractVector) = Plane(PL.stütz + v, PL.Vx, PL.Vy)
RotatePlane(PL::Plane, rads::Real=π/2) = ((S,C) = sincos(rads);   Plane(PL.stütz, C*PL.Vx + S*PL.Vy, C*PL.Vy - S*PL.Vx))
function RotationMatrix(PL::Plane, rads::Real)
    V = PL.Vx*transpose(PL.Vx) + PL.Vy*transpose(PL.Vy)
    W = PL.Vx*transpose(PL.Vy) - PL.Vy*transpose(PL.Vx)
    Diagonal(ones(length(PL.stütz))) + (cos(rads)-1.)*V -sin(rads)*W
end
RotateVector(PL::Plane, v::AbstractVector, rads::Real) = RotationMatrix(PL,rads) * v

function RotatedVector(α::Real, n1::Int, n2::Int, tot::Int)
    @assert (n1 ≤ tot && n2 ≤ tot && n1 != n2 && all(x->(x>0), [n1,n2,tot]))
    res = zeros(tot);   res[n1] = cos(α);   res[n2] = sin(α);   res
end


function DecomposeWRTPlane(PL::Plane, x::AbstractVector)
    @assert IsOnPlane(PL,x)
    V = x - PL.stütz
    [dot(V, PL.Vx), dot(V, PL.Vy)]
end

DistanceToPlane(PL::Plane, x::AbstractVector, ProjectionOp::AbstractMatrix=ProjectionOperator(PL)) = (Diagonal(ones(length(x))) - ProjectionOp) * (x - PL.stütz) |> norm
ProjectOntoPlane(PL::Plane, x::AbstractVector, ProjectionOp::AbstractMatrix=ProjectionOperator(PL)) = ProjectionOp * (x - PL.stütz) + PL.stütz

function ProjectionOperator(A::AbstractMatrix)
    size(A,2) != 2 && println("ProjectionOperator: Matrix size $(size(A)) not as expected.")
    A * inv(transpose(A) * A) * transpose(A)
end
ProjectionOperator(PL::Plane) = ProjectionOperator([PL.Vx PL.Vy])

IsNormalToPlane(PL::Plane, v::AbstractVector)::Bool = abs(dot(PL.Vx, v)) < 4e-15 && abs(dot(PL.Vy, v)) < 4e-15

function Make2ndOrthogonal(X::AbstractVector,Y::AbstractVector)
    Basis = GramSchmidt(floatify([X,Y]))
    # Maybe add check for orientation?
    return Basis[2]
end

"""
    MinimizeOnPlane(PL::Plane,F::Function,initial::AbstractVector=[1,-1.]; tol::Real=1e-5)
Minimizes given function in Plane and returns the optimal point in the ambient space in which the plane lies.
"""
function MinimizeOnPlane(PL::Plane, F::Function, initial::AbstractVector=[1e-2,-1e-2]; tol::Real=1e-5, meth::Optim.AbstractOptimizer=LBFGS(), kwargs...)
    # G(x) = F(PlaneCoordinates(PL,x))
    X = InformationGeometry.minimize(F∘PlaneCoordinates(PL), initial; tol=tol, meth=meth, kwargs...)
    # X = Optim.minimizer(optimize(G,initial, BFGS(), Optim.Options(g_tol=tol), autodiff = :forward))
    PlaneCoordinates(PL,X)
end

"""
    ProjectOnto(v::AbstractVector, u::AbstractVector)
Project `v` onto `u`.
"""
ProjectOnto(v::AbstractVector, u::AbstractVector) = (dot(v,u) / dot(u,u)) * u

"""
    ParallelPlanes(PL::Plane, v::AbstractVector, range) -> Vector{Plane}
Returns Vector of Planes which have been translated by `a .* v` for all `a` in `range`.
"""
function ParallelPlanes(PL::Plane, v::AbstractVector, range::AbstractVector{<:Real})
    norm(v) == 0. && throw("Direction cannot be null vector.")
    # PL.Projector * v == v && throw("Plane and vector linearly dependent.")
    ProjectOntoPlane(PL,v) == v && throw("Plane and vector linearly dependent.")
    [TranslatePlane(PL, ran .* v) for ran in range]
end


function GramSchmidt(v::AbstractVector,dim::Int=length(v))
    Basis = Vector{suff(v)}(undef,0)
    push!(Basis,v)
    for i in 2:length(v)
        push!(Basis,BasisVector(i,length(v)))
    end
    GramSchmidt([normalize(Basis[i]) for i in 1:length(Basis)])
end
function GramSchmidt(Basis::AbstractVector{<:AbstractVector})
    ONBasis = floatify(Basis)
    for j in 1:length(Basis)
        for i in 2:j
            ONBasis[j] .-= ProjectOnto(Basis[j],ONBasis[i-1])
        end
    end
    [normalize(ONBasis[i]) for i in 1:length(ONBasis)]
end


"""
    HyperPlane(basepoint, dir1, ...) -> Function
Returns an embedding function which translates points from HyperPlane coordinates to the ambient space.
"""
function HyperPlane(args...)
    @assert all(x->typeof(x) <: AbstractVector{<:Number},args) && ConsistentElDims(args) > 0
    argdim = length(args) -1
    function EmbedIntoHyperPlane(θ::AbstractVector{<:Number})
        @assert length(θ) == argdim
        args[1] + sum(θ[i] * args[i+1] for i in 1:argdim)
    end
end


"""
The `HyperCube` type is used to specify a cuboid region in the form of a cartesian product of ``N`` real intervals, thereby offering a convenient way of passing domains for integration or plotting between functions.
A `HyperCube` object `cube` type has two fields: `cube.L` and `cube.U` which are two vectors which respectively store the lower and upper boundaries of the real intervals in order.
Examples for constructing `HyperCube`s:
```julia
HyperCube([[1,3],[π,2π],[-500,100]])
HyperCube([1,π,-500],[3,2π,100])
HyperCube([[-1,1]])
HyperCube([-1,1])
HyperCube(collect([-7,7.] for i in 1:3))
```
Examples of quantities that can be computed from and operations involving a `HyperCube` object `X`:
```julia
CubeVol(X)
TranslateCube(X,v::AbstractVector)
CubeWidths(X)
```
"""
struct HyperCube{Q<:Number} <: Cuboid
    L::AbstractVector{Q}
    U::AbstractVector{Q}
    HyperCube(C::HyperCube; Padding::Number=0.) = HyperCube(C.L, C.U; Padding=Padding)
    function HyperCube(lowers::AbstractVector{<:Number}, uppers::AbstractVector{<:Number}; Padding::Number=0.)
        @assert length(lowers) == length(uppers)
        if Padding != 0.
            diff = (uppers - lowers) .* (Padding / 2.)
            lowers -= diff;     uppers += diff
        end
        !all(lowers .≤ uppers) && throw("First argument of HyperCube must be larger than second.")
        if length(lowers) < 20
            return new{suff(lowers)}(SVector{length(lowers)}(floatify(lowers)), SVector{length(uppers)}(floatify(uppers)))
        else
            return new{suff(lowers)}(floatify(lowers),floatify(uppers))
        end
    end
    function HyperCube(H::AbstractVector{<:AbstractVector{<:Number}}; Padding::Number=0.)
        len = length(H[1]);        !all(x->(length(x) == len),H) && throw("Inconsistent lengths.")
        M = Unpack(H);        HyperCube(M[:,1],M[:,2]; Padding=Padding)
    end
    function HyperCube(T::AbstractVector{<:Tuple{<:Real,<:Real}}; Padding::Number=0.)
        HyperCube([T[i][1] for i in 1:length(T)], [T[i][2] for i in 1:length(T)]; Padding=Padding)
    end
    function HyperCube(vals::AbstractVector{<:Number}; Padding::Number=0.)
        length(vals) != 2 && throw("Input has too many components.")
        HyperCube([vals[1]],[vals[2]]; Padding=Padding)
    end
    HyperCube(vals::Tuple{<:Number,<:Number}; Padding::Number=0.) = HyperCube([vals[1]],[vals[2]]; Padding=Padding)
    HyperCube(center::AbstractVector{<:Number}, width::Real) = HyperCube(center .- (width/2), center .+ (width/2))
end

length(Cube::HyperCube) = length(Cube.L)

"""
    Inside(Cube::HyperCube, p::AbstractVector{<:Number}) -> Bool
Checks whether a point `p` lies inside `Cube`.
"""
Inside(Cube::HyperCube, p::AbstractVector{<:Number}) = all(Cube.L .≤ p) && all(p .≤ Cube.U)


import Base.in
"""
    in(Cube::HyperCube, p::AbstractVector{<:Number}) -> Bool
Checks whether a point `p` lies inside `Cube`.
"""
in(p::AbstractVector{<:Number}, Cube::HyperCube) = Inside(Cube, p)

"""
    ConstructCube(M::Matrix{<:Number}; Padding::Number=1/50) -> HyperCube
Returns a `HyperCube` which encloses the extrema of the columns of the input matrix.
"""
ConstructCube(M::AbstractMatrix{<:Number}; Padding::Number=0.) = HyperCube([minimum(M[:,i]) for i in 1:size(M,2)], [maximum(M[:,i]) for i in 1:size(M,2)]; Padding=Padding)
ConstructCube(V::AbstractVector{<:Number}; Padding::Number=0.) = HyperCube(extrema(V); Padding=Padding)
ConstructCube(PL::Plane, sol::AbstractODESolution; Padding::Number=0.) = ConstructCube(Deplanarize(PL,sol; N=300); Padding=Padding)
ConstructCube(Ps::AbstractVector{<:AbstractVector{<:Number}}; Padding::Number=0.) = ConstructCube(Unpack(Ps); Padding=Padding)

# Could speed this up by just using the points in sol.u without interpolation.
function ConstructCube(sol::AbstractODESolution, Npoints::Int=200; Padding::Number=0.)
    ConstructCube(Unpack(map(sol,range(sol.t[1],sol.t[end];length=Npoints))); Padding=Padding)
end
function ConstructCube(sols::AbstractVector{<:AbstractODESolution}, Npoints::Int=200; Padding::Number=0.)
    mapreduce(sol->ConstructCube(sol, Npoints; Padding=Padding), union, sols)
end

ConstructCube(Tup::Tuple{AbstractVector{<:Plane},AbstractVector{<:AbstractODESolution}}; Padding=0.) = ConstructCube(Tup[1], Tup[2]; Padding=Padding)
function ConstructCube(Planes::AbstractVector{<:Plane}, sols::AbstractVector{<:AbstractODESolution}; Padding=0.)
    @assert length(Planes) == length(sols)
    reduce(union, map((x,y)->ConstructCube(x,y; Padding=Padding), Planes, sols))
end

"""
    CubeWidths(H::HyperCube) -> Vector
Returns vector of widths of the `HyperCube`.
"""
CubeWidths(Cube::HyperCube) = Cube.U - Cube.L

"""
    CubeVol(Cube::HyperCube) -> Real
Computes volume of a `HyperCube` as the product of its sidelengths.
"""
CubeVol(Cube::HyperCube) = prod(CubeWidths(Cube))

"""
    Center(Cube::HyperCube) -> Vector
Returns center of mass of `Cube`.
"""
Center(Cube::HyperCube) = 0.5 * (Cube.L + Cube.U)

"""
    TranslateCube(Cube::HyperCube,x::AbstractVector{<:Number}) -> HyperCube
Returns a `HyperCube` object which has been translated by `x`.
"""
TranslateCube(Cube::HyperCube, x::AbstractVector{<:Number}) = HyperCube(Cube.L + x, Cube.U + x)

"""
    ResizeCube(Cube::HyperCube, factor::Real=1.) -> HyperCube
Resizes given `Cube` evenly in all directions but keeps center of mass fixed.
"""
function ResizeCube(Cube::HyperCube, factor::Real=1.)
    @assert factor > 0.
    center = Center(Cube);      halfwidths = (0.5*factor) * CubeWidths(Cube)
    HyperCube(center-halfwidths, center+halfwidths)
end



DomainSamples(Domain::Union{Tuple{Real,Real}, HyperCube}; N::Int=500) = DomainSamples(Domain, N)
DomainSamples(Cube::HyperCube, N::Int) = length(Cube) == 1 ? DomainSamples((Cube.L[1],Cube.U[1]), N) : throw("Domain not suitable.")
function DomainSamples(Domain::Tuple{Real,Real}, N::Int)
    @assert N > 2 && Domain[1] < Domain[2]
    range(Domain[1], Domain[2]; length=N) |> collect
end

import Base.range
function range(C::HyperCube; length::Int=100, kwargs...)
    @assert size(C.L,1) == 1 && length > 1
    range(C.L[1], C.U[1]; length=length, kwargs...)
end

DropCubeDims(Cube::HyperCube, dim::Int) = DropCubeDims(Cube, [dim])
function DropCubeDims(Cube::HyperCube, dims::AbstractVector{<:Int})
    @assert all(dim -> 1 ≤ dim ≤ length(Cube), dims)
    HyperCube(Drop(Cube.L, dims), Drop(Cube.U, dims))
    # keep = trues(length(Cube));     keep[dims] .= false
    # HyperCube(Cube.L[keep], Cube.U[keep])
end

"""
    FaceCenters(Cube::HyperCube) -> Vector{Vector}
Returns a `Vector` of the `2n`-many face centers of a `n`-dimensional `Cube`.
"""
function FaceCenters(Cube::HyperCube)
    C = Center(Cube);   W = 0.5CubeWidths(Cube)
    vcat(map(i->C-W[i]*BasisVector(i,length(C)), 1:length(C)), map(i->C+W[i]*BasisVector(i,length(C)), 1:length(C)))
end

"""
    Corners(C::HyperCube) -> Vector{Vector}
Returns the `2^n` corner points of a `n`-dimensional `HyperCube`.
"""
Corners(C::HyperCube{T}) where T<:Number = Corners([T[]], [(C.L[i], C.U[i]) for i in 1:length(C)])
function Corners(Res::AbstractVector{<:AbstractVector}, C::AbstractVector{<:Tuple})
    length(C) == 0 && return Res
    Tup = popfirst!(C)
    Corners(vcat([append!(copy(r), Tup[1]) for r in Res], [append!(copy(r), Tup[2]) for r in Res]), C)
    # n = length(Res)
    # for r in Res
    #     append!(r, Tup[1])
    # end
    # NewRes = vcat(Res,Res)
    # for i in n+1:2n
    #     NewRes[i][end] = Tup[2]
    # end
    # Corners(NewRes,C)
end


import Base: vcat
vcat(C1::HyperCube, C2::HyperCube) = HyperCube(vcat(C1.L,C2.L), vcat(C1.U,C2.U); Padding=0.0)

import Base: union, intersect
"""
    intersect(A::HyperCube, B::HyperCube) -> HyperCube
    intersect(Cubes::AbstractVector{<:HyperCube}) -> HyperCube
Returns new `HyperCube` which is the intersection of the given `HyperCube`s.
"""
intersect(A::HyperCube, B::HyperCube) = intersect([A, B])
function intersect(Cubes::AbstractVector{<:HyperCube})
    LowerMatrix = [Cubes[i].L for i in 1:length(Cubes)] |> Unpack
    UpperMatrix = [Cubes[i].U for i in 1:length(Cubes)] |> Unpack
    HyperCube([maximum(col) for col in eachcol(LowerMatrix)], [minimum(col) for col in eachcol(UpperMatrix)])
end

"""
    union(A::HyperCube, B::HyperCube) -> HyperCube
    union(Cubes::AbstractVector{<:HyperCube}) -> HyperCube
Returns new `HyperCube` which contains both given `HyperCube`s.
That is, the returned cube is strictly speaking not the union, but a cover (which contains the union).
"""
union(A::HyperCube, B::HyperCube) = union([A, B])
function union(Cubes::AbstractVector{<:HyperCube})
    LowerMatrix = [Cubes[i].L for i in 1:length(Cubes)] |> Unpack
    UpperMatrix = [Cubes[i].U for i in 1:length(Cubes)] |> Unpack
    HyperCube([minimum(col) for col in eachcol(LowerMatrix)], [maximum(col) for col in eachcol(UpperMatrix)])
end

import Base.==
==(A::HyperCube, B::HyperCube) = A.L == B.L && A.U == B.U
==(A::Plane, B::Plane) = A.stütz == B.stütz && A.Vx == B.Vx && A.Vy == B.Vy

PositiveDomain(n::Int, maxval::Real=1e5) = (@assert maxval > 1e-16;     HyperCube(1e-16ones(n), fill(maxval,n)))
PositiveDomain(indxs::BoolVector, maxval::Real=1e5) = (@assert maxval > 1e-16;     HyperCube([(indxs[i] ? 1e-16 : -maxval) for i in eachindex(indxs)], fill(maxval,length(indxs))))
NegativeDomain(n::Int, maxval::Real=1e5) = (@assert maxval > 1e-16;     HyperCube(fill(-maxval,n), -1e-16ones(n)))
NegativeDomain(indxs::BoolVector, maxval::Real=1e5) = (@assert maxval > 1e-16;     HyperCube(fill(-maxval,length(indxs)), [(indxs[i] ? -1e-16 : maxval) for i in eachindex(indxs)]))
FullDomain(n::Int, maxval::Real=1e5) = (@assert maxval > 1e-16;     HyperCube(fill(-maxval,n), fill(maxval,n)))

import Base.rand
rand(C::HyperCube) = rand(C, Val(length(C)))
rand(C::HyperCube, ::Val{1}) = C.L[1] + (C.U[1] - C.L[1])*rand()
function rand(C::HyperCube, ::Val)
    f(l,u) = l + (u-l)*rand()
    map(f,C.L,C.U)
end

import Base.clamp
clamp(x::AbstractVector, C::HyperCube) = clamp(x, C.L, C.U)


struct EmbeddedODESolution{T,N,uType,uType2,EType,tType,rateType,P,A,IType,DE} <: AbstractODESolution{T,N,uType}
    u::uType
    u_analytic::uType2
    errors::EType
    t::tType
    k::rateType
    prob::P
    alg::A
    interp::IType
    dense::Bool
    tslocation::Int
    destats::DE
    retcode::Symbol
    Embedding::Function
end
(ES::EmbeddedODESolution)(t::Real,deriv::Type=Val{0};idxs=nothing,continuity=:left) = ES.Embedding(ES.interp(t,idxs,deriv,ES.prob.p,continuity))
(ES::EmbeddedODESolution)(Ts::AbstractVector{<:Real},deriv::Type=Val{0};kwargs...) = map(t->ES(t,deriv;kwargs...),Ts)
# Need to use push-forward to embed vectors, i.e. ForwardDiff.jacobian(Embedding, basepoint) * tangentvector
# (ES::EmbeddedODESolution)(v,t,deriv::Type=Val{0};idxs=nothing,continuity=:left) = sol.interp(v,t,idxs,deriv,sol.prob.p,continuity)


function EmbeddedODESolution(u, u_analytic, errors, t, k, prob, alg, interp, dense, tslocation, destats, retcode, Embedding)
    EmbeddedODESolution{typeof(u[1]), length(u[1]), typeof(u), typeof(u_analytic), typeof(errors),
    typeof(t), typeof(k), typeof(prob), typeof(alg), typeof(interp),typeof(destats)}(
    u, u_analytic, errors, t, k, prob, alg, interp, dense, tslocation, destats, retcode, Embedding)
end

"""
    EmbeddedODESolution(sol::AbstractODESolution, Embedding::Function) -> AbstractODESolution
    EmbeddedODESolution(sol::AbstractODESolution, PL::Plane) -> AbstractODESolution
Maps the solution `sol(t)` to some ODE into a larger space via `Embedding∘sol`.
"""
function EmbeddedODESolution(sol::AbstractODESolution{T,N,uType}, Embedding::Function=identity) where {T,N,uType}
    newu = map(Embedding, sol.u);    newk = [map(Embedding, k) for k in sol.k]
    EmbeddedODESolution(newu, isnothing(sol.u_analytic) ? nothing : Embedding∘sol.u_analytic, # Is this translation correct?
                 sol.errors, sol.t, newk, sol.prob, sol.alg,
                 sol.interp, # Leaving old interp object as is and only using embedding on calls of EmbeddedODESolution objects themselves.
                 sol.dense, sol.tslocation, sol.destats, sol.retcode, Embedding)
end
EmbeddedODESolution(sol::AbstractODESolution, PL::Plane) = EmbeddedODESolution(sol, PlaneCoordinates(PL))
function EmbeddedODESolution(sols::AbstractVector{<:AbstractODESolution}, Planes::AbstractVector{<:Plane})
    @assert length(sols) == length(Planes)
    map(EmbeddedODESolution, sols, Planes)
end
EmbeddedODESolution(PL::Union{Plane, AbstractVector{<:Plane}}, sol::Union{AbstractODESolution,AbstractVector{<:AbstractODESolution}}) = EmbeddedODESolution(sol, PL)
EmbeddedODESolution(Embedding::Function, sol::AbstractODESolution) = EmbeddedODESolution(sol, Embedding)
