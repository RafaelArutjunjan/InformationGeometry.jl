

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
            stütz = SVector{length(Vx)}(float.(stütz));     Vx = SVector{length(Vx)}(float.(Vx))
            Vy = SVector{length(Vx)}(float.(Vy))
            return new(stütz, Vx, Vy)
        else
            return new(float.(stütz), float.(Vx), float.(Vy))
        end
    end
end

length(PL::Plane) = length(PL.stütz)

function MLEinPlane(DM::AbstractDataModel, PL::Plane, start::AbstractVector{<:Number}=0.0001rand(2); tol::Real=1e-8)
    length(start) != 2 && throw("Dimensional Mismatch.");      model = Predictor(DM)
    planarmod(x, θ::AbstractVector{<:Number}; kwargs...) = model(x, PlaneCoordinates(PL,θ); kwargs...)
    curve_fit(Data(DM), planarmod, start; tol=tol).param
end

function PlanarDataModel(DM::AbstractDataModel, PL::Plane)
    @assert DM isa DataModel
    model = Predictor(DM);      dmodel = dPredictor(DM)
    newmod = (x,θ::AbstractVector{<:Number}; kwargs...) -> model(x, PlaneCoordinates(PL,θ); kwargs...)
    dnewmod = (x,θ::AbstractVector{<:Number}; kwargs...) -> dmodel(x, PlaneCoordinates(PL,θ); kwargs...) * [PL.Vx PL.Vy]
    mle = MLEinPlane(DM, PL)
    DataModel(Data(DM), newmod, dnewmod, mle, loglikelihood(DM,PlaneCoordinates(PL, mle)), true)
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
RotatePlane(PL::Plane, rads::Real=π/2) = Plane(PL.stütz,cos(rads)*PL.Vx + sin(rads)*PL.Vy, cos(rads)*PL.Vy - sin(rads)*PL.Vx)
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
    Basis = GramSchmidt(float.([X,Y]))
    # Maybe add check for orientation?
    return Basis[2]
end

"""
    MinimizeOnPlane(PL::Plane,F::Function,initial::AbstractVector=[1,-1.]; tol::Real=1e-5)
Minimizes given function in Plane and returns the optimal point in the ambient space in which the plane lies.
"""
function MinimizeOnPlane(PL::Plane, F::Function, initial::AbstractVector=[1,-1.]; tol::Real=1e-5)
    G(x) = F(PlaneCoordinates(PL,x))
    X = Optim.minimizer(optimize(G,initial, BFGS(), Optim.Options(g_tol=tol), autodiff = :forward))
    PlaneCoordinates(PL,X)
end

"""
    ProjectOnto(v::Vector,u::Vector)
Project `v` onto `u`.
"""
ProjectOnto(v::AbstractVector,u::AbstractVector) = (dot(v,u) / dot(u,u)) * u

"""
    ParallelPlanes(PL::Plane,v::AbstractVector,range) -> Vector{Plane}
Returns Vector of Planes which have been translated by `a .* v` for all `a` in `range`.
"""
function ParallelPlanes(PL::Plane,v::AbstractVector,range::Union{AbstractRange,AbstractVector})
    norm(v) == 0. && throw("Vector has length zero.")
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
    ONBasis = float.(Basis)
    for j in 1:length(Basis)
        for i in 2:j
            ONBasis[j] .-= ProjectOnto(Basis[j],ONBasis[i-1])
        end
    end
    [normalize(ONBasis[i]) for i in 1:length(ONBasis)]
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
TranslateCube(X,v::Vector)
CubeWidths(X)
```
"""
struct HyperCube{Q<:Number} <: Cuboid
    L::AbstractVector{Q}
    U::AbstractVector{Q}
    function HyperCube(lowers::AbstractVector{<:Number},uppers::AbstractVector{<:Number}; Padding::Number=0.)
        @assert length(lowers) == length(uppers)
        if Padding != 0.
            diff = (uppers - lowers) .* (Padding / 2.)
            lowers -= diff;     uppers += diff
        end
        !all(lowers .≤ uppers) && throw("First argument of HyperCube must be larger than second.")
        if length(lowers) < 20
            return new{suff(lowers)}(SVector{length(lowers)}(float.(lowers)), SVector{length(uppers)}(float.(uppers)))
        else
            return new{suff(lowers)}(float.(lowers),float.(uppers))
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


# Could speed this up by just using the points in sol.u without interpolation.
function ConstructCube(sol::AbstractODESolution, Npoints::Int=200; Padding::Number=0.)
    ConstructCube(Unpack(map(sol,range(sol.t[1],sol.t[end],length=Npoints))); Padding=Padding)
end
function ConstructCube(sols::Vector{<:AbstractODESolution}, Npoints::Int=200; Padding::Number=0.)
    mapreduce(sol->ConstructCube(sol, Npoints; Padding=Padding), union, sols)
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
    Center(Cube::HyperCube) |> Vector
Returns center of mass of `Cube`.
"""
Center(Cube::HyperCube) = 0.5 * (Cube.L + Cube.U)

"""
    TranslateCube(Cube::HyperCube,x::Vector{<:Number}) -> HyperCube
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


DropCubeDim(Cube::HyperCube, dim::Int) = DropCubeDims(Cube, [dim])
function DropCubeDims(Cube::HyperCube, dims::Union{AbstractVector{<:Int}, AbstractRange{<:Int}})
    @assert all(dim -> 1 ≤ dim ≤ length(Cube), dims)
    keep = trues(length(Cube));     keep[dims] .= false
    HyperCube(Cube.L[keep], Cube.U[keep])
end

"""
    FaceCenters(Cube::HyperCube) -> Vector{Vector}
Returns a `Vector` of the `2n`-many face centers of a `n`-dimensional `Cube`.
"""
function FaceCenters(Cube::HyperCube)
    C = Center(Cube);   W = 0.5CubeWidths(Cube)
    vcat(map(i->C-W[i]*BasisVector(i,length(C)), 1:length(C)), map(i->C+W[i]*BasisVector(i,length(C)), 1:length(C)))
end

### Slower than union
# """
#     CoverCubes(A::HyperCube, B::HyperCube)
# Return a new HyperCube which covers two other given HyperCubes.
# """
# function CoverCubes(A::HyperCube, B::HyperCube)
#     length(A) != length(B) && throw("CoverCubes: Cubes have different dims.")
#     lower = A.L; upper = A.U
#     for i in 1:length(A)
#         if A.L[i] > B.L[i]
#             lower[i] = B.L[i]
#         end
#         if A.U[i] < B.U[i]
#             upper[i] = B.U[i]
#         end
#     end
#     HyperCube(lower,upper)
# end
# CoverCubes(A::HyperCube, B::HyperCube) = Union(A, B)
# CoverCubes(args...) = CoverCubes([args...])
# CoverCubes(V::Vector{<:HyperCube}) = Union(V)

"""
    Intersect(A::HyperCube, B::HyperCube) -> HyperCube
    Intersect(Cubes::Vector{<:HyperCube}) -> HyperCube
Returns new `HyperCube` which is the intersection of the given `HyperCube`s.
"""
Intersect(A::HyperCube, B::HyperCube) = Intersect([A, B])
function Intersect(Cubes::Vector{<:HyperCube})
    LowerMatrix = [Cubes[i].L for i in 1:length(Cubes)] |> Unpack
    UpperMatrix = [Cubes[i].U for i in 1:length(Cubes)] |> Unpack
    HyperCube([maximum(col) for col in eachcol(LowerMatrix)], [minimum(col) for col in eachcol(UpperMatrix)])
end

"""
    Union(A::HyperCube, B::HyperCube) -> HyperCube
    Union(Cubes::Vector{<:HyperCube}) -> HyperCube
Returns new `HyperCube` which contains both given `HyperCube`s.
That is, the returned cube is strictly speaking not the union, but a cover (which contains the union).
"""
Union(A::HyperCube, B::HyperCube) = Union([A, B])
function Union(Cubes::Vector{<:HyperCube})
    LowerMatrix = [Cubes[i].L for i in 1:length(Cubes)] |> Unpack
    UpperMatrix = [Cubes[i].U for i in 1:length(Cubes)] |> Unpack
    HyperCube([minimum(col) for col in eachcol(LowerMatrix)], [maximum(col) for col in eachcol(UpperMatrix)])
end

import Base: union, intersect
union(A::HyperCube, B::HyperCube) = Union(A, B)
intersect(A::HyperCube, B::HyperCube) = Intersect(A, B)

import Base.==
==(A::HyperCube, B::HyperCube) = A.L == B.L && A.U == B.U
==(A::Plane, B::Plane) = A.stütz == B.stütz && A.Vx == B.Vx && A.Vy == B.Vy

PositiveDomain(n::Int) = HyperCube(zeros(n), fill(Inf,n))
PositiveDomain(indxs::BitVector) = HyperCube([(indxs[i] ? 0. : -Inf) for i in eachindex(indxs)], fill(Inf,length(indxs)))
NegativeDomain(n::Int) = HyperCube(fill(-Inf,n), zeros(n))
NegativeDomain(indxs::BitVector) = HyperCube(fill(-Inf,length(indxs)), [(indxs[i] ? 0. : Inf) for i in eachindex(indxs)])
FullDomain(n::Int) = HyperCube(fill(-Inf,n), fill(Inf,n))

import Base.rand
rand(Cube::HyperCube) = Cube.L + (Cube.U - Cube.L) .* rand(length(Cube.L))
