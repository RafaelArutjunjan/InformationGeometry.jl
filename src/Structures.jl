

suff(x::Number) = typeof(float(x))
suff(X::AbstractArray) = length(X) != 0 ? suff(X[1]) : error("Empty Array in suff.")

"""
The `DataSet` type is a container for datapoints. It holds 3 vectors `x, y, sigma` where the components of `sigma` quantify the error bars associated with each measurement.
"""
struct DataSet
    x::AbstractVector
    y::AbstractVector
    sigma::AbstractArray
    function DataSet(x,y,sigma)
        if length(x) != length(y)
            throw(ArgumentError("Dimension mismatch. length(x) = $(length(x)), length(y) = $(length(y))"))
        elseif length(sigma) != length(y) && length(sigma) != 1
            throw(ArgumentError("Dimension mismatch. length(y) = $(length(y)), length(sigma) = $(length(sigma))"))
        elseif length(sigma) == 1
            sigma = sigma*ones(length(y))
        end
        new(x,y,sigma)
    end
    function DataSet(DF::Union{DataFrame,AbstractMatrix})
        size(DF)[2] != 3 && throw("Unclear dimensions.")
        new(DF[:,1], DF[:,2], DF[:,3])
    end
end

"""
In addition to a `DataSet`, a `DataModel` contains the model as a function `model(x,p)` and its derivative `dmodel(x,p)` where `x` denotes the x value of the data and `p` is a vector of parameters on which the model depends. Crucially, `dmodel` contains the derivatives of the model with respect to the parameters `p`, not the x values.
"""
struct DataModel
    Data::DataSet
    model::Function
    dmodel::Function
    # Provide dModel using ForwardDiff if not given
    DataModel(DF::DataFrame, args...) = DataModel(DataSet(DF),args...)
    function DataModel(D::DataSet,F::Function)
        Autodmodel(x::Q,p) where Q<:Real = reshape(ForwardDiff.gradient(z->F(x,z),p),1,length(p))
        function Autodmodel(x::Vector{<:Real},p)
            Res = Array{suff(p)}(undef,length(x),length(p))
            for i in 1:length(x)
                Res[i,:] = Autodmodel(x[i],p)
            end
            Res
        end
        new(D,F,Autodmodel)
    end
    DataModel(DS::DataSet,M::Function,dM::Function) = new(DS,M,dM)
end

xdata(DS::DataSet) = DS.x;      xdata(DM::DataModel) = xdata(DM.Data)
ydata(DS::DataSet) = DS.y;      ydata(DM::DataModel) = ydata(DM.Data)
sigma(DS::DataSet) = DS.sigma;  sigma(DM::DataModel) = sigma(DM.Data)

function Sparsify(DS::DataSet,B::Vector=rand(Bool,length(DS.x)))
    length(B) != length(DS.x) && throw(ArgumentError("Sparsify: Vector not same number of components as datapoints."))
    !(length(DS.x[1]) == length(DS.y[1]) == length(DS.sigma[1])) && throw("Not programmed yet.")
    return DataSet(DS.x[B],DS.y[B],DS.sigma[B])
end
function Sparsify(DM::DataModel,B::Vector=rand(Bool,length(xdata(DM))))
    return DataModel(Sparsify(DM.Data,B),DM.model,DM.dmodel)
end



"""
Specifies a 2D plane in the so-called parameter form using 3 vectors.
"""
struct Plane
    stütz::Vector{<:Real}
    Vx::Vector{<:Real}
    Vy::Vector{<:Real}
    Projector::Matrix{<:Real}
    function Plane(stütz,Vx,Vy)
        if length(stütz) == 2 stütz = [stütz[1],stütz[2],0] end
        dim = length(stütz)
        if (length(Vx) != dim) || (length(Vy) != dim)
            throw(ArgumentError("Dimension mismatch. length(stütz) = $dim, length(Vx) = $(length(Vx)), length(Vy) = $(length(Vy))"))
        elseif dot(Vx,Vy) != 0
            println("Plane: Making Vy orthogonal to Vx.")
            new(float.(stütz),float.(normalize(Vx)),Make2ndOrthogonal(Vx,Vy),ProjectionOperator([Vx Vy]))
            # throw(ArgumentError("Basis Vectors of Plane not orthogonal."))
        else
            new(float.(stütz),float.(normalize(Vx)),float.(normalize(Vy)),float.(ProjectionOperator([Vx Vy])))
        end
    end
end

function BasisVector(Slot::Int,dims::Int)
    Res = zeros(dims);    Res[Slot] = 1;    Res
end
function PlaneCoordinates(PL::Plane,x::Vector)
    length(x) != 2 && throw(ArgumentError("length(coordinates) != 2"))
    return PL.stütz .+ x[1]*PL.Vx .+ x[2]*PL.Vy
end
using LinearAlgebra
import LinearAlgebra.dot
dot(A::AbstractMatrix, x::Vector, y::Vector=x) = transpose(x)*A*y
dot(x) = dot(x,x)
länge(x::Vector{<:Real}) = sqrt(dot(x,x))
EuclideanDistance(x::Vector, y::Vector) = länge(x.-y)
IsOnPlane(PL::Plane,x::Vector)::Bool = (DistanceToPlane(PL,x) == 0)
TranslatePlane(PL::Plane, v::Vector) = Plane(PL.stütz .+ v, PL.Vx, PL.Vy)
RotatePlane(PL::Plane, rads=pi/2) = Plane(PL.stütz,cos(rads)*PL.Vx + sin(rads)*PL.Vy, cos(rads)*PL.Vy - sin(rads)*PL.Vx)
function RotationMatrix(PL::Plane,rads::Q) where Q<:Real
    V = PL.Vx*transpose(PL.Vx) + PL.Vy*transpose(PL.Vy)
    W = PL.Vx*transpose(PL.Vy) - PL.Vy*transpose(PL.Vx)
    Diagonal(ones(length(PL.stütz))) + (cos(rads)-1.)*V -sin(rads)*W
end
RotateVector(PL::Plane,v::Vector,rads::Q) where Q<:Real = RotationMatrix(PL,rads)*v

function DecomposeWRTPlane(PL::Plane,x::Vector)
    !IsOnPlane(PL,x) && throw(ArgumentError("Decompose Error: Vector not on Plane."))
    V = x - PL.stütz
    [ProjectOnto(V,PL.Vx), ProjectOnto(V,PL.Vy)]
    # [dot(V,PL.Vx)/dot(PL.Vx,PL.Vx),dot(V,PL.Vy)/dot(PL.Vy,PL.Vy)]
end
DistanceToPlane(PL::Plane,x::Vector) = (diagm(ones(Float64,length(x))) .- PL.Projector) * (x .- PL.stütz) |> länge
ProjectOntoPlane(PL::Plane,x::Vector) = PL.Projector*(x .- PL.stütz) .+ PL.stütz

function ProjectionOperator(A::Matrix)
    size(A,2) != 2 && println("ProjectionOperator: Matrix size $(size(A)) not as expected.")
    A * inv(transpose(A) * A) * transpose(A)
end
ProjectionOperator(PL::Plane) = ProjectionOperator([PL.Vx PL.Vy])

function Make2ndOrthogonal(X::Vector,Y::Vector)
    Basis = GramSchmidt(float.([X,Y]))
    # Maybe add check for orientation?
    return Basis[2]
end

function MinimizeOnPlane(PL::Plane,F::Function,initial::Vector=[1,-1.]; tol=1e-5)
    G(x) = F(PlaneCoordinates(PL,x))
    X = Optim.minimizer(optimize(G,initial, BFGS(), Optim.Options(g_tol=tol), autodiff = :forward))
    PlaneCoordinates(PL,X)
end

# project v onto u
ProjectOnto(v::Vector,u::Vector) = dot(v,u)/dot(u,u) .* u

function GramSchmidt(v::Vector{<:Real},dim::Int=length(v))
    Basis = Vector{typeof(v)}(undef,0)
    push!(Basis,v)
    for i in 2:length(v)
        push!(Basis,BasisVector(i,length(v)))
    end
    GramSchmidt([normalize(Basis[i]) for i in 1:length(Basis)])
end
function GramSchmidt(Basis::Vector{Vector{Q}}) where Q
    ONBasis = float.(Basis)
    for j in 1:length(Basis)
        for i in 2:j
            ONBasis[j] .-= ProjectOnto(Basis[j],ONBasis[i-1])
        end
    end
    [normalize(ONBasis[i]) for i in 1:length(ONBasis)]
end

"""
The `HyperCube` type has the fields `vals::Vector{Vector}`, which stores the intervals which define the hypercube and `dim::Int`, which gives the dimension.
Overall it just offers a convenient and standardized way of passing domains for integration or plotting between functions without having to check that these domains are sensible every time.
Examples for constructing `HyperCube`s:
```
HyperCube([[1,3],[pi,2pi],[-500.0,100.0]])
HyperCube([[-1,1]])
HyperCube([-1,1])
HyperCube(LowerUpper([-1,-5],[0,-4]))
HyperCube(collect([-7,7.] for i in 1:3))
```
The `HyperCube` type is closely related to the `LowerUpper` type and they can be easily converted into each other.
Examples for quantities that can be computed from and operations involving `HyperCube` objects:
```
CubeVol(X)
TranslateCube(X,v::Vector)
CubeWidths(X)
```
"""
struct HyperCube{Q<:Real}
    vals::Vector{Vector{Q}}
    dim::Int
    function HyperCube(vals::Vector)
        vals = float.(vals)
        types = typeof(vals[1][1])
        for i in 1:length(vals)
            length(vals[i]) != 2 && error("Unsuitable Hypercube.")
            typeof(vals[i][1]) != types && error("Type Mismatch in Hypercube.")
            vals[i][1] > vals[i][2] && error("HyperCube: Orientation wrong, Interval $i: [$(vals[i][1]),$(vals[i][2])] not allowed.")
        end
        new{types}(vals,length(vals))
    end
    # Allow input [a,b] for 1D HyperCubes
    HyperCube(vals::Vector{<:Real}) = HyperCube([vals])
end

"""
The `LowerUpper` type has a field `L` and a field `U` which respectively store the lower and upper boundaries of an N-dimensional Hypercube.
It is very closely related to (and stores the same information as) the `HyperCube` type.
Examples for constructing `LowerUpper`s:
```
LowerUpper([-1,-5,pi],[0,-4,2pi])
LowerUpper(HyperCube([[5,6],[-pi,0.5]]))
LowerUpper(collect(1:5),collect(15:20))
```
Examples for quantities that can be computed from and operations involving `LowerUpper` objects:
```
CubeVol(X)
TranslateCube(X,v::Vector)
CubeWidths(X)
```
"""
struct LowerUpper{Q<:Real}
    L::Vector{Q}
    U::Vector{Q}
    function LowerUpper(lowers,uppers)
        lowers = float.(lowers); uppers = float.(uppers)
        length(lowers) != length(uppers) && throw(ArgumentError("Dimensional Mismatch in LowerUpper."))
        for i in 1:length(lowers)
            lowers[i] > uppers[i] && throw(ArgumentError("LowerUpper Constructor: lowers[$i] > uppers[$i]."))
        end
        new{suff(lowers)}(lowers,uppers)
    end
    function LowerUpper(H::HyperCube)
        S = H.vals; l = Vector{suff(S)}(undef,length(S))
        u = Vector{suff(S)}(undef,length(S))
        for i in 1:length(S)
            l[i] = S[i][1];        u[i] = S[i][2]
        end
        new{suff(S)}(l,u)
    end
    LowerUpper(H::Vector) = LowerUpper(HyperCube(H))
end

SensibleOutput(LU::LowerUpper) = LU.L,LU.U
function HyperCube(LU::LowerUpper)
    R = Vector{typeof(LU.U)}(undef,length(LU.U))
    for i in 1:length(LU.U)
        R[i] = [LU.L[i],LU.U[i]]
    end
    HyperCube(R)
end

function SensibleOutput(Res::Vector)
    if isa(Res[1],Real)
        return Res[1], Res[2]
    elseif isa(Res[1],Vector) && typeof(Res[1][1])<:Real
        u = Vector{suff(Res)}(undef,length(Res)); v = similar(u)
        for i in 1:length(Res)
            if length(Res[i]) != 2
                error("SensibleOutput only accepts inner vectors of length 2.")
            end
            u[i] = Res[i][1]
            v[i] = Res[i][2]
        end
        return u,v
    else
        throw(ArgumentError("Expected Vector{Real} or Vector{Vector{Real}}, got $(typeof(Res))"))
    end
end

Unpack(H::HyperCube) = Unpack(H.vals)
function Unpack(Z::Vector{Vector{Q}}) where Q
    N = length(Z); M = length(Z[1])
    A = Array{suff(Z)}(undef,N,M)
    for i in 1:N
        for j in 1:M
            A[i,j] = Z[i][j]
        end
    end
    A
end

function CubeVol(Space::Vector)
    lowers,uppers = SensibleOutput(Space)
    prod(uppers .- lowers)
end
CubeWidths(S::LowerUpper) = S.U .- S.L
CubeWidths(H::HyperCube) = CubeWidths(LowerUpper(H))

CubeVol(X::HyperCube) = CubeVol(LowerUpper(X))
CubeVol(S::LowerUpper) = prod(CubeWidths(S))

function TranslateCube(H::HyperCube,x::Vector)
    H.dim != length(x) && throw("Translation vector must be of same dimension as hypercube.")
    [H.vals[i] .+ x[i] for i in 1:length(x)] |> HyperCube
end
TranslateCube(LU::LowerUpper,x::Vector) = TranslateCube(HyperCube(LU),x) |> LowerUpper

"""
    CoverCubes(A::HyperCube,B::HyperCube)
Return new HyperCube which covers two other given HyperCubes.
"""
function CoverCubes(A::HyperCube,B::HyperCube)
    A.dim != B.dim && throw("CoverCubes: Cubes have different dims.")
    ALU = LowerUpper(A);    BLU = LowerUpper(B)
    lower = ALU.L; upper = ALU.U
    for i in 1:A.dim
        if ALU.L[i] > BLU.L[i]
            lower[i] = BLU.L[i]
        end
        if ALU.U[i] < BLU.U[i]
            upper[i] = BLU.U[i]
        end
    end
    HyperCube(LowerUpper(lower,upper))
end


normalize(x::Vector,scaling::Float64=1.0) = scaling.*x ./ sqrt(sum(x.^2))
function normalizeVF(u::Vector{<:Real},v::Vector{<:Real},scaling::Float64=1.0)
    newu = u;    newv = v;    factor = Float64
    for i in 1:length(u)
        factor = sqrt(u[i]^2 + v[i]^2)
        newu[i] = (scaling/factor)*u[i]
        newv[i] = (scaling/factor)*v[i]
    end
    newu,newv
end
function normalizeVF(u::Vector{<:Real},v::Vector{<:Real},PlanarCube::HyperCube,scaling::Float64=1.0)
    PlanarCube.dim != 2 && throw("normalizeVF: Cube not planar.")
    newu = u;    newv = v
    Widths = CubeWidths(PlanarCube) |> normalize
    for i in 1:length(u)
        factor = sqrt(u[i]^2 + v[i]^2)
        newu[i] = (scaling/factor)*u[i] * Widths[1]
        newv[i] = (scaling/factor)*v[i] * Widths[2]
    end
    newu,newv
end
