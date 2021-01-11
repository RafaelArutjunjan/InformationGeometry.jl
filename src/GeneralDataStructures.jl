

abstract type AbstractDataSet end
abstract type AbstractDataModel end
abstract type Cuboid end

struct DataSet <: AbstractDataSet end
struct DataSetExact <: AbstractDataSet end
struct DataModel <: AbstractDataModel end
struct HyperCube <: Cuboid end
struct ModelMap end
ModelOrFunction = Union{Function,ModelMap}




# Need to implement for each DataSet:   xdata, ydata, sigma, xsigma, ysigma, InvCov, Npoints, xdim, ydim,
#                                       WoundX (already generic), logdetInvCov (already generic), length (already generic)
# Need to implement for each DataModel: the above and: Data, model (Predictor), dmodel (dPredictor),
#                                       pdim, MLE, LogLikeMLE,
#                                       EmbeddingMap (already (kind of) generic), EmbeddingMatrix (already (kind of) generic)
#                                       Score (already (kind of) generic), FisherMetric (already (kind of) generic)



# Generic Methods for AbstractDataSets      -----       May be superceded by more specialized functions!
import Base.length
length(DS::AbstractDataSet) = Npoints(DS)
WoundX(DS::AbstractDataSet) = Windup(xdata(DS),xdim(DS))
logdetInvCov(DS::AbstractDataSet) = logdet(InvCov(DS))
DataspaceDim(DS::AbstractDataSet) = Npoints(DS) * ydim(DS)
# Data(DS::AbstractDataSet) = DS

xdist(DS::AbstractDataSet) = xDataDist(DS)
ydist(DS::AbstractDataSet) = yDataDist(DS)

Npoints(DS::AbstractDataSet) = Npoints(dims(DS))
xdim(DS::AbstractDataSet) = xdim(dims(DS))
ydim(DS::AbstractDataSet) = ydim(dims(DS))
Npoints(dims::Tuple{Int,Int,Int}) = dims[1]
xdim(dims::Tuple{Int,Int,Int}) = dims[2]
ydim(dims::Tuple{Int,Int,Int}) = dims[3]


# Generic Methods for AbstractDataModels      -----       May be superceded by more specialized functions!
pdim(DM::AbstractDataModel) = pdim(Data(DM), Predictor(DM))
MLE(DM::AbstractDataModel) = FindMLE(DM)
LogLikeMLE(DM::AbstractDataModel) = loglikelihood(DM, MLE(DM))


# Generic passthrough of queries from AbstractDataModel to AbstractDataSet for following functions:
# for F in [xdata, ydata, sigma, xsigma, ysigma, InvCov, dims, Npoints, length, xdim, ydim,
#                     logdetInvCov, WoundX, DataspaceDim, xnames, ynames, xdist, ydist]
#     F(DM::AbstractDataModel) = F(Data(DM))
# end
xdata(DM::AbstractDataModel) = xdata(Data(DM))
ydata(DM::AbstractDataModel) = ydata(Data(DM))
sigma(DM::AbstractDataModel) = sigma(Data(DM))
xsigma(DM::AbstractDataModel) = xsigma(Data(DM))
ysigma(DM::AbstractDataModel) = ysigma(Data(DM))
InvCov(DM::AbstractDataModel) = InvCov(Data(DM))

Npoints(DM::AbstractDataModel) = Npoints(Data(DM))
length(DM::AbstractDataModel) = length(Data(DM))
xdim(DM::AbstractDataModel) = xdim(Data(DM))
ydim(DM::AbstractDataModel) = ydim(Data(DM))
dims(DM::AbstractDataModel) = dims(Data(DM))

logdetInvCov(DM::AbstractDataModel) = logdetInvCov(Data(DM))
WoundX(DM::AbstractDataModel) = WoundX(Data(DM))
DataspaceDim(DM::AbstractDataModel) = DataspaceDim(Data(DM))

xnames(DM::AbstractDataModel) = xnames(Data(DM))
ynames(DM::AbstractDataModel) = ynames(Data(DM))

xdist(DM::AbstractDataModel) = xdist(Data(DM))
ydist(DM::AbstractDataModel) = ydist(Data(DM))


# Generic Methods which are not simply passed through
pnames(DM::AbstractDataModel) = pnames(DM, Predictor(DM))
pnames(DM::AbstractDataModel, M::ModelMap) = pnames(M)
pnames(DM::AbstractDataModel, F::Function) = CreateSymbolNames(pdim(DM),"θ")






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



function AutoDiffDmodel(DS::AbstractDataSet, model::Function; custom::Bool=false)
    Autodmodel(x::Number,θ::AbstractVector{<:Number}; kwargs...) = transpose(ForwardDiff.gradient(z->model(x,z; kwargs...),θ))
    NAutodmodel(x::AbstractVector{<:Number},θ::AbstractVector{<:Number}; kwargs...) = transpose(ForwardDiff.gradient(z->model(x,z; kwargs...),θ))
    AutodmodelN(x::Number,θ::AbstractVector{<:Number}; kwargs...) = ForwardDiff.jacobian(p->model(x,p; kwargs...),θ)
    NAutodmodelN(x::AbstractVector{<:Number},θ::AbstractVector{<:Number}; kwargs...) = ForwardDiff.jacobian(p->model(x,p; kwargs...),θ)
    # Getting extract_gradient! error from ForwardDiff when using gradient method with observables
    # CustomAutodmodel(x::Union{Number,AbstractVector{<:Number}},θ::AbstractVector{<:Number}) = transpose(ForwardDiff.gradient(p->model(x,p),θ))
    CustomAutodmodelN(x::Union{Number,AbstractVector{<:Number}},θ::AbstractVector{<:Number}; kwargs...) = ForwardDiff.jacobian(p->model(x,p; kwargs...),θ)
    if ydim(DS) == 1
        custom && return CustomAutodmodelN
        return xdim(DS) == 1 ? Autodmodel : NAutodmodel
    else
        custom && return CustomAutodmodelN
        return xdim(DS) == 1 ? AutodmodelN : NAutodmodelN
    end
end


"""
    DetermineDmodel(DS::AbstractDataSet, model::Function)::Function
Returns appropriate function which constitutes the automatic derivative of the `model(x,θ)` with respect to the parameters `θ` depending on the format of the x-values and y-values of the DataSet.
"""
function DetermineDmodel(DS::AbstractDataSet, model::Function, TryOptimize::Bool=false; custom::Bool=false)
    # Try to use symbolic dmodel:
    if TryOptimize
        Symbolic_dmodel = Optimize(DS, model; inplace=false)[2]
        Symbolic_dmodel != nothing && return Symbolic_dmodel
    end
    AutoDiffDmodel(DS, model; custom=custom)
end
function DetermineDmodel(DS::AbstractDataSet, M::ModelMap, TryOptimize::Bool=false; custom::Bool=ValToBool(M.CustomEmbedding))
    ModelMap(DetermineDmodel(DS, M.Map, TryOptimize; custom=custom), M)
end


function CheckModelHealth(DS::AbstractDataSet, model::ModelOrFunction)
    P = ones(pdim(DS,model));   X = xdim(DS) < 2 ? xdata(DS)[1] : xdata(DS)[1:xdim(DS)]
    try  model(X,P)   catch Err
        throw("Got xdim=$(xdim(DS)) but model appears to not accept x-values of this size.")
    end
    !(size(model(X,P),1) == ydim(DS)) && println("Got ydim=$(ydim(DS)) but output of model does not have this size.")
    !(model(X,P) isa SVector) && ydim(DS) > 1 && @warn "It may be beneficial for the overall performance to define the model function such that it outputs static vectors, i.e. SVectors."
    return
end



DataDist(Y::AbstractVector, Sig::AbstractVector, dist=Normal) = product_distribution([dist(Y[i],Sig[i]) for i in eachindex(Y)])
DataDist(Y::AbstractVector, Sig::AbstractMatrix, dist=MvNormal) = dist(Y, Symmetric(Sig))
yDataDist(DS::DataSet) = DataDist(ydata(DS), sigma(DS))
xDataDist(DS::DataSet) = InformationGeometry.Dirac(xdata(DS))
yDataDist(DM::DataModel) = yDataDist(Data(DM));    xDataDist(DM::DataModel) = xDataDist(Data(DM))



"""
    pdim(DS::AbstractDataSet, model::ModelOrFunction) -> Int
Infers the (minimal) number of components that the given function `F` accepts as input by successively testing it on vectors of increasing length.
"""
pdim(DS::AbstractDataSet, model::ModelOrFunction) = xdim(DS) < 2 ? GetArgLength(p->model(xdata(DS)[1],p)) : GetArgLength(p->model(xdata(DS)[1:xdim(DS)],p))

function GetArgLength(F::Function; max::Int=100)::Int
    max < 1 && throw("pdim: max = $max too small.")
    try     F(1.);  return 1    catch; end
    for i in 1:(max+1)
        try
            F(ones(i))
        catch y
            (isa(y, BoundsError) || isa(y, MethodError) || isa(y, DimensionMismatch) || isa(y, ArgumentError)) && continue
            println("pdim: Encountered error in specification of model function.");     rethrow()
        end
        i == (max + 1) ? throw(ArgumentError("pdim: Parameter space appears to have >$max dims. Aborting. Maybe wrong type of x was inserted?")) : return i
    end
end


import DataFrames.DataFrame
DataFrame(DM::DataModel) = DataFrame(Data(DM))
DataFrame(DS::DataSet) = SaveDataSet(DS)

import Base.join
function join(DS1::DataSet, DS2::DataSet)
    !(xdim(DS1) == xdim(DS2) && ydim(DS1) == ydim(DS2)) && throw("DataSets incompatible.")
    NewΣ = if typeof(sigma(DS1)) <: AbstractVector && typeof(sigma(DS2)) <: AbstractVector
        vcat(sigma(DS1), sigma(DS2))
    else
        BlockMatrix(sigma(DS1), sigma(DS2))
    end
    DataSet(vcat(xdata(DS1), xdata(DS2)), vcat(ydata(DS1), ydata(DS2)), NewΣ, (Npoints(DS1)+Npoints(DS2), xdim(DS1), ydim(DS1)))
end
join(DM1::DataModel, DM2::DataModel) = DataModel(join(Data(DM1),Data(DM2)), Predictor(DM1), dPredictor(DM1))
join(DS1::T, DS2::T, args...) where T <: Union{DataSet,DataModel} = join(join(DS1,DS2), args...)
join(DSVec::Vector{T}) where T <: Union{DataSet,DataModel} = join(DSVec...)

SortDataSet(DS::DataSet) = DS |> DataFrame |> sort |> DataSet
SortDataModel(DM::DataModel) = DataModel(SortDataSet(Data(DM)), Predictor(DM), dPredictor(DM))
function SubDataSet(DS::DataSet, range::Union{AbstractRange,AbstractVector})
    Npoints(DS) < length(range) && throw("Length of given range unsuitable for DataSet.")
    X = WoundX(DS)[range] |> Unwind
    Y = Windup(ydata(DS),ydim(DS))[range] |> Unwind
    Σ = sigma(DS)
    if typeof(Σ) <: AbstractVector
        Σ = Windup(Σ,ydim(DS))[range] |> Unwind
    elseif ydim(DS) == 1
        Σ = Σ[range,range]
    else
        throw("Under construction.")
    end
    DataSet(X,Y,Σ,(Int(length(X)/xdim(DS)),xdim(DS),ydim(DS)))
end
SubDataModel(DM::DataModel, range::Union{AbstractRange,AbstractVector}) = DataModel(SubDataSet(Data(DM),range), Predictor(DM), dPredictor(DM))

Sparsify(DS::DataSet) = SubDataSet(DS, rand(Bool,Npoints(DS)))
Sparsify(DM::DataModel) = SubDataSet(DS, rand(Bool,Npoints(DS)))








"""
Specifies a 2D plane in the so-called parameter form using 3 vectors.
"""
struct Plane
    stütz::AbstractVector
    Vx::AbstractVector
    Vy::AbstractVector
    function Plane(stütz::AbstractVector{<:Real}, Vx::AbstractVector{<:Real}, Vy::AbstractVector{<:Real}; Make2ndOrthogonal::Bool=true)
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

function PlanarDataModel(DM::DataModel, PL::Plane)
    model = Predictor(DM);      dmodel = dPredictor(DM)
    newmod = (x,θ::AbstractVector{<:Number}; kwargs...) -> model(x, PlaneCoordinates(PL,θ); kwargs...)
    dnewmod = (x,θ::AbstractVector{<:Number}; kwargs...) -> dmodel(x, PlaneCoordinates(PL,θ); kwargs...) * [PL.Vx PL.Vy]
    mle = MLEinPlane(DM, PL)
    DataModel(Data(DM), newmod, dnewmod, mle, loglikelihood(DM,PlaneCoordinates(PL, mle)), true)
end

# Performance gains of using static vectors is lost if their length exceeds 32
BasisVectorSV(Slot::Int, dims::Int) = dims < 20 ? BasisVectorSVdo(Slot,dims) : BasisVector(Slot,dims)
BasisVectorSVdo(Slot::Int, dims::Int) = Slot > dims ? throw("Dimensional Mismatch.") : SVector{dims}(Float64(i == Slot) for i in 1:dims)
function BasisVector(Slot::Int, dims::Int)
    Res = zeros(dims);    Res[Slot] = 1.;    Res
end


"""
    PlaneCoordinates(PL::Plane, v::AbstractVector{<:Real})
Returns an n-dimensional vector from a tuple of two real numbers which correspond to the coordinates in the 2D `Plane`.
"""
PlaneCoordinates(PL::Plane, v::AbstractVector) = PL.stütz + [PL.Vx PL.Vy] * v

Shift(PlaneBegin::Plane, PlaneEnd::Plane) = TranslatePlane(PlaneEnd, PlaneEnd.stütz - PlaneBegin.stütz)

IsOnPlane(PL::Plane, x::AbstractVector, ProjectionOp::AbstractMatrix=ProjectionOperator(PL))::Bool = DistanceToPlane(PL, x, ProjectionOp) < 4e-15
TranslatePlane(PL::Plane, v::AbstractVector) = Plane(PL.stütz + v, PL.Vx, PL.Vy)
RotatePlane(PL::Plane, rads::Real=pi/2) = Plane(PL.stütz,cos(rads)*PL.Vx + sin(rads)*PL.Vy, cos(rads)*PL.Vy - sin(rads)*PL.Vx)
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
HyperCube([[1,3],[pi,2π],[-500,100]])
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
struct HyperCube{Q<:Real} <: Cuboid
    L::AbstractVector{Q}
    U::AbstractVector{Q}
    function HyperCube(lowers::AbstractVector{<:Real},uppers::AbstractVector{<:Real}; Padding::Real=0.)
        @assert length(lowers) == length(uppers)
        if Padding != 0.
            diff = (uppers - lowers) .* Padding
            lowers -= diff;     uppers += diff
        end
        !all(lowers .≤ uppers) && throw("First argument of HyperCube must be larger than second.")
        if length(lowers) < 20
            return new{suff(lowers)}(SVector{length(lowers)}(float.(lowers)), SVector{length(uppers)}(float.(uppers)))
        else
            return new{suff(lowers)}(float.(lowers),float.(uppers))
        end
    end
    function HyperCube(H::AbstractVector{<:AbstractVector{<:Real}}; Padding::Real=0.)
        len = length(H[1]);        !all(x->(length(x) == len),H) && throw("Inconsistent lengths.")
        M = Unpack(H);        HyperCube(M[:,1],M[:,2]; Padding=Padding)
    end
    function HyperCube(vals::AbstractVector{<:Real}; Padding::Real=0.)
        length(vals) != 2 && throw("Input has too many components.")
        HyperCube([vals[1]],[vals[2]]; Padding=Padding)
    end
    HyperCube(vals::Tuple{<:Real,<:Real}; Padding::Real=0.) = HyperCube([vals[1]],[vals[2]]; Padding=Padding)
end

length(Cube::HyperCube) = length(Cube.L)

"""
    Inside(Cube::HyperCube, p::AbstractVector{<:Real}) -> Bool
Checks whether a point `p` lies inside `Cube`.
"""
Inside(Cube::HyperCube, p::AbstractVector{<:Real}) = all(Cube.L .≤ p) && all(p .≤ Cube.U)


import Base.in
"""
    in(Cube::HyperCube, p::AbstractVector{<:Real}) -> Bool
Checks whether a point `p` lies inside `Cube`.
"""
in(p::AbstractVector{<:Real}, Cube::HyperCube) = Inside(Cube, p)

"""
    ConstructCube(M::Matrix{<:Real}; Padding::Real=1/50) -> HyperCube
Returns a `HyperCube` which encloses the extrema of the columns of the input matrix.
"""
ConstructCube(M::AbstractMatrix{<:Real}; Padding::Real=0.) = HyperCube([minimum(M[:,i]) for i in 1:size(M,2)], [maximum(M[:,i]) for i in 1:size(M,2)]; Padding=Padding)
ConstructCube(V::AbstractVector{<:Real}; Padding::Real=0.) = HyperCube(extrema(V); Padding=Padding)
ConstructCube(PL::Plane, sol::ODESolution; Padding::Real=0.) = ConstructCube(Deplanarize(PL,sol; N=300); Padding=Padding)

function ConstructCube(sol::ODESolution, Npoints::Int=200; Padding::Real=0.)
    ConstructCube(Unpack(map(sol,range(sol.t[1],sol.t[end],length=Npoints))); Padding=Padding)
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
    TranslateCube(Cube::HyperCube,x::Vector{<:Real}) -> HyperCube
Returns a `HyperCube` object which has been translated by `x`.
"""
TranslateCube(Cube::HyperCube, x::AbstractVector{<:Real}) = HyperCube(Cube.L + x, Cube.U + x)


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
