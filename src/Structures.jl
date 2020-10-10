

suff(x::Number) = typeof(float(x))
suff(X::Union{AbstractArray,Tuple}) = length(X) != 0 ? suff(X[1]) : error("Empty Array in suff.")


abstract type AbstractDataSet end

"""
The `DataSet` type is a container for data points. It holds 3 vectors `x`, `y`, `sigma` where the components of `sigma` quantify the standard deviation associated with each measurement.
For example,
```julia
DS = DataSet([1,2,3.],[4,5,6.5],[0.5,0.45,0.6])
```
Its fields can be obtained via `xdata(DS)`, `ydata(DS)`, `sigma(DS)`.
"""
struct DataSet <: AbstractDataSet
    x::AbstractVector
    y::AbstractVector
    sigma::AbstractArray
    InvCov::AbstractMatrix
    # dims::Tuple{Int,Int,Int}
    function HealthyData(x::AbstractVector,y::AbstractVector)::Bool
        length(x) != length(y) && throw(ArgumentError("Dimension mismatch. length(x) = $(length(x)), length(y) = $(length(y))."))
        # Check that dimensions of x-values and y-values are consistent
        xdim = length(x[1]);    ydim = length(y[1])
        sum(length(x[i]) != xdim   for i in 1:length(x)) > 0 && throw("Inconsistent length of x-values.")
        sum(length(y[i]) != ydim   for i in 1:length(y)) > 0 && throw("Inconsistent length of y-values.")
        return true
    end
    function DataSet(x::AbstractVector,y::AbstractVector,sigma)
        HealthyData(x,y)
        if length(sigma) == 1 && length(y) > 1
            return DataSet(x, y, sigma*ones(length(y)))
        elseif size(sigma,1) == size(sigma,2) == length(y) && length(y) > 1
            throw(ArgumentError("DataSet not programmed for covariance matrices yet. Please decorrelate data and input as three vectors."))
        elseif length(sigma) == length(y) && length(sigma[1]) == 1
            return new(x,y,[sigma[i] for i in 1:length(y)],Diagonal([sigma[i]^-2 for i in 1:length(y)]))
        else
            throw(ArgumentError("Unsuitable specification of uncertainty: sigma = $sigma."))
        end
    end
    function DataSet(x::AbstractVector,y::AbstractVector)
        HealthyData(x,y)
        println("No uncertainties in the y-values were specified for this DataSet, assuming σ=1 for all y's.")
        DataSet(x,y,ones(length(y)))
    end
    function DataSet(DF::Union{DataFrame,AbstractMatrix})
        size(DF,2) > 3 && throw("Unclear dimensions of input $DF.")
        return DataSet(ToCols(convert(Matrix,DF))...)
    end
end


"""
    DetermineDmodel(DS::AbstractDataSet,model::Function)::Function
Returns appropriate function which constitutes the automatic derivative of the `model(x,θ)` with respect to the parameters `θ` depending on the format of the x-values and y-values of the DataSet.
"""
function DetermineDmodel(DS::AbstractDataSet,model::Function)::Function
    # xdim > 1, ydim = 1
    NAutodmodel(x::Vector{<:Real},θ::Vector{<:Number}) = reshape(ForwardDiff.gradient(z->model(x,z),θ),1,length(θ))
    function NAutodmodel(x::Vector{Vector{Q}},θ::Vector{<:Number}) where Q <: Real
        Res = Array{suff(θ)}(undef,ydim(DS)*length(x),length(θ))
        for i in 1:length(x)
            Res[i,:] = NAutodmodel(x[i],θ)
        end;    Res
    end
    # xdim = 1, ydim = 1
    Autodmodel(x::Real,θ::Vector{<:Number}) = reshape(ForwardDiff.gradient(z->model(x,z),θ),1,length(θ))
    function Autodmodel(x::Vector{<:Real},θ::Vector{<:Number})
        Res = Array{suff(θ)}(undef,length(x),length(θ))
        for i in 1:length(x)
            Res[i,:] = Autodmodel(x[i],θ)
        end;    Res
    end
    # xdim = 1, ydim > 1
    AutodmodelN(x::Number,θ::Vector{<:Number}) = ForwardDiff.jacobian(p->model(x,p),θ)
    AutodmodelN(x::Vector{<:Number},θ::Vector{<:Number}) = vcat([AutodmodelN(z,θ) for z in xdata(DS)]...)
    if xdim(DS) == 1
        if ydim(DS) == 1
            return Autodmodel
        else
            return AutodmodelN
        end
    else
        if ydim(DS) == 1
            return NAutodmodel
        else
            throw("Automatic differentiation for vector-valued xdata AND vector-valued ydata not pre-programmed yet.")
        end
    end
end


"""
In addition to a `DataSet`, a `DataModel` contains the model as a function `model(x,θ)` and its derivative `dmodel(x,θ)` where `x` denotes the x-value of the data and `θ` is a vector of parameters on which the model depends. Crucially, `dmodel` contains the derivatives of the model with respect to the parameters `θ`, not the x-values.
For example
```julia
DS = DataSet([1,2,3.],[4,5,6.5],[0.5,0.45,0.6])
model(x,θ::Vector) = θ[1] .* x .+ θ[2]
DM = DataModel(DS,model)
```
If provided like this, the gradient of the model with respect to the parameters `θ` (i.e. its "Jacobian") will be calculated using automatic differentiation. Alternatively, an explicit analytic expression for the Jacobian can be specified by hand:
```julia
function dmodel(x,θ::Vector)
   J = Array{Float64}(undef, length(x), length(θ))
   @. J[:,1] = x        # ∂(model)/∂θ₁
   @. J[:,2] = 1.       # ∂(model)/∂θ₂
   return J
end
DM = DataModel(DS,model,dmodel)
```
The output of the Jacobian must be a matrix whose columns correspond to the partial derivatives with respect to different components of `θ` and whose rows correspond to evaluations at different values of `x`.
"""
struct DataModel
    Data::AbstractDataSet
    model::Function
    dmodel::Function
    MLE::AbstractVector
    LogLikeMLE::Real
    # Provide dModel using ForwardDiff if not given
    DataModel(DF::DataFrame, args...) = DataModel(DataSet(DF),args...)
    DataModel(DS::AbstractDataSet,model::Function) = DataModel(DS,model,DetermineDmodel(DS,model))
    DataModel(DS::AbstractDataSet,model::Function,mle::AbstractVector) = DataModel(DS,model,DetermineDmodel(DS,model),mle)
    DataModel(DS::AbstractDataSet,M::Function,dM::Function) = DataModel(DS,M,dM,FindMLE(DS,M))
    function DataModel(DS::AbstractDataSet,M::Function,dM::Function,mle::AbstractVector)
        MLE = FindMLE(DS,M,mle);        LogLikeMLE = loglikelihood(DS,M,MLE)
        DataModel(DS,M,dM,MLE,LogLikeMLE)
    end
    # Check whether the determined MLE corresponds to a maximum of the likelihood unless sneak==true.
    function DataModel(DS::AbstractDataSet,M::Function,dM::Function,MLE::AbstractVector,LogLikeMLE::Real,sneak::Bool=false)
        sneak && new(DS,M,dM,MLE,LogLikeMLE)
        J = dM(xdata(DS),MLE);  g = transpose(J) * InvCov(DS) * J
        det(g) == 0. && throw("Model appears to contain superfluous parameters since it is not structurally identifiable at θ=$MLE.")
        !isposdef(Symmetric(g)) && throw("Hessian of likelihood at MLE not negative-definite: Could not determine MLE, got $MLE. Consider passing an appropriate initial parameter configuration 'init' for the estimation of the MLE to DataModel e.g. via DataModel(DS,model,init).")
        new(DS,M,dM,MLE,LogLikeMLE)
    end
end

xdata(DS::DataSet) = DS.x;                  xdata(DM::DataModel) = xdata(DM.Data)
ydata(DS::DataSet) = DS.y;                  ydata(DM::DataModel) = ydata(DM.Data)
sigma(DS::DataSet) = DS.sigma;              sigma(DM::DataModel) = sigma(DM.Data)
InvCov(DS::DataSet) = DS.InvCov;            InvCov(DM::DataModel) = InvCov(DM.Data)
xdim(DS::AbstractDataSet) = length(xdata(DS)[1]);   xdim(DM::DataModel) = xdim(DM.Data)
ydim(DS::AbstractDataSet) = length(ydata(DS)[1]);   ydim(DM::DataModel) = ydim(DM.Data)

# Eventually incorporate into DataSet type such that InvCov(DS::DataSet) = DS.InvCov
# function InvCov(DS::DataSet)
#     if typeof(sigma(DS)) <: AbstractVector
#         return diagm(sigma(DS).^-2)
#     elseif typeof(sigma(DS)) <: AbstractMatrix
#         return inv(sigma(DS))
#     end
# end

MLE(DM::DataModel) = DM.MLE;                LogLikeMLE(DM::DataModel) = DM.LogLikeMLE


yDataDist(DS::DataSet) = product_distribution([Normal(ydata(DS)[i],sigma(DS)[i]) for i in 1:length(ydata(DS))])
xDataDist(DS::DataSet) = product_distribution([Normal(xdata(DS)[i],sigma(DS)[i]) for i in 1:length(xdata(DS))])
yDataDist(DM::DataModel) = yDataDist(DM.Data);    xDataDist(DM::DataModel) = xDataDist(DM.Data)

# pdim(DM::DataModel; max::Int=50)::Int = pdim(DM.model,xdata(DM)[1]; max=max)
pdim(DM::DataModel; kwargs...) = length(MLE(DM))

"""
    pdim(model::Function,x::Union{T,Vector{T}}=1.; max::Int=50)::Int where T<:Real -> Int
Infers the number of parameters ``\\theta`` of the model function `model(x,θ)` by successively testing it on vectors of increasing length.
"""
function pdim(model::Function,x::Union{T,Vector{T}}=1.; max::Int=50)::Int where T<:Real
    max < 1 && throw("pdim: max = $max too small.")
    for i in 1:(max+1)
        try
            model(x,ones(i))
        catch y
            if isa(y, BoundsError)
                continue
            else
                println("pdim: Encountered error in specification of model function.")
                throw(y)
            end
        end
        i != (max+1) && return i
    end
    throw(ArgumentError("pdim: Parameter space appears to have >$max dims. Aborting. Maybe wrong type of x was inserted?"))
end


function Sparsify(DS::DataSet,B::Vector=rand(Bool,length(DS.x)))
    length(B) != length(DS.x) && throw(ArgumentError("Sparsify: Vector not same number of components as datapoints."))
    !(length(DS.x[1]) == length(DS.y[1]) == length(DS.sigma[1])) && throw("Not programmed yet.")
    return DataSet(DS.x[B],DS.y[B],DS.sigma[B])
end
function Sparsify(DM::DataModel,B::Vector=rand(Bool,length(xdata(DM))))
    return DataModel(Sparsify(DM.Data,B),DM.model,DM.dmodel)
end

import Base.length
length(DS::DataSet) = length(xdata(DS));    length(DM::DataModel) = length(DM.Data)
import DataFrames.DataFrame
DataFrame(DS::DataSet) = DataFrame([xdata(DS) ydata(DS) sigma(DS)]);    DataFrame(DM::DataModel) = DataFrame(DM.Data)

import Base.join
join(DS1::DataSet,DS2::DataSet) = DataSet([xdata(DS1)...,xdata(DS2)...],[ydata(DS1)...,ydata(DS2)...],[sigma(DS1)...,sigma(DS2)...])
join(DM1::DataModel,DM2::DataModel) = DataModel(join(DM1.Data,DM2.Data),DM1.model,DM1.dmodel)
join(DS1::T,DS2::T,args...) where T <: Union{DataSet,DataModel} = join(join(DS1,DS2),args...)
join(DSVec::Vector{T}) where T <: Union{DataSet,DataModel} = join(DSVec...)

SortDataSet(DS::DataSet) = DS |> DataFrame |> sort |> DataSet
SortDataModel(DM::DataModel) = DataModel(SortDataSet(DM.Data),DM.model,DM.dmodel)
SubDataSet(DS::AbstractDataSet,ran) = DataSet(xdata(DS)[ran],ydata(DS)[ran],sigma(DS)[ran])
SubDataModel(DM::DataModel,ran) = DataModel(SubDataSet(DM.Data,ran),DM.model,DM.dmodel)


"""
Specifies a 2D plane in the so-called parameter form using 3 vectors.
"""
struct Plane
    stütz::AbstractVector
    Vx::AbstractVector
    Vy::AbstractVector
    Projector::AbstractMatrix
    function Plane(stütz::Vector{<:Real},Vx::Vector{<:Real},Vy::Vector{<:Real})
        if length(stütz) == 2 stütz = [stütz[1],stütz[2],0] end
        if !(length(stütz) == length(Vx) == length(Vy))
            throw(ArgumentError("Dimension mismatch. length(stütz) = $(length(stütz)), length(Vx) = $(length(Vx)), length(Vy) = $(length(Vy))"))
        elseif dot(Vx,Vy) != 0
            println("Plane: Making Vy orthogonal to Vx.")
            new(float.(stütz),float.(normalize(Vx)),Make2ndOrthogonal(Vx,Vy),ProjectionOperator([Vx Vy]))
        else
            new(float.(stütz),float.(normalize(Vx)),float.(normalize(Vy)),ProjectionOperator([Vx Vy]))
        end
    end
    function Plane(stütz::Vector{<:Real},Vx::Vector{<:Real},Vy::Vector{<:Real}, Projector::Matrix{<:Real})
        !(length(stütz) == length(Vx) == length(Vy) == size(Projector,1) == size(Projector,2)) && throw("Plane: Dimensional Mismatch.")
        new(stütz,Vx,Vy,Projector)
    end
end

function PlanarDataModel(DM::DataModel,PL::Plane)
    newmod = (x,p::Vector{<:Number}) -> DM.model(x,PlaneCoordinates(PL,p))
    dnewmod = (x,p::Vector{<:Number}) -> DM.dmodel(x,PlaneCoordinates(PL,p)) * [PL.Vx PL.Vy]
    DataModel(DM.Data,newmod,dnewmod)
end

function BasisVector(Slot::Int,dims::Int)
    Res = zeros(dims);    Res[Slot] = 1;    Res
end
"""
    PlaneCoordinates(PL::Plane, v::Vector{<:Real})
Returns an n-dimensional vector from a tuple of two real numbers which
"""
function PlaneCoordinates(PL::Plane, v::Vector{<:Real})
    length(v) != 2 && throw(ArgumentError("PlaneCoordinates: length(v) != 2"))
    # PL.stütz .+ v[1]*PL.Vx .+ v[2]*PL.Vy
    # Faster:
    PL.stütz + [PL.Vx PL.Vy]*v
end
import LinearAlgebra.dot
dot(A::AbstractMatrix, x::Vector, y::Vector=x) = transpose(x)*A*y
dot(x) = dot(x,x)
länge(x::Vector{<:Real}) = sqrt(dot(x,x))
EuclideanDistance(x::Vector, y::Vector) = länge(x .- y)
IsOnPlane(PL::Plane,x::Vector)::Bool = (DistanceToPlane(PL,x) == 0)
TranslatePlane(PL::Plane, v::Vector) = Plane(PL.stütz + v, PL.Vx, PL.Vy, PL.Projector)
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
ProjectOntoPlane(PL::Plane,x::Vector) = PL.Projector*(x - PL.stütz) + PL.stütz

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
```julia
HyperCube([[1,3],[pi,2pi],[-500.0,100.0]])
HyperCube([[-1,1]])
HyperCube([-1,1])
HyperCube(LowerUpper([-1,-5],[0,-4]))
HyperCube(collect([-7,7.] for i in 1:3))
```
The `HyperCube` type is closely related to the `LowerUpper` type and they can be easily converted into each other.
Examples of quantities that can be computed from and operations involving a `HyperCube` object `X`:
```julia
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
```julia
LowerUpper([-1,-5,pi],[0,-4,2pi])
LowerUpper(HyperCube([[5,6],[-pi,0.5]]))
LowerUpper(collect(1:5),collect(15:20))
```
Examples for quantities that can be computed from and operations involving a `LowerUpper` object `X`:
```julia
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
        throw(ArgumentError("Expected Vector{Real} or Vector{Vector{Real}}, but got $(typeof(Res))"))
    end
end

Unpack(H::HyperCube) = Unpack(H.vals)
"""
    Unpack(Z::Vector{S}) where S <: Union{Vector,Tuple} -> Matrix
Converts vector of vectors to a matrix whose n-th column corresponds to the n-th component of the inner vectors.
"""
function Unpack(Z::Vector{S}) where S <: Union{Vector,Tuple}
    N = length(Z); M = length(Z[1])
    A = Array{suff(Z)}(undef,N,M)
    for i in 1:N
        for j in 1:M
            A[i,j] = Z[i][j]
        end
    end
    A
end
ToCols(M::Matrix) = Tuple(M[:,i] for i in 1:size(M,2))
Unwind(X::Vector{Vector{Q}}) where Q<:Number = vcat(X...)


function CubeVol(Space::Vector)
    lowers,uppers = SensibleOutput(Space)
    prod(uppers .- lowers)
end
CubeWidths(S::LowerUpper) = S.U .- S.L


"""
    CubeWidths(H::HyperCube) -> Vector
Returns vector of widths of the `HyperCube`.
"""
CubeWidths(H::HyperCube) = CubeWidths(LowerUpper(H))

"""
    CubeVol(X::HyperCube)
Computes volume of a `HyperCube` as the product of its sidelengths.
"""
CubeVol(X::HyperCube) = CubeVol(LowerUpper(X))
CubeVol(S::LowerUpper) = prod(CubeWidths(S))

"""
    TranslateCube(H::HyperCube,x::Vector)
Returns a `HyperCube` object which has been translated by `x`.
"""
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
CoverCubes(A::HyperCube,B::HyperCube,args...) = CoverCubes(CoverCubes(A,B),args...)
CoverCubes(V::Vector{T}) where T<:HyperCube = CoverCubes(V...)


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
