

# suff(x::Number) = typeof(float(x))
# suff(X::Union{AbstractArray,Tuple}) = length(X) != 0 ? suff(X[1]) : error("Empty Array in suff.")

"""
    suff(x) -> Type
If `x` stores BigFloats, `suff` returns BigFloat, else `suff` returns `Float64`.
"""
suff(x::BigFloat) = BigFloat
suff(x::Real) = Float64
suff(x::Complex) = real(x)
suff(x::Union{AbstractArray,Tuple}) = suff(x[1])

function HealthyData(x::AbstractVector,y::AbstractVector)
    length(x) != length(y) && throw(ArgumentError("Dimension mismatch. length(x) = $(length(x)), length(y) = $(length(y))."))
    # Check that dimensions of x-values and y-values are consistent
    xdim = length(x[1]);    ydim = length(y[1])
    sum(length(x[i]) != xdim   for i in 1:length(x)) > 0 && throw("Inconsistent length of x-values.")
    sum(length(y[i]) != ydim   for i in 1:length(y)) > 0 && throw("Inconsistent length of y-values.")
    return Tuple([length(x),xdim,ydim])
end

HealthyCovariance(sigma::AbstractVector{<:Real}) = !all(x->(0. < x),sigma) && throw("Some uncertainties not positive.")
HealthyCovariance(sigma::AbstractMatrix{<:Real}) = !isposdef(sigma) && throw("Covariance matrix not positive-definite.")


abstract type AbstractDataSet end
abstract type AbstractDataModel end

"""
The `DataSet` type is a versatile container for storing data. Typically, it is constructed by passing it three vectors `x`, `y`, `sigma` where the components of `sigma` quantify the standard deviation associated with each y-value.
Alternatively, a full covariance matrix can be supplied for the `ydata` instead of a vector of standard deviations. The contents of a `DataSet` `DS` can later be accessed via `xdata(DS)`, `ydata(DS)`, `sigma(DS)`.

Examples:

In the simplest case, where all data points are mutually independent and have a single ``x``-component and a single ``y``-component each, a `DataSet` consisting of four points can be constructed via
```julia
DataSet([1,2,3,4],[4,5,6.5,7.8],[0.5,0.45,0.6,0.8])
```
or alternatively by
```julia
DataSet([1,2,3,4],[4,5,6.5,7.8],Diagonal([0.5,0.45,0.6,0.8].^2))
```
where the diagonal covariance matrix in the second line is equivalent to the vector of uncertainties supplied in the first line.

More generally, if a dataset consists of ``N`` points where each ``x``-value has ``n`` many components and each ``y``-value has ``m`` many components, this can be specified to the `DataSet` constructor via a tuple ``(N,n,m)`` in addition to the vectors `x`, `y` and the covariance matrix.
For example:
```julia
X = [0.9, 1.0, 1.1, 1.9, 2.0, 2.1, 2.9, 3.0, 3.1, 3.9, 4.0, 4.1]
Y = [1.0, 5.0, 4.0, 8.0, 9.0, 13.0, 16.0, 20.0]
Cov = Diagonal([2.0, 4.0, 2.0, 4.0, 2.0, 4.0, 2.0, 4.0])
dims = Tuple([4,3,2])
DS = DataSet(X,Y,Cov,dims)
```
"""
struct DataSet <: AbstractDataSet
    x::AbstractVector
    y::AbstractVector
    # sigma::AbstractArray
    InvCov::AbstractMatrix
    dims::Tuple{Int,Int,Int}
    logdetInvCov::Real
    WoundX::Union{AbstractVector,Bool}
    function DataSet(x::AbstractVector,y::AbstractVector)
        println("No uncertainties in the y-values were specified for this DataSet, assuming σ=1 for all y's.")
        DataSet(x,y,ones(length(y)*length(y[1])))
    end
    function DataSet(DF::Union{DataFrame,AbstractMatrix})
        size(DF,2) > 3 && throw("Unclear dimensions of input $DF.")
        DataSet(ToCols(convert(Matrix,DF))...)
    end
    DataSet(x::AbstractVector,y::AbstractVector,sigma::AbstractArray) = DataSet(x,y,sigma,HealthyData(x,y))
    function DataSet(x::AbstractVector,y::AbstractVector,sigma::AbstractVector,dims::Tuple{Int,Int,Int})
        Sigma = Unwind(sigma)
        DataSet(Unwind(x),Unwind(y),Sigma,Diagonal([Sigma[i]^(-2) for i in 1:length(Sigma)]),dims)
    end
    DataSet(x::AbstractVector,y::AbstractVector,sigma::AbstractMatrix,dims::Tuple{Int,Int,Int}) = DataSet(Unwind(x),Unwind(y),sigma,inv(sigma),dims)
    function DataSet(x::AbstractVector{<:Real},y::AbstractVector{<:Real},sigma::AbstractArray{<:Real},
                                            InvCov::AbstractMatrix{<:Real},dims::Tuple{Int,Int,Int})
        !all(x->(x > 0), dims) && throw("Not all dims > 0: $dims.")
        !(N(dims) == Int(length(x)/xdim(dims)) == Int(length(y)/ydim(dims)) == Int(size(InvCov,1)/ydim(dims))) && throw("Inconsistent input dimensions.")
        InvCov = isdiag(InvCov) ? Diagonal(InvCov) : InvCov
        if !isposdef(InvCov)
            println("Inverse covariance matrix not perfectly positive-definite. Using only upper half and symmetrizing.")
            !isposdef(Symmetric(InvCov)) && throw("Inverse covariance matrix still not positive-definite after symmetrization.")
            InvCov = convert(Matrix,Symmetric(InvCov))
        end
        if xdim(dims) == 1
            return new(x,y,InvCov,dims,logdet(InvCov),false)
        else
            return new(x,y,InvCov,dims,logdet(InvCov),[SVector{xdim(dims)}(Z) for Z in Windup(x,xdim(dims))])
        end
    end
end


"""
    DetermineDmodel(DS::AbstractDataSet,model::Function)::Function
Returns appropriate function which constitutes the automatic derivative of the `model(x,θ)` with respect to the parameters `θ` depending on the format of the x-values and y-values of the DataSet.
"""
function DetermineDmodel(DS::AbstractDataSet,model::Function)::Function
    Autodmodel(x::Number,θ::AbstractVector{<:Number}) = transpose(ForwardDiff.gradient(z->model(x,z),θ))
    NAutodmodel(x::AbstractVector{<:Number},θ::AbstractVector{<:Number}) = transpose(ForwardDiff.gradient(z->model(x,z),θ))
    AutodmodelN(x::Number,θ::AbstractVector{<:Number}) = ForwardDiff.jacobian(p->model(x,p),θ)
    NAutodmodelN(x::AbstractVector{<:Number},θ::AbstractVector{<:Number}) = ForwardDiff.jacobian(p->model(x,p),θ)
    if ydim(DS) == 1
        if xdim(DS) == 1
            return Autodmodel
        else
            return NAutodmodel
        end
    else
        if xdim(DS) == 1
            return AutodmodelN
        else
            return NAutodmodelN
        end
    end
end


function CheckModelHealth(DS::AbstractDataSet,model::Function)
    P = ones(pdim(DS,model));   X = xdim(DS) < 2 ? xdata(DS)[1] : xdata(DS)[1:xdim(DS)]
    try  model(X,P)   catch Err
        throw("Got xdim=$(xdim(DS)) but model appears to not accept x-values of this size.")
    end
    !(size(model(X,P),1) == ydim(DS)) && println("Got ydim=$(ydim(DS)) but output of model does not have this size.")
    !(typeof(model(X,P)) <: SVector) && ydim(DS) > 1 && @warn "To increase overall performance, it is advisable to define the model function such that it outputs static vectors, i.e. SVectors."
    return
end



"""
In addition to storing a `DataSet`, a `DataModel` also contains a function `model(x,θ)` and its derivative `dmodel(x,θ)` where `x` denotes the x-value of the data and `θ` is a vector of parameters on which the model depends.
Crucially, `dmodel` contains the derivatives of the model with respect to the parameters `θ`, not the x-values.
For example
```julia
DS = DataSet([1,2,3,4],[4,5,6.5,7.8],[0.5,0.45,0.6,0.8])
model(x::Real,θ::AbstractVector{<:Real}) = θ[1] * x + θ[2]
DM = DataModel(DS,model)
```
In cases where the output of the model has more than one component (i.e. `ydim > 1`), it is advisable to define the model function in such a way that it outputs static vectors using **StaticArrays.jl** for increased performance.
For `ydim = 1`, **InformationGeometry.jl** expects the model to output a number instead of a vector with one component. In contrast, the parameter configuration `θ` must always be supplied as a vector.

If a `DataModel` is constructed as shown above, the gradient of the model with respect to the parameters `θ` (i.e. its "Jacobian") will be calculated using automatic differentiation. Alternatively, an explicit analytic expression for the Jacobian can be specified by hand:
```julia
using StaticArrays
function dmodel(x::Real,θ::AbstractVector{<:Real})
   @SMatrix [x  1.]     # ∂(model)/∂θ₁ and ∂(model)/∂θ₂
end
DM = DataModel(DS,model,dmodel)
```
The output of the Jacobian must be a matrix whose columns correspond to the partial derivatives with respect to different components of `θ` and whose rows correspond to evaluations at different components of `x`.
Again, although it is not strictly required, outputting the Jacobian in form of a static matrix is typically beneficial for the overall performance.

The `DataSet` contained in a `DataModel` named `DM` can be accessed via `DM.Data`, whereas the model and its Jacobian can be used via `DM.model` and `DM.dmodel` respectively.
"""
struct DataModel <: AbstractDataModel
    Data::AbstractDataSet
    model::Function
    dmodel::Function
    MLE::AbstractVector
    LogLikeMLE::Real
    # Provide dModel using ForwardDiff if not given
    DataModel(DF::DataFrame, args...) = DataModel(DataSet(DF),args...)
    DataModel(DS::AbstractDataSet,model::Function) = DataModel(DS,model,DetermineDmodel(DS,model))
    DataModel(DS::AbstractDataSet,model::Function,mle::AbstractVector) = DataModel(DS,model,DetermineDmodel(DS,model),mle)
    function DataModel(DS::AbstractDataSet,model::Function,dmodel::Function)
        DataModel(DS,model,dmodel,FindMLE(DS,model))
    end
    function DataModel(DS::AbstractDataSet,model::Function,dmodel::Function,mle::AbstractVector{<:Number},sneak::Bool=false)
        sneak && return DataModel(DS,model,dmodel,mle,-Inf,true)
        MLE = FindMLE(DS,model,mle);        LogLikeMLE = loglikelihood(DS,model,MLE)
        DataModel(DS,model,dmodel,MLE,LogLikeMLE)
    end
    # Check whether the determined MLE corresponds to a maximum of the likelihood unless sneak==true.
    function DataModel(DS::AbstractDataSet,model::Function,dmodel::Function,MLE::AbstractVector{<:Number},LogLikeMLE::Real,sneak::Bool=false)
        sneak && return new(DS,model,dmodel,MLE,LogLikeMLE)
        CheckModelHealth(DS,model)
        norm(AutoScore(DS,model,MLE)) > 1e-5 && throw("Norm of gradient of log-likelihood at supposed MLE=$MLE too large: $(norm(AutoScore(DS,M,MLE))).")
        g = AutoMetric(DS,model,MLE)
        det(g) == 0. && throw("Model appears to contain superfluous parameters since it is not structurally identifiable at supposed MLE=$MLE.")
        !isposdef(Symmetric(g)) && throw("Hessian of likelihood at supposed MLE=$MLE not negative-definite: Consider passing an appropriate initial parameter configuration 'init' for the estimation of the MLE to DataModel e.g. via DataModel(DS,model,init).")
        new(DS,model,dmodel,SVector{length(MLE)}(MLE),LogLikeMLE)
    end
end

xdata(DM::AbstractDataModel) = xdata(DM.Data)
ydata(DM::AbstractDataModel) = ydata(DM.Data)
sigma(DM::AbstractDataModel) = sigma(DM.Data)
InvCov(DM::AbstractDataModel) = InvCov(DM.Data)
N(DM::AbstractDataModel) = N(DM.Data)
xdim(DM::AbstractDataModel) = xdim(DM.Data)
ydim(DM::AbstractDataModel) = ydim(DM.Data)


N(dims::Tuple{Int,Int,Int}) = dims[1]
xdim(dims::Tuple{Int,Int,Int}) = dims[2]
ydim(dims::Tuple{Int,Int,Int}) = dims[3]

xdata(DS::DataSet) = DS.x
ydata(DS::DataSet) = DS.y
function sigma(DS::DataSet)
    sig = !issparse(InvCov(DS)) ? inv(InvCov(DS)) : inv(convert(Matrix,InvCov(DS)))
    sig = isdiag(sig) ? sqrt.(Diagonal(sig).diag) : sig
    return sig
end
# sigma(DS::DataSet) = DS.sigma
InvCov(DS::DataSet) = DS.InvCov
N(DS::DataSet) = N(DS.dims)
xdim(DS::DataSet) = xdim(DS.dims)
ydim(DS::DataSet) = ydim(DS.dims)
WoundX(DS::DataSet) = xdim(DS) < 2 ? xdata(DS) : DS.WoundX
WoundX(DS::AbstractDataSet) = Windup(xdata(DS),xdim(DS))


logdetInvCov(DM::AbstractDataModel) = logdetInvCov(DM.Data)
logdetInvCov(DS::AbstractDataSet) = logdet(InvCov(DS))
logdetInvCov(DS::DataSet) = DS.logdetInvCov

import Base.length
length(DS::AbstractDataSet) = N(DS);    length(DM::AbstractDataModel) = N(DM.Data)


"""
    MLE(DM::DataModel) -> Vector
Returns the parameter configuration ``\\theta_\\text{MLE} \\in \\mathcal{M}`` which is estimated to have the highest likelihood of producing the observed data (under the assumption that the specified model captures the true relationship present in the data).
For performance reasons, the maximum likelihood estimate is stored as a part of the `DataModel` type.
"""
MLE(DM::DataModel) = DM.MLE
"""
    LogLikeMLE(DM::DataModel) -> Real
Returns the value of the log-likelihood ``\\ell`` when evaluated at the maximum likelihood estimate, i.e. ``\\ell(\\mathrm{data} \\, | \\, \\theta_\\text{MLE})``.
For performance reasons, this value is stored as a part of the `DataModel` type.
"""
LogLikeMLE(DM::DataModel) = DM.LogLikeMLE

DataDist(Y::AbstractVector,Sig::AbstractVector,dist=Normal) = product_distribution([dist(Y[i],Sig[i]) for i in eachindex(Y)])
DataDist(Y::AbstractVector,Sig::AbstractMatrix,dist=MvNormal) = dist(Y,Sig)
yDataDist(DS::DataSet) = DataDist(ydata(DS),sigma(DS))
xDataDist(DS::DataSet) = DataDist(xdata(DS),sigma(DS))
yDataDist(DM::DataModel) = yDataDist(DM.Data);    xDataDist(DM::DataModel) = xDataDist(DM.Data)


pdim(DM::DataModel) = length(MLE(DM))
pdim(DM::AbstractDataModel) = pdim(DM.Data,DM.model)
pdim(DS::AbstractDataSet,model::Function) = xdim(DS) < 2 ? pdim(model,xdata(DS)[1]) : pdim(model,xdata(DS)[1:xdim(DS)])

"""
    pdim(model::Function,x::Union{T,Vector{T}}=1.; max::Int=50)::Int where T<:Real -> Int
Infers the number of parameters ``\\theta`` of the model function `model(x,θ)` by successively testing it on vectors of increasing length.
"""
function pdim(model::Function,x::Union{T,Vector{T}}=1.; max::Int=100)::Int where T<:Real
    max < 1 && throw("pdim: max = $max too small.")
    for i in 1:(max+1)
        try
            model(x,ones(i))
        catch y
            if isa(y, BoundsError) || isa(y,DimensionMismatch)
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


function Sparsify(DS::DataSet,B::Vector=rand(Bool,length(xdata(DS))))
    length(B) != length(xdata(DS)) && throw(ArgumentError("Sparsify: Vector not same number of components as datapoints."))
    !(length(xdata(DS)[1]) == length(ydata(DS)[1]) == length(sigma(DS)[1])) && throw("Not programmed yet.")
    return DataSet(xdata(DS)[B],ydata(DS)[B],sigma(DS)[B])
end
function Sparsify(DM::AbstractDataModel,B::Vector=rand(Bool,length(xdata(DM))))
    return DataModel(Sparsify(DM.Data,B),DM.model,DM.dmodel)
end


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


function BlockDiagonal(M::AbstractMatrix,N::Int)
    Res = zeros(size(M,1)*N,size(M,2)*N)
    for i in 1:N
        Res[((i-1)*size(M,1) + 1):(i*size(M,1)),((i-1)*size(M,1) + 1):(i*size(M,1))] = M
    end; Res
end


"""
Specifies a 2D plane in the so-called parameter form using 3 vectors.
"""
struct Plane
    stütz::AbstractVector
    Vx::AbstractVector
    Vy::AbstractVector
    # Projector::AbstractMatrix
    function Plane(stütz::AbstractVector{<:Real},Vx::AbstractVector{<:Real},Vy::AbstractVector{<:Real})
        if length(stütz) == 2 stütz = [stütz[1],stütz[2],0] end
        if !(length(stütz) == length(Vx) == length(Vy))
            throw(ArgumentError("Dimension mismatch. length(stütz) = $(length(stütz)), length(Vx) = $(length(Vx)), length(Vy) = $(length(Vy))"))
        elseif dot(Vx,Vy) != 0
            println("Plane: Making Vy orthogonal to Vx.")
            return Plane(stütz,Vx,Make2ndOrthogonal(Vx,Vy))
        else
            stütz = SVector{length(Vx)}(float.(stütz));     Vx = SVector{length(Vx)}(float.(Vx))
            Vy = SVector{length(Vx)}(float.(Vy))
            return new(stütz,Vx,Vy)
            # return new(stütz,Vx,Vy,ProjectionOperator([Vx Vy]))
        end
    end
    # function Plane(stütz::AbstractVector{<:Real},Vx::AbstractVector{<:Real},Vy::AbstractVector{<:Real}, Projector::AbstractMatrix{<:Real})
    #     !(length(stütz) == length(Vx) == length(Vy) == size(Projector,1) == size(Projector,2)) && throw("Plane: Dimensional Mismatch.")
    #     new(stütz,Vx,Vy,Projector)
    # end
end

length(PL::Plane) = length(PL.stütz)

function PlanarDataModel(DM::DataModel,PL::Plane)
    newmod = (x,p::AbstractVector{<:Number}) -> DM.model(x,PlaneCoordinates(PL,p))
    dnewmod = (x,p::AbstractVector{<:Number}) -> DM.dmodel(x,PlaneCoordinates(PL,p)) * [PL.Vx PL.Vy]
    DataModel(DM.Data,newmod,dnewmod,0.001ones(2))
end

function BasisVector(Slot::Int,dims::Int)
    Res = zeros(dims);    Res[Slot] = 1;    Res
end

"""
    PlaneCoordinates(PL::Plane, v::AbstractVector{<:Real})
Returns an n-dimensional vector from a tuple of two real numbers which correspond to the coordinates in the 2D `Plane`.
"""
PlaneCoordinates(PL::Plane, v::AbstractVector) = PL.stütz + [PL.Vx PL.Vy]*v
# length(v) != 2 && throw(ArgumentError("PlaneCoordinates: length(v) != 2"))


IsOnPlane(PL::Plane,x::AbstractVector)::Bool = (DistanceToPlane(PL,x) == 0)
#TranslatePlane(PL::Plane, v::AbstractVector) = Plane(PL.stütz + v, PL.Vx, PL.Vy, PL.Projector)
TranslatePlane(PL::Plane, v::AbstractVector) = Plane(PL.stütz + v, PL.Vx, PL.Vy)
RotatePlane(PL::Plane, rads::Real=pi/2) = Plane(PL.stütz,cos(rads)*PL.Vx + sin(rads)*PL.Vy, cos(rads)*PL.Vy - sin(rads)*PL.Vx)
function RotationMatrix(PL::Plane,rads::Real)
    V = PL.Vx*transpose(PL.Vx) + PL.Vy*transpose(PL.Vy)
    W = PL.Vx*transpose(PL.Vy) - PL.Vy*transpose(PL.Vx)
    Diagonal(ones(length(PL.stütz))) + (cos(rads)-1.)*V -sin(rads)*W
end
RotateVector(PL::Plane,v::AbstractVector,rads::Real) = RotationMatrix(PL,rads)*v

function RotatedVector(α::Real,n1::Int,n2::Int,tot::Int)
    !(n1 <= tot && n2 <= tot && n1 != n2 && all(x->(x>0),[n1,n2,tot])) && throw("Error")
    res = zeros(tot);   res[n1] = cos(α);   res[n2] = sin(α);   res
end


function DecomposeWRTPlane(PL::Plane,x::AbstractVector)
    !IsOnPlane(PL,x) && throw(ArgumentError("Decompose Error: Vector not on Plane."))
    V = x - PL.stütz
    [ProjectOnto(V,PL.Vx), ProjectOnto(V,PL.Vy)]
end
# DistanceToPlane(PL::Plane,x::AbstractVector) = (diagm(ones(Float64,length(x))) - PL.Projector) * (x - PL.stütz) |> norm
# ProjectOntoPlane(PL::Plane,x::AbstractVector) = PL.Projector*(x - PL.stütz) + PL.stütz
DistanceToPlane(PL::Plane,x::AbstractVector) = (diagm(ones(Float64,length(x))) - ProjectionOperator(PL)) * (x - PL.stütz) |> norm
ProjectOntoPlane(PL::Plane,x::AbstractVector) = ProjectionOperator(PL) * (x - PL.stütz) + PL.stütz

function ProjectionOperator(A::AbstractMatrix)
    size(A,2) != 2 && println("ProjectionOperator: Matrix size $(size(A)) not as expected.")
    A * inv(transpose(A) * A) * transpose(A)
end
ProjectionOperator(PL::Plane) = ProjectionOperator([PL.Vx PL.Vy])
# ProjectionOperator(PL::Plane) = PL.Projector

IsNormalToPlane(PL::Plane,v::AbstractVector)::Bool = (dot(PL.Vx,v) == dot(PL.Vy,v) == 0.)

function Make2ndOrthogonal(X::AbstractVector,Y::AbstractVector)
    Basis = GramSchmidt(float.([X,Y]))
    # Maybe add check for orientation?
    return Basis[2]
end

function MinimizeOnPlane(PL::Plane,F::Function,initial::AbstractVector=[1,-1.]; tol::Real=1e-5)
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




abstract type Cuboid end

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
struct HyperCube{Q<:Real} <: Cuboid
    vals::AbstractVector{<:AbstractVector{Q}}
    dim::Int
    function HyperCube(vals::AbstractVector)
        vals = float.(vals)
        types = typeof(vals[1][1])
        for i in 1:length(vals)
            length(vals[i]) != 2 && error("Unsuitable Hypercube.")
            typeof(vals[i][1]) != types && error("Type Mismatch in Hypercube.")
            vals[i][1] > vals[i][2] && error("HyperCube: Orientation wrong, Interval $i: [$(vals[i][1]),$(vals[i][2])] not allowed.")
        end
        new{types}(vals,length(vals))
    end
    HyperCube(vals::AbstractVector{<:Real}) = HyperCube([vals])
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
struct LowerUpper{Q<:Real} <: Cuboid
    L::AbstractVector{Q}
    U::AbstractVector{Q}
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
    LowerUpper(H::AbstractVector) = LowerUpper(HyperCube(H))
end

# SensibleOutput(LU::LowerUpper) = LU.L,LU.U
function HyperCube(LU::LowerUpper)
    R = Vector{typeof(LU.U)}(undef,length(LU.U))
    for i in 1:length(LU.U)
        R[i] = [LU.L[i],LU.U[i]]
    end
    HyperCube(R)
end

function SensibleOutput(Res::AbstractVector)
    if isa(Res[1],Real)
        return Res[1], Res[2]
    elseif isa(Res[1],AbstractVector) && typeof(Res[1][1]) <: Real
        u = Vector{suff(Res)}(undef,length(Res)); v = similar(u)
        for i in 1:length(Res)
            if length(Res[i]) != 2
                error("SensibleOutput only accepts inner vectors of length 2.")
            end
            u[i] = Res[i][1]
            v[i] = Res[i][2]
        end
        return u, v
    else
        throw(ArgumentError("Expected Vector{Real} or Vector{Vector{Real}}, but got $(typeof(Res))"))
    end
end

Unpack(H::HyperCube) = Unpack(H.vals)
"""
    Unpack(Z::Vector{S}) where S <: Union{Vector,Tuple} -> Matrix
Converts vector of vectors to a matrix whose n-th column corresponds to the n-th component of the inner vectors.
"""
function Unpack(Z::AbstractVector{S}) where S <: Union{AbstractVector,Tuple}
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
Unwind(X::AbstractVector{<:AbstractVector{<:Number}}) = reduce(vcat,X)
Unwind(X::AbstractVector{<:Number}) = X

Windup(v::AbstractVector{<:Number},n::Int) = n < 2 ? v : [v[(1+(i-1)*n):(i*n)] for i in 1:Int(length(v)/n)]



function CubeVol(Space::AbstractVector)
    lowers,uppers = Unpack(Space)
    prod(uppers - lowers)
end
CubeWidths(S::LowerUpper) = S.U - S.L

Center(LU::LowerUpper) = 0.5 * (LU.U + LU.L)
Center(C::HyperCube) = Center(LowerUpper(C))


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
function TranslateCube(H::HyperCube,x::AbstractVector)
    H.dim != length(x) && throw("Translation vector must be of same dimension as hypercube.")
    [H.vals[i] .+ x[i] for i in 1:length(x)] |> HyperCube
end
TranslateCube(LU::LowerUpper,x::AbstractVector) = TranslateCube(HyperCube(LU),x) |> LowerUpper

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
CoverCubes(V::Vector{<:HyperCube}) = CoverCubes(V...)


normalize(x::AbstractVector,scaling::Float64=1.0) = (scaling/norm(x)) * x
function normalizeVF(u::AbstractVector{<:Real},v::AbstractVector{<:Real},scaling::Float64=1.0)
    newu = u;    newv = v
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
