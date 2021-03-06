

function StandardInDomain(n::Int)
    IsInDomain(θ::AbstractVector{<:Number}) = length(θ) == n ? true : throw("Incorrect number of components was passed.")
end

# Callback triggers when Boundaries is `true`.
"""
Container for model functions which carries additional information, e.g. about the parameter domain on which it is valid.
"""
struct ModelMap
    Map::Function
    InDomain::Function
    Domain::Union{Cuboid,Bool}
    xyp::Tuple{Int,Int,Int}
    pnames::Vector{String}
    StaticOutput::Val
    inplace::Val
    CustomEmbedding::Val
    # Given: Bool-valued domain function
    function ModelMap(model::Function, InDomain::Function, xyp::Tuple{Int,Int,Int}; pnames::Union{Vector{String},Bool}=false)
        ModelMap(model, InDomain, false, xyp; pnames=pnames)
    end
    # Given: HyperCube
    function ModelMap(model::Function, Domain::Cuboid, xyp::Union{Tuple{Int,Int,Int},Bool}=false; pnames::Union{Vector{String},Bool}=false)
        # Change this to θ -> true to avoid double checking cuboid. Obviously make sure Boundaries() is constructed using both the function test
        # and the Cuboid test first before changing this.
        InDomain(θ::AbstractVector{<:Number})::Bool = θ ∈ Domain
        xyp isa Bool ? ModelMap(model, InDomain, Domain; pnames=pnames) : ModelMap(model, InDomain, Domain, xyp; pnames=pnames)
    end
    # Given: Function only (potentially) -> Find xyp
    function ModelMap(model::Function, InDomain::Function=θ::AbstractVector{<:Number}->true, Domain::Union{Cuboid,Bool}=false; pnames::Union{Vector{String},Bool}=false)
        xyp = if Domain isa Bool
            xlen, plen = GetArgSize(model);     testout = model((xlen < 2 ? 1. : ones(xlen)), GetStartP(plen))
            (xlen, size(testout,1), plen)
        else
            plen = length(Domain);      startp = GetStartP(plen)
            xlen = GetArgLength(x->model(x,startp));    testout = model((xlen < 2 ? 1. : ones(xlen)), startp)
            (xlen, size(testout,1), plen)
        end
        ModelMap(model, InDomain, Domain, xyp; pnames=pnames)
    end
    function ModelMap(model::Function, InDomain::Function, Domain::Union{Cuboid,Bool}, xyp::Tuple{Int,Int,Int}; pnames::Union{Vector{String},Bool}=false)
        Domain = typeof(Domain) == Bool ? FullDomain(xyp[3]) : Domain
        pnames = typeof(pnames) == Bool ? CreateSymbolNames(xyp[3],"θ") : pnames
        StaticOutput = typeof(model((xyp[1] < 2 ? 1. : ones(xyp[1])), ones(xyp[3]))) <: SVector
        ModelMap(model, InDomain, Domain, xyp, pnames, Val(StaticOutput), Val(false), Val(false))
    end
    "Construct new ModelMap from function `F` with data from `M`."
    ModelMap(F::Function, M::ModelMap) = ModelMap(F, M.InDomain, M.Domain, M.xyp, M.pnames, M.StaticOutput, M.inplace, M.CustomEmbedding)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Careful with inheriting CustomEmbedding to the Jacobian! For automatically generated dmodels (symbolic or autodiff) it should be OFF!
    function ModelMap(Map::Function, InDomain::Function, Domain::Union{Cuboid,Bool}, xyp::Tuple{Int,Int,Int},
                        pnames::Vector{String}, StaticOutput::Val, inplace::Val=Val(false), CustomEmbedding::Val=Val(false))
        new(Map, InDomain, Domain, xyp, pnames, StaticOutput, inplace, CustomEmbedding)
    end
end
(M::ModelMap)(x, θ::AbstractVector{<:Number}; kwargs...) = M.Map(x, θ; kwargs...)
ModelOrFunction = Union{Function,ModelMap}


function InformNames(M::ModelMap, pnames::Vector{String})
    @assert length(pnames) == M.xyp[3]
    ModelMap(M.Map, M.InDomain, M.Domain, M.xyp, pnames, M.StaticOutput, M.inplace, M.CustomEmbedding)
end


pnames(M::ModelMap) = M.pnames
Domain(M::ModelMap) = M.Domain
isinplace(M::ModelMap) = ValToBool(M.inplace)
iscustom(M::ModelMap) = ValToBool(M.CustomEmbedding)


MakeCustom(F::Function, Domain::Union{Cuboid, Bool}=false) = Domain isa Bool ? MakeCustom(ModelMap(F)) : MakeCustom(ModelMap(F, Domain))
function MakeCustom(M::ModelMap)
    if iscustom(M)
        println("Map already uses custom embedding.")
        return M
    else
        return ModelMap(M.Map, M.InDomain, M.Domain, M.xyp, M.pnames, M.StaticOutput, M.inplace, Val(true))
    end
end
function MakeNonCustom(M::ModelMap)
    if !iscustom(M)
        println("Map already not using custom embedding.")
        return M
    else
        return ModelMap(M.Map, M.InDomain, M.Domain, M.xyp, M.pnames, M.StaticOutput, M.inplace, Val(false))
    end
end


function ModelMap(F::Nothing, M::ModelMap)
    println("ModelMap: Got nothing instead of function to build new ModelMap")
    nothing
end
function CreateSymbolNames(n::Int, base::String="θ")
    n == 1 && return [base]
    D = Dict(string.(0:9) .=> ["₀","₁","₂","₃","₄","₅","₆","₇","₈","₉"])
    base .* [prod(get(D,"$x","Q") for x in string(digit)) for digit in 1:n]
end

pdim(DS::AbstractDataSet, model::ModelMap)::Int = model.xyp[3]
function ModelMappize(DM::AbstractDataModel)
    NewMod = Predictor(DM) isa ModelMap ? Predictor(DM) : ModelMap(Predictor(DM))
    NewdMod = dPredictor(DM) isa ModelMap ? dPredictor(DM) : ModelMap(dPredictor(DM))
    DataModel(Data(DM), NewMod, NewdMod, MLE(DM))
end


function OutsideBoundariesFunction(M::ModelMap)
    OutsideBoundaries(u,t,int)::Bool = !((Res ∈ M.Domain) && M.InDomain(Res))
end


_Apply(x::AbstractVector{<:Number}, Componentwise::Function, indxs::BitVector) = [(indxs[i] ? Componentwise(x[i]) : x[i]) for i in eachindex(indxs)]
_ApplyFull(x::AbstractVector{<:Number}, Vectorial::Function) = Vectorial(x)

MonotoneIncreasing(F::Function, Interval::Tuple{Number,Number})::Bool = Monotonicity(F, Interval) == :increasing
MonotoneDecreasing(F::Function, Interval::Tuple{Number,Number})::Bool = Monotonicity(F, Interval) == :decreasing
function Monotonicity(F::Function, Interval::Tuple{Number,Number})
    derivs = map(x->ForwardDiff.derivative(F, x), range(Interval[1], Interval[2]; length=200))
    all(x-> x≥0., derivs) && return :increasing
    all(x-> x≤0., derivs) && return :decreasing
    :neither
end

Transform(F::Function, indxs::BitVector, Transform::Function, InverseTransform::Function=x->invert(Transform,x)) = _Transform(F, indxs, Transform, InverseTransform)

# Try to do a bit of inference for the new domain here!
function Transform(M::ModelMap, indxs::BitVector, Transform::Function, InverseTransform::Function=x->invert(Transform,x))
    TranslatedDomain(θ::AbstractVector{<:Number}) = M.InDomain(_Apply(θ, Transform, indxs))
    mono = Monotonicity(Transform, (1e-12,50.))
    NewCube = if mono == :increasing
        HyperCube(_Apply(M.Domain.L, InverseTransform, indxs), _Apply(M.Domain.U, InverseTransform, indxs))
    elseif mono == :decreasing
        println("Detected monotone decreasing transformation.")
        HyperCube(_Apply(M.Domain.U, InverseTransform, indxs), _Apply(M.Domain.L, InverseTransform, indxs))
    else
        FullDomain(length(indxs))
        @warn "Transformation does not appear to be monotone."
    end
    ModelMap(_Transform(M.Map, indxs, Transform, InverseTransform), TranslatedDomain, NewCube,
                        M.xyp, M.pnames, M.StaticOutput, M.inplace, M.CustomEmbedding)
end
function Transform(M::ModelMap, Transform::Function, InverseTransform::Function=x->invert(Transform,x))
    Transform(M, trues(M.xyp[3]), Transform, InverseTransform)
end


function _Transform(F::Function, indxs::BitVector, Transform::Function, InverseTransform::Function)
    function TransformedModel(x::Union{Number, AbstractVector{<:Number}}, θ::AbstractVector{<:Number}; kwargs...)
        F(x, _Apply(θ, Transform, indxs); kwargs...)
    end
end

LogTransform(F::ModelOrFunction, indxs::BitVector) = Transform(F, indxs, log, exp)
LogTransform(M::ModelMap) = LogTransform(M, trues(M.xyp[3]))

Log10Transform(F::ModelOrFunction, indxs::BitVector) = Transform(F, indxs, log10, x->10^x)
Log10Transform(M::ModelMap) = Log10Transform(M, trues(M.xyp[3]))

ReflectionTransform(F::ModelOrFunction, indxs::BitVector) = Transform(F, indxs, x-> -x, x-> -x)
ReflectionTransform(M::ModelMap) = ReflectionTransform(M, trues(M.xyp[3]))

ScaleTransform(F::ModelOrFunction, indxs::BitVector, factor::Number) = Transform(F, indxs, x->factor*x, x->x/factor)
ScaleTransform(M::ModelMap, factor::Number) = ScaleTransform(M, trues(M.xyp[3]), factor)

function TranslationTransform(F::Function, v::AbstractVector{<:Number})
    TranslatedModel(x, θ::AbstractVector{<:Number}; kwargs...) = F(x, θ + v; kwargs...)
end
function TranslationTransform(M::ModelMap, v::AbstractVector{<:Number})
    ModelMap(TranslationTransform(M.Map, v), θ->M.InDomain(θ + v), TranslateCube(M.Domain, -v), M.xyp, M.pnames, M.StaticOutput,
                                    M.inplace, M.CustomEmbedding)
end

function LinearTransform(F::Function, A::AbstractMatrix{<:Number})
    TranslatedModel(x, θ::AbstractVector{<:Number}; kwargs...) = F(x, A*θ; kwargs...)
end

function LinearTransform(M::ModelMap, A::AbstractMatrix{<:Number})
    !isposdef(A) && println("Matrix in linear transform not positive definite.")
    Ainv = inv(A)
    ModelMap(LinearTransform(M.Map, A), θ->M.InDomain(A*θ), HyperCube(Ainv * M.Domain.L, Ainv * M.Domain.U),
                    M.xyp, M.pnames, M.StaticOutput, M.inplace, M.CustomEmbedding)
end

function AffineTransform(F::Function, A::AbstractMatrix{<:Number}, v::AbstractVector{<:Number})
    TranslatedModel(x, θ::AbstractVector{<:Number}; kwargs...) = F(x, A*θ + v; kwargs...)
end

function AffineTransform(M::ModelMap, A::AbstractMatrix{<:Number}, v::AbstractVector{<:Number})
    !isposdef(A) && println("Matrix in linear transform not positive definite.")
    Ainv = inv(A)
    ModelMap(AffineTransform(M.Map, A, v), θ->M.InDomain(A*θ+v), HyperCube(Ainv * (M.Domain.L-v), Ainv * (M.Domain.U-v)),
                    M.xyp, M.pnames, M.StaticOutput, M.inplace, M.CustomEmbedding)
end



LinearModel(x::Union{Number,AbstractVector{<:Number}}, θ::AbstractVector{<:Number}) = dot(θ[1:end-1], x) + θ[end]
QuadraticModel(x::Union{Number,AbstractVector{<:Number}}, θ::AbstractVector{<:Number}) = dot(θ[1:Int((end-1)/2)], x.^2) + dot(θ[Int((end-1)/2)+1:end-1], x) + θ[end]
ExponentialModel(x::Union{Number,AbstractVector{<:Number}}, θ::AbstractVector{<:Number}) = exp(LinearModel(x,θ))
SumExponentialsModel(x::Union{Number,AbstractVector{<:Number}},θ::AbstractVector{<:Number}) = sum(exp.(θ .* x))

function PolynomialModel(degree::Int)
    Polynomial(x::Number, θ::AbstractVector{<:Number}) = sum(θ[i] * x^(i-1) for i in 1:(degree+1))
end
