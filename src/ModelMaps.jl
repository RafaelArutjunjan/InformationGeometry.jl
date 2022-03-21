

function _TestOut(model::Function, startp::AbstractVector, xlen::Int; max::Int=100)
    if MaximalNumberOfArguments(model) == 2
        model((xlen < 2 ? rand() : rand(xlen)), startp)
    else
        Res = fill(-Inf, max)
        model(Res, (xlen < 2 ? rand() : rand(xlen)), startp)
        Res[1:(findfirst(isinf, Res) - 1)]
    end
end

# Callback triggers when Boundaries is `true`.
"""
    ModelMap(Map::Function, InDomain::Union{Nothing,Function}, Domain::HyperCube)
    ModelMap(Map::Function, InDomain::Function, xyp::Tuple{Int,Int,Int})
A container which stores additional information about a model map, in particular its domain of validity.
`Map` is the actual map `(x,θ) -> model(x,θ)`. `Domain` is a `HyperCube` which allows one to roughly specify the ranges of the various parameters.
For more complicated boundary constraints, scalar function `InDomain` can be specified, which should be strictly positive on the valid parameter domain.
!!! note
    A `Bool`-valued function which returns `true` in the valid domain also fits this description, which allows one to easily combine multiple constraints.
    Providing this information about the domain can be advantageous in the optimization process for complicated models.
"""
struct ModelMap{Inplace}
    Map::Function
    InDomain::Union{Nothing,Function}
    Domain::Cuboid
    xyp::Tuple{Int,Int,Int}
    pnames::AbstractVector{<:String}
    StaticOutput::Val
    inplace::Val
    CustomEmbedding::Val
    # Given: Bool-valued domain function
    function ModelMap(model::Function, InDomain::Function, xyp::Tuple{Int,Int,Int}; pnames::Union{AbstractVector{<:String},Bool}=false)
        ModelMap(model, InDomain, nothing, xyp; pnames=pnames)
    end
    # Given: HyperCube
    function ModelMap(model::Function, Domain::Cuboid, xyp::Union{Tuple{Int,Int,Int},Bool}=false; pnames::Union{AbstractVector{<:String},Bool}=false)
        xyp isa Bool ? ModelMap(model, nothing, Domain; pnames=pnames) : ModelMap(model, nothing, Domain, xyp; pnames=pnames)
    end
    # Given: xyp
    function ModelMap(model::Function, xyp::Tuple{Int,Int,Int}; pnames::Union{AbstractVector{<:String},Bool}=false)
        ModelMap(model, nothing, FullDomain(xyp[3]), xyp; pnames=pnames)
    end
    # Given: Function only (potentially) -> Find xyp
    function ModelMap(model::Function, InDomain::Union{Nothing,Function}=nothing, Domain::Union{Cuboid,Nothing}=nothing; pnames::Union{AbstractVector{<:String},Bool}=false)
        startp = isnothing(Domain) ? GetStartP(GetArgSize(model)[2]) : ElaborateGetStartP(Domain, InDomain)
        xlen = MaximalNumberOfArguments(model) > 2 ? GetArgLength((Res,x)->model(Res,x,startp)) : GetArgLength(x->model(x,startp))
        testout = _TestOut(model, startp, xlen)
        ModelMap(model, InDomain, Domain, (xlen, size(testout,1), length(startp)); pnames=pnames)
    end
    function ModelMap(model::Function, InDomain::Union{Nothing,Function}, Domain::Union{Cuboid,Nothing}, xyp::Tuple{Int,Int,Int}; pnames::Union{AbstractVector{<:String},Bool}=false)
        pnames = typeof(pnames) == Bool ? CreateSymbolNames(xyp[3],"θ") : pnames
        startp = isnothing(Domain) ? GetStartP(xyp[3]) : ElaborateGetStartP(Domain, InDomain)
        testout = _TestOut(model, startp, xyp[1])
        StaticOutput = testout isa SVector
        Inplace = MaximalNumberOfArguments(model) > 2
        ModelMap(model, InDomain, Domain, xyp, pnames, Val(StaticOutput), Val(Inplace), Val(false))
    end
    "Construct new ModelMap from function `F` with data from `M`."
    ModelMap(F::Function, M::ModelMap; inplace::Bool=isinplacemodel(M)) = ModelMap(F, M.InDomain, M.Domain, M.xyp, M.pnames, M.StaticOutput, Val(inplace), M.CustomEmbedding)
    # Careful with inheriting CustomEmbedding to the Jacobian! For automatically generated dmodels (symbolic or autodiff) it should be OFF!
    function ModelMap(Map::Function, InDomain::Union{Nothing,Function}, Domain::Union{Cuboid,Nothing}, xyp::Tuple{Int,Int,Int},
                        pnames::AbstractVector{String}, StaticOutput::Val, inplace::Val=Val(false), CustomEmbedding::Val=Val(false))
        isnothing(Domain) && (Domain = FullDomain(xyp[3], 1e5))
        InDomain isa Function && (@assert InDomain(Center(Domain)) isa Number "InDomain function must yield a scalar value, got $(typeof(InDomain(Center(Domain)))) at $(Center(Domain)).")
        new{ValToBool(inplace)}(Map, InDomain, Domain, xyp, pnames, StaticOutput, inplace, CustomEmbedding)
    end
end
(M::ModelMap{false})(x, θ::AbstractVector{<:Number}; kwargs...) = M.Map(x, θ; kwargs...)
(M::ModelMap{true})(y, x, θ::AbstractVector{<:Number}; kwargs...) = M.Map(y, x, θ; kwargs...)
(M::ModelMap{true})(x, θ::AbstractVector{<:Number}; kwargs...) = (Res=Vector{suff(θ)}(undef, M.xyp[2]);   M.Map(Res, x, θ; kwargs...);    Res)
const ModelOrFunction = Union{Function,ModelMap}


# For SciMLBase.remake
ModelMap(;
Map::Function=x->Inf,
InDomain::Union{Nothing,Function}=nothing,
Domain::Union{Cuboid,Nothing}=nothing,
xyp::Tuple{Int,Int,Int}=(1,1,1),
pnames::AbstractVector{String}=["θ"],
StaticOutput::Val=Val(true),
inplace::Val=Val(true),
CustomEmbedding::Val=Val(true)) = ModelMap(Map, InDomain, Domain, xyp, pnames, StaticOutput, inplace, CustomEmbedding)



function InformNames(M::ModelMap, pnames::AbstractVector{String})
    @assert length(pnames) == M.xyp[3]
    ModelMap(M.Map, M.InDomain, M.Domain, M.xyp, pnames, M.StaticOutput, M.inplace, M.CustomEmbedding)
end


pnames(M::ModelMap) = M.pnames
Domain(M::ModelMap) = M.Domain
isinplacemodel(M::ModelMap) = ValToBool(M.inplace)
iscustom(M::ModelMap) = ValToBool(M.CustomEmbedding)

isinplacemodel(F::Function) = MaximalNumberOfArguments(F) == 3

IsInDomain(M::ModelMap) = θ::AbstractVector -> IsInDomain(M, θ)
IsInDomain(M::ModelMap, θ::AbstractVector) = IsInDomain(M.InDomain, M.Domain, θ)
IsInDomain(InDomain::Union{Nothing,Function}, Domain::Union{Nothing,Cuboid}, θ::AbstractVector) = (_TestInDomain(InDomain, θ) && _TestDomain(Domain, θ))

# Eval InDomain function
_TestInDomain(::Nothing, θ::AbstractVector) = true
_TestInDomain(InDomain::Function, θ::AbstractVector) = InDomain(θ) > 0
# Eval Domain HyperCube
_TestDomain(::Nothing, θ::AbstractVector) = true       # Excluded
_TestDomain(Domain::Cuboid, θ::AbstractVector) = θ ∈ Domain


MakeCustom(F::Function, Domain::Union{Bool,Nothing}=nothing) = MakeCustom(ModelMap(F))
MakeCustom(F::Function, Domain::Cuboid) = MakeCustom(ModelMap(F, Domain))
function MakeCustom(M::ModelMap)
    if iscustom(M)
        @warn "MakeCustom: Given Map already uses custom embedding."
        return M
    else
        return ModelMap(M.Map, M.InDomain, M.Domain, M.xyp, M.pnames, M.StaticOutput, M.inplace, Val(true))
    end
end
function MakeNonCustom(M::ModelMap)
    if !iscustom(M)
        @warn "MakeNonCustom: Given Map already using non-custom embedding."
        return M
    else
        return ModelMap(M.Map, M.InDomain, M.Domain, M.xyp, M.pnames, M.StaticOutput, M.inplace, Val(false))
    end
end


function ModelMap(F::Nothing, M::ModelMap)
    @warn "ModelMap: Got Nothing instead of Function to build new ModelMap."
    nothing
end
function CreateSymbolNames(n::Int, base::String="θ")
    n == 1 && return [base]
    D = Dict(string.(0:9) .=> ["₀","₁","₂","₃","₄","₅","₆","₇","₈","₉"])
    base .* [prod(get(D,"$x","Q") for x in string(digit)) for digit in 1:n]
end

pdim(DS::AbstractDataSet, model::ModelMap)::Int = model.xyp[3]

function ModelMappize(DM::AbstractDataModel; pnames::Union{AbstractVector{<:String},Bool}=false)
    NewMod = Predictor(DM) isa ModelMap ? Predictor(DM) : ModelMap(Predictor(DM), (xdim(DM), ydim(DM), pdim(DM)); pnames=pnames)
    NewdMod = dPredictor(DM) isa ModelMap ? dPredictor(DM) : ModelMap(dPredictor(DM), (xdim(DM), ydim(DM), pdim(DM)); pnames=pnames)
    DataModel(Data(DM), NewMod, NewdMod, MLE(DM), LogLikeMLE(DM))
end


"""
Only works for `DataSet` and `DataSetExact` but will output wrong order of components for `CompositeDataSet`!
"""
function ConcatenateModels(Mods::AbstractVector{<:ModelMap})
    @assert ConsistentElDims((x->x.xyp[1]).(Mods)) > 0 && ConsistentElDims((x->x.xyp[3]).(Mods)) > 0
    if Mods[1].xyp[1] == 1
        function ConcatenatedModel(x::Number, θ::AbstractVector{<:Number}; kwargs...)
            map(model->model(x, θ; kwargs...), Mods) |> Reduction
        end
        EbdMap(model::Function, θ::AbstractVector, woundX::AbstractVector,custom::Val{false}; kwargs...) = Reduction(map(x->model(x,θ; kwargs...), woundX))
        EbdMap(model::Function, θ::AbstractVector, woundX::AbstractVector,custom::Val{true}; kwargs...) = model(woundX, θ; kwargs...)
        function ConcatenatedModel(X::AbstractVector{<:Number}, θ::AbstractVector{<:Number}; kwargs...)
            if any(iscustom, Mods)
                Res = if any(m->m.xyp[2]>1, Mods)
                    map(m->Windup(EbdMap(m.Map, θ, X, m.CustomEmbedding; kwargs...), m.xyp[2]), Mods)
                    # map(m->Windup(EmbeddingMap(DS, m, θ, X), m.xyp[2]), Mods)
                else
                    map(m->EbdMap(m.Map, θ, X, m.CustomEmbedding; kwargs...), Mods)
                    # map(m->EmbeddingMap(DS, m, θ, X), Mods)
                end
                return zip(Res...) |> Iterators.flatten |> collect |> Reduction
            else
                return map(z->ConcatenatedModel(z, θ; kwargs...), X) |> Reduction
            end
        end
        return ModelMap(ConcatenatedModel, reduce(union, (z->z.Domain).(Mods)), (Mods[1].xyp[1], sum((q->q.xyp[2]).(Mods)), Mods[1].xyp[3])) |> MakeCustom
    else
        function NConcatenatedModel(x::AbstractVector{<:Number}, θ::AbstractVector{<:Number}; kwargs...)
            map(model->model(x, θ; kwargs...), Mods) |> Reduction
        end
        function NConcatenatedModel(X::AbstractVector{<:AbstractVector{<:Number}}, θ::AbstractVector{<:Number}; kwargs...)
            if any(iscustom, Mods)
                Res = if any(m->m.xyp[2]>1, Mods)
                    map(m->Windup(EmbeddingMap(DS, m, θ, X), m.xyp[2]), Mods)
                else
                    map(m->EmbeddingMap(DS, m, θ, X), Mods)
                end
                return zip(Res...) |> Iterators.flatten |> collect |> Reduction
            else
                return map(z->NConcatenatedModel(z, θ; kwargs...), X) |> Reduction
            end
        end
        return ModelMap(NConcatenatedModel, reduce(union, (z->z.Domain).(Mods)), (Mods[1].xyp[1], sum((q->q.xyp[2]).(Mods)), Mods[1].xyp[3])) |> MakeCustom
    end
end



_Apply(x::AbstractVector{<:Number}, Componentwise::Function, idxs::BoolVector) = [(idxs[i] ? Componentwise(x[i]) : x[i]) for i in eachindex(idxs)]
_ApplyFull(x::AbstractVector{<:Number}, Vectorial::Function) = Vectorial(x)

MonotoneIncreasing(F::Function, Interval::Tuple{Number,Number})::Bool = Monotonicity(F, Interval) == :increasing
MonotoneDecreasing(F::Function, Interval::Tuple{Number,Number})::Bool = Monotonicity(F, Interval) == :decreasing
function Monotonicity(F::Function, Interval::Tuple{Number,Number})
    derivs = map(GetDeriv(Val(:ForwardDiff),F), range(Interval[1], Interval[2]; length=200))
    all(x-> x≥0., derivs) && return :increasing
    all(x-> x≤0., derivs) && return :decreasing
    :neither
end

Transform(model::Function, idxs::BoolVector, Transform::Function, InverseTransform::Function=x->invert(Transform,x)) = _Transform(model, idxs, Transform, InverseTransform)

# Try to do a bit of inference for the new domain here!
function Transform(M::ModelMap, idxs::BoolVector, Transform::Function, InverseTransform::Function=x->invert(Transform,x))
    TransformedDomain = M.InDomain isa Function ? (θ::AbstractVector{<:Number} -> M.InDomain(_Apply(θ, Transform, idxs))) : nothing
    mono = Monotonicity(Transform, (1e-12,50.))
    NewCube = if mono == :increasing
        HyperCube(_Apply(M.Domain.L, InverseTransform, idxs), _Apply(M.Domain.U, InverseTransform, idxs))
    elseif mono == :decreasing
        @warn "Detected monotone decreasing transformation."
        HyperCube(_Apply(M.Domain.U, InverseTransform, idxs), _Apply(M.Domain.L, InverseTransform, idxs))
    else
        @warn "Transformation does not appear to be monotone. Unable to infer new Domain."
        FullDomain(length(idxs))
    end
    ModelMap(_Transform(M.Map, idxs, Transform, InverseTransform), TransformedDomain, NewCube,
                        M.xyp, M.pnames, M.StaticOutput, M.inplace, M.CustomEmbedding)
end
# function Transform(M::ModelMap, Transform::Function, InverseTransform::Function=x->invert(Transform,x))
#     Transform(M, trues(M.xyp[3]), Transform, InverseTransform)
# end


function _Transform(F::Function, idxs::BoolVector, Transform::Function, InverseTransform::Function)
    function TransformedModel(x::Union{Number, AbstractVector{<:Number}}, θ::AbstractVector{<:Number}; kwargs...)
        F(x, _Apply(θ, Transform, idxs); kwargs...)
    end
end


"""
    Transform(DM::AbstractDataModel, F::Function, idxs=trues(pdim(DM))) -> DataModel
    Transform(model::Function, idxs, F::Function) -> Function
Transforms the parameters of the model by the given scalar function `F` such that `newmodel(x, θ) = oldmodel(x, F.(θ))`.
By providing `idxs`, one may restrict the application of the function `F` to specific parameter components.
"""
function Transform(DM::AbstractDataModel, F::Function, idxs::BoolVector=trues(pdim(DM)); kwargs...)
    @assert length(idxs) == pdim(DM)
    sum(idxs) == 0 && return DM
    DataModel(Data(DM), Transform(Predictor(DM), idxs, F), _Apply(MLE(DM), x->invert(F,x), idxs); kwargs...)
end
function Transform(DM::AbstractDataModel, F::Function, inverseF::Function, idxs::BoolVector=trues(pdim(DM)); kwargs...)
    @assert length(idxs) == pdim(DM)
    sum(idxs) == 0 && return DM
    DataModel(Data(DM), Transform(Predictor(DM), idxs, F, inverseF), _Apply(MLE(DM), inverseF, idxs); kwargs...)
end


LogTransform(M::ModelOrFunction, idxs::BoolVector=(M isa ModelMap ? trues(M.xyp[3]) : trues(GetArgSize(M)[2]))) = Transform(M, idxs, log, exp)
LogTransform(DM::AbstractDataModel, idxs::BoolVector=trues(pdim(DM)); kwargs...) = Transform(DM, log, exp, idxs; kwargs...)

ExpTransform(M::ModelOrFunction, idxs::BoolVector=(M isa ModelMap ? trues(M.xyp[3]) : trues(GetArgSize(M)[2]))) = Transform(M, idxs, exp, log)
ExpTransform(DM::AbstractDataModel, idxs::BoolVector=trues(pdim(DM)); kwargs...) = Transform(DM, exp, log, idxs; kwargs...)

Log10Transform(M::ModelOrFunction, idxs::BoolVector=(M isa ModelMap ? trues(M.xyp[3]) : trues(GetArgSize(M)[2]))) = Transform(M, idxs, log10, exp10)
Log10Transform(DM::AbstractDataModel, idxs::BoolVector=trues(pdim(DM)); kwargs...) = Transform(DM, log10, exp10, idxs; kwargs...)

Exp10Transform(M::ModelOrFunction, idxs::BoolVector=(M isa ModelMap ? trues(M.xyp[3]) : trues(GetArgSize(M)[2]))) = Transform(M, idxs, exp10, log10)
Exp10Transform(DM::AbstractDataModel, idxs::BoolVector=trues(pdim(DM)); kwargs...) = Transform(DM, exp10, log10, idxs; kwargs...)

@deprecate Power10Transform(args...; kwargs...) Exp10Transform(args...; kwargs...)

ReflectionTransform(M::ModelOrFunction, idxs::BoolVector=(M isa ModelMap ? trues(M.xyp[3]) : trues(GetArgSize(M)[2]))) = Transform(M, idxs, x-> -x, x-> -x)
ReflectionTransform(DM::AbstractDataModel, idxs::BoolVector=trues(pdim(DM)); kwargs...) = Transform(DM, x-> -x, x-> -x, idxs; kwargs...)

ScaleTransform(M::ModelOrFunction, factor::Number, idxs::BoolVector=(M isa ModelMap ? trues(M.xyp[3]) : trues(GetArgSize(M)[2]))) = Transform(M, idxs, x->factor*x, x->x/factor)
ScaleTransform(DM::AbstractDataModel, factor::Number, idxs::BoolVector=trues(pdim(DM)); kwargs...) = Transform(DM, x->factor*x, x->x/factor, idxs; kwargs...)


function TranslationTransform(F::Function, v::AbstractVector{<:Number})
    TranslatedModel(x, θ::AbstractVector{<:Number}; kwargs...) = F(x, θ + v; kwargs...)
end
function TranslationTransform(M::ModelMap, v::AbstractVector{<:Number})
    @assert length(M.Domain) == length(v)
    ModelMap(TranslationTransform(M.Map, v), (M.InDomain isa Function ? (θ->M.InDomain(θ + v)) : nothing), TranslateCube(M.Domain, -v), M.xyp, M.pnames, M.StaticOutput,
                                    M.inplace, M.CustomEmbedding)
end
function TranslationTransform(DM::AbstractDataModel, v::AbstractVector{<:Number}; kwargs...)
    @assert pdim(DM) == length(v)
    DataModel(Data(DM), TranslationTransform(Predictor(DM), v), MLE(DM)-v; kwargs...)
end


function LinearTransform(F::Function, A::AbstractMatrix{<:Number})
    TransformedModel(x, θ::AbstractVector{<:Number}; kwargs...) = F(x, A*θ; kwargs...)
end
function LinearTransform(M::ModelMap, A::AbstractMatrix{<:Number})
    @assert length(M.Domain) == size(A,1) == size(A,2)
    Ainv = inv(A)
    ModelMap(LinearTransform(M.Map, A), (M.InDomain isa Function ? (θ->M.InDomain(A*θ)) : nothing), HyperCube(Ainv * M.Domain.L, Ainv * M.Domain.U),
                    M.xyp, M.pnames, M.StaticOutput, M.inplace, M.CustomEmbedding)
end
function LinearTransform(DM::AbstractDataModel, A::AbstractMatrix{<:Number}; kwargs...)
    @assert pdim(DM) == size(A,1) == size(A,2)
    DataModel(Data(DM), LinearTransform(Predictor(DM), A), inv(A)*MLE(DM); kwargs...)
end


function AffineTransform(F::Function, A::AbstractMatrix{<:Number}, v::AbstractVector{<:Number})
    @assert size(A,1) == size(A,2) == length(v)
    TranslatedModel(x, θ::AbstractVector{<:Number}; kwargs...) = F(x, A*θ + v; kwargs...)
end
function AffineTransform(M::ModelMap, A::AbstractMatrix{<:Number}, v::AbstractVector{<:Number})
    @assert length(M.Domain) == size(A,1) == size(A,2) == length(v)
    Ainv = inv(A)
    ModelMap(AffineTransform(M.Map, A, v), (M.InDomain isa Function ? (θ->M.InDomain(A*θ+v)) : nothing), HyperCube(Ainv*(M.Domain.L-v), Ainv*(M.Domain.U-v)),
                    M.xyp, M.pnames, M.StaticOutput, M.inplace, M.CustomEmbedding)
end
function AffineTransform(DM::AbstractDataModel, A::AbstractMatrix{<:Number}, v::AbstractVector{<:Number}; kwargs...)
    @assert pdim(DM) == size(A,1) == size(A,2) == length(v)
    Ainv = inv(A)
    DataModel(Data(DM), AffineTransform(Predictor(DM), A, v), Ainv*(MLE(DM)-v); kwargs...)
end

_GetDecorrelationTransform(DM::AbstractDataModel) = (_GetDecorrelationTransform(FisherMetric(DM, MLE(DM))), MLE(DM))
_GetDecorrelationTransform(M::AbstractMatrix) = cholesky(Symmetric(inv(M))).L
LinearDecorrelation(DM::AbstractDataModel; kwargs...) = ((M,X) =_GetDecorrelationTransform(DM); AffineTransform(DM, M, X; kwargs...))

function DecorrelationTransforms(DM::AbstractDataModel)
    M, X = _GetDecorrelationTransform(DM);      iM = inv(M)
    ForwardTransform(x::AbstractVector) = M * x + X
    InvTransform(x::AbstractVector) = iM * (x - X)
    ForwardTransform, InvTransform
end


"""
    EmbedModelVia(model, F::Function; Domain::HyperCube=FullDomain(GetArgLength(F))) -> Union{Function,ModelMap}
Transforms a model function via `newmodel(x, θ) = oldmodel(x, F(θ))`.
A `Domain` for the new model can optionally be specified for `ModelMap`s.
"""
EmbedModelVia(model::Function, F::Function; Kwargs...) = EmbeddedModel(x, θ; kwargs...) = model(x, F(θ); kwargs...)
EmbedModelVia_inplace(model!::Function, F::Function; Kwargs...) = EmbeddedModel!(y, x, θ; kwargs...) = model!(y, x, F(θ); kwargs...)

function EmbedModelVia(M::ModelMap, F::Function; Domain::Union{Nothing,HyperCube}=nothing)
    if isnothing(Domain)
        Domain = FullDomain(GetArgLength(F))
        @warn "Cannot infer new Domain HyperCube for general embeddings, using $Domain."
    end
    ModelMap((isinplacemodel(M) ? EmbedModelVia_inplace : EmbedModelVia)(M.Map, F), (M.InDomain isa Function ? (M.InDomain∘F) : nothing),
            Domain, (M.xyp[1], M.xyp[2], length(Domain)), CreateSymbolNames(length(Domain), "θ"),
            M.StaticOutput, M.inplace, M.CustomEmbedding)
end

function EmbedDModelVia(dmodel::Function, F::Function, Size::Tuple=Tuple([]); ADmode::Union{Symbol,Val}=Val(:ForwardDiff), Kwargs...)
    Jac = GetJac(ADmode, F)
    EmbeddedJacobian(x, θ; kwargs...) = dmodel(x, F(θ); kwargs...) * Jac(θ)
end
function EmbedDModelVia_inplace(dmodel!::Function, F::Function, Size::Tuple{Int,Int}; ADmode::Union{Symbol,Val}=Val(:ForwardDiff), Kwargs...)
    Jac = GetJac(ADmode, F)
    function EmbeddedJacobian!(y, x, θ::AbstractVector{T}; kwargs...) where T<:Number
        Ycache = Matrix{T}(undef, Size)
        dmodel!(Ycache, x, F(θ); kwargs...)
        mul!(y, Ycache, Jac(θ))
    end
end
function EmbedDModelVia(dM::ModelMap, F::Function; Domain::HyperCube=FullDomain(GetArgLength(F)))
    # Pass the OLD pdim to EmbedDModelVia_inplace for cache
    ModelMap((isinplacemodel(dM) ? EmbedDModelVia_inplace : EmbedDModelVia)(dM.Map, F, dM.xyp[2:3]), (dM.InDomain isa Function ? (dM.InDomain∘F) : nothing),
            Domain, (dM.xyp[1], dM.xyp[2], length(Domain)), CreateSymbolNames(length(Domain), "θ"),
            dM.StaticOutput, dM.inplace, dM.CustomEmbedding)
end

"""
    Embedding(DM::AbstractDataModel, F::Function, start::AbstractVector; Domain::HyperCube=FullDomain(length(start))) -> DataModel
Transforms a model function via `newmodel(x, θ) = oldmodel(x, F(θ))` and returns the associated `DataModel`.
An initial parameter configuration `start` as well as a `Domain` can optionally be passed to the `DataModel` constructor.
"""
function Embedding(DM::AbstractDataModel, F::Function, start::AbstractVector{<:Number}=GetStartP(GetArgLength(F)); Domain::HyperCube=FullDomain(length(start)), kwargs...)
    DataModel(Data(DM), EmbedModelVia(Predictor(DM), F; Domain=Domain), EmbedDModelVia(dPredictor(DM), F; Domain=Domain), start; kwargs...)
end


## Transform dependent and independent variables of model

# in-place
EmbedModelXin(model::Function, Emb::Function) = XEmbeddedModel(y, x, θ::AbstractVector; kwargs...) = model(y, Emb(x), θ; kwargs...)
# out-of-place
EmbedModelXout(model::Function, Emb::Function) = XEmbeddedModel(x, θ::AbstractVector; kwargs...) = model(Emb(x), θ; kwargs...)

"""
    EmbedModelX(model::Function, Emb::Function)
Embeds the independent variables of a model function via `newmodel(x,θ) = oldmodel(Emb(x),θ)`.
"""
EmbedModelX(M::ModelMap, Emb::Function) = ModelMap((isinplacemodel(M) ? EmbedModelXin : EmbedModelXout)(M.Map, Emb), M)
EmbedModelX(model::Function, Emb::Function) = (MaximalNumberOfArguments(model) == 3 ? EmbedModelXin : EmbedModelXout)(model, Emb)

"""
    TransformXdata(DM::AbstractDataModel, Emb::Function, iEmb::Function, Name::String="Transform") -> AbstractDataModel
    TransformXdata(DS::AbstractDataSet, Emb::Function, Name::String="Transform") -> AbstractDataSet
Returns a modified `DataModel` where the x-variables have been transformed by a multivariable transform `Emb` both in the data as well as for the model via `newmodel(x,θ) = oldmodel(Emb(x),θ)`.
`iEmb` denotes the inverse of `Emb`.
"""
function TransformXdata(DM::AbstractDataModel, Emb::Function, iEmb::Function, Name::String="Transform"; kwargs...)
    @assert all(WoundX(DM) .≈ map(iEmb∘Emb, WoundX(DM))) # Check iEmb is correct inverse
    DataModel(TransformXdata(Data(DM), Emb, Name; kwargs...), EmbedModelX(Predictor(DM), iEmb), EmbedModelX(dPredictor(DM), iEmb), MLE(DM))
end
function TransformXdata(DS::AbstractDataSet, Emb::Function, Name::String="Transform"; xnames=Name*"(".*xnames(DS).*")", ADmode::Union{Val,Symbol}=Val(:ForwardDiff))
    NewX = Reduction(map(Emb, WoundX(DS)))
    if sum(abs, xsigma(DS)) == 0
        typeof(DS)(NewX, ydata(DS), ysigma(DS), dims(DS); xnames=xnames, ynames=ynames(DS))
    else
        @assert xsigma(DS) isa AbstractVector
        EmbJac = xdim(DS) > 1 ? GetJac(ADmode, Emb, xdim(DS)) : GetDeriv(ADmode, Emb)
        NewXsigma = map((xdat, xsig)->EmbJac(xdat)*xsig, WoundX(DS), Windup(xsigma(DS), xdim(DS))) # |> Reduction
        typeof(DS)(NewX, NewXsigma, ydata(DS), ysigma(DS), dims(DS); xnames=xnames, ynames=ynames(DS))
    end
end
# Drop iEmb
TransformXdata(DS::AbstractDataSet, Emb::Function, iEmb::Function, args...; kwargs...) = TransformXdata(DS, Emb, args...; kwargs...)


"""
    LogXdata(DM::AbstractDataModel)
    LogXdata(DS::AbstractDataSet)
Returns a modified `DataModel` where the x-variables have been logarithmized both in the data as well as for the model.
"""
LogXdata(DM::Union{AbstractDataModel,AbstractDataSet}; kwargs...) = TransformXdata(DM, x->broadcast(log,x), x->broadcast(exp,x), "log"; kwargs...)
Log10Xdata(DM::Union{AbstractDataModel,AbstractDataSet}; kwargs...) = TransformXdata(DM, x->broadcast(log10,x), x->broadcast(exp10,x), "log10"; kwargs...)
ExpXdata(DM::Union{AbstractDataModel,AbstractDataSet}; kwargs...) = TransformXdata(DM, x->broadcast(exp,x), x->broadcast(log,x), "exp"; kwargs...)
Exp10Xdata(DM::Union{AbstractDataModel,AbstractDataSet}; kwargs...) = TransformXdata(DM, x->broadcast(exp10,x), x->broadcast(log10,x), "exp10"; kwargs...)



# in-place
EmbedModelYin(model::Function, Emb::Function) = YEmbeddedModel(y, x, θ::AbstractVector; kwargs...) = (model(y, x, θ; kwargs...);   y=Emb(y))
# out-of-place
EmbedModelYout(model::Function, Emb::Function) = YEmbeddedModel(x, θ::AbstractVector; kwargs...) = Emb(model(x, θ; kwargs...))

"""
    EmbedModelY(model::Function, Emb::Function)
Embeds the independent variables of a model function via `newmodel(x,θ) = Emb(oldmodel(x,θ))`.
"""
EmbedModelY(M::ModelMap, Emb::Function) = ModelMap((isinplacemodel(M) ? EmbedModelYin : EmbedModelYout)(M.Map, Emb), M)
EmbedModelY(model::Function, Emb::Function) = (MaximalNumberOfArguments(model) == 3 ? EmbedModelYin : EmbedModelYout)(model, Emb)


# Unlike X-transform, model uses same embedding function for Y instead of inverse to compensate
"""
    TransformYdata(DM::AbstractDataModel, Emb::Function, Name::String="Transform") -> AbstractDataModel
    TransformYdata(DS::AbstractDataSet, Emb::Function, Name::String="Transform") -> AbstractDataSet
Returns a modified `DataModel` where the y-variables have been transformed by a multivariable transform `Emb` both in the data as well as for the model via `newmodel(x,θ) = Emb(oldmodel(x,θ))`.
"""
function TransformYdata(DM::AbstractDataModel, Emb::Function, Name::String="Transform"; kwargs...)
    DataModel(TransformYdata(Data(DM), Emb, Name; kwargs...), EmbedModelY(Predictor(DM), Emb), MLE(DM))
end
function TransformYdata(DS::AbstractDataSet, Emb::Function, Name::String="Transform"; ynames=Name*"(".*ynames(DS).*")", ADmode::Union{Val,Symbol}=Val(:ForwardDiff))
    @assert ysigma(DS) isa AbstractVector
    NewY = Reduction(map(Emb, WoundY(DS)));    EmbJac = ydim(DS) > 1 ? GetJac(ADmode, Emb, ydim(DS)) : GetDeriv(ADmode, Emb)
    NewYsigma = map((ydat, ysig)->EmbJac(ydat)*ysig, WoundY(DS), Windup(ysigma(DS), ydim(DS))) # |> Reduction
    if sum(abs, xsigma(DS)) == 0
        typeof(DS)(xdata(DS), NewY, NewYsigma, dims(DS); xnames=xnames(DS), ynames=ynames)
    else
        typeof(DS)(xdata(DS), xsigma(DS), NewY, NewYsigma, dims(DS); xnames=xnames(DS), ynames=ynames)
    end
end


"""
    LogYdata(DM::AbstractDataModel)
    LogYdata(DS::AbstractDataSet)
Returns a modified `DataModel` where the y-variables have been logarithmized both in the data as well as for the model.
"""
LogYdata(DM::Union{AbstractDataModel,AbstractDataSet}; kwargs...) = TransformYdata(DM, x->broadcast(log,x), "log")
Log10Ydata(DM::Union{AbstractDataModel,AbstractDataSet}; kwargs...) = TransformYdata(DM, x->broadcast(log10,x), "log10")
ExpYdata(DM::Union{AbstractDataModel,AbstractDataSet}; kwargs...) = TransformYdata(DM, x->broadcast(exp,x), "exp")
Exp10Ydata(DM::Union{AbstractDataModel,AbstractDataSet}; kwargs...) = TransformYdata(DM, x->broadcast(exp10,x), "exp10")




# Parameter transforms
ExpTransform(Sys::ODESystem, idxs::AbstractVector{<:Bool}=trues(length(parameters(Sys))); kwargs...) = SystemTransform(Sys, exp, idxs; kwargs...)
LogTransform(Sys::ODESystem, idxs::AbstractVector{<:Bool}=trues(length(parameters(Sys))); kwargs...) = SystemTransform(Sys, log, idxs; kwargs...)
Exp10Transform(Sys::ODESystem, idxs::AbstractVector{<:Bool}=trues(length(parameters(Sys))); kwargs...) = SystemTransform(Sys, exp10, idxs; kwargs...)
Log10Transform(Sys::ODESystem, idxs::AbstractVector{<:Bool}=trues(length(parameters(Sys))); kwargs...) = SystemTransform(Sys, log10, idxs; kwargs...)

"""
    SystemTransform(Sys::ODESystem, F::Function, idxs::AbstractVector{<:Bool}=trues(length(parameters(Sys)))) -> ODESystem
Transforms the parameters of a `ODESystem` according to `F`.
"""
function SystemTransform(Sys::ODESystem, F::Function, idxs::AbstractVector{<:Bool}=trues(length(parameters(Sys))))
    SubstDict = Dict(parameters(Sys) .=> [(idxs[i] ? F(x) : x) for (i,x) in enumerate(parameters(Sys))])
    NewEqs = [(equations(Sys)[i].lhs ~ substitute(equations(Sys)[i].rhs, SubstDict)) for i in 1:length(equations(Sys))]
    ODESystem(NewEqs, independent_variables(Sys)[1], states(Sys), parameters(Sys); name=nameof(Sys))
end

"""
    LinearModel(x::Union{Number,AbstractVector{<:Number}}, θ::AbstractVector{<:Number})
```math
y(x,θ) = θ_{n+1} + x_1 * θ_1 + x_2 * θ_2 + ... + x_n * θ_n
```
"""
LinearModel(x::Union{Number,AbstractVector{<:Number}}, θ::AbstractVector{<:Number}) = dot(view(θ, 1:(length(θ)-1)), x) + θ[end]
QuadraticModel(x::Union{Number,AbstractVector{<:Number}}, θ::AbstractVector{<:Number}) = (n=length(θ);  dot(view(θ,1:((n-1)÷2)), x.^2) + dot(view(θ,(n-1)÷2+1:n-1), x) + θ[end])
ExponentialModel = exp∘LinearModel
SumExponentialsModel(x::Union{Number,AbstractVector{<:Number}}, θ::AbstractVector{<:Number}) = sum(exp.(θ .* x))
PolynomialModel(degree::Int) = Polynomial(x::Number, θ::AbstractVector{<:Number}) = sum(θ[i] * x^(i-1) for i in 1:(degree+1))


function GetLinearModel(DS::AbstractDataSet)
    ydim(DS) != 1 && return GetGeneralLinearModel(DS)
    Names = "p_(" .* ynames(DS) .* " × " .* xnames(DS) .*")"
    push!(Names, "p_(" * ynames(DS)[1] * " × Offset)")
    ModelMap(LinearModel, (xdim(DS), ydim(DS), xdim(DS)+1); pnames=Names)
end

function GetGeneralLinearModel(DS::AbstractDataSet)
    ydim(DS) == 1 && return GetLinearModel(DS)
    Xdim, Ydim = xdim(DS), ydim(DS)
    NaiveGeneralLinearModel(x::AbstractVector{<:Number}, θ::AbstractVector{T}) where T <: Number = SVector{Ydim, T}(LinearModel(x, p) for p in Iterators.partition(θ, Xdim+1))
    Names = ["p_(" .* ynames(DS)[i] .* " × " .* xnames(DS) .*")" for i in 1:ydim(DS)]
    for (i,series) in enumerate(Names)
        push!(series, "p_(" * ynames(DS)[i] * " × Offset)")
    end
    Optimize(ModelMap(NaiveGeneralLinearModel, nothing, nothing, (xdim(DS), ydim(DS), ydim(DS)*(xdim(DS)+1)), reduce(vcat, Names), Val(true), Val(false), Val(false)); inplace=false)[1]
end


IsDEbased(F::Function) = occursin("DEmodel", string(nameof(typeof(F))))
IsDEbased(F::ModelMap) = IsDEbased(F.Map)
IsDEbased(DM::AbstractDataModel) = IsDEbased(Predictor(DM))
