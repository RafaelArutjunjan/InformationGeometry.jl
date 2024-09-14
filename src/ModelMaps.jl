

function _TestOut(model::Function, startp::AbstractVector, xlen::Int; max::Int=100)
    if !isinplacemodel(model)
        model((xlen < 2 ? rand() : rand(xlen)), startp)
    else
        Res = fill(-Inf, max)
        model(Res, (xlen < 2 ? rand() : rand(xlen)), startp)
        Res[1:(findfirst(isinf, Res) - 1)]
    end
end
function CheckIfIsCustom(model::Function, startp::AbstractVector, xyp::Tuple, IsInplace::Bool)
    woundX = Windup(collect(1:2xyp[1]) .+ 0.1rand(2xyp[1]), xyp[1])
    if !IsInplace
        try   length(model(woundX, startp)) == 2xyp[2]   catch;  false   end
    else
        Res = fill(-Inf, 2xyp[2])
        try   model(Res, woundX, startp);   !any(isinf, Res)    catch; false  end
    end
end

function ConstructModelxyp(model::Function)
    xp = GetArgSize(model)
    (xp[1], length(_TestOut(model, GetStartP(xp[2]), xp[1])), xp[2])
end

# Callback triggers when Boundaries is `true`.
"""
    ModelMap(Map::Function, InDomain::Union{Nothing,Function}, Domain::HyperCube; startp::AbstractVector)
    ModelMap(Map::Function, InDomain::Function, xyp::Tuple{Int,Int,Int})
A container which stores additional information about a model map, in particular its domain of validity.
`Map` is the actual map `(x,θ) -> model(x,θ)`. `Domain` is a `HyperCube` which allows one to roughly specify the ranges of the various parameters.
For more complicated boundary constraints, a function `InDomain(θ)` can be specified, for which all outputted components should be .≥ 0 on the valid parameter domain.
Alternatively, `InDomain` may also be a bool-valued function, evaluating to `true` in admissible parts of the parameter domain.

The kwarg `startp` may be used to pass a suitable parameter vector for the ModelMap.

!!! note
    A `Bool`-valued function which returns `true` in the valid domain also fits this description, which allows one to easily combine multiple constraints.
    Providing this information about the domain can be advantageous in the optimization process for complicated models.
"""
struct ModelMap{Inplace, Custom}
    Map::Function
    InDomain::Union{Nothing,Function}
    Domain::Cuboid
    xyp::Tuple{Int,Int,Int}
    pnames::AbstractVector{<:AbstractString}
    inplace::Val
    CustomEmbedding::Val
    name::Symbol
    Meta
    # Given: Bool-valued domain function
    function ModelMap(model::Function, InDomain::Function, xyp::Tuple{Int,Int,Int}; kwargs...)
        ModelMap(model, InDomain, nothing, xyp; kwargs...)
    end
    # Given: HyperCube
    function ModelMap(model::Function, Domain::Cuboid, xyp::Union{Tuple{Int,Int,Int},Bool}=false; kwargs...)
        xyp isa Bool ? ModelMap(model, nothing, Domain; kwargs...) : ModelMap(model, nothing, Domain, xyp; kwargs...)
    end
    # Given: xyp
    function ModelMap(model::Function, xyp::Tuple{Int,Int,Int}; kwargs...)
        ModelMap(model, nothing, nothing, xyp; kwargs...)
    end
    # Given: Function only (potentially) -> Find xyp
    function ModelMap(model::Function, InDomain::Union{Nothing,Function}=nothing, Domain::Union{Cuboid,Nothing}=nothing; 
                            startp::AbstractVector{<:Number}=isnothing(Domain) ? GetStartP(GetArgSize(model)[2]) : ElaborateGetStartP(Domain, InDomain), kwargs...)
        ModelMap(model, startp, InDomain, Domain; kwargs...)
    end
    function ModelMap(model::Function, startp::AbstractVector{<:Number}, InDomain::Union{Nothing,Function}=nothing, Domain::Union{Cuboid,Nothing}=nothing; kwargs...)
        xlen = isinplacemodel(model) ? GetArgLength((Res,x)->model(Res,x,startp)) : GetArgLength(x->model(x,startp))
        testout = _TestOut(model, startp, xlen)
        ModelMap(model, InDomain, Domain, (xlen, size(testout,1), length(startp)); startp=startp, kwargs...)
    end
    function ModelMap(model::Function, InDomain::Union{Nothing,Function}, Domain::Union{Cuboid,Nothing}, xyp::Tuple{Int,Int,Int}; pnames::AbstractVector{<:AbstractString}=String[], name::Union{<:AbstractString,Symbol}=Symbol(), Meta=nothing, 
                            startp::AbstractVector{<:Number}=isnothing(Domain) ? GetStartP(xyp[3]) : ElaborateGetStartP(Domain, InDomain), kwargs...)
        pnames = length(pnames) == 0 ? GetParameterNames(startp) : pnames
        # startp = isnothing(Domain) ? GetStartP(xyp[3]) : ElaborateGetStartP(Domain, InDomain)
        # testout = _TestOut(model, startp, xyp[1])
        # StaticOutput = testout isa SVector
        Inplace = isinplacemodel(model)
        # Given xyp, check if given model is custom, i.e. if it can output sensible values for woundX input
        IsCustom = CheckIfIsCustom(model, startp, xyp, Inplace)
        ModelMap(model, InDomain, Domain, xyp, pnames, Val(Inplace), Val(IsCustom), name, Meta; kwargs...)
    end
    "Construct new ModelMap from function `F` with data from `M`."
    ModelMap(F::Function, M::ModelMap; inplace::Bool=isinplacemodel(M)) = ModelMap(F, InDomain(M), Domain(M), M.xyp, M.pnames, Val(inplace), M.CustomEmbedding, name(M), M.Meta)
    # Careful with inheriting CustomEmbedding to the Jacobian! For automatically generated dmodels (symbolic or autodiff) it should be OFF!
    # function ModelMap(Map::Function, InDomain::Union{Nothing,Function}, Domain::Union{Cuboid,Nothing}, xyp::Tuple{Int,Int,Int},
    #                     pnames::AbstractVector{<:AbstractString}, StaticOutput::Val, inplace::Val=Val(false), CustomEmbedding::Val=Val(false), name::Symbol=Symbol())
    #     isnothing(Domain) && (Domain = FullDomain(xyp[3], 1e5))
    #     InDomain isa Function && (@assert InDomain(Center(Domain)) isa Number "InDomain function must yield a scalar value, got $(typeof(InDomain(Center(Domain)))) at $(Center(Domain)).")
    #     new{ValToBool(inplace)}(Map, InDomain, Domain, xyp, pnames, StaticOutput, inplace, CustomEmbedding, name)
    # end
    (@deprecate ModelMap(Map::Function, InDomain::Union{Nothing,Function}, Domain::Union{Cuboid,Nothing}, xyp::Tuple{Int,Int,Int},
                    pnames::AbstractVector{<:AbstractString}, StaticOutput::Val, inplace::Val, CustomEmbedding::Val, name::Symbol) ModelMap(Map, InDomain, Domain, xyp, pnames, inplace, CustomEmbedding, name, nothing))

    function ModelMap(Map::Function, InDomain::Union{Nothing,Function}, Domain::Union{Cuboid,Nothing}, xyp::Tuple{Int,Int,Int},
                        pnames::AbstractVector{<:AbstractString}, inplace::Val, CustomEmbedding::Val, name::Union{<:AbstractString,Symbol}=Symbol(), Meta=nothing)
        name isa AbstractString && (name = Symbol(name))
        @assert allunique(pnames) "Parameter names must be unique within a model, got $pnames."
        isnothing(Domain) ? (Domain = FullDomain(xyp[3], 1e5)) : (@assert length(Domain) == xyp[3] "Given Domain Hypercube $Domain does not fit inferred number of parameters $(xyp[3]).")
        InDomain isa Function && (@assert InDomain(Center(Domain)) isa Number "InDomain function must yield a scalar value, got $(typeof(InDomain(Center(Domain)))) at $(Center(Domain)).")
        new{ValToBool(inplace), ValToBool(CustomEmbedding)}(Map, InDomain, Domain, xyp, pnames, inplace, CustomEmbedding, name, Meta)
    end
end
(M::ModelMap{false})(x, θ::AbstractVector{<:Number}; kwargs...) = M.Map(x, θ; kwargs...)
(M::ModelMap{true})(y, x, θ::AbstractVector{<:Number}; kwargs...) = M.Map(y, x, θ; kwargs...)
(M::ModelMap{true})(x, θ::AbstractVector{T}; kwargs...) where T<:Number = (Res=Vector{T}(undef, M.xyp[2]);   M.Map(Res, x, θ; kwargs...);    Res)
const ModelOrFunction = Union{Function,ModelMap}


# For SciMLBase.remake
ModelMap(;
Map::Function=x->Inf,
InDomain::Union{Nothing,Function}=nothing,
Domain::Union{Cuboid,Nothing}=nothing,
xyp::Tuple{Int,Int,Int}=(1,1,1),
pnames::AbstractVector{<:AbstractString}=["θ"],
inplace::Val=Val(true),
CustomEmbedding::Val=Val(true),
name::Symbol=Symbol(),
Meta=nothing) = ModelMap(Map, InDomain, Domain, xyp, pnames, inplace, CustomEmbedding, name, Meta)



function InformNames(M::ModelMap, pnames::AbstractVector{<:AbstractString})
    @assert length(pnames) == M.xyp[3]
    ModelMap(M.Map, InDomain(M), Domain(M), M.xyp, pnames, M.inplace, M.CustomEmbedding, name(M), M.Meta)
end


pnames(M::ModelMap) = M.pnames
name(M::ModelMap) = M.name |> string
name(F::Function) = ""
Domain(M::ModelMap) = M.Domain
Domain(arg) = throw("$arg does not have a Domain.")
InDomain(M::ModelMap) = M.InDomain
InDomain(arg) = throw("$arg does not have an InDomain function.")

iscustommodel(M::ModelMap) = ValToBool(M.CustomEmbedding)
iscustommodel(F::Function) = false

isinplacemodel(M::ModelMap) = ValToBool(M.inplace)
isinplacemodel(F::Function) = MaximalNumberOfArguments(F) == 3
isinplacemodel(DM::AbstractDataModel) = isinplacemodel(Predictor(DM))

IsInDomain(DM::AbstractDataModel) = IsInDomain(Predictor(DM))
IsInDomain(F::Function) = θ::AbstractVector -> true

IsInDomain(DM::AbstractDataModel, θ::AbstractVector) = IsInDomain(Predictor(DM), θ)
IsInDomain(F::Function, θ::AbstractVector) = true

IsInDomain(M::ModelMap) = θ::AbstractVector -> IsInDomain(M, θ)
IsInDomain(M::ModelMap, θ::AbstractVector) = _IsInDomain(InDomain(M), Domain(M), θ)
_IsInDomain(InDomain::Union{Nothing,Function}, Domain::Union{Nothing,Cuboid}, θ::AbstractVector) = (_TestInDomain(InDomain, θ) && _TestDomain(Domain, θ))

# Eval InDomain function
_TestInDomain(::Nothing, θ::AbstractVector) = true
_TestInDomain(InDomain::Function, θ::AbstractVector) = all(InDomain(θ) .≥ 0)
# Eval Domain HyperCube
_TestDomain(::Nothing, θ::AbstractVector) = true       # Excluded
_TestDomain(Domain::Cuboid, θ::AbstractVector) = θ ∈ Domain


MakeCustom(F::Function, Domain::Union{Bool,Nothing}=nothing; kwargs...) = MakeCustom(ModelMap(F); kwargs...)
MakeCustom(F::Function, Domain::Cuboid; kwargs...) = MakeCustom(ModelMap(F, Domain); kwargs...)
function MakeCustom(M::ModelMap; Meta=M.Meta, verbose::Bool=true)
    if iscustommodel(M)
        verbose && @warn "MakeCustom: Given Map already uses custom embedding."
        return remake(M; Meta)
    else
        return ModelMap(M.Map, InDomain(M), Domain(M), M.xyp, M.pnames, M.inplace, Val(true), name(M), Meta)
    end
end
function MakeNonCustom(M::ModelMap; Meta=M.Meta, verbose::Bool=true)
    if !iscustommodel(M)
        verbose && @warn "MakeNonCustom: Given Map already using non-custom embedding."
        return remake(M; Meta)
    else
        return ModelMap(M.Map, InDomain(M), Domain(M), M.xyp, M.pnames, M.inplace, Val(false), name(M), Meta)
    end
end


function ModelMap(F::Nothing, M::ModelMap)
    @warn "ModelMap: Got Nothing instead of Function to build new ModelMap."
    nothing
end

const subscriptnumberdict = Dict(string.(0:9) .=> ["₀","₁","₂","₃","₄","₅","₆","₇","₈","₉"])
function CreateSymbolNames(n::Int, base::AbstractString="θ")
    n == 1 && return [base]
    base .* [prod(get(subscriptnumberdict, string(x), "Q") for x in string(digit)) for digit in 1:n]
end

## Read names from ComponentArrays
function GetNamesSymb(@nospecialize p::ComponentArray)
    GetVal(X::Val{T}) where T = T
    GetVal.(collect(valkeys(p)))
end
GetNamesSymb(p::Array{<:Number}) = length(p) > 1 ? Symbol.(1:length(p)) : [Symbol("")]
# No error for ReshapedArray{...,...,SubArray}
GetNamesSymb(@nospecialize p::Base.ReshapedArray) = length(p) > 1 ? Symbol.(1:length(p)) : [Symbol("")]
GetNamesSymb(p::Base.SubArray{A, B, Vector{C}}) where {A,B,C} = length(p) > 1 ? Symbol.(1:length(p)) : [Symbol("")]
function GetNamesSymb(p::AbstractArray{<:Number})
    @warn "Do not know how to read parameter names of $(typeof(p)), treating as type 'Array'."
    GetNamesSymb(convert(Array,p))
end

# Always output String.
GetNamesRecursive(p) = string.(GetNamesSymb(p))
function GetNamesRecursive(@nospecialize p_NN::ComponentVector)
    S = GetNamesSymb(p_NN)
    InnerS = map(s->GetNamesRecursive(getproperty(p_NN,s)), S)
    OuterNames = string.(S)
    InnerNames = [string.(s) for s in InnerS]
    Res = String[]
    for i in eachindex(OuterNames)
        if length(InnerNames[i]) > 1
            for j in eachindex(InnerNames[i])
                push!(Res, OuterNames[i] * "_" * InnerNames[i][j])
                # push!(Res, OuterNames[i] * "(" * InnerNames[i][j] * ")")
            end
        else
            push!(Res, OuterNames[i] * InnerNames[i][1])
        end
    end;    Res
end

# Functions to be called for construction
GetParameterNames(p::ComponentVector) = GetNamesRecursive(p)
GetParameterNames(p::AbstractVector) = CreateSymbolNames(length(p), "θ")


function _FullNames(DM::AbstractDataModel)
    if xdim(DM) == 1
        [CreateSymbolNames(Npoints(DM), "x"); pnames(DM)]
    else
        xMat = Matrix{String}(undef, Npoints(DM), xdim(DM))
        for i in 1:xdim(DM)
            xMat[:,i] = CreateSymbolNames(Npoints(DM), "("* xnames(DM)[i] *")")
        end
        [reduce(vcat,collect(eachrow(xMat))); pnames(DM)]
    end
end

pdim(DS::AbstractDataSet, model::ModelMap)::Int = model.xyp[3]

xdim(M::ModelMap)::Int = M.xyp[1]
ydim(M::ModelMap)::Int = M.xyp[2]
pdim(M::ModelMap)::Int = M.xyp[3]

function ModelMappize(DM::AbstractDataModel; pnames::AbstractVector{<:AbstractString}=String[])
    NewMod = Predictor(DM) isa ModelMap ? Predictor(DM) : ModelMap(Predictor(DM), (xdim(DM), ydim(DM), pdim(DM)); pnames=pnames)
    NewdMod = dPredictor(DM) isa ModelMap ? dPredictor(DM) : ModelMap(dPredictor(DM), (xdim(DM), ydim(DM), pdim(DM)); pnames=pnames)
    DataModel(Data(DM), NewMod, NewdMod, MLE(DM), LogLikeMLE(DM), LogPrior(DM), true)
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
            if any(iscustommodel, Mods)
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
        return ModelMap(ConcatenatedModel, reduce(union, Domain.(Mods)), (Mods[1].xyp[1], sum((q->q.xyp[2]).(Mods)), Mods[1].xyp[3])) |> MakeCustom
    else
        function NConcatenatedModel(x::AbstractVector{<:Number}, θ::AbstractVector{<:Number}; kwargs...)
            map(model->model(x, θ; kwargs...), Mods) |> Reduction
        end
        function NConcatenatedModel(X::AbstractVector{<:AbstractVector{<:Number}}, θ::AbstractVector{<:Number}; kwargs...)
            if any(iscustommodel, Mods)
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
        return ModelMap(NConcatenatedModel, reduce(union, Domain.(Mods)), (Mods[1].xyp[1], sum((q->q.xyp[2]).(Mods)), Mods[1].xyp[3])) |> MakeCustom
    end
end


_Apply(x::AbstractVector, ComponentwiseF::Function, idxs::BoolVector) = (@assert length(x) == length(idxs); [@inbounds (idxs[i] ? ComponentwiseF(x[i]) : x[i]) for i in eachindex(idxs)])

MonotoneIncreasing(F::Function, Interval::Tuple{Number,Number}; kwargs...) = Monotonicity(F, Interval; kwargs...) === :increasing
MonotoneDecreasing(F::Function, Interval::Tuple{Number,Number}; kwargs...) = Monotonicity(F, Interval; kwargs...) === :decreasing
function Monotonicity(F::Function, Interval::Tuple{Number,Number}; length::Int=100)
    derivs = map(GetDeriv(Val(:ForwardDiff),F), range(Interval[1], Interval[2]; length=length))
    all(x-> x≥0., derivs) && return :increasing
    all(x-> x≤0., derivs) && return :decreasing
    :neither
end

# Check if anonymous function
GetFunctionName(F::Function, Fallback::AbstractString) = (S = string(nameof(F));    !contains(S, "#") ? S : Fallback)
GetTrafoName(F::Function) = GetFunctionName(F, "Trafo")

ComponentwiseModelTransform(model::Function, idxs::BoolVector, Transform::Function, InverseTransform::Function=x->invert(Transform,x); pnames::AbstractVector{<:AbstractString}=String[]) = _Transform(model, idxs, Transform, InverseTransform)

# Try to do a bit of inference for the new domain here!
# Aim for user convenience and generality rather than performance
function ComponentwiseModelTransform(M::ModelMap, idxs::BoolVector, Transform::Function, InverseTransform::Function=x->invert(Transform,x); 
                        InverseTransformName::AbstractString=GetTrafoName(InverseTransform),
                        pnames::AbstractVector{<:AbstractString}=_Apply(pnames(M), (p->"$InverseTransformName("*p*")"), idxs))
    @assert !isinplacemodel(M)
    TransformedDomain = InDomain(M) isa Function ? (θ::AbstractVector{<:Number} -> InDomain(M)(_Apply(θ, Transform, idxs))) : nothing
    mono = Monotonicity(Transform, (1e-12,50.))
    NewCube = if mono == :increasing
        HyperCube(_Apply(Domain(M).L, InverseTransform, idxs), _Apply(Domain(M).U, InverseTransform, idxs))
    elseif mono == :decreasing
        @warn "Detected monotone decreasing transformation."
        HyperCube(_Apply(Domain(M).U, InverseTransform, idxs), _Apply(Domain(M).L, InverseTransform, idxs))
    else
        @warn "Transformation does not appear to be monotone. Unable to infer new Domain."
        FullDomain(length(idxs))
    end
    ModelMap(_Transform(M.Map, idxs, Transform, InverseTransform), TransformedDomain, NewCube,
                        M.xyp, pnames, M.inplace, M.CustomEmbedding, name(M), M.Meta)
end
# function ComponentwiseModelTransform(M::ModelMap, Transform::Function, InverseTransform::Function=x->invert(Transform,x))
#     ComponentwiseModelTransform(M, trues(M.xyp[3]), Transform, InverseTransform)
# end
@deprecate Transform(M::ModelOrFunction, args...; kwargs...) ComponentwiseModelTransform(M, args...; kwargs...)


function _Transform(F::Function, idxs::BoolVector, Transform::Function, InverseTransform::Function)
    function TransformedModel(x::Union{Number, AbstractVector{<:Number}}, θ::AbstractVector{<:Number}; kwargs...)
        F(x, _Apply(θ, Transform, idxs); kwargs...)
    end
end


"""
    ComponentwiseModelTransform(DM::AbstractDataModel, F::Function, idxs=trues(pdim(DM))) -> DataModel
    ComponentwiseModelTransform(model::Function, idxs, F::Function) -> Function
Transforms the parameters of the model by the given scalar function `F` such that `newmodel(x, θ) = oldmodel(x, F.(θ))`.
By providing `idxs`, one may restrict the application of the function `F` to broadcast only to specific parameter components.

For vector-valued transformations, see [`ModelEmbedding`](@ref).
"""
function ComponentwiseModelTransform(DM::AbstractDataModel, F::Function, idxs::BoolVector=trues(pdim(DM)); kwargs...)
    ComponentwiseModelTransform(DM, F, x->invert(F,x), idxs; kwargs...)
end
function ComponentwiseModelTransform(DM::AbstractDataModel, F::Function, inverseF::Function, idxs::BoolVector=trues(pdim(DM)); kwargs...)
    @assert length(idxs) == pdim(DM)
    sum(idxs) == 0 && return DM
    DataModel(Data(DM), ComponentwiseModelTransform(Predictor(DM), idxs, F, inverseF), _Apply(MLE(DM), inverseF, idxs), EmbedLogPrior(DM, θ->_Apply(θ, F, idxs)); kwargs...)
end


LogTransform(M::ModelOrFunction, idxs::BoolVector=(M isa ModelMap ? trues(M.xyp[3]) : trues(GetArgSize(M)[2]))) = ComponentwiseModelTransform(M, idxs, log, exp)
LogTransform(DM::AbstractDataModel, idxs::BoolVector=trues(pdim(DM)); kwargs...) = ComponentwiseModelTransform(DM, log, exp, idxs; kwargs...)

ExpTransform(M::ModelOrFunction, idxs::BoolVector=(M isa ModelMap ? trues(M.xyp[3]) : trues(GetArgSize(M)[2]))) = ComponentwiseModelTransform(M, idxs, exp, log)
ExpTransform(DM::AbstractDataModel, idxs::BoolVector=trues(pdim(DM)); kwargs...) = ComponentwiseModelTransform(DM, exp, log, idxs; kwargs...)

Log10Transform(M::ModelOrFunction, idxs::BoolVector=(M isa ModelMap ? trues(M.xyp[3]) : trues(GetArgSize(M)[2]))) = ComponentwiseModelTransform(M, idxs, log10, exp10)
Log10Transform(DM::AbstractDataModel, idxs::BoolVector=trues(pdim(DM)); kwargs...) = ComponentwiseModelTransform(DM, log10, exp10, idxs; kwargs...)

Exp10Transform(M::ModelOrFunction, idxs::BoolVector=(M isa ModelMap ? trues(M.xyp[3]) : trues(GetArgSize(M)[2]))) = ComponentwiseModelTransform(M, idxs, exp10, log10)
Exp10Transform(DM::AbstractDataModel, idxs::BoolVector=trues(pdim(DM)); kwargs...) = ComponentwiseModelTransform(DM, exp10, log10, idxs; kwargs...)

@deprecate Power10Transform(args...; kwargs...) Exp10Transform(args...; kwargs...)

ReflectionTransform(M::ModelOrFunction, idxs::BoolVector=(M isa ModelMap ? trues(M.xyp[3]) : trues(GetArgSize(M)[2]))) = ComponentwiseModelTransform(M, idxs, x-> -x, x-> -x)
ReflectionTransform(DM::AbstractDataModel, idxs::BoolVector=trues(pdim(DM)); kwargs...) = ComponentwiseModelTransform(DM, x-> -x, x-> -x, idxs; kwargs...)

ScaleTransform(M::ModelOrFunction, factor::Number, idxs::BoolVector=(M isa ModelMap ? trues(M.xyp[3]) : trues(GetArgSize(M)[2]))) = ComponentwiseModelTransform(M, idxs, x->factor*x, x->x/factor)
ScaleTransform(DM::AbstractDataModel, factor::Number, idxs::BoolVector=trues(pdim(DM)); kwargs...) = ComponentwiseModelTransform(DM, x->factor*x, x->x/factor, idxs; kwargs...)


function TranslationTransform(F::Union{ModelOrFunction, AbstractDataModel}, v::AbstractVector{T}; kwargs...) where T<:Number
    AffineTransform(F, Diagonal(ones(T,length(v))), v; kwargs...)
end
function LinearTransform(F::Union{ModelOrFunction, AbstractDataModel}, A::AbstractMatrix{T}; kwargs...) where T<:Number
    AffineTransform(F, A, zeros(T, size(A,1)); kwargs...)
end

function AffineTransform(F::Function, A::AbstractMatrix{<:Number}, v::AbstractVector{<:Number}; Domain::Union{HyperCube,Nothing}=nothing)
    @assert size(A,1) == size(A,2) == length(v)
    TranslatedModel(x, θ::AbstractVector{<:Number}; Kwargs...) = F(x, muladd(A,θ,v); Kwargs...)
end
function AffineTransform(M::ModelMap, A::AbstractMatrix{<:Number}, v::AbstractVector{<:Number}; Domain::Union{HyperCube,Nothing}=Domain(M))
    @assert isnothing(Domain) || (length(Domain) == size(A,1) == size(A,2) == length(v))
    Ainv = pinv(A)
    NewDomain = isnothing(Domain) ? HyperCube(Ainv*(Domain.L-v), Ainv*(Domain.U-v)) : nothing
    ModelMap(AffineTransform(M.Map, A, v), (!isnothing(InDomain(M)) ? (InDomain(M)∘(θ->muladd(A,θ,v))) : nothing), NewDomain,
                    M.xyp, M.pnames, M.inplace, M.CustomEmbedding, name(M), M.Meta)
end
function AffineTransform(DM::AbstractDataModel, A::AbstractMatrix{<:Number}, v::AbstractVector{<:Number}; kwargs...)
    @assert pdim(DM) == size(A,1) == size(A,2) == length(v)
    Ainv = pinv(A)
    DataModel(Data(DM), AffineTransform(Predictor(DM), A, v; kwargs...), Ainv*(MLE(DM)-v), EmbedLogPrior(DM, θ->muladd(A,θ,v)))
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

# Unlike "ComponentwiseModelTransform" EmbedModelVia should be mainly performant for use e.g. in ProfileLikelihood
# Also only vector-valued transformations
"""
    EmbedModelVia(model, F::Function; Domain::HyperCube=FullDomain(GetArgLength(F))) -> Union{Function,ModelMap}
Transforms a model function via `newmodel(x, θ) = oldmodel(x, F(θ))`.
A `Domain` for the new model can optionally be specified for `ModelMap`s.
"""
EmbedModelVia(model::Function, F::Function; Kwargs...) = EmbeddedModel(x, θ; kwargs...) = model(x, F(θ); Kwargs..., kwargs...)
EmbedModelVia_inplace(model!::Function, F::Function; Kwargs...) = EmbeddedModel!(y, x, θ; kwargs...) = model!(y, x, F(θ); Kwargs..., kwargs...)

function EmbedModelVia(M::ModelMap, F::Function; Domain::Union{Nothing,HyperCube}=nothing, pnames::Union{Nothing,AbstractVector{<:AbstractString}}=nothing, name::Union{<:AbstractString,Symbol}=name(M), Meta=M.Meta, kwargs...)
    if isnothing(Domain)
        Domain = FullDomain(GetArgLength(F))
        @warn "Cannot infer new Domain HyperCube for general embeddings, using $Domain."
    end
    Pnames = isnothing(pnames) ? CreateSymbolNames(length(Domain), "θ") : pnames
    ModelMap((isinplacemodel(M) ? EmbedModelVia_inplace : EmbedModelVia)(M.Map, F; kwargs...), (InDomain(M) isa Function ? (InDomain(M)∘F) : nothing),
            Domain, (M.xyp[1], M.xyp[2], length(Domain)), Pnames,
            M.inplace, M.CustomEmbedding, name, Meta)
end

function EmbedDModelVia(dmodel::Function, F::Function, Size::Tuple=Tuple([]); ADmode::Union{Symbol,Val}=Val(:ForwardDiff), Kwargs...)
    Jac = GetJac(ADmode, F)
    EmbeddedJacobian(x, θ; kwargs...) = dmodel(x, F(θ); Kwargs..., kwargs...) * Jac(θ)
end
function EmbedDModelVia_inplace(dmodel!::Function, F::Function, Size::Tuple{Int,Int}; ADmode::Union{Symbol,Val}=Val(:ForwardDiff), Kwargs...)
    Jac = GetJac(ADmode, F)
    function EmbeddedJacobian!(y, x, θ::AbstractVector{T}; kwargs...) where T<:Number
        Ycache = Matrix{T}(undef, Size)
        dmodel!(Ycache, x, F(θ); Kwargs..., kwargs...)
        mul!(y, Ycache, Jac(θ))
    end
end
function EmbedDModelVia(dM::ModelMap, F::Function; Domain::HyperCube=FullDomain(GetArgLength(F)), pnames::Union{Nothing,AbstractVector{<:AbstractString}}=nothing, name::Union{<:AbstractString,Symbol}=name(dM), Meta=dM.Meta, kwargs...)
    # Pass the OLD pdim to EmbedDModelVia_inplace for cache
    Pnames = isnothing(pnames) ? CreateSymbolNames(length(Domain), "θ") : pnames
    ModelMap((isinplacemodel(dM) ? EmbedDModelVia_inplace : EmbedDModelVia)(dM.Map, F, dM.xyp[2:3]; kwargs...), (!isnothing(InDomain(dM)) ? InDomain(dM)∘F : nothing),
            Domain, (dM.xyp[1], dM.xyp[2], length(Domain)), Pnames,
            dM.inplace, dM.CustomEmbedding, name, Meta)
end

"""
    ModelEmbedding(DM::AbstractDataModel, F::Function, start::AbstractVector; Domain::HyperCube=FullDomain(length(start))) -> DataModel
Transforms a model function via `newmodel(x, θ) = oldmodel(x, F(θ))` and returns the associated `DataModel`.
An initial parameter configuration `start` as well as a `Domain` can optionally be passed to the `DataModel` constructor.

For component-wise transformations see [`ComponentwiseModelTransform`](@ref).
"""
function ModelEmbedding(DM::AbstractDataModel, F::Function, start::AbstractVector{<:Number}=GetStartP(GetArgLength(F)); Domain::HyperCube=FullDomain(length(start)), kwargs...)
    DataModel(Data(DM), EmbedModelVia(Predictor(DM), F; Domain=Domain), EmbedDModelVia(dPredictor(DM), F; Domain=Domain), start, EmbedLogPrior(DM, F); kwargs...)
end
@deprecate Embedding ModelEmbedding

## Transform dependent and independent variables of model

# in-place
EmbedModelXin(model::Function, Emb::Function) = XEmbeddedModel(y, x, θ::AbstractVector; kwargs...) = model(y, Emb(x), θ; kwargs...)
# out-of-place
EmbedModelXout(model::Function, Emb::Function) = XEmbeddedModel(x, θ::AbstractVector; kwargs...) = model(Emb(x), θ; kwargs...)

"""
    EmbedModelX(model::Function, Emb::Function)
Embeds the independent variables of a model function via `newmodel(x,θ) = oldmodel(Emb(x),θ)`.
"""
EmbedModelX(M::ModelMap, Emb::Function, Inplace::Bool=isinplacemodel(M)) = ModelMap((Inplace ? EmbedModelXin : EmbedModelXout)(M.Map, Emb), M)
EmbedModelX(model::Function, Emb::Function, Inplace::Bool=isinplacemodel(model)) = (Inplace ? EmbedModelXin : EmbedModelXout)(model, Emb)

"""
    TransformXdata(DM::AbstractDataModel, Emb::Function, iEmb::Function, TransformName::String="Trafo") -> AbstractDataModel
    TransformXdata(DS::AbstractDataSet, Emb::Function, TransformName::String="Trafo") -> AbstractDataSet
Returns a modified `DataModel` where the x-variables have been transformed by a multivariable transform `Emb` both in the data as well as for the model via `newmodel(x,θ) = oldmodel(Emb(x),θ)`.
`iEmb` denotes the inverse of `Emb`.
The uncertainties are computed via linearized error propagation through the given transformation.
"""
function TransformXdata(DM::AbstractDataModel, Emb::Function, iEmb::Function, TransformName::AbstractString=GetTrafoName(Emb); kwargs...)
    @assert all(WoundX(DM) .≈ map(iEmb∘Emb, WoundX(DM))) # Check iEmb is correct inverse
    DataModel(TransformXdata(Data(DM), Emb, TransformName; kwargs...), EmbedModelX(Predictor(DM), iEmb), EmbedModelX(dPredictor(DM), iEmb), MLE(DM), LogPrior(DM))
end
function TransformXdata(DS::AbstractFixedUncertaintyDataSet, Emb::Function, TransformName::AbstractString=GetTrafoName(Emb); xnames=TransformName*"(".*xnames(DS).*")", ADmode::Union{Val,Symbol}=Val(:ForwardDiff))
    NewX = Reduction(map(Emb, WoundX(DS)))
    if !HasXerror(DS)
        DataSetType(DS)(NewX, ydata(DS), ysigma(DS), dims(DS); xnames=xnames, ynames=ynames(DS), name=name(DS))
    else
        EmbJac = xdim(DS) > 1 ? GetJac(ADmode, Emb, xdim(DS)) : GetDeriv(ADmode, Emb)
        NewXsigma = if xsigma(DS) isa AbstractVector
            map((xdat, xsig)->EmbJac(xdat)*xsig, WoundX(DS), Windup(xsigma(DS), xdim(DS))) # |> Reduction
        else
            J = reduce(BlockMatrix, map(EmbJac, WoundX(DS)))
            J * xsigma(DS) * transpose(J)
        end
        DataSetType(DS)(NewX, NewXsigma, ydata(DS), ysigma(DS), dims(DS); xnames=xnames, ynames=ynames(DS), name=name(DS))
    end
end
# Drop iEmb
TransformXdata(DS::AbstractDataSet, Emb::Function, iEmb::Function, args...; kwargs...) = TransformXdata(DS, Emb, args...; kwargs...)


for (Name, F, Finv, TrafoName) in [(:LogXdata, :log, :exp, :log),
                                (:Log10Xdata, :log10, :exp10, :log10),
                                (:ExpXdata, :exp, :log, :exp),
                                (:Exp10Xdata, :exp10, :log10, :exp10),
                                (:SqrtXdata, :sqrt, :(x->x^2), :sqrt)]
    @eval begin
        """
            $($Name)(DM::AbstractDataModel) -> AbstractDataModel
            $($Name)(DS::AbstractDataSet) -> AbstractDataSet
        Returns a modified `DataModel` or dataset object where $($TrafoName) has been applied component-wise to the x-variables both in the data as well as for the model.
        The uncertainties are computed via linearized error propagation through the given transformation.
        """
        $Name(DM::Union{AbstractDataModel,AbstractDataSet}; kwargs...) = TransformXdata(DM, x->broadcast($F,x), x->broadcast($Finv,x), "$($TrafoName)"; kwargs...)
        """
            $($Name)(DM::AbstractDataModel, idxs::BoolVector) -> AbstractDataModel
            $($Name)(DS::AbstractDataSet, idxs::BoolVector) -> AbstractDataSet
        Returns a modified `DataModel` or dataset object where $($TrafoName) has been applied to the components `i` of the x-variables for which `idxs[i]==true` both in the data as well as for the model.
        The uncertainties are computed via linearized error propagation through the given transformation.
        """
        function $Name(DM::Union{AbstractDataModel,AbstractDataSet}, idxs::BoolVector; kwargs...)
            @assert length(idxs) == xdim(DM) && xdim(DM) > 1
            TransformXdata(DM, x->_Apply(x, $F, idxs), x->_Apply(x, $Finv, idxs); xnames=_Apply(xnames(DM), (x->"$($TrafoName)("*x*")"), idxs), kwargs...)
        end
        export $Name
    end
end


# in-place
EmbedModelYin(model::Function, Emb::Function) = YEmbeddedModel(y, x, θ::AbstractVector; kwargs...) = (model(y, x, θ; kwargs...);   y=Emb(y))
# out-of-place
EmbedModelYout(model::Function, Emb::Function) = YEmbeddedModel(x, θ::AbstractVector; kwargs...) = Emb(model(x, θ; kwargs...))

"""
    EmbedModelY(model::Function, Emb::Function)
Embeds the independent variables of a model function via `newmodel(x,θ) = Emb(oldmodel(x,θ))`.
"""
EmbedModelY(M::ModelMap, Emb::Function, Inplace::Bool=isinplacemodel(M)) = ModelMap((Inplace ? EmbedModelYin : EmbedModelYout)(M.Map, Emb), M)
EmbedModelY(model::Function, Emb::Function, Inplace::Bool=isinplacemodel(model)) = (Inplace ? EmbedModelYin : EmbedModelYout)(model, Emb)


# Unlike X-transform, model uses same embedding function for Y instead of inverse to compensate
"""
    TransformYdata(DM::AbstractDataModel, Emb::Function, TransformName::String="Trafo") -> AbstractDataModel
    TransformYdata(DS::AbstractDataSet, Emb::Function, TransformName::String="Trafo") -> AbstractDataSet
Returns a modified `DataModel` where the y-variables have been transformed by a multivariable transform `Emb` both in the data as well as for the model via `newmodel(x,θ) = Emb(oldmodel(x,θ))`.
The uncertainties are computed via linearized error propagation through the given transformation.
"""
function TransformYdata(DM::AbstractDataModel, Emb::Function, TransformName::AbstractString=GetTrafoName(Emb); kwargs...)
    DataModel(TransformYdata(Data(DM), Emb, TransformName; kwargs...), EmbedModelY(Predictor(DM), Emb), MLE(DM), LogPrior(DM))
end
function TransformYdata(DS::AbstractFixedUncertaintyDataSet, Emb::Function, TransformName::AbstractString=GetTrafoName(Emb); ynames=TransformName*"(".*ynames(DS).*")", ADmode::Union{Val,Symbol}=Val(:ForwardDiff))
    NewY = Reduction(map(Emb, WoundY(DS)));    EmbJac = ydim(DS) > 1 ? GetJac(ADmode, Emb, ydim(DS)) : GetDeriv(ADmode, Emb)
    NewYsigma = if ysigma(DS) isa AbstractVector
        map((ydat, ysig)->EmbJac(ydat)*ysig, WoundY(DS), Windup(ysigma(DS), ydim(DS))) # |> Reduction
    else
        J = reduce(BlockMatrix, map(EmbJac, WoundY(DS)))
        J * ysigma(DS) * transpose(J)
    end
    if !HasXerror(DS)
        DataSetType(DS)(xdata(DS), NewY, NewYsigma, dims(DS); xnames=xnames(DS), ynames=ynames, name=name(DS))
    else
        DataSetType(DS)(xdata(DS), xsigma(DS), NewY, NewYsigma, dims(DS); xnames=xnames(DS), ynames=ynames, name=name(DS))
    end
end


for (Name, F, TrafoName) in [(:LogYdata, :log, :log),
                                (:Log10Ydata, :log10, :log10),
                                (:ExpYdata, :exp, :exp),
                                (:Exp10Ydata, :exp10, :exp10),
                                (:SqrtYdata, :sqrt, :sqrt)]
    @eval begin
        """
            $($Name)(DM::AbstractDataModel) -> AbstractDataModel
            $($Name)(DS::AbstractDataSet) -> AbstractDataSet
        Returns a modified `DataModel` or dataset object where $($TrafoName) has been applied component-wise to the y-variables both in the data as well as for the model.
        The uncertainties are computed via linearized error propagation through the given transformation.
        """
        $Name(DM::Union{AbstractDataModel,AbstractDataSet}; kwargs...) = TransformYdata(DM, y->broadcast($F,y), "$($TrafoName)"; kwargs...)
        """
            $($Name)(DM::AbstractDataModel, idxs::BoolVector) -> AbstractDataModel
            $($Name)(DS::AbstractDataSet, idxs::BoolVector) -> AbstractDataSet
        Returns a modified `DataModel` or dataset object where $($TrafoName) has been applied to the components `i` of the y-variables for which `idxs[i]==true` both in the data as well as for the model.
        The uncertainties are computed via linearized error propagation through the given transformation.
        """
        function $Name(DM::Union{AbstractDataModel,AbstractDataSet}, idxs::BoolVector; kwargs...)
            @assert length(idxs) == ydim(DM) && ydim(DM) > 1
            TransformYdata(DM, y->_Apply(y, $F, idxs); ynames=_Apply(ynames(DM), (y->"$($TrafoName)("*y*")"), idxs), kwargs...)
        end
        export $Name
    end
end


# in-place
EmbedModelXPin(model::Function, Emb::Function) = XPEmbeddedModel(y, x, θ::AbstractVector; kwargs...) = ((X,P)=Emb(x,θ);    model(y, X, P; kwargs...))
# out-of-place
EmbedModelXPout(model::Function, Emb::Function) = XPEmbeddedModel(x, θ::AbstractVector; kwargs...) = ((X,P)=Emb(x,θ);    model(X, P; kwargs...))

"""
    EmbedModelXP(model::Function, Emb::Function)
Embeds the independent variables of a model function via `newmodel(x,θ) = oldmodel(Emb(x,θ)...)`.
"""
EmbedModelXP(M::ModelMap, Emb::Function, Inplace::Bool=isinplacemodel(M)) = ModelMap((Inplace ? EmbedModelXPin : EmbedModelXPout)(M.Map, Emb), M)
EmbedModelXP(model::Function, Emb::Function, Inplace::Bool=isinplacemodel(model)) = (Inplace ? EmbedModelXPin : EmbedModelXPout)(model, Emb)



# Parameter transforms
ExpTransform(Sys::ODESystem, idxs::AbstractVector{<:Bool}=trues(length(parameters(Sys))); kwargs...) = SystemTransform(Sys, exp, idxs; kwargs...)
LogTransform(Sys::ODESystem, idxs::AbstractVector{<:Bool}=trues(length(parameters(Sys))); kwargs...) = SystemTransform(Sys, log, idxs; kwargs...)
Exp10Transform(Sys::ODESystem, idxs::AbstractVector{<:Bool}=trues(length(parameters(Sys))); kwargs...) = SystemTransform(Sys, exp10, idxs; kwargs...)
Log10Transform(Sys::ODESystem, idxs::AbstractVector{<:Bool}=trues(length(parameters(Sys))); kwargs...) = SystemTransform(Sys, log10, idxs; kwargs...)

"""
    SystemTransform(Sys::ODESystem, F::Function, idxs::AbstractVector{<:Bool}=trues(length(parameters(Sys)))) -> ODESystem
Transforms the parameters of an `ODESystem` according to a component-wise function `F`.
"""
function SystemTransform(Sys::AbstractODESystem, F::Function, idxs::AbstractVector{<:Bool}=trues(length(parameters(Sys))))
    SubstDict = Dict(parameters(Sys) .=> [(idxs[i] ? F(x) : x) for (i,x) in enumerate(parameters(Sys))])
    NewEqs = [(equations(Sys)[i].lhs ~ substitute(equations(Sys)[i].rhs, SubstDict)) for i in eachindex(equations(Sys))]
    # renamed "states" to "unknowns": https://github.com/SciML/ModelingToolkit.jl/pull/2432
    ODESystem(NewEqs, independent_variables(Sys)[1], try ModelingToolkit.unknowns(Sys) catch; ModelingToolkit.states(Sys) end, ModelingToolkit.parameters(Sys); name=nameof(Sys))
end



IsDEbased(F::Function) = occursin("DEmodel", string(nameof(typeof(F))))
IsDEbased(F::ModelMap) = IsDEbased(F.Map)
IsDEbased(DM::AbstractDataModel) = IsDEbased(Predictor(DM))

# Convert Vector to ComponentVector if required to retain type-stability
function GetComponentVectorEmbedding(P::ComponentVector)
    Ax = typeof(getaxes(P))
    ConvertToComponentVector(X::ComponentVector{T}) where T<:Number = X
    ConvertToComponentVector(X::AbstractVector{T}) where T<:Number  = convert(ComponentVector{T, Vector{T}, Ax}, X)
end
