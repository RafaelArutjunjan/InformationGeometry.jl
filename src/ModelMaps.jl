

function _TestOut(model::Function, startp::AbstractVector, xlen::Int; inplace::Bool=isinplacemodel(model), max::Int=100)
    if !inplace
        model((xlen < 2 ? rand() : rand(xlen)), startp)
    else
        Res = Fill(-Inf, max)
        ind = try
            model(Res, (xlen < 2 ? rand() : rand(xlen)), startp)
            findfirst(isinf, Res)
        catch E;
            @warn "Got error $E during attempted evaluation of model."
            ResInd = nothing
            for i in 1:max
                try (model(Res[1:i], (xlen < 2 ? rand() : rand(xlen)), startp);  ResInd=i) catch; end
            end
            if isnothing(ResInd)
                # model output scalar?
                try
                    res = zero(eltype(startp))
                    model(res, (xlen < 2 ? rand() : rand(xlen)), startp)
                    return res
                catch E2; @warn "Got error $E2 in attempt to get scalar output." end
            end; ResInd
        end
        isnothing(ind) && throw("Could not determine model output size. Consider providing tuple (xdim, ydim, pdim).")
        Res[1:(ind-1)]
    end
end
function CheckIfIsCustom(model::Function, startp::AbstractVector, xyp::Tuple, IsInplace::Bool)
    woundX = Windup(collect(1:2xyp[1]) .+ 0.1 .* rand(2xyp[1]), xyp[1])
    if !IsInplace
        try   length(model(woundX, startp)) == 2xyp[2]   catch;  false   end
    else
        Res = Fill(-Inf, 2xyp[2])
        try   model(Res, woundX, startp);   !any(isinf, Res)    catch; false  end
    end
end

function ConstructModelxyp(model::Function; max::Int=MaxArgLen)
    xp = GetArgSize(model; max)
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
"""
struct ModelMap{Inplace, Custom}
    Map::Function
    InDomain::Union{Nothing,Function}
    Domain::Cuboid
    xyp::Tuple{Int,Int,Int}
    pnames::AbstractVector{Symbol}
    inplace::Val
    CustomEmbedding::Val
    name::Symbol
    Meta
    SymbolicCache
    # Given: Bool-valued domain function
    ModelMap(model::Function, InDomain::Function, xyp::Tuple{Int,Int,Int}; kwargs...) = ModelMap(model, InDomain, nothing, xyp; kwargs...)
    # Given: HyperCube
    function ModelMap(model::Function, Domain::Cuboid, xyp::Union{Tuple{Int,Int,Int},Bool}=false; kwargs...)
        xyp isa Bool ? ModelMap(model, nothing, Domain; kwargs...) : ModelMap(model, nothing, Domain, xyp; kwargs...)
    end
    # Given: xyp
    ModelMap(model::Function, xyp::Tuple{Int,Int,Int}; kwargs...) = ModelMap(model, nothing, nothing, xyp; kwargs...)
    # Given: Function only (potentially) -> Find xyp
    function ModelMap(model::Function, InDomain::Union{Nothing,Function}=nothing, Domain::Union{Cuboid,Nothing}=nothing; 
                            startp::AbstractVector{<:Number}=isnothing(Domain) ? GetStartP(GetArgSize(model)[2]) : ElaborateGetStartP(Domain, InDomain), kwargs...)
        ModelMap(model, startp, InDomain, Domain; kwargs...)
    end
    function ModelMap(model::Function, startp::AbstractVector{<:Number}, InDomain::Union{Nothing,Function}=nothing, Domain::Union{Cuboid,Nothing}=nothing; inplace::Bool=isinplacemodel(model), kwargs...)
        xlen = inplace ? GetArgLength((Res,x)->model(Res,x,startp); max=MaxArgLen) : GetArgLength(x->model(x,startp); max=MaxArgLen)
        testout = _TestOut(model, startp, xlen; inplace)
        ModelMap(model, InDomain, Domain, (xlen, size(testout,1), length(startp)); inplace, startp=startp, kwargs...)
    end
    function ModelMap(model::Function, InDomain::Union{Nothing,Function}, Domain::Union{Cuboid,Nothing}, xyp::Tuple{Int,Int,Int}; name::StringOrSymb=Symbol(), Meta=nothing, 
                            startp::AbstractVector{<:Number}=isnothing(Domain) ? GetStartP(xyp[3]) : ElaborateGetStartP(Domain, InDomain), pnames::AbstractVector{<:StringOrSymb}=GetParameterNames(startp),
                            inplace::Bool=isinplacemodel(model), IsCustom::Bool=CheckIfIsCustom(model, startp, xyp, inplace), TrySymbolic::Bool=true, SymbolicCache=(TrySymbolic ? ToExpr(model, xyp) : nothing), kwargs...)
        ModelMap(model, InDomain, Domain, xyp, pnames, Val(inplace), Val(IsCustom), name, Meta, SymbolicCache; kwargs...)
    end
    "Construct new ModelMap from function `F` with data from `M`."
    # ModelMap(F::Function, M::ModelMap; inplace::Bool=isinplacemodel(M)) = ModelMap(F, InDomain(M), Domain(M), M.xyp, M.pnames, Val(inplace), M.CustomEmbedding, name(M), M.Meta)
    ModelMap(F::Function, M::ModelMap; inplace::Bool=isinplacemodel(M), IsCustom::Bool=iscustommodel(M), kwargs...) = remake(M; Map=F, inplace=Val(inplace), CustomEmbedding=Val(IsCustom), kwargs...)

    function ModelMap(Map::Function, InDomain::Union{Nothing,Function}, Domain::Union{Cuboid,Nothing}, xyp::Tuple{Int,Int,Int},
                        pnames::AbstractVector{<:StringOrSymb}, inplace::Val, CustomEmbedding::Val, name::StringOrSymb=Symbol(), Meta=nothing, symbolicCache=nothing; SymbolicCache=symbolicCache, SkipTests::Bool=false)
        if !SkipTests
            @assert allunique(pnames) "Parameter names must be unique within a model, got $pnames."
            isnothing(Domain) ? (Domain = FullDomain(xyp[3], Inf)) : (@assert length(Domain) == xyp[3] "Given Domain Hypercube $Domain does not fit inferred number of parameters $(xyp[3]).")
            InDomain isa Function && (@assert InDomain(Center(Domain)) isa Number "InDomain function must yield a scalar value, got $(typeof(InDomain(Center(Domain)))) at $(Center(Domain)).")
        end
        new{ValToBool(inplace), ValToBool(CustomEmbedding)}(Map, InDomain, Domain, xyp, Symbol.(pnames), inplace, CustomEmbedding, Symbol(name), Meta, SymbolicCache)
    end
end
(M::ModelMap{false})(x, θ::AbstractVector{<:Number}; kwargs...) = M.Map(x, θ; kwargs...)
(M::ModelMap{true})(y, x, θ::AbstractVector{<:Number}; kwargs...) = M.Map(y, x, θ; kwargs...)

(M::ModelMap{false})(y, x, θ::AbstractVector{<:Number}; kwargs...) = copyto!(y, M.Map(x, θ; kwargs...))
function (M::ModelMap{true})(x, θ::AbstractVector{T}; kwargs...) where T<:Number
    try
        Res=Vector{T}(undef, M.xyp[2]);   M.Map(Res, x, θ; kwargs...);    Res
    catch
        Res=Matrix{T}(undef, M.xyp[2], M.xyp[3]);   M.Map(Res, x, θ; kwargs...);    Res
    end
end
const ModelOrFunction = Union{Function,ModelMap}


# For SciMLBase.remake
ModelMap(;
Map::Function=x->Inf,
InDomain::Union{Nothing,Function}=nothing,
Domain::Union{Cuboid,Nothing}=nothing,
xyp::Tuple{Int,Int,Int}=(1,1,1),
pnames::AbstractVector{<:StringOrSymb}=[:θ],
inplace::Val=Val(true),
CustomEmbedding::Val=Val(true),
name::Symbol=Symbol(),
SymbolicCache=nothing,
Meta=nothing, kwargs...) = ModelMap(Map, InDomain, Domain, xyp, pnames, inplace, CustomEmbedding, name, Meta, SymbolicCache; kwargs...)



function InformNames(M::ModelMap, pnames::AbstractVector{<:StringOrSymb})
    @assert length(pnames) == M.xyp[3]
    ModelMap(M.Map, InDomain(M), Domain(M), M.xyp, Symbol.(pnames), M.inplace, M.CustomEmbedding, name(M), M.Meta)
end


pnames(M::ModelMap) = M.pnames .|> string
Pnames(M::ModelMap) = M.pnames
name(M::ModelMap) = M.name
name(F::Function) = Symbol()
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
CreateSymbolNames(n::Int, base::Symbol) = CreateSymbolNames(n, string(base)) .|> Symbol

## Read names from ComponentArrays
function GetNamesSymb(@nospecialize p::ComponentArray)
    GetVal(X::Val{T}) where T = T
    GetVal.(collect(valkeys(p)))
end
GetNamesSymb(p::Array{<:Number}) = length(p) > 1 ? Symbol.(1:length(p)) : [Symbol()]
# No error for ReshapedArray{...,...,SubArray}
GetNamesSymb(@nospecialize p::Base.ReshapedArray) = length(p) > 1 ? Symbol.(1:length(p)) : [Symbol()]
GetNamesSymb(p::Base.SubArray{A, B, Vector{C}}) where {A,B,C} = length(p) > 1 ? Symbol.(1:length(p)) : [Symbol()]
function GetNamesSymb(p::AbstractArray{<:Number})
    @warn "Do not know how to read parameter names of $(typeof(p)), treating as type 'Array'."
    GetNamesSymb(convert(Array,p))
end

# Always output String.
GetNamesRecursive(p) = string.(GetNamesSymb(p))
function GetNamesRecursive(@nospecialize p_NN::ComponentVector)
    S = GetNamesSymb(p_NN)
    OuterNames = string.(S)
    length(OuterNames) == length(p_NN) && return OuterNames
    InnerS = map(s->GetNamesRecursive(getproperty(p_NN,s)), S)
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


# Includes names of estimated x-parameters
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


xdim(M::ModelMap)::Int = M.xyp[1]
ydim(M::ModelMap)::Int = M.xyp[2]
pdim(M::ModelMap)::Int = M.xyp[3]

function ModelMappize(DM::AbstractDataModel; pnames::AbstractVector{<:StringOrSymb}=Symbol[])
    NewMod = Predictor(DM) isa ModelMap ? Predictor(DM) : ModelMap(Predictor(DM), (xdim(DM), ydim(DM), pdim(DM)); pnames=pnames)
    NewdMod = dPredictor(DM) isa ModelMap ? dPredictor(DM) : ModelMap(dPredictor(DM), (xdim(DM), ydim(DM), pdim(DM)); pnames=pnames)
    DataModel(Data(DM), NewMod, NewdMod, MLE(DM), LogLikeMLE(DM), LogPrior(DM), true)
end


## If ModelMap domain does not include error parameters from AbstractUnknownUncertaintyDataSet yet, try appending this at the end
function FixModelMapDomain(DS::AbstractUnknownUncertaintyDataSet, M::ModelMap; 
                            pnames::AbstractVector{<:StringOrSymb}=vcat(Symbol.(CreateSymbolNames(xpars(DS),"x")), Pnames(M), Symbol.(CreateSymbolNames(errormoddim(DS),"σ"))), 
                            σDomain::HyperCube=FullDomain(errormoddim(DS), 5),
                            Domain::HyperCube=xpars(DS) > 0 ? vcat(FullDomain(xpars(DS)), Domain(M), σDomain) : vcat(Domain(M), σDomain), kwargs...)
    Xyp = (xdim(M), ydim(M), length(Domain))
    model = remake(M; pnames=Symbol.(pnames), xyp=Xyp, Domain=Domain, kwargs...)
end


"""
Only works for `DataSet` and `DataSetExact` but will output wrong order of components for `CompositeDataSet`!
"""
# ConcatenateModels(Mods::Vararg{ModelMap}) = ConcatenateModels([X for X in Mods])
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

function SubComponentModel(M::ModelOrFunction, idxs::Union{Int, AbstractVector{<:Int}}, xdim::Int, ydim::Int)
    @assert all(1 .≤ idxs .≤ ydim) && allunique(idxs)
    V = ViewElements(idxs)
    SubModel(x::AbstractVector, p::AbstractVector{<:Number}) = (keep=repeat([j ∈ idxs for j in 1:ydim], length(x));   view(EmbeddingMap(Val(true), M, p, x), keep))
    SubModel(y, x, p) = copyto!(y, SubModel(x,p))
    SubModelOneDim(x::Number, p::AbstractVector{<:Number}) = V(M(x,p))
    SubModelOneDim(x, p) = SubModel(x, p);      SubModelOneDim(y, x, p) = SubModel(y, x, p)
    xdim == 1 ? SubModelOneDim : SubModel
end
function SubComponentDModel(dM::ModelOrFunction, idxs::Union{Int, AbstractVector{<:Int}}, xdim::Int, ydim::Int)
    @assert all(1 .≤ idxs .≤ ydim) && allunique(idxs)
    SubdModel(x::AbstractVector, p::AbstractVector{<:Number}) = (keep=repeat([j ∈ idxs for j in 1:ydim], length(x));   view(EmbeddingMatrix(Val(true), dM, p, x), keep, :))
    SubdModel(y, x, p) = copyto!(y, SubdModel(x,p))
    SubdModelOneDim(x::Number, p::AbstractVector{<:Number}) = view(dM(x,p), idxs, :)
    SubdModelOneDim(x, p) = SubdModel(x, p);      SubdModelOneDim(y, x, p) = SubdModel(y, x, p)
    xdim == 1 ? SubdModelOneDim : SubdModel
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

ComponentwiseModelTransform(model::Function, idxs::BoolVector, Transform::Function, InverseTransform::Function=x->invert(Transform,x); pnames::AbstractVector{<:StringOrSymb}=String[]) = _Transform(model, idxs, Transform, InverseTransform)

# Try to do a bit of inference for the new domain here!
# Aim for user convenience and generality rather than performance
function ComponentwiseModelTransform(M::ModelMap, idxs::BoolVector, Transform::Function, InverseTransform::Function=x->invert(Transform,x); 
                        InverseTransformName::AbstractString=GetTrafoName(InverseTransform),
                        pnames::AbstractVector{<:StringOrSymb}=_Apply(pnames(M), (p->"$InverseTransformName("*p*")"), idxs), kwargs...)
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
        FullDomain(length(idxs), Inf)
    end
    ModelMap(_Transform(M.Map, idxs, Transform, InverseTransform), TransformedDomain, NewCube,
                        M.xyp, pnames, M.inplace, M.CustomEmbedding, name(M), M.Meta; kwargs...)
end
# function ComponentwiseModelTransform(M::ModelMap, Transform::Function, InverseTransform::Function=x->invert(Transform,x))
#     ComponentwiseModelTransform(M, trues(M.xyp[3]), Transform, InverseTransform)
# end


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
function ComponentwiseModelTransform(DM::AbstractDataModel, F::Function, inverseF::Function, idxs::BoolVector=trues(pdim(DM)); SkipOptim::Bool=true, SkipTests::Bool=true, kwargs...)
    @assert length(idxs) == pdim(DM) || HasEstimatedUncertainties(DM) # Error parameters not forwarded to model map anyway
    sum(idxs) == 0 && return DM
    DataModel(Data(DM), ComponentwiseModelTransform(Predictor(DM), idxs, F, inverseF), _Apply(MLE(DM), inverseF, idxs), EmbedLogPrior(DM, θ->_Apply(θ, F, idxs)); SkipOptim, SkipTests, kwargs...)
end

for (Name, F, Finv, TrafoName) in [(:LogTransform, :log, :exp, :log),
                                (:Log10Transform, :log10, :exp10, :log10),
                                (:ExpTransform, :exp, :log, :exp),
                                (:Exp10Transform, :exp10, :log10, :exp10),
                                (:BiLogTransform, :BiLog, :BiExp, :BiLog),
                                (:BiExpTransform, :BiExp, :BiLog, :BiExp),
                                (:BiRootTransform, :BiRoot, :BiPower, :BiRoot),
                                (:BiPowerTransform, :BiPower, :BiRoot, :BiPower),]
    @eval begin
        $Name(M::ModelOrFunction, idxs::BoolVector=(M isa ModelMap ? trues(M.xyp[3]) : trues(GetArgSize(M)[2]))) = ComponentwiseModelTransform(M, idxs, $F, $Finv)
        $Name(DM::AbstractDataModel, idxs::BoolVector=trues(pdim(DM)); kwargs...) = ComponentwiseModelTransform(DM, $F, $Finv, idxs; kwargs...)
        export $Name
    end
end


@deprecate Power10Transform(args...; kwargs...) Exp10Transform(args...; kwargs...)

ReflectionTransform(M::ModelOrFunction, idxs::BoolVector=(M isa ModelMap ? trues(M.xyp[3]) : trues(GetArgSize(M)[2]))) = ComponentwiseModelTransform(M, idxs, x-> -x, x-> -x)
ReflectionTransform(DM::AbstractDataModel, idxs::BoolVector=trues(pdim(DM)); kwargs...) = ComponentwiseModelTransform(DM, x-> -x, x-> -x, idxs; kwargs...)

ScaleTransform(M::ModelOrFunction, factor::Number, idxs::BoolVector=(M isa ModelMap ? trues(M.xyp[3]) : trues(GetArgSize(M)[2]))) = ComponentwiseModelTransform(M, idxs, x->factor*x, x->x/factor)
ScaleTransform(DM::AbstractDataModel, factor::Number, idxs::BoolVector=trues(pdim(DM)); kwargs...) = ComponentwiseModelTransform(DM, x->factor*x, x->x/factor, idxs; kwargs...)


function TranslationTransform(F::Union{ModelOrFunction, AbstractDataModel}, v::AbstractVector{T}; kwargs...) where T<:Number
    AffineTransform(F, Diagonal(Ones(T,length(v))), v; kwargs...)
end
function LinearTransform(F::Union{ModelOrFunction, AbstractDataModel}, A::AbstractMatrix{T}; kwargs...) where T<:Number
    AffineTransform(F, A, Zeros(T, size(A,1)); kwargs...)
end

function AffineTransform(F::Function, A::AbstractMatrix{<:Number}, v::AbstractVector{<:Number}; Domain::Union{HyperCube,Nothing}=nothing)
    @assert size(A,1) == size(A,2) == length(v)
    TranslatedModel(x, θ::AbstractVector{<:Number}; Kwargs...) = F(x, muladd(A,θ,v); Kwargs...)
end
function AffineTransform(M::ModelMap, A::AbstractMatrix{<:Number}, v::AbstractVector{<:Number}; Domain::Union{HyperCube,Nothing}=Domain(M), kwargs...)
    @assert isnothing(Domain) || (length(Domain) == size(A,1) == size(A,2) == length(v))
    Ainv = pinv(A)
    NewDomain = isnothing(Domain) ? HyperCube(Ainv*(Domain.L-v), Ainv*(Domain.U-v)) : nothing
    ModelMap(AffineTransform(M.Map, A, v), (!isnothing(InDomain(M)) ? (InDomain(M)∘(θ->muladd(A,θ,v))) : nothing), NewDomain,
                    M.xyp, M.pnames, M.inplace, M.CustomEmbedding, name(M), M.Meta; kwargs...)
end
function AffineTransform(DM::AbstractDataModel, A::AbstractMatrix{<:Number}, v::AbstractVector{<:Number}; Domain::Union{HyperCube,Nothing}=GetDomain(DM), SkipOptim::Bool=true, SkipTests::Bool=true, kwargs...)
    @assert pdim(DM) == size(A,1) == size(A,2) == length(v)
    Ainv = pinv(A)
    DataModel(Data(DM), AffineTransform(Predictor(DM), A, v; Domain), Ainv*(MLE(DM)-v), EmbedLogPrior(DM, θ->muladd(A,θ,v)); SkipOptim, SkipTests, kwargs...)
end

_GetDecorrelationTransform(DM::AbstractDataModel, mle::AbstractVector=MLE(DM), F::AbstractMatrix=FisherMetric(DM, mle)) = (_GetDecorrelationTransform(F), mle)
_GetDecorrelationTransform(M::AbstractMatrix) = cholesky(Symmetric(inv(M))).L
LinearDecorrelation(DM::AbstractDataModel, mle::AbstractVector=MLE(DM), F::AbstractMatrix=FisherMetric(DM, mle); kwargs...) = (M =_GetDecorrelationTransform(F); AffineTransform(DM, M, mle; kwargs...))

function DecorrelationTransforms(DM::AbstractDataModel, mle::AbstractVector=MLE(DM), F::AbstractMatrix=FisherMetric(DM, mle))
    M = _GetDecorrelationTransform(F);      iM = inv(M)
    ForwardTransform(x::AbstractVector) = muladd(M, x, mle)
    InvTransform(x::AbstractVector) = iM * (x - mle)
    ForwardTransform, InvTransform
end

# Unlike "ComponentwiseModelTransform" EmbedModelVia should be mainly performant for use e.g. in ProfileLikelihood
# Also only vector-valued transformations
"""
    EmbedModelVia(model, F::Function; Domain::HyperCube=FullDomain(GetArgLength(F; max=200),Inf)) -> Union{Function,ModelMap}
Transforms a model function via `newmodel(x, θ) = oldmodel(x, F(θ))`.
A `Domain` for the new model can optionally be specified for `ModelMap`s.
"""
EmbedModelVia(model::Function, F::Function; Kwargs...) = EmbeddedModel(x, θ; kwargs...) = model(x, F(θ); kwargs...)
EmbedModelVia_inplace(model!::Function, F::Function; Kwargs...) = EmbeddedModel!(y, x, θ; kwargs...) = model!(y, x, F(θ); kwargs...)

function EmbedModelVia(M::ModelMap, F::Function; Domain::Union{Nothing,HyperCube}=nothing, pnames::Union{Nothing,AbstractVector{<:StringOrSymb}}=nothing,
                name::StringOrSymb=name(M), Meta=M.Meta, inplace::Bool=ValToBool(M.inplace), IsCustom::Bool=ValToBool(M.CustomEmbedding), TrySymbolic::Bool=false, kwargs...)
    if isnothing(Domain)
        Domain = FullDomain(GetArgLength(F), Inf)
        @warn "Cannot infer new Domain HyperCube for general embeddings, using $Domain."
    end
    PNames = isnothing(pnames) ? CreateSymbolNames(length(Domain), "θ") : pnames
    ModelMap((isinplacemodel(M) ? EmbedModelVia_inplace : EmbedModelVia)(M.Map, F), (InDomain(M) isa Function ? (InDomain(M)∘F) : nothing),
            Domain, (M.xyp[1], M.xyp[2], length(Domain)); pnames=Symbol.(PNames), name=Symbol(name), inplace, IsCustom, Meta, TrySymbolic, kwargs...)
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
function EmbedDModelVia(dM::ModelMap, F::Function; ADmode::Union{Symbol,Val}=Val(:ForwardDiff), Domain::HyperCube=FullDomain(GetArgLength(F; max=MaxArgLen),Inf), pnames::Union{Nothing,AbstractVector{<:StringOrSymb}}=nothing, 
                name::StringOrSymb=name(dM), Meta=dM.Meta, inplace::Bool=ValToBool(dM.inplace), IsCustom::Bool=ValToBool(dM.CustomEmbedding), TrySymbolic::Bool=false, kwargs...)
    # Pass the OLD pdim to EmbedDModelVia_inplace for cache
    PNames = isnothing(pnames) ? CreateSymbolNames(length(Domain), "θ") : pnames
    ModelMap((isinplacemodel(dM) ? EmbedDModelVia_inplace : EmbedDModelVia)(dM.Map, F, dM.xyp[2:3]; ADmode), (!isnothing(InDomain(dM)) ? InDomain(dM)∘F : nothing),
            Domain, (dM.xyp[1], dM.xyp[2], length(Domain)); pnames=Symbol.(PNames), name=Symbol(name), inplace, IsCustom, Meta, TrySymbolic, kwargs...)
end

"""
    ModelEmbedding(DM::AbstractDataModel, F::Function, start::AbstractVector; Domain::HyperCube=FullDomain(length(start),Inf)) -> DataModel
Transforms a model function via `newmodel(x, θ) = oldmodel(x, F(θ))` and returns the associated `DataModel`.
An initial parameter configuration `start` as well as a `Domain` can optionally be passed to the `DataModel` constructor.

For component-wise transformations see [`ComponentwiseModelTransform`](@ref).
"""
function ModelEmbedding(DM::AbstractDataModel, F::Function, start::AbstractVector{<:Number}=GetStartP(GetArgLength(F; max=MaxArgLen)); Domain::HyperCube=FullDomain(length(start),Inf), kwargs...)
    DataModel(Data(DM), EmbedModelVia(Predictor(DM), F; Domain=Domain), EmbedDModelVia(dPredictor(DM), F; Domain=Domain), start, EmbedLogPrior(DM, F); kwargs...)
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
EmbedModelX(M::ModelMap, Emb::Function, Inplace::Bool=isinplacemodel(M)) = ModelMap((Inplace ? EmbedModelXin : EmbedModelXout)(M.Map, Emb), M)
EmbedModelX(model::Function, Emb::Function, Inplace::Bool=isinplacemodel(model)) = (Inplace ? EmbedModelXin : EmbedModelXout)(model, Emb)

"""
    TransformXdata(DM::AbstractDataModel, Emb::Function, iEmb::Function, TransformName::String="Trafo") -> AbstractDataModel
    TransformXdata(DS::AbstractDataSet, Emb::Function, TransformName::String="Trafo") -> AbstractDataSet
Returns a modified `DataModel` where the x-variables have been transformed by a multivariable transform `Emb` both in the data as well as for the model via `newmodel(x,θ) = oldmodel(Emb(x),θ)`.
`iEmb` denotes the inverse of `Emb`.
The uncertainties are computed via linearized error propagation through the given transformation.
"""
function TransformXdata(DM::AbstractDataModel, Emb::Function, iEmb::Function, TransformName::AbstractString=GetTrafoName(Emb); xnames::AbstractVector{<:StringOrSymb}=TransformName*"(".*xnames(DM).*")", ADmode::Union{Val,Symbol}=Val(:ForwardDiff), kwargs...)
    @assert all(WoundX(DM) .≈ map(iEmb∘Emb, WoundX(DM))) # Check iEmb is correct inverse
    DataModel(TransformXdata(Data(DM), Emb, TransformName; xnames, ADmode), EmbedModelX(Predictor(DM), iEmb), EmbedModelX(dPredictor(DM), iEmb), MLE(DM), LogPrior(DM); ADmode, kwargs...)
end
function TransformXdata(DS::AbstractFixedUncertaintyDataSet, Emb::Function, TransformName::AbstractString=GetTrafoName(Emb); xnames=TransformName*"(".*xnames(DS).*")", ADmode::Union{Val,Symbol}=Val(:ForwardDiff))
    NewX = Reduction(map(Emb, WoundX(DS)))
    if !HasXerror(DS)
        DataSetType(DS)(NewX, ydata(DS), ysigma(DS), dims(DS); xnames=xnames, ynames=Ynames(DS), name=name(DS))
    else
        EmbJac = xdim(DS) > 1 ? GetJac(ADmode, Emb, xdim(DS)) : GetDeriv(ADmode, Emb)
        NewXsigma = if xsigma(DS) isa AbstractVector
            map((xdat, xsig)->EmbJac(xdat)*xsig, WoundX(DS), Windup(xsigma(DS), xdim(DS))) # |> Reduction
        else
            J = reduce(BlockMatrix, map(EmbJac, WoundX(DS)))
            J * xsigma(DS) * transpose(J)
        end
        DataSetType(DS)(NewX, NewXsigma, ydata(DS), ysigma(DS), dims(DS); xnames=xnames, ynames=Ynames(DS), name=name(DS))
    end
end
# Drop iEmb
TransformXdata(DS::AbstractDataSet, Emb::Function, iEmb::Function, args...; kwargs...) = TransformXdata(DS, Emb, args...; kwargs...)


for (Name, F, Finv, TrafoName) in [(:LogXdata, :log, :exp, :log),
                                (:Log10Xdata, :log10, :exp10, :log10),
                                (:ExpXdata, :exp, :log, :exp),
                                (:Exp10Xdata, :exp10, :log10, :exp10),
                                (:SqrtXdata, :sqrt, :(x->x^2), :sqrt),
                                (:BiLogXdata, :BiLog, :BiExp, :BiLog),
                                (:BiExpXdata, :BiExp, :BiLog, :BiExp),
                                (:BiRootXdata, :BiRoot, :BiPower, :BiRoot),
                                (:BiPowerXdata, :BiPower, :BiRoot, :BiPower),]
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
function TransformYdata(DM::AbstractDataModel, Emb::Function, TransformName::AbstractString=GetTrafoName(Emb); ynames::AbstractVector{<:StringOrSymb}=TransformName*"(".*ynames(DM).*")", ADmode::Union{Val,Symbol}=Val(:ForwardDiff), kwargs...)
    DataModel(TransformYdata(Data(DM), Emb, TransformName; ynames, ADmode), EmbedModelY(Predictor(DM), Emb), MLE(DM), LogPrior(DM); ADmode, kwargs...)
end
function TransformYdata(DS::AbstractFixedUncertaintyDataSet, Emb::Function, TransformName::AbstractString=GetTrafoName(Emb); ynames::AbstractVector{<:StringOrSymb}=TransformName*"(".*ynames(DS).*")", ADmode::Union{Val,Symbol}=Val(:ForwardDiff))
    NewY = Reduction(map(Emb, WoundY(DS)));    EmbJac = ydim(DS) > 1 ? GetJac(ADmode, Emb, ydim(DS)) : GetDeriv(ADmode, Emb)
    NewYsigma = if ysigma(DS) isa AbstractVector
        map((ydat, ysig)->EmbJac(ydat)*ysig, WoundY(DS), Windup(ysigma(DS), ydim(DS))) # |> Reduction
    else
        J = reduce(BlockMatrix, map(EmbJac, WoundY(DS)))
        J * ysigma(DS) * transpose(J)
    end
    if !HasXerror(DS)
        DataSetType(DS)(xdata(DS), NewY, NewYsigma, dims(DS); xnames=Xnames(DS), ynames=ynames, name=name(DS))
    else
        DataSetType(DS)(xdata(DS), xsigma(DS), NewY, NewYsigma, dims(DS); xnames=Xnames(DS), ynames=ynames, name=name(DS))
    end
end


for (Name, F, TrafoName) in [(:LogYdata, :log, :log),
                            (:Log10Ydata, :log10, :log10),
                            (:ExpYdata, :exp, :exp),
                            (:Exp10Ydata, :exp10, :exp10),
                            (:SqrtYdata, :sqrt, :sqrt),
                            (:BiLogYdata, :BiLog, :BiLog),
                            (:BiExpYdata, :BiExp, :BiExp),
                            (:BiRootYdata, :BiRoot, :BiRoot),
                            (:BiPowerYdata, :BiPower, :BiPower),]
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


# Provided by ModelingToolkitExt
SystemTransform() = throw("Need to load ModelingToolkit.jl first to use SystemTransform!")


IsDEbased(F::Function) = occursin("DEmodel", string(nameof(typeof(F))))
IsDEbased(F::ModelMap) = IsDEbased(F.Map)
IsDEbased(DM::AbstractDataModel) = IsDEbased(Predictor(DM))

# Convert Vector to ComponentVector if required to retain type-stability
function GetComponentVectorEmbedding(P::ComponentVector)
    Ax = typeof(getaxes(P))
    ConvertToComponentVector(X::ComponentVector{T}) where T<:Number = X
    ConvertToComponentVector(X::AbstractVector{T}) where T<:Number  = convert(ComponentVector{T, Vector{T}, Ax}, X)
end
