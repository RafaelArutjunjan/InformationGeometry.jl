

# Returns a copy of type `Vector`, i.e. is not typesafe!
SafeCopy(X::AbstractVector) = copy(X)
SafeCopy(X::FillArrays.AbstractFillVector) = collect(X)
SafeCopy(X::ComponentVector) = convert(Vector,X)
SafeCopy(X::AbstractVector{<:Num}) = X
SafeCopy(X::AbstractRange) = collect(X)
SafeCopy(X::Union{SVector,MVector}) = convert(Vector,X)

Drop(X::AbstractVector, i::Int) = (Z=SafeCopy(X);   splice!(Z,i);   Z)
Drop(X::ComponentVector, i::Int) = Drop(convert(Vector,X), i) # Why is this needed?
Drop(N::Nothing, i) = nothing
Drop(C::HyperCube, i) = DropCubeDims(C, i)

_Presort(Components::AbstractVector{<:Int}; rev::Bool=false) = issorted(Components; rev=rev) ? Components : sort(Components; rev=rev)
Drop(X::AbstractVector, Components::AbstractVector{<:Int}) = (Z=SafeCopy(X); for i in _Presort(Components; rev=true)    splice!(Z,i)    end;    Z)
Drop(X::AbstractVector, Components::AbstractVector{<:Bool}) = (@assert length(X) == length(Components);    Drop(X, IndVec(Components)))

# If known to be sorted already, can interate via Iterators.reverse(X)

"""
    Consecutive(X::AbstractVector) -> Bool
Checks whether all elements are separated by a distance of one in ascending order.
"""
Consecutive(X::AbstractUnitRange) = true
# Consecutive(X::Union{StepRange,StepRangeLen})= X.step == one(typeof(X.step))
function Consecutive(X::AbstractVector{T})::Bool where T <: Number
    F = isequal(one(T))
    @inbounds for i in 2:length(X)
        F(X[i]-X[i-1]) || return false
    end;    true
end

# https://discourse.julialang.org/t/how-to-sort-two-or-more-lists-at-once/12073/13
function _SortTogether(A::AbstractVector, B::AbstractVector, args...; rev::Bool=false, kwargs...)
    issorted(A; rev=rev) ? (A, B, args...) : getindex.((A, B, args...), (sortperm(A; rev=rev, kwargs...),))
end

# Use PreallocationTools for ValInserter and mutate same object?

# Insert value and convert to ComponentVector of prescribed type after
function ValInserter(Component::Int, Value::AbstractFloat, Z::T) where T <: ComponentVector{<:Number}
    # GetRanges(X::ComponentVector) = (A = only(getaxes(X)); [A[p].idx for p in propertynames(X)])
    # Ranges = GetRanges(X)
    # DropInd = (1:length(X))[BasisVector(Component, length(X))]
    
    # Inserter = ValInserter(Component, Value)
    # ValInsertionComponentVector(X::AbstractVector{<:Number}) = convert(T, Inserter(convert(Vector,X)))
    (x::Vector->convert(T,x))∘ValInserter(Component, Value, eltype(Z)[])∘(z::AbstractVector->convert(Vector,z))
end

## Allow for use of `insert` method for non-static array types
StaticArrays.insert(X::AbstractVector, I::Int, V::Number) = vcat((@view X[1:I-1]), V, (@view X[I:end]))

"""
    ValInserter(Component::Int, Value::AbstractFloat) -> Function
Returns an embedding function ``\\mathbb{R}^N \\longrightarrow \\mathbb{R}^{N+1}`` which inserts `Value` in the specified `Component`.
In effect, this allows one to pin an input component at a specific value.
"""
function ValInserter(Component::Int, Value::AbstractFloat, Z::T=Float64[]; nonmutating::Bool=false) where T <: AbstractVector{<:Number}
    nonmutating && return NonMutValInsertionEmbedding(P::AbstractVector) = insert(P, Component, Value)
    ValInsertionEmbedding(P::AbstractVector) = insert!(SafeCopy(P), Component, Value)
    ValInsertionEmbedding(P::Union{SVector,MVector}) = insert(P, Component, Value)
    ValInsertionEmbedding(P::AbstractVector{<:Num}) = @views [P[1:Component-1]; Value; P[Component:end]]
end


## Provides correct insertion embedding for in-place functions
function ValInserterTransform(Inds, Vals, mle::AbstractVector)
    Ins = ValInserter(Inds, Vals, mle)
    KeepInds = [i for i in eachindex(mle) if i ∉ Inds]
    ReductionTransform(x::Number) = x # Pass through if the function that is wrapped is scalar, like cost function itself
    ReductionTransform(x::AbstractVector) = view(x, KeepInds)
    ReductionTransform(x::AbstractMatrix) = view(x, KeepInds, KeepInds)
    function WrapFunctionWithInversion(F::Function, TestOut=try F(mle) catch; (S = similar(mle);    F(S,mle);   S) end)
        NewJ = TestOut isa Number ? nothing : similar(TestOut)
        WrappedFunction(x; kwargs...) = ReductionTransform(F(Ins(x); kwargs...))
        WrappedFunction(J, x, args...; kwargs...) = (F(NewJ, Ins(x), args...; kwargs...);    copyto!(J, ReductionTransform(NewJ)); J)
    end
end


"""
    ValInserter(Components::AbstractVector{<:Int}, Values::AbstractVector{<:AbstractFloat}) -> Function
Returns an embedding function which inserts `Values` in the specified `Components`.
In effect, this allows one to pin multiple input components at a specific values.
"""
function ValInserter(Components::AbstractVector{<:Int}, Values::AbstractVector{<:Number}, Z::T=Float64[]; nonmutating::Bool=false) where T <: AbstractVector{<:Number}
    @assert length(Components) == length(Values)
    if length(Components) == 0
        return Identity(X::AbstractVector{<:Number}) = X
    elseif !nonmutating && length(Components) ≥ 2 && Consecutive(Components) # consecutive components.
        return ConsecutiveInsertionEmbedding(P::AbstractVector) = (Res=SafeCopy(P);  splice!(Res, Components[1]:Components[1]-1, Values);    Res)
    end

    components, values = _SortTogether(Components, Values)
    function NonMutValInsertionEmbedding(P::AbstractVector{<:Number})
        Res = insert(P, components[1], values[1])
        for i in 2:length(components)
            Res = insert(Res, components[i], values[i])
        end;    Res
    end
    nonmutating && return NonMutValInsertionEmbedding
    function ValInsertionEmbedding(P::AbstractVector)
        Res = SafeCopy(P)
        for i in eachindex(components)
            insert!(Res, components[i], values[i])
        end;    Res
    end
    ValInsertionEmbedding(P::Union{AbstractVector{<:Num},SVector,MVector}) = NonMutValInsertionEmbedding(P)
end
function ValInserter(Components::AbstractVector{<:Bool}, Values::AbstractVector{<:Number}, Z::AbstractVector{<:Number}=Float64[]; kwargs...)
    @assert length(Components) == length(Values)
    ValInserter(IndVec(Components), (@view Values[Components]), Z; kwargs...)
end
function ValInserter(Components::AbstractVector{<:Int}, Value::Number, Z::AbstractVector{<:Number}=Float64[]; kwargs...)
    ValInserter(Components, Fill(Value, length(Components)), Z; kwargs...)
end

InsertIntoFirst(X::AbstractVector{<:Number}) = PassingIntoLast(θ::AbstractVector{<:Number}) = [X;θ]
InsertIntoLast(θ::AbstractVector{<:Number}) = PassingIntoFirst(X::AbstractVector{<:Number}) = [X;θ]


ProfilePredictor(DM::AbstractDataModel, args...; kwargs...) = ProfilePredictor(Predictor(DM), args...; kwargs...)
ProfilePredictor(M::ModelMap, Comp::Int, PinnedValue::AbstractFloat, mlestructure::AbstractVector=Float64[]; kwargs...) = EmbedModelVia(M, ValInserter(Comp, PinnedValue, mlestructure); Domain=DropCubeDims(Domain(M), Comp), kwargs...)
ProfilePredictor(M::ModelMap, Comps::AbstractVector{<:Int}, PinnedValues::AbstractVector{<:AbstractFloat}, mlestructure::AbstractVector=Float64[]; kwargs...) = EmbedModelVia(M, ValInserter(Comps, PinnedValues, mlestructure); Domain=DropCubeDims(Domain(M), Comps), kwargs...)

ProfilePredictor(M::Function, Comp::Int, PinnedValue::AbstractFloat, mlestructure::AbstractVector=Float64[]; kwargs...) = EmbedModelVia(M, ValInserter(Comp, PinnedValue, mlestructure); kwargs...)
ProfilePredictor(M::Function, Comps::AbstractVector{<:Int}, PinnedValues::AbstractVector{<:AbstractFloat}, mlestructure::AbstractVector=Float64[]; kwargs...) = EmbedModelVia(M, ValInserter(Comps, PinnedValues, mlestructure); kwargs...)


ProfileDPredictor(DM::AbstractDataModel, args...; kwargs...) = ProfileDPredictor(dPredictor(DM), args...; kwargs...)
ProfileDPredictor(dM::ModelMap, Comp::Int, PinnedValue::AbstractFloat, mlestructure::AbstractVector=Float64[]; kwargs...) = EmbedDModelVia(dM, ValInserter(Comp, PinnedValue, mlestructure); Domain=DropCubeDims(Domain(dM), Comp), kwargs...)
ProfileDPredictor(dM::ModelMap, Comps::AbstractVector{<:Int}, PinnedValues::AbstractVector{<:AbstractFloat}, mlestructure::AbstractVector=Float64[]; kwargs...) = EmbedDModelVia(dM, ValInserter(Comps, PinnedValues, mlestructure); Domain=DropCubeDims(Domain(dM), Comps), kwargs...)

ProfileDPredictor(dM::Function, Comp::Int, PinnedValue::AbstractFloat, mlestructure::AbstractVector=Float64[]; kwargs...) = EmbedDModelVia(dM, ValInserter(Comp, PinnedValue, mlestructure); kwargs...)
ProfileDPredictor(dM::Function, Comps::AbstractVector{<:Int}, PinnedValues::AbstractVector{<:AbstractFloat}, mlestructure::AbstractVector=Float64[]; kwargs...) = EmbedDModelVia(dM, ValInserter(Comps, PinnedValues, mlestructure); kwargs...)


"""
    FixParameters(DM::AbstractDataModel, Component::Int, Value::AbstractFloat=MLE(DM)[Component])
    FixParameters(DM::AbstractDataModel, Components::AbstractVector{<:Int}, Values::AbstractVector{<:AbstractFloat}=MLE(DM)[Components])
    FixParameters(DM::AbstractDataModel, Components::AbstractVector{<:Bool}, Values::AbstractVector{<:AbstractFloat}=MLE(DM)[Components])
    FixParameters(DM::AbstractDataModel, ParamDict::Dict{Union{String,Symbol}, Number})
Returns `DataModel` where one or more parameters have been pinned to specified values.
"""
function FixParameters(DM::AbstractDataModel, Components::Union{Int,AbstractVector{<:Int}}, Values::Union{AbstractFloat,AbstractVector{<:AbstractFloat}}=MLE(DM)[Components]; MLE::AbstractVector{<:Number}=MLE(DM), SkipOptim::Bool=false, SkipTests::Bool=true, TrySymbolic::Bool=true, kwargs...)
    @assert DM isa DataModel # Not implemented for ConditionGrids yet
    @assert length(Components) == length(Values) && length(Components) < pdim(DM)
    length(Components) == 0 && (@warn "Got no parameters to pin.";  return DM)
    PNames = [Pnames(DM)[i] for i in eachindex(Pnames(DM)) if i ∉ Components]
    ## Add SymbolicCache kwarg
    DataModel(Data(DM), ProfilePredictor(DM, Components, Values; TrySymbolic, pnames=PNames), ProfileDPredictor(DM, Components, Values; pnames=PNames), Drop(MLE, Components), EmbedLogPrior(DM, ValInserter(Components, Values)); SkipOptim, SkipTests, name=name(DM), kwargs...)
end
function FixParameters(CG::AbstractConditionGrid, Components::Union{Int,AbstractVector{<:Int}}, Values::Union{AbstractFloat,AbstractVector{<:AbstractFloat}}=MLE(CG)[Components]; MLE::AbstractVector{<:Number}=MLE(CG), SkipOptim::Bool=false, SkipTests::Bool=true, kwargs...)
    @assert length(Components) == length(Values) && length(Components) < pdim(CG)
    length(Components) == 0 && (@warn "Got no parameters to pin.";  return CG)
    Emb = ValInserter(Components, Values);      PNames = [Pnames(CG)[i] for i in eachindex(Pnames(CG)) if i ∉ Components]
    # remake(CG; Trafos=Trafos(CG)∘Emb, LogPriorFn=EmbedLogPrior(CG, Emb), MLE=Drop(MLE, Components), pnames=PNames, Domain=Drop(Domain(CG), Components), SkipOptim, SkipTests, kwargs...) ## Does not rewrite likelihood and score
    ConditionGrid(Conditions(CG), Trafos(CG)∘Emb, EmbedLogPrior(CG, Emb), Drop(MLE, Components); pnames=PNames, Domain=Drop(Domain(CG), Components), SkipOptim, SkipTests, kwargs...)
end
function FixParameters(DM::AbstractDataModel, Components::AbstractVector{<:Bool}, args...; kwargs...)
    @assert length(Components) == pdim(DM)
    FixParameters(DM, IndVec(Components), args...; kwargs...)
end

FixParameters(DM::AbstractDataModel, ParamDict::Dict{<:AbstractString,<:Number}; kwargs...) = FixParameters(DM, Dict(Symbol.(keys(ParamDict)) .=> values(ParamDict)); kwargs...)
function FixParameters(DM::AbstractDataModel, ParamDict::Dict{Symbol,T}; kwargs...) where T<:Number
    Comps = Int[];  Vals = T[];    Keys = keys(ParamDict)
    for i in 1:pdim(DM)
        Pnames(DM)[i] ∈ Keys && (push!(Comps, i);  push!(Vals, ParamDict[Pnames(DM)[i]]))
    end
    @assert length(Comps) > 0 "No overlap between parameters and given parameter dictionary: pnames=$(Pnames(DM)), keys=$(keys(ParamDict))."
    FixParameters(DM, Comps, float.(Vals); kwargs...)
end

@deprecate PinParameters FixParameters


"""
    FixNonIdentifiable(DM::AbstractDataModel, mle::AbstractVector=MLE(DM); verbose::Bool=true, kwargs...)
Fixes all structuraly non-identified parameters to their current values.
"""
function FixNonIdentifiable(DM::AbstractDataModel, mle::AbstractVector{<:AbstractFloat}=MLE(DM), args...; verbose::Bool=true, SkipOptim::Bool=true, kwargs...)
    Fix = .!isfinite.(MLEuncertStd(DM, mle, args...)) |> IndVec
    verbose && println("Fixing structurally non-identified parameter indices $Fix, i.e.: $(pnames(DM)[Fix])")
    FixParameters(DM, Fix; SkipOptim, kwargs...)
end


_WithoutInd(X::AbstractVector{<:Bool}, ind::Int=findfirst(X)) = (Z=copy(X);  Z[ind]=false;  Z)
function GetLinkEmbedding(Linked::AbstractVector{<:Bool}, MainIndBefore::Int=findfirst(Linked))
    @assert 1 ≤ MainIndBefore ≤ length(Linked) && sum(Linked) ≥ 2 "Got Linked=$Linked and MainIndBefore=$MainIndBefore."
    LinkedInds = [i for i in eachindex(Linked) if Linked[i] && i != MainIndBefore]
    LinkEmbedding(θ::AbstractVector{<:Number}) = ValInserter(LinkedInds, θ[MainIndBefore], θ)(θ)
end
"""
    LinkParameters(DM::AbstractDataModel, Linked::Union{AbstractVector{<:Bool},AbstractVector{<:Int}}, MainIndBefore::Int=findfirst(Linked); kwargs...)
Embeds the model such that all components `i` for which `Linked[i] == true` are linked to the parameter corresponding to component `MainIndBefore`.
`Linked` can also be a `String`: this creates a `BitVector` whose components are `true` whenever the corresponding parameter name contains `Linked`.
"""
function LinkParameters(DM::AbstractDataModel, Linked::AbstractVector{<:Bool}, MainIndBefore::Int=findfirst(Linked), args...; MLE::AbstractVector{<:Number}=MLE(DM), SkipOptim::Bool=false, SkipTests::Bool=false, kwargs...)
    DataModel(Data(DM), LinkParameters(Predictor(DM), Linked, MainIndBefore, args...; kwargs...), Drop(MLE, _WithoutInd(Linked, MainIndBefore)), EmbedLogPrior(DM, GetLinkEmbedding(Linked,MainIndBefore)); SkipOptim, SkipTests)
end
function LinkParameters(CG::AbstractConditionGrid, Linked::AbstractVector{<:Bool}, MainIndBefore::Int=findfirst(Linked), args...; MLE::AbstractVector{<:Number}=MLE(CG), SkipOptim::Bool=false, SkipTests::Bool=false, kwargs...)
    Emb = GetLinkEmbedding(Linked, MainIndBefore);    WoFirst = _WithoutInd(Linked, MainIndBefore);    PNames = _GetLinkedPnames(pnames(CG), Linked, MainIndBefore; WoFirst)
    # remake(CG; Trafos=Trafos(CG)∘Emb, LogPriorFn=EmbedLogPrior(CG, Emb), MLE=Drop(MLE, WoFirst), pnames=PNames, Domain=Drop(Domain(CG), WoFirst), SkipOptim, SkipTests, kwargs...) ## Does not rewrite likelihood and score
    ConditionGrid(Conditions(CG), Trafos(CG)∘Emb, EmbedLogPrior(CG, Emb), Drop(MLE, WoFirst); pnames=PNames, Domain=Drop(Domain(CG), WoFirst), SkipOptim, SkipTests, kwargs...)
end

function _GetLinkedPnames(pnames::AbstractVector{<:StringOrSymb}, Linked::AbstractVector{<:Bool}, MainIndBefore::Int=findfirst(Linked); WoFirst::AbstractVector{<:Bool}=_WithoutInd(Linked, MainIndBefore))
    @assert length(pnames) == length(Linked) == length(WoFirst)
    PNames = string.(copy(pnames))
    PNames[MainIndBefore] *= " =: " * join(string.(pnames[WoFirst]), " ≡ ")
    Symbol.(PNames[.!WoFirst])
end
function LinkParameters(M::ModelMap, Linked::AbstractVector{<:Bool}, MainIndBefore::Int=findfirst(Linked); kwargs...)
    WoFirst = _WithoutInd(Linked, MainIndBefore)
    PNames = _GetLinkedPnames(pnames(M), Linked, MainIndBefore; WoFirst)
    EmbedModelVia(M, GetLinkEmbedding(Linked, MainIndBefore); Domain=DropCubeDims(Domain(M), WoFirst), pnames=PNames, kwargs...)
end
function LinkParameters(F::Function, Linked::AbstractVector{<:Bool}, MainIndBefore::Int=findfirst(Linked); kwargs...)
    EmbedModelVia(F, GetLinkEmbedding(Linked, MainIndBefore); kwargs...)
end
LinkParameters(DM::Union{ModelOrFunction, AbstractDataModel}, Linked::AbstractVector{<:Int}, args...; kwargs...) = (@assert all(1 .≤ Linked .≤ pdim(DM)) && allunique(Linked);   LinkParameters(DM, [i ∈ Linked for i in 1:pdim(DM)], args...; kwargs...))
LinkParameters(DM::Union{ModelOrFunction, AbstractDataModel}, S::AbstractString, args...; kwargs...) = LinkParameters(DM, occursin.(S, pnames(DM)), args...; kwargs...)

function LinkParameters(DM, S::AbstractString, T::AbstractString, args...; kwargs...)
    Sparam = occursin.(S, pnames(DM));    Tparam = occursin.(T, pnames(DM))
    @assert (sum(Sparam) == sum(Tparam) == 1 && Sparam != Tparam) "Unable to link two distinct parameters uniquely: $S occurs in $(pnames(DM)[Sparam]) and $T occurs in $(pnames(DM)[Tparam])."
    LinkParameters(DM, Sparam .|| Tparam, args...; kwargs...)
end

function _WidthsFromFisher(F::AbstractMatrix, Confnum::Real; dof::Int=size(F,1), failed::Real=1e-10)
    widths = try
        sqrt.(Diagonal(inv(F)).diag)
    catch;
        # For structurally unidentifiable models, return value given by "failed".
        1 ./ sqrt.(Diagonal(F).diag)
    end
    sqrt(InvChisqCDF(dof, ConfVol(Confnum))) * clamp.(widths, failed, 1/failed)
end

GetProfileDomainCube(DM::AbstractDataModel, Confnum::Real; MLE::AbstractVector=MLE(DM), kwargs...) = GetProfileDomainCube(DM, MLE, Confnum; kwargs...)
function GetProfileDomainCube(DM::AbstractDataModel, mle::AbstractVector{<:Number}, Confnum::Real; Fisher::AbstractMatrix=FisherMetric(DM, mle), dof::Int=DOF(DM), kwargs...)
    Cube = GetProfileDomainCube(Fisher, mle, Confnum; dof=dof, kwargs...)
    Predictor(DM) isa ModelMap ? (Cube ∩ Predictor(DM).Domain) : Cube
end
"""
Computes approximate width of Confidence Region from Fisher Metric and return this domain as a `HyperCube`.
Ensures that this width is positive even for structurally unidentifiable models.
"""
function GetProfileDomainCube(F::AbstractMatrix, mle::AbstractVector, Confnum::Real; dof::Int=length(mle), ForcePositive::Bool=false, failed::Real=1e-10)
    @assert size(F,1) == size(F,2) == length(mle)
    widths = _WidthsFromFisher(F, Confnum; dof=dof, failed=failed)
    @assert all(x->x>0, widths)
    L = mle - widths;   U = mle + widths
    if ForcePositive
        L = clamp.(L, 1e-10, 1e12);     U = clamp.(U, 1e-10, 1e12)
    end
    HyperCube(L,U)
end

# USE NelderMead for ODEmodels!!!!!

### In LsqFitExt now
# GetMinimizer(Res::LsqFit.LsqFitResult) = Res.param
# GetMinimum(Res::LsqFit.LsqFitResult, L::Function) = GetMinimum(GetMinimizer(Res), L)
# HasConverged(Res::LsqFit.LsqFitResult; kwargs...) = Res.converged
# GetIterations(Res::LsqFit.LsqFitResult) = try Res.trace[end].iteration catch; -Inf end # needs kwarg store_trace=true to be available

GetMinimizer(Res::Optim.OptimizationResults) = Optim.minimizer(Res)
GetMinimum(Res::Optim.OptimizationResults, L::Function) = Res.minimum
HasConverged(Res::Optim.OptimizationResults; kwargs...) = Optim.converged(Res)
GetIterations(Res::Optim.OptimizationResults) = Res.iterations

GetMinimizer(Res::SciMLBase.OptimizationSolution) = Res.u
GetMinimum(Res::SciMLBase.OptimizationSolution, L::Function) = Res.objective
HasConverged(Res::SciMLBase.OptimizationSolution; kwargs...) = HasConverged(Res.retcode; kwargs...)
HasConverged(Ret::SciMLBase.ReturnCode.T; kwargs...) = Ret === ReturnCode.Success
GetIterations(Res::SciMLBase.OptimizationSolution) = Res.stats.iterations

# For Multistart fit
GetMinimizer(X::AbstractVector{<:Number}) = X
GetMinimum(X::AbstractVector{<:Number}, L::Function) = L(X)
HasConverged(X::AbstractVector{<:Number}; verbose::Bool=true) = (verbose && (@warn "HasConverged: Cannot infer convergence from vector, returning true if all finite."); !any(isinf, X))
GetIterations(X::AbstractVector{<:Number}) = 0

# For Multistart fit
GetMinimizer(R::AbstractMultistartResults) = MLE(R)
GetMinimum(R::AbstractMultistartResults, L::Function) = L(MLE(R))
HasConverged(R::AbstractMultistartResults; StepTol::Real=1e-3) = 1 < GetFirstStepInd(R; StepTol)
GetIterations(R::AbstractMultistartResults) = R.Iterations[1]

"""
    GetProfile(DM::AbstractDataModel, Comp::Int, dom::Tuple{<:Real, <:Real}; N::Int=50, dof::Int=DOF(DM), SaveTrajectories::Bool=true, SavePriors::Bool=false)
Computes profile likelihood associated with the component `Comp` of the parameters over the domain `dom`.
"""
function GetProfile(DM::AbstractDataModel, Comp::Int, dom::Tuple{<:Real, <:Real}; adaptive::Bool=true, N::Int=31, MLE::AbstractVector=MLE(DM), kwargs...)
    @assert dom[1] < dom[2] && (1 ≤ Comp ≤ length(MLE))
    ps = range(dom...; length=N)
    GetProfile(DM, Comp, ps; adaptive, N, MLE, kwargs...)
end


function GetProfile(DM::AbstractDataModel, Comp::Int, ps::AbstractVector{<:Real}; adaptive::Bool=true, Confnum::Real=2.0, N::Int=(adaptive ? 31 : length(ps)), min_steps::Int=Int(round(2N/5)), 
                        AllowNewMLE::Bool=true, general::Bool=true, IsCost::Bool=true, dof::Int=DOF(DM), SaveTrajectories::Bool=true, ApproximatePaths::Bool=false, 
                        LogLikelihoodFn::Function=loglikelihood(DM), CostFunction::Function=Negate(LogLikelihoodFn), UseGrad::Bool=true, CostGradient::Union{Function,Nothing}=(UseGrad ? NegScore(DM) : nothing),
                        UseHess::Bool=false, ADmode::Val=Val(:ForwardDiff), GenerateNewDerivatives::Bool=true, SavedPs::Union{AbstractVector{<:AbstractVector},Nothing}=nothing,
                        FisherMetricFn::Function=FisherMetric(DM), CostHessian::Union{Function,Nothing}=(!UseHess ? nothing : (GenerateNewDerivatives ? AutoMetricFromNegScore(CostGradient; ADmode) : FisherMetricFn)),
                        LogPriorFn::Union{Nothing,Function}=LogPrior(DM), SavePriors::Bool=!isnothing(LogPriorFn), Ndata::Int=DataspaceDim(DM), UseFscaling::Bool=false,
                        MLE::AbstractVector{<:Number}=InformationGeometry.MLE(DM), logLikeMLE::Real=LogLikeMLE(DM), KnownVariance::Bool=!HasEstimatedUncertainties(DM),
                        Fisher::Union{Nothing, AbstractMatrix}=(adaptive ? FisherMetricFn(MLE) : nothing), verbose::Bool=false, resort::Bool=true, Multistart::Int=0, maxval::Real=1e5, OnlyBreakOnBounds::Bool=false,
                        Domain::Union{Nothing, HyperCube}=GetDomain(DM), InDomain::Union{Nothing, Function}=GetInDomain(DM), ProfileDomain::Union{Nothing, HyperCube}=GetDomain(DM), tol::Real=1e-10,
                        meth=((isnothing(LogPriorFn) && !general && !HasEstimatedUncertainties(DM) && isloaded(:LsqFit)) ? nothing : LBFGS(;linesearch=LineSearches.BackTracking())), OptimMeth=meth, OffsetResults::Bool=true,
                        IC::Real=(!UseFscaling ? eltype(MLE)(InvChisqCDF(dof, ConfVol(Confnum); maxval=1e8)) : eltype(MLE)(dof*InvFDistCDF(ConfVol(Confnum), dof, Ndata-dof; maxval=1e8))), MinSafetyFactor::Real=1.05, MaxSafetyFactor::Real=3, 
                        stepfactor::Real=3.5, stepmemory::Real=0.2, terminatefactor::Real=10, flatstepconst::Real=3e-2, curvaturesensitivity::Real=0.7, gradientsensitivity::Real=0.05, kwargs...)
    SavePriors && isnothing(LogPriorFn) && @warn "Got kwarg SavePriors=true but model does not have prior."
    @assert Confnum > 0
    # stepfactor: overall multiplicative factor for step length
    # stepmemory: linear interpolation of new with previous step size
    # terminatefactor: terminate profile if distance from original MLE too large
    # curvaturesensitivity: step length dependence on current profile curvature
    # gradientsensitivity: step length dependence on profile slope
    # flatstepconst: mainly controls step size when profile exactly flat
    @assert !adaptive || (stepfactor > 0 && flatstepconst > 0 && 0 ≤ stepmemory < 1 && terminatefactor > 0 && curvaturesensitivity ≥ 0 && gradientsensitivity ≥ 0)
    @assert IC > 0 && MinSafetyFactor > 1 && MaxSafetyFactor > MinSafetyFactor
    OnlyBreakOnBounds && adaptive && @warn "OnlyBreakOnBounds does not currently work with adaptive=true!"

    # Use the given SavedPs as initial starts for the individual optimizations, e.g. from an earlier profile computation or approximation
    isnothing(SavedPs) || (@assert !adaptive && length(SavedPs) == length(ps) && length(SavedPs[1]) == length(MLE))
    UseStashOrSaved = isnothing(SavedPs) ? (Stash,i)->Stash : (Stash,i)->Drop(SavedPs[i],Comp)
    
    # Point IS OUTSIDE confidence interval if: loglike < logLikeMLE - 0.5InformationGeometry.InvChisqCDF(dof, ConfVol(Confnum))
    # Until final rescaling after profile computation Res is likelihood without offset (larger is better) ⟹ CostThresh and MaxThreshold < 0 with offset already included
    # Should technically be called ObjectiveThreshold inside function
    CostThreshold, MaxThreshold = logLikeMLE .- 0.5 .* (IC*MinSafetyFactor, IC*MaxSafetyFactor)

    OptimDomain = Drop(Domain, Comp)

    FitFunc = if !general && isnothing(OptimMeth) && !isnothing(LogPriorFn) && KnownVariance
        ((args...; Kwargs...)->Curve_fit(args...; tol, Domain=OptimDomain, verbose, Kwargs...))
    elseif Multistart > 0
        Meth = (!isnothing(LogPriorFn) && isnothing(OptimMeth)) ? LBFGS(;linesearch=LineSearches.BackTracking()) : OptimMeth
        verbose && @info "Using Multistart fitting with N=$Multistart in profile $Comp"
        ((args...; Kwargs...)->MultistartFit(args...; MultistartDomain=OptimDomain, N=Multistart, meth=Meth, showprogress=false, resampling=true, parallel=false, maxval, verbose, tol, Kwargs..., plot=false, Full=true))
    else
        Meth = (!isnothing(LogPriorFn) && isnothing(OptimMeth)) ? LBFGS(;linesearch=LineSearches.BackTracking()) : OptimMeth
        ((args...; Kwargs...)->InformationGeometry.minimize(args...; tol, meth=Meth, Domain=OptimDomain, verbose, Kwargs..., Full=true))
    end
    
    # Does not check proximity to boundary! Also does not check nonlinear constraints!
    InBounds = θ::AbstractVector{<:Number} -> _IsInDomain(nothing, Domain, θ)
    # InBounds = θ::AbstractVector{<:Number} -> _IsInDomain(InDomain, Domain, θ)


    ConditionalPush!(N::Nothing, args...) = N
    ConditionalPush!(X::AbstractArray, args...) = push!(X, args...)

    Res = eltype(MLE)[];    visitedps = eltype(MLE)[]
    Converged = BitVector()
    path = SaveTrajectories ? typeof(MLE)[] : nothing
    priors = SavePriors ? eltype(MLE)[] : nothing

    sizehint!(Res, N)
    sizehint!(visitedps, N)
    sizehint!(Converged, N)
    SaveTrajectories && sizehint!(path, N)
    SavePriors && sizehint!(priors, N)

    # Domain for Optimization, ProfileDomain just for early termination of profile
    ParamBounds = isnothing(ProfileDomain) ? (-Inf, Inf) : ProfileDomain[Comp]
    OnlyBreakOnBounds && @assert all(isfinite, ParamBounds)

    if length(MLE) == 1    # Cannot drop dims if pdim already 1
        Xs = [[x] for x in ps]
        Res = map(LogLikelihoodFn, Xs)
        Converged = .!isnan.(Res) .&& .!isinf.(Res) .&& map(x->InBounds([x]), ps)
        visitedps = ps
        SaveTrajectories && (path = Xs)
        SavePriors && map(x->EvalLogPrior(LogPriorFn, x), Xs)
    else
        MLEstash = Drop(MLE, Comp)
        
        PerformStep!!! = if ApproximatePaths
            # Perform steps based on profile direction at MLE
            dir = GetLocalProfileDir!(isnothing(Fisher) ? FisherMetricFn(MLE) : copy(Fisher), Comp; verbose)
            pmle = MLE[Comp]
            @inline function PerformApproximateStep!(Res, MLEstash, Converged, visitedps, path, priors, p, i=nothing)
                θ = @. (p-pmle) * dir + MLE

                push!(Res, LogLikelihoodFn(θ))
                # Ignore MLEstash
                push!(Converged, !isnan(Res[end]) && InBounds(θ))
                push!(visitedps, p)
                ConditionalPush!(path, θ)
                ConditionalPush!(priors, EvalLogPrior(LogPriorFn, θ))
            end
        elseif general || !KnownVariance # force also with CG?
            # Build objective function based on Neglikelihood only without touching internals
            @inline function PerformStepGeneral!(Res, MLEstash, Converged, visitedps, path, priors, p, i=nothing)
                Ins = ValInserter(Comp, p, MLE)
                L = CostFunction∘Ins
                F = isnothing(CostGradient) ? L : (Transform=ValInserterTransform(Comp, p, MLE);   isnothing(CostHessian) ? (L, Transform(CostGradient)) : (L, Transform(CostGradient), Transform(CostHessian)))
                R = FitFunc(F, UseStashOrSaved(MLEstash,i); kwargs...)
                
                push!(Res, -GetMinimum(R,L))
                FullP = Ins(GetMinimizer(R))
                copyto!(MLEstash, GetMinimizer(R))
                push!(Converged, HasConverged(R) && InBounds(FullP))
                push!(visitedps, p)
                ConditionalPush!(path, FullP)
                ConditionalPush!(priors, EvalLogPrior(LogPriorFn, FullP))
            end
        else
            # Build objective function manually by embedding model and LogPrior separately
            # Does not work combined with variance estimation, i.e. error models
            @inline function PerformStepManual!(Res, MLEstash, Converged, visitedps, path, priors, p, i=nothing)
                NewModel = ProfilePredictor(DM, Comp, p, MLE)
                Ins = ValInserter(Comp, p, MLE)
                DroppedLogPrior = EmbedLogPrior(DM, Ins)
                R = FitFunc(Data(DM), NewModel, UseStashOrSaved(MLEstash,i), DroppedLogPrior; kwargs...)

                push!(Res, -GetMinimum(R,x->-loglikelihood(Data(DM), NewModel, x, DroppedLogPrior)))
                FullP = Ins(GetMinimizer(R))
                copyto!(MLEstash, GetMinimizer(R))
                push!(Converged, HasConverged(R) && InBounds(FullP))
                push!(visitedps, p)
                ConditionalPush!(path, FullP)
                ConditionalPush!(priors, EvalLogPrior(LogPriorFn, FullP))
            end
        end
        # Adaptive instead of ps grid here?
        if adaptive
            approx_PL_curvature((x1, x2, x3), (y1, y2, y3)) = @fastmath -2 * (y1 * (x2 - x3) + y2 * (x3 - x1) + y3 * (x1 - x2)) / ((x1 - x2) * (x2 - x3) * (x3 - x1))
            
            maxstepnumber = N
            Fi = isnothing(Fisher) ? FisherMetricFn(MLE)[Comp,Comp] : Fisher[Comp,Comp]
            # Calculate initial stepsize based on curvature from fisher information
            initialδ = clamp(stepfactor * sqrt(IC) / (maxstepnumber * (flatstepconst + curvaturesensitivity*sqrt(Fi))) , 1e-12, 1e2)

            δ = initialδ
            minstep = 1e-2 * initialδ
            maxstep = 1e5 * initialδ

            # Second left point
            p = MLE[Comp] - δ
            PerformStep!!!(Res, MLEstash, Converged, visitedps, path, priors, p)

            # Input MLE
            push!(Res, LogLikelihoodFn(MLE))
            push!(Converged, InBounds(MLE))
            push!(visitedps, MLE[Comp])
            SaveTrajectories && push!(path, MLE)
            SavePriors && push!(priors, EvalLogPrior(LogPriorFn, MLE))

            # Second right point
            p = MLE[Comp] + δ
            PerformStep!!!(Res, MLEstash, Converged, visitedps, path, priors, p)

            visitedps2 = deepcopy(visitedps) |> reverse!
            Res2 = deepcopy(Res) |> reverse!
            path2 = SaveTrajectories ? reverse!(deepcopy(path)) : nothing
            priors2 = SavePriors ? reverse!(deepcopy(priors)) : nothing
            Converged2 = deepcopy(Converged) |> reverse!
            len = length(visitedps) -1
            
            @inline function DoAdaptive(visitedps, Res, path, priors, Converged)
                while !(Res[end] < CostThreshold) # break if threshold is passed
                    approx_curv = approx_PL_curvature((@view visitedps[end-2:end]), (@view Res[end-2:end]))
                    approx_grad = (Res[end]-Res[end-1]) / (visitedps[end]-visitedps[end-1])
                    newδ = stepfactor * sqrt(IC) / (maxstepnumber * (flatstepconst + curvaturesensitivity*sqrt(abs(approx_curv)) + gradientsensitivity*abs(approx_grad)))
                    δ = clamp(newδ > δ ? stepmemory*δ + (1-stepmemory)*newδ : newδ, minstep, maxstep)
                    
                    p = clamp(right ? visitedps[end] + δ : visitedps[end] - δ, ParamBounds...)

                    # Do the actual profile point calculation using the value p
                    PerformStep!!!(Res, MLEstash, Converged, visitedps, path, priors, p)

                    ## Early termination if profile flat or already wide enough
                    if right
                        p ≥ ParamBounds[2] && break
                        !OnlyBreakOnBounds && (length(visitedps) - len > maxstepnumber/2 || p > MLE[Comp] + terminatefactor*maxstepnumber*initialδ) && break
                    else
                        p ≤ ParamBounds[1] && break
                        !OnlyBreakOnBounds && (length(visitedps) - len > maxstepnumber/2 || p < MLE[Comp] - terminatefactor*maxstepnumber*initialδ) && break
                    end
                end
            end
            
            # Do right branch of profile
            right = true
            DoAdaptive(visitedps, Res, path, priors, Converged)
            
            # Do left branch of profile
            right = false
            δ = initialδ
            newδ = initialδ
            len = length(visitedps2) -1
            copyto!(MLEstash, Drop(MLE, Comp))
            DoAdaptive(visitedps2, Res2, path2, priors2, Converged2)
            
            visitedps = [(@view reverse!(visitedps2)[1:end-3]); visitedps]
            Res = [(@view reverse!(Res2)[1:end-3]); Res]
            path = SaveTrajectories ? [(@view reverse!(path2)[1:end-3]); path] : nothing
            priors = SavePriors ? [(@view reverse!(priors2)[1:end-3]); priors] : nothing
            Converged = [(@view reverse!(Converged2)[1:end-3]); Converged]
        else
            startind = (mlecomp = MLE[Comp];    try findfirst(x->x>mlecomp, ps)-1 catch; 1 end)
            if resort && isnothing(SavedPs) && startind > 1
                for p in sort((@view ps[startind:end]))
                    PerformStep!!!(Res, MLEstash, Converged, visitedps, path, priors, clamp(p, ParamBounds...))
                    p ≥ ParamBounds[2] && break
                    !OnlyBreakOnBounds && ((length(visitedps) - 1 > min_steps && Res[end] < CostThreshold) || (Res[end] < MaxThreshold)) && break
                end
                len = length(visitedps)
                copyto!(MLEstash, Drop(MLE, Comp))
                for p in sort((@view ps[startind-1:-1:1]); rev=true)
                    PerformStep!!!(Res, MLEstash, Converged, visitedps, path, priors, clamp(p, ParamBounds...))
                    p ≤ ParamBounds[1] && break
                    !OnlyBreakOnBounds && ((length(visitedps) - len > min_steps && Res[end] < CostThreshold) || (Res[end] < MaxThreshold)) && break
                end
            else # No early break, no clamping, just evaluate on given ps
                for (i,p) in enumerate(ps)
                    PerformStep!!!(Res, MLEstash, Converged, visitedps, path, priors, p, i)
                end
            end
        end
    end

    ProfileDataPostProcessing!!(Res, priors, Converged; MLE, dof, LogPriorFn, AllowNewMLE, IsCost, SavePriors, OffsetResults, logLikeMLE, Ndata, UseFscaling, verbose)
    perm = sortperm(visitedps)
    # Param always first col, Res second, Converged last. Third column always priors IF length(cols) ≥ 4, columns after prior may be other saved information.
    # Lazy array construction via RecursiveArrayTools allows for preserving type information of columns while still being indexable as matrix
    ResMat = SavePriors ? VectorOfArray([visitedps[perm], Res[perm], priors[perm], Converged[perm]]) : VectorOfArray([visitedps[perm], Res[perm], Converged[perm]])
    SaveTrajectories ? (ResMat, path[perm]) : ResMat
end

function GetProfile(DM::AbstractDataModel, Comp::Int, Confnum::Real; ForcePositive::Bool=false, kwargs...)
    GetProfile(DM, Comp, (C=GetProfileDomainCube(DM, Confnum; ForcePositive=ForcePositive); (C.L[Comp], C.U[Comp])); Confnum=Confnum, kwargs...)
end


function ProfileDataPostProcessing!!(Res::AbstractVector{<:Number}, priors::Union{Nothing,AbstractVector}, Converged::AbstractVector{<:Bool}; MLE::AbstractVector=Float64[], dof::Real=length(MLE), LogPriorFn::Union{Function,Nothing}=nothing, 
                        AllowNewMLE::Bool=true, IsCost::Bool=true, SavePriors::Bool=!isnothing(priors), OffsetResults::Bool=true, logLikeMLE::Real=-Inf, Ndata::Int=-5000, UseFscaling::Bool=false, verbose::Bool=true)
    Logmax = AllowNewMLE ? max(try maximum(view(Res, Converged)) catch; -Inf end, logLikeMLE) : logLikeMLE
    Logmax > logLikeMLE && @warn "Profile Likelihood analysis apparently found a likelihood value which is larger (i.e. better) than the previously stored LogLikeMLE. Continuing anyway."
    # Using pdim(DM) instead of 1 here, because it gives the correct result
    Priormax = SavePriors ? EvalLogPrior(LogPriorFn,MLE) : 0.0
    if IsCost
        if OffsetResults
            @. Res = 2*(Logmax - Res)
            SavePriors && (@. priors = 2*(Priormax - priors))
        else
            Res .*= -2
            SavePriors && (priors .*= -2)
        end
    else
        @assert OffsetResults
        Chi²ₖConfMapping(l::Real, lmle::Real) = l ≤ lmle ? InvConfVol(ChisqCDF(dof, 2(lmle - l))) : NaN
        Fₖ_ₙ₋ₖ_ConfMapping(l::Real, lmle::Real) = l ≤ lmle ? InvConfVol(FDistCDF(2(lmle - l)/dof, dof, Ndata-dof)) : NaN
        ConfMapping = UseFscaling ? Fₖ_ₙ₋ₖ_ConfMapping : Chi²ₖConfMapping
        @inbounds for i in eachindex(Res)
            Res[i] = ConfMapping(Res[i], Logmax)
        end
        if SavePriors
            verbose && @info "Got IsCost=true with SavePriors=true. Converting prior to confidence scale independently from data contribution. Strictly speaking, the isolated prior contributions are not meaningfully comparable to full profile on this scale!"
            @inbounds for i in eachindex(priors)
                priors[i] = ConfMapping(priors[i], Priormax)
            end
        end
    end
end


GetLocalProfileDir(DM::AbstractDataModel, Comp::Int, p::AbstractVector{<:Number}=MLE(DM); verbose::Bool=false) = GetLocalProfileDir!(FisherMetric(DM, p), Comp; verbose)
function GetLocalProfileDir!(F::AbstractMatrix, Comp::Int; verbose::Bool=false)
    # @boundscheck @assert size(F,1) == size(F,2) && 1 ≤ Comp ≤ size(F,1)
    verbose && NotPosDef(F) && @warn "Using pseudo-inverse to determine profile direction for parameter $Comp due to local non-identifiability."
    F[Comp, :] .= [(j == Comp) for j in axes(F,1)]
    dir = pinv(F)[:, Comp];    dir ./= dir[Comp];   dir
end

function ProfileLikelihood(DM::AbstractDataModel, Confnum::Real=2.0, inds::AbstractVector{<:Int}=1:pdim(DM); dof::Int=DOF(DM), ForcePositive::Bool=false, 
                            MLE::AbstractVector{<:Number}=MLE(DM), Fisher::AbstractMatrix=FisherMetric(DM, MLE), 
                            ProfileDomain::HyperCube=GetProfileDomainCube(Fisher, MLE, Confnum; dof, ForcePositive=ForcePositive), kwargs...)
    ProfileLikelihood(DM, ProfileDomain, inds; Confnum, MLE, Fisher, dof, kwargs...)
end

function ProfileLikelihood(DM::AbstractDataModel, ProfileDomain::HyperCube, inds::AbstractVector{<:Int}=1:pdim(DM); plot::Bool=isloaded(:Plots), Multistart::Int=0, parallel::Bool=(Multistart==0), verbose::Bool=true, showprogress::Bool=verbose, idxs::Tuple{Vararg{Int}}=length(pdim(DM))≥3 ? (1,2,3) : (1,2), 
                        MLE::AbstractVector{<:Number}=MLE(DM), kwargs...)
    # idxs for plotting only
    @assert 1 ≤ length(inds) ≤ length(MLE) && allunique(inds) && all(1 .≤ inds .≤ length(MLE)) && issorted(inds)
    @assert length(MLE) ≥ pdim(DM) # Allow for method reuse with FullParameterProfiles

    Prog = Progress(length(inds); enabled=showprogress, desc="Computing Profiles... "*(parallel ? "(parallel, $(nworkers()) workers) " : ""), dt=1, showspeed=true)
    Profiles = (parallel ? progress_pmap : progress_map)(i->GetProfile(DM, i, (ProfileDomain.L[i], ProfileDomain.U[i]); verbose, Multistart, MLE, kwargs...), inds; progress=Prog)

    plot && display(ProfilePlotter(DM, Profiles; idxs))
    Profiles
end

# x and y labels must be passed as kwargs
PlotSingleProfile(DM::AbstractDataModel, Prof::Tuple{<:Union{VectorOfArray,AbstractMatrix}, <:Any}, i::Int; kwargs...) = PlotSingleProfile(DM, Prof[1], i; kwargs...)
function PlotSingleProfile(DM::AbstractDataModel, Prof::Union{VectorOfArray,AbstractMatrix}, i::Int; kwargs...)
    P = RecipesBase.plot(view(Prof, :,1), Convergify(view(Prof, :,2), GetConverged(Prof)); leg=false, label=["Profile" nothing], kwargs...)
    HasPriors(Prof) && RecipesBase.plot!(P, view(Prof, :,1), Convergify(view(Prof, :,3), GetConverged(Prof)); label=["Prior" nothing], color=[:red :brown], line=:dash)
    P
end


GetConverged(M::AbstractMatrix) = BitVector(@view M[:, end])
GetConverged(M::VectorOfArray) = BitVector(M[end])
# Convergify(Values::AbstractVector{<:Number}, Converged::Union{BitVector,BoolVector}) = [Values .+ (NaN .* .!Converged) Values .+ (NaN .* ShrinkTruesByOne(Converged))]
Convergify(Values::AbstractVector{<:Number}, Converged::Union{BitVector,BoolVector}) = [Values .+ (Inf .* .!Converged) Values .+ (Inf .* ShrinkTruesByOne(Converged))]


# Grow Falses to their next neighbors to avoid holes in plot
function ShrinkTruesByOne(X::BoolVector)
    length(X) == 1 && return X
    Res = copy(X)
    X[1] && !X[2] && (Res[1] = false)
    X[end] && !X[end-1] && (Res[end] = false)
    for i in 2:length(Res)-1
        X[i] && (!X[i-1] || !X[i+1]) && (Res[i] = false)
    end;    Res
end

# What if trajectories NaN?
HasTrajectories(M::Tuple{Union{AbstractMatrix,VectorOfArray}, Nothing}) = false
HasTrajectories(M::Tuple{Union{AbstractMatrix,VectorOfArray}, AbstractVector}) = !all(x->all(isnan,x), M[2])
HasTrajectories(M::AbstractVector{<:Tuple}) = any(HasTrajectories, M)
HasTrajectories(M::AbstractMatrix) = false
HasTrajectories(M::AbstractVector{<:AbstractMatrix}) = false

HasPriors(M::AbstractVector{<:Union{<:AbstractMatrix, <:VectorOfArray}}) = any(HasPriors, M)
HasPriors(M::Tuple) = HasPriors(M[1])
HasPriors(M::Union{<:AbstractMatrix, <:VectorOfArray}) = size(M,2) > 3



function ProfilePlotter(DM::AbstractDataModel, Profiles::AbstractVector;
    PNames::AbstractVector{<:AbstractString}=(Predictor(DM) isa ModelMap ? pnames(Predictor(DM)) : CreateSymbolNames(pdim(DM), "θ")), idxs::Tuple{Vararg{Int}}=length(pdim(DM))≥3 ? (1,2,3) : (1,2), kwargs...)
    @assert length(Profiles) == length(PNames)
    Ylab = length(PNames) == pdim(DM) ? "Conf. level [σ]" : "W = 2[ℓ_mle - ℓ(θ)]"
    PlotObjects = [PlotSingleProfile(DM, Profiles[i], i; xlabel=PNames[i], ylabel=Ylab, kwargs...) for i in eachindex(Profiles)]
    length(Profiles) ≤ 3 && HasTrajectories(Profiles) && push!(PlotObjects, PlotProfileTrajectories(DM, Profiles; idxs))
    RecipesBase.plot(PlotObjects...; layout=length(PlotObjects), size=PlotSizer(length(PlotObjects)))
end
# Plot trajectories of Profile Likelihood
"""
    PlotProfileTrajectories(DM::AbstractDataModel, Profiles::AbstractVector{Tuple{AbstractMatrix,AbstractVector}}; idxs::Tuple=(1,2,3), OverWrite=true, kwargs...)
"""
function PlotProfileTrajectories(DM::AbstractDataModel, Profiles::AbstractVector; OverWrite::Bool=true, idxs::Tuple{Vararg{Int}}=length(pdim(DM))≥3 ? (1,2,3) : (1,2), kwargs...)
    @assert HasTrajectories(Profiles)
    @assert (2 ≤ length(idxs) ≤ 3 && allunique(idxs) && all(1 .≤ idxs .≤ pdim(DM)))
    P = OverWrite ? RecipesBase.plot() : RecipesBase.plot!()
    for i in eachindex(Profiles)
        HasTrajectories(Profiles[i]) && RecipesBase.plot!(P, map(x->getindex(x,collect(idxs)),Profiles[i][2]); marker=:circle, label="Comp: $i", kwargs...)
    end
    axislabels = (; xlabel=pnames(DM)[idxs[1]], ylabel=pnames(DM)[idxs[2]])
    length(idxs) == 3 && (axislabels = (; axislabels..., zlabel=pnames(DM)[idxs[3]]))
    RecipesBase.plot!(P, [MLE(DM)[collect(idxs)]]; linealpha=0, marker=:hex, markersize=3, label="MLE", axislabels..., kwargs...)
end

# Centralized Interpolation method where defaults for extension can be chosen - possibly change to ExtrapolationType.Extension?
GetInterpolator(X::AbstractVector{<:Number}, Y::AbstractVector{<:Number}, Interp::Type{<:AbstractInterpolation}; extrapolation=ExtrapolationType.None, kwargs...) = Interp(X, Y; extrapolation, kwargs...)

"""
    InterpolatedProfiles(M::AbstractVector{<:AbstractMatrix}, Interp::Type{<:AbstractInterpolation}=QuadraticInterpolation) -> Vector{Function}
Interpolates the `Vector{Matrix}` output of `ParameterProfiles`.
!!!note
    Does not distinguish between converged and non-converged points in the profile.
"""
function InterpolatedProfiles(Mats::AbstractVector{<:AbstractMatrix}, Interp::Type{<:AbstractInterpolation}=QuadraticInterpolation; kwargs...)
    [GetInterpolator(view(profile,:,2), view(profile,:,1), Interp; kwargs...) for profile in Mats]
end

"""
    ProfileBox(DM::AbstractDataModel, Fs::AbstractVector{<:AbstractInterpolation}, Confnum::Real=1.) -> HyperCube
Constructs `HyperCube` which bounds the confidence region associated with the confidence level `Confnum` from the interpolated likelihood profiles.
"""
ProfileBox(DM::AbstractDataModel, Fs::AbstractVector{<:Union{<:AbstractInterpolation,<:Nothing}}, Confnum::Real=1.; MLE::AbstractVector{<:Number}=MLE(DM), dof::Int=DOF(DM), kwargs...) = ProfileBox(Fs, MLE, Confnum; dof, kwargs...)

function ProfileBox(Fs::AbstractVector{<:Union{<:AbstractInterpolation,<:Nothing}}, mle::AbstractVector, Confnum::Real=1.; parallel::Bool=true, dof::Int=length(mle), kwargs...)
    @assert length(Fs) == length(mle)
    reduce(vcat, (parallel ? pmap : map)(i->_ProfileBox(Fs[i], Confnum; mleval=mle[i], dof, kwargs...), 1:length(Fs)))
end



FindSingleZeroWrapper(args...; kwargs...) = try Roots.find_zero(args...; kwargs...) catch;  NaN end

# Use Bracketing method from mle outwards by default since faster than find_zeros
FindZerosWrapper(F::Function, lb::AbstractFloat, ub::AbstractFloat; meth::Union{Nothing,Roots.AbstractUnivariateZeroMethod}=Roots.AlefeldPotraShi(), kwargs...) = FindZerosWrapper(F, lb, ub, meth; kwargs...)
# Catch unwanted kwargs: no_pts for single zero searches and mleval for AllZeros search
FindZerosWrapper(F::Function, lb::AbstractFloat, ub::AbstractFloat, ::Nothing; mleval::Real=0, kwargs...) = Roots.find_zeros(F, lb, ub; kwargs...)
FindZerosWrapper(F::Function, lb::AbstractFloat, ub::AbstractFloat, meth::Roots.AbstractBracketing; no_pts::Int=0, mleval::Real=(lb+ub)/2, kwargs...) = [FindSingleZeroWrapper(F, (lb, mleval), meth; kwargs...), FindSingleZeroWrapper(F, (mleval, ub), meth; kwargs...)]
FindZerosWrapper(F::Function, lb::AbstractFloat, ub::AbstractFloat, meth::Roots.AbstractNonBracketing; no_pts::Int=0, mleval::Real=(lb+ub)/2, kwargs...) = [FindSingleZeroWrapper(F, (lb+mleval)/2, meth; kwargs...), FindSingleZeroWrapper(F, (mleval+ub)/2, meth; kwargs...)]


_ProfileBox(F::Nothing, Confnum::Real=1.0; kwargs...) = HyperCube([-Inf], [Inf])

function _ProfileBox(F::AbstractInterpolation, Confnum::Real=1.0; IsCost::Bool=true, dof::Int=1, mleval::Real=F.t[findmin(F.u)[2]], 
                            CostThreshold::Union{<:Real, Nothing}=nothing, maxval::Real=Inf, tol::Real=1e-10, xrtol::Real=tol, xatol::Real=tol, kwargs...)
    Crossings = if !IsCost
        FindZerosWrapper(x->(F(x)-Confnum), F.t[1], F.t[end]; no_pts=length(F.t), xrtol, xatol, mleval, kwargs...)
    else
        # Already 2(loglikeMLE - loglike) in Profile
        CostThresh = if !isnothing(CostThreshold)
            # Allow for computation of F-based threshold here?
            CostThreshold
        else
            InvChisqCDF(dof, ConfVol(Confnum))
        end
        FindZerosWrapper(x->(F(x)-CostThresh), F.t[1], F.t[end]; no_pts=length(F.t), xrtol, xatol, mleval, kwargs...)
    end
    crossings = view(Crossings, .!isnan.(Crossings))
    if length(crossings) == 0
        crossings = [-maxval, maxval]
    elseif length(crossings) == 1
        if mleval < crossings[1]     # crossing is upper bound
            crossings = [-maxval, crossings[1]]
        else
            crossings = [crossings[1], maxval]
        end
    elseif length(crossings) > 2
        # think of cleverer way for checking slope
        @warn "Got $(length(crossings)) crossings: $crossings"
    end
    HyperCube([minimum(crossings)], [maximum(crossings)]; Padding=0.0)
end
ProfileBox(DM::AbstractDataModel, M::AbstractVector{<:AbstractMatrix}, Confnum::Real=2.0; Padding::Real=0., Interp::Type{<:AbstractInterpolation}=QuadraticInterpolation, kwargs...) = ProfileBox(DM, InterpolatedProfiles(M, Interp), Confnum; Padding, kwargs...)
ProfileBox(DM::AbstractDataModel, Confnum::Real; Padding::Real=0., add::Real=0.5, IsCost::Bool=true, Interp::Type{<:AbstractInterpolation}=QuadraticInterpolation, kwargs...) = ProfileBox(DM, ParameterProfiles(DM, Confnum+add; plot=false, IsCost, kwargs...), Confnum; IsCost, Interp, Padding)



"""
    PracticallyIdentifiable(DM::AbstractDataModel, Confnum::Real=3; plot::Bool=isloaded(:Plots), IsCost::Bool=false, kwargs...) -> Real
Determines the maximum confidence level (in units of standard deviations σ) at which the given `DataModel` is still practically identifiable.
"""
PracticallyIdentifiable(DM::AbstractDataModel, Confnum::Real=3; plot::Bool=false, N::Int=51, IsCost::Bool=false, kwargs...) = PracticallyIdentifiable(ParameterProfiles(DM, Confnum; plot, N, IsCost, kwargs...))

function PracticallyIdentifiable(Mats::AbstractVector{<:Union{<:AbstractMatrix,<:VectorOfArray}})
    function Minimax(M::Union{<:AbstractMatrix,<:VectorOfArray})
        finitevals = isfinite.(view(M,:,2))
        sum(finitevals) == 0 && return Inf
        V = @view M[finitevals, 2]
        split = findmin(V)[2]
        min(maximum(view(V,1:split)), maximum(view(V,split:length(V))))
    end
    minimum([Minimax(M) for M in Mats])
end



mutable struct ParameterProfiles <: AbstractProfiles
    Profiles::AbstractVector{<:Union{<:AbstractMatrix,<:VectorOfArray}}
    Trajectories::AbstractVector{<:Union{<:AbstractVector{<:AbstractVector{<:Number}}, <:Nothing}}
    Names::AbstractVector{Symbol}
    MLE::AbstractVector{<:Number}
    dof::Int
    IsCost::Bool
    Meta::Symbol
    # Allow for different inds and fill rest with nothing or NaN
    function ParameterProfiles(DM::AbstractDataModel, Confnum::Union{Real,HyperCube}=2., Inds::AbstractVector{<:Int}=1:pdim(DM); plot::Bool=isloaded(:Plots), SaveTrajectories::Bool=true, IsCost::Bool=true, dof::Int=DOF(DM), Meta::Symbol=:ParameterProfiles, pnames::AbstractVector{<:StringOrSymb}=pnames(DM), MLE::AbstractVector=MLE(DM), kwargs...)
        inds = sort(Inds)
        FullProfs = ProfileLikelihood(DM, Confnum, inds; plot=false, SaveTrajectories, IsCost, MLE, kwargs...)
        Profs = SaveTrajectories ? getindex.(FullProfs,1) : FullProfs
        Trajs = SaveTrajectories ? getindex.(FullProfs,2) : Fill(nothing, length(inds))
        if !(inds == 1:length(MLE))
            EmptyProf = VectorOfArray([Profs[1][1,i] isa Bool ? falses(1) : typeof(Profs[1][1,i])[NaN] for i in axes(Profs[1],2)])
            EmptyTraj = [Fill(NaN, length(MLE))]
            for i in 1:length(MLE) # Profs and Trajs already sorted by sorting inds
                if i ∉ inds
                    insert!(Profs, i, EmptyProf)
                    SaveTrajectories ? insert!(Trajs, i, EmptyTraj) : insert!(Trajs, i, nothing)
                end
            end
        end
        # Add check if new MLE was found
        P = ParameterProfiles(DM, Profs, Trajs, pnames; IsCost, dof, Meta, MLE)
        plot && display(RecipesBase.plot(P, false))
        P
    end
    function ParameterProfiles(DM::AbstractDataModel, Profiles::AbstractVector{<:Union{<:AbstractMatrix,<:VectorOfArray}}, Trajectories::AbstractVector=Fill(nothing,length(Profiles)), Names::AbstractVector{<:StringOrSymb}=pnames(DM); IsCost::Bool=true, dof::Int=DOF(DM), MLE::AbstractVector=MLE(DM), kwargs...)
        ParameterProfiles(Profiles, Trajectories, Names, MLE, dof, IsCost; kwargs...)
    end
    function ParameterProfiles(Profiles::AbstractVector{<:Union{<:AbstractMatrix,<:VectorOfArray}}, Trajectories::AbstractVector=Fill(nothing,length(Profiles)), Names::AbstractVector{<:StringOrSymb}=CreateSymbolNames(length(Profiles),"θ"); IsCost::Bool=true, dof::Int=length(Names), kwargs...)
        ParameterProfiles(Profiles, Trajectories, Names, Fill(NaN, length(Names)), dof, IsCost; kwargs...)
    end
    function ParameterProfiles(Profiles::AbstractVector{<:Union{<:AbstractMatrix,<:VectorOfArray}}, Trajectories::AbstractVector, Names::AbstractVector{<:StringOrSymb}, mle, dof::Int, IsCost::Bool, meta::Symbol=:ParameterProfiles; Meta::Symbol=meta, verbose::Bool=true)
        @assert length(Profiles) == length(Names) == length(mle) == length(Trajectories)
        verbose && !(1 ≤ dof ≤ length(mle)) && @warn "Got dof=$dof but length(MLE)=$(length(mle))."
        new(Profiles, Trajectories, Symbol.(Names), mle, dof, IsCost, Meta)
    end
end
(P::ParameterProfiles)(t::Real, i::Int, Interp::Type{<:AbstractInterpolation}=QuadraticInterpolation; kwargs...) = InterpolatedProfiles(P, i, Interp; kwargs...)(t)
(P::ParameterProfiles)(i::Int, Interp::Type{<:AbstractInterpolation}=QuadraticInterpolation; kwargs...) = InterpolatedProfiles(P, i, Interp; kwargs...)
InterpolatedProfiles(P::ParameterProfiles, i::Int, Interp::Type{<:AbstractInterpolation}=QuadraticInterpolation; kwargs...) = IsPopulated(P[i]) ? GetInterpolator(view(Profiles(P)[i],:,2), view(Profiles(P)[i],:,1), Interp; kwargs...) : nothing
InterpolatedProfiles(P::ParameterProfiles, Interp::Type{<:AbstractInterpolation}=QuadraticInterpolation; kwargs...) = [InterpolatedProfiles(P, i, Interp; kwargs...) for i in eachindex(P)]

# For SciMLBase.remake

"""
    ParameterProfiles(DM::AbstractDataModel, Confnum::Real=2, Inds::AbstractVector{<:Int}=1:pdim(DM); adaptive::Bool=true, N::Int=31, plot::Bool=isloaded(:Plots), SaveTrajectories::Bool=true, IsCost::Bool=true, parallel::Bool=true, dof::Int=DOF(DM), kwargs...)
Computes the profile likelihood for components `Inds` of the parameters ``θ \\in \\mathcal{M}`` over the given `Domain`.
Returns a vector of matrices where the first column of the n-th matrix specifies the value of the n-th component and the second column specifies the associated confidence level of the best fit configuration conditional to the n-th component being fixed at the associated value in the first column.
`Confnum` specifies the confidence level to which the profile should be computed if possible with `Confnum=2` corresponding to 2σ, i.e. approximately 95.4%.
Single profiles can be accessed via `P[i]`, given a profile object `P`.

The kwarg `IsCost=true` can be used to skip the transformation from the likelihood values to the associated confidence level such that `2(LogLikeMLE(DM) - loglikelihood(DM, θ))` is returned in the second columns of the profiles.
The trajectories followed during the reoptimization along the profile can be saved via `SaveTrajectories=true`.
For `adaptive=false` the size of the domain is estimated from the inverse Fisher metric and the profile is evaluated on a fixed stepsize grid.
Further `kwargs` can be passed to the optimization.

# Extended help

For visualization of the results, multiple methods are available, see e.g. [`PlotProfileTrajectories`](@ref), [`PlotRelativeParameterTrajectories`](@ref).
"""
ParameterProfiles(;
    Profiles::AbstractVector{<:Union{<:AbstractMatrix,<:VectorOfArray}}=[Zeros(1,3)],
    Trajectories::AbstractVector{<:Union{<:AbstractVector{<:AbstractVector{<:Number}}, <:Nothing}}=[nothing],
    Names::AbstractVector{<:StringOrSymb}=Symbol[],
    MLE::AbstractVector{<:Number}=Float64[],
    dof::Int=0,
    IsCost::Bool=false,
    Meta::Symbol=:remake) = ParameterProfiles(Profiles, Trajectories, Names, MLE, dof, IsCost; Meta)


Profiles(P::ParameterProfiles) = P.Profiles
Trajectories(P::ParameterProfiles) = P.Trajectories
pnames(P::ParameterProfiles) = P.Names .|> string
Pnames(P::ParameterProfiles) = P.Names
MLE(P::ParameterProfiles) = P.MLE
pdim(P::ParameterProfiles) = length(MLE(P))
DOF(P::ParameterProfiles) = P.dof
IsCost(P::ParameterProfiles) = P.IsCost
HasTrajectories(P::ParameterProfiles) = any(i->HasTrajectories(P[i]), 1:length(P))
IsPopulated(P::ParameterProfiles) = Bool[HasProfiles(P[i]) for i in eachindex(P)]
AllConverged(P::ParameterProfiles) = [AllConverged(P[i]) for i in eachindex(P)]
HasPriors(P::ParameterProfiles) = any(HasPriors, Profiles(P))

ProfileDomain(P::ParameterProfiles) = [IsPopulated(P[i]) ? collect(extrema(@view Profiles(P[i])[:,1])) : [-Inf,Inf] for i in eachindex(P)] |> HyperCube
HyperCube(P::ParameterProfiles) = ProfileDomain(P)
@deprecate Domain(P::ParameterProfiles) ProfileDomain(P)

Base.length(P::ParameterProfiles) = Profiles(P) |> length
Base.firstindex(P::ParameterProfiles) = Profiles(P) |> firstindex
Base.lastindex(P::ParameterProfiles) = Profiles(P) |> lastindex
Base.getindex(P::ParameterProfiles, i::Int) = ParameterProfilesView(P, i)
Base.getindex(P::ParameterProfiles, inds::AbstractVector{<:Int}) = [P[i] for i in inds]
Base.keys(P::ParameterProfiles) = 1:length(P)

ProfileBox(DM::AbstractDataModel, P::ParameterProfiles, Confnum::Real=1; kwargs...) = ProfileBox(P, Confnum; kwargs...)
"""
    ProfileBox(P::ParameterProfiles, Confnum::Real=1; Interp=DataInterpolations.QuadraticInterpolation, kwargs...)
Constructs `HyperCube` which bounds the confidence region associated with the confidence level `Confnum` from the interpolated likelihood profiles.
"""
ProfileBox(P::ParameterProfiles, Confnum::Real=1; IsCost::Bool=IsCost(P), dof::Int=DOF(P), Interp::Type{<:AbstractInterpolation}=QuadraticInterpolation, kwargs...) = ProfileBox(InterpolatedProfiles(P, Interp), MLE(P), Confnum; IsCost, dof, kwargs...)


"""
    PracticallyIdentifiable(P::ParameterProfiles) -> Real
Determines the maximum level at which ALL the given profiles in `ParameterProfiles` are still practically identifiable.
If `IsCost=true` was chosen for the profiles, the output is the maximal deviation in cost function value `W = 2(L_MLE - PL_i(θ))`.
If instead `IsCost=false` was chosen, so that cost function deviations have already been rescaled to confidence levels, the output of `PracticallyIdentifiable` is the maximal confidence level in units of standard deviations σ where the model is still practically identifiability.
"""
PracticallyIdentifiable(P::ParameterProfiles) = PracticallyIdentifiable(Profiles(P))



"""
    ParameterProfilesView(P::ParameterProfiles, i::Int)
Views `ParameterProfiles` object for the `i`th parameter.
"""
mutable struct ParameterProfilesView
    P::ParameterProfiles
    i::Int
    ParameterProfilesView(P::ParameterProfiles, i::Int) = (@assert 1 ≤ i ≤ pdim(P);     new(P,i))
end
(PV::ParameterProfilesView)(t::Real, Interp::Type{<:AbstractInterpolation}=QuadraticInterpolation; kwargs...) = PV.P(t, PV.i, Interp; kwargs...)
InterpolatedProfiles(PV::ParameterProfilesView, Interp::Type{<:AbstractInterpolation}=QuadraticInterpolation; kwargs...) = InterpolatedProfiles(PV.P, PV.i, Interp; kwargs...)

# Specialized
Profiles(PV::ParameterProfilesView) = getindex(Profiles(PV.P), PV.i)
Trajectories(PV::ParameterProfilesView) = getindex(Trajectories(PV.P), PV.i)
# Passthrough
pnames(PV::ParameterProfilesView) = pnames(PV.P)
Pnames(PV::ParameterProfilesView) = Pnames(PV.P)
MLE(PV::ParameterProfilesView) = MLE(PV.P)
pdim(PV::ParameterProfilesView) = pdim(PV.P)
DOF(PV::ParameterProfilesView) = DOF(PV.P)
IsCost(PV::ParameterProfilesView) = IsCost(PV.P)
HasTrajectories(PV::ParameterProfilesView) = !isnothing(Trajectories(PV)) && !all(x->all(isnan,x), Trajectories(PV))
HasProfiles(PV::ParameterProfilesView) = !all(isnan, view(Profiles(PV), :, 1))
IsPopulated(PV::ParameterProfilesView) = HasProfiles(PV)
GetConverged(PV::ParameterProfilesView) = GetConverged(Profiles(PV))
AllConverged(PV::ParameterProfilesView) = all(GetConverged(PV))
HasPriors(P::ParameterProfilesView) = HasPriors(Profiles(P))


# AbstractMatrix to the outside
Base.length(PV::ParameterProfilesView) = Profiles(PV) |> length
Base.size(PV::ParameterProfilesView) = Profiles(PV) |> size
Base.firstindex(PV::ParameterProfilesView) = Profiles(PV) |> firstindex
Base.lastindex(PV::ParameterProfilesView) = Profiles(PV) |> lastindex
Base.getindex(PV::ParameterProfilesView, args...) = getindex(Profiles(PV), args...)


ProfileBox(DM::AbstractDataModel, PV::ParameterProfilesView, Confnum::Real=1; kwargs...) = ProfileBox(PV, Confnum; kwargs...)
"""
    ProfileBox(PV::ParameterProfilesView, Confnum::Real=1; Interp=DataInterpolations.QuadraticInterpolation, kwargs...)
Constructs `HyperCube` which bounds the confidence region associated with the confidence level `Confnum` from the interpolated likelihood profiles.
"""
ProfileBox(PV::ParameterProfilesView, Confnum::Real=1; IsCost::Bool=IsCost(PV), dof::Int=DOF(PV), Interp::Type{<:AbstractInterpolation}=QuadraticInterpolation, kwargs...) = ProfileBox([InterpolatedProfiles(PV, Interp)], [MLE(PV)[PV.i]], Confnum; IsCost, dof, kwargs...)

PracticallyIdentifiable(PV::ParameterProfilesView) = PracticallyIdentifiable(view(Profiles(PV.P), PV.i:PV.i))


"""
    FullParameterProfiles(DM::AbstractDataModel, Confnum::Real=2., Inds::AbstractVector{<:Int}=(1:pdim(DM)) .+ length(xdata(DM)); LogLikelihoodFn::Function=LiftedLogLikelihood(DM)∘LiftedEmbedding(DM), MLE::AbstractVector{<:Number}=TotalLeastSquaresV(DM), Fisher::AbstractMatrix=FullFisherMetric(DM, MLE), kwargs...)
Compute parameter profiles while accounting for the uncertainties in the independent variables.
"""
function FullParameterProfiles(DM::AbstractDataModel, Confnum::Real=2., Inds::AbstractVector{<:Int}=(1:pdim(DM)) .+ length(xdata(DM)); ADmode=Val(:ForwardDiff), pnames::AbstractVector{<:StringOrSymb}=_FullNames(DM), 
                    LogLikelihoodFn::Function=(@assert !HasPrior(DM);   LiftedLogLikelihood(DM)∘LiftedEmbedding(DM)), CostFunction::Function=Negate(LogLikelihoodFn), CostGradient=GetGrad!(ADmode, CostFunction),
                    MLE::AbstractVector{<:Number}=TotalLeastSquaresV(DM), logLikeMLE::Real=LogLikelihoodFn(MLE), Fisher::AbstractMatrix=FullFisherMetric(DM,MLE), pDomain::Union{Nothing,HyperCube}=GetDomain(DM), 
                    xDomain::Union{Nothing,HyperCube}=(isnothing(pDomain) ? nothing : HyperCube(Fill(-Inf,length(xdata(DM))),Fill(Inf,length(xdata(DM))))), Domain::Union{Nothing,HyperCube}=(!isnothing(xDomain) && !isnothing(pDomain)) ? vcat(xDomain, pDomain) : nothing, 
                    ProfileDomain::Union{Nothing,HyperCube}=Domain, InDomain::Union{Nothing,Function}=isnothing(GetInDomain(DM)) ? nothing : GetInDomain(DM)∘(pd=pdim(DM);  x->(@view x[end-pd+1:end])), kwargs...)
    @assert HasXerror(DM)
    ParameterProfiles(DM, Confnum, Inds; ADmode, pnames, LogLikelihoodFn, CostFunction, CostGradient, MLE, logLikeMLE, Fisher, Domain, InDomain, ProfileDomain, kwargs...)
end


function PlotProfileTrajectories(DM::AbstractDataModel, P::ParameterProfiles; kwargs...)
    @assert HasTrajectories(P)
    PlotProfileTrajectories(DM, [(Profiles(P)[i], Trajectories(P)[i]) for i in eachindex(Profiles(P))]; kwargs...)
end

function ExtendProfiles(P::ParameterProfiles)
    throw("Not programmed yet.")
end


"""
    ProfileTransform(P::ParameterProfiles, F::Function; kwargs...)
Given a function `F(W::Vector)` or `F(p::Vector,W::Vector)`, the saved values of the likelihood ratio test statistic `w = 2(ℓ_mle - ℓ_profile(p))` are transformed.
"""
function ProfileTransform(P::ParameterProfiles, F::Function; Meta::Symbol=Symbol("Trafo: "*string(F)), TransformPriors::Bool=false, kwargs...)
    @assert IsCost(P);    HasPriors(P) && !TransformPriors && @warn "Saved priors of given profile are not transformed!"
    WrappedTrafo = MaximalNumberOfArguments(F) > 1 ? F : (x,y)->F(y)
    ObjectTrafo(Prof::VectorOfArray) = (S=copy(Prof);  S[2]=WrappedTrafo(S[1], S[2]);  TransformPriors && (S[3]=WrappedTrafo(S[1], S[3]));     S)
    NewProfs = [ObjectTrafo(Prof) for Prof in Profiles(P)]
    remake(P; Profiles=NewProfs, IsCost=false, Meta, kwargs...)
end


"""
    ProfileConfidenceTransform(DM::AbstractDataModel, P::ParameterProfiles, Trafo::Symbol=:Chi; dof::Real=DOF(DM), Ndata::Int=DataspaceDim(DM), kwargs...)
Applies confidence scaling based on `Trafo=:Chi` or `Trafo=:F` to given profiles `P`.
"""
function ProfileConfidenceTransform(DM::AbstractDataModel, P::ParameterProfiles, Trafo::Symbol=:Chi; dof::Real=DOF(DM), Ndata::Int=DataspaceDim(DM), kwargs...)
        _ProfileConfidenceTransform(P, Trafo, dof, Ndata; kwargs...)
end
function _ProfileConfidenceTransform(P::ParameterProfiles, Trafo::Symbol, dof::Real, Ndata::Int; TransformPriors::Bool=HasPriors(P), kwargs...)
    @assert Trafo === :Chi || Trafo === :F
    # Already transformed as 2(lmle-l)
    Chi²ₖConfMappingInd(w::Real) = w ≥ 0 ? InvConfVol(ChisqCDF(dof, w)) : NaN
    Chi²ₖConfMapping(Ps::AbstractVector, Ws::AbstractVector) = map(Chi²ₖConfMappingInd, Ws)
    Fₖ_ₙ₋ₖ_ConfMappingInd(w::Real) = w ≥ 0 ? InvConfVol(FDistCDF(w/dof, dof, Ndata-dof)) : NaN
    Fₖ_ₙ₋ₖ_ConfMapping(Ps::AbstractVector, Ws::AbstractVector) = map(Fₖ_ₙ₋ₖ_ConfMappingInd, Ws)
    ProfileTransform(P, Trafo === :Chi ? Chi²ₖConfMapping : Fₖ_ₙ₋ₖ_ConfMapping; TransformPriors, kwargs...)
end


"""
    PlotAlongProfilePaths(P::ParameterProfiles, F::Function; kwargs...)
Plot a scalar function `F` along the parameter trajectories of the profiles in `P`.
"""
function PlotAlongProfilePaths(P::ParameterProfiles, F::Function; kwargs...)
    PopulatedInds = IsPopulated(P)
    RecipesBase.plot([RecipesBase.plot(Profiles(P)[i][1], map(F, Trajectories(P)[i]); xlabel=pnames(P)[i], kwargs...) for i in eachindex(P) if IsPopulated(P)[i]]...; layout=sum(PopulatedInds), size=PlotSizer(sum(PopulatedInds)))
end

"""
    PlotAlongProfilePaths(P::ParameterProfiles, Fs::AbstractVector{<:Function}; kwargs...)
Plot i-th scalar function `F` in `Fs` along the parameter trajectories of the profiles in `P`.
"""
function PlotAlongProfilePaths(P::ParameterProfiles, Fs::AbstractVector{<:Function}; kwargs...)
    PopulatedInds = IsPopulated(P)
    @assert length(Fs) == pdim(P)
    RecipesBase.plot([RecipesBase.plot(Profiles(P)[i][1], map(Fs[i], Trajectories(P)[i]); xlabel=pnames(P)[i]) for i in eachindex(P) if IsPopulated(P)[i]]...; layout=sum(PopulatedInds), size=PlotSizer(sum(PopulatedInds)), kwargs...)
end


"""
    ReoptimizeProfile(DM::AbstractDataModel, P::ParameterProfiles, inds::AbstractVector{<:Int}=IndVec(IsPopulated(P)); plot::Bool=isloaded(:Plots), verbose::Bool=true, kwargs...)
Takes given profile `P` and reoptimizes each point in the trajectories.
"""
function ReoptimizeProfile(DM::AbstractDataModel, P::ParameterProfiles, inds::AbstractVector{<:Int}=IndVec(IsPopulated(P)); plot::Bool=isloaded(:Plots), Multistart::Int=0, parallel::Bool=(Multistart==0), pnames::AbstractVector{<:StringOrSymb}=pnames(DM),
                        verbose::Bool=true, showprogress::Bool=verbose, SaveTrajectories::Bool=true, MLE::AbstractVector=MLE(P), dof::Int=DOF(P), IsCost::Bool=IsCost(P), Meta::Symbol=:ParameterProfiles, kwargs...)
    @assert HasTrajectories(P);     @assert Multistart == 0
    @assert 1 ≤ length(inds) ≤ length(MLE) && allunique(inds) && all(1 .≤ inds .≤ length(MLE)) && issorted(inds)
    Prog = Progress(length(inds); enabled=showprogress, desc="Computing Profiles... "*(parallel ? "(parallel, $(nworkers()) workers) " : ""), dt=1, showspeed=true)
    FullProfs = (parallel ? progress_pmap : progress_map)(i->GetProfile(DM, i, getindex.(Trajectories(P)[i],i); adaptive=false, SavedPs=Trajectories(P)[i], verbose, Multistart, MLE, kwargs...), inds; progress=Prog)

    Profs = SaveTrajectories ? getindex.(FullProfs,1) : FullProfs
    Trajs = SaveTrajectories ? getindex.(FullProfs,2) : Fill(nothing, length(inds))
    if !(inds == 1:length(MLE))
        EmptyProf = VectorOfArray([Profs[1][1,i] isa Bool ? falses(1) : typeof(Profs[1][1,i])[NaN] for i in axes(Profs[1],2)])
        EmptyTraj = [Fill(NaN, length(MLE))]
        for i in 1:length(MLE) # Profs and Trajs already sorted by sorting inds
            if i ∉ inds
                insert!(Profs, i, EmptyProf)
                SaveTrajectories ? insert!(Trajs, i, EmptyTraj) : insert!(Trajs, i, nothing)
            end
        end
    end
    P2 = ParameterProfiles(DM, Profs, Trajs, pnames; IsCost, dof, Meta, MLE)
    plot && display(RecipesBase.plot(P2, false))
    P2
end


## Presumably more efficient for small coordinate distortions, less efficient for large coordinate distortions
function PreapproximatedParameterProfiles(DM::AbstractDataModel, Confnum::Union{Real,HyperCube}=2., Inds::AbstractVector{<:Int}=1:pdim(DM); 
                SaveTrajectories::Bool=true, SavePriors::Bool=false, plot::Bool=isloaded(:Plots), kwargs...)
    P = ParameterProfiles(DM, Confnum, Inds; ApproximatePaths=true, SaveTrajectories=true, SavePriors=false, verbose=false, plot=false, kwargs...)
    ReoptimizeProfile(DM, P, Inds; SaveTrajectories, SavePriors, plot, kwargs...)
end




"""
    IntegrationParameterProfiles(DM::AbstractDataModel, confnum::Real=2, inds::AbstractVector{<:Int}=1:pdim(DM); meth=BS3(), tol=1e-3, N::Union{Nothing,Int}=51, ProfileDomain::HyperCube=FullDomain(length(MLE), Inf), γ::Union{Nothing,Real}=nothing, kwargs...)
Computes profile likelihood path via integrating ODE derived via Lagrange multiplier based contraint by Chen and Jennrich (https://doi.org/10.1198/106186002493).
Unlike in Chen and Jennrich's approach, no stabilization term with constant `γ` is added by default, since the need for this stabilization is essentially obviated by the accuracy of autodiff Hessians and γ > 0 adds undesirable bias to the computed trajectory.
"""
function IntegrationParameterProfiles(DM::AbstractDataModel, confnum::Real=2, inds::AbstractVector{<:Int}=1:pdim(DM); plot::Bool=isloaded(:Plots), Multistart::Int=0, parallel::Bool=(Multistart==0), verbose::Bool=true, showprogress::Bool=verbose,
            Confnum::Real=confnum, SaveTrajectories::Bool=true, IsCost::Bool=true, dof::Int=DOF(DM), Meta::Symbol=:IntegrationParameterProfiles, pnames::AbstractVector{<:StringOrSymb}=pnames(DM), MLE::AbstractVector{<:Number}=MLE(DM), kwargs...)
    @assert 1 ≤ length(inds) ≤ length(MLE) && allunique(inds) && all(1 .≤ inds .≤ length(MLE)) && issorted(inds)
    @assert length(MLE) ≥ pdim(DM) # Allow for method reuse with FullParameterProfiles

    Prog = Progress(length(inds); enabled=showprogress, desc="Computing Profiles... "*(parallel ? "(parallel, $(nworkers()) workers) " : ""), dt=1, showspeed=true)
    FullProfs = (parallel ? progress_pmap : progress_map)(i->GetIntegrationProfile(DM, i, Float64[]; MLE, Confnum, IsCost, dof, verbose, kwargs...), inds; progress=Prog)

    Profs = SaveTrajectories ? getindex.(FullProfs,1) : FullProfs
    Trajs = SaveTrajectories ? getindex.(FullProfs,2) : Fill(nothing, length(inds))
    if !(inds == 1:length(MLE))
        EmptyProf = VectorOfArray([Profs[1][1,i] isa Bool ? falses(1) : typeof(Profs[1][1,i])[NaN] for i in axes(Profs[1],2)])
        EmptyTraj = [Fill(NaN, length(MLE))]
        for i in 1:length(MLE) # Profs and Trajs already sorted by sorting inds
            if i ∉ inds
                insert!(Profs, i, EmptyProf)
                SaveTrajectories ? insert!(Trajs, i, EmptyTraj) : insert!(Trajs, i, nothing)
            end
        end
    end
    # Add check if new MLE was found
    P = ParameterProfiles(DM, Profs, Trajs, pnames; IsCost, dof, Meta, MLE)
    plot && display(RecipesBase.plot(P, false))
    P
end


function GetIntegrationProfile(DM::AbstractDataModel, Comp::Int, ps::AbstractVector=Float64[]; ADmode::Val=Val(:ForwardDiff), N::Union{Nothing,Int}=51, tol::Real=1e-3,
                LogLikelihoodFn::Function=loglikelihood(DM), MLE::AbstractVector{<:Number}=MLE(DM), dof::Real=DOF(DM), Ndata::Int=DataspaceDim(DM), Confnum::Number=2, UseFscaling::Bool=false,
                IC::Real=(!UseFscaling ? eltype(MLE)(InvChisqCDF(dof, ConfVol(Confnum); maxval=1e8)) : eltype(MLE)(dof*InvFDistCDF(ConfVol(Confnum), dof, Ndata-dof; maxval=1e8))),
                LogPriorFn::Union{Function,Nothing}=LogPrior(DM), logLikeMLE::Real=LogLikeMLE(DM), Domain::Union{Nothing, HyperCube}=GetDomain(DM), verbose::Bool=true, 
                CostHessian::Function=GetHess!(ADmode, Negate(LogLikelihoodFn)),
                ### Pure PostProcessing:
                AllowNewMLE::Bool=true, IsCost::Bool=true, SavePriors::Bool=!isnothing(LogPriorFn), OffsetResults::Bool=true, SaveTrajectories::Bool=true, # Catch last
                kwargs...)
    LeftSol = IntegrationProfileArm(LogLikelihoodFn, MLE, Comp; Left=true,  ADmode, CostHessian, logLikeMLE, Confnum, IC, Domain, tol, verbose, kwargs...)
    RightSol= IntegrationProfileArm(LogLikelihoodFn, MLE, Comp; Left=false, ADmode, CostHessian, logLikeMLE, Confnum, IC, Domain, tol, verbose, kwargs...)

    # Need to make sure all elements unique for interpolation
    path = Vector{eltype(MLE)}[]
    if isnothing(N)
        sizehint!(path, length(LeftSol.t)+length(RightSol.t))
        for i in 2:length(LeftSol.t)-1
            push!(path, insert!(LeftSol.u[end+1-i], Comp, LeftSol.t[end+1-i]))
        end
        for i in 1:length(RightSol.t)-1
            push!(path, insert!(RightSol.u[i], Comp, RightSol.t[i]))
        end
    else # Interpolate ODE solution to achieve fixed profile N
        iseven(N) && (N += 1);    sizehint!(path, N)
        for t in range(extrema(LeftSol.t)...; length=1+(N-1)÷2)
            push!(path, insert!(LeftSol(t), Comp, t))
        end
        for t in range(extrema(RightSol.t)...; length=(N-1)÷2)[2:end]
            push!(path, insert!(RightSol(t), Comp, t))
        end
    end
    Res = map(LogLikelihoodFn, path)
    priors = SavePriors ? map(LogPriorFn, path) : nothing
    Converged = trues(length(Res))

    ProfileDataPostProcessing!!(Res, priors, Converged; MLE, dof, LogPriorFn, AllowNewMLE, IsCost, SavePriors, OffsetResults, logLikeMLE, Ndata, UseFscaling, verbose)
    ResMat = SavePriors ? VectorOfArray([getindex.(path,Comp), Res, priors, Converged]) : VectorOfArray([getindex.(path,Comp), Res, Converged])
    (ResMat, path)
end

function IntegrationProfileArm(LogLikelihoodFn::Function, MLE::AbstractVector{<:Number}, Comp::Int; ADmode::Val=Val(:ForwardDiff), Left::Bool=false,
                CostHessian::Function=GetHess!(ADmode, Negate(LogLikelihoodFn)),
                γ::Union{Nothing,Real}=nothing, CostGradient::Union{Nothing,Function}=isnothing(γ) ? nothing : GetGrad!(ADmode, LogLikelihoodFn),
                logLikeMLE::Real=LogLikelihoodFn(MLE), Confnum::Number=2, dof::Real=length(MLE), verbose::Bool=true, 
                IC::Real=eltype(MLE)(InvChisqCDF(dof, ConfVol(Confnum); maxval=1e8)), MinSafetyFactor::Real=1.05,
                Domain::Union{Nothing, HyperCube}=nothing, ProfileDomain::Union{Nothing, HyperCube}=Domain,
                Endtime::Real=1e2, psi_span::Tuple{<:Number,<:Number}=(MLE[Comp], Left ? MLE[Comp] -Endtime : MLE[Comp] +Endtime),
                meth::AbstractODEAlgorithm=BS3(), tol::Real=1e-3, reltol::Real=tol, abstol::Real=tol, kwargs...)

    @inline AddGradient!(Hλψ, γ::Nothing, CostGradient, G, θ, λ_indices) = nothing
    @inline function AddGradient!(Hλψ::AbstractVector, γ::Number, CostGradient::Function, G::AbstractVector, θ::AbstractVector, λ_indices)
        CostGradient(G, θ);     Hλψ .+= γ .* (@view G[λ_indices]);      nothing
    end
    function ProfileODE!(dλ_dψ::AbstractVector{<:Number}, λ::AbstractVector{<:Number}, params, ψ::Number)
        (; θ, H, G, Comp, λ_indices, CostHessian, CostGradient, γ) = params
        θ[Comp] = ψ;    θ[λ_indices] .= λ;     CostHessian(H, θ)
        Hλλ = (@view H[λ_indices, λ_indices]);     Hλψ = (@view H[λ_indices, Comp])
        ## Original Chen Jennrich with γ:  (Should not use γ > 0 unless accuracy of Hessian low)
        ## dλ_dψ .= -(Hλλ \ (Hλψ .+ γ .* (@view (GetGrad(ADmode,LogLikelihoodFn)(θ))[λ_indices])))
        AddGradient!(Hλψ, γ, CostGradient, G, θ, λ_indices)
        try
            dλ_dψ .= -(Hλλ \ Hλψ)
            # F = cholesky!(Symmetric(Hλλ); check=false)
            # ldiv!(dλ_dψ, F, Hλψ)
            # dλ_dψ .*= -1
        catch E;
            verbose && println("Error happened in profile $Comp at p=$ψ: $E")
            mul!(dλ_dψ, -pinv(Hλλ), Hλψ)
        end
        nothing
    end
    n = length(MLE);    λ_indices = setdiff(1:n, Comp)
    LogLikeThreshold = logLikeMLE - 0.5 * IC * MinSafetyFactor
    H = Matrix{eltype(MLE)}(undef, n, n);   G = isnothing(CostGradient) ? nothing : (@assert γ ≥ 0;   Vector{eltype(MLE)}(undef, n))
    
    params = (θ=copy(MLE), H=H, G=G, Comp=Comp, λ_indices=λ_indices, CostHessian=CostHessian, CostGradient=CostGradient, γ=γ, LogLikelihoodFn=LogLikelihoodFn, LogLikeThreshold=LogLikeThreshold)
    prob = ODEProblem(ODEFunction(ProfileODE!), MLE[λ_indices], psi_span, params)

    function EarlyTermination(λ::AbstractVector{<:Number}, ψ::Number, int)
        (; LogLikelihoodFn, θ, Comp, λ_indices, LogLikeThreshold) = int.p
        θ[Comp] = ψ;    θ[λ_indices] .= λ;        LogLikelihoodFn(θ) - LogLikeThreshold
    end
    DomainTermination = if !isnothing(ProfileDomain)
        if Left
            isfinite(ProfileDomain.L[Comp]) ? ContinuousCallback((λ,ψ,int)->(ψ-ProfileDomain.L[Comp]), terminate!) : nothing
        else
            isfinite(ProfileDomain.U[Comp]) ? ContinuousCallback((λ,ψ,int)->(ProfileDomain.U[Comp]-ψ), terminate!) : nothing
        end
    else nothing end
    solve(prob, meth; callback=CallbackSet(ContinuousCallback(EarlyTermination, terminate!), DomainTermination), reltol, abstol, kwargs...)
end





# Empty trafo name for identity
_GetTrafoName(F::typeof(identity), GenericName::Bool=false) = ""

## Pass through name as is - define this to avoid prefix InformationGeometry.BiLog with generic method
## Can be overloaded individually for custom trafos
# for Fname in [:BiLog, :BiRoot, :BiExp, :BiPower]
#     @eval _GetTrafoName(F::typeof($Fname), GenericName::Bool=false) = string($Fname)
# end

# Generic method
_GetTrafoName(F::Function, GenericName::Bool=false) = GenericName ? "Trafo" : string(Symbol(F))
# Nice function composition
_GetTrafoName(F::ComposedFunction, GenericName::Bool=false) = _GetTrafoName(F.outer, GenericName) *"∘"*_GetTrafoName(F.inner, GenericName)

function GetTrafoNames(F::Function, GenericName::Bool=false)
    TrafoName = _GetTrafoName(F, GenericName)
    length(TrafoName) > 0 && (TrafoName *= "(")
    TrafoName, (endswith(TrafoName, "(") ? ")" : "")
end
ApplyTrafoNames(S::AbstractString, F::Function; GenericName::Bool=false) = ((TrafoName, TrafoNameEnd)=GetTrafoNames(F, GenericName);    TrafoName * S * TrafoNameEnd)


# Size layout plot according to number of plots
PlotSizer(n::Int; size::Tuple{<:Int,<:Int}=(250,250)) = (s=Int(ceil(sqrt(n)));  s .* size)


@recipe f(P::Union{ParameterProfiles, ParameterProfilesView}, S::Symbol, args...) = P, Val(S), args...

# Plot trajectories by default
@recipe f(P::ParameterProfiles, PlotTrajectories::Bool=false) = P, Val(PlotTrajectories)

# DoBiLog for paths, i.e. TrafoPath
@recipe function f(P::ParameterProfiles, HasTrajectories::Val{true})
    Trafo = get(plotattributes, :Trafo, identity)
    DoBiLog = get(plotattributes, :BiLog, true)
    TrafoPath = get(plotattributes, :TrafoPath, DoBiLog ? BiLog : identity)

    layout := sum(IsPopulated(P)) + 1
    size --> PlotSizer(sum(IsPopulated(P)) + 1)

    @series begin
        layout := sum(IsPopulated(P)) + 1
        Trafo := Trafo
        P, Val(false)
    end

    @series begin
        subplot := sum(IsPopulated(P)) + 1
        idxs := get(plotattributes, :idxs, length(MLE(P))≥3 ? (1,2,3) : (1,2))
        legend --> nothing
        TrafoPath := TrafoPath
        P, Val(:PlotParameterTrajectories)
    end
end


@recipe function f(P::ParameterProfiles, HasTrajectories::Val{false})
    PopulatedInds = IsPopulated(P)
    layout --> sum(PopulatedInds)
    size --> PlotSizer(sum(PopulatedInds))

    P.Meta !== :ParameterProfiles && (plot_title --> string(P.Meta))
    Trafo = get(plotattributes, :Trafo, identity)
    tol = 0.05
    M = [maximum(view(T[2], GetConverged(T))) for T in Profiles(P) if !all(isnan, T[1]) && any(GetConverged(T)) && maximum(view(T[2], GetConverged(T))) > tol]
    maxy = length(M) > 0 ? median(M) : median([maximum(T[2]) for T in Profiles(P) if !all(isnan, T[1])])
    maxy = maxy < tol ? (maxy < 1e-8 ? tol : Inf) : maxy
    Ylims = get(plotattributes, :ylims, (Trafo.(-tol), Trafo.(maxy)))
    j = 1
    for i in eachindex(Profiles(P))
        if PopulatedInds[i]
            @series begin
                subplot := j
                ylims --> Ylims
                Trafo := Trafo
                ParameterProfilesView(P, i), Val(false)
            end
            j += 1
        end
    end
end

PlotProfileTrajectories(P::ParameterProfiles, args...; kwargs...) = RecipesBase.plot(P, args..., Val(:PlotParameterTrajectories); kwargs...)


@recipe function f(P::ParameterProfiles, trueparams::AbstractVector, ::Val{false}=Val(false))
    @assert length(trueparams) == length(Profiles(P))
    PopulatedInds = IsPopulated(P) 
    layout --> sum(PopulatedInds)
    size --> PlotSizer(sum(PopulatedInds))
    Interpolate = get(plotattributes, :Interpolate, false)
    Trafo = get(plotattributes, :Trafo, identity)
    tol = 0.05
    M = [maximum(view(T[2], GetConverged(T))) for T in Profiles(P) if !all(isnan, T[1]) && any(GetConverged(T)) && maximum(view(T[2], GetConverged(T))) > tol]
    maxy = length(M) > 0 ? median(M) : median([maximum(T[2]) for T in Profiles(P) if !all(isnan, T[1])])
    maxy = maxy < tol ? (maxy < 1e-8 ? tol : Inf) : maxy
    Ylims = get(plotattributes, :ylims, (Trafo.(-tol), Trafo.(maxy)))
    @series begin
        ylims --> Ylims
        Trafo := Trafo
        Interpolate := Interpolate
        P
    end
    j = 1
    for i in eachindex(trueparams)
        if PopulatedInds[i]
            @series begin
                subplot := j
                st := :vline
                line --> :dash
                color --> j+3
                label --> "True value"
                ylims --> Ylims
                xlabel --> pnames(P)[i]
                ylabel --> ApplyTrafoNames(IsCost(P) ? "W = 2[ℓ_mle - ℓ(θ)]" : "Conf. level [σ]", Trafo)
                @view trueparams[i:i]
            end
            j += 1
        end
    end
end


# BiLog kwarg for rescaling plotted trajectories
@recipe function f(P::ParameterProfiles, ::Val{:PlotParameterTrajectories})
    @assert HasTrajectories(P)

    idxs = get(plotattributes, :idxs, length(MLE(P))≥3 ? (1,2,3) : (1,2))
    if !((2 ≤ length(idxs) ≤ 3 && allunique(idxs) && all(1 .≤ idxs .≤ pdim(P))))
        @warn "Ignoring given idxs=$idxs because unsuitable."
        idxs = length(MLE(P))≥3 ? (1,2,3) : (1,2)
    end


    InterpolatePaths = get(plotattributes, :InterpolatePaths, false)
    Interp = QuadraticSpline
    # Should do rescaling with diagonal sqrt inv Fisher instead of BiLog
    DoBiLog = get(plotattributes, :BiLog, true)
    TrafoPath = get(plotattributes, :TrafoPath, DoBiLog ? BiLog : identity)
    TrafoName, TrafoNameEnd = GetTrafoNames(TrafoPath)
    xlabel --> TrafoName * pnames(P)[idxs[1]] * TrafoNameEnd
    ylabel --> TrafoName * pnames(P)[idxs[2]] * TrafoNameEnd
    if length(idxs) == 3
        zlabel --> TrafoName * pnames(P)[idxs[3]] * TrafoNameEnd
    end
    
    color_palette = get(plotattributes, :color_palette, :default)
    for i in eachindex(Profiles(P))
        if !isnothing(Trajectories(P)[i])
            @series begin
                label --> "Comp $i"
                color --> palette(color_palette)[(((2+i) % 15) +1)]
                lw --> 1.5
                M = Unpack(map(x->getindex(x, collect(idxs)), Trajectories(P)[i]))
                if length(idxs) == 3
                    TrafoPath.(view(M,:,1)), TrafoPath.(view(M,:,2)), TrafoPath.(view(M,:,3))
                else
                    if InterpolatePaths
                        Conv = GetConverged(Profiles(P[i]))
                        !all(Conv) && @warn "Interpolating profile $i but $(sum(.!Conv))/$(length(Conv)) points not converged."
                        F = Interp(TrafoPath.(view(M,:,2)), TrafoPath.(view(M,:,1)))
                        xran = range(F.t[1], F.t[end]; length=300)
                        xran, map(F, xran)
                    else
                        TrafoPath.(view(M,:,1)), TrafoPath.(view(M,:,2))
                    end
                end
            end
        end
    end
    @series begin
        label --> nothing
        markercolor --> :red
        marker --> :hex
        markersize --> 2.5
        markerstrokewidth --> 0
        [TrafoPath.(MLE(P)[collect(idxs)])]
    end
end

# Try to plot Trajectories if available
@recipe f(PV::ParameterProfilesView, PlotTrajectories::Bool=HasTrajectories(PV) && PV.P.Meta === :ParameterProfiles && length(Trajectories(PV)[1]) ≤ 3) = PV, Val(PlotTrajectories)

@recipe function f(PVs::AbstractVector{<:ParameterProfilesView}, V::Val=Val(false))
    layout --> length(PVs)
    size --> PlotSizer(length(PVs))
    Interpolate = get(plotattributes, :Interpolate, false)
    Trafo = get(plotattributes, :Trafo, identity)
    for i in eachindex(PVs)
        @series begin
            subplot := i
            Trafo := Trafo
            Interpolate := Interpolate
            PVs[i], V
        end
    end
end

# MaxLevel kwarg for checking which is the highest profile value which is still converged
# dof kwarg for plotting Confidence Levels
# Confnum kwarg for plotting specific levels
@recipe function f(PV::ParameterProfilesView, WithTrajectories::Val{false})
    i = PV.i
    Trafo = get(plotattributes, :Trafo, identity)
    legend --> nothing
    xguide --> pnames(PV)[i]
    yguide --> ApplyTrafoNames(IsCost(PV) ? "W = 2[ℓ_mle - ℓ(θ)]" : "Conf. level [σ]", Trafo)

    Interpolate = get(plotattributes, :Interpolate, false)
    Interp = QuadraticSpline
    @series begin
        lw --> 1.5
        if Interpolate
            label --> "Interpolated Profile"
            Conv = GetConverged(Profiles(PV))
            !all(Conv) && @warn "Interpolating profile $i but $(sum(.!Conv))/$(length(Conv)) points not converged."
            F = InterpolatedProfiles(PV, Interp)
            xran = range(F.t[1], F.t[end]; length=300)
            xran, map(Trafo∘F, xran)
        else
            label --> ["Profile Likelihood" nothing]
            Profiles(PV)[1], Trafo.(Convergify(Profiles(PV)[2], GetConverged(Profiles(PV))))
        end
    end
    # Draw prior contribution
    if HasPriors(PV)
        @series begin
            linealpha --> 0.75
            line --> :dash
            lw --> 1.5
            if Interpolate
                label --> "Interpolated Profile"
                Conv = GetConverged(Profiles(PV))
                !all(Conv) && @warn "Interpolating profile $i but $(sum(.!Conv))/$(length(Conv)) points not converged."
                F = Interp(Profiles(PV)[3], Profiles(PV)[1])
                xran = range(F.t[1], F.t[end]; length=300)
                xran, map(Trafo∘F, xran)
            else
                label --> ["Prior contribution" nothing]
                Profiles(PV)[1], Trafo.(Convergify(Profiles(PV)[3], GetConverged(Profiles(PV))))
            end
        end
    end
    ## Mark MLE in profile
    @series begin
        label --> nothing
        markercolor --> :red
        marker --> :hex
        markersize --> 2.5
        markerstrokewidth --> 0
        [MLE(PV)[i]], [Trafo.(0.0)]
    end
    # Mark threshold if not already rescaled to confidence scale
    Confnum = get(plotattributes, :Confnum, 1:5)
    if IsCost(PV) && all(Confnum .> 0)
        dof = get(plotattributes, :dof, DOF(PV))
        MaxLevel = get(plotattributes, :MaxLevel, maximum(view(Profiles(PV)[2],GetConverged(Profiles(PV))); init=-Inf))
        for (j,Thresh) in Iterators.zip(sort(Confnum; rev=true), convert.(eltype(MLE(PV)), InvChisqCDF.(dof, ConfVol.(sort(Confnum; rev=true)))))
            if Thresh < MaxLevel
                @series begin
                    st := :hline
                    line --> :dash
                    lw := 1.5
                    linecolor := palette(:viridis, length(Confnum); rev=true)[j]
                    label --> "$(j)σ level, dof=$dof"
                    [Trafo.(Thresh)]
                end
            end
        end
    end
end

@recipe function f(PV::ParameterProfilesView, WithTrajectories::Val{true})
    layout --> (2,1)
    Interpolate = get(plotattributes, :Interpolate, false)
    InterpolatePaths = get(plotattributes, :InterpolatePaths, Interpolate)
    Trafo = get(plotattributes, :Trafo, identity)
    DoBiLog = get(plotattributes, :BiLog, true)
    TrafoPath = get(plotattributes, :TrafoPath, DoBiLog ? BiLog : identity)
    @series begin
        subplot := 1
        Trafo := Trafo
        Interpolate := Interpolate
        PV, Val(false)
    end
    @series begin
        subplot := 2
        TrafoPath := TrafoPath
        InterpolatePaths := InterpolatePaths
        PV, Val(:PlotRelativeParamTrajectories)
    end
end


## Allow for plotting other scalar functions of the parameters, e.g. steady-state constraints
@recipe function f(P::ParameterProfiles, V::Union{Val{:PlotRelativeParamTrajectories},Val{:ProfilePaths}, Val{:ProfilePathDiffs},Val{:PathDiffs}, Val{:ProfilePathNormDiffs},Val{:PathNormDiffs}})
    @assert HasTrajectories(P)
    RelChange = get(plotattributes, :RelChange, false)
    idxs = get(plotattributes, :idxs, 1:pdim(P))
    mle = get(plotattributes, :MLE, MLE(P))
    OffsetResults = get(plotattributes, :OffsetResults, true)

    ParameterFunctions = get(plotattributes, :ParameterFunctions, nothing)
    @assert RelChange isa Bool
    @assert all(1 .≤ idxs .≤ pdim(P)) && allunique(idxs)

    ToPlotInds = [i for i in eachindex(MLE(P)) if i ∈ idxs && IsPopulated(P[i]) && !isnothing(Trajectories(P)[i])]

    layout --> length(ToPlotInds)
    size --> PlotSizer(length(ToPlotInds))
    DoBiLog = get(plotattributes, :BiLog, false)
    TrafoPath = get(plotattributes, :TrafoPath, DoBiLog ? BiLog : identity)

    for (plotnum, i) in enumerate(ToPlotInds)
        @series begin
            subplot := plotnum
            TrafoPath := TrafoPath
            RelChange --> RelChange
            idxs --> ToPlotInds
            MLE --> mle
            OffsetResults --> OffsetResults
            ParameterFunctions --> ParameterFunctions
            P[i], V
        end
    end
end

# Kwarg BiLog=true for BiLog scale
# Kwarg RelChange=false for parameter difference instead of ratio to MLE
# Kwarg idxs for trajectories to plot
# Kwarg MLE and OffsetResults
@recipe function f(PV::ParameterProfilesView, ::Union{Val{:PlotRelativeParamTrajectories},Val{:ProfilePaths}})
    @assert HasTrajectories(PV)
    RelChange = get(plotattributes, :RelChange, false)
    @assert RelChange isa Bool
    DoRelChange = RelChange && !any(MLE(PV) .== 0)
    Fisher = get(plotattributes, :Fisher, Diagonal(Ones(pdim(PV))))
    U = cholesky(Fisher).U
    ParameterFunctions = get(plotattributes, :ParameterFunctions, nothing)

    idxs = get(plotattributes, :idxs, 1:pdim(PV))
    @assert all(1 .≤ idxs .≤ pdim(PV)) && allunique(idxs)
    i = PV.i
    xguide --> pnames(PV)[i]

    mle = get(plotattributes, :MLE, MLE(PV))
    OffsetResults = get(plotattributes, :OffsetResults, true)
    DoBiLog = get(plotattributes, :BiLog, false)
    TrafoPath = get(plotattributes, :TrafoPath, DoBiLog ? BiLog : identity)
    ystring = DoRelChange ? "p_i" * (OffsetResults ? " / p_mle" : "") :  (U != Diagonal(Ones(pdim(PV))) ? "F^(1/2) * [p_i" * (OffsetResults ? " - p_mle" : "") * "]" : "p_i" * (OffsetResults ? " - p_mle" : ""))
    yguide --> ApplyTrafoNames(ystring, TrafoPath)
    # Also filter out 
    ToPlotInds = idxs[idxs .!= i]
    color_palette = get(plotattributes, :color_palette, :default)
    InterpolatePaths = get(plotattributes, :InterpolatePaths, false)
    Interp = QuadraticSpline
    # Colorize only parameters with 5 strongest changes
    for j in ToPlotInds
        @series begin
            color --> palette(color_palette)[(((2+j) % 15) +1)]
            label --> pnames(PV.P)[j]
            lw --> 1.5
            Change = if DoRelChange
                getindex.(Trajectories(PV), j) ./ (OffsetResults ? mle[j] : 1)
            else
                U[j,j] .* (getindex.(Trajectories(PV), j) .- (OffsetResults ? mle[j] : 0))
            end
            if InterpolatePaths
                Conv = GetConverged(Profiles(PV))
                !all(Conv) && @warn "Interpolating profile $i but $(sum(.!Conv))/$(length(Conv)) points not converged."
                F = Interp(TrafoPath.(Change), getindex.(Trajectories(PV), i))
                xran = range(F.t[1], F.t[end]; length=300)
                xran, map(F, xran)
            else
                getindex.(Trajectories(PV), i), TrafoPath.(Change)
            end
        end
    end
    # if !isnothing(ParameterFunctions)
    #     @assert ParameterFunctions isa AbstractArray{<:Function}
    #     for (j,F) in enumerate(ParameterFunctions)
    #         Vals = map(F, getindex.(Trajectories(PV), i))
    #         @series begin
    #             color --> palette(color_palette)[(((2+j+length(ToPlotInds)) % 15) +1)]
    #             label --> ParameterFunctions isa ParamTrafo ? string(ParameterFunctions.ConditionNames[j]) : "ParameterFunction $j"
    #             lw --> 1.5
    #             Change = if RelChange && !any(MLE(PV) .== 0)
    #             getindex.(Trajectories(PV), j) ./ MLE(PV)[j]
    #         else
    #             U[j,j] .* (getindex.(Trajectories(PV), j) .- MLE(PV)[j])
    #         end
    #             getindex.(Trajectories(PV), i), TrafoPath(Change)
    #         end
    #     end
    # end
    if OffsetResults
        # Mark MLE
        @series begin
            label --> nothing
            seriescolor --> :red
            marker --> :hex
            markersize --> 2.5
            markerstrokewidth --> 0
            [MLE(PV)[i]], (DoRelChange ? [TrafoPath(1.0)] : [TrafoPath(0.0)])
        end
    end
end

PlotProfilePaths(PV::Union{ParameterProfiles,ParameterProfilesView}; kwargs...) = RecipesBase.plot(PV, Val(:ProfilePaths); kwargs...)
@deprecate PlotRelativeParameterTrajectories PlotProfilePaths


@recipe function f(PV::ParameterProfilesView, ::Union{Val{:ProfilePathDiffs},Val{:PathDiffs}})
    @assert HasTrajectories(PV)
    RelChange = get(plotattributes, :RelChange, false)
    idxs = get(plotattributes, :idxs, 1:pdim(PV))
    mle = get(plotattributes, :MLE, MLE(PV))
    OffsetResults = get(plotattributes, :OffsetResults, true)
    ParameterFunctions = get(plotattributes, :ParameterFunctions, nothing)
    StepTol = get(plotattributes, :StepTol, 5)
    verbose = get(plotattributes, :verbose, true)

    idxs = get(plotattributes, :idxs, 1:pdim(PV))
    @assert all(1 .≤ idxs .≤ pdim(PV)) && allunique(idxs)
    i = PV.i
    xguide --> pnames(PV)[i]

    mle = get(plotattributes, :MLE, MLE(PV))
    OffsetResults = get(plotattributes, :OffsetResults, true)
    DoBiLog = get(plotattributes, :BiLog, false)
    TrafoPath = get(plotattributes, :TrafoPath, DoBiLog ? BiLog : identity)
    #ystring = DoRelChange ? "p_i" * (OffsetResults ? " / p_MLE" : "") :  (U != Diagonal(Ones(pdim(PV))) ? "F^(1/2) * [p_i" * (OffsetResults ? " - p_MLE" : "") * "]" : "p_i" * (OffsetResults ? " - p_MLE" : ""))
    ystring = "Finite Differences"
    yguide --> ApplyTrafoNames(ystring, TrafoPath)
    # Also filter out 
    ToPlotInds = idxs[idxs .!= i]
    color_palette = get(plotattributes, :color_palette, :default)
    # Colorize only parameters with 5 strongest changes
    @series begin
        st --> :vline
        label --> "MLE"
        seriescolor --> :red
        line --> :dash
        lw --> 1.5
        [MLE(PV)[i]]
    end
    for j in ToPlotInds
        @series begin
            color --> palette(color_palette)[(((2+j) % 15) +1)]
            label --> pnames(PV.P)[j]
            lw --> 1.5
            yDiffs = diff(getindex.(Trajectories(PV), j));      xDiffs = diff(getindex.(Trajectories(PV), i))
            FiniteDiffs = yDiffs ./ xDiffs
            X = CenteredVec(getindex.(Trajectories(PV), i))
            if verbose && any(abs.(median(TrafoPath.(FiniteDiffs)) .- collect(extrema(TrafoPath.(FiniteDiffs)))) .> StepTol)
                Jumps = X[abs.(median(TrafoPath.(FiniteDiffs)) .- TrafoPath.(FiniteDiffs)) .> StepTol]
                @info "Detected possible discrete jump"*(length(Jumps) > 1 ? "s" : "")*" in trajectory of parameter "*string(STRING_COLOR, pnames(PV)[j], NO_COLOR)*" of parameter profile "*string(STRING_COLOR, pnames(PV)[i], NO_COLOR)*" at: $(Jumps)"
            end
            X, TrafoPath.(FiniteDiffs)
        end
    end
end

PlotProfilePathDiffs(PV::Union{ParameterProfiles,ParameterProfilesView}; kwargs...) = RecipesBase.plot(PV, Val(:ProfilePathDiffs); kwargs...)

@recipe function f(PV::ParameterProfilesView, ::Union{Val{:ProfilePathNormDiffs},Val{:PathNormDiffs}})
    @assert HasTrajectories(PV)
    RelChange = get(plotattributes, :RelChange, false)
    idxs = get(plotattributes, :idxs, 1:pdim(PV))
    mle = get(plotattributes, :MLE, MLE(PV))
    OffsetResults = get(plotattributes, :OffsetResults, true)
    ParameterFunctions = get(plotattributes, :ParameterFunctions, nothing)
    StepTol = get(plotattributes, :StepTol, 5)
    verbose = get(plotattributes, :verbose, true)
    pnorm = get(plotattributes, :pnorm, 2)

    idxs = get(plotattributes, :idxs, 1:pdim(PV))
    @assert all(1 .≤ idxs .≤ pdim(PV)) && allunique(idxs)
    i = PV.i
    xguide --> pnames(PV)[i]

    mle = get(plotattributes, :MLE, MLE(PV))
    OffsetResults = get(plotattributes, :OffsetResults, true)
    DoBiLog = get(plotattributes, :BiLog, false)
    TrafoPath = get(plotattributes, :TrafoPath, DoBiLog ? BiLog : identity)
    #ystring = DoRelChange ? "p_i" * (OffsetResults ? " / p_MLE" : "") :  (U != Diagonal(Ones(pdim(PV))) ? "F^(1/2) * [p_i" * (OffsetResults ? " - p_MLE" : "") * "]" : "p_i" * (OffsetResults ? " - p_MLE" : ""))
    ystring = "Finite Differences $pnorm-norm"
    yguide --> ApplyTrafoNames(ystring, TrafoPath)
    # Consider all idxs
    ToPlotInds = idxs
    @series begin
        st --> :vline
        label --> "MLE"
        seriescolor --> :red
        line --> :dash
        lw --> 1.5
        [MLE(PV)[i]]
    end
    @series begin
        label --> "|| Δp⃗ ||_$pnorm / Δ($(pnames(PV)[i]))"
        lw --> 1.5
        yDiffs = diff(Trajectories(PV));      xDiffs = diff(getindex.(Trajectories(PV), i))
        FiniteDiffs = yDiffs ./ xDiffs;     NormFiniteDiffs = norm.(FiniteDiffs, pnorm)
        X = CenteredVec(getindex.(Trajectories(PV), i))
        if verbose && any(abs.(median(TrafoPath.(NormFiniteDiffs)) .- collect(extrema(TrafoPath.(NormFiniteDiffs)))) .> StepTol)
            Jumps = X[abs.(median(TrafoPath.(NormFiniteDiffs)) .- TrafoPath.(NormFiniteDiffs)) .> StepTol]
            @info "Detected possible discrete jump"*(length(Jumps) > 1 ? "s" : "")*" in trajectory of parameter profile "*string(STRING_COLOR, pnames(PV)[i], NO_COLOR)*" at: $(Jumps)"
        end
        X, TrafoPath.(NormFiniteDiffs)
    end
end

PlotProfilePathNormDiffs(PV::Union{ParameterProfiles,ParameterProfilesView}; kwargs...) = RecipesBase.plot(PV, Val(:ProfilePathNormDiffs); kwargs...)

for F in [:PlotProfilePaths, :PlotProfileTrajectories, :PlotProfilePathDiffs, :PlotProfilePathNormDiffs]
    @eval InformationGeometry.$F(DM::AbstractDataModel, args...; plot::Bool=false, kwargs...) = InformationGeometry.$F(ParameterProfiles(DM, args...; plot); kwargs...)
end



# Bad style but works for now for plotting profiles from different models in one:
function RecipesBase.plot(Ps::AbstractArray{<:Union{ParameterProfiles, ParameterProfilesView}}; kwargs...)
    Plts = [RecipesBase.plot(Ps[i]) for i in eachindex(Ps)]
    RecipesBase.plot(Plts...; layout=length(Plts), size=PlotSizer(length(Plts)), kwargs...)
end


# Generate validation profile centered on prediction at a single independent variable t
function GetValidationProfilePoint(DM::AbstractDataModel, yComp::Int, t::Union{AbstractVector{<:Number},Number}; Confnum::Real=2, N::Int=21, LogLikelihoodFn::Function=loglikelihood(DM),
                                MLE::AbstractVector{<:Number}=MLE(DM), ypred::Real=Predictor(DM)(t,MLE)[yComp], yoffset::Real=ypred, 
                                dof::Int=DOF(DM), LinPredictionUncert::Real=(C=VariancePropagation(DM, MLE; Confnum, dof)(t);   ydim(DM)>1 ? C[yComp, yComp] : C),
                                DivideBy::Real=5, σv::Real=LinPredictionUncert/DivideBy, IC::Real=InvChisqCDF(dof, ConfVol(Confnum)), ValidationSafetyFactor::Real=2, kwargs...) # Make Confnumsafety ratio σv/(obs + σv) to decrease computations when prediction profiles are desired?
    @assert IC > 0 && dof > 0
    M = Predictor(DM);    FicticiousPoint = Normal(0, σv)
    FictDataPointPrior(θnew::AbstractVector) = (θ=view(θnew, 1:lastindex(θnew)-1);   logpdf(FicticiousPoint, θnew[end] - M(t, θ)[yComp] + yoffset))
    VPL(θnew::AbstractVector) = LogLikelihoodFn(view(θnew, 1:lastindex(θnew)-1)) + FictDataPointPrior(θnew)
    mleNew = [MLE; (ypred-yoffset)];    Fisher = Diagonal(Fill(σv^-2,pdim(DM)+1))
    B = ValidationSafetyFactor*σv*sqrt(2*IC);    Ran = range(-B + (ypred-yoffset), B + (ypred-yoffset); length=N)
    GetProfile(DM, pdim(DM)+1, Ran; LogLikelihoodFn=VPL, LogPriorFn=FictDataPointPrior, dof, MLE=mleNew, logLikeMLE=VPL(mleNew), Fisher, Confnum, N, IsCost=true, Domain=nothing, InDomain=nothing, AllowNewMLE=false, general=true, SavePriors=true, kwargs...)
end

# Generate multiple validation profiles and add back offset to prediction scale
"""
    ValidationProfiles(DM::AbstractDataModel, yComp::Int, Ts::AbstractVector; Confnum::Real=2, dof::Int=DOF(DM), OffsetToZero::Bool=false, kwargs...)
Computes a set of validation profiles for the component `yComp` of the prediction at various values of the independent variables `Ts`.
The uncertainty of the ficticious validation data point can be optionally chosen via the keyword argument `σv`.
Most other kwargs are passed on to the `ParameterProfiles` function and thereby also to the optimizers, see e.g. [`ParameterProfiles`](@ref), [`InformationGeometry.minimize`](@ref).
"""
function ValidationProfiles(DM::AbstractDataModel, yComp::Int, Ts::AbstractVector=range(extrema(xdata(DM))...; length=3length(xdata(DM))); dof::Int=DOF(DM), MLE::AbstractVector{<:Number}=MLE(DM), IsCost::Bool=true, OffsetToZero::Bool=false, Meta=:ValidationProfiles, parallel::Bool=true, verbose::Bool=true, kwargs...)
    ypreds = [Predictor(DM)(t,MLE)[yComp] for t in Ts]      # Always compute with offset to zero internally
    Res = (parallel ? progress_pmap : progress_map)(i->GetValidationProfilePoint(DM, yComp, Ts[i]; ypred=ypreds[i], dof=dof, MLE, IsCost, verbose, kwargs...), 1:length(Ts); progress=Progress(length(Ts); enabled=verbose, desc="Computing Validation Profiles... (parallel, $(nworkers()) workers) ", dt=1, showspeed=true))
    Profs, Trajs = getindex.(Res,1), getindex.(Res,2)
    for i in eachindex(Ts)
        zProf = map(TrajPoint->Predictor(DM)(Ts[i], (@view TrajPoint[1:end-1]))[yComp], Trajs[i])
        Profs[i] = @views hcat(Profs[i][:,1:end-1], zProf, Profs[i][:,end])
    end
    # If should remain on true scale, add ypreds back on
    offsetvec = OffsetToZero ? Zeros(length(Ts)) : ypreds
    for i in eachindex(Ts)   Profs[i][:,1] .+= offsetvec[i];     for j in 1:size(Trajs[i],1)    Trajs[i][j][end] += offsetvec[i]    end    end
    VPL = "VPL"*(ydim(DM) > 1 ? "[$(yComp)]" : "");   ParameterProfiles(Profs, Trajs, [VPL*"($(Ts[i]))" for i in eachindex(Ts)], offsetvec, dof, IsCost; Meta)
end
ValidationProfiles(DM::AbstractDataModel, yComp::Int, t::Number; kwargs...) = ValidationProfile(DM, yComp, [t]; kwargs...)

# Add virtual point to validation profile again to obtain prediction profile
"""
    ConvertValidationToPredictionProfiles(VP::ParameterProfiles; kwargs...)
Converts a `ParameterProfile` object encoding a validation profile into a prediction profile by subtracting off the effect introduced by the ficticious validation data point.
"""
function ConvertValidationToPredictionProfiles(VP::ParameterProfiles; kwargs...)
    @assert HasPriors(VP) && IsCost(VP)
    @assert VP.Meta === :ValidationProfiles
    @assert all(size.(Profiles(VP),2) .> 4) # Make sure z-values saved in the Profile matrix, as well as priors
    # Use fourth column with z-values instead of original v-parameters from validation profile
    Profs = [(M=P[:, [4,2,size(P,2)]];   M[:,2] .-= P[:, 3]; M) for P in Profiles(VP)]
    remake(VP; Profiles=Profs, Meta=:PredictionProfiles, Names=Symbol.(map(x->replace(x, "VPL"=>"PPL"), string.(VP.Names))), kwargs...)
end
"""
    PredictionProfiles(DM::AbstractDataModel, yComp::Int, Ts::AbstractVector; Confnum::Real=2, dof::Int=DOF(DM), OffsetToZero::Bool=false, kwargs...)
Computes a set of prediction profiles for the component `yComp` of the prediction at various values of the independent variables `Ts`.
The prediction profiles are computed by means of intermediately generated validation profiles [`ValidationProfiles`](@ref).
Most other kwargs are passed on to the `ParameterProfiles` function and thereby also to the optimizers, see e.g. [`ParameterProfiles`](@ref), [`InformationGeometry.minimize`](@ref).
"""
PredictionProfiles(args...; DivideBy::Real=10, kwargs...) = ValidationProfiles(args...; DivideBy, kwargs...) |> ConvertValidationToPredictionProfiles


