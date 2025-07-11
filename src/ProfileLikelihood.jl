

# Returns a copy of type `Vector`, i.e. is not typesafe!
SafeCopy(X::AbstractVector) = copy(X)
SafeCopy(X::AbstractVector{<:Num}) = X
SafeCopy(X::AbstractRange) = collect(X)
SafeCopy(X::Union{SVector,MVector}) = convert(Vector,X)

Drop(X::AbstractVector, i::Int) = (Z=SafeCopy(X);   splice!(Z,i);   Z)
Drop(X::ComponentVector, i::Int) = Drop(convert(Vector,X), i)
Drop(N::Nothing, i::Int) = nothing
Drop(C::HyperCube, i::Int) = (inds = 1:length(C) .!= i; HyperCube(view(C.L, inds), view(C.U, inds)))

_Presort(Components::AbstractVector{<:Int}; rev::Bool=false) = issorted(Components; rev=rev) ? Components : sort(Components; rev=rev)
Drop(X::AbstractVector, Components::AbstractVector{<:Int}) = (Z=SafeCopy(X); for i in _Presort(Components; rev=true) splice!(Z,i) end;    Z)
Drop(X::AbstractVector, Components::AbstractVector{<:Bool}) = (@assert length(X) == length(Components);    Drop(X, (1:length(Components))[Components]))

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

"""
    ValInserter(Component::Int, Value::AbstractFloat) -> Function
Returns an embedding function ``\\mathbb{R}^N \\longrightarrow \\mathbb{R}^{N+1}`` which inserts `Value` in the specified `Component`.
In effect, this allows one to pin an input component at a specific value.
"""
function ValInserter(Component::Int, Value::AbstractFloat, Z::T=Float64[]) where T <: AbstractVector{<:Number}
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
    function WrapFunctionWithInversion(F::Function, TestOut=F(mle))
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
function ValInserter(Components::AbstractVector{<:Int}, Values::AbstractVector{<:Number}, Z::T=Float64[]) where T <: AbstractVector{<:Number}
    @assert length(Components) == length(Values)
    length(Components) == 0 && return Identity(X::AbstractVector{<:Number}) = X
    if length(Components) ≥ 2 && Consecutive(Components) # consecutive components.
        ConsecutiveInsertionEmbedding(P::AbstractVector) = (Res=SafeCopy(P);  splice!(Res, Components[1]:Components[1]-1, Values);    Res)
    else
        # Sort components to avoid shifts in indices through repeated insertion.
        components, values = _SortTogether(Components, Values)
        function ValInsertionEmbedding(P::AbstractVector)
            Res = SafeCopy(P)
            for i in eachindex(components)
                insert!(Res, components[i], values[i])
            end;    Res
        end
        function ValInsertionEmbedding(P::Union{SVector,MVector})
            Res = insert(P, components[1], values[1])
            for i in 2:length(components)
                Res = insert(Res, components[i], values[i])
            end;    Res
        end
    end
end
function ValInserter(Components::AbstractVector{<:Bool}, Values::AbstractVector{<:Number}, X::T=Float64[]) where T <: AbstractVector{<:Number}
    @assert length(Components) == length(Values)
    ValInserter((1:length(Components))[Components], Values[Components])
end
function ValInserter(Components::AbstractVector{<:Int}, Value::Number, Z::T=Float64[]) where T <: AbstractVector{<:Number}
    length(Components) == 0 && return Identity(X::AbstractVector{<:Number}) = X
    components = sort(Components)
    function ValInsertionEmbedding(P::AbstractVector)
        Res = SafeCopy(P)
        for i in eachindex(components)
            insert!(Res, components[i], Value)
        end;    Res
    end
    function ValInsertionEmbedding(P::Union{SVector,MVector})
        Res = insert(P, components[1], Value)
        for i in 2:length(components)
            Res = insert(Res, components[i], Value)
        end;    Res
    end
    function ValInsertionEmbedding(P::AbstractVector{<:Num})
        Res = [P[1:components[1]-1]; Value]
        for i in 2:length(components)
            Res = @views [Res; P[components[i-1]:components[i]-1]; Value]
        end;    [Res; @view P[components[end]:end]]
    end
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
    PinParameters(DM::AbstractDataModel, Component::Int, Value::AbstractFloat=MLE(DM)[Component])
    PinParameters(DM::AbstractDataModel, Components::AbstractVector{<:Int}, Values::AbstractVector{<:AbstractFloat}=MLE(DM)[Components])
    PinParameters(DM::AbstractDataModel, ParamDict::Dict{String, Number})
Returns `DataModel` where one or more parameters have been pinned to specified values.
"""
function PinParameters(DM::AbstractDataModel, Components::Union{Int,AbstractVector{<:Int}}, Values::Union{AbstractFloat,AbstractVector{<:AbstractFloat}}=MLE(DM)[Components]; SkipOptim::Bool=false, SkipTests::Bool=false)
    @assert length(Components) == length(Values) && length(Components) < pdim(DM)
    length(Components) == 0 && (@warn "Got no parameters to pin.";  return DM)
    Pnames = [pnames(DM)[i] for i in eachindex(pnames(DM)) if i ∉ Components]
    DataModel(Data(DM), ProfilePredictor(DM, Components, Values; pnames=Pnames), ProfileDPredictor(DM, Components, Values; pnames=Pnames), Drop(MLE(DM), Components), EmbedLogPrior(DM, ValInserter(Components, Values)); SkipOptim, SkipTests)
end

function PinParameters(DM::AbstractDataModel, ParamDict::Dict{<:AbstractString, Number}; kwargs...)
    Comps = Int[];  Vals = []
    for i in 1:pdim(DM)
        pnames(DM)[i] ∈ keys(ParamDict) && (push!(Comps, i);  push!(Vals, ParamDict[pnames(DM)[i]]))
    end
    @assert length(Comps) > 0 "No overlap between parameters and given parameter dictionary: pnames=$(pnames(DM)), keys=$(keys(ParamDict))."
    PinParameters(DM, Comps, Vals; kwargs...)
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
function LinkParameters(DM::AbstractDataModel, Linked::AbstractVector{<:Bool}, MainIndBefore::Int=findfirst(Linked), args...; SkipOptim::Bool=false, SkipTests::Bool=false, kwargs...)
    DataModel(Data(DM), LinkParameters(Predictor(DM), Linked, MainIndBefore, args...; kwargs...), Drop(MLE(DM), _WithoutInd(Linked, MainIndBefore)), EmbedLogPrior(DM, GetLinkEmbedding(Linked,MainIndBefore)); SkipOptim, SkipTests)
end
function LinkParameters(M::ModelMap, Linked::AbstractVector{<:Bool}, MainIndBefore::Int=findfirst(Linked); kwargs...)
    @assert length(Linked) == pdim(M)
    WoFirst = _WithoutInd(Linked, MainIndBefore)
    PNames = copy(pnames(M))
    PNames[MainIndBefore] *= " =: " * join(pnames(M)[WoFirst], " ≡ ")
    PNames = PNames[.!WoFirst]
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

function GetProfileDomainCube(DM::AbstractDataModel, Confnum::Real; dof::Int=DOF(DM), kwargs...)
    Cube = GetProfileDomainCube(FisherMetric(DM, MLE(DM)), MLE(DM), Confnum; dof=dof, kwargs...)
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
        L = clamp.(L, 1e-10, 1e12)
        U = clamp.(U, 1e-10, 1e12)
    end
    HyperCube(L,U)
end

# USE NelderMead for ODEmodels!!!!!

GetMinimizer(Res::LsqFit.LsqFitResult) = Res.param
GetMinimum(Res::LsqFit.LsqFitResult, L::Function) = GetMinimum(GetMinimizer(Res), L)
HasConverged(Res::LsqFit.LsqFitResult; kwargs...) = Res.converged
GetIterations(Res::LsqFit.LsqFitResult) = try Res.trace[end].iteration catch; -Inf end # needs kwarg store_trace=true to be available

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
function GetProfile(DM::AbstractDataModel, Comp::Int, dom::Tuple{<:Real, <:Real}; adaptive::Bool=true, N::Int=31, kwargs...)
    @assert dom[1] < dom[2] && (1 ≤ Comp ≤ pdim(DM))
    ps = DomainSamples(dom; N=N)
    GetProfile(DM, Comp, ps; adaptive, N, kwargs...)
end


function GetProfile(DM::AbstractDataModel, Comp::Int, ps::AbstractVector{<:Real}; adaptive::Bool=true, Confnum::Real=2.0, N::Int=(adaptive ? 31 : length(ps)), min_steps::Int=Int(round(2N/5)), 
                        AllowNewMLE::Bool=true, general::Bool=true, IsCost::Bool=true, dof::Int=DOF(DM), SaveTrajectories::Bool=true, SavePriors::Bool=false, ApproximatePaths::Bool=false, 
                        LogLikelihoodFn::Function=loglikelihood(DM), CostFunction::Function=Negate(LogLikelihoodFn), UseGrad::Bool=true, CostGradient::Union{Function,Nothing}=(UseGrad ? NegScore(DM) : nothing),
                        UseHess::Bool=false, ADmode::Val=Val(:ForwardDiff), GenerateNewDerivatives::Bool=true,
                        CostHessian::Union{Function,Nothing}=(!UseHess ? nothing : (GenerateNewDerivatives ? AutoMetricFromNegScore(CostGradient; ADmode) : FisherMetric(DM))),
                        LogPriorFn::Union{Nothing,Function}=LogPrior(DM), mle::AbstractVector{<:Number}=MLE(DM), logLikeMLE::Real=LogLikeMLE(DM),
                        Fisher::Union{Nothing, AbstractMatrix}=(adaptive ? FisherMetric(DM, mle) : nothing), verbose::Bool=false, resort::Bool=true, Multistart::Int=0, maxval::Real=1e5, OnlyBreakOnBounds::Bool=false,
                        Domain::Union{Nothing, HyperCube}=GetDomain(DM), InDomain::Union{Nothing, Function}=GetInDomain(DM), ProfileDomain::Union{Nothing, HyperCube}=Domain, tol::Real=1e-9,
                        meth=((isnothing(LogPriorFn) && !general && Data(DM) isa AbstractFixedUncertaintyDataSet) ? nothing : Optim.NewtonTrustRegion()), OptimMeth=meth, OffsetResults::Bool=true,
                        IC::Real=eltype(mle)(InvChisqCDF(dof, ConfVol(Confnum); maxval=1e12)), MinSafetyFactor::Real=1.05, MaxSafetyFactor::Real=3,
                        stepfactor::Real=3.5, stepmemory::Real=0.2, terminatefactor::Real=10, flatstepconst::Real=3e-2, curvaturesensitivity::Real=0.7, gradientsensitivity::Real=0.05, kwargs...)
    SavePriors && isnothing(LogPriorFn) && @warn "Got kwarg SavePriors=true but $(name(DM) === Symbol() ? "model" : string(name(DM))) does not have prior."
    @assert Confnum > 0
    # stepfactor: overall multiplicative factor for step length
    # stepmemory: linear interpolation of new with previous step size
    # terminatefactor: terminate profile if distance from original MLE too large
    # curvaturesensitivity: step length dependence on current profile curvature
    # gradientsensitivity: step length dependence on profile slope
    # flatstepconst: mainly controls step size when profile exactly flat
    @assert stepfactor > 0 && flatstepconst > 0 && 0 ≤ stepmemory < 1 && terminatefactor > 0 && curvaturesensitivity ≥ 0 && gradientsensitivity ≥ 0
    @assert IC > 0 && MinSafetyFactor > 1 && MaxSafetyFactor > MinSafetyFactor
    OnlyBreakOnBounds && adaptive && @warn "OnlyBreakOnBounds does not currently work with adaptive=true!"
    
    # logLikeMLE - 0.5InformationGeometry.InvChisqCDF(dof, ConfVol(Confnum)) > loglike
    CostThreshold, MaxThreshold = logLikeMLE .- 0.5 .* (IC*MinSafetyFactor, IC*MaxSafetyFactor)

    OptimDomain = Drop(Domain, Comp)

    FitFunc = if !general && isnothing(OptimMeth) && !isnothing(LogPriorFn) && Data(DM) isa AbstractFixedUncertaintyDataSet
        ((args...; Kwargs...)->curve_fit(args...; tol, Domain=OptimDomain, verbose, Kwargs...))
    elseif Multistart > 0
        Meth = (!isnothing(LogPriorFn) && isnothing(OptimMeth)) ? Optim.NewtonTrustRegion() : OptimMeth
        verbose && @info "Using Multistart fitting with N=$Multistart in profile $Comp"
        ((args...; Kwargs...)->MultistartFit(args...; MultistartDomain=OptimDomain, N=Multistart, meth=Meth, showprogress=false, resampling=true, parallel=false, maxval, verbose, tol, Kwargs..., plot=false, Full=true))
    else
        Meth = (!isnothing(LogPriorFn) && isnothing(OptimMeth)) ? Optim.NewtonTrustRegion() : OptimMeth
        ((args...; Kwargs...)->InformationGeometry.minimize(args...; tol, meth=Meth, Domain=OptimDomain, verbose, Kwargs..., Full=true))
    end
    
    # Does not check proximity to boundary! Also does not check nonlinear constraints!
    InBounds = θ::AbstractVector{<:Number} -> _IsInDomain(nothing, Domain, θ)
    # InBounds = θ::AbstractVector{<:Number} -> _IsInDomain(InDomain, Domain, θ)


    ConditionalPush!(N::Nothing, args...) = N
    ConditionalPush!(X::AbstractArray, args...) = push!(X, args...)

    Res = eltype(mle)[];    visitedps = eltype(mle)[]
    Converged = BitVector()
    path = SaveTrajectories ? typeof(mle)[] : nothing
    priors = SavePriors ? eltype(mle)[] : nothing

    sizehint!(Res, N)
    sizehint!(visitedps, N)
    sizehint!(Converged, N)
    SaveTrajectories && sizehint!(path, N)
    SavePriors && sizehint!(priors, N)

    # Domain for Optimization, ProfileDomain just for early termination of profile
    ParamBounds = isnothing(ProfileDomain) ? (-Inf, Inf) : ProfileDomain[Comp]
    OnlyBreakOnBounds && @assert all(isfinite, ParamBounds)

    if pdim(DM) == 1    # Cannot drop dims if pdim already 1
        Xs = [[x] for x in ps]
        Res = map(LogLikelihoodFn, Xs)
        Converged = .!isnan.(Res) .&& .!isinf.(Res) .&& map(x->InBounds([x]), ps)
        visitedps = ps
        SaveTrajectories && (path = Xs)
        SavePriors && map(x->EvalLogPrior(LogPriorFn, x), Xs)
    else
        MLEstash = Drop(mle, Comp)
        
        PerformStep!!! = if ApproximatePaths
            # Perform steps based on profile direction at MLE
            dir = GetLocalProfileDir(DM, Comp, mle)
            pmle = mle[Comp]
            @inline function PerformApproximateStep!(Res, MLEstash, Converged, visitedps, path, priors, p)
                θ = muladd(p-pmle, dir, mle)

                push!(Res, LogLikelihoodFn(θ))
                # Ignore MLEstash
                push!(Converged, !isnan(Res[end]) && InBounds(θ))
                push!(visitedps, p)
                ConditionalPush!(path, θ)
                ConditionalPush!(priors, EvalLogPrior(LogPriorFn, θ))
            end
        elseif general || Data(DM) isa AbstractUnknownUncertaintyDataSet
            # Build objective function based on Neglikelihood only without touching internals
            @inline function PerformStepGeneral!(Res, MLEstash, Converged, visitedps, path, priors, p)
                Ins = ValInserter(Comp, p, mle)
                L = CostFunction∘Ins
                F = isnothing(CostGradient) ? L : (Transform=ValInserterTransform(Comp, p, mle);   isnothing(CostHessian) ? (L, Transform(CostGradient)) : (L, Transform(CostGradient), Transform(CostHessian)))
                R = FitFunc(F, MLEstash; kwargs...)
                
                push!(Res, -GetMinimum(R,L))
                copyto!(MLEstash, GetMinimizer(R))
                FullP = Ins(copy(MLEstash))
                push!(Converged, HasConverged(R) && InBounds(FullP))
                push!(visitedps, p)
                ConditionalPush!(path, FullP)
                ConditionalPush!(priors, EvalLogPrior(LogPriorFn, FullP))
            end
        else
            # Build objective function manually by embedding model and LogPrior separately
            # Does not work combined with variance estimation, i.e. error models
            @inline function PerformStepManual!(Res, MLEstash, Converged, visitedps, path, priors, p)
                NewModel = ProfilePredictor(DM, Comp, p, mle)
                Ins = ValInserter(Comp, p, mle)
                DroppedLogPrior = EmbedLogPrior(DM, Ins)
                R = FitFunc(Data(DM), NewModel, MLEstash, DroppedLogPrior; kwargs...)

                push!(Res, -GetMinimum(R,x->-loglikelihood(Data(DM), NewModel, x, DroppedLogPrior)))
                copyto!(MLEstash, GetMinimizer(R))
                FullP = Ins(copy(MLEstash))
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
            Fi = isnothing(Fisher) ? FisherMetric(DM, mle)[Comp,Comp] : Fisher[Comp,Comp]
            # Calculate initial stepsize based on curvature from fisher information
            initialδ = clamp(stepfactor * sqrt(IC) / (maxstepnumber * (flatstepconst + curvaturesensitivity*sqrt(Fi))) , 1e-12, 1e2)

            δ = initialδ
            minstep = 1e-2 * initialδ
            maxstep = 1e5 * initialδ

            # Second left point
            p = mle[Comp] - δ
            PerformStep!!!(Res, MLEstash, Converged, visitedps, path, priors, p)

            # Input MLE
            push!(Res, LogLikelihoodFn(mle))
            push!(Converged, InBounds(mle))
            push!(visitedps, mle[Comp])
            SaveTrajectories && push!(path, mle)
            SavePriors && push!(priors, EvalLogPrior(LogPriorFn, mle))

            # Second right point
            p = mle[Comp] + δ
            PerformStep!!!(Res, MLEstash, Converged, visitedps, path, priors, p)

            visitedps2 = deepcopy(visitedps) |> reverse!
            Res2 = deepcopy(Res) |> reverse!
            path2 = SaveTrajectories ? reverse!(deepcopy(path)) : nothing
            priors2 = SavePriors ? reverse!(deepcopy(priors)) : nothing
            Converged2 = deepcopy(Converged) |> reverse!
            len = length(visitedps) -1
            
            @inline function DoAdaptive(visitedps, Res, path, priors, Converged)
                while Res[end] > CostThreshold

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
                        !OnlyBreakOnBounds && (length(visitedps) - len > maxstepnumber/2 || p > mle[Comp] + terminatefactor*maxstepnumber*initialδ) && break
                    else
                        p ≤ ParamBounds[1] && break
                        !OnlyBreakOnBounds && (length(visitedps) - len > maxstepnumber/2 || p < mle[Comp] - terminatefactor*maxstepnumber*initialδ) && break
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
            copyto!(MLEstash, Drop(mle, Comp))
            DoAdaptive(visitedps2, Res2, path2, priors2, Converged2)
            
            visitedps = [(@view reverse!(visitedps2)[1:end-3]); visitedps]
            Res = [(@view reverse!(Res2)[1:end-3]); Res]
            path = SaveTrajectories ? [(@view reverse!(path2)[1:end-3]); path] : nothing
            priors = SavePriors ? [(@view reverse!(priors2)[1:end-3]); priors] : nothing
            Converged = [(@view reverse!(Converged2)[1:end-3]); Converged]
        else
            startind = (mlecomp = mle[Comp];    findfirst(x->x>mlecomp, ps)-1)
            if resort && startind > 1
                for p in sort((@view ps[startind:end]))
                    PerformStep!!!(Res, MLEstash, Converged, visitedps, path, priors, clamp(p, ParamBounds...))
                    p ≥ ParamBounds[2] && break
                    !OnlyBreakOnBounds && ((length(visitedps) - 1 > min_steps && Res[end] < CostThreshold) || (Res[end] < MaxThreshold)) && break
                end
                len = length(visitedps)
                copyto!(MLEstash, Drop(mle, Comp))
                for p in sort((@view ps[startind-1:-1:1]); rev=true)
                    PerformStep!!!(Res, MLEstash, Converged, visitedps, path, priors, clamp(p, ParamBounds...))
                    p ≤ ParamBounds[1] && break
                    !OnlyBreakOnBounds && ((length(visitedps) - len > min_steps && Res[end] < CostThreshold) || (Res[end] < MaxThreshold)) && break
                end
            else # No early break, no clamping, just evaluate on given ps
                for p in ps
                    PerformStep!!!(Res, MLEstash, Converged, visitedps, path, priors, p)
                end
            end
        end
    end

    Logmax = AllowNewMLE ? max(try maximum(view(Res, Converged)) catch; -Inf end, logLikeMLE) : logLikeMLE
    Logmax > logLikeMLE && @warn "Profile Likelihood analysis apparently found a likelihood value which is larger than the previously stored LogLikeMLE. Continuing anyway."
    # Using pdim(DM) instead of 1 here, because it gives the correct result
    Priormax = SavePriors ? EvalLogPrior(LogPriorFn,mle) : 0.0
    if IsCost
        if OffsetResults
            @. Res = 2*(Logmax - Res)
            if SavePriors
                @. priors = 2*(Priormax - priors)
            end
        else
            Res .*= -2
            if SavePriors
                priors .*= -2
            end
        end
    else
        @inbounds for i in eachindex(Res)
            Res[i] = Res[i] ≤ Logmax ? InvConfVol(ChisqCDF(dof, 2(Logmax - Res[i]))) : NaN
        end
        if SavePriors
            throw("Not programmed for this case yet, please use kwarg IsCost=true with SavePriors=true.")
            # @inbounds for i in eachindex(priors)
            #     priors[i] = InvConfVol(ChisqCDF(dof, 2(Logmax - priors[i])))
            # end
        end
    end

    perm = sortperm(visitedps)
    # Param always first col, Res second, Converged last. Third column always priors IF length(cols) ≥ 4, columns after prior may be other saved information.
    # Lazy array construction via RecursiveArrayTools allows for preserving type information of columns while still being indexable as matrix
    ResMat = SavePriors ? VectorOfArray([visitedps[perm], Res[perm], priors[perm], Converged[perm]]) : VectorOfArray([visitedps[perm], Res[perm], Converged[perm]])
    SaveTrajectories ? (ResMat, path[perm]) : ResMat
end

function GetProfile(DM::AbstractDataModel, Comp::Int, Confnum::Real; ForcePositive::Bool=false, kwargs...)
    GetProfile(DM, Comp, (C=GetProfileDomainCube(DM, Confnum; ForcePositive=ForcePositive); (C.L[Comp], C.U[Comp])); Confnum=Confnum, kwargs...)
end



function GetLocalProfileDir(DM::AbstractDataModel, Comp::Int, p::AbstractVector{<:Number}=MLE(DM))
    F = FisherMetric(DM, p)
    F[Comp, :] .= [(j == Comp) for j in eachindex(p)]
    det(F) == 0 && @warn "Using pseudo-inverse to determine profile direction for parameter $Comp due to local non-identifiability."
    dir = pinv(F)[:, Comp];    dir ./= dir[Comp]
    dir
end


function ProfileLikelihood(DM::AbstractDataModel, Confnum::Real=2.0, inds::AbstractVector{<:Int}=1:pdim(DM); ForcePositive::Bool=false, Fisher=FisherMetric(DM, MLE(DM)), kwargs...)
    ProfileLikelihood(DM, GetProfileDomainCube(Fisher, MLE(DM), Confnum; ForcePositive=ForcePositive), inds; Confnum=Confnum, Fisher, kwargs...)
end

function ProfileLikelihood(DM::AbstractDataModel, Domain::HyperCube, inds::AbstractVector{<:Int}=1:pdim(DM); plot::Bool=isloaded(:Plots), parallel::Bool=true, verbose::Bool=true, idxs::Tuple{Vararg{Int}}=length(pdim(DM))≥3 ? (1,2,3) : (1,2), kwargs...)
    # idxs for plotting only
    @assert 1 ≤ length(inds) ≤ pdim(DM) && allunique(inds) && all(1 .≤ inds .≤ pdim(DM))

    Prog = Progress(length(inds); enabled=verbose, desc="Computing Profiles... "*(parallel ? "(parallel, $(nworkers()) workers) " : ""), dt=1, showspeed=true)
    Profiles = (parallel ? progress_pmap : progress_map)(i->GetProfile(DM, i, (Domain.L[i], Domain.U[i]); verbose, kwargs...), inds; progress=Prog)

    plot && display(ProfilePlotter(DM, Profiles; idxs))
    Profiles
end

# x and y labels must be passed as kwargs
PlotSingleProfile(DM::AbstractDataModel, Prof::Tuple{<:AbstractMatrix, <:Any}, i::Int; kwargs...) = PlotSingleProfile(DM, Prof[1], i; kwargs...)
function PlotSingleProfile(DM::AbstractDataModel, Prof::AbstractMatrix, i::Int; kwargs...)
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
    Res = copy(X)
    X[1] && !X[2] && (Res[1] = false)
    X[end] && !X[end-1] && (Res[end] = false)
    for i in 2:length(Res)-1
        X[i] && (!X[i-1] || !X[i+1]) && (Res[i] = false)
    end;    Res
end

# What if trajectories NaN?
HasTrajectories(M::Tuple{AbstractMatrix, Nothing}) = false
HasTrajectories(M::Tuple{AbstractMatrix, AbstractVector}) = !all(x->all(isnan,x), M[2])
HasTrajectories(M::AbstractVector{<:Tuple}) = any(HasTrajectories, M)
HasTrajectories(M::AbstractMatrix) = false
HasTrajectories(M::AbstractVector{<:AbstractMatrix}) = false

HasPriors(M::AbstractVector{<:Union{<:AbstractMatrix, <:VectorOfArray}}) = any(HasPriors, M)
HasPriors(M::Tuple) = HasPriors(M[1])
HasPriors(M::Union{<:AbstractMatrix, <:VectorOfArray}) = size(M,2) > 3



function ProfilePlotter(DM::AbstractDataModel, Profiles::AbstractVector;
    PNames::AbstractVector{<:AbstractString}=(Predictor(DM) isa ModelMap ? pnames(Predictor(DM)) : CreateSymbolNames(pdim(DM), "θ")), idxs::Tuple{Vararg{Int}}=length(pdim(DM))≥3 ? (1,2,3) : (1,2), kwargs...)
    @assert length(Profiles) == length(PNames)
    Ylab = length(PNames) == pdim(DM) ? "Conf. level [σ]" : "Cost Function"
    PlotObjects = [PlotSingleProfile(DM, Profiles[i], i; xlabel=PNames[i], ylabel=Ylab, kwargs...) for i in eachindex(Profiles)]
    length(Profiles) ≤ 3 && HasTrajectories(Profiles) && push!(PlotObjects, PlotProfileTrajectories(DM, Profiles; idxs))
    RecipesBase.plot(PlotObjects...; layout=length(PlotObjects))
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
function ProfileBox(DM::AbstractDataModel, Fs::AbstractVector{<:AbstractInterpolation}, Confnum::Real=1.; kwargs...)
    ProfileBox(Fs, MLE(DM), Confnum; dof=DOF(DM), kwargs...)
end
function ProfileBox(Fs::AbstractVector{<:AbstractInterpolation}, mle::AbstractVector, Confnum::Real=1.; parallel::Bool=true, dof::Int=length(mle), kwargs...)
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


function _ProfileBox(F::AbstractInterpolation, Confnum::Real=1.0; IsCost::Bool=true, dof::Int=1, mleval::Real=F.t[findmin(F.u)[2]], 
                            CostThreshold::Union{<:Real, Nothing}=nothing, maxval::Real=Inf, tol::Real=1e-10, xrtol::Real=tol, xatol::Real=tol, kwargs...)
    Crossings = if !IsCost
        FindZerosWrapper(x->(F(x)-Confnum), F.t[1], F.t[end]; no_pts=length(F.t), xrtol, xatol, mleval, kwargs...)
    else
        # Already 2(loglikeMLE - loglike) in Profile
        CostThresh = if !isnothing(CostThreshold)
            CostThreshold
        else
            # Allow for computation of F-based threshold here?
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
    PracticallyIdentifiable(DM::AbstractDataModel, Confnum::Real=1; plot::Bool=isloaded(:Plots), IsCost::Bool=false, kwargs...) -> Real
Determines the maximum confidence level (in units of standard deviations σ) at which the given `DataModel` is still practically identifiable.
"""
PracticallyIdentifiable(DM::AbstractDataModel, Confnum::Real=1.0; plot::Bool=isloaded(:Plots), N::Int=100, IsCost::Bool=false, kwargs...) = PracticallyIdentifiable(ParameterProfiles(DM, Confnum; plot=plot, N=N, IsCost=IsCost, kwargs...))

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
mutable struct ParameterProfiles <: AbstractProfiles
    Profiles::AbstractVector{<:Union{<:AbstractMatrix,<:VectorOfArray}}
    Trajectories::AbstractVector{<:Union{<:AbstractVector{<:AbstractVector{<:Number}}, <:Nothing}}
    Names::AbstractVector{Symbol}
    mle::AbstractVector{<:Number}
    dof::Int
    IsCost::Bool
    Meta::Symbol
    # Allow for different inds and fill rest with nothing or NaN
    function ParameterProfiles(DM::AbstractDataModel, Confnum::Union{Real,HyperCube}=2., Inds::AbstractVector{<:Int}=1:pdim(DM); plot::Bool=isloaded(:Plots), SaveTrajectories::Bool=true, IsCost::Bool=true, dof::Int=DOF(DM), Meta::Symbol=:ParameterProfiles, kwargs...)
        inds = sort(Inds)
        FullProfs = ProfileLikelihood(DM, Confnum, inds; plot=false, SaveTrajectories=SaveTrajectories, IsCost=IsCost, kwargs...)
        Profs = SaveTrajectories ? getindex.(FullProfs,1) : FullProfs
        Trajs = SaveTrajectories ? getindex.(FullProfs,2) : fill(nothing, length(inds))
        if !(inds == 1:pdim(DM))
            for i in 1:pdim(DM) # Profs and Trajs already sorted by sorting inds
                i ∉ inds && (insert!(Profs, i, fill(NaN, size(Profs[1])));  Profs[i][:,end] .= 0;    SaveTrajectories ? insert!(Trajs, i, [fill(NaN, pdim(DM))]) : insert!(Trajs, i, nothing))
            end
        end
        P = ParameterProfiles(DM, Profs, Trajs; IsCost, dof, Meta)
        plot && display(RecipesBase.plot(P, false))
        P
    end
    function ParameterProfiles(DM::AbstractDataModel, Profiles::AbstractVector{<:Union{<:AbstractMatrix,<:VectorOfArray}}, Trajectories::AbstractVector=fill(nothing,length(Profiles)), Names::AbstractVector{<:StringOrSymb}=pnames(DM); IsCost::Bool=true, dof::Int=DOF(DM), kwargs...)
        ParameterProfiles(Profiles, Trajectories, Names, MLE(DM), dof, IsCost; kwargs...)
    end
    function ParameterProfiles(Profiles::AbstractVector{<:Union{<:AbstractMatrix,<:VectorOfArray}}, Trajectories::AbstractVector=fill(nothing,length(Profiles)), Names::AbstractVector{<:StringOrSymb}=CreateSymbolNames(length(Profiles),"θ"); IsCost::Bool=true, dof::Int=length(Names), kwargs...)
        ParameterProfiles(Profiles, Trajectories, Names, fill(NaN, length(Names)), dof, IsCost; kwargs...)
    end
    function ParameterProfiles(Profiles::AbstractVector{<:Union{<:AbstractMatrix,<:VectorOfArray}}, Trajectories::AbstractVector, Names::AbstractVector{<:StringOrSymb}, mle, dof::Int, IsCost::Bool, meta::Symbol=:ParameterProfiles; Meta::Symbol=meta, verbose::Bool=true)
        @assert length(Profiles) == length(Names) == length(mle) == length(Trajectories)
        verbose && !(1 ≤ dof ≤ length(mle)) && @warn "Got dof=$dof but length(MLE)=$(length(mle))."
        new(Profiles, Trajectories, Symbol.(Names), mle, dof, IsCost, Meta)
    end
end
(P::ParameterProfiles)(t::Real, i::Int, Interp::Type{<:AbstractInterpolation}=QuadraticInterpolation; kwargs...) = InterpolatedProfiles(P, i, Interp; kwargs...)(t)
(P::ParameterProfiles)(i::Int, Interp::Type{<:AbstractInterpolation}=QuadraticInterpolation; kwargs...) = InterpolatedProfiles(P, i, Interp; kwargs...)
InterpolatedProfiles(P::ParameterProfiles, i::Int, Interp::Type{<:AbstractInterpolation}=QuadraticInterpolation; kwargs...) = GetInterpolator(view(Profiles(P)[i],:,2), view(Profiles(P)[i],:,1), Interp; kwargs...)
InterpolatedProfiles(P::ParameterProfiles, Interp::Type{<:AbstractInterpolation}=QuadraticInterpolation; kwargs...) = [GetInterpolator(view(Prof,:,2), view(Prof,:,1), Interp; kwargs...) for Prof in Profiles(P)]

# For SciMLBase.remake
ParameterProfiles(;
    Profiles::AbstractVector{<:AbstractMatrix}=[zeros(1,3)],
    Trajectories::AbstractVector{<:Union{<:AbstractVector{<:AbstractVector{<:Number}}, <:Nothing}}=[nothing],
    Names::AbstractVector{<:StringOrSymb}=Symbol[],
    mle::AbstractVector{<:Number}=Float64[],
    dof::Int=0,
    IsCost::Bool=false,
    Meta::Symbol=:remake,) = ParameterProfiles(Profiles, Trajectories, Names, mle, dof, IsCost; Meta)


Profiles(P::ParameterProfiles) = P.Profiles
Trajectories(P::ParameterProfiles) = P.Trajectories
pnames(P::ParameterProfiles) = P.Names .|> string
Pnames(P::ParameterProfiles) = P.Names
MLE(P::ParameterProfiles) = P.mle
pdim(P::ParameterProfiles) = length(MLE(P))
DOF(P::ParameterProfiles) = P.dof
IsCost(P::ParameterProfiles) = P.IsCost
HasTrajectories(P::ParameterProfiles) = any(i->HasTrajectories(P[i]), 1:length(P))
IsPopulated(P::ParameterProfiles) = Bool[HasProfiles(P[i]) for i in eachindex(P)]

HyperCube(P::ParameterProfiles) = [IsPopulated(P[i]) ? collect(extrema(@view Profiles(P[i])[:,1])) : [-Inf,Inf] for i in eachindex(P)] |> HyperCube

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
If `IsCost=true` was chosen for the profiles, the output is the maximal deviation in cost function value `2(L_MLE - PL_i(θ))`.
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


function PlotProfileTrajectories(DM::AbstractDataModel, P::ParameterProfiles; kwargs...)
    @assert HasTrajectories(P)
    PlotProfileTrajectories(DM, [(Profiles(P)[i], Trajectories(P)[i]) for i in eachindex(Profiles(P))]; kwargs...)
end

function ExtendProfiles(P::ParameterProfiles)
    throw("Not programmed yet.")
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


@recipe f(P::Union{ParameterProfiles, ParameterProfilesView}, S::Symbol, args...) = P, Val(S), args...

# Plot trajectories by default
@recipe f(P::ParameterProfiles, PlotTrajectories::Bool=HasTrajectories(P) && P.Meta === :ParameterProfiles && length(Trajectories(P)[1]) < 5) = P, Val(PlotTrajectories)

# DoBiLog for paths, i.e. TrafoPath
@recipe function f(P::ParameterProfiles, HasTrajectories::Val{true})
    layout := sum(IsPopulated(P)) + 1
    Trafo = get(plotattributes, :Trafo, identity)
    DoBiLog = get(plotattributes, :BiLog, true)
    TrafoPath = get(plotattributes, :TrafoPath, DoBiLog ? BiLog : identity)
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
    P.Meta !== :ParameterProfiles && (plot_title --> string(P.Meta))
    Trafo = get(plotattributes, :Trafo, identity)
    tol = 0.05
    maxy = median(vcat(0.0, [maximum(view(T, GetConverged(T), 2)) for T in Profiles(P) if !all(isnan, view(T, :, 1)) && sum(GetConverged(T)) > 0 && maximum(view(T, GetConverged(T), 2)) > tol]))
    maxy = maxy < tol ? (maxy < 1e-12 ? tol : Inf) : maxy
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
    Trafo = get(plotattributes, :Trafo, identity)
    tol = 0.05
    maxy = median(vcat(0.0, [maximum(view(T[2], GetConverged(T))) for T in Profiles(P) if !all(isnan, T[1]) && sum(GetConverged(T)) > 0 && maximum(view(T[2], GetConverged(T))) > tol]))
    maxy = maxy < tol ? (maxy < 1e-12 ? tol : Inf) : maxy
    Ylims = get(plotattributes, :ylims, (Trafo.(-tol), Trafo.(maxy)))
    @series begin
        ylims --> Ylims
        Trafo := Trafo
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
                ylabel --> ApplyTrafoNames(IsCost(P) ? "Cost Function" : "Conf. level [σ]", Trafo)
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
                    TrafoPath.(view(M,:,1)), TrafoPath.(view(M,:,2))
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
@recipe f(PV::ParameterProfilesView, PlotTrajectories::Bool=HasTrajectories(PV) && PV.P.Meta === :ParameterProfiles) = PV, Val(PlotTrajectories)

@recipe function f(PVs::AbstractVector{<:ParameterProfilesView}, V::Val=Val(false))
    layout --> length(PVs)
    Trafo = get(plotattributes, :Trafo, identity)
    for i in eachindex(PVs)
        @series begin
            subplot := i
            Trafo := Trafo
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
    yguide --> ApplyTrafoNames(IsCost(PV) ? "Cost Function" : "Conf. level [σ]", Trafo)

    @series begin
        label --> ["Profile Likelihood" nothing]
        lw --> 1.5
        Profiles(PV)[1], Trafo.(Convergify(Profiles(PV)[2], GetConverged(Profiles(PV))))
    end
    # Draw prior contribution
    if HasPriors(Profiles(PV))
        @series begin
            label --> ["Prior contribution" nothing]
            linealpha --> 0.75
            line --> :dash
            lw --> 1.5
            Profiles(PV)[1], Trafo.(Convergify(Profiles(PV)[3], GetConverged(Profiles(PV))))
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
                    lw --> 1.5
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
    Trafo = get(plotattributes, :Trafo, identity)
    DoBiLog = get(plotattributes, :BiLog, true)
    TrafoPath = get(plotattributes, :TrafoPath, DoBiLog ? BiLog : identity)
    @series begin
        subplot := 1
        Trafo := Trafo
        PV, Val(false)
    end
    @series begin
        subplot := 2
        TrafoPath := TrafoPath
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

    ToPlotInds = [i for i in eachindex(MLE(P)) if i ∈ idxs && !isnothing(Trajectories(P)[i])]

    layout --> length(ToPlotInds)
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
    Fisher = get(plotattributes, :Fisher, Diagonal(ones(pdim(PV))))
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
    ystring = DoRelChange ? "p_i" * (OffsetResults ? " / p_MLE" : "") :  (U != Diagonal(ones(pdim(PV))) ? "F^(1/2) * [p_i" * (OffsetResults ? " - p_MLE" : "") * "]" : "p_i" * (OffsetResults ? " - p_MLE" : ""))
    yguide --> ApplyTrafoNames(ystring, TrafoPath)
    # Also filter out 
    ToPlotInds = idxs[idxs .!= i]
    color_palette = get(plotattributes, :color_palette, :default)
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
            getindex.(Trajectories(PV), i), TrafoPath.(Change)
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
    #ystring = DoRelChange ? "p_i" * (OffsetResults ? " / p_MLE" : "") :  (U != Diagonal(ones(pdim(PV))) ? "F^(1/2) * [p_i" * (OffsetResults ? " - p_MLE" : "") * "]" : "p_i" * (OffsetResults ? " - p_MLE" : ""))
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
    #ystring = DoRelChange ? "p_i" * (OffsetResults ? " / p_MLE" : "") :  (U != Diagonal(ones(pdim(PV))) ? "F^(1/2) * [p_i" * (OffsetResults ? " - p_MLE" : "") * "]" : "p_i" * (OffsetResults ? " - p_MLE" : ""))
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


# Bad style but works for now for plotting profiles from different models in one:
function RecipesBase.plot(Ps::AbstractArray{<:Union{ParameterProfiles, ParameterProfilesView}}; kwargs...)
    Plts = [RecipesBase.plot(Ps[i]) for i in eachindex(Ps)]
    RecipesBase.plot(Plts...; layout=length(Plts), kwargs...)
end


# Generate validation profile centered on prediction at a single independent variable t
function GetValidationProfilePoint(DM::AbstractDataModel, yComp::Int, t::Union{AbstractVector{<:Number},Number}; Confnum::Real=2, N::Int=21, mle::AbstractVector{<:Number}=MLE(DM), ypred::Real=Predictor(DM)(t,mle)[yComp], yoffset::Real=ypred, 
                                dof::Int=DOF(DM), LinPredictionUncert::Real=(C=VariancePropagation(DM, mle; Confnum, dof)(t);   ydim(DM)>1 ? C[yComp, yComp] : C), DivideBy::Real=5, 
                                σv::Real=LinPredictionUncert/DivideBy, IC::Real=InvChisqCDF(dof, ConfVol(Confnum)), ValidationSafetyFactor::Real=2, kwargs...) # Make Confnumsafety ratio σv/(obs + σv) to decrease computations when prediction profiles are desired?
    @assert IC > 0 && dof > 0
    ℓ = loglikelihood(DM);    M = Predictor(DM);    FicticiousPoint = Normal(0, σv)
    FictDataPointPrior(θnew::AbstractVector) = (θ=view(θnew, 1:lastindex(θnew)-1);   logpdf(FicticiousPoint, θnew[end] - M(t, θ)[yComp] + yoffset))
    VPL(θnew::AbstractVector) = ℓ(view(θnew, 1:lastindex(θnew)-1)) + FictDataPointPrior(θnew)
    mleNew = [mle; (ypred-yoffset)];    Fisher = Diagonal(σv^-2 * ones(pdim(DM)+1))
    B = ValidationSafetyFactor*σv*sqrt(2*IC);    Ran = range(-B + (ypred-yoffset), B + (ypred-yoffset); length=N)
    GetProfile(DM, pdim(DM)+1, Ran; LogLikelihoodFn=VPL, LogPriorFn=FictDataPointPrior, dof, mle=mleNew, logLikeMLE=VPL(mleNew), Fisher, Confnum, N, IsCost=true, Domain=nothing, InDomain=nothing, AllowNewMLE=false, general=true, SavePriors=true, kwargs...)
end

# Generate multiple validation profiles and add back offset to prediction scale
"""
    ValidationProfiles(DM::AbstractDataModel, yComp::Int, Ts::AbstractVector; Confnum::Real=2, dof::Int=DOF(DM), OffsetToZero::Bool=false, kwargs...)
Computes a set of validation profiles for the component `yComp` of the prediction at various values of the independent variables `Ts`.
The uncertainty of the ficticious validation data point can be optionally chosen via the keyword argument `σv`.
Most other kwargs are passed on to the `ParameterProfiles` function and thereby also to the optimizers, see e.g. [`ParameterProfiles`](@ref), [`InformationGeometry.minimize`](@ref).
"""
function ValidationProfiles(DM::AbstractDataModel, yComp::Int, Ts::AbstractVector=range(extrema(xdata(DM))...; length=3length(xdata(DM))); dof::Int=DOF(DM), mle::AbstractVector{<:Number}=MLE(DM), IsCost::Bool=true, OffsetToZero::Bool=false, Meta=:ValidationProfiles, parallel::Bool=true, verbose::Bool=true, kwargs...)
    ypreds = [Predictor(DM)(t,mle)[yComp] for t in Ts]      # Always compute with offset to zero internally
    Res = (parallel ? progress_pmap : progress_map)(i->GetValidationProfilePoint(DM, yComp, Ts[i]; ypred=ypreds[i], dof=dof, mle, IsCost, verbose, kwargs...), 1:length(Ts); progress=Progress(length(Ts); enabled=verbose, desc="Computing Validation Profiles... (parallel, $(nworkers()) workers) ", dt=1, showspeed=true))
    Profs, Trajs = getindex.(Res,1), getindex.(Res,2)
    for i in eachindex(Ts)
        zProf = map(TrajPoint->Predictor(DM)(Ts[i], (@view TrajPoint[1:end-1]))[yComp], Trajs[i])
        Profs[i] = @views hcat(Profs[i][:,1:end-1], zProf, Profs[i][:,end])
    end
    # If should remain on true scale, add ypreds back on
    offsetvec = OffsetToZero ? zeros(length(Ts)) : ypreds
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
    @assert HasPriors(Profiles(VP)) && IsCost(VP)
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


