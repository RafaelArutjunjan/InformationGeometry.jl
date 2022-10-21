

# Returns a copy of type `Vector`, i.e. is not typesafe!
SafeCopy(X::AbstractVector) = copy(X)
SafeCopy(X::AbstractRange) = collect(X)
SafeCopy(X::Union{SVector,MVector}) = convert(Vector,X)

Drop(X::AbstractVector, i::Int) = (Z=SafeCopy(X);   splice!(Z,i);   Z)

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


"""
    ValInserter(Component::Int, Value::AbstractFloat) -> Function
Returns an embedding function ``\\mathbb{R}^N \\longrightarrow \\mathbb{R}^{N+1}`` which inserts `Value` in the specified `Component`.
In effect, this allows one to pin an input component at a specific value.
"""
function ValInserter(Component::Int, Value::AbstractFloat)
    ValInsertionEmbedding(P::AbstractVector) = insert!(SafeCopy(P), Component, Value)
    ValInsertionEmbedding(P::Union{SVector,MVector}) = insert(P, Component, Value)
end

# https://discourse.julialang.org/t/how-to-sort-two-or-more-lists-at-once/12073/13
function _SortTogether(A::AbstractVector, B::AbstractVector, args...; rev::Bool=false, kwargs...)
    issorted(A; rev=rev) ? (A, B, args...) : getindex.((A, B, args...), (sortperm(A; rev=rev, kwargs...),))
end
"""
    ValInserter(Components::AbstractVector{<:Int}, Values::AbstractVector{<:AbstractFloat}) -> Function
Returns an embedding function which inserts `Values` in the specified `Components`.
In effect, this allows one to pin multiple input components at a specific values.
"""
function ValInserter(Components::AbstractVector{<:Int}, Values::AbstractVector{<:Number})
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
function ValInserter(Components::AbstractVector{<:Bool}, Values::AbstractVector{<:Number})
    @assert length(Components) == length(Values)
    ValInserter((1:length(Components))[Components], Values[Components])
end
function ValInserter(Components::AbstractVector{<:Int}, Value::Number)
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
end

InsertIntoFirst(X::AbstractVector{<:Number}) = PassingIntoLast(θ::AbstractVector{<:Number}) = [X;θ]
InsertIntoLast(θ::AbstractVector{<:Number}) = PassingIntoFirst(X::AbstractVector{<:Number}) = [X;θ]


ProfilePredictor(DM::AbstractDataModel, args...; kwargs...) = ProfilePredictor(Predictor(DM), args...; kwargs...)
ProfilePredictor(M::ModelMap, Comp::Int, PinnedValue::AbstractFloat; kwargs...) = EmbedModelVia(M, ValInserter(Comp, PinnedValue); Domain=DropCubeDims(Domain(M), Comp), kwargs...)
ProfilePredictor(M::ModelMap, Comps::AbstractVector{<:Int}, PinnedValues::AbstractVector{<:AbstractFloat}; kwargs...) = EmbedModelVia(M, ValInserter(Comps, PinnedValues); Domain=DropCubeDims(Domain(M), Comps), kwargs...)

ProfilePredictor(M::Function, Comp::Int, PinnedValue::AbstractFloat; kwargs...) = EmbedModelVia(M, ValInserter(Comp, PinnedValue); kwargs...)
ProfilePredictor(M::Function, Comps::AbstractVector{<:Int}, PinnedValues::AbstractVector{<:AbstractFloat}; kwargs...) = EmbedModelVia(M, ValInserter(Comps, PinnedValues); kwargs...)


ProfileDPredictor(DM::AbstractDataModel, args...; kwargs...) = ProfileDPredictor(dPredictor(DM), args...; kwargs...)
ProfileDPredictor(dM::ModelMap, Comp::Int, PinnedValue::AbstractFloat; kwargs...) = EmbedDModelVia(dM, ValInserter(Comp, PinnedValue); Domain=DropCubeDims(Domain(dM), Comp), kwargs...)
ProfileDPredictor(dM::ModelMap, Comps::AbstractVector{<:Int}, PinnedValues::AbstractVector{<:AbstractFloat}; kwargs...) = EmbedDModelVia(dM, ValInserter(Comps, PinnedValues); Domain=DropCubeDims(Domain(dM), Comps), kwargs...)

ProfileDPredictor(dM::Function, Comp::Int, PinnedValue::AbstractFloat; kwargs...) = EmbedDModelVia(dM, ValInserter(Comp, PinnedValue); kwargs...)
ProfileDPredictor(dM::Function, Comps::AbstractVector{<:Int}, PinnedValues::AbstractVector{<:AbstractFloat}; kwargs...) = EmbedDModelVia(dM, ValInserter(Comps, PinnedValues); kwargs...)


"""
    PinParameters(DM::AbstractDataModel, Component::Int, Value::AbstractFloat=MLE(DM)[Component])
    PinParameters(DM::AbstractDataModel, Components::AbstractVector{<:Int}, Values::AbstractVector{<:AbstractFloat}=MLE(DM)[Components])
    PinParameters(DM::AbstractDataModel, ParamDict::Dict{String, Number})
Returns `DataModel` where one or more parameters have been pinned to specified values.
"""
function PinParameters(DM::AbstractDataModel, Components::Union{Int,AbstractVector{<:Int}}, Values::Union{AbstractFloat,AbstractVector{<:AbstractFloat}}=MLE(DM)[Components])
    @assert length(Components) == length(Values) && length(Components) < pdim(DM)
    length(Components) == 0 && (@warn "Got no parameters to pin.";  return DM)
    DataModel(Data(DM), ProfilePredictor(DM, Components, Values), ProfileDPredictor(DM, Components, Values), Drop(MLE(DM), Components), EmbedLogPrior(DM, ValInserter(Components, Values)))
end

function PinParameters(DM::AbstractDataModel, ParamDict::Dict{String, Number})
    Comps = Int[];  Vals = []
    for i in 1:pdim(DM)
        pnames(DM)[i] ∈ keys(ParamDict) && push!(Comps, i) && push!(Vals, ParamDict[pnames(DM)[i]])
    end
    @assert length(Comps) > 0 "No overlap between parameters and given parameter dictionary: pnames=$(pnames(DM)), keys=$(keys(ParamDict))."
    PinParameters(DM, Comps, Vals)
end


_WithoutFirst(X::AbstractVector{<:Bool}) = (Z=copy(X);  Z[findfirst(X)]=false;  Z)
function GetLinkEmbedding(Linked::AbstractVector{<:Bool}, MainInd::Int=findfirst(Linked))
    @assert MainInd ∈ 1:length(Linked) && sum(Linked) ≥ 2 "Got Linked=$Linked and MainInd=$MainInd."
    LinkedInds = (1:length(Linked))[Linked]
    LinkEmbedding(θ::AbstractVector{<:Number}) = ValInserter(LinkedInds, θ[MainInd])(θ)
end
"""
    LinkParameters(DM::AbstractDataModel, Linked::Union{AbstractVector{<:Bool},AbstractVector{<:Int}}, MainInd::Int=findfirst(Linked); kwargs...)
Embeds the model such that all components `i` for which `Linked[i] == true` are linked to the parameter corresponding to component `MainInd`.
`Linked` can also be a `String`: this creates a `BitVector` whose components are `true` whenever the corresponding parameter name contains `Linked`.
"""
function LinkParameters(DM::AbstractDataModel, Linked::AbstractVector{<:Bool}, MainInd::Int=findfirst(Linked), args...; kwargs...)
    DataModel(Data(DM), LinkParameters(Predictor(DM), Linked, MainInd, args...; kwargs...), Drop(MLE(DM), _WithoutFirst(Linked)), EmbedLogPrior(DM, GetLinkEmbedding(Linked,MainInd)))
end
function LinkParameters(M::ModelMap, Linked::AbstractVector{<:Bool}, MainInd::Int=findfirst(Linked); kwargs...)
    @assert length(Linked) == pdim(M)
    WoFirst = _WithoutFirst(Linked)
    Pnames = copy(pnames(M))
    Pnames[MainInd] *= " =: " * join(pnames(M)[WoFirst], " ≡ ")
    Pnames = Pnames[.!WoFirst]
    EmbedModelVia(M, GetLinkEmbedding(Linked, MainInd); Domain=DropCubeDims(Domain(M), WoFirst), pnames=Pnames, kwargs...)
end
function LinkParameters(F::Function, Linked::AbstractVector{<:Bool}, MainInd::Int=findfirst(Linked); kwargs...)
    EmbedModelVia(F, GetLinkEmbedding(Linked, MainInd); kwargs...)
end
LinkParameters(DM, Linked::AbstractVector{<:Int}, args...; kwargs...) = LinkParameters(DM, [i ∈ Linked for i in 1:pdim(DM)], args...; kwargs...)
LinkParameters(DM, S::String, args...; kwargs...) = LinkParameters(DM, occursin.(S, pnames(DM)), args...; kwargs...)

function LinkParameters(DM, S::String, T::String, args...; kwargs...)
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

function GetProfileDomainCube(DM::AbstractDataModel, Confnum::Real; kwargs...)
    Cube = GetProfileDomainCube(FisherMetric(DM, MLE(DM)), MLE(DM), Confnum; kwargs...)
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


"""
    GetProfile(DM::AbstractDataModel, Comp::Int, dom::Tuple{<:Real, <:Real}; N::Int=50, dof::Int=pdim(DM), SaveTrajectories::Bool=false, SavePriors::Bool=false)
Computes profile likelihood associated with the component `Comp` of the parameters over the domain `dom`.
"""
function GetProfile(DM::AbstractDataModel, Comp::Int, dom::Tuple{<:Real, <:Real}; N::Int=50, tol::Real=1e-9, IsCost::Bool=false, dof::Int=pdim(DM), SaveTrajectories::Bool=false, SavePriors::Bool=false, kwargs...)
    @assert dom[1] < dom[2] && (1 ≤ Comp ≤ pdim(DM))
    SavePriors && isnothing(LogPrior(DM)) && @warn "Got kwarg SavePriors=true but $(length(name(DM)) > 0 ? name(DM) : "model") does not have prior."

    ps = DomainSamples(dom; N=N)

    # Could use variable size array instead to cut off computation once Confnum+0.1 is reached?
    Res = eltype(MLE(DM))[];    visitedps = eltype(MLE(DM))[]
    path = SaveTrajectories ? Vector{Vector{eltype(MLE(DM))}}(undef, N) : nothing
    priors = SavePriors ? eltype(MLE(DM))[] : nothing
    if pdim(DM) == 1    # Cannot drop dims if pdim already 1
        Res = map(x->loglikelihood(DM, [x]), ps)
    else
        MLEstash = Drop(MLE(DM), Comp)
        for (i,p) in enumerate(ps)
            NewModel = ProfilePredictor(DM, Comp, p)
            DroppedLogPrior = EmbedLogPrior(DM, ValInserter(Comp,p))
            MLEstash = curve_fit(Data(DM), NewModel, ProfileDPredictor(DM, Comp, p), MLEstash, DroppedLogPrior; tol=tol, kwargs...).param
            push!(Res, loglikelihood(Data(DM), NewModel, MLEstash, DroppedLogPrior))
            push!(visitedps, p)
            SaveTrajectories && (push!(path,MLEstash);    insert!(path[end], Comp, p))
            SavePriors && push!(priors, EvalLogPrior(DroppedLogPrior, [p]))
        end
    end
    Logmax = max(maximum(Res), LogLikeMLE(DM))
    !(Logmax ≈ LogLikeMLE(DM)) && @warn "Profile Likelihood analysis apparently found a likelihood value which is larger than the previously stored LogLikeMLE. Continuing anyway."
    # Using pdim(DM) instead of 1 here, because it gives the correct result
    # Priormax = SavePriors ? EvalLogPrior(LogPrior(DM),MLE(DM)) : 0.0
    if IsCost
        @. Res = 2*(Logmax - Res)
        if SavePriors
            @. priors = 2*(Priormax - priors)
        end
    else
        @inbounds for i in eachindex(Res)
            Res[i] = InvConfVol(ChisqCDF(dof, 2(Logmax - Res[i])))
        end
        if SavePriors
            @inbounds for i in eachindex(priors)
                priors[i] = InvConfVol(ChisqCDF(dof, 2(Priormax - priors[i])))
            end
        end
    end

    ResMat = SavePriors ? [visitedps Res priors] : [visitedps Res]
    SaveTrajectories ? (ResMat, path) : ResMat
end

function GetProfile(DM::AbstractDataModel, Comp::Int, Confnum::Real; ForcePositive::Bool=false, kwargs...)
    GetProfile(DM, Comp, (C=GetProfileDomainCube(DM, Confnum; ForcePositive=ForcePositive); (C.L[Comp], C.U[Comp])); kwargs...)
end


"""
    ProfileLikelihood(DM::AbstractDataModel, Confnum::Real=2; N::Int=50, ForcePositive::Bool=false, plot::Bool=true, parallel::Bool=false, dof::Int=pdim(DM), SaveTrajectories::Bool=false) -> Vector{Matrix}
Computes the profile likelihood for each component of the parameters ``θ \\in \\mathcal{M}`` over the given `Domain`.
Returns a vector of N×2 matrices where the first column of the n-th matrix specifies the value of the n-th component and the second column specifies the associated confidence level of the best fit configuration conditional to the n-th component being fixed at the associated value in the first column.

The domain over which the profile likelihood is computed is not (yet) adaptively chosen. Instead the size of the domain is estimated from the inverse Fisher metric.
Therefore, often has to pass higher value for `Confnum` to this method than the confidence level one is actually interested in, to ensure that it is still covered (if the model is even practically identifiable in the first place).
"""
function ProfileLikelihood(DM::AbstractDataModel, Confnum::Real=2; ForcePositive::Bool=false, kwargs...)
    ProfileLikelihood(DM, GetProfileDomainCube(DM, Confnum; ForcePositive=ForcePositive); kwargs...)
end

function ProfileLikelihood(DM::AbstractDataModel, Domain::HyperCube; N::Int=50, plot::Bool=true, parallel::Bool=false, verbose::Bool=true, kwargs...)
    Profiles = if verbose
        Prog = Progress(pdim(DM); enabled=verbose, desc="Computing Profiles... ", dt=1, showspeed=true)
        (parallel ? progress_pmap : progress_map)(i->GetProfile(DM, i, (Domain.L[i], Domain.U[i]); N=N, kwargs...), 1:pdim(DM); progress=Prog)
    else
        (parallel ? pmap : map)(i->GetProfile(DM, i, (Domain.L[i], Domain.U[i]); N=N, kwargs...), 1:pdim(DM))
    end
    plot && display(ProfilePlotter(DM, Profiles))
    Profiles
end

# x and y labels must be passed as kwargs
PlotSingleProfile(DM::AbstractDataModel, Prof::Tuple{<:AbstractMatrix, <:Any}, i::Int; kwargs...) = PlotSingleProfile(DM, Prof[1], i; kwargs...)
function PlotSingleProfile(DM::AbstractDataModel, Prof::AbstractMatrix, i::Int; kwargs...)
    P = RecipesBase.plot(view(Prof, :,1), view(Prof, :,2); leg=false, label="Profile", kwargs...)
    size(Prof,2) == 3 && RecipesBase.plot!(P, view(Prof, :,1), view(Prof, :,3); label="Prior", color=:red, line=:dash)
    P
end

HasTrajectories(M::AbstractVector) = any(HasTrajectories, M)
HasTrajectories(M::Tuple) = true
HasTrajectories(M::AbstractMatrix) = false

function ProfilePlotter(DM::AbstractDataModel, Profiles::AbstractVector;
    Pnames::AbstractVector{<:String}=(Predictor(DM) isa ModelMap ? pnames(Predictor(DM)) : CreateSymbolNames(pdim(DM), "θ")), kwargs...)
    @assert length(Profiles) == length(Pnames)
    Ylab = length(Pnames) == pdim(DM) ? "Conf. level [σ]" : "Cost Function"
    PlotObjects = [PlotSingleProfile(DM, Profiles[i], i; xlabel=Pnames[i], ylabel=Ylab, kwargs...) for i in 1:length(Profiles)]
    length(Profiles) ≤ 3 && HasTrajectories(Profiles) && push!(PlotObjects, PlotProfileTrajectories(DM, Profiles))
    RecipesBase.plot(PlotObjects...; layout=length(PlotObjects))
end
# Plot trajectories of Profile Likelihood
"""
    PlotProfileTrajectories(DM::AbstractDataModel, Profiles::AbstractVector{Tuple{AbstractMatrix,AbstractVector}}; OverWrite=true, kwargs...)
"""
function PlotProfileTrajectories(DM::AbstractDataModel, Profiles::AbstractVector; OverWrite=true, kwargs...)
    @assert HasTrajectories(Profiles)
    P = OverWrite ? RecipesBase.plot() : RecipesBase.plot!()
    for i in 1:length(Profiles)
        HasTrajectories(Profiles[i]) && RecipesBase.plot!(P, Profiles[i][2]; marker=:circle, label="Comp: $i", kwargs...)
    end
    RecipesBase.plot!(P, [MLE(DM)]; linealpha=0, marker=:hex, markersize=3, label="MLE", kwargs...)
end


"""
    InterpolatedProfiles(M::AbstractVector{<:AbstractMatrix}) -> Vector{Function}
Interpolates the `Vector{Matrix}` output of ProfileLikelihood() with cubic splines.
"""
function InterpolatedProfiles(Mats::AbstractVector{<:AbstractMatrix})
    [QuadraticInterpolation(view(profile,:,2), view(profile,:,1)) for profile in Mats]
end

"""
    ProfileBox(DM::AbstractDataModel, Fs::AbstractVector{<:DataInterpolations.AbstractInterpolation}, Confnum::Real=1.) -> HyperCube
Constructs `HyperCube` which bounds the confidence region associated with the confidence level `Confnum` from the interpolated likelihood profiles.
"""
function ProfileBox(DM::AbstractDataModel, Fs::AbstractVector{<:AbstractInterpolation}, Confnum::Real=1.; kwargs...)
    ProfileBox(Fs, MLE(DM), Confnum; kwargs...)
end
function ProfileBox(Fs::AbstractVector{<:AbstractInterpolation}, mle::AbstractVector, Confnum::Real=1.; Padding::Real=0., max::Real=1e10, meth::Roots.AbstractUnivariateZeroMethod=Roots.Bisection())
    crossings = [find_zeros(x->(Fs[i](x)-Confnum), Fs[i].t[1], Fs[i].t[end]) for i in 1:length(Fs)]
    # crossings = map(F->[F.t[1], F.t[end]], Fs)
    # for i in 1:length(crossings)
    #     if any(isfinite, Fs[i].u)
    #         crossings[i][1] = find_zero(x->abs(Fs[i](x)-Confnum), (Fs[i].t[1], mle[i]), meth)
    #         crossings[i][2] = find_zero(x->abs(Fs[i](x)-Confnum), (mle[i], Fs[i].t[end]), meth)
    #     else
    #         crossings[i] .= SA[-max, max]
    #     end
    # end
    for i in 1:length(crossings)
        if length(crossings[i]) == 2
            continue
        elseif length(crossings[i]) == 1
            if mle[i] < crossings[i][1]     # crossing is upper bound
                crossings[i] = [-max, crossings[i][1]]
            else
                crossings[i] = [crossings[i][1], max]
            end
        else
            throw("Error for i = $i, got $(length(crossings[i])) crossings.")
        end
    end
    HyperCube(minimum.(crossings), maximum.(crossings); Padding=Padding)
end
ProfileBox(DM::AbstractDataModel, M::AbstractVector{<:AbstractMatrix}, Confnum::Real=1; Padding::Real=0.) = ProfileBox(DM, InterpolatedProfiles(M), Confnum; Padding=Padding)
ProfileBox(DM::AbstractDataModel, Confnum::Real; Padding::Real=0., add::Real=1.5, kwargs...) = ProfileBox(DM, ProfileLikelihood(DM, Confnum+add; plot=false, kwargs...), Confnum; Padding=Padding)



"""
    PracticallyIdentifiable(DM::AbstractDataModel, Confnum::Real=1; plot::Bool=true, kwargs...) -> Real
Determines the maximum confidence level (in units of standard deviations σ) at which the given `DataModel` is still practically identifiable.
"""
PracticallyIdentifiable(DM::AbstractDataModel, Confnum::Real=1; plot::Bool=true, kwargs...) = PracticallyIdentifiable(ProfileLikelihood(DM, Confnum; plot=plot, kwargs...))

function PracticallyIdentifiable(Mats::AbstractVector{<:AbstractMatrix{<:Number}})
    function Minimax(M::AbstractMatrix)
        finitevals = isfinite.(view(M,:,2))
        sum(finitevals) == 0 && return Inf
        V = M[finitevals, 2]
        split = findmin(V)[2]
        min(maximum(view(V,1:split)), maximum(view(V,split:length(V))))
    end
    minimum([Minimax(M) for M in Mats])
end


abstract type AbstractProfile end

struct ParameterProfile <: AbstractProfile
    Profiles::AbstractVector{<:AbstractMatrix}
    Trajectories::AbstractVector{<:Union{<:AbstractVector{<:AbstractVector{<:Number}}, <:Nothing}}
    Names::AbstractVector{<:String}
    mle::Union{Nothing,<:AbstractVector{<:Number}}
    IsCost::Bool
    function ParameterProfile(DM::AbstractDataModel, Confnum::Union{Real,HyperCube}=2.; SaveTrajectories::Bool=false, IsCost::Bool=false, kwargs...)
        Profs = ProfileLikelihood(DM, Confnum; SaveTrajectories=SaveTrajectories, IsCost=IsCost, kwargs...)
        SaveTrajectories ? ParameterProfile(DM, getindex.(Profs,1), getindex.(Profs,2); IsCost=IsCost) : ParameterProfile(DM, Profs; IsCost=IsCost)
    end
    function ParameterProfile(DM::AbstractDataModel, Profiles::AbstractVector{<:AbstractMatrix}, Trajectories::AbstractVector=fill(nothing,length(Profiles)), Names::AbstractVector{<:String}=pnames(DM); IsCost::Bool=false)
        ParameterProfile(Profiles, Trajectories, Names, MLE(DM), IsCost)
    end
    function ParameterProfile(Profiles::AbstractVector{<:AbstractMatrix}, Trajectories::AbstractVector=fill(nothing,length(Profiles)), Names::AbstractVector{<:String}=CreateSymbolNames(length(Profiles),"θ"); IsCost::Bool=false)
        ParameterProfile(Profiles, Trajectories, Names, nothing, IsCost)
    end
    function ParameterProfile(Profiles::AbstractVector{<:AbstractMatrix}, Trajectories::AbstractVector, Names::AbstractVector{<:String}, mle, IsCost::Bool)
        @assert length(Profiles) == length(Names) == length(mle) == length(Trajectories)
        new(Profiles, Trajectories, Names, mle, IsCost)
    end
end
(P::ParameterProfile)(t::Real, i::Int) = InterpolatedProfiles(P,i)(t)
(P::ParameterProfile)(i::Int) = InterpolatedProfiles(P,i)
InterpolatedProfiles(P::ParameterProfile, i::Int) = QuadraticInterpolation(view(Profiles(P)[i],:,2), view(Profiles(P)[i],:,1))
InterpolatedProfiles(P::ParameterProfile) = [QuadraticInterpolation(view(Prof,:,2), view(Prof,:,1)) for Prof in Profiles(P)]

Profiles(P::ParameterProfile) = P.Profiles
Trajectories(P::ParameterProfile) = P.Trajectories
pnames(P::ParameterProfile) = P.Names
MLE(P::ParameterProfile) = P.mle
IsCost(P::ParameterProfile) = P.IsCost

Base.length(P::ParameterProfile) = Profiles(P) |> length
Base.firstindex(P::ParameterProfile) = Profiles(P) |> firstindex
Base.lastindex(P::ParameterProfile) = Profiles(P) |> lastindex
Base.getindex(P::ParameterProfile, ind) = getindex(Profiles(P), ind)


ProfileBox(P::ParameterProfile, Confnum::Real; kwargs...) = ProfileBox(InterpolatedProfiles(P), MLE(P), Confnum; kwargs...)
ProfileBox(DM::AbstractDataModel, P::ParameterProfile, Confnum::Real; kwargs...) = ProfileBox(P, Confnum; kwargs...)

PracticallyIdentifiable(P::ParameterProfile) = PracticallyIdentifiable(Profiles(P))


@recipe f(P::ParameterProfile) = P, Val(all(!isnothing, Trajectories(P)))
@recipe function f(P::ParameterProfile, HasTrajectories::Val{true})
    @assert length(pnames(P)) ≤ 3
    layout := length(pnames(P)) + 1
    @series P, Val(false)
    label --> reshape(["Comp $i" for i in 1:length(pnames(P))], 1, :)
    for i in 1:length(pnames(P))
        @series begin
            subplot := length(pnames(P)) + 1
            Trajectories(P)[i]
        end
    end
    @series begin
        label := "MLE"
        xguide --> pnames(P)[1]
        yguide --> pnames(P)[2]
        if length(pnames(P)) == 3
            zguide --> pnames(P)[3]
        end
        subplot := length(pnames(P)) + 1
        [MLE(P)]
    end
end
@recipe function f(P::ParameterProfile, HasTrajectories::Val{false})
    layout := length(pnames(P))
    for i in 1:length(pnames(P))
        @series begin
            label --> "Profile Likelihood"
            xguide --> pnames(P)[i]
            yguide --> (IsCost(P) ? "Cost Function" : "Conf. level [σ]")
            subplot := i
            view(Profiles(P)[i],:,1), view(Profiles(P)[i],:,2)
            # P(i)
        end
        # Draw prior contribution
        if size(Profiles(P)[i],2) == 3
            @series begin
                label --> "Prior contribution"
                color --> :red
                line --> :dash
                subplot := i
                view(Profiles(P)[i],:,1), view(Profiles(P)[i],:,3)
                # P(i)
            end
        end
        ## Mark MLE in profiles
        @series begin
            subplot := i
            legend --> nothing
            xguide --> pnames(P)[i]
            yguide --> (IsCost(P) ? "Cost Function" : "Conf. level [σ]")
            seriescolor --> :red
            marker --> :hex
            markersize --> 3
            markerstrokewidth --> 0
            [MLE(P)[i]], [P(MLE(P)[1],1)]
        end
    end
    ## Mark Integer Confidence Levels in Profile
    ## Rootfinding errors sometimes
    # if !IsCost(P) && length(1:Int(floor(PracticallyIdentifiable(P)))) > 0
    #     maxlevel = Int(floor(PracticallyIdentifiable(P)))
    #     Boxes = map(level->ProfileBox(P, level), 1:maxlevel)
    #     for i in 1:length(pnames(P))
    #         for level in 1:maxlevel
    #             @series begin
    #                 subplot := i
    #                 legend --> nothing
    #                 markeralpha --> 0
    #                 line --> :dash
    #                 seriescolor --> :red
    #                 range(Boxes[level][i]...; length=5), level*ones(5)
    #             end
    #         end
    #     end
    # end
end
