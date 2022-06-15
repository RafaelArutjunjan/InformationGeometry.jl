

# Returns a copy of type `Vector`, i.e. is not typesafe!
SafeCopy(X::AbstractVector) = copy(X)
SafeCopy(X::AbstractRange) = collect(X)
SafeCopy(X::Union{SVector,MVector}) = convert(Vector,X)

Drop(X::AbstractVector, i::Int) = (Z=SafeCopy(X);   splice!(Z,i);   Z)

_Presort(Components::AbstractVector{<:Int}; rev::Bool=false) = issorted(Components; rev=rev) ? Components : sort(Components; rev=rev)
Drop(X::AbstractVector, Components::AbstractVector{<:Int}) = (Z=SafeCopy(X); for i in _Presort(Components; rev=true) splice!(Z,i) end;    Z)
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
function ValInserter(Components::AbstractVector{<:Int}, Values::AbstractVector{<:AbstractFloat})
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

InsertIntoFirst(X::AbstractVector{<:Number}) = PassingIntoLast(θ::AbstractVector{<:Number}) = [X;θ]
InsertIntoLast(θ::AbstractVector{<:Number}) = PassingIntoFirst(X::AbstractVector{<:Number}) = [X;θ]


ProfilePredictor(DM::AbstractDataModel, args...) = ProfilePredictor(Predictor(DM), args...)
ProfilePredictor(M::ModelOrFunction, Comp::Int, PinnedValue::AbstractFloat) = EmbedModelVia(M, ValInserter(Comp, PinnedValue); Domain=(M isa ModelMap ? DropCubeDims(M.Domain, Comp) : nothing))
ProfilePredictor(M::ModelOrFunction, Comps::AbstractVector{<:Int}, PinnedValues::AbstractVector{<:AbstractFloat}) = EmbedModelVia(M, ValInserter(Comps, PinnedValues); Domain=(M isa ModelMap ? DropCubeDims(M.Domain, Comps) : nothing))

ProfileDPredictor(DM::AbstractDataModel, args...) = ProfileDPredictor(dPredictor(DM), args...)
ProfileDPredictor(dM::ModelOrFunction, Comp::Int, PinnedValue::AbstractFloat) = EmbedDModelVia(dM, ValInserter(Comp, PinnedValue); Domain=(dM isa ModelMap ? DropCubeDims(dM.Domain, Comp) : nothing))
ProfileDPredictor(dM::ModelOrFunction, Comps::AbstractVector{<:Int}, PinnedValues::AbstractVector{<:AbstractFloat}) = EmbedDModelVia(dM, ValInserter(Comps, PinnedValues); Domain=(dM isa ModelMap ? DropCubeDims(dM.Domain, Comps) : nothing))

"""
    PinParameters(DM::AbstractDataModel, Component::Int, Value::AbstractFloat)
    PinParameters(DM::AbstractDataModel, Components::AbstractVector{<:Int}, Values::AbstractVector{<:AbstractFloat})
Returns `DataModel` where one or more parameters have been pinned to specified values.
"""
function PinParameters(DM::AbstractDataModel, Components::Union{Int,AbstractVector{<:Int}}, Values::Union{AbstractFloat,AbstractVector{<:AbstractFloat}})
    @assert length(Components) == length(Values) && length(Components) < pdim(DM)
    DataModel(Data(DM), ProfilePredictor(DM, Components, Values), ProfileDPredictor(DM, Components, Values), Drop(MLE(DM), Components), EmbedLogPrior(DM, ValInserter(Components, Values)))
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
    GetProfile(DM::AbstractDataModel, Comp::Int, dom::Tuple{<:Real, <:Real}; N::Int=50, dof::Int=pdim(DM), SaveTrajectories::Bool=false) -> N×2 Matrix
Computes profile likelihood associated with the component `Comp` of the parameters over the domain `dom`.
"""
function GetProfile(DM::AbstractDataModel, Comp::Int, dom::Tuple{<:Real, <:Real}; N::Int=50, tol::Real=1e-9, IsCost::Bool=false, dof::Int=pdim(DM), SaveTrajectories::Bool=false, kwargs...)
    @assert dom[1] < dom[2] && (1 ≤ Comp ≤ pdim(DM))
    ps = DomainSamples(dom; N=N)

    # Could use variable size array instead to cut off computation once Confnum+0.1 is reached?
    Res = fill(-Inf, N)
    path = SaveTrajectories ? Vector{Vector{Float64}}(undef, N) : nothing
    if pdim(DM) == 1    # Cannot drop dims if pdim already 1
        Res = map(x->loglikelihood(DM, [x]), ps)
    else
        MLEstash = Drop(MLE(DM), Comp)
        for (i,p) in enumerate(ps)
            NewModel = ProfilePredictor(DM, Comp, p)
            DroppedLogPrior = EmbedLogPrior(DM, ValInserter(Comp,p))
            MLEstash = curve_fit(Data(DM), NewModel, ProfileDPredictor(DM, Comp, p), MLEstash, DroppedLogPrior; tol=tol, kwargs...).param
            SaveTrajectories && (path[i] = MLEstash)
            Res[i] = loglikelihood(Data(DM), NewModel, MLEstash, DroppedLogPrior)
        end
    end
    Logmax = max(maximum(Res), LogLikeMLE(DM))
    !(Logmax ≈ LogLikeMLE(DM)) && @warn "Profile Likelihood analysis apparently found a likelihood value which is larger than the previously stored LogLikeMLE. Continuing anyway."
    # Using pdim(DM) instead of 1 here, because it gives the correct result
    if IsCost
        @. Res = 2*(Logmax - Res)
    else
        @inbounds for i in eachindex(Res)
            Res[i] = InvConfVol(ChisqCDF(dof, 2(Logmax - Res[i])))
        end
    end

    if SaveTrajectories
        for (i,p) in enumerate(ps)
            insert!(path[i], Comp, p)
        end
        [ps Res], path
    else
        [ps Res]
    end
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
    plot && ProfilePlotter(DM, Profiles)
    Profiles
end


function ProfilePlotter(DM::AbstractDataModel, Profiles::AbstractVector;
    Pnames::AbstractVector{<:String}=(Predictor(DM) isa ModelMap ? pnames(Predictor(DM)) : CreateSymbolNames(pdim(DM), "θ")), kwargs...)
    @assert length(Profiles) == length(Pnames)
    Ylab = length(Pnames) == pdim(DM) ? "Conf. level [σ]" : "Cost Function"
    PlotObjects = if Profiles isa AbstractVector{<:AbstractMatrix{<:Number}}
        [RecipesBase.plot(view(Profiles[i], :,1), view(Profiles[i], :,2); leg=false, xlabel=Pnames[i], ylabel=Ylab) for i in 1:length(Profiles)]
    else
        P1 = [RecipesBase.plot(view(Profiles[i][1], :,1), view(Profiles[i][1], :,2); leg=false, xlabel=Pnames[i], ylabel=Ylab) for i in 1:length(Profiles)]
        if length(Profiles) ≤ 3
            P2 = PlotProfileTrajectories(DM, Profiles)
            vcat(P1,[P2])
        else
            P1
        end
    end
    RecipesBase.plot(PlotObjects...; layout=length(PlotObjects)) |> display
end
# Plot trajectories of Profile Likelihood
"""
    PlotProfileTrajectories(DM::AbstractDataModel, Profiles::AbstractVector{Tuple{AbstractMatrix,AbstractVector}}; OverWrite=true, kwargs...)
"""
function PlotProfileTrajectories(DM::AbstractDataModel, Profiles::AbstractVector; OverWrite=true, kwargs...)
    @assert Profiles[1][1] isa AbstractMatrix{<:Number} && Profiles[1][2] isa AbstractVector{<:AbstractVector{<:Number}}
    P = OverWrite ? RecipesBase.plot() : RecipesBase.plot!()
    for i in 1:length(Profiles)
        RecipesBase.plot!(P, Profiles[i][2]; marker=:circle, label="Comp: $i", kwargs...)
    end
    RecipesBase.plot!(P, [MLE(DM)]; linealpha=0, marker=:hex, markersize=3, label="MLE", kwargs...)
    P
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
            legend --> nothing
            xguide --> pnames(P)[i]
            yguide --> (IsCost(P) ? "Cost Function" : "Conf. level [σ]")
            subplot := i
            view(Profiles(P)[i],:,1), view(Profiles(P)[i],:,2)
            # P(i)
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
