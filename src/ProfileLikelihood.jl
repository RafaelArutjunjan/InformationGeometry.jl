


function DropVec(i::Int, dim::Int)
    keep = trues(dim);    keep[i] = false;    keep
end
Drop(X::AbstractVector, i::Int) = X[DropVec(i, length(X))]



# vcat(X[1:Comp-1], [Val], X[Comp:end])
InsertValAt(X::AbstractVector{<:Number}, Comp::Int, Val::AbstractFloat) = insert!(copy(X), Comp, Val)
# Shouldn't need this:
InsertValAt(X::Number, Comp::Int, Val::AbstractFloat) = InsertValAt([X], Comp, Val)

"""
Naively computes approximate 1D domain from inverse Fisher metric at MLE.
"""
function GetDomainTuple(DM::AbstractDataModel, Comp::Int, Confnum::Real; ForcePositive::Bool=false)::Tuple
    @assert 1 ≤ Comp ≤ pdim(DM)
    ApproxDev = try
        sqrt(inv(FisherMetric(DM, MLE(DM)))[Comp,Comp])
    catch;
        try
            1 / sqrt(FisherMetric(DM, MLE(DM))[Comp,Comp])
        catch;
            1e-8
        end
    end
    ApproxDev *= sqrt(InvChisqCDF(pdim(DM), ConfVol(Confnum)))
    ForcePositive && @assert MLE(DM)[Comp]+ApproxDev > 0.

    # If Bio model, the pinned rate parameter should remain positive
    start = (MLE(DM)[Comp]-ApproxDev < 0. && ForcePositive) ? 1e-12 : MLE(DM)[Comp]-ApproxDev
    # ps = range(start, MLE(DM)[Comp]+ApproxDev; length=N)
    (start, MLE(DM)[Comp]+ApproxDev)
end



ProfilePredictor(DM::AbstractDataModel, Comp::Int, PinnedValue::AbstractFloat) = ProfilePredictor(Predictor(DM), Comp, PinnedValue)
ProfilePredictor(M::ModelOrFunction, Comp::Int, PinnedValue::AbstractFloat) = EmbedModelVia(M, X->InsertValAt(X, Comp, PinnedValue); Domain=(M isa ModelMap ? DropCubeDim(M.Domain, Comp) : nothing))

ProfileDPredictor(DM::AbstractDataModel, Comp::Int, PinnedValue::AbstractFloat) = ProfileDPredictor(dPredictor(DM), Comp, PinnedValue)
ProfileDPredictor(dM::ModelOrFunction, Comp::Int, PinnedValue::AbstractFloat) = EmbedDModelVia(dM, X->InsertValAt(X, Comp, PinnedValue); Domain=(dM isa ModelMap ? DropCubeDim(dM.Domain, Comp) : nothing))

"""
    GetProfile(DM::AbstractDataModel, Comp::Int, dom::Tuple{<:Real, <:Real}; N::Int=50) -> N×2 Matrix
Computes profile likelihood associated with the component `Comp` of the parameters over the domain `dom`.
"""
function GetProfile(DM::AbstractDataModel, Comp::Int, dom::Tuple{<:Real, <:Real}; N::Int=50)
    @assert dom[1] < dom[2] && (1 ≤ Comp ≤ pdim(DM))
    ps = DomainSamples(dom; N=N)

    # Could use variable size array instead to cut off computation once Confnum+0.1 is reached?
    Res = Vector{Float64}(undef, N)
    if pdim(DM) == 1    # Cannot drop dims if pdim already 1
        Res = map(x->loglikelihood(DM, [x]), ps)
    else
        MLEstash = Drop(MLE(DM), Comp)
        for (i,p) in enumerate(ps)
            NewModel = ProfilePredictor(DM, Comp, p)
            MLEstash = curve_fit(Data(DM), NewModel, ProfileDPredictor(DM, Comp, p), MLEstash).param
            Res[i] = loglikelihood(Data(DM), NewModel, MLEstash)
        end
    end
    Logmax = max(maximum(Res), LogLikeMLE(DM))
    # Using pdim(DM) instead of 1 here, because it gives the correct result
    Res = map(x->InvConfVol.(ChisqCDF.(pdim(DM), 2(Logmax - x))), Res)
    [ps Res]
end

function GetProfile(DM::AbstractDataModel, Comp::Int, Confnum::Real=2; N::Int=50, ForcePositive::Bool=false)
    GetProfile(DM, Comp, GetDomainTuple(DM, Comp, Confnum; ForcePositive=ForcePositive); N=N)
end


"""
    ProfileLikelihood(DM::AbstractDataModel, Confnum::Real=2; N::Int=50, ForcePositive::Bool=false, plot::Bool=true, parallel::Bool=false) -> Vector{Matrix}
Computes the profile likelihood for each component of the parameters ``θ \\in \\mathcal{M}`` over the given `Domain`.
Returns a vector of N×2 matrices where the first column of the n-th matrix specifies the value of the n-th component and the second column specifies the associated confidence level of the best fit configuration conditional to the n-th component being fixed at the associated value in the first column.

The domain over which the profile likelihood is computed is not (yet) adaptively chosen. Instead the size of the domain is estimated from the inverse Fisher metric.
Therefore, often has to pass higher value for `Confnum` to this method than the confidence level one is actually interested in, to ensure that it is still covered (if the model is even practically identifiable in the first place).
"""
function ProfileLikelihood(DM::AbstractDataModel, Confnum::Real=2; N::Int=50, ForcePositive::Bool=false, plot::Bool=true, parallel::Bool=false)
    ProfileLikelihood(DM, HyperCube([GetDomainTuple(DM, i, Confnum; ForcePositive=ForcePositive) for i in 1:pdim(DM)]); N=N, plot=plot, parallel=parallel)
end

function ProfileLikelihood(DM::AbstractDataModel, Domain::HyperCube; N::Int=50, plot::Bool=true, parallel::Bool=false)
    Map = parallel ? pmap : map
    Profiles = Map(i->GetProfile(DM, i, (Domain.L[i], Domain.U[i]); N=N), 1:pdim(DM))
    if plot
        Pnames = Predictor(DM) isa ModelMap ? pnames(Predictor(DM)) : CreateSymbolNames(pdim(DM), "θ")
        PlotObjects = [Plots.plot(view(Profiles[i], :,1), view(Profiles[i], :,2), leg=false, xlabel=Pnames[i], ylabel="Conf. level [σ]") for i in 1:pdim(DM)]
        Plots.plot(PlotObjects..., layout=pdim(DM)) |> display
    end
    Profiles
end

"""
    InterpolatedProfiles(M::AbstractVector{<:AbstractMatrix}) -> Vector{Function}
Interpolates the `Vector{Matrix}` output of ProfileLikelihood() with cubic splines.
"""
function InterpolatedProfiles(Mats::AbstractVector{<:AbstractMatrix})
    [CubicSpline(view(profile,:,2), view(profile,:,1)) for profile in Mats]
end

"""
    ProfileBox(DM::AbstractDataModel, Fs::AbstractVector{<:DataInterpolations.AbstractInterpolation}, Confnum::Real=1.) -> HyperCube
Constructs `HyperCube` which bounds the confidence region associated with the confidence level `Confnum` from the interpolated likelihood profiles.
"""
function ProfileBox(DM::AbstractDataModel, Fs::AbstractVector{<:DataInterpolations.AbstractInterpolation}, Confnum::Real=1.; Padding::Real=0.)
    domains = map(F->(F.t[1], F.t[end]), Fs)
    crossings = [find_zeros(x->(Fs[i](x)-Confnum), domains[i][1], domains[i][2]) for i in 1:length(Fs)]
    for i in 1:length(crossings)
        if length(crossings[i]) == 2
            continue
        elseif length(crossings[i]) == 1
            if MLE(DM)[i] < crossings[i][1]     # crossing is upper bound
                crossings[i] = [-10000.0, crossings[i][1]]
            else
                crossings[i] = [crossings[i][1], 10000.0]
            end
        else
            throw("Error for i = $i")
        end
    end
    HyperCube(minimum.(crossings), maximum.(crossings); Padding=Padding)
end
ProfileBox(DM::AbstractDataModel, M::AbstractVector{<:AbstractMatrix}, Confnum::Real=1; Padding::Real=0.) = ProfileBox(DM, InterpolatedProfiles(M), Confnum; Padding=Padding)
ProfileBox(DM::AbstractDataModel, Confnum::Real; N::Int=50, Padding::Real=0., add::Real=1.5) = ProfileBox(DM, ProfileLikelihood(DM, Confnum+add; N=N, plot=false), Confnum; Padding=Padding)



"""
    PracticallyIdentifiable(DM::AbstractDataModel, Confnum::Real=1; plot::Bool=true, kwargs...) -> Real
Determines the maximum confidence level (in units of standard deviations σ) at which the given `DataModel` is still practically identifiable.
"""
PracticallyIdentifiable(DM::AbstractDataModel, Confnum::Real=1; plot::Bool=true, kwargs...) = PracticallyIdentifiable(ProfileLikelihood(DM, Confnum; plot=plot, kwargs...))

function PracticallyIdentifiable(Mats::AbstractVector{<:AbstractMatrix{<:Number}})
    function Minimax(M::AbstractMatrix)
        finitevals = isfinite.(M[:,2])
        V = M[finitevals, 2]
        split = findmin(V)[2]
        min(maximum(V[1:split]), maximum(V[split:end]))
    end
    minimum([Minimax(M) for M in Mats])
end
