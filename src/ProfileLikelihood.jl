
RateParamCube(n::Int) = HyperCube(10^-5 * ones(n), 10^2 * ones(n))
function DropVec(i::Int, dim::Int)
    keep = trues(dim);    keep[i] = false;    keep
end
Drop(X::AbstractVector, i::Int) = X[DropVec(i, length(X))]

function EmbedModelVia(model::Function, F::Function)
    EmbeddedModel(x, θ; kwargs...) = model(x, F(θ); kwargs...)
end
function EmbedModelVia(M::ModelMap, F::Function; Domain::Union{Nothing, HyperCube}=nothing, ForcePositive::Bool=false)
    Finputdim = Domain isa Nothing ? InformationGeometry.GetArgLength(F, M.xyp[3]) : length(Domain)
    if Domain isa Nothing
        Domain = ForcePositive ? RateParamCube(Finputdim) : FullDomain(Finputdim)
    end
    ModelMap(EmbedModelVia(M.Map, F), (M.InDomain∘F), Domain, (M.xyp[1], M.xyp[2], Finputdim),InformationGeometry.CreateSymbolNames(Finputdim, "θ"), M.StaticOutput, M.inplace, M.CustomEmbedding)
end

ProfilePredictor(DM::AbstractDataModel, Comp::Int, PinnedValue::AbstractFloat) = ProfilePredictor(Data(DM), Predictor(DM), Comp, PinnedValue)
function ProfilePredictor(DS::AbstractDataSet, M::ModelMap, Comp::Int, PinnedValue::AbstractFloat)
    EmbedModelVia(M, X->InsertValAt(X, Comp, PinnedValue); Domain=DropCubeDim(M.Domain, Comp))
end
function ProfilePredictor(DS::AbstractDataSet, model::Function, Comp::Int, PinnedValue::AbstractFloat)
    EmbedModelVia(model, X->InsertValAt(X, Comp, PinnedValue))
end

# vcat(X[1:Comp-1], [Val], X[Comp:end])
InsertValAt(X::AbstractVector{<:Number}, Comp::Int, Val::AbstractFloat) = insert!(copy(X), Comp, Val)
# Shouldn't need this:
InsertValAt(X::Number, Comp::Int, Val::AbstractFloat) = InsertValAt([X], Comp, Val)

"""
Naively computes approximate 1D domain from inverse Fisher metric at MLE.
"""
function GetDomainTuple(DM::AbstractDataModel, Comp::Int, Confnum::Real; ForcePositive::Bool=false)::Tuple
    @assert 1 ≤ Comp ≤ pdim(DM)
    ApproxDev = Confnum * sqrt(inv(FisherMetric(DM, MLE(DM)))[Comp,Comp])
    ForcePositive && @assert MLE(DM)[Comp]+ApproxDev > 0.

    # If Bio model, the pinned rate parameter should remain positive
    start = (MLE(DM)[Comp]-ApproxDev < 0. && ForcePositive) ? 1e-12 : MLE(DM)[Comp]-ApproxDev
    # ps = range(start, MLE(DM)[Comp]+ApproxDev; length=N)
    (start, MLE(DM)[Comp]+ApproxDev)
end


"""
    GetProfile(DM::AbstractDataModel, Comp::Int, dom::Tuple{<:Real, <:Real}; N::Int=50) -> N×2 Matrix
Computes profile likelihood associated with the component `Comp` of the parameters over the domain `dom`.
"""
function GetProfile(DM::AbstractDataModel, Comp::Int, dom::Tuple{<:Real, <:Real}; N::Int=50)
    @assert dom[1] < dom[2] && (1 ≤ Comp ≤ pdim(DM))
    ps = range(dom[1], dom[2]; length=N)

    # Could use variable size array instead to cut off computation once Confnum+0.1 is reached?
    Res = Vector{Float64}(undef, N)
    if pdim(DM) == 1    # Cannot drop dims if pdim already 1
        Res = map(x->loglikelihood(DM, [x]), ps)
    else
        MLEstash = Drop(MLE(DM), Comp)
        for (i,p) in enumerate(ps)
            NewModel = ProfilePredictor(DM, Comp, p)
            MLEstash = FindMLE(Data(DM), NewModel, MLEstash)
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
        pnames = Predictor(DM) isa ModelMap ? InformationGeometry.pnames(Predictor(DM)) : InformationGeometry.CreateSymbolNames(pdim(DM), "θ")
        PlotObjects = [Plots.plot(view(Profiles[i], :,1), view(Profiles[i], :,2), leg=false, xlabel=pnames[i], ylabel="Conf. level [σ]") for i in 1:pdim(DM)]
        Plots.plot(PlotObjects..., layout=pdim(DM)) |> display
    end
    Profiles
end

"""
    InterpolatedProfiles(M::AbstractVector{<:AbstractMatrix}) -> Vector{Function}
Interpolates the `Vector{Matrix}` output of ProfileLikelihood() with cubic splines.
"""
function InterpolatedProfiles(M::AbstractVector{<:AbstractMatrix})
    [CubicSpline(view(profile,:,2), view(profile,:,1)) for profile in M]
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
