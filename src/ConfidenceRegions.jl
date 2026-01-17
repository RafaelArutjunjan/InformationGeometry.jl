


"""
Point θ lies outside confidence region of level `Confvol` if this function > 0.
"""
WilksCriterion(DM::AbstractDataModel, θ::AbstractVector{<:BigFloat}, Confvol::BigFloat=ConfVol(BigFloat(1.)); dof::Int=DOF(DM), kwargs...) = ChisqCDF(dof, 2(LogLikeMLE(DM) - loglikelihood(DM,θ; kwargs...))) - Confvol
WilksCriterion(DM::AbstractDataModel, θ::AbstractVector{<:Number}, Confvol::Real=ConfVol(1.); dof::Int=DOF(DM), kwargs...) = cdf(Chisq(dof), 2(LogLikeMLE(DM) - loglikelihood(DM, θ; kwargs...))) - Confvol

# Do not give default to third argument here such as to not overrule the defaults from above
WilksCriterion(DM::AbstractDataModel, θ::AbstractVector{<:Float64}, Confvol::BigFloat; kwargs...) = WilksCriterion(DM, BigFloat.(θ), Confvol; kwargs...)
WilksCriterion(DM::AbstractDataModel, θ::AbstractVector{<:BigFloat}, Confvol::Float64; kwargs...) = WilksCriterion(DM, θ, BigFloat(Confvol); kwargs...)

"""
    WilksTest(DM::DataModel, θ::AbstractVector{<:Number}, Confvol=ConfVol(1)) -> Bool
Checks whether a given parameter configuration `θ` is within a confidence interval of level `Confvol` using Wilks' theorem.
This makes the assumption, that the likelihood has the form of a normal distribution, which is asymptotically correct in the limit that the number of datapoints is infinite.
The keyword `dof` can be used to manually specify the degrees of freedom.
"""
WilksTest(DM::AbstractDataModel, θ::AbstractVector{<:Number}, Confvol::Real=ConfVol(one(suff(θ))); kwargs...)::Bool = WilksCriterion(DM, θ, Confvol; kwargs...) < 0.

# Convert BigFloat vector back to float if necessary
SmallFloat(X::AbstractVector{<:Number}, tol::Real) = X
SmallFloat(X::AbstractVector{<:BigFloat}, tol::Real) = tol < 2e-15 ? X : Float64.(X)

function _GetBoolTesterFunc(DM::AbstractDataModel, mle::AbstractVector, CF::Real; dof::Int=DOF(DM), Comp::Int=1, kwargs...)
    Wilks_Test(x::Real) = WilksTest(DM, muladd(x, BasisVector(Comp, length(mle)), mle), CF; dof, kwargs...)
end
function _GetFloatTesterFunc(DM::AbstractDataModel, mle::AbstractVector, CF::Real; dof::Int=DOF(DM), Comp::Int=1, kwargs...)
    Wilks_Criterion(x::Real) = WilksCriterion(DM, muladd(x, BasisVector(Comp, length(mle)), mle), CF; dof, kwargs...)
end
function _GetBoolFTesterFunc(DM::AbstractDataModel, mle::AbstractVector, CF::Real; Comp::Int=1, kwargs...)
    F_Test(x::Real) = Ftest(DM, muladd(x, BasisVector(Comp, length(mle)), mle), CF; kwargs...)
end
function _GetFloatFTesterFunc(DM::AbstractDataModel, mle::AbstractVector, CF::Real; Comp::Int=1, kwargs...)
    F_Criterion(x::Real) = FCriterion(DM, muladd(x, BasisVector(Comp, length(mle)), mle), CF; kwargs...)
end

"""
    FindConfBoundary(DM::AbstractDataModel, Confnum::Real; BoolTest::Bool=(Confnum > 8), Ftest::Bool=false, tol::Real=4e-15)
Finds parameter configuration which lies on the boundary of the confidence region of level `Confnum`σ.

* `BoolTest` can be used to specify whether the threshold is found using a `Bool`-valued test or a `Float`-valued test. Since it uses less memory, the `Bool`-valued test performs better when using `BigFloat` (i.e. when Confnum > 8).
* `Ftest=true` uses the F-test rather than the Wilks test to define the threshold. Typically, the F-test will yield more conservative estimates (i.e. larger confidence regions) since it accounts for small sample sizes.
"""
function FindConfBoundary(DM::AbstractDataModel, Confnum::Real; BoolTest::Bool=(Confnum > 8), Ftest::Bool=false, tol::Real=4e-15, dof::Int=DOF(DM), Comp::Int=1, verbose::Bool=true, factor::Real=10.0, kwargs...)
    CF = tol < 2e-15 ? ConfVol(BigFloat(Confnum); verbose=verbose) : ConfVol(Confnum; verbose=verbose)
    mle = if CF isa BigFloat
        verbose && suff(MLE(DM)) != BigFloat && @info "FindConfBoundary: Promoting MLE to BigFloat and continuing. However, it is advisable to promote the entire DataModel object via DM = BigFloat(DM) instead."
        BigFloat.(MLE(DM))
    else    MLE(DM)     end

    Res = if BoolTest || mle isa AbstractVector{<:BigFloat}
        verbose && !BoolTest && @info "FindConfBoundary: Promoting to BoolTest=true since MLE is a BigFloat."
        Test = (Ftest ? _GetBoolFTesterFunc : _GetBoolTesterFunc)(DM, mle, CF; dof=dof, Comp=Comp)
        _FindBoolBoundary(Test, mle; tol=tol, dof=dof, Comp=Comp, verbose=verbose, kwargs...)
    else
        Test = (Ftest ? _GetFloatFTesterFunc : _GetFloatTesterFunc)(DM, mle, CF; dof=dof, Comp=Comp)
        Interval = _BracketingInterval(DM, CF; dof=dof, Comp=Comp, factor=factor)
        _FindFloatBoundary(Test, Interval, mle; tol=tol, dof=dof, Comp=Comp, verbose=verbose, kwargs...)
    end;    SmallFloat(Res, tol)
end

function _FindBoolBoundary(Test::Function, mle::AbstractVector{<:Number}; tol::Real=4e-15, dof::Int=length(mle), Comp::Int=1, verbose::Bool=true, kwargs...)
    muladd(LineSearch(Test, zero(suff(mle)); tol=tol, verbose=verbose, kwargs...), BasisVector(Comp, length(mle)), mle)
end
function _FindFloatBoundary(DM::AbstractDataModel, Test::Function, mle::AbstractVector{<:Number}, CF::Real; dof::Int=DOF(DM), Comp::Int=1, factor::Real=10.0, kwargs...)
    Interval = _BracketingInterval(DM, CF; dof=dof, Comp=Comp, factor=factor)
    _FindFloatBoundary(Test, Interval, mle; dof=dof, Comp=Comp, kwargs...)
end
function _FindFloatBoundary(Test::Function, Interval::Tuple{<:Real,<:Real}, mle::AbstractVector{<:Number}; tol::Real=4e-15, Comp::Int=1, verbose::Bool=true,
                            meth::Roots.AbstractUnivariateZeroMethod=Roots.AlefeldPotraShi(), kwargs...)
    muladd(AltLineSearch(Test, Interval, meth; tol=tol), BasisVector(Comp, length(mle)), mle)
end

function _BracketingInterval(DM::AbstractDataModel, CF::Real; dof::Int=DOF(DM), Comp::Int=1, factor::Real=10.0)
    b = sqrt(InvChisqCDF(dof, CF) / FisherMetric(DM,MLE(DM))[Comp,Comp])
    (b/factor, factor*b)
end


## old
function FindConfBoundaryOld(DM::AbstractDataModel, Confnum::Real; tol::Real=4e-15, dof::Int=DOF(DM), verbose::Bool=true, kwargs...)
    CF = tol < 2e-15 ? ConfVol(BigFloat(Confnum); verbose=verbose) : ConfVol(Confnum; verbose=verbose)
    mle = if CF isa BigFloat
        verbose && suff(MLE(DM)) != BigFloat && @info "FindConfBoundary: Promoting MLE to BigFloat and continuing. However, it is advisable to promote the entire DataModel object via DM = BigFloat(DM) instead."
        BigFloat.(MLE(DM))
    else
        MLE(DM)
    end
    FindConfBoundaryOld(_GetBoolTesterFunc(DM, mle, CF; dof=dof), mle; tol=tol, verbose=verbose, kwargs...)
end
function FindConfBoundaryOld(Test::Function, mle::AbstractVector{<:Number}; tol::Real=4e-15, Comp::Int=1, kwargs...)
    Res = muladd(LineSearch(Test, zero(suff(mle)); tol=tol, kwargs...), BasisVector(Comp, length(mle)), mle)
    SmallFloat(Res, tol)
end
# Takes roughly 1/3 of the time of Boolean LineSearch
function FindConfBoundaryOld2(DM::AbstractDataModel, Confnum::Real; tol::Real=4e-15, dof::Int=DOF(DM), Comp::Int=1, maxiter::Int=-1, factor::Real=10.0, verbose::Bool=true,
                        meth::Roots.AbstractUnivariateZeroMethod=Roots.AlefeldPotraShi())
    CF = ConfVol(Confnum; verbose=verbose)
    muladd(AltLineSearch(_GetFloatTesterFunc(DM, MLE(DM), CF; dof=dof, Comp=Comp), _BracketingInterval(DM, CF; dof=dof, Comp=Comp, factor=factor), meth; tol), BasisVector(Comp, pdim(DM)), MLE(DM))
end


# equivalent to ResidualSquares(DM,MLE(DM))
RS_MLE(DM::AbstractDataModel) = logdetInvCov(DM) - DataspaceDim(DM)*log(2π) - 2LogLikeMLE(DM)
ResidualSquares(DM::AbstractDataModel, θ::AbstractVector{<:Number}=MLE(DM); kwargs...) = InnerProduct(yInvCov(DM, θ), ydata(DM) - EmbeddingMap(DM, θ; kwargs...))
ChisquaredReduced(DM::AbstractDataModel, θ::AbstractVector{<:Number}=MLE(DM); dof::Int=DOF(DM), kwargs...) = ResidualSquares(DM, θ; kwargs...) / (DataspaceDim(DM) - dof)
ChisquaredReduced(DM::AbstractDataModel, R::AbstractMultistartResults; kwargs...) = ChisquaredReduced(DM, MLE(R); kwargs...)

function FCriterion(DM::AbstractDataModel, θ::AbstractVector{<:Number}, Confvol::Real=ConfVol(one(suff(θ))); kwargs...)
    n = length(ydata(DM));  p = length(θ)
    ResidualSquares(DM,θ) - RS_MLE(DM) * (1. + length(θ)/(n - p)) * quantile(FDist(p, n-p), Confvol)
end
function FTest(DM::AbstractDataModel, θ::AbstractVector{<:Number}, Confvol::Real=ConfVol(one(suff(θ))); kwargs...)::Bool
    FCriterion(DM, θ, Confvol; kwargs...) < 0.0
end
"""
    FindFBoundary(DM::DataModel, Confnum::Real)
Finds parameter configuration for which the threshold of the F-test associated with a confidence level of `Confnum`σ is reached, i.e. which lies on the boundary of the F-test based confidence region of level `Confnum`σ.
"""
FindFBoundary(args...; kwargs...) = FindConfBoundary(args...; Ftest=true, kwargs...)


# Careful, often needs an additional factor of d1 (i.e. dof) multiplied additionally for comparability against chi^2
FDistCDF(x::T, d1::Number, d2::Number) where T<:Number = beta_inc(T(d1)/2.0, T(d2)/2.0, d1*x/(d1*x + d2))[1]



inversefactor(m) = 1. / sqrt((m - 1.) + (m - 1.)^2)
GetAlpha(x::AbstractVector{<:Number}) = GetAlpha(length(x))
@inline function GetAlpha(n::Int)
    V = Vector{Float64}(undef, n)
    Inv = inversefactor(n)
    fill!(V, -Inv)
    V[end] = (n-1) * Inv
    V
end

"""
    OrthVF(DM::DataModel, θ::AbstractVector{<:Number}; ADmode::Val=Val(:ForwardDiff)) -> Vector
Calculates a direction (in parameter space) in which the value of the log-likelihood does not change, given a parameter configuration ``\\theta``.
`ADmode=Val(false)` computes the Score by separately evaluating the `model` as well as the Jacobian `dmodel` provided in `DM`.
Other choices of `ADmode` directly compute the Score by differentiating the formula the log-likelihood, i.e. only one evaluation on a dual variable is performed.
"""
function OrthVF(DM::AbstractDataModel, θ::AbstractVector{<:Number}; alpha::AbstractVector=GetAlpha(length(θ)), ADmode::Val=Val(:ForwardDiff), kwargs...)
    length(θ) < 2 && throw(ArgumentError("dim(Parameter Space) < 2  --> No orthogonal VF possible."))
    # completely non-allocating version
    S = -Score(DM, θ; kwargs...);    P = prod(S)
    normalize(alpha .* P ./ S)
    # alpha .*= P;    alpha ./= S;    normalize!(alpha);    alpha
end
OrthVF(DM::AbstractDataModel; Kwargs...) = (args...; kwargs...)->OrthVF(DM, args...; Kwargs..., kwargs...)

# Faster Method for Planar OrthVF
"""
    OrthVF(DM::DataModel, PL::Plane, θ::AbstractVector{<:Number}; ADmode::Val=Val(:ForwardDiff)) -> Vector
Calculates a direction (in parameter space) in which the value of the log-likelihood does not change, given a parameter configuration ``\\theta``.
Since a 2D `Plane` is specified, both the input `θ` as well as well as the output have 2 components.
`ADmode=Val(false)` computes the Score by separately evaluating the `model` as well as the Jacobian `dmodel` provided in `DM`.
Other choices of `ADmode` directly compute the Score by differentiating the formula the log-likelihood, i.e. only one evaluation on a dual variable is performed.
"""
function OrthVF(DM::AbstractDataModel, PL::Plane, θ::AbstractVector{<:Number}, PlanarLogPrior::Union{Nothing,Function}=EmbedLogPrior(DM,PL); ADmode::Val=Val(:ForwardDiff), kwargs...)
    S = transpose(Projector(PL)) * (-Score(Data(DM), Predictor(DM), dPredictor(DM), PlaneCoordinates(PL,θ), PlanarLogPrior; ADmode=ADmode, kwargs...))
    P = prod(S);    normalize(SA[-P/S[1],P/S[2]])
end
# Catch ADmode
function OrthVF(DM::AbstractConditionGrid, PL::Plane, θ::AbstractVector{<:Number}, PlanarLogPrior::Union{Nothing,Function}=nothing; ADmode::Val=Val(:ForwardDiff), kwargs...)
    S = transpose(Projector(PL)) * (-Score(DM, PlaneCoordinates(PL,θ); kwargs...))
    P = prod(S);    normalize(SA[-P/S[1],P/S[2]])
end

"""
    OrthVF!(du::AbstractVector, DM::DataModel, θ::AbstractVector{<:Number}; ADmode::Val=Val(:ForwardDiff)) -> Vector
Calculates a direction (in parameter space) in which the value of the log-likelihood does not change, given a parameter configuration ``\\theta``.
`ADmode=Val(false)` computes the Score by separately evaluating the `model` as well as the Jacobian `dmodel` provided in `DM`.
Other choices of `ADmode` directly compute the Score by differentiating the formula the log-likelihood, i.e. only one evaluation on a dual variable is performed.
"""
function OrthVF!(du::AbstractVector, DM::AbstractDataModel, θ::AbstractVector; alpha::AbstractVector=GetAlpha(length(θ)), ADmode::Val=Val(:ForwardDiff), kwargs...)
    _turn!(du, -Score(DM, θ; kwargs...), alpha)
end

# Method for general functions F
function OrthVF(F::Function, θ::AbstractVector{<:Number}; ADmode::Union{Symbol,Val}=Val(:ForwardDiff), Grad=GetGrad(ADmode,F), alpha::AbstractVector=GetAlpha(length(θ)))
    _turn(Grad(θ), alpha)
end
function OrthVF!(du::AbstractVector, F::Function, θ::AbstractVector{<:Number}; ADmode::Union{Symbol,Val}=Val(:ForwardDiff), Grad=GetGrad(ADmode,F), alpha::AbstractVector=GetAlpha(length(θ)))
    _turn!(du, Grad(θ), alpha)
end
# Could try inplace Grad method into du, then divide inplace in alpha, then copy result back to du?

function OrthVF(F::Function; ADmode::Union{Symbol,Val}=Val(:ForwardDiff), Grad=GetGrad(ADmode,F), alpha::AbstractVector=GetAlpha(length(θ)))
    OrthogonalVectorField(θ::AbstractVector; alpha::AbstractVector=alpha) = _OrthVF(Grad, θ; alpha=alpha)
end

"""
    _OrthVF(dF::Function, θ::AbstractVector{<:Number}; alpha::AbstractVector=GetAlpha(length(θ)))
Computes OrthVF by evaluating the GRADIENT dF.
"""
_OrthVF(dF::Function, θ::AbstractVector{<:Number}; alpha::AbstractVector=GetAlpha(length(θ))) = _turn(dF(θ), alpha)
_OrthVF!(Res::AbstractVector{<:Number}, dF::Function, θ::AbstractVector{<:Number}; alpha::AbstractVector=GetAlpha(length(θ))) = _turn!(Res, dF(θ), alpha)

function _turn!(Res::AbstractVector, S::AbstractVector, α::AbstractVector)
    P = prod(S);    α .*= P;    map!(/, Res, α, S);    normalize!(Res)
end

# Out-of-place versions
_turn(S::AbstractVector, α::AbstractVector) = (Res=similar(S);  _turn!(Res, S, α);  Res)
_turn(S::AbstractVector, α::AbstractVector, normalize::Val{true}) = _turn(S, α)
_turn(S::AbstractVector, α::AbstractVector, normalize::Val{false}) = (P = prod(S);  P .* α ./ S)



GetStartP(DM::AbstractDataModel) = GetStartP(Data(DM), Predictor(DM), pdim(DM))
GetStartP(DS::AbstractDataSet, model::Function, hint::Int=pdim(DS,model)) = GetStartP(hint)
GetStartP(DS::AbstractFixedUncertaintyDataSet, model::ModelMap, hint::Int=-42) = ElaborateGetStartP(model)
function GetStartP(DS::AbstractUnknownUncertaintyDataSet, model::ModelMap, hint::Int=pdim(DS,model))
    hint == length(Domain(model)) && return ElaborateGetStartP(model)
    if hint == length(Domain(model)) + errormoddim(DS)
        # Append error parameter guesses slightly larger than zero
        vcat(ElaborateGetStartP(model), 0.1rand(errormoddim(DS)))
    elseif xpars(DS) > 0 && hint == xpars(DS) + length(Domain(model)) + errormoddim(DS)
        # Append error parameter guesses slightly larger than zero
        vcat(xdata(DS), ElaborateGetStartP(model), 0.1rand(errormoddim(DS)))
    else
        @warn "Got pdim=$hint but xpars + given ModelMap Domain + errormod = $(xpars(DS) + length(Domain(model)) + errormoddim(DS)) parameters."
        GetStartP(hint)
    end
end
GetStartP(hint::Int) = Ones(hint) .+ 0.05.*(rand(hint) .- 0.5)

ElaborateGetStartP(M::ModelMap; maxiters::Int=5000) = ElaborateGetStartP(Domain(M), InDomain(M); maxiters=maxiters)
function ElaborateGetStartP(C::HyperCube, InDom::Union{Nothing,Function}; maxiters::Int=5000)
    naivetry = GetStartP(length(C))
    _IsInDomain(InDom, C, naivetry) ? naivetry : SobolStartP(C, InDom; maxiters=maxiters)
end
function SobolStartP(C::HyperCube, InDom::Union{Nothing,Function}; maxiters::Int=5000)
    X = rand(length(C));    i = 0
    S = SOBOL.skip(SOBOL.SobolSeq(clamp(C.L, Fill(-1e5,length(C)), Fill(1e5,length(C))), clamp(C.U, Fill(-1e5,length(C)), Fill(1e5,length(C)))), rand(1:10*maxiters); exact=true)
    while i < maxiters
        SOBOL.next!(S, X);    (_TestInDomain(InDom, X) && break);    i += 1
    end
    i == maxiters && throw("Unable to find point p satisfying InDomain(p) > 0 inside HyperCube within $maxiters iterations.")
    X
end

# Make sure the given ADmode is valid for optimization in InformationGeometry.minimize
EnsureNoSymbolic(V::Val{:Symbolic}) = Val(EnsureNoSymbolic(:Symbolic))
EnsureNoSymbolic(V::Val) = V
function EnsureNoSymbolic(S::Symbol)
    S !== :Symbolic && return S
    if isloaded(:FiniteDifferences)
        @warn "Choosing ADmodeOptim=:FiniteDifferences since :Symbolic not possible. If undesired, supply other legal choice from diff_backends() via kwarg ADmodeOptim to DataModel constructor."
        :FiniteDifferences
    else
        @warn "Choosing ADmodeOptim=:ForwardDiff since :Symbolic not possible. If this does not work, load package FiniteDifferences.jl or supply other legal choice from diff_backends() via kwarg ADmodeOptim to DataModel constructor."
        :ForwardDiff
    end
end
# false should use FiniteDifferences, will throw error if not loaded but ForwardDiff clearly not intended anyway
EnsureNoSymbolic(B::Bool) = B ? Val(:ForwardDiff) : (@warn "Choosing ADmodeOptim=:FiniteDifferences. If undesired, supply other legal choice from diff_backends() via kwarg ADmodeOptim to DataModel constructor.";   Val(:FiniteDifferences))

function FindMLEBig(DM::AbstractDataModel,start::AbstractVector{<:Number}=MLE(DM),LogPriorFn::Union{Function,Nothing}=LogPrior(DM); LogLikelihoodFn::Function=loglikelihood(DM), kwargs...)
    FindMLEBig(Data(DM), Predictor(DM), start, LogPriorFn; LogLikelihoodFn, kwargs...)
end
function FindMLEBig(DS::AbstractDataSet,model::ModelOrFunction,start::AbstractVector{<:Number}=GetStartP(DS,model),LogPriorFn::Union{Function,Nothing}=nothing; ADmode::Union{Val,Symbol}=Val(:ForwardDiff),
                LogLikelihoodFn::Function=GetLogLikelihoodFn(DS,model,LogPriorFn), CostFunction::Function=Negate(LogLikelihoodFn), ScoreFn::Function=GetGrad!(EnsureNoSymbolic(ADmode), CostFunction), tol::Real=1e-14, verbose::Bool=true, kwargs...)
    verbose && HasXerror(DS) && !isa(DS, UnknownVarianceDataSet) && @warn "Ignoring x-uncertainties in maximum likelihood estimation. Can be incorporated using the TotalLeastSquares() method."
    InformationGeometry.minimize(CostFunction, ScoreFn, BigFloat.(start), (model isa ModelMap ? Domain(model) : nothing); tol, verbose, kwargs...)
end
function FindMLEBig(DS::AbstractDataSet,model::ModelOrFunction,dmodel::ModelOrFunction,start::AbstractVector{<:Number}=GetStartP(DS,model),LogPriorFn::Union{Function,Nothing}=nothing; kwargs...)
    FindMLEBig(DS, model, start, LogPriorFn; kwargs...)
end


function FindMLE(DM::AbstractDataModel, start::AbstractVector{<:Number}=MLE(DM), LogPriorFn::Union{Function,Nothing}=LogPrior(DM); LogLikelihoodFn::Function=loglikelihood(DM), kwargs...)
    FindMLE(Data(DM), Predictor(DM), start, LogPriorFn; LogLikelihoodFn, kwargs...)
end
function FindMLE(DS::AbstractDataSet, model::ModelOrFunction, Start::AbstractVector{<:Number}=GetStartP(DS,model), LogPriorFn::Union{Function,Nothing}=nothing; Big::Bool=false, ADmode::Union{Val,Symbol}=Val(:ForwardDiff), 
                LogLikelihoodFn::Function=GetLogLikelihoodFn(DS,model,LogPriorFn), CostFunction::Function=Negate(LogLikelihoodFn), ScoreFn::Function=GetGrad!(EnsureNoSymbolic(ADmode), CostFunction),
                tol::Real=1e-14, meth=nothing, verbose::Bool=true, kwargs...)
    start = floatify(Start)
    (Big || tol < 2.3e-15 || suff(start) == BigFloat) && return FindMLEBig(DS, model, start, LogPriorFn; LogLikelihoodFn, CostFunction, ScoreFn, ADmode, tol, kwargs...)
    verbose && HasXerror(DS) && !isa(DS, UnknownVarianceDataSet) && @warn "Ignoring x-uncertainties in maximum likelihood estimation. Can be incorporated using the TotalLeastSquares() method."
    if isnothing(meth) && isnothing(LogPriorFn) && DS isa DataSet
        curve_fit(DS, model, start; tol=tol).param
    else
        if isnothing(meth)
            InformationGeometry.minimize(CostFunction, ScoreFn, start, (model isa ModelMap ? Domain(model) : nothing); tol, verbose, kwargs...)
        else
            InformationGeometry.minimize(CostFunction, ScoreFn, start, (model isa ModelMap ? Domain(model) : nothing); tol, meth, verbose, kwargs...)
        end
    end
end


function FindMLE(DS::AbstractDataSet, model::ModelOrFunction, dmodel::ModelOrFunction, Start::AbstractVector{<:Number}=GetStartP(DS,model), LogPriorFn::Union{Function,Nothing}=nothing; Big::Bool=false, ADmode::Union{Val,Symbol}=Val(:ForwardDiff), 
                LogLikelihoodFn::Function=GetLogLikelihoodFn(DS,model,LogPriorFn), CostFunction::Function=Negate(LogLikelihoodFn), ScoreFn::Function=GetGrad!(EnsureNoSymbolic(ADmode), CostFunction),
                tol::Real=1e-14, meth=nothing, verbose::Bool=true, kwargs...)
    start = floatify(Start)
    (Big || tol < 2.3e-15 || suff(start) == BigFloat) && return FindMLEBig(DS, model, start, LogPriorFn; LogLikelihoodFn, CostFunction, ScoreFn, ADmode, tol, kwargs...)
    verbose && HasXerror(DS) && !isa(DS, UnknownVarianceDataSet) && @warn "Ignoring x-uncertainties in maximum likelihood estimation. Can be incorporated using the TotalLeastSquares() method."
    if isnothing(meth) && isnothing(LogPriorFn) && DS isa DataSet
        curve_fit(DS, model, dmodel, start; tol=tol).param
    else
        if isnothing(meth)
            InformationGeometry.minimize(CostFunction, ScoreFn, start, (model isa ModelMap ? Domain(model) : nothing); tol=tol, verbose=verbose, kwargs...)
        else
            InformationGeometry.minimize(CostFunction, ScoreFn, start, (model isa ModelMap ? Domain(model) : nothing); tol=tol, meth=meth, verbose=verbose, kwargs...)
        end
    end
end

"""
    ConfidenceInterval1D(DM::AbstractDataModel, Confnum::Real=1.; tol::Real=1e-14) -> Tuple{Number,Number}
Returns the confidence interval associated with confidence level `Confnum` in the case of one-dimensional parameter spaces.
"""
function ConfidenceInterval1D(DM::AbstractDataModel, Confnum::Real=1.; tol::Real=1e-14, ADmode::Union{Val,Symbol}=Val(:ForwardDiff), dof::Int=DOF(DM), kwargs...)
    (tol < 2e-15 || Confnum > 8) && throw("ConfidenceInterval1D not programmed for BigFloat yet.")
    pdim(DM) != 1 && throw("ConfidenceInterval1D not defined for p != 1.")
    A = LogLikeMLE(DM) - (1/2)*InvChisqCDF(pdim(DM),ConfVol(Confnum))
    Func(p::Number) = loglikelihood(DM, muladd(p, BasisVector(1,pdim(DM)), MLE(DM))) - A
    B = try
        AltLineSearch(Func, sqrt(InvChisqCDF(dof, ConfVol(Confnum)) * inv(FisherMetric(DM, MLE(DM)))[1]); tol, kwargs...)
    catch;
        LineSearch(Func, sqrt(InvChisqCDF(dof, ConfVol(Confnum)) * inv(FisherMetric(DM, MLE(DM)))[1]); tol)
    end
    A = try
        AltLineSearch(Func, -B; tol, kwargs...)
    catch;
        SFunc = x->Func(-x)
        LineSearch(SFunc, -B; tol)
    end
    # Df = GetDeriv(ADmode, Func)
    # B = find_zero((Func,Df),0.1,Roots.Order1(); xatol=tol)
    # A = find_zero((Func,Df),-B,Roots.Order1(); xatol=tol)
    rts = (MLE(DM)[1]+A, MLE(DM)[1]+B)
    rts[1] ≥ rts[2] ? throw("ConfidenceInterval1D errored by producing $rts.") : return rts
end

function SpatialBoundaryFunction(M::ModelMap)
    function ModelMapBoundaries(u,p,t)
        S = !IsInDomain(M, u)
        S && @warn "Curve ran into boundaries specified by ModelMap at $u."
        return S
    end
end


function Rescaling(M::AbstractMatrix, μ::AbstractVector=Zeros(size(M,1)); Full::Bool=false, Dirs::Tuple{Int,Int}=(1,2), factor::Real=1.0)
    @assert size(M,1) == size(M,2) == length(μ) && Dirs[1] != Dirs[2]
    iS = if Full
        SMatrix{2,2}(cholesky(M[[Dirs[1],Dirs[2]],[Dirs[1],Dirs[2]]]).U ./ factor)
    else
        (sqrt.(SA[M[Dirs[1],Dirs[1]], M[Dirs[2],Dirs[2]]]) ./ factor) |> Diagonal
    end
    S = inv(iS)
    if 0 == μ[Dirs[1]] == μ[Dirs[2]]
        ix -> iS*ix,          x -> S*x
    else
        mle2d = SVector{2}(μ[[Dirs[1],Dirs[2]]])
        ix -> iS*(ix - mle2d),  x -> muladd(S, x, mle2d)
    end
end

"""
    GenerateBoundary(DM::DataModel, u0::AbstractVector{<:Number}; tol::Real=1e-9, meth=Tsit5(), mfd::Bool=true) -> ODESolution
Basic method for constructing a curve lying on the confidence region associated with the initial configuration `u0`.
"""
function GenerateBoundary(DM::AbstractDataModel, u0::AbstractVector{<:Number}; tol::Real=1e-9, Boundaries::Union{Function,Nothing}=nothing,
                            meth::AbstractODEAlgorithm=GetBoundaryMethod(tol,DM), mfd::Bool=false, promote::Bool=!OrdinaryDiffEqCore.isimplicit(meth), ADmode::Val=Val(:ForwardDiff), 
                            autodiff::AbstractADType=ADtypeConverter(ADmode), kwargs...)
    promote && !mfd && (u0 = PromoteStatic(u0, true))
    LogLikeOnBoundary = loglikelihood(DM, u0)
    # Problem with inplace OrthVF! for implicit methods
    IntCurveODE! = if !OrdinaryDiffEqCore.isimplicit(meth)
        IntCurveODE_expl!(du,u,p,t)  =  (OrthVF!(du, DM, u; ADmode=ADmode);  du .*= 0.1)
    else
        function IntCurveODE_impl!(u::T,p,t)::T where T <: AbstractArray
            0.1 .* OrthVF(DM, u; ADmode=ADmode)
        end
    end
    g!(resid,u,p,t)  =  (resid[1] = LogLikeOnBoundary - loglikelihood(DM,u))
    terminatecondition(u,t,integrator) = u[2] - u0[2]
    # TerminateCondition only on upwards crossing --> supply two different affect functions, leave second free I
    CB = ContinuousCallback(terminatecondition,terminate!,nothing)
    CB = !isnothing(Boundaries) ? CallbackSet(CB, DiscreteCallback(Boundaries,terminate!)) : CB
    CB = Predictor(DM) isa ModelMap ? CallbackSet(CB, DiscreteCallback(SpatialBoundaryFunction(Predictor(DM)),terminate!)) : CB
    prob = ODEProblem(IntCurveODE!,u0,(0.,1e5))
    if mfd
        solve(prob, meth; reltol=tol, abstol=tol, callback=CallbackSet(CB,ManifoldProjection(g!; autodiff)), kwargs...)
    else
        solve(prob, meth; reltol=tol, abstol=tol, callback=CB, kwargs...)
    end
end
"""
    GenerateBoundary2(DM::AbstractDataModel, u0::AbstractVector{<:Number}; tol::Real=1e-9, meth=GetBoundaryMethod(tol,DM), mfd::Bool=false, ADmode::Val=Val(:ForwardDiff), FullRescale::Bool=false, Embedded::Bool=true, kwargs...)
"""
function GenerateBoundary2(DM::AbstractDataModel, U0::AbstractVector{<:Number}; tol::Real=1e-5, Boundaries::Union{Function,Nothing}=nothing,
                meth::AbstractODEAlgorithm=GetBoundaryMethod(tol,DM), mfd::Bool=false, promote::Bool=!OrdinaryDiffEqCore.isimplicit(meth), ADmode::Val=Val(:ForwardDiff), FullRescale::Bool=false,
                autodiff::AbstractADType=ADtypeConverter(ADmode), Embedded::Bool=true, factor::Real=1.0, kwargs...)
    iEmb, Emb = Rescaling(FisherMetric(DM, MLE(DM))/InvChisqCDF(pdim(DM),ConfVol(GetConfnum(DM,U0))), MLE(DM); Full=FullRescale, factor=factor)
    u0 = (promote && !mfd) ? PromoteStatic(iEmb(U0), true) : DeStatic(iEmb(U0))
    EmbLikelihood = loglikelihood(DM) ∘ Emb
    LogLikeOnBoundary = EmbLikelihood(u0)
    CheatingOrth!(du::AbstractVector, dF::AbstractVector) = (mul!(du, SA[0 1; -1 0.], dF);  normalize!(du))
    Grad! = GetGrad!(ADmode, EmbLikelihood);    GradCache = copy(u0)
    IntCurveODE!(du, u, p, t) = (Grad!(GradCache, u);  CheatingOrth!(du, GradCache))
    g!(resid, u, p, t)  =  (resid[1] = LogLikeOnBoundary - Emblikelihood(u))
    terminatecondition(u, t, integrator) = u[2] - u0[2]
    # TerminateCondition only on upwards crossing --> supply two different affect functions, leave second free I
    CB = ContinuousCallback(terminatecondition,terminate!,nothing)
    CB = !isnothing(Boundaries) ? CallbackSet(CB, DiscreteCallback(EmbedCallbackFunc(Boundaries, Emb),terminate!)) : CB
    CB = Predictor(DM) isa ModelMap ? CallbackSet(CB, DiscreteCallback(EmbedCallbackFunc(SpatialBoundaryFunction(Predictor(DM)), Emb),terminate!)) : CB
    prob = ODEProblem(IntCurveODE!, u0, (0.,1e5))
    sol = if mfd
        solve(prob, meth; reltol=tol, abstol=tol, callback=CallbackSet(CB,ManifoldProjection(g!; autodiff)), kwargs...)
    else
        solve(prob, meth; reltol=tol, abstol=tol, callback=CB, kwargs...)
    end
    Embedded ? EmbeddedODESolution(sol, Emb) : sol
end

EmbedCallbackFunc(Boundaries::Function, PL::Plane) = EmbedCallbackFunc(Boundaries, PlaneCoordinates(PL))
EmbedCallbackFunc(Boundaries::Function, Emb::Function) = (u,p,t)->Boundaries(Emb(u),p,t)

EmbedLogPrior(N::Nothing, args...) = nothing
EmbedLogPrior(DM::AbstractDataModel, PL) = EmbedLogPrior(LogPrior(DM), PL)
EmbedLogPrior(F::Function, PL::Plane) = EmbedLogPrior(F, PlaneCoordinates(PL))
EmbedLogPrior(F::Function, Emb::Function) = F∘Emb

function GenerateBoundary(DM::AbstractDataModel, PL::Plane, u0::AbstractVector{<:Number}; tol::Real=1e-9, mfd::Bool=false,
                            Boundaries::Union{Function,Nothing}=nothing, meth::AbstractODEAlgorithm=GetBoundaryMethod(tol,DM), promote::Bool=!OrdinaryDiffEqCore.isimplicit(meth),
                            ADmode::Val=Val(:ForwardDiff), autodiff::AbstractADType=ADtypeConverter(ADmode), Embedded::Bool=false, kwargs...)
    @assert length(u0) == 2
    promote && !mfd && (u0 = PromoteStatic(u0, true))
    PlanarLogPrior = EmbedLogPrior(DM, PL)
    PlanarLogLike = loglikelihood(DM)∘PlaneCoordinates(PL)
    LogLikeOnBoundary = PlanarLogLike(u0)
    IntCurveODE!(du,u,p,t)  =  (copyto!(du, 0.1 .* OrthVF(DM, PL, u, PlanarLogPrior; ADmode=ADmode)))
    g!(resid,u,p,t)  =  resid[1] = LogLikeOnBoundary - PlanarLogLike(u)
    terminatecondition(u,t,integrator) = u[2] - u0[2]
    CB = ContinuousCallback(terminatecondition,terminate!,nothing)
    CB = !isnothing(Boundaries) ? CallbackSet(CB, DiscreteCallback(EmbedCallbackFunc(Boundaries, PL),terminate!)) : CB
    CB = DM isa DataModel && Predictor(DM) isa ModelMap ? CallbackSet(CB, DiscreteCallback(EmbedCallbackFunc(SpatialBoundaryFunction(Predictor(DM)),PL),terminate!)) : CB
    prob = ODEProblem(IntCurveODE!,u0,(0.,1e5))
    sol = if mfd
        solve(prob, meth; reltol=tol, abstol=tol, callback=CallbackSet(CB, ManifoldProjection(g!; autodiff)), kwargs...)
    else
        solve(prob, meth; reltol=tol, abstol=tol, callback=CB, kwargs...)
    end
    Embedded ? EmbeddedODESolution(sol, PL) : sol
end

"""
General function `F` with 2D domain whose Hessian should be negative-definite everywhere, i.e. negative cost function.
"""
GenerateBoundary(args...; kwargs...) = _GenerateBoundary(args...; kwargs...)

function GenerateBoundary2(F::Function, u0::AbstractVector{<:Number}; Embedded::Bool=true, ADmode::Union{Val,Symbol}=Val(:ForwardDiff), kwargs...)
    iEmb, Emb = Rescaling(-GetHess(ADmode, F)(u0)) # ); Full::Bool=false, Dirs::Tuple{Int,Int}=(1,2), factor::Real=1e-2)
    sol = _GenerateBoundary(F∘Emb, iEmb(u0); ADmode=ADmode, kwargs...)
    Embedded ? EmbeddedODESolution(sol, Emb) : sol
end

_GenerateBoundary(F::Function, u0::AbstractVector{<:Number}; ADmode::Union{Val,Symbol}=Val(:ForwardDiff), kwargs...) = _GenerateBoundary(F, GetGrad!(ADmode,F), u0; ADmode, kwargs...)
function _GenerateBoundary(F::Function, dF::Function, u0::AbstractVector{<:Number}; kwargs...)
    inplaceDF = try    Res = copy(u0);    dF(Res, u0);    Val(true)    catch;    Val(false)    end
    _GenerateBoundary(F, dF, u0, inplaceDF; kwargs...)
end
function _GenerateBoundary(F::Function, dF::Function, u0::AbstractVector{<:Number}, inplaceDF::Val{true}; tol::Real=1e-9, mfd::Bool=false, 
                            ADmode::Union{Val,Symbol}=Val(:ForwardDiff), autodiff::AbstractADType=ADtypeConverter(ADmode),
                            Boundaries::Union{Function,Nothing}=nothing, meth::AbstractODEAlgorithm=GetMethod(tol), promote::Bool=!OrdinaryDiffEqCore.isimplicit(meth), kwargs...)
    @assert length(u0) == 2
    promote && !mfd && (u0 = PromoteStatic(u0, true))
    CheatingOrth!(du::AbstractVector, x::AbstractVector) = (mul!(du, SA[0 1; -1 0.], x);  normalize!(du))
    GradCache = copy(u0)
    IntCurveODE!inplace(du,u,p,t) = (dF(GradCache, u);  CheatingOrth!(du, GradCache))
    solve(ODEProblem(IntCurveODE!inplace,u0,(0.,1e5)), meth; reltol=tol, abstol=tol,
                callback=_CallbackConstructor(F, u0, F(u0); Boundaries, mfd, autodiff), kwargs...)
end
function _GenerateBoundary(F::Function, dF::Function, u0::AbstractVector{<:Number}, inplaceDF::Val{false}; tol::Real=1e-9, mfd::Bool=false,
                            ADmode::Union{Val,Symbol}=Val(:ForwardDiff), autodiff::AbstractADType=ADtypeConverter(ADmode),
                            Boundaries::Union{Function,Nothing}=nothing, meth::AbstractODEAlgorithm=GetMethod(tol), promote::Bool=!OrdinaryDiffEqCore.isimplicit(meth), kwargs...)
    @assert length(u0) == 2
    promote && !mfd && (u0 = PromoteStatic(u0, true))
    CheatingOrth!(du::AbstractVector, x::AbstractVector) = (mul!(du, SA[0 1; -1 0.], x);  normalize!(du))
    IntCurveODE!(du,u,p,t) = CheatingOrth!(du, dF(u))
    solve(ODEProblem(IntCurveODE!,u0,(0.,1e5)), meth; reltol=tol, abstol=tol,
                    callback=_CallbackConstructor(F, u0, F(u0); Boundaries, mfd, autodiff), kwargs...)
end

"""
Constructs appropriate callback kwarg for 2D cost function consisting of
- the termination condition when the integral curve has closed,
- a domain check based on `Boundaries` kwarg,
- optional manifold projection.
"""
function _CallbackConstructor(F::Function, u0::AbstractVector{<:Number}, FuncOnBoundary::Real; Boundaries::Union{Function,Nothing}=nothing, mfd::Bool=false, ADmode::Val=Val(:ForwardDiff), autodiff::AbstractADType=ADtypeConverter(ADmode),
                                    TerminateBackwards::Bool=false)
    terminatecondition(u,t,integrator) = u[2] - u0[2]
    CB = !TerminateBackwards ? ContinuousCallback(terminatecondition, terminate!, nothing) : ContinuousCallback(terminatecondition, nothing, terminate!)
    !isnothing(Boundaries) && (CB = CallbackSet(CB, DiscreteCallback(Boundaries,terminate!)))
    g!(resid,u,p,t) = (resid[1] = FuncOnBoundary - F(u))
    mfd && (CB = CallbackSet(ManifoldProjection(g!; autodiff), CB)) # Eval projection first
    CB
end

"""
    ConfidenceRegion(DM::AbstractDataModel, Confnum::Real=1.; tol::Real=1e-9, meth=Tsit5(), mfd::Bool=false, ADmode::Val=Val(:ForwardDiff), parallel::Bool=false, Dirs::Tuple{Int,Int,Int}=(1,2,3), N::Int=30)
Computes confidence region of level `Confnum`. For `pdim(DM) > 2`, the confidence region is intersected by a family of `Plane`s in the directions specified by the keyword `Dirs`.
The `Plane`s and their embedded 2D confidence boundaries are returned as the respective first and second arguments in this case.
"""
function ConfidenceRegion(DM::AbstractDataModel, Confnum::Real=1.; tol::Real=1e-9, meth::AbstractODEAlgorithm=GetBoundaryMethod(tol,DM), mfd::Bool=false, verbose::Bool=true, Padding::Real=1/10,
                            Boundaries::Union{Function,Nothing}=nothing, ADmode::Val=Val(:ForwardDiff), parallel::Bool=true, Dirs::Tuple{Int,Int,Int}=(1,2,3), N::Int=30, dof::Int=DOF(DM), kwargs...)
    if pdim(DM) == 1
        ConfidenceInterval1D(DM, Confnum; tol)
    elseif pdim(DM) == 2
        GenerateBoundary(DM, FindConfBoundary(DM, Confnum; tol, dof); tol, Boundaries, meth, mfd, ADmode, kwargs...)
    else
        verbose && Dirs == (1,2,3) && @info "ConfidenceRegion() computes solutions in the θ[1]-θ[2] plane which are separated in the θ[3] direction. For more explicit control, call MincedBoundaries() and set options manually."
        Cube = LinearCuboid(DM, Confnum; Padding, dof=dof)
        Planes = IntersectCube(DM, Cube, Confnum; Dirs, N, tol, dof)
        Planes, MincedBoundaries(DM, Planes, Confnum; tol, dof, Boundaries, ADmode, meth, mfd, parallel, kwargs...)
    end
end


IsStructurallyIdentifiableAlong(DM::AbstractDataModel, sol::AbstractODESolution; kwargs...)::Bool = length(StructurallyIdentifiableAlong(DM, sol; kwargs...)) == 0

function StructurallyIdentifiableAlong(DM::AbstractDataModel, sol::AbstractODESolution; kwargs...)
    Roots.find_zeros(t->GeometricDensity(DM, sol(t); kwargs...), sol.t[1], sol.t[end])
end
function StructurallyIdentifiableAlong(DM::AbstractDataModel, sols::AbstractVector{<:AbstractODESolution}; parallel::Bool=false, kwargs...)
    (parallel ? pmap : map)(x->StructurallyIdentifiableAlong(DM, x; kwargs...), sols)
end

"""
    StructurallyIdentifiable(DM::AbstractDataModel, mle::AbstractVector{<:Number}=MLE(DM); showall::Bool=false, noise::Real=1e-5, threshold::Real=1e-10, N::Int=3)
Checks if jacobian of model wrt parameters has singular values below threshold and provides associated singular directions.
"""
function StructurallyIdentifiable(DM::AbstractDataModel, mle::AbstractVector{<:Number}=MLE(DM); showall::Bool=false, noise::Real=1e-5, threshold::Real=1e-10, N::Int=3)
    J = reduce(vcat, [EmbeddingMatrix(DM, mle .+ noise .* (rand(length(mle)) .- 0.5)) for i in 1:N])
    _, S, Vt = svd(J)
    nonids = count(x->x<threshold, S)
    if nonids == 0
        println("$(name(DM) === Symbol() ? "DataModel" : string(name(DM))) is locally structurally identifiable at MLE!\nSmallest singular value $(round(S[end]; sigdigits=5)) > $threshold.")
    else
        println("$(name(DM) === Symbol() ? "DataModel" : string(name(DM))) NOT locally structurally identifiable at MLE!\n$nonids singular values < $threshold.")
    end
    if showall
        for ind in length(S):-1:1
            if S[ind] < threshold
                println("Singular direction associated with value $(round(S[ind]; sigdigits=6)):")
                println(DataFrame([[x] for x in Vt[:, ind]], string.(pnames(DM))))
            end
        end
    end;    S, Vt
end

IsStructurallyIdentifiable(DM::AbstractDataModel, args...; kwargs...) = all(x->x>0, StructurallyIdentifiable(DM, args...; kwargs...)[1])


"""
    ConfidenceRegions(DM::DataModel, Range::AbstractVector)
Computes the boundaries of confidence regions for two-dimensional parameter spaces given a vector or range of confidence levels.
A convenient interface which extends this to higher dimensions is currently still under development.

For example,
```julia
ConfidenceRegions(DM, 1:3; tol=1e-9)
```
computes the ``1\\sigma``, ``2\\sigma`` and ``3\\sigma`` confidence regions associated with a given `DataModel` using a solver tolerance of ``10^{-9}``.

Keyword arguments:
* `IsConfVol = true` can be used to specify the desired confidence level directly in terms of a probability ``p \\in [0,1]`` instead of in units of standard deviations ``\\sigma``,
* `tol` can be used to quantify the tolerance with which the ODE which defines the confidence boundary is solved (default `tol = 1e-9`),
* `meth` can be used to specify the solver algorithm (default `meth = Tsit5()`),
* `ADmode=Val(false)` computes the Score by separately evaluating the `model` as well as the Jacobian `dmodel` provided in `DM`. Other choices of `ADmode` directly compute the Score by differentiating the formula the log-likelihood, i.e. only one evaluation on a dual variable is performed.
* `parallel = true` parallelizes the computations of the separate confidence regions provided each process has access to the necessary objects,
* `dof` can be used to manually specify the degrees of freedom.
"""
function ConfidenceRegions(DM::AbstractDataModel, Confnums::AbstractVector{<:Real}=1:1; IsConfVol::Bool=false, verbose::Bool=true,
                        tol::Real=1e-9, meth::AbstractODEAlgorithm=GetBoundaryMethod(tol,DM), mfd::Bool=false, ADmode::Val=Val(:ForwardDiff),
                        Boundaries::Union{Function,Nothing}=nothing, tests::Bool=!(Predictor(DM) isa ModelMap), parallel::Bool=false, dof::Int=DOF(DM), kwargs...)
    verbose && NotPosDef(FisherMetric(DM,MLE(DM))) && throw("It appears as though the given model is not structurally identifiable.")
    Range = IsConfVol ? InvConfVol.(Confnums) : Confnums
    if pdim(DM) == 1
        return (parallel ? pmap : map)(x->ConfidenceRegion(DM, x; tol=tol, dof=dof), Range)
    elseif pdim(DM) == 2
        Prog = Progress(length(Range); enabled=verbose, desc="Computing boundaries... "*(parallel ? "(parallel, $(nworkers()) workers) " : ""), dt=1, showspeed=true)
        sols = (parallel ? progress_pmap : progress_map)(x->ConfidenceRegion(DM, x; tol=tol, dof=dof, Boundaries=Boundaries, meth=meth, mfd=mfd, ADmode=ADmode, kwargs...), Range; progress=Prog)
        if tests
            NotTerminated = map(x->!(x.retcode === SciMLBase.ReturnCode.Terminated), sols)
            verbose && sum(NotTerminated) != 0 && @warn "Solutions $((1:length(sols))[NotTerminated]) did not exit properly."
            roots = StructurallyIdentifiableAlong(DM, sols; parallel=parallel)
            Unidentifiables = map(x->(length(x) != 0), roots)
            for i in eachindex(roots)
                length(roots[i]) != 0 && verbose && @warn "Solution $i hits chart boundary at t = $(roots[i]) and should therefore be considered invalid."
            end
        end
        return sols
    else
        throw("This functionality is still under construction. Use ConfidenceRegion() instead.")
    end
end


"""
    InterruptedConfidenceRegion(DM::AbstractDataModel, Confnum::Real; Boundaries::Union{Function,Nothing}=nothing, tol::Real=1e-9,
                                redo::Bool=true, meth::AbstractODEAlgorithm=Tsit5(), mfd::Bool=true, ADmode::Val=Val(:ForwardDiff), kwargs...) -> ODESolution
Integrates along the level lines of the log-likelihood in the counter-clockwise direction until the model becomes either
1. structurally non-identifiable via `det(g) < tol`
2. the given `Boundaries(u,t,int)` method evaluates to `true`.
It then integrates from where this obstruction was met in the clockwise direction until said obstruction is hit again, resulting in a half-open confidence region.
"""
function InterruptedConfidenceRegion(DM::AbstractDataModel, Confnum::Real; Boundaries::Union{Function,Nothing}=nothing, tol::Real=1e-9,
                                redo::Bool=true, meth::AbstractODEAlgorithm=GetBoundaryMethod(tol,DM), mfd::Bool=false, ADmode::Val=Val(:ForwardDiff), kwargs...)
    GenerateInterruptedBoundary(DM, FindConfBoundary(DM, Confnum; tol=tol); Boundaries=Boundaries, tol=tol, meth=meth, mfd=mfd, ADmode=ADmode, kwargs...)
end

"""
    GenerateInterruptedBoundary(DM::AbstractDataModel, u0::AbstractVector{<:Number}; Boundaries::Union{Function,Nothing}=nothing, tol::Real=1e-9,
                                redo::Bool=true, meth::AbstractODEAlgorithm=Tsit5(), mfd::Bool=true, ADmode::Val=Val(:ForwardDiff), kwargs...) -> ODESolution
Integrates along the level lines of the log-likelihood in the counter-clockwise direction until the model becomes either
1. structurally non-identifiable via `det(g) < tol`
2. the given `Boundaries(u,t,int)` method evaluates to `true`.
It then integrates from where this obstruction was met in the clockwise direction until said obstruction is hit again, resulting in a half-open confidence region.
"""
function GenerateInterruptedBoundary(DM::AbstractDataModel, u0::AbstractVector{<:Number}; Boundaries::Union{Function,Nothing}=nothing, tol::Real=1e-9,
                                redo::Bool=true, meth::AbstractODEAlgorithm=GetBoundaryMethod(tol,DM), mfd::Bool=false, promote::Bool=!OrdinaryDiffEqCore.isimplicit(meth), 
                                ADmode::Val=Val(:ForwardDiff), autodiff::AbstractADType=ADtypeConverter(ADmode), kwargs...)
    promote && !mfd && (u0 = PromoteStatic(u0, true))
    LogLikeOnBoundary = loglikelihood(DM,u0)
    IntCurveODE!(du,u,p,t)  =  (du .= 0.1 .* OrthVF(DM, u; ADmode=ADmode))
    BackwardsIntCurveODE!(du,u,p,t)  =  (du .= -0.1 .* OrthVF(DM, u; ADmode=ADmode))
    g!(resid,u,p,t)  =  resid[1] = LogLikeOnBoundary - loglikelihood(DM, u)

    terminatecondition(u,t,integrator) = u[2] - u0[2]
    Singularity(u,t,integrator) = det(FisherMetric(DM, u)) - tol

    ForwardsTerminate = ContinuousCallback(terminatecondition,terminate!,nothing)

    CB = ContinuousCallback(Singularity,terminate!)
    CB = !isnothing(Boundaries) ? CallbackSet(CB, DiscreteCallback(Boundaries,terminate!)) : CB
    CB = Predictor(DM) isa ModelMap ? CallbackSet(CB, DiscreteCallback(SpatialBoundaryFunction(Predictor(DM)),terminate!)) : CB
    CB = mfd ? CallbackSet(CB, ManifoldProjection(g!; autodiff)) : CB

    Forwardprob = ODEProblem(IntCurveODE!, u0, (0., 1e5))
    sol1 = solve(Forwardprob, meth; reltol=tol, abstol=tol, callback=CallbackSet(ForwardsTerminate, CB), kwargs...)

    if norm(sol1.u[end] - sol1.u[1]) < 10tol
        # closed loop, no apparent interruption or direct termination in worst case
        return sol1
    else
        Backprob = redo ? ODEProblem(BackwardsIntCurveODE!, sol1.u[end], (0., 4e5)) : ODEProblem(BackwardsIntCurveODE!, u0, (0., 4e5))
        sol2 = solve(Backprob, meth; reltol=tol, abstol=tol, callback=CB, kwargs...)
        return redo ? sol2 : [sol1, sol2]
    end
end


# Assume that sums from Fisher metric defined with first derivatives of loglikelihood pull out

# FisherMetric(DM::AbstractDataModel; kwargs...) = θ::AbstractVector{<:Number} -> FisherMetric(DM)(θ; kwargs...)

"""
    FisherMetric(DM::DataModel, θ::AbstractVector{<:Number})
Computes the Fisher metric ``g`` given a `DataModel` and a parameter configuration ``\\theta`` under the assumption that the likelihood ``L(\\mathrm{data} \\, | \\, \\theta)`` is a multivariate normal distribution.
```math
g_{ab}(\\theta) \\coloneqq -\\int_{\\mathcal{D}} \\mathrm{d}^m y_{\\mathrm{data}} \\, L(y_{\\mathrm{data}} \\,|\\, \\theta) \\, \\frac{\\partial^2 \\, \\mathrm{ln}(L)}{\\partial \\theta^a \\, \\partial \\theta^b} = -\\mathbb{E} \\bigg( \\frac{\\partial^2 \\, \\mathrm{ln}(L)}{\\partial \\theta^a \\, \\partial \\theta^b} \\bigg)
```
"""
FisherMetric(DM::AbstractDataModel, θ::AbstractVector{<:Number}; kwargs...) = FisherMetric(DM)(θ; kwargs...)
FisherMetric(DM::AbstractDataModel, θ::AbstractVector{<:Number}, LogPriorFn::Union{Nothing,Function}; kwargs...) = (@warn "Will deprecate this FisherMetric method soon!";   FisherMetric(Data(DM), Predictor(DM), dPredictor(DM), θ, LogPriorFn; kwargs...))

FisherMetric(DS::AbstractDataSet, model::ModelOrFunction, dmodel::ModelOrFunction, θ::AbstractVector{<:Number}, LogPriorFn::Nothing; kwargs...) = _FisherMetric(DS, model, dmodel, θ; kwargs...)
# ADD MINUS SIGN FOR LogPrior TERM TO ACCOUNT FOR NEGATIVE SIGN IN DEFINTION OF FISHER METRIC
FisherMetric(DS::AbstractDataSet, model::ModelOrFunction, dmodel::ModelOrFunction, θ::AbstractVector{<:Number}, LogPriorFn::Function; kwargs...) = _FisherMetric(DS, model, dmodel, θ; kwargs...) - EvalLogPriorHess(LogPriorFn, θ)


# Specialize this for other DataSet types
_FisherMetric(DS::AbstractDataSet, model::ModelOrFunction, dmodel::ModelOrFunction, θ::AbstractVector{<:Number}; kwargs...) = Pullback(DS, dmodel, yInvCov(DS), θ; kwargs...)

# Does not work for CDS
# Data covariance matrix for a single data point, computed from the mean
function AverageSingleYsigmaMatrix(DM::AbstractDataModel, mle::AbstractVector=MLE(DM))
    @assert Data(DM) isa AbstractFixedUncertaintyDataSet
    Ysig = yInvCov(DM, mle) |> pinv
    mean([view(Ysig, inds, inds) for inds in Iterators.partition(1:size(Ysig,1), ydim(DM))])
end


"""
    VariancePropagation(DM::AbstractDataModel, mle::AbstractVector, Confnum::Real; dof::Int=DOF(DM), kwargs...)
    VariancePropagation(DM::AbstractDataModel, mle::AbstractVector, C::AbstractMatrix=quantile(Chisq(length(mle)), ConfVol(1)) * Symmetric(pinv(FisherMetric(DM, mle))); kwargs...)
Computes the forward propagation of the parameter covariance to the residuals. The output constitutes the cholesky decomposition (i.e. square root) of the variance associated with the residuals.
Matrix `C` corresponds to a parameter covariance matrix `Σ` which has been properly scaled according to a desired confidence level.
"""
function VariancePropagation(DM::AbstractDataModel, mle::AbstractVector, C::AbstractMatrix; Confnum::Real=1, dof::Int=DOF(DM), Validation::Bool=false, InterpolateDataUncertainty::Bool=false, ADmode::Val=Val(:ForwardDiff), verbose::Bool=true)
    JacobianWindup(J::AbstractMatrix, ydim::Int) = size(J,1) == ydim ? [J] : map(yinds->view(J, yinds, :), Iterators.partition(1:size(J,1), ydim))
    ConfScaling = InvChisqCDF(dof, ConfVol(Confnum))
    normalparams, yerrorparams = SkipXs(DM)((SplitErrorParams(DM)(mle))[1]), (SplitErrorParams(DM)(mle))[end]
    # As function of independent variable x
    YsigmaGenerator = if Data(DM) isa AbstractFixedUncertaintyDataSet
        # If Validation Band, add data uncertainty for single point to 
        Ysig = if Validation
            @assert !InterpolateDataUncertainty "InterpolateDataUncertainty not implemented yet!"
            verbose && !allequal(ysigma(DM)) && @warn "Need to compute average data uncertainty to plot validation profile between observations!"
            AverageSingleYsigmaMatrix(DM, mle)
        else    # No data uncertainty contribution
            Diagonal(Zeros(ydim(DM)))
        end
        Ysig *= ConfScaling
        ydim(DM) == 1 && (Ysig = Ysig[1])
        x -> Ysig
    else
        !Validation ? (x -> 0.0) : (x -> (S=inv(yinverrormodel(Data(DM))(x, Predictor(DM)(x,normalparams), yerrorparams));   ConfScaling * (S' * S)))
    end
    
    # Add data uncertainty here if Validation
    function CholeskyU(M::AbstractMatrix, x)
        # `M` denotes the pure prediction variance
        try     cholesky(Symmetric(YsigmaGenerator(x) .+ M)).U
        catch err
            !isa(err, PosDefException) && rethrow(err)
            UpperTriangular(Diagonal(Zeros(size(M,1))))
        end
    end
    Sqrt(M::Real, x) = sqrt(YsigmaGenerator(x) + M)
    SplitterJac = Data(DM) isa AbstractFixedUncertaintyDataSet ? (x->1.0) : GetJac(ADmode, x->SkipXs(DM)((SplitErrorParams(DM)(x))[1]))
    # Make sure that missings expected in data are not filtered out, e.g. by CompositeDataSet method
    SplitterJ = SplitterJac(mle)
    embeddingMatrix(DM::AbstractDataModel, normalparams::AbstractVector{<:Number}, X::AbstractVector) = EmbeddingMatrix(Val(true), dPredictor(DM), normalparams, X) * SplitterJ

    VarCholesky1(x::Number) = (J = dPredictor(DM)(x, normalparams)*SplitterJ;   CholeskyU(J * C * transpose(J),x))
    VarCholesky1(X::AbstractVector{<:Number}) = (Jf = embeddingMatrix(DM, normalparams, X);   map((J,x)->CholeskyU(J * C * transpose(J),x), JacobianWindup(Jf, ydim(DM)), X))
    VarSqrt1(x::Number) = (J = dPredictor(DM)(x, normalparams)*SplitterJ;   R = Sqrt((J * C * transpose(J))[1], x))
    VarSqrt1(X::AbstractVector{<:Number}) = (Jf = embeddingMatrix(DM, normalparams, X);   map((J,x)->Sqrt((J * C * transpose(J))[1], x), JacobianWindup(Jf, ydim(DM)), X))

    VarCholeskyN(x::AbstractVector{<:Number}) = (J = dPredictor(DM)(x, normalparams)*SplitterJ;   CholeskyU(J * C * transpose(J), x))
    VarCholeskyN(X::AbstractVector{AbstractVector{<:Number}}) = (Jf = embeddingMatrix(DM, normalparams, X);   map((J,x)->CholeskyU(J * C * transpose(J), x), JacobianWindup(Jf, ydim(DM)), X))
    VarSqrtN(x::AbstractVector{<:Number}) = (J = dPredictor(DM)(x, normalparams)*SplitterJ;   R = Sqrt((J * C * transpose(J))[1], x))
    VarSqrtN(X::AbstractVector{AbstractVector{<:Number}}) = (Jf = embeddingMatrix(DM, normalparams, X);   map((J,x)->Sqrt((J * C * transpose(J))[1], x), JacobianWindup(Jf, ydim(DM)), X))
    xdim(DM) == 1 ? (ydim(DM) > 1 ? VarCholesky1 : VarSqrt1) : (ydim(DM) > 1 ? VarCholeskyN : VarSqrtN)
end
function VariancePropagation(DM::AbstractDataModel, mle::AbstractVector=MLE(DM), confnum::Real=1; Confnum::Real=confnum, dof::Int=DOF(DM), F::AbstractMatrix=FisherMetric(DM, mle), verbose::Bool=true, kwargs...)
    verbose && NotPosDef(F) && @warn "Variance Propagation unreliable since det(FisherMetric)=0."
    VariancePropagation(DM, mle, InvChisqCDF(dof, ConfVol(Confnum)) * Symmetric(pinv(F)); Confnum, dof, kwargs...)
end


"""
    ValidationPropagation(DM::AbstractDataModel, mle::AbstractVector, Confnum::Real; dof::Int=DOF(DM), kwargs...)
    ValidationPropagation(DM::AbstractDataModel, mle::AbstractVector, C::AbstractMatrix=quantile(Chisq(length(mle)), ConfVol(1)) * Symmetric(pinv(FisherMetric(DM, mle))); kwargs...)
Computes the linearized validation band / prediction band, which quantifies the range where new successive measurements are expected to land with given probability.
Matrix `C` corresponds to a parameter covariance matrix `Σ` which has been properly scaled according to a desired confidence level.
"""
ValidationPropagation(DM::AbstractDataModel, args...; Validation::Bool=true, kwargs...) = VariancePropagation(DM, args...; kwargs..., Validation=true)



SafeSqrt(x::Real) = x < 0 ? -Inf : sqrt(x)
"""
    GeometricDensity(DM::AbstractDataModel, θ::AbstractVector) -> Real
Computes the square root of the determinant of the Fisher metric ``\\sqrt{\\mathrm{det}\\big(g(\\theta)\\big)}`` at the point ``\\theta``.
"""
GeometricDensity(DM::AbstractDataModel, θ::AbstractVector{<:Number}; kwargs...) = FisherMetric(DM, θ; kwargs...) |> det |> SafeSqrt
GeometricDensity(Metric::Function, θ::AbstractVector{<:Number}; kwargs...) = SafeSqrt(det(Metric(θ; kwargs...)))
GeometricDensity(DM::AbstractDataModel; kwargs...) = θ::AbstractVector{<:Number} -> GeometricDensity(DM, θ; kwargs...)

"""
    ConfidenceRegionVolume(DM::AbstractDataModel, Confnum::Real; N::Int=Int(1e5), WE::Bool=true, Approx::Bool=false, kwargs...) -> Real
Computes coordinate-invariant volume of confidence region associated with level `Confnum` via Monte Carlo by integrating the geometric density factor.
For likelihoods which are particularly expensive to evaluate, `Approx=true` can improve the performance by approximating the confidence region via polygons.
"""
function ConfidenceRegionVolume(DM::AbstractDataModel, Confnum::Real; N::Int=Int(1e5), WE::Bool=true, Approx::Bool=false, kwargs...)
    if Approx
        return ConfidenceRegionVolume(DM, ConfidenceRegion(DM,Confnum;tol=1e-6); N=N, WE=WE, Approx=Approx, kwargs...)
    else
        # Might not need to compute ConfidenceRegion if pdim > 2
        Domain = if pdim(DM) == 2
            # For pdim == 2, Bounding box from confidence region more performant than ProfileLikelihood
            ConstructCube(ConfidenceRegion(DM,Confnum;tol=1e-6); Padding=1e-2)
        else
            ProfileBox(DM, InterpolatedProfiles(ProfileLikelihood(DM, Confnum+2; plot=false)), Confnum; Padding=1e-2)
        end
        return IntegrateOverConfidenceRegion(DM, Domain, Confnum, z->GeometricDensity(DM,z; kwargs...); N=N, WE=WE, kwargs...)
    end
end
function ConfidenceRegionVolume(DM::AbstractDataModel, sol::AbstractODESolution; N::Int=Int(1e5), WE::Bool=true, Approx::Bool=false, kwargs...)
    @assert pdim(DM) == length(sol.u[1]) == 2
    Domain = ConstructCube(sol; Padding=1e-2)
    if Approx
        IntegrateOverApproxConfidenceRegion(DM, Domain, sol, z->GeometricDensity(DM,z;kwargs...); N=N, WE=WE)
    else
        IntegrateOverConfidenceRegion(DM, Domain, GetConfnum(DM, sol), z->GeometricDensity(DM,z;kwargs...); N=N, WE=WE, kwargs...)
    end
end
function ConfidenceRegionVolume(DM::AbstractDataModel, Tup::Tuple{<:AbstractVector{<:Plane},<:AbstractVector{<:AbstractODESolution}}; N::Int=Int(1e5), WE::Bool=true, Approx::Bool=false, kwargs...)
    ConfidenceRegionVolume(DM, Tup[1], Tup[2]; N=N, WE=WE, Approx=Approx, kwargs...)
end
function ConfidenceRegionVolume(DM::AbstractDataModel, Planes::AbstractVector{<:Plane}, sols::AbstractVector{<:AbstractODESolution}, Confnum::Real=GetConfnum(DM,Planes,sols); N::Int=Int(1e5), WE::Bool=true, Approx::Bool=false, kwargs...)
    # Domain = ProfileBox(DM, InterpolatedProfiles(ProfileLikelihood(DM, Confnum+2; plot=false)), Confnum; Padding=1e-2)
    Domain = ConstructCube(Planes, sols; Padding=1e-2)
    if Approx
        IntegrateOverApproxConfidenceRegion(DM, Domain, Planes, sols, GeometricDensity(DM); N=N, WE=WE)
    else
        IntegrateOverConfidenceRegion(DM, Domain, Confnum, GeometricDensity(DM); N=N, WE=WE, kwargs...)
    end
end



"""
    CoordinateVolume(DM::AbstractDataModel, Confnum::Real; N::Int=Int(1e5), WE::Bool=true, Approx::Bool=false, kwargs...) -> Real
Computes coordinate-dependent apparent volume of confidence region associated with level `Confnum` via Monte Carlo integration.
For likelihoods which are particularly expensive to evaluate, `Approx=true` can improve the performance by approximating the confidence region via polygons.
"""
function CoordinateVolume(DM::AbstractDataModel, Confnum::Real; N::Int=Int(1e5), WE::Bool=true, Approx::Bool=false, kwargs...)
    if Approx
        return CoordinateVolume(DM, ConfidenceRegion(DM,Confnum;tol=1e-6); N=N, WE=WE, Approx=Approx, kwargs...)
    else
        # Might not need to compute ConfidenceRegion if pdim > 2
        Domain = if pdim(DM) == 2
            # For pdim == 2, Bounding box from confidence region more performant than ProfileLikelihood
            ConstructCube(ConfidenceRegion(DM,Confnum;tol=1e-6); Padding=1e-2)
        else
            ProfileBox(DM, InterpolatedProfiles(ProfileLikelihood(DM, Confnum+2; plot=false)), Confnum; Padding=1e-2)
        end
        return IntegrateOverConfidenceRegion(DM, Domain, Confnum, ((z::T) where {T<:Number}) -> one(T); N=N, WE=WE, kwargs...)
    end
end
function CoordinateVolume(DM::AbstractDataModel, sol::AbstractODESolution; N::Int=Int(1e5), WE::Bool=true, Approx::Bool=false, kwargs...)
    @assert pdim(DM) == length(sol.u[1]) == 2
    Domain = ConstructCube(sol; Padding=1e-2)
    if Approx
        IntegrateOverApproxConfidenceRegion(DM, Domain, sol, ((z::T) where {T<:Number}) -> one(T); N=N, WE=WE)
    else
        IntegrateOverConfidenceRegion(DM, Domain, GetConfnum(DM, sol), ((z::T) where {T<:Number}) -> one(T); N=N, WE=WE, kwargs...)
    end
end
function CoordinateVolume(DM::AbstractDataModel, Tup::Tuple{<:AbstractVector{<:Plane},<:AbstractVector{<:AbstractODESolution}}; N::Int=Int(1e5), WE::Bool=true, Approx::Bool=false, kwargs...)
    CoordinateVolume(DM, Tup[1], Tup[2]; N=N, WE=WE, Approx=Approx, kwargs...)
end
function CoordinateVolume(DM::AbstractDataModel, Planes::AbstractVector{<:Plane}, sols::AbstractVector{<:AbstractODESolution}, Confnum::Real=GetConfnum(DM,Planes,sols); N::Int=Int(1e5), WE::Bool=true, Approx::Bool=false, kwargs...)
    Domain = ProfileBox(DM, InterpolatedProfiles(ProfileLikelihood(DM, Confnum+2; plot=false)), Confnum; Padding=1e-2)
    if Approx
        IntegrateOverApproxConfidenceRegion(DM, Domain, Planes, sols, ((z::T) where {T<:Number}) -> one(T); N=N, WE=WE)
    else
        IntegrateOverConfidenceRegion(DM, Domain, Confnum, ((z::T) where {T<:Number}) -> one(T); N=N, WE=WE, kwargs...)
    end
end

SphereVolumeFactor(n::Int) = π^(n/2) / gamma(n/2 + 1)
ExpectedInvariantVolume(DM::AbstractDataModel, Confnum::Real) = SphereVolumeFactor(pdim(DM)) * GeodesicRadius(DM, Confnum)^pdim(DM)

"""
    EllipsoidVolume(Σ::AbstractMatrix)
Computes volume of ellipsoid defined by symmetric positive-definite matrix whose eigenvalues constitute the squares of the half-axes of the ellipsoid.
"""
EllipsoidVolume(Σ::AbstractMatrix) = (n=size(Σ,1);   sqrt(exp(sum(log.(eigvals(Σ))))) * 2 * π^(n/2) / (n * gamma(n/2)))

GeodesicRadius(DM::AbstractDataModel, Confnum::Real) = GeodesicRadius(Confnum, pdim(DM))
GeodesicRadius(Confnum::Real, dim::Int) = sqrt(InvChisqCDF(dim, ConfVol(Confnum)))

"""
    CoordinateDistortion(DM::AbstractDataModel, Confnum::Real=1) -> Real
For CoordinateDistortions ≪ 1, the model predictions are extremely sensitive with respect to the parameters.
For CoordinateDistortion ⪎ 1, the model is comparatively insensitive towards the parameters.

This quantity is computed from the ratio of the coordinate-dependent apparent volume of a confidence region compared with the coordinate-invariant volume, which is obtained from integrating over the appropriate volume form / geometric density factor.
The unit of this quantity is ``[L^n]`` where ``L`` is the unit of length of each of the components.
"""
function CoordinateDistortion(DM::AbstractDataModel, Confnum::Real=1; Approx::Bool=false, WE::Bool=true, N::Int=Int(1e5), kwargs...)
    CoordinateVolume(DM, Confnum; N=N, Approx=Approx, WE=WE, kwargs...) / ExpectedInvariantVolume(DM, Confnum)
end

# Sensitivity independent of quality of measured datapoints (number and uncertainties), roughly independent of Confnum
function Sensitivity(DM::AbstractDataModel, Confnum::Real=1; Approx::Bool=false, WE::Bool=true, N::Int=Int(1e5), kwargs...)
    1 / CoordinateDistortion(DM, Confnum; Approx=Approx, WE=WE, N=N, kwargs...)
end




# M ⟵ D
Pullback(DM::AbstractDataModel, F::Function, θ::AbstractVector{<:Number}; kwargs...) = F(EmbeddingMap(DM, θ; kwargs...))

"""
    Pullback(DM::AbstractDataModel, ω::AbstractVector{<:Number}, θ::AbstractVector) -> Vector
Pull-back of a covector to the parameter manifold ``T^*\\mathcal{M} \\longleftarrow T^*\\mathcal{D}``.
"""
Pullback(DM::AbstractDataModel, ω::AbstractVector{<:Number}, θ::AbstractVector{<:Number}; kwargs...) = transpose(EmbeddingMatrix(DM, θ; kwargs...)) * ω


"""
    Pullback(DM::DataModel, G::AbstractArray{<:Number,2}, θ::AbstractVector) -> Matrix
Pull-back of a (0,2)-tensor `G` to the parameter manifold.
"""
Pullback(DM::AbstractDataModel, G::AbstractMatrix, θ::AbstractVector{<:Number}; kwargs...) = Pullback(Data(DM), dPredictor(DM), G, θ; kwargs...)
function Pullback(DS::AbstractDataSet, dmodel::ModelOrFunction, G::AbstractMatrix, θ::AbstractVector{<:Number}; kwargs...)
    J = EmbeddingMatrix(DS, dmodel, θ; kwargs...)
    transpose(J) * G * J
end

# M ⟶ D
"""
    Pushforward(DM::DataModel, X::AbstractVector, θ::AbstractVector) -> Vector
Calculates the push-forward of a vector `X` from the parameter manifold to the data space ``T\\mathcal{M} \\longrightarrow T\\mathcal{D}``.
"""
Pushforward(DM::AbstractDataModel, X::AbstractVector, θ::AbstractVector{<:Number}; kwargs...) = EmbeddingMatrix(DM, θ; kwargs...) * X


"""
    AIC(DM::DataModel, θ::AbstractVector) -> Real
Calculates the Akaike Information Criterion given a parameter configuration ``\\theta`` defined by ``\\mathrm{AIC} = 2 \\, \\mathrm{length}(\\theta) -2 \\, \\ell(\\mathrm{data} \\, | \\, \\theta)``.
Lower values for the AIC indicate that the associated model function is more likely to be correct. For linearly parametrized models and small sample sizes, it is advisable to instead use the AICc which is more accurate.
"""
AIC(DM::AbstractDataModel, θ::AbstractVector{<:Number}=MLE(DM); kwargs...) = 2length(θ) - 2loglikelihood(DM, θ; kwargs...)

"""
    AICc(DM::DataModel, θ::AbstractVector) -> Real
Computes Akaike Information Criterion with an added correction term that prevents the AIC from selecting models with too many parameters (i.e. overfitting) in the case of small sample sizes.
``\\mathrm{AICc} = \\mathrm{AIC} + \\frac{2\\mathrm{length}(\\theta)^2 + 2 \\mathrm{length}(\\theta)}{N - \\mathrm{length}(\\theta) - 1}`` where ``N`` is the number of data points.
Whereas AIC constitutes a first order estimate of the information loss, the AICc constitutes a second order estimate. However, this particular correction term assumes that the model is **linearly parametrized**.
"""
function AICc(DM::AbstractDataModel, θ::AbstractVector{<:Number}=MLE(DM); kwargs...)
    if (DataspaceDim(DM) - length(θ) - 1) != 0
        AIC(DM, θ; kwargs...) + (2length(θ)^2 + 2length(θ)) / (DataspaceDim(DM) - length(θ) - 1)
    else
        @warn "DataSet too small to apply AIC correction. Using AIC without correction instead."
        AIC(DM, θ; kwargs...)
    end
end

"""
    BIC(DM::DataModel, θ::AbstractVector) -> Real
Calculates the Bayesian Information Criterion given a parameter configuration ``\\theta`` defined by ``\\mathrm{BIC} = \\mathrm{ln}(N) \\cdot \\mathrm{length}(\\theta) -2 \\, \\ell(\\mathrm{data} \\, | \\, \\theta)`` where ``N`` is the number of data points.
"""
BIC(DM::AbstractDataModel, θ::AbstractVector{<:Number}=MLE(DM); kwargs...) = length(θ)*log(DataspaceDim(DM)) - 2loglikelihood(DM, θ; kwargs...)



"""
    GetSmoothnessLogPrior(DM::AbstractDataModel, λ=5e-2; N::Int=300, Ran::AbstractVector=range(XCube(DS)[1]...; length=N)[2:end-1])
Penalize mean curvature of model prediction to obtain smooth solution in maximum likelihood estimation.
"""
GetSmoothnessLogPrior(DM::AbstractDataModel, args...; kwargs...) = GetSmoothnessLogPrior(Data(DM), Predictor(DM), args...; kwargs...)
function GetSmoothnessLogPrior(DS::AbstractDataSet, Model::ModelOrFunction, λ=5e-2; N::Int=100, Ran::AbstractVector=range(XCube(DS)[1]...; length=N)[2:end-1])
    @assert xdim(DS) == 1 && all(x->x>0, λ)
    @assert λ isa Real || length(λ) == ydim(DS) # Allow different lambda in each y-component
    sqrt_λ = sqrt.(λ)
    function SmoothnessLogPrior(θ::AbstractVector{<:Number})
        PredHess(X::Real) = ForwardDiff.derivative(z->ForwardDiff.derivative(x->Model(x,θ),z), X)
        SquareHess(X) = (H=sqrt_λ.*PredHess(X);   dot(H,H))
        -sum(SquareHess, Ran)/(ydim(DS)*length(Ran))
    end
end


"""
    ModelComparison(DM1::AbstractDataModel, DM2::AbstractDataModel) -> Tuple{Int,Real}
Compares the AICc values of both models at best fit and estimates probability that one model is more likely than the other.
First entry of tuple returns which model is more likely to be correct (1 or 2) whereas the second entry returns the ratio of probabilities `p_better/p_worse`.
"""
function ModelComparison(DM1::AbstractDataModel, DM2::AbstractDataModel, Crit::Function=AICc; verbose::Bool=true, sigdigits::Int=5, kwargs...)
    Data(DM1) != Data(DM2) && throw("Not comparing model against same data!")
    Mod1 = Crit(DM1, MLE(DM1); kwargs...);      Mod2 = Crit(DM2, MLE(DM2); kwargs...)
    better = (Mod1 < Mod2 ? 1 : 2) # lower is better!
    res = round(exp(0.5*abs(Mod2-Mod1)); sigdigits=sigdigits)
    verbose && @info "Model $(better) is estimated to be $(res) times as likely to be correct from difference in AICc values."
    (better, res)
end

"""
    CompareCols(A::AbstractMatrix, B::AbstractMatrix) -> BitVector
Check whether two Jacobian matrices have any columns in common which are not completely filled with zeros.
"""
CompareCols(A::AbstractMatrix, B::AbstractMatrix) = (@assert size(A) == size(B);    BitArray(@views A[:,i] == B[:,i] && any(x->x!=0., A[:,i]) for i in axes(A,2)))

"""
    IsLinearParameter(DM::DataModel, MLE::AbstractVector) -> BitVector
Checks with respect to which parameters the model function `model(x,θ)` is linear and returns vector of booleans where `true` indicates linearity.
This test is performed by comparing the Jacobians of the model for two random configurations ``\\theta_1, \\theta_2 \\in \\mathcal{M}`` column by column.
"""
IsLinearParameter(DM::AbstractDataModel, mle::AbstractVector{<:Number}=MLE(DM), args...; factor::Real=0.1, kwargs...) = CompareCols(EmbeddingMatrix(DM, mle.+factor.*rand(pdim(DM)), args...; kwargs...), EmbeddingMatrix(DM, mle.+factor.*rand(pdim(DM)), args...; kwargs...))

"""
    IsLinear(DM::DataModel, MLE::AbstractVector) -> Bool
Checks whether the `model(x,θ)` function is linear with respect to all of its parameters ``\\theta \\in \\mathcal{M}``.
A componentwise check can be attained via the method `IsLinearParameter(DM)`.
"""
IsLinear(DM::AbstractDataModel, args...; kwargs...) = all(IsLinearParameter(DM, args...; kwargs...))

"""
    LeastInformativeDirection(DM::DataModel,θ::AbstractVector{<:Number}=MLE(DM)) -> Vector{Float64}
Returns a vector which points in the direction in which the likelihood decreases most slowly.
"""
function LeastInformativeDirection(DM::AbstractDataModel, θ::AbstractVector{<:Number}=MLE(DM); kwargs...)
    M = eigen(FisherMetric(DM,θ; kwargs...));  i = findmin(M.values)[2]
    M.vectors[:,i] ./ sqrt(M.values[i])
end


"""
    FindConfBoundaryOnPlane(DM::AbstractDataModel,PL::Plane,Confnum::Real=1.; tol::Real=1e-8) -> Union{AbstractVector{<:Number},Bool}
Computes point inside the plane `PL` which lies on the boundary of a confidence region of level `Confnum`.
If such a point cannot be found (i.e. does not seem to exist), the method returns `false`.
"""
function FindConfBoundaryOnPlane(DM::AbstractDataModel, PL::Plane, Confnum::Real=1.; tol::Real=1e-8, kwargs...)
    FindConfBoundaryOnPlane(DM, PL, MLEinPlane(DM, PL; tol=tol), Confnum; tol=tol, kwargs...)
end
function FindConfBoundaryOnPlane(DM::AbstractDataModel, PL::Plane, mle::AbstractVector{<:Number}, Confnum::Real=1.; dof::Int=DOF(DM), tol::Real=1e-8, LogLikelihoodFn::Function=loglikelihood(DM), meth=Roots.AlefeldPotraShi(), maxval::Real=1e-6, maxiter::Int=10000)
    CF = ConfVol(Confnum)
    # model = Predictor(DM);    PlanarLogPrior = EmbedLogPrior(DM, PL)
    # planarmod(x,p::AbstractVector{<:Number}) = model(x, PlaneCoordinates(PL,p))
    # Test(x::Number) = ChisqCDF(dof, abs(2(LogLikeMLE(DM) - loglikelihood(Data(DM), planarmod, mle + SA[x,0.], PlanarLogPrior)))) - CF < 0.
    EmbeddedLikelihood = LogLikelihoodFn∘PlaneCoordinates(PL)
    Test(x::Number) = ChisqCDF(dof, abs(2(LogLikeMLE(DM) - EmbeddedLikelihood(mle + SA[x,0.])))) - CF < 0.
    !Test(0.) && return false
    SA[LineSearch(Test, 0.; tol=tol, maxiter=maxiter), 0.] .+ mle
    # TestCont(x::Number) = ChisqCDF(dof, abs(2(LogLikeMLE(DM) - EmbeddedLikelihood(mle + SA[x,0.])))) - CF
    # TestCont(0.) ≥ 0 && return false
    # SA[AltLineSearch(TestCont, (0.0, maxval), meth; tol=tol), 0.] + mle
end


DoPruning(DM, Planes::AbstractVector{<:Plane}, Confnum; kwargs...) = Prune(DM, AntiPrune(DM, Planes, Confnum; kwargs...), Confnum; kwargs...)

function Prune(DM::AbstractDataModel, Pls::AbstractVector{<:Plane}, Confnum::Real=1.; tol::Real=1e-8, dof::Int=DOF(DM))
    CF = ConfVol(Confnum)
    Planes = copy(Pls)
    while length(Planes) > 2
        !WilksTest(DM, PlaneCoordinates(Planes[1],MLEinPlane(DM,Planes[1];tol=tol)), CF; dof) ? popfirst!(Planes) : break
    end
    while length(Planes) > 2
        !WilksTest(DM, PlaneCoordinates(Planes[end],MLEinPlane(DM,Planes[end];tol=tol)), CF; dof) ? pop!(Planes) : break
    end
    length(Planes) == 2 && throw("For some reason, all Planes were pruned away?!")
    return Planes
end

function AntiPrune(DM::AbstractDataModel, Pls::AbstractVector{<:Plane}, Confnum::Real=1.; tol::Real=1e-8, dof::Int=DOF(DM))
    Planes = copy(Pls)
    length(Planes) < 2 && throw("Not enough Planes to infer translation direction.")
    CF = ConfVol(Confnum)
    while true
        TestPlane = ShiftTo(Planes[2], Planes[1])
        WilksTest(DM, PlaneCoordinates(TestPlane,MLEinPlane(DM,TestPlane;tol=tol)), CF; dof) ? pushfirst!(Planes,TestPlane) : break
    end
    while true
        TestPlane = ShiftTo(Planes[end-1], Planes[end])
        WilksTest(DM, PlaneCoordinates(TestPlane,MLEinPlane(DM,TestPlane;tol=tol)), CF; dof) ? push!(Planes,TestPlane) : break
    end;    Planes
end


"""
    LinearCuboid(DM::AbstractDataModel, Confnum::Real=1.; dof::Int=DOF(DM), Padding::Number=1/30, N::Int=200) -> HyperCube
Returns `HyperCube` which bounds the linearized confidence region of level `Confnum` for a `DataModel`.
"""
function LinearCuboid(DM::AbstractDataModel, Confnum::Real=1.; dof::Int=DOF(DM), Padding::Number=1/30, N::Int=200)
    # LinearCuboid(Symmetric(InvChisqCDF(pdim(DM),ConfVol(Confnum))*inv(FisherMetric(DM, MLE(DM)))), MLE(DM); Padding=Padding, N=N)
    BoundingBox(Symmetric(InvChisqCDF(dof,ConfVol(Confnum))*inv(FisherMetric(DM, MLE(DM)))), MLE(DM); Padding=Padding)
end

# Formula from https://tavianator.com/2014/ellipsoid_bounding_boxes.html
function BoundingBox(Σ::AbstractMatrix, μ::AbstractVector=Zeros(size(Σ,1)); Padding::Number=1/30)
    @assert size(Σ,1) == size(Σ,2) == length(μ)
    E = eigen(Σ)
    @assert all(x->x>0, E.values)
    offsets = [dot(E.values, E.vectors[i,:].^2) for i in eachindex(E.values)] .|> sqrt
    HyperCube(μ-offsets, μ+offsets; Padding=Padding)
end


"""
    IntersectCube(DM::AbstractDataModel,Cube::HyperCube,Confnum::Real=1.; Dirs::Tuple{Int,Int,Int}=(1,2,3), N::Int=31) -> Vector{Plane}
Returns a set of parallel 2D planes which intersect `Cube`. The planes span the directions corresponding to the basis vectors corresponding to the first two components of `Dirs`.
They are separated in the direction of the basis vector associated with the third component of `Dirs`.
The keyword `N` can be used to approximately control the number of planes which are returned.
This depends on whether more (or fewer) planes than `N` are necessary to cover the whole confidence region of level `Confnum`.
"""
function IntersectCube(DM::AbstractDataModel, Cube::HyperCube, Confnum::Real=1.; N::Int=31, Dirs::Tuple{Int,Int,Int}=(1,2,3), tol::Real=1e-8, dof::Int=DOF(DM))
    (!allunique(Dirs) || !all(x->(1 ≤ x ≤ length(Cube)), Dirs)) && throw("Invalid choice of Dirs: $Dirs.")
    widths = CubeWidths(Cube)
    # PL = Plane(Center(Cube), BasisVector(Dirs[1], length(Cube)), BasisVector(Dirs[2],length(Cube)))
    PL = Plane(Center(Cube), 0.5widths[Dirs[1]]*BasisVector(Dirs[1], length(Cube)), 0.5widths[Dirs[2]]*BasisVector(Dirs[2],length(Cube)))
    IntersectRegion(DM, PL, widths[Dirs[3]] * BasisVector(Dirs[3],length(Cube)), Confnum; N, tol, dof)
end

"""
    IntersectRegion(DM::AbstractDataModel,PL::Plane,v::AbstractVector{<:Number},Confnum::Real=1.; N::Int=31) -> Vector{Plane}
Translates family of `N` planes which are translated approximately from `-v` to `+v` and intersect the confidence region of level `Confnum`.
If necessary, planes are removed or more planes added such that the maximal family of planes is found.
"""
function IntersectRegion(DM::AbstractDataModel, PL::Plane, v::AbstractVector{<:Number}, Confnum::Real=1.; N::Int=31, tol::Real=1e-8, dof::Int=DOF(DM))
    IsOnPlane(Plane(Zeros(length(v)), PL.Vx, PL.Vy),v) && throw("Translation vector v = $v lies in given Plane $PL.")
    # Planes = ParallelPlanes(PL, v, range(-0.5,0.5; length=N))
    # AntiPrune(DM, Prune(DM,Planes,Confnum;tol=tol), Confnum; tol=tol)
    DoPruning(DM, ParallelPlanes(PL, v, range(-0.5,0.5;length=N)), Confnum; tol, dof)
end


function GenerateEmbeddedBoundary(DM::AbstractDataModel, PL::Plane, Confnum::Real=1.; tol::Real=1e-8, dof::Int=DOF(DM), mfd::Bool=false, kwargs...)
    GenerateBoundary(DM, PL, FindConfBoundaryOnPlane(DM, PL, Confnum; tol, dof); tol, mfd, kwargs...)
end

"""
    MincedBoundaries(DM::AbstractDataModel, Planes::AbstractVector{<:Plane}, Confnum::Real=1.; tol::Real=1e-9, ADmode::Val=Val(:ForwardDiff), meth=Tsit5(), mfd::Bool=false)
Intersects the confidence boundary of level `Confnum` with `Planes` and computes `ODESolution`s which parametrize this intersection.
"""
function MincedBoundaries(DM::AbstractDataModel, Planes::AbstractVector{<:Plane}, Confnum::Real=1.; verbose::Bool=true, parallel::Bool=true, kwargs...)
    Prog = Progress(length(Planes); enabled=verbose, desc="Computing planar solutions... "*(parallel ? "(parallel, $(nworkers()) workers) " : ""), dt=1, showspeed=true)
    (parallel ? progress_pmap : progress_map)(X->GenerateEmbeddedBoundary(DM, X, Confnum; kwargs...), Planes; progress=Prog)
end



"""
    ContourDiagram(DM::AbstractDataModel, Confnum::Real, paridxs::AbstractVector{<:Int}=1:pdim(DM))
Plots 2D slices through confidence region for all parameter pairs to show non-linearity of parameter interdependence.
"""
function ContourDiagram(DM::AbstractDataModel, Confnum::Real=2, paridxs::AbstractVector{<:Int}=1:pdim(DM); idxs::AbstractVector{<:AbstractVector{<:Int}}=OrderedIndCombs2D(paridxs), 
                                tol::Real=1e-5, plot::Bool=isloaded(:Plots), pnames::AbstractVector{<:AbstractString}=pnames(DM), size=PlotSizer(length(idxs)), SkipTests::Bool=false, kwargs...)
    @assert pdim(DM) > 2 && Confnum > 0
    @assert allunique(idxs) && ConsistentElDims(idxs) == 2 && all(1 .≤ getindex.(idxs,1) .≤ pdim(DM)) && all(1 .≤ getindex.(idxs,2) .≤ pdim(DM))

    !SkipTests && !IsStructurallyIdentifiable(DM) && @warn "Model does not appear to be structurally identifiable. Continuing anyway."

    Cube = LinearCuboid(DM, Confnum);   widths = CubeWidths(Cube)
    Planes = [Plane(MLE(DM), 0.5widths[paridxs[j]]*BasisVector(paridxs[j], length(Cube)), 0.5widths[paridxs[i]]*BasisVector(paridxs[i], length(Cube))) for (i,j) in idxs]

    sols = MincedBoundaries(DM, Planes, Confnum; tol, kwargs...)
    esols = EmbeddedODESolution[EmbeddedODESolution(sols[k], ViewElements(inds)∘PlaneCoordinates(Planes[k])) for (k,inds) in enumerate(idxs)]
    eCubes = map(sol->ConstructCube(sol; Padding=0.075), esols)
    if plot
        Plts = [(p = RecipesBase.plot([MLE(DM)[inds]]; label="MLE$(inds)", xlabel=pnames[inds[1]], ylabel=pnames[inds[2]], seriestype=:scatter);
                RecipesBase.plot!(p, esols[k]; idxs=(1,2), label="$(Confnum)σ Slice", xlims=eCubes[k][1], ylims=eCubes[k][2])) for (k,inds) in enumerate(idxs)]
        RecipesBase.plot(Plts...; layout=length(Plts), size) |> display
    end;    esols
end

"""
    ContourDiagramLowerTriangular(DM::AbstractDataModel, Confnum::Real, paridxs::AbstractVector{<:Int}=1:pdim(DM))
Plots 2D slices through confidence region for all parameter pairs to show non-linearity of parameter interdependence.
"""
function ContourDiagramLowerTriangular(DM::AbstractDataModel, Confnum::Real=2, paridxs::AbstractVector{<:Int}=1:pdim(DM); 
                tol::Real=1e-5, plot::Bool=isloaded(:Plots), pnames::AbstractVector{<:AbstractString}=pnames(DM), size=PlotSizer(length(idxs)), SkipTests::Bool=false, 
                IndMat::AbstractMatrix{<:AbstractVector{<:Int}}=[[x,y] for y in paridxs, x in paridxs],
                idxs::AbstractVector{<:AbstractVector{<:Int}}=vec(IndMat), comparison::Function=Base.isless, kwargs...)
    @assert pdim(DM) > 2 && Confnum > 0
    @assert allunique(idxs) && ConsistentElDims(idxs) == 2 && all(1 .≤ getindex.(idxs,1) .≤ pdim(DM)) && all(1 .≤ getindex.(idxs,2) .≤ pdim(DM))

    !SkipTests && !IsStructurallyIdentifiable(DM) && @warn "Model does not appear to be structurally identifiable. Continuing anyway."

    Cube = LinearCuboid(DM, Confnum);   widths = CubeWidths(Cube)
    finalidxs = [[i,j] for (i,j) in idxs if comparison(i,j)]
    Planes = [Plane(MLE(DM), 0.5widths[i]*BasisVector(i, length(Cube)), 0.5widths[j]*BasisVector(j, length(Cube))) for (i,j) in finalidxs]
    sols = MincedBoundaries(DM, Planes, Confnum; tol, kwargs...)
    esols = EmbeddedODESolution[EmbeddedODESolution(sols[k], ViewElements(inds)∘PlaneCoordinates(Planes[k])) for (k,inds) in enumerate(finalidxs)]
    eCubes = map(sol->ConstructCube(sol; Padding=0.075), esols)
    if plot
        k = 0;  Plts = []
        for i in 2:length(paridxs), j in 1:(length(paridxs)-1)
            inds = IndMat[i,j]
            if comparison(j,i)
                k += 1
                plt = RecipesBase.plot([MLE(DM)[inds]]; label="MLE$(inds)", xlabel=pnames[inds[1]], ylabel=pnames[inds[2]], seriestype=:scatter)
                RecipesBase.plot!(plt, esols[k]; idxs=(1,2), label="$(Confnum)σ Slice", xlims=eCubes[k][1], ylims=eCubes[k][2])
                push!(Plts, plt)
            else
                push!(Plts, RecipesBase.plot(; framestyle = :none))
            end
        end;    RecipesBase.plot(Plts...; layout=(length(paridxs)-1, length(paridxs)-1), size) |> display
    end;    esols
end


function Thinner(S::AbstractVector{<:AbstractVector{<:Number}}; threshold::Real=0.2)
    M = S |> diff .|> norm;    b = threshold * median(M);    Res = [S[1]]
    for i in 2:length(M)
        M[i-1] + M[i] < b ? continue : push!(Res, S[i])
    end;    Res
end

CastShadow(DM::AbstractDataModel, Tup::Tuple{<:AbstractVector{<:Plane},<:AbstractVector{<:AbstractODESolution}}, args...; kwargs...) = CastShadow(DM, Tup[1], Tup[2], args...; kwargs...)
CastShadow(DM::DataModel, Planes::AbstractVector{<:Plane}, sols::AbstractVector{<:AbstractODESolution}, dirs::Tuple{<:Int,<:Int}; kwargs...) = CastShadow(DM, Planes, sols, dirs[1], dirs[2]; kwargs...)
function CastShadow(DM::DataModel, Planes::AbstractVector{<:Plane}, sols::AbstractVector{<:AbstractODESolution}, dir1::Int, dir2::Int; threshold::Real=0.2)
    @assert length(Planes) == length(sols)
    @assert dir1 != dir2

    (1 ≤ dir1 ≤ pdim(DM) && 1 ≤ dir2 ≤ pdim(DM)) || @warn "Projection directions > pdim(DM)."
    pdim(DM) == length(Planes[1]) || @warn "Pdim = $(pdim(DM)) but ambient dim of planes is $(length(Planes[1]))."

    Project(p::AbstractVector{<:Number}, dir1::Int, dir2::Int) = SA[p[dir1], p[dir2]]

    poly = map(x->Project(PlaneCoordinates(Planes[1],x), dir1, dir2), sols[1])
    for i in 2:length(Planes)
        poly = UnionPolygons(poly, map(x->Project(PlaneCoordinates(Planes[i],x), dir1, dir2), sols[i]))
    end;    Thinner(poly; threshold=threshold)
end


# function ToGeos(pointlist::AbstractVector{<:AbstractVector{<:Number}})
#     @assert 2 == InformationGeometry.ConsistentElDims(pointlist)
#     text = "POLYGON(("
#     for point in pointlist
#         text *= string(point[1]) *" "* string(point[2]) *","
#     end
#     text *= string(pointlist[1][1]) *" "* string(pointlist[1][2]) * "))"
#     LibGEOS.readgeom(text)
# end
# UnionPolygons(p1::AbstractVector{<:AbstractVector{<:Number}}, p2::AbstractVector{<:AbstractVector{<:Number}}) = LibGEOS.GeoInterface.coordinates(UnionPolygons(ToGeos(p1), ToGeos(p2)))[1]
# UnionPolygons(p1::LibGEOS.Polygon, p2::LibGEOS.Polygon) = LibGEOS.union(p1,p2)


ToGeos(args...) = throw("Need to load LibGEOS.jl first.")
UnionPolygons(args...) = throw("Need to load LibGEOS.jl first.")



ToAmbient(DM::AbstractDataModel, pointlist::AbstractVector{<:AbstractVector{<:Number}}, dirs::Tuple{<:Int, <:Int}) = ToAmbient(DM, pointlist, dirs[1], dirs[2])
function ToAmbient(DM::AbstractDataModel, pointlist::AbstractVector{<:AbstractVector{<:Number}}, dir1::Int, dir2::Int)
    @assert 2 == InformationGeometry.ConsistentElDims(pointlist)
    @assert 1 ≤ dir1 ≤ pdim(DM) && 1 ≤ dir2 ≤ pdim(DM) && dir1 != dir2
    mle = copy(MLE(DM));      mle[[dir1,dir2]] .= 0.0
    PL = Plane(mle, BasisVector(dir1, pdim(DM)), BasisVector(dir2, pdim(DM)))
    map(x->PlaneCoordinates(PL,x), pointlist)
end

function ShadowTheatre(DM::AbstractDataModel, Confnum::Real=1, dirs::Tuple{<:Int,<:Int}=(1,2); tol::Real=1e-7, N::Int=50)
    @assert (1 ≤ dirs[1] ≤ pdim(DM)) && (1 ≤ dirs[2] ≤ pdim(DM)) && dirs[1] != dirs[2] && pdim(DM) > 2
    keep = trues(pdim(DM));     keep[dirs[1]] = false;      keep[dirs[2]] = false
    translationdirs = collect(1:pdim(DM))[keep]

    Planes, sols = ConfidenceRegion(DM, Confnum; tol=tol, N=N, Dirs=(dirs[1],dirs[2],translationdirs[1]))
    list = CastShadow(DM, Planes, sols, dirs)
    if length(translationdirs) > 1
        for i in (@view translationdirs[2:end])
            Planes, sols = ConfidenceRegion(DM, Confnum; tol=tol, N=N, Dirs=(dirs[1],dirs[2],i))
            list = UnionPolygons(list, CastShadow(DM, Planes, sols, dirs))
        end
    end
    ToAmbient(DM, list, dirs)
end



"""
    LeftOfLine(q₁::AbstractVector, q₂::AbstractVector, p::AbstractVector) -> Bool
Checks if point `p` is left of the line from `q₁` to `q₂` via `det([q₁-p  q₂-p]) > 0` for 2D points.
"""
function LeftOfLine(q₁::AbstractVector, q₂::AbstractVector, p::AbstractVector)::Bool
    # @assert length(q₁) == length(q₂) == length(p) == 2
    (q₁[1] - p[1]) * (q₂[2] - p[2]) - (q₂[1] - p[1]) * (q₁[2] - p[2]) > 0
end

# Copied from Luxor.jl
"""
    isinside(p, pointlist) -> Bool
Is a point `p` inside a polygon defined by a counterclockwise list of points.
"""
function isinside(p::AbstractVector{<:Number}, pointlist::AbstractVector{<:AbstractVector})
    @assert ConsistentElDims(pointlist) == length(p) == 2
    c = false
    @inbounds for counter in eachindex(pointlist)
        q1 = pointlist[counter]
        # if reached last point, set "next point" to first point
        if counter == length(pointlist)
            q2 = pointlist[1]
        else
            q2 = pointlist[counter + 1]
        end
        if (q1[2] < p[2]) != (q2[2] < p[2]) # crossing
            if q1[1] >= p[1]
                if q2[1] > p[1]
                    c = !c
                elseif (LeftOfLine(q1, q2, p) == (q2[2] > q1[2]))
                    c = !c
                end
            elseif q2[1] > p[1]
                if (LeftOfLine(q1, q2, p) == (q2[2] > q1[2]))
                    c = !c
                end
            end
        end
    end
    c
end


"""
    ApproxInRegion(sol::AbstractODESolution, p::AbstractVector{<:Number}) -> Bool
Blazingly fast approximative test whether a point lies within the polygon defined by the base points of a 2D ODESolution.
Especially well-suited for hypothesis testing once a confidence boundary has been explicitly computed.
"""
ApproxInRegion(sol::AbstractODESolution, p::AbstractVector{<:Number}) = isinside(p, sol.u)

function ApproxInRegion(Planes::AbstractVector{<:Plane}, sols::AbstractVector{<:AbstractODESolution}, p::AbstractVector{<:Number})
    !(ConsistentElDims(Planes) == length(p) == 3) && throw("ApproxInRegion: Cannot determine for length(p) > 3.")      # Unclear how to do this for higher dimensions.
    @assert length(Planes) == length(sols) && all(x->length(x.u[1])==2, sols)
    # Assuming all planes parallel
    ProjectionOp = InformationGeometry.ProjectionOperator(Planes[1])
    minind = findmin([DistanceToPlane(Planes[i], p, ProjectionOp) for i in eachindex(Planes)])[2]
    ApproxInRegion(sols[minind], DecomposeWRTPlane(Planes[minind], ProjectOntoPlane(Planes[minind], p)))
end




function GetConfnum(DM::AbstractDataModel, θ::AbstractVector{<:Number}; dof::Int=DOF(DM), kwargs...)
    if length(θ) == pdim(DM)
        InvConfVol(ChisqCDF(dof, 2(LogLikeMLE(DM) - loglikelihood(DM, θ; kwargs...))))
    elseif length(θ) == xpdim(DM)
        @warn "Using LiftedEmbedding for determining the Confnum here."
        LiftedLogLike = FullLiftedLogLikelihood(DM)
        InvConfVol(ChisqCDF(dof, 2(LiftedLogLike(TotalLeastSquaresV(DM)) - LiftedLogLike(θ; kwargs...))))
    else
        throw("Length of θ $(length(θ)) neither corresponds to pdim=$(pdim(DM)) nor xpdim=$(xpdim(DM)).")
    end
end
GetConfnum(DM::AbstractDataModel, sol::AbstractODESolution; kwargs...) = GetConfnum(DM, sol.u[end]; kwargs...)
GetConfnum(DM::AbstractDataModel, PL::Plane, sol::AbstractODESolution; kwargs...) = GetConfnum(DM, PlaneCoordinates(PL, sol.u[end]); kwargs...)

function GetConfnum(DM::AbstractDataModel, Planes::AbstractVector{<:Plane}, sols::AbstractVector{<:AbstractODESolution}; kwargs...)
    Nums = [GetConfnum(DM, Planes[i], sols[i]; kwargs...) for i in eachindex(Planes)]
    mean = sum(Nums) / length(Nums)
    !all(x->abs(x-mean) < 1e-5, Nums) && @warn "High Variance in given Confnums, continuing anyway with arithmetic mean."
    return mean
end


"""
    CrossValidation(DM::AbstractDataModel)
    CrossValidation(DM::AbstractDataModel, keeps::AbstractVector{<:AbstractVector})
Leave-one-out cross validation by default. Alternatively, combinations which are to be kept can be specified via `keeps`.
"""
CrossValidation(DM::AbstractDataModel) = CrossValidation(DM, [(x->x!=i).(1:Npoints(DM)) for i in 1:Npoints(DM)])
function CrossValidation(DM::AbstractDataModel, keeps::AbstractVector{<:AbstractVector})
    DMs = [DataModel(Data(DM)[keeper], Predictor(DM), dPredictor(DM), MLE(DM), LogPrior(DM)) for keeper in keeps]
    Res = eltype(MLE(DM))[]
    for (i, keeper) in enumerate(keeps)
        newpoints = map(!, keeper)
        res = InnerProduct(Diagonal(WoundInvCov(DM)[newpoints]), WoundY(DM)[newpoints] - EmbeddingMap(DMs[i], MLE(DMs[i]), WoundX(DM)[newpoints])) |> sqrt
        push!(Res, res)
    end
    Res, Measurements.measurement(mean(Res), std(Res)/sqrt(length(Res)))
end




abstract type AbstractBoundarySlice end

struct ConfidenceBoundarySlice <: AbstractBoundarySlice
    sols::AbstractVector{<:AbstractODESolution}
    Dirs::Tuple{Int,Int,Int}
    Confnum::Real
    mle::AbstractVector{<:Number}
    pnames::AbstractVector{<:AbstractString}
    Full::Bool
end
Sols(CB::ConfidenceBoundarySlice) = CB.sols
Dirs(CB::ConfidenceBoundarySlice) = CB.Dirs

Confnum(CB::AbstractBoundarySlice) = CB.Confnum
MLE(CB::AbstractBoundarySlice) = CB.mle
pnames(CB::AbstractBoundarySlice) = CB.pnames

function ConfidenceBoundarySlice(DM::AbstractDataModel, sols::AbstractVector{<:AbstractODESolution}, Dirs::Tuple{Int,Int,Int})
    Full = length(sols[1].u[1]) == xpdim(DM)
    @assert Full || length(sols[1].u[1]) == pdim(DM)
    Confnum = GetConfnum(DM, sols[1].u[1]; dof=DOF(DM))
    mle = (Full ? TotalLeastSquaresV(DM) : MLE(DM))[Dirs...]
    Names = (Full ? _FullNames(DM) : pnames(DM))[Dirs...]
    ConfidenceBoundarySlice(DM, sols, Dirs, Confnum, mle, Names, Full)
end
function ConfidenceBoundarySlice(DM::AbstractDataModel, Planes::AbstractVector{<:Plane}, sols::AbstractVector{<:AbstractODESolution}, Dirs::Tuple{Int,Int,Int})
    ConfidenceBoundarySlice([EmbeddedODESolution(Planes[i], sols[i]) for i in eachindex(sols)], Dirs)
end
function ConfidenceBoundarySlice(DM::AbstractDataModel, sol::AbstractODESolution, Dirs::Tuple{Int,Int,Int}=(1,2,0))
    @assert length(sol.u[1]) == 2
    Dirs[3] != 0 && (Dirs = (Dirs[1], Dirs[2],0))
    Full = xpdim(DM) == 2
    Confnum = GetConfnum(DM, sol.u[1]; dof=DOF(DM))
    mle = (Full ? TotalLeastSquaresV(DM) : MLE(DM))[[Dirs[1],Dirs[2]]]
    Names = (Full ? _FullNames(DM) : pnames(DM))[[Dirs[1],Dirs[2]]]
    ConfidenceBoundarySlice([sol], Dirs, Confnum, mle, Names, Full)
end
function ConfidenceBoundarySlice(DM::AbstractDataModel, CI::Tuple{<:Number,<:Number})
    Confnum = GetConfnum(DM, [CI[1]])
    ConfidenceInterval(CI, GetConfnum(DM, [CI[1]]), MLE(DM), pnames(DM))
end

function ConfidenceBoundarySlice(DM::AbstractDataModel, Confnum::Real; Dirs::Tuple{Int,Int,Int}=(1,2,3), kwargs...)
    Res = ConfidenceRegion(DM, Confnum; Dirs=Dirs, kwargs...)
    if Res isa Tuple{<:AbstractVector{<:Plane},<:AbstractVector{<:AbstractODESolution}}
        ConfidenceBoundarySlice(DM, Res[1], Res[2], Dirs)
    else
        ConfidenceBoundarySlice(DM, Res, Dirs)
    end
end

function ConfidenceBands(DM::AbstractDataModel, S::AbstractBoundarySlice; kwargs...)
    ConfidenceBands(DM, Sols(S); kwargs...)
end

@recipe function f(CB::AbstractBoundarySlice)
    Cube = ConstructCube(Sols(CB))
    xguide := pnames(CB)[1];    yguide := pnames(CB)[2]
    xlims --> Cube[Dirs(CB)[1]];    ylims --> Cube[Dirs(CB)[2]]
    if length(pnames(CB)) ≥ 3
        zguide := pnames(CB)[3]
        zlims --> Cube[Dirs(CB)[3]]
    end
    title --> "CB slice $(Dirs(CB)), level=$(round(Confnum(CB); digits=2))σ"
    titlefontsize --> 8
    @series begin
        linecolor --> nothing
        markercolor --> :black
        label --> "MLE"
        marker --> :hex
        [MLE(CB)]
    end
    for i in eachindex(Sols(CB))
        @series begin
            idxs := Dirs(CB)[3] == 0 ? (Dirs(CB)[1], Dirs(CB)[2]) : Dirs(CB)
            label --> ""
            linecolor --> get(plotattributes, :seriescolor, :red)
            Sols(CB)[i]
        end
    end
end





abstract type AbstractConfidenceBoundary end

struct ConfidenceBoundary <: AbstractConfidenceBoundary
    Slices::AbstractVector{<:AbstractBoundarySlice}
end

function Slices end # Do not intend to refer to Base.Slices here
Slices(CB::ConfidenceBoundary) = CB.Slices
Base.length(CB::ConfidenceBoundary) = CB |> Slices |> length
Base.firstindex(CB::ConfidenceBoundary) = firstindex(Slices(CB))
Base.lastindex(CB::ConfidenceBoundary) = lastindex(Slices(CB))

function ConfidenceBoundary(DM::AbstractDataModel, Res::AbstractVector{<:Tuple{<:AbstractVector{<:Plane}, <:AbstractVector{<:AbstractODESolution}}}, Trips::AbstractVector{<:Tuple{<:Int, <:Int, <:Int}})
    @assert length(Trips) == length(Res)
    Full = length(Res[1][1][1]) == xpdim(DM)
    @assert Full || length(Res[1][1][1]) == pdim(DM) "Ambient space of solutions incompatible with DataModel."
    sort(unique(reduce(vcat, Trips))) != (Full ? (1:xpdim(DM)) : (1:pdim(DM))) && @warn "Not all directions present in ConfidenceBoundary."

    Confnums = map(i->GetConfnum(DM,PlaneCoordinates(Res[i][1][1],Res[i][2][1].u[1]); dof=DOF(DM)), 1:length(Res))
    @assert all(Confnums .≈ Confnums[1]) "Confnums unequal."
    mle = (Full ? TotalLeastSquaresV(DM) : MLE(DM))
    Names = (Full ? _FullNames(DM) : pnames(DM))
    [ConfidenceBoundarySlice(EmbeddedODESolution.(Res[i]...), Trips[i], Confnums[1], mle[[Trips[i]...]], Names[[Trips[i]...]], Full) for i in eachindex(Trips)] |> ConfidenceBoundary
end

@recipe function f(CB::ConfidenceBoundary)
    layout --> CB |> Slices |> length
    for i in eachindex(Slices(CB))
        @series begin
            subplot := i
            leg --> false
            Slices(CB)[i]
        end
    end
end


# struct ConfidenceInterval <: AbstractBoundarySlice
#     Interval::Tuple{<:Number,<:Number}
#     Confnum::Real
#     mle::AbstractVector{<:Number}
#     pnames::AbstractVector{<:AbstractString}
#     function ConfidenceInterval(Interval::Tuple{<:Number,<:Number}, Confnum::Real, mle::AbstractVector{<:Number}, pnames::AbstractVector{<:AbstractString})
#         @assert Interval[1] ≤ mle[1] ≤ Interval[2]
#         @assert length(mle) == length(pnames) == 1
#         new(Interval, Confnum, mle, pnames)
#     end
# end
# Interval(CI::ConfidenceInterval) = CI.Interval


