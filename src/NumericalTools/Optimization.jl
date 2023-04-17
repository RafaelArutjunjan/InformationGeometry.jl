
"""
    muladd!(C, M, X, Y)
C = M*X + Y
"""
muladd!(C, M, X, Y) = (mul!(C, M, X);   C .+= Y;    C)

function LsqFit.curve_fit(DM::AbstractDataModel, initial::AbstractVector{<:Number}=MLE(DM), LogPriorFn::Union{Nothing,Function}=LogPrior(DM); kwargs...)
    curve_fit(Data(DM), Predictor(DM), dPredictor(DM), initial, LogPriorFn; kwargs...)
end

LsqFit.curve_fit(DS::AbstractDataSet, model::Function, args...; kwargs...) = _curve_fit(DS, model, args...; kwargs...)
function LsqFit.curve_fit(DS::AbstractDataSet, M::ModelMap, initial::AbstractVector{T}=GetStartP(DS,M), args...; verbose::Bool=true, kwargs...) where T<:Number
    if initial ∉ Domain(M)
        verbose && @warn "Initial guess $initial not within given bounds. Clamping to bounds and continuing."
        initial = clamp(initial, HyperCube(Domain(M); Padding=-0.01))
    end
    _curve_fit(DS, M, initial, args...; lower=convert(Vector{T},Domain(M).L), upper=convert(Vector{T},Domain(M).U), kwargs...)
end
function LsqFit.curve_fit(DS::AbstractDataSet, M::ModelMap, dM::ModelOrFunction, initial::AbstractVector{T}=GetStartP(DS,M), args...; verbose::Bool=true, kwargs...) where T<:Number
    if initial ∉ Domain(M)
        verbose && @warn "Initial guess $initial not within given bounds. Clamping to bounds and continuing."
        initial = clamp(initial, HyperCube(Domain(M); Padding=-0.01))
    end
    _curve_fit(DS, M, dM, initial, args...; lower=convert(Vector{T},Domain(M).L), upper=convert(Vector{T},Domain(M).U), kwargs...)
end


function _curve_fit(DS::AbstractDataSet, model::ModelOrFunction, initial::AbstractVector{<:Number}=GetStartP(DS,model), LogPriorFn::Union{Nothing,Function}=nothing; tol::Real=1e-14, kwargs...)
    !isnothing(LogPriorFn) && @warn "curve_fit() cannot account for priors. Throwing away given prior and continuing anyway."
    LsqFit.check_data_health(xdata(DS), ydata(DS))
    u = cholesky(yInvCov(DS)).U;    Ydat = - u * ydata(DS)
    F(θ::AbstractVector) = muladd(u, EmbeddingMap(DS, model, θ), Ydat)
    iF(Yres::AbstractVector, θ::AbstractVector) = muladd!(Yres, u, EmbeddingMap(DS, model, θ), Ydat)
    p0 = convert(Vector, initial)
    R = LsqFit.OnceDifferentiable(iF, p0, copy(F(p0)); inplace = true, autodiff = :forward)
    LsqFit.lmfit(R, p0, yInvCov(DS); x_tol=tol, g_tol=tol, kwargs...)
end

function _curve_fit(DS::AbstractDataSet, model::ModelOrFunction, dmodel::ModelOrFunction, initial::AbstractVector{<:Number}=GetStartP(DS,model), LogPriorFn::Union{Nothing,Function}=nothing; tol::Real=1e-14, kwargs...)
    !isnothing(LogPriorFn) && @warn "curve_fit() cannot account for priors. Throwing away given prior and continuing anyway."
    LsqFit.check_data_health(xdata(DS), ydata(DS))
    u = cholesky(yInvCov(DS)).U;    Ydat = - u * ydata(DS)
    F(θ::AbstractVector) = u * (EmbeddingMap(DS, model, θ) - ydata(DS))
    dF(θ::AbstractVector) = u * EmbeddingMatrix(DS, dmodel, θ)
    iF(Yres::AbstractVector, θ::AbstractVector) = muladd!(Yres, u, EmbeddingMap(DS, model, θ), Ydat)
    idF(Jac::AbstractMatrix, θ::AbstractVector) = mul!(Jac, u, EmbeddingMatrix(DS, dmodel, θ))
    p0 = convert(Vector, initial)
    R = LsqFit.OnceDifferentiable(iF, idF, p0, copy(F(p0)); inplace = true)
    LsqFit.lmfit(R, p0, yInvCov(DS); x_tol=tol, g_tol=tol, kwargs...)
end

function rescaledjac(M::AbstractMatrix{<:Number}, xlen::Int)
    M[:,1:xlen] .*= sqrt(size(M,1)/xlen -1.);    return M
end


function TotalLeastSquares(DM::AbstractDataModel, args...; kwargs...)
    !isnothing(LogPrior(DM)) && @warn "TotalLeastSquares() cannot account for priors. Throwing away given prior and continuing anyway."
    TotalLeastSquares(Data(DM), Predictor(DM), args...; kwargs...)
end
"""
    TotalLeastSquares(DSE::DataSetExact, model::ModelOrFunction, initial::AbstractVector{<:Number}; tol::Real=1e-13, kwargs...) -> Vector
Experimental feature which takes into account uncertainties in x-values to improve the accuracy of the fit.
Returns concatenated vector of x-values and parameters. Assumes that the uncertainties in the x-values and y-values are normal, i.e. Gaussian!
"""
function TotalLeastSquares(DSE::DataSetExact, model::ModelOrFunction, initialp::AbstractVector{<:Number}=GetStartP(DSE, model); tol::Real=1e-13,
                                ADmode::Union{Symbol,Val}=Val(:ForwardDiff), rescale::Bool=true, verbose::Bool=true, Full::Bool=false, kwargs...)
    # Improve starting values by fitting with ordinary least squares first
    initialp = curve_fit(DataSet(WoundX(DSE),Windup(ydata(DSE),ydim(DSE)),ysigma(DSE)), model, initialp; tol=tol, kwargs...).param
    if xdist(DSE) isa InformationGeometry.Dirac
        verbose && @warn "TLS: xdist of given data is Dirac, can only perform ordinary least squares."
        return xdata(DSE), initialp
    end

    plen = pdim(DSE,model);  xlen = Npoints(DSE) * xdim(DSE)
    function predictY(ξ::AbstractVector)
        x = view(ξ, 1:xlen);        p = view(ξ, (xlen+1):length(ξ))
        # INPLACE EmbeddingMap!() would be great here!
        vcat(x, EmbeddingMap(DSE, model, p, Windup(x,xdim(DSE))))
    end
    u = cholesky(BlockMatrix(InvCov(xdist(DSE)),InvCov(ydist(DSE)))).U;    Ydata = vcat(xdata(DSE), ydata(DSE))
    f(θ::AbstractVector) = u * (predictY(θ) - Ydata)
    Jac = GetJac(ADmode, predictY)
    dfrescaled(θ::AbstractVector) = u * rescaledjac(Jac(θ), xlen)
    df(θ::AbstractVector) = u * Jac(θ)
    p0 = vcat(xdata(DSE), initialp)
    R = LsqFit.OnceDifferentiable(f, (rescale ? dfrescaled : df), p0, copy(f(p0)); inplace = false)
    fit = LsqFit.lmfit(R, p0, BlockMatrix(InvCov(xdist(DSE)), InvCov(ydist(DSE))); g_tol=tol, kwargs...)
    verbose && !fit.converged && @warn "TLS appears to not have converged."
    Full ? fit : (Windup(fit.param[1:xlen],xdim(DSE)), fit.param[xlen+1:end])
end


## Remove above custom method and implement LiftedCost∘LiftedEmbedding for both
function TotalLeastSquares(DS::AbstractDataSet, model::ModelOrFunction, initialp::AbstractVector{<:Number}=GetStartP(DS, model); kwargs...)
    !HasXerror(DS) && throw("Cannot perform Total Least Squares Fitting for DataSets without x-uncertainties.")
    xlen = Npoints(DS)*xdim(DS);    Cost(x::AbstractVector) = -logpdf(dist(DS), x)
    function predictY(ξ::AbstractVector)
        x = view(ξ, 1:xlen);        p = view(ξ, (xlen+1):length(ξ))
        vcat(x, EmbeddingMap(DS, model, p, Windup(x,xdim(DS))))
    end
    InformationGeometry.minimize(Cost∘predictY, [xdata(DS); initialp]; kwargs...)
end

"""
Concatenated total least squares vector [xdata; pdim].
"""
TotalLeastSquaresV(args...; kwargs...) = reduce(vcat, TotalLeastSquares(args...; kwargs...))


# Currently, Fminbox errors for: SimulatedAnnealing, ParticleSwarm, AbstractNGMRES
# Apparently by design cannot handle: AcceleratedGradientDescent, MomentumGradientDescent, Newton, NewtonTrustRegion, KrylovTrustRegion

ConstrainMeth(meth::Optim.AbstractOptimizer, Domain::Nothing; verbose::Bool=true) = meth
function ConstrainMeth(meth::Optim.AbstractOptimizer, Domain::HyperCube; verbose::Bool=true)
    if meth isa Optim.AbstractConstrainedOptimizer
        meth
    elseif meth isa Union{NelderMead, BFGS, LBFGS, GradientDescent, ConjugateGradient}
        Fminbox(meth)
    elseif meth isa Optim.SecondOrderOptimizer # meth isa Optim.Newton || meth isa Optim.NewtonTrustRegion
        verbose && @warn "$(nameof(typeof(meth)))() does not support constrained optimization, switching to IPNewton()."
        IPNewton()
    else
        verbose && @warn "$(nameof(typeof(meth)))() currently does not support constrained optimization, ignoring given domain boundaries and continuing."
        meth
    end
end

function AutoDiffble(F::Function, x::AbstractVector)
    try
        DerivableFunctionsBase._GetGrad(Val(true))(F, x) isa AbstractVector
        true
    catch;  false   end
end

function ConstrainStart(Start::AbstractVector{T}, Dom::HyperCube; verbose::Bool=true) where T <: Number
    if Start ∈ Dom
        convert(Vector{T}, Start)
    else
        verbose && @warn "Initial guess $Start not within given bounds. Clamping to bounds and continuing."
        convert(Vector{T}, clamp(Start, HyperCube(Dom; Padding=-0.01)))
    end
end
ConstrainStart(Start::AbstractVector{T}, Dom::Nothing; kwargs...) where T <: Number = convert(Vector{T}, Start)


"""
    minimize(F::Function, start::AbstractVector{<:Number}; tol::Real=1e-10, meth=NelderMead(), Full::Bool=false, timeout::Real=600, kwargs...) -> Vector
    minimize(F, dF, start::AbstractVector{<:Number}; tol::Real=1e-10, meth=LBFGS(), Full::Bool=false, timeout::Real=600, kwargs...) -> Vector
    minimize(F, dF, ddF, start::AbstractVector{<:Number}; tol::Real=1e-10, meth=NewtonTrustRegion(), Full::Bool=false, timeout::Real=600, kwargs...) -> Vector
Minimizes the scalar input function using the given `start` using algorithms from `Optim.jl` specified via the keyword `meth`.
`Full=true` returns the full solution object instead of only the minimizing result.
Optionally, the search domain can be bounded by passing a suitable `HyperCube` object as the third argument (ignoring derivatives).
"""
minimize(F::Function, start::AbstractVector, args...; kwargs...) = InformationGeometry.minimize((F,), start, args...; kwargs...)
minimize(F::Function, dF::Function, start::AbstractVector, args...; kwargs...) = InformationGeometry.minimize((F,dF), start, args...; kwargs...)
minimize(F::Function, dF::Function, ddF::Function, start::AbstractVector, args...; kwargs...) = InformationGeometry.minimize((F,dF,ddF), start, args...; kwargs...)
function minimize(Fs::Tuple{Vararg{Function}}, Start::AbstractVector{T}, domain::Union{HyperCube,Nothing}=nothing; Domain::Union{HyperCube,Nothing}=domain, Fthresh::Union{Nothing,Real}=nothing, tol::Real=1e-10,
                            g_tol::Real=tol, x_tol::Real=0.0, meth::Optim.AbstractOptimizer=(length(Fs) == 1 ? NelderMead() : (length(Fs) == 2 ? LBFGS() : NewtonTrustRegion())),
                            timeout::Real=600, Full::Bool=false, verbose::Bool=true, iterations::Int=10000, kwargs...) where T <: Number
    @assert 1 ≤ length(Fs) ≤ 3
    start = ConstrainStart(Start, Domain; verbose=verbose)
    length(Fs) == 3 && @assert MaximalNumberOfArguments(Fs[2]) == MaximalNumberOfArguments(Fs[3]) "Derivatives dF and ddF need to be either both in-place or both out-of-place."
    !(Fs[1](start) isa Number) && throw("Given function must return scalar values, got $(typeof(Fs[1](start))) instead.")
    options = if isnothing(Fthresh)
        Optim.Options(; g_tol=g_tol, x_tol=x_tol, time_limit=floatify(timeout), iterations)
    else  # stopping criterion via callback kwarg
        Optim.Options(; callback=(z->z.value<Fthresh), g_tol=g_tol, x_tol=x_tol, time_limit=floatify(timeout), iterations)
    end
    Cmeth = ConstrainMeth(meth, Domain; verbose=verbose)
    Res = if Cmeth isa Optim.AbstractConstrainedOptimizer
        start ∉ Domain && @warn "Given starting value $start not in specified domain $Domain."
        if length(Fs) == 1
            Optim.optimize(Fs[1], convert(Vector{T},Domain.L), convert(Vector{T},Domain.U), floatify(start), Cmeth, options; kwargs...)
        else
            if Cmeth isa Fminbox
                # Optim.optimize only accepts inplace kwarg for Fminbox
                Optim.optimize(Fs..., convert(Vector{T},Domain.L), convert(Vector{T},Domain.U), floatify(start), Cmeth, options; inplace=MaximalNumberOfArguments(Fs[2])>1, kwargs...)
            else
                Optim.optimize(Fs..., convert(Vector{T},Domain.L), convert(Vector{T},Domain.U), floatify(start), Cmeth, options; kwargs...)
            end
        end
    else
        if length(Fs) == 1
            Optim.optimize(Fs[1], floatify(start), Cmeth, options; kwargs...)
        else
            Optim.optimize(Fs..., floatify(start), Cmeth, options; inplace=MaximalNumberOfArguments(Fs[2])>1, kwargs...)
        end
    end
    verbose && !Optim.converged(Res) && @warn "minimize(): Optimization appears to not have converged."
    Full ? Res : Optim.minimizer(Res)
end

GetDomain(DM::AbstractDataModel) = GetDomain(Predictor(DM))
GetDomain(M::ModelMap) = Domain(M)
GetDomain(F::Function) = nothing
function minimize(DS::AbstractDataSet, Model::ModelOrFunction, start::AbstractVector{<:Number}, LogPriorFn::Union{Nothing,Function}; Domain::Union{HyperCube,Nothing}=GetDomain(Model), kwargs...)
    F = HasXerror(DS) ? FullLiftedNegLogLikelihood(DS,Model,LogPriorFn) : (θ->-loglikelihood(DS,Model,θ,LogPriorFn))
    minimize(F, start, Domain; kwargs...)
end
"""
    RobustFit(DM::AbstractDataModel, start::AbstractVector{<:Number}; tol::Real=1e-10, p::Real=1, kwargs...)
Uses `p`-Norm to judge distance on Dataspace as specified by the keyword.
"""
RobustFit(DM::AbstractDataModel, start::AbstractVector{<:Number}=MLE(DM); kwargs...) = RobustFit(Data(DM), Predictor(DM), start, LogPrior(DM); kwargs...)
function RobustFit(DS::AbstractDataSet, M::ModelOrFunction, start::AbstractVector{<:Number}=GetStartP(DS,M), LogPriorFn::Union{Nothing,Function}=nothing; Domain::Union{HyperCube,Nothing}=(M isa ModelMap ? Domain(M) : nothing), tol::Real=1e-10, p::Real=1, kwargs...)
    HalfSig = cholesky(yInvCov(DS)).U
    # Since F is minimized, need to subtract LogPrior
    F(θ::AbstractVector) = norm(HalfSig * (ydata(DS) - EmbeddingMap(DS, M, θ)), p) - EvalLogPrior(LogPriorFn, θ)
    InformationGeometry.minimize(F, start, Domain; tol=tol, kwargs...)
end
function RobustFit(DS::AbstractDataSet, M::ModelOrFunction, dM::ModelOrFunction, start::AbstractVector{<:Number}=GetStartP(DS,M), LogPriorFn::Union{Nothing,Function}=nothing; Domain::Union{HyperCube,Nothing}=(M isa ModelMap ? Domain(M) : nothing), tol::Real=1e-10, p::Real=1, kwargs...)
    HalfSig = cholesky(yInvCov(DS)).U
    # Since F is minimized, need to subtract LogPrior
    F(θ::AbstractVector) = norm(HalfSig * (EmbeddingMap(DS, M, θ) - ydata(DS)), p) - EvalLogPrior(LogPriorFn, θ)
    function dFp(θ::AbstractVector)
        z = HalfSig * (EmbeddingMap(DS, M, θ) - ydata(DS))
        n = sum(z.^p)^(1/p - 1) * z.^(p-1)
        transpose(HalfSig * EmbeddingMatrix(DS, dM, θ)) * n - EvalLogPriorGrad(LogPriorFn, θ)
    end
    dF1(θ::AbstractVector) = transpose(HalfSig * EmbeddingMatrix(DS, dM, θ)) *  sign.(HalfSig * (EmbeddingMap(DS, M, θ) - ydata(DS))) - EvalLogPrior(LogPriorFn, θ)
    InformationGeometry.minimize(F, (p == 1 ? dF1 : dFp), start, Domain; tol=tol, kwargs...)
end




"""
    LineSearch(Test::Function, start::Number=0.; tol::Real=8e-15, maxiter::Int=10000) -> Number
Finds real number `x` where the boolean-valued `Test(x::Number)` goes from `true` to `false`.
"""
function LineSearch(Test::Function, start::Number=0.; tol::Real=8e-15, maxiter::Int=10000, verbose::Bool=true)
    if ((suff(start) != BigFloat) && tol < 1e-15)
        verbose && @info "LineSearch: start not BigFloat but tol=$tol. Promoting and continuing."
        start = BigFloat(start)
    end
    if !Test(start)
        start += 1e-10
        verbose && @warn "LineSearch: Test(start) did not work, trying Test(start + 1e-10)."
        !Test(start) && throw(ArgumentError("LineSearch: Test not true for starting value."))
    end
    # For some weird reason, if the division by 4 is removed, the loop never terminates for BigFloat-valued "start"s - maybe the compiler erroneously tries to optimize the variable "stepsize" away or something?! (Julia version ≤ 1.6.0)
    stepsize = one(suff(start)) / 2.;       value = start
    for i in 1:maxiter
        if Test(value + stepsize) # inside
            value += stepsize
            # value - start > 2000. && throw("FindConfBoundary: Value larger than 2000.")
        else            #outside
            if stepsize < tol
                return value + stepsize
            else
                stepsize /= 5.
            end
        end
    end
    throw("$maxiter iterations over. Value=$value, Stepsize=$stepsize")
end

function AltLineSearch(Test::Function, Domain::Tuple{T,T}=(0., 1e4), meth::Roots.AbstractBracketingMethod=Roots.AlefeldPotraShi(); tol::Real=1e-12, kwargs...) where T<:Real
    find_zero(Test, Domain, meth; xatol=tol, xrtol=tol, kwargs...)
end
function AltLineSearch(Test::Function, Domain::Tuple{T,T}, meth::Roots.AbstractBracketingMethod=Roots.AlefeldPotraShi(); tol::Real=convert(BigFloat,exp10(-precision(BigFloat)/10))) where T<:BigFloat
    Res = find_zero(Test, (Float64(Domain[1]), Float64(Domain[2])), meth; xatol=1e-14, xrtol=1e-14)
    find_zero(Test, (BigFloat(Res-3e-14),BigFloat(Res+3e-14)), Roots.Bisection(); xatol=tol, xrtol=tol)
end

function AltLineSearch(Test::Function, start::Real, meth::Roots.AbstractNonBracketingMethod=Roots.Order2(); tol::Real=1e-12, kwargs...)
    find_zero(Test, start, meth; xatol=tol, xrtol=tol, kwargs...)
end
function AltLineSearch(Test::Function, start::BigFloat, meth::Roots.AbstractNonBracketingMethod=Roots.Order2(); tol::Real=convert(BigFloat,exp10(-precision(BigFloat)/10)), kwargs...)
    Res = find_zero(Test, Float64(start), meth; xatol=1e-14, xrtol=1e-14)
    find_zero(Test, BigFloat(Res), meth; xatol=tol, xrtol=tol, kwargs...)
end
