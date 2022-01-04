


import LsqFit.curve_fit
function curve_fit(DM::AbstractDataModel, initial::AbstractVector{<:Number}=MLE(DM), LogPriorFn::Union{Nothing,Function}=LogPrior(DM); tol::Real=1e-14, kwargs...)
    curve_fit(Data(DM), Predictor(DM), dPredictor(DM), initial, LogPriorFn; tol=tol, kwargs...)
end

function curve_fit(DS::AbstractDataSet, M::ModelMap, initial::AbstractVector{<:Number}=GetStartP(DS,M), LogPriorFn::Union{Nothing,Function}=nothing; tol::Real=1e-14, kwargs...)
    curve_fit(DS, M.Map, initial, LogPriorFn; tol=tol, lower=convert(Vector,M.Domain.L), upper=convert(Vector,M.Domain.U), kwargs...)
end

function curve_fit(DS::AbstractDataSet, M::ModelMap, dM::ModelOrFunction, initial::AbstractVector{<:Number}=GetStartP(DS,M), LogPriorFn::Union{Nothing,Function}=nothing; tol::Real=1e-14, kwargs...)
    curve_fit(DS, M.Map, dM, initial, LogPriorFn; tol=tol, lower=convert(Vector,M.Domain.L), upper=convert(Vector,M.Domain.U), kwargs...)
end

function curve_fit(DS::AbstractDataSet, model::Function, initial::AbstractVector{<:Number}=GetStartP(DS,model), LogPriorFn::Union{Nothing,Function}=nothing; tol::Real=1e-14, kwargs...)
    !isnothing(LogPriorFn) && @warn "curve_fit() cannot account for priors. Throwing away given prior and continuing anyway."
    X = xdata(DS);  Y = ydata(DS);    LsqFit.check_data_health(X, Y)
    u = cholesky(yInvCov(DS)).U
    f(θ::AbstractVector) = u * (EmbeddingMap(DS, model, θ) - Y)
    p0 = convert(Vector, initial)
    R = LsqFit.OnceDifferentiable(f, p0, copy(f(p0)); inplace = false, autodiff = :forward)
    LsqFit.lmfit(R, p0, yInvCov(DS); x_tol=tol, g_tol=tol, kwargs...)
end

function curve_fit(DS::AbstractDataSet, model::Function, dmodel::ModelOrFunction, initial::AbstractVector{<:Number}=GetStartP(DS,model), LogPriorFn::Union{Nothing,Function}=nothing; tol::Real=1e-14, kwargs...)
    !isnothing(LogPriorFn) && @warn "curve_fit() cannot account for priors. Throwing away given prior and continuing anyway."
    X = xdata(DS);  Y = ydata(DS);    LsqFit.check_data_health(X, Y)
    u = cholesky(yInvCov(DS)).U
    f(θ::AbstractVector) = u * (EmbeddingMap(DS, model, θ) - Y)
    df(θ::AbstractVector) = u * EmbeddingMatrix(DS, dmodel, θ)
    p0 = convert(Vector, initial)
    R = LsqFit.OnceDifferentiable(f, df, p0, copy(f(p0)); inplace = false)
    LsqFit.lmfit(R, p0, yInvCov(DS); x_tol=tol, g_tol=tol, kwargs...)
end

function normalizedjac(M::AbstractMatrix{<:Number}, xlen::Int)
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
    dfnormalized(θ::AbstractVector) = u * normalizedjac(Jac(θ), xlen)
    df(θ::AbstractVector) = u * Jac(θ)
    p0 = vcat(xdata(DSE), initialp)
    R = rescale ? LsqFit.OnceDifferentiable(f, dfnormalized, p0, copy(f(p0)); inplace = false) : LsqFit.OnceDifferentiable(f, df, p0, copy(f(p0)); inplace = false)
    fit = LsqFit.lmfit(R, p0, BlockMatrix(InvCov(xdist(DSE)), InvCov(ydist(DSE))); x_tol=tol, g_tol=tol, kwargs...)
    verbose && !fit.converged && @warn "TLS appears to not have converged."
    Full ? fit : (Windup(fit.param[1:xlen],xdim(DSE)), fit.param[xlen+1:end])
end


## Remove above custom method and implement LiftedCost∘LiftedEmbedding for both
function TotalLeastSquares(DS::AbstractDataSet, model::ModelOrFunction, initialp::AbstractVector{<:Number}=GetStartP(DS, model); kwargs...)
    sum(abs, xsigma(DS)) == 0.0 && throw("Cannot perform Total Least Squares Fitting for DataSets without x-uncertainties.")
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
        DerivableFunctions._GetGrad(Val(true))(F, x) isa AbstractVector
        true
    catch;  false   end
end


"""
    minimize(F::Function, start::AbstractVector{<:Number}; tol::Real=1e-10, meth=NelderMead(), Full::Bool=false, timeout::Real=200, kwargs...) -> Vector
Minimizes the scalar input function using the given `start` using algorithms from `Optim.jl` specified via the keyword `meth`.
`Full=true` returns the full solution object instead of only the minimizing result.
Optionally, the search domain can be bounded by passing a suitable `HyperCube` object as the third argument.
"""
function minimize(F::Function, start::AbstractVector{<:Number}, Domain::Union{HyperCube,Nothing}=nothing; Fthresh::Union{Nothing,Real}=nothing, tol::Real=1e-10,
                            meth::Optim.AbstractOptimizer=NelderMead(), timeout::Real=200, Full::Bool=false, verbose::Bool=true, kwargs...)
    !(F(start) isa Number) && throw("Given function must return scalar values, got $(typeof(F(start))) instead.")
    options = if isnothing(Fthresh)
        Optim.Options(g_tol=tol, x_tol=tol, time_limit=floatify(timeout))
    else  # stopping criterion via callback kwarg
        Optim.Options(callback=(z->z.value<Fthresh), g_tol=tol, x_tol=tol, time_limit=floatify(timeout))
    end
    Cmeth = ConstrainMeth(meth,Domain)
    Res = if Cmeth isa Optim.AbstractConstrainedOptimizer
        start ∉ Domain && @warn "Given starting value $start not in specified domain $Domain."
        Optim.optimize(F, convert(Vector{Float64},Domain.L), convert(Vector{Float64},Domain.U), floatify(start), Cmeth, options; kwargs...)
    else
        Optim.optimize(F, floatify(start), Cmeth, options; kwargs...)
    end
    verbose && !Optim.converged(Res) && @warn "minimize(): Optimization appears to not have converged."
    Full ? Res : Optim.minimizer(Res)
end
function minimize(F::Function, dF::Function, start::AbstractVector{<:Number}, Domain::Union{HyperCube,Nothing}=nothing; Fthresh::Union{Nothing,Real}=nothing, tol::Real=1e-10,
                            meth::Optim.AbstractOptimizer=BFGS(), timeout::Real=200, Full::Bool=false, verbose::Bool=true, kwargs...)
    !(F(start) isa Number) && throw("Given function must return scalar values, got $(typeof(F(start))) instead.")
    options = if isnothing(Fthresh)
        Optim.Options(g_tol=tol, x_tol=tol, time_limit=floatify(timeout))
    else  # stopping criterion via callback kwarg
        Optim.Options(callback=(z->z.value<Fthresh), g_tol=tol, x_tol=tol, time_limit=floatify(timeout))
    end
    Cmeth = ConstrainMeth(meth,Domain)
    Res = if Cmeth isa Optim.AbstractConstrainedOptimizer
        start ∉ Domain && @warn "Given starting value $start not in specified domain $Domain."
        Optim.optimize(F, dF, convert(Vector{Float64},Domain.L), convert(Vector{Float64},Domain.U), floatify(start), Cmeth, options; inplace=MaximalNumberOfArguments(dF)>1, kwargs...)
    else
        Optim.optimize(F, dF, floatify(start), Cmeth, options; inplace=MaximalNumberOfArguments(dF)>1, kwargs...)
    end
    verbose && !Optim.converged(Res) && @warn "minimize(): Optimization appears to not have converged."
    Full ? Res : Optim.minimizer(Res)
end
function minimize(F::Function, dF::Function, ddF::Function, start::AbstractVector{<:Number}, Domain::Union{HyperCube,Nothing}=nothing; Fthresh::Union{Nothing,Real}=nothing, tol::Real=1e-10,
                            meth::Optim.AbstractOptimizer=NewtonTrustRegion(), timeout::Real=200, Full::Bool=false, verbose::Bool=true, kwargs...)
    !(F(start) isa Number) && throw("Given function must return scalar values, got $(typeof(F(start))) instead.")
    @assert MaximalNumberOfArguments(dF) == MaximalNumberOfArguments(ddF) "Derivatives dF and ddF need to be either both in-place or both not in-place"
    options = if isnothing(Fthresh)
        Optim.Options(g_tol=tol, x_tol=tol, time_limit=floatify(timeout))
    else  # stopping criterion via callback kwarg
        Optim.Options(callback=(z->z.value<Fthresh), g_tol=tol, x_tol=tol, time_limit=floatify(timeout))
    end
    Cmeth = ConstrainMeth(meth,Domain)
    Res = if Cmeth isa Optim.AbstractConstrainedOptimizer
        start ∉ Domain && @warn "Given starting value $start not in specified domain $Domain."
        Optim.optimize(F, dF, ddF, convert(Vector{Float64},Domain.L), convert(Vector{Float64},Domain.U), floatify(start), Cmeth, options; inplace=MaximalNumberOfArguments(dF)>1, kwargs...)
    else
        Optim.optimize(F, dF, ddF, floatify(start), Cmeth, options; inplace=MaximalNumberOfArguments(dF)>1, kwargs...)
    end
    verbose && !Optim.converged(Res) && @warn "minimize(): Optimization appears to not have converged."
    Full ? Res : Optim.minimizer(Res)
end
minimize(FdF::Tuple, args...; kwargs...) = InformationGeometry.minimize(FdF..., args...; kwargs...)

"""
    RobustFit(DM::AbstractDataModel, start::AbstractVector{<:Number}; tol::Real=1e-10, p::Real=1, kwargs...)
Uses `p`-Norm to judge distance on Dataspace as specified by the keyword.
"""
RobustFit(DM::AbstractDataModel, start::AbstractVector{<:Number}=MLE(DM); kwargs...) = RobustFit(Data(DM), Predictor(DM), start, LogPrior(DM); kwargs...)
function RobustFit(DS::AbstractDataSet, M::ModelOrFunction, start::AbstractVector{<:Number}=GetStartP(DS,M), LogPriorFn::Union{Nothing,Function}=nothing; Domain::Union{HyperCube,Nothing}=(M isa ModelMap ? M.Domain : nothing), tol::Real=1e-10, p::Real=1, kwargs...)
    HalfSig = cholesky(yInvCov(DS)).U
    # Since F is minimized, need to subtract LogPrior
    F(θ::AbstractVector) = norm(HalfSig * (ydata(DS) - EmbeddingMap(DS, M, θ)), p) - EvalLogPrior(LogPriorFn, θ)
    InformationGeometry.minimize(F, start, Domain; tol=tol, kwargs...)
end
function RobustFit(DS::AbstractDataSet, M::ModelOrFunction, dM::ModelOrFunction, start::AbstractVector{<:Number}=GetStartP(DS,M), LogPriorFn::Union{Nothing,Function}=nothing; Domain::Union{HyperCube,Nothing}=(M isa ModelMap ? M.Domain : nothing), tol::Real=1e-10, p::Real=1, kwargs...)
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
function LineSearch(Test::Function, start::Number=0.; tol::Real=8e-15, maxiter::Int=10000)
    if ((suff(start) != BigFloat) && tol < 1e-15)
        println("LineSearch: start not BigFloat but tol=$tol. Promoting and continuing.")
        start = BigFloat(start)
    end
    if !Test(start)
        start += 1e-10
        println("LineSearch: Test(start) did not work, trying Test(start + 1e-10).")
        !Test(start) && throw(ArgumentError("LineSearch: Test not true for starting value."))
    end
    # For some weird reason, if the division by 4 is removed, the loop never terminates for BigFloat-valued "start"s - maybe the compiler erroneously tries to optimize the variable "stepsize" away or something?! (Julia version ≤ 1.6.0)
    stepsize = one(suff(start)) / 4.;       value = start
    for i in 1:maxiter
        if Test(value + stepsize) # inside
            value += stepsize
            # value - start > 2000. && throw("FindConfBoundary: Value larger than 2000.")
        else            #outside
            if stepsize < tol
                return value + stepsize
            end
            stepsize /= 5.
        end
    end
    throw("$maxiter iterations over. Value=$value, Stepsize=$stepsize")
end

function AltLineSearch(Test::Function, Domain::Tuple{T,T}=(0., 1e4), meth::Roots.AbstractUnivariateZeroMethod=Roots.AlefeldPotraShi(); tol::Real=1e-12) where T<:Real
    find_zero(Test, Domain, meth; xatol=tol, xrtol=tol)
end
function AltLineSearch(Test::Function, Domain::Tuple{T,T}, meth::Roots.AbstractUnivariateZeroMethod=Roots.AlefeldPotraShi(); tol::Real=convert(BigFloat,10^(-precision(BigFloat)/10))) where T<:BigFloat
    Res = find_zero(Test, (Float64(Domain[1]), Float64(Domain[2])), meth; xatol=1e-14, xrtol=1e-14)
    find_zero(Test, (BigFloat(Res-2e-14),BigFloat(Res+2e-14)), Bisection(); xatol=tol, xrtol=tol)
end
