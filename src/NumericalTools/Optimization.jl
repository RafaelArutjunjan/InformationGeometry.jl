


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
    X = xdata(DS);  Y = ydata(DS);    LsqFit.check_data_health(X, Y)
    u = cholesky(yInvCov(DS)).U
    !isnothing(LogPriorFn) && @warn "curve_fit() cannot account for priors. Throwing away given prior and continuing anyway."
    f(p) = u * (EmbeddingMap(DS, model, p) - Y)
    p0 = convert(Vector, initial)
    R = LsqFit.OnceDifferentiable(f, p0, copy(f(p0)); inplace = false, autodiff = :forward)
    LsqFit.lmfit(R, p0, yInvCov(DS); x_tol=tol, g_tol=tol, kwargs...)
end

function curve_fit(DS::AbstractDataSet, model::Function, dmodel::ModelOrFunction, initial::AbstractVector{<:Number}=GetStartP(DS,model), LogPriorFn::Union{Nothing,Function}=nothing; tol::Real=1e-14, kwargs...)
    X = xdata(DS);  Y = ydata(DS);    LsqFit.check_data_health(X, Y)
    u = cholesky(yInvCov(DS)).U
    !isnothing(LogPriorFn) && @warn "curve_fit() cannot account for priors. Throwing away given prior and continuing anyway."
    f(p) = u * (EmbeddingMap(DS, model, p) - Y)
    df(p) = u * EmbeddingMatrix(DS, dmodel, p)
    p0 = convert(Vector, initial)
    R = LsqFit.OnceDifferentiable(f, df, p0, copy(f(p0)); inplace = false)
    LsqFit.lmfit(R, p0, yInvCov(DS); x_tol=tol, g_tol=tol, kwargs...)
end

function normalizedjac(M::AbstractMatrix{<:Number}, xlen::Int)
    M[:,1:xlen] .*= sqrt(size(M,1)/xlen -1.);    return M
end


TotalLeastSquares(DM::AbstractDataModel, args...; kwargs...) = TotalLeastSquares(Data(DM), Predictor(DM), args...; kwargs...)
"""
    TotalLeastSquares(DSE::DataSetExact, model::ModelOrFunction, initial::AbstractVector{<:Number}; tol::Real=1e-13, kwargs...) -> Vector
Experimental feature which takes into account uncertainties in x-values to improve the accuracy of the fit.
Returns concatenated vector of x-values and parameters. Assumes that the uncertainties in the x-values and y-values are normal, i.e. Gaussian!
"""
function TotalLeastSquares(DSE::DataSetExact, model::ModelOrFunction, initialp::AbstractVector{<:Number}=GetStartP(DSE, model); ADmode::Union{Symbol,Val}=Val(:ForwardDiff), tol::Real=1e-13, rescale::Bool=true, kwargs...)
    # Improve starting values by fitting with ordinary least squares first
    initialp = curve_fit(DataSet(WoundX(DSE),Windup(ydata(DSE),ydim(DSE)),ysigma(DSE)), model, initialp; tol=tol, kwargs...).param
    if xdist(DSE) isa InformationGeometry.Dirac
        println("xdist of given data is Dirac, can only use ordinary least squares.")
        return xdata(DSE), initialp
    end

    plen = pdim(DSE,model);  xlen = Npoints(DSE) * xdim(DSE)
    function predictY(ξ::AbstractVector)
        x = view(ξ, 1:xlen);        p = view(ξ, (xlen+1):length(ξ))
        # INPLACE EmbeddingMap!() would be great here!
        vcat(x, EmbeddingMap(DSE, model, p, Windup(x,xdim(DSE))))
    end
    u = cholesky(BlockMatrix(InvCov(xdist(DSE)),InvCov(ydist(DSE)))).U;    Ydata = vcat(xdata(DSE), ydata(DSE))
    f(p) = u * (predictY(p) - Ydata)
    Jac = GetJac(ADmode, predictY)
    dfnormalized(p) = u * normalizedjac(Jac(p), xlen)
    df(p) = u * Jac(p)
    p0 = vcat(xdata(DSE), initialp)
    R = rescale ? LsqFit.OnceDifferentiable(f, dfnormalized, p0, copy(f(p0)); inplace = false) : LsqFit.OnceDifferentiable(f, df, p0, copy(f(p0)); inplace = false)
    fit = LsqFit.lmfit(R, p0, BlockMatrix(InvCov(xdist(DSE)), InvCov(ydist(DSE))); x_tol=tol, g_tol=tol, kwargs...)
    Windup(fit.param[1:xlen],xdim(DSE)), fit.param[xlen+1:end]
end

function TotalLeastSquares(DS::AbstractDataSet, model::ModelOrFunction, initialp::AbstractVector{<:Number}=GetStartP(DS, model); tol::Real=1e-13, kwargs...)
    sum(abs, xsigma(DS)) == 0.0 && throw("Cannot perform Total Least Squares Fitting for DataSets without x-uncertainties.")
    xlen = Npoints(DS)*xdim(DS);    Cost(x::AbstractVector) = -logpdf(dist(DS), x)
    function predictY(ξ::AbstractVector)
        x = view(ξ, 1:xlen);        p = view(ξ, (xlen+1):length(ξ))
        vcat(x, EmbeddingMap(DS, model, p, Windup(x,xdim(DS))))
    end
    InformationGeometry.minimize(Cost∘predictY, [xdata(DS); initialp]; tol=tol, kwargs...)
end


"""
    minimize(F::Function, start::AbstractVector{<:Number}; tol::Real=1e-10, meth=NelderMead(), Full::Bool=false, timeout::Real=200, kwargs...) -> Vector
Minimizes the scalar input function using the given `start` using algorithms from `Optim.jl` specified via the keyword `meth`.
`Full=true` returns the full solution object instead of only the minimizing result.
Optionally, the search domain can be bounded by passing a suitable `HyperCube` object as the third argument.
"""
function minimize(F::Function, start::AbstractVector{<:Number}, Domain::Union{HyperCube,Nothing}=nothing; Fthresh::Union{Nothing,Real}=nothing, tol::Real=1e-10, meth::Optim.AbstractOptimizer=NelderMead(), timeout::Real=200, Full::Bool=false, kwargs...)
    !(F(start) isa Number) && throw("Given function must return scalar values, got $(typeof(F(start))) instead.")
    options = if isnothing(Fthresh)
        Optim.Options(g_tol=tol, x_tol=tol, time_limit=floatify(timeout))
    else  # stopping criterion via callback kwarg
        Optim.Options(callback=(z->z.value<Fthresh), g_tol=tol, x_tol=tol, time_limit=floatify(timeout))
    end
    Res = if isnothing(Domain)
        optimize(F, floatify(start), meth, options; kwargs...)
    else
        start ∉ Domain && throw("Given starting value not in specified domain.")
        optimize(F, convert(Vector{Float64},Domain.L), convert(Vector{Float64},Domain.U), floatify(start), meth, options; kwargs...)
    end
    Full ? Res : Optim.minimizer(Res)
end
minimize(FdF::Tuple{Function,Function}, args...; kwargs...) = minimize(FdF[1], FdF[2], args...; kwargs...)
function minimize(F::Function, dF::Function, start::AbstractVector{<:Number}, Domain::Union{HyperCube,Nothing}=nothing; Fthresh::Union{Nothing,Real}=nothing, tol::Real=1e-10, meth::Optim.AbstractOptimizer=BFGS(), timeout::Real=200, Full::Bool=false, kwargs...)
    !(F(start) isa Number) && throw("Given function must return scalar values, got $(typeof(F(start))) instead.")
    # Wrap dF to make it inplace
    newdF = MaximalNumberOfArguments(dF) < 2 ? ((G,x)->copyto!(G,dF(x))) : dF
    options = if isnothing(Fthresh)
        Optim.Options(g_tol=tol, x_tol=tol, time_limit=floatify(timeout))
    else  # stopping criterion via callback kwarg
        Optim.Options(callback=(z->z.value<Fthresh), g_tol=tol, x_tol=tol, time_limit=floatify(timeout))
    end
    Res = if isnothing(Domain)
        Optim.optimize(F, newdF, floatify(start), meth, options; inplace=true, kwargs...)
    else
        start ∉ Domain && throw("Given starting value not in specified domain.")
        Optim.optimize(F, newdF, convert(Vector{Float64},Domain.L), convert(Vector{Float64},Domain.U), floatify(start), meth, options; inplace=true, kwargs...)
    end
    Full ? Res : Optim.minimizer(Res)
end

"""
    RobustFit(DM::AbstractDataModel, start::AbstractVector{<:Number}; tol::Real=1e-10, p::Real=1, kwargs...)
Uses `p`-Norm to judge distance on Dataspace as specified by the keyword.
"""
RobustFit(DM::AbstractDataModel, args...; kwargs...) = RobustFit(Data(DM), Predictor(DM), args...; kwargs...)
function RobustFit(DS::AbstractDataSet, M::ModelOrFunction, start::AbstractVector{<:Number}=GetStartP(DS,M), Domain::Union{HyperCube,Nothing}=(M isa ModelMap ? M.Domain : nothing); tol::Real=1e-10, p::Real=1, kwargs...)
    HalfSig = cholesky(yInvCov(DS)).U
    F(x::AbstractVector) = norm(HalfSig * (ydata(DS) - EmbeddingMap(DS, M, x)), p)
    InformationGeometry.minimize(F, start, Domain; tol=tol, kwargs...)
end
function RobustFit(DS::AbstractDataSet, M::ModelOrFunction, dM::ModelOrFunction, start::AbstractVector{<:Number}=GetStartP(DS,M), Domain::Union{HyperCube,Nothing}=(M isa ModelMap ? M.Domain : nothing); tol::Real=1e-10, p::Real=1, kwargs...)
    HalfSig = cholesky(yInvCov(DS)).U
    F(x::AbstractVector) = norm(HalfSig * (EmbeddingMap(DS, M, x) - ydata(DS)), p)
    function dFp(x::AbstractVector)
        z = HalfSig * (EmbeddingMap(DS, M, x) - ydata(DS))
        n = sum(z.^p)^(1/p - 1) * z.^(p-1)
        transpose(HalfSig * EmbeddingMatrix(DS, dM, x)) * n
    end
    dF1(x::AbstractVector) = transpose(HalfSig * EmbeddingMatrix(DS, dM, x)) *  sign.(HalfSig * (EmbeddingMap(DS, M, x) - ydata(DS)))
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
