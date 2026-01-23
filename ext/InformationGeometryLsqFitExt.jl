module InformationGeometryLsqFitExt

    using InformationGeometry, LsqFit, LinearAlgebra, DerivableFunctionsBase


    import InformationGeometry: GetMinimizer, GetMinimum, HasConverged, GetIterations
    GetMinimizer(Res::LsqFit.LsqFitResult) = Res.param
    GetMinimum(Res::LsqFit.LsqFitResult, L::Function) = GetMinimum(GetMinimizer(Res), L)
    HasConverged(Res::LsqFit.LsqFitResult; kwargs...) = Res.converged
    GetIterations(Res::LsqFit.LsqFitResult) = try Res.trace[end].iteration catch; -Inf end # needs kwarg store_trace=true to be available

    

    import InformationGeometry: Curve_fit, muladd!, rescaledjac, MLE, LogPrior, EmbeddingMap, EmbeddingMatrix, Predictor, dPredictor, Data
    import InformationGeometry: GetDomain, yInvCov, InvCov, WoundX, WoundY, GetStartP, Windup, BlockMatrix, ysigma, xdim, ydim, pdim, Npoints, xdist, ydist

    function Curve_fit(DM::AbstractDataModel, initial::AbstractVector{<:Number}=MLE(DM), LogPriorFn::Union{Nothing,Function}=LogPrior(DM); kwargs...)
        Curve_fit(Data(DM), Predictor(DM), dPredictor(DM), initial, LogPriorFn; kwargs...)
    end

    # Always returning complete LsqFitResult object, but use kwarg Full to indicate trace should also be stored
    function Curve_fit(DS::AbstractDataSet, model::ModelOrFunction, initial::AbstractVector{T}=GetStartP(DS,model), LogPriorFn::Union{Nothing,Function}=nothing; verbose::Bool=true, tol::Real=1e-12, 
                    Domain::Union{Nothing, HyperCube}=GetDomain(model), lb=(!isnothing(Domain) ? Domain.L : T[]), ub=(!isnothing(Domain) ? Domain.U : T[]), 
                    # kwarg compatibility with Optimization.jl and Optim.jl backends
                    Full::Bool=false, store_trace::Bool=Full, maxiters::Int=10000, iterations::Int=maxiters, maxtime::Real=600.0, timeout::Real=maxtime, kwargs...) where T<:Number
        verbose && !isnothing(LogPriorFn) && @warn "LsqFit.curve_fit() cannot account for priors. Throwing away given prior and continuing anyway."
        LsqFit.check_data_health(xdata(DS), ydata(DS))
        u = cholesky(yInvCov(DS)).U;    Ydat = - u * ydata(DS)
        F(θ::AbstractVector) = muladd(u, EmbeddingMap(DS, model, θ), Ydat)
        iF(Yres::AbstractVector, θ::AbstractVector) = muladd!(Yres, u, EmbeddingMap(DS, model, θ), Ydat)
        R = LsqFit.OnceDifferentiable(iF, initial, copy(F(initial)); inplace = true, autodiff = :forward)
        LsqFit.lmfit(R, initial, yInvCov(DS); x_tol=tol, g_tol=tol, lower=convert(Vector{T},lb), upper=convert(Vector{T},ub), store_trace, maxIter=iterations, maxTime=Float64(timeout), kwargs...)
    end

    function Curve_fit(DS::AbstractDataSet, model::ModelOrFunction, dmodel::ModelOrFunction, initial::AbstractVector{T}=GetStartP(DS,model), LogPriorFn::Union{Nothing,Function}=nothing; verbose::Bool=true, tol::Real=1e-12, 
                    Domain::Union{Nothing, HyperCube}=GetDomain(model), lb=(!isnothing(Domain) ? Domain.L : T[]), ub=(!isnothing(Domain) ? Domain.U : T[]), 
                    Full::Bool=false, store_trace::Bool=Full, maxiters::Int=10000, iterations::Int=maxiters, maxtime::Real=600.0, timeout::Real=maxtime, kwargs...) where T<:Number
        verbose && !isnothing(LogPriorFn) && @warn "LsqFit.curve_fit() cannot account for priors. Throwing away given prior and continuing anyway."
        LsqFit.check_data_health(xdata(DS), ydata(DS))
        u = cholesky(yInvCov(DS)).U;    Ydat = - u * ydata(DS)
        F(θ::AbstractVector) = u * (EmbeddingMap(DS, model, θ) - ydata(DS))
        dF(θ::AbstractVector) = u * EmbeddingMatrix(DS, dmodel, θ)
        iF(Yres::AbstractVector, θ::AbstractVector) = muladd!(Yres, u, EmbeddingMap(DS, model, θ), Ydat)
        idF(Jac::AbstractMatrix, θ::AbstractVector) = mul!(Jac, u, EmbeddingMatrix(DS, dmodel, θ))
        # dFrescaled(θ::AbstractVector) = u * rescaledjac(Jac(θ), xlen)
        R = LsqFit.OnceDifferentiable(iF, idF, initial, copy(F(initial)); inplace = true)
        LsqFit.lmfit(R, initial, yInvCov(DS); x_tol=tol, g_tol=tol, lower=convert(Vector{T},lb), upper=convert(Vector{T},ub), store_trace, maxIter=iterations, maxTime=Float64(timeout), kwargs...)
    end


    """
        TotalLeastSquaresOLD(DSE::DataSetExact, model::ModelOrFunction, initial::AbstractVector{<:Number}; tol::Real=1e-13, kwargs...) -> Vector
    Experimental feature which takes into account uncertainties in x-values to improve the accuracy of the fit.
    Returns concatenated vector of x-values and parameters. Assumes that the uncertainties in the x-values and y-values are normal, i.e. Gaussian!
    """
    function InformationGeometry.TotalLeastSquaresOLD(DSE::DataSetExact, model::ModelOrFunction, initialp::AbstractVector{<:Number}=GetStartP(DSE, model); tol::Real=1e-13,
                                    ADmode::Union{Symbol,Val}=Val(:ForwardDiff), rescale::Bool=true, verbose::Bool=true, Full::Bool=false, store_trace::Bool=Full, 
                                    maxiters::Int=10000, iterations::Int=maxiters, maxtime::Real=600.0, timeout::Real=maxtime, kwargs...)
        # Improve starting values by fitting with ordinary least squares first
        initialp = Curve_fit(DataSet(WoundX(DSE),Windup(ydata(DSE),ydim(DSE)),ysigma(DSE)), model, initialp; tol=tol, kwargs...).param
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
        # Use inplace Jacobian
        dfrescaled(θ::AbstractVector) = u * rescaledjac(Jac(θ), xlen)
        df(θ::AbstractVector) = u * Jac(θ)
        p0 = vcat(xdata(DSE), initialp)
        R = LsqFit.OnceDifferentiable(f, (rescale ? dfrescaled : df), p0, copy(f(p0)); inplace = false)
        fit = LsqFit.lmfit(R, p0, BlockMatrix(InvCov(xdist(DSE)), InvCov(ydist(DSE))); x_tol=tol, g_tol=tol, store_trace, maxIter=iterations, maxTime=Float64(timeout), kwargs...)
        verbose && !fit.converged && @warn "TLS appears to not have converged."
        Full ? fit : (Windup(fit.param[1:xlen],xdim(DSE)), fit.param[xlen+1:end])
    end

end # module