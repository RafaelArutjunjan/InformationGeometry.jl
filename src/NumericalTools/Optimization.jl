
"""
    muladd!(C, M, X, Y)
C = M*X + Y
"""
muladd!(C, M, X, Y) = (mul!(C, M, X);   C .+= Y;    C)

function LsqFit.curve_fit(DM::AbstractDataModel, initial::AbstractVector{<:Number}=MLE(DM), LogPriorFn::Union{Nothing,Function}=LogPrior(DM); kwargs...)
    curve_fit(Data(DM), Predictor(DM), dPredictor(DM), initial, LogPriorFn; kwargs...)
end

# Always returning complete LsqFitResult object, but use kwarg Full to indicate trace should also be stored
function LsqFit.curve_fit(DS::AbstractDataSet, model::ModelOrFunction, initial::AbstractVector{T}=GetStartP(DS,model), LogPriorFn::Union{Nothing,Function}=nothing; verbose::Bool=true, tol::Real=1e-12, 
                Domain::Union{Nothing, HyperCube}=GetDomain(model), lb=(!isnothing(Domain) ? Domain.L : T[]), ub=(!isnothing(Domain) ? Domain.U : T[]), 
                # kwarg compatibility with Optimization.jl and Optim.jl backends
                Full::Bool=false, store_trace::Bool=Full, maxiters::Int=10000, iterations::Int=maxiters, maxtime::Real=600.0, timeout::Real=maxtime, kwargs...) where T<:Number
    verbose && !isnothing(LogPriorFn) && @warn "curve_fit() cannot account for priors. Throwing away given prior and continuing anyway."
    LsqFit.check_data_health(xdata(DS), ydata(DS))
    u = cholesky(yInvCov(DS)).U;    Ydat = - u * ydata(DS)
    F(θ::AbstractVector) = muladd(u, EmbeddingMap(DS, model, θ), Ydat)
    iF(Yres::AbstractVector, θ::AbstractVector) = muladd!(Yres, u, EmbeddingMap(DS, model, θ), Ydat)
    R = LsqFit.OnceDifferentiable(iF, initial, copy(F(initial)); inplace = true, autodiff = :forward)
    LsqFit.lmfit(R, initial, yInvCov(DS); x_tol=tol, g_tol=tol, lower=convert(Vector{T},lb), upper=convert(Vector{T},ub), store_trace, maxIter=iterations, maxTime=Float64(timeout), kwargs...)
end

function LsqFit.curve_fit(DS::AbstractDataSet, model::ModelOrFunction, dmodel::ModelOrFunction, initial::AbstractVector{T}=GetStartP(DS,model), LogPriorFn::Union{Nothing,Function}=nothing; verbose::Bool=true, tol::Real=1e-12, 
                Domain::Union{Nothing, HyperCube}=GetDomain(model), lb=(!isnothing(Domain) ? Domain.L : T[]), ub=(!isnothing(Domain) ? Domain.U : T[]), 
                Full::Bool=false, store_trace::Bool=Full, maxiters::Int=10000, iterations::Int=maxiters, maxtime::Real=600.0, timeout::Real=maxtime, kwargs...) where T<:Number
    verbose && !isnothing(LogPriorFn) && @warn "curve_fit() cannot account for priors. Throwing away given prior and continuing anyway."
    LsqFit.check_data_health(xdata(DS), ydata(DS))
    u = cholesky(yInvCov(DS)).U;    Ydat = - u * ydata(DS)
    F(θ::AbstractVector) = u * (EmbeddingMap(DS, model, θ) - ydata(DS))
    dF(θ::AbstractVector) = u * EmbeddingMatrix(DS, dmodel, θ)
    iF(Yres::AbstractVector, θ::AbstractVector) = muladd!(Yres, u, EmbeddingMap(DS, model, θ), Ydat)
    idF(Jac::AbstractMatrix, θ::AbstractVector) = mul!(Jac, u, EmbeddingMatrix(DS, dmodel, θ))
    R = LsqFit.OnceDifferentiable(iF, idF, initial, copy(F(initial)); inplace = true)
    LsqFit.lmfit(R, initial, yInvCov(DS); x_tol=tol, g_tol=tol, lower=convert(Vector{T},lb), upper=convert(Vector{T},ub), store_trace, maxIter=iterations, maxTime=Float64(timeout), kwargs...)
end

function rescaledjac(M::AbstractMatrix{T}, xlen::Int) where T<:Number
    M[:,1:xlen] .*= sqrt(size(M,1)/xlen -one(T));    return M
end


function TotalLeastSquaresOLD(DM::AbstractDataModel, args...; kwargs...)
    !isnothing(LogPrior(DM)) && @warn "TotalLeastSquares() cannot account for priors. Throwing away given prior and continuing anyway."
    TotalLeastSquaresOLD(Data(DM), Predictor(DM), args...; kwargs...)
end
"""
    TotalLeastSquaresOLD(DSE::DataSetExact, model::ModelOrFunction, initial::AbstractVector{<:Number}; tol::Real=1e-13, kwargs...) -> Vector
Experimental feature which takes into account uncertainties in x-values to improve the accuracy of the fit.
Returns concatenated vector of x-values and parameters. Assumes that the uncertainties in the x-values and y-values are normal, i.e. Gaussian!
"""
function TotalLeastSquaresOLD(DSE::DataSetExact, model::ModelOrFunction, initialp::AbstractVector{<:Number}=GetStartP(DSE, model); tol::Real=1e-13,
                                ADmode::Union{Symbol,Val}=Val(:ForwardDiff), rescale::Bool=true, verbose::Bool=true, Full::Bool=false, store_trace::Bool=Full, 
                                maxiters::Int=10000, iterations::Int=maxiters, maxtime::Real=600.0, timeout::Real=maxtime, kwargs...)
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
    fit = LsqFit.lmfit(R, p0, BlockMatrix(InvCov(xdist(DSE)), InvCov(ydist(DSE))); x_tol=tol, g_tol=tol, store_trace, maxIter=iterations, maxTime=Float64(timeout), kwargs...)
    verbose && !fit.converged && @warn "TLS appears to not have converged."
    Full ? fit : (Windup(fit.param[1:xlen],xdim(DSE)), fit.param[xlen+1:end])
end



"""
    TotalLeastSquares(DM::DataModel, initialθ::AbstractVector=MLE(DM), start::AbstractVector=[xdata(DM);initialp]; tol::Real=1e-13, kwargs...) -> Vector
Experimental feature which takes into account uncertainties in x-values to improve the accuracy of the fit.
Returns concatenated vector of x-values and parameters. Assumes that the uncertainties in the x-values and y-values are normal, i.e. Gaussian!
"""
function TotalLeastSquares(DM::AbstractDataModel, initialp::AbstractVector{<:Number}=MLE(DM), start::AbstractVector{<:Number}=[xdata(DM);initialp]; Full::Bool=false, 
                            rescale::Bool=false, ADmode::Union{Symbol,Val}=Val(:ForwardDiff), kwargs...)
    if rescale
        @assert Data(DM) isa DataSetExact "rescale kwarg only implemented for DataSetExact."
        !isnothing(LogPrior(DM)) && @warn "TotalLeastSquaresOLD() cannot account for priors. Throwing away given prior and continuing anyway."
        return TotalLeastSquaresOLD(Data(DM), Predictor(DM), initialp; rescale, Full, ADmode, kwargs...)
    end
    !HasXerror(DM) && throw("Cannot perform Total Least Squares Fitting for DataSets without x-uncertainties.")
    Res = InformationGeometry.minimize(FullLiftedNegLogLikelihood(DM), start; Full, kwargs...)
    Full ? Res : (Windup(Res[1:length(xdata(DM))],xdim(DM)), Res[length(xdata(DM))+1:end])
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
    start = if Start ∈ Dom
        Start
    else
        verbose && @warn "Initial guess $Start not within given bounds. Clamping to bounds and continuing."
        clamp(Start, HyperCube(Dom; Padding=-1e-3))
    end
    StaticArrays.isstatic(start) ? convert(Vector{T}, start) : start
end
ConstrainStart(start::AbstractVector{T}, Dom::Nothing; kwargs...) where T<:Number = StaticArrays.isstatic(start) ? convert(Vector{T}, start) : start


"""
    minimize(F::Function, start::AbstractVector{<:Number}; tol::Real=1e-10, meth=NelderMead(), Full::Bool=false, maxtime::Real=600, kwargs...) -> Vector
    minimize(F, dF, start::AbstractVector{<:Number}; tol::Real=1e-10, meth=LBFGS(), Full::Bool=false, maxtime::Real=600, kwargs...) -> Vector
    minimize(F, dF, ddF, start::AbstractVector{<:Number}; tol::Real=1e-10, meth=NewtonTrustRegion(), Full::Bool=false, maxtime::Real=600, kwargs...) -> Vector
Minimizes the scalar input function using the given `start` using any algorithms from the `Optimation.jl` ecosystem specified via the keyword `meth`.
`Full=true` returns the full solution object instead of only the minimizing result.
Optionally, the search domain can be bounded by passing a suitable `HyperCube` object as the third argument (ignoring derivatives).
"""
minimize(F::Function, start::AbstractVector, args...; kwargs...) = InformationGeometry.minimize((F,), start, args...; kwargs...)
minimize(F::Function, dF::Function, start::AbstractVector, args...; kwargs...) = InformationGeometry.minimize((F,dF), start, args...; kwargs...)
minimize(F::Function, dF::Function, ddF::Function, start::AbstractVector, args...; kwargs...) = InformationGeometry.minimize((F,dF,ddF), start, args...; kwargs...)


function minimize(Fs::Tuple{Vararg{Function}}, Start::AbstractVector{<:Number}, domain::Union{HyperCube,Nothing}=nothing; Domain::Union{HyperCube,Nothing}=domain,
                meth=(length(Fs) == 1 ? Optim.NelderMead() : (length(Fs) == 2 ? Optim.LBFGS(;linesearch=LineSearches.BackTracking()) : Optim.NewtonTrustRegion())), timeout::Real=600.0, maxtime::Real=timeout, kwargs...)
    minimize(Fs, Start, meth; Domain, maxtime, kwargs...)
end


# Force use of custom function instead of using OptimizationOptimJL.jl as intermediate layer
minimize(Fs::Tuple{Vararg{Function}}, Start::AbstractVector{<:Number}, meth::Optim.AbstractOptimizer; OptimJL::Bool=true, kwargs...) = (OptimJL ? minimizeOptimJL : minimizeOptimizationJL)(Fs, Start, meth; kwargs...)
minimize(Fs::Tuple{Vararg{Function}}, Start::AbstractVector{<:Number}, meth; kwargs...) = minimizeOptimizationJL(Fs, Start, meth; kwargs...)

# Not economocal use of kwargs for passthrough but all options for Optim.jl listed in one place
function minimizeOptimJL(Fs::Tuple{Vararg{Function}}, Start::AbstractVector{T}, meth::Optim.AbstractOptimizer; Domain::Union{HyperCube,Nothing}=nothing, 
                Fthresh::Union{Nothing,Real}=nothing, tol::Real=1e-10, Full::Bool=false, verbose::Bool=true, maxtime::Real=600.0, time_limit::Real=maxtime,
                # catch for now:
                cons=nothing, lcons=nothing, ucons=nothing, 
                lb=(!isnothing(Domain) ? convert(Vector{T},Domain.L) : nothing), ub=(!isnothing(Domain) ? convert(Vector{T},Domain.U) : nothing),
                g_tol::Real=tol, x_tol=nothing, f_tol=nothing, x_abstol::Real=0.0, x_reltol::Real=0.0, f_abstol::Real=0.0, f_reltol::Real=0.0, g_abstol::Real=1e-8, 
                maxiters::Int=10000, iterations::Int=maxiters, callback=nothing, f_calls_limit::Int=0, allow_f_increases::Bool=true, 
                store_trace::Bool=false, show_trace::Bool=false, extended_trace::Bool=false, show_every::Int=1, 
                retry::Bool=false, retrymeth=(retry ? Optim.NelderMead() : nothing), kwargs...) where T <: Number
    @assert !retry || !isnothing(retrymeth)
    @assert 1 ≤ length(Fs) ≤ 3
    start = ConstrainStart(Start, Domain; verbose=verbose)
    length(Fs) == 3 && @assert MaximalNumberOfArguments(Fs[2]) == MaximalNumberOfArguments(Fs[3]) "Derivatives dF and ddF need to be either both in-place or both out-of-place."
    !(Fs[1](start) isa Number) && throw("Given function must return scalar values, got $(typeof(Fs[1](start))) instead.")
    
    # Construct callback for early stopping if objective function below Fthresh unless any other callback is given
    cb = isnothing(callback) ? (!isnothing(Fthresh) ? (z->z.value<Fthresh) : nothing) : callback
    options = Optim.Options(; x_tol, f_tol, g_tol, x_abstol, x_reltol, f_abstol, f_reltol, g_abstol, f_calls_limit, show_every, show_trace,
                            allow_f_increases, iterations, store_trace, extended_trace, time_limit, callback=cb)
    
    Cmeth = ConstrainMeth(meth, Domain; verbose=verbose)
    Res = if Cmeth isa Optim.AbstractConstrainedOptimizer
        start ∉ Domain && @warn "Given starting value $start not in specified domain $Domain."
        if length(Fs) == 1
            @assert !isnothing(lb) && !isnothing(ub)
            Optim.optimize(Fs[1], lb, ub, floatify(start), Cmeth, options; kwargs...)
        else
            if Cmeth isa Fminbox
                # Optim.optimize only accepts inplace kwarg for Fminbox
                Optim.optimize(Fs..., lb, ub, floatify(start), Cmeth, options; inplace=MaximalNumberOfArguments(Fs[2])>1, kwargs...)
            else
                Optim.optimize(Fs..., lb, ub, floatify(start), Cmeth, options; kwargs...)
            end
        end
    else
        if length(Fs) == 1
            Optim.optimize(Fs[1], floatify(start), Cmeth, options; kwargs...)
        else
            Optim.optimize(Fs..., floatify(start), Cmeth, options; inplace=MaximalNumberOfArguments(Fs[2])>1, kwargs...)
        end
    end
    if !Optim.converged(Res) 
        verbose && @warn "minimize(): Optimization appears to not have converged."
        if retry
            Res = InformationGeometry.minimize(Fs, Optim.minimizer(Res), retrymeth; Domain, Fthresh, tol, Full=true, verbose, maxtime, time_limit,
                cons, lcons, ucons, lb, ub, g_tol, x_tol, f_tol, x_abstol, x_reltol, f_abstol, f_reltol, g_abstol, maxiters, iterations, callback, f_calls_limit, allow_f_increases, 
                store_trace, show_trace, extended_trace, show_every, retry=false, kwargs...)
        end
    end
    Full ? Res : Optim.minimizer(Res)
end

ADtypeConverter(V::Val{true}) = Optimization.AutoForwardDiff()
ADtypeConverter(V::Val{:ForwardDiff}) = Optimization.AutoForwardDiff()
ADtypeConverter(V::Val{:ReverseDiff}) = Optimization.AutoReverseDiff()
ADtypeConverter(V::Val{:Zygote}) = Optimization.AutoZygote()
ADtypeConverter(V::Val{:FiniteDifferences}) = Optimization.AutoFiniteDifferences()
ADtypeConverter(V::Val{:FiniteDiff}) = Optimization.AutoFiniteDiff()
ADtypeConverter(V::Val{:Symbolic}) = Optimization.AutoSymbolics()
ADtypeConverter(S::Symbol) = ADtypeConverter(Val(S))

# Extend with constraint Functions
function minimizeOptimizationJL(Fs::Tuple{Vararg{Function}}, Start::AbstractVector{<:Number}, meth; ADmode::Union{Val,Symbol}=Val(:ForwardDiff), adtype::AbstractADType=ADtypeConverter(ADmode), cons=nothing, lcons=nothing, ucons=nothing, kwargs...)
    @assert 1 ≤ length(Fs) ≤ 3
    
    if !SciMLBase.allowsconstraints(meth)
        cons, lcons, ucons = nothing, nothing, nothing
    end

    optf = if length(Fs) == 1
        OptimizationFunction{true}((x,p)->Fs[1](x), adtype; cons=cons)
    else
        @warn "minimize(): Currently ignoring manually given derivatives and using adtype=$adtype instead."
        OptimizationFunction{true}((x,p)->Fs[1](x), adtype; cons=cons)
    # elseif length(Fs) == 2
    #     numarg = MaximalNumberOfArguments(Fs[2])
    #     if numarg == 1
    #         OptimizationFunction{(numarg > 1)}((x,p)->Fs[1](x), adtype; grad=(x,p)->Fs[2](x))
    #     else
    #         OptimizationFunction{(numarg > 1)}((x,p)->Fs[1](x), adtype; grad=(G,x,p)->Fs[2](G,x))
    #     end
    # else
    #     @assert MaximalNumberOfArguments(Fs[2]) == MaximalNumberOfArguments(Fs[3]) "Derivatives dF and ddF need to be either both in-place or both out-of-place."
    #     numarg = MaximalNumberOfArguments(Fs[2])
    #     if numarg == 1
    #         OptimizationFunction{(numarg > 1)}((x,p)->Fs[1](x), adtype; grad=(x,p)->Fs[2](x), hess=(x,p)->Fs[3](x))
    #     else
    #         OptimizationFunction{(numarg > 1)}((x,p)->Fs[1](x), adtype; grad=(G,x,p)->Fs[2](G,x), hess=(H,x,p)->Fs[3](H,x))
    #     end
    end
    minimizeOptimizationJL(optf, Start, meth; lcons=lcons, ucons=ucons, kwargs...)
end

# For Optimizers from the Optimization.jl ecosystem
function minimizeOptimizationJL(optf::OptimizationFunction, Start::AbstractVector{<:Number}, meth; Domain::Union{HyperCube,Nothing}=nothing, Full::Bool=false, verbose::Bool=true, 
                    tol::Real=1e-10, maxiters::Int=10000, maxtime::Real=600.0, abstol::Real=tol, reltol::Real=tol, 
                    lb=((SciMLBase.allowsbounds(meth) && !isnothing(Domain)) ? Domain.L : nothing), ub=((SciMLBase.allowsbounds(meth) && !isnothing(Domain)) ? Domain.U : nothing), lcons=nothing, ucons=nothing, 
                    Fthresh::Union{Nothing,Real}=nothing, callback=(!isnothing(Fthresh) ? (z->z.objective<Fthresh) : nothing), 
                    retry::Bool=false, retrymeth=(retry ? Optim.NelderMead() : nothing), kwargs...)
    
    @assert !retry || !isnothing(retrymeth)
    SciMLBase.requiresbounds(meth) && isnothing(lb) && (lb = fill(-Inf, length(Start)))
    SciMLBase.requiresbounds(meth) && isnothing(ub) && (ub = fill(Inf, length(Start)))
    
    prob = OptimizationProblem(optf, ConstrainStart(Start, Domain; verbose=verbose); lcons, ucons, lb=lb, ub=ub, sense=MinSense)

    sol = Optimization.solve(prob, meth; maxiters, maxtime, abstol, reltol, (!isnothing(callback) ? (;callback=callback) : (;))..., kwargs...) # callback
    if sol.retcode !== ReturnCode.Success 
        verbose && @warn "minimize(): Optimization appears to not have converged."
        if retry
            verbose && @warn "minimize(): Try to continue with NelderMead()."
            prob = OptimizationProblem(optf, ConstrainStart(sol.u, Domain; verbose=verbose); lcons, ucons, lb=lb, ub=ub, sense=MinSense)
            sol = Optimization.solve(prob, retrymeth; maxiters, maxtime, abstol, reltol, (!isnothing(callback) ? (;callback=callback) : (;))..., kwargs...)
            if sol.retcode !== ReturnCode.Success
                verbose && @warn "minimize(): Repeated Optimization with NelderMead() appears to not have converged, too."
            end
        end
    end;    Full ? sol : sol.u
end


GetDomain(DM::AbstractDataModel) = GetDomain(Predictor(DM))
GetDomain(M::ModelMap) = Domain(M)
GetDomain(F::Function) = nothing

GetInDomain(DM::AbstractDataModel) = GetInDomain(Predictor(DM))
GetInDomain(M::ModelMap) = InDomain(M)
GetInDomain(F::Function) = nothing

GetConstraintFunc(DM::AbstractDataModel, startp::AbstractVector{<:Number}=MLE(DM); kwargs...) = GetConstraintFunc(Predictor(DM), startp; kwargs...)
GetConstraintFunc(F::Function, startp::AbstractVector{<:Number}=Float64[]; kwargs...) = (nothing, nothing, nothing)
function GetConstraintFunc(M::ModelMap, startp::AbstractVector{<:Number}=GetStartP(M); inplace::Bool=true)
    if isnothing(InDomain(M))
        (nothing, nothing, nothing)
    else
        Cons = InDomain(M);     testout = Cons(startp)
        if testout isa AbstractVector
            (0.0 .* testout, testout .+ Inf, inplace ? ((y::AbstractVector,x::AbstractVector,p)-> copyto!(y,Cons(x))) : ((x::AbstractVector,p)->Cons(x)))
        elseif testout isa Number
            ([0.0], [Inf], inplace ? ((y::AbstractVector,x::AbstractVector,p)-> copyto!(y,Cons(x))) : ((x::AbstractVector,p)->[Cons(x)]))
        end
    end
end

function minimize(DM::AbstractDataModel, start::AbstractVector{<:Number}=MLE(DM); Lifted::Bool=false, Domain::Union{HyperCube,Nothing}=GetDomain(DM), meth=missing, kwargs...)
    F = (Lifted && HasXerror(DM)) ? FullLiftedNegLogLikelihood(DM) : Negloglikelihood(DM)
    # Get constraint function and Hypercube from ModelMap if available?
    Lcons, Ucons, Cons = GetConstraintFunc(DM, start; inplace=true) # isinplacemodel(DM)
    # Allow meth=nothing if no constraints to use LsqFit
    PassMeth = ((!ismissing(meth) && !isnothing(meth)) ? (; meth=meth) : (;))
    !Lifted && isnothing(meth) && isnothing(LogPrior(DM)) && isnothing(Cons) && (return curve_fit(DM, start, LogPrior(DM); Domain, kwargs...))
    isnothing(Cons) ? minimize(F, start, Domain; PassMeth..., kwargs...) : minimize(F, start, Domain; lcons=Lcons, ucons=Ucons, cons=Cons, PassMeth..., kwargs...)
end

# If DM not constructed yet
function minimize(DS::AbstractDataSet, Model::ModelOrFunction, start::AbstractVector{<:Number}, LogPriorFn::Union{Nothing,Function}; Lifted::Bool=false, Domain::Union{HyperCube,Nothing}=GetDomain(Model), meth=missing, kwargs...)
    F = (Lifted && HasXerror(DS)) ? FullLiftedNegLogLikelihood(DS,Model,LogPriorFn,length(start)) : (θ->-loglikelihood(DS,Model,θ,LogPriorFn))
    Lcons, Ucons, Cons = GetConstraintFunc(Model, start; inplace=true) # isinplacemodel(DM)
    # Allow meth=nothing if no constraints to use LsqFit
    PassMeth = ((!ismissing(meth) && !isnothing(meth)) ? (; meth=meth) : (;))
    !Lifted && isnothing(meth) && isnothing(LogPriorFn) && isnothing(Cons) && (return curve_fit(DS, Model, start, LogPriorFn; Domain, kwargs...))
    isnothing(Cons) ? minimize(F, start, Domain; PassMeth..., kwargs...) : minimize(F, start, Domain; lcons=Lcons, ucons=Ucons, cons=Cons, PassMeth..., kwargs...)
end


"""
    ParameterSavingCallback(start::AbstractVector) -> (Vector{typeof(start)}, Function)
Produces tuple where first entry constitutes empty `Vector{typeof(start)}` array into which the parameter trajectory will be saved.
Second entry is the callback function itself, which is to be added to the optimization method via the `callback` keyword argument.

# Examples
```julia
Pars, Func = ParameterSavingCallback(MLE(DM))
InformationGeometry.minimize(DM, rand(3); meth=Optim.NewtonTrustRegion(), OptimJL=false, callback=Func)
TracePlot(Pars)
```
!!! note
    Does not work for optimizers from Optim.jl unless wrapped with OptimizationOptimJL.jl, i.e. when setting keyword `OptimJL=false`.
"""
function ParameterSavingCallback(X::AbstractVector{<:Number})
    SavedParams = typeof(X)[]
    GetCurPar(S::Optim.OptimizationState) = ((@warn "Cannot access current parameters in OptimizationState for Optim.jl");    fill(Inf, length(X)))
    GetCurPar(S::Optimization.OptimizationState) = S.u
    GetCurPar(S) = throw("Got $S instead of OptimizationState.")
    SaveOptimizationpath(State, args...) = (push!(SavedParams, GetCurPar(State));   false)
    SavedParams, SaveOptimizationpath
end


GetRobustCostFunction(DM::AbstractDataModel; kwargs...) = GetRobustCostFunction(Data(DM), Predictor(DM), LogPrior(DM); kwargs...)
function GetRobustCostFunction(DS::AbstractFixedUncertaintyDataSet, M::ModelOrFunction, LogPriorFn::Union{Nothing,Function}=nothing; p::Real=1)
    HalfSig = cholesky(yInvCov(DS)).U    # Since F is minimized, need to subtract LogPrior
    pNormCostFunction(θ::AbstractVector) = norm(HalfSig * (ydata(DS) - EmbeddingMap(DS, M, θ)), p) - EvalLogPrior(LogPriorFn, θ)
end

"""
    RobustFit(DM::AbstractDataModel, start::AbstractVector{<:Number}; tol::Real=1e-10, p::Real=1, kwargs...)
Uses `p`-Norm to judge distance on Dataspace as specified by the keyword.
"""
RobustFit(DM::AbstractDataModel, start::AbstractVector{<:Number}=MLE(DM); kwargs...) = RobustFit(Data(DM), Predictor(DM), start, LogPrior(DM); kwargs...)
function RobustFit(DS::AbstractDataSet, M::ModelOrFunction, start::AbstractVector{<:Number}=GetStartP(DS,M), LogPriorFn::Union{Nothing,Function}=nothing; Domain::Union{HyperCube,Nothing}=(M isa ModelMap ? Domain(M) : nothing), tol::Real=1e-10, p::Real=1, kwargs...)
    InformationGeometry.minimize(GetRobustCostFunction(DS, M, LogPriorFn; p), start, Domain; tol=tol, kwargs...)
end
# function RobustFit(DS::AbstractDataSet, M::ModelOrFunction, dM::ModelOrFunction, start::AbstractVector{<:Number}=GetStartP(DS,M), LogPriorFn::Union{Nothing,Function}=nothing; Domain::Union{HyperCube,Nothing}=(M isa ModelMap ? Domain(M) : nothing), tol::Real=1e-10, p::Real=1, kwargs...)
#     HalfSig = cholesky(yInvCov(DS)).U
#     # Since F is minimized, need to subtract LogPrior
#     F(θ::AbstractVector) = norm(HalfSig * (EmbeddingMap(DS, M, θ) - ydata(DS)), p) - EvalLogPrior(LogPriorFn, θ)
#     function dFp(θ::AbstractVector)
#         z = HalfSig * (EmbeddingMap(DS, M, θ) - ydata(DS))
#         n = sum(z.^p)^(1/p - 1) * z.^(p-1)
#         transpose(HalfSig * EmbeddingMatrix(DS, dM, θ)) * n - EvalLogPriorGrad(LogPriorFn, θ)
#     end
#     dF1(θ::AbstractVector) = transpose(HalfSig * EmbeddingMatrix(DS, dM, θ)) *  sign.(HalfSig * (EmbeddingMap(DS, M, θ) - ydata(DS))) - EvalLogPrior(LogPriorFn, θ)
#     InformationGeometry.minimize(F, (p == 1 ? dF1 : dFp), start, Domain; tol=tol, kwargs...)
# end


"""
    IncrementalTimeSeriesFit(DM::AbstractDataModel, initial::AbstractVector{<:Number}=MLE(DM); steps::Int=length(Data(DM))÷5, Method::Function=InformationGeometry.minimize, kwargs...) -> Vector
Fits DataModel incrementally by splitting up the times series into chunks, e.g. fitting only the first quarter of data points, then half and so on.
This can yield much better fitting results from random starting points, particularly for autocorrelated time series data.
For example when the time series data oscillates in such a way that the optimization often gets stuck in a local optimum where the model fits a mostly straight line through the data, not correctly recognizing the oscillations.
"""
IncrementalTimeSeriesFit(DM::AbstractDataModel, initial::AbstractVector{<:Number}=MLE(DM); Method::Function=InformationGeometry.minimize, kwargs...) = IncrementalTimeSeriesFit(Method, DM, initial; kwargs...)

"""
    IncrementalTimeSeriesFit(Method::Function, DM::AbstractDataModel, initial::AbstractVector{<:Number}=MLE(DM); steps::Int=length(Data(DM))÷5, kwargs...) -> Vector
Uses `Method` for fitting, which should be of the form `(::DataModel, ::AbstractVector) -> ::AbstractVector`
"""
function IncrementalTimeSeriesFit(Method::Function, DM::AbstractDataModel, initial::AbstractVector{<:Number}=MLE(DM); steps::Int=length(Data(DM))÷5, kwargs...)
    @assert steps ≤ Npoints(DM)
    res = copy(initial)
    Results = typeof(initial)[]
    for chunk in vcat([1:i*(Npoints(DM)÷steps) for i in 1:steps-1], [1:Npoints(DM)])
        FullRes = Method(SubDataModel(DM, chunk), res; kwargs...)
        res = GetMinimizer(FullRes)
        push!(Results, res)
    end;    res
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
    Res = find_zero(Test, (Float64(Domain[1]), Float64(Domain[2])), meth; xatol=1e-12, xrtol=1e-12)
    find_zero(Test, (BigFloat(Res-3e-12),BigFloat(Res+3e-12)), Roots.Bisection(); xatol=tol, xrtol=tol)
end

function AltLineSearch(Test::Function, start::Real, meth::Roots.AbstractNonBracketingMethod=Roots.Order2(); tol::Real=1e-12, kwargs...)
    find_zero(Test, start, meth; xatol=tol, xrtol=tol, kwargs...)
end
function AltLineSearch(Test::Function, start::BigFloat, meth::Roots.AbstractNonBracketingMethod=Roots.Order2(); tol::Real=convert(BigFloat,exp10(-precision(BigFloat)/10)), kwargs...)
    Res = find_zero(Test, Float64(start), meth; xatol=1e-12, xrtol=1e-12)
    find_zero(Test, BigFloat(Res), meth; xatol=tol, xrtol=tol, kwargs...)
end
