

# RecipesBase.@recipe f(DM::AbstractDataModel) = DM, MLE(DM)
RecipesBase.@recipe function f(DM::AbstractDataModel, mle::AbstractVector{<:Number}=MLE(DM), xpositions::AbstractVector{<:Number}=xdata(DM))
    (xdim(DM) != 1 && Npoints(DM) > 1) && throw("Not programmed for plotting xdim != 1 yet.")
    xguide -->              (ydim(DM) > Npoints(DM) ? "Positions" : xnames(DM)[1])
    yguide -->              (ydim(DM) ==1 ? ynames(DM)[1] : "Observations")
    @series begin
        Data(DM), xpositions
    end
    markeralpha :=      0.
    linewidth -->       2
    seriescolor :=     (ydim(DM) == 1 ? get(plotattributes, :seriescolor, :red) : (:auto))
    linestyle -->       :solid
    RSEs = ResidualStandardError(DM, mle)
    RSEs = !isnothing(RSEs) ? convert.(Float64, RSEs) : RSEs
    label -->  if ydim(DM) == 1
        # "Fit with RSE≈$(RSEs[1])"
        "Fit" * (isnothing(RSEs) ? "" : " with RSE≈$(round(RSEs[1]; sigdigits=3))")
    elseif ydim(DM) ≤ Npoints(DM)
        # reshape([ynames(DM)[i] * " Fit with RSE≈$(RSEs[i])" for i in 1:ydim(DM)], 1, ydim(DM))
        reshape([ynames(DM)[i] * " Fit"*(isnothing(RSEs) ? "" : " with RSE≈$(round(RSEs[i]; sigdigits=3))")  for i in 1:ydim(DM)], 1, ydim(DM))
    else
        reshape("Fit for $(xnames(DM)[1])=" .* string.(round.(xdata(DM); sigdigits=3)), 1, length(xdata(DM)))
    end
    # ydim(DS) > Npoints(DS) && length(xpositions) != ydim(DS)
    X = ydim(DM) ≤ Npoints(DM) ? DomainSamples(extrema(xdata(DM)); N=500) : xdata(DM)
    Y = predictedY(DM, mle, X)
    # Y = EmbeddingMap(Data(DM), Predictor(DM), mle, X)
    # Y = ydim(DM) == 1 ? Y : (ydim(DM) ≤ Npoints(DM) ? Unpack(Windup(Y, ydim(DM))) : transpose(Unpack(Windup(Y, ydim(DM)))))
    if ydim(DM) ≤ Npoints(DM)
        return X, Y
    elseif xpositions != xdata(DM)
        return xpositions, Y
    else
        return Y
    end
end
function predictedY(DM::AbstractDataModel, mle::AbstractVector{<:Number}=MLE(DM), X::AbstractVector=xdata(DM))
    predictedY(Data(DM), Predictor(DM), mle, X)
end
function predictedY(DS::AbstractDataSet, model::ModelOrFunction, mle::AbstractVector{<:Number}, X::AbstractVector=xdata(DS))
    # Ignore structure of missing values for CompositeDataSet for dense prediction curve
    Y = DS isa CompositeDataSet ? EmbeddingMap(Val(true), model, mle, X) : EmbeddingMap(DS, model, mle, X)
    ydim(DS) == 1 ? Y : (ydim(DS) ≤ Npoints(DS) ? Unpack(Windup(Y, ydim(DS))) : transpose(Unpack(Windup(Y, ydim(DS)))))
end

function PlotFit(DM::AbstractDataModel, mle::AbstractVector=MLE(DM), X::Union{AbstractVector,Nothing}=nothing; N::Int=500, kwargs...)
    X = ydim(DM) ≤ Npoints(DM) ? DomainSamples(extrema(xdata(DM)); N=N) : xdata(DM)
    RecipesBase.plot!(X, predictedY(DM, mle, X); label="Fit", kwargs...)
end

# xpositions for PDE Datasets
RecipesBase.@recipe function f(DS::AbstractDataSet, xpositions::AbstractVector{<:Number}=xdata(DS))
    xdim(DS) != 1 && throw("Not programmed for plotting xdim != 1 yet.")
    Σ_y = typeof(ysigma(DS)) <: AbstractVector ? ysigma(DS) : sqrt.(Diagonal(ysigma(DS)).diag)
    Σ_x = DS isa DataSetExact ? (xsigma(DS) isa AbstractVector ? xsigma(DS) : sqrt.(Diagonal(xsigma(DS)).diag)) : nothing
    line -->                (:scatter, 0.8)
    xguide -->              (ydim(DS) > Npoints(DS) ? "Positions" : xnames(DS)[1])
    yguide -->              (ydim(DS) == 1 ? ynames(DS)[1] : "Observations")
    seriescolor := :auto
    if ydim(DS) == 1
        label --> "Data"
        yerror --> Σ_y
        xerror --> Σ_x
    elseif ydim(DS) ≤ Npoints(DS)       # Series per y-component
        label --> reshape("Data: " .* ynames(DS), 1, ydim(DS))
        yerror --> Unpack(Windup(Σ_y, ydim(DS)))
        xerror --> Σ_x
    else                                # Series per x-component
        # Use of xdata instead of xpositions deliberate here!
        label --> reshape("Data for $(xnames(DS)[1])=" .* string.(round.(xdata(DS); sigdigits=3)), 1, length(xdata(DS)))
        yerror --> transpose(Unpack(Windup(Σ_y, ydim(DS))))
        # No way to incorporate errors in xpositions here....
    end
    Y = if ydim(DS) == 1
        ydata(DS)
    elseif ydim(DS) ≤ Npoints(DS)
        Unpack(Windup(ydata(DS), ydim(DS)))
    else
        Unpack(Windup(ydata(DS), ydim(DS))) |> transpose
    end
    if ydim(DS) > Npoints(DS) && length(xpositions) != ydim(DS)
        # Discard xpositions if xpositions == xdata but ydim > Npoints
        return Y
    else
        return xpositions, Y
    end
end


RecipesBase.@recipe function f(DSs::AbstractVector{<:AbstractDataSet})
    layout := length(DSs)
    for x in DSs
        @series begin x end
    end
end


RecipesBase.@recipe function f(LU::HyperCube)
    length(LU.U) != 2 && throw("Cube not Planar, cannot Plot Box.")
    rectangle(LU)[:,1], rectangle(LU)[:,2]
end

RecipesBase.@recipe function f(X::AbstractVector{<:AbstractVector{<:Number}})
    marker --> :hex
    linealpha --> 0
    markersize --> 1.8
    markerstrokewidth --> 0.5
    ToCols(Unpack(X))
end

"""
    Rsquared(DM::DataModel) -> Real
Calculates the R² value associated with the maximum likelihood estimate of a `DataModel`. It should be noted that the R² value is only a valid measure for the goodness of a fit for linear relationships.
"""
function Rsquared(DM::DataModel, mle::AbstractVector{<:Number}=MLE(DM))
    !(xdim(DM) == ydim(DM) == 1) && return -1
    mean = sum(ydata(DM)) / length(ydata(DM))
    Stot = sum(abs2, ydata(DM) .- mean)
    Sres = sum(abs2, ydata(DM) - EmbeddingMap(DM,mle))
    1 - Sres / Stot
end



ResidualStandardError(DM::AbstractDataModel, mle::AbstractVector{<:Number}=MLE(DM)) = ResidualStandardError(Data(DM), Predictor(DM), mle)
function ResidualStandardError(DS::AbstractDataSet, model::ModelOrFunction, MLE::AbstractVector{<:Number}; verbose::Bool=true)
    Npoints(DS) ≤ length(MLE) && ((verbose && @warn "Too few data points to compute RSE"); return nothing)
    ydiff = ydata(DS) - EmbeddingMap(DS, model, MLE)
    Res = map(i->sqrt(sum(abs2, view(ydiff, i:ydim(DS):length(ydiff))) / (Npoints(DS) - length(MLE))), 1:ydim(DS))
    ydim(DS) == 1 ? Res[1] : Res
end

TotalRSE(DS::AbstractDataSet, model::ModelOrFunction, MLE::AbstractVector{<:Number}) = norm(ResidualStandardError(DS, model, MLE))


@deprecate FittedPlot(args...) RecipesBase.plot(args...)

ResidualPlot(DM::AbstractDataModel; kwargs...) = ResidualPlot(Data(DM), Predictor(DM), MLE(DM); kwargs...)
function ResidualPlot(DS::DataSet, model::ModelOrFunction, mle::AbstractVector{<:Number}; kwargs...)
    RecipesBase.plot(DataModel(DataSet(xdata(DS), ydata(DS)-EmbeddingMap(DS,model,mle), ysigma(DS), dims(DS)),
                ((x,p)->(ydim(DS) == 1 ? 0.0 : zeros(ydim(DS)))),
                (x,p)->zeros(ydim(DS), length(mle)),
                mle, _loglikelihood(DS, model, mle), true); kwargs...)
end
function ResidualPlot(DS::DataSetExact, model::ModelOrFunction, mle::AbstractVector{<:Number}; kwargs...)
    RecipesBase.plot(DataModel(DataSetExact(xdata(DS), xsigma(DS), ydata(DS)-EmbeddingMap(DS,model,mle), ysigma(DS), dims(DS)),
                ((x,p)->(ydim(DS) == 1 ? 0.0 : zeros(ydim(DS)))),
                (x,p)->zeros(ydim(DS), length(mle)),
                mle, _loglikelihood(DS, model, mle), true); kwargs...)
end




function _PrepareEllipsePlot(pos::AbstractVector{<:Number}, cov::AbstractMatrix{<:Number}; N::Int=100)
    @assert length(pos) == size(cov,1) == size(cov,2)
    C = cholesky(collect(cov)).U
    if length(pos) == 2
        ran = range(0, 2π; length=N)
        M = Unpack([pos + C*[cos(α),sin(α)] for α in ran])
        M[:,1], M[:,2]
    elseif length(pos) == 3
        θran = range(0, π; length=N);  ϕran = range(0, 2π; length=N)
        # M = mapreduce(x->transpose(pos + C * x), vcat, eachrow([cos.(ϕran).*sin.(θran) sin.(ϕran).*sin.(θran) cos.(θran)]))
        M = Unpack([pos + C*[cos(ϕ)*sin(θ),sin(ϕ)*sin(θ),cos(θ)] for θ in θran for ϕ in ϕran])
        M[:,1], M[:,2], M[:,3]
    else
        throw("Cannot plot Ellipses for dim > 3.")
    end
end


function PlotEllipse(pos::AbstractVector{<:Number}, cov::AbstractMatrix{<:Number}; OverWrite::Bool=false, N::Int=100, c=:blue, kwargs...)
    @assert length(pos) == size(cov,1) == size(cov,2)
    @assert 2 ≤ length(pos) ≤ 3
    F = OverWrite ? RecipesBase.plot : RecipesBase.plot!
    F([pos]; marker=:hex, linealpha=0, markersize=1.5, markeralpha=1, c=c, label="")
    M = _PrepareEllipsePlot(pos, cov; N=N)
    RecipesBase.plot!(M...; label="", fillalpha=0.2, lw=0, seriestype=:shape, c=c, kwargs...)
end



function ToEllipsoidTuples(DS::AbstractDataSet)
    X, Σ = if xdim(DS) > 0
        vcat(xdata(DS), ydata(DS)), BlockMatrix(HealthyCovariance(xsigma(DS)), HealthyCovariance(ysigma(DS)))
    elseif xdim(DS) == 0 && ydim(DS) > 1
        throw("Not programmed yet.")
    else
        throw("Error.")
    end
    ToEllipsoidTuples(X, Σ, dims(DS))
end

function ToEllipsoidTuples(X::AbstractVector, Σ::AbstractMatrix)
    @assert length(X) % 2 == 0
    @info "ToEllipsoidTuples: No dims given, assuming xdim = ydim = 1"
    ToEllipsoidTuples(X, Σ, (length(X)÷2, 1, 1))
end

"""
Expects [xdata;ydata] as format for X (and Σ).
"""
function ToEllipsoidTuples(X::AbstractVector, Σ::AbstractMatrix, dims::Tuple{Int,Int,Int})
    @assert length(X) == size(Σ,1) == size(Σ,2) == dims[1]*(dims[2] + dims[3])
    Xit = Iterators.partition(1:dims[1]*dims[2], dims[2])
    Yit = Iterators.partition(dims[1]*dims[2]+1:dims[1]*(dims[2] + dims[3]), dims[3])
    function _Sub(X,Σ,tup)
        inds = vcat(tup[1],tup[2])
        view(X,inds), view(Σ, inds, inds)
    end
    [_Sub(X,Σ,tup) for tup in Iterators.zip(Xit,Yit)]
end


"""
Returns [xdata;ydata] as format for X (and Σ).
"""
function FromEllipsoidTuples(M::AbstractVector{<:Tuple{AbstractVector{<:Number}, AbstractMatrix{<:Number}}})
    @info "FromEllipsoidTuples: No dims given, assuming xdim = 1, ydim = $(length(M[1][1])-1)"
    FromEllipsoidTuples(M, (length(M), 1, length(M[1][1])-1))
end

function FromEllipsoidTuples(M::AbstractVector{<:Tuple{AbstractVector{<:Number}, AbstractMatrix{<:Number}}}, dims::Tuple{Int,Int,Int})
    @assert length(M) == dims[1] && ConsistentElDims(getindex.(M,1)) == dims[2] + dims[3]
    X = vcat(mapreduce(x->x[1][1:dims[2]], vcat, M), mapreduce(x->x[1][1+dims[2]:end], vcat, M))
    Σ = zeros(dims[1]*(dims[2] + dims[3]), dims[1]*(dims[2] + dims[3]))
    for (i,tup) in enumerate(Iterators.zip(Iterators.partition(1:dims[1]*dims[2], dims[2]),Iterators.partition(dims[1]*dims[2]+1:dims[1]*(dims[2] + dims[3]), dims[3])))
        inds = vcat(tup[1],tup[2])
        view(Σ,inds,inds) .= M[i][2]
    end;    X, HealthyCovariance(Σ)
end

PlotEllipses(GDS::GeneralizedDataSet; kwargs...) = PlotEllipses(dist(GDS), dims(GDS); kwargs...)
PlotEllipses(dist::ContinuousMultivariateDistribution, dims::Tuple{Int,Int,Int}; kwargs...) = PlotEllipses(GetMean(dist), Sigma(dist), dims; kwargs...)
function PlotEllipses(X::AbstractVector, Σ::AbstractMatrix, dims::Tuple{Int,Int,Int}; OverWrite::Bool=true, c=:blue, kwargs...)
    @assert length(X) == size(Σ,1) == size(Σ,2) == dims[1]*(dims[2] + dims[3])
    @assert 2 ≤ dims[2] + dims[3] ≤ 3
    p = []
    OverWrite && RecipesBase.plot()
    Ellipses = ToEllipsoidTuples(X,Σ,dims)
    for (x, σ) in Ellipses
        p = PlotEllipse(x, σ; OverWrite=false, c=c, kwargs...)
    end;    display(p)
end


RecipesBase.@recipe function f(GDS::GeneralizedDataSet, xpositions::AbstractVector{<:Number}=xdata(GDS))
    @assert xdim(GDS) + ydim(GDS) == 2
    names = vcat(xnames(GDS),ynames(GDS))
    xguide --> names[1]
    yguide --> names[2]
    N = xdim(GDS) + ydim(GDS) == 2 ? 200 : 80
    for (x,σ) in ToEllipsoidTuples(GetMean(dist(GDS)),Sigma(dist(GDS)),dims(GDS))
        A = cholesky(collect(σ)).U * [cos.(range(0, 2π; length=N))'; sin.(range(0, 2π; length=N))']
        @series begin
            seriestype := :shape
            fillalpha --> 0.3
            fillcolor --> :blue
            label := ""
            x[1] .+ A[1,:], x[2] .+ A[2,:]
        end
    end
end



meshgrid(x, y) = (repeat(x, outer=length(y)), repeat(y, inner=length(x)))

"""
    PlotScalar(F::Function, PlanarCube::HyperCube; N::Int=100, Save::Bool=false, parallel::Bool=false, nlevels::Int=40, kwargs...)
Plots a scalar function `F` over the 2D domain `PlanarCube` by `N^2` evaluations on a regular grid.
"""
function PlotScalar(F::Function, PlanarCube::HyperCube; N::Int=100, Save::Bool=false, parallel::Bool=false, OverWrite::Bool=true, nlevels::Int=40, kwargs...)
    length(PlanarCube) != 2 && throw(ArgumentError("Cube not Planar."))
    Lims = PlanarCube;    A = range(Lims.L[1], Lims.U[1], length=N);    B = range(Lims.L[2], Lims.U[2], length=N)
    func(args...) = F([args...])
    if Save
        X,Y = meshgrid(A, B)
        Z = (parallel ? pmap : map)(func, X, Y)
        (OverWrite ? RecipesBase.plot : RecipesBase.plot!)(X, Y, Z; seriestype=:contour, fill=true, nlevels=nlevels, kwargs...) |> display
        return [X Y Z]
    else
        (OverWrite ? RecipesBase.plot : RecipesBase.plot!)(A, B, func; seriestype=:contour, fill=true, nlevels=nlevels, kwargs...)
    end
end

"""
    PlotScalar(F::Function, PlotPlane::Plane, PlanarCube::HyperCube; N::Int=100, Save::Bool=false, parallel::Bool=false, nlevels::Int=40, kwargs...)
Plots a scalar function `F` by evaluating the given `PlotPlane` over the 2D domain `PlanarCube` by `N^2` evaluations on a regular grid.
"""
function PlotScalar(F::Function, PlotPlane::Plane, PlanarCube::HyperCube; N::Int=100, Save::Bool=true, parallel::Bool=false, nlevels::Int=40, kwargs...)
    length(PlanarCube) != 2 && throw(ArgumentError("Cube not Planar."))
    Lims = PlanarCube;    A = range(Lims.L[1], Lims.U[1], length=N);    B = range(Lims.L[2], Lims.U[2], length=N)
    Lcomp(x,y) = F(PlaneCoordinates(PlotPlane,[x,y]))
    if Save
        X,Y = meshgrid(A,B)
        Z = (parallel ? pmap : map)(Lcomp,X,Y)
        p = RecipesBase.plot(X, Y, Z; seriestype=:contour, fill=true, leg=false, nlevels=nlevels, title="Plot centered around: $(round.(PlotPlane.stütz,sigdigits=4))",
            xlabel="$(PlotPlane.Vx) direction", ylabel="$(PlotPlane.Vy) direction", kwargs...)
        p = RecipesBase.plot!([0],[0]; seriestype=:scatter, lab="Center", marker=:hex)
        display(p)
        return [X Y Z]
    else
        p = RecipesBase.plot(A, B, Lcomp; seriestype=:contour, fill=true, leg=false, nlevels=nlevels, title="Plot centered around: $(round.(PlotPlane.stütz,sigdigits=4))",
            xlabel="$(PlotPlane.Vx) direction", ylabel="$(PlotPlane.Vy) direction", kwargs...)
        p = RecipesBase.plot!([0],[0]; seriestype=:scatter, lab="Center", marker=:hex)
        return p
    end
end

"""
    PlotLogLikelihood(DM::AbstractDataModel, PlanarCube::HyperCube; N::Int=100, Save::Bool=true, parallel::Bool=false, nlevels::Int=40, kwargs...)
    PlotLogLikelihood(DM::AbstractDataModel, PlotPlane::Plane, PlanarCube::HyperCube; N::Int=100, Save::Bool=true, parallel::Bool=false, nlevels::Int=40, kwargs...)
Evaluates the loglikelihood on a 2D domain via the `PlotScalar()` function.
"""
PlotLogLikelihood(DM::AbstractDataModel, args...; kwargs...) = PlotScalar(loglikelihood(DM), args...; kwargs...)



function ConstructBox(fit::LsqFit.LsqFitResult, Confnum::Real; AxisCS::Bool=true)
    E = Confnum * stderror(fit)
    HyperCube(fit.param - E, fit.param + E)
end


rectangle(ax,ay,bx,by) = [ax ay; bx ay; bx by; ax by; ax ay]
function rectangle(LU::HyperCube)
    length(LU.L) != 2 && throw(ArgumentError("Cube not Planar."))
    rectangle((LU.L)...,(LU.U)...)
end


"""
    VFRescale(ZeilenVecs::Array{<:Number,2},C::HyperCube;scaling=0.85)
Rescale vector to look good in 2D plot.
"""
function VFRescale(ZeilenVecs::Array{<:Number,2},C::HyperCube;scaling=0.85)
    VecsPerLine = sqrt(size(ZeilenVecs)[1])
    SpacePerVec = (scaling/VecsPerLine) * CubeWidths(C)
    for i in 1:size(ZeilenVecs)[1]
        if SpacePerVec[1] < abs(ZeilenVecs[i,1])
            rescale = SpacePerVec[1] / abs(ZeilenVecs[i,1])
            ZeilenVecs[i,:] .*= rescale
        end
        if SpacePerVec[2] < abs(ZeilenVecs[i,2])
            rescale = SpacePerVec[2] / abs(ZeilenVecs[i,2])
            ZeilenVecs[i,:] .*= rescale
        end
    end
    ZeilenVecs[:,1],ZeilenVecs[:,2]
end


function Plot2DVF(V::Function, Lims::HyperCube; N::Int=25, scaling::Float64=0.85, OverWrite::Bool=false, kwargs...)
    @assert length(Lims) == length(V(Center(Lims))) == 2
    AV, BV = meshgrid(range(Lims.L[1], Lims.U[1]; length=N), range(Lims.L[2], Lims.U[2]; length=N))
    Vcomp(a,b) = V([a,b])
    u, v = VFRescale(Unpack(Vcomp.(AV,BV)), Lims; scaling=scaling)
    (OverWrite ? RecipesBase.plot : RecipesBase.plot!)(AV, BV; seriestype=:quiver, quiver=(u,v), kwargs...) |> display
    [AV BV u v]
end



"""
    Deplanarize(PL::Plane,sol::AbstractODESolution; N::Int=500) -> Matrix
    Deplanarize(PL::Plane,sol::AbstractODESolution, Ts::AbstractVector{<:Number}) -> Matrix
Converts the 2D outputs of `sol` from planar coordinates associated with `PL` to the coordinates of the ambient space of `PL`.
"""
Deplanarize(PL::Plane, sol::AbstractODESolution; N::Int=500) = Deplanarize(PL, sol, range(sol.t[1],sol.t[end];length=N))
Deplanarize(PL::Plane, sol::AbstractODESolution, Ts::AbstractVector{<:Number}) = map(t->PlaneCoordinates(PL,sol(t)),Ts) |> Unpack

"""
    VisualizeSols(sols::AbstractVector{<:AbstractODESolution}; OverWrite::Bool=true)
Visualizes vectors of type `ODESolution` using the `Plots.jl` package. If `OverWrite=false`, the solution is displayed on top of the previous plot object.
"""
function VisualizeSols(sols::AbstractVector{<:AbstractODESolution}; vars::Tuple=Tuple(1:length(sols[1].u[1])), OverWrite::Bool=true, leg::Bool=false, kwargs...)
    p = [];     OverWrite && RecipesBase.plot()
    for sol in sols
        p = VisualizeSols(sol; vars=vars, leg=leg, kwargs...)
    end;    p
end
function VisualizeSols(sol::AbstractODESolution; vars::Tuple=Tuple(1:length(sol.u[1])), leg::Bool=false, OverWrite::Bool=false,
                                        ModelMapMeta::Union{ModelMap,Bool}=false, kwargs...)
    OverWrite && RecipesBase.plot()
    if ModelMapMeta isa ModelMap
        names = pnames(ModelMapMeta)
        if length(vars) == 2
            return RecipesBase.plot!(sol; xlabel=names[vars[1]], ylabel=names[vars[2]], vars=vars, leg=leg, kwargs...)
        elseif length(vars) == 3
            return RecipesBase.plot!(sol; xlabel=names[vars[1]], ylabel=names[vars[2]], zlabel=names[vars[3]], vars=vars, leg=leg, kwargs...)
        end
        # What if vars > 3? Does Plots.jl throw an error?
    end
    RecipesBase.plot!(sol; vars=vars, leg=leg, kwargs...)
end

function VisualizeSols(PL::Plane, sol::AbstractODESolution; vars::Tuple=Tuple(1:length(PL)), leg::Bool=false, N::Int=500,
                            ModelMapMeta::Union{ModelMap,Bool}=false, kwargs...)
    H = Deplanarize(PL, sol; N=N)
    if ModelMapMeta isa ModelMap
        names = pnames(ModelMapMeta)
        return RecipesBase.plot!(H[:,vars[1]], H[:,vars[2]], H[:,vars[3]]; xlabel=names[vars[1]], ylabel=names[vars[2]], zlabel=names[vars[3]], leg=leg, kwargs...)
    else
        return RecipesBase.plot!(H[:,vars[1]], H[:,vars[2]], H[:,vars[3]]; leg=leg, kwargs...)
    end
end
function VisualizeSols(PL::Plane, sols::AbstractVector{<:AbstractODESolution}; vars::Tuple=Tuple(1:length(PL)), N::Int=500, OverWrite::Bool=true, leg::Bool=false, kwargs...)
    p = [];     OverWrite && RecipesBase.plot()
    for sol in sols
        p = VisualizeSols(PL, sol; N=N, vars=vars, leg=leg, kwargs...)
    end;    p
end

VisualizeSols(X::Tuple, args...; kwargs...) = VisualizeSols(X..., args...; kwargs...)
function VisualizeSols(PL::AbstractVector{<:Plane},sols::AbstractVector{<:AbstractODESolution}; vars::Tuple=Tuple(1:length(PL[1])), N::Int=500,
                OverWrite::Bool=true,leg::Bool=false, color=rand([:red,:blue,:green,:orange,:grey]), kwargs...)
    length(PL) != length(sols) && throw("VisualizeSols: Must receive same number of Planes and Solutions.")
    p = [];     OverWrite && RecipesBase.plot()
    for i in 1:length(sols)
        p = VisualizeSols(PL[i], sols[i]; N=N, vars=vars, leg=leg, color=color, kwargs...)
    end;    p
end
function VisualizeSols(DM::AbstractDataModel, args...; OverWrite::Bool=true, kwargs...)
    (OverWrite ? RecipesBase.plot : RecipesBase.plot!)([MLE(DM)]; seriestype=:scatter, label="MLE")
    if Predictor(DM) isa ModelMap
        VisualizeSols(args...; OverWrite=false, ModelMapMeta=Predictor(DM), kwargs...)
    else
        VisualizeSols(args...; OverWrite=false, kwargs...)
    end
end


function VisualizeSols(CB::ConfidenceBoundary; vars::Tuple=Tuple(1:length(CB.MLE)), OverWrite::Bool=true, color=rand([:red,:blue,:green,:orange,:grey]), kwargs...)

    p = OverWrite ? RecipesBase.plot([CB.MLE]; seriestype=:scatter, label="MLE") : []
    if length(vars) == 2
        VisualizeSols(CB.sols[1]; vars=vars, color=color, OverWrite=false, label="$(round(CB.Confnum, sigdigits=3))σ Conf. Boundary",
                        xlabel=CB.pnames[vars[1]], ylabel=CB.pnames[vars[2]], leg=true, kwargs...)
    else
        VisualizeSols(CB.sols[1]; vars=vars, color=color, OverWrite=false, label="$(round(CB.Confnum, sigdigits=3))σ Conf. Boundary",
                        xlabel=CB.pnames[vars[1]], ylabel=CB.pnames[vars[2]], zlabel=CB.pnames[vars[3]],leg=true, kwargs...)
    end
    for sol in CB.sols[2:end]
        p = VisualizeSols(sol; vars=vars, color=color, OverWrite=false, label="", leg=true, kwargs...)
    end; p
end

function VisualizeSols(CBs::AbstractVector{<:ConfidenceBoundary}; vars::Tuple=Tuple(1:length(CBs[1].MLE)), OverWrite::Bool=true, kwargs...)
    @assert all(x->x.MLE==CBs[1].MLE, CBs)
    @assert allunique(map(x->x.Confnum,CBs))
    p = OverWrite ? RecipesBase.plot([CBs[1].MLE]; seriestype=:scatter, label="MLE") : []
    for CB in CBs
        VisualizeSols(CB; vars=vars, OverWrite=false, kwargs...)
    end; p
end

VisualizeGeos(sol::AbstractODESolution; kwargs...) = VisualizeGeos([sol]; kwargs...)
function VisualizeGeos(sols::AbstractVector{<:AbstractODESolution}; OverWrite::Bool=false, leg::Bool=false, kwargs...)
    VisualizeSols(sols; vars=Tuple(1:Int(length(sols[1].u[1])/2)), OverWrite=OverWrite, leg=leg, kwargs...)
end


function VisualizeSolPoints(sol::AbstractODESolution; kwargs...)
    RecipesBase.plot!(sol.u; marker=:hex, line=:dash, linealpha=1, markersize=2, kwargs...)
    VisualizeSols(sol; OverWrite=false, kwargs...)
end
function VisualizeSolPoints(sols::AbstractVector{<:AbstractODESolution}; OverWrite::Bool=false, kwargs...)
    p = [];     OverWrite && RecipesBase.plot()
    for sol in sols
        p = VisualizeSolPoints(sol; kwargs...)
    end;    p
end



XCube(DS::AbstractDataSet; Padding::Number=0.) = ConstructCube(Unpack(WoundX(DS)); Padding=Padding)
XCube(DM::AbstractDataModel; Padding::Number=0.) = XCube(Data(DM); Padding=Padding)
Grid(Cube::HyperCube, N::Int=5) = [range(Cube.L[i], Cube.U[i]; length=N) for i in 1:length(Cube)]


# function GetExtrema(DM::AbstractDataModel,sols::Union{AbstractODESolution,AbstractVector{<:AbstractODESolution}},X::Union{<:Number,AbstractVector{<:Number}}; N::Int=200)
#     low = Inf;   up = -Inf
#     for sol in sols
#         Y = map(Z->DM.model(X,sol(Z)), range(sol.t[1],sol.t[end]; length=N))
#         templow = minimum(Y);   tempup = maximum(Y)
#         if templow < low    low = templow       end
#         if up < tempup      up = tempup         end
#     end;    (low, up)
# end
# function GetExtrema(DM::AbstractDataModel,PL::Plane,sols::Union{AbstractODESolution,AbstractVector{<:AbstractODESolution}},X::Union{<:Number,AbstractVector{<:Number}}; N::Int=200)
#     low = Inf;   up = -Inf
#     for sol in sols
#         templow, tempup = map(t->DM.model(X,PlaneCoordinates(PL,sol(t))), range(sol.t[1],sol.t[end]; length=N)) |> extrema
#         if templow < low    low = templow       end
#         if up < tempup      up = tempup         end
#     end;    (low, up)
# end
# function GetExtrema(DM::AbstractDataModel,PL::AbstractVector{<:Plane},sols::AbstractVector{<:AbstractODESolution},X::Union{<:Number,AbstractVector{<:Number}}; N::Int=200)
#     length(PL) != length(sols) && throw("Dimensional Mismatch.")
#     low = Inf;   up = -Inf
#     for i in 1:length(sols)
#         templow, tempup = GetExtrema(DM,PL[i],sols[i],X; N=N)
#         if templow < low    low = templow       end
#         if up < tempup      up = tempup         end
#     end;    (low, up)
# end
#
# """
#     ConfidenceBands(DM::DataModel,sol::AbstractODESolution,domain::HyperCube; N::Int=200)
# Given a confidence interval `sol`, the pointwise confidence band around the model prediction is computed for x values in `domain` by evaluating the model on the boundary of the confidence region.
# """
# function ConfidenceBands(DM::AbstractDataModel,sols::Union{AbstractODESolution,AbstractVector{<:AbstractODESolution}},domain::HyperCube=XCube(DM); N::Int=200, Np::Int=200)
#     !(length(domain) == xdim(DM)) && throw("Dimensionality of domain inconsistent with xdim.")
#     low = Vector{Float64}(undef,N^xdim(DM));     up = Vector{Float64}(undef,N^xdim(DM))
#     X = xdim(DM) == 1 ? range(domain.L[1],domain.U[1]; length=N) : Iterators.product(Grid(domain,N)...)
#     for i in 1:length(X)
#         low[i], up[i] = GetExtrema(DM,sols,X[i]; N=Np)
#     end
#     col = rand([:red,:blue,:green,:orange,:grey])
#     RecipesBase.plot!(X,low,color=col,label="Lower Conf. Band")
#     RecipesBase.plot!(X,up,color=col,label="Upper Conf. Band") |> display
#     return [Unpack(collect(X)) low up]
# end
#
# function ConfidenceBands(DM::AbstractDataModel,Planes::Union{Plane,AbstractVector{<:Plane}},sols::Union{AbstractODESolution,AbstractVector{<:AbstractODESolution}},
#                                             domain::HyperCube=XCube(DM); N::Int=200, Np::Int=300)
#     length(domain) != xdim(DM) && throw("Dimensionality of domain inconsistent with xdim.")
#     !(length(Planes) == 1 || length(Planes) == length(sols)) && throw("Number of Planes inconsisten with number of ODESolutions.")
#     low = Vector{Float64}(undef,N^xdim(DM));     up = Vector{Float64}(undef,N^xdim(DM))
#     X = xdim(DM) == 1 ? range(domain.L[1],domain.U[1]; length=N) : Iterators.product(Grid(domain,N)...)
#     for i in 1:length(X)
#         low[i], up[i] = GetExtrema(DM,Planes,sols,X[i]; N=Np)
#     end
#     col = rand([:red,:blue,:green,:orange,:grey])
#     RecipesBase.plot!(X,low,color=col,label="Lower Conf. Band")
#     RecipesBase.plot!(X,up,color=col,label="Upper Conf. Band") |> display
#     return [Unpack(collect(X)) low up]
# end


function PlotConfidenceBands(DM::AbstractDataModel, M::AbstractMatrix{<:Number}, xpositions::Union{AbstractVector{<:Number},Nothing}=nothing;
                                        Confnum::Real=-1)
    lab = 0 < Confnum ? "$(Confnum)σ " : ""
    if size(M,2) == 3
        RecipesBase.plot!(view(M,:,1), view(M,:,2:3); label=[lab*"Conf. Band" ""], color=rand([:red,:blue,:green,:orange,:grey])) |> display
    else # Assume the FittedPlot splits every y-component into a separate series of points and have same number of rows as x-values
        @assert xdim(DM) == 1 && size(M,2) == 1 + 2ydim(DM)
        if xpositions isa Nothing
            for i in 1:(size(M,1)-1)
                RecipesBase.plot!([M[i,2:2:end] M[i,3:2:end]]; color=rand([:red,:blue,:green,:orange,:grey]), label=["" ""])
            end
            RecipesBase.plot!([M[end,2:2:end] M[end,3:2:end]]; color=rand([:red,:blue,:green,:orange,:grey]), label=["" lab*"Conf. Band"]) |> display
        elseif length(xpositions) == (size(M,2)-1) / 2
            for i in 1:(size(M,1)-1)
                RecipesBase.plot!(xpositions, [M[i,2:2:end] M[i,3:2:end]]; color=rand([:red,:blue,:green,:orange,:grey]), label=["" ""])
            end
            RecipesBase.plot!(xpositions, [M[end,2:2:end] M[end,3:2:end]]; color=rand([:red,:blue,:green,:orange,:grey]), label=["" lab*"Conf. Band"]) |> display
        else
            throw("Vector of xpositions wrong length.")
        end
    end
end

function ConfidenceBands(DM::AbstractDataModel, Confnum::Real, Xdomain::HyperCube=XCube(DM); N::Int=300, plot::Bool=true, samples::Int=200, kwargs...)
    ConfidenceBands(DM, ConfidenceRegion(DM,Confnum; kwargs...), Xdomain; N=N, plot=plot, samples=samples)
end

"""
    ConfidenceBands(DM::DataModel, sol::AbstractODESolution, Xdomain::HyperCube; N::Int=300, plot::Bool=true) -> Matrix
Given a confidence interval `sol`, the pointwise confidence band around the model prediction is computed for x values in `Xdomain`
by evaluating the model on the boundary of the confidence region.
"""
function ConfidenceBands(DM::AbstractDataModel, sols::Union{AbstractODESolution,AbstractVector{<:AbstractODESolution}}, Xdomain::HyperCube=XCube(DM);
                            N::Int=300, plot::Bool=true, samples::Int=max(2*length(sols),100), kwargs...)
    ConfidenceBands(DM, sols, DomainSamples(Xdomain; N=N); plot=plot, samples=samples, kwargs...)
end

function ConfidenceBands(DM::AbstractDataModel, Tup::Tuple{<:AbstractVector{<:Plane},AbstractVector{<:AbstractODESolution}}, woundX=XCube(DM);
                            N::Int=300, plot::Bool=true, samples::Int=max(2*length(Tup[1]),100), kwargs...)
    ConfidenceBands(DM, Tup[1], Tup[2], woundX; plot=plot, samples=samples, kwargs...)
end
function ConfidenceBands(DM::AbstractDataModel, Planes::AbstractVector{<:Plane}, sols::AbstractVector{<:AbstractODESolution}, Xdomain::HyperCube=XCube(DM);
                            N::Int=300, plot::Bool=true, samples::Int=max(2*length(sols),100), kwargs...)
    ConfidenceBands(DM, Planes, sols, DomainSamples(Xdomain; N=N); plot=plot, samples=samples, kwargs...)
end

function ConfidenceBands(DM::AbstractDataModel, sols::Union{AbstractODESolution,AbstractVector{<:AbstractODESolution}}, woundX::AbstractVector{<:Number};
                            plot::Bool=true, samples::Int=max(2*length(sols),100))
    Res = Array{Float64,2}(undef, length(woundX), 2*ydim(DM))
    for col in 1:2:2ydim(DM)    fill!(view(Res,:,col), Inf);     fill!(view(Res,:,col+1), -Inf)    end
    # gradually refine Res for each solution to avoid having to allocate a huge list of points
    for sol in sols
        _ConfidenceBands!(Res, DM, map(sol, range(sol.t[1], sol.t[end]; length=samples)), woundX)
    end
    M = hcat(woundX, Res)
    if plot
        Confnum = round(GetConfnum(DM, sols[1]); sigdigits=2)
        PlotConfidenceBands(DM, M; Confnum=Confnum)
    end;    M
end

function ConfidenceBands(DM::AbstractDataModel, Planes::AbstractVector{<:Plane}, sols::AbstractVector{<:AbstractODESolution}, woundX::AbstractVector{<:Number};
                            plot::Bool=true, samples::Int=max(2*length(sols),100))
    @assert length(Planes) == length(sols)
    Res = Array{Float64,2}(undef, length(woundX), 2*ydim(DM))
    for col in 1:2:2ydim(DM)    fill!(view(Res,:,col), Inf);     fill!(view(Res,:,col+1), -Inf)    end
    # gradually refine Res for each solution to avoid having to allocate a huge list of points
    for i in 1:length(sols)
        _ConfidenceBands!(Res, DM, map(x->PlaneCoordinates(Planes[i], sols[i](x)), range(sols[i].t[1], sols[i].t[end]; length=samples)), woundX)
    end
    M = hcat(woundX, Res)
    if plot
        Confnum = round(GetConfnum(DM, PlaneCoordinates(Planes[1], sols[1].u[1])); sigdigits=2)
        PlotConfidenceBands(DM, M; Confnum=Confnum)
    end;    M
end


"""
ApproxConfidenceBands(DM::AbstractDataModel, Confnum::Real, Xdomain=XCube(DM); N::Int=300, plot::Bool=true, add::Real=1.5)
"""
function ApproxConfidenceBands(DM::AbstractDataModel, Confnum::Real, Xdomain=XCube(DM); N::Int=300, plot::Bool=true, add::Real=1.5)
    higherConfnum = Confnum + add
    @warn "Trying to establish HyperCube which circumscribes confidence region by passing $higherConfnum to ProfileLikelihood(). If this errors, establish a suitable box manually."
    Box = ProfileBox(DM, InterpolatedProfiles(ProfileLikelihood(DM, higherConfnum; plot=false)), Confnum)
    ApproxConfidenceBands(DM, Box, Xdomain; N=N, plot=plot)
end

"""
    ApproxConfidenceBands(DM::AbstractDataModel, ParameterCube::HyperCube, Xdomain=XCube(DM); N::Int=300, plot::Bool=true)
Computes confidence bands associated with the face centers of the `ParameterCube`.
If the `ParameterCube` circumscribes a given confidence region, this will typically result in a gross and asymmetric overestimation of the true pointwise confidence bands associated with this confidence level.
"""
function ApproxConfidenceBands(DM::AbstractDataModel, ParameterCube::HyperCube, Xdomain::HyperCube=XCube(DM); N::Int=300, plot::Bool=true)
    length(Xdomain) != xdim(DM) && throw("Dimensionality of domain inconsistent with xdim.")
    ApproxConfidenceBands(DM, ParameterCube, DomainSamples(Xdomain; N=N); plot=plot)
end

function ApproxConfidenceBands(DM::AbstractDataModel, ParameterCube::HyperCube, woundX::AbstractVector{<:Number}; plot::Bool=true)
    ConfidenceBands(DM, FaceCenters(ParameterCube), woundX; plot=plot)
end

# Devise version with woundX::AbstractVector{<:AbstractVector{<:Number}} for xdim > 1
function ConfidenceBands(DM::AbstractDataModel, points::AbstractVector{<:AbstractVector{<:Number}}, woundX::AbstractVector{<:Number}; plot::Bool=true)
    xdim(DM) != 1 && throw("Not programmed for xdim != 1 yet.")
    Res = Array{Float64,2}(undef, length(woundX), 2*ydim(DM))
    for col in 1:2:2ydim(DM)    fill!(view(Res,:,col), Inf);     fill!(view(Res,:,col+1), -Inf)    end
    _ConfidenceBands!(Res, DM, points, woundX)
    M = hcat(woundX, Res)
    if plot
        Confnum = round(GetConfnum(DM, points[1]); sigdigits=2)
        PlotConfidenceBands(DM, M; Confnum=Confnum)
    end;    M
end

# Does the computations
function _ConfidenceBands!(Res::AbstractMatrix, DM::AbstractDataModel, points::AbstractVector{<:AbstractVector{<:Number}}, woundX::AbstractVector{<:Number})
    @assert pdim(DM) == ConsistentElDims(points)
    xdim(DM) != 1 && throw("Not programmed for xdim != 1 yet.")
    @assert size(Res) == (length(woundX), 2ydim(DM))
    # BELOW LOOP WILL NOT WORK FOR CompositeDataSet!!!!!!
    @assert Data(DM) isa DataSet || Data(DM) isa DataSetExact
    for point in points
        # Do it like this to exploit CustomEmbeddings
        Y = Windup(EmbeddingMap(Data(DM), Predictor(DM), point, woundX), ydim(DM)) |> Unpack
        for col in 1:2:2ydim(DM)
            Ycol = Int(ceil(col/2))
            for row in 1:size(Res,1)
                Res[row, col] = min(Res[row, col], Y[row,Ycol])
                Res[row, col+1] = max(Res[row, col+1], Y[row,Ycol])
            end
        end
    end
end

"""
Computes width of confidence bands.
"""
function ConfidenceBandWidth(args...; plot::Bool=true, OverWrite::Bool=true, kwargs...)
    band = ConfidenceBands(args...; plot=false, kwargs...)
    Res = hcat(view(band,:,1), view(band,:,3)-view(band,:,2))
    F = OverWrite ? RecipesBase.plot : RecipesBase.plot!
    plot && display(F(view(Res,:,1), view(Res,:,2), label="Conf. band width"))
    Res
end

"""
    PredictionEnsemble(DM::AbstractDataModel, pDomain::HyperCube, Xs::AbstractVector{<:Number}=DomainSamples(XCube(DM),300); N::Int=50, uniform::Bool=true, MaxConfnum::Real=3, plot::Bool=true, kwargs...)
Plots `N` model predictions which are randomly chosen from a confidence region of level `MaxConfnum`.
"""
function PredictionEnsemble(DM::AbstractDataModel, pDomain::HyperCube, Xs::AbstractVector{<:Number}=DomainSamples(XCube(DM),300); N::Int=50, uniform::Bool=true, MaxConfnum::Real=3, plot::Bool=true, kwargs...)
    @assert xdim(DM) == 1
    function GenerateUniformPoint(DM::AbstractDataModel, pDomain::HyperCube, MaxConfnum::Real)
        p = SVector{length(pDomain)}(rand(pDomain));        Conf = GetConfnum(DM, p)
        Conf ≤ MaxConfnum ? (p, Conf) : GenerateUniformPoint(DM, pDomain, MaxConfnum)
    end
    iF = inv(FisherMetric(DM, MLE(DM))) |> Symmetric
    function GenerateGaussianPoint(DM::AbstractDataModel, MaxConfnum::Real)
        p = rand(MvNormal(MLE(DM), iF));    Conf = GetConfnum(DM, p)
        Conf ≤ MaxConfnum ? (p, Conf) : GenerateGaussianPoint(DM, MaxConfnum)
    end
    function MakePrediction(DM::AbstractDataModel, pDomain::HyperCube, Xs::AbstractVector{<:Number}; MaxConfnum::Real=3)
        point, Conf = uniform ? GenerateUniformPoint(DM, pDomain, MaxConfnum) : GenerateGaussianPoint(DM, MaxConfnum)
        MakePrediction(DM, point, Xs), Conf
    end
    function MakePrediction(DM::AbstractDataModel, point::AbstractVector{<:Number}, Xs::AbstractVector{<:Number})
        Unpack(Windup(EmbeddingMap(Data(DM), Predictor(DM), point, Xs), ydim(DM)))
    end
    Preds = [MakePrediction(DM, pDomain, Xs; MaxConfnum=MaxConfnum) for i in 1:N]
    plot && display(PlotEnsemble(Xs, getindex.(Preds,1), getindex.(Preds,2); kwargs...))
    Xs, getindex.(Preds,1), getindex.(Preds,2)
end
PredictionEnsemble(DM::AbstractDataModel; MaxConfnum::Real=3, kwargs...) = PredictionEnsemble(DM, ProfileBox(DM,MaxConfnum); MaxConfnum=MaxConfnum, kwargs...)

function PlotEnsemble(Xs::AbstractVector{<:Number}, Preds::Union{AbstractVector{<:AbstractArray{<:Number}}}, Confs::AbstractVector{<:Real}; palette::Symbol=:Oranges_5, OverWrite::Bool=false, kwargs...)
    extr = extrema(Confs);    p = OverWrite ? RecipesBase.plot() : RecipesBase.plot!()
    for i in eachindex(Confs)
        p = RecipesBase.plot!(Xs, Preds[i]; color=get(cgrad(palette), Confs[i], extr), label="", alpha=0.3, kwargs...)
    end;    p
end


PointwiseConfidenceBandFULL(DM::DataModel,sol::AbstractODESolution,Cube::HyperCube,Confnum::Real=1; N::Int=500) = PointwiseConfidenceBandFULL(DM,sol,FindMLE(DM),Cube,Confnum; N=N)
function PointwiseConfidenceBandFULL(DM::DataModel,sol::AbstractODESolution,MLE::AbstractVector,Cube::HyperCube,Confnum::Real=1; N::Int=500)
    !(length(Cube) == xdim(DM)) && throw("PWConfBand: Wrong Cube dim.")
    if ydim(DM) == 1
        Lims = ConstructCube(sol)
        low = Vector{Float64}(undef,N); up = Vector{Float64}(undef,N)
        X = DomainSamples(Cube; N=N)
        for i in 1:length(X)
            Y = Predictor(DM)(X[i],MLE)
            up[i] = maximum(Y); low[i] = minimum(Y)
        end
        LogLikeMLE = loglikelihood(DM,MLE)
        Confvol = ConfVol(Confnum)
        for i in 1:N
            num = rand.(Uniform.(Lims.L,Lims.U))
            if WilksTestPrepared(DM,num,LogLikeMLE,Confvol)
                for i in 1:length(X)
                    Y = Predictor(DM)(X[i],num)
                    Y > up[i] && (up[i] = Y)
                    Y < low[i] && (low[i] = Y)
                end
            else i = i-1 end
        end
        RecipesBase.plot!(X,low)
        RecipesBase.plot!(X,up) |> display
        return [X low up]
    else
        throw("Not programmed yet.")
        # Evaluate on boundary of cube
    end
end

"""
    PlotMatrix(Mat::AbstractMatrix, mle::AbstractVector; Confnum::Real=0., dims::Tuple{Int,Int}=(1,2), N::Int=400, plot::Bool=true, OverWrite::Bool=true, kwargs...)
Plots ellipse corresponding to a given covariance matrix which may additionally be offset by a vector `mle`.
By providing information on the confidence level of the given matrix via the `Confnum` kwarg, a correction factor is computed to rescale the given matrix appropriately.

Example:
```
PlotMatrix(inv(FisherMetric(DM,mle)),mle)
```
"""
function PlotMatrix(Mat::AbstractMatrix, MLE::AbstractVector{<:Number}=zeros(size(Mat,1)); Confnum::Real=0., dims::Tuple{Int,Int}=(1,2), N::Int=400, plot::Bool=true, OverWrite::Bool=true, kwargs...)
    !(length(MLE) == size(Mat,1) == size(Mat,2)) && throw("PlotMatrix: Dimensional mismatch.")
    corr = Confnum != 0. ? sqrt(quantile(Chisq(length(MLE)),ConfVol(Confnum))) : 1.0
    C = corr .* cholesky(Symmetric(Mat)).L
    angles = range(0, 2π; length=N)
    F(α::Number) = MLE + C * RotatedVector(α, dims[1], dims[2], length(MLE))
    Data = Unpack(F.(angles))
    plot && display((OverWrite ? RecipesBase.plot : RecipesBase.plot!)(ToCols(Data)...; label="Matrix", kwargs...))
    Data
end



function PlotCurves(Curves::AbstractVector{<:AbstractODESolution}; N::Int=100)
    p = [];    A = Array{Float64,2}(undef,N,2)
    for sol in Curves
        ran = range(sol.t[1],sol.t[end],length=N)
        for i in 1:length(ran)    A[i,:] = sol(ran[i])[1:2]  end
        p = RecipesBase.plot!(A[:,1],A[:,2])
        # p = RecipesBase.plot!(sol,vars=(1,2))
    end
    p
end

EvaluateAlongGeodesic(F::Function, sol::AbstractODESolution, Interval::Tuple{<:Number,<:Number}=(sol.t[1],sol.t[end]); N::Int=300) = [F(sol(t)[1:Int(length(sol.u[1])/2)]) for t in range(Interval[1],Interval[2],length=N)]
function PlotAlongGeodesic(F::Function,sol::AbstractODESolution, Interval::Tuple{<:Number,<:Number}=(sol.t[1],sol.t[end]); N::Int=300, OverWrite::Bool=false)
    Z = EvaluateAlongGeodesic(F, sol, Interval; N=N)
    @assert ConsistentElDims(Z) == 1
    X = DomainSamples(Interval; N=N)
    display((OverWrite ? RecipesBase.plot : RecipesBase.plot!)(X,Z))
    [X Z]
end
EvaluateAlongGeodesicLength(DM::AbstractDataModel, F::Function, sol::AbstractODESolution, Interval::Tuple{<:Number,<:Number}=(sol.t[1],sol.t[end]); N::Int=300) = EvaluateAlongGeodesic(F,sol,Interval, N=N)
function PlotAlongGeodesicLength(DM::AbstractDataModel, F::Function, sol::AbstractODESolution, Interval::Tuple{<:Number,<:Number}=(sol.t[1],sol.t[end]); N::Int=300, OverWrite::Bool=false)
    Z = EvaluateAlongGeodesic(F, sol, Interval; N=N)
    @assert ConsistentElDims(Z) == 1
    X = DomainSamples(Interval; N=N)
    Geo = GeodesicLength(x->FisherMetric(DM,x), sol, sol.t[end]; FullSol=true, tol=1e-14)
    Ls = map(Geo, X)
    display((OverWrite ? RecipesBase.plot : RecipesBase.plot!)(Ls, Z))
    [Ls Z]
end
EvaluateAlongCurve(F::Function, sol::AbstractODESolution, Interval::Tuple{<:Number,<:Number}=(sol.t[1],sol.t[end]); N::Int=300) = [F(sol(t)) for t in range(Interval[1],Interval[2],length=N)]
function PlotAlongCurve(F::Function, sol::AbstractODESolution, Interval::Tuple{<:Number,<:Number}=(sol.t[1],sol.t[end]); N::Int=300, OverWrite::Bool=false)
    Z = EvaluateAlongCurve(F, sol, Interval, N=N)
    @assert ConsistentElDims(Z) == 1
    X = DomainSamples(Interval; N=N)
    display((OverWrite ? RecipesBase.plot : RecipesBase.plot!)(X, Z))
    [X Z]
end

PhaseSpacePlot(DM::AbstractDataModel; kwargs...) = PhaseSpacePlot(DM, (C=InformationGeometry.XCube(DM); range(C.L[1], C.U[1]; length=300)); kwargs...)
function PhaseSpacePlot(DM::AbstractDataModel, ts::AbstractVector{<:Number}, mle::AbstractVector{<:Number}=MLE(DM); OverWrite::Bool=true, kwargs...)
    OverWrite && RecipesBase.plot()
    p = RecipesBase.plot!(collect(Iterators.partition(ydata(DM),ydim(DM))); seriestype=:scatter, label="Observed Data")
    p = if ydim(DM) == 2
        RecipesBase.plot!(collect(Iterators.partition(EmbeddingMap(DM, mle, ts),ydim(DM))); label="Phase Space Trajectory", xlabel="$(ynames(DM)[1])", ylabel="$(ynames(DM)[2])", markeralpha=0, linealpha=1, kwargs...)
    elseif ydim(DM) == 3
        RecipesBase.plot!(collect(Iterators.partition(EmbeddingMap(DM, mle, ts),ydim(DM))); label="Phase Space Trajectory", xlabel="$(ynames(DM)[1])", ylabel="$(ynames(DM)[2])", zlabel="$(ynames(DM)[3])", markeralpha=0, linealpha=1, kwargs...)
    else
        throw("Cannot display phase space for ydim=$(ydim(DM)).")
    end;    display(p);    p
end


# Helper methods for creating a rectangular or triangular mesh from the outputs of MincedBoundaries()


# function RectangularFaceIndices(N::Int=20, zerobased::Bool=false)
#     G = Matrix{Int64}(undef,N,4)
#     for i in 1:(N-1)
#         G[i,:] = SA[i, i+1, N+i+1, N+i]
#     end;    G[N,:] = SA[N, 1, N+1, 2N]
#     zerobased && (G .-= 1)
#     G
# end
"""
Returns Array of vertex numbers where every row constitutes a trapezoid for two adjacent curves from which N samples have been drawn.
"""
RectangularFaceIndices(N::Int, zerobased::Bool=false) = (G=Matrix{Int64}(undef,N,4); RectangularFaceIndices!(G, zerobased); G)
function RectangularFaceIndices!(G::AbstractMatrix{<:Int}, zerobased::Bool=false)
    N, cols = size(G);    @assert cols == 4
    @inbounds for i in Base.OneTo(N-1)
        G[i,:] = SA[i, i+1, N+i+1, N+i]
    end;    G[N,:] = SA[N, 1, N+1, 2N]
    zerobased && (G .-= 1)
    nothing
end

"""
Turns Array which specifies rectangular faces into triangular faces.
"""
function RectToTriangFaces(M::AbstractMatrix{<:Int})
    G = Matrix{Int64}(undef, 2size(M,1), 3)
    @inbounds for i in Base.OneTo(size(M,1))
        G[2i-1,:], G[2i,:] = SA[M[i,1], M[i,2], M[i,4]], SA[M[i,2], M[i,3], M[i,4]]
    end;    G
end
"""
    CreateMesh(Planes::AbstractVector{<:Plane}, Sols::AbstractVector{<:AbstractODESolution}; N::Int=2*length(Sols), rectangular::Bool=true) -> (Matrix{Float64}, Matrix{Int64})
Returns a N×3 matrix whose rows correspond to the coordinates of various points in 3D space as the first argument.
The second Matrix is either N×4 or N×3 depending on the value of `rectangular` and enumerates the points which are to be connected up to a rectangular or triangular face in counter-clockwise fashion. The indices of the points correspond to the lines in the first Matrix.
"""
function CreateMesh(Planes::AbstractVector{<:Plane}, Sols::AbstractVector{<:AbstractODESolution}, fact::Real=1.4; N::Int=2*length(Sols), rectangular::Bool=true, pointy::Bool=true)
    Vertices = reduce(vcat, [Deplanarize(Planes[i], Sols[i]; N=N) for i in 1:length(Sols)])
    M = RectangularFaceIndices(N)
    Faces = reduce(vcat, [M .+ (i-1)*N for i in 1:(length(Sols)-1)])
    !rectangular && (Faces = RectToTriangFaces(Faces))
    pointy && return AddCaps(Planes, Sols, Vertices, Faces, fact; N=N)
    Vertices, Faces
end
function AddCaps(Planes::AbstractVector{<:Plane}, Sols::AbstractVector{<:AbstractODESolution}, Vertices::AbstractMatrix, Faces::AbstractMatrix, fact::Real=1.4; N::Int=2*length(Sols))
    @assert length(Planes[1]) == 3 # Can relax this if cross() is extended to higher dims
    @assert fact ≥ 1
    linep1 = size(Vertices,1) + 1
    function rowmean(PL::Plane, sol::AbstractODESolution, N::Int)
        PlaneCoordinates(PL, sum(sol(t) for t in range(sol.t[1], sol.t[end]; length=N))/N)
    end
    # add two points on the top and bottom of the confidence region
    n = cross(Planes[1].Vx, Planes[1].Vy) |> normalize
    height = dot(n, Planes[1].stütz - Planes[2].stütz)
    p1 = rowmean(Planes[1], Sols[1], N) + (fact-1)*height*n
    p2 = rowmean(Planes[end], Sols[end], N) - (fact-1)*height*n
    # p1 = (fact-1) * (Planes[1].stütz - Planes[2].stütz) + Planes[1].stütz
    # p2 = (fact-1) * (Planes[end].stütz - Planes[end-1].stütz) + Planes[end].stütz

    connectp1, connectp2 = if size(Faces,2) == 4
        vcat(reduce(vcat, [[i i+1 linep1 linep1] for i in 1:(N-1)]), [N 1 linep1 linep1]),
        vcat(reduce(vcat, [[i i+1 linep1+1 linep1+1] for i in ((length(Sols)-1)*N + 1):(linep1-2)]), [linep1-1 ((length(Sols)-1)*N+1) linep1+1 linep1+1])
    else
        vcat(reduce(vcat, [[i i+1 linep1] for i in 1:(N-1)]), [N 1 linep1]),
        vcat(reduce(vcat, [[i i+1 linep1+1] for i in ((length(Sols)-1)*N + 1):(linep1-2)]), [linep1-1 ((length(Sols)-1)*N+1) linep1+1])
    end
    vcat(Vertices, vcat(transpose(p1), transpose(p2))), vcat(Faces, connectp1, connectp2)
end

function ToObj(Vertices::AbstractMatrix, Faces::AbstractMatrix)
    text = ""
    for i in 1:size(Vertices,1)
        text *= "v"
        for j in 1:size(Vertices,2)
            text *= " $(Vertices[i,j])"
        end
        text *= '\n'
    end
    # ONE-BASED facet indices for .obj files
    for i in 1:size(Faces,1)
        text *= "f"
        for j in 1:size(Faces,2)
            text *= " $(Faces[i,j])"
        end
        text *= '\n'
    end;    text
end
function WriteObj(Vertices::AbstractMatrix, Faces::AbstractMatrix, path::String="D:/Boundary.obj")
    open(path,"w") do f
        write(f,ToObj(Vertices,Faces))
    end;    return nothing
end


"""
    PolarSolution(Planes::AbstractVector{<:Plane}, sols::AbstractVector{<:AbstractODESolution}) -> Function
Returns spherical function `f: [0,π]×[0,2π] ⟶ R^3` `(θ, ϕ) ⟼ [x(θ, ϕ), y(θ, ϕ), z(θ, ϕ)]` which returns closest point on confidence boundary corresponding to the given angular coordinates.
"""
function PolarSolution(Planes::AbstractVector{<:Plane}, sols::AbstractVector{<:AbstractODESolution})
    @assert 3 == length(Planes[1])
    len = length(Planes)
    PolarToInd(θ::Real, n::Int=len) = ceil(Int, θ/π * (n-1)) + 1
    function SurfacePoint(θ::Real, ϕ::Real)
        @assert 0 ≤ θ ≤ π "θ not in range."
        ind = PolarToInd(θ)
        SurfacePoint(Planes[ind], sols[ind], ϕ)
    end
    function SurfacePoint(PL::Plane, sol::AbstractODESolution, ϕ::Real)
        # @assert 0 ≤ ϕ ≤ 2π "ϕ not in range."
        PlaneCoordinates(PL, sol(sol.t[end] * ϕ / (2π)))
    end
end


function AddCaps(X::M, Y::M, Z::M, fact::Real=1.4) where M <: AbstractMatrix{<:Number}
    # thetas varies along rows, phis varies along columns
    @assert size(X) == size(Y) == size(Z)
    @assert fact ≥ 1
    function rowmean(X::M, Y::M, Z::M, i::Int) where M <: AbstractMatrix{<:Number}
        point = zeros(3)
        for j in 1:size(X,2)
            point += SA[X[i,j], Y[i,j], Z[i,j]]
        end;    point / size(X,2)
    end
    # cap1 = (z = rowmean(X,Y,Z,2);   z + fact*(rowmean(X,Y,Z,1) - z))
    # cap2 = (z = rowmean(X,Y,Z, size(X,1)-1);   z + fact*(rowmean(X,Y,Z, size(X,1)) - z))
    n = (i=1; normalize(cross([X[i,3], Y[i,3], Z[i,3]]-[X[i,2], Y[i,2], Z[i,2]], [X[i,1], Y[i,1], Z[i,1]]-[X[i,2], Y[i,2], Z[i,2]])))
    height = dot(n, [X[1,1], Y[1,1], Z[1,1]]-[X[2,1], Y[2,1], Z[2,1]])
    cap1 = rowmean(X, Y, Z, 1) + (fact-1)*height*n
    cap2 = rowmean(X, Y, Z, size(X,1)) - (fact-1)*height*n
    vcat(transpose(cap1[1]*ones(size(X,2))), X, transpose(cap2[1]*ones(size(X,2)))),
    vcat(transpose(cap1[2]*ones(size(X,2))), Y, transpose(cap2[2]*ones(size(X,2)))),
    vcat(transpose(cap1[3]*ones(size(X,2))), Z, transpose(cap2[3]*ones(size(X,2))))
end

"""
    ToMatrices(Planes::AbstractVector{<:Plane}, sols::AbstractVector{<:AbstractODESolution}; N::Int=2*length(sols)) -> Tuple{Matrix,Matrix,Matrix}
Returns tuple of three `length(Planes) × N` matrices `X, Y, Z` for plotting of confidence boundary surfaces in 3D with tools such as Makie.
"""
function ToMatrices(Planes::AbstractVector{<:Plane}, sols::AbstractVector{<:AbstractODESolution}, fact::Real=1.4; N::Int=2*length(sols), pointy::Bool=true)
    θs = range(0, π; length=size(Planes,1));    ϕs = range(0, 2π; length=N);    P = PolarSolution(Planes, sols)
    X = Array{Float64}(undef, length(θs), length(ϕs));    Y = Array{Float64}(undef, length(θs), length(ϕs));    Z = Array{Float64}(undef, length(θs), length(ϕs))
    for i in 1:length(θs)
        for j in 1:length(ϕs)
            # X[i,j], Y[i,j], Z[i,j] = P(θs[i], ϕs[j])
            X[i,j], Y[i,j], Z[i,j] = P(Planes[i], sols[i], ϕs[j])
        end
    end
    pointy ? AddCaps(X, Y, Z, fact) : (X, Y, Z)
end
