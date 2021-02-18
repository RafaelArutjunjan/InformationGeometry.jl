

RecipesBase.@recipe f(DM::AbstractDataModel) = DM, MLE(DM)

RecipesBase.@recipe function f(DM::AbstractDataModel, MLE::AbstractVector{<:Number})
    (xdim(DM) != 1 && Npoints(DM) > 1) && throw("Not programmed for plotting xdim != 1 yet.")
    # legendtitle --> "RSE ≈ $(round(ResidualStandardError(DM, MLE), sigdigits=3))"
    xguide -->              xnames(DM)[1]
    yguide -->              (ydim(DM) ==1 ? ynames(DM)[1] : "Observations")
    @series begin
        Data(DM)
    end
    markeralpha :=      0.
    linewidth -->       2
    if ydim(DM) == 1
        seriescolor -->     :red
        linestyle -->       :solid
    end
    label -->   if ydim(DM) == 1
        "Fit with RSE≈$(round(ResidualStandardError(DM, MLE), sigdigits=3))"
    else
        RSEs = round.(ResidualStandardError(DM, MLE), sigdigits=3)
        reshape([ynames(DM)[i] * " Fit with RSE≈$(RSEs[i])" for i in 1:ydim(DM)], 1, ydim(DM))
    end
    Xbounds = extrema(xdata(DM))
    X = range(Xbounds[1], Xbounds[2]; length=500) |> collect
    Y = EmbeddingMap(Data(DM), Predictor(DM), MLE, X)
    X, (ydim(DM) ==1 ? Y : Unpack(Windup(Y, ydim(DM))))
end

RecipesBase.@recipe function f(DS::DataSet)
    xdim(DS) != 1 && throw("Not programmed for plotting xdim != 1 yet.")
    Σ_y = typeof(sigma(DS)) <: AbstractVector ? sigma(DS) : sqrt.(Diagonal(sigma(DS)).diag)
    line -->                (:scatter, 1)
    xguide -->              xnames(DS)[1]
    yguide -->              (ydim(DS) ==1 ? ynames(DS)[1] : "Observations")
    label -->               (ydim(DS) ==1 ? "Data" : reshape("Data: " .* ynames(DS), 1, ydim(DS)))
    yerror -->              (ydim(DS) ==1 ? Σ_y : Unpack(Windup(Σ_y, ydim(DS))))
    if ydim(DS) == 1
        linecolor   -->         :blue
        markercolor -->         :blue
        markerstrokecolor -->   :blue
    end
    xdata(DS), (ydim(DS) ==1 ? ydata(DS) : Unpack(Windup(ydata(DS), ydim(DS))))
end

RecipesBase.@recipe function f(DS::DataSetExact)
    xdim(DS) != 1 && throw("Not programmed for plotting xdim != 1 yet.")
    xdist(DS) isa InformationGeometry.Dirac && return DataSet(xdata(DS), ydata(DS), ysigma(DS), dims(DS))
    Σ_x = typeof(xsigma(DS)) <: AbstractVector ? xsigma(DS) : sqrt.(Diagonal(xsigma(DS)).diag)
    Σ_y = typeof(ysigma(DS)) <: AbstractVector ? ysigma(DS) : sqrt.(Diagonal(ysigma(DS)).diag)
    line -->                (:scatter, 1)
    xguide -->              xnames(DS)[1]
    yguide -->              (ydim(DS) ==1 ? ynames(DS)[1] : "Observations")
    label -->               (ydim(DS) ==1 ? "Data" : reshape("Data: " .* ynames(DS), 1, ydim(DS)))
    yerror -->              (ydim(DS) ==1 ? Σ_y : Unpack(Windup(Σ_y, ydim(DS))))
    xerror -->              Σ_x
    if ydim(DS) == 1
        linecolor   -->         :blue
        markercolor -->         :blue
        markerstrokecolor -->   :blue
    end
    xdata(DS), (ydim(DS) ==1 ? ydata(DS) : Unpack(Windup(ydata(DS), ydim(DS))))
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
function Rsquared(DM::DataModel, MLE::AbstractVector{<:Number})
    !(xdim(DM) == ydim(DM) == 1) && return -1
    mean = sum(ydata(DM)) / length(ydata(DM))
    Stot = sum(abs2, ydata(DM) .- mean)
    Sres = sum(abs2, ydata(DM) - EmbeddingMap(DM,MLE))
    1 - Sres / Stot
end
Rsquared(DM::DataModel) = Rsquared(DM, MLE(DM))



ResidualStandardError(DM::AbstractDataModel) = ResidualStandardError(DM, MLE(DM))
ResidualStandardError(DM::AbstractDataModel, MLE::AbstractVector{<:Number}) = ResidualStandardError(Data(DM), Predictor(DM), MLE)
function ResidualStandardError(DS::AbstractDataSet, model::ModelOrFunction, MLE::AbstractVector{<:Number})
    @assert Npoints(DS) > length(MLE)
    ydiff = ydata(DS) - EmbeddingMap(DS, model, MLE)
    Res = map(i->sqrt(sum(abs2, view(ydiff, i:ydim(DS):length(ydiff))) / (Npoints(DS) - length(MLE))), 1:ydim(DS))
    ydim(DS) == 1 ? Res[1] : Res
end

TotalRSE(DS::AbstractDataSet, model::ModelOrFunction, MLE::AbstractVector{<:Number}) = norm(ResidualStandardError(DS, model, MLE))



FittedPlot(DM::AbstractDataModel, args...; kwargs...) = Plots.plot(DM, args...; kwargs...)

ResidualPlot(DM::AbstractDataModel; kwargs...) = ResidualPlot(Data(DM), Predictor(DM), MLE(DM); kwargs...)
function ResidualPlot(DS::DataSet, model::ModelOrFunction, mle::AbstractVector{<:Number}; kwargs...)
    Plots.plot(DataModel(DataSet(xdata(DS), ydata(DS)-EmbeddingMap(DS,model,mle), sigma(DS), dims(DS)), (x,p)->0., mle, true); kwargs...)
end
function ResidualPlot(DS::DataSetExact, model::ModelOrFunction, mle::AbstractVector{<:Number}; kwargs...)
    Plots.plot(DataModel(DataSetExact(xdata(DS), xsigma(DS), ydata(DS)-EmbeddingMap(DS,model,mle), ysigma(DS), dims(DS)), (x,p)->0., mle, true); kwargs...)
end
# function ResidualPlot(DM::AbstractDataModel; kwargs...)
#     !(xdim(DS) == ydim(DS) == 1) && throw("Not programmed for plotting xdim != 1 or ydim != 1 yet.")
#     Plots.plot(DataSetExact(xdata(DM),resid,ysigma(DM));kwargs...)
#     Plots.plot!(x->0,[xdata(DM)[1],xdata(DM)[end]],label="Fit")
#     Plots.plot!(legendtitle="R² ≈ $(round(Rsquared(DM),sigdigits=3))")
# end

meshgrid(x, y) = (repeat(x, outer=length(y)), repeat(y, inner=length(x)))


function PlotScalar(F::Function, PlanarCube::HyperCube; N::Int = 100, Save::Bool = false, parallel::Bool=false)
    length(PlanarCube) != 2 && throw(ArgumentError("Cube not Planar."))
    Lims = PlanarCube;    A = range(Lims.L[1], Lims.U[1], length=N);    B = range(Lims.L[2], Lims.U[2], length=N)
    func(args...) = F([args...])
    Map = parallel ? pmap : map
    if Save
        X,Y = meshgrid(A,B)
        Z = Map(func,X,Y)
        p = contour(X,Y,Z, fill=true, nlevels=40)
        display(p)
        return [X Y Z]
    else
        p = contour(A,B,func, fill=true, nlevels=40)
        return p
    end
end

function PlotScalar(F::Function, PlotPlane::Plane, PlanarCube::HyperCube, N::Int=100; Save::Bool=true, parallel::Bool=false)
    length(PlanarCube) != 2 && throw(ArgumentError("Cube not Planar."))
    Lims = PlanarCube;    A = range(Lims.L[1], Lims.U[1], length=N);    B = range(Lims.L[2], Lims.U[2], length=N)
    Lcomp(x,y) = F(PlaneCoordinates(PlotPlane,[x,y]))
    Map = parallel ? pmap : map
    if Save
        X,Y = meshgrid(A,B)
        Z = Map(Lcomp,X,Y)
        p = contour(X,Y,Z, fill=true, leg=false, nlevels=40, title="Plot centered around: $(round.(PlotPlane.stütz,sigdigits=4))",
            xlabel="$(PlotPlane.Vx) direction", ylabel="$(PlotPlane.Vy) direction")
        p = scatter!([0],[0],lab="Center", marker=:hex)
        display(p)
        return [X Y Z]
    else
        p = contour(A,B,Lcomp, fill=true, leg=false, nlevels=40, title="Plot centered around: $(round.(PlotPlane.stütz,sigdigits=4))",
            xlabel="$(PlotPlane.Vx) direction", ylabel="$(PlotPlane.Vy) direction")
        p = scatter!([0],[0],lab="Center", marker=:hex)
        return p
    end
end



############################# Plotting
function PlotLoglikelihood(DM::DataModel, MLE::AbstractVector, PlanarCube::HyperCube, N::Int=100; Save::Bool=true, parallel::Bool=false)
    length(MLE) !=2 && throw(ArgumentError("Only 2D supported."))
    length(PlanarCube) != 2 && throw(ArgumentError("Cube not Planar."))
    Lcomp(args...) = loglikelihood(DM,[args...])
    Lims = TranslateCube(PlanarCube,MLE)
    A = range(Lims.L[1], Lims.U[1], length=N);  B = range(Lims.L[2], Lims.U[2], length=N)
    Map = parallel ? pmap : map
    if Save
        X,Y = meshgrid(A,B)
        Z = Map(Lcomp,X,Y)
        p = contour(X,Y,Z, fill=true, leg=false, nlevels=40)
        p = scatter!([MLE[1]],[MLE[2]], lab="MLE: [$(round(MLE[1],sigdigits=4)),$(round(MLE[2],sigdigits=4))]", marker=:hex)
        display(p)
        return [X Y Z]
    else
        p = contour(A,B,Lcomp, fill=true, leg=false, nlevels=40)
        p = scatter!([MLE[1]],[MLE[2]], lab="MLE: [$(round(MLE[1],sigdigits=4)),$(round(MLE[2],sigdigits=4))]", marker=:hex)
        return p
    end
end

PlotLoglikelihood(DM::DataModel,MLE::AbstractVector,size::Float64=0.5,N::Int=100) = PlotLoglikelihood(DM, MLE, HyperCube([[-size,size],[-size,size]]), N)

function PlotLoglikelihood(DM::DataModel, PlotPlane::Plane, PlanarCube::HyperCube, N::Int=100; Save::Bool=true, parallel::Bool=false)
    Lcomp(x,y) = loglikelihood(DM,PlaneCoordinates(PlotPlane,[x,y]))
    length(PlanarCube) != 2 && throw(ArgumentError("Cube not Planar."))
    Lims = PlanarCube
    A = range(Lims.L[1], Lims.U[1], length=N)
    B = range(Lims.L[2], Lims.U[2], length=N)
    Map = parallel ? pmap : map
    if Save
        X,Y = meshgrid(A,B)
        Z = Map(Lcomp,X,Y)
        p = contour(X,Y,Z, fill=true, leg=false, nlevels=40, title="Plot centered around: $(round.(PlotPlane.stütz,sigdigits=4))",
            xlabel="$(PlotPlane.Vx) direction", ylabel="$(PlotPlane.Vy) direction")
        p = scatter!([0],[0],lab="Center", marker=:hex)
        display(p)
        return [X Y Z]
    else
        p = contour(A,B,Lcomp, fill=true, leg=false, nlevels=40, title="Plot centered around: $(round.(PlotPlane.stütz,sigdigits=4))",
            xlabel="$(PlotPlane.Vx) direction", ylabel="$(PlotPlane.Vy) direction")
        p = scatter!([0],[0],lab="Center", marker=:hex)
        return p
    end
end

PlotLoglikelihood(DM::DataModel, PlotPlane::Plane, size::Float64=0.5,N::Int=100;Save::Bool=true) = PlotLoglikelihood(DM, PlotPlane, HyperCube([[-size,size],[-size,size]]), N, Save=Save)


function ConstructBox(fit::LsqFit.LsqFitResult, Confnum::Real; AxisCS::Bool=true)
    E = Confnum * stderror(fit)
    HyperCube(fit.param - E, fit.param + E)
end

# Choose a Plane?
VisualizeMC(Test::Function, Boundaries::AbstractVector,N::Int=2000) = VisualizeMC(Test, HyperCube(Boundaries), N)

function VisualizeMC(Test::Function, PlanarCube::HyperCube, N::Int=2000)
    length(PlanarCube) != 2 && throw(ArgumentError("Cube not Planar."))
    Lims = PlanarCube
    YesPoints = Vector{Vector{Float64}}(undef,0)
    NoPoints = Vector{Vector{Float64}}(undef,0)
    for i in 1:N
        num = rand.(Uniform.(Lims.L,Lims.U))
        if Test(num)
            push!(YesPoints,num)
        else
            push!(NoPoints,num)
        end
    end
    Plots.plot!(rectangle(Lims),lw=2,lab="Sample Space")
    p = scatter!(SensibleOutput(YesPoints), marker=(0.7,:hex,:green), markersize=2,lab="Inside")
    p = scatter!(SensibleOutput(NoPoints), marker=(0.7,:circle,:red), markersize=2,lab="Outside")
    p
end

function VisualizeMC(Test::Function, sol::ODESolution, N::Int=2000)
    Cube = ConstructCube(sol)
    lowers, uppers = Cube.L, Cube.U
    YesPoints = Vector{Vector{Float64}}(undef,0)
    NoPoints = Vector{Vector{Float64}}(undef,0)
    for i in 1:N
        num = rand.(Uniform.(lowers,uppers))
        if Test(num)
            push!(YesPoints,num)
        else
            push!(NoPoints,num)
        end
    end
    p = Plots.plot(sol,vars=(1,2), lw=2,xlims=(lowers[1],uppers[1]), ylims=(lowers[2],uppers[2]))
        try
        Box = rectangle(lowers...,uppers...)
        Plots.plot!(Box[:,1],Box[:,2],lw=2,lab="Sample Space")
    catch x
        println("Could not plot sampling rectangle. Fix Error:")
        println(x)
    end
    p = scatter!(SensibleOutput(YesPoints), marker=(0.5,:hex,:green), markersize=2,lab="Inside")
    p = scatter!(SensibleOutput(NoPoints), marker=(0.5,:circle,:red), markersize=2,lab="Outside")
    p
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
    SpacePerVec = scaling/VecsPerLine .* CubeWidths(C)
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

function Plot2DVF(DM::DataModel,V::Function,MLE::AbstractVector,PlanarCube::HyperCube,N::Int=25;scaling::Float64=0.85, OverWrite::Bool=false)
    length(MLE) !=2 && throw(ArgumentError("Only 2D supported."))
    length(PlanarCube) != 2 && throw(ArgumentError("Cube not Planar."))
    Lims = TranslateCube(PlanarCube,MLE)
    AV, BV  = meshgrid(range(Lims.L[1], Lims.U[1], length=N), range(Lims.L[2], Lims.U[2], length=N))
    Vcomp(a,b) = V([a,b])
    u,v = Vcomp.(AV,BV) |> SensibleOutput
    u,v = VFRescale([u v],PlanarCube,scaling=scaling)
    if OverWrite
        quiver(AV,BV,quiver=(u,v)) |> display
    else
        quiver!(AV,BV,quiver=(u,v)) |> display
    end
    [AV BV u v]
end

function Plot2DVF(DM::DataModel,V::Function,MLE::AbstractVector,size::Float64=0.5,N::Int=25;scaling::Float64=0.85, OverWrite::Bool=false)
    Plot2DVF(DM,V, MLE, HyperCube([[-size,size],[-size,size]]), N; scaling=scaling, OverWrite=OverWrite)
end


function Plot2DVF(DM::DataModel, V::Function, PlotPlane::Plane, PlanarCube::HyperCube, N::Int=25; scaling::Float64=0.85, OverWrite::Bool=false)
    length(PlanarCube) != 2 && throw(ArgumentError("Cube not Planar."))
    Lims = PlanarCube
    AV, BV  = meshgrid(range(Lims.L[1], Lims.U[1], length=N), range(Lims.L[2], Lims.U[2], length=N))
    Vcomp(a,b) = V(PlaneCoordinates(PlotPlane,[a,b]))
    u,v = Vcomp.(AV,BV) |> SensibleOutput
    u,v = VFRescale([u v],PlanarCube,scaling=scaling)
    if OverWrite
        quiver(AV,BV,quiver=(u,v)) |> display
    else
        quiver!(AV,BV,quiver=(u,v)) |> display
    end
    [AV BV u v]
end

function Plot2DVF(DM::DataModel,V::Function, PlotPlane::Plane,size::Float64=0.5, N::Int=25; scaling::Float64=0.85, OverWrite::Bool=false)
    Plot2DVF(DM,V, PlotPlane, HyperCube([[-size,size],[-size,size]]), N; scaling=scaling, OverWrite=OverWrite)
end

"""
    Deplanarize(PL::Plane,sol::ODESolution; N::Int=500) -> Matrix
    Deplanarize(PL::Plane,sol::ODESolution,Ts::Union{AbstractVector{<:Number},AbstractRange}) -> Matrix
Converts the 2D outputs of `sol` from planar coordinates associated with `PL` to the coordinates of the ambient space of `PL`.
"""
Deplanarize(PL::Plane,sol::ODESolution; N::Int=500) = Deplanarize(PL,sol,range(sol.t[1],sol.t[end]; length=N))
Deplanarize(PL::Plane,sol::ODESolution,Ts::Union{AbstractVector{<:Number},AbstractRange}) = map(t->PlaneCoordinates(PL,sol(t)),Ts) |> Unpack

"""
    VisualizeSols(sols::Vector{<:ODESolution}; OverWrite::Bool=true)
Visualizes vectors of type `ODESolution` using the `Plots.jl` package. If `OverWrite=false`, the solution is displayed on top of the previous plot object.
"""
function VisualizeSols(sols::Vector{<:ODESolution}; vars::Tuple=Tuple(1:length(sols[1].u[1])), OverWrite::Bool=true, leg::Bool=false, kwargs...)
    p = [];     OverWrite && Plots.plot()
    for sol in sols
        p = VisualizeSols(sol; vars=vars, leg=leg, kwargs...)
    end;    p
end
function VisualizeSols(sol::ODESolution; vars::Tuple=Tuple(1:length(sol.u[1])), leg::Bool=false, OverWrite::Bool=false,
                                        ModelMapMeta::Union{ModelMap,Bool}=false, kwargs...)
    OverWrite && Plots.plot()
    if ModelMapMeta isa ModelMap
        names = pnames(ModelMapMeta)
        if length(vars) == 2
            return Plots.plot!(sol; xlabel=names[vars[1]], ylabel=names[vars[2]], vars=vars, leg=leg, kwargs...)
        elseif length(vars) == 3
            return Plots.plot!(sol; xlabel=names[vars[1]], ylabel=names[vars[2]], zlabel=names[vars[3]], vars=vars, leg=leg, kwargs...)
        end
        # What if vars > 3? Does Plots.jl throw an error?
    end
    Plots.plot!(sol; vars=vars, leg=leg, kwargs...)
end

function VisualizeSols(PL::Plane, sol::ODESolution; vars::Tuple=Tuple(1:length(PL)), leg::Bool=false, N::Int=500,
                            ModelMapMeta::Union{ModelMap,Bool}=false, kwargs...)
    H = Deplanarize(PL, sol; N=N)
    if ModelMapMeta isa ModelMap
        names = pnames(ModelMapMeta)
        return Plots.plot!(H[:,vars[1]], H[:,vars[2]], H[:,vars[3]]; xlabel=names[vars[1]], ylabel=names[vars[2]], zlabel=names[vars[3]], leg=leg, kwargs...)
    else
        return Plots.plot!(H[:,vars[1]], H[:,vars[2]], H[:,vars[3]]; leg=leg, kwargs...)
    end
end
function VisualizeSols(PL::Plane, sols::Vector{<:ODESolution}; vars::Tuple=Tuple(1:length(PL)), N::Int=500, OverWrite::Bool=true, leg::Bool=false, kwargs...)
    p = [];     OverWrite && Plots.plot()
    for sol in sols
        p = VisualizeSols(PL, sol; N=N, vars=vars, leg=leg, kwargs...)
    end;    p
end

VisualizeSols(X::Tuple, args...; kwargs...) = VisualizeSols(X..., args...; kwargs...)
function VisualizeSols(PL::Vector{<:Plane},sols::Vector{<:ODESolution}; vars::Tuple=Tuple(1:length(PL[1])), N::Int=500,
                OverWrite::Bool=true,leg::Bool=false, color=rand([:red,:blue,:green,:orange,:grey]), kwargs...)
    length(PL) != length(sols) && throw("VisualizeSols: Must receive same number of Planes and Solutions.")
    p = [];     OverWrite && Plots.plot()
    for i in 1:length(sols)
        p = VisualizeSols(PL[i], sols[i]; N=N, vars=vars, leg=leg, color=color, kwargs...)
    end;    p
end
function VisualizeSols(DM::AbstractDataModel, args...; OverWrite::Bool=true, kwargs...)
    OverWrite ? scatter([MLE(DM)]; label="MLE") : scatter!([MLE(DM)]; label="MLE")
    if Predictor(DM) isa ModelMap
        return VisualizeSols(args...; OverWrite=false, ModelMapMeta=Predictor(DM), kwargs...)
    else
        return VisualizeSols(args...; OverWrite=false, kwargs...)
    end
end


function VisualizeSols(CB::ConfidenceBoundary; vars::Tuple=Tuple(1:length(CB.MLE)), OverWrite::Bool=true, color=rand([:red,:blue,:green,:orange,:grey]), kwargs...)
    p = OverWrite ? scatter([CB.MLE]; label="MLE") : []
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

function VisualizeSols(CBs::Vector{<:ConfidenceBoundary}; vars::Tuple=Tuple(1:length(CBs[1].MLE)), OverWrite::Bool=true, kwargs...)
    @assert all(x->x.MLE==CBs[1].MLE, CBs)
    @assert allunique(map(x->x.Confnum,CBs))
    p = OverWrite ? Plots.scatter([CBs[1].MLE]; label="MLE") : []
    for CB in CBs
        VisualizeSols(CB; vars=vars, OverWrite=false, kwargs...)
    end; p
end

function VisualizeGeos(sols::Union{ODESolution,Vector{<:ODESolution}}; OverWrite::Bool=true, leg::Bool=false)
    VisualizeSols(sols; vars=Tuple(1:Int(length(sols[1].u[1])/2)), OverWrite=OverWrite, leg=leg)
end


function VisualizeSolPoints(sol::ODESolution; kwargs...)
    Plots.plot!([sol.u[i][1] for i in 1:length(sol.t)], [sol.u[i][2] for i in 1:length(sol.t)]; marker=:hex, markersize=2, kwargs...)
end
function VisualizeSolPoints(sols::Vector{<:ODESolution}; OverWrite::Bool=false, kwargs...)
    p = [];     OverWrite && Plots.plot()
    for sol in sols
        p = VisualizeSolPoints(sol; kwargs...)
    end;    p
end


function ConstructAmbientSolution(PL::Plane, sol::ODESolution{T,N}) where {T,N}
    @warn "ConstructAmbientSolution() is an experimental feature. In particular, interpolation for the ambient solution object is not implemented yet."
    ODESolution{T,N,typeof(sol.u),typeof(sol.u_analytic),typeof(sol.errors),
                 typeof(sol.t),typeof(sol.k),
                 typeof(sol.prob),typeof(sol.alg),typeof(sol.interp),typeof(sol.destats)}(
                 map(x->PlaneCoordinates(PL,x), sol.u), #sol.u[I],
                 sol.u_analytic === nothing ? nothing : throw("Not programmed for u_analytic yet."), # Also translate u_analytic??
                 sol.errors, sol.t,
                 [map(x->PlaneCoordinates(PL,x), k) for k in sol.k], #sol.dense ? sol.k[I] : sol.k,
                 sol.prob,
                 sol.alg,
                 sol.interp, #(args...)->PlaneCoordinates(PL,sol.interp(args...)),
                 false,sol.tslocation,sol.destats,sol.retcode)
end


XCube(DS::AbstractDataSet; Padding::Number=0.) = ConstructCube(Unpack(WoundX(DS)); Padding=Padding)
XCube(DM::AbstractDataModel; Padding::Number=0.) = XCube(Data(DM); Padding=Padding)
Grid(Cube::HyperCube, N::Int=5) = [range(Cube.L[i], Cube.U[i]; length=N) for i in 1:length(Cube)]


# function GetExtrema(DM::AbstractDataModel,sols::Union{ODESolution,Vector{<:ODESolution}},X::Union{<:Number,AbstractVector{<:Number}}; N::Int=200)
#     low = Inf;   up = -Inf
#     for sol in sols
#         Y = map(Z->DM.model(X,sol(Z)), range(sol.t[1],sol.t[end]; length=N))
#         templow = minimum(Y);   tempup = maximum(Y)
#         if templow < low    low = templow       end
#         if up < tempup      up = tempup         end
#     end;    (low, up)
# end
# function GetExtrema(DM::AbstractDataModel,PL::Plane,sols::Union{ODESolution,Vector{<:ODESolution}},X::Union{<:Number,AbstractVector{<:Number}}; N::Int=200)
#     low = Inf;   up = -Inf
#     for sol in sols
#         templow, tempup = map(t->DM.model(X,PlaneCoordinates(PL,sol(t))), range(sol.t[1],sol.t[end]; length=N)) |> extrema
#         if templow < low    low = templow       end
#         if up < tempup      up = tempup         end
#     end;    (low, up)
# end
# function GetExtrema(DM::AbstractDataModel,PL::Vector{<:Plane},sols::Vector{<:ODESolution},X::Union{<:Number,AbstractVector{<:Number}}; N::Int=200)
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
#     ConfidenceBands(DM::DataModel,sol::ODESolution,domain::HyperCube; N::Int=200)
# Given a confidence interval `sol`, the pointwise confidence band around the model prediction is computed for x values in `domain` by evaluating the model on the boundary of the confidence region.
# """
# function ConfidenceBands(DM::AbstractDataModel,sols::Union{ODESolution,Vector{<:ODESolution}},domain::HyperCube=XCube(DM); N::Int=200, Np::Int=200)
#     !(length(domain) == xdim(DM)) && throw("Dimensionality of domain inconsistent with xdim.")
#     low = Vector{Float64}(undef,N^xdim(DM));     up = Vector{Float64}(undef,N^xdim(DM))
#     X = xdim(DM) == 1 ? range(domain.L[1],domain.U[1]; length=N) : Iterators.product(Grid(domain,N)...)
#     for i in 1:length(X)
#         low[i], up[i] = GetExtrema(DM,sols,X[i]; N=Np)
#     end
#     col = rand([:red,:blue,:green,:orange,:grey])
#     Plots.plot!(X,low,color=col,label="Lower Conf. Band")
#     Plots.plot!(X,up,color=col,label="Upper Conf. Band") |> display
#     return [Unpack(collect(X)) low up]
# end
#
# function ConfidenceBands(DM::AbstractDataModel,Planes::Union{Plane,Vector{<:Plane}},sols::Union{ODESolution,Vector{<:ODESolution}},
#                                             domain::HyperCube=XCube(DM); N::Int=200, Np::Int=300)
#     length(domain) != xdim(DM) && throw("Dimensionality of domain inconsistent with xdim.")
#     !(length(Planes) == 1 || length(Planes) == length(sols)) && throw("Number of Planes inconsisten with number of ODESolutions.")
#     low = Vector{Float64}(undef,N^xdim(DM));     up = Vector{Float64}(undef,N^xdim(DM))
#     X = xdim(DM) == 1 ? range(domain.L[1],domain.U[1]; length=N) : Iterators.product(Grid(domain,N)...)
#     for i in 1:length(X)
#         low[i], up[i] = GetExtrema(DM,Planes,sols,X[i]; N=Np)
#     end
#     col = rand([:red,:blue,:green,:orange,:grey])
#     Plots.plot!(X,low,color=col,label="Lower Conf. Band")
#     Plots.plot!(X,up,color=col,label="Upper Conf. Band") |> display
#     return [Unpack(collect(X)) low up]
# end


"""
    ConfidenceBands(DM::DataModel, sol::ODESolution, Xdomain::HyperCube; N::Int=300) -> Matrix
Given a confidence interval `sol`, the pointwise confidence band around the model prediction is computed for x values in `Xdomain` by evaluating the model on the boundary of the confidence region.
"""
function ConfidenceBands(DM::AbstractDataModel, sols::Union{ODESolution,Vector{<:ODESolution}}, Xdomain::HyperCube=XCube(DM);
                            N::Int=300, plot::Bool=true, samples::Int=200)
    length(Xdomain) != xdim(DM) && throw("Dimensionality of Xdomain inconsistent with xdim.")
    if xdim(DM) == 1
        X = range(Xdomain.L[1], Xdomain.U[1]; length=N)
        Res = Array{Float64,2}(undef, N, 2*ydim(DM))
        for col in 1:2:2ydim(DM)
            fill!(view(Res,:,col), Inf)
            fill!(view(Res,:,col+1), -Inf)
        end
        for sol in sols
            for t in range(sol.t[1], sol.t[end]; length=samples)
                # Do it like this to exploit CustomEmbeddings
                # WILL NOT WORK FOR CompositeDataSet!!!!!!
                Y = Windup(EmbeddingMap(Data(DM), Predictor(DM), sol(t), X), ydim(DM)) |> Unpack
                for col in 1:2:2ydim(DM)
                    Ycol = Int(ceil(col/2))
                    for row in 1:size(Res,1)
                        Res[row, col] = min(Res[row, col], Y[row,Ycol])
                        Res[row, col+1] = max(Res[row, col+1], Y[row,Ycol])
                    end
                end
            end
        end
        if plot
            for col in 1:2:2ydim(DM)
                PlotConfidenceBands(hcat(X,view(Res,:,col:col+1)))
            end
        end
        return hcat(X, Res)
    else
        throw("Not programmed for xdim != 1 yet.")
    end
end

function ConfidenceBands(DM::AbstractDataModel, Confnum::Real, Xdomain::HyperCube=XCube(DM); N::Int=300, plot::Bool=true, samples::Int=200)
    ConfidenceBands(DM, ConfidenceRegion(DM,Confnum), Xdomain; N=N, plot=plot, samples=samples)
end

function PlotConfidenceBands(M::AbstractMatrix)
    size(M,2) != 3 && throw("Matrix dimensions inconsistent: $(size(M)).")
    X = view(M,:,1);    low = view(M,:,2);      up = view(M,:,3)
    col = rand([:red,:blue,:green,:orange,:grey])
    # Plots.plot!(X, low; ribbon=(zeros(length(X)), up-low), linealpha=0, color=col, fillalpha=0.25, label="")
    Plots.plot!(X, low; color=col, label="")
    Plots.plot!(X, up; color=col, label="Conf. Band") |> display
end

"""
ApproxConfidenceBands(DM::AbstractDataModel, Confnum::Real, Xdomain=XCube(DM); N::Int=300, plot::Bool=true, add::Real=2.0)
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
function ApproxConfidenceBands(DM::AbstractDataModel, ParameterCube::HyperCube, Xdomain=XCube(DM); N::Int=300, plot::Bool=true)
    length(Xdomain) != xdim(DM) && throw("Dimensionality of domain inconsistent with xdim.")
    if xdim(DM) == 1
        X = range(Xdomain.L[1], Xdomain.U[1]; length=N)
        Res = Array{Float64,2}(undef, N, 2*ydim(DM))
        for col in 1:2:2ydim(DM)
            fill!(view(Res,:,col), Inf)
            fill!(view(Res,:,col+1), -Inf)
        end
        for point in FaceCenters(ParameterCube)
            # Do it like this to exploit CustomEmbeddings
            # WILL NOT WORK FOR CompositeDataSet!!!!!!
            Y = Windup(EmbeddingMap(Data(DM), Predictor(DM), point, X), ydim(DM)) |> Unpack
            for col in 1:2:2ydim(DM)
                Ycol = Int(ceil(col/2))
                for row in 1:size(Res,1)
                    Res[row, col] = min(Res[row, col], Y[row,Ycol])
                    Res[row, col+1] = max(Res[row, col+1], Y[row,Ycol])
                end
            end
        end
        if plot
            for col in 1:2:2ydim(DM)
                PlotConfidenceBands(hcat(X,view(Res,:,col:col+1)))
            end
        end
        return hcat(X, Res)
    else
        throw("Not programmed for xdim != 1 yet.")
    end
end



PointwiseConfidenceBandFULL(DM::DataModel,sol::ODESolution,Cube::HyperCube,Confnum::Real=1; N::Int=500) = PointwiseConfidenceBandFULL(DM,sol,FindMLE(DM),Cube,Confnum; N=N)
function PointwiseConfidenceBandFULL(DM::DataModel,sol::ODESolution,MLE::AbstractVector,Cube::HyperCube,Confnum::Real=1; N::Int=500)
    !(length(Cube) == xdim(DM)) && throw("PWConfBand: Wrong Cube dim.")
    if ydim(DM) == 1
        Lims = ConstructCube(sol)
        low = Vector{Float64}(undef,N); up = Vector{Float64}(undef,N)
        X = range(Cube.L[1],Cube.U[1],length=N)
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
                    if Y > up[i]
                        up[i] = Y
                    end
                    if Y < low[i]
                        low[i] = Y
                    end
                end
            else
                i = i-1
            end
        end
        Plots.plot!(X,low)
        Plots.plot!(X,up) |> display
        return [X low up]
    else
        throw("Not programmed yet.")
        # Evaluate on boundary of cube
    end
end

"""
    PlotMatrix(Mat::Matrix, MLE::Vector; N::Int=400)
Plots ellipse corresponding to a given covariance matrix which may additionally be offset by a vector `MLE`.

Example:
```
PlotMatrix(inv(FisherMetric(DM,MLE)),MLE)
```
"""
function PlotMatrix(Mat::AbstractMatrix, MLE::AbstractVector{<:Number}=zeros(size(Mat,1)); dims::Tuple{Int,Int}=(1,2), N::Int=400, plot::Bool=true, kwargs...)
    !(length(MLE) == size(Mat,1) == size(Mat,2)) && throw("PlotMatrix: Dimensional mismatch.")
    C = sqrt(quantile(Chisq(length(MLE)),ConfVol(1))) .* cholesky(Symmetric(Mat)).L;  angles = range(0,2π;length=N)
    F(α::Number) = MLE + C * RotatedVector(α,dims[1],dims[2],length(MLE))
    Data = Unpack(F.(angles))
    if plot   display(Plots.plot!(ToCols(Data)...;label="Matrix", kwargs...))  end
    Data
end



function PlotCurves(Curves::Vector{<:ODESolution}; N::Int=100)
    p = [];    A = Array{Float64,2}(undef,N,2)
    for sol in Curves
        ran = range(sol.t[1],sol.t[end],length=N)
        for i in 1:length(ran)    A[i,:] = sol(ran[i])[1:2]  end
        p = Plots.plot!(A[:,1],A[:,2])
        # p = Plots.plot!(sol,vars=(1,2))
    end
    p
end

EvaluateAlongGeodesic(F::Function,sol::ODESolution, Interval::Tuple{<:Number,<:Number}=(sol.t[1],sol.t[end]); N::Int=300) = [F(sol(t)[1:Int(length(sol.u[1])/2)]) for t in range(Interval[1],Interval[2],length=N)]
function PlotAlongGeodesic(F::Function,sol::ODESolution, Interval::Tuple{<:Number,<:Number}=(sol.t[1],sol.t[end]); N::Int=300, OverWrite::Bool=false)
    Z = EvaluateAlongGeodesic(F,sol,Interval, N=N)
    if length(Z[1]) == 1
        if OverWrite
            Plots.plot(range(Interval[1],Interval[2],length=N),Z) |> display
        else
            Plots.plot!(range(Interval[1],Interval[2],length=N),Z) |> display
        end
    end
    [collect(range(Interval[1],Interval[2],length=N)) Z]
end
EvaluateAlongGeodesicLength(DM::AbstractDataModel,F::Function,sol::ODESolution, Interval::Tuple{<:Number,<:Number}=(sol.t[1],sol.t[end]); N::Int=300) = EvaluateAlongGeodesic(F,sol,Interval, N=N)
function PlotAlongGeodesicLength(DM::AbstractDataModel,F::Function,sol::ODESolution, Interval::Tuple{<:Number,<:Number}=(sol.t[1],sol.t[end]); N::Int=300, OverWrite::Bool=false)
    Z = EvaluateAlongGeodesic(F,sol,Interval; N=N)
    Geo = GeodesicLength(x->FisherMetric(DM,x), sol, sol.t[end]; fullSol=true, Auto=true, tol=1e-14)
    Ls = map(Geo,range(Interval[1],Interval[2],length=N))
    if length(Z[1]) == 1
        if OverWrite
            Plots.plot(Ls,Z) |> display
        else
            Plots.plot!(Ls,Z) |> display
        end
    end
    [Ls Z]
end
EvaluateAlongCurve(F::Function,sol::ODESolution, Interval::Tuple{<:Number,<:Number}=(sol.t[1],sol.t[end]); N::Int=300) = [F(sol(t)) for t in range(Interval[1],Interval[2],length=N)]
function PlotAlongCurve(F::Function,sol::ODESolution, Interval::Tuple{<:Number,<:Number}=(sol.t[1],sol.t[end]); N::Int=300, OverWrite::Bool=false)
    Z = EvaluateAlongCurve(F,sol,Interval, N=N)
    if length(Z[1]) == 1
        if OverWrite
            Plots.plot(range(Interval[1],Interval[2],length=N),Z) |> display
        else
            Plots.plot!(range(Interval[1],Interval[2],length=N),Z) |> display
        end
    end
    Z
end



# Helper methods for creating a rectangular or triangular mesh from the outputs of MincedBoundaries()

"""
Returns Array of vertex numbers where every row constitutes a trapezoid for two adjacent curves from which N samples have been drawn.
"""
function RectangularFacetIndices(N::Int=20, zerobased::Bool=false)
    G = Matrix{Int64}(undef,0,4)
    for i in 0:(N-2)
        G = vcat(G,[i i+1 N+i+1 N+i])
    end
    # Facet indices zero-based or one-based?
    if zerobased
        return vcat(G,[N-1 0 N 2N-1])
    else
        return vcat(G,[N-1 0 N 2N-1]) .+ 1
    end
end
"""
Turns Array which specifies trapezoidal faces into triangular connections.
"""
function RectToTriangFacets(M::Matrix{<:Int})
    G = Matrix{Int64}(undef,2size(M,1),3)
    for i in 1:size(M,1)
        G[2i-1,:] = [M[i,1], M[i,2], M[i,4]]
        G[2i,:] = M[i,2:4]
    end;    G
end
function CreateMesh(Planes::Vector{<:Plane}, Sols::Vector{<:ODESolution}; N::Int=3*length(Sols), rectangular::Bool=true)
    Vertices = vcat([Deplanarize(Planes[i],Sols[i], N=N) for i in 1:length(Sols)]...)
    M = RectangularFacetIndices(N)
    Facets = vcat([M .+ (i-1)*N for i in 1:(length(Sols)-1)]...)
    if rectangular
        return Vertices, Facets
    else
        return Vertices, RectToTriangFacets(Facets)
    end
end
function ToObj(Vertices::Matrix,Facets::Matrix)
    text = ""
    for i in 1:size(Vertices,1)
        text *= "v"
        for j in 1:size(Vertices,2)
            text *= " $(Vertices[i,j])"
        end
        text *= '\n'
    end
    # ONE-BASED facet indices for .obj files
    for i in 1:size(Facets,1)
        text *= "f"
        for j in 1:size(Facets,2)
            text *= " $(Facets[i,j])"
        end
        text *= '\n'
    end
    text
end
function WriteObj(Vertices::Matrix,Facets::Matrix, path::String="D:/Boundary.obj")
    open(path,"w") do f
        write(f,ToObj(Vertices,Facets))
    end
    return
end
