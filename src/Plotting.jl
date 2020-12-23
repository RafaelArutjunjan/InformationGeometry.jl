

RecipesBase.@recipe f(DM::AbstractDataModel) = DM, MLE(DM)

RecipesBase.@recipe function f(DM::AbstractDataModel, MLE::AbstractVector{<:Number})
    (!(xdim(DM) == ydim(DM) == 1) && Npoints(DM) > 1) && throw("Not programmed for plotting xdim != 1 or ydim != 1 yet.")
    legendtitle --> "R² ≈ $(round(Rsquared(DM),sigdigits=3))"
    xguide -->              xnames(DM)[1]
    yguide -->              ynames(DM)[1]
    @series begin
        Data(DM)
    end
    markeralpha :=      0.
    label -->           "Fit"
    seriescolor -->     :red
    linestyle -->       :solid
    linewidth -->       2
    Xbounds = extrema(xdata(DM))
    X = range(Xbounds[1], Xbounds[2]; length=500)
    Y = map(z->DM.model(z,MLE), X)
    ToCols([X Y])
end

RecipesBase.@recipe function f(DS::DataSet)
    !(xdim(DS) == ydim(DS) == 1) && throw("Not programmed for plotting xdim != 1 or ydim != 1 yet.")
    Σ_y = typeof(sigma(DS)) <: AbstractVector ? sigma(DS) : sqrt.(Diagonal(sigma(DS)).diag)
    line -->                (:scatter,1)
    xguide -->              xnames(DS)[1]
    yguide -->              ynames(DS)[1]
    label -->               "Data"
    yerror -->              Σ_y
    linecolor   -->         :blue
    markercolor -->         :blue
    markerstrokecolor -->   :blue
    xdata(DS), ydata(DS)
end

RecipesBase.@recipe function f(DS::DataSetExact)
    !(xdim(DS) == ydim(DS) == 1) && throw("Not programmed for plotting xdim != 1 or ydim != 1 yet.")
    typeof(xdist(DS)) <: InformationGeometry.Dirac && return DataSet(xdata(DS), ydata(DS), ysigma(DS), DS.dims)
    Σ_x = typeof(xsigma(DS)) <: AbstractVector ? xsigma(DS) : sqrt.(Diagonal(xsigma(DS)).diag)
    Σ_y = typeof(ysigma(DS)) <: AbstractVector ? ysigma(DS) : sqrt.(Diagonal(ysigma(DS)).diag)
    line -->                (:scatter,1)
    xguide -->              xnames(DS)[1]
    yguide -->              ynames(DS)[1]
    label -->               "Data"
    xerror -->              Σ_x
    yerror -->              Σ_y
    linecolor   -->         :blue
    markercolor -->         :blue
    markerstrokecolor -->   :blue
    xdata(DS), ydata(DS)
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
function Rsquared(DM::DataModel)
    !(xdim(DM) == ydim(DM) == 1) && return -1
    mean = sum(ydata(DM)) / length(ydata(DM))
    Stot = (ydata(DM) .- mean).^2 |> sum
    Sres = (ydata(DM) - EmbeddingMap(DM,MLE(DM))).^2 |> sum
    1 - Sres / Stot
end


FittedPlot(DM::AbstractDataModel, args...; kwargs...) = Plots.plot(DM, args...; kwargs...)

ResidualPlot(DM::AbstractDataModel; kwargs...) = ResidualPlot(Data(DM), DM.model, MLE(DM); kwargs...)
function ResidualPlot(DS::DataSet, model::ModelOrFunction, mle::AbstractVector{<:Number}; kwargs...)
    Plots.plot(DataModel(DataSet(xdata(DS), ydata(DS)-EmbeddingMap(DS,model,mle), sigma(DS), DS.dims), (x,p)->0., mle, true); kwargs...)
end
function ResidualPlot(DS::DataSetExact, model::ModelOrFunction, mle::AbstractVector{<:Number}; kwargs...)
    Plots.plot(DataModel(DataSetExact(xdata(DS), xsigma(DS), ydata(DS)-EmbeddingMap(DS,model,mle), ysigma(DS), DS.dims), (x,p)->0., mle, true); kwargs...)
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
    VFRescale(ZeilenVecs::Array{<:Real,2},C::HyperCube;scaling=0.85)
Rescale vector to look good in 2D plot.
"""
function VFRescale(ZeilenVecs::Array{<:Real,2},C::HyperCube;scaling=0.85)
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
    Deplanarize(PL::Plane,sol::ODESolution,Ts::Union{AbstractVector{<:Real},AbstractRange}) -> Matrix
Converts the 2D outputs of `sol` from planar coordinates associated with `PL` to the coordinates of the ambient space of `PL`.
"""
Deplanarize(PL::Plane,sol::ODESolution; N::Int=500) = Deplanarize(PL,sol,range(sol.t[1],sol.t[end]; length=N))
Deplanarize(PL::Plane,sol::ODESolution,Ts::Union{AbstractVector{<:Real},AbstractRange}) = map(t->PlaneCoordinates(PL,sol(t)),Ts) |> Unpack

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
    if typeof(ModelMapMeta) <: ModelMap
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

function VisualizeSols(PL::Plane, sol::ODESolution; vars::Tuple=Tuple(1:length(sol.u[1])), leg::Bool=false, N::Int=500,
                            ModelMapMeta::Union{ModelMap,Bool}=false, kwargs...)
    H = Deplanarize(PL, sol; N=N);    Plots.plot!(H[:,1], H[:,2], H[:,3]; leg=leg, kwargs...)
end
function VisualizeSols(PL::Plane, sols::Vector{<:ODESolution}; vars::Tuple=Tuple(1:length(sols[1].u[1])), N::Int=500,
                            ModelMapMeta::Union{ModelMap,Bool}=false, OverWrite::Bool=true, leg::Bool=false, kwargs...)
    p = [];     OverWrite && Plots.plot()
    for sol in sols
        p = VisualizeSols(PL, sol; N=N, vars=vars, leg=leg, kwargs...)
    end;    p
end

VisualizeSols(X::Tuple{Vector{<:Plane},Vector{<:ODESolution}}, args...; kwargs...) = VisualizeSols(X[1], X[2], args...; kwargs...)
function VisualizeSols(PL::Vector{<:Plane},sols::Vector{<:ODESolution}; vars::Tuple=Tuple(1:length(sols[1].u[1])), N::Int=500,
            OverWrite::Bool=true,leg::Bool=false, color=rand([:red,:blue,:green,:orange,:grey]), ModelMapMeta::Union{ModelMap,Bool}=false, kwargs...)
    length(PL) != length(sols) && throw("VisualizeSols: Must receive same number of Planes and Solutions.")
    p = [];     OverWrite && Plots.plot()
    for i in 1:length(sols)
        p = VisualizeSols(PL[i], sols[i]; N=N, vars=vars, leg=leg, color=color, kwargs...)
    end;    p
end
function VisualizeSols(DM::AbstractDataModel, args...; OverWrite::Bool=true, kwargs...)
    OverWrite ? scatter([MLE(DM)]; label="MLE") : scatter!([MLE(DM)]; label="MLE")
    if typeof(Predictor(DM)) <: ModelMap
        return VisualizeSols(args...; OverWrite=false, ModelMapMeta=Predictor(DM), kwargs...)
    else
        return VisualizeSols(args...; OverWrite=false, kwargs...)
    end
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


XCube(DS::AbstractDataSet; Padding::Real=0.) = ConstructCube(Unpack(WoundX(DS)); Padding=Padding)
XCube(DM::AbstractDataModel; Padding::Real=0.) = XCube(Data(DM); Padding=Padding)
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
    ConfidenceBands(DM::DataModel,sol::ODESolution,domain::HyperCube; N::Int=300)
Given a confidence interval `sol`, the pointwise confidence band around the model prediction is computed for x values in `domain` by evaluating the model on the boundary of the confidence region.
"""
function ConfidenceBands(DM::AbstractDataModel,sol::ODESolution,domain::HyperCube=XCube(DM); N::Int=300, plot::Bool=true)
    !(length(domain) == xdim(DM) == 1) && throw("Dimensionality of domain inconsistent with xdim.")
    if ydim(DM) == 1
        T = range(sol.t[1],sol.t[end]; length=300)
        X = range(domain.L[1],domain.U[1]; length=N)
        low = Vector{Float64}(undef,N); up = Vector{Float64}(undef,N)
        for i in 1:length(X)
            Y = map(t->DM.model(X[i],sol(t)),T)
            low[i], up[i] = extrema(Y)
        end
        # plot && Plots.plot!(X,low; ribbon=(zeros(length(X)), up-low),linealpha=0,fillalpha=0.3,label="Conf. Band") |> display
        if plot
            col = rand([:red,:blue,:green,:orange,:grey])
            Plots.plot!(X,low,color=col,label="Lower Conf. Band");     Plots.plot!(X,up,color=col,label="Upper Conf. Band") |> display
        end
        return [X low up]
    else
        throw("Not programmed yet.")
    end
end

function ConfidenceBands(DM::AbstractDataModel, Confnum::Real, domain::HyperCube=XCube(DM); N::Int=300, plot::Bool=true)
    ConfidenceBands(DM, ConfidenceRegion(DM,Confnum), domain; N=N, plot=plot)
end


PointwiseConfidenceBandFULL(DM::DataModel,sol::ODESolution,Cube::HyperCube,Confnum::Real=1; N::Int=500) = PointwiseConfidenceBandFULL(DM,sol,FindMLE(DM),Cube,Confnum; N=N)
function PointwiseConfidenceBandFULL(DM::DataModel,sol::ODESolution,MLE::AbstractVector,Cube::HyperCube,Confnum::Real=1; N::Int=500)
    !(length(Cube) == xdim(DM)) && throw("PWConfBand: Wrong Cube dim.")
    if ydim(DM) == 1
        Lims = ConstructCube(sol)
        low = Vector{Float64}(undef,N); up = Vector{Float64}(undef,N)
        X = range(Cube.L[1],Cube.U[1],length=N)
        for i in 1:length(X)
            Y = DM.model(X[i],MLE)
            up[i] = maximum(Y); low[i] = minimum(Y)
        end
        LogLikeMLE = loglikelihood(DM,MLE)
        Confvol = ConfVol(Confnum)
        for i in 1:N
            num = rand.(Uniform.(Lims.L,Lims.U))
            if WilksTestPrepared(DM,num,LogLikeMLE,Confvol)
                for i in 1:length(X)
                    Y = DM.model(X[i],num)
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
    C = sqrt(quantile(Chisq(length(MLE)),ConfVol(1))) .* cholesky(Symmetric(Mat)).L;  angles = range(0,2pi,length=N)
    F(α::Real) = MLE + C * RotatedVector(α,dims[1],dims[2],length(MLE))
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

EvaluateAlongGeodesic(F::Function,sol::ODESolution, Interval::Tuple{<:Real,<:Real}=(sol.t[1],sol.t[end]); N::Int=300) = [F(sol(t)[1:Int(length(sol.u[1])/2)]) for t in range(Interval[1],Interval[2],length=N)]
function PlotAlongGeodesic(F::Function,sol::ODESolution, Interval::Tuple{<:Real,<:Real}=(sol.t[1],sol.t[end]); N::Int=300, OverWrite::Bool=false)
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
EvaluateAlongGeodesicLength(DM::AbstractDataModel,F::Function,sol::ODESolution, Interval::Tuple{<:Real,<:Real}=(sol.t[1],sol.t[end]); N::Int=300) = EvaluateAlongGeodesic(F,sol,Interval, N=N)
function PlotAlongGeodesicLength(DM::AbstractDataModel,F::Function,sol::ODESolution, Interval::Tuple{<:Real,<:Real}=(sol.t[1],sol.t[end]); N::Int=300, OverWrite::Bool=false)
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
EvaluateAlongCurve(F::Function,sol::ODESolution, Interval::Tuple{<:Real,<:Real}=(sol.t[1],sol.t[end]); N::Int=300) = [F(sol(t)) for t in range(Interval[1],Interval[2],length=N)]
function PlotAlongCurve(F::Function,sol::ODESolution, Interval::Tuple{<:Real,<:Real}=(sol.t[1],sol.t[end]); N::Int=300, OverWrite::Bool=false)
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
