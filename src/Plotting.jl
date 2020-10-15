




using RecipesBase
RecipesBase.@recipe function plot(DS::AbstractDataSet,args...)
    line -->            (:scatter,1)
    yerror -->          sigma(DS)
    linecolor   -->     :blue
    markercolor -->     :blue
    markerstrokecolor --> :blue
    xdata(DS),ydata(DS),args...
end
RecipesBase.@recipe function plot(DM::DataModel,args...)
    DM.Data,args...
end
RecipesBase.@recipe function plot(H::HyperCube,args...)
    LowerUpper(H),args...
end
RecipesBase.@recipe function plot(LU::LowerUpper,args...)
    length(LU.U) != 2 && throw("Cube not Planar, cannot Plot Box.")
    rectangle(LU)[:,1],rectangle(LU)[:,2],args...
end
RecipesBase.@recipe function plot(X::AbstractVector{<:AbstractVector{<:Number}},args...)
    G = Unpack(X)
    marker --> :hex
    linealpha --> 0
    markersize --> 1.8
    markerstrokewidth --> 0.5
    G[:,1],G[:,2]
end


# LSQFIT
import LsqFit.curve_fit
curve_fit(DM::AbstractDataModel,initial::AbstractVector{<:Number}=MLE(DM);tol::Real=6e-15,kwargs...) = curve_fit(DM.Data,DM.model,initial;tol=tol,kwargs...)
function curve_fit(DS::AbstractDataSet,model::Function,initial::AbstractVector{<:Number}=ones(pdim(F,xdata(DS)[1]))+0.01rand(pdim(F,xdata(DS)[1]));tol::Real=6e-15,kwargs...)
    X = xdata(DS);  Y = ydata(DS)
    LsqFit.check_data_health(X, Y)
    u = cholesky(InvCov(DS)).U
    f(p) = u * ( model(X, p) - Y )
    LsqFit.lmfit(f,initial,InvCov(DS);x_tol=tol,g_tol=tol,kwargs...)
end

# RecipesBase.@recipe function RecipeTester(args...)
#     rand(10,5)
# end
#
# RecipesBase.@recipe function FittedPlot2(DM::DataModel,Fit::LsqFit.LsqFitResult)
#     legendtitle     --> "R² ≈ $(round(Rsquared(DM,Fit),sigdigits=3))"
#     xlabel          --> "x"
#     ylabel          --> "y"
#     RecipesBase.@series begin
#         DM
#     end
#     RecipesBase.@series begin
#         X = range(xdata(DM)[1],xdata(DM)[end],length=500)
#         X, map(z->DM.model(z,Fit.param),X)
#     end
# end
# export FittedPlot2, RecipeTester

FittedPlot(DM::DataModel) = FittedPlot(DM,curve_fit(DM))
FittedPlot(DM::DataModel,p::AbstractVector) = FittedPlot(DM,curve_fit(DM,p))
function FittedPlot(DM::DataModel,Fit::LsqFit.LsqFitResult)
    Plots.plot(DM,label="Data",legendtitle="R² ≈ $(round(Rsquared(DM,Fit),sigdigits=3))")
    Plots.plot!(x->DM.model(x,Fit.param),range(xdata(DM)[1],xdata(DM)[end],length=600),label="Fit")
end

ResidualPlot(args...) = ResidPlot(args...)
ResidPlot(DM::DataModel) = ResidPlot(DM,curve_fit(DM))
ResidPlot(DM::DataModel,p::AbstractVector) = ResidPlot(DM,curve_fit(DM,p))
function ResidPlot(DM::DataModel,Fit::LsqFit.LsqFitResult)
    Plots.plot(x->0,[xdata(DM)[1],xdata(DM)[end]],label="Fit")
    Plots.plot!(DataSet(xdata(DM),-Fit.resid.*sigma(DM),sigma(DM)))
    Plots.plot!(legendtitle="R² ≈ $(round(Rsquared(DM,Fit),sigdigits=3))")
end



function PlotScalar(F::Function, PlanarCube::HyperCube; N::Int = 100, Save::Bool = false, parallel::Bool=false)
    PlanarCube.dim != 2 && throw(ArgumentError("Cube not Planar."))
    Lims = LowerUpper(PlanarCube)
    A = range(Lims.L[1], Lims.U[1], length=N)
    B = range(Lims.L[2], Lims.U[2], length=N)
    if Save
        X,Y = meshgrid(A,B)
        Z = similar(X)
        if parallel
            Z = pmap(F,X,Y)
        else
            Z = map(F,X,Y)
        end
        p = contour(X,Y,Z, fill=true, nlevels=40)
        display(p)
        return [X Y Z]
    else
        p = contour(A,B,F, fill=true, nlevels=40)
        return p
    end
end

function PlotScalar(F::Function, PlotPlane::Plane, PlanarCube::HyperCube, N::Int=100; Save::Bool=true, parallel::Bool=false)
    Lcomp(x,y) = F(PlaneCoordinates(PlotPlane,[x,y]))
    PlanarCube.dim != 2 && throw(ArgumentError("Cube not Planar."))
    Lims = LowerUpper(PlanarCube)
    A = range(Lims.L[1], Lims.U[1], length=N)
    B = range(Lims.L[2], Lims.U[2], length=N)
    if Save
        X,Y = meshgrid(A,B)
        Z = similar(X)
        if parallel
            Z = pmap(Lcomp,X,Y)
        else
            Z = map(Lcomp,X,Y)
        end
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
    PlanarCube.dim != 2 && throw(ArgumentError("Cube not Planar."))
    Lcomp(args...) = loglikelihood(DM,[args...])
    Lims = LowerUpper(TranslateCube(PlanarCube,MLE))
    A = range(Lims.L[1], Lims.U[1], length=N);  B = range(Lims.L[2], Lims.U[2], length=N)
    if Save
        X,Y = meshgrid(A,B)
        Z = similar(X)
        if parallel
            Z = pmap(Lcomp,X,Y)
        else
            Z = map(Lcomp,X,Y)
        end
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
    PlanarCube.dim != 2 && throw(ArgumentError("Cube not Planar."))
    Lims = LowerUpper(PlanarCube)
    A = range(Lims.L[1], Lims.U[1], length=N)
    B = range(Lims.L[2], Lims.U[2], length=N)
    if Save
        X,Y = meshgrid(A,B)
        Z = similar(X)
        if parallel
            Z = pmap(Lcomp,X,Y)
        else
            Z = map(Lcomp,X,Y)
        end
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


function ConstructBox(fit::LsqFit.LsqFitResult,Confnum::Real; AxisCS::Bool=true)
    E = Confnum * stderror(fit)
    LowerUpper(fit.param .- E, fit.param .+ E)
end

# Choose a Plane?
VisualizeMC(Test::Function, Boundaries::AbstractVector,N::Int=2000) = VisualizeMC(Test, HyperCube(Boundaries), N)

function VisualizeMC(Test::Function, PlanarCube::HyperCube, N::Int=2000)
    PlanarCube.dim != 2 && throw(ArgumentError("Cube not Planar."))
    Lims = LowerUpper(PlanarCube)
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
    lowers, uppers = ConstructLowerUpper(sol)
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
function rectangle(LU::LowerUpper)
    length(LU.L) != 2 && throw(ArgumentError("Cube not Planar."))
    rectangle((LU.L)...,(LU.U)...)
end
rectangle(H::HyperCube) = rectangle(LowerUpper(H))


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
    PlanarCube.dim != 2 && throw(ArgumentError("Cube not Planar."))
    Lims = LowerUpper(TranslateCube(PlanarCube,MLE))
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


function Plot2DVF(DM::DataModel,V::Function, PlotPlane::Plane, PlanarCube::HyperCube, N::Int=25; scaling::Float64=0.85, OverWrite::Bool=false)
    PlanarCube.dim != 2 && throw(ArgumentError("Cube not Planar."))
    Lims = LowerUpper(PlanarCube)
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




function Deplanarize(PL::Plane,sol::ODESolution;N::Int=500)
    map(t->PlaneCoordinates(PL,sol(t)),range(sol.t[1],sol.t[end],length=N)) |> Unpack
end

"""
    VisualizeSols(sols::Vector; OverWrite::Bool=true)
Visualizes vectors of type `ODESolution` using the `Plots.jl` package. If `OverWrite=false`, the solution is displayed on top of the previous plot object.
"""
function VisualizeSols(sols::Vector{<:ODESolution}; vars::Tuple=Tuple(1:length(sols[1].u[1])), OverWrite::Bool=true,leg::Bool=false,kwargs...)
    p = [];     OverWrite && Plots.plot()
    for sol in sols
        p = VisualizeSol(sol;vars=vars,leg=leg,kwargs...)
    end;    p
end
VisualizeSol(sol::ODESolution; vars::Tuple=Tuple(1:length(sol.u[1])), leg::Bool=false,kwargs...) = Plots.plot!(sol;vars=vars,leg=leg,kwargs...)


function VisualizeSol(PL::Plane,sol::ODESolution; vars::Tuple=Tuple(1:length(sol.u[1])), leg::Bool=false, N::Int=500, kwargs...)
    H = Deplanarize(PL,sol;N=N);    Plots.plot!(H[:,1],H[:,2],H[:,3];leg=leg, kwargs...)
end
function VisualizeSols(PL::Plane,sols::Vector{<:ODESolution}; vars::Tuple=Tuple(1:length(sols[1].u[1])), N::Int=500, OverWrite::Bool=true,leg::Bool=false, kwargs...)
    p = [];     OverWrite && Plots.plot()
    for sol in sols
        p = VisualizeSol(PL,sol; N=N,vars=vars,leg=leg, kwargs...)
    end;    p
end
function VisualizeSols(PL::Vector{<:Plane},sols::Vector{<:ODESolution}; vars::Tuple=Tuple(1:length(sols[1].u[1])), N::Int=500,
            OverWrite::Bool=true,leg::Bool=false, color=rand([:red,:blue,:green,:orange,:grey]), kwargs...)
    length(PL) != length(sols) && throw("VisualizeSols: Must receive same number of Planes and Solutions.")
    p = [];     OverWrite && Plots.plot()
    for i in 1:length(sols)
        p = VisualizeSol(PL[i],sols[i]; N=N,vars=vars,leg=leg, color=color, kwargs...)
    end;    p
end


function VisualizeSolPoints(sol::ODESolution; kwargs...)
    Plots.plot!([sol.u[i][1] for i in 1:length(sol.t)], [sol.u[i][2] for i in 1:length(sol.t)]; marker=:hex,markersize=2, kwargs...)
end
function VisualizeSolPoints(sols::Vector{<:ODESolution}; OverWrite::Bool=false, kwargs...)
    p = [];     OverWrite && Plots.plot()
    for sol in sols
        p = VisualizeSolPoints(sol; kwargs...)
    end;    p
end

VisualizeGeo(sol::ODESolution;OverWrite::Bool=false,leg::Bool=false) = VisualizeSol(sol,vars=Tuple(1:Int(length(sol.u[1])/2)),OverWrite=OverWrite,leg=leg)
VisualizeGeos(sols::Vector{<:ODESolution}; OverWrite::Bool=true,leg::Bool=false) = VisualizeSols(sols,vars=Tuple(1:Int(length(sols[1].u[1])/2)),OverWrite=OverWrite,leg=leg)


"""
    PointwiseConfidenceBand(DM::DataModel,sol::ODESolution,domain::HyperCube; N::Int=200)
Given a confidence interval `sol`, the pointwise confidence band around the model prediction is computed for x values in `domain` by evaluating the model on the boundary of the confidence interval.
"""
function PointwiseConfidenceBand(DM::DataModel,sol::ODESolution,domain::HyperCube; N::Int=300)
    domain.dim != length(xdata(DM)[1]) && throw("PWConfBand: Wrong Cube dim.")
    if length(ydata(DM)[1]) == 1
        T = range(sol.t[1],sol.t[end],length=300)
        X = range(domain.vals[1][1],domain.vals[1][2],length=N)
        low = Vector{Float64}(undef,N); up = Vector{Float64}(undef,N)
        for i in 1:length(X)
            Y = map(t->DM.model(X[i],sol(t)),T)
            up[i] = maximum(Y); low[i] = minimum(Y)
        end
        Plots.plot!(X,low)
        Plots.plot!(X,up) |> display
        return [X low up]
    else
        throw("Not programmed yet.")
        # Evaluate on boundary of cube
    end
end


PointwiseConfidenceBandFULL(DM::DataModel,sol::ODESolution,Cube::HyperCube,Confnum::Real=1; N::Int=500) = PointwiseConfidenceBandFULL(DM,sol,FindMLE(DM),Cube,Confnum; N=N)
function PointwiseConfidenceBandFULL(DM::DataModel,sol::ODESolution,MLE::AbstractVector,Cube::HyperCube,Confnum::Real=1; N::Int=500)
    Cube.dim != length(xdata(DM)[1]) && throw("PWConfBand: Wrong Cube dim.")
    if length(ydata(DM)[1]) == 1
        Lims = ConstructCube(sol) |> LowerUpper
        low = Vector{Float64}(undef,N); up = Vector{Float64}(undef,N)
        X = range(Cube.vals[1][1],Cube.vals[1][2],length=N)
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
    PlotMatrix(Mat::Matrix,MLE::Vector,N::Int=400)
Plots ellipse corresponding to a given covariance matrix which may additionally be offset by a vector `MLE`.

Example:
```
PlotMatrix(inv(FisherMetric(DM,MLE)),MLE)
```
"""
function PlotMatrix(Mat::Matrix,MLE::AbstractVector{<:Number}=zeros(size(Mat,1)),N::Int=400)
    !(length(MLE) == size(Mat,1) == size(Mat,2) == 2) && throw("PlotMatrix: Dimensional mismatch.")
    C = cholesky(Symmetric(Mat)).L;    angles = range(0,2pi,length=N)
    F(angle::Real) = MLE .+ C* [cos(angle),sin(angle)]
    Data = Unpack(F.(angles))
    display(plot!(Data[:,1],Data[:,2],label="Matrix")); Data
end



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
