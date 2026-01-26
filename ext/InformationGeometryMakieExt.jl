module InformationGeometryMakieExt

    using InformationGeometry, Makie
    using Distributions, DataInterpolations, LinearAlgebra, PlotUtils

    import InformationGeometry: ParameterProfilesView, IsPopulated, PlotSizer, GetConverged, Profiles, Trajectories, DOF, ApplyTrafoNames, pnames
    import InformationGeometry: IsCost, Convergify, HasPriors

    ### Taken from AlgebraOfGraphics.jl
    """
        adapted_aog_theme(; fonts=[Makie.to_font("Fira Sans-Medium"), Makie.to_font("Fira Sans-Light")])
    Use via `Makie.set_theme!(; InformationGeometry.adapted_aog_theme()...)`
    """
    function InformationGeometry.adapted_aog_theme(; fonts = [Makie.to_font("Fira Sans-Medium"), Makie.to_font("Fira Sans-Light")],
                        mediumfont = first(fonts), lightfont = last(fonts))
        marker = :circle

        colormap = :batlow
        linecolor = :gray25
        markercolor = :gray25
        patchcolor = :gray25

        palette = (
            color = Makie.wong_colors(),
            patchcolor = Makie.wong_colors(),
            marker = [:circle, :utriangle, :cross, :rect, :diamond, :dtriangle, :pentagon, :xcross],
            linestyle = [:solid, :dash, :dot, :dashdot, :dashdotdot],
            side = [:left, :right],
        )
        # setting marker here is a temporary hack
        # it should either respect `marker = :circle` globally
        # or `:circle` and `Circle` should have the same size
        BoxPlot = (mediancolor = :white, marker = :circle)
        Scatter = (marker = :circle,)
        Violin = (mediancolor = :white,)

        Axis = (
            xgridvisible = false, ygridvisible = false,
            # topspinevisible = true, rightspinevisible = true, 
            topspinecolor = :darkgray,  rightspinecolor = :darkgray, bottomspinecolor = :darkgray, leftspinecolor = :darkgray, 
            xtickcolor = :darkgray, ytickcolor = :darkgray, xminortickcolor = :darkgray, yminortickcolor = :darkgray,
            xticklabelfont = lightfont, yticklabelfont = lightfont, xlabelfont = mediumfont, ylabelfont = mediumfont, titlefont = mediumfont,
        )
        Axis3 = (
            protrusions = 55, # to include label on z axis, should be fixed in Makie
            xgridvisible = false, ygridvisible = false, zgridvisible = false,
            xspinecolor_1 = :darkgray, yspinecolor_1 = :darkgray, zspinecolor_1 = :darkgray, 
            xspinecolor_2 = :transparent, yspinecolor_2 = :transparent, zspinecolor_2 = :transparent, 
            xspinecolor_3 = :transparent, yspinecolor_3 = :transparent, zspinecolor_3 = :transparent,
            xtickcolor = :darkgray, ytickcolor = :darkgray, ztickcolor = :darkgray,
            xticklabelfont = lightfont, yticklabelfont = lightfont, zticklabelfont = lightfont, 
            xlabelfont = mediumfont, ylabelfont = mediumfont, zlabelfont = mediumfont, titlefont = mediumfont,
        )
        Legend = (
            framevisible = false,
            gridshalign = :left,
            padding = (0.0f0, 0.0f0, 0.0f0, 0.0f0),
            labelfont = lightfont,
            titlefont = mediumfont,
        )
        Colorbar = (
            flip_vertical_label = true,
            spinewidth = 0,
            ticklabelfont = lightfont,
            labelfont = mediumfont,
        )
        return (;
            joinstyle=:round,
            fonts = (; regular = lightfont, bold = mediumfont), 
            marker, colormap, linecolor, markercolor, patchcolor, palette, BoxPlot, Scatter, Violin, Axis, Axis3, Legend, Colorbar,
        )
    end

    

    ### ParameterProfiles

    function Makie.plot(P::ParameterProfiles; Trafo::Function=identity, ylims::Union{Nothing,Tuple}=nothing, Interpolate::Bool=false, Confnum::AbstractVector{<:Integer}=1:5, dof::Integer=DOF(P), MaxLevel::Union{Nothing,Real}=nothing, kwargs...)
        PopulatedInds = IsPopulated(P);     n_plots = sum(PopulatedInds)
        n_plots == 0 && error("No populated profiles to plot")
        if ylims === nothing
            tol = 0.05
            M = [maximum(view(T[2], GetConverged(T))) for T in Profiles(P) if !all(isnan, T[1]) && any(GetConverged(T)) && maximum(view(T[2], GetConverged(T))) > tol]
            maxy = length(M) > 0 ? median(M) : median([maximum(T[2]) for T in Profiles(P) if !all(isnan, T[1])])
            maxy = maxy < tol ? (maxy < 1e-8 ? tol : Inf) : maxy
            ylims = (Trafo(-tol), Trafo(maxy))
        end
        fig_size = PlotSizer(n_plots);  fig = Figure(size=fig_size)
        if P.Meta !== :ParameterProfiles
            fig[0, :] = Label(fig, string(P.Meta), fontsize=18, font=:bold)
        end
        n_cols = min(n_plots, 3); n_rows = ceil(Int, n_plots / n_cols)
        j = 1
        for i in eachindex(Profiles(P))
            if PopulatedInds[i]
                row = (j - 1) ÷ n_cols + 1; col = (j - 1) % n_cols + 1
                ax = fig[row, col] = Axis(fig)
                Makie.plot!(ax, ParameterProfilesView(P, i), Val(false); Trafo=Trafo, ylims=ylims, Interpolate=Interpolate, Confnum=Confnum, dof=dof, MaxLevel=MaxLevel, kwargs...)
                j += 1
            end
        end;    fig
    end
    function Makie.plot!(ax::Axis, PV::ParameterProfilesView, V::Val=Val(false); Trafo::Function=identity, ylims::Union{Nothing,Tuple}=nothing, Interpolate::Bool=false, Confnum::AbstractVector{<:Integer}=1:5, dof::Integer=DOF(PV), MaxLevel::Union{Nothing,Real}=nothing, kwargs...)
        i = PV.i
        ax.xlabel = pnames(PV)[i]
        ax.ylabel = ApplyTrafoNames(IsCost(PV) ? "W = 2[ℓ_mle - ℓ(θ)]" : "Conf. level [σ]", Trafo)
        
        if Interpolate
            Interp = QuadraticSpline;   Conv = GetConverged(Profiles(PV))
            !all(Conv) && @warn "Interpolating profile $i but $(sum(.!Conv))/$(length(Conv)) points not converged."
            F = InterpolatedProfiles(PV, Interp); xran = range(F.t[1], F.t[end]; length=300)
            lines!(ax, xran, map(Trafo∘F, xran); color=Makie.wong_colors()[1], linewidth=2, label="Interpolated Profile", linecap=:round, joinstyle=:round, kwargs...)
        else
            Conv = GetConverged(Profiles(PV)); y_vals = Convergify(Profiles(PV)[2], Conv)
            lines!(ax, Profiles(PV)[1], Trafo.(@view y_vals[:,1]); linewidth=2, color=Makie.wong_colors()[1], label="Profile Likelihood", linecap=:round, joinstyle=:round, kwargs...)
            !all(Conv) && lines!(ax, Profiles(PV)[1], Trafo.(@view y_vals[:,2]); color=Makie.wong_colors()[2], linewidth=2, label=nothing, linecap=:round, joinstyle=:round, kwargs...)
        end
        if HasPriors(PV)
            if Interpolate
                Interp = QuadraticSpline;   Conv = GetConverged(Profiles(PV))
                !all(Conv) && @warn "Interpolating profile $i but $(sum(.!Conv))/$(length(Conv)) points not converged."
                F = Interp(Profiles(PV)[3], Profiles(PV)[1]); xran = range(F.t[1], F.t[end]; length=300)
                lines!(ax, xran, vec(map(Trafo∘F, xran)); color=Makie.wong_colors()[3], linewidth=2, linestyle=:dash, alpha=0.85, label="Prior contribution", linecap=:round, joinstyle=:round)
            else
                Conv = GetConverged(Profiles(PV)); y_vals = Convergify(Profiles(PV)[3], Conv)
                lines!(ax, Profiles(PV)[1], Trafo.(@view y_vals[:,1]); color=Makie.wong_colors()[3], linewidth=2, linestyle=:dash, alpha=0.85, label="Prior contribution", linecap=:round, joinstyle=:round)
                !all(Conv) && lines!(ax, Profiles(PV)[1], Trafo.(@view y_vals[:,2]); color=Makie.wong_colors()[4], linewidth=2, linestyle=:dash, alpha=0.85, label="Prior contribution", linecap=:round, joinstyle=:round)
            end
        end
        scatter!(ax, [MLE(PV)[i]], [Trafo(0.0)]; marker=:hexagon, markersize=12, color=:red, strokewidth=0, label=nothing)
        if IsCost(PV) && all(Confnum .> 0)
            MaxLevel === nothing && (MaxLevel = maximum(view(Profiles(PV)[2], GetConverged(Profiles(PV))); init=-Inf))
            sorted_conf = sort(Confnum; rev=true)
            for (j, (Conf, Thresh)) in enumerate(zip(sorted_conf, convert.(eltype(MLE(PV)), InvChisqCDF.(dof, ConfVol.(sorted_conf)))))
                Thresh < MaxLevel && hlines!(ax, [Trafo(Thresh)]; linewidth=1.5, linestyle=:dash, color=palette(:viridis, length(sorted_conf); rev=true)[j], linecap=:round, label="$(j)σ level, dof=$dof")
            end
        end
        ylims !== nothing && ylims!(ax, ylims)
        return ax
    end
    function Makie.plot(PV::ParameterProfilesView, V::Val=Val(false); kwargs...)
        fig = Figure(); ax = Axis(fig[1, 1])
        Makie.plot!(ax, PV, V; kwargs...)
        return fig
    end

end # module