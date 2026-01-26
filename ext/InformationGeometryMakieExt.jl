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




    #### Datasets and DataModels
    import InformationGeometry: xdata, name, ynames, Npoints, DOF, ysigma, Windup, Unpack, DomainSamples, xdata, ResidualStandardError, xnames
    import InformationGeometry: AbstractUnknownUncertaintyDataSet, HasXerror
    import InformationGeometry: SubDataSetComponent, predictedY, NotPosDef

    """
        Makie.plot(DM::AbstractDataModel, mle, xpositions; Confnum=1.0, PlotVariance=false, dof=DOF(DM), Validation=false)
    Inner method: builds a Makie Figure and Axis, plots the data (with errors
    if present), the model fit, and optional variance bands.
    """
    function Makie.plot(DM::AbstractDataModel, mle::AbstractVector{<:Number} = MLE(DM), xpositions::AbstractVector{<:Number} = xdata(DM); Confnum::Real = 1.0, PlotVariance::Bool = false, 
                        dof::Real = DOF(DM), Validation::Bool = false, Padding::Real = 0.05, kwargs...)

        (xdim(DM) != 1 && Npoints(DM) > 1) && error("Not programmed for plotting xdim != 1 yet.")

        xlabel = (ydim(DM) > Npoints(DM) ? "Positions" : xnames(DM)[1])
        ylabel = (ydim(DM) == 1 ? ynames(DM)[1] : "Observations")
        ttl    = string(name(DM))

        fig = Makie.Figure()
        ax  = Makie.Axis(fig[1, 1]; xlabel = xlabel, ylabel = ylabel, title = ttl, limits=(HyperCube(extrema(xdata(DM)); Padding)[1], nothing))

        Makie.plot!(ax, Data(DM), Val(:Default), xpositions; kwargs...)

        X = (ydim(DM) ≤ Npoints(DM)) ? DomainSamples(extrema(xdata(DM)); N=401) : xdata(DM)
        Y = InformationGeometry.predictedY(DM, mle, X)

        # color palette
        palette_colors = Makie.wong_colors()

        # Prepare line labels + colors
        RSEs = ResidualStandardError(DM, mle)
        RSEs = isnothing(RSEs) ? nothing : convert.(Float64, RSEs)

        # ------------------------------------------------------------------
        # Plot the fit lines
        # ------------------------------------------------------------------
        if ydim(DM) == 1
            lbl = "Fit" * (isnothing(RSEs) ? "" : " with RSE≈$(round(RSEs[1]; sigdigits=3))")
            Makie.lines!(ax, X, Y; color = :red, linewidth = 2, linestyle = :solid, label = lbl)
        elseif ydim(DM) ≤ Npoints(DM)
            # One series per y-component
            for i in 1:ydim(DM)
                lbl = ynames(DM)[i] * " Fit" *
                    (isnothing(RSEs) ? "" : " with RSE≈$(round(RSEs[i]; sigdigits=3))")
                Makie.lines!(ax, X, view(Y, :, i);
                    color     = palette_colors[mod1(i, length(palette_colors))],
                    linewidth = 2,
                    linestyle = :solid,
                    label     = lbl)
            end
        else
            # One series per x-component
            for (i, xval) in enumerate(xdata(DM))
                lbl = "Fit for $(xnames(DM)[1])=$(round(xval; sigdigits=3))"
                Makie.lines!(ax, X, Y[i];
                    color     = palette_colors[mod1(i, length(palette_colors))],
                    linewidth = 2,
                    linestyle = :solid,
                    label     = lbl)
            end
        end

        if any(Confnum .> 0)
            F = FisherMetric(DM, mle)
            if PlotVariance || !NotPosDef(F)
                for (j, Conf) in enumerate(Confnum[Confnum .> 0])
                    if ydim(DM) == 1
                        SqrtVar = VariancePropagation(
                            DM, mle,
                            InvChisqCDF(dof, ConfVol(Conf)) * pinv(F);
                            Validation = Validation,
                            Confnum    = Conf,
                            dof        = dof
                        )(Windup(X, xdim(DM)))

                        Makie.band!(ax, X, Y .- SqrtVar, Y .+ SqrtVar;
                            color = (palette_colors[mod1(4+j, length(palette_colors))], 0.25),
                            label = "Linearized $(Conf)σ $((Validation ? "Validation" : "Conf.")) Band")
                    else
                        SqrtVar = VariancePropagation(
                            DM, mle,
                            InvChisqCDF(dof, ConfVol(Conf)) * pinv(F);
                            Validation = Validation,
                            Confnum    = Conf,
                            dof        = dof
                        )(Windup(X, xdim(DM))) .|> x -> Diagonal(x).diag

                        for i in 1:ydim(DM)
                            clr = palette_colors[mod1(i*ydim(DM)+j, length(palette_colors))]
                            Makie.band!(ax, X,
                                view(Y, :, i) .- getindex.(SqrtVar, i),
                                view(Y, :, i) .+ getindex.(SqrtVar, i);
                                color = (clr, 0.25),
                                label = "Linearized $(Conf)σ $((Validation ? "Validation" : "Conf.")) Band")
                        end
                    end
                end
            end
        end;    fig
    end

    """
        Makie.plot(DS::AbstractDataSet, positions=xdata(DS))
    Outer wrapper: chooses a Val depending on xdim(DS).
    """
    function Makie.plot(DS::AbstractDataSet, positions::AbstractVector{<:Number}=xdata(DS); kwargs...)
        if xdim(DS) == 1
            return Makie.plot(DS, Val(:Default), positions; kwargs...)
        elseif xdim(DS) ≤ 10
            return Makie.plot(DS, Val(:ManyPredictors), positions; kwargs...)  # not implemented here
        else
            return Makie.plot(DS, Val(:IgnoreX), positions; kwargs...)         # not implemented here
        end
    end

    """
        Makie.plot(DS::AbstractDataSet, ::Val{:Default}, xpositions=xdata(DS))
    Plots 1D predictors vs data with error bars (if present).
    """
    function Makie.plot(DS::AbstractDataSet, V::Val{:Default}, xpositions::AbstractVector{<:Number}=xdata(DS); Padding::Real=0.05, kwargs...)
        @assert xdim(DS) == 1 "Use Val(:ManyPredictors) or Val(:IgnoreX) for xdim != 1"

        # Axis labels / title
        xlabel = (ydim(DS) > Npoints(DS) ? "Positions" : xnames(DS)[1])
        ylabel = (ydim(DS) == 1 ? ynames(DS)[1] : "Observations")
        ttl    = string(name(DS))

        fig = Makie.Figure()
        ax  = Makie.Axis(fig[1,1]; xlabel = xlabel, ylabel = ylabel, title = ttl, limits=(HyperCube(extrema(xdata(DS)); Padding)[1], nothing))
        Makie.plot!(ax, DS, V, xpositions; kwargs...)
        fig
    end
    function Makie.plot!(ax, DS::AbstractDataSet, ::Val{:Default}, xpositions::AbstractVector{<:Number}=xdata(DS); palette_colors = Makie.wong_colors(), kwargs...)
        @assert xdim(DS) == 1 "Use Val(:ManyPredictors) or Val(:IgnoreX) for xdim != 1"

        # compute sigmas
        ysig = ysigma(DS)
        Σ_y  = ysig isa AbstractVector ? ysig : sqrt.(Diagonal(ysig).diag)
        xsig = xsigma(DS)
        Σ_x  = HasXerror(DS) ? (xsig isa AbstractVector ? xsig : sqrt.(Diagonal(xsig).diag)) : nothing

        if ydim(DS) == 1
            # single series
            yvals = ydata(DS)
            Makie.scatter!(ax, xpositions, yvals; label = "Data", color=palette_colors[1], kwargs...)
            Makie.errorbars!(ax, xpositions, yvals, Σ_y; color=palette_colors[1], kwargs...)
            # errors
            HasXerror(DS) && Makie.errorbars!(ax, xpositions, yvals, Σ_y; direction=:x, color=palette_colors[1], kwargs...)
        elseif ydim(DS) ≤ Npoints(DS)
            # one series per y-component
            yvals = Unpack(Windup(ydata(DS), ydim(DS)))
            yerrs = Unpack(Windup(Σ_y, ydim(DS)))
            for i in 1:ydim(DS)
                Makie.scatter!(ax, xpositions, yvals[:, i];
                        color = palette_colors[mod1(i, length(palette_colors))],
                        label = "Data: $(ynames(DS)[i])", kwargs...)
                if HasXerror(DS)
                    Makie.errorbars!(ax, xpositions, yvals[:, i], yerrs[:, i], Σ_x; color = palette_colors[mod1(i, length(palette_colors))], kwargs...)
                else
                    Makie.errorbars!(ax, xpositions, yvals[:, i], yerrs[:, i]; color = palette_colors[mod1(i, length(palette_colors))], kwargs...)
                end
            end
        else
            # one series per x-component (transposed)
            yvals = transpose(Unpack(Windup(ydata(DS), ydim(DS))))
            yerrs = transpose(Unpack(Windup(Σ_y, ydim(DS))))
            for i in 1:length(xdata(DS))
                lbl = "Data for $(xnames(DS)[1])=$(round(xdata(DS)[i]; sigdigits=3))"
                Makie.scatter!(ax, xpositions, yvals[i, :];
                        color = palette_colors[mod1(i, length(palette_colors))],
                        label = lbl, kwargs...)
                # no robust way to incorporate xpositions if different from xdata
                Makie.errorbars!(ax, xpositions, yvals[i, :], yerrs[i, :]; color = palette_colors[mod1(i, length(palette_colors))], kwargs...)
            end
        end

        Makie.axislegend(ax; position = :rb)
        return ax
    end

    """
        Makie.plot(DS::AbstractDataSet, ::Val{:ManyPredictors}; rectangular = (xdim(DS) ≤ 10))
    Plot multiple (≈10) independent variables separately. If `rectangular` is true,
    create a ydim×xdim grid of subplots; otherwise stack the subplots in one column.
    """
    function Makie.plot(DS::AbstractDataSet, ::Val{:ManyPredictors}; rectangular::Bool = (xdim(DS) ≤ 10))

        # overall figure + optional “title” label
        fig = Makie.Figure()
        fig[0, :] = Makie.Label(fig, string(name(DS)); fontsize = 18, tellheight = true)

        # error magnitudes
        ysig = ysigma(DS)
        Σ_y  = ysig isa AbstractVector ? ysig : sqrt.(Diagonal(ysig).diag)
        xsig = xsigma(DS)
        Σ_x  = HasXerror(DS) ? (xsig isa AbstractVector ? xsig : sqrt.(Diagonal(xsig).diag)) : nothing

        # helper to place axes
        function place_axis(fig, idx, totalrows, totalcols)
            r = fld(idx-1, totalcols) + 1
            c = mod(idx-1, totalcols) + 1
            return Makie.Axis(fig[r, c])
        end

        # rectangular grid
        if rectangular
            nr, nc = ydim(DS), xdim(DS)
            # set column widths / row heights as needed
            for j in 1:ydim(DS)
                for i in 1:xdim(DS)
                    ax = Makie.Axis(fig[j, i]; xlabel = xnames(DS)[i], ylabel = ynames(DS)[j])
                    # data
                    xvals = getindex.(WoundX(DS), i)
                    yvals = getindex.(WoundY(DS), j)
                    Makie.scatter!(ax, xvals, yvals; color = :black, label = "Data")
                    # errors
                    yerr = getindex.(Windup(Σ_y, ydim(DS)), j)
                    if isnothing(Σ_x)
                        Makie.errorbars!(ax, xvals, yvals, yerr)
                    else
                        xerr = getindex.(Windup(Σ_x, xdim(DS)), i)
                        Makie.errorbars!(ax, xvals, yvals, yerr, xerr)
                    end
                end
            end
        else
            # stack all subplots in one column
            idx = 1
            total = xdim(DS) * ydim(DS)
            grid = GridLayout(fig[1, 1]; nrow = total, ncol = 1)
            for j in 1:ydim(DS)
                for i in 1:xdim(DS)
                    ax = Makie.Axis(grid[idx, 1]; xlabel = xnames(DS)[i], ylabel = ynames(DS)[j])
                    xvals = getindex.(WoundX(DS), i)
                    yvals = getindex.(WoundY(DS), j)
                    Makie.scatter!(ax, xvals, yvals; color = :black, label = "Data")
                    yerr = getindex.(Windup(Σ_y, ydim(DS)), j)
                    if isnothing(Σ_x)
                        Makie.errorbars!(ax, xvals, yvals, yerr)
                    else
                        xerr = getindex.(Windup(Σ_x, xdim(DS)), i)
                        Makie.errorbars!(ax, xvals, yvals, yerr, xerr)
                    end
                    idx += 1
                end
            end
        end;    fig
    end

    """
        Makie.plot(DS::AbstractDataSet, ::Val{:IgnoreX}; collapse = (ydim(DS) ≤ 5))
    Plot data points in order, ignoring the actual x-values. If `collapse` is true,
    all y-dimensions are overlaid on a single axis; otherwise each y-dimension gets
    its own subplot.
    """
    function Makie.plot(DS::AbstractDataSet, ::Val{:IgnoreX}; collapse::Bool = (ydim(DS) ≤ 5))

        fig = Makie.Figure()
        fig[1, :] = Makie.Label(fig, string(name(DS)); fontsize = 18, tellheight = true)

        # y-error magnitudes
        ysig = ysigma(DS)
        Σ_y  = ysig isa AbstractVector ? ysig : sqrt.(Diagonal(ysig).diag)

        palette_colors = Makie.wong_colors()

        if collapse
            ax = Makie.Axis(fig[1, 1]; xlabel = "Data Point Index", ylabel = "Observations")
            for j in 1:ydim(DS)
                yvals = getindex.(WoundY(DS), j)
                yerr  = getindex.(Windup(Σ_y, ydim(DS)), j)
                xvals = 1:length(yvals)
                Makie.scatter!(ax, xvals, yvals;
                        color = palette_colors[mod1(j, length(palette_colors))],
                        label = "Data: $(ynames(DS)[j])")
                Makie.errorbars!(ax, xvals, yvals, yerr;
                        color = palette_colors[mod1(j, length(palette_colors))])
            end
            Makie.axislegend(ax; position = :rb)
        else
            grid = GridLayout(fig[1, 1]; nrow = ydim(DS), ncol = 1)
            for j in 1:ydim(DS)
                ax = Makie.Axis(grid[j, 1];
                        xlabel = "Data Point Index",
                        ylabel = ynames(DS)[j])
                yvals = getindex.(WoundY(DS), j)
                yerr  = getindex.(Windup(Σ_y, ydim(DS)), j)
                xvals = 1:length(yvals)
                Makie.scatter!(ax, xvals, yvals; color = :black, label = "Data")
                Makie.errorbars!(ax, xvals, yvals, yerr)
            end
        end;    fig
    end
end # module