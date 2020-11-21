

function SaveAdaptive(sol::ODESolution, N::Int=500; curvature = 0.003, Ntol=0.08)
    Tspan = (sol.t[1],sol.t[end]);      maxiter=30
    for _ in 1:maxiter
        T = reduce(vcat,[refine_grid(x->sol(x)[i],Tspan; max_curvature=curvature)[1] for i in 1:length(sol.u[1])]) |> unique |> sort
        if length(T) > N
            curvature *= 1.2
        elseif length(T) < (1-Ntol) * N
            curvature *= 0.5
        else
            return Homogenize(T,N)
        end
    end
    throw("SaveAdaptive: DNF in $maxiter iterations.")
end

Homogenize(sol::ODESolution,N::Int=500) = Homogenize(sol.t,N)
function Homogenize(V::AbstractVector,N::Int=500)
    Ts = unique(V)
    for i in 1:(N-length(Ts))
        s = findmax(diff(Ts))[2]
        insert!(Ts,s+1,Ts[s] + (Ts[s+1]-Ts[s])/2)
    end;    Ts
end

Dehomogenize(sol::ODESolution,N::Int=500) = Dehomogenize(sol.t,N)
function Dehomogenize(V::AbstractVector,N::Int=500)
    Ts = unique(V)
    for i in 1:(length(Ts)-N)
        s = findmin(diff(Ts))[2]
        deleteat!(Ts,s+1)
    end;    Ts
end

"""
    SaveConfidence(sols::Vector{<:ODESolution}, N::Int=500; sigdigits::Int=7, adaptive::Bool=true) -> Matrix
Returns a `Matrix` of with `N` rows corresponding to the number of evaluations of each `ODESolution` in `sols`.
The colums correspond to the various components of the evaluated solutions.
E.g. for an `ODESolution` with 3 components, the 4. column in the `Matrix` corresponds to the evaluated first components of `sols[2]`.
"""
function SaveConfidence(sols::Vector{<:ODESolution}, N::Int=500; sigdigits::Int=7, adaptive::Bool=true)
    mapreduce(sol->SaveConfidence(sol,N; sigdigits=sigdigits,adaptive=adaptive), hcat, sols)
end
function SaveConfidence(sol::ODESolution, N::Int=500; sigdigits::Int=7, adaptive::Bool=true)
    Ts = adaptive ? SaveAdaptive(sol, N) : range(sol.t[1], sol.t[end]; length=N)
    round.(Unpack(map(sol,Ts)); sigdigits=sigdigits)
end

function SaveConfidence(Planes::Vector{<:Plane}, sols::Vector{<:ODESolution}, N::Int=500; sigdigits::Int=7, adaptive::Bool=true)
    length(Planes) != length(sols) && throw("Dimensional Mismatch: length(Planes)=$(length(Planes)), length(sols)=$(length(sols)).")
    mapreduce((PL,sol)->SaveConfidence(PL, sol, N; sigdigits=sigdigits,adaptive=adaptive), hcat, Planes, sols)
end
function SaveConfidence(PL::Plane, sols::Vector{<:ODESolution}, N::Int=500; sigdigits::Int=7, adaptive::Bool=true)
    mapreduce(sol->SaveConfidence(PL,sol,N; sigdigits=sigdigits,adaptive=adaptive), hcat, sols)
end
function SaveConfidence(PL::Plane, sol::ODESolution, N::Int=500; sigdigits::Int=7, adaptive::Bool=true)
    Ts = adaptive ? SaveAdaptive(sol, N) : range(sol.t[1], sol.t[end]; length=N)
    round.(Deplanarize(PL,sol,Ts); sigdigits=sigdigits)
end
# function SaveConfidence(sols::Vector{<:ODESolution}, N::Int=500; sigdigits::Int=7, adaptive::Bool=true)
#     d = length(sols[1].u[1]);    Res = Array{Float64}(undef,N,d*length(sols))
#     for (i,sol) in enumerate(sols)
#         Res[:,((i-1)*d+1):(d*i)] = SaveConfidence(sol, N; sigdigits=sigdigits, adaptive=adaptive)
#     end;    Res
# end
# function SaveConfidence(sol::ODESolution, N::Int=500; sigdigits::Int=7, adaptive::Bool=true)
#     T = adaptive ? SaveAdaptive(sol, N) : range(sol.t[1], sol.t[end]; length=N)
#     round.(Unpack(sol.(T)); sigdigits=sigdigits)
# end

"""
    SaveGeodesics(sols::Vector{<:ODESolution}, N::Int=500; sigdigits::Int=7, adaptive::Bool=true) -> Matrix
Returns a `Matrix` of with `N` rows corresponding to the number of evaluations of each `ODESolution` in `sols`.
The colums correspond to the various components of the evaluated solutions.
Since the solution objects for geodesics contain the velocity as the second half of the components, only the first half of the components is saved.
"""
function SaveGeodesics(sols::Vector{<:ODESolution},N::Int=500; sigdigits::Int=7,adaptive::Bool=true)
    d = length(sols[1].u[1])/2 |> Int;      Res = Array{Float64}(undef,N,d*length(sols))
    for (i,sol) in enumerate(sols)
        Res[:,((i-1)*d+1):(d*i)] = SaveGeodesics(sol, N; sigdigits=sigdigits, adaptive=adaptive)
    end;    Res
end
function SaveGeodesics(sol::ODESolution, N::Int=500; sigdigits::Int=7, adaptive::Bool=true)
    SaveConfidence(sol, N; sigdigits=sigdigits, adaptive=adaptive)[:, 1:Int(length(sol.u[1])/2)]
end

"""
    SaveDataSet(DS::DataSet; sigdigits::Int=0)
Returns a `DataFrame` whose columns respectively constitute the x-values, y-values and standard distributions associated with the data points.
For `sigdigits > 0` the values are rounded to the specified number of significant digits.
"""
function SaveDataSet(DS::AbstractDataSet; sigdigits::Int=0)
    !(xdim(DS) == ydim(DS) == Int(size(sigma(DS),1)/Npoints(DS))) && throw("Not programmed yet.")
    sig = sigma(DS)
    !(typeof(sig) <: AbstractVector) && throw("Sigma not a vector, but instead $(typeof(sig)).")
    if sigdigits < 1
        return DataFrame([xdata(DS) ydata(DS) sigma(DS)])
    else
        return DataFrame(round.([xdata(DS) ydata(DS) sigma(DS)]; sigdigits=sigdigits))
    end
end
SaveDataSet(DM::AbstractDataModel; sigdigits::Int=0) = SaveDataSet(Data(DM); sigdigits=sigdigits)



# As soon as https://github.com/JuliaPlots/PlotUtils.jl/pull/103 is merged:
## import PlotUtils: adapted_grid
# function refine_grid(f::Function, minmax::Tuple{Real, Real}; max_recursions::Int = 10, max_curvature::Real = 0.003)
#     PlotUtils.adapted_grid(f, minmax; max_recursions=max_recursions, max_curvature=max_curvature)
# end

# Adaptation from PlotUtils.jl
function refine_grid(f, minmax::Tuple{Real, Real}; max_recursions = 10, max_curvature = 0.003)
    if minmax[1] > minmax[2]
        throw(ArgumentError("interval must be given as (min, max)"))
    elseif minmax[1] == minmax[2]
        x = minmax[1]
        return [x], [f(x)]
    end

    # Initial number of points
    n_points = 21
    n_intervals = n_points ÷ 2
    @assert isodd(n_points)

    xs = collect(range(minmax[1]; stop=minmax[2], length=n_points))
    # Move the first and last interior points a bit closer to the end points
    xs[2] = xs[1] + (xs[2] - xs[1]) * 0.25
    xs[end-1] = xs[end] - (xs[end] - xs[end-1]) * 0.25

    # Wiggle interior points a bit to prevent aliasing and other degenerate cases
    rng = MersenneTwister(1337)
    rand_factor = 0.05
    for i in 2:length(xs)-1
        xs[i] += rand_factor * 2 * (rand(rng) - 0.5) * (xs[i+1] - xs[i-1])
    end

    n_tot_refinements = zeros(Int, n_intervals)

    # Replace DomainErrors with NaNs
    g = function(x)
        local y
        try
            y = f(x)
        catch err
            if err isa DomainError
                y = NaN
            else
                rethrow(err)
            end
        end
        return y
    end
    # We evaluate the function on the whole interval
    fs = g.(xs)
    while true
        curvatures = zeros(n_intervals)
        active = falses(n_intervals)
        isfinite_f = isfinite.(fs)
        min_f, max_f = any(isfinite_f) ? extrema(fs[isfinite_f]) : (0.0, 0.0)
        f_range = max_f - min_f
        # Guard against division by zero later
        if f_range == 0 || !isfinite(f_range)
            f_range = one(f_range)
        end
        # Skip first and last interval
        for interval in 1:n_intervals
            p = 2 * interval
            if n_tot_refinements[interval] >= max_recursions
                # Skip intervals that have been refined too much
                active[interval] = false
            elseif !all(isfinite.(fs[[p-1,p,p+1]]))
                active[interval] = true
            else
                tot_w = 0.0
                # Do a small convolution
                for (q,w) in ((-1, 0.25), (0, 0.5), (1, 0.25))
                    interval == 1 && q == -1 && continue
                    interval == n_intervals && q == 1 && continue
                    tot_w += w
                    i = p + q
                    # Estimate integral of second derivative over interval, use that as a refinement indicator
                    # https://mathformeremortals.wordpress.com/2013/01/12/a-numerical-second-derivative-from-three-points/
                    curvatures[interval] += abs(2 * ((fs[i+1] - fs[i]) / ((xs[i+1]-xs[i]) * (xs[i+1]-xs[i-1]))
                                                    -(fs[i] - fs[i-1]) / ((xs[i]-xs[i-1]) * (xs[i+1]-xs[i-1])))
                                                    * (xs[i+1] - xs[i-1])^2) / f_range * w
                end
                curvatures[interval] /= tot_w
                # Only consider intervals with a high enough curvature
                active[interval] = curvatures[interval] > max_curvature
            end
        end
        # Approximate end intervals as being the same curvature as those next to it.
        # This avoids computing the function in the end points
        curvatures[1] = curvatures[2]
        active[1] = active[2]
        curvatures[end] = curvatures[end-1]
        active[end] = active[end-1]
        if all(x -> x >= max_recursions, n_tot_refinements[active])
            break
        end
        n_target_refinements = n_intervals ÷ 2
        interval_candidates = collect(1:n_intervals)[active]
        n_refinements = min(n_target_refinements, length(interval_candidates))
        perm = sortperm(curvatures[active])
        intervals_to_refine = sort(interval_candidates[perm[length(perm) - n_refinements + 1:end]])
        n_intervals_to_refine = length(intervals_to_refine)
        n_new_points = 2*length(intervals_to_refine)

        # Do division of the intervals
        new_xs = zeros(eltype(xs), n_points + n_new_points)
        new_fs = zeros(eltype(fs), n_points + n_new_points)
        new_tot_refinements = zeros(Int, n_intervals + n_intervals_to_refine)
        k = 0
        kk = 0
        for i in 1:n_points
            if iseven(i) # This is a point in an interval
                interval = i ÷ 2
                if interval in intervals_to_refine
                    kk += 1
                    new_tot_refinements[interval - 1 + kk] = n_tot_refinements[interval] + 1
                    new_tot_refinements[interval + kk] = n_tot_refinements[interval] + 1

                    k += 1
                    new_xs[i - 1 + k] = (xs[i] + xs[i-1]) / 2
                    new_fs[i - 1 + k] = g(new_xs[i-1 + k])

                    new_xs[i + k] = xs[i]
                    new_fs[i + k] = fs[i]

                    new_xs[i + 1 + k] = (xs[i+1] + xs[i]) / 2
                    new_fs[i + 1 + k] = g(new_xs[i + 1 + k])
                    k += 1
                else
                    new_tot_refinements[interval + kk] = n_tot_refinements[interval]
                    new_xs[i + k] = xs[i]
                    new_fs[i + k] = fs[i]
                end
            else
                new_xs[i + k] = xs[i]
                new_fs[i + k] = fs[i]
            end
        end
        xs = new_xs
        fs = new_fs
        n_tot_refinements = new_tot_refinements
        n_points = n_points + n_new_points
        n_intervals = n_points ÷ 2
    end
    return xs, fs
end
