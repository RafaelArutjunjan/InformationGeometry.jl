


SaveAdaptive(sol::AbstractODESolution, N::Int=500; kwargs...) = SaveAdaptive(sol, (sol.t[1],sol.t[end]); N=N, kwargs...)
function SaveAdaptive(sol::Union{Function,AbstractODESolution}, Tspan::Tuple{Real,Real}; N::Int=500, curvature::Real=0.003, Ntol::Real=0.08, maxiter::Int=30)
    @assert Tspan[1] < Tspan[2]
    test = sol(Tspan[1])
    for _ in 1:maxiter
        T = reduce(vcat, [PlotUtils.adapted_grid(x->sol(x)[i], Tspan; max_curvature=curvature)[1] for i in eachindex(test)]) |> unique! |> sort!
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


Homogenize(sol::AbstractODESolution, N::Int=500) = Homogenize(sol.t, N)
function Homogenize(V::AbstractVector, N::Int=500)
    Ts = unique(V)
    for i in 1:(N-length(Ts))
        s = FindMaxDiff(Ts)[2]
        insert!(Ts,s+1,Ts[s] + (Ts[s+1]-Ts[s])/2)
    end;    Ts
end

Dehomogenize(sol::AbstractODESolution, N::Int=500) = Dehomogenize(sol.t, N)
function Dehomogenize(V::AbstractVector, N::Int=500)
    Ts = unique(V)
    for i in 1:(length(Ts)-N)
        s = FindMinDiff(Ts)[2]
        deleteat!(Ts,s+1)
    end;    Ts
end

"""
    SaveConfidence(sols::AbstractVector{<:AbstractODESolution}, N::Int=500; sigdigits::Int=7, adaptive::Bool=true) -> Matrix
    SaveConfidence(Planes::AbstractVector{<:Plane}, sols::AbstractVector{<:AbstractODESolution}, N::Int=500; sigdigits::Int=7, adaptive::Bool=true) -> Matrix
Returns a `Matrix` of with `N` rows corresponding to the number of evaluations of each `ODESolution` in `sols`.
The colums correspond to the various components of the evaluated solutions.
E.g. for an `ODESolution` with 3 components, the 4. column in the `Matrix` corresponds to the evaluated first components of `sols[2]`.
"""
function SaveConfidence(sols::AbstractVector{<:AbstractODESolution}, N::Int=500; sigdigits::Int=7, adaptive::Bool=true)
    mapreduce(sol->SaveConfidence(sol,N; sigdigits=sigdigits,adaptive=adaptive), hcat, sols)
end
function SaveConfidence(sol::AbstractODESolution, N::Int=500; sigdigits::Int=7, adaptive::Bool=true)
    Ts = adaptive ? SaveAdaptive(sol, N) : range(sol.t[1], sol.t[end]; length=N)
    round.(Unpack(map(sol,Ts)); sigdigits=sigdigits)
end

function SaveConfidence(Planes::AbstractVector{<:Plane}, sols::AbstractVector{<:AbstractODESolution}, N::Int=500; sigdigits::Int=7, adaptive::Bool=true)
    length(Planes) != length(sols) && throw("Dimensional Mismatch: length(Planes)=$(length(Planes)), length(sols)=$(length(sols)).")
    mapreduce((PL,sol)->SaveConfidence(PL, sol, N; sigdigits=sigdigits,adaptive=adaptive), hcat, Planes, sols)
end
function SaveConfidence(PL::Plane, sols::AbstractVector{<:AbstractODESolution}, N::Int=500; sigdigits::Int=7, adaptive::Bool=true)
    mapreduce(sol->SaveConfidence(PL,sol,N; sigdigits=sigdigits,adaptive=adaptive), hcat, sols)
end
function SaveConfidence(PL::Plane, sol::AbstractODESolution, N::Int=500; sigdigits::Int=7, adaptive::Bool=true)
    Ts = adaptive ? SaveAdaptive(sol, N) : range(sol.t[1], sol.t[end]; length=N)
    round.(Deplanarize(PL,sol,Ts); sigdigits=sigdigits)
end
# function SaveConfidence(sols::AbstractVector{<:AbstractODESolution}, N::Int=500; sigdigits::Int=7, adaptive::Bool=true)
#     d = length(sols[1].u[1]);    Res = Array{Float64}(undef,N,d*length(sols))
#     for (i,sol) in enumerate(sols)
#         Res[:,((i-1)*d+1):(d*i)] = SaveConfidence(sol, N; sigdigits=sigdigits, adaptive=adaptive)
#     end;    Res
# end
# function SaveConfidence(sol::AbstractODESolution, N::Int=500; sigdigits::Int=7, adaptive::Bool=true)
#     T = adaptive ? SaveAdaptive(sol, N) : range(sol.t[1], sol.t[end]; length=N)
#     round.(Unpack(sol.(T)); sigdigits=sigdigits)
# end

"""
    SaveGeodesics(sols::AbstractVector{<:AbstractODESolution}, N::Int=500; sigdigits::Int=7, adaptive::Bool=true) -> Matrix
Returns a `Matrix` of with `N` rows corresponding to the number of evaluations of each `ODESolution` in `sols`.
The colums correspond to the various components of the evaluated solutions.
Since the solution objects for geodesics contain the velocity as the second half of the components, only the first half of the components is saved.
"""
function SaveGeodesics(sols::AbstractVector{<:AbstractODESolution}, N::Int=500; sigdigits::Int=7, adaptive::Bool=true)
    d = length(sols[1].u[1])/2 |> Int;      Res = Array{Float64}(undef,N,d*length(sols))
    for (i,sol) in enumerate(sols)
        Res[:,((i-1)*d+1):(d*i)] = SaveGeodesics(sol, N; sigdigits=sigdigits, adaptive=adaptive)
    end;    Res
end
function SaveGeodesics(sol::AbstractODESolution, N::Int=500; sigdigits::Int=7, adaptive::Bool=true)
    SaveConfidence(sol, N; sigdigits=sigdigits, adaptive=adaptive)[:, 1:Int(length(sol.u[1])/2)]
end

"""
    SaveDataSet(DS::DataSet; sigdigits::Int=0)
Returns a `DataFrame` whose columns respectively constitute the x-values, y-values and standard distributions associated with the data points.
For `sigdigits > 0` the values are rounded to the specified number of significant digits.
"""
function SaveDataSet(DS::AbstractDataSet; sigdigits::Int=0, kwargs...)
    df = DataFrame(DS; kwargs...);    sigdigits > 0 ? round.(df; sigdigits=sigdigits) : df
end
SaveDataSet(DM::AbstractDataModel; kwargs...) = SaveDataSet(Data(DM); kwargs...)

DataFrames.DataFrame(DM::AbstractDataModel; kwargs...) = DataFrame(Data(DM); kwargs...)

function DataFrames.DataFrame(DS::DataSet; kwargs...)
    @assert ysigma(DS) isa AbstractVector
    M = [Unpack(WoundX(DS)) UnpackWindup(ydata(DS), ydim(DS)) UnpackWindup(ysigma(DS), ydim(DS))]
    DataFrame(M, vcat(xnames(DS), ynames(DS), ynames(DS) .* "_σ"); kwargs...)
end
function DataFrames.DataFrame(DS::DataSetExact; kwargs...)
    @assert ysigma(DS) isa AbstractVector && xsigma(DS) isa AbstractVector
    !HasXerror(DS) && return DataFrame(DataSet(DS))
    M = [Unpack(WoundX(DS)) UnpackWindup(xsigma(DS), xdim(DS)) UnpackWindup(ydata(DS), ydim(DS)) UnpackWindup(ysigma(DS), ydim(DS))]
    DataFrame(M, vcat(xnames(DS), xnames(DS) .* "_σ", ynames(DS), ynames(DS) .* "_σ"); kwargs...)
end


function DataFrames.DataFrame(CDS::CompositeDataSet; kwargs...)
    @assert ysigma(CDS) isa AbstractVector && xsigma(CDS) isa AbstractVector
    @assert !HasXerror(CDS) "Not programmed for x-uncertainties yet."
    Vectorize(x::AbstractVector) = x;   Vectorize(x::Union{Number,Missing}) = [x]
    # How many observations per WoundX(CDS) in subdatasets?
    _Counts(CDS::CompositeDataSet) = [[count(isequal(x), Windup(xdata(DS),xdim(DS))) for x in WoundX(CDS)] for DS in Data(CDS)]
    # Check if there are any missings or if can convert to DS
    counts = _Counts(CDS);    maxcounts = map((args...)->maximum([args...]), counts...)
    Xmat = Vector{eltype(xdata(CDS))}[]
    for i in eachindex(WoundX(CDS))
        for c in 1:maxcounts[i]
            push!(Xmat, WoundX(CDS)[i])
        end
    end;    Xmat = Unpack(Xmat) # rows sorted such that repeating are adjacent for loop later
    ResY = Vector{Vector{Union{Missing,eltype(ydata(CDS))}}}(undef, 0)
    ResYσ = Vector{Vector{Union{Missing,eltype(ysigma(CDS))}}}(undef, 0)
    for (j,DS) in enumerate(Data(CDS))
        ydat = Windup(ydata(DS),ydim(DS));        ysig = Windup(ysigma(DS),ydim(DS))
        for (i,X) in enumerate(WoundX(CDS))
            if counts[j][i] > 0
                ind = 0
                for c in 1:counts[j][i]
                    ind = findnext(isequal(X), WoundX(DS), ind+1)
                    push!(ResY, Vectorize(ydat[ind]))
                    push!(ResYσ, Vectorize(ysig[ind]))
                end
            end
            for _ in 1:(maxcounts[i]-counts[j][i])
                push!(ResY, fill(missing, ydim(DS)))
                push!(ResYσ, fill(missing, ydim(DS)))
            end
        end
    end
    M = [Xmat UnpackWindup(reduce(vcat, ResY), ydim(CDS)) UnpackWindup(reduce(vcat, ResYσ), ydim(CDS))]
    DataFrame(M, vcat(xnames(CDS), ynames(CDS), ynames(CDS) .* "_σ"); kwargs...)
end
