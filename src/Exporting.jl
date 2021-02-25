

function SaveAdaptive(sol::AbstractODESolution, N::Int=500; curvature = 0.003, Ntol=0.08)
    Tspan = (sol.t[1],sol.t[end]);      maxiter=30
    for _ in 1:maxiter
        T = reduce(vcat,[PlotUtils.adapted_grid(x->sol(x)[i],Tspan; max_curvature=curvature)[1] for i in 1:length(sol.u[1])]) |> unique |> sort
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
        s = findmax(diff(Ts))[2]
        insert!(Ts,s+1,Ts[s] + (Ts[s+1]-Ts[s])/2)
    end;    Ts
end

Dehomogenize(sol::AbstractODESolution, N::Int=500) = Dehomogenize(sol.t, N)
function Dehomogenize(V::AbstractVector, N::Int=500)
    Ts = unique(V)
    for i in 1:(length(Ts)-N)
        s = findmin(diff(Ts))[2]
        deleteat!(Ts,s+1)
    end;    Ts
end

"""
    SaveConfidence(sols::Vector{<:AbstractODESolution}, N::Int=500; sigdigits::Int=7, adaptive::Bool=true) -> Matrix
    SaveConfidence(Planes::Vector{<:Plane}, sols::Vector{<:AbstractODESolution}, N::Int=500; sigdigits::Int=7, adaptive::Bool=true) -> Matrix
Returns a `Matrix` of with `N` rows corresponding to the number of evaluations of each `ODESolution` in `sols`.
The colums correspond to the various components of the evaluated solutions.
E.g. for an `ODESolution` with 3 components, the 4. column in the `Matrix` corresponds to the evaluated first components of `sols[2]`.
"""
function SaveConfidence(sols::Vector{<:AbstractODESolution}, N::Int=500; sigdigits::Int=7, adaptive::Bool=true)
    mapreduce(sol->SaveConfidence(sol,N; sigdigits=sigdigits,adaptive=adaptive), hcat, sols)
end
function SaveConfidence(sol::AbstractODESolution, N::Int=500; sigdigits::Int=7, adaptive::Bool=true)
    Ts = adaptive ? SaveAdaptive(sol, N) : range(sol.t[1], sol.t[end]; length=N)
    round.(Unpack(map(sol,Ts)); sigdigits=sigdigits)
end

function SaveConfidence(Planes::Vector{<:Plane}, sols::Vector{<:AbstractODESolution}, N::Int=500; sigdigits::Int=7, adaptive::Bool=true)
    length(Planes) != length(sols) && throw("Dimensional Mismatch: length(Planes)=$(length(Planes)), length(sols)=$(length(sols)).")
    mapreduce((PL,sol)->SaveConfidence(PL, sol, N; sigdigits=sigdigits,adaptive=adaptive), hcat, Planes, sols)
end
function SaveConfidence(PL::Plane, sols::Vector{<:AbstractODESolution}, N::Int=500; sigdigits::Int=7, adaptive::Bool=true)
    mapreduce(sol->SaveConfidence(PL,sol,N; sigdigits=sigdigits,adaptive=adaptive), hcat, sols)
end
function SaveConfidence(PL::Plane, sol::AbstractODESolution, N::Int=500; sigdigits::Int=7, adaptive::Bool=true)
    Ts = adaptive ? SaveAdaptive(sol, N) : range(sol.t[1], sol.t[end]; length=N)
    round.(Deplanarize(PL,sol,Ts); sigdigits=sigdigits)
end
# function SaveConfidence(sols::Vector{<:AbstractODESolution}, N::Int=500; sigdigits::Int=7, adaptive::Bool=true)
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
    SaveGeodesics(sols::Vector{<:AbstractODESolution}, N::Int=500; sigdigits::Int=7, adaptive::Bool=true) -> Matrix
Returns a `Matrix` of with `N` rows corresponding to the number of evaluations of each `ODESolution` in `sols`.
The colums correspond to the various components of the evaluated solutions.
Since the solution objects for geodesics contain the velocity as the second half of the components, only the first half of the components is saved.
"""
function SaveGeodesics(sols::Vector{<:AbstractODESolution}, N::Int=500; sigdigits::Int=7, adaptive::Bool=true)
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
function SaveDataSet(DS::AbstractDataSet; sigdigits::Int=0)
    !(xdim(DS) == ydim(DS) == Int(size(sigma(DS),1)/Npoints(DS))) && throw("Not programmed yet.")
    sig = sigma(DS)
    !(typeof(sig) <: AbstractVector) && throw("Sigma not a vector, but instead $(typeof(sig)).")
    if sigdigits < 1
        return DataFrame([xdata(DS) ydata(DS) sigma(DS)], :auto)
    else
        return DataFrame(round.([xdata(DS) ydata(DS) sigma(DS)]; sigdigits=sigdigits), :auto)
    end
end
SaveDataSet(DM::AbstractDataModel; sigdigits::Int=0) = SaveDataSet(Data(DM); sigdigits=sigdigits)
