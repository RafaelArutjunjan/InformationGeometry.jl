

"""
    LinearModel(x::Union{Number,AbstractVector{<:Number}}, θ::AbstractVector{<:Number})
```math
y(x,θ) = θ_{n+1} + x_1 * θ_1 + x_2 * θ_2 + ... + x_n * θ_n
```
"""
LinearModel(x::Union{Number,AbstractVector{<:Number}}, θ::AbstractVector{<:Number}) = dot(view(θ, 1:(length(θ)-1)), x) + θ[end]
"""
    QuadraticModel(x::Union{Number,AbstractVector{<:Number}}, θ::AbstractVector{<:Number})
```math
y(x,θ) = θ_1 * x^2 + θ_2 * x + θ_3
```
"""
QuadraticModel(x::Union{Number,AbstractVector{<:Number}}, θ::AbstractVector{<:Number}) = (n=length(θ);  dot(view(θ,1:((n-1)÷2)), x.^2) + dot(view(θ,(n-1)÷2+1:n-1), x) + θ[end])
"""
    ExponentialModel(x::Union{Number,AbstractVector{<:Number}}, θ::AbstractVector{<:Number})
```math
y(x,θ) = exp(θ_{n+1} + x_1 * θ_1 + x_2 * θ_2 + ... + x_n * θ_n)
```
"""
ExponentialModel = exp∘LinearModel
"""
    SumExponentialsModel(x::Union{Number,AbstractVector{<:Number}}, θ::AbstractVector{<:Number})    
```math
y(x,θ) = θ_{n+1} + exp(x_1 * θ_1) + exp(x_2 * θ_2) + ... + exp(x_n * θ_n)
```
"""
SumExponentialsModel(x::Union{Number,AbstractVector{<:Number}}, θ::AbstractVector{<:Number}) = sum(exp.(view(θ,1:length(θ)-1) .* x)) + θ[end]
"""
    PolynomialModel(n::Int)
Creates a polynomial of degree `n`:
```math
y(x,θ) = θ_1 * x^n + θ_2 * x^{n-1} + ... θ_{n} * x + θ_{n+1}
```
"""
PolynomialModel(degree::Int) = Polynomial(x::Number, θ::AbstractVector{<:Number}) = sum(θ[i] * x^(i-1) for i in 1:(degree+1))


function GetLinearModel(DS::AbstractDataSet)
    ydim(DS) != 1 && return GetGeneralLinearModel(DS)
    Names = "p_(" .* ynames(DS) .* " × " .* xnames(DS) .*")"
    push!(Names, "p_(" * ynames(DS)[1] * " × Offset)")
    ModelMap(LinearModel, (xdim(DS), ydim(DS), xdim(DS)+1); pnames=Names)
end

function GetGeneralLinearModel(DS::AbstractDataSet)
    ydim(DS) == 1 && return GetLinearModel(DS)
    Xdim, Ydim = xdim(DS), ydim(DS)
    NaiveGeneralLinearModel(x::AbstractVector{<:Number}, θ::AbstractVector{T}) where T <: Number = SVector{Ydim, T}(LinearModel(x, p) for p in Iterators.partition(θ, Xdim+1))
    Names = ["p_(" .* ynames(DS)[i] .* " × " .* xnames(DS) .*")" for i in 1:ydim(DS)]
    for (i,series) in enumerate(Names)
        push!(series, "p_(" * ynames(DS)[i] * " × Offset)")
    end
    OptimizeModel(ModelMap(NaiveGeneralLinearModel, nothing, nothing, (xdim(DS), ydim(DS), ydim(DS)*(xdim(DS)+1)), reduce(vcat, Names), Val(true), Val(false), Val(false)); inplace=false)[1]
end




function TimeShiftEmb(xyp::Tuple{Int,Int,Int}, IsCustom::Bool)
    if xyp[1] == 1
        if IsCustom
            TimeShiftC(x::Union{Number, AbstractVector{<:Number}}, p::AbstractVector{<:Number}) = (x .+ p[xyp[3]+1], view(p,1:xyp[3]))
        else
            TimeShift(x::Number, p::AbstractVector{<:Number}) = (x + p[xyp[3]+1], view(p,1:xyp[3]))
        end
    else
        if IsCustom
            TimeShiftNC(x::AbstractVector{<:Number}, p::AbstractVector{<:Number}) = (x .+ view(p, xyp[3]+1:xyp[3]+xyp[1]), view(p,1:xyp[3]))
            TimeShiftNC(x::AbstractVector{<:AbstractVector{<:Number}}, p::AbstractVector{<:Number}) = (broadcast(z->z .+ view(p, xyp[3]+1:xyp[3]+xyp[1]),x), view(p,1:xyp[3]))
        else
            TimeShiftN(x::AbstractVector{<:Number}, p::AbstractVector{<:Number}) = (x .+ view(p, xyp[3]+1:xyp[3]+xyp[1]), view(p,1:xyp[3]))
        end
    end
end
function TimeShiftTransform(model::Function, xyp::Tuple{Int,Int,Int}=ConstructModelxyp(model), IsCustom::Bool=iscustommodel(model), Inplace::Bool=isinplacemodel(model), Dom::HyperCube=5FullDomain(xyp[1], 1.0))
    EmbedModelXP(model, TimeShiftEmb(xyp, IsCustom), Inplace)
end
"""
    TimeShiftTransform(DM::AbstractDataModel)
    TimeShiftTransform(M::ModelMap, xyp::Tuple{Int,Int,Int}=Getxyp(M), IsCustom::Bool=iscustommodel(M), dom::Tuple{<:Real,<:Real}=(-5,5))
Extends the model by adding `xdim(M)` many timeshift parameters `p_time` as the last accepted parameters such that the new model is given by
`newmodel(x,p) = oldmodel(x .+ p[end-xdim(M)+1:end], p[1:end-xdim(M)])`.
"""
function TimeShiftTransform(M::ModelMap, xyp::Tuple{Int,Int,Int}=Getxyp(M), IsCustom::Bool=iscustommodel(M), Inplace::Bool=isinplacemodel(M), Dom::HyperCube=5FullDomain(xyp[1], 1.0))
    ModelMap(TimeShiftTransform(M.Map, xyp, IsCustom, Inplace),
        !isnothing(InDomain(M)) ? InDomain(M)∘ViewElements(1:xyp[3]) : nothing,
        vcat(!isnothing(Domain(M)) ? Domain(M) : FullDomain(xyp[3], Inf), Dom),
        xyp .+ (0,0,xyp[1]),
        vcat(pnames(M), CreateSymbolNames(xyp[1],"Timeshift")),
        Val(isinplacemodel(M)),
        Val(IsCustom),
        name(M) === Symbol() ? name(M) : Symbol("Time-shifted " *string(name(M))),
        M.Meta
    )
end

function TimeShiftTransform(DM::AbstractDataModel; factor::Real=0.85, kwargs...)
    Cube = (C = XCube(DM);    factor*TranslateCube(C, -Center(C)))
    DataModel(Data(DM), TimeShiftTransform(Predictor(DM), (xdim(DM), ydim(DM), pdim(DM)), iscustommodel(Predictor(DM)), isinplacemodel(Predictor(DM)), Cube), 
                [MLE(DM); 1e-8ones(xdim(DM))], LogPrior(DM); kwargs...)
end



"""
    TimeRetardation(t, T_shift, r; t_range=1)
Implements time retardation of the form
```math
(\\log_{10}(10^{r \\cdot t / t_\\text{range}} + 10^{r \\cdot T_\\text{shift} / t_\\text{range}}) - \\log_{10}(1 + 10^{r \\cdot T_\\text{shift} / t_\\text{range}}))/r
```
adapted from
https://www.researchgate.net/publication/340061135_A_New_Approximation_Approach_for_Transient_Differential_Equation_Models
with additional curvature parameter `r`.
"""
function TimeRetardation(t::Union{S, <:AbstractVector{S}}, T_shift::Number, r::Number; t_range::Real=1) where S<:Number
    @. (log10(exp10(r*t/t_range) + exp10(r*T_shift/t_range)) - log10(one(S) + exp10(r*T_shift/t_range)))/r
end

TransientApproximation(; kwargs...) = (args...; Kwargs...) -> TransientApproximation(args...; kwargs..., Kwargs...)
function TransientApproximation(t::Union{<:Number,AbstractVector{<:Number}}, θ::AbstractVector{<:Number}; t_range::Real=1)
    A_sus, A_trans, τ_1, τ_3, τ_2, T_shift, r, offset = θ
    sustained(t_ret) = @. A_sus * (one(eltype(t_ret)) - exp(-t_ret/τ_1))
    transient(t_ret) = @. A_trans * (one(eltype(t_ret)) - exp(-t_ret/τ_3)) * exp(-t_ret/τ_2)
    Res(t_ret) = @. sustained(t_ret) + transient(t_ret) + offset
    @. Res(TimeRetardation(t, T_shift, r; t_range))
end

"""
    GetRetardedTransientFunction(DS::AbstractDataSet; exp10::Bool=false, TimeRangeAdjustment::Bool=true, t_range::Real=(E=extrema(xdata(DS));  E[2]-E[1]))

Implements `RetardedTransientFunction(t::Real, θ::AbstractVector)` from
https://www.researchgate.net/publication/340061135_A_New_Approximation_Approach_for_Transient_Differential_Equation_Models
with additional curvature parameter `r` for time retardation (see also [`TimeRetardation`](@ref)).

Parameters: `A_sus, A_trans, τ_1, τ_1′, τ_2, T_shift, r, offset`

Kwarg `exp10` controls whether rate parameters `τ` and curvature parameter `r` are transformed to log10-scale.
"""
function GetRetardedTransientFunction(DS::AbstractDataSet; AddDomain::Bool=false, exp10::Bool=false, TimeRangeAdjustment::Bool=true, t_range::Real=(E=extrema(xdata(DS));  E[2]-E[1]))
    @assert xdim(DS) == 1
    @assert ydim(DS) == 1 # Relax this later?
    GetRan(X::AbstractVector{<:Number}) = (E = extrema(X); E[2]-E[1])
    yran = GetRan(ydata(DS))
    mintdiff = FindMinDiff(sort(xdata(DS)))[1]
    lb = [-2yran, -2yran, mintdiff/2, mintdiff/2, mintdiff/2, -t_range/5, 0.1, minimum(ydata(DS))-0.2yran]
    ub = [+2yran, +2yran,      2t_range,      2t_range,      2t_range, +t_range/2,  10, maximum(ydata(DS))+0.2yran]
    # τ_1′ > τ_1
    DomainConstraint(X::AbstractVector) = X[4] - X[3]
    # DomainConstraint = nothing
    
    # pnames=["A_sus", "A_trans", "τ_1", "τ_1′", "τ_2", "T_shift", "r", "offset"]
    RTF = if AddDomain
        ModelMap(TransientApproximation(; t_range=(TimeRangeAdjustment ? t_range : 1)), DomainConstraint, HyperCube(lb,ub), (xdim(DS), ydim(DS), length(lb)); pnames=["A_sus", "A_trans", "τ_1", "τ_1′", "τ_2", "T_shift", "r", "offset"])
    else
        ModelMap(TransientApproximation(; t_range=(TimeRangeAdjustment ? t_range : 1)), (xdim(DS), ydim(DS), length(lb)); pnames=["A_sus", "A_trans", "τ_1", "τ_1′", "τ_2", "T_shift", "r", "offset"])
    end
    exp10 ? Exp10Transform(RTF, [false, false, true, true, true, false, true, false]) : RTF
end


# Also adding ODESystemTimeRetardation method for ODESystem in ModelingToolkitExt

"""
    ODESystemTimeRetardation(Sys::ODEFunction) -> ODEFunction
Applies [`TimeRetardation`](@ref) the to given `ODEFunction` by multiplying all equations with the sigmodial derivative of the time retardation transformation.
The new parameters `T_shift` and `r` are appended to the ODE parameters.
"""
function ODESystemTimeRetardation(Func::AbstractODEFunction{T}) where T
    oldf! = Func.f
    function newf!(du,u,p,t)
        oldf!(du,u,(@view p[1:end-2]), t)
        T_shift, r_coupling = @view p[end-1:end]
        du .*= exp10(r_coupling * t) / (exp10(r_coupling * t) + exp10(r_coupling * T_shift))
        nothing
    end
    function newf(u,p,t)
        T_shift, r_coupling = @view p[end-1:end]
        oldf!(u,(@view p[1:end-2]),t) .* (exp10(r_coupling * t) / (exp10(r_coupling * t) + exp10(r_coupling * T_shift)))
    end
    ODEFunction{T}(T ? newf! : newf)
end
