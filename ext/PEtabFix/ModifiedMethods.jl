

# Copied from _get_nllh
function GetNllh(probinfo::PEtabODEProblemInfo, model_info::ModelInfo,
                    prior::Function=PEtab._get_prior(model_info)[1], residuals::Bool=false; cids::AbstractVector{Symbol}=[:all])::Function
    _nllh = let pinfo = probinfo, minfo = model_info, res = residuals, _prior = prior, Cids = cids
        (x::AbstractVector; prior::Bool = true, cids::AbstractVector{Symbol}=Cids) -> begin
            PEtab._test_ordering(x, minfo.xindices.xids[:estimate_ps])
            _x = x |> collect
            nllh_val = PEtab.nllh(_x, pinfo, minfo, cids, false, res)
            if prior == true && res == false
                # nllh -> negative prior
                return nllh_val - _prior(_x)
            else
                return nllh_val
            end
        end
    end
    return _nllh
end


function GetNllhGrads(method, probinfo::PEtabODEProblemInfo, model_info::ModelInfo,
                   grad_prior::Function; cids::AbstractVector{Symbol}=[:all])::Tuple{Function, Function}
    if probinfo.gradient_method == :ForwardDiff
        _grad_nllh! = _get_grad_forward_AD_adapted(probinfo, model_info; cids = cids)
    end
    if probinfo.gradient_method == :ForwardEquations
        _grad_nllh! = _get_grad_forward_eqs_adapted(probinfo, model_info; cids = cids)
    end

    _grad! = let _grad_nllh! = _grad_nllh!, grad_prior = grad_prior, Cids = cids
        (g, x; prior = true, isremade = false, cids = Cids) -> begin
            _x = x |> collect
            _g = similar(_x)
            _grad_nllh!(_g, _x; isremade = isremade)
            if prior
                # nllh -> negative prior
                _g .+= grad_prior(_x) .* -1
            end
            g .= _g
            return nothing
        end
    end
    _grad = let _grad! = _grad!, Cids = cids
        (x; prior = true, isremade = false, cids = Cids) -> begin
            gradient = similar(x)
            _grad!(gradient, x; prior = prior, isremade = isremade)
            return gradient
        end
    end
    return _grad!, _grad
end

function GetNllhHesses(probinfo::PEtabODEProblemInfo, model_info::ModelInfo,
                   hess_prior::Function; ret_jacobian::Bool = false,
                   FIM::Bool = false, cids::AbstractVector{Symbol}=[:all])::Tuple{Function, Function}
    @unpack hessian_method, split_over_conditions, chunksize, cache = probinfo
    @unpack xdynamic = cache
    if FIM == true
        hessian_method = probinfo.FIM_method
    end

    if hessian_method === :ForwardDiff
        _hess_nllh! = _get_hess_forward_AD_adapted(probinfo, model_info; cids = cids)
    elseif hessian_method === :BlockForwardDiff
        _hess_nllh! = _get_hess_block_forward_AD_adapted(probinfo, model_info; cids = cids)
    elseif hessian_method == :GaussNewton
        _hess_nllh! = _get_hess_gaussnewton_adapted(probinfo, model_info, ret_jacobian; cids = cids)
    end

    _hess! = let _hess_nllh! = _hess_nllh!, hess_prior = hess_prior
        (H, x; prior = true, isremade = false) -> begin
            _x = x |> collect
            _H = H |> collect
            if hessian_method == :GaussNewton
                _hess_nllh!(_H, _x; isremade = isremade)
            else
                _hess_nllh!(_H, _x)
            end
            if prior && ret_jacobian == false
                # nllh -> negative prior
                _H .+= hess_prior(_x) .* -1
            end
            H .= _H
            return nothing
        end
    end
    _hess = (x; prior = true) -> begin
        if hessian_method == :GaussNewton && ret_jacobian == true
            H = zeros(eltype(x), length(x), length(model_info.petab_measurements.time))
        else
            H = zeros(eltype(x), length(x), length(x))
        end
        _hess!(H, x; prior = prior)
        return H
    end
    return _hess!, _hess
end



## Adapted from derivative_functions.jl
# const _get_grad_forward_AD_adapted = PEtab._get_grad_forward_AD
function _get_grad_forward_AD_adapted(probinfo::PEtabODEProblemInfo,
                              model_info::ModelInfo; cids::AbstractVector{Symbol}=[:all])::Function
    @unpack split_over_conditions, gradient_method, chunksize = probinfo
    @unpack sensealg, cache = probinfo
    _nllh_not_solveode = _get_nllh_not_solveode_adapted(probinfo, model_info;
                                                grad_forward_AD = true, cids = cids)

    if split_over_conditions == false
        _nllh_solveode = PEtab._get_nllh_solveode(probinfo, model_info; grad_xdynamic = true)

        @unpack xdynamic = cache
        chunksize_use = PEtab._get_chunksize(chunksize, xdynamic)
        cfg = ForwardDiff.GradientConfig(_nllh_solveode, xdynamic, chunksize_use)
        _grad! = let _nllh_not_solveode = _nllh_not_solveode,
            _nllh_solveode = _nllh_solveode, cfg = cfg, minfo = model_info, pinfo = probinfo

            (grad, x; isremade = false) -> PEtab.grad_forward_AD!(grad, x, _nllh_not_solveode,
                                                            _nllh_solveode, cfg, pinfo,
                                                            minfo; isremade = isremade)
        end
    end

    if split_over_conditions == true
        _nllh_solveode = PEtab._get_nllh_solveode(probinfo, model_info; cid = true,
                                            grad_xdynamic = true)

        _grad! = let _nllh_not_solveode = _nllh_not_solveode,
            _nllh_solveode = _nllh_solveode, minfo = model_info, pinfo = probinfo

            (g, x; isremade = false) -> PEtab.grad_forward_AD_split!(g, x, _nllh_not_solveode,
                                                               _nllh_solveode, pinfo, minfo)
        end
    end
    return _grad!
end


# const _get_grad_forward_eqs_adapted = PEtab._get_grad_forward_eqs
function _get_grad_forward_eqs_adapted(probinfo::PEtabODEProblemInfo,
                               model_info::ModelInfo; cids::AbstractVector{Symbol}=[:all])::Function
    @unpack split_over_conditions, gradient_method, chunksize = probinfo
    @unpack sensealg, cache = probinfo
    @unpack xdynamic, odesols = cache
    chunksize_use = PEtab._get_chunksize(chunksize, xdynamic)

    if sensealg == :ForwardDiff && split_over_conditions == false
        _solve_conditions! = let pinfo = probinfo, minfo = model_info
            (sols, x) -> PEtab.solve_conditions!(sols, x, pinfo, minfo; sensitivites_AD = true)
        end
        cfg = ForwardDiff.JacobianConfig(_solve_conditions!, odesols, xdynamic,
                                         chunksize_use)
    end

    if sensealg == :ForwardDiff && split_over_conditions == true
        _solve_conditions! = let pinfo = probinfo, minfo = model_info
            (sols, x, cid) -> PEtab.solve_conditions!(sols, x, pinfo, minfo; cids = cid,
                                                sensitivites_AD = true)
        end
        cfg = ForwardDiff.JacobianConfig(_solve_conditions!, odesols, xdynamic,
                                         chunksize_use)
    end

    if sensealg != :ForwardDiff
        _solve_conditions! = let pinfo = probinfo, minfo = model_info
            (x, cid) -> PEtab.solve_conditions!(minfo, x, pinfo; cids = cid, sensitivites = true)
        end
        cfg = nothing
    end

    _nllh_not_solveode = _get_nllh_not_solveode_adapted(probinfo, model_info;
                                                grad_forward_eqs = true, cids = cids)

    _grad! = let _nllh_not_solveode = _nllh_not_solveode,
        _solve_conditions! = _solve_conditions!, minfo = model_info, pinfo = probinfo,
        cfg = cfg, Cids = cids

        (g, x; isremade = false, cids = Cids) -> PEtab.grad_forward_eqs!(g, x, _nllh_not_solveode,
                                                      _solve_conditions!, pinfo, minfo,
                                                      cfg; cids = cids,
                                                      isremade = isremade)
    end
    return _grad!
end

#=
    Hessian functions
=#
# const _get_hess_forward_AD_adapted = PEtab._get_hess_forward_AD
function _get_hess_forward_AD_adapted(probinfo::PEtabODEProblemInfo,
                              model_info::ModelInfo; cids::AbstractVector{Symbol}=[:all])::Function
    @unpack split_over_conditions, chunksize = probinfo
    if split_over_conditions == false
        _nllh = let pinfo = probinfo, minfo = model_info, Cids = cids
            (x) -> PEtab.nllh(x, pinfo, minfo, Cids, true, false)
        end
        nestimate = length(model_info.xindices.xids[:estimate])
        chunksize_use = PEtab._get_chunksize(chunksize, zeros(nestimate))
        cfg = ForwardDiff.HessianConfig(_nllh, zeros(nestimate), chunksize_use)
        _hess_nllh! = let _nllh = _nllh, cfg = cfg, minfo = model_info
            (H, x) -> PEtab.hess!(H, x, _nllh, minfo, cfg)
        end
    end

    if split_over_conditions == true
        _nllh = let pinfo = probinfo, minfo = model_info
            (x, cid) -> PEtab.nllh(x, pinfo, minfo, cid, true, false)
        end
        _hess_nllh! = let _nllh = _nllh, minfo = model_info
            (H, x) -> PEtab.hess_split!(H, x, _nllh, minfo)
        end
    end
    return _hess_nllh!
end

# const _get_hess_block_forward_AD_adapted = PEtab._get_hess_block_forward_AD
function _get_hess_block_forward_AD_adapted(probinfo::PEtabODEProblemInfo,
                                    model_info::ModelInfo; cids::AbstractVector{Symbol}=[:all])::Function
    @unpack split_over_conditions, chunksize = probinfo
    xdynamic = probinfo.cache.xdynamic

    _nllh_not_solveode = _get_nllh_not_solveode_adapted(probinfo, model_info;
                                                grad_forward_AD = true, cids = cids)

    if split_over_conditions == false
        _nllh_solveode = let pinfo = probinfo, minfo = model_info, Cids = cids
            @unpack xnoise, xobservable, xnondynamic = pinfo.cache
            (x) -> PEtab.nllh_solveode(x, xnoise, xobservable, xnondynamic, pinfo, minfo;
                                 grad_xdynamic = true, cids = Cids)
        end

        chunksize_use = PEtab._get_chunksize(chunksize, xdynamic)
        cfg = ForwardDiff.HessianConfig(_nllh_solveode, xdynamic, chunksize_use)
        _hess_nllh! = let _nllh_solveode = _nllh_solveode,
            _nllh_not_solveode = _nllh_not_solveode, pinfo = probinfo, minfo = model_info, Cids = cids,
            cfg = cfg

            (H, x) -> PEtab.hess_block!(H, x, _nllh_not_solveode, _nllh_solveode, pinfo,
                                  minfo, cfg; cids = Cids)
        end
    end

    if split_over_conditions == true
        _nllh_solveode = let pinfo = probinfo, minfo = model_info
            @unpack xnoise, xobservable, xnondynamic = pinfo.cache
            (x, cid) -> PEtab.nllh_solveode(x, xnoise, xobservable, xnondynamic,
                                      pinfo, minfo,
                                      grad_xdynamic = true,
                                      cids = cid)
        end

        _hess_nllh! = let _nllh_solveode = _nllh_solveode,
            _nllh_not_solveode = _nllh_not_solveode, pinfo = probinfo, minfo = model_info, Cids = cids

            (H, x) -> PEtab.hess_block_split!(H, x, _nllh_not_solveode, _nllh_solveode,
                                        pinfo, minfo; cids = Cids)
        end
    end
    return _hess_nllh!
end

# const _get_hess_gaussnewton_adapted = PEtab._get_hess_gaussnewton
function _get_hess_gaussnewton_adapted(probinfo::PEtabODEProblemInfo, model_info::ModelInfo,
                               ret_jacobian::Bool; cids::AbstractVector{Symbol}=[:all])::Function
    @unpack split_over_conditions, chunksize, cache = probinfo
    xdynamic = probinfo.cache.xdynamic

    if split_over_conditions == false
        _solve_conditions! = let pinfo = probinfo, minfo = model_info, Cids = cids
            (sols, x) -> PEtab.solve_conditions!(sols, x, pinfo, minfo; cids = Cids,
                                           sensitivites_AD = true)
        end
    end
    if split_over_conditions == true
        _solve_conditions! = let pinfo = probinfo, minfo = model_info
            (sols, x, cid) -> PEtab.solve_conditions!(sols, x, pinfo, minfo; cids = cid,
                                                sensitivites_AD = true)
        end
    end
    chunksize_use = PEtab._get_chunksize(chunksize, xdynamic)
    cfg = cfg = ForwardDiff.JacobianConfig(_solve_conditions!,
                                           cache.odesols,
                                           cache.xdynamic, chunksize_use)

    _residuals_not_solveode = let pinfo = probinfo, minfo = model_info, Cids = cids
        ixnoise = minfo.xindices.xindices_notsys[:noise]
        ixobservable = minfo.xindices.xindices_notsys[:observable]
        ixnondynamic = minfo.xindices.xindices_notsys[:nondynamic]
        (residuals, x) -> begin
            PEtab.residuals_not_solveode(residuals, x[ixnoise], x[ixobservable],
                                   x[ixnondynamic], pinfo, minfo; cids = Cids)
        end
    end

    xnot_ode = zeros(Float64, length(model_info.xindices.xids[:not_system]))
    cfg_notsolve = ForwardDiff.JacobianConfig(_residuals_not_solveode,
                                              cache.residuals_gn, xnot_ode,
                                              ForwardDiff.Chunk(xnot_ode))
    _hess_nllh! = let _residuals_not_solveode = _residuals_not_solveode,
        pinfo = probinfo, minfo = model_info, cfg = cfg, cfg_notsolve = cfg_notsolve,
        ret_jacobian = ret_jacobian, _solve_conditions! = _solve_conditions!, Cids = cids

        (H, x; isremade = false) -> PEtab.hess_GN!(H, x, _residuals_not_solveode,
                                             _solve_conditions!, pinfo, minfo, cfg,
                                             cfg_notsolve; cids = Cids,
                                             isremade = isremade,
                                             ret_jacobian = ret_jacobian)
    end
    return _hess_nllh!
end

#=
    Helpers
=#
# const _get_nllh_not_solveode_adapted = PEtab._get_nllh_not_solveode
function _get_nllh_not_solveode_adapted(probinfo::PEtabODEProblemInfo, model_info::ModelInfo;
                                grad_forward_AD::Bool = false, grad_adjoint::Bool = false,
                                grad_forward_eqs::Bool = false, cids::AbstractVector{Symbol}=[:all])::Function
    _nllh_not_solveode = let pinfo = probinfo, minfo = model_info, Cids = cids
        ixnoise = minfo.xindices.xindices_notsys[:noise]
        ixobservable = minfo.xindices.xindices_notsys[:observable]
        ixnondynamic = minfo.xindices.xindices_notsys[:nondynamic]
        (x) -> PEtab.nllh_not_solveode(x[ixnoise], x[ixobservable], x[ixnondynamic], pinfo, minfo;
                                 cids = Cids, grad_forward_AD = grad_forward_AD,
                                 grad_forward_eqs = grad_forward_eqs,
                                 grad_adjoint = grad_adjoint)
    end
    return _nllh_not_solveode
end


function PEtab._grad_forward_eqs!(grad::Vector{T}, _solve_conditions!::Function,
                            probinfo::PEtabODEProblemInfo, model_info::ModelInfo,
                            cfg::Union{ForwardDiff.JacobianConfig, Nothing};
                            cids::Vector{Symbol} = [:all],
                            isremade::Bool = false)::Nothing where {T <: Union{BigFloat, Float16, Float32, Float64}}
    @unpack cache, sensealg = probinfo
    @unpack xindices, simulation_info = model_info
    xnoise_ps = PEtab.transform_x(cache.xnoise, xindices, :xnoise, cache)
    xobservable_ps = PEtab.transform_x(cache.xobservable, xindices, :xobservable, cache)
    xnondynamic_ps = PEtab.transform_x(cache.xnondynamic, xindices, :xnondynamic, cache)
    xdynamic_ps = PEtab.transform_x(cache.xdynamic, xindices, :xdynamic, cache)

    # Solve the expanded ODE system for the sensitivites
    success = PEtab.solve_sensitivites!(model_info, _solve_conditions!, xdynamic_ps, sensealg,
                                  probinfo, cids, cfg, isremade)
    if success != true
        @warn "Failed to solve sensitivity equations"
        fill!(grad, 0.0)
        return nothing
    end
    if isempty(xdynamic_ps)
        return nothing
    end

    fill!(grad, 0.0)
    for icid in eachindex(simulation_info.conditionids[:experiment])
        if cids[1] != :all && !(simulation_info.conditionids[:experiment][icid] in cids)
            continue
        end
        PEtab._grad_forward_eqs_cond!(grad, xdynamic_ps, xnoise_ps, xobservable_ps,
                                xnondynamic_ps,
                                icid, sensealg, probinfo, model_info)
    end
    return nothing
end

function PEtab._jac_residuals_xdynamic!(jac::AbstractMatrix{T}, _solve_conditions!::Function,
                                  probinfo::PEtabODEProblemInfo,
                                  model_info::ModelInfo, cfg::ForwardDiff.JacobianConfig;
                                  cids::Vector{Symbol} = [:all],
                                  isremade::Bool = false)::Nothing where {T <: Union{BigFloat, Float16, Float32, Float64}}
    @unpack cache, sensealg, reuse_sensitivities = probinfo
    @unpack xindices, simulation_info = model_info
    xnoise_ps = PEtab.transform_x(cache.xnoise, xindices, :xnoise, cache)
    xobservable_ps = PEtab.transform_x(cache.xobservable, xindices, :xobservable, cache)
    xnondynamic_ps = PEtab.transform_x(cache.xnondynamic, xindices, :xnondynamic, cache)
    xdynamic_ps = PEtab.transform_x(cache.xdynamic, xindices, :xdynamic, cache)

    if reuse_sensitivities == false
        success = PEtab.solve_sensitivites!(model_info, _solve_conditions!, xdynamic_ps,
                                      :ForwardDiff, probinfo, cids, cfg, isremade)
        if success != true
            @warn "Failed to solve sensitivity equations"
            fill!(jac, 0.0)
            return nothing
        end
    end
    if isempty(xdynamic_ps)
        fill!(jac, 0.0)
        return nothing
    end

    # Compute the gradient by looping through all experimental conditions.
    for icid in eachindex(simulation_info.conditionids[:experiment])
        if cids[1] != :all && !(simulation_info.conditionids[:experiment][icid] in cids)
            continue
        end
        PEtab._jac_residuals_cond!(jac, xdynamic_ps, xnoise_ps, xobservable_ps, xnondynamic_ps,
                             icid, probinfo, model_info)
    end
    return nothing
end