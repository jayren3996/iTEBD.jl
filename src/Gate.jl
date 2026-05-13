#---------------------------------------------------------------------------------------------------
# QUANTUM GATE
#---------------------------------------------------------------------------------------------------
"""
    tensor_applygate!(G, Γs, λl; keywords...)

Apply a dense local gate `G` to a contiguous block of stored local tensors.

Conceptually this routine:

1. groups the local tensors `Γs` into a single block tensor,
2. inserts the incoming Schmidt values `λl` on the left bond,
3. applies the dense gate `G` on the physical legs,
4. decomposes the updated block back into site-local tensors.

Parameters:
- `G`
  Dense local operator acting on the physical Hilbert space of the grouped
  block. Its matrix dimension must match the total physical dimension of `Γs`.
- `Γs`
  Stored local tensors for the contiguous block to be updated.
- `λl`
  Schmidt values on the bond immediately to the left of that block.

Keyword arguments:
- `maxdim=MAXDIM`
  Hard cap on the retained bond dimension during the decomposition of the
  updated block.
- `mindim=1`
  Minimum retained bond dimension on each updated internal bond.
- `truncerr=0.0`
  Target discarded weight used to choose the smallest admissible kept bond
  dimension on each updated internal bond. Set `truncerr = 0` for fixed-cap
  evolution.
- `svd_min=SVDTOL`
  Optional absolute singular-value floor applied after the discarded-weight
  target is evaluated.
- `renormalize=false`
  Whether to renormalize Schmidt values produced by the decomposition.

Returns:
- `(Γs_new, λs_new)` where `Γs_new` are the updated stored local tensors and
  `λs_new` are the Schmidt spectra on the internal bonds of the updated block.

Notes:
- This is a low-level helper. Most user-facing evolution code should call
  [`applygate!`](@ref) or [`evolve!`](@ref) instead.
- The tensors are assumed to already follow the package storage convention
  `B_i = Γ_i λ_i`.
"""
function tensor_applygate!(
    G::AbstractMatrix{<:Number}, Γs::AbstractVector{<:AbstractArray{<:Number, 3}},
    λl::AbstractVector{<:Number};
    maxdim::Integer=MAXDIM,
    mindim::Integer=1,
    truncerr::Real=0.0,
    svd_min::Real=SVDTOL,
    cutoff::Union{Nothing,Real}=nothing,
    renormalize::Bool=false,
    return_stats::Bool=false,
    bond_indices::Union{Nothing,AbstractVector{<:Integer}}=nothing
)
    svd_floor = _resolve_svd_min(svd_min, cutoff)
    _validate_truncation_args(maxdim, mindim, truncerr, svd_floor)
    n = length(Γs)
    isone(n) && return return_stats ? ([tensor_umul(G, Γs[1])], Vector{Vector{eltype(λl)}}(), Any[]) : ([tensor_umul(G, Γs[1])], Vector{Vector{eltype(λl)}}())
    Γ = tensor_group(Γs)
    tensor_lmul!(λl, Γ)
    GΓ = tensor_umul(G, Γ)
    tensor_decomp!(
        GΓ,
        λl,
        n;
        maxdim,
        mindim,
        truncerr,
        svd_min=svd_floor,
        renormalize,
        return_stats,
        bond_indices,
    )
end
#---------------------------------------------------------------------------------------------------
export applygate!
"""
    applygate!(ψ, G, i, j; maxdim=MAXDIM, mindim=1, truncerr=0.0, svd_min=SVDTOL, renormalize=true, return_stats=false)

Apply a local gate `G` in place to the contiguous region from site `i` to site
`j` of the periodic unit cell.

Parameters:
- `ψ`
  State to update in place.
- `G`
  Dense local operator. Its dimension must match the total physical dimension of
  the block from site `i` to site `j`.
- `i`, `j`
  Start and end sites of the support inside the unit cell. If `j < i`, the
  support is interpreted with periodic wraparound.

Keyword arguments:
- `maxdim=MAXDIM`
  Hard cap on the retained bond dimension during the decomposition of the
  updated block.
- `mindim=1`
  Minimum retained bond dimension on each updated internal bond.
- `truncerr=0.0`
  Target discarded weight used to choose the smallest admissible kept bond
  dimension on each updated internal bond. Set `truncerr = 0` for fixed-cap
  evolution.
- `svd_min=SVDTOL`
  Optional absolute singular-value floor applied after the discarded-weight
  target is evaluated.
- `renormalize=true`
  Whether to renormalize the retained Schmidt values.
- `return_stats=false`
  If `true`, also return per-bond truncation diagnostics for this gate update.

Returns:
- `ψ` when `return_stats = false`.
- `(ψ, stats)` when `return_stats = true`, where `stats.bond_stats` contains one
  entry per updated internal bond.

Notes:
- For a one-site update with `i == j`, the operator is applied directly to the
  stored local tensor without a block decomposition.
- For multi-site updates, this routine works directly with the stored
  right-canonical tensors `ψ.Γ`.
- Site indices are interpreted periodically through the finite unit cell.

Example:
```julia
psi = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
X = [0 1; 1 0]
applygate!(psi, kron(X, X), 1, 2; maxdim=4)
```
"""
function applygate!(
    ψ::iMPS, G::AbstractMatrix,
    i::Integer, j::Integer;
    maxdim::Integer=MAXDIM,
    mindim::Integer=1,
    truncerr::Real=0.0,
    svd_min::Real=SVDTOL,
    cutoff::Union{Nothing,Real}=nothing,
    renormalize::Bool=true,
    return_stats::Bool=false
)
    svd_floor = _resolve_svd_min(svd_min, cutoff)
    _validate_truncation_args(maxdim, mindim, truncerr, svd_floor)
    if isequal(i, j)
        tensor_umul!(G, ψ.Γ[i])
        if return_stats
            return ψ, (
                support=(i, j),
                bond_stats=Any[],
                max_discarded_weight=0.0,
                num_saturated=0,
            )
        end
        return ψ
    end
    inds = _gate_indices(ψ, i, j)
    Γs = ψ.Γ[inds]
    λl = ψ.λ[mod(i-2,ψ.n)+1]
    if return_stats
        Γs, λs, bond_stats = tensor_applygate!(
            G,
            Γs,
            λl;
            maxdim,
            mindim,
            truncerr,
            svd_min=svd_floor,
            renormalize,
            return_stats=true,
            bond_indices=inds[1:end-1],
        )
    else
        Γs, λs = tensor_applygate!(
            G,
            Γs,
            λl;
            maxdim,
            mindim,
            truncerr,
            svd_min=svd_floor,
            renormalize,
        )
    end
    push!(λs, ψ.λ[j])
    for i in eachindex(inds) 
        ψ.Γ[inds[i]] = Γs[i]
        ψ.λ[inds[i]] = λs[i]
    end
    return_stats && return ψ, _gate_update_stats(i, j, bond_stats)
    return ψ
end

"""
    _gate_indices(ψ, i, j)

Return the periodic list of unit-cell indices covered by the contiguous support
from site `i` to site `j`.

Notes:
- If `j < i`, the support is interpreted with periodic wraparound.
- This is an internal helper used by [`evolve!`](@ref).
"""
function _gate_indices(ψ::iMPS, i::Integer, j::Integer)
    j > i ? (i:j) : [i:ψ.n; 1:j]
end

const _SUZUKI_FOURTH_P = 1 / (4 - 4^(1 / 3))
const _BARTHEL_A1 = 0.09584850274120368
const _BARTHEL_A2 = -0.07811115892163792
const _BARTHEL_A3 = 0.5 - (_BARTHEL_A1 + _BARTHEL_A2)
const _BARTHEL_B1 = 0.42652466131587616
const _BARTHEL_B2 = -0.12039526945509727
const _BARTHEL_B3 = 1 - 2 * (_BARTHEL_B1 + _BARTHEL_B2)

function _validate_truncation_args(
    maxdim::Integer,
    mindim::Integer,
    truncerr::Real,
    svd_min::Real
)
    maxdim > 0 || throw(ArgumentError("maxdim must be positive"))
    mindim > 0 || throw(ArgumentError("mindim must be positive"))
    maxdim >= mindim || throw(ArgumentError("maxdim must be at least mindim"))
    isfinite(truncerr) && truncerr >= 0 || throw(ArgumentError("truncerr must be finite and non-negative"))
    isfinite(svd_min) && svd_min >= 0 || throw(ArgumentError("svd_min must be finite and non-negative"))
    return nothing
end

function _resolve_svd_min(svd_min::Real, cutoff::Union{Nothing,Real})
    if isnothing(cutoff)
        return svd_min
    end
    isfinite(cutoff) && cutoff >= 0 ||
        throw(ArgumentError("cutoff must be finite and non-negative"))
    svd_min == SVDTOL ||
        throw(ArgumentError("cutoff and svd_min are aliases; pass only one"))
    return cutoff
end

function _validate_chi_policy(chi_policy::Symbol)
    chi_policy in (:fixed, :adaptive) ||
        throw(ArgumentError("chi_policy must be one of :fixed or :adaptive"))
    return nothing
end

function _validate_evolve_args(
    steps::Integer,
    maxdim::Integer,
    mindim::Integer,
    truncerr::Real,
    svd_min::Real
)
    steps >= 0 || throw(ArgumentError("steps must be non-negative"))
    _validate_truncation_args(maxdim, mindim, truncerr, svd_min)
    return nothing
end

function _gate_update_stats(i::Integer, j::Integer, bond_stats)
    max_discarded_weight = isempty(bond_stats) ? 0.0 : maximum(stat.discarded_weight for stat in bond_stats)
    num_saturated = count(stat -> stat.saturated, bond_stats)
    (
        support=(i, j),
        bond_stats=bond_stats,
        max_discarded_weight=max_discarded_weight,
        num_saturated=num_saturated,
    )
end

function _aggregate_evolution_stats(gate_updates)
    bond_stats = Any[]
    for update in gate_updates
        append!(bond_stats, update.bond_stats)
    end

    max_discarded_weight = isempty(bond_stats) ? 0.0 : maximum(stat.discarded_weight for stat in bond_stats)
    mean_discarded_weight = isempty(bond_stats) ? 0.0 : sum(stat.discarded_weight for stat in bond_stats) / length(bond_stats)
    max_kept_dim = isempty(bond_stats) ? 0 : maximum(stat.chi_keep for stat in bond_stats)
    num_saturated = count(stat -> stat.saturated, bond_stats)

    (
        gate_updates=gate_updates,
        max_discarded_weight=max_discarded_weight,
        mean_discarded_weight=mean_discarded_weight,
        max_kept_dim=max_kept_dim,
        num_saturated=num_saturated,
    )
end

function _validate_trotter_scheme(num_layers::Integer, trotter::Symbol)
    num_layers > 0 || throw(ArgumentError("layers must contain at least one commuting layer"))
    trotter in (:second, :fourth, :fourth_opt) ||
        throw(ArgumentError("unknown trotter scheme $(repr(trotter)); use :second, :fourth, or :fourth_opt"))
    if trotter === :fourth_opt && num_layers != 2
        throw(ArgumentError("trotter = :fourth_opt requires exactly two commuting layers"))
    end
    return nothing
end

function _validate_trotter_evolution(trotter::Symbol, evolution::Symbol)
    evolution in (:real, :imaginary) ||
        throw(ArgumentError("unknown evolution $(repr(evolution)); use :real or :imaginary"))
    if evolution === :imaginary && trotter !== :second
        throw(ArgumentError("higher-order Trotterization is currently supported only for evolution = :real"))
    end
    return nothing
end

function _push_trotter_stage!(
    stages::Vector{Tuple{Int, Float64}},
    layer::Integer,
    coeff::Real
)
    abs(coeff) <= ZEROTOL && return stages

    coeff_f = Float64(coeff)
    if !isempty(stages) && last(stages)[1] == layer
        merged = last(stages)[2] + coeff_f
        if abs(merged) <= ZEROTOL
            pop!(stages)
        else
            stages[end] = (layer, merged)
        end
    else
        push!(stages, (layer, coeff_f))
    end
    return stages
end

function _append_strang_step!(
    stages::Vector{Tuple{Int, Float64}},
    num_layers::Integer,
    scale::Real
)
    if num_layers == 1
        return _push_trotter_stage!(stages, 1, scale)
    end

    for layer in 1:num_layers-1
        _push_trotter_stage!(stages, layer, scale / 2)
    end
    _push_trotter_stage!(stages, num_layers, scale)
    for layer in (num_layers - 1):-1:1
        _push_trotter_stage!(stages, layer, scale / 2)
    end
    return stages
end

function _append_trotter_macro_step!(
    stages::Vector{Tuple{Int, Float64}},
    num_layers::Integer,
    trotter::Symbol
)
    if trotter === :second
        return _append_strang_step!(stages, num_layers, 1.0)
    elseif trotter === :fourth
        p = _SUZUKI_FOURTH_P
        q = 1 - 4p
        _append_strang_step!(stages, num_layers, p)
        _append_strang_step!(stages, num_layers, p)
        _append_strang_step!(stages, num_layers, q)
        _append_strang_step!(stages, num_layers, p)
        _append_strang_step!(stages, num_layers, p)
        return stages
    elseif trotter === :fourth_opt
        _push_trotter_stage!(stages, 1, _BARTHEL_A1)
        _push_trotter_stage!(stages, 2, _BARTHEL_B1)
        _push_trotter_stage!(stages, 1, _BARTHEL_A2)
        _push_trotter_stage!(stages, 2, _BARTHEL_B2)
        _push_trotter_stage!(stages, 1, _BARTHEL_A3)
        _push_trotter_stage!(stages, 2, _BARTHEL_B3)
        _push_trotter_stage!(stages, 1, _BARTHEL_A3)
        _push_trotter_stage!(stages, 2, _BARTHEL_B2)
        _push_trotter_stage!(stages, 1, _BARTHEL_A2)
        _push_trotter_stage!(stages, 2, _BARTHEL_B1)
        _push_trotter_stage!(stages, 1, _BARTHEL_A1)
        return stages
    end

    throw(ArgumentError("unknown trotter scheme $(repr(trotter)); use :second, :fourth, or :fourth_opt"))
end

function _trotter_stage_schedule(num_layers::Integer, trotter::Symbol, steps::Integer)
    steps >= 0 || throw(ArgumentError("steps must be non-negative"))
    _validate_trotter_scheme(num_layers, trotter)

    stages = Tuple{Int, Float64}[]
    for _ in 1:steps
        _append_trotter_macro_step!(stages, num_layers, trotter)
    end
    return stages
end

function _trotter_time_prefactor(dt::Real, coeff::Real, evolution::Symbol)
    if evolution === :real
        return -1im * coeff * dt
    elseif evolution === :imaginary
        return -coeff * dt
    end
    throw(ArgumentError("unknown evolution $(repr(evolution)); use :real or :imaginary"))
end

function _materialize_trotter_gates(
    layers,
    dt::Real,
    stages::AbstractVector{<:Tuple{Int, <:Real}};
    evolution::Symbol=:real
)
    cache = Dict{Tuple{Int, Int, Float64}, Any}()
    gates = Tuple{Any, Int, Int}[]

    for (layer_idx, coeff) in stages
        for (term_idx, term) in enumerate(layers[layer_idx])
            h, i, j = term
            key = (layer_idx, term_idx, Float64(coeff))
            G = get!(cache, key) do
                exp(_trotter_time_prefactor(dt, coeff, evolution) * h)
            end
            push!(gates, (G, i, j))
        end
    end

    return gates
end

function _fuse_consecutive_gates(gates)
    isempty(gates) && return gates
    fused = [gates[1]]
    for k in 2:length(gates)
        G_prev, i_prev, j_prev = fused[end]
        G_curr, i_curr, j_curr = gates[k]
        if i_curr == i_prev && j_curr == j_prev
            fused[end] = (G_curr * G_prev, i_prev, j_prev)
        else
            push!(fused, gates[k])
        end
    end
    return fused
end

function _evolve_gate_sequence!(
    ψ::iMPS,
    gates;
    maxdim::Integer=MAXDIM,
    mindim::Integer=1,
    truncerr::Real=0.0,
    svd_min::Real=SVDTOL,
    cutoff::Union{Nothing,Real}=nothing,
    renormalize::Bool=true,
    return_stats::Bool=false,
    chi_policy::Symbol=:fixed,
    q::Real=1.0,
    alpha::Real=0.1
)
    gates = _fuse_consecutive_gates(gates)
    svd_floor = _resolve_svd_min(svd_min, cutoff)
    _validate_chi_policy(chi_policy)
    gate_updates = return_stats ? Any[] : nothing
    χ = min(maxdim, max(mindim, maximum(length, ψ.λ; init=mindim)))
    for gate in gates
        G, i, j = gate
        if return_stats
            _, stats = applygate!(
                ψ,
                G,
                i,
                j;
                maxdim,
                mindim,
                truncerr,
                svd_min=svd_floor,
                renormalize,
                return_stats=true,
            )
            push!(gate_updates, stats)
        else
            applygate!(ψ, G, i, j; maxdim, mindim, truncerr, svd_min=svd_floor, renormalize)
        end

        if chi_policy === :adaptive
            for bond in _gate_indices(ψ, i, j)
                χ = adaptive_bonddim(
                    χ,
                    ψ.λ[bond];
                    mindim,
                    maxdim,
                    q,
                    alpha,
                    cutoff=svd_floor,
                )
            end
            canonical!(ψ; maxdim=χ, cutoff=svd_floor, renormalize)
        end
    end

    return gate_updates
end

export trotter_gates
"""
    trotter_gates(layers, dt; trotter=:second, evolution=:real)

Materialize one Trotter macro-step from a Hamiltonian split into commuting
layers.

Each element of `layers` is a collection of local Hamiltonian terms
`(h, i, j)`. Terms inside a single layer are assumed to commute, so their dense
gates can be applied sequentially in the supplied order. The package does not
check commutation automatically.

Parameters:
- `layers`
  Hamiltonian split into commuting layers. Each local term reuses the tuple
  shape `(h, i, j)` with dense operator `h` supported on the unit-cell region
  `i:j`.
- `dt`
  Microscopic time step.

Keyword arguments:
- `trotter=:second`
  Trotter scheme. Use `:second` for Strang splitting, `:fourth` for Suzuki's
  recursive fourth-order composition, or `:fourth_opt` for the two-layer
  optimized real-time formula of Barthel and Zhang.
- `evolution=:real`
  Use `:real` to build gates `exp(-1im * c * dt * h)` or `:imaginary` to build
  second-order imaginary-time gates `exp(-c * dt * h)`.

Returns:
- A vector of dense local gates `(G, i, j)` implementing one macro-step.

Notes:
- `trotter = :fourth_opt` requires exactly two layers.
- Higher-order Trotter schemes currently reject `evolution = :imaginary`
  because their negative substeps are only supported for real-time evolution.

Example:
```julia
X = [0 1; 1 0]
H = kron(X, X)
layers = [[(H, 1, 2)], [(H, 2, 1)]]
gates = trotter_gates(layers, 0.1; trotter=:fourth_opt)
```
"""
function trotter_gates(
    layers,
    dt::Real;
    trotter::Symbol=:second,
    evolution::Symbol=:real
)
    _validate_trotter_scheme(length(layers), trotter)
    _validate_trotter_evolution(trotter, evolution)

    stages = _trotter_stage_schedule(length(layers), trotter, 1)
    return _materialize_trotter_gates(layers, dt, stages; evolution)
end

export evolve!
"""
    evolve!(ψ, gates, steps; maxdim=MAXDIM, mindim=1, truncerr=0.0, svd_min=SVDTOL, renormalize=true, return_stats=false)

Apply a sequence of local gates repeatedly for `steps` sweeps.

Each element of `gates` must be a tuple `(G, i, j)` consisting of the local
operator `G` and the support `i:j` inside the unit cell.

Parameters:
- `ψ`
  State to evolve in place.
- `gates`
  Iterable of tuples `(G, i, j)`. Each tuple specifies a dense local operator
  `G` and the contiguous support `i:j` inside the unit cell.
- `steps`
  Number of full sweeps through the gate list.

Keyword arguments:
- `maxdim=MAXDIM`
  Hard cap on the retained bond dimension during each local update.
- `mindim=1`
  Minimum retained bond dimension on each updated internal bond.
- `truncerr=0.0`
  Target discarded weight used to choose the smallest admissible kept bond
  dimension on each updated internal bond. Set `truncerr = 0` for fixed-cap
  evolution.
- `svd_min=SVDTOL`
  Optional absolute singular-value floor applied after the discarded-weight
  target is evaluated.
- `renormalize=true`
  Whether to renormalize Schmidt values after decomposition.
- `return_stats=false`
  If `true`, also return aggregated truncation diagnostics for the full
  evolution.

Returns:
- `ψ` when `return_stats = false`.
- `(ψ, stats)` when `return_stats = true`.

Notes:
- `truncerr = 0` reproduces the standard fixed-cap behavior with `maxdim` as the
  hard cap.
- `truncerr > 0` enables standard local SVD truncation control based on the
  discarded weight of the post-gate singular values.
- The `gates` argument is kept explicit rather than hiding the sweep pattern in
  a higher-level model object.

Example:
```julia
X = [0 1; 1 0]
psi = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
gates = [(kron(X, X), 1, 2), (kron(X, X), 2, 1)]
evolve!(psi, gates, 5; maxdim=4)
```
"""
function evolve!(
    ψ::iMPS,
    gates,
    steps::Integer;
    maxdim::Integer=MAXDIM,
    mindim::Integer=1,
    truncerr::Real=0.0,
    svd_min::Real=SVDTOL,
    cutoff::Union{Nothing,Real}=nothing,
    chi_policy::Symbol=:fixed,
    q::Real=1.0,
    alpha::Real=0.1,
    renormalize::Bool=true,
    return_stats::Bool=false
)
    svd_floor = _resolve_svd_min(svd_min, cutoff)
    _validate_evolve_args(steps, maxdim, mindim, truncerr, svd_floor)
    _validate_chi_policy(chi_policy)

    gate_updates = return_stats ? Any[] : nothing
    for _ in 1:steps
        updates = _evolve_gate_sequence!(
            ψ,
            gates;
            maxdim,
            mindim,
            truncerr,
            svd_min=svd_floor,
            chi_policy,
            q,
            alpha,
            renormalize,
            return_stats,
        )
        return_stats && append!(gate_updates, updates)
    end

    return_stats ? (ψ, _aggregate_evolution_stats(gate_updates)) : ψ
end

"""
    evolve!(ψ, layers, dt, steps; trotter=:second, evolution=:real, maxdim=MAXDIM, mindim=1, truncerr=0.0, svd_min=SVDTOL, renormalize=true, return_stats=false)

Evolve `ψ` under a Hamiltonian split into commuting layers.

Each layer is a collection of local Hamiltonian terms `(h, i, j)`. One
Trotterized macro-step is first expanded into a gate sequence with
[`trotter_gates`](@ref), after which the resulting dense gates are applied in
place with the same local truncation controls as the gate-list [`evolve!`](@ref)
method.

Parameters:
- `ψ`
  State to evolve in place.
- `layers`
  Hamiltonian split into commuting layers. Each local term is a tuple
  `(h, i, j)` where `h` is a dense local operator supported on `i:j`.
- `dt`
  Microscopic time step.
- `steps`
  Number of Trotter macro-steps.

Keyword arguments:
- `trotter=:second`
  Trotter scheme. Supported values are `:second`, `:fourth`, and
  `:fourth_opt`.
- `evolution=:real`
  Use `:real` for real-time gates `exp(-1im * c * dt * h)` or `:imaginary` for
  second-order imaginary-time gates `exp(-c * dt * h)`.
- `maxdim`, `mindim`, `truncerr`, `svd_min`, `renormalize`, `return_stats`
  Match the gate-list [`evolve!`](@ref) interface.

Returns:
- The same object `ψ`, mutated in place.

Notes:
- Terms inside one layer are applied in the supplied order, and the caller is
  responsible for ensuring that they commute.
- `trotter = :fourth_opt` requires exactly two layers.
- Higher-order Trotter schemes currently reject `evolution = :imaginary`.

Example:
```julia
X = [0 1; 1 0]
H = kron(X, X)
layers = [[(H, 1, 2)], [(H, 2, 1)]]
psi = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
evolve!(psi, layers, 0.1, 5; trotter=:fourth, maxdim=8)
```
"""
function evolve!(
    ψ::iMPS,
    layers,
    dt::Real,
    steps::Integer;
    trotter::Symbol=:second,
    evolution::Symbol=:real,
    maxdim::Integer=MAXDIM,
    mindim::Integer=1,
    truncerr::Real=0.0,
    svd_min::Real=SVDTOL,
    cutoff::Union{Nothing,Real}=nothing,
    chi_policy::Symbol=:fixed,
    q::Real=1.0,
    alpha::Real=0.1,
    renormalize::Bool=true,
    return_stats::Bool=false
)
    svd_floor = _resolve_svd_min(svd_min, cutoff)
    _validate_evolve_args(steps, maxdim, mindim, truncerr, svd_floor)
    _validate_chi_policy(chi_policy)
    _validate_trotter_scheme(length(layers), trotter)
    _validate_trotter_evolution(trotter, evolution)

    stages = _trotter_stage_schedule(length(layers), trotter, steps)
    gates = _materialize_trotter_gates(layers, dt, stages; evolution)

    updates = _evolve_gate_sequence!(
        ψ,
        gates;
        maxdim,
        mindim,
        truncerr,
        svd_min=svd_floor,
        chi_policy,
        q,
        alpha,
        renormalize,
        return_stats,
    )
    return_stats ? (ψ, _aggregate_evolution_stats(updates)) : ψ
end

#---------------------------------------------------------------------------------------------------
# Multi-Site Operators
#---------------------------------------------------------------------------------------------------
"""
    convert_operator(mat, d, n)

Convert a `d^n × d^n` local operator into the column-major tensor convention
used internally by this package.

Parameters:
- `mat`
  Dense operator written in the conventional site ordering.
- `d`
  Local Hilbert-space dimension per site.
- `n`
  Number of sites acted on by the operator.

Returns:
- A dense matrix with the same shape as `mat`, but reordered to match the
  package's column-major tensor convention.

Notes:
- This is useful when importing local operators written in a different index
  ordering.
"""
function convert_operator(mat::AbstractMatrix, d::Integer, n::Integer)
    tensor = reshape(mat, fill(d, 2n)...)
    perm = [n:-1:1; 2n:-1:n+1]
    tensor = permutedims(tensor, perm)
    reshape(tensor, size(mat))
end
