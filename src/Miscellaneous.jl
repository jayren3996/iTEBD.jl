#---------------------------------------------------------------------------------------------------
# Entropy
#---------------------------------------------------------------------------------------------------
export entanglement_entropy, natural_bonddim, adaptive_bonddim

"""
    entanglement_entropy(p; cutoff=1e-10)

Compute the von Neumann entropy `-sum(p .* log.(p))` of a probability vector
or Schmidt spectrum `p`, ignoring entries smaller than `cutoff`.

Parameters:
- `p`
  Probability vector or Schmidt-weight vector. The function normalizes `p` by
  `sum(p)` before applying the entropy formula.

Keyword arguments:
- `cutoff=1e-10`
  Entries not exceeding `cutoff` are ignored to avoid spurious contributions
  from numerical noise.

Returns:
- The scalar von Neumann entropy.

Example:
```julia
S = entanglement_entropy([0.5, 0.5])
```
"""
function entanglement_entropy(
    S::AbstractVector;
    cutoff::AbstractFloat=1e-10
)
    EE = 0.0
    total = sum(S)
    if total > 0 && isfinite(total)
        for si in S
            p = si / total
            if p > cutoff
                EE -= p * log(p)
            end
        end
    end
    EE
end

"""
    natural_bonddim(λ; q=1.0, alpha=0.1, cutoff=1e-12)

Return a smooth spectrum-derived estimate of the intrinsic bond dimension
associated with Schmidt values `λ`.

The score is based on the Rényi effective rank of order `q` on the normalized
weights `p_i = abs2(λ_i) / sum(abs2, λ)`, optionally amplified by the total
tail weight `1 - p_1` (where `p_1` is the largest normalized weight). The
boundary case `q = 0` reduces to the support-count rank (number of
above-cutoff Schmidt values); `q = 1` is the entropy rank; `q = 2` is the
participation-ratio rank.
"""
function natural_bonddim(
    λ::AbstractVector;
    q::Real=1.0,
    alpha::Real=0.1,
    cutoff::Real=1e-12
)
    q >= 0 || throw(ArgumentError("q must be non-negative"))
    alpha >= 0 || throw(ArgumentError("alpha must be non-negative"))
    cutoff >= 0 || throw(ArgumentError("cutoff must be non-negative"))

    weights = Float64[]
    total = 0.0
    for value in λ
        weight = abs2(value)
        if weight > cutoff
            push!(weights, weight)
            total += weight
        end
    end

    total > 0 || return 1.0

    probs = sort!(weights ./ total; rev=true)
    entropy = if isapprox(q, 0; atol=sqrt(eps(Float64)))
        # Support-count rank: exp(entropy) = number of above-cutoff modes.
        log(length(probs))
    elseif isapprox(q, 1.0; atol=sqrt(eps(Float64)))
        entanglement_entropy(probs; cutoff=cutoff)
    else
        log(sum(p -> p^q, probs)) / (1 - q)
    end
    rank = exp(min(entropy, log(prevfloat(typemax(Float64)))))
    if !isfinite(rank)
        rank = prevfloat(typemax(Float64))
    end
    tail_weight = 1 - first(probs)
    result = rank * (1 + alpha * tail_weight)
    isfinite(result) ? result : prevfloat(typemax(Float64))
end

"""
    adaptive_bonddim(previous, λ; mindim=1, maxdim=MAXDIM, q=1.0, alpha=0.1, cutoff=1e-12)

Return a non-decreasing bond dimension by ratcheting the smooth intrinsic
estimate from [`natural_bonddim`](@ref) between `mindim` and `maxdim`.
"""
function adaptive_bonddim(
    previous::Integer,
    λ::AbstractVector;
    mindim::Integer=1,
    maxdim::Integer=MAXDIM,
    q::Real=1.0,
    alpha::Real=0.1,
    cutoff::Real=1e-12,
    ratchet::Bool=true
)
    mindim > 0 || throw(ArgumentError("mindim must be positive"))
    maxdim >= mindim || throw(ArgumentError("maxdim must be at least mindim"))

    raw = natural_bonddim(λ; q, alpha, cutoff)
    target = if ratchet
        max(previous, mindim, ceil(Int, raw))
    else
        max(mindim, ceil(Int, raw))
    end
    min(maxdim, target)
end

#---------------------------------------------------------------------------------------------------
# Inner Product
#---------------------------------------------------------------------------------------------------
export inner_product
"""
    inner_product(T)
    inner_product(T1, T2)

Return the overlap per unit cell defined by the dominant eigenvalue of a
transfer matrix.

For two translationally invariant tensor networks or `iMPS` objects with the
same unit cell, this routine first constructs the mixed unit-cell transfer
matrix

`E_cell(T1, T2)`,

and then returns

`|λ_max(E_cell(T1, T2))|`,

the magnitude of its dominant eigenvalue. For the one-argument form
`inner_product(T)`, this reduces to the corresponding norm per unit cell,
obtained from the dominant eigenvalue of the self-transfer matrix.

Returns:
- A scalar overlap or norm per unit cell extracted from the dominant
  transfer-matrix eigenvalue.

Notes:
- This is not the ordinary finite-system inner product obtained by contracting a
  finite chain.
- Instead, it is the infinite-system quantity associated with one periodic unit
  cell of the transfer matrix.
- For `iMPS` inputs, the relevant transfer matrix is the product over the whole
  unit cell, not a single local tensor.
- The implementation returns the absolute value of the dominant eigenvalue.
"""
inner_product(T::iMPS) = inner_product(T, T)
inner_product(mps1::iMPS, mps2::iMPS) = inner_product(mps1.Γ, mps2.Γ)
#---------------------------------------------------------------------------------------------------
inner_product(T::AbstractArray{<:Number, 3}) = inner_product(T, T)
function inner_product(
    T1::AbstractArray{<:Number, 3},
    T2::AbstractArray{<:Number, 3},
)
    _dominant_chain_eigenvalue([T1], [T2])
end
#---------------------------------------------------------------------------------------------------
function inner_product(Ts::AbstractVector{<:AbstractArray{<:Number, 3}})
    inner_product(Ts, Ts)
end
function inner_product(
    T1s::AbstractVector{<:AbstractArray{<:Number, 3}},
    T2s::AbstractVector{<:AbstractArray{<:Number, 3}},
)
    _dominant_chain_eigenvalue(T1s, T2s)
end

# Threshold below which the dense `gtrm + eigen` path is competitive with the
# matrix-free Krylov sweep. For χ ≤ 8 the Arnoldi startup overhead matches or
# exceeds the cost of a full χ²×χ² eigendecomposition; above that, the
# matrix-free path is asymptotically better by O(χ³).
const _CHAIN_EIG_DENSE_THRESHOLD = 8

function _dominant_chain_eigenvalue(
    T1s::AbstractVector{<:AbstractArray{<:Number, 3}},
    T2s::AbstractVector{<:AbstractArray{<:Number, 3}};
    dense_threshold::Integer=_CHAIN_EIG_DENSE_THRESHOLD,
)
    n = length(T1s)
    n == length(T2s) || throw(ArgumentError(
        "T1s and T2s have different lengths: $n vs $(length(T2s))"
    ))
    n > 0 || throw(ArgumentError("T1s and T2s must be non-empty"))

    χL_T1 = size(T1s[1], 1)
    χR_T1 = size(T1s[end], 3)
    χL_T2 = size(T2s[1], 1)
    χR_T2 = size(T2s[end], 3)
    χL_T1 == χR_T1 && χL_T2 == χR_T2 || throw(DimensionMismatch(
        "chain transfer not square: T1 has left=$χL_T1 right=$χR_T1, " *
        "T2 has left=$χL_T2 right=$χR_T2"
    ))

    if max(χL_T1, χL_T2) <= dense_threshold
        return _dominant_eigenvalue_dense(gtrm(T1s, T2s))
    end

    T = promote_type(eltype(T1s[1]), eltype(T2s[1]))
    seed = zeros(T, χR_T2, χR_T1)
    @inbounds for i in 1:min(χR_T2, χR_T1)
        seed[i, i] = one(T)
    end

    # Reuse a single workspace across all Krylov matvecs. Without this the
    # chain sweep allocates ~n * χ² per matvec, which dominates wall time for
    # large unit cells (e.g. n=8 χ=32 → ~800 KB / matvec).
    ws = ChainTransferWorkspace(T1s, T2s)
    out_size = χR_T2 * χR_T1
    f = function (ρ_vec)
        ρ_mat = reshape(ρ_vec, χR_T2, χR_T1)
        result_view = apply_chain_transfer!(ws, T1s, T2s, ρ_mat; dir=:r)
        # Copy out of workspace into a fresh vector — eigsolve retains its
        # iterates, so they cannot alias the ping-pong buffers.
        out = Vector{T}(undef, out_size)
        copyto!(out, vec(result_view))
        return out
    end

    vals, _, info = eigsolve(f, vec(seed), 1, :LM; ishermitian=false)
    if info.converged < 1
        # Random unstructured transfers occasionally fail to converge with the
        # default Krylov budget; fall back to the dense path for correctness.
        @warn "Dominant chain eigenvalue Krylov solver did not converge; " *
              "falling back to dense path" info
        return _dominant_eigenvalue_dense(gtrm(T1s, T2s))
    end
    return abs(vals[1])
end

function _dominant_eigenvalue_dense(trmat::AbstractMatrix)
    if size(trmat, 1) <= 4096
        vals, _ = eigen(trmat)
        idx = argmax(abs.(vals))
        return abs(vals[idx])
    end
    vals, _, info = eigsolve(trmat, 1, :LM; ishermitian=false)
    if info.converged < 1
        # Previously this warned and returned abs(vals[1]) anyway, masking the
        # divergence. Callers downstream (norm, overlap, fixed-point checks)
        # would silently propagate the wrong value. Returning NaN forces the
        # caller to notice; the warning still fires so the cause is visible.
        @warn "Dominant eigenvalue Krylov solver did not converge; returning NaN" info
        return NaN
    end
    return abs(vals[1])
end
