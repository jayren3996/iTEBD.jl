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
  Probability vector or Schmidt-weight vector. The function does not normalize
  `p`, so callers are expected to pass a properly normalized spectrum when that
  interpretation matters.

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
    for si in S
        if si > cutoff
            EE -= si * log(si)
        end
    end
    EE
end

"""
    natural_bonddim(λ; q=1.0, alpha=0.1, cutoff=1e-12)

Return a smooth spectrum-derived estimate of the intrinsic bond dimension
associated with Schmidt values `λ`.

The score is based on the Renyi effective rank of the normalized weights
`p_i = abs2(λ_i) / sum(abs2, λ)` and is optionally amplified by the total tail
weight `1 - p_1`, where `p_1` is the largest normalized weight.
"""
function natural_bonddim(
    λ::AbstractVector;
    q::Real=1.0,
    alpha::Real=0.1,
    cutoff::Real=1e-12
)
    q > 0 || throw(ArgumentError("q must be positive"))
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
    entropy = if isapprox(q, 1.0; atol=sqrt(eps(Float64)))
        entanglement_entropy(probs; cutoff=cutoff)
    else
        log(sum(p -> p^q, probs)) / (1 - q)
    end
    rank = exp(entropy)
    tail_weight = 1 - first(probs)
    rank * (1 + alpha * tail_weight)
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
    cutoff::Real=1e-12
)
    mindim > 0 || throw(ArgumentError("mindim must be positive"))
    maxdim >= mindim || throw(ArgumentError("maxdim must be at least mindim"))

    raw = natural_bonddim(λ; q, alpha, cutoff)
    target = max(previous, mindim, ceil(Int, raw))
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
function inner_product(T)
    trmat = trm(T)
    val, vec = eigsolve(trmat)
    abs(val[1])
end
#---------------------------------------------------------------------------------------------------
inner_product(T::iMPS) = inner_product(T, T)
#---------------------------------------------------------------------------------------------------
function inner_product(T1, T2)
    trmat = gtrm(T1, T2)
    val, vec = eigsolve(trmat)
    abs(val[1])
end
