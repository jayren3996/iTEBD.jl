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

The score is based on the Rényi effective rank of the normalized weights
`p_i = abs2(λ_i) / sum(abs2, λ)` and is optionally amplified by the total tail
weight `1 - p_1`, where `p_1` is the largest normalized weight:

`r_q = exp(-sum(p_i * log(p_i)))` when `q = 1`,

`r_q = (sum(p_i^q))^(1 / (1 - q))` when `q != 1`,

`χ_nat = r_q * (1 + alpha * (1 - p_1))`.

At `q = 1`, this reduces to the entropy rank `exp(-sum(p_i log p_i))`.
At `q = 2`, it reduces to the participation ratio / inverse participation
ratio (IPR) rank `1 / sum(p_i^2)`.

Parameters:
- `λ`
  Schmidt values on a bond. The routine internally uses the normalized weights
  `abs2.(λ) / sum(abs2, λ)`.

Keyword arguments:
- `q=1.0`
  Rényi-rank parameter controlling how conservative the truncation rule is.
  In the bond-truncation sense used by this package:
  smaller `q` gives a larger recommended bond dimension and is therefore more
  conservative, while larger `q` gives a smaller recommended bond dimension and
  is more aggressive.
- `alpha=0.1`
  Tail-weight amplification factor. Larger `alpha` further increases the
  recommended bond dimension when the Schmidt spectrum has a distributed tail,
  making truncation more conservative.
- `cutoff=1e-12`
  Values with squared magnitude below this threshold are ignored.

Returns:
- Smooth real-valued estimate of the intrinsic bond dimension suggested by the
  spectrum.

Qualitative behavior:
- `q = 1` corresponds to the entropy rank and is the default recommendation.
- `q = 2` corresponds to the participation ratio / IPR and is more aggressive
  because it discounts tails more strongly.
- `q < 1` is even more conservative than entropy rank because it gives more
  weight to small Schmidt values.
- `alpha = 0` disables the explicit tail-amplification factor.
- Increasing `alpha` never decreases `χ_nat`; it only protects long Schmidt
  tails more strongly.

For users who care primarily about truncation fidelity:
- `q = 1.0, alpha = 0.1` is the default compromise.
- Move to `q < 1` if the entropy-rank rule is still too aggressive.
- Move to `q > 1` or smaller `alpha` if the recommended bond dimensions are too
  large for the intended computation.

Example:
```julia
χnat = natural_bonddim([1 / sqrt(2), 1 / sqrt(2)])
```
"""
function natural_bonddim(
    λ::AbstractVector;
    q::Real=1.0,
    alpha::Real=0.1,
    cutoff::Real=1e-12
)
    q > 0 || throw(ArgumentError("q must be positive"))
    alpha >= 0 || throw(ArgumentError("alpha must be non-negative"))

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

Return a non-decreasing bond dimension for time evolution by ratcheting the
smooth intrinsic estimate from [`natural_bonddim`](@ref) between `mindim` and
`maxdim`.

The returned bond dimension is

`χ_new = min(maxdim, max(previous, mindim, ceil(Int, χ_nat)))`,

where `χ_nat = natural_bonddim(λ; q=q, alpha=alpha, cutoff=cutoff)`.

Parameters:
- `previous`
  Previously selected bond dimension.
- `λ`
  Current Schmidt values used to score the new bond demand.

Keyword arguments:
- `mindim=1`
  Lower bound on the returned bond dimension.
- `maxdim=MAXDIM`
  Upper bound on the returned bond dimension.
- `q=1.0`, `alpha=0.1`, `cutoff=1e-12`
  Parameters forwarded to [`natural_bonddim`](@ref).

Returns:
- Integer bond dimension satisfying
  `mindim <= χ_new <= maxdim` and `χ_new >= previous`.

Notes:
- In the truncation language used by this package, smaller `q` and larger
  `alpha` both make the returned bond dimension more conservative.
- The returned value is monotone in time because of the explicit ratchet with
  `previous`.

Example:
```julia
χ = adaptive_bonddim(4, [0.9, 0.3]; mindim=2, maxdim=16)
```
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
function inner_product(T1, T2)
    trmat = gtrm(T1, T2)
    val, vec = eigsolve(trmat)
    abs(val[1])
end
