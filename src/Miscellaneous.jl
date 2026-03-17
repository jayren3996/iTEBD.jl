#---------------------------------------------------------------------------------------------------
# Entropy
#---------------------------------------------------------------------------------------------------
export entanglement_entropy, natural_bonddim, adaptive_bonddim

"""
    entanglement_entropy(p; cutoff=1e-10)

Compute the von Neumann entropy `-sum(p .* log.(p))` of a probability vector
or Schmidt spectrum `p`, ignoring entries smaller than `cutoff`.
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
    natural_bonddim(λ; q=1.5, alpha=0.5, cutoff=1e-12)

Return a smooth spectrum-derived estimate of the intrinsic bond dimension
associated with Schmidt values `λ`.

The score is based on the Rényi effective rank of the normalized weights
`p_i = λ_i^2 / sum(abs2, λ)` and is optionally amplified by the total tail
weight `1 - p_1`:

`r_q = exp(S_q)` with `S_q = log(sum(p_i^q)) / (1 - q)`,

`χ_nat = r_q * (1 + alpha * (1 - p_1))`.

Smaller `q` is bolder, larger `q` is more conservative. The default
`q = 1.5` interpolates between entropy rank and participation ratio.
"""
function natural_bonddim(
    λ::AbstractVector;
    q::Real=1.5,
    alpha::Real=0.5,
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
    adaptive_bonddim(previous, λ; mindim=1, maxdim=MAXDIM, q=1.5, alpha=0.5, cutoff=1e-12)

Return a non-decreasing bond dimension for time evolution by ratcheting the
smooth intrinsic estimate from [`natural_bonddim`](@ref) between `mindim` and
`maxdim`.
"""
function adaptive_bonddim(
    previous::Integer,
    λ::AbstractVector;
    mindim::Integer=1,
    maxdim::Integer=MAXDIM,
    q::Real=1.5,
    alpha::Real=0.5,
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

Return the dominant transfer-matrix overlap.

For `inner_product(T)`, this is the norm of a single tensor network transfer
matrix. For `inner_product(T1, T2)`, it is the overlap per unit cell between
two tensor networks or `iMPS` objects.
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

