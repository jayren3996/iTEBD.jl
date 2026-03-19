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
  Maximum bond dimension retained during the decomposition of the updated block.
- `cutoff=SVDTOL`
  Singular-value cutoff used during that decomposition.
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
    maxdim=MAXDIM, cutoff=SVDTOL, renormalize=false
)
    n = length(Γs)
    isone(n) && return ([GΓ], [])
    Γ = tensor_group(Γs)
    tensor_lmul!(λl, Γ)
    GΓ = tensor_umul(G, Γ)
    tensor_decomp!(GΓ, λl, n; maxdim, cutoff, renormalize)
end
#---------------------------------------------------------------------------------------------------
export applygate!
"""
    applygate!(ψ, G, i, j; maxdim=MAXDIM, cutoff=SVDTOL, renormalize=true)

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
  Maximum temporary bond dimension used when decomposing the updated block.
- `cutoff=SVDTOL`
  Singular-value cutoff used during that decomposition.
- `renormalize=true`
  Whether to renormalize the retained Schmidt values.

Returns:
- The same object `ψ`, mutated in place.

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
    cutoff::Real=SVDTOL, 
    renormalize::Bool=true
)
    if isequal(i, j)
        ψ.Γ[i] = tensor_umul(G, ψ.Γ[i])
        return ψ
    end
    inds = j>i ? collect(i:j) : [i:ψ.n; 1:j]
    Γs = ψ.Γ[inds]
    λl = ψ.λ[mod(i-2,ψ.n)+1]
    Γs, λs = tensor_applygate!(G, Γs, λl; maxdim, cutoff, renormalize)
    push!(λs, ψ.λ[j])
    for i in eachindex(inds) 
        ψ.Γ[inds[i]] = Γs[i]
        ψ.λ[inds[i]] = λs[i]
    end
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
    j > i ? collect(i:j) : [i:ψ.n; 1:j]
end

export evolve!
"""
    evolve!(ψ, gates, steps; chi_policy=:fixed, maxdim=MAXDIM, mindim=1, q=1.0, alpha=0.1, cutoff=SVDTOL, renormalize=true)

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
- `chi_policy=:fixed`
  Bond-dimension policy. Use `:fixed` for standard fixed-`maxdim` evolution or
  `:adaptive` to ratchet the bond dimension with [`adaptive_bonddim`](@ref).
- `maxdim=MAXDIM`
  Maximum temporary bond dimension used during gate application. In adaptive
  mode this is also the hard upper cap.
- `mindim=1`
  Minimum bond dimension allowed in adaptive mode.
- `q=1.0`, `alpha=0.1`
  Parameters passed to [`adaptive_bonddim`](@ref) in adaptive mode.
- `cutoff=SVDTOL`
  Singular-value cutoff used during gate application and any subsequent
  canonicalization.
- `renormalize=true`
  Whether to renormalize Schmidt values after decomposition and canonicalization.

Returns:
- The same object `ψ`, mutated in place.

Behavior:
- With `chi_policy = :fixed`, each update is applied with the supplied
  `maxdim`.
- With `chi_policy = :adaptive`, each update is first applied up to `maxdim`,
  then the state is compressed back to a non-decreasing bond dimension chosen
  from the updated Schmidt spectra.

Notes:
- Adaptive mode is intentionally conservative: it probes with the larger working
  dimension `maxdim`, then projects back down to the ratcheted `χ`.
- The `gates` argument is kept explicit rather than hiding the sweep pattern in
  a higher-level model object.

Example:
```julia
X = [0 1; 1 0]
psi = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
gates = [(kron(X, X), 1, 2), (kron(X, X), 2, 1)]
evolve!(psi, gates, 5; chi_policy=:fixed, maxdim=4)
```
"""
function evolve!(
    ψ::iMPS,
    gates,
    steps::Integer;
    chi_policy::Symbol=:fixed,
    maxdim::Integer=MAXDIM,
    mindim::Integer=1,
    q::Real=1.0,
    alpha::Real=0.1,
    cutoff::Real=SVDTOL,
    renormalize::Bool=true
)
    steps >= 0 || throw(ArgumentError("steps must be non-negative"))
    maxdim > 0 || throw(ArgumentError("maxdim must be positive"))
    mindim > 0 || throw(ArgumentError("mindim must be positive"))
    maxdim >= mindim || throw(ArgumentError("maxdim must be at least mindim"))

    χ = min(maxdim, max(mindim, maximum(length.(ψ.λ))))

    for _ in 1:steps
        for gate in gates
            G, i, j = gate
            if chi_policy === :fixed
                applygate!(ψ, G, i, j; maxdim, cutoff, renormalize)
            elseif chi_policy === :adaptive
                applygate!(ψ, G, i, j; maxdim, cutoff, renormalize)
                for k in _gate_indices(ψ, i, j)
                    χ = adaptive_bonddim(χ, ψ.λ[k]; mindim, maxdim, q, alpha, cutoff)
                end
                canonical!(ψ; maxdim=χ, cutoff, renormalize)
            else
                throw(ArgumentError("unknown chi_policy $(repr(chi_policy)); use :fixed or :adaptive"))
            end
        end
    end

    ψ
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
