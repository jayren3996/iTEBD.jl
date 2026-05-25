#---------------------------------------------------------------------------------------------------
# iMPS 
#---------------------------------------------------------------------------------------------------
export iMPS
"""
    iMPS

Infinite matrix-product state with a finite unit cell.

Fields:
- `Γ::Vector{Array{T,3}}`
  Stored local tensors for one periodic unit cell. In this package these are
  the right-canonical tensors `B_i = Γ_i λ_i`, not the bare Vidal tensors
  usually denoted `Γ_i` in the iTEBD literature.
- `λ::Vector{Vector{Float64}}`
  Schmidt spectra across the bonds of the unit cell. `λ[i]` is the Schmidt
  spectrum on the bond to the right of site `i`, with periodic wraparound.
- `n::Int`
  Number of sites in the periodic unit cell.

Convention:
- `ψ.Γ[i]` stores the right-canonical tensor `B_i`.
- `ψ[i]` returns the bare Vidal tensor `Γ_i` together with `λ[i]`.

After canonicalization, the entanglement structure of the state is stored
explicitly in `λ`, while the local tensors in `Γ` are right-canonical.

Notes:
- The package is written for the injective setting used by standard iTEBD
  workflows.
- Many internal contractions operate directly on the stored tensors `B_i`
  rather than reconstructing bare Vidal tensors first.
"""
struct iMPS{T<:Number, S<:Real}
    Γ::Vector{Array{T, 3}}
    λ::Vector{Vector{S}}
    n::Int

    function iMPS(
        Γ::Vector{Array{T, 3}},
        λ::Vector{Vector{S}},
        n::Integer,
    ) where {T<:Number, S<:Real}
        n > 0 || throw(ArgumentError(
            "iMPS unit-cell length n must be positive (got $n)"))
        length(Γ) == n || throw(ArgumentError(
            "iMPS: length(Γ) = $(length(Γ)) but n = $n"))
        length(λ) == n || throw(ArgumentError(
            "iMPS: length(λ) = $(length(λ)) but n = $n"))
        for i in 1:n
            Dr_i = size(Γ[i], 3)
            Dl_next = size(Γ[mod1(i + 1, n)], 1)
            length(λ[i]) == Dr_i || throw(DimensionMismatch(
                "iMPS bond $i: length(λ[$i]) = $(length(λ[i])) " *
                "but size(Γ[$i], 3) = $Dr_i"))
            Dr_i == Dl_next || throw(DimensionMismatch(
                "iMPS bond $i: size(Γ[$i], 3) = $Dr_i but " *
                "size(Γ[$(mod1(i + 1, n))], 1) = $Dl_next " *
                "(bond dims must match at the wraparound seam)"))
            all(isfinite, λ[i]) || throw(ArgumentError(
                "iMPS λ[$i] contains non-finite values"))
            any(<(zero(S)), λ[i]) && throw(ArgumentError(
                "iMPS λ[$i] contains negative values"))
        end
        return new{T,S}(Γ, λ, Int(n))
    end
end
#---------------------------------------------------------------------------------------------------
"""
    iMPS(T, Γs; renormalize=true)

Construct an `iMPS` from a list of local tensors with element type `T`.

Parameters:
- `T`
  Element type used for the stored tensors.
- `Γs`
  Vector of local three-leg tensors, one per site in the unit cell. These are
  interpreted as raw input tensors; they do not need to already satisfy the
  package's canonical convention.

Keyword arguments:
- `renormalize=true`
  If `true`, the constructed state is immediately brought to the package's
  Schmidt-canonical form with [`canonical!`](@ref). If `false`, the tensors are
  stored as given and the Schmidt vectors are initialized as trivial all-ones
  placeholders.

Returns:
- A new [`iMPS`](@ref) with unit-cell length `length(Γs)`.

Notes:
- For normal user-facing construction, leaving `renormalize=true` is strongly
  recommended.
- When `renormalize=false`, the resulting object may not yet satisfy the stored
  right-canonical convention expected by many higher-level routines.

Example:
```julia
Γs = [rand(ComplexF64, 2, 2, 2), rand(ComplexF64, 2, 2, 2)]
psi = iMPS(ComplexF64, Γs; renormalize=true)
```
"""
function iMPS(
    T::DataType,
    Γs::AbstractVector{<:AbstractArray{<:Number, 3}};
    renormalize::Bool=true
)
    n = length(Γs)
    Γ = Array{T}.(Γs)
    λ = [ones(_schmidt_value_type(T), size(Γi, 3)) for Γi in Γs]
    ψ = iMPS(Γ, λ, n)
    return renormalize ? canonical!(ψ) : ψ
end

# Helper to get the appropriate real type for Schmidt values
_schmidt_value_type(::Type{T}) where T = real(T)

# Type-stable override for _support_tol (the generic method in TensorAlgebra.jl
# hard-codes Float64).  This more-specific method takes precedence for real
# Schmidt-value vectors and preserves the element type.
function _support_tol(vals::AbstractVector{S}; atol::Real=ZEROTOL, rtol::Real=sqrt(eps(S))) where {S<:Real}
    mags = abs.(vals)
    scale = isempty(mags) ? zero(S) : maximum(mags)
    return max(S(atol), S(rtol) * max(scale, one(S)))
end
#---------------------------------------------------------------------------------------------------
"""
    iMPS(Γs; renormalize=true)

Construct a complex-valued [`iMPS`](@ref) from a list of local tensors.

This is a convenience wrapper for `iMPS(ComplexF64, Γs; renormalize=...)`.
"""
function iMPS(Γs; renormalize::Bool=true)
    return iMPS(ComplexF64, Γs; renormalize)
end

#---------------------------------------------------------------------------------------------------
# INITIATE MPS
#---------------------------------------------------------------------------------------------------
export rand_iMPS
"""
    rand_iMPS(T, n, d, dim)
    rand_iMPS(n, d, dim)

Generate a random `iMPS` with:
- unit-cell size `n`,
- local Hilbert-space dimension `d`,
- bond dimension `dim`.

Parameters:
- `T`
  Element type of the random tensors in the typed method.
- `n`
  Number of sites in the unit cell.
- `d`
  Local Hilbert-space dimension at each site.
- `dim`
  Initial bond dimension used to sample the random local tensors.

Returns:
- A random [`iMPS`](@ref) already canonicalized with the package's default
  canonicalization settings.

Notes:
- The stored tensors in the returned state satisfy the package convention
  `ψ.Γ[i] = B_i = Γ_i λ_i`.
- The untyped method `rand_iMPS(n, d, dim)` uses `Float64`.

Example:
```julia
psi = rand_iMPS(ComplexF64, 2, 2, 4)
```
"""
function rand_iMPS(
    T::DataType,
    n::Integer,
    d::Integer,
    dim::Integer
)
    Γ = [rand(T, dim, d, dim) for i=1:n]
    iMPS(T, Γ; renormalize=true)
end
rand_iMPS(n, d, dim) = rand_iMPS(Float64, n, d, dim)
#---------------------------------------------------------------------------------------------------
export product_iMPS
"""
    product_iMPS(T, vectors)
    product_iMPS(vectors)

Construct a bond-dimension-1 `iMPS` from a list of local state vectors, one
for each site in the unit cell.

Parameters:
- `T`
  Element type used for the stored tensors in the typed method.
- `vectors`
  Vector of local state vectors, one per site in the unit cell. All vectors are
  assumed to have the same local dimension.

Returns:
- A bond-dimension-1 [`iMPS`](@ref), canonicalized on construction.

Notes:
- This is the easiest way to create simple product states such as N\'eel or
  `Z_2` states.

Example:
```julia
psi = product_iMPS(ComplexF64, [[0, 1], [1, 0], [0, 1], [1, 0]])
```
"""
function product_iMPS(
    T::DataType,
    v::AbstractVector{<:AbstractVector{<:Number}}
)
    !isempty(v) || throw(ArgumentError("vectors must contain at least one local state"))
    n = length(v)
    d = length(v[1])
    d > 0 || throw(ArgumentError("local state vectors must be nonempty"))
    Γ = [zeros(T, 1, d, 1) for i=1:n]
    λ = [ones(_schmidt_value_type(T), 1) for i=1:n]
    for i=1:n
        length(v[i]) == d || throw(ArgumentError("all local state vectors must have the same length"))
        vec = T.(v[i])
        all(isfinite, vec) || throw(ArgumentError("local state vectors must contain only finite values"))
        nrm = norm(vec)
        isfinite(nrm) && nrm > 0 || throw(ArgumentError("local state vectors must be nonzero"))
        Γ[i][1,:,1] .= vec ./ nrm
    end
    ψ = iMPS(Γ, λ, n)
    canonical!(ψ)
end
#---------------------------------------------------------------------------------------------------
function product_iMPS(v::AbstractVector{<:AbstractVector{<:Number}})
    T = float(promote_type(eltype.(v)...))
    product_iMPS(T, v)
end
#---------------------------------------------------------------------------------------------------
export canonical!
function _validate_canonical_options(maxdim, cutoff, noninjective::Symbol, symmetry_break::Symbol)
    maxdim isa Integer || throw(ArgumentError("maxdim must be an integer"))
    maxdim > 0 || throw(ArgumentError("maxdim must be positive"))
    cutoff isa Real || throw(ArgumentError("cutoff must be real"))
    isfinite(cutoff) && cutoff >= 0 || throw(ArgumentError("cutoff must be finite and non-negative"))
    noninjective in (:warn, :error, :ignore) ||
        throw(ArgumentError("noninjective must be one of :warn, :error, or :ignore"))
    symmetry_break in (:auto, :none) ||
        throw(ArgumentError("symmetry_break must be one of :auto or :none"))
    return nothing
end

"""
    canonical!(
        ψ;
        maxdim=MAXDIM,
        cutoff=SVDTOL,
        renormalize=true,
        noninjective=:warn,
        symmetry_break=:none,
    )

Bring `ψ` to Schmidt-canonical form in place.

This is the standard normalization step used throughout the package after
applying gates or rebuilding an `iMPS` from raw tensors.

Convention note:
- The package stores right-canonical tensors `B_i = Γ_i λ_i`.
- The Schmidt spectrum on bond `i` is still kept separately as `λ[i]`.
- Calling `ψ[i]` returns the bare Vidal tensor `Γ_i` together with `λ[i]`
  by dividing out the absorbed right Schmidt values.

This routine uses the standard injective canonicalization path, with a
conservative warning/sector-selection policy for likely non-injective inputs.

Parameters:
- `ψ`
  State to canonicalize in place.

Keyword arguments:
- `maxdim=MAXDIM`
  Maximum number of Schmidt values retained on each bond during the internal
  SVD step. This acts as a hard bond-dimension cap.
- `cutoff=SVDTOL`
  Singular values smaller than this threshold are discarded during
  canonicalization.
- `renormalize=true`
  If `true`, renormalize the retained Schmidt values after truncation.
- `noninjective=:warn`
  Policy for likely non-injective or degenerate transfer spectra. Supported
  values are `:warn`, `:error`, and `:ignore`.
- `symmetry_break=:none`
  Do not run the explicit deterministic sector-selection heuristic for likely
  non-injective tensors. Use `:auto` to enable that heuristic for simple
  block-diagonal/GHZ-like tensors.

Returns:
- The same object `ψ`, mutated in place.

Notes:
- The stored tensors after this call satisfy the package convention
  `ψ.Γ[i] = B_i = Γ_i λ_i`.
- If you canonicalize with a small `maxdim` or a large `cutoff`, the resulting
  state is the truncated state.
- The bare Vidal tensors can be recovered afterwards with `ψ[i]`.

Example:
```julia
psi = rand_iMPS(ComplexF64, 1, 2, 4)
canonical!(psi; maxdim=4, cutoff=1e-12, renormalize=true)
Gamma1, lambda1 = psi[1]
```
"""
function canonical!(
    ψ::iMPS;
    maxdim=MAXDIM,
    cutoff=SVDTOL,
    renormalize=true,
    noninjective::Symbol=:warn,
    symmetry_break::Symbol=:none,
)
    _validate_canonical_options(maxdim, cutoff, noninjective, symmetry_break)
    bond_dims_before = length.(ψ.λ)
    ψ.Γ[:], ψ.λ[:] = schmidt_canonical(
        ψ.Γ,
        ψ.λ[end];
        maxdim,
        cutoff,
        renormalize,
        noninjective,
        symmetry_break,
    )
    # When truncation changes any bond dimension, the per-bond SVDs in
    # tensor_decomp! leave the state only approximately right-canonical: the
    # discarded singular modes break the matrix identity that the per-site
    # U-factor rescaling relies on for its gauge. A second `schmidt_canonical`
    # pass on the already-truncated state restores the exact gauge.
    #
    # A cheaper LQ-sweep gauge restoration was prototyped (see
    # bench/bench_canonical_truncation.jl history) and found to produce a
    # state with 1e-4 lower fidelity to the input than the full re-pass: the
    # singular factor `S` left at the wrap bond after the LQ sweep cannot be
    # absorbed back into either bordering site without breaking the
    # per-site right-isometric condition, so dropping it changes the physical
    # state. The full re-pass remains the correct algorithm.
    if length.(ψ.λ) != bond_dims_before
        ψ.Γ[:], ψ.λ[:] = schmidt_canonical(
            ψ.Γ,
            ψ.λ[end];
            maxdim,
            cutoff,
            renormalize,
            noninjective=noninjective == :error ? :error : :ignore,
            symmetry_break=:none,
        )
    end
    return ψ
end

#---------------------------------------------------------------------------------------------------
# BASIC PROPERTIES
#---------------------------------------------------------------------------------------------------
eltype(::iMPS{T}) where T = T
#---------------------------------------------------------------------------------------------------
"""
    ψ[i]

Return the bare Vidal tensor `Γ_i` and Schmidt values `λ[i]` for site `i`.

The stored tensor in `ψ.Γ[i]` is the right-canonical tensor
`B_i = Γ_i λ_i`, so indexing divides out the absorbed right Schmidt
values before returning the local tensor.

Parameters:
- `i`
  Site index inside the periodic unit cell. Indices are wrapped periodically, so
  values outside `1:ψ.n` are reduced modulo the unit-cell length.

Returns:
- `(Γ_i, λ_i)` where `Γ_i` is the bare Vidal tensor on site `i` and `λ_i` is
  the Schmidt spectrum on the bond to its right.

Notes:
- This allocates a copy of the stored tensor before dividing out the absorbed
  Schmidt values.
- Use `ψ.Γ[i]` directly if you want the stored right-canonical tensor instead.
"""
function getindex(mps::iMPS{T,S}, i::Integer) where {T,S}
    i = mod(i-1, mps.n) + 1
    Γ = copy(mps.Γ[i])
    λ_internal = mps.λ[i]
    # Divide out the right Schmidt values in-place, avoiding the allocation
    # of a full reciprocal vector. Use rtol = 0 (only the absolute ZEROTOL
    # cuts in) so that small-but-physical Schmidt values are inverted, not
    # zeroed: a state with λ = [1.0, 1e-9] should still produce a Vidal Γ
    # with the second mode intact. If your state has noise-level Schmidt
    # values that you want dropped, recanonicalize with a tighter `cutoff`
    # first rather than relying on getindex to filter.
    β = size(Γ, 3)
    Γ_reshaped = reshape(Γ, :, β)
    tol = _support_tol(λ_internal; atol=ZEROTOL, rtol=zero(S))
    for j in 1:β
        scale = abs(λ_internal[j]) > tol ? inv(λ_internal[j]) : zero(S)
        @inbounds Γ_reshaped[:, j] .*= scale
    end
    # Return a copy of the Schmidt vector so callers cannot accidentally mutate
    # the iMPS's internal state.
    Γ, copy(λ_internal)
end
#---------------------------------------------------------------------------------------------------
"""
    mps_promote_type(T, mps)

Rebuild an `iMPS` with local tensors converted to element type `T`.

Parameters:
- `T`
  Target element type for the local tensors.
- `mps`
  Input state to convert.

Returns:
- A new [`iMPS`](@ref) whose local tensors have element type `T`.

Notes:
- The Schmidt spectra are reused unchanged.
- This is an internal helper used when adapting tensor element types.
"""
function mps_promote_type(
    T::DataType,
    mps::iMPS
)
    Γ_new = Array{T}.(mps.Γ)
    λ_new = [copy(s) for s in mps.λ]
    iMPS(Γ_new, λ_new, mps.n)
end
#---------------------------------------------------------------------------------------------------
"""
    ent_S(ψ, i)

Return the bipartite entanglement entropy across bond `i`.

Parameters:
- `ψ`
  Input state.
- `i`
  Bond index, wrapped periodically onto the unit cell.

Returns:
- The von Neumann entanglement entropy computed from `ψ.λ[i].^2`.

Notes:
- This uses the Schmidt spectrum stored in the canonical form. It is therefore
  most meaningful after the state has been canonicalized.

Example:
```julia
psi = rand_iMPS(ComplexF64, 2, 2, 4)
S = ent_S(psi, 1)
```
"""
function ent_S(mps::iMPS, i::Integer)
    j = mod(i-1, mps.n) + 1
    ρ = mps.λ[j] .^ 2
    entanglement_entropy(ρ)
end
#---------------------------------------------------------------------------------------------------
"""
    expect(ψ, O, i, j)

Expectation value of the local operator `O` acting on the contiguous region
from site `i` to site `j` inside the periodic unit cell of `ψ`.

The contraction is performed directly with the stored tensors
`B_i = Γ_i λ_i`.

Parameters:
- `ψ`
  Input state.
- `O`
  Dense local operator acting on the contiguous block from site `i` to site
  `j`. Its matrix dimension must match the local Hilbert-space dimension raised
  to the block length.
- `i`, `j`
  Start and end sites of the support inside the unit cell. If `j < i`, the
  support is interpreted with periodic wraparound.

Returns:
- The real expectation value of `O` in the state `ψ`.

Notes:
- This routine works directly with the stored right-canonical tensors rather
  than reconstructing bare Vidal tensors.
- The operator must act on a contiguous region matching the chosen interval.

Example:
```julia
psi = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
Z = [1 0; 0 -1]
val = expect(psi, kron(Z, Z), 1, 2)
```
"""
function expect(ψ::iMPS, O::AbstractMatrix, i::Integer, j::Integer)
    inds = j >= i ? collect(i:j) : [i:ψ.n; 1:j]
    Γ = ψ.Γ[inds]
    λl = ψ.λ[mod(i-2,ψ.n)+1]
    ocontract(Γ, O, λl) |> real
end


#---------------------------------------------------------------------------------------------------
# MANIPULATION
#---------------------------------------------------------------------------------------------------
"""
    conj(mps)

Complex-conjugate an `iMPS`.

Parameters:
- `mps`
  Input state.

Returns:
- A new [`iMPS`](@ref) whose stored local tensors are complex conjugated.

Notes:
- The Schmidt spectra are copied unchanged because they are stored as real
  vectors.
"""
function conj(mps::iMPS)
    Γ_new = [conj(B) for B in mps.Γ]
    λ_new = [copy(s) for s in mps.λ]
    iMPS(Γ_new, λ_new, mps.n)
end

#---------------------------------------------------------------------------------------------------
"""
    gtrm(mps1, mps2)

General transfer matrix between two `iMPS` objects.

If `mps1` and `mps2` have a common unit-cell length `n`, this returns the
unit-cell transfer matrix

`E_cell(mps1, mps2) = E_1 * E_2 * ... * E_n`,

where each `E_i` is the local mixed transfer matrix built from the stored local
tensors `mps1.Γ[i]` and `mps2.Γ[i]`.

This is a thin convenience wrapper for `gtrm(mps1.Γ, mps2.Γ)`.

Notes:
- The dominant eigenvalue of this unit-cell transfer matrix is the quantity used
  by [`inner_product`](@ref) to define the overlap per unit cell.
"""
gtrm(mps1::iMPS, mps2::iMPS) = gtrm(mps1.Γ, mps2.Γ)
