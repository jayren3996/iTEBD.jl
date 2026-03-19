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
struct iMPS{T<:Number}
    Γ::Vector{Array{T, 3}}
    λ::Vector{Vector{Float64}}
    n::Int64
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
    λ = [ones(Float64, size(Γi, 3)) for Γi in Γs]
    ψ = iMPS(Γ, λ, n)
    return renormalize ? canonical!(ψ) : ψ
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
    λ = [ones(dim) for i=1:n]
    iMPS(Γ, λ, n)
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
    n = length(v)
    d = length(v[1])
    Γ = [zeros(T, 1, d, 1) for i=1:n]
    λ = [ones(1) for i=1:n]
    for i=1:n
        Γ[i][1,:,1] .= v[i]
    end
    iMPS(Γ, λ, n)
end
#---------------------------------------------------------------------------------------------------
function product_iMPS(v::AbstractVector{<:AbstractVector{<:Number}})
    T = promote_type(eltype.(v)...)
    product_iMPS(T, v)
end
#---------------------------------------------------------------------------------------------------
export canonical!
"""
    canonical!(ψ; maxdim=MAXDIM, cutoff=SVDTOL, renormalize=true)

Bring `ψ` to Schmidt-canonical form in place.

This is the standard normalization step used throughout the package after
applying gates or rebuilding an `iMPS` from raw tensors.

Convention note:
- The package stores right-canonical tensors `B_i = Γ_i λ_i`.
- The Schmidt spectrum on bond `i` is still kept separately as `λ[i]`.
- Calling `ψ[i]` returns the bare Vidal tensor `Γ_i` together with `λ[i]`
  by dividing out the absorbed right Schmidt values.

This routine assumes the state is injective and is the standard
canonicalization path used by the package.

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
function canonical!(ψ::iMPS; maxdim=MAXDIM, cutoff=SVDTOL, renormalize=true)
    ψ.Γ[:], ψ.λ[:] = schmidt_canonical(ψ.Γ, ψ.λ[end]; maxdim, cutoff, renormalize)
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
function getindex(mps::iMPS, i::Integer)
    i = mod(i-1, mps.n) + 1
    Γ, λ = copy(mps.Γ[i]), mps.λ[i]
    tensor_rmul!(Γ, 1 ./ λ)
    Γ, λ
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
    Γ, λ, n = get_data(mps)
    Γ_new = Array{T}.(Γ)
    iMPS(Γ_new, λ, n)
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
    Γ = ψ.Γ[j>=i ? (i:j) : [i:ψ.n; 1:j]]
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
    Γ, λ, n = get_data(mps)
    iMPS(conj.(Γ), λ, n)
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
