"""
    imps2mps(ψ, sites; L=length(sites))

Convert an `iMPS` into a length-`L` `ITensorMPS.MPS` by unrolling the periodic
unit cell across the supplied physical-site index collection. Site `i` of the
output borrows its tensor from `ψ.Γ[mod1(i, ψ.n)]`, so the unit cell is
repeated as many times as needed to reach length `L`.

Parameters:
- `ψ`
  Source `iMPS`. Read-only; not mutated.
- `sites`
  Collection of `ITensors.Index` objects of length at least `L`, one per
  physical site in the unrolled chain.

Keyword arguments:
- `L=length(sites)`
  Number of physical sites in the output `MPS`. Must be `>= 0` and no greater
  than `length(sites)`.

Returns:
- A finite `ITensorMPS.MPS` of length `L` whose virtual bonds are reused from
  the iMPS unit cell. For `L >= 2` the wraparound bond is reused as the left
  boundary of site 1; for `L == 1` a fresh `Index` is allocated for the left
  boundary to avoid collapsing the only tensor's left and right legs onto the
  same `Index` object.

Notes:
- Bond dimensions of `ψ.Γ` must match the supplied `sites` (physical leg) and
  the unit-cell wraparound (virtual legs). Mismatches throw `ArgumentError`.
- This is the bridge to `ITensorMPS.jl` for downstream analysis (DMRG, finite
  observables, etc.) using a finite-chain projection of the infinite state.
"""
function imps2mps(ψ::iMPS, s; L::Integer=length(s))
    L >= 0 || throw(ArgumentError("L must be nonnegative"))
    length(s) >= L || throw(ArgumentError("site index collection must contain at least L indices"))
    L == 0 && return MPS(ITensor[])

    sites = collect(Iterators.take(s, L))
    links = [Index(size(ψ.Γ[mod1(i, ψ.n)], 3), "Link,l=$i") for i in 1:L]
    # The wrap-around convention reuses `links[L]` as the left bond of site 1.
    # For `L == 1` that would collapse the left and right legs of the only
    # tensor onto the same Index object, producing a self-contracted ITensor.
    # Allocate a distinct boundary Index in that case.
    boundary_left = if L == 1
        Index(size(ψ.Γ[1], 1), "Link,l=0")
    else
        links[L]
    end
    tensors = Vector{ITensor}(undef, L)

    for i in 1:L
        Γ = ψ.Γ[mod1(i, ψ.n)]
        size(Γ, 2) == dim(sites[i]) ||
            throw(ArgumentError("site index dimension does not match iMPS physical dimension at site $i"))
        left = i == 1 ? boundary_left : links[i - 1]
        right = links[i]
        size(Γ, 1) == dim(left) && size(Γ, 3) == dim(right) ||
            throw(ArgumentError("iMPS bond dimensions are incompatible at site $i"))
        tensors[i] = ITensor(Γ, left, sites[i], right)
    end

    return MPS(tensors)
end
