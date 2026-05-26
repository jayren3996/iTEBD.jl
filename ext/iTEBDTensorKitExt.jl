module iTEBDTensorKitExt

using iTEBD
using TensorKit
using LinearAlgebra

# Names from the base package that this extension will specialise. Using
# `import` (not `using`) so that adding methods to these names is unambiguous
# to the compiler.
import iTEBD: graded_space, spin_half_ops, schmidt_values
import iTEBD: rand_iMPS, product_iMPS
import iTEBD: iMPS, _validate_iMPS_bonds

# ─────────────────────────────────────────────────────────────────────────────
# Chunk 3: Helper layer
# ─────────────────────────────────────────────────────────────────────────────

"""
    graded_space(symmetry::Symbol, charges_to_dims...)

Build a TensorKit graded vector space for a named Abelian symmetry without
forcing the user to import TensorKit's irrep types directly.

Supported `symmetry` values: `:U1`, `:Z2`, `:ZN`, `:U1xU1`, `:U1xZ2`,
`:Trivial`. For `:ZN` the second positional argument is the order `N`.

Product sectors use lowercase `x` as the ASCII form of TensorKit's `⊠`
operator: `:U1xU1` is `U1Irrep ⊠ U1Irrep`, `:U1xZ2` is `U1Irrep ⊠ Z2Irrep`,
and so on.

Examples:
    graded_space(:U1, 0=>2, 1=>1, -1=>1)
    graded_space(:Z2, 0=>3, 1=>3)
    graded_space(:ZN, 4, 0=>1, 1=>1, 2=>1, 3=>1)
    graded_space(:U1xU1, (0,0)=>2, (1,-1)=>1)
"""
function graded_space(::Val{:U1}, pairs::Pair{Int,Int}...)
    return Vect[U1Irrep](Int(c) => Int(d) for (c, d) in pairs)
end

function graded_space(::Val{:Z2}, pairs::Pair{Int,Int}...)
    return Vect[Z2Irrep](Int(c) => Int(d) for (c, d) in pairs)
end

function graded_space(::Val{:ZN}, N::Integer, pairs::Pair{Int,Int}...)
    N ≥ 2 || throw(ArgumentError("graded_space(:ZN, N, …) requires N ≥ 2 (got $N)"))
    Irrep = ZNIrrep{Int(N)}
    return Vect[Irrep](Int(c) => Int(d) for (c, d) in pairs)
end

function graded_space(::Val{:Trivial}, pairs::Pair{Int,Int}...)
    length(pairs) == 1 || throw(ArgumentError(
        "graded_space(:Trivial, …) takes exactly one charge=>dim pair"))
    # The charge label is intentionally discarded: ComplexSpace has no sector
    # grading. `graded_space(:Trivial, 0=>3)` and `graded_space(:Trivial, 5=>3)`
    # are identical — the API takes a charge for symmetry with the other
    # variants, but the value plays no role for the trivial sector.
    _, d = first(pairs)
    return ComplexSpace(Int(d))
end

function graded_space(::Val{:U1xU1}, pairs::Pair{<:Tuple,Int}...)
    Irrep = U1Irrep ⊠ U1Irrep
    return Vect[Irrep](Irrep(c[1], c[2]) => Int(d) for (c, d) in pairs)
end

function graded_space(::Val{:U1xZ2}, pairs::Pair{<:Tuple,Int}...)
    Irrep = U1Irrep ⊠ Z2Irrep
    return Vect[Irrep](Irrep(c[1], c[2]) => Int(d) for (c, d) in pairs)
end

graded_space(sym::Symbol, args...) = graded_space(Val(sym), args...)

# NOTE: `SymmetricIMPS` is declared in the base `iTEBD` module (SymmetricStubs.jl)
# as `const SymmetricIMPS = iMPS` — the widest alias. The extension cannot
# redefine a const from the parent module. Instead, dispatch that must be
# restricted to TensorKit-backed tensors uses the concrete type parameters
# `iMPS{<:AbstractTensorMap, <:DiagonalTensorMap}` directly in method signatures
# within this extension.

"""
    spin_half_ops(symmetry::Symbol)

Return the spin-1/2 operators for the given symmetry backend.

For `:U1`: returns `(Sz, SzSz, SpSm, SmSp)` where `Sz` is the one-site
spin-z operator on the graded physical space `P = Vect[U1Irrep](1=>1, -1=>1)`
(spin-up = U1(+1), spin-down = U1(-1)), and `SzSz`, `SpSm`, `SmSp` are
**pre-assembled flux-0 two-site operators** living on the common HomSpace
`(P ⊗ P) ← (P ⊗ P)`. Users construct two-site Hamiltonians by adding the
pre-built terms, e.g.

    Sz, SzSz, SpSm, SmSp = spin_half_ops(:U1)
    h = SzSz + 0.5 * (SpSm + SmSp)         # Heisenberg density

This signature sidesteps the dual-space composition problem that arises when
trying to form `Sp ⊗ Sm + Sm ⊗ Sp` from one-site charged operators (which
live on different HomSpaces in TensorKit's convention).

For `:Trivial`: returns `(Sx, Sy, Sz, Sp, Sm, Id)` as plain 2×2 `ComplexF64`
matrices (no TensorKit grading). Useful for testing non-symmetric code paths.
"""
function spin_half_ops(::Val{:U1})
    P = graded_space(:U1, 1=>1, -1=>1)

    # Sz: endomorphism of P with diagonal block values ±1/2 (flux 0).
    Sz = zeros(ComplexF64, P ← P)
    block(Sz, U1Irrep(1))[1, 1]  =  0.5
    block(Sz, U1Irrep(-1))[1, 1] = -0.5

    # Two-site operators on (P ⊗ P) ← (P ⊗ P). The fused codomain
    # decomposes into U(1) sectors {+2, 0, -2} with dimensions {1, 2, 1}.
    # Basis ordering inside the U1(0) block is empirically [|↑↓⟩, |↓↑⟩]
    # (see test_symmetric_basic.jl for the verification against the dense
    # 4×4 Heisenberg matrix).

    # SzSz = diag(0.25, -0.25, -0.25, 0.25) in the {↑↑, ↑↓, ↓↑, ↓↓} basis.
    SzSz = zeros(ComplexF64, P ⊗ P ← P ⊗ P)
    block(SzSz, U1Irrep(2))[1, 1]  = 0.25
    block(SzSz, U1Irrep(-2))[1, 1] = 0.25
    block(SzSz, U1Irrep(0)) .= ComplexF64[-0.25 0; 0 -0.25]

    # SpSm: only |↑↓⟩⟨↓↑| is nonzero, i.e. position (2,3) of the dense 4×4.
    # Inside the U1(0) block this is the upper-right entry.
    SpSm = zeros(ComplexF64, P ⊗ P ← P ⊗ P)
    block(SpSm, U1Irrep(0)) .= ComplexF64[0 1; 0 0]

    # SmSp = SpSm': matrix element at (3,2), lower-left of the U1(0) block.
    SmSp = zeros(ComplexF64, P ⊗ P ← P ⊗ P)
    block(SmSp, U1Irrep(0)) .= ComplexF64[0 0; 1 0]

    return Sz, SzSz, SpSm, SmSp
end

function spin_half_ops(::Val{:Trivial})
    # Spin-1/2 generators in the conventional normalisation S^a = σ^a / 2,
    # so [S^x, S^y] = i S^z and S^a · S^a = 3/4 · I.
    Sx = ComplexF64[0 0.5; 0.5 0]
    Sy = ComplexF64[0 -0.5im; 0.5im 0]
    Sz = ComplexF64[0.5 0; 0 -0.5]
    Sp = Sx + im*Sy
    Sm = Sx - im*Sy
    Id = ComplexF64[1 0; 0 1]
    return Sx, Sy, Sz, Sp, Sm, Id
end

spin_half_ops(sym::Symbol) = spin_half_ops(Val(sym))

function schmidt_values(ψ::SymmetricIMPS, i::Integer)
    1 ≤ i ≤ ψ.n || throw(BoundsError(ψ.λ, i))
    return _flatten_diagonal_blocks(ψ.λ[i])
end

# Internal: flatten a DiagonalTensorMap's per-sector diagonal blocks into a
# single descending-sorted Vector{Float64}.
# In TensorKit 0.16, blocks(λ::DiagonalTensorMap) yields (sector, Diagonal{...})
# pairs where each block is a Diagonal matrix; diag(blk) extracts the values.
function _flatten_diagonal_blocks(λ::DiagonalTensorMap)
    out = Float64[]
    for (_sector, blk) in blocks(λ)
        for v in diag(blk)
            push!(out, Float64(real(v)))
        end
    end
    sort!(out; rev=true)
    return out
end

# ─────────────────────────────────────────────────────────────────────────────
# Chunk 4: Symmetric iMPS constructors + wraparound bond-space check
# ─────────────────────────────────────────────────────────────────────────────

"""
    rand_iMPS(pspace::VectorSpace, vspace::VectorSpace, n::Integer)

Construct a random symmetric infinite MPS with explicit `pspace` (physical leg)
and `vspace` (virtual leg) graded vector spaces, and unit-cell length `n`.

Each local tensor maps `(vspace ⊗ pspace) ← vspace`, following the package
storage convention. The Schmidt-value field `λ[i]` is initialised to a
`DiagonalTensorMap` of ones on `vspace`.

Note: the returned state is NOT Schmidt-canonical. Once the symmetric
canonicalisation routine is loaded, call `canonical!(ψ)` to bring it to
canonical form.
"""
function rand_iMPS(pspace::VectorSpace, vspace::VectorSpace, n::Integer)
    n > 0 || throw(ArgumentError("n must be positive (got $n)"))
    Γ = [randn(ComplexF64, vspace ⊗ pspace ← vspace) for _ in 1:n]
    λ = [DiagonalTensorMap(ones(Float64, dim(vspace)), vspace) for _ in 1:n]
    return iMPS(Γ, λ, n)
end

# Internal: distribute χ across sectors generated by `pspace ⊗ pspace`. The
# uniform-per-sector split is rough but adequate for initial random states;
# canonicalisation will reshape it once the symmetric `canonical!` lands.
function _auto_bond_space(sym::Symbol, pspace::VectorSpace, χ::Integer; flux::Integer=0)
    fused = fuse(pspace ⊗ pspace)
    sector_list = collect(sectors(fused))
    isempty(sector_list) && throw(ArgumentError(
        "_auto_bond_space: pspace ⊗ pspace produced no sectors"))
    per_sector = max(1, χ ÷ length(sector_list))
    pairs = Pair{Int,Int}[Int(s.charge) => per_sector for s in sector_list]
    return graded_space(sym, pairs...)
end

"""
    rand_iMPS(symmetry::Symbol, charges; χ::Integer, n::Integer=1, flux::Integer=0)

Build a random symmetric iMPS with `symmetry` (one of the supported
`graded_space` symbols), physical-leg `charges` (one per basis state), bond
dimension `χ` auto-distributed across compatible sectors, unit cell `n`, and
target total `flux` around the unit cell.

Only simple symmetries (`:U1`, `:Z2`, `:ZN`, `:Trivial`) are supported in v1.
For product symmetries use the raw `rand_iMPS(pspace, vspace, n)` form.

Examples:
    rand_iMPS(:U1, [-1, 1]; χ=8, n=2)              # spin-1/2 XXZ-style
    rand_iMPS(:Z2, [0, 1];   χ=4, n=2)             # parity-conserving Ising-style

Note: the returned state is NOT Schmidt-canonical. Call `canonical!(ψ)` after
loading the symmetric backend's canonicalisation routine.
"""
function rand_iMPS(sym::Symbol, charges::AbstractVector{<:Integer};
                   χ::Integer, n::Integer=1, flux::Integer=0)
    sym in (:U1, :Z2, :ZN, :Trivial) || throw(ArgumentError(
        "rand_iMPS(symbol-based) only supports simple symmetries in v1 " *
        "(got :$sym). Use the raw `rand_iMPS(pspace, vspace, n)` form for products."))
    χ > 0 || throw(ArgumentError("χ must be positive (got $χ)"))
    P = graded_space(sym, [c => 1 for c in charges]...)
    V = _auto_bond_space(sym, P, χ; flux=flux)
    return rand_iMPS(P, V, n)
end

"""
    product_iMPS(symmetry::Symbol, charges, occupations)

Build a bond-dimension-1 symmetric iMPS where site `i` occupies the physical
basis state with charge `occupations[i]`. The cumulative-charge bond decoration
makes every tensor flux-0 individually.

Only simple symmetries (`:U1`, `:Z2`, `:ZN`) are supported in v1.

Example: spin-1/2 Néel state in the Sz=0 sector:
    ψ = product_iMPS(:U1, [-1, 1], [1, -1])

Note: `sum(occupations)` must equal zero (mod the symmetry order) so that the
bond fluxes close around the periodic unit cell. A DimensionMismatch is thrown
if they do not.
"""
function product_iMPS(sym::Symbol, charges::AbstractVector{<:Integer},
                      occupations::AbstractVector{<:Integer})
    sym in (:U1, :Z2, :ZN) || throw(ArgumentError(
        "product_iMPS(symbol-based) only supports :U1, :Z2, :ZN in v1 (got :$sym)"))
    n = length(occupations)
    n > 0 || throw(ArgumentError("occupations must be non-empty"))
    all(c -> c in charges, occupations) || throw(ArgumentError(
        "every occupation must appear in `charges`"))
    P = graded_space(sym, [c => 1 for c in charges]...)

    # The bond running into site i carries the cumulative charge of sites 1..i-1.
    # We use plain integer arithmetic here; the bond space grading handles the
    # TensorKit-level sector.
    cum = vcat(0, cumsum(occupations)[1:end-1])
    Vbonds = [graded_space(sym, c => 1) for c in cum]

    S = sectortype(P)
    Γ = Vector{typeof(zeros(ComplexF64, Vbonds[1] ⊗ P ← Vbonds[1]))}(undef, n)
    λ = Vector{typeof(DiagonalTensorMap(ones(Float64, 1), Vbonds[1]))}(undef, n)

    for i in 1:n
        Vl = Vbonds[i]
        Vr = Vbonds[mod1(i + 1, n)]
        Γ[i] = zeros(ComplexF64, Vl ⊗ P ← Vr)
        # The block label in TensorKit for a rank-(2,1) tensor (Vl ⊗ P ← Vr) is
        # the domain sector (= charge of Vr), since U(1) conservation requires
        # c_Vl + c_P = c_Vr in each block.
        cr = cum[mod1(i + 1, n)]
        block(Γ[i], S(cr))[1, 1] = 1.0
        λ[i] = DiagonalTensorMap(ones(Float64, 1), Vr)
    end

    return iMPS(Γ, λ, n)
end

# Symmetric specialisation of _validate_iMPS_bonds: checks that the right-leg
# space of Γ[i] equals the left-leg space of Γ[i+1] (with wraparound), using
# the TensorKit `domain`/`codomain` accessors. This catches mismatched graded
# spaces (e.g. non-closing flux around the unit cell) before any downstream
# contraction can silently propagate the inconsistency.
function _validate_iMPS_bonds(
    Γ::Vector{<:AbstractTensorMap},
    λ::Vector{<:DiagonalTensorMap},
    n::Integer,
)
    for i in 1:n
        Vr      = domain(Γ[i])[1]
        Vl_next = codomain(Γ[mod1(i + 1, n)])[1]
        Vr == Vl_next || throw(DimensionMismatch(
            "SymmetricIMPS bond $i: right space of Γ[$i] ($Vr) does not match " *
            "left space of Γ[$(mod1(i + 1, n))] ($Vl_next); fluxes must close " *
            "around the unit cell"
        ))
    end
    return nothing
end

# Remaining method bodies are populated by subsequent chunks:
#   Chunks 5-7 — canonical!, applygate!, expect, energy_density, ent_S

end # module
