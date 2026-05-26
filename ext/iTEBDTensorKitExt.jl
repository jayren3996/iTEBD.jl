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

# `SymmetricIMPS` — the TensorKit-backed variant of `iMPS`. The exact
# `AbstractTensorMap` parameters are deliberately unconstrained here so the
# alias matches any U(1)/Z_N/product-sector graded-space tensors that later
# constructors produce. The wraparound bond-space check (Chunk 4) and all
# algorithm specialisations (Chunks 5-7) dispatch on this alias.
const SymmetricIMPS = iMPS{<:AbstractTensorMap, <:DiagonalTensorMap}
export SymmetricIMPS

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

# Remaining method bodies are populated by subsequent chunks:
#   Chunk 4 — rand_iMPS / product_iMPS symmetric constructors,
#             _validate_iMPS_bonds specialisation
#   Chunks 5-7 — canonical!, applygate!, expect, energy_density, ent_S

end # module
