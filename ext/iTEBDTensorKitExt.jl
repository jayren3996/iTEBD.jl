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

For `:U1`: returns `(Sz, Sp, Sm)` as TensorKit TensorMaps on the graded physical
space `P = Vect[U1Irrep](1=>1, -1=>1)` (spin-up = U1(+1), spin-down = U1(-1)).
`Sz` is an endomorphism of `P`; `Sp` lives on `P ← dual(P)` and `Sm = Sp'`.

For `:Trivial`: returns `(Sx, Sy, Sz, Sp, Sm, Id)` as plain 2×2 `ComplexF64`
matrices (no TensorKit grading). Useful for testing non-symmetric code paths.
"""
function spin_half_ops(::Val{:U1})
    P = graded_space(:U1, 1=>1, -1=>1)
    Pd = dual(P)

    # Sz: endomorphism of P with diagonal block values ±1/2
    Sz = zeros(ComplexF64, P ← P)
    block(Sz, U1Irrep(1))[1, 1]  =  0.5
    block(Sz, U1Irrep(-1))[1, 1] = -0.5

    # Sp: P ← dual(P). In TensorKit's dual-space convention for U1,
    # block(Sp, U1(+1)) is the 1×1 matrix that maps from dual(-1) [i.e. the
    # spin-down input in the contragredient sense] to the spin-up output.
    # This is the standard raising block.
    Sp = zeros(ComplexF64, P ← Pd)
    block(Sp, U1Irrep(1))[1, 1] = 1.0

    # Sm: the adjoint of Sp; lives on dual(P) ← P
    Sm = Sp'

    return Sz, Sp, Sm
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

# This module is currently a skeleton. Method bodies are populated by
# subsequent chunks of the implementation plan:
#   Chunk 3 — graded_space, spin_half_ops, schmidt_values
#   Chunk 4 — rand_iMPS / product_iMPS symmetric constructors,
#             _validate_iMPS_bonds specialisation
#   Chunks 5-7 — canonical!, applygate!, expect, energy_density, ent_S

end # module
