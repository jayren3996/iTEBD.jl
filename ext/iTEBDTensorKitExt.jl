module iTEBDTensorKitExt

using iTEBD
using TensorKit
using LinearAlgebra
using KrylovKit: eigsolve

# Names from the base package that this extension will specialise. Using
# `import` (not `using`) so that adding methods to these names is unambiguous
# to the compiler.
import iTEBD: graded_space, spin_half_ops, schmidt_values
import iTEBD: rand_iMPS, product_iMPS
import iTEBD: iMPS, _validate_iMPS_bonds
import iTEBD: canonical!

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

# `SymmetricIMPS` — the TensorKit-backed variant of `iMPS`. The narrow type
# parameters constrain it to symmetric tensors only; dense `Array`-backed
# states do NOT satisfy this alias. Available as
# `Base.get_extension(iTEBD, :iTEBDTensorKitExt).SymmetricIMPS` from user code,
# and unqualified as `SymmetricIMPS` inside the extension's own methods.
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

# Internal: extract the integer-valued label from a single Abelian sector.
_sector_int(s::U1Irrep)   = Int(s.charge)
_sector_int(s::ZNIrrep)   = Int(s.n)
_sector_int(::Trivial)    = 0

# Internal: distribute χ across sectors generated by `pspace ⊗ pspace`. The
# uniform-per-sector split is rough but adequate for initial random states;
# canonicalisation will reshape it once the symmetric `canonical!` lands.
function _auto_bond_space(sym::Symbol, pspace::VectorSpace, χ::Integer; flux::Integer=0)
    fused = fuse(pspace ⊗ pspace)
    sector_list = collect(sectors(fused))
    isempty(sector_list) && throw(ArgumentError(
        "_auto_bond_space: pspace ⊗ pspace produced no sectors"))
    per_sector = max(1, χ ÷ length(sector_list))
    pairs = Pair{Int,Int}[_sector_int(s) => per_sector for s in sector_list]
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
    flux == 0 || throw(ArgumentError(
        "rand_iMPS(symbol-based) currently only supports flux=0 in v1. " *
        "For non-zero target flux, use the raw `rand_iMPS(pspace, vspace, n)` " *
        "form and build the bond space yourself."))
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
    sum_occ = sum(occupations)
    sum_occ == 0 || throw(ArgumentError(
        "product_iMPS: total flux of occupations is $(sum_occ), must be 0 around " *
        "the unit cell. Pass occupations that sum to zero, or build the state " *
        "directly via the raw constructor and accept the non-zero global flux."))
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

# ─────────────────────────────────────────────────────────────────────────────
# Chunk 5: Symmetric truncated SVD primitive
# ─────────────────────────────────────────────────────────────────────────────

"""
    _symmetric_tsvd(A; maxdim, cutoff)

Truncated SVD of a four-leg symmetric block tensor `A` with the codomain shape
`V_left ⊗ P_1 ⊗ P_2` and domain shape `V_right`. The split happens at the
canonical "two-site" axis — the second physical leg and the right virtual
leg move to the domain side, while `V_left` and the first physical leg stay
in the codomain. Returns `(U, S, Vt, info)` where:

- `U` carries the left bond and the first physical leg (`codomain (V_left, P_1)`).
- `S::DiagonalTensorMap` holds the truncated singular values on the new bond.
- `Vt` carries the second physical leg and the right virtual leg.
- `info` is TensorKit/MatrixAlgebraKit's truncation diagnostic (a real number
  reporting the discarded weight) — used by callers for sanity checks.

This is the workhorse primitive used by `applygate!` to recanonicalise a
two-site block after applying a gate. It is also exercised directly by the
`canonical!` implementation when grouping the unit cell.
"""
function _symmetric_tsvd(A::AbstractTensorMap;
                         maxdim::Integer=iTEBD.MAXDIM,
                         cutoff::Real=iTEBD.SVDTOL)
    # Repartition: codomain (V_left, P_1), domain (P_2, V_right).
    # In TensorKit 0.16 this is `permute(A, ((1, 2), (3, 4)))`.
    B = permute(A, ((1, 2), (3, 4)))
    # Truncation strategy: cap rank at maxdim AND drop singular values below
    # `cutoff` (`trunctol` filters by absolute value, default `by = abs`).
    strategy = truncrank(Int(maxdim)) & trunctol(; atol=Float64(cutoff))
    return svd_trunc!(B; trunc=strategy)
end

# ─────────────────────────────────────────────────────────────────────────────
# Chunk 5: Symmetric canonical!
# ─────────────────────────────────────────────────────────────────────────────

# Apply the unit-cell transfer map to a bond-space density matrix.
#
# `dir = :r` applies T_n first (innermost), then T_{n-1}, …, T_1 last
# (outermost): (T_1 ∘ T_2 ∘ ... ∘ T_n)(ρ). The loop walks `i = n, n-1, …, 1`.
# `dir = :l` reverses the order: i = 1, 2, …, n.
#
# The single-site right transfer is T_i(ρ)[a, b] = Σ_{s, c, d} Γ_i[a, s, c]
# ρ[c, d] conj(Γ_i[b, s, d]).
function _apply_transfer_unit_cell(ψ::iMPS, ρ::AbstractTensorMap; dir::Symbol)
    n = ψ.n
    out = ρ
    indices = dir === :r ? (n:-1:1) : (1:n)
    for i in indices
        Γ = ψ.Γ[i]
        if dir === :r
            @tensor out_new[a; b] := Γ[a, s, c] * out[c; d] * conj(Γ[b, s, d])
        else
            @tensor out_new[a; b] := conj(Γ[c, s, a]) * out[c; d] * Γ[d, s, b]
        end
        out = out_new
    end
    return out
end

# Dominant fixed point of the unit-cell transfer map. Returns
# `(λ_max, ρ)` where `λ_max` is the magnitude of the dominant transfer
# eigenvalue (real, positive) and `ρ` is the corresponding Hermitian
# positive-semidefinite fixed point normalised to trace 1.
#
# Notes on sign / phase: the transfer map preserves the cone of PSD
# operators, so its dominant *PSD* eigenvalue is real and non-negative. For
# random iMPSes with discrete symmetries in the bond space, the Arnoldi
# spectrum often contains paired ±λ_max eigenvalues; the negative one
# corresponds to an indefinite eigenvector and is NOT the physical fixed
# point. We therefore request the top few Krylov vectors and pick the one
# whose blockwise spectrum is closest to PSD.
function _dominant_fixed_point(ψ::iMPS; dir::Symbol, tol::Real, maxiter::Integer)
    n = ψ.n
    Vbond = if dir === :r
        # Right fixed point lives on the right virtual leg of Γ[n].
        domain(ψ.Γ[n])[1]
    else
        # Left fixed point lives on the left virtual leg of Γ[1].
        codomain(ψ.Γ[1])[1]
    end
    ρ0 = id(ComplexF64, Vbond)
    matvec = ρ -> _apply_transfer_unit_cell(ψ, ρ; dir=dir)
    # Request a handful of eigenvalues so we can pick the genuine PSD one in
    # case of ±λ_max degeneracy.
    howmany = min(4, dim(Vbond))
    vals, vecs, info = eigsolve(matvec, ρ0, howmany, :LM;
                                tol=tol, maxiter=maxiter, ishermitian=false)
    info.converged ≥ 1 || @warn "Transfer map fixed-point eigsolve did not " *
        "converge (dir=$dir); proceeding with the leading Ritz vector" info

    # Identify the eigenvector closest to PSD: hermitianise each, compute the
    # negative spectral weight (sum of negative eigenvalues per block), and
    # take the one minimising that weight. The "right" eigenvector then has
    # mostly non-negative eigenvalues (numerical noise aside).
    best_idx = 1
    best_neg = Inf
    best_ρ   = nothing
    for k in eachindex(vecs)
        ρ_h = (vecs[k] + vecs[k]') / 2
        # Flip the global sign if the trace is negative — this is a free
        # gauge choice and gives the "positive face" of the candidate.
        if real(tr(ρ_h)) < 0
            ρ_h = -ρ_h
        end
        neg_weight = _negative_spectral_weight(ρ_h)
        if neg_weight < best_neg
            best_neg = neg_weight
            best_idx = k
            best_ρ   = ρ_h
        end
    end

    ρ_dom = best_ρ
    trace = real(tr(ρ_dom))
    if trace > sqrt(eps(Float64))
        ρ_dom = ρ_dom / trace
    else
        @warn "canonical!: dominant fixed point has near-zero trace ($trace); " *
              "the canonical form will be unreliable. This typically signals a " *
              "non-injective input."
    end
    λ_max = abs(vals[best_idx])
    return λ_max, ρ_dom
end

# Compute the sum of the magnitudes of negative eigenvalues across all blocks
# of a Hermitian TensorMap. Zero (within tolerance) means PSD; large values
# mean the operator is sign-indefinite.
function _negative_spectral_weight(M::AbstractTensorMap)
    w = 0.0
    for (_sector, blk) in blocks(M)
        H = (Matrix(blk) + Matrix(blk)') / 2
        evs = eigvals(Hermitian(H))
        for v in evs
            r = real(v)
            if r < 0
                w += -r
            end
        end
    end
    return w
end

# Block-wise positive square root of a Hermitian, positive-semidefinite
# TensorMap. The block algebra preserves U(1) sector structure trivially since
# every block is processed independently.
function _block_sqrt(M::AbstractTensorMap)
    out = similar(M)
    for (sector, blk) in blocks(M)
        H = (Matrix(blk) + Matrix(blk)') / 2
        F = eigen(Hermitian(H))
        # Clamp tiny negatives from numerical noise to avoid complex sqrt.
        evs = max.(real.(F.values), 0)
        sq = F.vectors * Diagonal(sqrt.(evs)) * F.vectors'
        copyto!(block(out, sector), sq)
    end
    return out
end

# Block-wise pseudo-inverse square root: M^{-1/2}, zeroing eigenvalues below a
# per-block tolerance derived from the largest eigenvalue magnitude.
function _block_isqrt(M::AbstractTensorMap)
    out = similar(M)
    for (sector, blk) in blocks(M)
        H = (Matrix(blk) + Matrix(blk)') / 2
        F = eigen(Hermitian(H))
        evs = real.(F.values)
        scale = isempty(evs) ? 0.0 : maximum(abs, evs)
        tol = sqrt(eps(Float64)) * max(scale, 1.0)
        invs = [v > tol ? inv(sqrt(v)) : 0.0 for v in evs]
        isq = F.vectors * Diagonal(invs) * F.vectors'
        copyto!(block(out, sector), isq)
    end
    return out
end

"""
    canonical!(ψ::iMPS{<:AbstractTensorMap, <:DiagonalTensorMap};
               maxdim=MAXDIM, cutoff=SVDTOL, renormalize=true,
               tol=1e-12, maxiter=200)

Bring a symmetric (TensorKit-backed) `iMPS` to Schmidt canonical form using
TensorKit primitives.

Algorithm (injective setting):

1. Solve for the dominant right and left fixed points `R`, `L` of the
   unit-cell transfer map (T_R from the stored tensors B = Γ λ; T_L is its
   adjoint chain). KrylovKit's `eigsolve` is used as a matrix-free solver.
2. Take Hermitian square roots: `X = sqrt(R)`, `Y = sqrt(L)`.
3. SVD: `X · Y = U · S · V'`. The canonical Schmidt spectrum on the
   wraparound bond is `Λ = S` (optionally renormalised so `sum(Λ²) = 1`).
4. Apply the wraparound gauge `G = X · U`:
   - For `n = 1`, the gauge sandwiches the single tensor:
     `B_new = U' · X⁻¹ · B · X · U` and `ψ.λ[1] = Λ`.
   - For `n > 1`, the gauge splits between the two boundary sites:
     `B_n_new = B_n · X · U`, `B_1_new = U' · X⁻¹ · B_1`. The internal bonds
     are then re-canonicalised by an SVD chain on the grouped unit cell so
     that each individual `B_i` satisfies the right-canonical condition
     `Σ_s B_i B_i' = I`.

Assumptions: the unit cell is *injective* — the dominant eigenvalue of the
transfer map is unique. For non-injective inputs (e.g. states with broken
translation symmetry within the unit cell, or transfer maps with
nearly-degenerate paired dominant eigenvalues), this routine may silently
produce an arbitrary or empty Schmidt spectrum on one or more bonds. Always
verify after canonicalisation by inspecting `schmidt_values(ψ, i)`. A future
release will add non-injective / multi-block canonical form. For now, when
the assumption is violated, a `@warn` is emitted from the asymmetric-
eigenvalue check, and downstream evolution should not be trusted.

Keyword arguments:
- `maxdim::Integer = MAXDIM` — Hard rank cap on each bond.
- `cutoff::Real = SVDTOL`    — Discard singular values below this threshold.
- `renormalize::Bool = true` — Rescale every bond so `sum(Λ²) = 1`.
- `tol::Real = 1e-12`        — Convergence tolerance for `eigsolve`.
- `maxiter::Integer = 200`   — Maximum Krylov restarts for `eigsolve`.
"""
function canonical!(ψ::iMPS{<:AbstractTensorMap, <:DiagonalTensorMap};
                    maxdim::Integer=iTEBD.MAXDIM,
                    cutoff::Real=iTEBD.SVDTOL,
                    renormalize::Bool=true,
                    tol::Real=1e-12,
                    maxiter::Integer=200)
    n = ψ.n
    # 1. Dominant right and left fixed points of the unit-cell transfer map.
    λ_r, R = _dominant_fixed_point(ψ; dir=:r, tol=tol, maxiter=maxiter)
    λ_l, L = _dominant_fixed_point(ψ; dir=:l, tol=tol, maxiter=maxiter)

    # The dominant transfer eigenvalue should match in both directions for an
    # injective state (up to numerical noise). Take the average for a more
    # robust per-site scaling factor.

    # Injective states have λ_r ≈ λ_l. A meaningful asymmetry signals non-injectivity
    # (e.g. block-diagonal transfer with paired dominant eigenvalues, or a state
    # with broken translation symmetry within the unit cell). Surface this rather
    # than silently averaging and proceeding.
    λ_scale = max(abs(λ_r), abs(λ_l), 1.0)
    if abs(λ_r - λ_l) > sqrt(eps(Float64)) * λ_scale * 100
        @warn "canonical!: asymmetric transfer eigenvalues (λ_r=$(λ_r), λ_l=$(λ_l)); " *
              "the input may be non-injective. The injective canonicalisation path " *
              "is unreliable here; consider seeding the random state differently or " *
              "constructing a state in a fixed flux sector."
    end
    λ_max = (real(λ_r) + real(λ_l)) / 2

    # Rescale each Γ by λ_max^(1/(2n)) so the new unit-cell transfer eigenvalue
    # becomes 1. After this, `Σ_s B_new · B_new' = I` will hold for the
    # gauge-transformed state.
    if λ_max > 0
        scale = λ_max^(1 / (2 * n))
        for i in 1:n
            ψ.Γ[i] = ψ.Γ[i] / scale
        end
    end

    # 2. Principal square roots (Hermitian, block-positive).
    X    = _block_sqrt(R)
    Xinv = _block_isqrt(R)
    Y    = _block_sqrt(L)

    # 3. SVD of M = X · Y. The new Schmidt values on the wraparound bond are
    # the singular values; the gauge `G = X · U` rotates into the canonical
    # basis (see derivation in the docstring).
    M = X * Y
    U, Λ, _Vt, _info = svd_trunc!(M;
        trunc=truncrank(Int(maxdim)) & trunctol(; atol=Float64(cutoff)))
    if renormalize
        nrm = norm(Λ)
        if !iszero(nrm)
            Λ = Λ / nrm
        end
    end

    # 4. Apply the wraparound gauge G = X·U and split for multi-site cells.
    _absorb_gauge!(ψ, X, Xinv, U, Λ;
                   maxdim=maxdim, cutoff=cutoff, renormalize=renormalize)
    return ψ
end

# Reabsorb the wraparound gauge into ψ.Γ and ψ.λ, then re-canonicalise any
# internal bonds via an SVD chain so every B_i is individually right-canonical.
function _absorb_gauge!(ψ::iMPS{<:AbstractTensorMap, <:DiagonalTensorMap},
                        X, Xinv, U, Λ;
                        maxdim::Integer, cutoff::Real, renormalize::Bool)
    n = ψ.n
    # G_right = X · U is the gauge applied on the right of Γ[n] (and on the
    # right of the single site for n = 1).
    G_right = X * U
    # G_left = U' · X⁻¹ is the gauge applied on the left of Γ[1] (or on the
    # left of the single site for n = 1).
    G_left = U' * Xinv

    if n == 1
        Γ_old = ψ.Γ[1]
        @tensor Γ_new[a, s; c] := G_left[a; a'] * Γ_old[a', s; c'] *
                                  G_right[c'; c]
        ψ.Γ[1] = Γ_new
        ψ.λ[1] = Λ
        return ψ
    end

    # n > 1: gauge the boundary sites, then split internal bonds.
    Γ_first = ψ.Γ[1]
    Γ_last  = ψ.Γ[n]
    # Absorb the wraparound Schmidt Λ on the LEFT of B_1 (in addition to the
    # gauge transformation). This puts the grouped tensor into "centered
    # canonical" form (Λ on both sides), which is the input shape the SVD
    # chain in `_split_unit_cell!` expects — see derivation in that function.
    @tensor Γ_first_new[a, s; c] := Λ[a; a''] * G_left[a''; a'] * Γ_first[a', s; c]
    @tensor Γ_last_new[a, s; c]  := Γ_last[a, s; c'] * G_right[c'; c]
    ψ.Γ[1] = Γ_first_new
    ψ.Γ[n] = Γ_last_new
    ψ.λ[n] = Λ

    # Recanonicalise internal bonds by a left-to-right SVD chain over the
    # grouped unit cell. With Λ already left-mul'd into ψ.Γ[1], the SVD chain
    # starts with `λi = Λ` and produces individually right-canonical B_i's
    # whose left-Schmidt factor matches the wraparound Λ.
    _split_unit_cell!(ψ, Λ; maxdim=maxdim, cutoff=cutoff, renormalize=renormalize)
    return ψ
end

# Split a multi-site unit cell back into individually right-canonical site
# tensors. After the wraparound gauge, the grouped tensor
# `T = B_1 · B_2 · ... · B_n` is right-canonical as a whole (Σ_phys T·T' = I).
#
# The algorithm mirrors the dense `tensor_decomp!` (left-to-right SVD sweep):
# at each step the leftmost site is split off; the new Schmidt is absorbed on
# its right (to maintain `B_i = Γ_i · λ_i` storage convention) and on the LEFT
# of the remaining tensor (so the next iteration sees a "left-Schmidt-absorbed"
# tensor analogous to the wraparound starting state). The final remaining
# tensor is the rightmost site's stored B (it is already right-canonical as an
# SVD isometry).
function _split_unit_cell!(ψ::iMPS{<:AbstractTensorMap, <:DiagonalTensorMap},
                           Λwrap::DiagonalTensorMap;
                           maxdim::Integer, cutoff::Real, renormalize::Bool)
    n = ψ.n
    n > 1 || return ψ

    # Build the full grouped tensor T = B_1 · B_2 · ... · B_n.
    T = ψ.Γ[1]
    for i in 2:n
        T = _contract_right(T, ψ.Γ[i])
    end
    # T has codomain (V_left, P_1, ..., P_n), domain V_right.

    truncstrat = truncrank(Int(maxdim)) & trunctol(; atol=Float64(cutoff))
    λi = Λwrap            # previous Schmidt entering the leftmost site
    λi_inv = _diag_inverse(λi)
    Ti = T
    for site in 1:(n - 1)
        # At iteration `site`, Ti has codomain (V_l_current, P_site, P_{site+1}, …, P_n)
        # of rank (n - site + 2) and domain (V_right) of rank 1.
        # Permute Ti to isolate site `site`: keep (V_l_current, P_site) in the
        # codomain and move (P_{site+1}, …, P_n, V_right) into the domain,
        # then SVD to extract the right-canonical site tensor A_site.
        num_cod = numout(Ti)
        cod_keep = (1, 2)
        dom_move = ntuple(k -> 2 + k, num_cod - 2)
        dom_inds = (dom_move..., num_cod + 1)
        Ti_perm = permute(Ti, (cod_keep, dom_inds))

        Ai, Λnew, Vt_part, _info = svd_trunc!(Ti_perm; trunc=truncstrat)
        if renormalize
            nrm = norm(Λnew)
            if !iszero(nrm)
                Λnew = Λnew / nrm
            end
        end

        # Stored site tensor: B_site = λi⁻¹ · Ai · Λnew.
        # Ai has codomain (V_l_current, P_site), domain (new_bond).
        # λi has codomain (V_l_current) ← (V_l_current).
        # Λnew has codomain (new_bond) ← (new_bond).
        Ai_div = _absorb_diag_left(Ai, λi_inv)
        Bi = _absorb_diag_right(Ai_div, Λnew)
        ψ.Γ[site] = Bi
        ψ.λ[site] = Λnew

        # Prepare Ti for the next iteration: Ti_new = Λnew · Vt_part (left-mul
        # by the new Schmidt). This way the next iteration's "λi" is Λnew.
        if site < n - 1
            Ti_new = _absorb_diag_left_on_codomain(Vt_part, Λnew)
            Ti = Ti_new
            λi = Λnew
            λi_inv = _diag_inverse(λi)
        else
            # Last iteration: do NOT left-mul by Λnew. The remaining Vt_part
            # becomes the stored B_n directly (it is right-canonical by virtue
            # of being an SVD isometry). However, we still need to reshape it
            # to match the storage convention (V_l, P_n) ← V_right.
            ψ.Γ[n] = _vt_to_site_tensor(Vt_part)
        end
    end
    return ψ
end

# Helper: contract T with the next site G on T's right virtual leg.
# T has codomain `(V_left, P_1, ..., P_{k-1})` and domain `V_mid`.
# G has codomain `(V_mid, P_k)` and domain `V_right`.
# Result has codomain `(V_left, P_1, ..., P_k)` and domain `V_right`.
function _contract_right(T::AbstractTensorMap, G::AbstractTensorMap)
    # Repartition G: codomain (V_mid,), domain (P_k, V_right).
    G_perm = permute(G, ((1,), (2, 3)))
    M = T * G_perm
    num_cod_M = numout(M)
    cod_inds = (ntuple(i -> i, num_cod_M)..., num_cod_M + 1)
    dom_inds = (num_cod_M + 2,)
    return permute(M, (cod_inds, dom_inds))
end

# Helper: convert a Vt-style tensor with codomain (new_bond,) and domain
# (P, V_right) into the iMPS storage convention (V_left, P) ← V_right.
function _vt_to_site_tensor(Vt::AbstractTensorMap)
    @assert numout(Vt) == 1 && numin(Vt) == 2
    return permute(Vt, ((1, 2), (3,)))
end

# Helper: absorb diagonal map λ on the RIGHT virtual leg of U.
# U has codomain shape (..., V_mid) actually — wait we need to be careful.
# In our convention, U from SVD has codomain (V_l, P), domain (V_mid). We want
# U_new = U · λ where λ has codomain (V_mid) ← (V_mid). This is straightforward
# composition: U * λ gives codomain (V_l, P), domain (V_mid_new).
function _absorb_diag_right(U::AbstractTensorMap, λ::AbstractTensorMap)
    return U * λ
end

# Helper: absorb diagonal map λi on the LEFT virtual leg of U.
# U has codomain (V_l, P), domain (V_mid). λi has codomain (V_l) ← (V_l).
# We need to compute λi⁻¹ · U where λi⁻¹ acts on U's first codomain leg.
# Using a contraction macro is cleanest here.
function _absorb_diag_left(U::AbstractTensorMap, λi_inv::AbstractTensorMap)
    @tensor out[a, s; c] := λi_inv[a; a'] * U[a', s; c]
    return out
end

# Helper: left-multiply a diagonal on the codomain side. For a Vt-style tensor
# `Vt` with codomain (new_bond,) and domain (P_2, ..., P_n, V_r), absorb λ
# (codomain (new_bond) ← (new_bond)) on the LEFT: λ * Vt.
function _absorb_diag_left_on_codomain(Vt::AbstractTensorMap, λ::AbstractTensorMap)
    return λ * Vt
end

# Helper: compute the pseudo-inverse of a DiagonalTensorMap (sector-wise).
function _diag_inverse(λ::DiagonalTensorMap)
    out = similar(λ)
    for (sector, blk) in blocks(λ)
        out_blk = block(out, sector)
        for i in 1:size(blk, 1)
            v = blk[i, i]
            inv_v = abs(v) > sqrt(eps(real(eltype(blk)))) ? inv(v) : zero(v)
            out_blk[i, i] = inv_v
        end
    end
    return out
end

# ─────────────────────────────────────────────────────────────────────────────
# Chunk 6: Symmetric applygate! for nearest-neighbour two-site gates
# ─────────────────────────────────────────────────────────────────────────────

# `_evolve_gate_sequence!` in Gate.jl calls `maximum(length, ψ.λ; init=mindim)`
# to compute the current bond dimension. For SymmetricIMPS each λ[i] is a
# DiagonalTensorMap, so we expose `length` as the total virtual dimension.
Base.length(λ::DiagonalTensorMap) = dim(domain(λ)[1])

import iTEBD: applygate!

"""
    applygate!(ψ::SymmetricIMPS, G::AbstractTensorMap, i::Integer, j::Integer;
               maxdim=MAXDIM, cutoff=SVDTOL, renormalize=true, kwargs...)

Apply a two-site gate `G` to neighbouring sites `i` and `j = mod1(i+1, ψ.n)`
of a symmetric infinite MPS in place. `G` must be a TensorMap with codomain
`P ⊗ P` and domain `P ⊗ P`, where `P` matches the physical leg of `ψ`.

Algorithm:
1. Group: `B[a,s,t;c] = Γi[a,s;m] * Γj[m,t;c]`.
2. Apply gate: `B'[a,s,t;c] = G[s,t;u,v] * B[a,u,v;c]`.
3. SVD `B'` on the `(a,s) | (t,c)` cut via `_symmetric_tsvd`.
4. Store `U` as `ψ.Γ[i]`, `S * Vt` as `ψ.Γ[j]`, `S` as `ψ.λ[i]`.

Only nearest-neighbour gates (`j = mod1(i+1, ψ.n)`) are supported.
Extra keyword arguments (`mindim`, `truncerr`, `svd_min`, `return_stats`, etc.)
from the base `evolve!` routing are accepted and silently ignored so that
dispatch through `_evolve_gate_sequence!` works without modification.
"""
function applygate!(ψ::iMPS{<:AbstractTensorMap, <:DiagonalTensorMap},
                    G::AbstractTensorMap, i::Integer, j::Integer;
                    maxdim::Integer=iTEBD.MAXDIM,
                    cutoff::Real=iTEBD.SVDTOL,
                    renormalize::Bool=true,
                    kwargs...)
    j == mod1(i + 1, ψ.n) || throw(ArgumentError(
        "v1 symmetric applygate! supports nearest-neighbour two-site gates only " *
        "(got i=$i, j=$j on n=$(ψ.n))"))

    Γi = ψ.Γ[i]
    Γj = ψ.Γ[j]

    # Step 1: group the two-site block.
    # Γi has codomain (V_l, P), domain (V_mid).
    # Γj has codomain (V_mid, P), domain (V_r).
    # B has codomain (V_l, P1, P2), domain (V_r).
    @tensor B[a, s, t; c] := Γi[a, s; m] * Γj[m, t; c]

    # Step 2: apply the gate.
    # G has codomain (P1, P2), domain (P1, P2) — the convention used by
    # spin_half_ops(:U1) and id(P ⊗ P).
    @tensor B′[a, s, t; c] := G[s, t; u, v] * B[a, u, v; c]

    # Step 3: SVD with (V_l, P1) | (P2, V_r) cut.
    U, S, Vt, _ = _symmetric_tsvd(B′; maxdim=maxdim, cutoff=cutoff)

    if renormalize
        nrm = norm(S)
        nrm > 0 && (S = S / nrm)
    end

    # Step 4: store results. Package convention: absorb Schmidt values S into
    # the right tensor Γj so that Γi = U is a left isometry.
    #
    # After _symmetric_tsvd:
    #   U   has codomain (V_l, P1), domain (V_mid) — shape (2,1), ready for ψ.Γ[i].
    #   Vt  has codomain (V_mid,), domain (P2, V_r) — shape (1,2).
    #
    # Storage convention for Γ is (V_l, P) ← V_r — shape (2,1).
    # Absorb S on the left of Vt to get S·Vt with shape (1,2), then
    # permute to (2,1) via _vt_to_site_tensor to match the convention.
    ψ.Γ[i] = U
    ψ.λ[i] = S
    ψ.Γ[j] = _vt_to_site_tensor(S * Vt)

    return ψ
end

end # module
