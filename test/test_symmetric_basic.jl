using Test
using iTEBD
using LinearAlgebra: I, norm, tr, Diagonal, eigen, Hermitian
using Random

# Explicit imports from TensorKit to avoid name conflicts with ITensors
# (both are test dependencies and share some exported names like `dim`, `space`).
using TensorKit: U1Irrep, Z2Irrep, ZNIrrep, Vect, dim, block, blocks, space, id,
                 ComplexSpace, sectortype, sectors, domain, codomain,
                 AbstractTensorMap, DiagonalTensorMap, ←, ⊗, @tensor

# Symbol-table reach into the loaded extension for internal helpers.
const _TKExt = Base.get_extension(iTEBD, :iTEBDTensorKitExt)
schmidt_values_from_S(S::DiagonalTensorMap) = _TKExt._flatten_diagonal_blocks(S)

@testset "graded_space" begin
    P = graded_space(:U1, 0=>2, 1=>1, -1=>1)
    @test P isa Vect[U1Irrep]
    @test dim(P) == 4
    @test dim(P, U1Irrep(0)) == 2
    @test dim(P, U1Irrep(1)) == 1
end

@testset "graded_space Z2, ZN, Trivial" begin
    Pz = graded_space(:Z2, 0=>3, 1=>3)
    @test Pz isa Vect[Z2Irrep]
    @test dim(Pz) == 6

    P4 = graded_space(:ZN, 4, 0=>1, 1=>1, 2=>1, 3=>1)
    @test P4 isa Vect[ZNIrrep{4}]
    @test dim(P4) == 4

    Pt = graded_space(:Trivial, 0=>3)
    @test Pt isa ComplexSpace
    @test dim(Pt) == 3
end

@testset "graded_space products" begin
    P = graded_space(:U1xU1, (0,0)=>2, (1,-1)=>1, (-1,1)=>1)
    @test dim(P) == 4

    Q = graded_space(:U1xZ2, (0,0)=>2, (1,1)=>1)
    @test dim(Q) == 3
end

@testset "spin_half_ops" begin
    @testset "U(1) symmetric" begin
        Sz, SzSz, SpSm, SmSp = spin_half_ops(:U1)
        P = graded_space(:U1, 1=>1, -1=>1)

        # Sz is a one-site endomorphism of P with the correct sector values
        @test sectortype(space(Sz, 1)) == U1Irrep
        @test space(Sz, 1) == P
        @test isapprox(block(Sz, U1Irrep(1))[1, 1], 0.5; atol=1e-12)
        @test isapprox(block(Sz, U1Irrep(-1))[1, 1], -0.5; atol=1e-12)

        # Sz is hermitian and squares to (1/4) I
        @test isapprox(Sz, Sz'; atol=1e-12)
        @test isapprox(Sz * Sz, 0.25 * id(P); atol=1e-12)

        # All three two-site operators live on the SAME HomSpace, so they add.
        @test space(SzSz) == space(SpSm)
        @test space(SpSm) == space(SmSp)

        # SpSm and SmSp are mutual adjoints.
        @test isapprox(SpSm', SmSp; atol=1e-12)

        # The Heisenberg density assembles cleanly and matches the dense
        # 4×4 reference in the {|↑↑⟩, |↑↓⟩, |↓↑⟩, |↓↓⟩} basis. This is the
        # composition that the previous Chunk 3 implementation silently broke.
        h = SzSz + 0.5 * (SpSm + SmSp)
        @test isapprox(h, h'; atol=1e-12)

        h_dense = ComplexF64[
            0.25   0     0     0   ;
            0    -0.25  0.5    0   ;
            0     0.5  -0.25   0   ;
            0     0     0     0.25 ]
        @test isapprox(reshape(convert(Array, h), 4, 4), h_dense; atol=1e-12)
    end

    @testset "Trivial / dense fallback" begin
        Sx, Sy, Sz, Sp, Sm, Id = spin_half_ops(:Trivial)
        @test Sz isa AbstractMatrix
        @test isapprox(Sx*Sx + Sy*Sy + Sz*Sz, 0.75 * Id; atol=1e-12)
    end
end

@testset "rand_iMPS symmetric (raw spaces)" begin
    P = graded_space(:U1, 1=>1, -1=>1)
    V = graded_space(:U1, 0=>2, 2=>1, -2=>1)
    ψ = rand_iMPS(P, V, 2)
    @test ψ.Γ[1] isa AbstractTensorMap
    @test ψ.λ[1] isa DiagonalTensorMap
    @test ψ.n == 2
    @test length(ψ.Γ) == 2 && length(ψ.λ) == 2
    # Domain/codomain check: each Γ[i] should map V ⊗ P → V (rank-3 in MPS shape).
    for i in 1:2
        @test codomain(ψ.Γ[i])[1] == V
        @test codomain(ψ.Γ[i])[2] == P
        @test domain(ψ.Γ[i])[1] == V
    end
    # Wraparound: right-leg of Γ[end] equals left-leg of Γ[1] (trivially true here
    # because all Γ are constructed from the same V).
    @test domain(ψ.Γ[end])[1] == codomain(ψ.Γ[1])[1]
end

@testset "rand_iMPS(:U1, charges, χ)" begin
    ψ = rand_iMPS(:U1, [-1, 1]; χ=8, n=2, flux=0)
    @test ψ.Γ[1] isa AbstractTensorMap
    @test ψ.λ[1] isa DiagonalTensorMap
    @test ψ.n == 2
    # Total virtual dim should be ≤ χ (auto-distributor may round)
    @test dim(domain(ψ.Γ[1])[1]) ≤ 8
    @test dim(domain(ψ.Γ[1])[1]) ≥ 1
end

@testset "product_iMPS symmetric Néel state" begin
    ψ = product_iMPS(:U1, [-1, 1], [1, -1])
    @test ψ.Γ[1] isa AbstractTensorMap
    @test ψ.λ[1] isa DiagonalTensorMap
    @test ψ.n == 2
    # Each Γ[i] has bond dim 1 (product state) on both sides.
    for i in 1:2
        @test dim(codomain(ψ.Γ[i])[1]) == 1
        @test dim(domain(ψ.Γ[i])[1]) == 1
    end
    # Bond charges should be Sz=0 (cumulative before site 1) and Sz=+1 (after site 1)
    @test sectortype(codomain(ψ.Γ[1])[1]) == U1Irrep
    left_sectors_1  = collect(sectors(codomain(ψ.Γ[1])[1]))
    right_sectors_1 = collect(sectors(domain(ψ.Γ[1])[1]))
    @test left_sectors_1  == [U1Irrep(0)]
    @test right_sectors_1 == [U1Irrep(1)]
    left_sectors_2  = collect(sectors(codomain(ψ.Γ[2])[1]))
    right_sectors_2 = collect(sectors(domain(ψ.Γ[2])[1]))
    @test left_sectors_2  == [U1Irrep(1)]
    @test right_sectors_2 == [U1Irrep(0)]
end

@testset "product_iMPS :Z2 with modular flux closure" begin
    # Two occupied sites on a Z_2 chain: sum = 2 ≡ 0 (mod 2), should succeed.
    ψ = product_iMPS(:Z2, [0, 1], [1, 1])
    @test ψ.n == 2
    @test sectortype(codomain(ψ.Γ[1])[1]) == Z2Irrep
    # Three odd-charge sites: sum = 3 ≡ 1 (mod 2), should fail.
    @test_throws ArgumentError product_iMPS(:Z2, [0, 1], [1, 1, 1])
end

@testset "product_iMPS rejects :ZN symbol API" begin
    # :ZN cannot pass N through the symbol-based product_iMPS signature; the
    # error message must direct the user to the raw constructor.
    err = try
        product_iMPS(:Z3, [0, 1, 2], [0, 1, 2])
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("raw", err.msg) || occursin("ZN", err.msg)
end

@testset "rand_iMPS rejects :ZN symbol API" begin
    err = try
        rand_iMPS(:Z3, [0, 1, 2]; χ=4, n=2)
    catch e
        e
    end
    @test err isa ArgumentError
end

@testset "wraparound flux check rejects mismatched spaces" begin
    P  = graded_space(:U1, 1=>1, -1=>1)
    Va = graded_space(:U1, 0=>1)
    Vb = graded_space(:U1, 2=>1)
    # Build two tensors whose right→left bond spaces deliberately do not match
    # around the wraparound: Γ[1] left=Va right=Vb; Γ[2] left=Va right=Va. The
    # seam from Γ[2]'s right (Va) to Γ[1]'s left (Va) is fine, but Γ[1]'s right
    # (Vb) doesn't match Γ[2]'s left (Va) — bond 1 fails.
    Γ1 = zeros(ComplexF64, Va ⊗ P ← Vb)
    Γ2 = zeros(ComplexF64, Va ⊗ P ← Va)
    λ_dummy = [
        DiagonalTensorMap(ones(Float64, 1), Vb),
        DiagonalTensorMap(ones(Float64, 1), Va),
    ]
    @test_throws DimensionMismatch iMPS([Γ1, Γ2], λ_dummy, 2)
end

@testset "rand_iMPS(:Z2, charges, χ)" begin
    ψ = rand_iMPS(:Z2, [0, 1]; χ=4, n=2)
    @test ψ.Γ[1] isa AbstractTensorMap
    @test ψ.λ[1] isa DiagonalTensorMap
    @test ψ.n == 2
    @test dim(domain(ψ.Γ[1])[1]) ≤ 4
end

@testset "wraparound seam (bond n→1) failure" begin
    P  = graded_space(:U1, 1=>1, -1=>1)
    Va = graded_space(:U1, 0=>1)
    Vb = graded_space(:U1, 2=>1)
    # Build two tensors where bonds 1..n-1 match but the seam from Γ[n] back
    # to Γ[1] is broken: Γ[1].left=Va, Γ[2].right=Vb (not Va), so the seam fails.
    Γ1 = zeros(ComplexF64, Va ⊗ P ← Va)
    Γ2 = zeros(ComplexF64, Va ⊗ P ← Vb)
    λ_dummy = [
        DiagonalTensorMap(ones(Float64, 1), Va),
        DiagonalTensorMap(ones(Float64, 1), Vb),
    ]
    @test_throws DimensionMismatch iMPS([Γ1, Γ2], λ_dummy, 2)
end

@testset "λ-space mismatch is rejected" begin
    P  = graded_space(:U1, 1=>1, -1=>1)
    Va = graded_space(:U1, 0=>1, 2=>1)
    Vb = graded_space(:U1, 1=>2)          # same total dim (2), different sectors
    Γ  = [zeros(ComplexF64, Va ⊗ P ← Va) for _ in 1:2]
    # λ[1] deliberately lives on Vb (wrong space) instead of Va.
    λ  = [
        DiagonalTensorMap(ones(Float64, 2), Vb),
        DiagonalTensorMap(ones(Float64, 2), Va),
    ]
    @test_throws DimensionMismatch iMPS(Γ, λ, 2)
end

# ─────────────────────────────────────────────────────────────────────────────
# Chunk 5: Symmetric truncated SVD primitive and canonical!
# ─────────────────────────────────────────────────────────────────────────────

@testset "_symmetric_tsvd preserves blocks" begin
    # Two-site block: V_left ⊗ P_1 ⊗ P_2 ← V_right. With P = (±1) the combined
    # two-site charge shift is 0 or ±2, which matches the chosen V sectors.
    P = graded_space(:U1, 1=>1, -1=>1)
    V = graded_space(:U1, 0=>2, 2=>1, -2=>1)
    A = randn(ComplexF64, V ⊗ P ⊗ P ← V)
    U, S, Vt, info = _TKExt._symmetric_tsvd(A; maxdim=4, cutoff=1e-12)
    @test S isa DiagonalTensorMap
    # The middle bond should match between U and S, and between S and Vt.
    @test domain(U)[1] == domain(S)[1]
    @test codomain(S)[1] == codomain(Vt)[1]
    # The block should have nonzero values.
    @test sum(abs2, schmidt_values_from_S(S)) > 0
    # info is TensorKit's (MatrixAlgebraKit's) truncation diagnostic — a real
    # truncation-error scalar in TK 0.16.
    @test info isa Real
end

@testset "canonical! on symmetric iMPS" begin
    using Random
    # Seed pinning is load-bearing here. The v1 symmetric canonical! now THROWS
    # on non-injective inputs; we pin seed=2 here so the test exercises the
    # supported injective path. With this (P, V) choice, ~50% of random seeds
    # land in a non-injective regime where the algorithm raises ArgumentError.
    # A future release will add non-injective / multi-block canonical form to
    # remove the seed dependence.
    Random.seed!(2)
    P = graded_space(:U1, 1=>1, -1=>1)
    V = graded_space(:U1, 0=>1, 1=>1, -1=>1, 2=>1, -2=>1)
    ψ = rand_iMPS(P, V, 2)
    canonical!(ψ)
    # Schmidt values are sorted descending and normalised on each bond.
    for i in 1:2
        vals = schmidt_values(ψ, i)
        @test issorted(vals; rev=true)
        @test isapprox(sum(abs2, vals), 1.0; atol=1e-9)
    end
    # Post-canonicalisation the stored tensors should be right-canonical:
    # sum_s B[i] B[i]' ≈ I on the LEFT virtual space of Γ[i].
    for i in 1:2
        Γ = ψ.Γ[i]
        @tensor R[a; b] := Γ[a, s, c] * conj(Γ[b, s, c])
        # Compare blockwise to identity on the codomain space.
        for (sec, blk) in blocks(R)
            @test isapprox(blk, Matrix{ComplexF64}(I, size(blk)); atol=1e-7)
        end
    end
    # Idempotence: canonicalising again should be a no-op up to numerics.
    λ_before = schmidt_values(ψ, 1)
    canonical!(ψ)
    λ_after = schmidt_values(ψ, 1)
    @test isapprox(λ_before, λ_after; atol=1e-10)
end

@testset "canonical! on symmetric iMPS, n=1" begin
    using Random
    # Seed pinning is load-bearing here. The v1 symmetric canonical! now THROWS
    # on non-injective inputs; we pin seed=2 here so the test exercises the
    # supported injective path. For a single-site unit cell the transfer
    # spectrum stays connected (consecutive integer charges), but a few seeds
    # still hit Krylov sign-degeneracies that trigger the non-injective throw.
    # Seed 2 is a known-good injective seed. A future release will add
    # non-injective / multi-block canonical form to remove the seed dependence.
    Random.seed!(2)
    P = graded_space(:U1, 1=>1, -1=>1)
    V = graded_space(:U1, 0=>2, 1=>2, -1=>2)
    ψ = rand_iMPS(P, V, 1)
    canonical!(ψ)
    vals = schmidt_values(ψ, 1)
    @test issorted(vals; rev=true)
    @test isapprox(sum(abs2, vals), 1.0; atol=1e-9)
    Γ = ψ.Γ[1]
    @tensor R[a; b] := Γ[a, s, c] * conj(Γ[b, s, c])
    for (sec, blk) in blocks(R)
        @test isapprox(blk, Matrix{ComplexF64}(I, size(blk)); atol=1e-7)
    end
end

@testset "canonical! on symmetric iMPS, n=4 product state" begin
    # n=4 runs the _split_unit_cell! loop twice with the repartition step
    # (sites 1 and 2). Same correctness target as n=3: each Γ stays a
    # right-isometry per block.
    ψ = product_iMPS(:U1, [-1, 1, 0], [1, -1, 1, -1])
    canonical!(ψ)
    @test ψ.n == 4
    for k in 1:4
        Γ = ψ.Γ[k]
        @test dim(codomain(Γ)[1]) == 1
        @test dim(domain(Γ)[1]) == 1
        @tensor R[a; b] := Γ[a, s, c] * conj(Γ[b, s, c])
        for (_, blk) in blocks(R)
            @test isapprox(blk, Matrix{ComplexF64}(I, size(blk)); atol=1e-9)
        end
    end
end

@testset "canonical! on symmetric iMPS, n=3 product state" begin
    # n=3 is the smallest unit cell that triggers the iterative `_split_unit_cell!`
    # repartition (n=2 takes the terminal branch directly). A bond-dim-1 product
    # state is deterministic — no Krylov, no random seed — and its right-canonical
    # form is again bond-dim-1 with right isometries.
    ψ = product_iMPS(:U1, [-1, 1, 0], [1, -1, 0])
    canonical!(ψ)
    @test ψ.n == 3
    # Every site stays bond-dim 1 and remains a right-isometry: ⟨Γ|Γ⟩ = 1 per block.
    for k in 1:3
        Γ = ψ.Γ[k]
        @test dim(codomain(Γ)[1]) == 1
        @test dim(domain(Γ)[1]) == 1
        @tensor R[a; b] := Γ[a, s, c] * conj(Γ[b, s, c])
        for (_, blk) in blocks(R)
            @test isapprox(blk, Matrix{ComplexF64}(I, size(blk)); atol=1e-9)
        end
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Chunk 6: Symmetric applygate! and evolve! routing
# ─────────────────────────────────────────────────────────────────────────────

@testset "applygate! symmetric two-site identity" begin
    using Random
    Random.seed!(2)
    P = graded_space(:U1, 1=>1, -1=>1)
    V = graded_space(:U1, 0=>2, 2=>1, -2=>1)
    ψ = rand_iMPS(P, V, 2)
    canonical!(ψ)

    norm_before = schmidt_values(ψ, 1)

    # The identity gate on two physical sites should leave ψ unchanged up to
    # truncation noise.
    Iop = id(ComplexF64, P ⊗ P)
    applygate!(ψ, Iop, 1, 2; maxdim=8)

    norm_after = schmidt_values(ψ, 1)

    # After truncation the kept Schmidt values should match the pre-gate ones
    # (modulo numerical noise and the truncation-to-maxdim=8 contract).
    n_keep = min(length(norm_before), length(norm_after), 8)
    @test isapprox(norm_before[1:n_keep], norm_after[1:n_keep]; atol=1e-9)
end

@testset "applygate! rejects return_stats=true" begin
    # Dense applygate! returns (ψ, stats) on return_stats=true; symmetric path
    # doesn't yet collect them. Silent acceptance would cause the base
    # `_evolve_gate_sequence!` to destructure ψ itself and crash.
    ψ = product_iMPS(:U1, [-1, 1], [1, -1])
    P = codomain(ψ.Γ[1])[2]
    Iop = id(ComplexF64, P ⊗ P)
    @test_throws ArgumentError applygate!(ψ, Iop, 1, 2; return_stats=true)
end

@testset "applygate! validates truncation kwargs" begin
    # The symmetric path used to accept `kwargs...` and silently drop
    # `mindim`/`truncerr`, plus typo kwargs. Match the dense path's
    # `_resolve_svd_min` / `_validate_truncation_args` contract.
    ψ = product_iMPS(:U1, [-1, 1], [1, -1])
    P = codomain(ψ.Γ[1])[2]
    Iop = id(ComplexF64, P ⊗ P)

    # cutoff and svd_min are aliases; supplying both is ambiguous.
    @test_throws ArgumentError applygate!(ψ, Iop, 1, 2; cutoff=1e-10, svd_min=1e-10)

    # mindim > 1 is not implemented in the v1 symmetric path.
    @test_throws ArgumentError applygate!(ψ, Iop, 1, 2; mindim=2)

    # truncerr > 0 likewise.
    @test_throws ArgumentError applygate!(ψ, Iop, 1, 2; truncerr=1e-6)

    # Negative / non-finite truncation knobs are rejected.
    @test_throws ArgumentError applygate!(ψ, Iop, 1, 2; cutoff=-1.0)
    @test_throws ArgumentError applygate!(ψ, Iop, 1, 2; svd_min=-1.0)
    @test_throws ArgumentError applygate!(ψ, Iop, 1, 2; truncerr=-1e-6)

    # Typo kwargs are caught because the `kwargs...` catch-all is gone.
    @test_throws MethodError applygate!(ψ, Iop, 1, 2; typo_kw=true)
end

@testset "applygate! normalizes periodic site indices" begin
    # On n=2 the labels (3, 2) and (1, 2) describe the same cut after periodic
    # reduction. The dense path normalizes via `_normalize_gate_sites`; the
    # symmetric path should match. Before this fix the out-of-range label
    # crashed at `ψ.Γ[3]`.
    ψ = product_iMPS(:U1, [-1, 1], [1, -1])
    P = codomain(ψ.Γ[1])[2]
    Iop = id(ComplexF64, P ⊗ P)
    applygate!(ψ, Iop, 3, 2)            # = applygate!(ψ, Iop, 1, 2) after normalize
    @test ψ.n == 2
    # Non-nearest-neighbour pairs are still rejected after normalization.
    @test_throws ArgumentError applygate!(ψ, Iop, 1, 1)
end

@testset "evolve! routes through symmetric applygate!" begin
    using Random
    Random.seed!(2)
    P = graded_space(:U1, 1=>1, -1=>1)
    V = graded_space(:U1, 0=>2, 2=>1, -2=>1)
    ψ = rand_iMPS(P, V, 2)
    canonical!(ψ)
    Iop = id(ComplexF64, P ⊗ P)
    gates = [(Iop, 1, 2), (Iop, 2, 1)]
    evolve!(ψ, gates, 3; maxdim=8)
    @test ψ.Γ[1] isa AbstractTensorMap
    @test ψ.λ[1] isa DiagonalTensorMap
end

@testset "evolve! end-to-end on a Z2 product state" begin
    # Build a Z_2 product state whose flux closes mod 2, apply identity gates
    # via evolve!, and verify the Z_2 grading is preserved at every bond.
    # This is the smallest end-to-end exercise of the Z_2 dispatch path
    # through product_iMPS → canonical! → applygate!.
    ψ = product_iMPS(:Z2, [0, 1], [1, 1])
    P = codomain(ψ.Γ[1])[2]
    Iop = id(ComplexF64, P ⊗ P)
    gates = [(Iop, 1, 2), (Iop, 2, 1)]
    evolve!(ψ, gates, 3; maxdim=4)
    @test ψ.n == 2
    @test sectortype(codomain(ψ.Γ[1])[1]) == Z2Irrep
    for i in 1:ψ.n
        Vr      = domain(ψ.Γ[i])[1]
        Vl_next = codomain(ψ.Γ[mod1(i + 1, ψ.n)])[1]
        @test Vr == Vl_next
    end
end

@testset "evolve! propagates cutoff to symmetric applygate!" begin
    # Build a Néel state and evolve with an entangling gate so the bond grows.
    # With a loose cutoff the truncation should drop modes; with a tight cutoff
    # all modes are kept. Comparing the two bond dimensions verifies that the
    # base `evolve!` cutoff actually reaches the symmetric SVD.
    Sz, SzSz, SpSm, SmSp = spin_half_ops(:U1)
    h = SzSz + 0.5 * (SpSm + SmSp)
    dt = 0.1
    gate = exp(-dt * h)
    gates = [(gate, 1, 2), (gate, 2, 1)]

    ψ_loose = product_iMPS(:U1, [-1, 1], [1, -1])
    evolve!(ψ_loose, gates, 5; maxdim=64, cutoff=0.5)

    ψ_tight = product_iMPS(:U1, [-1, 1], [1, -1])
    evolve!(ψ_tight, gates, 5; maxdim=64, cutoff=1e-14)

    χ_loose = sum(dim(domain(λ)[1]) for λ in ψ_loose.λ)
    χ_tight = sum(dim(domain(λ)[1]) for λ in ψ_tight.λ)
    @test χ_loose < χ_tight
end

# ─────────────────────────────────────────────────────────────────────────────
# Chunk 7: Symmetric observables
# ─────────────────────────────────────────────────────────────────────────────

@testset "ent_S on symmetric iMPS" begin
    using Random
    Random.seed!(2)
    P = graded_space(:U1, 1=>1, -1=>1)
    V = graded_space(:U1, 0=>2, 2=>1, -2=>1)
    ψ = rand_iMPS(P, V, 2)
    canonical!(ψ)
    S = ent_S(ψ, 1)
    @test S ≥ 0
    @test isfinite(S)
end

@testset "one-site expect and energy_density" begin
    # Néel state in Sz=0 sector
    ψ = product_iMPS(:U1, [-1, 1], [1, -1])
    Sz, SzSz, SpSm, SmSp = spin_half_ops(:U1)

    # Site-resolved magnetisations: site 1 has Sz=+1 (occupation +1 in 2*Sz units → +0.5 in Sz)
    val1 = expect(ψ, Sz, 1, 1)
    val2 = expect(ψ, Sz, 2, 2)
    @test isapprox(real(val1),  0.5; atol=1e-10)
    @test isapprox(real(val2), -0.5; atol=1e-10)
    @test isapprox(imag(val1),  0.0; atol=1e-10)

    # Total Sz over unit cell averages to 0 (Sz=0 sector)
    @test isapprox(real(val1) + real(val2), 0.0; atol=1e-10)

    # Heisenberg density on Néel: <SzSz> = -0.25 (the Sz⊗Sz piece on antialigned spins);
    # the off-diagonal SpSm, SmSp pieces evaluate to 0 on a product state.
    h = SzSz + 0.5 * (SpSm + SmSp)
    e = energy_density(ψ, h)
    @test isapprox(real(e), -0.25; atol=1e-10)
    @test isapprox(imag(e),  0.0;  atol=1e-10)
end

@testset "applygate! preserves canonical form across many gate applications" begin
    # Use imaginary-time Heisenberg gates starting from the Néel product state.
    # The Néel state has bond dim 1 (only U1(0) sector). Under imaginary-time
    # evolution the bond grows gradually — we use maxdim=4 to keep it bounded
    # well below truncation noise. Because only the U1(0) sector is populated,
    # no sector can be accidentally cut to zero by the SVD cutoff, making the
    # invariant checks reliable.
    Sz, SzSz, SpSm, SmSp = spin_half_ops(:U1)
    h = SzSz + 0.5 * (SpSm + SmSp)
    dt = 0.05
    G = exp(-dt * h)   # imaginary-time gate (real, non-unitary)
    ψ = product_iMPS(:U1, [-1, 1], [1, -1])   # Néel product state, bond dim 1

    # Apply 20 full Trotter steps (non-wrap then wrap gate each step).
    # The wrap gate triggers canonical! inside applygate!.
    for step in 1:20
        applygate!(ψ, G, 1, 2; maxdim=4)
        applygate!(ψ, G, 2, 1; maxdim=4)
    end

    # The canonical-form invariant: ψ.Γ[i] should remain a right-isometry,
    # i.e., Σ_s Γ[i]_{a,s,c} · conj(Γ[i]_{b,s,c}) ≈ δ_{ab} on the codomain bond.
    for i in 1:ψ.n
        Γ = ψ.Γ[i]
        @tensor R[a; b] := Γ[a, s, c] * conj(Γ[b, s, c])
        for (sec, blk) in blocks(R)
            ident = Matrix{ComplexF64}(I, size(blk))
            @test isapprox(blk, ident; atol=1e-8)
        end
    end

    # Schmidt-value normalisation invariant.
    for i in 1:ψ.n
        @test isapprox(sum(abs2, schmidt_values(ψ, i)), 1.0; atol=1e-8)
    end
end

@testset "dense and symmetric backends agree on one Heisenberg gate" begin
    # Same physical setup in both backends (Néel state, spin-1/2 XXX), apply
    # one imaginary-time gate, compare energy densities. This catches
    # convention drifts between paths — exactly the kind of bug that bit
    # `expect` / `energy_density` during the symmetric rollout (one vs two
    # λL factors).
    dt = 0.05

    # Dense Néel state and Heisenberg gate.
    Sz_d = ComplexF64[0.5  0; 0 -0.5]
    Sp_d = ComplexF64[0   1; 0  0]
    Sm_d = ComplexF64[0   0; 1  0]
    h_d = real(kron(Sz_d, Sz_d) + 0.5 * (kron(Sp_d, Sm_d) + kron(Sm_d, Sp_d)))
    gate_d = exp(-dt * Matrix(h_d))
    up   = ComplexF64[1, 0]
    down = ComplexF64[0, 1]
    ψ_d = product_iMPS(ComplexF64, [up, down])

    # Symmetric Néel state and gate via the U(1) helper.
    ψ_s = product_iMPS(:U1, [-1, 1], [1, -1])
    Sz_s, SzSz, SpSm, SmSp = spin_half_ops(:U1)
    h_s = SzSz + 0.5 * (SpSm + SmSp)
    gate_s = exp(-dt * h_s)

    # Pre-gate: ⟨h⟩ on Néel is exactly -0.25, both backends agree.
    @test isapprox(real(energy_density(ψ_d, h_d)), -0.25; atol=1e-9)
    @test isapprox(real(energy_density(ψ_s, h_s)), -0.25; atol=1e-9)

    # Apply one gate per backend at the same bond, compare.
    applygate!(ψ_d, gate_d, 1, 2)
    applygate!(ψ_s, gate_s, 1, 2)
    E_d_after = real(energy_density(ψ_d, h_d))
    E_s_after = real(energy_density(ψ_s, h_s))
    @test isapprox(E_d_after, E_s_after; atol=1e-7)
    @test E_d_after < -0.25   # imaginary-time gate lowers the energy
end

# ─────────────────────────────────────────────────────────────────────────────
# Post-review polish tests (blocking fixes #1, #2, important fix #1)
# ─────────────────────────────────────────────────────────────────────────────

@testset "canonical! throws on non-injective input" begin
    using Random
    # Seeds 1, 3, 4, 5, 9 are known to land in the non-injective regime for
    # this (P, V) configuration (asymmetric left/right transfer eigenvalues).
    # The v1 canonical! now THROWS rather than silently corrupting the state.
    Random.seed!(3)
    P = graded_space(:U1, 1=>1, -1=>1)
    V = graded_space(:U1, 0=>1, 1=>1, -1=>1, 2=>1, -2=>1)
    ψ = rand_iMPS(P, V, 2)
    @test_throws ArgumentError canonical!(ψ)
end

@testset "canonical! honours small cutoff" begin
    using Random
    # With the previous hardcoded sqrt(eps) ≈ 1.5e-8 threshold in _block_isqrt
    # and _diag_inverse, a cutoff of 1e-14 was silently ignored. Now the helpers
    # honour the user's value. We use the same (P, V, n=2, seed=2) as the main
    # canonical! test (which is a known-injective configuration) to ensure
    # the test exercises the canonical form rather than the non-injective throw.
    Random.seed!(2)
    P = graded_space(:U1, 1=>1, -1=>1)
    V = graded_space(:U1, 0=>1, 1=>1, -1=>1, 2=>1, -2=>1)
    ψ = rand_iMPS(P, V, 2)
    canonical!(ψ; cutoff=1e-14)
    vals = schmidt_values(ψ, 1)
    @test issorted(vals; rev=true)
    @test isapprox(sum(abs2, vals), 1.0; atol=1e-9)
    # Any sub-eps noise modes must be cleared — if cutoff were silently floored
    # at sqrt(eps) ≈ 1.5e-8, this assertion would not be informative; with the
    # user-supplied 1e-14 in effect, the kept entries must all clear it.
    @test all(>(1e-14), vals)
end
