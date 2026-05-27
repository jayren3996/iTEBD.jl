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
    @test isapprox(sum(abs2, schmidt_values(ψ, 1)), 1.0; atol=1e-9)
end
