using Test
using iTEBD

# Explicit imports from TensorKit to avoid name conflicts with ITensors
# (both are test dependencies and share some exported names like `dim`, `space`).
using TensorKit: U1Irrep, Z2Irrep, ZNIrrep, Vect, dim, block, space, id,
                 ComplexSpace, sectortype, sectors, domain, codomain,
                 AbstractTensorMap, DiagonalTensorMap, ←, ⊗

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
