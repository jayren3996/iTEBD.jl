using Test
using iTEBD

# Explicit imports from TensorKit to avoid name conflicts with ITensors
# (both are test dependencies and share some exported names like `dim`, `space`).
using TensorKit: U1Irrep, Z2Irrep, ZNIrrep, Vect, dim, block, space, id,
                 ComplexSpace, sectortype, domain, codomain

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
    @test ψ isa iTEBD.SymmetricIMPS
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
