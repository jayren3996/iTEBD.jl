using Test
using iTEBD

# Explicit imports from TensorKit to avoid name conflicts with ITensors
# (both are test dependencies and share some exported names like `dim`, `space`).
using TensorKit: U1Irrep, Z2Irrep, ZNIrrep, Vect, dim, block, space, id,
                 blocks, dual, ComplexSpace, sectortype, blocksectors, ←

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
        Sz, Sp, Sm = spin_half_ops(:U1)
        P = graded_space(:U1, 1=>1, -1=>1)

        # Sz is an endomorphism of P with the correct sector values
        @test sectortype(space(Sz, 1)) == U1Irrep
        @test space(Sz, 1) == P
        @test isapprox(block(Sz, U1Irrep(1))[1, 1], 0.5; atol=1e-12)
        @test isapprox(block(Sz, U1Irrep(-1))[1, 1], -0.5; atol=1e-12)

        # Sz*Sz = 0.25 * id(P): valid endomorphism algebra check
        @test isapprox(Sz * Sz, 0.25 * id(P); atol=1e-12)

        # Sp : P ← dual(P),  Sm = Sp' : dual(P) ← P
        # Sp*Sm : P ← P = projector onto spin-up sector
        proj_up = zeros(ComplexF64, P ← P)
        block(proj_up, U1Irrep(1))[1, 1] = 1.0
        @test isapprox(Sp * Sm, proj_up; atol=1e-12)

        # Raising block: block(Sp, U1(1)) maps dual(-1) → (+1)
        @test isapprox(block(Sp, U1Irrep(1))[1, 1], 1.0; atol=1e-12)
    end

    @testset "Trivial / dense fallback" begin
        Sx, Sy, Sz, Sp, Sm, Id = spin_half_ops(:Trivial)
        @test Sz isa AbstractMatrix
        @test isapprox(Sx*Sx + Sy*Sy + Sz*Sz, 0.75 * Id; atol=1e-12)
    end
end
