using Test
using LinearAlgebra
using iTEBD

if !isdefined(Main, :TestUtils)
    include(joinpath(@__DIR__, "test_utils.jl"))
end
using .TestUtils: pauli_matrices

@testset "PRODUCT_STATE_EXPECTATIONS" begin
    P = pauli_matrices()

    ψ00 = product_iMPS(ComplexF64, [[1, 0], [1, 0]])
    ψ01 = product_iMPS(ComplexF64, [[1, 0], [0, 1]])

    @test iTEBD.expect(ψ00, P.Z, 1, 1) ≈ 1.0 atol=1e-12
    @test iTEBD.expect(ψ01, P.Z, 2, 2) ≈ -1.0 atol=1e-12
    @test iTEBD.expect(ψ01, kron(P.Z, P.Z), 1, 2) ≈ -1.0 atol=1e-12
    @test iTEBD.expect(ψ01, kron(P.Z, P.Z), 2, 1) ≈ -1.0 atol=1e-12
    @test iTEBD.expect(ψ00, P.X, 1, 1) ≈ 0.0 atol=1e-12
    @test iTEBD.expect(ψ00, P.I, 1, 1) ≈ 1.0 atol=1e-12
end

@testset "INNER_PRODUCT_PRODUCT_STATES" begin
    ψ00 = product_iMPS(ComplexF64, [[1, 0], [1, 0]])
    ψ01 = product_iMPS(ComplexF64, [[1, 0], [0, 1]])

    @test iTEBD.inner_product(ψ00) ≈ 1.0 atol=1e-12
    @test iTEBD.inner_product(ψ00, ψ00) ≈ 1.0 atol=1e-12
    @test iTEBD.inner_product(ψ00, ψ01) ≈ 0.0 atol=1e-12
end

@testset "ENTANGLEMENT_ENTROPY_EDGES" begin
    @test entanglement_entropy([1.0, 0.0]) ≈ 0.0 atol=1e-12
    @test entanglement_entropy([0.5, 0.5]) ≈ log(2) atol=1e-12
    @test entanglement_entropy([0.5, 0.5, 1e-12]; cutoff=1e-10) ≈ log(2) atol=1e-12
end
