using Test
using LinearAlgebra
using iTEBD

if !isdefined(Main, :TestUtils)
    include(joinpath(@__DIR__, "test_utils.jl"))
end
using .TestUtils: deterministic_tensor, pauli_matrices

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

@testset "INNER_PRODUCT_ACCEPTS_TENSOR_VECTOR_SELF_OVERLAP" begin
    ψ = product_iMPS(ComplexF64, [[1, 0], [0, 1]])

    @test iTEBD.inner_product(ψ.Γ) ≈ iTEBD.inner_product(ψ.Γ, ψ.Γ) atol=1e-12
end

@testset "OCONTRACT_MATCHES_MATERIALIZED_PATH_WITH_LIMITED_ALLOCATIONS" begin
    Γ1 = deterministic_tensor(16, 2, 32)
    Γ2 = deterministic_tensor(32, 2, 16)
    Ts = [Γ1, Γ2]
    Oraw = reshape(ComplexF64.(1:16), 4, 4)
    O = Oraw + Oraw'
    λl = collect(range(0.5, 1.5; length=16))

    function materialized_contract()
        Γ = iTEBD.tensor_group(Ts)
        iTEBD.tensor_lmul!(λl, Γ)
        return dot(Γ, iTEBD.tensor_umul(O, Γ))
    end

    f() = iTEBD.ocontract(Ts, O, λl)

    @test f() ≈ materialized_contract()
    f()
    f()
    @test (@allocated f()) < 45_000
end

@testset "ENTANGLEMENT_ENTROPY_EDGES" begin
    @test entanglement_entropy([1.0, 0.0]) ≈ 0.0 atol=1e-12
    @test entanglement_entropy([0.5, 0.5]) ≈ log(2) atol=1e-12
    @test entanglement_entropy([0.5, 0.5, 1e-12]; cutoff=1e-10) ≈ log(2) atol=1e-12
end
