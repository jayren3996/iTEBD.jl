using Test
using LinearAlgebra
using iTEBD

if !isdefined(Main, :TestUtils)
    include(joinpath(@__DIR__, "test_utils.jl"))
end
using .TestUtils: bell_gate, pauli_matrices

@testset "OPERATOR_SPAN_VALIDATION" begin
    P = pauli_matrices()
    ψ = product_iMPS(ComplexF64, [[1, 0], [0, 1]])

    @test operator_span(ψ, P.Z) == 1
    @test operator_span(ψ, kron(P.Z, P.Z)) == 2
    @test_throws ArgumentError operator_span(ψ, ones(ComplexF64, 2, 3))
    @test_throws ArgumentError operator_span(ψ, Matrix{ComplexF64}(I, 3, 3))
end

@testset "ENERGY_DENSITY_PRODUCT_STATE" begin
    P = pauli_matrices()
    ψ00 = product_iMPS(ComplexF64, [[1, 0], [1, 0]])
    ψ01 = product_iMPS(ComplexF64, [[1, 0], [0, 1]])

    @test energy_density(ψ00, P.Z) ≈ 1.0 atol=1e-12
    @test energy_density(ψ01, P.Z) ≈ 0.0 atol=1e-12
    @test energy_density(ψ01, kron(P.Z, P.Z)) ≈ -1.0 atol=1e-12
end

@testset "SCARFINDER_NSTEP_VALIDATION" begin
    @test_throws ArgumentError iTEBD._warn_scarfinder_nstep(0, :gate)
    @test iTEBD._warn_scarfinder_nstep(2, :gate) === nothing
end

@testset "SCARFINDER_UNIFORM_EVOLVE_VALIDATION" begin
    ψ = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
    G = Matrix{ComplexF64}(I, 2, 2)

    @test_throws ArgumentError iTEBD._evolve_uniform!(ψ, G; span=0)
    @test iTEBD._evolve_uniform!(ψ, G; span=1, maxdim=1) === ψ
end

@testset "SCARFINDER_UNIFORM_EVOLVE_SPAN2_MATCHES_EXPLICIT_SWEEP" begin
    ψ = product_iMPS(ComplexF64, [[1, 0], [1, 0]])
    G = bell_gate()

    ψ_uniform = deepcopy(ψ)
    ψ_explicit = deepcopy(ψ)

    iTEBD._evolve_uniform!(ψ_uniform, G; span=2, maxdim=4)
    applygate!(ψ_explicit, G, 1, 2; maxdim=4)
    applygate!(ψ_explicit, G, 2, 1; maxdim=4)

    @test size.(ψ_uniform.Γ) == size.(ψ_explicit.Γ)
    @test length.(ψ_uniform.λ) == length.(ψ_explicit.λ)
    @test ψ_uniform.λ[1] ≈ ψ_explicit.λ[1] atol=1e-12
    @test ψ_uniform.λ[2] ≈ ψ_explicit.λ[2] atol=1e-12
    if size.(ψ_uniform.Γ) == size.(ψ_explicit.Γ)
        @test iTEBD.inner_product(ψ_uniform, ψ_explicit) ≈ 1.0 atol=1e-12
    end
end
