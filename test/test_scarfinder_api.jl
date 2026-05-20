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
    @test_throws ArgumentError operator_span(ψ, ones(ComplexF64, 1, 1))
    @test_throws ArgumentError operator_span(ψ, ones(ComplexF64, 2, 3))
    @test_throws ArgumentError operator_span(ψ, Matrix{ComplexF64}(I, 3, 3))

    ψ_scalar = product_iMPS(ComplexF64, [[1]])
    @test_throws ArgumentError operator_span(ψ_scalar, Matrix{ComplexF64}(I, 2, 2))
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
    @test_throws ArgumentError iTEBD._evolve_uniform!(ψ, Matrix{ComplexF64}(I, 8, 8); span=3)
    @test iTEBD._evolve_uniform!(ψ, G; span=1, maxdim=1) === ψ
end

@testset "SCARFINDER_TRUNCATE_ONE_SITE_UNITCELL" begin
    ψ = product_iMPS(ComplexF64, [[1, 0]])

    @test iTEBD._truncate_unitcell!(ψ, 1) === ψ
    @test ψ.n == 1
    @test length(ψ.λ[1]) == 1
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

@testset "MINIMIZE_ON_TRAJECTORY_LANDS_AT_BEST_VALUE" begin
    # Build a synthetic step!/f pair where the minimum occurs strictly
    # mid-trajectory. The fixed-trajectory test verifies that the final
    # state matches the trial state at the minimum point, instead of
    # whatever a replayed step! would produce.
    ψ = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
    f = x -> -float(x.λ[1][1])         # decreasing in λ[1][1]
    counter = Ref(0)
    function step!(x)
        counter[] += 1
        # Mutate the Schmidt vector so that f(x) traces 0, -1, -2, -1, 0...
        # over four step!s. The minimum is therefore at sample index 3.
        idx = mod(counter[] - 1, 4) + 1
        x.λ[1][1] = (idx == 3 ? 2.0 : idx == 2 ? 1.0 : idx == 4 ? 1.0 : 0.0)
        return x
    end
    iTEBD._minimize_on_trajectory!(f, step!, ψ, 4)
    @test f(ψ) == -2.0
    @test ψ.λ[1][1] == 2.0
end
