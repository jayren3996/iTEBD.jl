using Test
using LinearAlgebra
using iTEBD

if !isdefined(Main, :TestUtils)
    include(joinpath(@__DIR__, "test_utils.jl"))
end
using .TestUtils: bell_gate, pauli_matrices

@testset "APPLYGATE_ONE_SITE_PATH_RETURNS_EMPTY_STATS" begin
    P = pauli_matrices()
    ψ = product_iMPS(ComplexF64, [[1, 0]])

    ψ_after, stats = applygate!(ψ, P.X, 1, 1; return_stats=true)

    @test ψ_after === ψ
    @test iTEBD.expect(ψ, P.Z, 1, 1) ≈ -1.0 atol=1e-12
    @test stats.support == (1, 1)
    @test isempty(stats.bond_stats)
    @test stats.max_discarded_weight == 0.0
    @test stats.num_saturated == 0
end

@testset "APPLYGATE_IDENTITY_PRESERVES_PRODUCT_STATE" begin
    ψ0 = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
    ψ = product_iMPS(ComplexF64, [[1, 0], [0, 1]])

    applygate!(ψ, Matrix{ComplexF64}(I, 4, 4), 1, 2; maxdim=4)

    @test iTEBD.inner_product(ψ, ψ0) ≈ 1.0 atol=1e-12
    @test maximum(length.(ψ.λ)) == 1
end

@testset "APPLYGATE_WRAPAROUND_INVERSE_RESTORES_STATE" begin
    G = bell_gate()
    ψ0 = product_iMPS(ComplexF64, [[1, 0], [1, 0]])
    ψ = product_iMPS(ComplexF64, [[1, 0], [1, 0]])

    applygate!(ψ, G, 2, 1; maxdim=4)
    applygate!(ψ, adjoint(G), 2, 1; maxdim=4)

    @test iTEBD.inner_product(ψ, ψ0) ≈ 1.0 atol=1e-12
    @test maximum(length.(ψ.λ)) == 1
end

@testset "LEGACY_CUTOFF_KEYWORD_IS_ACCEPTED" begin
    G = bell_gate()
    ψ = product_iMPS(ComplexF64, [[1, 0], [1, 0]])

    @test iTEBD._resolve_svd_min(iTEBD.SVDTOL, nothing) == iTEBD.SVDTOL
    @test iTEBD._resolve_svd_min(iTEBD.SVDTOL, 0.0) == 0.0
    @test applygate!(ψ, G, 1, 2; maxdim=4, cutoff=0.0) === ψ
    @test maximum(length.(ψ.λ)) == 2
    @test_throws ArgumentError applygate!(ψ, G, 1, 2; cutoff=1e-12, svd_min=2e-12)
end

@testset "EVOLVE_ZERO_STEPS_IS_NOOP_WITH_EMPTY_STATS" begin
    ψ = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
    before_Γ = deepcopy(ψ.Γ)
    before_λ = deepcopy(ψ.λ)

    ψ_after, stats = evolve!(
        ψ,
        [(Matrix{ComplexF64}(I, 4, 4), 1, 2)],
        0;
        return_stats=true
    )

    @test ψ_after === ψ
    @test ψ.Γ == before_Γ
    @test ψ.λ == before_λ
    @test isempty(stats.gate_updates)
    @test stats.max_discarded_weight == 0.0
    @test stats.mean_discarded_weight == 0.0
    @test stats.max_kept_dim == 0
    @test stats.num_saturated == 0
end

@testset "LEGACY_ADAPTIVE_EVOLVE_KEYWORDS_ARE_ACCEPTED" begin
    G = bell_gate()
    ψ = product_iMPS(ComplexF64, [[1, 0], [1, 0]])

    @test evolve!(
        ψ,
        [(G, 1, 2)],
        1;
        maxdim=4,
        mindim=1,
        cutoff=0.0,
        chi_policy=:adaptive,
        q=1.0,
        alpha=0.1,
    ) === ψ
    @test maximum(length.(ψ.λ)) == 2
    @test all(length(λ) <= 4 for λ in ψ.λ)

    @test_throws ArgumentError evolve!(ψ, [(G, 1, 2)], 1; chi_policy=:unknown)
end

@testset "PUBLIC_GATE_VALIDATION_ERRORS" begin
    ψ = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
    G = Matrix{ComplexF64}(I, 4, 4)

    @test_throws ArgumentError applygate!(ψ, G, 1, 2; maxdim=0)
    @test_throws ArgumentError applygate!(ψ, G, 1, 2; mindim=0)
    @test_throws ArgumentError applygate!(ψ, G, 1, 2; maxdim=1, mindim=2)
    @test_throws ArgumentError applygate!(ψ, G, 1, 2; truncerr=-1e-3)
    @test_throws ArgumentError applygate!(ψ, G, 1, 2; svd_min=-1e-3)

    @test_throws ArgumentError evolve!(ψ, [(G, 1, 2)], -1)
    @test_throws ArgumentError evolve!(ψ, [(G, 1, 2)], 1; maxdim=0)
    @test_throws ArgumentError evolve!(ψ, [(G, 1, 2)], 1; mindim=0)
    @test_throws ArgumentError evolve!(ψ, [(G, 1, 2)], 1; maxdim=1, mindim=2)
    @test_throws ArgumentError evolve!(ψ, [(G, 1, 2)], 1; truncerr=-1e-3)
    @test_throws ArgumentError evolve!(ψ, [(G, 1, 2)], 1; svd_min=-1e-3)
end

@testset "CONVERT_OPERATOR_ORDERING" begin
    M = reshape(collect(1.0:16.0), 4, 4)
    I4 = Matrix{Float64}(I, 4, 4)
    C = iTEBD.convert_operator(M, 2, 2)

    @test size(C) == (4, 4)
    @test iTEBD.convert_operator(C, 2, 2) == M
    @test iTEBD.convert_operator(I4, 2, 2) == I4
    @test iTEBD.convert_operator([1.0 2.0; 3.0 4.0], 2, 1) == [1.0 2.0; 3.0 4.0]
end
