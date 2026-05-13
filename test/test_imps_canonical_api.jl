using Test
using LinearAlgebra
using iTEBD
using iTEBD: rand_iMPS, product_iMPS, canonical!, iMPS

if !isdefined(Main, :TestUtils)
    include(joinpath(@__DIR__, "test_utils.jl"))
end
using .TestUtils: assert_normalized_schmidt_spectra, assert_stored_tensor_convention, right_canonical_error

function _assert_finite_nonempty(psi)
    @test all(!isempty(λ) for λ in psi.λ)
    @test all(all(isfinite, λ) for λ in psi.λ)
    @test all(all(isfinite, Γ) for Γ in psi.Γ)
    @test all(all(>(0), size(Γ)) for Γ in psi.Γ)
end

@testset "IMPS_CANONICAL_CONVENTION" begin
    psi = rand_iMPS(ComplexF64, 1, 2, 1)

    @test canonical!(psi) === psi
    @test length(psi.Γ) == 1
    @test length(psi.λ) == 1
    @test right_canonical_error(psi) <= 1e-10
end

@testset "IMPS_MULTI_SITE_CANONICAL_INVARIANTS" begin
    psi = rand_iMPS(ComplexF64, 2, 2, 1)

    assert_normalized_schmidt_spectra(psi; atol=1e-10)
    assert_stored_tensor_convention(psi; atol=1e-12)
    @test right_canonical_error(psi) <= 1e-10
    @test iTEBD.inner_product(psi) ≈ 1.0 atol=1e-10
end

@testset "PRODUCT_IMPS_CANONICAL_ON_CONSTRUCTION" begin
    psi = product_iMPS(ComplexF64, [[2, 0]])

    assert_normalized_schmidt_spectra(psi; atol=1e-12)
    assert_stored_tensor_convention(psi; atol=1e-12)
    @test right_canonical_error(psi) <= 1e-12
    @test iTEBD.inner_product(psi) ≈ 1.0 atol=1e-12

    psi_untyped = product_iMPS([[1, 1]])
    @test eltype(psi_untyped) == Float64
    @test psi_untyped.Γ[1][1, :, 1] ≈ fill(inv(sqrt(2)), 2) atol=1e-12
end

@testset "INDEXING_DIVIDES_SMALL_RETAINED_SCHMIDT_VALUES" begin
    λ = [1.0, 1e-9]
    B = zeros(ComplexF64, 2, 2, 2)
    B[1, 1, 1] = 1.0
    B[2, 2, 2] = λ[2]
    psi = iMPS([B], [λ], 1)

    Γ, returned_λ = psi[1]

    @test returned_λ === psi.λ[1]
    @test Γ[1, 1, 1] ≈ 1.0 atol=1e-12
    @test Γ[2, 2, 2] ≈ 1.0 atol=1e-12
end

@testset "CANONICAL_ARGUMENT_VALIDATION" begin
    psi = product_iMPS(ComplexF64, [[1, 0]])

    @test_throws ArgumentError canonical!(psi; maxdim=0)
    @test_throws ArgumentError canonical!(psi; cutoff=-1e-12)
    @test_throws ArgumentError canonical!(psi; noninjective=:invalid)
    @test_throws ArgumentError canonical!(psi; symmetry_break=:invalid)
end

@testset "CANONICAL_STABILITY_EDGE_CASES" begin
    psi = rand_iMPS(ComplexF64, 1, 2, 1)
    canonical!(psi; cutoff=2.0, noninjective=:ignore)
    _assert_finite_nonempty(psi)
    @test all(size(Γ, 1) > 0 && size(Γ, 3) > 0 for Γ in psi.Γ)

    G = zeros(ComplexF64, 2, 2, 2)
    G[1, 1, 1] = 1
    G[1, 2, 2] = 1
    G[2, 1, 1] = 1
    G[2, 2, 2] = 1
    for λ in ([1.0, 1e-300], [1.0, 0.0])
        psi_tiny = iMPS([copy(G)], [collect(λ)], 1)
        canonical!(psi_tiny; noninjective=:ignore)
        _assert_finite_nonempty(psi_tiny)
    end
end

@testset "GHZ_LIKE_NONINJECTIVE_WARNING" begin
    G = zeros(ComplexF64, 2, 2, 2)
    G[1, 1, 1] = 1
    G[2, 2, 2] = 1
    psi_default = iMPS([copy(G)], [ones(2)], 1)

    @test_logs (:warn, r"without symmetry-sector selection") (:warn,) (:warn,) canonical!(psi_default; noninjective=:warn)
    _assert_finite_nonempty(psi_default)

    psi_auto = iMPS([copy(G)], [ones(2)], 1)
    @test_logs (:warn, r"selected virtual sector") canonical!(psi_auto; noninjective=:warn, symmetry_break=:auto)
    _assert_finite_nonempty(psi_auto)

    @test_throws ArgumentError canonical!(
        iMPS([copy(G)], [ones(2)], 1);
        noninjective=:error,
        symmetry_break=:auto,
    )
end

@testset "NO_BLOCK_CANONICAL_EXPORT" begin
    @test !isdefined(iTEBD, :block_canonical)
end
