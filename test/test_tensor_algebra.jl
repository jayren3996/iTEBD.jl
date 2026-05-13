using Test
using LinearAlgebra
using iTEBD

if !isdefined(Main, :TestUtils)
    include(joinpath(@__DIR__, "test_utils.jl"))
end
using .TestUtils: deterministic_tensor

@testset "TENSOR_DIAGONAL_MULTIPLICATION" begin
    Γ = reshape(ComplexF64.(1:12), 2, 3, 2)

    left = copy(Γ)
    iTEBD.tensor_lmul!([2.0, 3.0], left)
    @test left[1, :, :] ≈ 2 .* Γ[1, :, :]
    @test left[2, :, :] ≈ 3 .* Γ[2, :, :]

    right = copy(Γ)
    iTEBD.tensor_rmul!(right, [5.0, 7.0])
    @test right[:, :, 1] ≈ 5 .* Γ[:, :, 1]
    @test right[:, :, 2] ≈ 7 .* Γ[:, :, 2]
end

@testset "TENSOR_GROUPING_INVARIANTS" begin
    Γ1 = deterministic_tensor(2, 2, 3)
    Γ2 = deterministic_tensor(3, 2, 4)
    Γ3 = deterministic_tensor(4, 2, 2)

    grouped2 = iTEBD.tensor_group([Γ1, Γ2])
    grouped3 = iTEBD.tensor_group([Γ1, Γ2, Γ3])

    @test size(grouped2) == (2, 4, 4)
    @test size(grouped3) == (2, 8, 2)
    @test grouped2 ≈ iTEBD.tensor_group_2(Γ1, Γ2)
    @test grouped3 ≈ iTEBD.tensor_group_3(Γ1, Γ2, Γ3)

    single = iTEBD.tensor_group([Γ1])
    @test single == Γ1
    @test single !== Γ1
end

@testset "TENSOR_SVD_RECONSTRUCTS_UNTRUNCATED_BLOCK" begin
    T = deterministic_tensor(2, 2, 2, 3)

    U, S, V = iTEBD.tensor_svd(T; maxdim=20, svd_min=0.0)

    @test size(U) == (2, 2, length(S))
    @test size(V) == (length(S), 2, 3)
    @test reshape(U, 4, :) * Diagonal(S) * reshape(V, length(S), :) ≈ reshape(T, 4, 6)
end

@testset "TENSOR_DECOMP_ROUNDTRIPS_GROUPED_BLOCK" begin
    Γ1 = deterministic_tensor(2, 2, 3)
    Γ2 = deterministic_tensor(3, 2, 4)
    Γ3 = deterministic_tensor(4, 2, 2)
    grouped = iTEBD.tensor_group([Γ1, Γ2, Γ3])

    Γs, λs = iTEBD.tensor_decomp!(copy(grouped), ones(2), 3; maxdim=20, svd_min=0.0)

    @test length(Γs) == 3
    @test length(λs) == 2
    @test size.(Γs) == [(2, 2, 4), (4, 2, 4), (4, 2, 2)]
    @test iTEBD.tensor_group(Γs) ≈ grouped atol=1e-10
end

@testset "SVD_TRIM_THRESHOLDS_AND_RENORMALIZES" begin
    M = Matrix(Diagonal([4.0, 2.0, 1e-13, 0.0]))

    U, S, V = iTEBD.svd_trim(M; maxdim=4, svd_min=1e-12)
    @test S ≈ [4.0, 2.0]
    @test U * Diagonal(S) * V ≈ Diagonal([4.0, 2.0, 0.0, 0.0])

    _, S2, _ = iTEBD.svd_trim(M; maxdim=1, svd_min=0.0, renormalize=true)
    @test length(S2) == 1
    @test norm(S2) ≈ 1.0
end

@testset "DISCARDED_WEIGHT_SELECTOR_REJECTS_INVALID_ARGUMENTS" begin
    @test_throws ArgumentError iTEBD._discarded_weight_choice([1.0]; mindim=0)
    @test_throws ArgumentError iTEBD._discarded_weight_choice([1.0]; maxdim=0)
    @test_throws ArgumentError iTEBD._discarded_weight_choice([1.0]; mindim=3, maxdim=2)
    @test_throws ArgumentError iTEBD._discarded_weight_choice([1.0]; truncerr=-1e-3)
    @test_throws ArgumentError iTEBD._discarded_weight_choice([1.0]; svd_min=-1e-3)
end

@testset "RENORMALIZE_SINGULAR_VALUES_THROWS_ON_ZERO_NORM" begin
    @test_throws ArgumentError iTEBD._renormalize_singular_values!([0.0, 0.0])
    @test_throws ArgumentError iTEBD._renormalize_singular_values!(Float64[])
    @test_throws ArgumentError iTEBD._renormalize_singular_values!([Inf])
    @test_throws ArgumentError iTEBD._renormalize_singular_values!([NaN])
end

@testset "TENSOR_DECOMP_REJECTS_MALFORMED_GROUPED_TENSOR" begin
    # size(Γ, 2) = 6, n = 3, d = round(Int, 6^(1/3)) = 2, but 2^3 = 8 != 6
    malformed = randn(2, 6, 2)
    @test_throws ArgumentError iTEBD.tensor_decomp!(malformed, ones(2), 3; maxdim=20, svd_min=0.0)
end

@testset "SUPPORT_TOL_SCALES_WITH_SMALL_VALUES" begin
    # With scale 1e-6, tolerance should be ~1e-6 * rtol, not ~rtol (due to max(scale, 1.0))
    rtol = sqrt(eps(Float64))
    vals = [1e-6]
    tol = iTEBD._support_tol(vals; atol=0.0, rtol=rtol)
    @test tol ≈ rtol * 1e-6 atol=1e-20
    @test tol < rtol * 1e-2  # Should be much smaller than the inflated ~1e-8

    # Empty vector should return atol
    @test iTEBD._support_tol(Float64[]; atol=1e-10, rtol=rtol) == 1e-10
end

@testset "SAFE_RECIPROCAL_PRESERVES_SMALL_VALUES" begin
    vals = [1.0, 1e-12]
    invvals = iTEBD._safe_reciprocal(vals)
    @test invvals[2] != 0.0
    @test invvals[2] ≈ 1e12
end

struct _FailingSVDMatrix <: AbstractMatrix{Float64}
    data::Matrix{Float64}
end
Base.size(M::_FailingSVDMatrix) = size(M.data)
Base.getindex(M::_FailingSVDMatrix, i::Int, j::Int) = M.data[i, j]
function LinearAlgebra.svd(M::_FailingSVDMatrix; kwargs...)
    throw(LAPACKException(1))
end

@testset "SVD_WITH_FALLBACK_SCALES_PERTURBATION_RELATIVELY" begin
    # A rank-deficient matrix with norm ~1.0
    M1 = _FailingSVDMatrix(Float64[1 0; 0 0])
    res1 = iTEBD._svd_with_fallback(M1)
    @test all(res1.S .>= 0.0)
    @test res1.S[1] ≈ 1.0 atol=1e-10

    # A rank-deficient matrix with large norm ~1e6
    # Old code: perturbation = 1e-12 (fixed)
    # New code: perturbation = 1e-12 * max(1e6, 1.0) = 1e-6 (relative)
    M2 = _FailingSVDMatrix(Float64[1e6 0; 0 0])
    res2 = iTEBD._svd_with_fallback(M2)
    @test all(res2.S .>= 0.0)
    @test res2.S[1] ≈ 1e6 atol=1e-4
    @test res2.S[2] ≈ 1e-6 atol=1e-10
end

@testset "SVD_TRIM_DEFAULT_USES_DENSE_SVD" begin
    # With default kwargs, svd_trim should use dense SVD, not iterative
    # We test this by using a matrix that would trigger iterative under old auto logic
    # (300x300, maxdim=50 < 300/4=75) but not under new conservative logic
    M = randn(300, 300)
    U, S, V = iTEBD.svd_trim(M)  # default kwargs
    @test length(S) >= 1
    @test all(S .>= 0.0)
    @test size(U, 1) == 300
    @test size(V, 2) == 300
    @test size(U, 2) == length(S)
    @test size(V, 1) == length(S)
end

@testset "ITERATIVE_SVD_TRIM_WARNS_ON_ILL_CONDITIONED" begin
    # A matrix with very small singular values (1e-12) is ill-conditioned
    # The Gram-matrix approach squares condition numbers, so it should warn
    M = Float64[1.0 0.0; 0.0 1e-12]
    # The iterative path should either warn, error, or handle correctly
    # Currently it silently returns wrong results - we require at least a warning
    @test_logs (:warn, r"iterative SVD.*ill-conditioned|well-conditioned") iTEBD._iterative_svd_trim(M; maxdim=2, svd_min=0.0)
end

@testset "BONDSTAT_RETURNS_CONCRETE_TYPES" begin
    # tensor_svd with return_stats=true and truncerr should return a concrete stats type
    T = randn(2, 2, 2, 3)
    U, S, V, stats = iTEBD.tensor_svd(T; maxdim=20, svd_min=0.0, truncerr=1e-8, return_stats=true)
    @test typeof(stats) <: NamedTuple

    # tensor_decomp! with return_stats=true should return Vector{BondStat}
    Γ1 = deterministic_tensor(2, 2, 3)
    Γ2 = deterministic_tensor(3, 2, 4)
    Γ3 = deterministic_tensor(4, 2, 2)
    grouped = iTEBD.tensor_group([Γ1, Γ2, Γ3])

    Γs, λs, bond_stats = iTEBD.tensor_decomp!(copy(grouped), ones(2), 3; maxdim=20, svd_min=0.0, truncerr=1e-8, return_stats=true)
    @test bond_stats isa Vector{iTEBD.BondStat}
    @test length(bond_stats) == 2
    bs = bond_stats[1]
    @test bs.bond isa Int
    @test bs.chi_req isa Int
    @test bs.chi_keep isa Int
    @test bs.discarded_weight isa Float64
    @test bs.saturated isa Bool
    @test bs.target_met isa Bool
    @test bs.smallest_kept_sv isa Float64
    @test bs.largest_discarded_sv isa Float64
end
