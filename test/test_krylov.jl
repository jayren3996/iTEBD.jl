using Test
using LinearAlgebra
using iTEBD

if !isdefined(Main, :TestUtils)
    include(joinpath(@__DIR__, "test_utils.jl"))
end
using .TestUtils: deterministic_tensor

@testset "KRAUS_MATRIX_MATCHES_DIRECT_ACTION" begin
    KL = deterministic_tensor(2, 3, 2)
    KU = deterministic_tensor(2, 3, 2) ./ 3
    ρ = deterministic_tensor(2, 2)

    for dir in (:r, :l)
        direct = iTEBD.kraus(KL, KU, ρ; dir)
        matrix_action = reshape(iTEBD.kraus_mat(KL, KU; dir) * vec(ρ), 2, 2)
        @test matrix_action ≈ direct atol=1e-12
end

@testset "STEADY_MAT_SYMMETRIZES_BEFORE_HERMITIAN" begin
    # A tensor that produces a nearly non-Hermitian fixed point due to numerical noise
    K = deterministic_tensor(3, 2, 3) .+ 0.01im .* deterministic_tensor(3, 2, 3)
    mat = iTEBD.steady_mat(K; dir=:r)
    @test mat isa Hermitian
    # Check that the matrix is truly symmetric
    M = Matrix(mat)
    @test M ≈ M' atol=1e-10
end

@testset "KRYLOV_EIGEN_CONVERGENCE_CHECK" begin
    K = deterministic_tensor(2, 2, 2)
    # Should not warn for well-behaved tensor
    @test_logs min_level=Logging.Warn iTEBD.krylov_eigen(K, conj(K); dir=:r)
end

@testset "KRYLOV_EIGEN_RETURNS_PSD" begin
    K = deterministic_tensor(2, 2, 2)
    _, ρ = iTEBD.krylov_eigen(K, conj(K); dir=:r)
    evals = eigvals(Hermitian(ρ))
    @test all(evals .>= -1e-8)  # Allow tiny negative from numerical noise
end

@testset "FIXED_POINT_MAT_RANDOM_PSD_START" begin
    K = deterministic_tensor(2, 2, 2)
    mat = iTEBD.fixed_point_mat(K; dir=:r)
    @test mat isa Hermitian
    evals = eigvals(mat)
    @test all(evals .>= -1e-8)
end

@testset "INNER_PRODUCT_SIZE_GATE" begin
    # Small transfer matrix should use dense eigen
    K = deterministic_tensor(2, 2, 2)
    val = iTEBD.inner_product(K)
    @test val > 0
    @test isfinite(val)
end
end

@testset "KRAUS_REJECTS_ILLEGAL_DIRECTION" begin
    K = ones(ComplexF64, 1, 1, 1)
    ρ = ones(ComplexF64, 1, 1)

    @test_throws ErrorException iTEBD.kraus(K, K, ρ; dir=:sideways)
    @test_throws ErrorException iTEBD.kraus_mat(K, K; dir=:sideways)
end

@testset "STEADY_MAT_RETURNS_HERMITIAN_FIXED_POINT" begin
    K = ones(ComplexF64, 1, 1, 1)

    right = iTEBD.steady_mat(K; dir=:r)
    left = iTEBD.steady_mat(K; dir=:l)

    @test right isa Hermitian
    @test left isa Hermitian
    @test Matrix(right) ≈ ones(ComplexF64, 1, 1)
    @test Matrix(left) ≈ ones(ComplexF64, 1, 1)
end
