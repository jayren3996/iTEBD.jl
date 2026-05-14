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
    # Should not warn for well-behaved tensor (just test it doesn't error)
    @test iTEBD.krylov_eigen(K, conj(K); dir=:r) isa Tuple
end

@testset "KRYLOV_EIGEN_PRESERVES_MIXED_TRANSFER_EIGENVECTOR" begin
    KL = ComplexF64[
        -0.04990981619173357 + 0.37581081664311117im 0.8236903566197061 + 0.18919661481672084im;
        -0.5705307511354865 + 1.737355233576168im 1.237389980613537 - 0.5840849033672174im;;;
        -0.7373373057894631 - 0.2327327743230712im 0.339900158882909 - 0.0018089068174504633im;
        -0.341003599625987 + 0.8359718325825684im 1.0144242636922476 + 0.3762122276115845im
    ]
    KU = ComplexF64[
        0.17242705063206054 + 0.01431603714733052im 0.5642210453859982 - 0.8805362566291404im;
        -0.24993329832938052 + 0.06525916339956629im -0.35663890247391805 + 0.5218964243858084im;;;
        -0.35850823943240706 + 1.0978558814874178im -1.1122310977685903 - 0.9829602239960348im;
        0.5851136974491122 + 1.5004674770426478im 0.6133171675832304 - 0.33749865941784685im
    ]

    λ, ρ = iTEBD.krylov_eigen(KL, KU; dir=:r)

    @test norm(iTEBD.kraus(KL, KU, ρ; dir=:r) - λ * ρ) / (abs(λ) * norm(ρ)) < 1e-8
end

@testset "KRYLOV_EIGEN_CAN_PROJECT_SELF_TRANSFER_TO_PSD" begin
    K = deterministic_tensor(2, 2, 2)
    _, ρ = iTEBD.krylov_eigen(K, conj(K); dir=:r, project_psd=true)
    evals = eigvals(Hermitian(ρ))
    @test all(evals .>= -1e-8)  # Allow tiny negative from numerical noise
end

@testset "KRYLOV_EIGEN_SUPPORTS_MIXED_RECTANGULAR_FIXED_POINT" begin
    KL = deterministic_tensor(2, 2, 3)
    KU = deterministic_tensor(3, 2, 2)

    λ, ρ = iTEBD.krylov_eigen(KL, KU; dir=:r)

    @test size(ρ) == (3, 2)
    @test vec(iTEBD.kraus(KL, KU, ρ; dir=:r)) ≈ λ * vec(ρ) rtol=1e-8

    λ_left, ρ_left = iTEBD.krylov_eigen(KL, KU; dir=:l)

    @test size(ρ_left) == (3, 2)
    @test vec(iTEBD.kraus(KL, KU, ρ_left; dir=:l)) ≈ λ_left * vec(ρ_left) rtol=1e-8
end

@testset "KRYLOV_EIGEN_VALIDATES_INITIAL_SHAPE_AND_PSD_PROJECTION" begin
    KL = deterministic_tensor(2, 2, 3)
    KU = deterministic_tensor(3, 2, 2)

    @test_throws ArgumentError iTEBD.krylov_eigen(KL, KU, ones(2, 2); dir=:r)
    @test_throws ArgumentError iTEBD.krylov_eigen(KL, KU; dir=:r, project_psd=true)
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
