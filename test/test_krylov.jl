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
