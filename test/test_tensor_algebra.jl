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
