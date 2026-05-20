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

@testset "SVD_TRIM_ITERATIVE_KEEPS_ONE_VALUE_BELOW_THRESHOLD" begin
    M = Matrix(Diagonal([1e-14, 5e-15, 0.0]))

    U, S, V = iTEBD.svd_trim(M; maxdim=3, svd_min=1e-12, use_iterative=true)

    @test U isa Matrix{Float64}
    @test S isa Vector{Float64}
    @test V isa Matrix{Float64}
    @test size(U) == (3, 1)
    @test length(S) == 1
    @test size(V) == (1, 3)
    @test isfinite(S[1])
    @test 0 <= S[1] < 1e-12
end

@testset "SVD_TRIM_ITERATIVE_RECONSTRUCTS_TINY_RETAINED_VALUE" begin
    M = Matrix(Diagonal([1e-30, 0.0, 0.0]))

    U, S, V = iTEBD.svd_trim(M; maxdim=3, svd_min=1e-12, use_iterative=true)

    @test U * Diagonal(S) * V ≈ M atol=1e-40 rtol=1e-8
end

@testset "SVD_TRUNCATE_BY_ERROR_REJECTS_NONFINITE_INPUT" begin
    M = [1.0 Inf; 0.0 1.0]

    @test_throws ArgumentError iTEBD._svd_truncate_by_error(M; maxdim=2)
end

@testset "DISCARDED_WEIGHT_SELECTOR_REJECTS_INVALID_ARGUMENTS" begin
    @test_throws ArgumentError iTEBD._discarded_weight_choice([1.0]; mindim=0)
    @test_throws ArgumentError iTEBD._discarded_weight_choice([1.0]; maxdim=0)
    @test_throws ArgumentError iTEBD._discarded_weight_choice([1.0]; mindim=3, maxdim=2)
    @test_throws ArgumentError iTEBD._discarded_weight_choice([1.0]; truncerr=-1e-3)
    @test_throws ArgumentError iTEBD._discarded_weight_choice([1.0]; svd_min=-1e-3)
end

@testset "DISCARDED_WEIGHT_SELECTOR_AVOIDS_VECTOR_TEMPORARIES" begin
    s = collect(range(1.0, 0.001; length=1000))
    choice = iTEBD._discarded_weight_choice(s; mindim=4, maxdim=128, truncerr=1e-8, svd_min=1e-12)

    @test choice.chi_keep == 128
    @test choice.saturated
    @test choice.smallest_kept_sv == s[128]
    @test choice.largest_discarded_sv == s[129]

    f() = iTEBD._discarded_weight_choice(s; mindim=4, maxdim=128, truncerr=1e-8, svd_min=1e-12)
    f()
    @test (@allocated f()) < 12_000
end

@testset "APPLY_TRANSFER_MATCHES_GTRM_DENSE_ACTION" begin
    using Random
    Random.seed!(20260520)
    for (χ1, d, χ2) in [(3, 2, 3), (4, 3, 5), (6, 2, 4), (8, 2, 8)]
        T1 = randn(ComplexF64, χ1, d, χ1)
        T2 = randn(ComplexF64, χ2, d, χ2)
        M = iTEBD.gtrm(T1, T2)

        ρ_r = randn(ComplexF64, χ2, χ1)
        ref_r = reshape(M * vec(ρ_r), χ2, χ1)
        new_r = iTEBD.apply_transfer(T1, T2, ρ_r; dir=:r)
        @test new_r ≈ ref_r rtol=1e-10

        ρ_l = randn(ComplexF64, χ2, χ1)
        ref_l = reshape(transpose(M) * vec(ρ_l), χ2, χ1)
        new_l = iTEBD.apply_transfer(T1, T2, ρ_l; dir=:l)
        @test new_l ≈ ref_l rtol=1e-10
    end
end

@testset "APPLY_CHAIN_TRANSFER_MATCHES_GTRM_PRODUCT" begin
    using Random
    Random.seed!(20260520)
    for (n, χ, d) in [(2, 4, 2), (3, 6, 2), (4, 8, 3), (3, 12, 2)]
        T1s = [randn(ComplexF64, χ, d, χ) for _ in 1:n]
        T2s = [randn(ComplexF64, χ, d, χ) for _ in 1:n]
        M = iTEBD.gtrm(T1s, T2s)
        ρ = randn(ComplexF64, χ, χ)

        ref_r = reshape(M * vec(ρ), χ, χ)
        new_r = iTEBD.apply_chain_transfer(T1s, T2s, ρ; dir=:r)
        @test new_r ≈ ref_r rtol=1e-10

        ref_l = reshape(transpose(M) * vec(ρ), χ, χ)
        new_l = iTEBD.apply_chain_transfer(T1s, T2s, ρ; dir=:l)
        @test new_l ≈ ref_l rtol=1e-10
    end
end

@testset "APPLY_TRANSFER_REJECTS_BAD_SHAPE_AND_DIR" begin
    T1 = randn(ComplexF64, 4, 2, 4)
    T2 = randn(ComplexF64, 4, 2, 4)
    @test_throws DimensionMismatch iTEBD.apply_transfer(T1, T2, zeros(ComplexF64, 3, 3); dir=:r)
    @test_throws DimensionMismatch iTEBD.apply_transfer(T1, T2, zeros(ComplexF64, 3, 3); dir=:l)
    @test_throws ArgumentError iTEBD.apply_transfer(T1, T2, zeros(ComplexF64, 4, 4); dir=:up)
    bad = randn(ComplexF64, 4, 3, 4)  # different physical dim
    @test_throws DimensionMismatch iTEBD.apply_transfer(T1, bad, zeros(ComplexF64, 4, 4); dir=:r)
end

@testset "APPLY_CHAIN_TRANSFER_REJECTS_BAD_INPUT" begin
    T = [randn(ComplexF64, 4, 2, 4) for _ in 1:3]
    @test_throws ArgumentError iTEBD.apply_chain_transfer(T, T[1:2], zeros(ComplexF64, 4, 4))
    @test_throws ArgumentError iTEBD.apply_chain_transfer(typeof(T)(), typeof(T)(), zeros(ComplexF64, 4, 4))
    @test_throws ArgumentError iTEBD.apply_chain_transfer(T, T, zeros(ComplexF64, 4, 4); dir=:bad)
end

@testset "APPLY_TRANSFER_RECTANGULAR_BONDS_AND_FLOAT64" begin
    using Random
    Random.seed!(20260520)
    # Cross-element-type and asymmetric (χL ≠ χR) tensor shapes — both code paths
    # the existing tests skip. Float64 takes the no-conj branch where adjoint==transpose.
    for ET in (Float64, ComplexF64)
        for (χL1, χR1, χL2, χR2, d) in [(3, 5, 4, 6, 2), (2, 7, 5, 3, 3), (8, 4, 6, 5, 2)]
            T1 = randn(ET, χL1, d, χR1)
            T2 = randn(ET, χL2, d, χR2)
            M = iTEBD.gtrm(T1, T2)

            ρr = randn(ET, χR2, χR1)
            ref_r = reshape(M * vec(ρr), χL2, χL1)
            @test iTEBD.apply_transfer(T1, T2, ρr; dir=:r) ≈ ref_r rtol=1e-10

            ρl = randn(ET, χL2, χL1)
            ref_l = reshape(transpose(M) * vec(ρl), χR2, χR1)
            @test iTEBD.apply_transfer(T1, T2, ρl; dir=:l) ≈ ref_l rtol=1e-10
        end
    end
end

@testset "CHAIN_TRANSFER_WORKSPACE_REUSES_ACROSS_DIFFERENT_TENSORS" begin
    using Random
    Random.seed!(20260520)
    # Workspace sized for the larger chain must safely process a smaller one too.
    big = [randn(ComplexF64, 16, 2, 16) for _ in 1:6]
    small = [randn(ComplexF64, 10, 2, 10) for _ in 1:3]
    ws = iTEBD.ChainTransferWorkspace(big, big)  # infer T via convenience ctor
    ρ_small = randn(ComplexF64, 10, 10)
    ref = iTEBD.apply_chain_transfer(small, small, ρ_small; dir=:r)
    @test iTEBD.apply_chain_transfer!(ws, small, small, ρ_small; dir=:r) ≈ ref rtol=1e-10
end

@testset "APPLY_CHAIN_TRANSFER_INPLACE_ALLOCATIONS_BOUNDED" begin
    using Random
    Random.seed!(20260520)
    # Locks in the contract: in-place sweep is O(1) allocations per call,
    # independent of unit-cell length. Without this guard a regression that
    # reintroduced per-site allocations (e.g. dropping an `@view`) would only be
    # noticed via benchmarks.
    n, χ, d = 6, 16, 2
    T1s = [randn(ComplexF64, χ, d, χ) for _ in 1:n]
    T2s = [randn(ComplexF64, χ, d, χ) for _ in 1:n]
    ρ = randn(ComplexF64, χ, χ)
    ws = iTEBD.ChainTransferWorkspace(T1s, T2s)
    iTEBD.apply_chain_transfer!(ws, T1s, T2s, ρ; dir=:r)  # warmup
    iTEBD.apply_chain_transfer!(ws, T1s, T2s, ρ; dir=:r)
    # Budget covers per-site reshape SubArray wrappers (n*~80 B) plus a little slack.
    @test (@allocated iTEBD.apply_chain_transfer!(ws, T1s, T2s, ρ; dir=:r)) < 4_000
end
