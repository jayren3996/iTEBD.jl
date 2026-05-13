using Test
using LinearAlgebra
using iTEBD

@testset "Schmidt Fix Tests" begin

    @testset "POSITIVE_EIGENSYSTEM_NO_INFLATION" begin
        # Issue 1: eigenvalues of 1e-12 must NOT be inflated to tol (~1.5e-8)
        H = Hermitian(diagm([1e-12, 1e-13]))
        vals, vecs = iTEBD._positive_eigensystem(H)
        # With the bug, the sole retained eigenvalue is inflated to ~sqrt(eps()).
        # With the fix it should stay at the original scale (1e-12).
        @test length(vals) == 1
        @test vals[1] ≈ 1e-12 atol=1e-13
        @test vals[1] < 1e-8  # make sure it was NOT inflated to tol
    end

    @testset "POSITIVE_EIGENSYSTEM_RTOL_KEYWORD" begin
        # Issue 2: user should be able to pass a tighter rtol
        H = Hermitian(diagm([1.0, 1e-10]))
        # Default rtol (~1.5e-8) drops 1e-10
        vals_default, _ = iTEBD._positive_eigensystem(H)
        # Tighter rtol=1e-15 keeps 1e-10
        vals_tight, _ = iTEBD._positive_eigensystem(H; rtol=1e-15)
        @test length(vals_default) == 1
        @test length(vals_tight) == 2
        @test sort(vals_tight) ≈ [1e-10, 1.0] atol=1e-11
    end

    @testset "TRANSFER_DEGENERACY_SIZE_GATE" begin
        # Issue 3: _transfer_degeneracy should work for both small and large bond dims
        # Small bond dimension (dense path)
        Γ_small = randn(ComplexF64, 4, 2, 4)
        res_small = iTEBD._transfer_degeneracy(Γ_small)
        @test res_small.degenerate isa Bool
        @test res_small.count isa Int

        # Large bond dimension (should not crash; uses iterative path)
        Γ_large = randn(ComplexF64, 52, 2, 52)
        res_large = iTEBD._transfer_degeneracy(Γ_large)
        @test res_large.degenerate isa Bool
        @test res_large.count isa Int
    end

    @testset "TOLERANCE_HELPER_EXISTS_AND_CONSISTENT" begin
        # Issue 4: there should be a shared tolerance helper
        @test isdefined(iTEBD, :_tolerance)
        vals = [1.0, 1e-8, 1e-16]
        tol = iTEBD._tolerance(vals; zerotol=1e-20, rtol=sqrt(eps(Float64)))
        @test tol > 0
        @test tol isa Float64
        # The helper should be used by _simple_sector_selection implicitly,
        # so we just verify the helper exists and behaves reasonably.
    end

    @testset "SCHMIDT_CANONICAL_NO_DIAGONAL_ALLOCATION" begin
        # Issue 5: canonicalization should still produce correct results
        Γ = randn(ComplexF64, 3, 2, 3)
        S = [1.0, 0.5, 0.25]
        Γ_new, S_new = iTEBD.schmidt_canonical(Γ, S; maxdim=10, cutoff=1e-15, renormalize=false)
        @test size(Γ_new, 1) == size(Γ_new, 3)
        @test length(S_new) == size(Γ_new, 3)
        @test all(isfinite, Γ_new)
        @test all(isfinite, S_new)
        @test all(>=(0), S_new)
    end

    @testset "SCHMIDT_CANONICAL_LOW_RANK_WARNING" begin
        # Issue 6: when eigenvalues are filtered, a warning should be emitted
        # Build a tensor whose transfer matrix has a tiny eigenvalue that gets filtered
        Γ = zeros(ComplexF64, 3, 2, 3)
        Γ[1, 1, 1] = 1.0
        Γ[2, 2, 2] = 1e-12
        S = [1.0, 1.0, 1.0]
        @test_logs (:warn, r"Low.rank compression") (:warn, r"Low.rank compression") iTEBD.schmidt_canonical(Γ, S; maxdim=10, cutoff=1e-15, renormalize=false, noninjective=:ignore)
    end

end
