using Test
using LinearAlgebra
using Random
using iTEBD
using iTEBD: rand_iMPS, product_iMPS, canonical!, iMPS

Random.seed!(20260522)

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

    @test returned_λ == psi.λ[1]
    @test returned_λ !== psi.λ[1]  # copy, not aliased — see GETINDEX_RETURNS_INDEPENDENT_SCHMIDT_COPY
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

    # noninjective=:error should throw before symmetry_break is considered,
    # regardless of which symmetry_break value is set. Previously only the
    # :error+:auto path was tested.
    @test_throws ArgumentError canonical!(
        iMPS([copy(G)], [ones(2)], 1);
        noninjective=:error,
        symmetry_break=:none,
    )

    # noninjective=:ignore should silently produce a usable state on the same
    # degenerate input, with no @warn from the noninjective code path. The
    # state's tensors and Schmidt vectors should be finite and non-empty.
    psi_ignore = iMPS([copy(G)], [ones(2)], 1)
    canonical!(psi_ignore; noninjective=:ignore)
    _assert_finite_nonempty(psi_ignore)
end

@testset "NO_BLOCK_CANONICAL_EXPORT" begin
    @test !isdefined(iTEBD, :block_canonical)
end

@testset "CONJ_IMPS_INDEPENDENT_SCHMIDT" begin
    psi = rand_iMPS(ComplexF64, 2, 2, 4)
    psi_conj = conj(psi)
    @test psi_conj.n == psi.n
    for i in 1:psi.n
        @test psi_conj.Γ[i] ≈ conj(psi.Γ[i])
        @test psi_conj.λ[i] ≈ psi.λ[i]
    end
    # Mutating the conjugated state's Schmidt vector must not alias the
    # original (regression test for the `get_data` UndefVarError + λ aliasing).
    sentinel = -123.0
    psi_conj.λ[1][1] = sentinel
    @test psi.λ[1][1] != sentinel
end

@testset "MPS_PROMOTE_TYPE_CONVERTS_AND_COPIES" begin
    psi = rand_iMPS(ComplexF64, 2, 2, 4)
    psi32 = iTEBD.mps_promote_type(ComplexF32, psi)
    @test all(eltype(Γ) == ComplexF32 for Γ in psi32.Γ)
    @test psi32.n == psi.n
    # λ must be deep-copied so the promoted state has independent storage.
    sentinel = Float32(-7.0)
    psi32.λ[1][1] = sentinel
    @test psi.λ[1][1] != sentinel
end

@testset "GETINDEX_RETURNS_INDEPENDENT_SCHMIDT_COPY" begin
    psi = rand_iMPS(ComplexF64, 2, 2, 4)
    _, lambda1 = psi[1]
    original = psi.λ[1][1]
    lambda1[1] = -42.0
    @test psi.λ[1][1] == original
end

@testset "CANONICAL_AFTER_TRUNCATION_IS_FULLY_CANONICAL" begin
    # Before the re-pass fix, canonical!(psi; maxdim=χ) with χ < natural bond dim
    # left the state only approximately right-canonical (truncated SVDs in
    # tensor_decomp! break the per-site U gauge). canonical! now does a second
    # pass when truncation actually occurred, which restores exact gauge with
    # no further information loss.
    for n in (1, 2, 3)
        psi = rand_iMPS(ComplexF64, n, 2, 6)
        canonical!(psi; maxdim=2)
        @test maximum(length, psi.λ) <= 2
        @test right_canonical_error(psi) <= 1e-10
        assert_normalized_schmidt_spectra(psi; atol=1e-10)
        @test iTEBD.inner_product(psi) ≈ 1.0 atol=1e-9
    end
end

@testset "SCHMIDT_IS_TRUE_BOND_SPECTRUM" begin
    # After canonical!, the stored λ[n] (wraparound bond) must be the
    # *physical* Schmidt spectrum on that bond, i.e., the diagonal of the
    # transfer matrix's left fixed point should be Diagonal(λ_n²) up to
    # overall scale. This pins down the Schmidt-form property of the
    # canonical decomposition, not just the right-canonical condition.
    for n in (1, 2, 3)
        psi = rand_iMPS(ComplexF64, n, 2, 4)
        canonical!(psi)
        Γ = iTEBD.tensor_group(psi.Γ)
        L = Matrix(iTEBD.steady_mat(Γ; dir=:l))
        L_norm = L / tr(L)
        # off-diagonal must be small (state is already in the basis where λ²
        # is diagonal)
        @test norm(L_norm - Diagonal(diag(L_norm))) < 1e-8
        # diagonal entries match λ² up to permutation
        λ2 = sort(real.(psi.λ[end]) .^ 2; rev=true)
        λ2 ./= sum(λ2)
        diag_part = sort(real.(diag(L_norm)); rev=true)
        @test diag_part ≈ λ2 atol=1e-8
    end
end

@testset "CANONICAL_UNITCELL_TRANSFER_HAS_DOMINANT_EIGENVALUE_ONE" begin
    for n in (1, 2, 3)
        psi = rand_iMPS(ComplexF64, n, 2, 4)
        canonical!(psi)
        Γ = iTEBD.tensor_group(psi.Γ)
        T = iTEBD.kraus_mat(Γ, conj(Γ); dir=:r)
        λ_max = maximum(abs, eigvals(T))
        @test λ_max ≈ 1.0 atol=1e-9
    end
end

@testset "STABILITY_MULTISITE_GHZ_LIKE" begin
    # GHZ-like block-diagonal tensors at n > 1: existing GHZ test only covers
    # n = 1. Make sure the warning fires once per pass and the resulting state
    # is still finite and right-canonical.
    G = zeros(ComplexF64, 2, 2, 2)
    G[1, 1, 1] = 1; G[2, 2, 2] = 1
    for n in (2, 3)
        psi = iMPS([copy(G) for _ in 1:n], [ones(2) for _ in 1:n], n)
        canonical!(psi; noninjective=:ignore)
        @test right_canonical_error(psi) <= 1e-9
        @test all(all(isfinite, λ) for λ in psi.λ)
        @test all(all(isfinite, Γ) for Γ in psi.Γ)
    end
end

@testset "STABILITY_NEARLY_NONINJECTIVE_PERTURBATION" begin
    # Tensors that *look* block-diagonal but have a tiny off-diagonal mixing
    # should still canonicalize cleanly across many orders of perturbation
    # magnitude.
    for ε in (1e-2, 1e-6, 1e-10)
        G = zeros(ComplexF64, 2, 2, 2)
        G[1, 1, 1] = 1; G[2, 2, 2] = 1
        G[1, 1, 2] = ε; G[2, 1, 1] = ε
        psi = iMPS([G], [ones(2)], 1)
        canonical!(psi; noninjective=:ignore)
        @test right_canonical_error(psi) <= 1e-9
        @test all(isfinite, psi.λ[1])
    end
end

@testset "STABILITY_ILL_CONDITIONED_INPUT" begin
    # Tensors whose entries span many orders of magnitude (e.g. one row scaled
    # by 1e10) shouldn't crash canonical!. The natural result is heavy
    # rank-compression onto the dominant subspace.
    for scale in (1e6, 1e10, 1e14)
        Random.seed!(42)
        G = randn(ComplexF64, 4, 2, 4)
        G[1, :, :] .*= scale
        psi = iMPS([G], [ones(4)], 1)
        canonical!(psi; noninjective=:ignore)
        @test right_canonical_error(psi) <= 1e-9
        @test all(isfinite, psi.λ[1])
    end
end

@testset "STABILITY_HARD_ZEROS_IN_SCHMIDT" begin
    # Schmidt vectors with exact-zero entries — `_safe_reciprocal` and the
    # support-truncation logic must handle these without producing NaN/Inf.
    for kill_count in 1:3
        Γ = randn(ComplexF64, 4, 2, 4)
        λ = ones(Float64, 4); λ[1:kill_count] .= 0.0
        psi = iMPS([Γ], [λ], 1)
        canonical!(psi; noninjective=:ignore)
        @test right_canonical_error(psi) <= 1e-7
        @test all(isfinite, psi.λ[1])
    end
end

@testset "STABILITY_NAN_INF_REJECTED" begin
    Γ_nan = randn(ComplexF64, 2, 2, 2); Γ_nan[1, 1, 1] = NaN
    @test_throws Exception canonical!(iMPS([Γ_nan], [ones(2)], 1); noninjective=:ignore)

    Γ_inf = randn(ComplexF64, 2, 2, 2); Γ_inf[1, 1, 1] = Inf
    @test_throws Exception canonical!(iMPS([Γ_inf], [ones(2)], 1); noninjective=:ignore)
end

@testset "STABILITY_KRYLOV_PATH_LARGE_BOND" begin
    # bond > 32 forces the Krylov solver inside steady_mat. Exercise both
    # single- and multi-site cases at moderate bond dim.
    for (bond, n) in [(48, 1), (40, 2)]
        Random.seed!(bond + n)
        Γs = [randn(ComplexF64, bond, 2, bond) for _ in 1:n]
        λs = [10.0 .^ (-3 .* rand(bond)) for _ in 1:n]
        psi = iMPS(Γs, λs, n)
        canonical!(psi; noninjective=:ignore)
        @test right_canonical_error(psi) <= 1e-7
        @test all(all(isfinite, λ) for λ in psi.λ)
    end
end

@testset "STABILITY_NOISE_PLUS_RECANONICALIZE" begin
    # Standard TEBD scenario: after applying gates, the state is no longer
    # perfectly canonical. canonical! must converge to a canonical form on a
    # noise-corrupted input.
    Random.seed!(11)
    psi = rand_iMPS(ComplexF64, 2, 2, 4)
    canonical!(psi)
    for Γ in psi.Γ
        Γ .+= 0.1 .* randn(ComplexF64, size(Γ)...)
    end
    @test right_canonical_error(psi) > 1e-2  # we did break it
    canonical!(psi; noninjective=:ignore)
    @test right_canonical_error(psi) <= 1e-9
    assert_normalized_schmidt_spectra(psi; atol=1e-9)
end
