using Test
using LinearAlgebra
using iTEBD

if !isdefined(Main, :TestUtils)
    include(joinpath(@__DIR__, "test_utils.jl"))
end
using .TestUtils: deterministic_tensor, pauli_matrices

@testset "PRODUCT_STATE_EXPECTATIONS" begin
    P = pauli_matrices()

    ψ00 = product_iMPS(ComplexF64, [[1, 0], [1, 0]])
    ψ01 = product_iMPS(ComplexF64, [[1, 0], [0, 1]])

    @test iTEBD.expect(ψ00, P.Z, 1, 1) ≈ 1.0 atol=1e-12
    @test iTEBD.expect(ψ01, P.Z, 2, 2) ≈ -1.0 atol=1e-12
    @test iTEBD.expect(ψ01, kron(P.Z, P.Z), 1, 2) ≈ -1.0 atol=1e-12
    @test iTEBD.expect(ψ01, kron(P.Z, P.Z), 2, 1) ≈ -1.0 atol=1e-12
    @test iTEBD.expect(ψ00, P.X, 1, 1) ≈ 0.0 atol=1e-12
    @test iTEBD.expect(ψ00, P.I, 1, 1) ≈ 1.0 atol=1e-12
end

@testset "INNER_PRODUCT_PRODUCT_STATES" begin
    ψ00 = product_iMPS(ComplexF64, [[1, 0], [1, 0]])
    ψ01 = product_iMPS(ComplexF64, [[1, 0], [0, 1]])

    @test iTEBD.inner_product(ψ00) ≈ 1.0 atol=1e-12
    @test iTEBD.inner_product(ψ00, ψ00) ≈ 1.0 atol=1e-12
    @test iTEBD.inner_product(ψ00, ψ01) ≈ 0.0 atol=1e-12
end

@testset "INNER_PRODUCT_ACCEPTS_TENSOR_VECTOR_SELF_OVERLAP" begin
    ψ = product_iMPS(ComplexF64, [[1, 0], [0, 1]])

    @test iTEBD.inner_product(ψ.Γ) ≈ iTEBD.inner_product(ψ.Γ, ψ.Γ) atol=1e-12
end

@testset "INNER_PRODUCT_MATRIX_FREE_MATCHES_DENSE_PATH" begin
    using Random
    Random.seed!(20260520)
    # χ > dense threshold (8) → matrix-free Krylov is active.
    for (n, χ, d) in [(1, 12, 2), (2, 12, 2), (3, 16, 2), (4, 16, 2), (1, 20, 4)]
        # Right-isometric inputs give a clean dominant eigenvalue at 1.
        T = AbstractArray{ComplexF64, 3}[]
        for _ in 1:n
            G = randn(ComplexF64, χ, d, χ)
            Q, _ = qr(transpose(reshape(G, χ, d * χ)))
            push!(T, reshape(transpose(Matrix(Q)[:, 1:χ]), χ, d, χ))
        end
        matfree = iTEBD.inner_product(T)
        dense = iTEBD._dominant_eigenvalue_dense(iTEBD.gtrm(T, T))
        @test matfree ≈ dense rtol=1e-8
    end
end

@testset "INNER_PRODUCT_MATRIX_FREE_LARGE_UNIT_CELL" begin
    using Random
    Random.seed!(20260520)
    # n ≥ 4 unit cells with χ above the dense threshold — guards the
    # high-impact regime (4-site or longer iTEBD evolutions).
    for (n, χ, d) in [(4, 12, 2), (4, 16, 2), (6, 12, 2), (6, 16, 2),
                       (8, 12, 2), (8, 16, 2), (4, 12, 3)]
        T = AbstractArray{ComplexF64, 3}[]
        for _ in 1:n
            G = randn(ComplexF64, χ, d, χ)
            Q, _ = qr(transpose(reshape(G, χ, d * χ)))
            push!(T, reshape(transpose(Matrix(Q)[:, 1:χ]), χ, d, χ))
        end
        matfree = iTEBD.inner_product(T)
        dense = iTEBD._dominant_eigenvalue_dense(iTEBD.gtrm(T, T))
        @test matfree ≈ dense rtol=1e-8
    end
end

@testset "INNER_PRODUCT_RIGHT_ISOMETRIC_DOMINANT_EIGVAL_IS_ONE" begin
    using Random
    Random.seed!(20260520)
    # Right-isometric tensors satisfy ∑_s B_s B_s† = I, so the dominant transfer
    # eigenvalue is exactly 1. Pins down both the matrix-free Krylov path (χ>8)
    # and the dense path (χ<=8) end-to-end without relying on the dense path as
    # the reference.
    for (n, χ) in [(1, 4), (1, 12), (4, 6), (4, 16), (8, 12)]
        T = AbstractArray{ComplexF64, 3}[]
        for _ in 1:n
            G = randn(ComplexF64, χ, 2, χ)
            Q, _ = qr(transpose(reshape(G, χ, 2χ)))
            push!(T, reshape(transpose(Matrix(Q)[:, 1:χ]), χ, 2, χ))
        end
        @test iTEBD.inner_product(T) ≈ 1.0 atol=1e-8
    end
end

@testset "APPLY_CHAIN_TRANSFER_INPLACE_MATCHES_ALLOCATING" begin
    using Random
    Random.seed!(20260520)
    # Cross-check the in-place workspace path against the allocating reference
    # at unit-cell sizes up to n=8 with non-uniform bond dims.
    for (n, χ_max, d) in [(4, 12, 2), (6, 12, 2), (8, 10, 2), (4, 16, 3)]
        χs = [rand(max(2, χ_max ÷ 2):χ_max) for _ in 1:n]
        χs[end] = χs[1]  # close periodicity on the iMPS-like setting
        T1s = [randn(ComplexF64, χs[mod1(i-1, n)], d, χs[i]) for i in 1:n]
        T2s = [randn(ComplexF64, χs[mod1(i-1, n)], d, χs[i]) for i in 1:n]
        # apply_chain_transfer expects ρ shape matching the rightmost bond
        # pair (χR_T2_n, χR_T1_n) = (χs[n], χs[n]).
        ρ = randn(ComplexF64, χs[n], χs[n])

        ref = iTEBD.apply_chain_transfer(T1s, T2s, ρ; dir=:r)
        ws = iTEBD.ChainTransferWorkspace(ComplexF64, T1s, T2s)
        new = iTEBD.apply_chain_transfer!(ws, T1s, T2s, ρ; dir=:r)
        @test new ≈ ref rtol=1e-10

        # Workspace must produce identical results on repeated calls (catches
        # ping-pong index bugs where the wrong buffer is returned).
        new2 = iTEBD.apply_chain_transfer!(ws, T1s, T2s, ρ; dir=:r)
        @test new2 ≈ ref rtol=1e-10
    end
end

@testset "OCONTRACT_MATCHES_MATERIALIZED_PATH_WITH_LIMITED_ALLOCATIONS" begin
    Γ1 = deterministic_tensor(16, 2, 32)
    Γ2 = deterministic_tensor(32, 2, 16)
    Ts = [Γ1, Γ2]
    Oraw = reshape(ComplexF64.(1:16), 4, 4)
    O = Oraw + Oraw'
    λl = collect(range(0.5, 1.5; length=16))

    function materialized_contract()
        Γ = iTEBD.tensor_group(Ts)
        iTEBD.tensor_lmul!(λl, Γ)
        return dot(Γ, iTEBD.tensor_umul(O, Γ))
    end

    f() = iTEBD.ocontract(Ts, O, λl)

    @test f() ≈ materialized_contract()
    f()
    f()
    @test (@allocated f()) < 45_000
end

@testset "ENTANGLEMENT_ENTROPY_EDGES" begin
    @test entanglement_entropy([1.0, 0.0]) ≈ 0.0 atol=1e-12
    @test entanglement_entropy([0.5, 0.5]) ≈ log(2) atol=1e-12
    @test entanglement_entropy([0.5, 0.5, 1e-12]; cutoff=1e-10) ≈ log(2) atol=1e-12
end

@testset "INNER_PRODUCT_ACCEPTS_KRYLOV_KWARGS" begin
    # The tol/maxiter kwargs were previously not exposed; users couldn't
    # tighten convergence for high-precision overlaps or loosen it for cheap
    # probes. Verify the kwargs are accepted on every public method and that
    # default behavior is unchanged.
    using Random
    Random.seed!(2026_05_25)

    psi = rand_iMPS(ComplexF64, 2, 2, 4)
    canonical!(psi)
    default = iTEBD.inner_product(psi)
    tight   = iTEBD.inner_product(psi; tol=1e-14)
    loose   = iTEBD.inner_product(psi; tol=1e-6, maxiter=20)

    @test default ≈ 1.0 atol=1e-10
    @test tight   ≈ 1.0 atol=1e-10
    @test loose   ≈ 1.0 atol=1e-4

    # Two-arg form on tensor vectors.
    @test iTEBD.inner_product(psi.Γ, psi.Γ; tol=1e-12, maxiter=50) ≈ 1.0 atol=1e-10

    # Single-tensor form.
    T = first(psi.Γ)
    @test iTEBD.inner_product(T; tol=1e-12) ≈ iTEBD.inner_product(T) atol=1e-10
end

@testset "ENT_S_PRODUCT_STATE_IS_ZERO" begin
    # Product states have Schmidt rank 1, so the bipartite entropy across
    # every bond is zero.
    ψ = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
    @test ent_S(ψ, 1) ≈ 0.0 atol=1e-12
    @test ent_S(ψ, 2) ≈ 0.0 atol=1e-12
end

@testset "ENT_S_INDEX_WRAPS_PERIODICALLY" begin
    # ent_S(ψ, i) should treat i modulo ψ.n, so accessing bond 0, -1, or n+1
    # returns the entropy of the corresponding wrapped bond.
    using Random
    Random.seed!(2026_05_25)
    ψ = rand_iMPS(ComplexF64, 4, 2, 4)
    canonical!(ψ)

    @test ent_S(ψ, 0)       ≈ ent_S(ψ, ψ.n)       atol=1e-12
    @test ent_S(ψ, ψ.n + 1) ≈ ent_S(ψ, 1)         atol=1e-12
    @test ent_S(ψ, ψ.n + 2) ≈ ent_S(ψ, 2)         atol=1e-12
    @test ent_S(ψ, -1)      ≈ ent_S(ψ, ψ.n - 1)   atol=1e-12
end

@testset "ENT_S_MAXIMALLY_ENTANGLED_PAIR_IS_LOG2" begin
    # A 1-site unit cell representing the (|00⟩ + |11⟩)/√2 repeating pattern
    # has uniform Schmidt λ = [1/√2, 1/√2] on every bond, so the bipartite
    # entropy is log 2.
    B = zeros(ComplexF64, 2, 2, 2)
    B[1, 1, 1] = inv(sqrt(2))
    B[2, 2, 2] = inv(sqrt(2))
    λ = [inv(sqrt(2)), inv(sqrt(2))]
    ψ = iMPS([B], [λ], 1)
    @test ent_S(ψ, 1) ≈ log(2) atol=1e-12
end

@testset "EXPECT_REJECTS_MISMATCHED_OPERATOR_DIMENSION" begin
    # expect(ψ, O, i, j) requires size(O) == (d^len, d^len) for a contiguous
    # block of length |j - i + 1| (with wraparound). A wrong-sized operator
    # should fail cleanly rather than crash deep inside the contraction.
    ψ = product_iMPS(ComplexF64, [[1, 0], [0, 1]])  # d=2, 2-site cell
    Z = ComplexF64[1 0; 0 -1]
    bad_3site = ComplexF64.(Matrix{Float64}(I, 8, 8))   # 2^3 = 8 but block is 2 sites
    @test_throws Exception iTEBD.expect(ψ, bad_3site, 1, 2)
    # A 1-site operator on a 2-site block also wrong-sized.
    @test_throws Exception iTEBD.expect(ψ, Z, 1, 2)
end

@testset "INNER_PRODUCT_ORTHOGONAL_PRODUCT_STATES" begin
    # Two distinct product states are orthogonal, so the per-cell overlap
    # magnitude collapses to zero. Verifies the cross-transfer-matrix path
    # on a case with known analytic value.
    up   = ComplexF64[1, 0]
    down = ComplexF64[0, 1]
    ψ_up   = product_iMPS(ComplexF64, [up, up])
    ψ_down = product_iMPS(ComplexF64, [down, down])
    @test isapprox(iTEBD.inner_product(ψ_up, ψ_down), 0.0; atol=1e-12)
    @test isapprox(iTEBD.inner_product(ψ_up,   ψ_up),   1.0; atol=1e-12)
end

@testset "ENERGY_SPAN_BRACKETS_HEISENBERG" begin
    # energy_span runs short imaginary-time iTEBD with exp(±dτ h) starting
    # from random unit cells. The two returned energies should bracket the
    # one-site-cell expectation of |↑↓⟩ under spin-1/2 Heisenberg.
    Sx = 0.5 * ComplexF64[0 1; 1 0]
    Sy = 0.5 * ComplexF64[0 -im; im 0]
    Sz = 0.5 * ComplexF64[1 0; 0 -1]
    h = real(kron(Sx, Sx) + kron(Sy, Sy)) + real(kron(Sz, Sz))

    Emin, Emax, (ψ_min, ψ_max) = energy_span(2, 2, h; dτ=0.05, Nτ=200, maxdim=8)
    @test Emin <= Emax
    @test ψ_min.n == 2 && ψ_max.n == 2
    @test size(ψ_min.Γ[1], 2) == 2 && size(ψ_max.Γ[1], 2) == 2
    # Spin-1/2 Heisenberg per-bond ground-state energy is exactly -ln(2) + 1/4
    # ≈ -0.4431; the search should reach close to it under imaginary time.
    @test Emin <= -0.30
    # And the high-energy end should be positive (well above the GS).
    @test Emax >= 0.0
end
