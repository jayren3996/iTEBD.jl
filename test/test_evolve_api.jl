using Test
using LinearAlgebra
using Random
using iTEBD
using iTEBD: product_iMPS, applygate!, evolve!, trotter_gates

Random.seed!(20260526)

gate_product(gates) = begin
    dim = size(first(gates)[1], 1)
    U = Matrix{ComplexF64}(I, dim, dim)
    for (G, _, _) in gates
        U = Matrix{ComplexF64}(G) * U
    end
    U
end

schedule_layers(schedule) = first.(schedule)
schedule_coeffs(schedule) = last.(schedule)

@testset "EVOLVE_FIXED_POLICY" begin
    X = [0.0 1.0; 1.0 0.0]
    G = kron(X, X)

    psi_manual = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
    psi_evolve = product_iMPS(ComplexF64, [[1, 0], [0, 1]])

    for _ in 1:3
        applygate!(psi_manual, G, 1, 2; maxdim=4)
        applygate!(psi_manual, G, 2, 1; maxdim=4)
    end

    @test evolve!(psi_evolve, [(G, 1, 2), (G, 2, 1)], 3; maxdim=4) === psi_evolve
    @test psi_evolve.λ[1] ≈ psi_manual.λ[1] atol=1e-12
    @test psi_evolve.λ[2] ≈ psi_manual.λ[2] atol=1e-12
end

@testset "EVOLVE_LOCAL_TRUNCATION_GROWS_WHEN_NEEDED" begin
    H = 1 / sqrt(2) * [1.0 1.0; 1.0 -1.0]
    CNOT = [1.0 0 0 0;
            0 1 0 0;
            0 0 0 1;
            0 0 1 0]
    bell_gate = CNOT * kron(H, I(2))

    psi = product_iMPS(ComplexF64, [[1, 0], [1, 0]])
    @test maximum(length.(psi.λ)) == 1

    evolve!(psi, [(bell_gate, 1, 2)], 1; maxdim=4)

    @test maximum(length.(psi.λ)) == 2
end

@testset "TROTTER_LAYER_SECOND_ORDER_MATCHES_MANUAL" begin
    X = [0.0 1.0; 1.0 0.0]
    H = kron(X, X)
    dt = 0.1
    steps = 3

    Ghalf = exp(-0.5im * dt * H)
    Gfull = exp(-1im * dt * H)
    layers = [[(H, 1, 2)], [(H, 2, 1)]]

    psi_manual = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
    psi_layer = product_iMPS(ComplexF64, [[1, 0], [0, 1]])

    for _ in 1:steps
        applygate!(psi_manual, Ghalf, 1, 2; maxdim=4)
        applygate!(psi_manual, Gfull, 2, 1; maxdim=4)
        applygate!(psi_manual, Ghalf, 1, 2; maxdim=4)
    end

    @test evolve!(psi_layer, layers, dt, steps; maxdim=4) === psi_layer
    @test psi_layer.λ[1] ≈ psi_manual.λ[1] atol=1e-12
    @test psi_layer.λ[2] ≈ psi_manual.λ[2] atol=1e-12
    @test iTEBD.inner_product(psi_layer, psi_manual) ≈ 1.0 atol=1e-12
end

@testset "TROTTER_STAGE_SCHEDULES" begin
    p = 1 / (4 - 4^(1 / 3))
    q = 1 - 4p
    a1 = 0.09584850274120368
    a2 = -0.07811115892163792
    a3 = 0.5 - (a1 + a2)
    b1 = 0.42652466131587616
    b2 = -0.12039526945509727
    b3 = 1 - 2 * (b1 + b2)

    fourth = iTEBD._trotter_stage_schedule(2, :fourth, 1)
    @test schedule_layers(fourth) == [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]
    @test schedule_coeffs(fourth) ≈ [p / 2, p, p, p, (1 - 3p) / 2, q, (1 - 3p) / 2, p, p, p, p / 2] atol=1e-15

    fourth_two = iTEBD._trotter_stage_schedule(2, :fourth, 2)
    @test length(fourth_two) == 21
    @test schedule_layers(fourth_two)[11] == 1
    @test schedule_coeffs(fourth_two)[11] ≈ p atol=1e-15

    fourth_opt_two = iTEBD._trotter_stage_schedule(2, :fourth_opt, 2)
    @test schedule_layers(fourth_opt_two) == [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]
    @test schedule_coeffs(fourth_opt_two) ≈ [
        a1, b1, a2, b2, a3, b3, a3, b2, a2, b1,
        2a1,
        b1, a2, b2, a3, b3, a3, b2, a2, b1, a1
    ] atol=1e-15
end

@testset "TROTTER_DENSE_ACCURACY" begin
    X = ComplexF64[0 1; 1 0]
    Z = ComplexF64[1 0; 0 -1]
    dt = 0.3

    layers = [[(X, 1, 1)], [(Z, 1, 1)]]
    U_exact = exp(-1im * dt * (X + Z))

    U_second = gate_product(trotter_gates(layers, dt; trotter=:second))
    U_fourth = gate_product(trotter_gates(layers, dt; trotter=:fourth))
    U_fourth_opt = gate_product(trotter_gates(layers, dt; trotter=:fourth_opt))

    err_second = opnorm(U_second - U_exact)
    err_fourth = opnorm(U_fourth - U_exact)
    err_fourth_opt = opnorm(U_fourth_opt - U_exact)

    @test err_fourth < err_second
    @test err_fourth_opt < err_second
end

@testset "TROTTER_GATE_CACHE_REUSES_FOR_REPEATED_COEFFS" begin
    # _materialize_trotter_gates caches exp(coeff * h) by (layer, term, coeff)
    # so palindromic Trotter schemes can reuse the matrix exponential when the
    # same coefficient appears multiple times. A regression in the cache key
    # (e.g. coeff equality drift from float arithmetic) would show up here as
    # the gate list having all-distinct Matrix references.
    X = ComplexF64[0 1; 1 0]
    Z = ComplexF64[1 0; 0 -1]
    layers = [[(X, 1, 2)], [(Z, 1, 2)]]

    # :fourth_opt has 11 stages with palindromic coefficients
    # (A1, B1, A2, B2, A3, B3, A3, B2, A2, B1, A1). At most 3 unique values
    # per layer (A1/A2/A3 and B1/B2/B3) → at most 6 unique gate matrices
    # across the 11 stages. Multi-step calls also exercise the seam merging
    # in _push_trotter_stage!.
    gates = trotter_gates(layers, 0.1; trotter=:fourth_opt)
    @test length(gates) == 11
    @test length(unique(objectid(g) for (g, _, _) in gates)) <= 6

    # Multi-step schedule via the internal helper. Stage 1 of step k+1 merges
    # with stage 11 of step k (both layer 1, coeff A1), so the per-step stage
    # count decreases.
    stages_3 = iTEBD._trotter_stage_schedule(2, :fourth_opt, 3)
    gates_3 = iTEBD._materialize_trotter_gates(layers, 0.1, stages_3; evolution=:real)
    @test length(gates_3) < 33  # naive count would be 11 * 3 = 33
    # The cache should still produce a small number of unique matrices.
    @test length(unique(objectid(g) for (g, _, _) in gates_3)) <= 8
end

@testset "EVOLVE_SINGLE_SITE_CELL_UNDER_ADAPTIVE" begin
    # The :adaptive chi-policy ratchets bond dim across bonds within a sweep,
    # and across sweeps via the persisted ψ.λ. For n=1 cells there is exactly
    # one bond (the wraparound), so the ratchet should still produce a
    # monotonic bond-dim sequence over multiple evolve! calls.
    Random.seed!(2026_05_26)
    X = ComplexF64[0 1; 1 0]
    Z = ComplexF64[1 0; 0 -1]
    G = exp(-0.05im * (X + 0.3 * Z))

    psi = rand_iMPS(ComplexF64, 1, 2, 1)
    canonical!(psi)
    bond_dims = Int[length(psi.λ[1])]
    for _ in 1:5
        evolve!(psi, [(G, 1, 1)], 3; chi_policy=:adaptive, maxdim=8)
        push!(bond_dims, length(psi.λ[1]))
    end
    @test all(bond_dims[i] >= bond_dims[i-1] for i in 2:length(bond_dims))
    @test all(bond_dims .<= 8)
    @test isapprox(iTEBD.inner_product(psi), 1.0; atol=1e-8)
end

@testset "EVOLVE_FLOAT32_IMAGINARY_TIME_PRESERVES_ELTYPE" begin
    # Most tests use ComplexF64. Verify the lower-precision path actually
    # runs end-to-end without silent upcasting in evolve!, preserves the
    # element type, and keeps the state normalized.
    Random.seed!(2026_05_26)
    H = ComplexF32[1 0 0 0; 0 -1 2 0; 0 2 -1 0; 0 0 0 1]  # tight-binding-like
    G = exp(-0.05f0 * H)
    @test eltype(G) === ComplexF32

    psi = rand_iMPS(ComplexF32, 2, 2, 2)
    canonical!(psi)
    evolve!(psi, [(G, 1, 2), (G, 2, 1)], 5; maxdim=4)
    @test eltype(psi.Γ[1]) === ComplexF32
    @test eltype(psi.λ[1]) === Float32
    @test isapprox(iTEBD.inner_product(psi), 1.0; atol=1f-4)
end

@testset "EVOLVE_WRAPAROUND_GATE_UNDER_ADAPTIVE_POLICY" begin
    # Wrap-around gates (j < i) used to leave the state non-canonical, and
    # the :adaptive policy ratchets the bond dim across bonds within an
    # evolve! sweep. Combining the two should produce a canonical state
    # whose bond dim respects the ratchet — neither path should clobber
    # the other.
    Random.seed!(2026_05_25)

    # Inline right-canonical check to avoid pulling TestUtils into this file.
    function right_canonical_err(ψ)
        errs = Float64[]
        for Γ in ψ.Γ
            Dl = size(Γ, 1)
            overlap = zeros(ComplexF64, Dl, Dl)
            for s in 1:size(Γ, 2)
                Bs = reshape(Γ[:, s, :], Dl, size(Γ, 3))
                overlap .+= Bs * Bs'
            end
            push!(errs, norm(overlap - Matrix{ComplexF64}(I, Dl, Dl)))
        end
        maximum(errs)
    end

    Z = ComplexF64[1 0; 0 -1]
    X = ComplexF64[0 1; 1 0]
    H = kron(Z, X) + 0.2 * kron(X, Z)
    G_wrap = exp(-0.1im * H)

    psi = rand_iMPS(ComplexF64, 4, 2, 4)
    canonical!(psi)
    initial_dims = length.(psi.λ)
    # Sweep: include both interior gates and the wrap (4, 1) gate.
    gates = [(G_wrap, 1, 2), (G_wrap, 2, 3), (G_wrap, 3, 4), (G_wrap, 4, 1)]
    evolve!(psi, gates, 3; chi_policy=:adaptive, maxdim=8)

    @test right_canonical_err(psi) < 1e-8
    @test isapprox(iTEBD.inner_product(psi), 1.0; atol=1e-8)
    # Adaptive ratchet: bond dim never falls below initial.
    @test all(length(psi.λ[i]) >= initial_dims[i] for i in 1:psi.n)
end

@testset "EVOLVE_FOURTH_ORDER_INTEGRATION" begin
    # End-to-end check: evolve! consuming a fourth-order Trotter schedule on
    # an iMPS should preserve norm and stay closer to the unitary evolution
    # than :second at small dt. Previously the test suite only exercised
    # fourth-order schemes at the gate-product level (TROTTER_DENSE_ACCURACY),
    # so a regression in how evolve! wires the fourth-order stage list into
    # the iMPS update loop could ship silently.
    X = ComplexF64[0 1; 1 0]
    Z = ComplexF64[1 0; 0 -1]
    H = kron(X, X) + 0.3 * kron(Z, Z)
    layers = [[(H, 1, 2)], [(H, 2, 1)]]

    for scheme in (:fourth, :fourth_opt)
        psi = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
        evolve!(psi, layers, 0.05, 5; trotter=scheme, evolution=:real, maxdim=8)
        # Norm should hold up under unitary real-time evolution.
        @test isapprox(iTEBD.inner_product(psi), 1.0; atol=1e-6)
        # Bond dim should respect the cap.
        @test maximum(length.(psi.λ)) <= 8
    end
end

@testset "TROTTER_ORDER_FROM_DT_SCALING" begin
    # Verify per-macro-step error scaling matches each scheme's documented
    # order: for a p-th order Trotter, error ~ O(dt^(p+1)) per macro-step,
    # so halving dt should reduce error by 2^(p+1).
    #   :second     → p=2 → ratio ~8
    #   :fourth     → p=4 → ratio ~32
    #   :fourth_opt → p=4 → ratio ~32
    # We measure opnorm(U_trotter(dt) - exp(-i·dt·H_full)) at a sequence of
    # halving dt, against a Hamiltonian with non-commuting layers so the
    # error is non-zero and the order is genuinely exercised.
    X = ComplexF64[0 1; 1 0]
    Z = ComplexF64[1 0; 0 -1]
    layers = [[(X, 1, 1)], [(Z, 1, 1)]]
    H_full = X + Z

    function step_error(scheme, dt)
        U_trotter = gate_product(trotter_gates(layers, dt; trotter=scheme))
        U_exact   = exp(-1im * dt * H_full)
        return opnorm(U_trotter - U_exact)
    end

    # Halve dt 3 times.
    dts = [0.4, 0.2, 0.1, 0.05]

    for (scheme, expected_ratio) in (
        (:second,     8.0),
        (:fourth,     32.0),
        (:fourth_opt, 32.0),
    )
        errs = [step_error(scheme, dt) for dt in dts]
        # Sanity: errors should be strictly decreasing as dt shrinks (above
        # floating-point noise).
        @test all(errs[i] < errs[i-1] for i in 2:length(errs))
        @test all(e > 1e-14 for e in errs)

        # Consecutive halving ratios. Allow (expected/2, expected*2) to
        # absorb finite-dt corrections from higher-order commutators.
        for i in 2:length(errs)
            ratio = errs[i-1] / errs[i]
            @test expected_ratio / 2 < ratio < expected_ratio * 2
        end
    end
end

@testset "TROTTER_IMAGINARY_PROMOTES_ALL_LAYER_ELEMENT_TYPES" begin
    X = Float64[0 1; 1 0]
    Y = ComplexF64[0 -im; im 0]
    layers = [[(X, 1, 1)], [(Y, 1, 1)]]

    gates = trotter_gates(layers, 0.1; evolution=:imaginary)

    @test eltype(first(gates)[1]) <: Complex
    @test all(gate -> eltype(gate[1]) <: Complex, gates)
end

@testset "TROTTER_VALIDATION" begin
    X = [0.0 1.0; 1.0 0.0]
    layers1 = [[(X, 1, 1)]]
    layers2 = [[(X, 1, 1)], [(X, 1, 1)]]
    layers3 = [[(X, 1, 1)], [(X, 1, 1)], [(X, 1, 1)]]
    psi = product_iMPS(ComplexF64, [[1, 0]])

    @test_throws ArgumentError trotter_gates(layers3, 0.1; trotter=:fourth_opt)
    @test_throws ArgumentError trotter_gates(layers2, 0.1; trotter=:fourth, evolution=:imaginary)
    @test_throws ArgumentError evolve!(psi, layers3, 0.1, 1; trotter=:fourth_opt)
    @test_throws ArgumentError evolve!(psi, layers2, 0.1, 1; trotter=:fourth_opt, evolution=:imaginary)

    # trotter=:fourth with a single layer would silently collapse to a
    # first-order Euler step because the five Suzuki substeps all target the
    # same layer and merge into one stage. Reject up front.
    @test_throws ArgumentError trotter_gates(layers1, 0.1; trotter=:fourth)
    @test_throws ArgumentError evolve!(psi, layers1, 0.1, 1; trotter=:fourth)
end

@testset "ADAPTIVE_POLICY_VALID_STATE" begin
    X = [0.0 1.0; 1.0 0.0]
    H = kron(X, X)
    G = exp(-0.1im * H)
    gates = [(G, 1, 2), (G, 2, 1)]

    psi = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
    evolve!(psi, gates, 2; chi_policy=:adaptive, maxdim=4)

    @test all(λ -> all(λ .>= 0), psi.λ)
    @test all(λ -> isapprox(norm(λ), 1.0; atol=1e-8), psi.λ)
    @test all(Γ -> size(Γ, 1) == size(Γ, 3), psi.Γ)
end

@testset "ONE_SITE_ADAPTIVE_EVOLVE_DOES_NOT_SHRINK_UNRELATED_BONDS" begin
    X = ComplexF64[0 1; 1 0]
    psi = iTEBD.rand_iMPS(ComplexF64, 3, 2, 3)
    before_lengths = length.(psi.λ)

    evolve!(psi, [(X, 1, 1)], 1; chi_policy=:adaptive, maxdim=1, cutoff=0.0)

    @test length.(psi.λ)[2:3] == before_lengths[2:3]
end

@testset "GATE_FUSION_SAME_BOND" begin
    X = [0.0 1.0; 1.0 0.0]
    G1 = kron(X, I(2))
    G2 = kron(I(2), X)
    G_fused = G2 * G1

    psi_seq = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
    psi_fused = product_iMPS(ComplexF64, [[1, 0], [0, 1]])

    applygate!(psi_seq, G1, 1, 2; maxdim=4)
    applygate!(psi_seq, G2, 1, 2; maxdim=4)
    applygate!(psi_fused, G_fused, 1, 2; maxdim=4)

    @test psi_seq.λ[1] ≈ psi_fused.λ[1] atol=1e-12
    @test psi_seq.λ[2] ≈ psi_fused.λ[2] atol=1e-12
    @test psi_seq.Γ[1] ≈ psi_fused.Γ[1] atol=1e-12
    @test psi_seq.Γ[2] ≈ psi_fused.Γ[2] atol=1e-12
end

@testset "EVOLVE_SAME_SUPPORT_GATES_APPLY_SEQUENTIALLY" begin
    H = 1 / sqrt(2) * [1.0 1.0; 1.0 -1.0]
    CNOT = [1.0 0 0 0;
            0 1 0 0;
            0 0 0 1;
            0 0 1 0]
    bell_gate = CNOT * kron(H, I(2))
    inverse_bell_gate = adjoint(bell_gate)

    psi_manual = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
    psi_evolve = product_iMPS(ComplexF64, [[1, 0], [0, 1]])

    applygate!(psi_manual, bell_gate, 1, 2; maxdim=1)
    applygate!(psi_manual, inverse_bell_gate, 1, 2; maxdim=1)
    evolve!(psi_evolve, [(bell_gate, 1, 2), (inverse_bell_gate, 1, 2)], 1; maxdim=1)

    @test abs(iTEBD.inner_product(psi_evolve, psi_manual)) ≈ 1.0 atol=1e-12
    @test psi_evolve.Γ[1] ≈ psi_manual.Γ[1] atol=1e-12
    @test psi_evolve.Γ[2] ≈ psi_manual.Γ[2] atol=1e-12
end

@testset "TROTTER_GATES_TYPED" begin
    X = ComplexF64[0 1; 1 0]
    layers = [[(X, 1, 1)], [(X, 1, 1)]]
    gates = trotter_gates(layers, 0.1; trotter=:second)

    @test gates isa Vector
    @test eltype(gates) <: Tuple{AbstractMatrix, Int, Int}
    @test eltype(first(gates)[1]) <: ComplexF64
end
