using Test
using LinearAlgebra
using iTEBD
using iTEBD: product_iMPS, applygate!, evolve!, trotter_gates

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
    layers2 = [[(X, 1, 1)], [(X, 1, 1)]]
    layers3 = [[(X, 1, 1)], [(X, 1, 1)], [(X, 1, 1)]]
    psi = product_iMPS(ComplexF64, [[1, 0]])

    @test_throws ArgumentError trotter_gates(layers3, 0.1; trotter=:fourth_opt)
    @test_throws ArgumentError trotter_gates(layers2, 0.1; trotter=:fourth, evolution=:imaginary)
    @test_throws ArgumentError evolve!(psi, layers3, 0.1, 1; trotter=:fourth_opt)
    @test_throws ArgumentError evolve!(psi, layers2, 0.1, 1; trotter=:fourth_opt, evolution=:imaginary)
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
