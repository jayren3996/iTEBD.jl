@isdefined(iTEBD) || include("../src/iTEBD.jl")
using .iTEBD: iTEBD, product_iMPS, applygate!, evolve!, trotter_gates
using Test
using LinearAlgebra

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

@testset "EVOLVE_ADAPTIVE_POLICY" begin
    H = 1 / sqrt(2) * [1.0 1.0; 1.0 -1.0]
    CNOT = [1.0 0 0 0;
            0 1 0 0;
            0 0 0 1;
            0 0 1 0]
    bell_gate = CNOT * kron(H, I(2))

    psi = product_iMPS(ComplexF64, [[1, 0], [1, 0]])
    @test maximum(length.(psi.λ)) == 1

    evolve!(psi, [(bell_gate, 1, 2)], 1; chi_policy=:adaptive, maxdim=4)

    @test maximum(length.(psi.λ)) == 2
end

@testset "EVOLVE_ADAPTIVE_AKLT_STABILITY" begin
    X = sqrt(2) / 2 * [0 1 0; 1 0 1; 0 1 0]
    Y = sqrt(2) / 2 * 1im * [0 -1 0; 1 0 -1; 0 1 0]
    Z = [1 0 0; 0 0 0; 0 0 -1]

    H = begin
        SS = kron(X, X) + kron(Y, Y) + kron(Z, Z)
        0.5 * SS + SS^2 / 6 + I / 3
    end
    G = exp(-0.1 * H)
    gates = [(G, 1, 2), (G, 2, 1)]

    aklt_tensor = zeros(2, 3, 2)
    aklt_tensor[1, 1, 2] = +sqrt(2 / 3)
    aklt_tensor[1, 2, 1] = -sqrt(1 / 3)
    aklt_tensor[2, 2, 2] = +sqrt(1 / 3)
    aklt_tensor[2, 3, 1] = -sqrt(2 / 3)
    aklt = iTEBD.iMPS([aklt_tensor, aklt_tensor])

    psi = iTEBD.rand_iMPS(ComplexF64, 2, 3, 1)
    evolve!(psi, gates, 200; chi_policy=:adaptive, maxdim=8)

    @test all(1 <= length(λ) <= 8 for λ in psi.λ)
    @test all(isapprox(norm(λ), 1.0; atol=1e-8) for λ in psi.λ)
    @test iTEBD.inner_product(psi, aklt) > 0.999
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
