@isdefined(iTEBD) || include("../src/iTEBD.jl")
using .iTEBD: iTEBD, product_iMPS, applygate!, evolve!
using Test
using LinearAlgebra

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
