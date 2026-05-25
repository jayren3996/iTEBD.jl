using Test
using LinearAlgebra
using Random
using iTEBD

Random.seed!(20260520)

if !isdefined(Main, :TestUtils)
    include(joinpath(@__DIR__, "test_utils.jl"))
end
using .TestUtils: AKLT_TENSOR

const AKLT_STEPS = parse(Int, get(ENV, "ITEBD_AKLT_STEPS", "200"))

function aklt_gate(dt)
    S1X = sqrt(2) / 2 * [0 1 0; 1 0 1; 0 1 0]
    S1Y = sqrt(2) / 2 * 1im * [0 -1 0; 1 0 -1; 0 1 0]
    S1Z = [1 0 0; 0 0 0; 0 0 -1]
    ss = kron(S1X, S1X) + kron(S1Y, S1Y) + kron(S1Z, S1Z)
    H = ss + ss^2 / 3 + 2I / 3
    return exp(-dt * H)
end

@testset "AKLT_TWO_SITE_IMAGINARY_TIME_CONVERGENCE" begin
    rdim = 16
    G = aklt_gate(0.1)
    aklt = iMPS([AKLT_TENSOR, AKLT_TENSOR])
    psi = rand_iMPS(ComplexF64, 2, 3, 1)

    evolve!(psi, [(G, 1, 2), (G, 2, 1)], AKLT_STEPS; maxdim=rdim, truncerr=1e-10)
    canonical!(psi; maxdim=rdim)

    @test iTEBD.inner_product(psi, aklt) > 1 - 1e-6
    @test all(size(Γ, 2) == 3 for Γ in psi.Γ)
    @test all(1 <= length(λ) <= rdim for λ in psi.λ)
    @test all(isapprox(norm(λ), 1.0; atol=1e-8) for λ in psi.λ)
end

@testset "AKLT_THREE_SITE_IMAGINARY_TIME_CONVERGENCE" begin
    rdim = 16
    G = aklt_gate(0.1)
    aklt = iMPS([AKLT_TENSOR, AKLT_TENSOR, AKLT_TENSOR])
    psi = rand_iMPS(ComplexF64, 3, 3, 1)

    evolve!(psi, [(G, 1, 2), (G, 2, 3), (G, 3, 1)], AKLT_STEPS; maxdim=rdim, truncerr=1e-10)
    canonical!(psi; maxdim=rdim)

    @test iTEBD.inner_product(psi, aklt) > 1 - 1e-6
    @test all(size(Γ, 2) == 3 for Γ in psi.Γ)
    @test all(1 <= length(λ) <= rdim for λ in psi.λ)
    @test all(isapprox(norm(λ), 1.0; atol=1e-8) for λ in psi.λ)
end
