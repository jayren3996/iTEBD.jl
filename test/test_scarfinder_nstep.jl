@isdefined(iTEBD) || include("../src/iTEBD.jl")
using .iTEBD: iTEBD, product_iMPS, scarfinder_step!
using LinearAlgebra
using Test

@testset "SCARFINDER_NSTEP" begin
    psi_h = product_iMPS(ComplexF64, [[1, 0], [1, 0]])
    h_gate = Matrix{ComplexF64}(I, 2, 2)
    @test_logs (:warn, r"nstep = 1") scarfinder_step!(psi_h, h_gate, 0.1, 1; nstep=1, maxdim=1)

    psi_gate = product_iMPS(ComplexF64, [[1, 0], [1, 0]])
    G1 = Matrix{ComplexF64}(I, 2, 2)

    @test_logs (:warn, r"nstep = 1") scarfinder_step!(psi_gate, G1, 1; nstep=1, maxdim=1)

    psi_mixed = product_iMPS(ComplexF64, [[1, 0], [1, 0]])
    h1 = Matrix{ComplexF64}(I, 2, 2)

    @test_logs (:warn, r"nstep = 1") scarfinder_step!(psi_mixed, G1, h1, 1; nstep=1, maxdim=1)

    psi_default = product_iMPS(ComplexF64, [[1, 0], [1, 0]])
    @test_logs scarfinder_step!(psi_default, G1, 1; maxdim=1)
end
