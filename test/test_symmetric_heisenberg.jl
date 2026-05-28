using Test
using iTEBD
using TensorKit
using LinearAlgebra
using Random

@testset "Heisenberg XXZ symmetric iTEBD" begin
    Random.seed!(2)
    P = graded_space(:U1, 1=>1, -1=>1)
    Sz, SzSz, SpSm, SmSp = spin_half_ops(:U1)

    # Heisenberg density h = Sz⊗Sz + 0.5*(S+⊗S- + S-⊗S+)
    h = SzSz + 0.5 * (SpSm + SmSp)

    # Start in the Néel state (Sz=0 sector)
    ψ = product_iMPS(:U1, [-1, 1], [1, -1])
    dt = 0.05
    gates = [(exp(-dt * h), 1, 2), (exp(-dt * h), 2, 1)]
    evolve!(ψ, gates, 400; maxdim=32, cutoff=1e-10, recanonicalize=true)

    e = energy_density(ψ, h)
    # Bethe-ansatz ground-state energy density of the spin-1/2 Heisenberg chain
    # is e_∞ = 1/4 - ln(2) ≈ -0.4431.
    @test isapprox(e, 1/4 - log(2); atol=1e-3)
end

@testset "Dense and symmetric agree at χ=8" begin
    Random.seed!(2)

    # Dense path: use spin_half_ops(:Trivial) to build the dense Heisenberg density
    Sx, Sy, Sz_d, Sp_d, Sm_d, _ = spin_half_ops(:Trivial)
    SzSz_d = kron(Sz_d, Sz_d)
    XY_d   = 0.5 * (kron(Sp_d, Sm_d) + kron(Sm_d, Sp_d))
    h_d    = SzSz_d + XY_d
    ψd = product_iMPS(ComplexF64, [[0+0im, 1+0im], [1+0im, 0+0im]])
    dt = 0.05
    gates_d = [(exp(-dt * h_d), 1, 2), (exp(-dt * h_d), 2, 1)]
    evolve!(ψd, gates_d, 400; maxdim=8, cutoff=1e-10, recanonicalize=true)
    e_dense = energy_density(ψd, h_d)

    # Symmetric path: same Hamiltonian via spin_half_ops(:U1)
    Sz, SzSz, SpSm, SmSp = spin_half_ops(:U1)
    h_s = SzSz + 0.5*(SpSm + SmSp)
    ψs = product_iMPS(:U1, [-1, 1], [1, -1])
    gates_s = [(exp(-dt * h_s), 1, 2), (exp(-dt * h_s), 2, 1)]
    evolve!(ψs, gates_s, 400; maxdim=8, cutoff=1e-10, recanonicalize=true)
    e_sym = energy_density(ψs, h_s)

    @test isapprox(real(e_dense), real(e_sym); atol=1e-4)
end
