using Test
using LinearAlgebra
using Random
using iTEBD

if !isdefined(Main, :TestUtils)
    include(joinpath(@__DIR__, "test_utils.jl"))
end
using .TestUtils: pauli_matrices

# Public-API convergence tests for all three `scarfinder!` interfaces. Each
# test seeds the RNG, runs a small number of iterations, and asserts that the
# returned state is normalized, respects the bond-dimension cap χ, and (where
# applicable) holds its energy near the supplied target.

# -----------------------------------------------------------------------------
# Hamiltonian-based interface: scarfinder!(ψ, h, dt, χ, N)
# -----------------------------------------------------------------------------

@testset "SCARFINDER_HAMILTONIAN_INTERFACE_RUNS_AND_PRESERVES_NORM" begin
    Random.seed!(42)

    P0 = Float64[0 0; 0 1]
    X  = Float64[0 1; 1 0]
    h  = kron(P0, X, P0)  # 3-site PXP density

    psi = product_iMPS(ComplexF64, [[0, 1], [1, 0], [0, 1], [1, 0]])
    χ, N, dt = 2, 3, 0.02

    # No `target` → energy fixing disabled; pure projection-after-evolution loop.
    scarfinder!(psi, h, dt, χ, N; span=3, nstep=3, maxdim=8, refine=false)

    @test maximum(length.(psi.λ)) <= χ
    @test isapprox(inner_product(psi), 1.0; atol=1e-8)
end

@testset "SCARFINDER_HAMILTONIAN_INTERFACE_HOLDS_TARGET_ENERGY" begin
    Random.seed!(43)

    P0 = Float64[0 0; 0 1]
    X  = Float64[0 1; 1 0]
    h  = kron(P0, X, P0)

    psi = product_iMPS(ComplexF64, [[0, 1], [1, 0], [0, 1], [1, 0]])
    target = energy_density(psi, h; span=3)

    χ, N, dt = 2, 3, 0.02
    scarfinder!(psi, h, dt, χ, N;
        span=3, nstep=3, maxdim=8, target=target, refine=false)

    @test maximum(length.(psi.λ)) <= χ
    final_energy = energy_density(psi, h; span=3)
    # Loose bound: energy fixing should keep us within an O(1) fraction of the
    # target for this small system; a regression that breaks the fixing loop
    # entirely would blow this out by orders of magnitude.
    @test abs(final_energy - target) < 0.5
end

# -----------------------------------------------------------------------------
# Gate-based interface (no energy fixing): scarfinder!(ψ, G, χ, N)
# -----------------------------------------------------------------------------

@testset "SCARFINDER_GATE_INTERFACE_RUNS_AND_RESPECTS_CHI" begin
    Random.seed!(44)

    P = pauli_matrices()
    dt = 0.02
    # Two-site gate so that the projection has something to do.
    G = exp(-1im * dt * (kron(P.X, P.X) + 0.3 * kron(P.Z, P.Z)))

    psi = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
    χ, N = 2, 3

    scarfinder!(psi, G, χ, N; span=2, nstep=3, maxdim=4, refine=false)

    @test maximum(length.(psi.λ)) <= χ
    @test isapprox(inner_product(psi), 1.0; atol=1e-8)
end

# -----------------------------------------------------------------------------
# Mixed interface (custom gate G, energy fixed against h):
# scarfinder!(ψ, G, h, χ, N) — the recommended form for constrained models.
# -----------------------------------------------------------------------------

@testset "SCARFINDER_MIXED_INTERFACE_RUNS_ON_PXP" begin
    Random.seed!(45)

    P0 = Float64[0 0; 0 1]
    N1 = Float64[1 0; 0 0]
    X  = Float64[0 1; 1 0]

    h_pxp = kron(P0, X, P0)
    no_double_2 = Matrix{Float64}(I, 4, 4) - kron(N1, N1)

    dt = 0.02
    G = kron(no_double_2, I(2)) * kron(I(2), no_double_2) * exp(-1im * dt * h_pxp)

    psi = product_iMPS(ComplexF64, [[0, 1], [1, 0], [0, 1], [1, 0]])
    target = energy_density(psi, h_pxp; span=3)

    χ, N = 2, 3
    scarfinder!(psi, G, h_pxp, χ, N;
        span=3, hspan=3, nstep=3, target=target, maxdim=12, refine=false)

    @test maximum(length.(psi.λ)) <= χ
    final_energy = energy_density(psi, h_pxp; span=3)
    @test abs(final_energy - target) < 0.5
end

@testset "SCARFINDER_MIXED_INTERFACE_REFINEMENT_KEEPS_LOWEST_ENTROPY" begin
    # With refine=true the routine scans a short trajectory and returns the
    # minimum-entanglement point along it. Verify the returned state is at
    # least as good (lower or equal entropy on the scanned bond) as the
    # post-iteration state would be — i.e. refinement is monotone.
    Random.seed!(46)

    P0 = Float64[0 0; 0 1]
    N1 = Float64[1 0; 0 0]
    X  = Float64[0 1; 1 0]

    h_pxp = kron(P0, X, P0)
    no_double_2 = Matrix{Float64}(I, 4, 4) - kron(N1, N1)

    dt = 0.02
    G = kron(no_double_2, I(2)) * kron(I(2), no_double_2) * exp(-1im * dt * h_pxp)

    psi_no_refine = product_iMPS(ComplexF64, [[0, 1], [1, 0], [0, 1], [1, 0]])
    psi_refine    = deepcopy(psi_no_refine)
    target = energy_density(psi_no_refine, h_pxp; span=3)

    χ, N = 2, 3
    common_kwargs = (span=3, hspan=3, nstep=3, target=target, maxdim=12)

    scarfinder!(psi_no_refine, G, h_pxp, χ, N; refine=false,                            common_kwargs...)
    scarfinder!(psi_refine,    G, h_pxp, χ, N; refine=true, refine_step=20,             common_kwargs...)

    # The refinement scan's selection criterion is `ent_S(x, x.n)`, so the
    # returned state should have entropy ≤ the non-refined state on that bond.
    @test ent_S(psi_refine, psi_refine.n) <= ent_S(psi_no_refine, psi_no_refine.n) + 1e-10
end
