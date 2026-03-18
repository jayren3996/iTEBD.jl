# ScarFinder

`iTEBD.jl` includes a general ScarFinder workflow for searching for
low-entanglement, weakly thermalizing trajectories directly in the iMPS
manifold.

## Idea

One ScarFinder iteration in this package has three conceptual parts:

1. Evolve the current iMPS for a short real-time interval inside a larger
   temporary bond dimension.
2. Project back to the chosen variational manifold by truncating to the target
   bond dimension `χ`.
3. If a target energy density is supplied, apply a small imaginary-time
   correction so the projected state stays near that target energy rather than
   drifting toward a trivial low-energy state.

The high-level `scarfinder!` routines simply repeat this iteration `N` times.
By default they then perform a short refinement scan along the optimized
trajectory and keep the minimum-entanglement point.

## Public Interfaces

There are three public ScarFinder interfaces:

- `scarfinder!(ψ, h, dt, χ, N; ...)`
  Hamiltonian-based interface. The local Hamiltonian density `h` is exponentiated
  internally as `exp(-1im * dt * h)`.
- `scarfinder!(ψ, G, χ, N; ...)`
  Gate-based interface. Use this when the evolution rule is already given as a
  local gate `G` and no energy fixing is needed.
- `scarfinder!(ψ, G, h, χ, N; ...)`
  Mixed `gate + Hamiltonian` interface. Use this when the real-time update is a
  custom gate `G`, but the energy correction should still be measured with
  respect to a Hamiltonian density `h`.

The mixed interface is usually the right choice for constrained models such as
PXP, where the gate used during ScarFinder may contain projectors that restore
the physical subspace after truncation, while the target energy should still be
computed from the unprojected Hamiltonian density.

## Positional Arguments

The meaning of the positional arguments is:

- `ψ`
  The trial `iMPS` state, updated in place.
- `h`
  A local Hamiltonian density acting on a contiguous window of the unit cell.
  Its support can be inferred automatically with `operator_span(ψ, h)`.
- `G`
  A local gate acting on a contiguous window of the unit cell. This is used in
  the gate-based and mixed interfaces.
- `dt`
  Microscopic real-time step used to build `exp(-1im * dt * h)` in the
  Hamiltonian-based interface.
- `χ`
  Target bond dimension after the projection step. This is the bond dimension of
  the variational manifold that ScarFinder repeatedly projects back to.
- `N`
  Number of ScarFinder iterations.

## Keyword Arguments

All interfaces support the refinement controls:

- `refine=true`
  After the main `N` iterations, scan a short trajectory and keep the
  minimum-entanglement point.
- `refine_step=1000`
  Number of trial points used in that scan.

The Hamiltonian-based interface also supports:

- `refine_dt=dt / 10`
  Microscopic time step used during the refinement scan.

The evolution and projection controls are:

- `nstep`
  Number of local evolution steps before each projection. For
  `scarfinder!(ψ, h, dt, χ, N; ...)`, one ScarFinder iteration represents a
  physical interval `Δt ≈ nstep * dt`.
- `maxdim=MAXDIM`
  Temporary bond dimension allowed during the real-time evolution stage, before
  the projection back to `χ`.
- `cutoff=SVDTOL`
  Truncation cutoff used during the projection step.

The support controls are:

- `span=operator_span(ψ, G_or_h)`
  Number of sites acted on by the real-time evolution object. In the
  Hamiltonian-based interface this refers to `h`; in the gate-based and mixed
  interfaces it refers to `G`.
- `hspan=operator_span(ψ, h)`
  Number of sites acted on by the Hamiltonian density used for energy fixing in
  the mixed interface.

The energy-fixing controls are:

- `target=nothing`
  Target energy density. If left as `nothing`, no energy correction is applied.
- `tol=1e-6`
  Stop the energy-fixing loop once the energy density is within `tol` of the
  target.
- `α=0.1`
  Step-size parameter for the imaginary-time energy correction.
- `maxstep=50`
  Maximum number of energy-fixing substeps after each projection.

## How The Arguments Fit Together

The most important practical distinction is between the two time scales:

- `dt`
  Microscopic gate time used to construct a local evolution operator.
- `nstep`
  Number of microscopic applications inside one ScarFinder iteration.

If you want one ScarFinder iteration to represent a larger physical interval
`Δt`, choose `nstep ≈ Δt / dt`. For example, if your microscopic gate uses
`dt = 0.01` but you want each ScarFinder iteration to represent `Δt = 0.05`,
set `nstep = 5`.

The two bond dimensions also play different roles:

- `maxdim`
  Temporary working bond dimension during real-time evolution.
- `χ`
  Target bond dimension of the projected manifold.

Typically `maxdim >= χ`, often strictly larger.

## Helper Functions

The following helpers are useful when setting up ScarFinder runs:

- `operator_span(ψ, O)`
  Infer how many sites a local operator `O` acts on from the local Hilbert-space
  dimension of `ψ`.
- `energy_density(ψ, h; span=...)`
  Compute the unit-cell averaged expectation value of `h`.
- `energy_span(n, d, h; dτ=0.1, Nτ=1000, maxdim=32)`
  Estimate the low- and high-energy densities reachable by imaginary-time iTEBD.
  This is convenient when choosing a `target` energy for ScarFinder.

## Complete PXP Example

The example below uses the mixed interface
`scarfinder!(ψ, G, h, χ, N; ...)`, which is the recommended form for the PXP
model.

```@example
using iTEBD
using LinearAlgebra

P0 = [0 0; 0 1]
N1 = [1 0; 0 0]
X = [0 1; 1 0]

# Three-site PXP Hamiltonian density.
h_pxp = kron(P0, X, P0)

# Two-site blockade projector used to repair truncation artifacts.
no_double_2 = Matrix{Float64}(I, 4, 4) - kron(N1, N1)

# Gate used during ScarFinder evolution.
dt = 0.01
G = kron(no_double_2, I(2)) * kron(I(2), no_double_2) * exp(-1im * dt * h_pxp)

# Four-site Z2 product state.
psi = product_iMPS(ComplexF64, [[0, 1], [1, 0], [0, 1], [1, 0]])

# Match the energy density of the initial Z2 state.
target = energy_density(psi, h_pxp; span=3)

# Here one ScarFinder step represents roughly Δt = 0.05.
scarfinder!(psi, G, h_pxp, 2, 5;
    span=3,
    hspan=3,
    nstep=5,
    target=target,
    maxdim=12,
    refine=false,
)

(;
    target_energy=target,
    final_energy=energy_density(psi, h_pxp; span=3),
    maxbond=maximum(length.(psi.λ)),
)
```

## Choosing An Interface

- Use `scarfinder!(ψ, h, dt, χ, N; ...)` when the local evolution rule is just
  the Hamiltonian density `h`.
- Use `scarfinder!(ψ, G, χ, N; ...)` when you already have a local gate and do
  not want any energy correction.
- Use `scarfinder!(ψ, G, h, χ, N; ...)` when you evolve with a custom gate `G`
  but still want the truncation-induced energy drift corrected with respect to
  `h`.

## Reference

The implementation here follows the algorithmic idea introduced in:

- J. Ren, A. Hallam, L. Ying, and Z. Papić, [ScarFinder: A Detector of Optimal
  Scar Trajectories in Quantum Many-Body Dynamics](https://eprints.whiterose.ac.uk/id/eprint/238368/),
  PRX Quantum 6, 040332 (2025).

The generated function reference lives on [API Reference](api.md).
