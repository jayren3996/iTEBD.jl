# ScarFinder

ScarFinder searches for low-entanglement, weakly thermalizing trajectories
directly on the iMPS manifold. Given a trial state, a local update rule, and a
target bond dimension `χ`, it repeatedly evolves, projects back to `χ`, and
optionally corrects the energy density so the trajectory does not drift toward
a featureless low-energy state.

`iTEBD.jl` exposes three entry points:

- `scarfinder!(ψ, h, dt, χ, N; ...)` builds `exp(-1im * dt * h)` from the local
  Hamiltonian density `h` and uses it as the real-time gate.
- `scarfinder!(ψ, G, χ, N; ...)` applies a user-supplied gate `G` with no
  energy correction.
- `scarfinder!(ψ, G, h, χ, N; ...)` applies a user-supplied gate `G` but
  measures the energy drift against `h`.

Each iteration has the same shape: real-time evolution at a temporary
bond dimension `maxdim`, projection back to `χ`, and an optional imaginary-time
nudge toward a target energy density. After `N` iterations the routine
optionally scans a short refinement trajectory and keeps the minimum-entanglement
point along it.

## Which interface should I use?

Pick by the form of the update rule.

- If your dynamics is `H` and you only have the local density `h`, use
  `scarfinder!(ψ, h, dt, χ, N; ...)`. This is the simplest path.
- If your update is already packaged as a gate (for example a Floquet step, or
  a precomputed `exp(-1im * dt * h)` you intend to reuse), and you do not need
  an energy constraint, use `scarfinder!(ψ, G, χ, N; ...)`.
- For constrained models such as PXP, where `G` carries projectors that repair
  the physical subspace after truncation, use `scarfinder!(ψ, G, h, χ, N; ...)`.
  The projected gate drives the dynamics while the unprojected Hamiltonian
  density `h` defines the energy target. Conflating the two leads to wrong
  energy corrections.

## Positional arguments

- `ψ` — the trial `iMPS`, updated in place.
- `h` — a local Hamiltonian density on a contiguous window of the unit cell.
  Its support can be inferred by `operator_span(ψ, h)`.
- `G` — a local gate on a contiguous window. Used by the gate-based and mixed
  interfaces.
- `dt` — microscopic real-time step used to build `exp(-1im * dt * h)` in the
  Hamiltonian-based interface.
- `χ` — bond dimension of the projected manifold. ScarFinder always returns to
  this size after each iteration.
- `N` — number of ScarFinder iterations.

## Keyword arguments

Evolution and projection:

- `nstep` — microscopic evolution steps before each projection. One ScarFinder
  iteration covers physical time `Δt ≈ nstep * dt`.
- `maxdim=MAXDIM` — temporary bond dimension during real-time evolution.
- `cutoff=SVDTOL` — SVD cutoff used during the projection step.

Support widths:

- `span=operator_span(ψ, G_or_h)` — number of sites the real-time gate acts on.
  In the Hamiltonian-based interface this refers to `h`; in the gate-based and
  mixed interfaces it refers to `G`.
- `hspan=operator_span(ψ, h)` — number of sites the Hamiltonian density acts on
  during energy fixing in the mixed interface.

Energy fixing:

- `target=nothing` — target energy density. Leave as `nothing` to disable
  energy correction.
- `tol=1e-6` — stop the energy-fixing loop once `|E - target| < tol`.
- `α=0.1` — step-size parameter for the imaginary-time correction.
- `maxstep=50` — maximum substeps inside one energy-fixing call.

Refinement scan (run once after the `N` iterations):

- `refine=true` — scan a short trajectory and keep the minimum-entanglement
  point.
- `refine_step=100` — number of trial points used in the scan.
- `refine_dt=dt/10` — microscopic step used during the scan (Hamiltonian-based
  interface only).

## Two time scales: `dt` vs `nstep`

`dt` is the microscopic gate time; `nstep` is how many of those microscopic
steps fit inside one ScarFinder iteration. If you want each iteration to cover
physical time `Δt`, set `nstep ≈ Δt / dt`. With `dt = 0.01` and `Δt = 0.05`,
that means `nstep = 5`.

`nstep = 1` is accepted for backward compatibility but emits a warning: one
microscopic step followed by a hard projection back to `χ` is usually too
aggressive a truncation cycle to represent the intended coarse-grained flow.

## Two bond dimensions: `maxdim` vs `χ`

`maxdim` is the temporary headroom that real-time evolution may grow into
before each projection. `χ` is where the state lands after the projection.
Typically `maxdim ≥ χ`, often strictly larger so the evolution has room to
represent transient growth that the projection then prunes.

## Helper functions

- [`operator_span`](@ref) infers the support of a local operator from the
  local Hilbert-space dimension of `ψ`.
- [`energy_density`](@ref) returns the unit-cell averaged expectation value of
  a local operator.
- [`energy_span`](@ref) estimates the lowest and highest energy densities
  reachable by imaginary-time iTEBD with `h`. Useful for picking a `target`.

## A complete PXP example

The example below uses the mixed interface and reproduces the standard
ScarFinder workflow on the PXP model. The gate `G` contains a two-site
blockade projector that restores the Rydberg-constraint subspace after each
truncation; the energy target is matched to the energy density of the
initial Z2 product state, which is what the scar trajectory should approximately
preserve.

After the run, two numbers tell you whether ScarFinder converged in the
expected sense: `final_energy` should be close to `target_energy`, and
`maxbond` should sit at or below `χ` (here `χ = 2`). The state is also
small enough to inspect bond-by-bond if you want.

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

## Diagnostics during a run

Between successive `scarfinder!` calls (or between blocks of iterations) it is
worth tracking two cheap quantities:

- `energy_density(psi, h; span=...)` — should stay close to `target` once
  energy fixing has settled. A drift that the energy-fixing loop cannot close
  usually means `α` is too small, `maxstep` is too small, or `target` is
  outside the manifold reachable at the chosen `χ`.
- `maximum(length.(psi.λ))` — the largest bond dimension currently used. If
  this saturates at `χ` after every iteration, the projection is doing real
  work; if it sits well below `χ`, you can probably reduce `χ` (and the cost).

## Reference

The implementation follows the algorithm in:

- J. Ren, A. Hallam, L. Ying, and Z. Papić, [ScarFinder: A Detector of Optimal
  Scar Trajectories in Quantum Many-Body Dynamics](https://eprints.whiterose.ac.uk/id/eprint/238368/),
  PRX Quantum 6, 040332 (2025).

The generated function reference lives on [API Reference](api.md).
