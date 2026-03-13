# iTEBD.jl

`iTEBD.jl` is a Julia package for infinite time-evolving block decimation on
translationally invariant one-dimensional systems.

The package is built around a compact `iMPS` representation and provides:

- infinite matrix-product states with a finite unit cell,
- gate-based real- and imaginary-time evolution,
- Schmidt canonicalization and basic transfer-matrix observables,
- a ScarFinder workflow for low-entanglement state searches,
- a mixed `gate + Hamiltonian` ScarFinder interface for constrained models such as PXP.

## Installation

Install from a Julia REPL:

```julia
pkg> add https://github.com/jayren3996/iTEBD.jl
```

Then load the package with:

```julia
using iTEBD
```

## Core Ideas

### `iMPS`

An `iMPS` stores one periodic unit cell of an infinite matrix-product state:

- `Γ`: local tensors,
- `λ`: Schmidt values on each bond,
- `n`: number of sites in the unit cell.

This package uses a right-canonical convention in which the right Schmidt values
are absorbed into each local tensor. After calling `canonical!`, the entanglement
structure is stored in `λ` and the local tensors are right-canonical.

### Local Gates

Time evolution is implemented by repeatedly applying a local gate:

```julia
applygate!(ψ, G, i, j; maxdim=MAXDIM, cutoff=SVDTOL, renormalize=true)
```

where:

- `ψ` is an `iMPS`,
- `G` is a dense local operator,
- `i:j` specifies the support within the periodic unit cell,
- `maxdim` controls the temporary bond dimension during truncation.

## Main API

These are the entry points you are most likely to use:

- `iMPS(Γs; renormalize=true)`
- `rand_iMPS(T, n, d, dim)`
- `product_iMPS(vectors)`
- `canonical!(ψ)`
- `applygate!(ψ, G, i, j; ...)`
- `inner_product(ψ1, ψ2)`
- `energy_density(ψ, h; span=...)`
- `energy_span(n, d, h; ...)`
- `scarfinder_step!`
- `scarfinder!`

## Quick Start

### Random State

```julia
using iTEBD

psi = rand_iMPS(ComplexF64, 2, 2, 4)
```

This creates a random two-site unit-cell state with local dimension `2` and
bond dimension `4`.

### Product State

```julia
using iTEBD

psi = product_iMPS(ComplexF64, [[0, 1], [1, 0], [0, 1], [1, 0]])
```

This is the $Z_2$ product state on a four-site unit cell.

### Imaginary-Time AKLT Example

```julia
using iTEBD, LinearAlgebra

X = sqrt(2) / 2 * [0 1 0; 1 0 1; 0 1 0]
Y = sqrt(2) / 2 * 1im * [0 -1 0; 1 0 -1; 0 1 0]
Z = [1 0 0; 0 0 0; 0 0 -1]

H = begin
    SS = kron(X, X) + kron(Y, Y) + kron(Z, Z)
    0.5 * SS + SS^2 / 6 + I / 3
end

G = exp(-0.1 * H)
psi = rand_iMPS(ComplexF64, 2, 3, 1)

for _ in 1:300
    applygate!(psi, G, 1, 2; maxdim=8)
    applygate!(psi, G, 2, 1; maxdim=8)
end
```

## ScarFinder

The package includes a general ScarFinder workflow for low-entanglement state
searches. There are two useful interfaces:

- Hamiltonian-based:
  `scarfinder!(ψ, h, dt, χ, N; ...)`
- Gate-based with Hamiltonian energy fixing:
  `scarfinder!(ψ, G, h, χ, N; ...)`

The second form is especially useful in constrained models where the update rule
is more naturally written as a projected gate rather than a pure exponential
$e^{-i dt h}$.

### PXP ScarFinder Example

This is the recommended ScarFinder example for the package.

```julia
using iTEBD, LinearAlgebra

# Projector entering the PXP Hamiltonian.
P0 = [0 0; 0 1]

# Local excitation projector used for the blockade constraint.
N1 = [1 0; 0 0]

X = [0 1; 1 0]

# Three-site local PXP Hamiltonian density.
h_pxp = kron(P0, X, P0)

# Local no-double-excitation projector used to repair truncation artifacts.
no_double_2 = Matrix{Float64}(I, 4, 4) - kron(N1, N1)
proj_pxp = kron(no_double_2, I(2)) * kron(I(2), no_double_2)

# Microscopic gate and projected ScarFinder gate.
dt = 0.01
G = proj_pxp * exp(-1im * dt * h_pxp)

# Z2 initial state on a four-site unit cell.
psi = product_iMPS(ComplexF64, [[0, 1], [1, 0], [0, 1], [1, 0]])

# Use the Z2 energy density as the target.
target = energy_density(psi, h_pxp; span=3)

# If one ScarFinder step is meant to represent Δt = 0.1, use nstep = 10.
scarfinder!(psi, G, h_pxp, 2, 200;
    span=3,
    hspan=3,
    nstep=10,
    target=target,
    maxdim=32,
)
```

### Important ScarFinder Notes

- If the gate is built from a microscopic time step $dt$ but each ScarFinder
  iteration is supposed to represent a larger interval $\Delta t$, choose
  `nstep ≈ Δt / dt`.
- In constrained models such as PXP, the projector used in the ScarFinder gate
  is often **not** the same object as the projector appearing in the Hamiltonian
  density. Keeping `G` and `h` separate in the API is intentional.
- The mixed interface
  `scarfinder!(ψ, G, h, χ, N; ...)`
  is the one to use when the evolution rule is gate-based but energy fixing
  should still be performed with respect to a Hamiltonian density.

## Example Notebooks

The `examples/` folder contains runnable notebooks:

- `CanonicalForm.ipynb`
  canonicalization diagnostics and sanity checks.
- `AKLT_GS.ipynb`
  imaginary-time AKLT ground-state evolution.
- `PXP.ipynb`
  short-time PXP dynamics from the $Z_2$ product state.
- `PXP_ScarFinder.ipynb`
  PXP ScarFinder search using the mixed `gate + Hamiltonian` interface.

Each notebook begins with:

```julia
import Pkg
Pkg.activate("..")
```

so it can be run directly from the repository checkout.

## Current Scope

`iTEBD.jl` is intentionally fairly direct and low-level. The package assumes:

- dense local operators,
- explicit control over unit-cell structure,
- manual selection of gate supports and truncation parameters.

That keeps it flexible for exploratory work, especially for custom ScarFinder
protocols and constrained dynamics.
