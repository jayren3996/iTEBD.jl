# iTEBD.jl

`iTEBD.jl` is a Julia package for infinite time-evolving block decimation (iTEBD)
calculations on translationally invariant one-dimensional systems.

The package focuses on:
- infinite matrix-product states with a finite unit cell,
- gate-based real- and imaginary-time evolution,
- canonicalization and basic transfer-matrix observables,
- a compact hybrid `ScarFinder` workflow for low-entanglement state searches.

## Installation

Install from a Julia REPL with:

```julia
pkg> add https://github.com/jayren3996/iTEBD.jl
```

## Core API

The most commonly used entry points are:

- `iMPS(Γs; renormalize=true)`: build an infinite MPS from a list of local tensors.
- `rand_iMPS(T, n, d, dim)`: random canonicalized `iMPS`.
- `product_iMPS(vectors)`: bond-dimension-1 product state on a finite unit cell.
- `canonical!(ψ)`: bring an `iMPS` to Schmidt-canonical form.
- `applygate!(ψ, G, i, j; maxdim, cutoff, renormalize)`: apply a local gate.
- `inner_product(ψ1, ψ2)`: overlap per unit cell.
- `energy_density(ψ, h)`: unit-cell averaged expectation value of a local operator.
- `scarfinder_step!` and `scarfinder!`: hybrid scar-search routines.
- `floquet_scarfinder_step!` and `floquet_scarfinder!`: Floquet scar-search routines.

## Quick Start

### States

An `iMPS` stores:
- `Γ`: local tensors for one unit cell,
- `λ`: Schmidt spectra on each bond,
- `n`: number of sites in the unit cell.

The convention used in this package absorbs the right Schmidt values into each
local tensor. After canonicalization, the tensors are right-canonical and the
vectors `λ[i]` contain the entanglement data.

### Example: AKLT Ground State By Imaginary-Time iTEBD

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

### Example: General ScarFinder Workflow

```julia
using iTEBD, LinearAlgebra

h = begin
    zz = -[1 0 0 -1;
           0 -1 -1 0;
           0 -1 -1 0;
          -1 0 0 1]
    x = [0 1; 1 0]
    z = [1 0; 0 -1]
    h1 = -0.5 * z - 1.05 * x
    zz + (kron(h1, I(2)) + kron(I(2), h1)) / 2
end

psi = rand_iMPS(ComplexF64, 2, 2, 2)

scarfinder!(psi, h, 0.01, 2, 200;
    nstep=10,
    target=-1.0,
    maxdim=64,
)

println("energy density = ", energy_density(psi, h))
println("entanglement   = ", iTEBD.ent_S(psi, psi.n))
```

`scarfinder!` infers the operator range from the local Hilbert-space dimension
of `ψ`, so the same interface works for 2-site, 3-site, and longer local operators
as long as they are passed as dense matrices compatible with `applygate!`.

### Example: Floquet ScarFinder

For periodically driven systems, use `floquet_scarfinder!` with either a single
one-period gate or a sequence of substep gates:

```julia
using iTEBD, LinearAlgebra

P = [0 0; 0 1]
X = [0 1; 1 0]
PXP = kron(P, X, P)

U1 = exp(-1im * 0.04 * PXP)
U2 = exp(-1im * 0.02 * PXP)

psi = product_iMPS(ComplexF64, [[0, 1], [1, 0], [0, 1], [1, 0]])

floquet_scarfinder!(psi, [U1, U2], 2, 50;
    spans=3,
    ncycle=1,
    maxdim=32,
    refine=true,
    refine_step=100,
)
```

This applies one Floquet period, truncates back to bond dimension `2`, and then
searches for a low-entanglement Floquet state by repeating the procedure.

## Example Notebooks

The `examples/` folder contains runnable notebooks:

- `CanonicalForm.ipynb`: canonicalization diagnostics.
- `AKLT_GS.ipynb`: imaginary-time AKLT ground-state calculation.
- `PXP.ipynb`: short-time PXP dynamics from the `Z₂` state.
- `PXP_ScarFinder.ipynb`: compact PXP scar-search workflow using `scarfinder!`.

Each notebook starts with:

```julia
import Pkg
Pkg.activate("..")
```

so they can be run directly from the repository checkout.
