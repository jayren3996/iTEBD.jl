# Getting started

## Installation

From a Julia REPL:

```julia
pkg> add https://github.com/jayren3996/iTEBD.jl
```

Then load it with:

```julia
using iTEBD
```

## Checking your install

A quick sanity check: construct a small state, canonicalize it, and look at
the Schmidt spectrum on the first bond. If this runs without error and prints
a normalized vector, the package is working.

```@example install
using iTEBD

psi = rand_iMPS(ComplexF64, 2, 2, 4)
canonical!(psi)
psi.λ[1]
```

Because `canonical!` renormalizes by default, `sum(psi.λ[1].^2)` should equal
`1` up to floating-point error.

## Your first `iMPS`

A random state is the fastest way to get going. The signature
`rand_iMPS(T, n, d, dim)` takes

- `T`: element type of the local tensors (e.g. `Float64`, `ComplexF64`),
- `n`: number of physical sites in the periodic unit cell,
- `d`: local Hilbert-space dimension at each site,
- `dim`: bond dimension used to sample the initial random tensors.

```@example first
using iTEBD

psi = rand_iMPS(ComplexF64, 2, 2, 4)
```

This is a two-site unit cell of qubits (`d = 2`) with bond dimension `4`. The
returned state is already canonicalized, so `psi.λ[i]` holds the Schmidt
spectrum on the bond to the right of site `i` and `psi.Γ[i]` holds the
right-canonical tensor $B_i = \Gamma_i \lambda_i$.

To inspect the entanglement structure on the first bond:

```@example first
psi.λ[1]
```

If you later mutate the tensors directly, call `canonical!(psi)` again before
computing observables. The defaults `MAXDIM` and `SVDTOL` set the bond-dimension
cap and the singular-value cutoff used during the SVD sweep.

## Product states

`product_iMPS` takes a vector of local site vectors, one per site of the unit
cell, and builds the corresponding bond-dimension-1 product state. All site
vectors must share the same local dimension; each is normalized internally.

```@example product
using iTEBD

z2 = product_iMPS(ComplexF64, [[0, 1], [1, 0], [0, 1], [1, 0]])
```

The example above is a four-site $Z_2$ (Néel) state in the $\sigma^z$ basis.
Bond dimension is `1`, so each `z2.λ[i]` is `[1.0]`.

## Examples in the repository

The [`examples/`](https://github.com/jayren3996/iTEBD.jl/tree/master/examples)
directory contains runnable notebooks for the stable workflows:

- `CanonicalForm.ipynb`
- `AKLT_GS.ipynb`
- `PXP.ipynb`
- `PXP_ScarFinder.ipynb`

Each notebook begins with

```julia
import Pkg
Pkg.activate("..")
```

so it can be run directly from a repository checkout.

## Where to go next

If you came here asking "how do I evolve a state in time?", go straight to
[Time Evolution](time-evolution.md) for the gate and Trotter interface.

Other useful next stops:

- [States and Canonical Form](imps.md) for the storage convention used for
  `Γ` and `λ`, and the relationship to the bare Vidal form;
- [Observables](observables.md) for overlaps, local expectation values, and
  energy densities;
- [ScarFinder Workflow](scarfinder.md) for the iterative low-entanglement
  search routines.
