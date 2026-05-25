```@raw html
<div style="text-align: center; margin-bottom: 1.5rem;">
  <img src="assets/logo.svg" alt="iTEBD.jl" width="160"/>
</div>
```

# iTEBD.jl

`iTEBD.jl` is a Julia package for infinite time-evolving block decimation on
translationally invariant one-dimensional quantum systems. It targets the
thermodynamic limit directly: instead of a long finite chain with open or
periodic boundaries, you specify a periodic unit cell and the package works
with the resulting infinite matrix-product state. This is convenient when the
quantity you care about (entanglement structure, bulk energy density, scar
trajectories) is defined per unit cell rather than for a particular system
size.

The package deliberately stays narrow. It does not implement finite-size DMRG,
mixed boundary conditions, or symmetric tensors, and it assumes the injective
setting throughout canonicalization. If you need any of those, a finite-MPS
package such as `ITensors.jl` is a better starting point. What you do get here
is a compact `iMPS` type, explicit control over truncation and Trotter order,
and a ScarFinder routine for low-entanglement search inside the iMPS manifold.

## Manual layout

The manual is organized around the package's main workflows. Each page is
written for human readers and emphasizes conventions, caveats, and runnable
examples; the API reference is generated from the docstrings and lists exact
signatures and defaults.

- [Getting Started](getting-started.md) covers installation and how to build a
  first state with `rand_iMPS` or `product_iMPS`.
- [iMPS and Canonical Form](imps.md) explains the storage convention — in
  particular, that `ψ.Γ[i]` holds the right-canonical tensor `B_i = Γ_i λ_i`
  rather than the bare Vidal `Γ_i` — and what `canonical!` does to a state.
- [Time Evolution](time-evolution.md) describes `applygate!` for single
  updates, `evolve!` for repeated sweeps, the layered Trotter interface
  (`:second`, `:fourth`, `:fourth_opt`), and the adaptive bond-dimension
  helpers `natural_bonddim` and `adaptive_bonddim`.
- [Observables](observables.md) covers transfer-matrix overlaps with
  `inner_product`, expectation values with `expect`, entanglement entropy with
  `ent_S`, and `energy_density` / `energy_span` for local Hamiltonians.
- [ScarFinder](scarfinder.md) documents the three `scarfinder!` interfaces
  (Hamiltonian, gate, mixed), how the two time scales `dt` and `nstep` combine,
  and a complete PXP example.
- [API Reference](api.md) gives the docstring-driven signature list, including
  keyword defaults and the constants `MAXDIM` and `SVDTOL`.

A reasonable reading order for a new user is Getting Started → iMPS and
Canonical Form → Time Evolution → Observables, then ScarFinder if you need the
low-entanglement search routines and the API Reference when you want exact
signatures.

## A minimal example

The snippet below builds a random two-site unit cell with bond dimension 4,
brings it to Schmidt-canonical form, and inspects the leading Schmidt spectrum.
For a random initial state the dominant Schmidt value is close to 1 and the
rest of the spectrum is small, which is what `canonical!` exposes by rotating
the gauge into the Schmidt basis on every bond.

```@example
using iTEBD

psi = rand_iMPS(ComplexF64, 2, 2, 4)
canonical!(psi)

(;
    n_sites = length(psi.Γ),
    n_bonds = length(psi.λ),
    lambda1 = psi.λ[1],
)
```

After `canonical!`, the stored tensors `psi.Γ[i]` are right-canonical and the
entanglement structure on each bond is carried explicitly by `psi.λ[i]`. See
[iMPS and Canonical Form](imps.md) for the full convention and
[Time Evolution](time-evolution.md) for how local gates are applied on top of
this representation.

## Package scope

`iTEBD.jl` is intentionally direct. You specify the unit cell, the local
operators, and the truncation settings (`maxdim`, `cutoff`) explicitly. There
is no automatic Hamiltonian builder, no symmetry sectors, and no abelian or
non-abelian quantum-number bookkeeping. That keeps the package small enough to
read end-to-end and flexible for exploratory tensor-network work and custom
ScarFinder protocols.
