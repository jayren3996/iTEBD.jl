# iTEBD.jl

`iTEBD.jl` is a Julia package for infinite time-evolving block decimation on
translationally invariant one-dimensional systems.

The package is built around a compact `iMPS` representation and currently
focuses on stable workflows:

- finite-unit-cell infinite matrix-product states,
- Schmidt canonicalization in the injective setting,
- local gate-based real- and imaginary-time evolution,
- basic transfer-matrix observables,
- ScarFinder routines for low-entanglement state searches.

## What This Manual Covers

This manual is organized around the package's main workflows:

- [Getting Started](getting-started.md) for installation and a first state.
- [iMPS and Canonical Form](imps.md) for the package's tensor convention.
- [Time Evolution](time-evolution.md) for local gate updates.
- [Observables](observables.md) for overlaps and energy densities.
- [ScarFinder](scarfinder.md) for low-entanglement search workflows.
- [API Reference](api.md) for docstring-driven function reference.

## A Minimal Example

```@example
using iTEBD

psi = rand_iMPS(ComplexF64, 2, 2, 4)
canonical!(psi)
length(psi.Γ), length(psi.λ)
```

## Package Scope

`iTEBD.jl` stays intentionally direct and low-level. You specify the unit cell,
the local operators, and the truncation settings explicitly, which keeps the
package flexible for exploratory tensor-network work and custom ScarFinder
protocols.
