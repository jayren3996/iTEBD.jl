# Observables

The package provides a compact set of routines for overlaps and local energy
density measurements in translationally invariant states.

## Overlaps

Use `inner_product(ψ1, ψ2)` to compare two states:

```@example
using iTEBD

psi1 = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
psi2 = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
inner_product(psi1, psi2)
```

## Local Energy Densities

`energy_density(ψ, h; span=...)` evaluates a local Hamiltonian density on the
current state. `energy_span(n, d, h; ...)` estimates suitable measurement spans
for a periodic unit cell and local term.

The generated function reference lives on [API Reference](api.md).
