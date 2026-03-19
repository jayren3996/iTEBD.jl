# Observables

The package provides a compact set of routines for overlaps and local energy
density measurements in translationally invariant states.

## Overlaps

For infinite, translationally invariant states, `inner_product` is defined from
the transfer matrix of one periodic unit cell, not from a finite-chain
contraction.

If `ψ_1` and `ψ_2` have unit-cell tensors
`[\Gamma_1^{(1)}, ..., \Gamma_n^{(1)}]` and
`[\Gamma_1^{(2)}, ..., \Gamma_n^{(2)}]`, the package builds the unit-cell
transfer matrix

```math
E_{\mathrm{cell}}(\psi_1, \psi_2)
= E_1(\Gamma_1^{(1)}, \Gamma_1^{(2)})
  E_2(\Gamma_2^{(1)}, \Gamma_2^{(2)}) \cdots
  E_n(\Gamma_n^{(1)}, \Gamma_n^{(2)}),
```

where each local factor contracts the physical leg and keeps the virtual legs
open:

```math
E_i{}_{(\beta_{i-1}, \alpha_{i-1}), (\beta_i, \alpha_i)}
= \sum_s
\overline{\Gamma_i^{(1)}(\alpha_{i-1}, s, \alpha_i)}
\Gamma_i^{(2)}(\beta_{i-1}, s, \beta_i).
```

The overlap reported by the package is then

```math
\mathrm{inner\_product}(\psi_1, \psi_2)
= \left| \lambda_{\max}\!\bigl(E_{\mathrm{cell}}(\psi_1, \psi_2)\bigr) \right|,
```

namely the magnitude of the dominant eigenvalue of the unit-cell transfer
matrix. The one-argument form `inner_product(ψ)` is the corresponding norm per
unit cell.

So `inner_product(ψ1, ψ2)` should be interpreted as an overlap per unit cell for
the infinite system, not as the ordinary inner product of a finite MPS.

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
