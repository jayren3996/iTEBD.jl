# Observables

This page covers the measurement routines that act on a translationally
invariant [`iMPS`](@ref). From a canonicalized state you can compute:

- Self-norms and cross overlaps with [`inner_product`](@ref).
- Single-site and multi-site expectation values with [`expect`](@ref).
- Bipartite entanglement entropy across any bond with [`ent_S`](@ref).
- Unit-cell averaged energy density with [`energy_density`](@ref).
- Reachable low- and high-energy brackets with [`energy_span`](@ref).
- The operator support inferred from `size(O)` with [`operator_span`](@ref).

The signatures and edge-case behavior are documented in the
[API Reference](api.md).

## Overlaps

`inner_product(ψ)` returns the norm per unit cell, and `inner_product(ψ1, ψ2)`
returns the overlap per unit cell of two states sharing a common unit-cell
length. Both quantities are built from the unit-cell transfer matrix, not from
a finite-chain contraction.

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

The reported value is the magnitude of the dominant eigenvalue of this product:

```math
\mathrm{inner\_product}(\psi_1, \psi_2)
= \left| \lambda_{\max}\!\bigl(E_{\mathrm{cell}}(\psi_1, \psi_2)\bigr) \right|.
```

This convention differs from the usual finite-MPS inner product. The phase of
the dominant eigenvalue is discarded, and the result is one per cell rather
than for the whole chain. Two states that agree up to a global phase therefore
report `1`, and orthogonal states report `0`.

```@example
using iTEBD

psi1 = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
psi2 = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
inner_product(psi1, psi2)
```

For a single argument, `inner_product(ψ)` reduces to the norm per unit cell of
`ψ`:

```@example
using iTEBD

psi = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
inner_product(psi)
```

## Local expectation values

`expect(ψ, O, i, j)` returns the real part of the expectation value of `O`
acting on the contiguous block of sites `i:j` inside the periodic unit cell.
When `j < i`, the support wraps around the cell. The matrix dimension of `O`
must equal `d^(j - i + 1)` where `d` is the local Hilbert-space dimension.

Single-site Pauli measurement on a two-site cell:

```@example
using iTEBD

psi = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
Z = [1 0; 0 -1]
expect(psi, Z, 1, 1)
```

The site index addresses the second basis state on site 2, so the same operator
gives the opposite sign there:

```@example
using iTEBD

psi = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
Z = [1 0; 0 -1]
expect(psi, Z, 2, 2)
```

Multi-site operators use `kron` (with the leftmost site as the leftmost factor)
and span the contiguous interval `i:j`:

```@example
using iTEBD
using LinearAlgebra

psi = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
Z = [1 0; 0 -1]
expect(psi, kron(Z, Z), 1, 2)
```

## Entanglement entropy

`ent_S(ψ, i)` returns the bipartite von Neumann entropy across bond `i`,
computed directly from the stored Schmidt spectrum `ψ.λ[i]`. The bond index is
wrapped periodically, so any integer is valid. The state should be
canonicalized before measurement, otherwise `ψ.λ[i]` is not the Schmidt
spectrum of a cut.

```@example
using iTEBD

psi = rand_iMPS(ComplexF64, 2, 2, 4)
canonical!(psi)
ent_S(psi, 1)
```

For a product state the entropy is zero across every bond:

```@example
using iTEBD

psi = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
ent_S(psi, 1)
```

## Energy density

`energy_density(ψ, h; span)` returns the unit-cell averaged expectation value
of a local Hamiltonian density `h`. The package shifts `h` across every
starting site `i = 1, ..., n` of the unit cell and averages the
[`expect`](@ref) values:

```math
\frac{1}{n}\sum_{i=1}^{n}\langle h_{i,\, i+\mathrm{span}-1}\rangle.
```

When the keyword `span` is omitted, it is inferred from `size(h)` and the local
dimension of `ψ` through [`operator_span`](@ref); `size(h, 1)` must be `d^span`
for some integer `span`.

A two-site Heisenberg-like density on a two-site unit cell:

```@example
using iTEBD
using LinearAlgebra

Sx = [0 1; 1 0] / 2
Sy = [0 -im; im 0] / 2
Sz = [1 0; 0 -1] / 2
h = real(kron(Sx, Sx) + kron(Sy, Sy)) + kron(Sz, Sz)

psi = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
energy_density(psi, h)            # span inferred as 2 from size(h) = (4, 4)
```

The same call with an explicit `span` argument:

```@example
using iTEBD
using LinearAlgebra

Sx = [0 1; 1 0] / 2
Sy = [0 -im; im 0] / 2
Sz = [1 0; 0 -1] / 2
h = real(kron(Sx, Sx) + kron(Sy, Sy)) + kron(Sz, Sz)

psi = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
energy_density(psi, h; span=2)
```

`span` must be a positive integer not exceeding the unit-cell length.

## Energy bracket

`energy_span(n, d, h; dτ, Nτ, maxdim)` runs short imaginary-time iTEBD sweeps
with `exp(-dτ h)` and `exp(+dτ h)` starting from random unit cells, then
returns

```julia
Emin, Emax, (psi_min, psi_max) = energy_span(n, d, h)
```

`Emin` and `Emax` are the energy densities of the relaxed low- and high-energy
trial states, and the accompanying states are useful for warm starts. The
bracket is a heuristic estimate of the range reachable by iTEBD at the chosen
`maxdim`, not a rigorous variational bound.

```julia
using iTEBD
using LinearAlgebra

Sx = [0 1; 1 0] / 2
Sy = [0 -im; im 0] / 2
Sz = [1 0; 0 -1] / 2
h = real(kron(Sx, Sx) + kron(Sy, Sy)) + kron(Sz, Sz)

Emin, Emax, _ = energy_span(2, 2, h; dτ=0.05, Nτ=200, maxdim=16)
```

The returned `(Emin, Emax)` is a sensible range from which to pick a `target`
energy density when calling `scarfinder!`; see [ScarFinder](scarfinder.md) for
the energy-fixing protocol that consumes it.
