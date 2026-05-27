# Symmetric infinite MPS

`iTEBD.jl` ships with an optional symmetric backend that lets you exploit
Abelian conservation laws — total `Sz`, particle number `N`, parity, or any
combination of these — without ever leaving the `iMPS` API you already know.

This page assumes you have run the dense examples in
[States and Canonical Form](imps.md) and now want to take advantage of a
symmetry your model has. **You do not need any prior experience with
TensorKit.**

## What you gain

For the spin-1/2 XXZ chain, total `Sz` is conserved. If you tell `iTEBD.jl`
this, every internal tensor splits into independent blocks labelled by `Sz`,
and roughly `1/√χ` of the data stops being stored. The numerical answer is
unchanged; the runtime and memory both shrink.

## Sectors, charges, graded spaces

A *charge* (synonyms: *sector*, *irrep*, *QN*) is a label that lives on a
basis vector. For spin-1/2 the physical leg has two basis vectors, one with
`Sz = +1/2` and one with `Sz = -1/2`. We use the integer convention `2*Sz`
throughout, so the charges are `+1` and `-1`.

A *graded vector space* is a regular vector space whose basis vectors each
carry a charge. The spin-1/2 physical leg is the graded vector space
`P = (Sz=+1) ⊕ (Sz=-1)`. In `iTEBD.jl` you build it with:

```julia
using iTEBD
using TensorKit                # loads the symmetric extension
P = graded_space(:U1, 1=>1, -1=>1)
```

The pair `1=>1` reads "the `+1` charge has dimension 1 (one basis state)";
similarly for `-1=>1`.

## Flux

Every symmetric tensor has a *flux*: the net charge it adds (or removes) to a
state when it acts. For Abelian symmetries the flux is a single integer (or
tuple, for product symmetries).

The intuition:

- **Flux = 0** — the tensor doesn't move charge. Most operators you care
  about have flux 0: the identity, projectors, conserved quantities like
  `Sz` or particle number, Hamiltonian densities, and the MPS tensors of a
  state in a fixed charge sector.
- **Flux ≠ 0** — the tensor moves charge by that amount. `S+` adds +2 to
  `Sz` (in our `2·Sz` integer convention) so it has flux **+2**. `S-` has
  flux **−2**.

When you multiply or contract two symmetric tensors, the flux must "match
up" the way charges do on the contracted legs — otherwise the result is
zero or TensorKit raises a `SpaceMismatch` error. Adding two tensors with
different fluxes is also forbidden.

Worked example. With `Sz, SzSz, SpSm, SmSp = spin_half_ops(:U1)`:

| Operator | Flux | Why |
|---|---|---|
| `Sz`       | 0  | diagonal, doesn't move spin |
| `SzSz`     | 0  | two-site, both legs flux 0 |
| `SpSm`     | 0  | composite: `+2` on site 1 cancels `-2` on site 2 |
| `SmSp`     | 0  | symmetric to above |

**Practical consequence for `iTEBD.jl`:** the two-site Hamiltonian density
`h = Sz⊗Sz + (1/2)(S+⊗S- + S-⊗S+)` is meaningful because every term has
total flux 0: `Sz⊗Sz` is two flux-0 operators on adjacent sites; `S+⊗S-`
pairs a +2 with a −2, summing to 0; same for `S-⊗S+`. `iTEBD.jl`'s
`spin_half_ops(:U1)` returns these pre-assembled two-site terms
(`SzSz`, `SpSm`, `SmSp`) so you can add them and `exp` them freely. The
individual single-site `S+` and `S-` are not returned because composing
them naively (`S+ ⊗ S-` as a four-leg operator) requires extra "sided
operator" machinery that lives outside the v1 helper layer.

If you want to dig into how TensorKit represents flux internally — the
codomain/domain split, dual spaces, arrows on diagrams — see the
[TensorKit manual](https://quantumkithub.github.io/TensorKit.jl/stable/man/sectors/).
For most users of this package, "operators have a flux; flux-0 things
compose and add cleanly" is all you need to know.

## End-to-end walkthrough: spin-1/2 XXZ in the Sz=0 sector

```julia
using iTEBD, TensorKit

# Build the U(1)-symmetric spin-1/2 operators (pre-assembled two-site forms)
Sz, SzSz, SpSm, SmSp = spin_half_ops(:U1)

# Heisenberg density h = Sz⊗Sz + (1/2)(S+⊗S- + S-⊗S+)
h = SzSz + 0.5 * (SpSm + SmSp)

# Néel initial state in the Sz=0 sector
ψ = product_iMPS(:U1, [-1, 1], [1, -1])

# Imaginary-time iTEBD, two-site Trotter
dt = 0.05
gates = [(exp(-dt * h), 1, 2), (exp(-dt * h), 2, 1)]
evolve!(ψ, gates, 400; maxdim=32, cutoff=1e-10)

# Energy density approaches the Bethe-ansatz value 1/4 - log(2) ≈ -0.4431
energy_density(ψ, h)

# Sector-resolved Schmidt spectrum
schmidt_values(ψ, 1)
```

## Common errors

- `DimensionMismatch: ... fluxes must close around the unit cell` — the
  right virtual space of `Γ[n]` differs from the left virtual space of
  `Γ[1]`. In practice this happens when the running flux around the unit
  cell does not return to zero. Either set every tensor's flux to 0, or
  build the wraparound space explicitly.
- `MethodError: no method matching ...` from a `graded_space` call without
  loading TensorKit — you need `using TensorKit` to pull in the symmetric
  extension.
- `Argument N must be ≥ 2` in `graded_space(:ZN, N, …)` — Z_1 is trivial;
  use `:Trivial` instead.
- `ArgumentError: canonical!: asymmetric transfer eigenvalues (λ_r=..., λ_l=...)` —
  your state is in a non-injective regime that the v1 symmetric `canonical!`
  refuses to handle (to avoid silently producing a corrupted state). Try a
  different random seed or construct the state in a fixed flux sector via
  `product_iMPS`.

## See also

- [MPSKitModels.jl](https://github.com/QuantumKitHub/MPSKitModels.jl) for
  ready-made symmetric Hamiltonians (Heisenberg, Hubbard, ...) that work
  unchanged with this backend.
- [TensorKit.jl manual](https://quantumkithub.github.io/TensorKit.jl/stable/)
  for the underlying symmetric-tensor library.
