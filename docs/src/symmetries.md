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

> **What is the "flux" of a tensor?**
>
> Every leg of a symmetric tensor carries a charge label on each of its
> basis states. When you contract two legs together, the charges on the
> contracted basis states must match — otherwise the matrix element is zero
> by symmetry. The *flux* of a tensor is the **net charge it carries
> between its incoming and outgoing legs**: how much charge goes *in* on
> one side minus how much goes *out* on the other.
>
> A tensor with **flux = 0** is the most common case — it neither creates
> nor destroys charge. Examples: the identity, a U(1)-symmetric Hamiltonian
> density `Sz⊗Sz`, an MPS tensor at the ground state of an `Sz`-conserving
> model.
>
> A tensor with **flux ≠ 0** *moves* charge. `S+` has flux `+2` (in our
> `2*Sz` convention) because it raises spin. A flux-`q` MPS tensor inserts
> `q` units of total `Sz` at that site.
>
> In `iTEBD.jl`, when you build an iMPS in a fixed-`Sz` sector you set
> every MPS tensor to flux=0, and the wraparound bond closes onto itself
> with consistent charges. If you wanted to study a state with a single
> magnon (one extra `Sz = +1`), you would put one flux-`+2` site somewhere
> in the unit cell.

Worked example. With `Sz, SzSz, SpSm, SmSp = spin_half_ops(:U1)`:

| Operator | Flux | Why |
|---|---|---|
| `Sz`       | 0  | diagonal, doesn't move spin |
| `SzSz`     | 0  | two-site, both legs flux 0 |
| `SpSm`     | 0  | composite: `+2` on site 1 cancels `-2` on site 2 |
| `SmSp`     | 0  | symmetric to above |

Note: `iTEBD.jl`'s `spin_half_ops(:U1)` deliberately returns the
pre-assembled flux-0 two-site terms `SzSz`, `SpSm`, `SmSp` rather than the
individual `S+` and `S-` operators. The individual operators are non-zero
flux and cannot be directly added in the symmetric tensor framework
without sided-operator machinery; the pre-assembled two-site forms close
the flux locally and can be combined and added freely.

## Arrow convention on diagrams

> **Arrow convention.** Every leg in our diagrams has an arrow.
>
> - An arrow pointing **into** the tensor means that leg's charges are
>   read *as given*.
> - An arrow pointing **out** of the tensor means that leg's charges are
>   read *negated* (mathematically: this leg lives in the dual space).
>
> **Why this matters.** When you connect two legs together, the arrows
> have to be **consistent**: one arrow leaves one tensor, the other arrow
> enters the next tensor. (Connecting "out" to "out" or "in" to "in"
> would sum charges that should subtract — TensorKit raises an error.)
>
> **The standard MPS convention used in this package:** physical legs
> point *out* (kets), bonds point *right* (left bond into the tensor,
> right bond out of it). The flux equation reads
> `(in charges) − (out charges) = flux`.

```
       P↑ (physical, out)
        │
        ▼
   ───►─[ Γ_i ]─►───
   V_left (in)     V_right (out)

   flux(Γ_i) = (charges into Γ_i) − (charges out of Γ_i)
             = V_left_charge − V_right_charge − P_charge
```

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
- `canonical!: asymmetric transfer eigenvalues (λ_r=..., λ_l=...)` — your
  state is in a non-injective regime that the v1 symmetric `canonical!`
  doesn't handle. Try a different random seed or initial state.

## See also

- [MPSKitModels.jl](https://github.com/QuantumKitHub/MPSKitModels.jl) for
  ready-made symmetric Hamiltonians (Heisenberg, Hubbard, ...) that work
  unchanged with this backend.
- [TensorKit.jl manual](https://quantumkithub.github.io/TensorKit.jl/stable/)
  for the underlying symmetric-tensor library.
