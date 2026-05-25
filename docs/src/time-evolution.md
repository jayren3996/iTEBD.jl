# Time evolution

Time evolution proceeds by applying local gates to the unit cell and
re-canonicalizing the bonds that the gate touched. This page covers the local
update, the layered Trotter helpers, and the optional adaptive bond-dimension
controller.

## Local gate updates

The in-place update is

```julia
applygate!(ψ, G, i, j; maxdim=MAXDIM, mindim=1, truncerr=0.0,
           svd_min=SVDTOL, renormalize=true, return_stats=false)
```

where `i:j` is the contiguous support inside the periodic unit cell (the range
may wrap around). The gate `G` is a dense operator whose dimension must match
the total physical dimension of the block.

Truncation on the updated internal bonds is governed by the same controller
documented on the [truncation](truncation-note.md) page:

- `maxdim` is a hard cap on the kept bond dimension.
- `mindim` is the minimum kept dimension (default `1`).
- `truncerr` is a target discarded weight. The default `0.0` selects fixed-cap
  behavior; positive values let the controller drop modes once the discarded
  weight rises above the target.
- `svd_min` is an absolute singular-value floor applied after the
  discarded-weight rule.
- `renormalize=true` rescales the kept Schmidt values so the state stays
  normalized.

For repeated sweeps, use

```julia
evolve!(ψ, gates, steps; chi_policy=:fixed, maxdim=MAXDIM, ...)
```

with each element of `gates` written as `(G, i, j)`. The truncation keywords
have the same meaning as above and are applied to every gate in the sweep.

## Hamiltonian layers and Trotter order

Given a Hamiltonian split into commuting layers, the package can build the
Trotter gates internally:

```julia
evolve!(ψ, layers, dt, steps; trotter=:second, evolution=:real, ...)
```

Each `layers[k]` is a vector of local Hamiltonian terms `(h, i, j)`. Terms in
one layer are assumed to commute and are applied in the supplied order; the
caller is responsible for checking commutation. To inspect or reuse the gate
list explicitly,

```julia
trotter_gates(layers, dt; trotter=:second, evolution=:real)
```

returns one macro-step as a vector of `(G, i, j)` tuples.

### Choosing a Trotter scheme

Three schemes are available:

- `trotter=:second` — Strang splitting on any number of layers. Required for
  imaginary-time evolution. A sensible default for ground-state search.
- `trotter=:fourth_opt` — Barthel-Zhang optimized fourth-order formula for
  exactly two layers. Fewest substeps among the fourth-order options, so it is
  the cheapest accurate choice for typical nearest-neighbor models in real time
  at small `dt`.
- `trotter=:fourth` — Suzuki's recursive fourth-order composition. Works for
  any number of layers and is the right choice when the Hamiltonian needs more
  than two commuting blocks.

Both fourth-order schemes contain negative substeps, so they are restricted to
real-time evolution; `evolution=:imaginary` is only valid with
`trotter=:second`. Substep counts also differ: `:second` produces one Strang
sweep per macro-step, `:fourth` produces five Strang sweeps, and `:fourth_opt`
produces eleven elementary substeps over two layers. Account for that when
comparing wall-clock costs.

```@example
using iTEBD
using LinearAlgebra

X = [0 1; 1 0]
Z = [1 0; 0 -1]
H = kron(X, X) + 0.2 * kron(Z, Z)

layers = [[(H, 1, 2)], [(H, 2, 1)]]
psi = product_iMPS(ComplexF64, [[1, 0], [0, 1]])

evolve!(psi, layers, 0.1, 5; trotter=:fourth, maxdim=8)

length(trotter_gates(layers, 0.1; trotter=:fourth_opt))
```

## Adaptive bond dimension

By default, `evolve!` runs with `chi_policy=:fixed`, which combines the
`maxdim` cap with the discarded-weight controller documented above. That fixed
policy is the recommended starting point. The adaptive policy is opt-in and
mostly useful when you want the bond dimension to grow gradually during a
long real- or imaginary-time run rather than being saturated to `maxdim` from
the first step.

Two helpers support the adaptive policy:

- `natural_bonddim(λ; q=1.0, alpha=0.1)` reads the current Schmidt spectrum
  `λ` and returns a smooth estimate of how many modes the state actually
  needs. Larger `q` (e.g. `q = 2`) gives a smaller, more aggressive estimate;
  smaller `q` keeps more of the tail and is more conservative. The `alpha`
  factor adds a small tail-protection bonus when the spectrum is broad.
- `adaptive_bonddim(previous, λ; mindim, maxdim, ...)` turns that estimate
  into the actual bond dimension used by the next step, with two guardrails:
  the result never decreases between steps, and it is clamped to
  `[mindim, maxdim]`.

The formulas below define those helpers precisely. They are reference
material, not required reading for normal use.

`natural_bonddim` is built from the normalized Schmidt weights

```math
p_i = \frac{|\lambda_i|^2}{\sum_j |\lambda_j|^2},
```

the Rényi effective rank

```math
r_q =
\begin{cases}
\exp\!\left(-\sum_i p_i \log p_i\right), & q = 1, \\
\left(\sum_i p_i^q\right)^{1/(1-q)}, & q \ne 1,
\end{cases}
```

and the tail-sensitive score

```math
\chi_{\mathrm{nat}} = r_q \bigl(1 + \alpha (1 - p_1)\bigr),
```

where `p_1` is the largest normalized Schmidt weight. `adaptive_bonddim` then
applies the ratchet

```math
\chi_{\mathrm{new}} =
\min\!\left(\chi_{\max},
\max\!\left(\chi_{\mathrm{prev}}, \chi_{\min},
\left\lceil \chi_{\mathrm{nat}} \right\rceil\right)\right).
```

This package uses the following truncation language throughout:

- aggressive truncation means choosing a bond dimension that is too small and
  therefore introduces larger truncation error and lower fidelity;
- conservative truncation means keeping a larger bond dimension, usually
  preserving fidelity better at higher cost.

With that convention:

- `q = 1` is the entropy-rank rule and is the default;
- `q = 2` is the participation ratio / IPR and is more aggressive;
- `q < 1` is more conservative because it reacts more strongly to small Schmidt
  values;
- larger `alpha` is more conservative because it explicitly protects long
  tails;
- `alpha = 0` disables the tail-amplification factor.

The default `q = 1.0, alpha = 0.1` is a mild, fidelity-oriented setting. It
keeps the entropy-rank baseline while only slightly enlarging the recommended
bond dimension when the Schmidt spectrum develops a broad tail.

To enable adaptive growth, pass `chi_policy=:adaptive` together with the same
gate list:

```@example
using iTEBD
using LinearAlgebra

X = sqrt(2) / 2 * [0 1 0; 1 0 1; 0 1 0]
Y = sqrt(2) / 2 * 1im * [0 -1 0; 1 0 -1; 0 1 0]
Z = [1 0 0; 0 0 0; 0 0 -1]

H = begin
    SS = kron(X, X) + kron(Y, Y) + kron(Z, Z)
    0.5 * SS + SS^2 / 6 + I / 3
end

G = exp(-0.1 * H)
psi = rand_iMPS(ComplexF64, 2, 3, 1)
gates = [(G, 1, 2), (G, 2, 1)]

evolve!(psi, gates, 10; chi_policy=:adaptive, maxdim=16)

maximum(length.(psi.λ))
```

## Imaginary-time AKLT example

```@example
using iTEBD
using LinearAlgebra

X = sqrt(2) / 2 * [0 1 0; 1 0 1; 0 1 0]
Y = sqrt(2) / 2 * 1im * [0 -1 0; 1 0 -1; 0 1 0]
Z = [1 0 0; 0 0 0; 0 0 -1]

H = begin
    SS = kron(X, X) + kron(Y, Y) + kron(Z, Z)
    0.5 * SS + SS^2 / 6 + I / 3
end

G = exp(-0.1 * H)
psi = rand_iMPS(ComplexF64, 2, 3, 1)

for _ in 1:10
    applygate!(psi, G, 1, 2; maxdim=8)
    applygate!(psi, G, 2, 1; maxdim=8)
end

inner_product(psi)
```

The generated function reference lives on [API Reference](api.md).
