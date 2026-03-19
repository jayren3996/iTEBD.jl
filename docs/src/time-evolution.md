# Time Evolution

Time evolution is implemented by repeatedly applying local gates to the unit
cell and re-canonicalizing the updated bonds.

## Local Gate Updates

The main in-place update is:

```julia
applygate!(ψ, G, i, j; maxdim=MAXDIM, cutoff=SVDTOL, renormalize=true)
```

where `i:j` specifies the gate support inside the periodic unit cell.

For repeated sweeps, you can also use:

```julia
evolve!(ψ, gates, steps; chi_policy=:fixed, maxdim=MAXDIM, ...)
```

where each element of `gates` is a tuple `(G, i, j)`.

## Adaptive Bond Dimension

For adaptive real- or imaginary-time evolution, `iTEBD.jl` provides two helper
functions:

- `natural_bonddim(λ; q=1.0, alpha=0.1)` estimates a smooth intrinsic bond
  dimension from a Schmidt spectrum `λ`.
- `adaptive_bonddim(previous, λ; mindim, maxdim, ...)` turns that score into a
  non-decreasing bond dimension bounded between `mindim` and `maxdim`.

The first helper is defined from the normalized Schmidt weights

```math
p_i = \frac{|\lambda_i|^2}{\sum_j |\lambda_j|^2},
```

through the Rényi effective rank

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

where `p_1` is the largest normalized Schmidt weight. The second helper turns
that smooth score into an actual working bond dimension through the ratchet

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

The default `q = 1.0, alpha = 0.1` is chosen as a mild, fidelity-oriented
setting. It keeps the entropy-rank baseline while only slightly enlarging the
recommended bond dimension when the Schmidt spectrum develops a broad tail.

The high-level wrapper accepts the same gate list together with the adaptive
policy:

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

## Imaginary-Time AKLT Example

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

size.(psi.Γ)
```

The generated function reference lives on [API Reference](api.md).
