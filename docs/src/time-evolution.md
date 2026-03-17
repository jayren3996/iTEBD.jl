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

- `natural_bonddim(λ; q=1.5, alpha=0.5)` estimates a smooth intrinsic bond
  dimension from a Schmidt spectrum `λ`.
- `adaptive_bonddim(previous, λ; mindim, maxdim, ...)` turns that score into a
  non-decreasing bond dimension bounded between `mindim` and `maxdim`.

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

The default `q = 1.5` interpolates between entropy rank (`q -> 1`) and the
participation ratio (`q = 2`). Larger `alpha` makes the score more sensitive to
long Schmidt tails.

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
