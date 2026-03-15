# Time Evolution

Time evolution is implemented by repeatedly applying local gates to the unit
cell and re-canonicalizing the updated bonds.

## Local Gate Updates

The main in-place update is:

```julia
applygate!(ψ, G, i, j; maxdim=MAXDIM, cutoff=SVDTOL, renormalize=true)
```

where `i:j` specifies the gate support inside the periodic unit cell.

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
