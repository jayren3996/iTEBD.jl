# iMPS and Canonical Form

## Stored Representation

An `iMPS` stores one periodic unit cell of an infinite matrix-product state:

- `Γ`: local tensors,
- `λ`: Schmidt values on each bond,
- `n`: number of sites in the unit cell.

The package uses a right-canonical storage convention in which the right
Schmidt values are absorbed into each local tensor:

```math
B_i = \Gamma_i \lambda_i.
```

That means `ψ.Γ[i]` stores the right-canonical tensor `B_i`, while `ψ[i]`
returns the bare Vidal tensor `Γ_i` together with `λ[i]`.

## Canonicalization

`canonical!(ψ)` is the standard normalization step used throughout the package.
It updates the state in place and stores the resulting Schmidt spectra in
`ψ.λ`.

```@example
using iTEBD

psi = rand_iMPS(ComplexF64, 1, 2, 4)
canonical!(psi)
Gamma1, lambda1 = psi[1]
size(Gamma1), lambda1
```

The generated reference for constructors and canonicalization routines lives on
[API Reference](api.md).
