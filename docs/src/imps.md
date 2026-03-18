# iMPS and Canonical Form

## What An `iMPS` Stores

An `iMPS` in this package stores one periodic unit cell of an infinite
matrix-product state:

- `ψ.Γ`
  A vector of local three-leg tensors, one for each site in the unit cell.
- `ψ.λ`
  A vector of Schmidt spectra, one for each bond in the unit cell.
- `ψ.n`
  The unit-cell length.

If the unit cell has sites `1, ..., n`, then bond `i` means the bond between
site `i` and site `i + 1`, with periodic wraparound at the end of the unit
cell.

## Important Convention: Stored `Γ` Is Not The Bare Vidal `Γ`

The most important caveat in this package is that the stored tensor `ψ.Γ[i]`
is **not** the bare Vidal tensor usually written as `Γ_i` in the iTEBD
literature.

Instead, after canonicalization the package stores the right-canonical tensor

```math
B_i = \Gamma_i \lambda_i,
```

where:

- `Γ_i`
  is the bare Vidal tensor for site `i`,
- `λ_i`
  is the Schmidt spectrum on the bond to the right of site `i`,
- `B_i`
  is the right-canonical tensor actually stored in `ψ.Γ[i]`.

So the package data layout is:

- `ψ.Γ[i] = B_i`
- `ψ.λ[i] = λ_i`

This is why the symbol `Γ` in the struct field should be read as "stored local
tensor" rather than automatically assuming it is the bare Vidal `Γ_i`.

## How To Recover The Bare Vidal Tensor

If you want the Vidal tensor and the Schmidt values in the more standard form,
index the state:

```julia
Gamma1, lambda1 = psi[1]
```

This returns:

- `Gamma1 = Γ_1`
- `lambda1 = λ_1`

Internally, `ψ[i]` divides the absorbed right Schmidt values back out of the
stored tensor `ψ.Γ[i]`.

That means there are two different but related objects:

- `psi.Γ[i]`
  right-canonical stored tensor `B_i`
- `psi[i][1]`
  bare Vidal tensor `Γ_i`

This distinction matters whenever you compare formulas in the literature with
what is stored in memory in `iTEBD.jl`.

## Canonical Form Used Here

After calling `canonical!(ψ)`, the state is in the package's Schmidt-canonical
representation:

- each stored local tensor `B_i = Γ_i λ_i` is right-canonical,
- each `ψ.λ[i]` stores the Schmidt values across bond `i`,
- the entanglement structure is therefore carried explicitly by `ψ.λ`.

Operationally, this means:

- contractions such as `expect(ψ, ...)` are performed directly with the stored
  tensors `B_i`,
- bare Vidal tensors are reconstructed only when explicitly requested through
  `ψ[i]`,
- local gates can be applied in the stored representation and the result can
  then be brought back to Schmidt-canonical form.

## What `canonical!` Does

The canonicalization entry point is:

```julia
canonical!(ψ; maxdim=MAXDIM, cutoff=SVDTOL, renormalize=true)
```

This routine:

1. groups the unit cell into a single periodic object,
2. computes the left and right fixed-point gauges needed for a Schmidt-canonical
   representation,
3. performs an SVD on the effective bond problem,
4. truncates according to `maxdim` and `cutoff`,
5. stores the resulting right-canonical tensors back into `ψ.Γ` and the Schmidt
   spectra back into `ψ.λ`.

It mutates `ψ` in place and returns the same object.

## Meaning Of The `canonical!` Arguments

The keyword arguments control both gauge-fixing and truncation:

- `maxdim=MAXDIM`
  Maximum number of Schmidt values kept on each bond during canonicalization.
  This is a hard bond-dimension cap.
- `cutoff=SVDTOL`
  Singular values smaller than this threshold are discarded.
- `renormalize=true`
  Whether to renormalize the retained Schmidt values after truncation.

In practice:

- use `maxdim` when you want to limit the bond dimension explicitly,
- use `cutoff` when you want to suppress very small Schmidt values,
- leave `renormalize=true` in normal use, especially after truncation.

## Constructor Behavior

The constructors:

```julia
iMPS(Γs; renormalize=true)
iMPS(T, Γs; renormalize=true)
```

assume the input tensors are raw local tensors and initialize the Schmidt
vectors as trivial all-ones placeholders before canonicalizing.

So:

- `renormalize=true`
  means "construct the state and immediately bring it to the package's
  Schmidt-canonical form".
- `renormalize=false`
  means "store the tensors as given, with placeholder Schmidt vectors".

The latter is mainly useful for controlled internal workflows; most users should
keep the default `renormalize=true`.

## Single-Site Example

```@example
using iTEBD

psi = rand_iMPS(ComplexF64, 1, 2, 4)
canonical!(psi; maxdim=4, cutoff=1e-12, renormalize=true)

stored_B = psi.Γ[1]
Gamma1, lambda1 = psi[1]

(;
    stored_size=size(stored_B),
    vidal_size=size(Gamma1),
    lambda=lambda1,
)
```

In this example:

- `stored_B` is the right-canonical tensor actually stored in the struct,
- `Gamma1` is the bare Vidal tensor reconstructed by indexing,
- `lambda1` is the Schmidt spectrum on the bond to the right of the site.

## Practical Caveats

- When reading papers, check whether `Γ_i` means the bare Vidal tensor or a
  tensor with one of the Schmidt vectors absorbed. In this package the stored
  tensor is the absorbed version `B_i = Γ_i λ_i`.
- `canonical!` assumes the state is in the injective setting used throughout
  this package.
- If you canonicalize with a small `maxdim` or aggressive `cutoff`, then `ψ[i]`
  reconstructs the Vidal tensor for the **truncated** state, not for the
  pre-truncation state.

## References

For the standard iTEBD and canonical-form background, the most useful starting
points are:

- G. Vidal, [Classical Simulation of Infinite-Size Quantum Lattice Systems in
  One Spatial Dimension](https://link.aps.org/doi/10.1103/PhysRevLett.98.070201),
  Physical Review Letters 98, 070201 (2007).
- R. Orús and G. Vidal, [Infinite Time-Evolving Block Decimation Algorithm
  Beyond Unitary Evolution](https://link.aps.org/doi/10.1103/PhysRevB.78.155117),
  Physical Review B 78, 155117 (2008).
- U. Schollwöck, [The Density-Matrix Renormalization Group in the Age of Matrix
  Product States](https://doi.org/10.1016/j.aop.2010.09.012), Annals of Physics
  326, 96-192 (2011).

The generated reference for constructors and canonicalization routines lives on
[API Reference](api.md).
