# iTEBD.jl

`iTEBD.jl` is now just a compatibility wrapper around
[`InfiniteTEBD.jl`](https://github.com/jayren3996/InfiniteTEBD.jl). It contains
no implementation code of its own; all algorithms, documentation, examples, and
future development live in `InfiniteTEBD.jl`.

The old implementation has moved into the registered `InfiniteTEBD` package.
This repository keeps the historical package name working for code that still
does:

```julia
using iTEBD
```

New projects should use:

```julia
pkg> add InfiniteTEBD
```

```julia
using InfiniteTEBD
```

## Installation

For existing projects that still depend on the old URL package name:

```julia
pkg> add https://github.com/jayren3996/iTEBD.jl
```

This installs `iTEBD` plus its sole runtime dependency, `InfiniteTEBD`.

## Compatibility

The wrapper re-exports the `InfiniteTEBD` API and forwards qualified access for
the compatibility names that users historically reached as `iTEBD.foo`.
For example:

```julia
using iTEBD

psi = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
inner_product(psi)

iTEBD.tensor_decomp! === InfiniteTEBD.tensor_decomp!
```

See [`InfiniteTEBD.jl`](https://github.com/jayren3996/InfiniteTEBD.jl) for the
core package, documentation, examples, benchmarks, and future development.
