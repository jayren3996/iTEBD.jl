# Getting Started

## Installation

Install the package from a Julia REPL:

```julia
pkg> add https://github.com/jayren3996/iTEBD.jl
```

Then load it with:

```julia
using iTEBD
```

## Create Your First `iMPS`

```@example
using iTEBD

psi = rand_iMPS(ComplexF64, 2, 2, 4)
```

This creates a random two-site unit-cell state with local dimension `2` and
bond dimension `4`.

You can also build a product state directly:

```@example
using iTEBD

z2 = product_iMPS(ComplexF64, [[0, 1], [1, 0], [0, 1], [1, 0]])
```

## Examples In The Repository

The [`examples/`](https://github.com/jayren3996/iTEBD.jl/tree/master/examples)
directory contains runnable notebooks for the stable workflows:

- `CanonicalForm.ipynb`
- `AKLT_GS.ipynb`
- `PXP.ipynb`
- `PXP_ScarFinder.ipynb`

Each notebook begins with:

```julia
import Pkg
Pkg.activate("..")
```

so it can be run directly from a repository checkout.
