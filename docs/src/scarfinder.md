# ScarFinder

`iTEBD.jl` includes a general ScarFinder workflow for low-entanglement state
searches.

## Available Interfaces

Two public interfaces are especially useful:

- `scarfinder!(ψ, h, dt, χ, N; ...)` for Hamiltonian-based updates,
- `scarfinder!(ψ, G, h, χ, N; ...)` for gate-based updates with Hamiltonian
  energy fixing.

The mixed `gate + Hamiltonian` form is particularly useful for constrained
models such as PXP, where the evolution rule is naturally written as a
projected gate.

## PXP Example

```@example
using iTEBD
using LinearAlgebra

P0 = [0 0; 0 1]
N1 = [1 0; 0 0]
X = [0 1; 1 0]

h_pxp = kron(P0, X, P0)
no_double_2 = Matrix{Float64}(I, 4, 4) - kron(N1, N1)
proj_pxp = kron(no_double_2, I(2)) * kron(I(2), no_double_2)

dt = 0.01
G = proj_pxp * exp(-1im * dt * h_pxp)
psi = product_iMPS(ComplexF64, [[0, 1], [1, 0], [0, 1], [1, 0]])
target = energy_density(psi, h_pxp; span=3)

typeof(target)
```

## Notes

- If one ScarFinder iteration should represent a larger interval than the
  microscopic gate time step, choose `nstep ≈ Δt / dt`.
- Keeping `G` and `h` separate in the mixed interface is intentional: the
  projector used in the ScarFinder gate need not be identical to the projector
  entering the Hamiltonian density.

The generated function reference lives on [API Reference](api.md).
