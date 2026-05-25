<p align="center">
  <img src="docs/src/assets/logo.svg" alt="iTEBD.jl" width="180"/>
</p>

<h1 align="center">iTEBD.jl</h1>

<p align="center">
  <em>Infinite time-evolving block decimation for translationally invariant 1D quantum systems.</em>
</p>

<p align="center">
  <a href="https://jayren3996.github.io/iTEBD.jl/dev/"><img alt="Documentation (dev)" src="https://img.shields.io/badge/docs-dev-blue.svg"/></a>
  <a href="https://github.com/jayren3996/iTEBD.jl/actions/workflows/documentation.yml"><img alt="Documentation build status" src="https://github.com/jayren3996/iTEBD.jl/actions/workflows/documentation.yml/badge.svg"/></a>
  <a href="https://julialang.org/"><img alt="Julia" src="https://img.shields.io/badge/made%20with-Julia-9558B2.svg?logo=julia"/></a>
</p>

---

`iTEBD.jl` targets the thermodynamic limit directly. Instead of running on a long finite chain, you specify a periodic unit cell and the package works with the resulting infinite matrix-product state. That makes it a natural fit when the quantity of interest (entanglement structure, bulk energy density, scar trajectories) is defined per unit cell rather than for a particular system size.

The package stays narrow on purpose. It does not implement finite-size DMRG, mixed boundary conditions, or symmetric tensors, and it assumes the injective setting throughout canonicalization. If you need any of those, [`ITensors.jl`](https://github.com/ITensor/ITensors.jl) is a better starting point.

## Installation

From a Julia REPL:

```julia
pkg> add https://github.com/jayren3996/iTEBD.jl
```

Then load it with `using iTEBD`.

## Quick start

Imaginary-time iTEBD relaxes a random spin-1 state into the AKLT ground state in about a hundred Trotter steps:

```julia
using iTEBD, LinearAlgebra

# Spin-1 operators
X = sqrt(2)/2 * [0 1 0; 1 0 1; 0 1 0]
Y = sqrt(2)/2 * 1im * [0 -1 0; 1 0 -1; 0 1 0]
Z = [1 0 0; 0 0 0; 0 0 -1]

# AKLT bilinear-biquadratic Hamiltonian density
SS = kron(X, X) + kron(Y, Y) + kron(Z, Z)
H  = 0.5 * SS + SS^2 / 6 + I / 3

# Two-site unit cell, imaginary-time evolution
psi   = rand_iMPS(ComplexF64, 2, 3, 1)
gates = [(exp(-0.1 * H), 1, 2), (exp(-0.1 * H), 2, 1)]
evolve!(psi, gates, 300; maxdim=8)

energy_density(psi, H)         # → ≈ 0.0  (AKLT ground state)
maximum(length.(psi.λ))        # → 2      (converged bond dimension)
```

A walkthrough of this example with more diagnostics is on the [Time Evolution](https://jayren3996.github.io/iTEBD.jl/dev/time-evolution/) page.

## Features

- Infinite matrix-product states with arbitrary periodic unit cells.
- Local-gate updates with a discarded-weight truncation controller (`applygate!`, `evolve!`).
- Trotter helpers for second-order Strang splitting and two fourth-order schemes (`trotter_gates`).
- Schmidt canonicalization in the injective setting (`canonical!`).
- Transfer-matrix overlaps, multi-site expectation values, entanglement entropy, energy density (`inner_product`, `expect`, `ent_S`, `energy_density`, `energy_span`).
- ScarFinder workflow for low-entanglement state search, with Hamiltonian, gate, and mixed `gate + Hamiltonian` interfaces (`scarfinder!`).
- Optional adaptive bond-dimension policy as a ratchet on bond growth (`adaptive_bonddim`, `natural_bonddim`).

## iMPS convention

The single rule to remember: **the stored tensor has the right Schmidt values already multiplied in.** Formally, after `canonical!`,

```
B_i = Γ_i · λ_i
```

so `psi.Γ[i]` returns the right-canonical tensor `B_i`, not the bare Vidal `Γ_i`. To get the Vidal pair, index the state: `Γ_i, λ_i = psi[i]`. The [States and Canonical Form](https://jayren3996.github.io/iTEBD.jl/dev/imps/) page explains this in detail.

## ScarFinder

The package includes a general ScarFinder workflow that searches for low-entanglement, weakly-thermalizing trajectories directly on the iMPS manifold. Each iteration evolves the state for a short real-time interval, projects back to a target bond dimension `χ`, and optionally applies a small imaginary-time correction to hold the energy density near a chosen target.

Three interfaces are exposed:

```julia
scarfinder!(ψ, h, dt, χ, N; ...)    # Hamiltonian-based
scarfinder!(ψ, G, χ, N; ...)        # gate-based, no energy correction
scarfinder!(ψ, G, h, χ, N; ...)     # mixed: custom gate G, energy fixed against h
```

The mixed form is the recommended one for constrained models like PXP, where the gate carries projectors that repair truncation artifacts while the energy target is defined against the unprojected Hamiltonian density. See the [ScarFinder Workflow](https://jayren3996.github.io/iTEBD.jl/dev/scarfinder/) page for the complete PXP example.

## Documentation

Full manual at <https://jayren3996.github.io/iTEBD.jl/dev/>.

- [Getting Started](https://jayren3996.github.io/iTEBD.jl/dev/getting-started/) — installation, the first state, sanity checks.
- [States and Canonical Form](https://jayren3996.github.io/iTEBD.jl/dev/imps/) — the storage convention, `canonical!`, runnable examples.
- [Time Evolution](https://jayren3996.github.io/iTEBD.jl/dev/time-evolution/) — `applygate!`, `evolve!`, Trotter order, adaptive bond dimension.
- [Observables](https://jayren3996.github.io/iTEBD.jl/dev/observables/) — overlaps, expectation values, entropy, energy density.
- [ScarFinder Workflow](https://jayren3996.github.io/iTEBD.jl/dev/scarfinder/) — the three interfaces, the two time scales, the PXP example.
- [API Reference](https://jayren3996.github.io/iTEBD.jl/dev/api/) — generated from docstrings.

The `examples/` directory contains runnable Jupyter notebooks: `CanonicalForm.ipynb`, `AKLT_GS.ipynb`, `PXP.ipynb`, `PXP_ScarFinder.ipynb`.

## Citation

If the ScarFinder routines are useful in your published work, please cite:

```bibtex
@article{Ren2025ScarFinder,
  title   = {ScarFinder: A Detector of Optimal Scar Trajectories in
             Quantum Many-Body Dynamics},
  author  = {Ren, Jie and Hallam, Andrew and Ying, Lei and Papi\'c, Zlatko},
  journal = {PRX Quantum},
  volume  = {6},
  pages   = {040332},
  year    = {2025},
  doi     = {10.1103/PRXQuantum.6.040332},
}
```

## Acknowledgements

`iTEBD.jl` builds on [`ITensors.jl`](https://github.com/ITensor/ITensors.jl) / [`ITensorMPS.jl`](https://github.com/ITensor/ITensorMPS.jl) for tensor-network primitives, [`TensorOperations.jl`](https://github.com/Jutho/TensorOperations.jl) for contractions, and [`KrylovKit.jl`](https://github.com/Jutho/KrylovKit.jl) for the dominant transfer-matrix eigenvalue.
