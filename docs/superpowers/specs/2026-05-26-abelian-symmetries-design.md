# Abelian Symmetric Tensors in iTEBD.jl — Design

**Status:** Approved design, ready for implementation planning
**Date:** 2026-05-26
**Author:** Brainstormed with Claude, decisions by JieRen

## Goal

Add Abelian symmetric-tensor support to `iTEBD.jl` so users can run iTEBD on symmetric infinite MPS — XXZ in fixed Sz, spinful Hubbard in fixed `(N, Sz)`, transverse-field Ising in fixed Z₂ parity, Z_N clock models — with a flat user-facing API that does not require prior TensorKit experience.

The scope of "v1" is the core iTEBD pipeline only: canonicalization, gate evolution, basic observables. ScarFinder and ITensor interop remain dense-only.

## Why this, why now

`iTEBD.jl` is currently the only Julia package with a native iTEBD kernel, but every comparable library in the field (TeNPy, MPSKit.jl, ITensorInfiniteMPS.jl, Cytnx, block2) supports at least U(1) symmetric tensors. Working in a fixed sector typically gives a 5–50× speed/memory win at the same physical χ, and it is the cleanest mathematical setting for several physics targets the package already cares about (Heisenberg, XXZ, Hubbard). Closing this gap is the single highest-leverage feature addition identified in the 2026-05-26 competitive analysis.

## Scope

**In scope for v1:**

- Symmetric infinite MPS via `TensorKit.jl` as an optional backend
- Abelian symmetries: U(1), Z_N, and arbitrary Abelian products (U(1)×U(1), U(1)×Z₂, …)
- Core iTEBD pipeline: `canonical!`, `applygate!`, `evolve!`, `trotter_gates`, `expect`, `ent_S`, `energy_density`, `rand_iMPS`, `product_iMPS`
- A small helper layer so users do not need to import TensorKit types
- A new documentation page that explains symmetric tensors from zero, including the meaning of "flux" and arrow conventions

**Out of v1 (revisit later):**

- SU(2) and other non-Abelian symmetries
- Fermion parity / graded vector spaces
- Symmetric ScarFinder
- TensorKit ↔ ITensorMPS bridge for symmetric states
- Auto-Hamiltonian / model library (users are pointed at `MPSKitModels.jl`)
- Non-injective canonical form on the symmetric backend

## Architecture

### Parametric `iMPS`

The current struct (in `src/iMPS.jl:34-67`) is:

```julia
struct iMPS{T<:Number, S<:Real}
    Γ::Vector{Array{T,3}}
    λ::Vector{Vector{S}}
    n::Int
end
```

It is changed to parameterise on the tensor types themselves, not on element types:

```julia
struct iMPS{ΓT, λT}
    Γ::Vector{ΓT}
    λ::Vector{λT}
    n::Int
end

const DenseIMPS{T<:Number,S<:Real} = iMPS{Array{T,3}, Vector{S}}
const SymmetricIMPS                = iMPS{<:TensorMap, <:DiagonalTensorMap}
```

Existing constructors and method signatures that today reference `iMPS{T,S}` are migrated to the new parameterisation; user-facing call sites that did not rely on the element-type parameters are unaffected.

### The 6-method dispatch waist

All iTEBD algorithms route through six tensor-algebra primitives, each implemented twice (dense `Array`, symmetric `TensorMap`):

| Method | Purpose |
| --- | --- |
| `_lmul_diag!(λ, Γ)`      | apply Schmidt values on left bond |
| `_rmul_diag!(Γ, λ)`      | apply Schmidt values on right bond |
| `_apply_gate(G, Γs)`     | contract a k-site gate with k contiguous tensors |
| `_svd_split(B; maxdim, cutoff)` | truncated SVD into `(U, S, V)` |
| `_rand_tensor(spec...)`  | random local tensor with the right legs |
| `_product_tensor(spec...)` | local tensor for a product state |

**Files that change:** `src/iMPS.jl`, `src/TensorAlgebra.jl`, `src/Schmidt.jl`, `src/Gate.jl`, `src/Contractions.jl`, `src/Krylov.jl`.

**Files that do not change:** `src/ScarFinder.jl` (stays dense-only — explicit scope cut), `src/ITensorsInterop.jl`, `src/Miscellaneous.jl`.

## Data model and user-facing API

### Sectors live in TensorKit, not in our types

The `iMPS{TensorMap,…}` value does not carry a `Sector` type parameter. TensorKit already encodes the sector type in each tensor's leg structure, so we do not introduce a parallel sector abstraction.

Dense path (unchanged):

```julia
ψ = rand_iMPS(ComplexF64, 2, 3, 1)            # n=2 sites, d=3, χ=1
```

Symmetric path (new):

```julia
using TensorKit
P = Vect[U1Irrep](-1=>1, 1=>1)                # spin-1/2, Sz=±1
V = Vect[U1Irrep](0=>2, 2=>1, -2=>1)          # virtual leg
ψ = rand_iMPS(P, V, 2)                        # 2-site unit cell, flux=0
```

Dispatch on the first positional argument: `AbstractVectorSpace` selects the symmetric path, `Type{<:Number}` selects the dense path.

### Schmidt values: `DiagonalTensorMap` for the symmetric case

TensorKit's `DiagonalTensorMap` stores singular values together with their sector labels and is the natural output of `tsvd`. Using it avoids maintaining a parallel `Vector{Vector{Sector}}` for labels alongside `Vector{Vector{Float64}}`. Public observables that today consume `Vector{Float64}` (`ent_S`, `natural_bonddim`) get a thin adapter that extracts the singular values; the existing formula code is reused.

### Gates: TensorMap on the symmetric path

`trotter_gates(layers, dt)` requires only that `exp(coeff * h)` is defined; TensorKit supports this. On the symmetric path, the user supplies `h` as a `TensorMap{T,2,2}` instead of a dense matrix. We do not build Hamiltonians for the user beyond the spin-1/2 micro-helper described below.

### Wraparound flux check

Today, the constructor at `src/iMPS.jl:50-59` validates bond dimensions only. On the symmetric path, the right-leg vector space of `Γ[n]` must equal the left-leg vector space of `Γ[1]` (not just match in total dimension). This is added as one extra check; failure throws a `DimensionMismatch` with a sector-level diagnostic.

## Helper layer

Five helpers live in a Julia package extension `ext/iTEBDTensorKitExt.jl`, loaded automatically when the user does `using TensorKit` (Julia ≥ 1.9). Each is small and removes a concrete friction point for users who do not know TensorKit. The extension is declared in `Project.toml` under `[weakdeps]` / `[extensions]`; the base package gains no hard dependency on TensorKit.

```julia
# 1) Symbol → graded-space, avoiding direct TensorKit imports
graded_space(:U1,   0=>2, 1=>1, -1=>1)        # ≡ Vect[U1Irrep](0=>2, 1=>1, -1=>1)
graded_space(:Z2,   0=>3, 1=>3)
graded_space(:ZN,   4, 0=>1, 1=>1, 2=>1, 3=>1)
graded_space(:U1xU1, (0,0)=>2, (1,1)=>1)

# 2) High-level symmetric rand_iMPS, auto-distributing χ across sectors
ψ = rand_iMPS(:U1, [-1, 1]; χ=8, n=2, flux=0)

# 3) Symmetric product state from occupation labels
ψ = product_iMPS(:U1, [-1, 1], [1, -1])       # Néel state in Sz=0 sector

# 4) Uniform Schmidt-value accessor, both backends
λvals = schmidt_values(ψ, i)                  # always Vector{Float64}

# 5) Spin-1/2 operator micro-set (the only operator helper)
Sz, Sp, Sm     = spin_half_ops(:U1)
Sx, Sy, Sz, Sp, Sm, Id = spin_half_ops(:Trivial)
```

**What is deliberately not a helper:** model builders (`xxz_hamiltonian`, …) — outside scope and already provided by `MPSKitModels.jl`; gate wrappers around `exp(-1im*dt*h)`; QN string DSLs; pretty-printers.

End-to-end target flow on the symmetric path:

```julia
P = graded_space(:U1, -1=>1, 1=>1)
ψ = rand_iMPS(:U1, [-1, 1]; χ=8, n=2)
Sz, Sp, Sm = spin_half_ops(:U1)
h = Jz*(Sz⊗Sz) + 0.5*Jxy*(Sp⊗Sm + Sm⊗Sp)
gates = trotter_gates([(h, [1,2]), (h, [2,1])], 0.05)
evolve!(ψ, gates, 300; maxdim=16)
```

The user writes no `using TensorKit`; the unavoidable TensorKit-isms are `⊗` when composing operators and the act of thinking in charges, both inherent to the problem.

## Documentation deliverables

A new manual page **`docs/src/symmetries.md`**, slotted between `imps.md` and `time-evolution.md` in `docs/make.jl`'s nav. Written for a reader who has used `iTEBD.jl` on dense tensors but has never worked with symmetric tensors.

The page covers, in this order:

1. **What a symmetry buys you.** One paragraph, concrete (XXZ + Sz example), no abstract algebra.
2. **Sectors, charges, graded spaces, defined from zero.** "Charge" / "sector" / "irrep" / "QN" are all the same thing; a graded vector space is a vector space whose basis vectors carry charge labels. Worked example on the spin-1/2 physical leg.
3. **Flux, explained pedagogically.** Reproduced here for clarity since it is the term the user flagged as needing care:

   > **What is the "flux" of a tensor?**
   >
   > Every leg of a symmetric tensor carries a charge label on each of its basis states. When you contract two legs together, the charges on the contracted basis states must match — otherwise the matrix element is zero by symmetry. The *flux* of a tensor is the **net charge it carries between its incoming and outgoing legs**: how much charge goes *in* on one side minus how much goes *out* the other.
   >
   > A tensor with **flux = 0** is the most common case — it neither creates nor destroys charge. Examples: the identity, a U(1)-symmetric Hamiltonian density `Sz⊗Sz`, an MPS tensor at the ground state of an Sz-conserving model.
   >
   > A tensor with **flux ≠ 0** *moves* charge. `S+` has flux `+2` (in the 2×Sz convention) because it raises spin. A flux-`q` MPS tensor inserts `q` units of total Sz at that site.
   >
   > In `iTEBD.jl`, when you build an iMPS in a fixed-Sz sector you set every MPS tensor to flux=0, and the wraparound bond closes onto itself with consistent charges. If you wanted to study a state with a single magnon (one extra Sz=+1), you'd put one flux=+2 site somewhere in the unit cell.

   Backed by a worked example showing the flux of `Sz`, `S+`, `S-`, and a spin-1/2 MPS site.
4. **Arrow convention on diagrams**, defined from zero and reused throughout the rest of the manual:

   > **Arrow convention.** Every leg in our diagrams has an arrow.
   >
   > - An arrow pointing **into** the tensor means that leg's charges are read *as given*.
   > - An arrow pointing **out** of the tensor means that leg's charges are read *negated* (mathematically: this leg lives in the dual space).
   >
   > **Why this matters.** When you connect two legs together, the arrows have to be **consistent**: one arrow leaves one tensor, the other arrow enters the next tensor. (Connecting "out" to "out" or "in" to "in" would sum charges that should subtract — TensorKit raises an error.)
   >
   > **The standard MPS convention used in this package:** physical legs point *out* (kets), bonds point *right* (left bond into the tensor, right bond out of it). The flux equation reads `(in charges) − (out charges) = flux`.

   A canonical SVG diagram in `docs/src/assets/` shows a 2-site iMPS tensor with arrows drawn and the flux equation written underneath. Other manual pages reference it rather than redefining the convention.
5. **End-to-end walkthrough.** The exact XXZ flow above, run, with the output of `schmidt_values` and the per-sector block structure printed so the reader sees what the symmetric path actually produces.
6. **Common errors and what they mean.** Three or four real error messages (mismatched spaces at the wraparound, gate flux ≠ 0 when expected, dimension-0 sector after truncation) with one-sentence explanations.

A smaller edit to **`docs/src/imps.md`** adds a one-paragraph "Symmetric variant" callout pointing at the new page so users discover it from existing navigation.

Initial diagrams may ship as ASCII art committed alongside the prose; SVG upgrades are a follow-up doc pass and do not block v1.

## Testing strategy

Three concrete success criteria, each runnable and binary-pass:

| Goal | Check | Where |
| --- | --- | --- |
| Dense regression | All existing tests in `test/` pass unchanged after the parametric refactor; zero numerical drift on the dense path | CI |
| Block-structure correctness | After 300 imaginary-time iTEBD steps on the spin-1/2 Heisenberg chain in `:U1` mode, each Schmidt spectrum lies entirely in the `Sz=0` sector with the expected χ-per-sector pattern | new `test/symmetric/heisenberg_sectors.jl` |
| Quantitative golden | The same Heisenberg run reproduces the known ground-state energy density `-ln(2) + 1/4 ≈ -0.4431` to four digits at χ=32, and matches the dense run at the same χ to roughly SVD tolerance | same file |

**Stopping rule:** v1 is done when all three goals are green on a clean CI run.

## Risks and mitigations

1. **The 6-method waist might leak.** If `src/Schmidt.jl`'s fixed-point solver, or any other internal routine, currently calls `reshape` or other `Array`-specific operations outside the six dispatch primitives, the waist is not actually narrow. *Mitigation:* the first task of implementation is to audit every `Array`-specific call site and confirm it routes through the six. If it does not, the method set grows and the design is revisited here rather than papered over.
2. **`DiagonalTensorMap` interop.** The plan assumes `DiagonalTensorMap` exposes singular values cleanly enough for `ent_S` / `natural_bonddim`. If it does not, the adapter widens. *Mitigation:* spike this early in implementation; if gnarly, fall back to `Vector{Float64}` + companion sector labels and update this design.
3. **Package extension boilerplate.** Julia's weak-deps / extensions feature (1.9+) has some sharp edges. *Mitigation:* budget time for `Project.toml` configuration; no design change required.
4. **Doc diagrams.** SVG diagrams take real time to draw well. *Mitigation:* ship ASCII art first; SVG follow-up does not block v1.

## Decisions recorded

| Decision | Choice | Considered alternative |
| --- | --- | --- |
| Symmetry scope | U(1), Z_N, Abelian products | U(1) only |
| Backend | TensorKit.jl as optional dependency | Hand-rolled block tensors |
| Integration | Parametric `iMPS{ΓT,λT}` | Parallel `SymmetricIMPS` type; full replacement |
| v1 algorithms | Core iTEBD only | Plus ScarFinder; plus ITensors interop |
| Helper layer | Five small helpers, no model builders | Zero helpers; full model library |
| Documentation | New `symmetries.md` page with pedagogical flux + arrow definitions | Brief docstring-only coverage |
