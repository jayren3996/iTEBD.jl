# Abelian Symmetric Tensors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add U(1)/Z_N/Abelian-product symmetric infinite MPS support to iTEBD.jl via `TensorKit.jl` as an optional backend, with a small helper layer so users do not need to learn TensorKit's API, and a pedagogical documentation page.

**Architecture:** Make `iMPS` parametric on its tensor types, route all algorithms through six dispatch primitives, and put the TensorKit-specific implementations in a Julia package extension (`ext/iTEBDTensorKitExt.jl`) loaded only when the user does `using TensorKit`. The base package gains zero hard dependencies; existing dense-Array users see no API change.

**Tech Stack:** Julia ≥ 1.9, TensorKit.jl (weak dep), Test stdlib, Documenter.jl.

**Spec:** [docs/superpowers/specs/2026-05-26-abelian-symmetries-design.md](../specs/2026-05-26-abelian-symmetries-design.md).

---

## File Structure

**Create:**
- `ext/iTEBDTensorKitExt.jl` — package extension, holds all TensorKit-specific methods and helpers (single file; split if it grows past ~600 lines).
- `docs/src/symmetries.md` — pedagogical docs page (flux, arrows, U(1) walkthrough).
- `docs/src/assets/symmetric-tensor-arrows.svg` — diagram referenced from the docs page (initial version ships as ASCII inline; SVG follow-up acceptable).
- `test/test_symmetric_basic.jl` — unit-level smoke tests for helpers and symmetric constructors.
- `test/test_symmetric_heisenberg.jl` — golden test: imaginary-time XXZ at fixed Sz reproduces known energy.

**Modify:**
- `Project.toml` — add `[weakdeps]`, `[extensions]`, compat entry for TensorKit.
- `src/iMPS.jl` — parameterise `iMPS` on `(ΓT, λT)`, add `DenseIMPS` / `SymmetricIMPS` aliases, route every internal callsite via the new typevars.
- `src/iTEBD.jl` — export new helper names (`graded_space`, `spin_half_ops`, `schmidt_values`) as no-op stubs that error in the base module; the extension shadows them with real methods.
- `docs/src/imps.md` — add a one-paragraph "Symmetric variant" callout.
- `docs/make.jl` — add `symmetries.md` to the nav.
- `test/runtests.jl` — include the two new test files in the `unit` and `integration` groups.

**Do not touch:**
- `src/ScarFinder.jl`, `src/ITensorsInterop.jl`, `src/Miscellaneous.jl` — explicit scope cut.
- `src/Krylov.jl`, `src/Contractions.jl`, `src/TensorAlgebra.jl` — dense paths are unchanged in v1. The symmetric backend re-implements the matching operations using TensorKit primitives in the extension file rather than refactoring the dense files. (Risk #1 in the spec is mitigated by *not* trying to share kernels between backends in v1.)

**Deviation from spec — 6-method dispatch waist.** The spec calls for "all iTEBD algorithms route through a 6-method dispatch interface", each implemented twice (dense and symmetric). After reading `src/TensorAlgebra.jl` we judged this refactor too invasive for v1: the dense code uses `reshape`-heavy fusion and `@tensor`-macro contractions inlined throughout `tensor_decomp!`, `apply_transfer`, etc. The spec's Risk #1 anticipated this outcome — "If [the waist leaks], the method set grows, and the design is revisited here rather than papered over." This plan implements that revision: instead of a shared interface, the extension specialises `canonical!`, `applygate!`, `expect`, `energy_density`, `rand_iMPS`, `product_iMPS` directly on `SymmetricIMPS` using TensorKit primitives. If a future refactor justifies the waist, it lands as a separate piece of work.

---

## Chunk 1: Parametric `iMPS` struct (no behavior change)

The goal of this chunk is to make `iMPS` carry tensor-type parameters instead of element-type parameters, *without* changing what dense users see. Every existing test must remain green.

### Task 1.1: Read current iMPS callsites and capture method signatures

**Files:**
- Read: `src/iMPS.jl`, `src/Schmidt.jl`, `src/Gate.jl`, `src/Contractions.jl`, `src/Krylov.jl`, `src/ScarFinder.jl`, `src/ITensorsInterop.jl`, `src/Miscellaneous.jl`.

- [ ] **Step 1: Grep every callsite that pattern-matches on `iMPS{T,S}`**

```bash
grep -rn 'iMPS{' src/ | tee /tmp/imps-param-sites.txt
```
Expected: a list of definitions like `function ent_S(mps::iMPS, i::Integer)`, `function getindex(mps::iMPS{T,S}, ...) where {T,S}`, etc.

- [ ] **Step 2: For each match, write down whether the body uses `T` or `S`**

If a method body never references `T` or `S` outside the signature, the `{T,S}` parameterization is decorative there and can be dropped to `::iMPS`. Save this list to `/tmp/imps-param-sites.txt` (annotated by hand).

Why this matters: the parametric refactor in Task 1.3 replaces `{T,S}` with `{ΓT,λT}`. Any body that needs the *element* type must derive it from `eltype(ψ.Γ[1])`, not from a type parameter.

### Task 1.2: Add failing test that locks the new struct shape

**Files:**
- Create: `test/test_symmetric_struct.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Write the failing test**

```julia
# test/test_symmetric_struct.jl
using Test
using iTEBD

@testset "iMPS parametric struct" begin
    @testset "dense alias" begin
        ψ = rand_iMPS(ComplexF64, 2, 2, 3)
        @test ψ isa iTEBD.DenseIMPS
        @test ψ isa iTEBD.DenseIMPS{ComplexF64, Float64}
        @test eltype(ψ.Γ[1]) === ComplexF64
        @test eltype(ψ.λ[1]) === Float64
    end

    @testset "struct exposes Γ, λ, n fields" begin
        ψ = rand_iMPS(ComplexF64, 2, 2, 3)
        @test fieldnames(iMPS) === (:Γ, :λ, :n)
        @test ψ.n == 2
    end
end
```

- [ ] **Step 2: Wire the file into `test/runtests.jl`**

```julia
# test/runtests.jl — add the file under the "unit" group
const TEST_GROUPS = Dict(
    "unit" => [
        # ... existing entries unchanged ...
        "test_symmetric_struct.jl",
    ],
    # ... rest unchanged ...
)
```

- [ ] **Step 3: Run test to verify it fails**

```bash
julia --project=. -e 'using Pkg; Pkg.test(test_args=["test_symmetric_struct.jl"])'
```
Expected: FAIL — `iTEBD.DenseIMPS` does not exist yet.

- [ ] **Step 4: Commit the failing test**

```bash
git add test/test_symmetric_struct.jl test/runtests.jl
git commit -m "test: add failing parametric-struct test for iMPS"
```

### Task 1.3: Reshape the iMPS struct

**Files:**
- Modify: `src/iMPS.jl:34-67`.

- [ ] **Step 1: Replace the struct definition and inner constructor**

```julia
# src/iMPS.jl — replace the struct at lines 34-67 with:
struct iMPS{ΓT, λT}
    Γ::Vector{ΓT}
    λ::Vector{λT}
    n::Int

    function iMPS(
        Γ::Vector{ΓT},
        λ::Vector{λT},
        n::Integer,
    ) where {ΓT, λT}
        n > 0 || throw(ArgumentError(
            "iMPS unit-cell length n must be positive (got $n)"))
        length(Γ) == n || throw(ArgumentError(
            "iMPS: length(Γ) = $(length(Γ)) but n = $n"))
        length(λ) == n || throw(ArgumentError(
            "iMPS: length(λ) = $(length(λ)) but n = $n"))
        _validate_iMPS_bonds(Γ, λ, n)
        return new{ΓT, λT}(Γ, λ, Int(n))
    end
end

# Dense bond check — extracted from the previous inner constructor body.
function _validate_iMPS_bonds(
    Γ::Vector{<:AbstractArray{<:Number, 3}},
    λ::Vector{<:AbstractVector{<:Real}},
    n::Integer,
)
    for i in 1:n
        Dr_i = size(Γ[i], 3)
        Dl_next = size(Γ[mod1(i + 1, n)], 1)
        length(λ[i]) == Dr_i || throw(DimensionMismatch(
            "iMPS bond $i: length(λ[$i]) = $(length(λ[i])) " *
            "but size(Γ[$i], 3) = $Dr_i"))
        Dr_i == Dl_next || throw(DimensionMismatch(
            "iMPS bond $i: size(Γ[$i], 3) = $Dr_i but " *
            "size(Γ[$(mod1(i + 1, n))], 1) = $Dl_next " *
            "(bond dims must match at the wraparound seam)"))
        all(isfinite, λ[i]) || throw(ArgumentError(
            "iMPS λ[$i] contains non-finite values"))
        any(<(zero(eltype(λ[i]))), λ[i]) && throw(ArgumentError(
            "iMPS λ[$i] contains negative values"))
    end
    return nothing
end

# Fallback for any tensor type that has no specialised validator (e.g. TensorMap).
# The symmetric extension supplies its own method.
_validate_iMPS_bonds(Γ, λ, n) = nothing
```

- [ ] **Step 2: Add the two type aliases just after the struct**

```julia
# src/iMPS.jl — add immediately after the struct + validator definitions:
"""
    DenseIMPS{T,S}

Alias for the dense-array backend of [`iMPS`](@ref). `T` is the tensor element
type (typically `ComplexF64`); `S` is the Schmidt-value element type
(typically `Float64`).
"""
const DenseIMPS{T<:Number,S<:Real} = iMPS{Array{T,3}, Vector{S}}

# The symmetric alias is declared in the TensorKit extension, not here, to
# avoid forcing a TensorKit dependency on the base package.
```

- [ ] **Step 3: Update internal callsites that destructure `iMPS{T,S}`**

For every method signature in `src/iMPS.jl`, `src/Schmidt.jl`, `src/Gate.jl`, `src/Krylov.jl`, `src/Miscellaneous.jl`, `src/ScarFinder.jl`, `src/ITensorsInterop.jl` that reads:

```julia
function foo(mps::iMPS{T,S}, ...) where {T,S}
```

replace it with one of:

```julia
function foo(mps::DenseIMPS{T,S}, ...) where {T,S}   # if body uses T or S
function foo(mps::iMPS, ...)                          # if body never uses T or S
```

The method at `src/iMPS.jl:396` (`getindex`) and `src/iMPS.jl:437` (`mps_promote_type`) both use `T` in their bodies — those need `DenseIMPS{T,S}`. Use the `/tmp/imps-param-sites.txt` list from Task 1.1 as the worklist.

- [ ] **Step 4: Run the new struct test**

```bash
julia --project=. -e 'using Pkg; Pkg.test(test_args=["test_symmetric_struct.jl"])'
```
Expected: PASS.

- [ ] **Step 5: Run the full test suite**

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```
Expected: every previously-passing test still passes. If any fail, the failure is a regression from Step 3; fix by replacing `iMPS` with the correct alias at the failing callsite.

- [ ] **Step 6: Commit**

```bash
git add src/iMPS.jl src/Schmidt.jl src/Gate.jl src/Krylov.jl \
        src/Miscellaneous.jl src/ScarFinder.jl src/ITensorsInterop.jl
git commit -m "refactor: parameterise iMPS on tensor types, add DenseIMPS alias"
```

---

## Chunk 2: Wire up the TensorKit package extension

The extension is the only part of the package that imports TensorKit. Loading it is automatic the moment the user does `using TensorKit` in their session.

### Task 2.1: Add TensorKit as a weak dependency

**Files:**
- Modify: `Project.toml`.

- [ ] **Step 1: Add weakdeps and extensions stanzas**

```toml
# Project.toml — add at the bottom (alongside the existing [deps] and [extras])
[weakdeps]
TensorKit = "07d1fe3e-3e46-537d-9eac-e9e13d0d4cec"

[extensions]
iTEBDTensorKitExt = "TensorKit"

[compat]
TensorKit = "0.14, 0.15, 0.16"
julia = "1.9"
```

- [ ] **Step 2: Verify package still loads without TensorKit**

```bash
julia --project=. -e 'using iTEBD; println(pathof(iTEBD))'
```
Expected: prints the path to `src/iTEBD.jl` with no errors. (The extension is dormant when TensorKit is not loaded.)

### Task 2.2: Add user-facing extension stubs in the base package

The base package declares the *names* `graded_space`, `spin_half_ops`, and `schmidt_values` and the symbol-based methods of `rand_iMPS` / `product_iMPS`. Without TensorKit loaded, calling them raises a clear "load TensorKit first" error. The extension shadows them with real methods.

**Files:**
- Modify: `src/iTEBD.jl`.
- Create: `src/SymmetricStubs.jl`.

- [ ] **Step 1: Add the stubs file**

```julia
# src/SymmetricStubs.jl
#
# Names declared here are real-implemented inside ext/iTEBDTensorKitExt.jl. In
# the base package they throw a clear message telling the user to `using
# TensorKit` first.

export graded_space, spin_half_ops, schmidt_values

const _NEEDS_TENSORKIT = """
This entry point requires the TensorKit backend. Run `using TensorKit` in your \
session (or add it to your project) to load the iTEBD symmetric extension.\
"""

graded_space(args...; kwargs...) = error(_NEEDS_TENSORKIT)
spin_half_ops(args...; kwargs...) = error(_NEEDS_TENSORKIT)

# schmidt_values has a dense fallback. The symmetric extension specialises on
# DiagonalTensorMap-backed states; the dense fallback below handles the
# existing Vector{Float64} layout.
"""
    schmidt_values(ψ, i)

Return the Schmidt spectrum on bond `i` as a `Vector{Float64}`, independent of
whether `ψ` is dense or symmetric.
"""
schmidt_values(ψ::DenseIMPS, i::Integer) = Float64.(ψ.λ[i])

# Symbol-based rand_iMPS / product_iMPS stubs — base package version is the
# error path, the extension shadows them.
function rand_iMPS(sym::Symbol, args...; kwargs...)
    error(_NEEDS_TENSORKIT)
end
function product_iMPS(sym::Symbol, args...; kwargs...)
    error(_NEEDS_TENSORKIT)
end
```

- [ ] **Step 2: Include the stubs file in `src/iTEBD.jl`**

```julia
# src/iTEBD.jl — add a line just below the existing includes (after ScarFinder.jl):
include("SymmetricStubs.jl")
```

- [ ] **Step 3: Smoke test the stubs**

```bash
julia --project=. -e '
using iTEBD
try
    graded_space(:U1, 0=>1)
catch e
    @assert occursin("TensorKit", e.msg) "Expected TensorKit error message"
    println("OK: graded_space raises the expected error.")
end
ψ = rand_iMPS(ComplexF64, 2, 2, 3)
@assert schmidt_values(ψ, 1) isa Vector{Float64}
println("OK: dense schmidt_values returns Vector{Float64}.")
'
```
Expected: prints both `OK:` lines.

- [ ] **Step 4: Commit**

```bash
git add Project.toml src/iTEBD.jl src/SymmetricStubs.jl
git commit -m "feat: add TensorKit as a weak dep and stub symmetric entry points"
```

### Task 2.3: Create the extension skeleton

**Files:**
- Create: `ext/iTEBDTensorKitExt.jl`.

- [ ] **Step 1: Write the skeleton**

```julia
# ext/iTEBDTensorKitExt.jl
module iTEBDTensorKitExt

using iTEBD
using TensorKit
using LinearAlgebra

# Imports of names we will specialise. Adding `import` (not `using`) so that
# extending these names is unambiguous to the compiler.
import iTEBD: graded_space, spin_half_ops, schmidt_values
import iTEBD: rand_iMPS, product_iMPS
import iTEBD: iMPS

# SymmetricIMPS alias — referenced by every symmetric specialisation below.
# ΓT is a TensorMap with 1 in-leg and 2 out-legs (or whatever the canonical
# convention turns out to be — verify against TensorKit 0.16 docs at
# https://quantumkithub.github.io/TensorKit.jl/stable/man/tensors/ when wiring
# up the first constructor in Chunk 4).
const SymmetricIMPS = iMPS{<:AbstractTensorMap, <:DiagonalTensorMap}

# Stubs that will be filled in by the chunks below. Leaving the methods undefined
# at this point is fine — calling them will raise MethodError, which is exactly
# what we want until Chunks 3-7 implement them.

end # module
```

- [ ] **Step 2: Verify extension loads**

```bash
julia --project=. -e '
using iTEBD
using TensorKit
println("Loaded iTEBD with TensorKit extension.")
println("SymmetricIMPS alias: ", iTEBD.SymmetricIMPS)
' 2>&1 | tee /tmp/ext-load.txt
```
Expected: no errors, prints both lines. If Julia complains the extension didn't load, double-check the UUID in `Project.toml` and that `ext/iTEBDTensorKitExt.jl` is exactly that path.

- [ ] **Step 3: Commit**

```bash
git add ext/iTEBDTensorKitExt.jl
git commit -m "feat: scaffold ext/iTEBDTensorKitExt.jl, define SymmetricIMPS alias"
```

---

## Chunk 3: Helper layer — `graded_space`, `spin_half_ops`, `schmidt_values`

### Task 3.1: `graded_space(:U1, ...)`

**Files:**
- Create: `test/test_symmetric_basic.jl`.
- Modify: `ext/iTEBDTensorKitExt.jl`.
- Modify: `test/runtests.jl`.

- [ ] **Step 1: Write the failing test**

```julia
# test/test_symmetric_basic.jl
using Test
using iTEBD
using TensorKit

@testset "graded_space" begin
    P = graded_space(:U1, 0=>2, 1=>1, -1=>1)
    @test P isa Vect[U1Irrep]
    @test dim(P) == 4
    @test dim(P, U1Irrep(0)) == 2
    @test dim(P, U1Irrep(1)) == 1
end
```

Add `"test_symmetric_basic.jl"` to the `"unit"` group in `test/runtests.jl`.

- [ ] **Step 2: Run test to verify it fails**

```bash
julia --project=. -e 'using Pkg; Pkg.test(test_args=["test_symmetric_basic.jl"])'
```
Expected: FAIL with `MethodError: no method matching graded_space(::Symbol, ...)`.

- [ ] **Step 3: Implement `graded_space(:U1, ...)` in the extension**

```julia
# ext/iTEBDTensorKitExt.jl — inside the module, add:

"""
    graded_space(symmetry::Symbol, charges_to_dims...)

Build a TensorKit graded vector space for a named Abelian symmetry without
forcing the user to import TensorKit's irrep types directly.

Supported `symmetry` values: `:U1`, `:Z2`, `:ZN`, `:U1xU1`, `:U1xZ2`,
`:Trivial`. For `:ZN` the second positional argument is the order `N`.

Examples:
    graded_space(:U1, 0=>2, 1=>1, -1=>1)
    graded_space(:Z2, 0=>3, 1=>3)
    graded_space(:ZN, 4, 0=>1, 1=>1, 2=>1, 3=>1)
    graded_space(:U1xU1, (0,0)=>2, (1,-1)=>1)
"""
function graded_space(::Val{:U1}, pairs::Pair{Int,Int}...)
    return Vect[U1Irrep](Int(c) => Int(d) for (c, d) in pairs)
end

graded_space(sym::Symbol, args...) = graded_space(Val(sym), args...)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
julia --project=. -e 'using Pkg; Pkg.test(test_args=["test_symmetric_basic.jl"])'
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ext/iTEBDTensorKitExt.jl test/test_symmetric_basic.jl test/runtests.jl
git commit -m "feat: graded_space(:U1, ...) helper"
```

### Task 3.2: `graded_space(:Z2, ...)`, `(:ZN, N, ...)`, and trivial sector

**Files:**
- Modify: `test/test_symmetric_basic.jl`, `ext/iTEBDTensorKitExt.jl`.

- [ ] **Step 1: Extend the test**

Append to `test/test_symmetric_basic.jl`:

```julia
@testset "graded_space Z2, ZN, Trivial" begin
    Pz = graded_space(:Z2, 0=>3, 1=>3)
    @test Pz isa Vect[Z2Irrep]
    @test dim(Pz) == 6

    P4 = graded_space(:ZN, 4, 0=>1, 1=>1, 2=>1, 3=>1)
    @test P4 isa Vect[ZNIrrep{4}]
    @test dim(P4) == 4

    Pt = graded_space(:Trivial, 0=>3)
    @test Pt isa ComplexSpace
    @test dim(Pt) == 3
end
```

- [ ] **Step 2: Run test to verify it fails**

```bash
julia --project=. -e 'using Pkg; Pkg.test(test_args=["test_symmetric_basic.jl"])'
```
Expected: FAIL on the new testsets.

- [ ] **Step 3: Implement the additional methods**

Append to `ext/iTEBDTensorKitExt.jl`:

```julia
function graded_space(::Val{:Z2}, pairs::Pair{Int,Int}...)
    return Vect[Z2Irrep](Int(c) => Int(d) for (c, d) in pairs)
end

function graded_space(::Val{:ZN}, N::Integer, pairs::Pair{Int,Int}...)
    N ≥ 2 || throw(ArgumentError("graded_space(:ZN, N, …) requires N ≥ 2 (got $N)"))
    Irrep = ZNIrrep{Int(N)}
    return Vect[Irrep](Int(c) => Int(d) for (c, d) in pairs)
end

function graded_space(::Val{:Trivial}, pairs::Pair{Int,Int}...)
    length(pairs) == 1 || throw(ArgumentError(
        "graded_space(:Trivial, …) takes exactly one charge=>dim pair"))
    _, d = first(pairs)
    return ComplexSpace(Int(d))
end
```

- [ ] **Step 4: Run test to verify it passes**

```bash
julia --project=. -e 'using Pkg; Pkg.test(test_args=["test_symmetric_basic.jl"])'
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ext/iTEBDTensorKitExt.jl test/test_symmetric_basic.jl
git commit -m "feat: graded_space supports :Z2, :ZN, :Trivial"
```

### Task 3.3: `graded_space(:U1xU1, ...)` and `(:U1xZ2, ...)` product sectors

**Files:**
- Modify: `test/test_symmetric_basic.jl`, `ext/iTEBDTensorKitExt.jl`.

- [ ] **Step 1: Extend the test**

```julia
@testset "graded_space products" begin
    P = graded_space(:U1xU1, (0,0)=>2, (1,-1)=>1, (-1,1)=>1)
    @test dim(P) == 4

    Q = graded_space(:U1xZ2, (0,0)=>2, (1,1)=>1)
    @test dim(Q) == 3
end
```

- [ ] **Step 2: Implement the methods**

```julia
function graded_space(::Val{:U1xU1}, pairs::Pair{<:Tuple,Int}...)
    Irrep = U1Irrep ⊠ U1Irrep
    return Vect[Irrep](Irrep(c[1], c[2]) => Int(d) for (c, d) in pairs)
end

function graded_space(::Val{:U1xZ2}, pairs::Pair{<:Tuple,Int}...)
    Irrep = U1Irrep ⊠ Z2Irrep
    return Vect[Irrep](Irrep(c[1], c[2]) => Int(d) for (c, d) in pairs)
end
```

- [ ] **Step 3: Run test, then commit**

```bash
julia --project=. -e 'using Pkg; Pkg.test(test_args=["test_symmetric_basic.jl"])'
git add ext/iTEBDTensorKitExt.jl test/test_symmetric_basic.jl
git commit -m "feat: graded_space supports product Abelian sectors"
```

### Task 3.4: `spin_half_ops(:U1)` and `spin_half_ops(:Trivial)`

**Files:**
- Modify: `test/test_symmetric_basic.jl`, `ext/iTEBDTensorKitExt.jl`.

- [ ] **Step 1: Add the failing test**

The U(1) helper returns **pre-assembled two-site terms** rather than charged
one-site raising/lowering operators. The reason is that single-site charged
operators in TensorKit 0.16 must live on different HomSpaces (e.g. `P ← P`
with coupled sector ±2 each), which means `Sp ⊗ Sm + Sm ⊗ Sp` fails with
`SpaceMismatch` — the two summands land on incompatible spaces. Pre-assembling
the two-site terms on the single common space `(P ⊗ P) ← (P ⊗ P)` sidesteps
the issue entirely and lets the user write `h = SzSz + 0.5*(SpSm + SmSp)`.

```julia
@testset "spin_half_ops" begin
    @testset "U(1) symmetric" begin
        Sz, SzSz, SpSm, SmSp = spin_half_ops(:U1)
        P = graded_space(:U1, 1=>1, -1=>1)

        # Sz: one-site, hermitian, squares to (1/4) I
        @test sectortype(space(Sz, 1)) == U1Irrep
        @test space(Sz, 1) == P
        @test isapprox(Sz, Sz'; atol=1e-12)
        @test isapprox(Sz * Sz, 0.25 * id(P); atol=1e-12)

        # Two-site operators all live on the same HomSpace and compose
        @test space(SzSz) == space(SpSm) == space(SmSp)
        @test isapprox(SpSm', SmSp; atol=1e-12)

        # The Heisenberg density assembles cleanly; verify against the dense
        # 4×4 reference in the {|↑↑⟩, |↑↓⟩, |↓↑⟩, |↓↓⟩} basis.
        h = SzSz + 0.5 * (SpSm + SmSp)
        @test isapprox(h, h'; atol=1e-12)
        h_dense = ComplexF64[
            0.25   0     0     0   ;
            0    -0.25  0.5    0   ;
            0     0.5  -0.25   0   ;
            0     0     0     0.25 ]
        @test isapprox(reshape(convert(Array, h), 4, 4), h_dense; atol=1e-12)
    end

    @testset "Trivial / dense fallback" begin
        Sx, Sy, Sz, Sp, Sm, Id = spin_half_ops(:Trivial)
        @test Sz isa AbstractMatrix
        @test isapprox(Sx*Sx + Sy*Sy + Sz*Sz, 0.75 * Id; atol=1e-12)
    end
end
```

- [ ] **Step 2: Implement the U(1) variant**

```julia
function spin_half_ops(::Val{:U1})
    P = graded_space(:U1, 1=>1, -1=>1)

    # Sz: one-site endomorphism (flux 0)
    Sz = zeros(ComplexF64, P ← P)
    block(Sz, U1Irrep(1))[1, 1]  =  0.5
    block(Sz, U1Irrep(-1))[1, 1] = -0.5

    # Two-site operators on the single common space (P ⊗ P) ← (P ⊗ P).
    # The fused codomain decomposes into sectors {+2, 0, -2}; the U1(0)
    # block is the 2×2 mixed-spin subspace.
    SzSz = zeros(ComplexF64, P ⊗ P ← P ⊗ P)
    block(SzSz, U1Irrep(2))[1, 1]  = 0.25
    block(SzSz, U1Irrep(-2))[1, 1] = 0.25
    block(SzSz, U1Irrep(0)) .= ComplexF64[-0.25 0; 0 -0.25]

    # SpSm: |↑↓⟩⟨↓↑| — upper-right of the U1(0) block
    SpSm = zeros(ComplexF64, P ⊗ P ← P ⊗ P)
    block(SpSm, U1Irrep(0)) .= ComplexF64[0 1; 0 0]

    # SmSp: |↓↑⟩⟨↑↓| = (SpSm)'
    SmSp = zeros(ComplexF64, P ⊗ P ← P ⊗ P)
    block(SmSp, U1Irrep(0)) .= ComplexF64[0 0; 1 0]

    return Sz, SzSz, SpSm, SmSp
end
```

> **TensorKit version note:** the code above targets TensorKit 0.16's
> `zeros(ComplexF64, codomain ← domain)` constructor for `TensorMap`, and
> the `block(t, sector)` accessor returning a mutable dense view. The
> basis ordering inside the `U1(0)` block (whether `[|↑↓⟩, |↓↑⟩]` or its
> swap) is empirically verified by the test in Step 1 — if the dense
> reference matrix ever stops matching after a TensorKit upgrade, transpose
> the off-diagonal entries inside the `U1(0)` blocks of `SpSm`/`SmSp`.

- [ ] **Step 3: Implement the Trivial / dense variant**

Add to the extension (the dense variant doesn't need TensorKit-loaded code but lives in the same helper for discoverability):

```julia
function spin_half_ops(::Val{:Trivial})
    Sx = ComplexF64[0 0.5; 0.5 0]
    Sy = ComplexF64[0 -0.5im; 0.5im 0]
    Sz = ComplexF64[0.5 0; 0 -0.5]
    Sp = Sx + im*Sy
    Sm = Sx - im*Sy
    Id = ComplexF64[1 0; 0 1]
    return Sx, Sy, Sz, Sp, Sm, Id
end

spin_half_ops(sym::Symbol) = spin_half_ops(Val(sym))
```

- [ ] **Step 4: Run test, then commit**

```bash
julia --project=. -e 'using Pkg; Pkg.test(test_args=["test_symmetric_basic.jl"])'
git add ext/iTEBDTensorKitExt.jl test/test_symmetric_basic.jl
git commit -m "feat: spin_half_ops(:U1) and :Trivial operator helpers"
```

### Task 3.5: `schmidt_values` for symmetric states

**Files:**
- Modify: `test/test_symmetric_basic.jl`, `ext/iTEBDTensorKitExt.jl`.

This task is intentionally light. The full `schmidt_values` symmetric implementation depends on the symmetric `iMPS` constructor (Chunk 4) and its `DiagonalTensorMap`-backed `λ` field. We only stub here; the test gets fleshed out in Chunk 4.

- [ ] **Step 1: Add the symmetric specialisation**

```julia
# ext/iTEBDTensorKitExt.jl
function schmidt_values(ψ::SymmetricIMPS, i::Integer)
    1 ≤ i ≤ ψ.n || throw(BoundsError(ψ.λ, i))
    return Float64.(sort!(collect(values(blocks(ψ.λ[i]))); rev=true) |> _flatten_block_values)
end

# Helper: flatten a vector of per-block diagonal arrays into a single sorted vector.
function _flatten_block_values(per_block::AbstractVector)
    out = Float64[]
    for blk in per_block
        for v in blk
            push!(out, Float64(real(v)))
        end
    end
    sort!(out; rev=true)
    return out
end
```

> **Note:** the exact way to extract block diagonal values from a
> `DiagonalTensorMap` in your TensorKit version may be `blocks(λ)` returning
> `Dict{Sector,Vector}` or a similar shape — adapt the helper to the actual
> return type. Verify with `using TensorKit; D = DiagonalTensorMap(rand(3),
> ℂ^3); blocks(D)` in a REPL before committing.

- [ ] **Step 2: Commit**

```bash
git add ext/iTEBDTensorKitExt.jl
git commit -m "feat: schmidt_values stub for SymmetricIMPS"
```

---

## Chunk 4: Symmetric `iMPS` constructors

### Task 4.1: `rand_iMPS(pspace, vspace, n)` raw TensorKit constructor

**Files:**
- Modify: `test/test_symmetric_basic.jl`, `ext/iTEBDTensorKitExt.jl`.

- [ ] **Step 1: Add the failing test**

```julia
@testset "rand_iMPS symmetric (raw spaces)" begin
    P = graded_space(:U1, 1=>1, -1=>1)
    V = graded_space(:U1, 0=>2, 2=>1, -2=>1)
    ψ = rand_iMPS(P, V, 2)
    @test ψ isa iTEBD.SymmetricIMPS
    @test ψ.n == 2
    @test length(ψ.Γ) == 2 && length(ψ.λ) == 2
    @test domain(ψ.Γ[1])[1] == V  # right virtual leg
    @test codomain(ψ.Γ[1])[2] == V   # left virtual leg matches the next site
end
```

> Verify the exact codomain/domain layout against the TensorKit MPS conventions used in MPSKit.jl at <https://quantumkithub.github.io/MPSKit.jl/dev/man/states/> before locking the test. Standard convention is `(V_left ⊗ P) ← V_right`.

- [ ] **Step 2: Implement the constructor**

```julia
function rand_iMPS(pspace::VectorSpace, vspace::VectorSpace, n::Integer)
    n > 0 || throw(ArgumentError("n must be positive (got $n)"))
    Γ = [randn(ComplexF64, vspace ⊗ pspace ← vspace) for _ in 1:n]
    λ = [DiagonalTensorMap(ones(Float64, dim(vspace)), vspace) for _ in 1:n]
    ψ = iMPS(Γ, λ, n)
    return canonical!(ψ)        # routed in Chunk 5
end
```

> Until Chunk 5 implements `canonical!` for `SymmetricIMPS`, the call in the
> final line will hit a `MethodError`. Mark this task complete only once the
> test in Step 1 passes — that becomes the gating condition for Chunk 5.

- [ ] **Step 3: Commit (test will currently fail; that's fine — gated by Chunk 5)**

```bash
git add ext/iTEBDTensorKitExt.jl test/test_symmetric_basic.jl
git commit -m "feat: rand_iMPS(pspace, vspace, n) symmetric raw constructor"
```

### Task 4.2: Symbol-based `rand_iMPS(:U1, charges, χ; n, flux)`

**Files:**
- Modify: `test/test_symmetric_basic.jl`, `ext/iTEBDTensorKitExt.jl`.

- [ ] **Step 1: Test**

```julia
@testset "rand_iMPS(:U1, charges, χ)" begin
    ψ = rand_iMPS(:U1, [-1, 1]; χ=8, n=2, flux=0)
    @test ψ isa iTEBD.SymmetricIMPS
    @test ψ.n == 2
end
```

- [ ] **Step 2: Implement the helper**

```julia
"""
    rand_iMPS(symmetry::Symbol, charges; χ::Integer, n::Integer=1, flux=0)

Build a random symmetric iMPS in one call. `symmetry` is one of the supported
`graded_space` symbols. `charges` is the list of charges on the physical leg,
one entry per basis state. `χ` is the total bond dimension auto-distributed
across sectors compatible with the requested `flux`.
"""
function rand_iMPS(sym::Symbol, charges::AbstractVector{<:Integer};
                   χ::Integer, n::Integer=1, flux::Integer=0)
    P = graded_space(sym, [c => 1 for c in charges]...)
    V = _auto_bond_space(sym, P, χ; flux=flux)
    return rand_iMPS(P, V, n)
end

# Internal: distribute χ across sectors of V compatible with cycling around
# the unit cell at the requested total flux. For v1 we use a uniform split
# weighted by the natural sector multiplicity from fusing pspace ⊗ pspace.
function _auto_bond_space(sym::Symbol, pspace::VectorSpace, χ::Integer; flux::Integer=0)
    fused = fuse(pspace ⊗ pspace)
    per_sector = max(1, χ ÷ length(sectors(fused)))
    pairs = Pair{Int,Int}[Int(charge(s)[1]) => per_sector for s in sectors(fused)]
    return graded_space(sym, pairs...)
end
```

> The signature `charge(s)[1]` and the `sectors(fused)` iterator come from
> TensorKit; if your version returns sector iterators differently (e.g.
> `keys(blocks(...))`) adapt this helper. The test in Step 1 is the success
> criterion.

- [ ] **Step 3: Commit**

```bash
git add ext/iTEBDTensorKitExt.jl test/test_symmetric_basic.jl
git commit -m "feat: rand_iMPS(:U1, charges; χ, n, flux) symbol-based ctor"
```

### Task 4.3: `product_iMPS(:U1, charges, occupations)`

**Files:**
- Modify: `test/test_symmetric_basic.jl`, `ext/iTEBDTensorKitExt.jl`.

- [ ] **Step 1: Test**

```julia
@testset "product_iMPS symmetric Néel state" begin
    ψ = product_iMPS(:U1, [-1, 1], [1, -1])
    @test ψ isa iTEBD.SymmetricIMPS
    @test ψ.n == 2
    # Each site lives in a 1-d sector matching the occupation
    for i in 1:2
        @test dim(domain(ψ.Γ[i])[1]) == 1
    end
end
```

- [ ] **Step 2: Implement**

```julia
function product_iMPS(sym::Symbol, charges::AbstractVector{<:Integer},
                      occupations::AbstractVector{<:Integer})
    n = length(occupations)
    n > 0 || throw(ArgumentError("occupations must be non-empty"))
    all(c -> c in charges, occupations) || throw(ArgumentError(
        "occupations must be drawn from `charges`"))
    P = graded_space(sym, [c => 1 for c in charges]...)

    # Bond at site i carries the cumulative charge running from left to right.
    cum = cumsum(occupations)
    pushfirst!(cum, 0)
    pop!(cum)
    V = [graded_space(sym, c => 1) for c in cum]

    Γ = Vector{Any}(undef, n)
    λ = Vector{Any}(undef, n)
    for i in 1:n
        Vl = V[i]
        Vr = V[mod1(i + 1, n)]
        Γ[i] = TensorMap(zeros, ComplexF64, Vl ⊗ P ← Vr)
        # Set the single nonzero block: charge on physical leg = occupations[i]
        # Implementation detail — for product states the only allowed block has
        # ones in the single basis element.
        block(Γ[i], sectortype(P)(occupations[i]))[1, 1, 1] = 1.0
        λ[i] = DiagonalTensorMap(ones(Float64, 1), Vr)
    end
    return iMPS(Vector{eltype(Γ)}(Γ), Vector{eltype(λ)}(λ), n)
end
```

> The exact way to index a single block of a TensorMap is `block(t, sector)`
> returning a dense array. If your TensorKit version uses `blocks(t)[sector]`
> instead, swap the call.

- [ ] **Step 3: Commit**

```bash
git add ext/iTEBDTensorKitExt.jl test/test_symmetric_basic.jl
git commit -m "feat: product_iMPS(:U1, charges, occupations) symmetric ctor"
```

### Task 4.4: Wraparound flux check in the inner constructor

**Files:**
- Modify: `ext/iTEBDTensorKitExt.jl`.

- [ ] **Step 1: Specialise `_validate_iMPS_bonds` for the symmetric case**

```julia
# ext/iTEBDTensorKitExt.jl
import iTEBD: _validate_iMPS_bonds

function _validate_iMPS_bonds(
    Γ::Vector{<:AbstractTensorMap},
    λ::Vector{<:DiagonalTensorMap},
    n::Integer,
)
    for i in 1:n
        Vr = domain(Γ[i])[1]
        Vl_next = codomain(Γ[mod1(i + 1, n)])[1]
        Vr == Vl_next || throw(DimensionMismatch(
            "SymmetricIMPS bond $i: right space of Γ[$i] ($Vr) does not match " *
            "left space of Γ[$(mod1(i + 1, n))] ($Vl_next); fluxes must close " *
            "around the unit cell"
        ))
    end
    return nothing
end
```

- [ ] **Step 2: Add an explicit failing-flux test**

In `test/test_symmetric_basic.jl`:

```julia
@testset "wraparound flux check rejects mismatched spaces" begin
    P = graded_space(:U1, 1=>1, -1=>1)
    Va = graded_space(:U1, 0=>1)
    Vb = graded_space(:U1, 2=>1)            # deliberately wrong
    Γ1 = TensorMap(randn, ComplexF64, Va ⊗ P ← Vb)
    Γ2 = TensorMap(randn, ComplexF64, Vb ⊗ P ← Va)       # closes back
    Γ_bad = [Γ1, TensorMap(randn, ComplexF64, Va ⊗ P ← Va)]   # broken seam
    λ_dummy = [DiagonalTensorMap(ones(1), Vb), DiagonalTensorMap(ones(1), Va)]
    @test_throws DimensionMismatch iMPS(Γ_bad, λ_dummy, 2)
end
```

- [ ] **Step 3: Run, commit**

```bash
julia --project=. -e 'using Pkg; Pkg.test(test_args=["test_symmetric_basic.jl"])'
git add ext/iTEBDTensorKitExt.jl test/test_symmetric_basic.jl
git commit -m "feat: symmetric wraparound bond-space check"
```

---

## Chunk 5: Symmetric canonicalization

This is the largest single chunk. The implementation reuses TensorKit's `tsvd` and follows the same fixed-point + sweep structure as the dense path, but TensorKit's primitives let us skip the explicit reshape/fuse plumbing.

### Task 5.1: Truncated SVD on a two-site block

**Files:**
- Modify: `test/test_symmetric_basic.jl`, `ext/iTEBDTensorKitExt.jl`.

- [ ] **Step 1: Test**

```julia
@testset "_symmetric_tsvd preserves blocks" begin
    P = graded_space(:U1, 1=>1, -1=>1)
    V = graded_space(:U1, 0=>2, 2=>1, -2=>1)
    A = TensorMap(randn, ComplexF64, V ⊗ P ⊗ P ← V)
    U, S, Vt, info = iTEBDTensorKitExt._symmetric_tsvd(A; maxdim=4, cutoff=1e-12)
    @test S isa DiagonalTensorMap
    @test space(U)[end] == space(S)[1]
    @test space(Vt)[1] == space(S)[2]
    @test sum(values(blocks(S))) |> v -> sum(sum(abs2, b) for b in v) > 0
end
```

- [ ] **Step 2: Implement**

```julia
function _symmetric_tsvd(A::AbstractTensorMap; maxdim::Integer, cutoff::Real)
    # tsvd over a 2-site block: fuse legs (V ⊗ P) on the left and (P ← V) on the right
    # so that we get the standard left/right canonical pieces.
    U, S, V_, info = tsvd!(copy(A); trunc=truncbelow(cutoff) & truncdim(maxdim))
    return U, S, V_, info
end
```

> `truncbelow(cutoff) & truncdim(maxdim)` is the documented TensorKit
> truncation-policy combinator at
> <https://quantumkithub.github.io/TensorKit.jl/stable/man/truncation/>. If
> your version uses different policy names (e.g. `truncerr`, `truncdim`),
> consult that page.

- [ ] **Step 3: Run, commit**

```bash
julia --project=. -e 'using Pkg; Pkg.test(test_args=["test_symmetric_basic.jl"])'
git add ext/iTEBDTensorKitExt.jl test/test_symmetric_basic.jl
git commit -m "feat: symmetric truncated SVD on two-site blocks"
```

### Task 5.2: Symmetric `canonical!`

**Files:**
- Modify: `test/test_symmetric_basic.jl`, `ext/iTEBDTensorKitExt.jl`.

- [ ] **Step 1: Test**

```julia
@testset "canonical! on symmetric iMPS" begin
    P = graded_space(:U1, 1=>1, -1=>1)
    V = graded_space(:U1, 0=>2, 2=>1, -2=>1)
    Γ = [TensorMap(randn, ComplexF64, V ⊗ P ← V) for _ in 1:2]
    λ = [DiagonalTensorMap(ones(Float64, dim(V)), V) for _ in 1:2]
    ψ = iMPS(Γ, λ, 2)
    canonical!(ψ)
    # After canonicalisation: λ on each bond is sorted descending and normalised
    for i in 1:2
        vals = schmidt_values(ψ, i)
        @test issorted(vals; rev=true)
        @test isapprox(sum(abs2, vals), 1.0; atol=1e-10)
    end
end
```

- [ ] **Step 2: Implement**

```julia
import iTEBD: canonical!

"""
    canonical!(ψ::SymmetricIMPS; maxdim=MAXDIM, cutoff=SVDTOL, renormalize=true)

Bring `ψ` to Schmidt canonical form using TensorKit primitives. The injective
fixed-point is solved by power iteration on the transfer map.
"""
function canonical!(ψ::SymmetricIMPS;
                    maxdim::Integer=iTEBD.MAXDIM,
                    cutoff::Real=iTEBD.SVDTOL,
                    renormalize::Bool=true,
                    tol::Real=1e-12,
                    maxiter::Integer=200)
    # 1) Solve for the dominant right/left fixed points of the unit-cell
    #    transfer map via simple power iteration (KrylovKit is overkill here
    #    given the symmetric blocks usually have small dim).
    R = _power_fixed_point(ψ; dir=:r, tol=tol, maxiter=maxiter)
    L = _power_fixed_point(ψ; dir=:l, tol=tol, maxiter=maxiter)

    # 2) Take square roots in each block (R and L are positive in the injective case).
    Xinv = _symmetric_isqrt(R)
    Y    = _symmetric_sqrt(L)

    # 3) SVD of Y · X⁻¹ gives the new gauge transformation; absorb it back into ψ.Γ.
    U, Λ, Vt = tsvd!(Y * Xinv; trunc=truncbelow(cutoff) & truncdim(maxdim))
    if renormalize
        Λ = Λ / norm(Λ)
    end

    # 4) Sweep through the unit cell rewriting tensors and Schmidt values.
    _apply_gauge_sweep!(ψ, U, Λ, Vt)
    return ψ
end

# --- helpers ---

function _power_fixed_point(ψ::SymmetricIMPS; dir::Symbol, tol::Real, maxiter::Integer)
    Vbond = (dir === :r) ? domain(ψ.Γ[end])[1] : codomain(ψ.Γ[1])[1]
    ρ = id(ComplexF64, Vbond)
    for _ in 1:maxiter
        ρ_new = _apply_transfer_unit_cell(ψ, ρ; dir=dir)
        ρ_new = ρ_new / norm(ρ_new)
        if norm(ρ_new - ρ) < tol
            return ρ_new
        end
        ρ = ρ_new
    end
    return ρ
end

function _apply_transfer_unit_cell(ψ::SymmetricIMPS, ρ::AbstractTensorMap; dir::Symbol)
    out = ρ
    indices = dir === :r ? reverse(1:ψ.n) : (1:ψ.n)
    for i in indices
        Γ = ψ.Γ[i]
        if dir === :r
            @tensor out_new[a; b] := Γ[a, s, c] * out[c; d] * conj(Γ)[b, s, d]
        else
            @tensor out_new[a; b] := conj(Γ)[c, s, a] * out[c; d] * Γ[d, s, b]
        end
        out = out_new
    end
    return out
end

function _symmetric_sqrt(ρ::AbstractTensorMap)
    # ρ is a positive operator on a single space; sqrt acts block-wise.
    out = similar(ρ)
    for (sector, blk) in blocks(ρ)
        block(out, sector) .= sqrt(Hermitian(blk))
    end
    return out
end

function _symmetric_isqrt(ρ::AbstractTensorMap)
    out = similar(ρ)
    for (sector, blk) in blocks(ρ)
        F = eigen(Hermitian(blk))
        invsqrt = F.vectors * Diagonal(1 ./ sqrt.(F.values)) * F.vectors'
        block(out, sector) .= invsqrt
    end
    return out
end

function _apply_gauge_sweep!(ψ::SymmetricIMPS, U, Λ, Vt)
    # Reabsorb (U, Λ, Vt) into the boundary tensors of the unit cell. For n=1
    # this is a single mult; for n>1 it falls out of the gauge fix.
    ψ.Γ[1] = U * ψ.Γ[1]
    ψ.Γ[end] = ψ.Γ[end] * Vt
    for i in 1:ψ.n
        ψ.λ[i] = (i == ψ.n) ? Λ : ψ.λ[i]
    end
    return ψ
end
```

> **TensorKit version note:** the code above targets TensorKit 0.16's
> `@tensor`-macro support for `AbstractTensorMap`. If your version requires
> explicit `permute` calls instead of slot-based indexing, rewrite each
> `@tensor` block as a `permute` + matrix-multiply pair preserving the
> contracted index pattern. The block iteration `for (sector, blk) in
> blocks(ρ)` follows the TensorKit 0.16 convention; older versions may
> expose this as `for sector in sectors(ρ); blk = block(ρ, sector); …`.
>
> If `_power_fixed_point` does not converge within 200 iterations on the
> Heisenberg test in Chunk 8, switch it to KrylovKit's `eigsolve` using the
> same callable as the matvec. The convergence target is the same.

- [ ] **Step 3: Run all symmetric tests**

```bash
julia --project=. -e 'using Pkg; Pkg.test(test_args=["test_symmetric_basic.jl"])'
```
Expected: PASS, including the previously-blocked `rand_iMPS` and `product_iMPS` tests.

- [ ] **Step 4: Commit**

```bash
git add ext/iTEBDTensorKitExt.jl test/test_symmetric_basic.jl
git commit -m "feat: canonical! for symmetric iMPS via TensorKit primitives"
```

---

## Chunk 6: Gate evolution on the symmetric backend

### Task 6.1: `applygate!` for `SymmetricIMPS`

**Files:**
- Modify: `test/test_symmetric_basic.jl`, `ext/iTEBDTensorKitExt.jl`.

- [ ] **Step 1: Test**

```julia
@testset "applygate! symmetric two-site identity" begin
    P = graded_space(:U1, 1=>1, -1=>1)
    ψ = rand_iMPS(:U1, [-1, 1]; χ=4, n=2, flux=0)
    Iop = id(P ⊗ P)
    norm_before = schmidt_values(ψ, 1)
    applygate!(ψ, Iop, 1, 2; maxdim=4)
    norm_after = schmidt_values(ψ, 1)
    @test isapprox(norm_before, norm_after; atol=1e-10)
end
```

- [ ] **Step 2: Implement**

```julia
import iTEBD: applygate!

function applygate!(ψ::SymmetricIMPS, G::AbstractTensorMap, i::Integer, j::Integer;
                    maxdim::Integer=iTEBD.MAXDIM,
                    cutoff::Real=iTEBD.SVDTOL,
                    renormalize::Bool=true,
                    kwargs...)
    j == mod1(i + 1, ψ.n) || throw(ArgumentError(
        "v1 symmetric applygate! supports nearest-neighbour two-site gates only " *
        "(got i=$i, j=$j on n=$(ψ.n))"))
    Γi, Γj = ψ.Γ[i], ψ.Γ[j]

    # Fuse the two-site block, apply the gate, SVD back into two tensors.
    @tensor B[a, s, t, b] := Γi[a, s, c] * Γj[c, t, b]
    @tensor B[a, s, t, b] := G[s, t, u, v] * B[a, u, v, b]

    U, S, Vt, _ = tsvd!(B; trunc=truncbelow(cutoff) & truncdim(maxdim))
    if renormalize
        S = S / norm(S)
    end
    ψ.Γ[i] = U
    ψ.Γ[j] = S * Vt           # absorb Schmidt values into the right tensor (package convention)
    ψ.λ[i] = S
    return ψ
end
```

- [ ] **Step 3: Run, commit**

```bash
julia --project=. -e 'using Pkg; Pkg.test(test_args=["test_symmetric_basic.jl"])'
git add ext/iTEBDTensorKitExt.jl test/test_symmetric_basic.jl
git commit -m "feat: applygate! for two-site nearest-neighbour symmetric gates"
```

### Task 6.2: `evolve!` routing

**Files:**
- (no changes — `evolve!` in `src/Gate.jl` already loops on `applygate!`, so it picks up the symmetric specialisation by dispatch).

- [ ] **Step 1: Add a smoke test that confirms `evolve!` works symmetrically**

In `test/test_symmetric_basic.jl`:

```julia
@testset "evolve! routes through symmetric applygate!" begin
    P = graded_space(:U1, 1=>1, -1=>1)
    ψ = rand_iMPS(:U1, [-1, 1]; χ=4, n=2)
    Iop = id(P ⊗ P)
    gates = [(Iop, 1, 2), (Iop, 2, 1)]
    evolve!(ψ, gates, 3; maxdim=4)
    @test ψ isa iTEBD.SymmetricIMPS
end
```

- [ ] **Step 2: Run, commit**

```bash
julia --project=. -e 'using Pkg; Pkg.test(test_args=["test_symmetric_basic.jl"])'
git add test/test_symmetric_basic.jl
git commit -m "test: evolve! dispatches into symmetric applygate!"
```

---

## Chunk 7: Symmetric observables

### Task 7.1: `ent_S(ψ::SymmetricIMPS, i)`

**Files:**
- Modify: `test/test_symmetric_basic.jl`, `ext/iTEBDTensorKitExt.jl`.

- [ ] **Step 1: Test**

```julia
@testset "ent_S on symmetric iMPS" begin
    ψ = rand_iMPS(:U1, [-1, 1]; χ=4, n=2)
    canonical!(ψ)
    S = ent_S(ψ, 1)
    @test S ≥ 0
    @test isfinite(S)
end
```

- [ ] **Step 2: Implement**

```julia
import iTEBD: ent_S, entanglement_entropy

function ent_S(ψ::SymmetricIMPS, i::Integer)
    vals = schmidt_values(ψ, i)
    p = abs2.(vals)
    p ./= sum(p)
    return entanglement_entropy(p)
end
```

- [ ] **Step 3: Commit**

```bash
git add ext/iTEBDTensorKitExt.jl test/test_symmetric_basic.jl
git commit -m "feat: ent_S(::SymmetricIMPS, i)"
```

### Task 7.2: One-site `expect` and `energy_density`

**Files:**
- Modify: `test/test_symmetric_basic.jl`, `ext/iTEBDTensorKitExt.jl`.

- [ ] **Step 1: Test**

```julia
@testset "expect one-site and energy_density" begin
    ψ = product_iMPS(:U1, [-1, 1], [1, -1])     # Néel state
    Sz, SzSz, _, _ = spin_half_ops(:U1)
    val1 = expect(ψ, Sz, 1, 1)
    val2 = expect(ψ, Sz, 2, 2)
    @test isapprox(real(val1) + real(val2), 0.0; atol=1e-10)

    # Heisenberg SzSz density on a Néel state — expected value -1/4 per bond.
    e = energy_density(ψ, SzSz)
    @test isfinite(e)
end
```

- [ ] **Step 2: Implement**

```julia
import iTEBD: expect, energy_density

function expect(ψ::SymmetricIMPS, O::AbstractTensorMap, i::Integer, j::Integer)
    i == j || throw(ArgumentError(
        "v1 symmetric expect supports one-site operators only (got i=$i, j=$j)"))
    Γ = ψ.Γ[i]
    λL = ψ.λ[mod1(i - 1, ψ.n)]
    @tensor val = conj(Γ)[a, s, b] * λL[a;a'] * Γ[a', t, b] * O[s, t]
    return val
end

function energy_density(ψ::SymmetricIMPS, h::AbstractTensorMap)
    # Two-site Hamiltonian density; one bond per unit cell, averaged over unit cell.
    n = ψ.n
    total = zero(ComplexF64)
    for i in 1:n
        j = mod1(i + 1, n)
        Γi, Γj = ψ.Γ[i], ψ.Γ[j]
        λL = ψ.λ[mod1(i - 1, n)]
        @tensor val = conj(Γi)[a, s, c] * conj(Γj)[c, t, b] * λL[a;a'] *
                      h[s, t, u, v] * Γi[a', u, c'] * Γj[c', v, b]
        total += val
    end
    return real(total) / n
end
```

> The exact `@tensor` index patterns depend on TensorKit's leg-order
> conventions; the test in Step 1 is the success criterion. If `@tensor`
> errors on the index order, swap to `permute` + matrix contractions.

- [ ] **Step 3: Run, commit**

```bash
julia --project=. -e 'using Pkg; Pkg.test(test_args=["test_symmetric_basic.jl"])'
git add ext/iTEBDTensorKitExt.jl test/test_symmetric_basic.jl
git commit -m "feat: expect one-site and energy_density for SymmetricIMPS"
```

---

## Chunk 8: Golden test — Heisenberg XXZ Sz-conserving iTEBD

This is the v1 acceptance test from the spec.

### Task 8.1: Spin-1/2 Heisenberg Hamiltonian under U(1)

**Files:**
- Create: `test/test_symmetric_heisenberg.jl`.
- Modify: `test/runtests.jl` — add `"test_symmetric_heisenberg.jl"` to the `"integration"` group.

- [ ] **Step 1: Write the integration test**

```julia
# test/test_symmetric_heisenberg.jl
using Test
using iTEBD
using TensorKit
using LinearAlgebra

@testset "Heisenberg XXZ symmetric iTEBD" begin
    P = graded_space(:U1, 1=>1, -1=>1)

    # spin_half_ops(:U1) returns pre-assembled two-site terms so that
    # `SpSm + SmSp` composes cleanly on a single HomSpace (see Task 3.4).
    Sz, SzSz, SpSm, SmSp = spin_half_ops(:U1)
    h = SzSz + 0.5 * (SpSm + SmSp)

    # Start in the Néel state (Sz=0 sector)
    ψ = product_iMPS(:U1, [-1, 1], [1, -1])
    dt = 0.05
    gates = [(exp(-dt * h), 1, 2), (exp(-dt * h), 2, 1)]
    evolve!(ψ, gates, 400; maxdim=32, cutoff=1e-10)

    e = energy_density(ψ, h)
    # Bethe-ansatz ground-state energy density of the spin-1/2 Heisenberg chain
    # is e_∞ = 1/4 - ln(2) ≈ -0.4431.
    @test isapprox(e, 1/4 - log(2); atol=1e-3)

    # Block structure: at fixed Sz=0, only even-charge sectors appear on each bond.
    for i in 1:ψ.n
        for s in sectors(domain(ψ.Γ[i])[1])
            @test iseven(Int(charge(s)[1]))
        end
    end
end
```

- [ ] **Step 2: Run**

```bash
julia --project=. -e 'using Pkg; Pkg.test(test_args=["test_symmetric_heisenberg.jl"])'
```
Expected: PASS. If the energy is off by more than 5e-3, raise `maxdim` to 48 and rerun before declaring failure — Bethe convergence is sensitive to truncation.

- [ ] **Step 3: Commit**

```bash
git add test/test_symmetric_heisenberg.jl test/runtests.jl
git commit -m "test: golden Heisenberg XXZ symmetric iTEBD convergence"
```

### Task 8.2: Dense vs symmetric crosscheck at small χ

**Files:**
- Modify: `test/test_symmetric_heisenberg.jl`.

- [ ] **Step 1: Add a crosscheck against the dense path**

```julia
@testset "Dense and symmetric agree at χ=8" begin
    # Dense path
    Sx, Sy, Sz_d, Sp_d, Sm_d, _ = spin_half_ops(:Trivial)
    SzSz = kron(Sz_d, Sz_d)
    XY_d = 0.5 * (kron(Sp_d, Sm_d) + kron(Sm_d, Sp_d))
    h_d = SzSz + XY_d
    ψd = product_iMPS(ComplexF64, [[0+0im, 1+0im], [1+0im, 0+0im]])
    dt = 0.05
    gates_d = [(exp(-dt * h_d), 1, 2), (exp(-dt * h_d), 2, 1)]
    evolve!(ψd, gates_d, 400; maxdim=8, cutoff=1e-10)
    e_dense = energy_density(ψd, h_d)

    # Symmetric path
    Sz, SzSz, SpSm, SmSp = spin_half_ops(:U1)
    h_s = SzSz + 0.5 * (SpSm + SmSp)
    ψs = product_iMPS(:U1, [-1, 1], [1, -1])
    gates_s = [(exp(-dt * h_s), 1, 2), (exp(-dt * h_s), 2, 1)]
    evolve!(ψs, gates_s, 400; maxdim=8, cutoff=1e-10)
    e_sym = energy_density(ψs, h_s)

    @test isapprox(e_dense, e_sym; atol=1e-4)
end
```

- [ ] **Step 2: Run, commit**

```bash
julia --project=. -e 'using Pkg; Pkg.test(test_args=["test_symmetric_heisenberg.jl"])'
git add test/test_symmetric_heisenberg.jl
git commit -m "test: cross-check dense and symmetric XXZ energy at χ=8"
```

---

## Chunk 9: Documentation

### Task 9.1: Write `docs/src/symmetries.md`

**Files:**
- Create: `docs/src/symmetries.md`.

- [ ] **Step 1: Write the page**

Copy this exact content (drawn from the approved spec) into `docs/src/symmetries.md`:

```markdown
# Symmetric infinite MPS

`iTEBD.jl` ships with an optional symmetric backend that lets you exploit
Abelian conservation laws — total `Sz`, particle number `N`, parity, or any
combination of these — without ever leaving the `iMPS` API you already know.

This page assumes you have run the dense examples in
[States and Canonical Form](imps.md) and now want to take advantage of a
symmetry your model has. **You do not need any prior experience with
TensorKit.**

## What you gain

For the spin-1/2 XXZ chain, total `Sz` is conserved. If you tell `iTEBD.jl`
this, every internal tensor splits into independent blocks labelled by `Sz`,
and roughly `1/√χ` of the data stops being stored. The numerical answer is
unchanged; the runtime and memory both shrink.

## Sectors, charges, graded spaces

A *charge* (synonyms: *sector*, *irrep*, *QN*) is a label that lives on a
basis vector. For spin-1/2 the physical leg has two basis vectors, one with
`Sz = +1/2` and one with `Sz = -1/2`. We use the integer convention `2*Sz`
throughout, so the charges are `+1` and `-1`.

A *graded vector space* is a regular vector space whose basis vectors each
carry a charge. The spin-1/2 physical leg is the graded vector space
`P = (Sz=+1) ⊕ (Sz=-1)`. In `iTEBD.jl` you build it with:

```julia
using iTEBD
using TensorKit                # loads the symmetric extension
P = graded_space(:U1, 1=>1, -1=>1)
```

The pair `1=>1` reads "the `+1` charge has dimension 1 (one basis state)";
similarly for `-1=>1`.

## Flux

> **What is the "flux" of a tensor?**
>
> Every leg of a symmetric tensor carries a charge label on each of its
> basis states. When you contract two legs together, the charges on the
> contracted basis states must match — otherwise the matrix element is zero
> by symmetry. The *flux* of a tensor is the **net charge it carries
> between its incoming and outgoing legs**: how much charge goes *in* on
> one side minus how much goes *out* on the other.
>
> A tensor with **flux = 0** is the most common case — it neither creates
> nor destroys charge. Examples: the identity, a U(1)-symmetric Hamiltonian
> density `Sz⊗Sz`, an MPS tensor at the ground state of an `Sz`-conserving
> model.
>
> A tensor with **flux ≠ 0** *moves* charge. `S+` has flux `+2` (in our
> `2*Sz` convention) because it raises spin. A flux-`q` MPS tensor inserts
> `q` units of total `Sz` at that site.
>
> In `iTEBD.jl`, when you build an iMPS in a fixed-`Sz` sector you set
> every MPS tensor to flux=0, and the wraparound bond closes onto itself
> with consistent charges. If you wanted to study a state with a single
> magnon (one extra `Sz = +1`), you would put one flux-`+2` site somewhere
> in the unit cell.

Worked example. The fluxes of the conceptual spin-1/2 operators are:

| Operator | Flux | Why |
|---|---|---|
| `Sz`       | 0  | diagonal, doesn't move spin |
| `Sp`       | +2 | raises `Sz` by 1, i.e. `2*Sz` by 2 |
| `Sm`       | −2 | lowers `Sz` |
| `Sz ⊗ Sz`  | 0  | sum of two flux-0 ops |
| `Sp ⊗ Sm`  | 0  | +2 and −2 cancel |

In `iTEBD.jl`, `spin_half_ops(:U1)` returns `(Sz, SzSz, SpSm, SmSp)` with the
**two-site terms already assembled** as flux-0 operators on `(P ⊗ P) ← (P ⊗ P)`.
You do not see `Sp`/`Sm` as separate charged objects — they are folded into the
two-site combinations that the Hamiltonian needs. This avoids a TensorKit
subtlety where summing single-site charged operators (which live on different
HomSpaces) raises a `SpaceMismatch` error.

## Arrow convention on diagrams

> **Arrow convention.** Every leg in our diagrams has an arrow.
>
> - An arrow pointing **into** the tensor means that leg's charges are
>   read *as given*.
> - An arrow pointing **out** of the tensor means that leg's charges are
>   read *negated* (mathematically: this leg lives in the dual space).
>
> **Why this matters.** When you connect two legs together, the arrows
> have to be **consistent**: one arrow leaves one tensor, the other arrow
> enters the next tensor. (Connecting "out" to "out" or "in" to "in"
> would sum charges that should subtract — TensorKit raises an error.)
>
> **The standard MPS convention used in this package:** physical legs
> point *out* (kets), bonds point *right* (left bond into the tensor,
> right bond out of it). The flux equation reads
> `(in charges) − (out charges) = flux`.

```
       P↑ (physical, out)
        │
        ▼
   ───►─[ Γ_i ]─►───
   V_left (in)     V_right (out)

   flux(Γ_i) = (charges into Γ_i) − (charges out of Γ_i)
             = V_left_charge − V_right_charge − P_charge
```

## End-to-end walkthrough: spin-1/2 XXZ in the Sz=0 sector

```julia
using iTEBD, TensorKit

# Build the U(1)-symmetric spin-1/2 operators. The helper returns the
# one-site Sz and the three pre-assembled flux-0 two-site terms.
Sz, SzSz, SpSm, SmSp = spin_half_ops(:U1)

# Heisenberg density h = Sz⊗Sz + (1/2)(S+⊗S- + S-⊗S+)
h = SzSz + 0.5 * (SpSm + SmSp)

# Néel initial state in the Sz=0 sector
ψ = product_iMPS(:U1, [-1, 1], [1, -1])

# Imaginary-time iTEBD, two-site Trotter
dt = 0.05
gates = [(exp(-dt * h), 1, 2), (exp(-dt * h), 2, 1)]
evolve!(ψ, gates, 400; maxdim=32, cutoff=1e-10)

# Energy density approaches the Bethe-ansatz value 1/4 - log(2) ≈ -0.4431
energy_density(ψ, h)

# Sector-resolved Schmidt spectrum
schmidt_values(ψ, 1)
```

## Common errors

- `DimensionMismatch: ... fluxes must close around the unit cell` — the
  right virtual space of `Γ[n]` differs from the left virtual space of
  `Γ[1]`. In practice this happens when the running flux around the unit
  cell does not return to zero. Either set every tensor's flux to 0, or
  build the wraparound space explicitly.
- `MethodError: no method matching ...` from a `graded_space` call without
  loading TensorKit — you need `using TensorKit` to pull in the symmetric
  extension.
- `Argument N must be ≥ 2` in `graded_space(:ZN, N, …)` — Z_1 is trivial;
  use `:Trivial` instead.

## See also

- [MPSKitModels.jl](https://github.com/QuantumKitHub/MPSKitModels.jl) for
  ready-made symmetric Hamiltonians (Heisenberg, Hubbard, ...) that work
  unchanged with this backend.
- [TensorKit.jl manual](https://quantumkithub.github.io/TensorKit.jl/stable/)
  for the underlying symmetric-tensor library.
```

- [ ] **Step 2: Commit**

```bash
git add docs/src/symmetries.md
git commit -m "docs: pedagogical page for symmetric iMPS, flux, arrows"
```

### Task 9.2: Add a callout to `docs/src/imps.md`

**Files:**
- Modify: `docs/src/imps.md`.

- [ ] **Step 1: Append the callout at the end of the page**

```markdown
## Symmetric variant

If your model conserves `Sz`, particle number, parity, or any Abelian
combination of those, the same `iMPS` API works on top of an optional
[TensorKit.jl](https://github.com/Jutho/TensorKit.jl)-backed symmetric
infrastructure. See [Symmetric infinite MPS](symmetries.md) for a
walkthrough that explains charges, flux, and arrow conventions from zero.
```

- [ ] **Step 2: Commit**

```bash
git add docs/src/imps.md
git commit -m "docs: link to the new symmetries page from the iMPS page"
```

### Task 9.3: Wire the new page into the nav

**Files:**
- Modify: `docs/make.jl`.

- [ ] **Step 1: Add the entry under `Guide`**

```julia
# docs/make.jl — modify the pages list:
pages=[
    "Overview" => "index.md",
    "Guide" => [
        "Getting Started" => "getting-started.md",
        "States and Canonical Form" => "imps.md",
        "Symmetric MPS" => "symmetries.md",          # <-- add this line
        "Time Evolution" => "time-evolution.md",
        "Observables" => "observables.md",
        "ScarFinder Workflow" => "scarfinder.md",
    ],
    "Reference" => [
        "API Reference" => "api.md",
    ],
],
```

- [ ] **Step 2: Build docs locally**

```bash
julia --project=docs docs/make.jl
```
Expected: docs build with no warnings related to the new file. Any
"reference to nonexistent docstring" warnings are unrelated to this task
and pre-exist on `master`; ignore them.

- [ ] **Step 3: Commit**

```bash
git add docs/make.jl
git commit -m "docs: add symmetries page to the nav"
```

---

## Final verification

- [ ] **Step 1: Run the full test suite**

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```
Expected: every test passes — both the original dense tests (regression
goal) and the new symmetric tests (block-structure + quantitative goals
from the spec).

- [ ] **Step 2: Verify package loads cleanly without TensorKit**

```bash
julia --project=. -e 'using iTEBD; @assert isdefined(iTEBD, :iMPS)'
```
Expected: prints nothing, exits 0.

- [ ] **Step 3: Verify package loads cleanly with TensorKit**

```bash
julia --project=. -e 'using iTEBD; using TensorKit; @assert isdefined(iTEBD, :SymmetricIMPS)'
```
Expected: prints nothing, exits 0.

- [ ] **Step 4: Confirm spec acceptance criteria**

Walk through the three goals in `docs/superpowers/specs/2026-05-26-abelian-symmetries-design.md` §Testing strategy:
1. Dense regression — covered by full test-suite run in Step 1.
2. Block-structure correctness — covered by `test_symmetric_heisenberg.jl` block-parity assertion.
3. Quantitative golden — covered by the `e ≈ 1/4 - log(2)` assertion in the same file.

If all three are green, v1 is done.

- [ ] **Step 5: Tag the milestone**

```bash
git tag -a v1.6.0 -m "Abelian symmetric backend (TensorKit, U(1)/Z_N/products)"
```

(Push and release are deferred to the user.)
