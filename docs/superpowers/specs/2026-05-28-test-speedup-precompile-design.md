# Speed up local test runs via PrecompileTools workloads

**Date:** 2026-05-28
**Status:** Approved (design)
**Topic:** Reduce time-to-first-execution (TTFX) for `Pkg.test()` from a fresh Julia process.

## Problem

The full test suite feels slow. The dominant cost is **not** the test logic — it is
Julia startup plus package/method compilation (TTFX), paid in full on every run
because tests are run as `Pkg.test()` from a fresh `julia` process.

Test surface today (`test/runtests.jl`): ~26 files, ~4,200 lines, run sequentially
with group selection (`unit`/`api`/`smoke`/`bench`/`integration`) via
`ITEBD_TEST_GROUP` or filename `ARGS`. The heaviest single file is
`test_symmetric_basic.jl` (834 lines, 167 `@test*` annotations); it lives in the
default `unit` group and exercises the TensorKit extension, which is the most
compile-heavy path.

CI already uses `julia-actions/cache@v2`, so a precompile-image win carries over to CI.

## Goal

Cache native code for the hot inference paths at package-precompile time so that the
first `using iTEBD` + first real call inside a test process is near-instant, without
changing any runtime behavior.

Non-goals: parallelizing the suite, changing the run workflow, or trimming test
computation. Those were considered (approaches B and C) and deferred; revisit only
if measurement shows TTFX is no longer the bottleneck.

## Approach (chosen)

Add `PrecompileTools.@compile_workload` blocks that run a **minimal** representative
workflow for each of the two compile-heavy paths:

1. **Dense core** — in `src/iTEBD.jl`.
2. **Symmetric / graded** — in `ext/iTEBDTensorKitExt.jl` (only precompiles when
   TensorKit is loaded, which is exactly when the symmetric tests run).

`PrecompileTools` does not alter runtime behavior; it only pre-runs code paths during
precompilation so the native code is baked into the cached image (Julia ≥ 1.9).

### Why both, and why minimal

- **Both:** the symmetric path is the heaviest compile and the default test group
  loads TensorKit, so most of the TTFX win is in the extension. The dense block is
  cheap and covers the non-symmetric tests.
- **Minimal:** one small representative call per path bounds the added
  precompile/install cost. We can widen coverage later if measurement shows specific
  cold paths remain.

## Components

### 1. Dependency change — `Project.toml`

- Add `PrecompileTools` to `[deps]` (NOT `[weakdeps]`). It must be a regular dep so the
  TensorKit extension module can `using PrecompileTools`.
- Add a `[compat]` entry for `PrecompileTools` (e.g. `"1"`).

`PrecompileTools` is a tiny, pure-Julia package with negligible dependency footprint.

### 2. Dense workload — `src/iTEBD.jl`

Add at the bottom of the module, after all `include`s:

```julia
using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    X = ComplexF64[0 1; 1 0]
    G = kron(X, X)
    @compile_workload begin
        ψ = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
        applygate!(ψ, G, 1, 2; maxdim=4)
        evolve!(ψ, [(G, 1, 2), (G, 2, 1)], 3; maxdim=4)
        expect(ψ, G, 1, 2)
        ent_S(ψ, 1)
    end
end
```

Sizes deliberately tiny (2-site cell, `maxdim=4`, 3 steps). Inputs that need no caching
live in `@setup_workload`; only API calls live in `@compile_workload`.

### 3. Symmetric workload — `ext/iTEBDTensorKitExt.jl`

Add at the bottom of the extension module. Uses only patterns already covered by
passing tests in `test_symmetric_basic.jl`:

```julia
using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    @compile_workload begin
        ψ = product_iMPS(:U1, [-1, 1], [1, -1])      # U(1) Néel
        P = codomain(ψ.Γ[1])[2]
        Iop = id(ComplexF64, P ⊗ P)
        applygate!(ψ, Iop, 1, 2; maxdim=8)
        evolve!(ψ, [(Iop, 1, 2), (Iop, 2, 1)], 3; maxdim=8)
        Sz, _, _, _ = spin_half_ops(:U1)
        expect(ψ, Sz, 1, 1)
        ent_S(ψ, 1)
    end
end
```

Scope choice (minimal): cover **U(1)** only. Z2/ZN share the same dispatch machinery,
so U(1) coverage captures most of the TTFX win; widen later only if needed.

**Name resolution (implementation must verify):** the iTEBD API functions
(`product_iMPS`, `applygate!`, `evolve!`, `spin_half_ops`, `expect`, `ent_S`) are
defined in the parent module; the extension adds *methods* to them, so inside the ext
they may be reachable only as `iTEBD.foo` unless explicitly imported. Likewise the
TensorKit symbols (`id`, `⊗`, `codomain`) must be in scope. Before writing the block,
check the existing `import`/`using` lines at the top of `ext/iTEBDTensorKitExt.jl` and
qualify or import names as needed so the workload compiles. The snippet above shows the
*shape*; exact qualification is settled against the actual ext imports.

## Stability guardrails

The user's hard constraint: do not affect overall stability.

- **No behavior change:** `@compile_workload` runs only at precompile time and changes
  no exported function's runtime semantics.
- **The one real risk:** if a workload call *throws*, precompilation fails and the
  package won't load. Mitigation: every call above is copied from a currently-passing
  test with valid inputs. Implementation must run the workloads (precompile must
  succeed) before considering the task done.

## Verification plan

Evidence required before claiming done:

1. **No behavior regression** — `Pkg.test()` is green (default group) on the local
   Julia version.
2. **Workloads don't error** — `using iTEBD` and `using TensorKit; using iTEBD`
   both precompile cleanly (watch for precompile-time errors/warnings).
3. **TTFX delta** — measure first-call latency in a fresh process before vs. after,
   for both paths, and report the numbers:
   - Dense: `julia --project -e 'using iTEBD; @time (ψ=product_iMPS(ComplexF64,[[1,0],[0,1]]); evolve!(ψ,[(kron([0 1;1 0]+0im,[0 1;1 0]+0im),1,2)],3;maxdim=4))'`
   - Symmetric: analogous first `evolve!` on a `:U1` state with TensorKit loaded.
   - Compare against the same commands on the pre-change commit.

A positive result is a clear drop in the measured first-call time; precompilation time
(`Precompiling iTEBD...`) is expected to rise modestly — that is the accepted tradeoff.

## Rollout / reversibility

Fully reversible: deleting the two `@compile_workload` blocks and the `PrecompileTools`
dep restores the prior state. No data, API, or on-disk format changes.
