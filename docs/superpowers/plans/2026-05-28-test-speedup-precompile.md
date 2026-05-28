# TTFX Speedup via PrecompileTools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut local `Pkg.test()` time by precompiling the hot dense and symmetric code paths so the first call in a fresh Julia process is near-instant.

**Architecture:** Add `PrecompileTools.@compile_workload` blocks running a minimal representative workflow in two places — `src/iTEBD.jl` (dense core) and `ext/iTEBDTensorKitExt.jl` (symmetric/graded path). Julia ≥ 1.9 caches the native code into the precompile image. No runtime behavior changes.

**Tech Stack:** Julia ≥ 1.9, PrecompileTools.jl, TensorKit (extension trigger), existing `Test`-based suite.

**Spec:** `docs/superpowers/specs/2026-05-28-test-speedup-precompile-design.md`

---

## File Structure

- `Project.toml` — add `PrecompileTools` to `[deps]` and `[compat]`.
- `src/iTEBD.jl` — add dense `@compile_workload` block before the module's final `end`.
- `ext/iTEBDTensorKitExt.jl` — add symmetric `@compile_workload` block before `end # module`.
- `test/test_precompile_workload.jl` — NEW: regression test that runs both workload bodies so API drift fails loudly in the suite instead of as an opaque precompile error.
- `test/runtests.jl` — register the new test file in the `smoke` group.

## Note on TDD for this change

This is a build/precompile change; there is no "new behavior" to drive red→green. The genuine guard is **Task 3's regression test**, which executes the exact workload bodies and asserts they run. It passes immediately (the API already works) — its value is catching future API drift that would otherwise break precompilation. Primary verification is: precompile cleanly + full suite green + measured TTFX drop.

All commands assume the working directory is the repo root and `julia` is on `PATH`.

---

### Task 1: Capture TTFX baseline (no code change)

Record "before" numbers on current `HEAD` so the final task can show the delta.

**Files:** none.

- [ ] **Step 1: Warm the precompile cache (dense)**

Run:
```bash
julia --project=. -e 'using iTEBD'
```
Expected: may print `Precompiling iTEBD...` once, then exit 0.

- [ ] **Step 2: Measure dense first-call latency (fresh process)**

Run this twice; **record the SECOND run's `@time`** (first run may re-pay precompile, second is the fair warm-cache/fresh-process number):
```bash
julia --project=. -e '
using iTEBD
X = ComplexF64[0 1; 1 0]; G = kron(X, X)
@time begin
    ψ = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
    applygate!(ψ, G, 1, 2; maxdim=4)
    evolve!(ψ, [(G, 1, 2), (G, 2, 1)], 3; maxdim=4)
end'
```
Expected: prints a `@time` line like `X.XXX seconds (… allocations)`. Note the seconds value as **DENSE_BASELINE**.

- [ ] **Step 3: Measure symmetric path latency via the heaviest test file**

Warm once, then time it (run the timed command 2–3 times, take the smallest `real`):
```bash
julia --project=. -e 'using Pkg; Pkg.test(test_args=["test_symmetric_basic.jl"])'
time julia --project=. -e 'using Pkg; Pkg.test(test_args=["test_symmetric_basic.jl"])'
```
Expected: tests pass; note the `real` time as **SYM_BASELINE**.

- [ ] **Step 4: Record the baseline**

Edit the `## Measurements` table at the bottom of this plan: put DENSE_BASELINE and SYM_BASELINE in the **Baseline** column. Leave the **After** column for Task 6. No commit yet — Task 6 commits the plan once both columns are filled.

---

### Task 2: Add the PrecompileTools dependency

**Files:**
- Modify: `Project.toml`

- [ ] **Step 1: Add to `[deps]`**

In `Project.toml`, add this line inside the `[deps]` block (alphabetical-ish; placement is not significant to Julia):
```toml
PrecompileTools = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
```

- [ ] **Step 2: Add to `[compat]`**

In the `[compat]` block add:
```toml
PrecompileTools = "1"
```

- [ ] **Step 3: Resolve and verify it loads**

Run:
```bash
julia --project=. -e 'using Pkg; Pkg.resolve(); Pkg.instantiate(); using PrecompileTools; println("PrecompileTools OK")'
```
Expected: ends with `PrecompileTools OK`, exit 0. `Manifest.toml` may update.

- [ ] **Step 4: Commit**

```bash
git add Project.toml Manifest.toml
git commit -m "build: add PrecompileTools dependency for precompile workloads"
```
(If `Manifest.toml` is gitignored, omit it from the `git add`.)

---

### Task 3: Add the workload-body regression test

Write the test first so the workload operations are validated as sound before they run at precompile time.

**Files:**
- Create: `test/test_precompile_workload.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Create `test/test_precompile_workload.jl`**

```julia
using Test
using iTEBD
using iTEBD: product_iMPS, applygate!, evolve!, expect, ent_S, spin_half_ops
using TensorKit: id, ⊗, codomain

# These two @testsets mirror, line for line, the bodies of the
# @compile_workload blocks in src/iTEBD.jl and ext/iTEBDTensorKitExt.jl.
# If the public API drifts, this test fails with a clear error instead of
# surfacing as an opaque precompilation failure that blocks `using iTEBD`.

@testset "precompile workload bodies execute" begin
    @testset "dense core" begin
        X = ComplexF64[0 1; 1 0]
        G = kron(X, X)
        ψ = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
        applygate!(ψ, G, 1, 2; maxdim=4)
        evolve!(ψ, [(G, 1, 2), (G, 2, 1)], 3; maxdim=4)
        @test expect(ψ, G, 1, 2) isa Number
        @test ent_S(ψ, 1) isa Real
    end

    @testset "symmetric U(1)" begin
        ψ = product_iMPS(:U1, [-1, 1], [1, -1])
        P = codomain(ψ.Γ[1])[2]
        Iop = id(ComplexF64, P ⊗ P)
        applygate!(ψ, Iop, 1, 2; maxdim=8)
        evolve!(ψ, [(Iop, 1, 2), (Iop, 2, 1)], 3; maxdim=8)
        Sz, _, _, _ = spin_half_ops(:U1)
        @test expect(ψ, Sz, 1, 1) isa Number
        @test ent_S(ψ, 1) isa Real
    end
end
```

- [ ] **Step 2: Register it in the `smoke` group**

In `test/runtests.jl`, change the `"smoke"` entry of `TEST_GROUPS` from:
```julia
    "smoke" => [
        "test_docs_smoke.jl",
    ],
```
to:
```julia
    "smoke" => [
        "test_docs_smoke.jl",
        "test_precompile_workload.jl",
    ],
```

- [ ] **Step 3: Run the new test and verify it PASSES**

Run:
```bash
julia --project=. -e 'using Pkg; Pkg.test(test_args=["test_precompile_workload.jl"])'
```
Expected: `Test Summary:` shows all passing, exit 0. (It passes immediately — the API is already correct; this is a regression guard, not red→green.)

- [ ] **Step 4: Commit**

```bash
git add test/test_precompile_workload.jl test/runtests.jl
git commit -m "test: guard precompile workload bodies against API drift"
```

---

### Task 4: Add the dense-core compile workload

**Files:**
- Modify: `src/iTEBD.jl` (insert before the module's final `end` on the last line)

- [ ] **Step 1: Add the workload block**

In `src/iTEBD.jl`, the file currently ends with:
```julia
include("ScarFinder.jl")
include("SymmetricStubs.jl")

end
```
Insert the workload between the last `include` and the final `end`:
```julia
include("ScarFinder.jl")
include("SymmetricStubs.jl")

#---------------------------------------------------------------------------------------------------
# PRECOMPILE WORKLOAD
#---------------------------------------------------------------------------------------------------
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

end
```

- [ ] **Step 2: Verify the package precompiles cleanly**

Run:
```bash
julia --project=. -e 'using iTEBD; println("dense load OK")'
```
Expected: prints `Precompiling iTEBD...` (workload runs during this), then `dense load OK`, exit 0. **No errors or stacktraces.** If the workload throws, this step fails — fix the call before proceeding (do not commit a package that won't precompile).

- [ ] **Step 3: Run the dense-touching tests to confirm no behavior change**

Run:
```bash
ITEBD_TEST_GROUP=unit julia --project=. -e 'using Pkg; Pkg.test()'
```
Expected: all green, exit 0.

- [ ] **Step 4: Commit**

```bash
git add src/iTEBD.jl
git commit -m "perf: add dense-core PrecompileTools workload to cut TTFX"
```

---

### Task 5: Add the symmetric-extension compile workload

**Files:**
- Modify: `ext/iTEBDTensorKitExt.jl` (insert before `end # module` on the last line)

- [ ] **Step 1: Add the workload block**

In `ext/iTEBDTensorKitExt.jl`, the file ends with `end # module`. Immediately before that final line, insert:
```julia
# ─────────────────────────────────────────────────────────────────────────────
# Precompile workload (symmetric / graded path)
# ─────────────────────────────────────────────────────────────────────────────
using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    @compile_workload begin
        ψ = product_iMPS(:U1, [-1, 1], [1, -1])
        P = codomain(ψ.Γ[1])[2]
        Iop = id(ComplexF64, P ⊗ P)
        applygate!(ψ, Iop, 1, 2; maxdim=8)
        evolve!(ψ, [(Iop, 1, 2), (Iop, 2, 1)], 3; maxdim=8)
        Sz, _, _, _ = spin_half_ops(:U1)
        expect(ψ, Sz, 1, 1)
        ent_S(ψ, 1)
    end
end

end # module
```
Name resolution check: `product_iMPS` and `spin_half_ops` are imported at the top of the file; `applygate!`/`evolve!`/`expect`/`ent_S` come via `using iTEBD`; `id`/`⊗`/`codomain` come via `using TensorKit`. All resolve as bare names here — no `iTEBD.` qualification needed.

- [ ] **Step 2: Verify the extension precompiles cleanly**

The extension only loads when TensorKit is present, which is true inside the test environment. Trigger it:
```bash
julia --project=. -e 'using Pkg; Pkg.test(test_args=["test_precompile_workload.jl"])'
```
Expected: prints precompilation of `iTEBDTensorKitExt` (workload runs here), then the regression test passes, exit 0. **No errors during precompilation.** If the workload throws, this fails — fix before committing.

- [ ] **Step 3: Run the full symmetric test file to confirm no behavior change**

Run:
```bash
julia --project=. -e 'using Pkg; Pkg.test(test_args=["test_symmetric_basic.jl"])'
```
Expected: all green, exit 0.

- [ ] **Step 4: Commit**

```bash
git add ext/iTEBDTensorKitExt.jl
git commit -m "perf: add symmetric-path PrecompileTools workload to cut TTFX"
```

---

### Task 6: Verify suite green and measure the TTFX delta

**Files:**
- Modify: `docs/superpowers/plans/2026-05-28-test-speedup-precompile.md` (fill the `## Measurements` section)

- [ ] **Step 1: Full default suite green**

Run:
```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```
Expected: all groups pass, exit 0. This is the stability gate — no behavior regressions.

- [ ] **Step 2: Re-measure dense first-call latency**

Same probe as Task 1 Step 2 (run twice, record the second run's `@time`):
```bash
julia --project=. -e '
using iTEBD
X = ComplexF64[0 1; 1 0]; G = kron(X, X)
@time begin
    ψ = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
    applygate!(ψ, G, 1, 2; maxdim=4)
    evolve!(ψ, [(G, 1, 2), (G, 2, 1)], 3; maxdim=4)
end'
```
Record as **DENSE_AFTER**.

- [ ] **Step 3: Re-measure symmetric path**

Same as Task 1 Step 3 (warm once, time 2–3 runs, take smallest `real`):
```bash
julia --project=. -e 'using Pkg; Pkg.test(test_args=["test_symmetric_basic.jl"])'
time julia --project=. -e 'using Pkg; Pkg.test(test_args=["test_symmetric_basic.jl"])'
```
Record as **SYM_AFTER**.

- [ ] **Step 4: Fill in the Measurements section and commit**

Replace the placeholder `## Measurements` block at the bottom of this plan with the real numbers and a one-line verdict (expect DENSE_AFTER < DENSE_BASELINE and SYM_AFTER ≤ SYM_BASELINE; precompile time itself rises — that's the accepted tradeoff).

```bash
git add docs/superpowers/plans/2026-05-28-test-speedup-precompile.md
git commit -m "docs: record TTFX before/after for precompile workloads"
```

---

## Measurements

_(Filled in during Task 1 and Task 6.)_

| Probe | Baseline | After | Delta |
|-------|----------|-------|-------|
| Dense first-call `@time` (s) | TBD (Task 1) | TBD (Task 6) | — |
| `test_symmetric_basic.jl` wall-clock `real` (s) | TBD (Task 1) | TBD (Task 6) | — |

Verdict: _(filled in Task 6)_
