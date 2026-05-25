# Code-review P0 triage — 2026-05-25

This file triages the five P0 findings produced by the fan-out review of
2026-05-25 (7 parallel agents, one per source slice). Each claim is verified by
a minimal reproduction or direct code reading. No source code is changed in
this pass — the goal is to confirm what is real before allocating a fix budget.

## Summary

| # | Claim | Status | Reachable from public API? |
|---|---|---|---|
| 1 | `trotter=:fourth` collapses to first-order Euler on single-layer Hamiltonians | **Confirmed bug** | Yes |
| 2 | `tensor_decomp!` crashes with `BoundsError` when `n == 1` | Confirmed in private function | **No** — `canonical!` guards |
| 3 | `iMPS` struct has no inner constructor enforcing invariants | **Confirmed hazard** | Yes (two paths) |
| 4 | Mixed `scarfinder!(ψ, G, h, χ, N; ...)` has zero test coverage | **Confirmed** | (test-suite gap, not a runtime bug) |
| 5 | `test_scarfinder_performance.jl` is excluded from the default CI run | **Confirmed** | (CI configuration) |

Three real bugs (1, 3) and a runtime-unreachable correctness defect (2). Two
test-suite gaps (4, 5) that should be addressed alongside any source fix so a
regression doesn't slip back in.

---

## P0 #1 — `trotter=:fourth` silently degrades to Euler on a single layer

**Claim.** The 5 Suzuki substeps `(p, p, q, p, p)` of `trotter=:fourth` all
target layer 1 when `num_layers == 1`, so `_push_trotter_stage!` merges them
into a single stage with coefficient `2p + q + 2p = 4p + q = 1`, equivalent to
one Euler step `exp(-i·dt·H)`.

**Repro.**

```julia
using iTEBD
H = ComplexF64[1 0; 0 -1]
layers = [[(H, 1, 1)]]  # one commuting layer with one 1-site term

gates_second = iTEBD.trotter_gates(layers, 0.1; trotter=:second, evolution=:real)
gates_fourth = iTEBD.trotter_gates(layers, 0.1; trotter=:fourth, evolution=:real)

length(gates_second)   # => 1
length(gates_fourth)   # => 1   (should be 5 substeps' worth)
gates_fourth[1][1][1,1] # => 0.9950041652780258 - 0.09983341664682815im
exp(-0.1im)             # => 0.9950041652780258 - 0.09983341664682815im
```

The `:fourth` gate is bit-identical to one Euler step, confirming the collapse.

**Conclusion.** Real, user-reachable, silent accuracy regression. The
existing validator [`_validate_trotter_scheme`](src/Gate.jl:313) rejects
`:fourth_opt` with `num_layers != 2` but does not reject `:fourth` with
`num_layers == 1`.

**Status: CONFIRMED P0.**

---

## P0 #2 — `tensor_decomp!` BoundsError with `n == 1`

**Claim.** The function pre-allocates `Γs` of length `n` and `λs` of length
`n-1`, then the trailing epilogue at [TensorAlgebra.jl:742-743](src/TensorAlgebra.jl:742)
writes `Γs[n-1] = Γs[0]` and `λs[n-1] = λs[0]` when `n == 1`.

**Repro.**

```julia
using iTEBD
Γ = randn(ComplexF64, 4, 2, 4)
λ = ones(4)
iTEBD.tensor_decomp!(Γ, λ, 1)
# => BoundsError: attempt to access 1-element Vector{Array{ComplexF64, 3}} at index [0]

# But the user-facing path:
psi = rand_iMPS(ComplexF64, 1, 2, 4)
canonical!(psi)   # => succeeds; ψ.n = 1, length(ψ.λ) = 1
```

**Conclusion.** The bug is real, but the only caller in the source tree
([Schmidt.jl:275](src/Schmidt.jl:275)) short-circuits with `if isone(n)` and
returns directly without ever calling `tensor_decomp!`. The commented-out
`canonical_trim` at Schmidt.jl:295 would have hit it but is dead code.

**Status: Real bug, but P0 → P2** — defense-in-depth fix only;
not currently reachable through documented APIs.

---

## P0 #3 — `iMPS` struct has no invariants enforced

**Claim.** The struct ([iMPS.jl:34-38](src/iMPS.jl:34))

```julia
struct iMPS{T<:Number, S<:Real}
    Γ::Vector{Array{T, 3}}
    λ::Vector{Vector{S}}
    n::Int
end
```

has no inner constructor, so `iMPS(Γ, λ, n)` will accept any combination of
field values regardless of whether `length(Γ) == length(λ) == n` and regardless
of whether bond dimensions agree at the wraparound seam.

**Repro.**

```julia
using iTEBD

# 1. Mismatched field lengths.
psi = iMPS([randn(ComplexF64, 2, 2, 2)], [ones(2), ones(2)], 3)
# => Constructed without error: length(Γ)=1, length(λ)=2, n=3.

# 2. Outer constructor with renormalize=false bypasses canonical! and accepts
#    bond-dim mismatch at the wraparound:
Γs = [randn(ComplexF64, 2, 2, 3), randn(ComplexF64, 5, 2, 2)]
psi = iMPS(ComplexF64, Γs; renormalize=false)
# => Constructed without error: Γ[1] right=3, Γ[2] left=5 — mismatch.
#    Any subsequent contraction (e.g. inner_product) will crash deep inside
#    a reshape or kron with an unhelpful error.
```

**Conclusion.** Real. The default `renormalize=true` keeps users out of trouble
because `canonical!` rebuilds a consistent state, but the `renormalize=false`
path and the bare `iMPS(Γ, λ, n)` constructor are silent footguns.

**Status: CONFIRMED P0.** Add an inner constructor that asserts
`length(Γ) == length(λ) == n`, that each `size(Γ[i], 3) == length(λ[i])`, that
`size(Γ[mod1(i+1, n)], 1) == length(λ[i])`, and that `λ[i]` are real, finite,
non-negative.

---

## P0 #4 — Mixed `scarfinder!(ψ, G, h, χ, N; ...)` has no test coverage

**Claim.** The marquee mixed-interface form of `scarfinder!` is never invoked
in `test/`.

**Verification.**

```
$ grep -rn 'scarfinder!' test/
test/test_scarfinder_performance.jl:192:    @test_logs (:warn, r"nstep = 1") iTEBD.scarfinder!(ψ_copy,  h, dt, χ, N; ...)
test/test_scarfinder_performance.jl:197:    @test_logs (:warn, r"nstep = 1") iTEBD.scarfinder!(ψ_copy2, h, dt, χ, N; ...)
```

Two matches, both calling the Hamiltonian-based interface `(ψ, h, dt, χ, N)`.
None call the mixed `(ψ, G, h, χ, N)` form. Both calls live in
`test_scarfinder_performance.jl`, which is in the `bench` group and is
**excluded from the default CI run** (see #5).

So neither the mixed interface nor the Hamiltonian interface has effective
default-CI coverage. The gate-only interface is exercised indirectly through
the internal helpers in `test_scarfinder_api.jl` but not at the public level.

**Conclusion.** The marquee feature has zero default-CI convergence coverage on
two of its three public signatures.

**Status: CONFIRMED P0.** Add a convergence test for each of the three
interfaces.

---

## P0 #5 — `test_scarfinder_performance.jl` is excluded from the default run

**Claim.** Despite being a regression suite (not a benchmark), the file is
grouped under `bench`, which `default` does not include.

**Verification.**

```julia
# test/runtests.jl
const TEST_GROUPS = Dict(
    ...
    "bench" => [
        "test_bench_smoke.jl",
        "test_scarfinder_performance.jl",
        "test_pxp_legacy_smoke.jl",
    ],
    ...
)

const TEST_ALIASES = Dict(
    "default" => ["unit", "api", "smoke", "integration"],
    "fast"    => ["unit", "api", "smoke"],
    "all"     => ["unit", "api", "smoke", "bench", "integration"],
    ...
)
```

And `test_scarfinder_performance.jl` itself is a regression suite — its
top-of-file comments are titled `Fix 1: _truncate_unitcell! should not crash on
large unit cells`, `Fix 2: ...`, etc. There are no `@elapsed`, `@allocated`, or
timing assertions in the file.

**Conclusion.** Misnamed and misclassified. Renaming to
`test_scarfinder_regression.jl` and moving into the `unit` group would make the
regressions execute on every CI run.

**Status: CONFIRMED P0.** Fix is a one-line edit in runtests.jl plus a file
rename.

---

## Recommended fix order

When a fix budget opens up, work in this order. Each item is a single PR:

1. **(P0 #5) Move `test_scarfinder_performance.jl` into the `unit` group**, optionally renaming to `test_scarfinder_regression.jl`. Single-line edit. Done first so subsequent fixes are guarded by these regression tests immediately.

2. **(P0 #4) Add convergence tests for all three `scarfinder!` interfaces.** Three short tests, each seeded RNG, each asserts an objective improves (e.g. AKLT energy gap closing, or PXP scar overlap increasing). Done before #3 so any fix to ScarFinder has a regression net.

3. **(P0 #3) Add an inner constructor to `iMPS`.** Enforce `length(Γ) == length(λ) == n`, bond-dim consistency at the wraparound, and `λ ≥ 0`. Decide policy for `renormalize=false`: still permit construction, but consider running the same invariant check after `Γ` and `λ` are populated. Add a test that the constructor rejects malformed inputs with a clear `ArgumentError`.

4. **(P0 #1) Fix `trotter=:fourth` for single-layer Hamiltonians.** Two reasonable resolutions:
   - **Reject at the validator**: extend [`_validate_trotter_scheme`](src/Gate.jl:313) to require `num_layers >= 2` for `:fourth` (mirroring the existing `:fourth_opt` rule). Users wanting fourth-order on a single-term Hamiltonian must split it artificially or use `:second`.
   - **Special-case the merge**: in `_push_trotter_stage!`, don't merge two consecutive same-layer stages when they come from a higher-order Trotter decomposition. (Riskier; affects the cache key path too.)
   - The first is conservative and easier to ship; the second changes semantics for any same-layer adjacent stages from `:fourth_opt` as well, so it needs more thought.
   - Add a test that builds a single-layer Hamiltonian and asserts `length(trotter_gates(layers, dt; trotter=:fourth))` matches the expected substep count, or that an `ArgumentError` is raised — whichever resolution is chosen.

5. **(P0 #2 → P2) Defense-in-depth fix to `tensor_decomp!`.** Guard `n == 1` upfront, returning `([Γ], Vector{Vector{eltype(λl)}}())`. Add a unit test calling `tensor_decomp!` directly with `n == 1`. Not urgent — no caller currently passes `n == 1` — but the function is reachable via `iTEBD.tensor_decomp!` from user code.

## Notes for the next pass

- The P1 list from the original review (silent-correctness items: wrap-around re-canonicalization, single-site normalization, `_energy_fix!` warning, `_safe_reciprocal` default, refinement bond choice, hard-coded Krylov tolerances, etc.) was not verified in this pass. Each should be re-checked the same way (read code + minimal repro) before fixing.

- The agents flagged `imps2mps` ([src/ITensors.jl](src/ITensors.jl)) as undocumented. This is true but lower-impact: the function works, it just isn't reachable via `using iTEBD.ITensors` because that resolves to the upstream `ITensors` package, not the local file. Worth renaming the file to `IMPSInterop.jl` or similar in the same pass as the inner-constructor work.

- The fan-out review itself was useful: 7 agents, ~60 raw findings, ~30 unique
  after dedup, of which 5 reached P0. Most P0 items survived verification —
  the agents were calibrated well.
