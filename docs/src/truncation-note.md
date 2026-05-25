# Truncation note

In one line: the gate-evolution controller picks the smallest bond dimension
whose discarded weight falls below `truncerr`, then clamps the result to
`[mindim, maxdim]`. The older entropy-rank heuristic is still available as an
opt-in policy (`chi_policy=:adaptive`) through [`adaptive_bonddim`](@ref) and
[`natural_bonddim`](@ref).

## Gate-evolution semantics

`applygate!` and both `evolve!` methods share the same truncation interface:

- `maxdim` — hard cap on the kept bond dimension.
- `mindim` — minimum kept bond dimension.
- `truncerr` — target discarded weight per updated bond.
- `svd_min` — absolute floor on retained singular values.
- `return_stats` — return per-bond and per-gate diagnostics alongside the
  mutated state.
- `chi_policy=:fixed` (default) — use the discarded-weight controller described
  below.
- `chi_policy=:adaptive` — ratchet the bond dimension with `adaptive_bonddim`,
  parameterized by `q`, `alpha`, and the `svd_min`-derived cutoff.

Given the post-gate singular values `s`, the `:fixed` controller proceeds as:

1. normalize `p = abs2.(s) / sum(abs2.(s))`,
2. define `w_disc(χ) = sum(p[χ+1:end])`,
3. choose the smallest `χ` with `w_disc(χ) <= truncerr`,
4. clamp `χ` to `[mindim, maxdim]`.

When `truncerr` cannot be reached within `maxdim`, the update keeps `maxdim`
and flags the bond as saturated in the returned statistics.

## Typical settings

A conservative default for moderate Trotter steps is
`truncerr=1e-10, maxdim=64`. Tighten `truncerr` (for example `1e-12`) when
running long imaginary-time cooling or when an observable depends on
small singular values; loosen it (toward `1e-8`) when the simulation is
saturating `maxdim` and you accept a controlled loss of accuracy in exchange
for a smaller bond dimension. `mindim` mainly matters during the early steps
of a quench, where it prevents the bond from collapsing before correlations
build up. Leave `svd_min` near machine epsilon unless the singular spectrum
has a clear numerical noise floor that should be cut.

Passing `return_stats=true` gives back per-bond `discarded_weight` and
`chi_keep`, the per-gate `max_discarded_weight` and `num_saturated`, and the
run-level aggregates `mean_discarded_weight` and `max_kept_dim`. These are
the right quantities to log when verifying that a chosen `(truncerr, maxdim)`
pair is actually doing what you expect: persistent saturation means `maxdim`
is too small for the requested accuracy, while a `max_discarded_weight`
several orders of magnitude below `truncerr` means you can shrink `maxdim`
without harm.

## When to use `:adaptive`

`chi_policy=:adaptive` remains useful when you want a ratchet on bond growth
rather than a per-step optimum — for example, slow thermalization runs where
the entanglement creeps up over many Trotter steps and you would rather have
the bond dimension grow monotonically through `adaptive_bonddim` than be
recomputed independently at every bond. It is also a reasonable choice when
the Schmidt spectrum has a long, slow tail and the bond-local discarded
weight underestimates the dimension actually needed downstream.

## `truncerr` versus `cutoff`

Gate-evolution truncation and canonicalization use distinct thresholds, and
this is the most common source of confusion. `truncerr` is the discarded
weight target for the `:fixed` controller during `applygate!` and `evolve!`.
The `cutoff` keyword used by `canonical!` and the ScarFinder projection
helpers is an absolute floor on retained singular values during
canonicalization and fixed-dimension compression. Setting one does not affect
the other; both are typically active during a full evolution.

## Why the old heuristic was replaced

The previous policy estimated `χ` from entropy rank, participation ratio,
Rényi ranks, and a tail-amplification factor. That estimator inferred bond
dimension from summary statistics rather than from the discarded weight that
the truncation actually introduces, so the relationship between its inputs
and the simulation error was indirect. The current `:fixed` controller is
bond-local, operates on the post-gate singular values produced by each
update, allows `χ` to grow or shrink in response to the spectrum, and
reports saturation when the requested accuracy is not reachable under
`maxdim`. The `:adaptive` policy remains available for the cases above.
