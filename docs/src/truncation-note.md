# Truncation Note

The default gate-evolution truncation policy is a bond-local controller based
on the actual discarded weight of the post-gate singular spectrum. The older
entropy-rank / effective-rank adaptive heuristic is no longer the default; it
remains available as an opt-in policy (`chi_policy=:adaptive`) via
[`adaptive_bonddim`](@ref) and [`natural_bonddim`](@ref).

## Current Gate-Evolution Semantics

For `applygate!` and both `evolve!` interfaces:

- `maxdim`: hard cap on the kept bond dimension,
- `mindim`: minimum kept bond dimension,
- `truncerr`: target local discarded weight,
- `svd_min`: optional absolute singular-value floor,
- `return_stats`: return truncation diagnostics instead of only mutating `ψ`,
- `chi_policy=:fixed` (default) uses the discarded-weight controller below;
  `chi_policy=:adaptive` ratchets the bond dimension with `adaptive_bonddim`
  using the keyword arguments `q`, `alpha`, and the `svd_min`-derived cutoff.

The local controller works directly from the post-gate singular values `s`:

1. normalize `p = abs2.(s) / sum(abs2.(s))`,
2. define `w_disc(χ) = sum(p[χ+1:end])`,
3. choose the smallest `χ` with `w_disc(χ) <= truncerr`,
4. clamp that choice between `mindim` and `maxdim`.

If the requested `truncerr` cannot be met within `maxdim`, the update keeps
`maxdim` and reports a saturated truncation in the returned statistics.

## Why The Old Heuristic Was Removed

The previous scheme estimated bond dimension from entropy rank, participation
ratio, Rényi ranks, and a tail-amplification factor. That heuristic was not the
standard quantity controlled in tensor-network time evolution: it inferred `χ`
from a summary statistic instead of using the actual discarded weight of the
post-gate singular spectrum.

The current controller is scientifically cleaner because:

- it is bond-local,
- it uses the actual singular values produced by the update,
- it lets `χ` grow or shrink naturally,
- it reports when the requested accuracy was not achievable under `maxdim`.

`canonical!` and the ScarFinder projection helpers still use their existing
`cutoff` keyword as an absolute singular-value threshold for canonicalization
and fixed-dimension compression. That is separate from the `truncerr`-based
gate-evolution controller.
