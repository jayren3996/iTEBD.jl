# Adaptive Bond-Dimension Benchmark Report

## Executive Summary

The discarded-weight-controlled adaptive scheme is **scientifically usable**, but the benchmark evidence does **not** support making it the default recommendation for `iTEBD.jl`.

Final recommendation: **keep only for certain regimes**.

The data support three main conclusions:

1. In low-entanglement or already-simple flows, adaptive truncation can be efficient, but only with relatively loose thresholds. Tight thresholds tend to overgrow `chi` and hit the hard cap without improving observables.
2. In the exact TFIM real-time benchmark, adaptive truncation is only competitive at very low wall-time budgets and moderate accuracy targets. On the best high-accuracy frontier, fixed `chi` is better.
3. In the interacting XXZ and structured PXP revival tests, adaptive truncation is competitive as a convenience heuristic, but it does not dominate fixed `chi`. Its benefits are incremental, not decisive.

All raw data, figures, and per-benchmark summaries are saved under `benchmark_results/`.

## Method Summary

Each benchmark was run in the required order:

1. timestep study first,
2. trusted reference construction,
3. fixed-cap sweep,
4. adaptive discarded-weight sweep,
5. cost/accuracy interpretation.

Recorded diagnostics include wall time, allocated bytes, observable time series, entanglement entropy, bond dimensions on inequivalent bonds, discarded-weight histories, and saturation counts. Reported trust times are defined from observable error relative to the selected reference.

## Table A: Reference Setups

| Benchmark | Model | Initial state | Production `dt` | Reference method | Reference settings | Comparison window |
| --- | --- | --- | --- | --- | --- | --- |
| AKLT imaginary time | Spin-1 AKLT projector Hamiltonian | Random bond-1 two-site iMPS | `0.1` | Exact AKLT state plus tight dt study | Exact AKLT MPS; dt study with `maxdim=128`, `truncerr=1e-14` | `τ ∈ [0, 8]` |
| TFIM quench | `H = -∑ X_i X_{i+1} - 1.3 ∑ Z_i` | Uniform `|↑_z⟩` product state | `0.0125` | Exact free-fermion Majorana evolution on large open chain | Exact covariance evolution, `L=128`; validation error `~1e-16`; finite-size drift negligible | `t ∈ [0, 4]` |
| XXZ Néel quench | `Jxy=1`, `Jz=1.2` XXZ chain | Two-site Néel product state | `0.025` | Highly converged fixed-cap iTEBD | Reference `dt=0.0125`, `chi=128`; probe `chi=96`; max reference disagreement up to `T=2` was `|Δm_stag|≈3.4e-3` | `t ∈ [0, 2]` |
| PXP revivals | 6-site unit-cell PXP Trotterization with 3-site terms | 6-site `Z2` product state | `0.1` | Highly converged fixed-cap iTEBD | Reference `dt=0.025`, `chi=128`; probe `chi=96`; no observable difference at recorded precision | `t ∈ [0, 10]` |

## Benchmark 1: AKLT Imaginary Time

Headline result: adaptive truncation is only clearly useful here when the target is loose enough that it settles back to the true AKLT bond dimension instead of chasing tiny local tails.

The timestep study showed that halving `dt` from `0.1` to `0.05` changed the final energy by only `2.27e-7` and the final `SzSz` correlator by `1.65e-6`, so `dt=0.1` was used for the production comparison. The exact AKLT fixed point has bond dimension `2`, which makes this benchmark a clean overgrowth test.

The best fixed-cap run was `chi=4`, with final energy error `5.28e-6`, peak `chi=4`, and wall time `0.186 s`. The best adaptive run was `truncerr=1e-4, maxdim=8`, with final energy error `4.33e-6`, peak `chi=5`, and wall time `0.134 s`. That adaptive run also settled to late-time `chi=2`, so it behaved sensibly. Tightening the adaptive target was not monotone: for `truncerr <= 1e-6`, the method overgrew, frequently saturated the ceiling, and kept late-time `chi` well above the exact AKLT value even though the physics was already converged.

Verdict for AKLT: adaptive is acceptable as an efficiency convenience when the user does **not** know the required `chi` a priori, but fixed `chi=2` or `4` remains simpler and more predictable for this regime.

Figures:

- [Energy error vs imaginary time](benchmark_results/figures/aklt_imaginary_time/energy_error_vs_imaginary_time.png)
- [Bond dimension vs imaginary time](benchmark_results/figures/aklt_imaginary_time/chi_vs_imaginary_time.png)
- [Entanglement vs imaginary time](benchmark_results/figures/aklt_imaginary_time/entanglement_vs_imaginary_time.png)
- [Final error vs wall time](benchmark_results/figures/aklt_imaginary_time/final_error_vs_wall_time.png)

## Benchmark 2: TFIM Exact Quench

Headline result: adaptive truncation does **not** beat fixed `chi` on the high-accuracy frontier, although it is competitive at very low wall-time budgets and moderate error thresholds.

This benchmark used an exact free-fermion reference. The Majorana reference matched small-chain dense exact evolution at machine precision, and the open-chain finite-size check was negligible. The smallest tested timestep, `dt=0.0125`, was needed because `dt=0.025` still gave `max |Δ⟨Z⟩| ≈ 1.31e-3` over `T=4`. As a result, the `1e-3` trust-time threshold is partly Trotter-limited even in the tight run, so the most informative comparisons are `T_{5e-3}` and `T_{1e-2}`.

At the useful `5e-3` threshold, fixed `chi=48` reached the full window `T=4.0` with RMS magnetization error `8.78e-4` in `0.465 s`. The best adaptive full-window run was `truncerr=1e-8, maxdim=64`, with RMS error `1.20e-3` in `0.839 s`. So fixed `chi` was better at the same trust time and lower cost. Adaptive only helped at the very low-budget end: below about `0.1 s`, `truncerr=1e-8, maxdim=16` extended `T_{5e-3}` beyond the cheapest fixed run, but that advantage disappeared once moderate fixed caps (`chi≈24-48`) were allowed.

Verdict for TFIM: adaptive is **not** the preferred default for controlled real-time work. It is a low-budget heuristic, not the best accuracy/cost strategy.

Figures:

- [Magnetization vs time](benchmark_results/figures/tfim_quench/zmag_vs_time.png)
- [Magnetization error vs time](benchmark_results/figures/tfim_quench/zmag_error_vs_time.png)
- [Energy drift vs time](benchmark_results/figures/tfim_quench/energy_drift_vs_time.png)
- [Discarded weight vs time](benchmark_results/figures/tfim_quench/discarded_weight_vs_time.png)
- [Trust time vs peak chi](benchmark_results/figures/tfim_quench/trust_time_vs_peak_chi.png)
- [Error vs wall time](benchmark_results/figures/tfim_quench/error_vs_wall_time.png)

## Benchmark 3: XXZ Néel Quench

Headline result: adaptive truncation is competitive in the interacting case, but it still does not dominate fixed `chi`.

The first pass to `T=3` showed that the nominal reference was not converged at late times, so the official comparison window was shortened to `T=2`. That matters: the final XXZ numbers below are taken only from the interval where `chi=96` and `chi=128` agreed closely enough to use `chi=128, dt=0.0125` as a credible reference. With that change, the production timestep became `dt=0.025`.

Fixed `chi=96` reproduced the full `T=2` window at the `5e-3` threshold with RMS staggered-magnetization error `2.39e-4` in `1.55 s`. Adaptive `truncerr=1e-6, maxdim=96` also reached the full window, but with larger RMS error `1.09e-3`; its advantage was cost, not accuracy, at `1.17 s`. Tighter adaptive settings did not help much because they drove the run toward the same hard ceiling. So adaptive can save some time at a fixed ceiling, but it is not a clear win in observable accuracy.

Verdict for XXZ: adaptive is plausible as an optional budget-saving heuristic, especially when the user is unsure what `chi` to choose, but it should not be sold as a superior accuracy controller in generic interacting dynamics.

Figures:

- [Staggered magnetization vs time](benchmark_results/figures/xxz_neel_quench/mstag_vs_time.png)
- [Magnetization error vs time](benchmark_results/figures/xxz_neel_quench/mstag_error_vs_time.png)
- [Energy drift vs time](benchmark_results/figures/xxz_neel_quench/energy_drift_vs_time.png)
- [Bond dimension vs time](benchmark_results/figures/xxz_neel_quench/chi_vs_time.png)
- [Trust time vs peak chi](benchmark_results/figures/xxz_neel_quench/trust_time_vs_peak_chi.png)
- [Error vs wall time](benchmark_results/figures/xxz_neel_quench/error_vs_wall_time.png)

## Benchmark 4: PXP Revivals

Headline result: adaptive truncation preserves the revival structure well, but the gain over a small fixed cap is modest because the structured dynamics are already cheap.

The PXP reference was well converged: the `chi=96` and `chi=128` runs at `dt=0.025` were identical at recorded precision, and the timestep study showed that even `dt=0.1` kept the imbalance error comfortably below the chosen `2e-2` trust threshold over the full `T=10` window. That made the production comparison inexpensive and clean.

Fixed `chi=8` already preserved the full revival window with RMS imbalance error `9.49e-4`, max peak-height error `1.50e-3`, and wall time `0.648 s`. Adaptive `truncerr=1e-6, maxdim=32` also preserved the full window and was slightly faster at `0.550 s`, but its RMS imbalance error and peak-height error were both somewhat worse. So adaptive is viable here, but the benefit is incremental because the fixed-`chi` baseline is already strong.

Verdict for PXP: adaptive is neutral-to-mildly-useful. It does not appear harmful for revival physics, but it is not essential to reach good results.

Figures:

- [Revival observable vs time](benchmark_results/figures/pxp_revivals/imbalance_vs_time.png)
- [Imbalance error vs time](benchmark_results/figures/pxp_revivals/imbalance_error_vs_time.png)
- [Peak error vs index](benchmark_results/figures/pxp_revivals/peak_error_vs_index.png)
- [Bond dimension vs time](benchmark_results/figures/pxp_revivals/chi_vs_time.png)
- [Discarded weight vs time](benchmark_results/figures/pxp_revivals/discarded_weight_vs_time.png)
- [Error vs wall time](benchmark_results/figures/pxp_revivals/error_vs_wall_time.png)

## Table B: Fixed vs Adaptive Headline Comparison

| Benchmark | Best fixed run | Best adaptive run | Observable metric | Trust time | Peak `chi` | Wall time | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AKLT | `chi=4` | `truncerr=1e-4, maxdim=8` | Final energy error `5.28e-6` vs `4.33e-6` | `—` | `4` vs `5` | `0.186 s` vs `0.134 s` | Adaptive helps only when loose enough to shrink back to `chi=2`; tight targets overgrow |
| TFIM | `chi=48` | `truncerr=1e-8, maxdim=64` | RMS `|Δ⟨Z⟩|`: `8.78e-4` vs `1.20e-3` | `T_{5e-3}=4.0` for both | `48` vs `64` | `0.465 s` vs `0.839 s` | Fixed is better on the full-window frontier |
| XXZ | `chi=96` | `truncerr=1e-6, maxdim=96` | RMS `|Δm_stag|`: `2.39e-4` vs `1.09e-3` | `T_{5e-3}=2.0` for both | `96` vs `96` | `1.55 s` vs `1.17 s` | Adaptive is cheaper, but noticeably less accurate |
| PXP | `chi=8` | `truncerr=1e-6, maxdim=32` | RMS `|ΔI|`: `9.49e-4` vs `1.41e-3` | `T_{2e-2}=10.0` for both | `8` vs `14` | `0.648 s` vs `0.550 s` | Adaptive is slightly cheaper but not meaningfully better |

## Table C: Failure Modes

| Benchmark | Excessive `chi` overgrowth | `chi` chatter / instability | Frequent saturation | Non-monotonic tightening | Wrong physics despite small local discarded weight |
| --- | --- | --- | --- | --- | --- |
| AKLT | Yes, for tight thresholds | Mild | Yes | Yes | No |
| TFIM | Moderate at tight thresholds | No clear chatter | Yes | Yes | Yes, at moderate thresholds and low ceilings |
| XXZ | Moderate at tight thresholds | No clear chatter | Yes | Mildly | No clear case once the ceiling was generous |
| PXP | Mild | No clear chatter | Only for the tightest low-ceiling cases | Mildly | No |

## Final Recommendation

Recommendation: **keep only for certain regimes**.

Why:

- The adaptive scheme is never catastrophically bad in the benchmarked regimes when the user monitors saturation and uses a reasonable `maxdim`.
- It does produce genuine wins in a few narrow cases: low-budget TFIM runs, AKLT with a loose target, and modest speedups in PXP or XXZ at the same hard ceiling.
- But it is not robustly better than fixed `chi`, and on the strongest controlled benchmark we have, the exact TFIM quench, fixed `chi` is the better high-accuracy choice.

What should be documented for users:

- `truncerr` is a **local** controller, not a global accuracy guarantee.
- Small discarded weight does **not** guarantee small observable error at long times.
- Tight `truncerr` with a small `maxdim` often just produces repeated saturation and unnecessary `chi` growth.
- If the required bond dimension is already known or easy to estimate, fixed `chi` remains the more predictable workflow.
- Adaptive truncation is safest as an exploratory mode with explicit monitoring of `num_saturated`, `max_discarded_weight`, and the actual observable convergence.
