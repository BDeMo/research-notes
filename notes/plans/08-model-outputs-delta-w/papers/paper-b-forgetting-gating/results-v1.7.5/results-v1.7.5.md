# Results v1.7.5 (Qwen3-8B) — scaling diagnosis, minimize-scale, stable training

> **Builds on [v1.7.3](../results-v1.7.3/results-v1.7.3.md)** (the leak-free clean re-run + do-no-harm gate).
> v1.7.4 added a length-adaptive memory + the 2025 baselines + the scaling study; v1.7.5 adds **training-stability
> rigor** and the **HP sweep / scaling critical review / minimize-scale** investigation. Setup is shared with
> v1.7.3 ([experimental-setup.md](../results-v1.7.3/experimental-setup.md)) plus the v1.7.5 deltas below.
> One-page brief: [`../summary-matrix-v1.7.3.md`](../summary-matrix-v1.7.3.md). Paper draft: [`../paper/main_v1.7.5.tex`](../paper/main_v1.7.5.tex).

## What v1.7.5 changed vs v1.7.3/v1.7.4
- **Training stability** (so the negatives are not an optimization artifact): gradient accumulation (effective
  batch 8), LR warmup + cosine decay, EMA-smoothed loss logging, and periodic held-out **val** loss. Default on.
- **Faithful 2025 baselines** (ICAE/AOC/Beacon/500x/ComprExIT/LCC) at matched budget — audit:
  [`baselines-faithfulness.md`](baselines-faithfulness.md).
- **Broad HP sweep + scaling critical review** (this version's main investigation).

## The documents (this version)
| doc | content |
|---|---|
| [`longcontext-improvement.md`](longcontext-improvement.md) | length-adaptive vs fixed-K (no long-ctx rescue), no-scaling result, OOD soft-memory probe, 2025-baseline table, training-dynamics fix (§11) |
| [`scaling-critical-review.md`](scaling-critical-review.md) | critical review of the scaling design (the capacity-K axis was untested), per-bench re-analysis, hypotheses H1–H5 + verdicts |
| [`baselines-faithfulness.md`](baselines-faithfulness.md) | per-method faithfulness audit of the 2025 context-compression baselines vs their papers |

## Headline findings (so far)
1. **Length-adaptive memory ≈ fixed-K** (Δ±0.02) — chunking does not rescue long context.
2. **No scaling** — avg compress flat across 24× FLOPs / data / depth; and (HP sweep) **capacity K does not scale either** (K16 ≈ K64 ≥ K256).
3. **Minimize-scale is the cure** — the smallest config (N≈4, K≈16) matches the big one at ~10× less compute.
4. **The memory is OOD** (nearest-vocab cos ≈ 0.10) — a mechanistic reason for the ceiling.
5. Pending: **H5 (is the ceiling OOD-induced? the manifold/norm fix)** + the minimize-scale Pareto + best-HP — see `scaling-critical-review.md` §5b.

→ Net: the **capacity-bound diagnosis** holds and strengthens; the contribution is the **do-no-harm gate** + minimize-scale, not a better compressor (diagnosis-led paper, `main_v1.7.5.tex`).

## Raw outputs (pods)
`out/clean/hp/` (HP sweep), `out/clean/min/` (minimize N×K grid), `out/clean/v175/` (stable cohort), `out/clean/scale/` (scaling grid). Analyzers: `analyze_hp.py`, `analyze_scale.py`, `plot_dynamics.py`.
