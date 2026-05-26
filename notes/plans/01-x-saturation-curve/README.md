# Plan 01 — X-Saturation Curve + Dataset Curator

> **Status**: drafting
> **Created**: 2026-05-26
> **Owner**: Mingjia
> **Parent idea**: brainstorm I1 (also entangles I3, I4, G6)
> **One-liner**: For every candidate training example, sweep the $X$-budget and measure when accuracy plateaus. The examples whose plateau is *below target* define the **W-residual** — these are exactly the problems where more inference compute can't help, only weight updates can. Curate training data on the W-residual.

---

## Problem

Today's data pipelines for SFT/RL/distillation use one of:
- random / class-balanced sampling (cheap, wasteful)
- loss-based hard mining (penalizes things the model can't yet handle, but *might* learn with more $X$)
- reward-variance / response-length heuristics (proxy at best)

None of these distinguish between:
- **Type-X-fixable**: the model fails at $X=$small, but succeeds at $X=$large. → No $W$ update needed; just allocate more $X$ at inference.
- **Type-W-residual**: even with maximal $X$ (long CoT, retrieval, search), the model still fails. → $W$ must change.

These two failure modes need *different* interventions. Mixing them wastes compute and confuses learning. The naive symptom: SFT runs that look productive on training loss but yield diminishing transfer on held-out.

## Core idea

Given a candidate set $\mathcal{D}_{\text{cand}} = \{(q_i, y_i^*)\}$ and a base model $f(\cdot; W_0)$:

1. For each $q_i$, compute the **$X$-saturation curve** $\alpha_i(b) = V\big(f(X_b(q_i); W_0)\big)$ where $b$ indexes $X$-budget (CoT length, sample count, retrieval depth, search width).
2. Define an example as **$X$-saturated** if $\alpha_i$ plateaus: $\alpha_i(b_{\max}) - \alpha_i(b_{\max}/4) < \epsilon$.
3. Define the **$W$-residual** subset $\mathcal{D}_W = \{i : X\text{-saturated} \land \alpha_i(b_{\max}) < \tau\}$.
4. Run SFT / DPO / RL on $\mathcal{D}_W$ only. Compare to full-set training at matched compute.

The hypothesis: **per dollar of training compute, SFT on $\mathcal{D}_W$ produces larger held-out gains than SFT on $\mathcal{D}_{\text{cand}}$ or any heuristic subset of the same size.**

## Success criteria

Primary: on a held-out evaluation set, at matched training-compute budget,
- SFT on $\mathcal{D}_W$ outperforms SFT on a same-size random subset by ≥ 5% absolute accuracy on at least one of GSM8K / MATH / HumanEval+.
- SFT on $\mathcal{D}_W$ matches or beats SFT on the *full* $\mathcal{D}_{\text{cand}}$ while using ≤ 30% of the data.

Secondary:
- The curated subset is < 30% of the full data, demonstrating real selection signal.
- Curve sweep cost is amortized: sweep-once-train-many is cheaper than full-data SFT on iter-2.
- The $X{\to}W$ exchange rate is empirically measurable: $\Delta\text{acc}_W / \Delta\text{FLOPs}$.

Kill criteria:
- If at any $X$-budget the "$X$-saturated AND wrong" set is < 5% of data → topic is X-bounded; further W-side work is premature.
- If random subset of same size beats $\mathcal{D}_W$ → the curve signal is noise.

## Files in this plan
- [`README.md`](README.md) — this file
- [`validation.md`](validation.md) — full experimental protocol, baselines, ablations
- [`channels.md`](channels.md) — specific benchmarks, datasets, scorers
- [`budget.md`](budget.md) — GPU-hours, $, wall-clock, headcount
- [`references.md`](references.md) — closest related work
