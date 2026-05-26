# Plan 03 — W-Space Best-of-N (Test-Time Search Along the Weights Axis)

> **Status**: drafting
> **Created**: 2026-05-26
> **Owner**: Mingjia
> **Parent idea**: brainstorm D1 (also entangles D2, B1, H7)
> **One-liner**: All current "test-time compute" is search along the **$X$-axis** (sample more tokens, more CoT, more retrieval). This plan introduces search along the **$W$-axis**: sample $N$ candidate weight perturbations $\{\Delta W_i\}$, run inference under each, let a verifier pick the best. Same Best-of-N pattern, orthogonal diversity source.

---

## Problem

The dominant scaling axis for test-time inference is *$X$-diversity*: temperature sampling, beam search, tree search, self-consistency. These all *vary the input/output sequence* under a fixed $W$. They share one limitation: every sample is drawn from the same probability distribution $p(\cdot \mid X; W)$. The diversity is bounded by the model's existing posterior.

What if the underlying model itself were different for each sample? A LoRA $\Delta W_i$ shifts the entire posterior — *much* further than temperature can. This opens a new axis of test-time search.

**Open questions this plan answers**:
- Does $W$-diversity provide *different coverage* than $X$-diversity at matched FLOPs?
- Where on the compute-vs-quality Pareto front does $W$-BoN sit relative to $X$-BoN?
- Can we *learn* the $W$-perturbation distribution rather than sampling it blindly?

## Core idea

For each query $q$:

1. Generate $N$ candidate weight deltas $\{\Delta W_1, \dots, \Delta W_N\}$.
2. Run inference $y_i = f(q; W_0 + \Delta W_i)$ for each.
3. Score with a verifier: $i^\star = \arg\max_i V(q, y_i)$.
4. Return $y_{i^\star}$.

The interesting variable is **where the $\Delta W_i$ come from**. We test three families, in increasing sophistication:

- **WBoN-Noise**: $\Delta W_i \sim \mathcal{N}(0, \sigma^2 I)$ in LoRA space. Calibration of $\sigma$ matters.
- **WBoN-Library**: $\Delta W_i$ drawn from a pre-built library of LoRA experts (e.g., trained on different sub-distributions, or [Lots-of-LoRAs](https://huggingface.co/datasets/Lots-of-LoRAs/Lots-of-LoRAs)).
- **WBoN-Hypernet**: $\Delta W_i = H_\phi(q, z_i)$ with $z_i$ a noise seed, $H_\phi$ a (small) hypernet trained for diversity + quality.

## Success criteria

Primary: on a held-out reasoning benchmark with a cheap verifier,
- WBoN-Hypernet at $N = 8$ outperforms $X$-BoN at $N = 8$ by ≥ 3% absolute on at least one of MATH / HumanEval+.
- WBoN-Hypernet at $N = 8$ matches $X$-BoN at $N = 32$, at substantially fewer total tokens generated.

Secondary:
- **Disjoint successes**: there exist problems where WBoN solves but $X$-BoN does not, and vice-versa. → diversity is genuinely orthogonal.
- **Compute-Pareto**: WBoN dominates $X$-BoN at matched FLOPs for some operating points.

Kill criteria:
- If WBoN-Noise is no better than $X$-BoN at any $\sigma$ → the entire premise is suspect; reframe before proceeding.
- If WBoN-Hypernet collapses to "always same $\Delta W$" → diversity is the bottleneck, need a different architecture.

## Why now

- Doc-to-LoRA proved hypernets *can* produce usable LoRAs in one forward pass — making per-query LoRA cheap enough to be in the inference budget.
- S-LoRA proved batched LoRA serving works — making WBoN deployable.
- Verifiers for math / code are now reliable enough to do honest Best-of-N work.

## Files in this plan
- [`README.md`](README.md) — this file
- [`validation.md`](validation.md) — experimental protocol
- [`channels.md`](channels.md) — benchmarks, datasets, verifiers
- [`budget.md`](budget.md) — compute and money
- [`references.md`](references.md) — closest prior work
