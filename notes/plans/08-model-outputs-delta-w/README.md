# Plan 08 — Model Outputs ΔW as Part of Generation (Self-Modifying LLMs)

> **Status**: **v1 paper submitting (characterization)** · v1.5 intrinsic gating in progress · north star (self-modifying ΔW) on hold pending v2
> **Created**: 2026-05-26 · **Last updated**: 2026-06-04
> **Owner**: Mingjia
> **Parent idea**: brainstorm H6 (also entangles H3, H7, I3)
> **One-liner**: At each turn, the model produces not just an answer $y$ but also a **weight delta $\Delta W$** representing what it just learned. A verifier decides whether to apply, reject, or scale $\Delta W$. Over time, the model literally rewrites itself.

## v1 (mem-X / soft-prompt) — current paper-submitting branch

**v1 ≠ north star.** v1 implements the simplest carrier of the "learned wrapper around a frozen base" idea: memory as input embeddings (mem-X axis), no $\Delta W$ yet. Sister axes mem-H (hidden state) and mem-W (LoRA-delta) are parked.

**The 3-regime law (Phase Y, 4 seeds, 2026-06-03)** is the v1 paper's contribution:
- Regime A — QuALITY: wrapper matches Gist within seed noise, **+5 to +12 pp over no-training baselines**, 1/3 fewer soft tokens.
- Regime B — MuSR-mm: **both wrappers at chance**; full_context wins.
- Regime C — RULER-NIAH: **OURS = GIST = 0.000 ± 0.000** across 4 seeds; full_context = 0.995.

Full fact-dump at [`v1-results-2026-06-03.md`](v1-results-2026-06-03.md). Source repo: `~/workspace/mem-test/mem-embedding/`. Paper repo: `~/workspace/latent-mem-paper/`.

**Pivot menu**: [`v1-if-wrapper-doesnt-work-2026-06-03.md`](v1-if-wrapper-doesnt-work-2026-06-03.md) — 10 directions, top-3 testable-today: hybrid wrapper+retrieval (B), infilling objective (D), unfreeze last layer (I).

**v1.5 gating study**: [`v1.5-intrinsic-gating-study-2026-06-04.tex`](v1.5-intrinsic-gating-study-2026-06-04.tex) / [`pdf`](v1.5-intrinsic-gating-study-2026-06-04.pdf) studies intrinsic "do-no-harm" signals for suppressing the wrapper when it would hurt a frozen-base model.

## Deliverables hub

Use this section as the main human entry point for Plan 08 deliverables.

### v1 / wrapper facts and main result

- **Main v1 result summary**: [`v1-results-2026-06-03.md`](v1-results-2026-06-03.md)
  — the full `mem-embedding` result harvest and three-regime transfer law.
- **Pivot / lessons after v1**:
  [`v1-if-wrapper-doesnt-work-2026-06-03.md`](v1-if-wrapper-doesnt-work-2026-06-03.md)
  — why v1 becomes a characterization paper plus v1.5 options.
- **How v0/v1 got here**: [`v0-how-we-got-here.md`](v0-how-we-got-here.md)
  — architecture and experimental chronology.

### v1.5 / signal grid and gating deliverables

- **Compiled technical note**:
  [`v1.5-intrinsic-gating-study-2026-06-04.pdf`](v1.5-intrinsic-gating-study-2026-06-04.pdf)
  — PDF report with settings, metrics, results, and next gate target.
- **LaTeX source**:
  [`v1.5-intrinsic-gating-study-2026-06-04.tex`](v1.5-intrinsic-gating-study-2026-06-04.tex)
  — source for the compiled report.
- **Signal grid folder**: [`grids-2026-06-04/`](grids-2026-06-04/)
  — correlation grids and ranking figures for v1.5 gating.
- **Help ranking figure**:
  [`grids-2026-06-04/rank_help.png`](grids-2026-06-04/rank_help.png)
  — top signals for wrapper help.
- **Non-obvious ranking figure**:
  [`grids-2026-06-04/rank_interesting.png`](grids-2026-06-04/rank_interesting.png)
  — confidence-filtered signals, highlighting divergence/drift.
- **Grid CSVs**:
  [`grid_auroc_correct.csv`](grids-2026-06-04/grid_auroc_correct.csv),
  [`grid_auroc_noharm.csv`](grids-2026-06-04/grid_auroc_noharm.csv),
  [`grid_spearman_cont.csv`](grids-2026-06-04/grid_spearman_cont.csv),
  [`corr_long.csv`](grids-2026-06-04/corr_long.csv).

### v2 / next version

- **v2 plan**: [`v2-plan.md`](v2-plan.md)
  — cross-session latent memory with read/write tokens.
- **v2 related work**: [`v2-related-work.md`](v2-related-work.md)
  — crowded latent reasoning / memory-token landscape and differentiation.
- **Q2 activation-memory probe**:
  [`q2-activation-memory-probe.md`](q2-activation-memory-probe.md)
  — candidate layer/site selection for later memory interfaces.

### Idea table

- **Ideas index table**: [`../README.md`](../README.md)
  — plan-level index for Plans 01/03/08/09.
- **Full idea table**: [`../../ideas/README.md`](../../ideas/README.md)
  — source idea rows, including H6 (Plan 08) and related follow-ups H3/H7/G4/G5.
- **Original inference-time training brainstorm**:
  [`../../ideas/inference-time-training.md`](../../ideas/inference-time-training.md)
  — source brainstorm behind the X/W framing and Plan 08.

## Weekly human update — week of 2026-06-01

**High-level message.** Plan 08 now has a more positive and actionable story:
the learned memory wrapper is not a universal replacement for long context, but
it is a useful **controlled compression module**. The v1 result tells us where it
helps; v1.5 turns that boundary into a method by learning when to open or close
the wrapper.

**Good news.**
- The v1 experiments produced a clear characterization rather than a dead end:
  soft-prompt memory helps when gist-level compression is enough, and fails when
  the task needs exact detail preservation.
- This naturally motivates a do-no-harm gate: keep the frozen base model safe,
  use the wrapper on compressible inputs, and fall back to base / full context
  when the wrapper is likely to hurt.
- Multi-seed probing found practical gate signals. Wrapper confidence and
  sequence log probability are useful correctness indicators; large
  wrapper-to-base divergence is a warning signal.

**Lesson.** The right framing is controlled adaptation, not "one wrapper solves
all long-context tasks." For RCA, this is exactly the behavior we want: improve
long-context evidence handling without erasing the base model's general ability.

**Next.** Implement the first gated wrapper using confidence + divergence
features, then evaluate whether it preserves Regime-A gains while closing on
Regime-B/C cases where lossy memory is unsafe.

---

## North star (what this plan was originally about)

---

## Problem

Current LLMs are static at inference: the same query, the same conversation history, the same weights. Memory lives in either (a) the prompt (volatile, expensive, lost at session end) or (b) external memory tools (RAG, vector DBs — bolt-on, not native).

Two existing classes of solutions:
- **Hypernet-amortized updates** (Doc-to-LoRA, Text-to-LoRA): cheap, single-pass, but driven by an *external* context-encoder, not by the model's own reasoning.
- **Inference-time gradient updates** (TTT, online RFT, Cartridges): real backprop, expensive, infrequent.

What's missing: **the model itself reasoning about what to learn, and writing it directly into its own weights as part of generation.** This is what most "agentic memory" papers gesture at but never deliver as actual weight updates.

## Core idea

The model's output is augmented:

$$
(y_t, \Delta W_t) \sim p_\theta(\,\cdot\,|\,q_t, \text{history})
$$

where $\Delta W_t$ is the proposed weight update for *itself* after this turn. A verifier gates application:

$$
\theta_{t+1} \;=\; \theta_t + \alpha_t \cdot \Delta W_t,
\quad \alpha_t \in \{0, 1\} \text{ or } [0, 1]
$$

with $\alpha_t$ chosen by:
- a rule-based verifier (did this turn succeed?), or
- a learned critic, or
- the user (explicit thumbs up/down → DPO-style $\alpha$).

In implementation, $\Delta W_t$ is produced by a hypernetwork module $H_\phi$ that takes the model's own hidden states / reasoning trace as input, conditional on a special `<learn>...</learn>` segment in the output.

```
User: <task>
Model: <reasoning> ... </reasoning>
       <answer> ... </answer>
       <learn> I should remember that X implies Y in this domain. </learn>
       <delta_w hash="..." /> ← actually produced by H_φ from <learn> content
```

The `<delta_w>` is what gets applied.

## Why this is harder than D2L

1. The update is conditioned on the model's *own output*, not on a fixed context. There's no oracle context distillation teacher.
2. The verifier signal arrives *after* the application — we need a way to score speculatively or roll back.
3. Stability is the central question: the model must not drift catastrophically over a session.
4. Composition across turns: $\Delta W_1 + \Delta W_2 + \dots$ may interfere.

## Why this is plausible now

- Doc-to-LoRA proved a hypernet can produce a usable LoRA from a context in one pass — so the *mechanism* exists.
- Process Reward Models (PRMs) provide reliable per-step scoring — so the *verifier* exists.
- AlphaEdit / null-space-constrained editing showed how to apply weight updates without breaking other capabilities — so the *stability tool* exists.

What's new: stitching these into a closed loop where the model is the agent.

## Success criteria

**Phase 1 — feasibility (3 months)**: in a controlled continual-learning setting, the trained system applies > 0 useful updates (rate of beneficial updates significantly above random; rate of catastrophic updates near zero).

**Phase 2 — utility (3 months)**: on multi-turn personalization (LaMP, PerLTQA) or sequential task adaptation, the self-modifying model outperforms a frozen base model by ≥ 5% absolute, *without* requiring out-of-band fine-tuning.

**Phase 3 — agent integration (6 months)**: in long-horizon agent tasks (SWE-Bench-Verified, WebArena, OSWorld), the self-modifying agent improves *within* a single task rollout (better step-2 actions than step-1) — measurable by per-step success rate.

Kill criteria:
- If Phase 1 cannot achieve "rate of beneficial > rate of catastrophic" → kill.
- If learned $\Delta W$ is observably indistinguishable from random noise (verifier accepts uniformly) → killing learnt-update; fall back to D2L-style context-conditioned updates.

## Files in this plan
- [`README.md`](README.md) — this file
- [`v0-learned-memory-wrapper.md`](v0-learned-memory-wrapper.md) — practical v0 scope
- [`v0-learned-memory-wrapper_zh.md`](v0-learned-memory-wrapper_zh.md) — Chinese v0 scope
- [`v0-budget.md`](v0-budget.md) — practical v0 budget
- [`v0-budget_zh.md`](v0-budget_zh.md) — Chinese v0 budget
- [`v1-results-2026-06-03.md`](v1-results-2026-06-03.md) — v1 empirical state and paper claim
- [`v1.5-intrinsic-gating-study-2026-06-04.tex`](v1.5-intrinsic-gating-study-2026-06-04.tex) — v1.5 intrinsic gating note
- [`validation.md`](validation.md) — experimental protocol, baselines
- [`channels.md`](channels.md) — benchmarks, datasets
- [`budget.md`](budget.md) — costs by phase
- [`references.md`](references.md) — prior work
