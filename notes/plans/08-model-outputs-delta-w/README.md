# Plan 08 — Model Outputs ΔW as Part of Generation (Self-Modifying LLMs)

> **Status**: drafting (most speculative of the three)
> **Created**: 2026-05-26
> **Owner**: Mingjia
> **Parent idea**: brainstorm H6 (also entangles H3, H7, I3)
> **One-liner**: At each turn, the model produces not just an answer $y$ but also a **weight delta $\Delta W$** representing what it just learned. A verifier decides whether to apply, reject, or scale $\Delta W$. Over time, the model literally rewrites itself.

---

## Reader guide

This directory now has three scopes:

- **Practical v0 / v1** (in flight, paper now): frozen 7B/8B base model + trainable learned-memory wrapper at *encode time*. Characterizes the bit-capacity wall. **This is the version producing the ICLR 2027 paper draft (`latent-mem-paper`).**
- **v2** (designed, paper next): same wrapper protocol, but memory is appended *during* autoregressive generation and persists *across sessions*. Inherits the bit-capacity diagnostic from v1.
- **Full Plan 08** (long-term, parked): model-emitted weight deltas and verifier-gated self-modification.

Plus one **parallel methodology research question**:

- **Q2** — Which activations carry memory? Probe the frozen base for sparse / dormant channels and write the wrapper into those only. Its own paper.

Start here for v0 / v1 (the paper being written now):

- [`v0-learned-memory-wrapper.md`](v0-learned-memory-wrapper.md) — concise English note for the executable v0.
- [`v0-learned-memory-wrapper_zh.md`](v0-learned-memory-wrapper_zh.md) — Chinese version.
- [`v0-paper-target-2026-05-31.md`](v0-paper-target-2026-05-31.md) — paper framing decision (Candidate A: Bit-Capacity Limits).
- [`v0-sweep-plan-2026-05-31.md`](v0-sweep-plan-2026-05-31.md) — sweep design.
- [`v0-sweep-results-2026-05-31.md`](v0-sweep-results-2026-05-31.md) — sweep results scaffold.
- [`v0-results-2026-05-30.md`](v0-results-2026-05-30.md) — earlier interim results.
- [`v0-datasets-plan.md`](v0-datasets-plan.md) — dataset plan.
- [`v0-budget.md`](v0-budget.md) — v0 budget and decision gates.
- [`v0-budget_zh.md`](v0-budget_zh.md) — Chinese budget.
- [`misc/`](misc/) — versioned architecture and training diagrams (PNG + TikZ).

Start here for v2 (next paper):

- [`v2-plan.md`](v2-plan.md) — design + wrapper-to-model integration + open questions.
- [`v2-related-work.md`](v2-related-work.md) — literature survey (2024–25 latent-reasoning landscape; ~20 papers).

Start here for Q2 (parked methodology):

- [`q2-activation-memory-probe.md`](q2-activation-memory-probe.md) — research questions, methodology sketch, compute estimate.

Then read the original files below for the full self-modifying-LLM version (long-term scope).

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

### Index
- [`README.md`](README.md) — this file

### v0 / v1 — paper now (ICLR 2027 draft in `latent-mem-paper`)
- [`v0-learned-memory-wrapper.md`](v0-learned-memory-wrapper.md) — practical v0 scope (EN)
- [`v0-learned-memory-wrapper_zh.md`](v0-learned-memory-wrapper_zh.md) — practical v0 scope (ZH)
- [`v0-paper-target-2026-05-31.md`](v0-paper-target-2026-05-31.md) — paper framing decision
- [`v0-sweep-plan-2026-05-31.md`](v0-sweep-plan-2026-05-31.md) — sweep design
- [`v0-sweep-results-2026-05-31.md`](v0-sweep-results-2026-05-31.md) — sweep results scaffold
- [`v0-results-2026-05-30.md`](v0-results-2026-05-30.md) — interim results
- [`v0-datasets-plan.md`](v0-datasets-plan.md) — dataset plan
- [`v0-budget.md`](v0-budget.md) — v0 budget (EN)
- [`v0-budget_zh.md`](v0-budget_zh.md) — v0 budget (ZH)
- [`misc/`](misc/) — versioned diagrams (PNG + TikZ)

### v2 — paper next (cross-session latent memory)
- [`v2-plan.md`](v2-plan.md) — A+B framing, wrapper-to-model integration design space, open questions, next steps
- [`v2-related-work.md`](v2-related-work.md) — 2024–25 literature survey (~20 papers)

### Q2 — methodology sub-paper (parked)
- [`q2-activation-memory-probe.md`](q2-activation-memory-probe.md) — which activations carry memory; constrained-write wrapper

### Long-term self-modifying-LLM scope (full Plan 08)
- [`validation.md`](validation.md) — experimental protocol, baselines
- [`channels.md`](channels.md) — benchmarks, datasets
- [`budget.md`](budget.md) — costs by phase
- [`references.md`](references.md) — prior work
