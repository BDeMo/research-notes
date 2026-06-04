# Plan 09 — Intrinsic-site protection: long-context ↔ forgetting coupling → anti-forgetting method

> **Status**: drafting
> **Created**: 2026-06-03
> **Owner**: Mingjia
> **Parent**: P0 thesis in [`../../ideas/rca-transformer-intrinsic-2026-06-03.md`](../../ideas/rca-transformer-intrinsic-2026-06-03.md) (§1 + §6). Design rules: §0 of that file.
> **One-liner**: First **measure** whether the transformer-intrinsic sites that carry long-context behavior are the *same* sites whose perturbation during fine-tuning causes catastrophic forgetting (a broad, multi-level observation study on RCA + code + math). Then, **only if the coupling is real**, design a data-agnostic anti-forgetting method that protects exactly those sites.

---

## Why this plan is structured measure-first

Per the audit (brainstorm §5), every *single mechanism* for either problem is already published. The only defensible contribution is the **unifying observation** (DR8) + a **site-selection criterion** (DR9). An observation can't be assumed — it must be measured first. So Phase 1 is a **de-risk measurement study that needs no task-specific training and no RCA labels in the method**; Phase 2 designs the method *conditioned on what Phase 1 finds*; Phase 3 is the paper-grade general eval.

This ordering also protects us: if the coupling is weak, we learn it in Phase 1 (cheap) instead of after building a method.

## Problem

Two failure modes are usually studied separately:
- **Long-context inference** (read-time): as context grows, attention entropy explodes, "lost-in-the-middle", KV blows up, retrieval degrades.
- **Catastrophic forgetting** (write-time): SFT on a new domain drifts pretrained code/math/general ability.

**Hypothesis (DR8, the thing we test)**: both are governed by the *same* small set of **intrinsic load-bearing sites** — attention sinks / massive-activation channels (dense), super experts that induce those sinks (MoE), retrieval/induction heads. Long context *overloads* these sites; fine-tuning *perturbs* them. If true, one data-agnostic rule — *detect these sites on generic text, then protect them during any SFT* — improves long-context retention **and** prevents forgetting, with **zero task-specific trainable parameters** (DR3, DR6).

## Core hypotheses (testable)

- **H1 (coexistence)**: a forward pass on generic text identifies a small set of sites that are disproportionately important for long-context behavior (high sink mass / massive activation / retrieval-head score / super-expert score).
- **H2 (coupling — the key claim)**: those same sites are where fine-tuning concentrates its largest, most capability-damaging perturbations. Formally, the site-ranking by *long-context importance* and the site-ranking by *forgetting disruption* are **positively correlated** (Spearman ρ well above chance), across RCA / code / math SFT.
- **H3 (causal)**: **protecting** the long-context-important sites during SFT (freeze / grad-mask / orthogonalize) **reduces forgetting** of code/math/general *and* preserves long-context ability, at equal or better new-domain learning, beating capability-agnostic baselines.

## Success criteria

Primary (decides the paper):
- **H2**: Spearman ρ between the long-ctx-importance ranking and the forgetting-disruption ranking ≥ 0.4 (per-site), holding on ≥ 2 of 3 SFT domains and ≥ 2 models, 4-seed (DR14).
- **H3**: at matched new-domain accuracy, intrinsic-site protection cuts the average forgetting (BWT / retention drop) by ≥ 30% relative to full SFT, and beats the best capability-agnostic baseline (MoFO/OPLoRA/EWC) by ≥ 5% retention — on both a dense and a MoE model.
- Long-context is **not** degraded by the protection (Δ effective-context-length ≥ 0 vs full SFT).

Secondary:
- The protected set is **small** (≤ 5–10% of params / heads / experts) — demonstrates real localization.
- The criterion is **data-agnostic**: sites detected on generic text transfer to protecting against *any* SFT domain (cross-domain transfer of the site set).
- The criterion is **model-agnostic**: the same recipe works across families (Qwen3, Llama-3.1) and the dense↔MoE split.

Kill criteria:
- **Phase 1 gate**: if H2's ρ is near zero across models/domains (long-ctx sites and forgetting sites are *unrelated*), the unifying thesis is false → stop, write the negative observation as a short note (still publishable as "these two problems are decoupled, contrary to intuition").
- **Phase 2 gate**: if protecting the coupled sites does *not* reduce forgetting (H3 fails) even though H2 held → the correlation is non-causal; pivot to a pure-measurement paper.

## Phases

| Phase | What | Needs RCA data? | Gate |
|---|---|---|---|
| **0** | Tooling: site detectors + drift meters + hooks | no | detectors reproduce known sinks/massive-acts |
| **1** | **Observation study** (multi-level/granularity/metric) → coupling map | no (generic + code/math/RCA inputs, no labels in method) | H1 + H2 |
| **2** | **Method design** (protection rule, contingent on Phase 1) + causal test | light, task-agnostic | H3 |
| **3** | **Paper eval**: benchmarks, metrics, baselines, cross-domain/task/model | RCA only as showcase | headline numbers |

## Files in this plan
- [`README.md`](README.md) — this file (problem, hypotheses, phases, success criteria)
- [`validation.md`](validation.md) — **Phase 1 observation matrix** (levels × granularities × metrics × probes) + Phase 2 method design + ablations + stats
- [`channels.md`](channels.md) — **Phase 3 eval**: benchmarks, metrics, baselines, cross-domain/task/model settings, base models, detailed protocol
- [`budget.md`](budget.md) — GPU-hours, $, wall-clock, decision gates
- [`references.md`](references.md) — closest prior (links to `knowledge-sources.md` IDs)
