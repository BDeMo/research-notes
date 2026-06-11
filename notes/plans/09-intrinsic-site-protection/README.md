# Plan 09 — **Janus**: intrinsic-site protection (long-context ↔ forgetting coupling)

> ⛔ **PIVOTED (2026-06-04 eve) — read first.** The unifying **head-level** coupling thesis below
> (H1/H2/H3, "same heads, two frontiers") was **falsified under controls**
> ([`mechanism-results-2026-06-04.md`](mechanism-results-2026-06-04.md)). The work split into **two
> papers**: **Paper A** = *long-context heads are gradient-**shielded** during fine-tuning* (the
> surviving, opposite fact) → [`paperA-plan-2026-06-05.md`](paperA-plan-2026-06-05.md); **Paper B** =
> the v1.5 wrapper/gating line *"know when to fall back"* →
> [`gate-pipeline-2026-06-05.md`](gate-pipeline-2026-06-05.md) (filed here for now; belongs to the
> Plan 08 wrapper line, to be relocated). Everything below is the original pre-pivot plan, kept for
> history — its success/kill criteria refer to the falsified H2.

> **Codename**: **Janus** (working; one site, two faces — *read*-time long-context overload vs *write*-time forgetting perturbation. "dual-frontier" is the descriptive alias. Name is not load-bearing; change freely.)
> **Status**: drafting (novelty audit done 2026-06-04, brainstorm §5.5–5.6). **Phase-0 detectors validated on Qwen3-8B 2026-06-03** → [`phase0-results-2026-06-03.md`](phase0-results-2026-06-03.md). **Phase-1 exploration 8 models / 7 H100s 2026-06-04** → [`phase1-results-2026-06-04.md`](phase1-results-2026-06-04.md). Key empirical findings: (1) sink heads ≠ retrieval heads (Jaccard 0) across 0.6→14B → protect the **retrieval heads**, not the sink; (2) ρ(retrieval, SFT-drift) > ρ(sink, SFT-drift) in nearly every model (direction robust, clears 0.4 gate only on smallest 2 Qwen3 → **H2 partial**); (3) grad-mask protection causally removes the coupling; (4) **to fix**: harden NIAH probe + induce real forgetting (math-SFT was too gentle to test H3).
> **Created**: 2026-06-03
> **Owner**: Mingjia
> **Parent**: P0 thesis in [`../../ideas/rca-transformer-intrinsic-2026-06-03.md`](../../ideas/rca-transformer-intrinsic-2026-06-03.md) (§1 + §6). Design rules: §0. Novelty audit: §5.5–5.6.
> **One-liner**: First **measure** whether the transformer-intrinsic sites that carry long-context behavior are the *same* sites whose perturbation during fine-tuning causes catastrophic forgetting (a broad, multi-level observation study on RCA + code + math). Then, **only if the coupling is real**, design a data-agnostic anti-forgetting method that protects exactly those sites.

## Novelty position (locked after the 2026-06-04 audit — read before building)

Every *single* leg is already published (long-ctx via retrieval/sink heads `[retrieval-head]` `[duo-attn]`; forgetting-localizes-to-heads `[mech-forget]`; MoE-forgetting via task-affinity expert freezing DAS/ESFT/DES-MoE; feature-space anti-forgetting `[sae-fd]`; the two single-leg sink fixes `[focusft]` long-ctx + `[sink-forget]` forgetting). **The contribution is the conjunction, stated precisely:**

1. **The *predictive* coupling (science).** The read-side, **data-agnostic** intrinsic-importance ranking (retrieval-head / sink-mass / super-expert score, detected on generic text) **predicts a priori** the write-side forgetting-disruption ranking. This lets us protect the load-bearing sites *before* SFT with a **task-agnostic** criterion. The thing to beat is `[mech-forget]`, whose damaged-set is **post-hoc by ΔW**; ours is **a-priori by long-context importance**, and we show the two coincide (H2).
2. **The headline instantiation = MoE super-expert protection** (cleanest, least-crowded; §5.6a). Nobody selects the protected set by the **intrinsic super-expert / sink-induction** criterion for FT-without-forgetting — all MoE work uses task/domain affinity.
3. **One lever, both axes, dual eval** — must beat the *stack* of the two single-leg fixes (`[focusft]` + `[sink-forget]`), else there is no paper (the bar Phase-1/P0c must clear).

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
- [`paper-draft-2026-06-04.md`](paper-draft-2026-06-04.md) — **paper-level writeup** (abstract → method → results → repro): the consolidated story
- [`metrics-reference.md`](metrics-reference.md) — **metric codebook**: every metric's derivation / rationale / interpretation / range + the full correlation ranking
- [`ideas-brainstorm-2026-06-04.md`](ideas-brainstorm-2026-06-04.md) — **idea list** (62 methods): one per coupling cell + criterion / inference-time / architectural / optimizer / theory / blue-sky, each with motivation + expected insight
- [`metrics-curation-2026-06-04.md`](metrics-curation-2026-06-04.md) — **metric triage**: soundness × interestingness rating table (39 metrics) → KEEP/SUPPORT/FILTER, + 12 metric-driven ideas (I1–I12)
- [`matrix.md`](matrix.md) — **living done/todo ledger** (one-screen status + results + queue); update after every work block
- [`README.md`](README.md) — this file (problem, hypotheses, phases, success criteria)
- [`validation.md`](validation.md) — **Phase 1 observation matrix** (levels × granularities × metrics × probes) + Phase 2 method design + ablations + stats
- [`channels.md`](channels.md) — **Phase 3 eval**: benchmarks, metrics, baselines, cross-domain/task/model settings, base models, detailed protocol
- [`budget.md`](budget.md) — GPU-hours, $, wall-clock, decision gates
- [`references.md`](references.md) — closest prior (links to `knowledge-sources.md` IDs)
- [`phase0-results-2026-06-03.md`](phase0-results-2026-06-03.md) — Phase-0 detector validation on Qwen3-8B (actual run)
- [`phase1-results-2026-06-04.md`](phase1-results-2026-06-04.md) — **Phase-1 exploration, 8 models / 7 H100s (actual run)** — H2 partial, sink≠retrieval robust, NIAH+forgetting setup needs hardening
- [`facts-2026-06-04.md`](facts-2026-06-04.md) — **broad fact-finding on the 7–9B cohort** (Qwen3.5-9B, GLM-4-9B, Qwen3-8B, Qwen2.5-7B-Inst; ~30 metrics × 5 angles, native bf16): cross-family collapse-axis facts + the drift↔retrieval coupling
- [`grid-metrics-2026-06-04.md`](grid-metrics-2026-06-04.md) — v1 pooled metric grid (41 cells). **⛔ Its "uniformly positive coupling" headline is superseded** (falsified under controls — see `mechanism-results-2026-06-04.md`).
- [`mechanism-results-2026-06-04.md`](mechanism-results-2026-06-04.md) — **the controlled re-analysis that falsified head-level coupling** and established gradient-shielding (authoritative).
- [`paperA-plan-2026-06-05.md`](paperA-plan-2026-06-05.md) — **Paper A plan** (gradient-shielding): main tables, ablations, pre-exps.
- [`paper-preexp-plan-2026-06-04.md`](paper-preexp-plan-2026-06-04.md) — Paper A pre-experiment plan (A1 carrier, A2 a-priori→drift).
- [`cohort-grid-results-2026-06-04.md`](cohort-grid-results-2026-06-04.md) — cohort grid read-out (flags the sign-inconsistent pairs).
- [`gate-pipeline-2026-06-05.md`](gate-pipeline-2026-06-05.md) — **Paper B (v1.5)** pipeline spec + "good gating signal" definition (D1–D6).
- [`gate-scope-litreview-2026-06-05.md`](gate-scope-litreview-2026-06-05.md) — **Paper B** scope (parametric agentic memory) + literature review + novelty defense.
- [`gate-critical-review-2026-06-05.md`](gate-critical-review-2026-06-05.md) — **Paper B** ICLR-2027 self-review (3 rounds).
- [`matrix-archive-2026-06-04.md`](matrix-archive-2026-06-04.md) — archived v1 measurement ledger.
- `figs/` — headline + facts figures; `runs/` — detector/runner/viz/analysis code + GPU strategy

## Status vs success criteria (updated 2026-06-05, post-pivot)
- **H1 (coexistence)** — ✅ detectors find the small site sets at every scale (Phase-1 R1).
- **H2 (head-level coupling)** — ⛔ **FALSIFIED under controls** (partial|out_norm + within-layer + CIs): the pooled "+" was a depth/activation/hybrid-Qwen3.5 artifact; within-layer the sign reverses. See [`mechanism-results-2026-06-04.md`](mechanism-results-2026-06-04.md).
- **Surviving fact → Paper A** — long-context heads are gradient-**shielded** (under-updated) during task-SFT; activation-mediated, task-objective-specific, scale-fading. Plan: [`paperA-plan-2026-06-05.md`](paperA-plan-2026-06-05.md).
- **H3 (causal protection)** — superseded by the Paper A framing; the original protection-sweep is archived (no head-level coupling to protect against). Forgetting "dial" found (lr 1e-3 = selective MMLU drop, NIAH intact).
- **Paper B (v1.5 gating)** — separate line, running: see §0.GATE in [`matrix.md`](matrix.md).
