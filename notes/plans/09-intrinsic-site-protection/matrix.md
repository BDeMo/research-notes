# Matrix — Janus (Plan 09): long-context ↔ forgetting coupling

> **Living ledger.** Top = one-screen status. Then the **DONE** ledger (what we've
> shipped, with results) and the **TODO** queue (what's next, prioritized).
> Keep both sides in sync after every work block. Last updated **2026-06-04 ~19:35 UTC**.
>
> **Thesis (Janus):** the transformer-intrinsic sites that carry **long-context
> behaviour at inference (LC)** are the *same* sites that **fine-tuning perturbs
> most at training (CF)**. If true → one data-agnostic rule (detect on generic
> text, protect during SFT) improves long-ctx retention *and* prevents forgetting.
> Two frontiers, one site → "Janus."

---

## 0 · Status (one screen)

| # | work item | state | evidence |
|---|---|---|---|
| N | Novelty audit (Path B closed; super-expert gate passes; wedge = *predictive* coupling) | ✅ | brainstorm §5.5–5.6; refs + knowledge-sources |
| 0 | Phase-0 detectors (Qwen3-8B): sink / massive-act / retrieval reproduce | ✅ | `phase0-results-2026-06-03.md` |
| 1 | Phase-1 exploration (8 models / 7 H100s): **sink≠retrieval** robust 0.6→14B; H2 partial | ◐ | `phase1-results-2026-06-04.md` |
| F | Facts study (7–9B cohort, ~30 metrics×5 angles): collapse-axis facts; drift↔retrieval +0.52 | ✅ | `facts-2026-06-04.md` |
| Q | Qwen3.5 env solved (torch 2.11 iso-venv) + hybrid 3:1 attention handling | ✅ | `runs/GPU_STRATEGY.md` |
| G | **Unified grid** (12 angles/39 metrics × 12 benchmarks × 4 families = 41 cells) | ✅ | `grid-metrics-2026-06-04.md` |
| G2 | **Headline: LC×CF coupling uniformly positive (+0.17…0.56)**, cross-family/dataset | ✅ | `figs/G1_LC_CF_coupling.png` |
| H3 | **Causal test**: protect LC-coupled heads during SFT → CF↓ without LC↓ | ⏳ next | — |
| B | Behavioural LC (NIAH acc@length) + retention Δ per dataset (close angle 12) | ⏳ | — |
| S | Scale axis 0.6→32B on the confirmed couplings | ⏳ | ladder data partly collected |
| M | Method design (Phase-2 variant per granularity) + paper framing | ⏳ | `validation.md` §Phase-2 |

**Headline result (G2).** Per-head, pooled over 41 model×dataset cells, **every
long-context head-metric is positively correlated with every forgetting metric**
(0.17–0.56, no negative cell). Sign-consistent across all 41 cells:
`prev_token~grad_mag +0.48`, `attn_distance~act_drift +0.42`, `prev_token~fisher
+0.42`, `retrieval~act_drift +0.37`. Strongest write-side signals are **weight-
drift & activation-drift** (what SFT *actually does*), not just a-priori Fisher.

**Compute now.** ray (`sam-dev-ray`) 4×H100 **idle/free**. sam-dev 4×H100 **busy
with v1 `mem-test` probe_v2 (4 seeds, not Janus)** — leave alone. → run next Janus
step on **ray only**.

---

## 1 · DONE ledger (chronological, with results)

### N — Novelty audit (2026-06-04)
- Path B (sink-key-bias-only tuning) **closed**: ZeroTuning/PASTA/`[sink-emerge]` preempt it; both single legs preempted (`[focusft]` long-ctx, `[sink-forget]` forgetting).
- **Super-expert protection gate PASSES**: all MoE anti-forgetting selects by task/domain affinity (DAS/ESFT/DES-MoE) or long-tail (ExpertCondenser); nobody uses the intrinsic super-expert criterion → P0b headline.
- Surviving wedge = the **predictive** coupling (read-side ranking predicts write-side ranking *a priori*). To beat: `[mech-forget]` (post-hoc ΔW).
- Fixed 9 broken citation refs in `knowledge-sources.md`.

### 0 — Phase-0 detectors (Qwen3-8B, 2026-06-03)
- All three phenomena reproduce; **5 massive-act channels** (ch2276 ≈410× median); retrieval-head hub at L23.
- **sink heads ≠ retrieval heads** (Jaccard 0) → the long-ctx site to protect is **retrieval heads**, not sinks.

### 1 — Phase-1 exploration (8 models / 7 H100s, 2026-06-04)
- sink≠retrieval **robust across 0.6→14B** (Jaccard ≈0).
- H2 **partial**: ρ(retrieval,drift) > ρ(sink,drift) in ~all models, but clears the 0.4 gate only on the smallest 2 Qwen3.
- Grad-mask head protection **works mechanically**. Gaps: NIAH probe too hard on base models; math-SFT too gentle to induce forgetting.

### F — Facts study (7–9B cohort, 2026-06-04)
- Cross-family **representation-collapse axis** (vendor-independent, incl. GLM-4): late layers → resid-norm↑, prediction crystallizes, eff-rank↓, massive activations emerge, MLP down-proj spectrum co-moves.
- `drift ↔ retrieval = +0.52` per-head across all 4 families.

### Q — Qwen3.5 + infra (2026-06-04)
- Isolated venv `/root/q35iso` (torch 2.11+cu128 / tf 5.10) loads Qwen3.5-9B; hybrid 3:1 Gated-DeltaNet handled (`_iter_attn_maps` remap; linear-attn layers guarded).
- All models **native bf16, no quantization**; metrics computed in fp32.
- Freed sam-dev GPU0 (killed idle rca-demo `app.py`).

### G — Unified grid (2026-06-04)
- Decoupled into **5 pass types** (A fwd+attn ⊕ C backward share batches; B weights once; D SFT; E behavioural) → minimal compute.
- **39 metrics / 12 angles**, frontier-tagged LC/CF/ST.
- **41 (model×dataset) cells**, 4 families × 12 benchmarks, all 8 H100s, 0 failures.
- → **G2 headline** (above).

---

## 2 · TODO queue (prioritized)

1. **H3 causal test (the method).** On ray: take top-k LC-coupled heads (by `attn_distance`/`retrieval`/`prev_token`), protect during SFT (freeze / grad-mask / orthogonal) vs {none, random, ΔW-set `[mech-forget]`}. Measure Δforgetting (retention) **and** Δlong-ctx (NIAH). *Gate: ≥30% less forgetting, no LC regression, beat baselines.*
2. **Close angle 12 (behavioural).** NIAH acc@length×depth + capability retention Δ (before/after SFT) per dataset → ties the structural couplings to actual LC/CF *outcomes*.
3. **Scale axis (0.6→32B).** Re-run the grid (or just the LC×CF couplings) on the ladder; check the coupling is scale-invariant (esp. does it strengthen/weaken with size?).
4. **MoE instantiation (P0b headline).** Port detect+protect to a super-expert MoE (Qwen3-30B-A3B) — the least-crowded novelty.
5. **Eval channels.** Wire LongMemEval / TRACE (continual) for the paper-grade forgetting axis.
6. **Method design memo** (Phase-2 variant chosen by which granularity carries the coupling) → `validation.md`.

## 3 · Decisions / open questions
- Headline framing: lead with the **predictive coupling** (G2) — strongest, cleanest, cross-family. RCA stays the application showcase, not the method.
- Which write-side target is canonical: **act_drift / dW_drift** (couple hardest) vs Fisher (a-priori, the "predict before training" story). Likely report both; Fisher for the a-priori claim.
- Kill criterion still live: if H3 fails despite G2 holding → pivot to a pure-measurement paper (the coupling is itself a contribution).

## 4 · Pointers
- Plan: `README.md` · `validation.md` · `channels.md` · `budget.md` · `references.md`
- Results: `phase0-results-*.md` · `phase1-results-*.md` · `facts-*.md` · `grid-metrics-*.md`
- Code: `runs/janus_run.py` (detect/niah/sft/capeval/intrinsic/facts/**grid**) · `runs/analyze_grid.py` · `runs/janus_grid_viz.py` · `runs/orch_grid.sh` · `runs/GPU_STRATEGY.md`
- Figures: `figs/G1_LC_CF_coupling.png` (headline) · `figs/14_facts_consistent.png` · `figs/09_scaling_ladder.png`
