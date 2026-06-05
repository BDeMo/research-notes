# 2026-06-04 — Empirical meta-methodology + cross-model intrinsic-signal reproduction (plan 08 v1.5)

## Activities
- Distilled the recurring **execution principles** (honesty/fairness, measure-first,
  native metrics, multi-granularity×angle, filter-to-non-obvious, rank/visualize,
  metric codebook, multi-seed×cross-model robustness, baselines/controls, compute discipline,
  living ledger) into [`../meta-methodology.md`](../meta-methodology.md) — 12 principles,
  a worked-instances table (08 v1.5 vs 09), and an execution checklist.
- Reproduced the plan-08 **v1.5 intrinsic gating signals on two new base families** to test
  which correlations are *consistent*:
  - **GLM-4-9B-0414** — native `Glm4ForCausalLM`, probe ported with **zero code changes**
    (`model.norm` + `inputs_embeds` + `output_hidden_states` all present). 3-seed sweep
    (5 core benches) on sam-dev cuda:0/1/2.
  - **Qwen3.5-9B** — hybrid Gated-DeltaNet + sparse-MoE multimodal; only in transformers-`main`
    (needs torch ≥2.7). Built an **isolated venv** (transformers-main + torch 2.8 + einops)
    so the working env is untouched; loads via `AutoModelForCausalLM`→`Qwen3_5ForCausalLM`,
    forwards through a **torch fallback** (fla kernels incompatible with torch 2.8 → removed).
    1-seed sweep on cuda:3 (slow ~3 s/step but correct).
- Built [`scripts/xmodel_consistency.py`](../../../mem-test/mem-embedding/scripts/xmodel_consistency.py):
  per-model pooled AUROC(help)/AUROC(no-harm)/Spearman(lift), restricted to
  **architecture-comparable** signals (scalars + fractional-depth layer-curve features;
  raw `@layer-idx` dropped because depths differ 36/40/32).
- Pulled Qwen3-8B **core baseline** (8 runs, 6000 items) from ray as the comparison anchor;
  validated the pipeline reproduces the confidence-dominant ranking (`top5_mass_w` AUROC 0.702).

## Decisions
- **The v1.5 intrinsic-gating doc is plan 08, step v1.5 — NOT 08-v2, and NOT plan 09.**
  - It gates the *mem-X soft-prompt wrapper* (a v1 refinement), whereas **08-v2** is a different
    extension (cross-session suffix read/write latent memory).
  - **Plan 09 (Janus)** rhymes but is a distinct object: write-time ($W$) protection of
    pretrained-capability *sites* against SFT-forgetting, vs 08-v1.5's read-time ($X$) per-input
    *gating* of an additive wrapper. **Siblings under one methodology, not the same thing.**
  - 09 and 08-v2 are **not** the same either (anti-forgetting site-protection vs memory-architecture).
- **Overlap to manage, not merge**: 08-v1.5's cross-model cohort (Qwen3-8B / GLM-4-9B / Qwen3.5-9B)
  overlaps the cohort in plan 09's [`facts-2026-06-04`](../../notes/plans/09-intrinsic-site-protection/facts-2026-06-04.md)
  / `grid-metrics-2026-06-04`. Different signal families (wrapper-relative vs head/expert/sink),
  same models + same metric-grid methodology → **cross-reference**, don't duplicate.
- Cross-model fast-path kernels (fla) abandoned for Qwen3.5: incompatible with torch 2.8; the
  torch fallback is correct (validated, just slower) — correctness over speed.

## Output artifacts
- [`docs/meta-methodology.md`](../meta-methodology.md) — the distilled methodology (this session's headline output).
- `mem-test/mem-embedding/scripts/xmodel_consistency.py` — cross-model consistency analyzer.
- `mem-test/mem-embedding/xmodel/qwen3_8b/` — pulled Qwen3-8B baseline (6000 items).
- (pending runs) GLM-4-9B 3-seed + Qwen3.5-9B 4-seed signal jsonls → consistency CSV + chart.
- [`notes/plans/08-model-outputs-delta-w/summary/2026-06-04/v1.5-metric-candidates-2026-06-04.md`](../../notes/plans/08-model-outputs-delta-w/summary/2026-06-04/v1.5-metric-candidates-2026-06-04.md)
  — plan-09-style gate-signal curation (generality × novelty × interest × soundness) + 14 new
  candidates (memory-geometry / counterfactual / 09-bridge structure & attention-routing / stability)
  + code-&-validate shortlist + the facts/grid/analysis location index.

## Knowledge sources used
- Benchmark-native metric papers (QuALITY / MuSR / SQuAD-v2 / RULER / TriviaQA / NarrativeQA / MS-MARCO).
- HF model cards: `Qwen/Qwen3.5-9B` (hybrid MoE, transformers-main), `THUDM/GLM-4-9B-0414` (native `Glm4`).

## Next steps
- Finish GLM (3 seeds, ~1 bench left) + Qwen3.5 (1 seed; add 2 more when GLM frees GPUs); run
  `xmodel_consistency.py` → "which signals are consistent across families" CSV + ranked chart.
- Fold the consistent-signal set into `v1.5-intrinsic-gating-study-2026-06-04.tex` (new cross-model
  section) and the `mem-embedding/summary/matrix.md`.
- Cross-link this cohort with plan 09 `facts-2026-06-04` so the two intrinsic-metric studies reciprocate.
