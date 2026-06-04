# Plan 08 Settings and Provenance

> Central setting/provenance registry for Plan 08 result cells.
> Any result table, figure, benchmark cell, or headline number should cite one
> of these setting IDs instead of repeating every hyperparameter locally.

## How to cite settings

- For compact tables, add a `setting` column with IDs such as `P08-S1`.
- For slides, write `setting: P08-S1` in the table note or link text.
- For long prose, use a nearby sentence: "Setting: [`P08-S1`](settings.md#p08-s1--v1-canonical-wrapper-recipe)."
- If a result mixes settings, cite all relevant IDs.

## P08-S0 — Plan-level source and environment

Use for high-level deliverables and non-numeric summaries.

| field | value |
|---|---|
| Setting ID | `P08-S0` |
| Plan | Plan 08 — Model Outputs ΔW / learned-memory wrapper line |
| Primary implementation | `~/workspace/mem-test/mem-embedding/` |
| Shared infra | `~/workspace/mem-test/llm-infra/`, `~/workspace/mem-test/encoder-infra/` |
| Paper repo | `~/workspace/latent-mem-paper/` |
| Research notes hub | [`README.md`](README.md) |
| Idea source | [`../../ideas/inference-time-training.md`](../../ideas/inference-time-training.md), idea H6 |
| Plan / idea tables | [`../README.md`](../README.md), [`../../ideas/README.md`](../../ideas/README.md) |
| Compute context | `sam-dev`, `sam-dev-ray`, `sam-dev-test` Kubernetes pods |
| Caveat | Local source paths are provenance pointers; public-facing links should point to committed markdown/PDF/CSV artifacts in this repo. |

## P08-S1 — v1 canonical wrapper recipe

Use for v1 architecture/training recipe descriptions and result cells produced
by the final v1 `mem-X` wrapper recipe.

| field | value |
|---|---|
| Setting ID | `P08-S1` |
| Source repo | `~/workspace/mem-test/mem-embedding/` |
| Canonical code pointer | `k8s/sweep/build_sweep.py::BEST_RECIPE + BEST_ARCH` |
| Base model | Qwen3-8B, frozen, bf16 |
| Wrapper axis | `mem-X` — memory as input embeddings / soft prompt |
| Trainable params | ~218M, 2.67% of base |
| Memory state | `K = 32` soft tokens, hidden size `d = 4096` |
| Write side | 1 Perceiver-IO-style cross-attention + FFN block |
| Read side | `xattn` combine mode; refined memory prepended as soft prefix |
| Training data | `categorical_niah`, 500 train items, 50 held-out |
| Training stage | Final v1 shipped as SFT-only |
| Optimizer | AdamW |
| Steps | 1800 |
| Key losses | CE + KL + consecutive-Δm diversity; answer-head CE for categorical tasks |
| Key hyperparameters | `lambda_div = 0.1`, `lambda_ah ≈ 0.5`, `K = 32`, heads `h = 4`, layers `L = 1` |
| Seeds used for headline v1 bands | `{42, 7, 11, 13}` |
| Baselines | `no_context`, `full_context`, `retrieval`, `summary`, matched Gist Tokens |
| Main facts doc | [`v1-results-2026-06-03.md`](v1-results-2026-06-03.md) |
| Caveats | Single-task synthetic SFT; not a universal memory claim; exact-retrieval tasks expose the bit-capacity / lossy-compression limit. |

## P08-S2 — v1 Phase Y three-regime benchmark cells

Use for the v1 headline result table and any direct citation of QuALITY,
MuSR-mm, or RULER-NIAH headline cells.

| field | value |
|---|---|
| Setting ID | `P08-S2` |
| Inherits | `P08-S1` |
| Benchmark protocol | zero-shot public eval through `scripts/train_smoke.py --eval-dataset`; wrapper never trained on eval data |
| Headline run source | `runs/sweep-phaseY-20260603-195211UTC/` |
| RULER assist source | `runs/sweep-phaseY-rulerassist-20260603-205145UTC/` |
| Benchmarks | QuALITY-val, MuSR-mm, RULER-NIAH |
| Metrics | QuALITY/MuSR: `accuracy_letter`; RULER: `exact_value_match` |
| Seeds | `{42, 7, 11, 13}` |
| Reported statistic | mean ± std over seeds |
| Baseline comparison | matched Gist + no-training baselines; `full_context` used as reference where available |
| Main result doc | [`v1-results-2026-06-03.md#5-the-3-regime-law--headline-result-phase-y-4-seeds`](v1-results-2026-06-03.md#5-the-3-regime-law--headline-result-phase-y-4-seeds) |
| Caveats | `full_context` is an upper/reference condition, not always available under equal context budget; result is a characterization of regimes, not universal superiority. |

## P08-S3 — v1.5 intrinsic signal probe

Use for v1.5 signal/correlation results, including the PDF note and signal grid
figures.

| field | value |
|---|---|
| Setting ID | `P08-S3` |
| Inherits | `P08-S1` |
| Main probe script | `scripts/probe_v2.py` |
| Analysis scripts | `scripts/signal_corr.py`, `scripts/signal_study.py` |
| Base/wrapper | frozen Qwen3-8B + v1 trained wrapper |
| Per-item passes | wrapper, no-context base, memory-ablation, MC likelihood, greedy generation, teacher-forced answer-span pass |
| Main seeds | `{42, 7, 11, 13, 1, 5, 17, 99, 23}` |
| Main benchmark set | QuALITY, MuSR-mm, RULER-NIAH, HotpotQA, TriviaQA |
| Main sample size | 9 seeds × 150 items / benchmark = 1350 items per main benchmark |
| Additional benchmark sweep | QuALITY-hard, SQuAD-v2, NarrativeQA, MS-MARCO |
| Candidate signals | confidence / entropy / margin, sequence logprob, wrapper-to-base KL/JS/TV, per-layer logit-lens divergence, memory influence, residual drift, slot geometry |
| Targets | native correctness, `help`, `noharm`, continuous lift |
| Correlation metrics | AUROC for binary targets, Spearman for continuous targets |
| Artifact folder | [`grids-2026-06-04/`](grids-2026-06-04/) |
| Grid CSVs | [`grid_auroc_correct.csv`](grids-2026-06-04/grid_auroc_correct.csv), [`grid_auroc_noharm.csv`](grids-2026-06-04/grid_auroc_noharm.csv), [`grid_spearman_cont.csv`](grids-2026-06-04/grid_spearman_cont.csv), [`corr_long.csv`](grids-2026-06-04/corr_long.csv) |
| Figures | [`rank_help.png`](grids-2026-06-04/rank_help.png), [`rank_interesting.png`](grids-2026-06-04/rank_interesting.png) |
| PDF note | [`v1.5-intrinsic-gating-study-2026-06-04.pdf`](v1.5-intrinsic-gating-study-2026-06-04.pdf) |
| Caveats | Confidence signals are useful but partly obvious; non-obvious design signal is late-layer divergence/drift used inversely. Some single-seed layer signals did not survive multi-seed checks. |

## P08-S4 — v2 design setting

Use for v2 design claims, not for verified result cells.

| field | value |
|---|---|
| Setting ID | `P08-S4` |
| Status | design / next-version setting, not a completed benchmark result |
| Base model | frozen Qwen3-8B by default; Llama-3.1-8B as possible cross-family validation |
| Core extension | `update_from_generation(m, GenStep)`, `serialize(m)`, `load(bytes)` |
| Memory timing | suffix-style write after generation; default trigger = end-of-turn |
| Read path | reuse v1 read/apply modes first; residual/multi-layer injection as secondary track |
| Persistence | serialize latent memory state and reload across sessions |
| Target datasets | LongMemEval, LOCOMO; synthetic multi-session smoke first |
| Main baseline | TokMem; related latent reasoning / memory-token methods in related-work doc |
| Design doc | [`v2-plan.md`](v2-plan.md) |
| Related work | [`v2-related-work.md`](v2-related-work.md) |
| Caveats | No v2 benchmark result should be reported as verified until this setting is implemented and evaluated. |

## P08-S5 — RCA / project motivation setting

Use for RCA motivation cells or slides that cite prior RCA SFT observations as
motivation rather than Plan 08 benchmark results.

| field | value |
|---|---|
| Setting ID | `P08-S5` |
| Source repo | `~/workspace/rca-demo/` |
| RCA base family discussed | Qwen2.5-7B-Instruct (`q25`) |
| Datasets | Nezha, OpenRCA-500, RCAEval, lincyaw/rca; private Liang DTS only as internal qualitative probe |
| Role in Plan 08 | motivation: direct SFT can teach RCA behavior but can also regress general capabilities / format / evidence grounding |
| Public/private boundary | Do not expose raw Nokia DTS prompts or private local paths in public slides |
| Relevant Plan 08 doc | [`v0-learned-memory-wrapper.md`](v0-learned-memory-wrapper.md), [`v0-how-we-got-here.md`](v0-how-we-got-here.md) |
| Caveats | RCA-demo numbers are not Plan 08 wrapper results unless explicitly labeled as RCA motivation. |

## Maintenance rule

When adding a new result table:

1. Add or update the relevant setting entry here.
2. Add a `setting` column or caption note in the result table.
3. Link the setting ID from any slide or summary that repeats the result.
4. If the result uses an external dataset/paper/tool, include its link or stable
   source identifier in the setting entry.
