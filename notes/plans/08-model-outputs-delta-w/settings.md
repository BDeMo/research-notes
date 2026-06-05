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
| Artifact folder | [`summary/2026-06-04/grids-2026-06-04/`](summary/2026-06-04/grids-2026-06-04/) |
| Grid CSVs | [`grid_auroc_correct.csv`](summary/2026-06-04/grids-2026-06-04/grid_auroc_correct.csv), [`grid_auroc_noharm.csv`](summary/2026-06-04/grids-2026-06-04/grid_auroc_noharm.csv), [`grid_spearman_cont.csv`](summary/2026-06-04/grids-2026-06-04/grid_spearman_cont.csv), [`corr_long.csv`](summary/2026-06-04/grids-2026-06-04/corr_long.csv) |
| Figures | [`rank_help.png`](summary/2026-06-04/grids-2026-06-04/rank_help.png), [`rank_interesting.png`](summary/2026-06-04/grids-2026-06-04/rank_interesting.png) |
| PDF note | [`v1.5-intrinsic-gating-study-2026-06-04.pdf`](summary/2026-06-04/v1.5-intrinsic-gating-study-2026-06-04.pdf) |
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

## P08-S6 — v1.5 cross-model 7-family signal probe (2026-06-05)

Use for the 7-model signal-consistency result and the gate ceiling / transfer cells.

| field | value |
|---|---|
| Setting ID | `P08-S6` |
| Inherits | `P08-S1` (canonical wrapper recipe), but swept over 7 base families |
| Probe script | `scripts/probe_v3.py` (native MC + per-layer logit-lens + 16 candidate metrics) |
| Analysis | `scripts/xmodel_consistency.py` (per-signal AUROC/Spearman × model), `scripts/gate_ceiling.py` (logistic gate 5-fold CV + leave-one-model-out transfer) |
| Base models (frozen, bf16) | Qwen3-8B, Qwen3-14B, Qwen3.5-9B, Qwen2.5-7B-Instruct, GLM-4-9B-0414, Phi-3.5-mini-instruct, Mistral-7B-Instruct-v0.3. Qwen3.5 via isolated `transformers-main + torch 2.8` venv (Gated-DeltaNet **torch fallback**); pad_token=eos for Mistral |
| Wrapper | trained on `categorical_niah` per P08-S1 (K=32, `xattn`, 1800 steps) |
| Seeds | Qwen3-8B / GLM / Qwen3.5: `{1,7,11,13}`; others: `{1,7}` |
| Benchmarks | core: quality, musr_mm, ruler_niah, hotpot_qa, trivia_qa; new: quality_hard, squad_v2, narrativeqa, ms_marco |
| Sample size | 13.2k core items across 7 models |
| Targets / metrics | `help`, `no-harm`, `useful=lift>0`; AUROC + Spearman; gate transfer = leave-one-model-out AUROC |
| Comparability | only scalar + fractional-depth layer-curve signals compared (raw `@idx` dropped) |
| Artifacts | [`summary/2026-06-05/grids-xmodel-2026-06-05/xmodel_consistency_7models.csv`](summary/2026-06-05/grids-xmodel-2026-06-05/xmodel_consistency_7models.csv) (+`.png`), [`gate_ceiling_cv.csv`](summary/2026-06-05/grids-xmodel-2026-06-05/gate_ceiling_cv.csv), [`gate_transfer_lomo.csv`](summary/2026-06-05/grids-xmodel-2026-06-05/gate_transfer_lomo.csv); verdicts [`v1.5-metric-candidates-2026-06-04.md#7`](summary/2026-06-04/v1.5-metric-candidates-2026-06-04.md) §7 |
| Headline | `delta_last` general (AUROC 0.59–0.80 in all 7); gate LOMO AUROC 0.71; logit-lens model-specific |
| Caveats | wrappers here are cat_niah-trained ⇒ benches are OOD; per-model seed counts differ (4 vs 2). |

## P08-S7 — train×test capability-boundary grid (2026-06-05)

Use for the wrapper transfer/boundary heatmap and "in-distribution 提点" cells.

| field | value |
|---|---|
| Setting ID | `P08-S7` |
| Inherits | `P08-S1`, with `--train-dataset` (wrapper trains on each dataset) + `--n-memory 64` |
| Script | `scripts/probe_v3.py --train-dataset <D> --combine xattn --n-memory 64 --train-steps 1800 --n-eval 100 --no-cf-metrics --no-attn-metrics`; analysis `scripts/grid_transfer.py` |
| Base model | Qwen3-8B (fixed — this is the *data* axis) |
| Train datasets (9, grid rows) | categorical_niah, ruler_niah, quality, musr_mm, hotpot_qa, trivia_qa, squad_v2, narrativeqa, ms_marco (train seed 42) |
| Test benches (8, grid cols) | quality, musr_mm, ruler_niah, hotpot_qa, trivia_qa, squad_v2, narrativeqa, ms_marco (eval seed 43, held-out items) |
| Metric | Δ = mean(native_w) − mean(native_0); native = MC accuracy or SQuAD-F1 / ROUGE-L per bench |
| Distance taxonomy | same-dataset / same-task {retrieval, MC, exQA, absQA} / same-domain {synthetic, wiki, literary, web} / cross |
| Artifacts | [`summary/2026-06-05/grids-xmodel-2026-06-05/transfer_matrix.csv`](summary/2026-06-05/grids-xmodel-2026-06-05/transfer_matrix.csv), [`transfer_long.csv`](summary/2026-06-05/grids-xmodel-2026-06-05/transfer_long.csv), [`transfer_heatmap.png`](summary/2026-06-05/grids-xmodel-2026-06-05/transfer_heatmap.png) |
| Headline | in-dist Δ +0.017 (QA +2..+9 pt); same-task −.013, same-domain −.034, cross −.075 (distribution-bound) |
| Caveats | answer-head disabled for non-categorical train datasets; K=64 mild capacity bump vs P08-S1's K=32; 1 seed/cell (significance pending). |

## P08-S8 — do-no-harm gate (routing + gated combine) (2026-06-05)

Use for gate result cells (main-table gated column, do-no-harm, gate transfer).

| field | value |
|---|---|
| Setting ID | `P08-S8` |
| Inherits | signals from `P08-S6`; wrapper from `P08-S1` |
| Gated combine | `mem_embedding/wrapper.py` `combine='gated'`: `e'=e+g·α·Δe`, `g=σ(MLP(memory-write feats[,query]))`; **do-no-harm exact** (g→0 ⇒ base identity); knobs `g_bias_init`, `gate_use_query` |
| Routing gate (headline) | `scripts/gate_route.py` — logistic on general signals (`delta_last`, geometry, `mem_influence_span`) → P(useful); route base↔wrapper at τ; 5-fold CV + leave-one-model-out |
| Soft-gate sweep (ablation) | `scripts/gate_train.py` (foreign-decoy + do-no-harm KL + gate BCE), tmux `gs_c1..c9` — **fails** (gate ~0.5, residual corrupts gen) |
| Result | gated ≥ base on 32/35 cells; LOMO transfer AUROC 0.71 |
| Artifacts | [`summary/2026-06-05/grids-xmodel-2026-06-05/gate_route_main.csv`](summary/2026-06-05/grids-xmodel-2026-06-05/gate_route_main.csv) |
| Caveats | routing table evaluated on cat_niah-trained (OOD) grid; **locking experiment** (gate over the P08-S7 in-distribution wrappers) is the open item before the in-dist-gain + OOD-preserve claim is end-to-end. |

## Maintenance rule

When adding a new result table:

1. Add or update the relevant setting entry here.
2. Add a `setting` column or caption note in the result table.
3. Link the setting ID from any slide or summary that repeats the result.
4. If the result uses an external dataset/paper/tool, include its link or stable
   source identifier in the setting entry.
