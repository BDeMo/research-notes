# Plan 08 — raw result data: what every number means

Result data (CSVs + figures) for plan 08. **Read this first** — it defines the
evaluation protocol, the per-benchmark metric, the condition tags, and a
column-by-column dictionary for every file. Written analysis is in
[`../summary/`](../summary/); exact recipes in [`../settings/settings.md`](../settings/settings.md)
(`P08-S*`); raw per-item probe dumps / checkpoints stay in `mem-test/mem-embedding/` (pods).

---

## 1 · Evaluation protocol (how every score is produced)

For each eval item we build the base-model prompt under a **condition** (below),
then score with the **benchmark-native** metric — no proxy losses.

- **Multiple-choice** (`quality`, `quality_hard`, `musr_mm`): score each answer
  option by its mean token log-likelihood, pick the argmax → `mc_acc` ∈ {0,1}
  (exact match to the gold letter). `mc_margin` = gold logprob − best distractor.
- **Generative / extractive QA**: greedy-generate ≤16 tokens, then
  `score_item(prediction, gold)`; the headline number is the **PRIMARY** metric
  for that bench (below). Implementation: `llm-infra/.../benchmark_metrics.py`.

A reported cell = **mean** of that metric over `n` items (eval seed = train seed+1,
`n_chunks=8`). `n` ≈ 100 in the transfer grid, 150–200 in baselines/ceiling/SFT.

### Per-benchmark PRIMARY metric (the number the tables report)
| bench | metric | meaning | source |
|---|---|---|---|
| quality, quality_hard, musr_mm | `accuracy_letter` (`mc_acc`) | closed multiple-choice accuracy | QuALITY (Pang 2022), MuSR (Sprague 2024) |
| hotpot_qa, squad_v2, trivia_qa | `squad_f1` | token-overlap F1, SQuAD normalization (trivia = max over answer aliases) | HotpotQA (Yang 2018), SQuAD v2 (Rajpurkar 2018), TriviaQA |
| narrativeqa, ms_marco | `rouge_l` | ROUGE-L F1 over reference answers | NarrativeQA (Kočiský 2018), MS MARCO (Bajaj 2018) |
| ruler_niah | `exact_value_match` | exact retrieval of the needle string (0/1) | RULER (Hsieh 2024) |

### Condition tags (suffix on `native_*` / `mc_acc_*`)
| tag | condition | role |
|---|---|---|
| `_0` | **no-context**: base sees the query only | floor |
| `_w` | **ungated wrapper**: K soft memory tokens prepended to the query (no raw text) | the learned module |
| `_full` | **full-context**: raw chunks + query (no wrapper) | ceiling |
| `_sft0` / `_sftfull` | **LoRA-SFT** base (trained on mix), no-context / full-context | adapt-by-fine-tuning baseline |
| gated (routing) | route per item between `_w` and `_0` by a learned gate | the do-no-harm method |

---

## 2 · Settings (which model / recipe)

Headline base = **Qwen3-8B**; 7-family cohort = Qwen3-8B/14B, Qwen3.5-9B,
Qwen2.5-7B, GLM-4-9B, Phi-3.5-mini, Mistral-7B. Wrapper recipe = **K=64 memory
tokens, `combine=xattn`, 1 layer, 1800 steps**. Each file's exact model/setting is
noted below; full recipes (`P08-S0…S8`) in [`../settings/settings.md`](../settings/settings.md):
`P08-S6` = 7-model signal probe · `P08-S7` = train×test transfer grid · `P08-S8` = gate.

---

## 3 · File-by-file column dictionary

### [`grids-2026-06-04/`](grids-2026-06-04/) — single-model signal study (Qwen3-8B, `P08-S3`)
- `corr_long.csv` — `signal, angle, granularity, bench, auroc_correct, auroc_noharm, spearman_cont`: per-signal correlation with usefulness. `auroc_correct` = AUROC(signal ⟶ wrapper-correct); `auroc_noharm` = AUROC(signal ⟶ wrapper ≥ base); `spearman_cont` = rank-corr with the continuous native score. AUROC 0.5 = no signal.
- `grid_auroc_correct.csv`, `grid_auroc_noharm.csv`, `grid_spearman_cont.csv` — `signal, angle, <bench…>`: the same three quantities as signal×bench matrices.
- figures `rank_help.png`, `rank_interesting.png` — signals ranked by help-prediction / by "non-obvious" interest.

### [`grids-xmodel-2026-06-05/`](grids-xmodel-2026-06-05/) — 7-model cohort + gate + boundary
- `xmodel_consistency_7models.csv` — `signal, angle, auroc_help[<model>]×7, spear_lift[<model>]×7, same_direction, strong_all_ge_0.07, mean_|auroc-.5|, min_|auroc-.5|`: per-signal AUROC(predict wrapper helps) and Spearman(signal,lift) for each of the 7 families; `same_direction` = consistent sign across all 7; `strong_all_ge_0.07` = |AUROC−.5|≥.07 in all 7 (the generality test). (`H1`, `P08-S6`.)
- `gate_ceiling_cv.csv` — `feature_set, scope, auroc_useful, auroc_help, auroc_noharm`: how well a logistic gate on a feature set predicts usefulness; `scope` = pooled 5-fold CV vs leave-one-model-out. (`H2`.)
- `gate_transfer_lomo.csv` — `held_out_model, feature_set, auroc_useful_transfer`: train gate on the other 6 families, AUROC on the held-out one (model-agnostic test). (`H2`.)
- `gate_route_main.csv` — `model, bench, regime, no_ctx, ungated, gated_cv, gated_lomo, use_frac, tau`: routing-gate table; `gated_*` = mean native score after routing, `use_frac` = fraction routed to the wrapper, `tau` = threshold. ⚠️ uses **optimistic per-model τ**; the **honest CV** version is `baselines-2026-06-05/gate_baselines.csv`. (`H2`/`§7`.)
- `transfer_matrix.csv` — `train\test, <bench…>`: Δ = (native_w − native_0) for a wrapper trained on `train`, tested on `test` (the capability-boundary heatmap). `transfer_long.csv` — `train, test, distance, native_w, native_0, delta, n`: long form; `distance` ∈ {same-dataset, same-task, same-domain, cross}. (`H3`, `P08-S7`.)
- figures `transfer_heatmap.png`, `xmodel_consistency_7models.png`.

### [`baselines-2026-06-05/`](baselines-2026-06-05/) — honest gate baselines + locking (offline, `gate_baselines.py`)
- `gate_baselines.csv` — `model, bench, no_ctx, ungated, threshold, learned, oracle, n`: mean PRIMARY score under each condition. `no_ctx`/`ungated` as above; `threshold` = route by a single-signal (`delta_last`) cut; `learned` = logistic routing gate; both via **honest 5-fold CV** (cut/τ chosen on train fold). `oracle` = per-item `max(ungated, no_ctx)` (ceiling of any gate). do-no-harm = (gate ≥ no_ctx).
- `locking_by_distance.csv` — `distance, n, delta_ungated, delta_gated, help_ungated, help_gated`: one pooled gate over the transfer grid; `delta_*` = mean (score − no_ctx) for ungated vs gated; `help_*` = fraction with Δ>0. (`§7b`.)

### [`mix-2026-06-05/`](mix-2026-06-05/) — multi-task (mix) training (Qwen3-8B, 3 seeds)
- `mix_results.csv` — `bench, seed, native_w, native_0, delta, n`: one wrapper trained on ALL datasets mixed (`train-dataset=mix`), `native_w`/`native_0` = wrapper vs no-context PRIMARY score, `delta` = their difference, per seed (42/7/11). (`§8`; shows mix does **not** broaden competence.)

### [`seed_std-2026-06-05.csv`](seed_std-2026-06-05.csv) — per-benchmark seed variance (the noise floor)
`setting, bench, n_seeds, no_ctx_mean, no_ctx_std, wrapper_mean, wrapper_std`: mean±std of the PRIMARY score across seeds (Qwen3-8B; `qwen3_8b_catniah` = 4 seeds, `qwen3_8b_mix` = 3 seeds). **Use as the reference error bar:** no-context is stable (std ≈0.01–0.03) but the **wrapper is seed-unstable on QA** (trivia/hotpot wrapper std ≈0.08–0.18), so a wrapper delta smaller than ~1 std on those benches is not meaningful. (v1 Phase-Y 4-seed std is in `v1-results-2026-06-03.md §5` / matrix §1; single-seed transfer-grid cells get CIs from the running 5-seed significance batch.)

> Landing next (will be added here with the same protocol): full-context ceiling
> (`native_full`) and mix+SFT (`native_sft*`) per bench × model.
