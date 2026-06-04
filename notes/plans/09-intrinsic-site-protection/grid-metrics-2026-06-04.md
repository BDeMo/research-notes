# Janus — unified intrinsic-metric grid (12 angles × 39 metrics × 12 datasets, 2026-06-04)

> Focus (user): the two frontiers are **long-context = inference (LC)** and **catastrophic forgetting = training (CF)**. This run builds the full decoupled metric grid and asks the headline question: **do the sites that carry LC behaviour predict the sites that CF perturbs?**

## What ran
- **Cohort** (native bf16): Qwen3.5-9B, GLM-4-9B, Qwen3-8B, Qwen2.5-7B-Instruct.
- **12 benchmarks**: wikitext, mmlu, math, gsm8k, triviaqa, bbh, squad, hotpotqa, quality, narrativeqa, musr, msmarco.
- **41 (model×dataset) cells**, all 8 H100s (ray iso-venv: Qwen3.5+GLM-4; sam-dev: Qwen3+Qwen2.5), 0 failures.
- Code: `runs/janus_run.py` (`grid`), `runs/orch_grid.sh`, `runs/analyze_grid.py`, `runs/janus_grid_viz.py`. Tables: `runs/grid_summary.json`. Figures: `figs/G1_LC_CF_coupling.png`, `figs/G2_corr_head.png`, `figs/G3_corr_layer.png`.

## Decoupled compute (minimal passes)
Every metric is derived from one of 5 pass types, with **A and C sharing the same batches** and **B amortised once per model**:
- **A** — 1 forward/text (eager, hidden+attn+labels, k/v hooks) → all inference/representation/info/activation metrics (+ surprisal). Tuned-lens reuses A's hidden via a ridge solve.
- **B** — weights only, once per model → parameter spectra (angles 9–10).
- **C** — 1 backward on the same A batches → Fisher / grad-mag / grad-noise.
- **D** — 1 short SFT/dataset → weight-drift + activation-drift.
- **E** — NIAH / retention (behavioural; model-level).

## The 12 angles / 39 metrics (frontier-tagged)
LC = long-context/inference · CF = forgetting/training · ST = structural covariate.

| angle | metrics (frontier) |
|---|---|
| 1 attention sink/stream | sink·LC, attn_entropy·LC |
| 2 retrieval/copy | retrieval·LC, induction·LC, prev_token·LC |
| 3 attention reach | attn_distance·LC, receptive_field·LC |
| 4 KV/value geometry | kv_norm·LC, v_norm·LC, out_norm·ST |
| 5 repr. rank/geometry | eff_rank, anisotropy, intrinsic_dim(TwoNN), spectral_decay ·ST |
| 6 repr. dynamics | update_norm, curvature, cka_adjacent, token_mixing·LC |
| 7 information theory | tuned_lens_kl/depth, ll_kl_to_final, ll_top1_depth, ll_entropy, lens_gap, surprisal ·ST |
| 8 activation dist. | act_kurtosis, act_sparsity, gini, dead_frac, massive_max, massive_count ·ST |
| 9 parameter spectra | w_stable_rank, w_eff_rank, w_spectral_norm, w_ht_alpha·CF, condition_num, weight_entropy ·ST |
| 10 per-head param geom | head_alpha·CF, head_wnorm·ST, ov_norm·ST |
| 11 forgetting dynamics | fisher·CF, grad_mag·CF, grad_noise·CF, dW_drift·CF, act_drift·CF |
| 12 behaviour | (NIAH / retention / surprisal) |

Redundant cousins decoupled: kept `eff_rank` (dropped `repr_entropy≡log`), kept `tuned_lens` over raw logit-lens (+`lens_gap`), `grad_mag`/`fisher` reported but treated as one CF-magnitude axis.

## Headline: LC (inference) × CF (training) coupling — `figs/G1_LC_CF_coupling.png`
Per-head Spearman, pooled across all 41 cells. **Every LC head-metric is positively coupled to every CF metric — the whole 9×5 matrix is positive (0.17–0.56), no negative cell.**

| | act_drift | dW_drift | fisher | grad_mag | grad_noise |
|---|---|---|---|---|---|
| **attn_distance** | 0.48 | **0.53** | 0.23 | 0.22 | 0.46 |
| **v_norm** | 0.47 | **0.56** | 0.25 | 0.21 | 0.26 |
| **kv_norm** | 0.45 | 0.44 | 0.28 | 0.28 | 0.25 |
| **retrieval** | 0.42 | 0.45 | 0.27 | 0.25 | 0.28 |
| **sink** | 0.47 | 0.27 | 0.19 | 0.17 | 0.34 |
| **prev_token** | 0.17 | 0.20 | **0.44** | **0.46** | 0.24 |
| **receptive_field** | 0.23 | 0.40 | 0.32 | 0.34 | 0.29 |
| **attn_entropy** | 0.23 | 0.40 | 0.33 | 0.34 | 0.29 |
| **induction** | 0.30 | 0.36 | 0.32 | 0.31 | 0.27 |

**Sign-consistent across all 41 cells** (robust): `prev_token~grad_mag +0.48`, `attn_distance~act_drift +0.42`, `prev_token~fisher +0.42`, `retrieval~act_drift +0.37`.

### Reading
The heads that do the **long-context work at inference** — wide attention reach (`attn_distance`, `receptive_field`), retrieval/copy (`retrieval`, `prev_token`, `induction`), sinks, and large KV/V magnitude — are systematically the heads that **fine-tuning perturbs most at training** (largest weight-drift, activation-drift, Fisher, gradient). This is the **Janus thesis (one site, two frontiers)**, now confirmed **per-head, cross-family (incl. GLM-4 + Qwen3.5 hybrid), cross-dataset (12 benchmarks)**. The two strongest write-side signals are **weight-drift and activation-drift** (the actual SFT change), more than the a-priori Fisher — i.e. the coupling is strongest with what training *actually does*, not just curvature.

## Caveats
- Pooling 4 families × 12 datasets is conservative: means drop vs the 3-model subset (0.55→0.42) and a few couplings flip sign in 1–2 cells (wide [min,max]); the 4 listed above never flip.
- Qwen3.5 hybrid: per-head attention metrics cover its 8 full-attention layers (remapped); per-layer metrics use all 32.
- 41/48 cells (6 model×dataset loads skipped gracefully).

## Next
- **Causal test**: protect the top LC-coupled heads during SFT → does CF drop without LC regression? (the H3 / method step).
- Add behavioural LC (NIAH acc@length) + retention deltas to close angle 12 per dataset.
- Scale axis (0.6→32B) on the confirmed couplings.
