# Janus: One Site, Two Frontiers — Long-Context (Inference) and Catastrophic Forgetting (Training) Are Governed by the Same Intrinsic Heads

*Working paper draft · Plan 09 · 2026-06-04. Status: measurement complete (G); causal test (H3) running.*

---

## Abstract

Long-context degradation at **inference** and catastrophic forgetting at **training** are normally studied as separate problems with separate fixes. We test a unifying hypothesis (**Janus**): the transformer-intrinsic sites that *carry long-context behaviour* are the *same* sites that *fine-tuning perturbs most*. We build a unified, decoupled instrumentation harness — **39 intrinsic metrics across 12 angles**, computed in a **minimal 5-pass schedule** — and run it as a grid over **4 model families (Qwen3-8B, Qwen2.5-7B-Instruct, GLM-4-9B, Qwen3.5-9B) × 12 benchmarks = 41 cells**, all in native bf16. We find that the per-head long-context metrics (attention reach, retrieval/induction, sink, KV/value magnitude) are **uniformly positively correlated** with the per-head forgetting metrics (weight-drift, activation-drift, Fisher, gradient) — every cell of the 9×5 coupling matrix is positive (ρ = 0.17–0.56), and four couplings are sign-consistent across all 41 model×dataset cells. The coupling is strongest with what SFT *actually changes* (weight/activation drift) and holds across vendors, including the hybrid linear-attention Qwen3.5. We then test causally whether **protecting the long-context heads during SFT** preserves long-context and general ability without blocking new-domain learning (H3, in progress). If confirmed, a single data-agnostic rule — detect the load-bearing heads on generic text, protect them during any fine-tune — addresses both frontiers at once.

---

## 1. Introduction

Two failure modes:
- **Long-context (read-time / inference).** As context grows, attention disperses, "lost-in-the-middle" appears, KV-cache balloons, and retrieval degrades.
- **Catastrophic forgetting (write-time / training).** Fine-tuning on a new domain erodes pre-trained code/math/general ability.

**Thesis (Janus).** Both are governed by the same small set of **intrinsic load-bearing heads** — retrieval/induction heads, attention sinks, and heads with large KV/value magnitude. Long context *overloads* them; fine-tuning *perturbs* them. If the read-side importance ranking predicts the write-side disruption ranking, then one data-agnostic rule (detect on generic text → protect during SFT) improves long-context retention **and** prevents forgetting, with zero task-specific parameters.

**Contributions.**
1. A **unified intrinsic-metric instrument** (39 metrics / 12 angles) with a **decoupled minimal-compute** schedule (§3).
2. The **measurement result**: a per-head **long-context × forgetting coupling** that is uniformly positive across 4 families × 12 benchmarks (§5.4) — the central empirical claim.
3. A **causal protection test** (H3, §5.5): does protecting the long-context heads during SFT preserve both frontiers?

**Novelty vs prior work (audit, §2).** Every *single leg* is published — retrieval heads `[retrieval-head]`, attention sinks `[sink-streaming]`, forgetting-localizes-to-heads `[mech-forget]`, MoE expert-freezing by task-affinity (ESFT/DES-MoE/DAS), feature-space anti-forgetting `[sae-fd]`, and the two single-leg sink fixes `[focusft]` (long-ctx) / `[sink-forget]` (forgetting). **Nobody connects the two legs.** Our contribution is the *conjunction*: the read-side criterion **predicts** the write-side disruption a-priori (vs `[mech-forget]`'s post-hoc ΔW set), and one criterion governs both.

---

## 2. Related work & novelty position
See [`references.md`](references.md) and the audit in [`../../ideas/rca-transformer-intrinsic-2026-06-03.md`](../../ideas/rca-transformer-intrinsic-2026-06-03.md) §5.5–5.6. Headline: the **predictive joint coupling** is the wedge; **MoE super-expert protection** is the least-crowded instantiation.

---

## 3. Method: intrinsic-metric instrument

### 3.1 The 12 angles / 39 metrics (frontier-tagged)
LC = long-context/inference · CF = forgetting/training · ST = structural covariate.

| angle | metrics (definition) | frontier |
|---|---|---|
| 1 sink/stream | **sink** (mean attn prob to position 0), **attn_entropy** (−Σp log p of the attention row) | LC |
| 2 retrieval/copy | **retrieval** (attn from last query to a planted needle's value tokens), **induction** (attn to t−offset on a repeated random sequence), **prev_token** (attn to t−1) | LC |
| 3 attention reach | **attn_distance** (Σ attn·\|i−j\|), **receptive_field** (=exp(attn_entropy); *redundant, pruned*) | LC |
| 4 KV/value geometry | **kv_norm** (‖K‖ per head), **v_norm** (‖V‖), **out_norm** (‖attn output‖), **kv_outlier** | LC/ST |
| 5 repr. rank | **eff_rank** (participation ratio of hidden-state singular values), **anisotropy** (mean pairwise cosine of token reps), **intrinsic_dim** (TwoNN), **spectral_decay** | ST |
| 6 repr. dynamics | **update_norm** (‖h_l−h_{l−1}‖), **curvature** (‖h_{l+1}−2h_l+h_{l−1}‖), **cka_adjacent** (linear CKA of consecutive layers), **token_mixing** (off-diagonal attn mass) | ST/LC |
| 7 information theory | **tuned_lens_kl/depth** (ridge h_l→h_final then unembed; KL & top-1 agreement to final), **ll_kl_to_final**, **ll_top1_depth**, **ll_entropy**, **lens_gap**, **surprisal** (NLL/token) | ST |
| 8 activation dist. | **act_kurtosis**, **act_sparsity** (frac \|x\|<0.1σ), **massive_max** (max channel \|act\|/median), **massive_count**, **gini**, **dead_frac** | ST |
| 9 parameter spectra | **stable_rank** (‖W‖_F²/‖W‖₂²), **eff_rank**, **spectral_norm**, **HT-SR α** (Hill power-law exponent of the eigenvalue tail), **condition_num**, **weight_entropy** | ST/CF |
| 10 per-head param geom | **head_alpha** (HT-SR on the per-head q/o slice), **head_wnorm**, **ov_norm** (‖W_O W_V‖) | CF/ST |
| 11 forgetting dynamics | **fisher** (Σ g² per head, empirical Fisher diag), **grad_mag** (mean \|g\|), **grad_noise** (per-head gradient CV² across batches), **dW_drift** (‖ΔW‖ per head after SFT), **act_drift** (per-head activation change after SFT) | CF |
| 12 behaviour | **niah_acc** (needle-in-haystack accuracy @ length×depth), **mmlu/gsm8k acc**, **retention Δ** (after−before SFT) | LC/CF |

### 3.2 Decoupled minimal-compute schedule
Every metric is derived from one of **5 pass types**; A and C share the *same* batches and B is amortised once per model:
- **A** — 1 forward / text (eager, hidden + attention + labels, with K/V hooks) → angles 1–8 + surprisal; tuned-lens reuses A's hidden via a ridge solve.
- **B** — weights only, once per model → angles 9–10 (SVD per matrix).
- **C** — 1 backward on the same A batches → Fisher / grad / grad-noise.
- **D** — 1 short SFT / dataset → weight- & activation-drift.
- **E** — behavioural (NIAH, capability, retention).

Code: [`runs/janus_run.py`](runs/janus_run.py) (`grid`/`h3`), [`runs/analyze_grid.py`](runs/analyze_grid.py), [`runs/orch_grid.sh`](runs/orch_grid.sh).

### 3.3 Coupling statistic
For two per-head metrics A, B with vectors over all (layer, head) sites, the coupling is the **Spearman rank correlation** ρ(A, B), computed per (model, dataset) cell and pooled across cells. A pair is **consistent** if sign(ρ) is identical across all cells.

---

## 4. Experimental setup
- **Models (native bf16, no quantization; metrics in fp32):** Qwen3-8B, Qwen2.5-7B-Instruct, GLM-4-9B (cross-vendor), Qwen3.5-9B (hybrid 3:1 Gated-DeltaNet/full-attention; per-head attention metrics on its 8 full-attention layers, remapped).
- **Benchmarks (12):** wikitext, MMLU, MATH, GSM8K, TriviaQA, BBH, SQuAD-v2, HotpotQA, QuALITY, NarrativeQA, MuSR, MS-MARCO.
- **Compute:** 2× 4×H100. Qwen3.5 in an isolated torch-2.11/transformers-5.10 venv. 41/48 (model×dataset) cells (6 graceful skips), 0 failures.

---

## 5. Results

### 5.1 Detectors reproduce known phenomena (Phase-0)
On Qwen3-8B: BOS attention sink (L13H12 mass = 1.0), **5 massive-activation channels** (ch2276 ≈ 410× median), retrieval-head hub at L23. → [`phase0-results-2026-06-03.md`](phase0-results-2026-06-03.md).

### 5.2 Sink heads ≠ retrieval heads (Phase-1)
Top-15 sink vs retrieval head sets are **disjoint (Jaccard 0)** across 0.6B→14B and across families. → the long-context site to protect is the **retrieval heads**, not the sink. [`phase1-results-2026-06-04.md`](phase1-results-2026-06-04.md), [`facts-2026-06-04.md`](facts-2026-06-04.md).

### 5.3 A cross-family representation-collapse axis
Vendor-independent (incl. GLM-4): late layers → residual-norm↑, prediction crystallizes (KL-to-final↓), eff-rank↓, **massive activations emerge** (act_kurtosis~massive_max +0.93), and the MLP down-proj spectrum co-moves (down stable-rank ~ KL-to-final −0.71). [`facts-2026-06-04.md`](facts-2026-06-04.md), `figs/14_facts_consistent.png`.

### 5.4 ★ Headline: long-context × forgetting coupling
Per-head Spearman ρ, pooled over 41 cells (`figs/G1_LC_CF_coupling.png`). **Every LC head-metric is positively coupled to every CF metric (0.17–0.56; no negative cell).**

| LC ↓ \ CF → | act_drift | dW_drift | fisher | grad_mag | grad_noise |
|---|---|---|---|---|---|
| attn_distance | 0.48 | **0.53** | 0.23 | 0.22 | 0.46 |
| v_norm | 0.47 | **0.56** | 0.25 | 0.21 | 0.26 |
| kv_norm | 0.45 | 0.44 | 0.28 | 0.28 | 0.25 |
| retrieval | 0.42 | 0.45 | 0.27 | 0.25 | 0.28 |
| sink | 0.47 | 0.27 | 0.19 | 0.17 | 0.34 |
| prev_token | 0.17 | 0.20 | **0.44** | **0.46** | 0.24 |
| receptive_field | 0.23 | 0.40 | 0.32 | 0.34 | 0.29 |
| attn_entropy | 0.23 | 0.40 | 0.33 | 0.34 | 0.29 |
| induction | 0.30 | 0.36 | 0.32 | 0.31 | 0.27 |

**Sign-consistent across all 41 cells:** `prev_token~grad_mag +0.48`, `attn_distance~act_drift +0.42`, `prev_token~fisher +0.42`, `retrieval~act_drift +0.37`. The coupling is strongest with **drift** (what SFT actually does), not just a-priori Fisher. → the heads doing long-context work are exactly the heads SFT perturbs most. [`grid-metrics-2026-06-04.md`](grid-metrics-2026-06-04.md).

### 5.5 Causal protection test (H3) — COMPLETE, inconclusive (setup-limited)
Cross-family (GLM-4-9B + Qwen3-8B), SFT GSM8K (600 steps, attention-proj) × 5 variants {none / lc / retrieval / random / deltaw}, NIAH+MMLU+GSM8K before/after. **The setup did not induce long-context forgetting**, so the headline causal claim could not be tested: GLM-4-9B NIAH stayed 1.0→1.0 (nothing to forget); Qwen3-8B NIAH *improved* 0.02→0.18–0.62 (the SFT taught the answer format, so protection was counterproductive). MMLU showed only slight movement (GLM-4: retrieval-protect best retention −0.05 vs random/deltaw −0.09 — a weak positive that retrieval-head protection helps *when* there is forgetting). **Conclusion:** GSM8K-SFT is not long-context-degrading on these models (same gap as Phase-1); H3 must be re-run with a forgetting-inducing setup (strong-NIAH base + narrow non-retrieval heavy SFT + longer NIAH). Until then this remains a **measurement** paper with a strong unifying observation (§5.4), not yet a causal one.

---

## 6. Limitations
- Pooling 4 families × 12 datasets is conservative (means drop vs a 3-model subset; a few couplings flip sign in 1–2 cells — the four listed never flip).
- Qwen3.5's per-head attention metrics cover its 8 full-attention layers only.
- NIAH probe is hard on some base models; H3 uses heavy SFT to ensure measurable forgetting. MMLU eval at small n is noisy (verifying the axis).
- H3 (causality) not yet complete; until then this is a **measurement** paper with a strong unifying observation.

## 7. Reproducibility & artifacts
- **Repo:** `github.com/BDeMo/research-notes` → `notes/plans/09-intrinsic-site-protection/`
- **Plan:** `README.md` · `matrix.md` (live ledger) · `validation.md` · `channels.md` · `budget.md` · `references.md`
- **Results notes:** `phase0-results-2026-06-03.md` · `phase1-results-2026-06-04.md` · `facts-2026-06-04.md` · `grid-metrics-2026-06-04.md` · this file
- **Code:** `runs/janus_run.py` · `runs/analyze_grid.py` · `runs/janus_grid_viz.py` · `runs/orch_grid.sh` · `runs/orch_h3.sh` · `runs/GPU_STRATEGY.md`
- **Figures:** `figs/G1_LC_CF_coupling.png` (headline) · `figs/G2_corr_head.png` · `figs/G3_corr_layer.png` · `figs/14_facts_consistent.png` · `figs/09_scaling_ladder.png` · `figs/02_sink_vs_retrieval.png`
