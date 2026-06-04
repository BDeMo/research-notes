# Janus — cross-model cohort + metric grid (2026-06-04)

> Harvest of the overnight Janus exploration (repo `../../../../janus`, 8×H100: `sam-dev` + `sam-dev-ray`). Forward-only **FACTS** (broad per-layer metrics) + backward **INTRINSIC** (per-head Fisher/SFT drift coupling) + **NIAH/capability** outcome probes. This extends the single-model Phase-0 ([`phase0-results-2026-06-03.md`](phase0-results-2026-06-03.md)) to a cohort and to the H2 coupling test.
> Source tables: `janus/grid_out/_grid_tables/summary.json`; galleries `janus/gallery/INDEX.md` (Qwen3 ladder + Qwen2.5) and `janus/gallery_cohort/INDEX.md` (7–9B cross-vendor cohort).

## Cohort

- **Scale ladder**: Qwen3 0.6B · 1.7B · 4B · 8B · 14B.
- **Cross-vendor 7–9B cohort** (controlled scale, per user 2026-06-04): Qwen3-8B · Qwen3.5-9B · Qwen2.5-7B-Instruct · GLM-4-9B (+ Qwen2.5-1.5B(-Instruct), GLM-4-32B for facts).
- Grid summary aggregates **41 cells** across {glm4-9b, qwen2.5-7b-instruct, qwen3-8b, qwen3.5-9b}.
- All dense. (MoE super-expert headline, P0b, still to run — Qwen3-30B-A3B.)

## Finding 1 — the metric → "frontier" taxonomy (which face each intrinsic metric serves)

Each per-layer intrinsic metric classifies cleanly as long-context (**LC**, read), forgetting (**CF**, write), or neither/stable (**ST**):

- **LC (read-side)**: `retrieval`, `sink`, `induction`, `prev_token`, `attn_distance`, `attn_entropy`, `receptive_field`, `kv_norm`, `v_norm`, `token_mixing`.
- **CF (write-side)**: `dW_drift`, `act_drift`, `fisher`, `grad_mag`, `grad_noise`, `w_ht_alpha`.
- **ST (neither)**: ~25 metrics — `massive_count/max`, `eff_rank`, all lens metrics (`ll_*`, `tuned_lens_*`), `resid_norm`, `anisotropy`, `condition_num`, `spectral_decay`, `out_norm`, etc.

→ Two distinct families of sites exist; the question (H2) is whether the LC ranking predicts the CF ranking.

## Finding 2 — H2 coupling: **qualified PASS** (real, model-consistent, but narrow & moderate)

Across the 4-model cohort (n=41 cells), the read↔write metric pairs that are **positive and consistent in every model**:

| read (LC) ~ write (CF) | mean ρ | min | max | consistent |
|---|---|---|---|---|
| **prev_token ~ grad_mag** | **0.475** | 0.179 | 0.902 | ✓ |
| **attn_distance ~ act_drift** | **0.423** | 0.096 | 0.877 | ✓ |
| **prev_token ~ fisher** | **0.417** | 0.164 | 0.894 | ✓ |
| **retrieval ~ act_drift** | **0.372** | 0.074 | 0.874 | ✓ |

Higher *mean* but **sign-inconsistent** across models (do NOT rely on): `v_norm~dW_drift` 0.552, `sink~act_drift` 0.409, `retrieval~dW_drift` 0.382, `kv_norm~dW_drift` 0.367.

**Reading**: the predictive coupling (DR8/H2) is **real and model-consistent** for *previous-token / attention-distance / retrieval* read metrics against *gradient-magnitude / activation-drift / Fisher* write metrics, at **ρ ≈ 0.37–0.48** — i.e. around the pre-registered H2 ≥ 0.4 bar, three of four clearing it. It is **not** a blanket coupling: sink- and dW-based pairs flip sign across models. This matches the audit's "narrow & predictive" wedge (brainstorm §5.6): the read side predicts *which sites the optimizer will hit hardest*, but the cleanest signal is the **retrieval / previous-token / attention-distance** axis, not the sink axis.

## Finding 3 — sink ≠ retrieval heads (confirmed across the cohort)

Phase-0 on Qwen3-8B: top sink heads (L7–13) vs top retrieval heads (L17–31) → **Jaccard 0, Spearman 0.09**. The cohort gallery reproduces the disjointness across models (`02_sink_vs_retrieval.png`). **Consequence**: the long-context site to *protect* is the **retrieval heads**, not the sink (sink is stabilization/streaming). Kills sink-only Path B again, empirically.

## Finding 4 — cross-model / cross-scale

Site maps, NIAH grids, forgetting/plasticity bars, and the H3 retention–plasticity Pareto reproduce across the Qwen3 0.6B→14B ladder, Qwen3.5, Qwen2.5, and GLM-4 (cross-vendor) → the taxonomy + the consistent couplings are **model-agnostic** (DR5). A `14_facts_consistent.png` "cross-model consistent facts" panel is the cohort headline figure.

## Implications for the plan

1. **H2 gate = qualified pass** → proceed to H3 (causal protection), but key the read-side criterion on **retrieval / previous-token / attention-distance** metrics (the consistent ones), not sink.
2. **Headline still = MoE super-expert protection** (P0b, §6.2) — but that grid was dense; run Qwen3-30B-A3B next.
3. **`grad_mag` / `act_drift` / `fisher`** are the write-side disruption targets to predict; `[mech-forget]` uses a post-hoc ΔW set — our consistent read-side predictors (ρ≈0.4) are the a-priori alternative to beat.
4. Mind the **inconsistent** pairs: don't headline `v_norm~dW_drift` (0.552) — it flips sign across models.

## Provenance
- `janus/grid_out/_grid_tables/summary.json` (frontier map + T4 couplings, n=41).
- `janus/gallery/INDEX.md`, `janus/gallery_cohort/INDEX.md` (14 figure families).
- `janus/GPU_STRATEGY.md` (8×H100 layout; Qwen3.5 hybrid-attn loading solved; ≤32B Fisher/SFT memory rules).
