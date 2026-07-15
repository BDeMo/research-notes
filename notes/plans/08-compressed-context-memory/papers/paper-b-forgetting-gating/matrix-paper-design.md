# Matrix 3 — PAPER DESIGN (A & B: claims → method → evidence → status → gaps)

> Two papers off one fact-base. Status: ✅ have · 🏃 in progress · ⏳ todo. Last synced 2026-07-06.

## Paper A — the do-no-harm GATE  ·  **RE-OPENED 2026-07-06** to combine KVzip (see `PAPER-A-v1.8.1-complete.md`); was archived
**Thesis:** the compressor is a commodity (F12); the contribution is a **compressor-agnostic, out-of-sample-verified gate** that makes *any* lossy compressor "never worse than the feasible baseline." Focus = **robustness + gating performance + generality across compression methods**. Central question: **base-model intrinsic signal vs a trained gate?**

| dimension | plan | supported by | status | gap |
|---|---|---|---|---|
| C-A1 do-no-harm (OUT-of-sample) | gated ≥ feasible baseline with a **conformal/held-out** threshold + stated coverage | F13 (in-sample was tautological) | ⏳ | **X-C3 not run** — the headline needs this |
| C-A2 intrinsic-vs-trained gate | compare base-margin/conf/entropy/recon signals vs a trained gate head | v1.8 signals (`conf`,`neg_recon`,`targ`,`dlogit`) | 🏃 partial | need clean intrinsic-vs-trained comparison (X-C6) |
| C-A3 **generality across compressors** | one gate over KV-evict / prompt-compress / merge / RAG | fact-base has all compressor outputs | ⏳ | **X-C6** — the differentiator; not yet run |
| C-A4 robustness (cross-model/size) | gate transfers across bases/sizes | scaling/cross-family checkpoints (X-scale) | ⏳ | X-C13 |
| method | 1 frozen base, 2 losses (answer+reconstruct), 1 gate signal (=reconstruction), 1 guarantee (conformal) | `method-elegance-plan-v1.8.x.md` | ✅ designed | validate pruned≈full (T1 ✅) + conformal (⏳) |
| positioning | selective prediction / calibration for compressed memory | — | ✅ | neighbors: cascades, speculative decoding, conformal |
| **honest caveats** | gist/cart NOT faithful (drop or exact-repro); "gated≥full" only out-of-sample | matrix-facts F13, excluded | ✅ noted | — |

## Paper B — identify & keep the IMPORTANT information (impactful)
**Current method version: `IMP-v2.1.0`** (span-level, Mode A training-free, span=32, keep=0.5, signals={query-relevance, surprisal}) — the version behind all current result tables (token-level `IMP-v2.0` superseded).

**Thesis (reframed 2026-07-06):** long-context accuracy is bottlenecked by *finding the important info*; every training-free baseline fails in a characterizable regime (F1–F11). Build **IMP** — a **pluggable importance-router** that uses length-robust signals to keep important tokens & drop redundant ones, **distilled (transferred) from an existing base's own importance** (no base retrain, compute/data-frugal), works on **linear+quadratic**, **solves long-prefill**, and ships as a **fast-reproducible checkpoint**. Focus = **generality · performance · efficient · simple · insightful**. Design: `v2.1.0-paperB-method-designs.md`. **No gate.**

| dimension | plan | supported by | status | gap |
|---|---|---|---|---|
| C-B1 necessity | compress ≫ feasible no-compression baseline when input over-runs window | QuALITY over-window 3.5× (v1.8); necessity sweep (X-nec) | ✅ core evidence | — |
| C-B2 diagnosis (why baselines fail) | 10 stories + dive-ins + scaling/cross-family | F1–F11, Figs 1–7 | ✅ strong | corrected kvzip/full-hurts (done) |
| C-B3 **importance identification** | dive-in metrics (attention, MINE-MI, key-norm, reconstructability, redundancy) → per-token importance | 16 metrics in `v2.1.0 §3` | ⏳ | **X-C2 (MINE) + X-C8** not run |
| C-B4 **method = IMP** (pluggable importance-router: cheap signal-scoring → keep-verbatim/merge-redundant → short prefill; distilled from base's own importance; + linear side-cache) | `v2.1.0-paperB-method-designs.md` (D1–D6); ToMe ✅ = naive baseline (F14) | 🏃 design done | build order in design doc; needs C-B3 signals |
| C-B5 linear/GDN arm | merge + brief re-train (R-MeeTo) on GDN base | F10, R-MeeTo | ⏳ | X-C5 |
| C-B6 domain generalization | 5×5 transfer + intrinsic predictors | T1 rows (aggregate) | ⏳ | **X-C1 full 5×5 + X-C7** not done |
| C-B7 ~~do-no-harm gate~~ | **REMOVED from Paper B** (2026-07-06) — the gate/C3 is Paper A's (archived) | — | ✂️ dropped | Paper B is now a performance/compression method, not a gate |
| eval suite | RULER length-sweep + real long-doc + (add ∞Bench/NoCha/LongBench-v2) | X-nec, fact-base | 🏃 | X-C12 (global-reasoning bench) |
| baselines (faithful only) | ~20 kvpress KV + LLMLingua-2 + (Long)LLMLingua + BM25-RAG + window + full/no_ctx | catalog-faithfulness (D21) | ✅ | gist/cart excluded unless exact (X-C10) |
| **honest caveats** | RAG hurts QuALITY; multi_needle broken; kvzip retrieval-only | matrix-facts | ✅ noted | — |

## Shared / differences (see OVERVIEW for prose)
- **Shared facts:** F12 (commodity compressor), F1–F11 (no universal baseline), F6/F13 (full-not-safe, gate-not-tautological), F10 (GDN interface).
- **A-only:** conformal gate machinery, intrinsic-vs-trained, compressor-generality.
- **B-only:** necessity crossover, importance identification + guided-merging method, 5×5 transfer, SSM merge+retrain, the 10-story diagnosis.

## Critical open risks (must close before submission)
1. **X-C3 conformal gate** — without it, Paper A's headline (do-no-harm) rests on a tautology. **Top priority.**
2. **X-C4/X-C2** — Paper B's *method* (importance-guided merging) doesn't exist yet; ToMe alone is a negative result (F14).
3. **X-C1** — the 5×5 transfer matrix (C2/C-B6) is only aggregate rows.
4. **X-C6** — Paper A's generality claim (one gate over all compressors) is unvalidated.

*Provenance: `method-elegance-plan-v1.8.x.md`, `v2.0.0-plan.md`, `v2.1.0-direction-and-method-plan.md`, `matrix-facts.md`, `matrix-experiments.md`.*
