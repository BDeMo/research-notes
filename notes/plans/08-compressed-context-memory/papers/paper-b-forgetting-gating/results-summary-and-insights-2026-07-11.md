# Results summary, reliability review, insights & validation plan (2026-07-11)

> ⚠️ **`full` = TRUNCATED to MAXCTX (16k)**, not the whole document (`embed(ctx[:,:MAXCTX])`). It equals *true full* only when a doc ≤ 16k — true for every bench here **except ∞Bench** (131k docs → `full` sees ~12%). Only `rag` and our `auto`(chunk) read the whole doc; `full/window/tome/kvzip/knorm` are MAXCTX-bounded. See [`correctness-audit-2026-07-13.md`](correctness-audit-2026-07-13.md).

Consolidates everything into (1) grounded results, (2) code-review/reliability statement, (3) multi-level insights, (4) validation experiments (the 24h task). **Every number below cites a fact ID + its log/doc**; nothing un-sourced.

---

## 1. Grounded results (each traceable)
Source of truth: [`matrix-facts.md`](matrix-facts.md) F1–F35; per-number provenance in the linked docs.

### 1a. Diagnosis (the durable contribution)
| # | claim | evidence |
|---|---|---|
| F27 | FULL-test map: "more context hurts" (QuALITY full 7.2 < blind 17.9; LongBench-v2 full 33.6 ≈ blind 33.4); compression/retrieval beats full (∞Bench RAG 64.2>53.7); IMP≈full on RULER where ToMe/window collapse | `ft_*` [`main-table-fulltest.md`](main-table-fulltest.md) |
| F31 | Diagnosis is **model-invariant** across 10 models (1.7B→14B, Qwen2.5/3/3.5, GLM-4, Ministral, quad+linear) | `g_*` [`generality-model-matrix.md`](generality-model-matrix.md) |
| F30 | IMP dominated by free RAG on accuracy (only synthetic RULER +2.6; squad −32.6) | `ft_*` [`dive-in-imp-weakness-and-baselines.md`](dive-in-imp-weakness-and-baselines.md) |
| F32 | No signal×span combo closes IMP's QA gap (squad all 11–24 vs RAG 71) | `ia_*` |

### 1b. Redesign (the method work)
| # | claim | evidence |
|---|---|---|
| F33 | +IDF-lexical signal (`all`) closes most multi-hop gap (lb_hotpotqa 17.4→24.1≈RAG); keeps retrieval | `implex_*` |
| F34 | **`chunk` (RAG-borrowing) matches RAG on QA** (squad `span` 46.2→`chunk` **69.3 ≈ RAG 69.5**); but ties, doesn't beat; trades off needle-retrieval (RULER chunk 77 vs span 86) | `rd_*` [`imp-redesign-results.md`](imp-redesign-results.md) |
| F35 | **gold-recall is the most significant intrinsic metric** — predicts downstream (squad span goldR 0.33→acc46, chunk 1.0→acc69); query-cov/kept-frac do not | `_diag_intrinsic2` [`intrinsic-probe-results.md`](intrinsic-probe-results.md) |

### 1c. Paper-A negatives (settled)
F12 compressor is commodity · F13 gated≥full tautological · F22 reconstruction-gate a non-effect · F26 learned soft-memory can't compress extractive QA (beaten by training-free IMP).

---

## 2. Reliability / code review (2026-07-11)
- **IMP modes** ({span,chunk,hier,bm25span,qfree}) + signals ({qrel,surp,IDF-lex}) + Okapi-BM25 (k1=1.5,b=0.75) verified correct against `run_baseline.py` (§_imp). Frozen-base for eval-only modes (`method_um=False`) confirmed → IMP numbers are true frozen-base.
- **Scoring:** MC = length-normalized letter log-likelihood (`mc_loglik`); gen = EM/F1/ROUGE via `score_item`. **Caveat (must state in paper):** gen scoring has a **gold-substring fallback** (`if F1==0 and gold∈generation → 1`) applied *uniformly to every method* — preserves relative rankings but can slightly inflate absolute gen F1. no_ctx/full references computed per-cell with the same scorer (consistent).
- **Sampling disclosure:** long-context headline = FULL split; short-context huge QA (squad/hotpot/trivia/ms_marco) = disclosed N (500 or 2000); ablations N=100/200 disclosed. Registry in [`experiment-config-and-sampling.md`](experiment-config-and-sampling.md).
- **Infra reliability note:** free1/free2/free3 are **shared nodes** — contended in daytime → OOM → partial grids. **Only d15-25/d15-30 are reliable.** The paper-grade FULL-N main comparison is therefore (re)run on d1525+d1530 (this task); flaky-node partial runs are treated as exploratory only.

---

## 3. Insights by level (and how significant)
| level | insight | strength | fact |
|---|---|---|---|
| **regime** | which method wins flips by task; no universal compressor | very strong, model-invariant | F3/F18/F27/F31 |
| **phenomenon** | "more context hurts" on literary-MC & hardest-MC; compression *denoises* | strong, all 10 models | F6/F23/F27/F31 |
| **baseline** | RAG (free, BM25 passage) dominates on QA; needs lexical anchor (fails abstractive) | strong | F9/F30 |
| **method** | IMP matches RAG only after adding chunk-BM25 (`chunk`); ties not beats; span/hier keep retrieval | strong | F33/F34 |
| **signal** | IDF-lexical was the missing ingredient; surprisal keeps noise on structured text; no single signal wins | strong | F20/F33/CMG dive |
| **intrinsic** | **gold-recall predicts downstream; the QA gap = answer-retention gap** (chunk keeps short passages whole) | strong, mechanistic | F35 |
| **architecture** | everything reproduces on cache-free linear GDN; KV-methods can't run there, IMP/RAG can | strong | F10/F28/F31 |

**The one meta-insight:** long-context accuracy = a **keep-the-answer-span** problem; **gold-recall is the intrinsic lever**, and *chunk-granularity + BM25* is what maximizes it on real QA (why RAG & `chunk` win), while token-importance maximizes it only on sparse needles (why `span` wins RULER). ⇒ the *right granularity is input-dependent* → an **adaptive selector keyed on gold-recall / signal-sparsity** is the principled "best of both."

---

## 4. Validation experiments (what proves each insight)
1. **[RUNNING, 24h] FULL-test redesign main comparison** — Qwen3-8B, 5 schemes+RAG, keep 0.5, 25 benchmarks, FULL N (huge short QA N=2000 disclosed), on **d1525×3 + d1530×2** (reliable). Tags `rf_q8_*`. *Validates F34 at full scale + the regime map (F27/F31) with mainstream benches (LongBench-full, BABILong).*
2. **[pending] Adaptive mode-selector** — pick chunk vs span per input by a cheap gold-recall proxy (needle-probe) / signal-sparsity; target ">both fixed modes AND ≥RAG on QA while keeping RULER". *Validates the meta-insight.* Falsifier: if the selector can't beat the better single mode, the "input-dependent granularity" claim is weak.
3. **[pending] gold-recall→accuracy correlation** (full-signal, on a reliable GPU) — regress downstream acc on intrinsic gold-recall across schemes×benches; report R². *Validates F35 as a label-light predictor.*
4. **[pending] query-agnostic amortization** — `qfree` compress-once vs RAG re-retrieve on QuALITY multi-Q-per-article; amortized acc/cost. *The only axis IMP could beat RAG.*

**24h task = #1 (running on our 6 reliable GPUs).** #2–#4 queued behind it.
