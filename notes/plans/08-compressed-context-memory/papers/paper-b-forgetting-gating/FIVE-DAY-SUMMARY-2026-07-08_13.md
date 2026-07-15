# Five-day summary (2026-07-08 → 07-13) — experiments, problems, analyses

> ⚠️ **`full` = TRUNCATED to MAXCTX (16k)**, not the whole document (`embed(ctx[:,:MAXCTX])`). It equals *true full* only when a doc ≤ 16k — true for every bench here **except ∞Bench** (131k docs → `full` sees ~12%). Only `rag` and our `auto`(chunk) read the whole doc; `full/window/tome/kvzip/knorm` are MAXCTX-bounded. See [`correctness-audit-2026-07-13.md`](correctness-audit-2026-07-13.md).

Everything run over the five days, every problem found, every analysis — as tables. Paper B ("observe long-context failures → bolt a training-free importance-router onto a frozen base"). Base grids on `grid_impredesign/` + `grid_fulltable/`. All scores ×100. Cross-refs: facts [`matrix-facts.md`](matrix-facts.md) (F18–F44), tables [`imp-redesign-results.md`](imp-redesign-results.md), [`main-table-fulltest.md`](main-table-fulltest.md), sampling [`experiment-config-and-sampling.md`](experiment-config-and-sampling.md).

---

## 1. Experiments run (grids)
| # | grid / tag | date | scope | N | GPUs | status |
|---|---|---|---|---|---|---|
| E1 | `ft_*` first full main table | 07-08 | 9 methods × 16 benches, Qwen3-8B | full/2000 | flaky free | done (superseded by E5) |
| E2 | `rd_q8_*` IMP-redesign keep-sweep | 07-09→11 | 5 IMP modes+RAG × 4 keeps × benches | 200 | d1525/free1 | **done 359/360** |
| E3 | multi-model generality `g_*` | 07-09 | 10 models (1.7B–14B, Qwen2.5/3/3.5, GLM-4, Ministral) | 100 | free/d15 | done (F31) |
| E4 | intrinsic probe `_diag_intrinsic2` | 07-09 | gold-recall/query-cov/kept-frac per mode | — | CPU | done (F35) |
| E5 | **`rf_q8_*` reliable FULL-N main comparison** | 07-11 | 6 methods × 25 benches, Qwen3-8B, **1 worker/GPU** | **FULL** (short QA 2000) | d1525×3+d1530×2 | **done 150/150, 0 fail** |
| E6 | `au_q8_*` auto-baseline (peak-τ3) | 07-11→12 | auto over all 20 discriminative benches | full/2000 | d1525+d1530 | done 20/20 |
| E7 | `ax_q8_*` router + adaptive-budget study | 07-12 | routers {peak2,peak4,mc,qover} + budget sweep + rag tight | full/2000 | free1×4 | done 58/68 (10 16k-OOM, non-blocking) |
| E8 | **`fam_*` ALL-family sweep** | 07-12 | 9 families × 5 crossover benches × {auto,chunk,span,rag} | 100 | free3×2 | **done 180/180, 0 fail** |
| E9 | `case_diag` IMP-vs-RAG case diff | 07-13 | gold-recall + beyond-MAXCTX, tokenizer-only | 30 | CPU | done (F44) |
| E10 | `rf2_*` truncation-fix validation | 07-13 | chunk/auto full-doc vs truncated, ∞Bench + squad | 100/200 | d1525+free1 | squad ✓; ∞Bench running |

**Coverage:** benches = RULER-NIAH, ∞Bench-choice, LongBench-v1 (2wikimqa/hotpotqa/musique/multifieldqa/narrativeqa/qasper/triviaqa/repobench/trec/samsum/qmsum/passageretrieval), LongBench-v2, BABILong (qa1-4k/qa1/qa2/qa3-16k), QuALITY-hard, MuSR, squad_v2, hotpot_qa, trivia_qa, ms_marco. Methods = IMP{span,chunk,hier,bm25span,qfree,auto}, RAG(BM25), full, no_ctx, (+ kvzip/knorm/ll2/tome/window in E1/E5 main table).

---

## 2. Problems / bugs found & resolved
| # | problem | impact | root cause | fix | validity of old results |
|---|---|---|---|---|---|
| P1 | free-node grids kept dying | grids stalled (150→24 etc.) | **self-inflicted**: multiple grids stacked on one node + 16k forward ≈72GB → own jobs OOM (NOT contention) | discipline: 1 worker/GPU, 1 grid/node; move to d1525/d1530 | re-ran E5 cleanly |
| P2 | `lb_triviaqa`/`lb_repobench` 16k OOM | 12 cells failed | longest single-item prefills (~72GB) | report at **12k** (disclosed) | others at 16k valid |
| P3 | MC scoring inconsistency (earlier) | MC numbers not comparable | gen-based vs loglik | unified to `mc_loglik` (length-norm letter LL) | fixed pre-window |
| P4 | gen-scoring gold-substring fallback | may inflate absolute gen F1 | `if F1==0 and gold∈text →1` | kept (applied to ALL methods uniformly) → rankings valid | disclosed caveat |
| P5 | 5 LongBench gen tasks near-floor | look like method failure | substring-F1 unfit for summ/classif/code (qmsum/samsum/passageretrieval/repobench/trec) | **excluded from method claims** (F40) | not method signal |
| P6 | `peak`/`qover` auto-routers | auto mis-routed (squad→span, −20) | BM25 peakiness ≠ the chunk/span decision | switch default router → **`mc`** (options-aware) (F41) | peak-τ3 `au_*` superseded |
| P7 | adaptive budget didn't help ∞Bench | H2 disproved | gap is retrieval-mechanism, not budget | negative result recorded (F42) | honest ceiling |
| P8 | **∞Bench −19 truncation asymmetry** | IMP blind to 88% of doc; RAG sees all | `_imp` truncates to MAXCTX *before* selecting; `_rag` doesn't | chunk/BM25 family retrieves **full doc** (`GCM_IMP_FULLDOC`) (F44) | **valid for ctx≤MAXCTX; only ∞Bench affected** |
| P9 | Qwen3-14B RULER = 0/0/0 | one cell unusable | harness/scoring artifact for 14B on RULER | excluded | isolated |

---

## 3. Facts & analyses (F18–F44, newest first)
| fact | one-line claim | polarity |
|---|---|---|
| F44 | ∞Bench −19 was a **truncation artifact** (IMP saw first 16k of 131k; RAG saw all) — fixed via full-doc retrieval | bug→fix |
| F43 | diagnosis is **model-invariant across 9 families / 3 archs / 1.7B–14B** (crossover, chunk=RAG, RAG≫full all hold) | + strong |
| F42 | adaptive/tighter budget does **NOT** close ∞Bench (even RAG loses as budget shrinks) — retrieval-mechanism gap | − |
| F41 | **`mc` auto-router recovers best-of-both**; peak/qover fail (mis-route squad→span) | + method |
| F40 | 5 LongBench gen tasks scorer-limited → excluded | honesty |
| F39 | **RAG denoises far above `full`** on ∞Bench (+17.5) & needle — biggest single effect; IMP's main gap | + / gap |
| F38 | `qfree` (query-agnostic) weakest everywhere → query-conditioning essential | mechanism |
| F37 | **chunk↔span crossover is real & input-predictable** (lexical anchor→chunk; reasoning→span) | + key |
| F36 | **`chunk` = RAG on extractive QA at full N** (squad 66.3≈67.2) | + |
| F35 | **gold-recall predicts downstream accuracy** (query-cov & kept-frac don't) | mechanism |
| F34 | chunk-BM25 borrowed into IMP closes the QA gap (N=200; upgraded by F36) | + |
| F33 | IDF-lexical signal (`all`) closes most multi-hop gap | + |
| F32 | no signal×span combo closes IMP's QA gap — structural, not hyperparameter | − |
| F31 | diagnosis model-invariant on 10 models (first pass) | + |
| F30 | IMP dominated by free RAG on accuracy (only synthetic RULER +2.6) | − boundary |
| F27 | full-test map: "more context hurts"; compression/retrieval > full | + diagnosis |
| F22–F26 | reconstruction-gate a non-effect; learned soft-memory can't beat training-free IMP on extractive QA | − (Paper A) |
| F18–F21 | head-to-head; MC-fix; signal probe; KV-free family baselines | setup |

---

## 4. Method evolution
IMP-v2.0 (token-level, failed coherent QA) → v2.1.0 (span-level) → v2.1.1 (+IDF-lexical, `signal=all`) → **redesign** (5 modes: span/chunk/hier/bm25span/qfree; F34/F36) → **`auto` selector** (mc-router, F41) → **full-doc retrieval** (F44). Default now: `auto`, router `mc`, `signal=all`, span 32 / chunk 256, `FULLDOC=1`.

---

## 5. Best result & where the gap is (Qwen3-8B, keep 0.5, full-N)
| regime | best | ours `auto`(mc) | RAG | full | ours−RAG | note |
|---|---|--:|--:|--:|--:|---|
| extractive QA (squad) | tie | 66.3 | 67.2 | 65.9 | −0.9 | same lexical retrieval |
| hard-MC, ctx hurts (quality_hard) | **ours** | **12.8** | 9.6 | 9.2 | **+3.2** | RAG retrieves noise; span keeps reasoning tokens |
| needle (ruler) | tie | 98 | 98.8 | 99 | −0.8 | both isolate needle |
| synthetic reasoning (babilong_qa3) | **ours(span)** | 29 | 27 | 29 | **+2** | importance keeps distributed facts |
| multi-hop (hotpot 4k) | RAG | 45.6 | 55.8 | 57.2 | −10.2 | passage-RAG gathers evidence better (true gap) |
| single-passage MC (∞Bench) | **ours(chunk-FD)** | **76.0** (chunk-FD) | 69.0 | 57 | **+7.0 WIN** (same N=100) | **F44 fix flipped −19→+7: truncation was the whole gap** |
| open QA (trivia/musr) | RAG | 66.5/48.9 | 72.7/56.7 | — | −6/−8 | parametric + passage retrieval |

**Overall best method = RAG** (free BM25). **Our `auto`(mc) ties RAG on extractive/needle, beats it on no-lexical hard-MC & reasoning, runs on linear + hard-MC where RAG loses.** Genuine remaining gap = multi-hop (−10) and (pending F44 re-run) ∞Bench: both are RAG's *passage-level* retrieval > IMP's fixed-chunk/token importance. **Paper framing = diagnosis (F27/F37/F39) + model-invariant generality (F43) + a regime-spanning training-free router — not "beats RAG".**

---

## 6. Not-yet-done (in flight / next)
- ⏳ **`rf2_*` ∞Bench full-doc re-run** (F44 fix) — quantify how much of −19 was the artifact.
- ▶ **best-of-ALL-baselines table** (vs kvzip/knorm/ll2/tome/window, not only RAG) — E1/E5 data, pending compile.
- ▶ **tie-time advantage quantification** (compression ratio / prefill / query-agnostic reuse / linear) when IMP ≈ RAG.
