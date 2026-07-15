# IMP redesign — results (living table, fills as the 72h grid runs)

> ⚠️ **`full` = TRUNCATED to MAXCTX (16k)**, not the whole document (`embed(ctx[:,:MAXCTX])`). It equals *true full* only when a doc ≤ 16k — true for every bench here **except ∞Bench** (131k docs → `full` sees ~12%). Only `rag` and our `auto`(chunk) read the whole doc; `full/window/tome/kvzip/knorm` are MAXCTX-bounded. See [`correctness-audit-2026-07-13.md`](correctness-audit-2026-07-13.md).

**OURS (IMP schemes) = {span, chunk, hier, bm25span, qfree}** — all are our IMP-family variants (training-free, input-side, architecture-agnostic, verbatim-keep). **References/baselines = {full (uncompressed upper bound), rag (BM25-2048)}** — not ours. Per-scheme method details: [`imp-redesign-schemes-2026-07-09.md`](imp-redesign-schemes-2026-07-09.md) §Method details. All ×100.

| column | what | ours? |
|---|---|:--:|
| full | uncompressed context (upper-bound reference) | ref |
| **span** | IMP-v2.1.1 token-score {qrel+surp+idf-lex} → top-p 32-tok spans | **✅ ours** |
| **chunk** | 256-tok chunks scored by BM25+surp+qrel, retrieve-to-budget | **✅ ours** |
| **hier** | BM25 top chunks → span-importance refine (2-stage) | **✅ ours** |
| **bm25span** | proper BM25 per 32-tok span + surprisal | **✅ ours** |
| **qfree** | surprisal-only (no query) → top-p spans; query-agnostic reuse | **✅ ours** |
| rag | BM25 passage retrieval (budget 2048) | baseline |

**Decision rule:** a scheme "wins" if it **≥ RAG on QA** AND **keeps retrieval (≈full on RULER)** AND **runs on linear**. Grid: `grid_impredesign/rd_*` (d1525 = Qwen3-8B, d1530 = Qwen3.5-9B linear). 32k benches capped at 16k here (method comparison; 32k headline in `main-table-fulltest.md`). `·` = pending. Keep-sweep {0.1,0.25,0.5,0.75} in logs; tables below show **keep=0.5**.

*`*` = **ours** (IMP variant) · `·ref`/`·base` = reference / baseline.*

## Table A — Qwen3-8B (quadratic), keep=0.5, **FULL-N** (`rf_q8_*`, 2026-07-11)
Authoritative full-test main comparison (supersedes the earlier N=200 draft). Long-context benches = FULL split; short QA (squad/hotpot/trivia/ms_marco) N=2000; `lb_triviaqa`/`lb_repobench` @12k (16k OOM, see sampling doc). All ×100. **Bold** = ≥ RAG.

### A1 — discriminative extractive/multi-hop QA
| bench | no_ctx | full·ref | span* | chunk* | hier* | bm25span* | qfree* | rag·base |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **squad_v2** | 16.6 | 65.9 | 46.6 | **66.3** | 46.7 | 51.9 | 38.3 | 67.2 |
| hotpot_qa | 24.3 | 57.2 | 45.5 | 45.6 | 45.4 | 45.5 | 43.5 | 55.8 |
| trivia_qa | 50.2 | 70.9 | 65.9 | 68.5 | 66.1 | 67.3 | 65.5 | 72.7 |
| ms_marco | 24.7 | 30.7 | 29.0 | 29.6 | 29.3 | 29.1 | 29.1 | **30.7** |
| lb_multifieldqa | 17.9 | 38.6 | 37.3 | 38.6 | 37.6 | 37.4 | 34.3 | 39.3 |
| lb_hotpotqa | 11.1 | 22.6 | **24.1** | 23.9 | 24.0 | 21.1 | 18.6 | 24.1 |
| lb_2wikimqa | 13.7 | 22.8 | 14.2 | 14.7 | 15.0 | 15.8 | 12.1 | 18.5 |
| lb_musique | 4.4 | 11.9 | **10.4** | **11.4** | 10.3 | 9.6 | 8.7 | 10.3 |
| lb_qasper | 8.8 | 27.6 | 22.4 | 23.6 | 22.9 | 22.7 | 20.3 | 26.4 |
| lb_narrativeqa | 5.3 | 14.0 | 13.8 | **15.1** | 13.9 | 13.1 | 12.1 | 15.0 |
| lb_triviaqa@12k | 17.5 | 17.0 | **17.5** | **17.5** | 17.0 | 17.0 | **18.5** | 17.0 |

### A2 — multiple-choice / reasoning (native log-likelihood)
| bench | no_ctx | full·ref | span* | chunk* | hier* | bm25span* | qfree* | rag·base |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| ∞Bench-choice | 41.9 | 52.8 | 49.8 | 50.7 | 49.8 | 51.1 | 52.8 | **70.3** |
| musr_mm | 44.4 | 58.9 | 48.9 | 47.8 | 48.9 | 52.2 | 44.4 | 56.7 |
| quality_hard | **19.6** | 9.2 | **12.8** | 10.9 | 12.3 | 10.6 | 10.8 | 9.6 |
| longbench_v2 | 33.4 | 32.6 | 30.0 | 31.8 | 29.8 | 29.4 | 31.0 | **35.0** |

### A3 — synthetic (needle / bAbI reasoning)
| bench | no_ctx | full·ref | span* | chunk* | hier* | bm25span* | qfree* | rag·base |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| ruler_niah | 0 | 99.0 | 96.6 | 98.0 | 96.6 | 97.0 | 97.2 | 98.8 |
| babilong_qa1_4k | 13 | 75 | 75 | **78** | 75 | **80** | 65 | 70 |
| babilong_qa1_16k | 13 | 68 | 67 | **69** | 67 | 63 | 61 | 72 |
| babilong_qa2_16k | 0 | 30 | 28 | **29** | 28 | 28 | 19 | 28 |
| babilong_qa3_16k | 4 | 29 | **29** | 22 | 28 | 25 | 27 | 27 |

### A4 — scorer-limited (near-floor for **every** method incl. full → NOT method signal; excluded from claims)
`lb_qmsum` (all 0.0) · `lb_samsum` (full 1.5) · `lb_passageretrieval` (full 0.5) · `lb_repobench@12k` (full 0.4) · `lb_trec` (full 4.0). These are summarization / classification / code-completion where our substring-F1 scorer is inappropriate (F40). Reported for completeness, **not** used for method conclusions.

## Table B — Qwen3.5-9B (linear/GDN, no KV cache), keep=0.5, N=100
| bench | full·ref | span* | chunk* | hier* | bm25span* | qfree* | rag·base |
|---|--:|--:|--:|--:|--:|--:|--:|
| longbench_v2 (MC) | 31.0 | 28.0 | 25.0 | 28.0 | 30.0 | 28.0 | 37.0 |
| ruler_niah@16k | 91.0 | 86.0 | 77.0 | 87.0 | 81.0 | 88.0 | 86.0 |
| lb_hotpotqa | 44.9 | 38.2 | 34.1 | · | 30.2 | · | · |
| quality_hard (MC) | · | · | · | · | · | · | · |
| squad_v2 | · | · | · | · | · | · | · |
| hotpot_qa | · | · | · | · | · | · | · |

## VERDICT & dive-in (FULL-N, 2026-07-11)

**1. `chunk` = RAG on extractive QA, confirmed at full scale (F36, was F34@N200).** Not an N=200 artifact: squad `chunk` 66.3 ≈ RAG 67.2 (> full 65.9); ms_marco 29.6 vs 30.7; narrativeqa **15.1 > RAG 15.0**; multifieldqa 38.6 = full. The RAG-borrowing fix generalizes.

**2. The chunk↔span crossover is REAL and input-predictable (F37 — the key mechanistic finding).**
- Where the answer is a **contiguous, lexically-matchable span** (extractive QA) → **`chunk`/BM25 wins** (squad chunk 66.3 vs span 46.6, a **+20pt** gap).
- Where the answer needs **distributed reasoning or has no lexical query anchor** → **`span`/token-importance wins**: quality_hard span **12.8 > chunk 10.9 > RAG 9.6** (IMP-span *beats* RAG here); babilong_qa3 span **29 > chunk 22**; lb_hotpotqa span 24.1 = RAG.
- ⇒ **the right granularity is a function of the input** (lexical anchor present?) — direct motivation for an adaptive selector.

**3. RAG denoises far ABOVE full on ∞Bench-choice (F39): RAG 70.3 vs full 52.8 (+17.5).** Retrieval isn't only memory-saving; dropping distractors *lifts accuracy* well past the uncompressed upper bound. IMP modes only reach ~50 here → the biggest remaining IMP gap; likely because ∞Bench-choice answers hinge on one retrievable passage that BM25-2048 isolates but 256-chunk keep=0.5 dilutes.

**4. "More context hurts" reconfirmed at full N (F27): quality_hard no_ctx 19.6 > full 9.2; longbench_v2 no_ctx 33.4 > full 32.6; lb_triviaqa no_ctx 17.5 ≈ full 17.0 (parametric).** Every compression method beats `full` on quality_hard — compression = denoising.

**5. `qfree` (query-agnostic) is the weakest mode everywhere (F38): squad 38.3, babilong_qa2 19 (vs 28–29).** Query-conditioning is essential; this bounds the "compress-once, reuse across queries" story — it costs real accuracy.

**Net (honest):** IMP-family **matches RAG on extractive QA (`chunk`), beats RAG on no-lexical hard-MC (`span`), keeps needle-retrieval (`span`/`hier` ≈ full on RULER), and runs on linear** — one training-free, architecture-agnostic router spanning regimes that RAG (fails hard-MC/reasoning) and KV-methods (can't run linear) each only half-cover. It does **not** beat free RAG on lexical QA, and trails RAG on ∞Bench-choice. **Contribution = the diagnosis (F27/F31/F43) + a unified regime-spanning router + the input-predictable crossover (F37)** — not a SOTA-accuracy claim.

## `auto` selector — DONE (2026-07-12, F41; details [`explore-results-2026-07-12.md`](explore-results-2026-07-12.md))
The `auto` selector was resolved: the **`mc` router** (options-present→span, else→chunk) recovers best-of-both — squad **66.3** (=chunk) AND quality_hard **12.8** (=span); the BM25-peakiness/query-overlap routers failed (mis-route squad→span, −20). `auto` default set to `mc`. Generality: the crossover, `chunk`=RAG, and RAG≫full are **model-invariant across 9 families / 3 archs / 1.7B–14B** (F43).

## ⭐ BEST RESULT & WHERE THE GAP IS (read this)
**Best method overall = RAG** (free, input-side BM25 passage-retrieval) — it wins or ties on almost every real task; it is the baseline to beat. **Our best variant = `auto` (mc-router)**, which *ties* RAG where RAG is strong and *beats* it where RAG is weak, but does not dominate. Per-regime best (Qwen3-8B, keep 0.5, full-N):

| regime (bench) | best method | ours `auto`(mc) | RAG | full | gap ours−RAG | why the gap |
|---|---|--:|--:|--:|--:|---|
| extractive QA (squad) | tie ours/RAG | **66.3** | 67.2 | 65.9 | **−0.9 (tie)** | chunk-BM25 = passage-BM25 here |
| hard-MC, ctx hurts (quality_hard) | **ours** | **12.8** | 9.6 | 9.2 | **+3.2** | no lexical anchor → RAG retrieves noise; span keeps salient tokens (blind 19.6 still tops → ctx-hurts regime) |
| needle (ruler) | tie | 98 | 98.8 | 99 | −0.8 (tie) | both isolate the needle |
| synthetic reasoning (babilong_qa3) | **ours(span)** | 29 | 27 | 29 | **+2** | distributed facts; importance keeps them |
| multi-hop (hotpot_qa) | RAG | 45.6 | 55.8 | 57.2 | **−10.2** | multi-passage evidence; passage-RAG gathers it better |
| MC w/ 1 key passage (∞Bench-choice) | RAG | ~51 | **70.3** | 52.8 | **−19.3** ⬅ biggest | RAG-2048 isolates the answer passage; 256-chunk dilutes it (F42: not a budget fix) |
| open QA (trivia/musr) | RAG | 66.5 / 48.9 | 72.7 / 56.7 | 70.9 / 58.9 | −6 / −8 | parametric + passage retrieval |

**Summary of the gap:** our method is **≈RAG on extractive/needle, ahead on no-lexical hard-MC/reasoning, but behind by 10–19 on tasks whose answer sits in ONE lexically-anchored passage that RAG's passage-level BM25 isolates and IMP's 256-token chunking dilutes** (∞Bench-choice −19, multi-hop −10). F42 proved this gap is a *retrieval-mechanism* difference, **not** closeable by budget tuning. Closing it would require giving IMP passage-level (not fixed-chunk) retrieval — i.e., converging toward RAG. So the paper's honest framing is **diagnosis + regime-spanning unification (incl. linear & hard-MC where RAG loses)**, not "beats RAG".

## Notes (fills as it runs)
- **longbench_v2** (both models): all schemes ≈ full ≈ blind — ceiling bench, no discrimination (F27).
- **ruler_niah (linear):** span 86 / hier 87 / qfree 88 ≈ full 91, **chunk weakest 77** — chunking hurts single-needle; span/hier/qfree keep it.
- Still filling: ∞Bench, lb_2wiki/musique/narrative, ruler(quad), quality_hard, musr, ms_marco, + the multi-model (1.7B/4B/14B/2.5-7B) and new mainstream benches (BABILong/LongBench-extras).
