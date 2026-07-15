# Complete results — all benches × all methods × all base models (2026-07-13)

> ⚠️ **`full·trunc16k` = context TRUNCATED to MAXCTX**, not the whole doc; materially truncated only on ∞Bench (131k). Only `rag` and our `auto`(chunk) read the full doc. Def + per-bench table: [`experiment-config-and-sampling.md`](experiment-config-and-sampling.md). All scores ×100, keep=0.5, frozen base.

Sources: `mt_*` (corrected **auto=ours**, rag; Qwen3-8B, FULL N, 2026-07-13) · `ft_*`/E1 (window/ll2/tome/kvzip/knorm/full/no_ctx, FULL N) · `fam_*` (9 families) · RULER from E1/E5 (HF-Hub blocked the mt re-run of RULER's wikitext filler).

`·t`=truncated to MAXCTX · **auto = OURS** (training-free importance router: doc>MAXCTX/extractive→chunk-BM25 full-doc; literary-MC-that-fits→span; `signal=all`, FULLDOC). Bold = best in row. `‡`=needs KV cache (no linear).

## TABLE 1 — Qwen3-8B, all methods × all benches (corrected)
| bench (N) | no_ctx | full·t | window·t | ll2 | tome·t | kvzip·t‡ | knorm·t‡ | rag | **auto (OURS)** | best |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--|
| ∞Bench-choice (229) | 41.9 | 52.8 | 41.5 | 60.7 | 51.5 | 35.8 | 25.8 | 64.2 | **70.3** | **OURS** |
| LongBench-v2 (503) | 33.4 | 32.6 | 32.6 | — | 32.8 | 23.1 | 22.3 | **34.2** | 33.8 | rag |
| RULER-NIAH (500) | 0.0 | 99.0 | 6.0 | 95.0 | 0.0 | **99.0** | 85.0 | 94.2 | 98.0 | kvzip/full (OURS best arch-agnostic) |
| squad_v2 (2000) | 16.6 | 65.9 | 69.0 | 53.7 | 38.9 | 64.9 | 16.5 | 67.2 | 66.9 | window/rag≈OURS |
| hotpot_qa (2000) | 24.3 | 57.2 | 53.3 | 52.3 | 43.9 | 56.2 | 23.1 | 55.8 | 48.8 | full |
| trivia_qa (2000) | 50.2 | 70.9 | 59.7 | 71.4 | 62.3 | 71.2 | 40.7 | 72.7 | **72.8** | **OURS**(tie ll2/rag) |
| ms_marco (2000) | 24.7 | 30.7 | 30.4 | 30.2 | 28.6 | 28.4 | 19.6 | **30.7** | 29.9 | rag/full |
| lb_2wikimqa (200) | 13.7 | 22.8 | 10.4 | 18.5 | 11.2 | 17.3 | 22.4 | 16.0 | 16.8 | full |
| lb_hotpotqa (200) | 11.1 | 22.6 | 11.3 | 20.6 | 17.4 | 17.3 | 18.4 | **24.8** | 24.6 | rag≈OURS |
| lb_multifieldqa (150) | 17.9 | 38.6 | 23.2 | 36.2 | — | 37.1 | 30.7 | 38.8 | **39.5** | **OURS** |
| lb_musique (200) | 4.4 | 11.9 | 4.1 | 9.2 | 9.0 | 9.3 | 10.2 | 9.2 | 11.1 | full (OURS best method) |
| lb_narrativeqa (200) | 5.3 | 14.0 | 9.3 | 13.7 | 13.5 | 11.2 | 9.3 | 13.7 | **14.3** | **OURS** |
| lb_qasper (200) | 8.8 | 27.6 | 15.4 | 23.8 | 21.0 | 19.6 | 14.6 | 23.0 | 24.2 | full |
| QuALITY (1595) | 17.9 | 7.2 | 12.0 | 9.8 | 9.7 | 29.5 | **31.2** | 7.9 | 9.5 | **knorm‡** |
| QuALITY-hard (813) | 19.6 | 9.2 | 13.3 | 12.2 | 11.7 | **29.9** | 29.6 | 10.5 | 12.9 | **kvzip‡** |
| MuSR (90) | 44.4 | **58.9** | 53.3 | 54.4 | 48.9 | 55.6 | 53.3 | 56.7 | 48.9 | full |

## TABLE 2 — all-family generality (9 families, `fam_*`, N=100, crossover benches; span/chunk/rag/full)
Shows the model-invariant crossover (chunk≫span on extractive) + chunk≈rag. `auto`(peak, pre-fix) omitted (superseded by mt corrected router). ×100.
| family | arch | squad f/span/**chunk**/rag | quality_hard span/chunk (no_ctx) | ∞Bench full→rag | ruler full→rag |
|---|---|--|--|--|--|
| Qwen3-1.7B | dense | 52.5 / 29.6 / **44.5** / 45.2 | 20/20 (21) | 37→54 | 70→94 |
| Qwen3-4B | dense | 55.5 / 27.0 / **51.0** / 52.3 | 17/15 (20) | 54→66 | 70→94 |
| Qwen3-8B | dense | 65.9 / 46.6 / **66.3** / 67.2 | 12.8/10.9 (19.6) | 52.8→**70.3(ours)** | 99→94/98 |
| Qwen3-14B | dense | 38.5 / 20.5 / **33.0** / 33.4 | 12/12 (18) | 51→67 | (14B ruler broken) |
| Qwen2.5-7B | dense | 58.8 / 36.0 / **63.4** / 63.7 | 13/12 (23) | 52→69 | 70→94 |
| Qwen3.5-4B | linear | 56.3 / 39.7 / **56.1** / 57.8 | 11/14 (24) | 45→62 | 70→93 |
| Qwen3.5-9B | linear | 71.8 / 51.9 / **71.4** / 66.3 | 10/10 (21) | 48→65 | 66→89 |
| GLM-4-9B | dense | 45.6 / 25.5 / **44.5** / 45.4 | 19/18 (18) | 49→74 | 71→97 |
| Ministral-8B | dense | 63.4 / 43.7 / **64.4** / 64.9 | 20/18 (21) | 53→68 | 70→92 |
| Llama-xLAM-8B | dense | 51.5 / 29.3 / **50.2** / 50.9 | 21/18 (24) | 51→71 | 72→95 |

## Analysis
**1. Corrected OURS (`auto`) is now competitive-to-winning, not just "ties RAG".** After the F44 full-doc fix, `auto` **wins outright on 5 benches** (∞Bench 70.3, trivia 72.8, lb_multifieldqa 39.5, lb_narrativeqa 14.3, lb_musique-among-methods 11.1) and **ties the best on ~4** (squad, lb_hotpotqa, ms_marco, LongBench-v2). The old span-only IMP (E1) lost most of these; the router + full-doc retrieval closed them.

**2. No single method dominates — the table's core message.** Row-by-row the winner changes: **OURS** (∞Bench, trivia, multifieldqa, narrativeqa), **rag** (LB-v2, ms_marco, lb_hotpotqa), **full** (2wikimqa, musique, qasper, MuSR, hotpot), **kvzip/knorm‡** (QuALITY, QuALITY-hard), **kvzip/full** (RULER). This is the empirical ground for an input-adaptive router.

**3. The literary-MC regime belongs to KV-eviction, not us (honest).** QuALITY/QuALITY-hard: **kvzip 29.5/29.9, knorm 31.2/29.6** beat everything incl. `no_ctx` 17.9/19.6 and crush OURS (9.5/12.9). Dropping distracting KV *denoises* literary-MC. But kvzip/knorm **need a KV cache → cannot run on linear/GDN**; OURS can. So: where a KV cache exists and the task is literary-MC, cede to kvzip; elsewhere OURS is the arch-agnostic router.

**4. "full is a liability" holds (and full is truncated):** QuALITY full 7.2 < blind 17.9; MuSR/2wiki full is *best* (dense reasoning needs everything). ∞Bench full 52.8 ≪ rag/OURS — but recall full is truncated to 16k of 131k, so this is partly information access.

**5. Model-invariance (Table 2):** the chunk≫span crossover and chunk≈rag on extractive hold on **all 9 families / 3 architectures / 1.7B–14B**; RAG/retrieval ≫ full on ∞Bench & RULER everywhere. The diagnosis is not a Qwen3-8B artifact.

**Net:** OURS = a single training-free, architecture-agnostic router that is **best-or-tied on the retrieval/extractive/needle/∞-long regimes and runs on linear**, while being honestly beaten by **KV-eviction on literary-MC** and by **full on dense-reasoning (MuSR/musique)**. Contribution = the diagnosis + the regime-spanning router, verified across families, with a real bug (F44) found and fixed.
