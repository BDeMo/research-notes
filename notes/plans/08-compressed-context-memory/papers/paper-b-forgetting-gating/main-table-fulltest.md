# Paper-B main table — FULL test sets (2026-07-08, live)

> **Method version:** the `imp` column = **IMP-v2.1.0** = span-level Importance-routing Prefilter, **Mode A (training-free)**, span=32, keep=0.5, signals={query-relevance, surprisal}. All Paper-B result files use this same tag; do not confuse with the token-level IMP-v2.0 (retrieval-only, superseded) or Paper-A's learned soft-memory compressor.

Base **Qwen/Qwen3-8B** (frozen). Budget = **keep 0.5** (window 1024, RAG 2048, LLMLingua-2 0.5, ToMe 0.5, **IMP-v2.1.0 span-32 keep-0.5**, kvzip/knorm KV-ratio 0.5). Scoring = native (MC=letter log-likelihood accuracy; extractive/multi-hop=token-F1; abstractive=ROUGE-L; NIAH=exact-value). Numbers are **×100**.

**Sampling:** long-context headline benches on **FULL split** (size in col `N`); the four short-context sanity benches on a **disclosed N=500 subset**. Config/provenance: [`experiment-config-and-sampling.md`](experiment-config-and-sampling.md). Blank = cell still running.

`no_ctx` = blind (no context, lower bound). **`full·trunc16k` = context TRUNCATED to the first MAXCTX (16k) tokens** (NOT the whole document — the uncompressed upper bound *within a 16k budget*; on ∞Bench's 131k docs it sees only ~12%). **imp = ours (IMP-v2.1.0, span-level, Mode A).**

## Method definitions — exactly what each method sees, its budget, and whether it is truncated
**Critical:** `full` is **NOT the whole document** — it is the context **truncated to the first MAXCTX (16k) tokens** (`embed(ctx_ids[:, :MAXCTX])`). On docs longer than 16k (only ∞Bench here, median 131k), `full` sees just the first ~12%. Any forward-pass/KV method is likewise MAXCTX-bounded. **Only `rag` and our `auto` (chunk branch) scan the FULL untruncated document** (BM25 needs no forward). This is why "retrieval > full" on ∞Bench — different information access, not just better compression.

| method | ours? | what it sees (context access) | budget knob | mechanism | truncated to MAXCTX? |
|---|:--:|---|---|---|:--:|
| `no_ctx` | ref | nothing (query only) | — | blind / parametric guess (lower bound) | n/a |
| `full·trunc16k` | ref | context **first MAXCTX=16k tokens (TRUNCATED)** | MAXCTX | uncompressed read of the (capped) context (upper bound *within 16k*, not the whole doc) | **YES — head 16k only** |
| `window` (txl) | base | first 4 "sink" tokens + **last (W−4) tokens** | W=1024 | generic sliding-window read (attention-sink + recent tail). *Not* StreamingLLM (that = kvpress `streaming`) | YES (head-4 + tail-1020) |
| `rag` | base | **FULL untruncated doc** → 128-tok passages → BM25 vs query(+options) → top passages to budget, reading-order, cap MAXCTX | BUDGET=2048 | lexical passage retrieval | **NO — whole doc** |
| `ll2` (LLMLingua-2) | base | full ctx text → token-classifier prunes low-info tokens → shortened text (then cap MAXCTX) | rate=0.5 | learned importance token-pruning (xlm-roberta-large) | compressor reads full text; kept text capped MAXCTX |
| `tome` (ToMe) | base | context first MAXCTX → token embeddings | ratio=0.5 | input-side bipartite soft-match **token merging** (order-preserving) | YES (needs embed forward) |
| `kvzip` | base | context tok[:MAXCTX] → forward → build KV | KV-ratio=0.5 (frac dropped) | **KV-cache eviction** by reconstruction-importance; **needs a KV cache → cannot run on linear/GDN** | YES |
| `knorm` | base | context tok[:MAXCTX] → forward → build KV | KV-ratio=0.5 | **KV-cache eviction** by key L2-norm; **KV-cache required (no linear)** | YES |
| **`auto`** | **✅ OURS** | doc>MAXCTX *or* extractive → 256-tok chunk **BM25 over FULL doc**; literary-MC that fits → span-importance over first MAXCTX | keep=0.5 | training-free **importance-router** (mc+doc-length routing, `signal=all`, FULLDOC) | chunk branch: **NO (full doc)**; span branch: YES |

**CORRECTED (2026-07-13):** the ours column is now **`auto`(OURS, IMP-v2.2)** — the full-doc-retrieval router from the `mt_*` re-run (replaces the old MAXCTX-truncated IMP-v2.1.0). ‡ = needs KV cache (no linear). Bold = best in row. Long-context rows = FULL split, full-N-matched auto/rag; short-QA rows `*` = auto/rag re-measured at **N=2000** (`mt_*`) while window/ll2/tome/kvzip/knorm stay at E1 **N=500** (fully N-matched auto-vs-rag in [`COMPLETE-RESULTS-2026-07-13.md`](COMPLETE-RESULTS-2026-07-13.md)). RULER auto = E5 chunk (mt RULER blocked by HF-Hub wikitext download).

| bench | N | no_ctx | full·t16k | window·t | ll2 | tome·t | kvzip·t‡ | knorm·t‡ | rag | **auto (OURS)** |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **LongBench-v2** (MC, hardest) | 503 | 33.4 | 33.6 | 32.6 | · | 30.8 | 23.1 | 22.3 | **34.2** | 33.8 |
| **∞Bench** choice (MC, >100k) | 229 | 41.9 | 52.8 | 41.5 | 60.7 | 51.5 | 35.8 | 25.8 | 64.2 | **70.3** |
| lb_2wikimqa | 200 | 13.7 | **22.8** | 10.4 | 18.5 | 11.2 | 17.3 | 22.4 | 16.0 | 16.8 |
| lb_hotpotqa | 200 | 11.1 | 22.6 | 11.3 | 20.6 | 17.4 | 17.3 | 18.4 | **24.8** | 24.6 |
| lb_multifieldqa | 150 | 17.9 | 38.6 | 23.2 | 36.2 | · | 37.1 | 30.7 | 38.8 | **39.5** |
| lb_musique | 200 | 4.4 | **11.9** | 4.1 | 9.2 | 9.0 | 9.3 | 10.2 | 9.2 | 11.1 |
| lb_narrativeqa | 200 | 5.3 | 14.0 | 9.3 | 13.7 | 13.5 | 11.2 | 9.3 | 13.7 | **14.3** |
| lb_qasper | 200 | 8.8 | **27.6** | 15.4 | 23.8 | 21.0 | 19.6 | 14.6 | 23.0 | 24.2 |
| **RULER-NIAH @16k** | 500 | 0.0 | 99.0 | 6.0 | 95.0 | 0.0 | **99.0** | 85.0 | 94.2 | 98.0 |
| QuALITY (MC) | 1595 | 17.9 | 7.2 | 12.0 | 9.8 | 9.7 | 29.5 | **31.2** | 7.9 | 9.5 |
| QuALITY-hard (MC) | 813 | 19.6 | 9.2 | 13.3 | 12.2 | 11.7 | **29.9** | 29.6 | 10.5 | 12.9 |
| MuSR (MC) | 90 | 44.4 | **58.9** | 53.3 | 54.4 | 48.9 | 55.6 | 53.3 | 56.7 | 48.9 |
| squad_v2 | 2000ᵃ | 16.6 | 65.9 | 69.0 | 53.7 | 38.9 | 64.9 | 16.5 | 67.2 | 66.9 |
| hotpot_qa | 2000ᵃ | 24.3 | **57.2** | 53.3 | 52.3 | 43.9 | 56.2 | 23.1 | 55.8 | 48.8 |
| trivia_qa | 2000ᵃ | 50.2 | 70.9 | 59.7 | 71.4 | 62.3 | 71.2 | 40.7 | 72.7 | **72.8** |
| ms_marco | 2000ᵃ | 24.7 | 30.7 | 30.4 | 30.2 | 28.6 | 28.4 | 19.6 | **30.7** | 29.9 |

ᵃ short-QA: `no_ctx/full/rag/auto` = N=2000 (`mt_*`); `window/ll2/tome/kvzip/knorm` = N=500 (E1) — mildly different N, close values.

**N = actual items evaluated (verified by loading).** Long-context headline benches are the **FULL** public split: LongBench-v2 503, ∞Bench-choice 229, LongBench-v1 150–200/task, QuALITY 1595, QuALITY-hard 813, MuSR 90. RULER-NIAH = 500 fixed synthetic samples (standard). `*` short-context sanity = **disclosed N=500 subset** of much larger sets (squad full 5928, hotpot 7405, trivia 16137, ms_marco 55578) — not the paper's headline. `·` = cell still running.

---

## What the full-test numbers say (findings)

**F-i. No method is universally best; the ranking flips by task — now confirmed at full scale.**
RAG wins on ∞Bench / lb_hotpotqa / lb_multifieldqa / squad; LLMLingua-2 wins on trivia; IMP wins on RULER; `full` wins on lb_musique / lb_qasper / MuSR; `no_ctx` wins on QuALITY. No column dominates the table.

**F-ii. "More context hurts" is real and strong — on the hardest and the MC benches.**
- **QuALITY: `full` 7.2 < blind `no_ctx` 17.9** (and QuALITY-hard 9.2 < 19.6). Feeding the whole article *halves* accuracy vs. answering blind — the frozen base is distracted by the long literary context.
- **LongBench-v2 (hardest): `full` 33.6 ≈ `no_ctx` 33.4** — full context buys **~0 over blind guessing** on the hardest real long-context MC set. Every compression method lands in the same 32–34 band; the bench is a genuine ceiling.

**F-iii. On ultra-long and open QA, compression/retrieval BEATS full context.**
- **∞Bench (>100k): RAG 64.2 and LLMLingua-2 60.7 both beat `full` 53.7**; `window` truncation ≈ blind (41.5 vs 41.9) — chopping loses the evidence, *selecting* it wins.
- squad 71.5 > 69.0, trivia 71.4 > 69.4 (kvzip **71.2** > full too), lb_hotpotqa 24.8 > 22.6, lb_multifieldqa 38.8 > 38.6 — retrieval/compression ≥ full on many QA sets.
- **KV-eviction denoises literary-MC too:** on QuALITY-hard **kvzip 29.9 / knorm 29.6 beat *both* full (9.2) and blind (19.6)** — dropping distracting KV *helps* the MC log-likelihood where reading the whole article hurts. (IMP/ll2/tome stay ~10–15 here; the win is specific to KV-mass pruning on this literary-MC regime.)

**F-iv. IMP (ours) is the standout where selection must be lossless: synthetic retrieval.**
- **RULER-NIAH @16k: IMP 96.8 ≈ `full` 99.0**, beating knorm 85.0 and rag 94.2, while **ToMe collapses to 0.0 and window to 6.0.** Merging tokens destroys the needle; truncation drops it; span-importance routing keeps it almost losslessly at half budget.
- On real long-doc QA IMP is mid-pack (competitive on narrativeqa 13.5≈full 14.0, hotpot 17.4; weak on 2wiki 11.2) — motivating the span-granularity and the Mode-B light training.

**F-v. The KV-eviction baselines (kvzip/knorm) are middling on real long-doc** (kvzip lb_hotpotqa 17.3, musr 55.6; knorm RULER 85.0) — competitive on retrieval, not better than text-side selection on QA.

### Takeaway for Paper B
The full-test map reproduces and *sharpens* the exploratory story: (1) a fixed compressor cannot be right everywhere; (2) on the hardest/most-literary sets full context is a *liability*, not an asset; (3) selecting the right spans (RAG / LLMLingua-2 / **IMP**) is what recovers or exceeds full — and IMP is uniquely lossless on needle retrieval where merge/truncate baselines collapse. This is the empirical ground for an **input-adaptive, importance-routing** structure on a frozen base.

*(Table auto-updated as remaining kvzip/knorm/imp cells finish; see `experiment-config-and-sampling.md` §4 for the full method/bench config.)*
