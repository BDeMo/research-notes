# IMP vs RAG — case-level diff & the truncation bug (2026-07-13)

Goal: understand *case-by-case* why IMP trails RAG, find naive bugs, verify prior results.

## The smoking gun (F44): IMP was truncated before selecting; RAG was not
Code paths in `run_baseline.py`:
- `_imp`: `rt.tok(ctx, truncation=True, max_length=MAXCTX)` → **only the first MAXCTX (16k) tokens exist**, then score/select within them.
- `_rag`: `rt.tok(ctx)` (no truncation) → BM25 over the **whole** document, select to budget, `[:MAXCTX]` *after*.

Case diagnostic (30 items/bench, tokenizer-only, no model):
| bench | MAXCTX | median raw ctx | frac>MAXCTX | RAG kept-content from >MAXCTX | RAG top-passage >MAXCTX |
|---|--:|--:|--:|--:|--:|
| **∞Bench-choice** | 16k | **131,737** | 1.00 | **0.87** | **0.90** |
| lb_hotpotqa | 16k | 16,017 | 0.40 | 0.01 | 0.00 |
| hotpot_qa / squad | 4k | short | 0 | 0 | 0 |

⇒ On ∞Bench, **87% of what RAG uses lives past token 16k — the region IMP is blind to.** The −19 gap is mostly "RAG reads 131k, IMP reads the first 16k (12%)", not IMP selecting badly. (∞Bench gold is an MC letter → substring-recall is uninformative there; the beyond-MAXCTX fraction is the right measure.)

## Fix & validity
- **Fix (F44):** the chunk/BM25/hier family (and `auto`→chunk) needs **no forward pass**, so it now retrieves over the **full untruncated doc** (`GCM_IMP_FULLDOC=1`, default on). span/qfree/surprisal modes still truncate (they run an O(L²) forward — inherent).
- **Sanity (squad, docs ≤4k):** chunk-FD **69.0** ≈ old chunk 66.3 (> full 65.9) → fulldoc does not hurt short docs.
- **Prior-results validity:** UNAFFECTED for every bench with ctx ≤ MAXCTX (squad, hotpot, and LongBench where RAG-beyond ≈ 0.01). **Only ultra-long ∞Bench was confounded** → re-running `rf2_*` (chunk-FD, auto-FD, and a truncated control to reproduce the old ~51).

## Fix result on ∞Bench-choice (N=100) — the gap FLIPS
| config | ∞Bench-choice | note |
|---|--:|---|
| chunk **truncated** (old, control) | 53.0 | reproduces old rf_q8 chunk 50.7 → harness consistent |
| **chunk FULL-DOC (fix)** | **76.0** | **beats full 57 and RAG 64.2** |
| chunk FULL-DOC keep0.25 | 74.0 | robust to budget |
| **RAG (same N=100)** | 69.0 | apples-to-apples baseline |
| auto(mc) full-doc | 52.0 | mc router mis-routes ∞Bench (has options→span) → misses chunk-FD |
| full / no_ctx | 57 / 44 | reference |

⇒ **The −19 "loss" was entirely the truncation artifact. CONFIRMED same-N (N=100, same items): IMP-chunk-FD 76.0 > RAG 69.0 > full 57** — IMP now *beats* RAG on ∞Bench by +7. (control truncated-chunk 53.0 reproduces old rf_q8 50.7.)

**Router bug exposed (F41 refinement needed):** the `mc` router sends ∞Bench to `span` because it *has* options — but ∞Bench is "MC whose answer sits in a retrievable passage", which needs **chunk-full-doc**. The right axis is not MC-vs-extractive but **"answer-in-a-retrievable-passage" vs "distributed-reasoning/context-hurts"**. Fix candidate: route to chunk whenever a strong BM25 passage exists (peakiness) OR the doc ≫ MAXCTX, else span for the genuine context-hurts literary-MC (QuALITY).

## Where IMP genuinely differs from RAG (not bugs — mechanism)
1. **Extractive QA:** once both see the same doc, IMP-chunk BM25 ≈ RAG (squad 69 ≈ 67) — same lexical retrieval.
2. **Hard-MC where context hurts (quality_hard):** IMP-span **12.8 > RAG 9.6** — RAG retrieves *noise* (no lexical anchor between options & context); token-importance keeps salient reasoning tokens. IMP's real edge.
3. **Multi-hop (hotpot 4k, no truncation):** RAG 55.8 > IMP 45.6 — genuine: passage-level retrieval gathers distributed evidence better than 256-chunk/token importance. A true (small) method gap, not a bug.
