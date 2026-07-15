# Method vNext — archive of current best + Occam simplification (2026-07-13)

The current-best method (`auto`, wins the most benchmarks among compression methods) is archived here as the vNext **starting point**, followed by the Occam-razor analysis and the next-version design.

---

## PART A — ARCHIVE: current best method `auto` (IMP v2.2)
**Selection basis:** among compression *methods* (excluding the `full`/`no_ctx` references), `auto` wins the most benchmarks on Qwen3-8B (∞Bench 70.3, trivia 72.8, lb_multifieldqa 39.5, lb_narrativeqa 14.3, lb_musique-among-methods) and ties best on ~4 more; verified model-invariant ([`COMPLETE-RESULTS-2026-07-13.md`](COMPLETE-RESULTS-2026-07-13.md)).

### A.1 One-line
A **training-free, architecture-agnostic, input-side prefilter** on a **frozen** base: score context by cheap signals, keep the top spans/chunks **verbatim** to a budget, drop the rest; **route** the granularity per input.

### A.2 Interface
`compress(context, query, keep) -> kept_token_ids` (then `frozen_base(embed([kept ; query]))`). No weights trained, no base modification. Inference: prefill on `keep·L` tokens instead of `L`.

### A.3 Signals (per token, O(L), forward-free except surprisal)
- **lex** — IDF-weighted query-term match (BM25 core). *forward-free.*
- **qdot** — cosine(token input-embedding, mean query embedding). *forward-free.*
- **surp** — token surprisal (needs ONE base forward over the kept-window). *needs forward.*
- `signal=all` = z(qdot)+z(surp)+z(lex) (used by span branch).

### A.4 Routing (the `auto` rule, as shipped)
```
if doc_len > MAXCTX  OR  not has_options(query):   # too long to forward, or extractive/retrieval
      -> CHUNK over the FULL doc (BM25 on 256-tok chunks, no forward)   # FULLDOC
else (fits AND multiple-choice/literary):
      -> SPAN (token-importance signal=all, top-p 32-tok spans, over first MAXCTX)
```
- **chunk branch** (`_fulldoc_bm25`): 256-tok chunks over the untruncated doc → Okapi-BM25 vs query(+options) → greedy keep to `keep·min(L,MAXCTX)` → reading order. **No forward, scales to 131k, runs on linear.**
- **span branch**: forward for surprisal over first MAXCTX → z(qdot+surp+lex) → top-p 32-tok spans.

### A.5 Cost
- chunk branch: O(L) BM25, no model forward for selection; then base prefill on `keep·MAXCTX`.
- span branch: one O(MAXCTX²) forward for surprisal + base prefill.
- vs `full`: prefill cost ×keep; vs RAG: same order (both BM25).

### A.6 Exact code path
`run_baseline.py::_imp` (+`_fulldoc_bm25`), `GCM_IMP_MODE=auto GCM_IMP_AUTO_ROUTER=mc GCM_IMP_FULLDOC=1 GCM_IMP_SIGNAL=all GCM_IMP_SPAN=32 GCM_IMP_CHUNK=256 GCM_LL_RATE=0.5`. Snapshot → `configs/IMP-v2.2-auto.code.py` (to archive).

### A.7 Where it wins / ties / loses (honest)
- **Wins:** ∞Bench (70.3>rag 64.2), trivia (72.8), multifieldqa (39.5), narrativeqa (14.3), musique-among-methods; best arch-agnostic on RULER (98).
- **Ties best:** squad, lb_hotpotqa, ms_marco, LongBench-v2.
- **Loses:** literary-MC (kvzip/knorm ~30 ≫ auto ~10-13, but they need KV → no linear); dense-reasoning (full best: 2wiki/musique/MuSR/qasper/hotpot).

---

## PART B — Occam analysis: what ACTUALLY drives the wins
Strip the method to its load-bearing parts:

1. **The workhorse is the CHUNK branch = full-doc BM25 lexical retrieval.** Every `auto` win comes from routing to chunk-FD; the chunk branch alone ≈ RAG (F36) and, with 256-tok chunks + keep·MAXCTX budget, sometimes beats RAG (∞Bench, trivia, 2wiki).
2. **The importance signals (surprisal, qdot) add ~nothing to the wins.** F32: no signal×span combo closes QA; F41: the winning router just *selects chunk*, not a clever signal. The span branch only matters in literary-MC — where it still **loses to KV-eviction** and to `no_ctx`.
3. **gold-recall is the single mechanistic lever (F35):** accuracy ≈ P(gold span kept). BM25-chunk maximizes it on lexical tasks; nothing text-side maximizes it on no-lexical literary-MC (that regime is a *base-capability ceiling*, F6 — full<blind).
4. **Two regimes are simply not ours:** literary-MC → KV-eviction (kvzip); dense-reasoning → keep everything (`full`).

**Occam conclusion:** the baroque `auto` (6 modes × 3 signals × routers × flags) is empirically **≈ its one chunk branch**. The extra machinery is complexity without verified payoff.

---

## PART C — vNext (v3.0) design: the simplest thing that keeps the wins
**Core insight (the paper's one sentence):**
> On a frozen LLM, long-context accuracy is a *keep-the-gold-span* problem, and the cheapest length- & architecture-agnostic way to keep it is **full-document lexical chunk retrieval** — no forward pass, no training, no KV cache — which matches or beats every text- and KV-side compressor **except** where the base itself can't use long context (literary-MC → KV-eviction) or needs all of it (dense reasoning → no compression).

**v3.0 method = ONE mode, ONE signal, ONE knob:**
- `IMP-lite`: 256-token chunks over the **full untruncated document** → Okapi-BM25 vs query(+options) → keep top chunks to budget `k·MAXCTX` → reading order → frozen base. **That's it.** No span mode, no surprisal, no qdot, no router, no adaptive budget.
- Deployment: plug-and-play (training-free), any architecture (dense/linear), any length (no forward for selection).

**Why this is the right Occam cut:** it drops every component that Part B shows is non-load-bearing, keeps the one that wins, and makes the method *trivially* describable and reproducible.

### C.1 Validation experiments (must-run before committing v3.0)
1. **Ablation-to-lite:** `IMP-lite` (pure BM25 chunk-FD) vs full `auto` across the 16-bench table. **Hypothesis:** lite ≈ auto (within noise) on all wins/ties. *If true, Occam wins — ship lite.* Falsifier: if any bench needs the span/surprisal branch, keep exactly that.
2. **The one principled add-on, tested as a single ablation:** replace BM25 with **cheap semantic chunk-retrieval** = mean input-embedding cosine per chunk (still forward-free) — does it close the no-lexical gap (quality_hard, some hotpot) *without* a forward? Keep it only if it helps and stays free.
3. **Chunk-size & budget sweep** (the only two knobs): chunk∈{128,256,512}, k∈{0.1,0.25,0.5}. Find the single default.
4. **Cost/efficiency table:** prefill FLOPs & latency of IMP-lite vs full vs RAG at 16k/131k — the efficiency story (IMP-lite = no selection forward, O(L) BM25).

### C.2 What the paper claims after v3.0
- **Method:** IMP-lite — a one-knob, training-free, architecture-agnostic, full-document lexical compressor that is best-or-tied across retrieval/extractive/needle/ultra-long regimes and the *only* such method that runs on cache-free linear bases.
- **Diagnosis (the real contribution):** the regime map + gold-recall mechanism + model-invariance (F27/F35/F37/F43) + the honest boundaries (literary-MC→KV-eviction, dense-reasoning→full).
- **Scope honesty:** we do **not** beat KV-eviction on literary-MC or `full` on dense reasoning; we own the arch-agnostic long-retrieval regime.

### C.3 Open question to resolve in v3.0
Is IMP-lite meaningfully different from RAG? **Yes, if** (a) 256-chunk+larger-budget consistently beats RAG-128/2048 (∞Bench/trivia/2wiki suggest yes), and/or (b) the semantic add-on (C.1-2) gives a forward-free edge RAG lacks. **If not**, the honest framing is "RAG, correctly configured for compression, is the training-free long-context baseline to beat — and here is the diagnosis of when it wins." Decide empirically from C.1.
