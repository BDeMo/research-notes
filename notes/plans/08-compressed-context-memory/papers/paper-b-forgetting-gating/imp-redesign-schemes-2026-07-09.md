# IMP redesign — borrowing from RAG (schemes + 72h experiment plan, 2026-07-09)

**Why redesign.** F30/F32 showed vanilla IMP (v2.1.0, {qrel,surp}) is dominated by RAG on QA; F33 showed adding an **IDF-lexical** signal (v2.1.1, `all`) closes most of the multi-hop gap and keeps retrieval — proving *the missing ingredient was RAG's discriminative lexical matching*. So: **borrow RAG's mechanics (chunk granularity + proper BM25 ranking + retrieve-to-budget) into the IMP framework, without losing IMP's base properties.**

**Base properties to KEEP (non-negotiable):** training-free (Mode A) · input-side (no KV cache needed → works on **linear/GDN** bases where kvzip/knorm can't) · keeps important content **verbatim** (no merge, protects the needle) · reading-order preserved · budget-controlled (keep-fraction).

**What RAG has that IMP lacked:** (a) **coherent passage granularity** (whole chunks, not scattered tokens); (b) **proper BM25 ranking** (IDF + tf saturation + length-norm), not just embedding-cosine; (c) **retrieve-to-budget** selection.

## Schemes (`GCM_IMP_MODE`)
| id | mode | scoring | selection unit | borrows from RAG | keeps IMP property |
|---|---|---|---|---|---|
| **S1** | `span` (v2.1.1 baseline) | z(qrel)+z(surp)+z(idf-lex), token | top-p **32-tok spans** | idf-lex | verbatim spans, non-lexical needle via surp |
| **S2** | `chunk` | z(BM25)+z(mean surp)+z(mean qrel), per chunk | top **256-tok chunks** to budget | chunk granularity + BM25 + retrieve-to-budget | verbatim, surp rescues non-lexical, works on linear |
| **S3** | `hier` | BM25 top chunks (2×budget) → span-importance refine | 2-stage: chunk then span | retrieve stage | fine verbatim refinement keeps only needle spans |
| **S4** | `bm25span` | proper **BM25(q, span)** + z(surp) | top-p **32-tok spans** | exact BM25 ranker | span verbatim + surp for non-lexical |
| **S5** | `qfree` | z(surp) only (**no query**) | top-p spans | — (anti-RAG niche) | **query-agnostic**: compress once, reuse for many queries (RAG must re-retrieve) |

**Hypotheses:**
- S2/S3 (chunk + BM25) should **match RAG on extractive/multi-hop** (squad, hotpot) — recovering the coherence+BM25 RAG has — while S4 keeps finer granularity.
- All of S1–S4 keep **retrieval** (ruler) where ToMe/window collapse and RAG is also fine — the unified-router story.
- S5 targets the **only axis RAG can't win**: amortized multi-query (compress once → K queries); if S5 ≥ RAG amortized, that's a genuine method contribution beyond RAG.

## Method details (each scheme, precisely) — all are OURS (IMP family)

Shared setup for every scheme: context tokens `c[1..L]` (truncated to `MAXCTX`), query `q`, keep-fraction `p`, budget `keep=max(8,⌊pL⌋)`, frozen base `F` (no weights touched). Per-token signals computed once (as in v2.1.1): `qrel(t)=cos(e(c_t),ē_q)` (embedding-only, prefill-free), `surp(t)=−log p_F(c_t|c_{<t})` (one forward), `lex(t)=𝟙[c_t∈q]·log((L+1)/(tf(c_t)+0.5))` (IDF-weighted query match). `z(·)` = z-score. Output = `embed(c[S])`, `S` = kept indices in reading order. Env: `GCM_BASELINE=imp GCM_IMP_MODE=<scheme> GCM_LL_RATE=p GCM_IMP_SPAN=32 GCM_IMP_CHUNK=256 GCM_IMP_SIGNAL=all`.

### S1 · `span` (IMP-v2.1.1 baseline — ours)
- **Score:** token `s(t)=z(qrel)+z(surp)+z(lex)`.
- **Select:** split into contiguous 32-tok spans; span score = **max** token `s` in span; keep top `⌊keep/32⌋` spans **whole**, reading order.
- **Borrows from RAG:** the IDF-lexical term (F33). **Keeps IMP:** fine 32-tok verbatim spans; surprisal rescues non-lexical needles.

### S2 · `chunk` (RAG-granularity — ours)
- **Score (per 256-tok chunk `u`):** `S(u)=z(BM25(q,u))+z(mean_t∈u surp)+z(mean_t∈u qrel)`, where **BM25** = Okapi (k1=1.5, b=0.75), df over chunks.
- **Select:** greedily add highest-scoring chunks until token budget `keep` filled; restore reading order; keep chunks **verbatim**.
- **Borrows from RAG:** passage granularity + proper BM25 ranking + retrieve-to-budget (this is essentially RAG's ranker, but fused with surprisal/qrel so non-lexical evidence isn't lost). **Keeps IMP:** input-side, runs on linear (no KV), verbatim.

### S3 · `hier` (retrieve-then-refine — ours)
- **Stage 1 (retrieve):** BM25 top chunks (256-tok) up to **2×budget** tokens — coarse RAG-style recall.
- **Stage 2 (refine):** within the retained tokens, apply S1 span-importance (`z(qrel)+z(surp)+z(lex)`, max-pool 32-tok spans) to cut down to `keep`.
- **Borrows from RAG:** the recall stage (get the right passages first). **Keeps IMP:** fine verbatim refinement drops within-passage redundancy → higher effective density than pure chunk.

### S4 · `bm25span` (BM25 at span granularity — ours)
- **Score (per 32-tok span `u`):** `S(u)=z(BM25(q,u))+z(mean surp)` — RAG's exact lexical ranker but at fine 32-tok spans instead of 256-tok chunks.
- **Select:** greedily add top spans to budget, reading order.
- **Borrows from RAG:** proper BM25 (tf saturation + IDF + length-norm), stronger than S1's token-sum lexical. **Keeps IMP:** fine granularity; surprisal for non-lexical.

### S5 · `qfree` (query-agnostic — ours, the anti-RAG niche)
- **Score:** token `s(t)=z(surp)` **only — no query used.**
- **Select:** top-p 32-tok spans (as S1).
- **Why:** RAG/S1–S4 all need the query at compression time → must **re-compress per query**. `qfree` compresses **once**, reusable for **any** query (KV-cache reuse / multi-query amortization) — the one axis where an importance-router can beat RAG. Expected lower single-query accuracy, but wins on **amortized cost** when a long context serves many queries.
- **Keeps IMP:** verbatim spans, input-side, linear-friendly; the unique deployment story RAG cannot match.

## 72h experiment grid
**Axes:** modes {S1 span, S2 chunk, S3 hier, S4 bm25span, S5 qfree} + references {full, no_ctx, rag, ll2, tome, kvzip, knorm} · benches (full suite incl. hardest) · keep {0.25, 0.5} · bases {Qwen3-8B quad, **Qwen3.5-9B linear** for arch-generality}.
- **Wave-1 (Qwen3-8B, N=200, all benches × 5 modes × keep{0.25,0.5}):** the head-to-head redesign comparison vs RAG.
- **Wave-2 (best mode, FULL N, headline benches):** paper-grade numbers for the winning scheme; update main table imp column.
- **Wave-3 (Qwen3.5-9B linear, 5 modes × core benches):** confirm redesign keeps the architecture-generality edge (RAG works on linear too, but does the router still win retrieval?).
- **Wave-4 (S5 query-agnostic amortized): compress once → answer all queries of a doc** (QuALITY multi-Q per article); IMP-qfree vs RAG-per-query. The decisive "beyond RAG" test.
Launchers/logs: `grid_impredesign/`. This fills ~72h across all idle GPUs (free1/free3/d1530/d1525/free2).

**Decision rule (honest):** if a redesigned scheme **≥ RAG on QA AND keeps retrieval AND runs on linear** → IMP has a real method contribution (unified, training-free, architecture-agnostic). If none beats RAG anywhere RAG works → keep the diagnosis as the contribution (F31) and report IMP as the training-free/linear reference.
