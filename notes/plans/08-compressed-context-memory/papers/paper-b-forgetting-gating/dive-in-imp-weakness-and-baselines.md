# Dive-in: where IMP is weak, what the winners have in common, and is the direction worth it?

> ⚠️ **`full` = TRUNCATED to MAXCTX (16k)**, not the whole document. The ∞Bench IMP−RAG gaps quoted here are pre-fix (truncation artifact, F44); with full-doc retrieval IMP-chunk ≥ RAG on ∞Bench. `full` = *true full* only for docs ≤ 16k (all benches except ∞Bench). See [`correctness-audit-2026-07-13.md`](correctness-audit-2026-07-13.md).

**Framing (per the design-phase mandate):** we are **not committed to IMP**. The goal is *facts* — find the best method, learn whether the "importance-routing" direction is worth pursuing, and locate the **performance boundary** — not to force IMP to look good. Evidence: [`main-table-fulltest.md`](main-table-fulltest.md) (full-test, Qwen3-8B, keep 0.5). All ×100.

## 1. The single most important number: **IMP vs RAG** (both training-free, input-side, architecture-agnostic)
| bench | kind | full | IMP | RAG | **IMP−RAG** | best method |
|---|---|--:|--:|--:|--:|--|
| ruler16k | synthetic needle | 99.0 | 96.8 | 94.2 | **+2.6** | kvzip 99 |
| quality | literary-MC | 7.2 | 9.7 | 7.9 | +1.8* | knorm 31 |
| quality_hard | literary-MC | 9.2 | 11.7 | 10.5 | +1.2* | kvzip 30 |
| lb_narrativeqa | abstractive | 14.0 | 13.5 | 13.7 | −0.1 | ll2 13.7 |
| lb_musique | multi-hop | 11.9 | 9.0 | 9.2 | −0.2 | knorm 10.2 |
| longbench_v2 | hard-MC | 33.6 | 32.8 | 34.2 | −1.4 | rag 34.2 |
| lb_qasper | extractive-long | 27.6 | 21.0 | 23.0 | −1.9 | ll2 23.8 |
| ms_marco | open-QA | 30.5 | 28.6 | 30.9 | −2.2 | rag 30.9 |
| lb_2wikimqa | multi-hop | 22.8 | 11.2 | 16.0 | −4.8 | knorm 22.4 |
| lb_hotpotqa | multi-hop | 22.6 | 17.4 | 24.8 | −7.4 | rag 24.8 |
| musr_mm | reasoning-MC | 58.9 | 48.9 | 56.7 | −7.8 | rag 56.7 |
| trivia_qa | open-QA | 69.4 | 62.3 | 70.5 | −8.2 | ll2 71.4 |
| hotpot_qa | multi-hop | 56.2 | 43.9 | 55.0 | −11.2 | kvzip 56.2 |
| infbench_choice | ultra-long-MC | 53.7 | 51.5 | 64.2 | −12.7 | rag 64.2 |
| **squad_v2** | extractive-short | 69.0 | 38.9 | 71.5 | **−32.6** | rag 71.5 |

\* the two "IMP wins" outside RULER are literary-MC where *everything* is near-random and full-hurts — not real wins (the actual winner there is kvzip/knorm at ~30).

**⇒ IMP beats RAG cleanly on exactly ONE benchmark (synthetic RULER, +2.6). On real tasks RAG ≥ IMP everywhere, up to −32.6 on squad.** RAG is training-free, input-side, and architecture-agnostic — the *same* deployment properties we sell for IMP. **This is the boundary: the importance-routing direction has ~0 accuracy headroom over a free retrieval baseline.**

## 2. What do the benches where IMP is WEAK have in common?
| failure cluster | benches | why IMP loses |
|---|---|---|
| **high lexical-overlap extractive** | squad (−32.6 vs RAG) | the question's words *are* in the answer sentence → **BM25 retrieves it near-perfectly**; IMP's semantic token-scoring is coarser and dilutes the budget across the passage |
| **multi-hop / distributed evidence** | hotpot, 2wiki, musique | the answer needs **several evidence spans + their links**; independent per-token importance keeps some hops and **drops others / breaks the chain** |
| **dense reasoning** | musr (−10 vs full) | the whole chain is load-bearing → **any 50% pruning hurts**; `full` wins |
| **passage-retrievable open-QA** | trivia, ms_marco, infbench | a single relevant *passage* answers it → **passage-level retrieval (RAG/LLMLingua) beats span-level importance** |

**Common characteristic:** IMP is a **sparse token-importance selector**. It only helps when the useful signal is **sparse + localized + surrounded by genuine filler** (= synthetic NIAH). Real long-doc QA has **dense / distributed / weakly-relevant** context where (a) you often need coherent *passages* not top-scoring spans, and (b) lexical retrieval pinpoints the answer better than model-internal importance.

## 3. What characterizes the WINNERS?
- **RAG (BM25 passage retrieval)** — modal winner (7 benches). Wins when the answer sits in a **lexically-retrievable passage** (extractive, open-QA, single-field). Operates at **passage** granularity → preserves coherence IMP's span-selection breaks. Fails only on abstractive/global (F9) — where nothing works anyway.
- **kvzip / knorm (KV eviction)** — win **literary-MC** (QuALITY-hard 30 ≫ full 9): dropping distracting **KV mass** *denoises* exactly where full-context hurts (F6). But **cannot run on cache-free linear models** (F10) — their only structural weakness vs IMP.
- **LLMLingua-2 (perplexity compression)** — wins trivia/qasper/narrative: keeps informative tokens, denoises; a learned small-LM scorer beats IMP's raw surprisal.
- **full** — wins multi-hop/reasoning: needs everything.

## 4. Is the direction worth doing? Where is the boundary?
**As a raw-accuracy method: no — the ceiling is RAG's curve, and RAG is free.** IMP does not clear it on any real task. Forcing IMP to "win" on accuracy would be overclaiming.

**IMP's only *structural* properties RAG lacks:**
1. **Query-agnostic** — surprisal-based importance needs **no query at compression time** → compress the context **once**, reuse across **many queries** (RAG must re-retrieve per query). *Untested, and the one axis where IMP could genuinely beat RAG (amortized/multi-query / KV-cache-reuse setting).*
2. **Needle-exactness** — keeps the exact token span; RAG's chunking can split a needle. (Marginal: RULER +2.6.)
3. **Runs on cache-free linear bases** — but **RAG runs there too** (input-side), so this is NOT unique to IMP.

**Honest strategic options (pick based on facts, not attachment to IMP):**
- **(A) Reposition Paper B around the DIAGNOSIS, not IMP.** The novel, defensible contribution is the **fact-base** (no universal method F3/F18/F27; full-hurts F6; compression-denoises F23; architecture/size-invariance F11/F28; keep-curve inversion F29). IMP becomes a *training-free reference point*, not a SOTA claim. **Strongest honest paper.**
- **(B) Pivot the method to query-agnostic reusable compression** — the one setting where IMP can beat RAG. Test: compress once, evaluate M queries; report amortized accuracy/cost. If IMP > RAG there → a real method contribution.
- **(C) Efficiency-first** — if IMP ≈ RAG accuracy at lower prefill/latency, sell cost (needs the tokens-read/latency table; note the surprisal-forward caveat, method doc §7).
- **(D) Drop importance-routing** if (B)/(C) don't beat RAG; fold the effort into the diagnosis paper (A).

**Recommended next experiments (falsifiers, cheap):**
1. **IMP vs RAG, query-agnostic multi-query** on squad/hotpot (compress once → K questions). If IMP doesn't win amortized, (B) is dead.
2. **Add a cost column** (tokens read, prefill FLOPs) to the main table — decide if (C) is real.
3. **RAG⊕IMP hybrid** (BM25 passages → importance-rerank within) — quick check for a cheap accuracy bump.

## 5. Update (2026-07-09) — the two open questions from §4 are now ANSWERED by data
- **IMP design-ceiling (F32):** the signal×span ablation (Qwen3-8B, 15 configs) shows **no combination closes the QA gap** — squad all 11–24 (RAG 71), hotpot all 22–32 (RAG 55). IMP's QA deficit is **structural, not a hyperparameter.** ⇒ "tune IMP to beat RAG" is dead.
- **Diagnosis generality (F31):** across **10 models** (1.7B→14B, Qwen2.5/3/3.5, GLM-4, Ministral, quad+linear), all three signatures reproduce — IMP≈full on retrieval / IMP-weakest-on-QA / full-hurts-on-literary-MC ([`generality-model-matrix.md`](generality-model-matrix.md)). ⇒ the diagnosis is a **property of frozen LLMs + long context**, not a single-model artifact.

**Net:** option **(A) is now the evidence-backed choice — Paper B's contribution is the model-invariant DIAGNOSIS**, with IMP as a training-free reference point (unique only on retrieval + cache-free-linear). Options (B) query-agnostic and (C) efficiency remain the *only* ways IMP could add a method claim; they need the (still-unbuilt) multi-query harness + cost column. Do **not** claim IMP as a SOTA-accuracy compressor.

---
**Bottom line (fact, not spin):** the importance-routing *direction* is **not worth pursuing as an accuracy method** — a free RAG baseline dominates it on every real task and every model tested (only synthetic RULER +2.6), and **no IMP hyperparameter closes the gap (F32)**. The robust, model-invariant **diagnosis (F31) is the contribution**; IMP survives only as a training-free retrieval/linear reference or (pending falsification) a query-agnostic/efficiency play.
