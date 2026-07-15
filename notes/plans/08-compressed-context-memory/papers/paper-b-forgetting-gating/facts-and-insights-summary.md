# Facts & extracted insights — consolidated summary (2026-07-08)

> ⚠️ **`full` = TRUNCATED to MAXCTX (16k)**, not the whole document. `full` = *true full* only for docs ≤ 16k — true for every bench **except ∞Bench** (131k → sees ~12%). ∞Bench "RAG≫full / IMP<RAG" reflects access, not just compression; corrected by F44. See [`correctness-audit-2026-07-13.md`](correctness-audit-2026-07-13.md).

Synthesis of the 27 established facts ([`matrix-facts.md`](matrix-facts.md) F1–F27) into thematic clusters + the higher-level insights that drive both papers. Every claim traces to a fact ID and its log. Method behind current IMP numbers = **`IMP-v2.1.0`** (span-level, Mode A; registry D28).

---

## Cluster 1 — No universal method; the ranking flips by regime
- **F3 / F18 / F27:** which method wins is *regime-specific*. Retrieval: kvzip/knorm/RAG/LLMLingua ≈ full (0.8–1.0), snapkv & ToMe fail; multi-hop: window/RAG/LLMLingua ~0.5–0.6; abstractive: nothing beats full; literary-MC: compression ≈ no-ctx > full. Confirmed at FULL scale (F27).
- **F11:** these regime failures are **size-invariant (Qwen3 1.7B→14B) and cross-family (Qwen2.5, Llama).**
> **Insight ①:** there is **no fixed compressor that is right everywhere** → the decision of *what to keep* must be made **per input**, not baked into a method.

## Cluster 2 — Exactly where each baseline family breaks
- **F1** attention-KV eviction (snapkv/h2o/tova/expected) **collapses at a sharp ~16k cliff** (0.62→0.17).
- **F2 / F4** attention-free eviction (knorm/streaming/kvzip) is length-robust to 32k, **but kvzip is retrieval-only** with a real ratio cliff at ~0.9 (not cliff-free).
- **F5** window truncation = a **locality assumption** — lossless when the answer is local, fails on retrieval/multi-hop.
- **F8 / F16** LLMLingua **denoises but drops needles**; original perplexity-LLMLingua ≥ LongLLMLingua (question-awareness doesn't help here).
- **F9** BM25-RAG is length-robust but **lexical-overlap-bound** — fails abstractive/global, *hurts* QuALITY, budget-invariant.
- **F14 / F17** naive token-merging (ToMe) **destroys the needle** (RULER 0.0); similarity-matching beats random only on *redundant* content, never preserves a needle.
> **Insight ②:** every family's failure is **structural** (a built-in assumption: locality / lexical overlap / mergeability / attention-mass), so it fails *predictably* in the regime that violates it.

## Cluster 3 — "More context hurts", and compression as denoising
- **F6** "full context hurts" is **literary-MC-specific** (QuALITY full < blind) and is a **base-capability ceiling, NOT distractor-driven** (full-read is distractor-robust on RCA).
- **F7** distractors degrade window & prompt-compression, **not** full-read or RAG.
- **F23 / F27** compression **denoises retrieval**: LLMLingua/RAG/IMP beat `full` on trivia, categorical/coding NIAH, squad, ∞Bench. On the hardest MC (LongBench-v2) `full` ≈ blind (~0 gain).
> **Insight ③:** long context is often a **liability, not an asset** — the win from compression is partly *denoising*, not just cost. This reframes the goal from "fit more" to "keep the right part."

## Cluster 4 — Architecture / size / family generality
- **F10** GDN base (Qwen3.5/3.6) has **no KV cache** → KV-eviction methods are N/A, but the *task-level* failures (full-hurts on literary MC) reproduce.
- **F15** Qwen3.6 uses the **same GDN linear module** as 3.5 (3:1 hybrid) → findings carry forward.
- **F11** failures invariant across size & family.
> **Insight ④:** because the failures are architecture/size/family-invariant, an **importance signal learned/derived from one base transfers** → licenses "bolt IMP onto *any* frozen base" (incl. cache-free linear models, where KV-eviction can't even apply).

## Cluster 5 — The signal that makes training-free routing feasible
- **F20** cheap **O(L) signals locate the needle token, length-invariantly**: query-relevance AUROC 0.95 (word-needle), surprisal 0.84 (numeric); **neither wins both**; embedding/hidden-norm useless.
> **Insight ⑤:** a **training-free, multi-signal importance router is feasible** — and because no single signal wins, the **combination** (z(query)+z(surprisal)) is load-bearing. This is the (a)-SCORE stage of IMP.

## Cluster 6 — IMP (our method) results
- **F24** token-level IMP (`v2.0`) matches `full` on retrieval at every length (ruler 1.0, where ToMe=0.0) **but shatters coherent QA** (squad 0.15) — per-token top-p breaks sentences.
- **F25** span-level IMP (`v2.1.0`, the method) **rescues QA** (squad 0.15→0.46, hotpot 0.21→0.42) **while keeping retrieval = full**, and **beats full on trivia/categorical by denoising**; keep-rate monotone; span-size 16–64 insensitive.
- **F27** at FULL test scale: **RULER-16k 96.8 ≈ full 99.0** where ToMe→0.0 / window→6.0 collapse; mid-pack on real long-doc QA.
> **Insight ⑥:** the right *granularity* is the **span**, not the token — the answer lives in a coherent span. Span-level importance-routing keeps the needle's span intact (retrieval) *and* the answer sentence (QA). This is IMP's core design choice and its distinctive win vs merge/truncate baselines.

## Cluster 7 — Paper A (learned soft-memory + gate): negative results
- **F12** the compressor is a **commodity** (flip any single loss → ±0.03; 2-loss core ≈ kitchen-sink).
- **F13** "gated ≥ full" was **tautological in-sample** (out-of-sample/conformal not yet validated).
- **F22** repeat-prompt **reconstruction is not a usable do-no-harm gate** — indistinguishable from a random-memory control at every K (mean real−random ≈ 0); only `conf` (~0.61–0.69) gates.
- **F26** the learned soft-memory compressor **cannot compress extractive QA** (compress 0.13–0.20 ≈ blind ≪ full 0.69); longer/enc-LoRA/higher-LR make it worse. **Training-free IMP (0.46) beats the learned compressor (0.19)** on the same task.
> **Insight ⑦:** Paper A's learned compressor is the **blocker**, not a knob-tuning issue; the reconstruction-as-gate hypothesis is dead. This **tilts the project to Paper B** — a training-free/light structural router already outperforms the learned memory.

---

## The six meta-insights (what it all means)
1. **Bottleneck = finding/keeping the important info, not raw capacity.** Every fixed method fails in a characterizable regime (Cluster 1–2).
2. **Failures are structural & predictable** → they map cleanly to each method's built-in assumption (Insight ②).
3. **More context is often a liability**; compression frequently *denoises* and beats full (Cluster 3).
4. **Failures are architecture/size/family-invariant** → an importance router transfers; works even on cache-free linear models (Cluster 4).
5. **Cheap O(L) signals can locate the important tokens** → training-free routing is feasible; combine signals (Cluster 5).
6. **Select coherent SPANS, not tokens** → span-level IMP is the general method: retrieval-lossless *and* QA-coherent (Cluster 6). Paper A's learned compressor is blocked and is beaten by training-free IMP (Cluster 7).

## Status of the evidence (2026-07-08, live)
- **FULL-test main table** (F27): running, ~93/112 cells (long-context headline on FULL split). [`main-table-fulltest.md`](main-table-fulltest.md)
- **Cross-architecture generality** (Qwen3.5-9B GDN + others): running. [`generality-model-matrix.md`](generality-model-matrix.md)
- **Keep-rate ablation, all methods** (validates Insight ① budget axis + IMP monotonicity): running. [`hyperparameters.md`](hyperparameters.md)
- Excluded/broken items (never cite): `multi_needle_niah` scoring bug; pyramidkv/adakv/qfilter/duo/kvzap (don't run on Qwen3 GQA); gist-lite/cart-lite (not the published methods); CMG-demo accuracy (near-chance).
