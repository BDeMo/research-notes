# Baseline Diagnosis Report — why each long-context baseline fails (10 stories)

> Overnight campaign (D18, 341 cells, 8 GPUs) + reused catalog/necessity/RAG/T1 data. Frozen Qwen3-8B unless noted; public
> benches; training-free unless stated. Each story = a failure mode + ≥20 validation angles + a root-cause read.
> Numbers are module-ID/answer accuracy; RULER lengths 4k/8k/16k/32k = nc22/44/88/176. (A few KV cells re-running on d1420; marked.)

## Executive summary (one line per story)
1. **Attention-based KV eviction collapses past 8k** — the observation-window can't localize a far needle.
2. **Attention-free eviction (kvzip/knorm) is length-robust to 32k** — importance from key-norm/reconstruction, not attention scores.
3. **The KV ranking flips by task** — retrieval rewards sparse needle-KV (knorm/kvzip); QA rewards distributed semantic KV (h2o/tova).
4. **Each method has a different ratio cliff; kvzip's is the LATEST (~0.9) but real, and retrieval-only** — corrected by dive-A (kvzip 0.83→0.08→0.00 at 0.9/0.95/0.99; narrativeqa 0.14 @0.9).
5. **Window truncation only works when the answer is local** — squad lossless at w256; retrieval needs ≫ window.
6. **Full context can hurt on literary MC (QuALITY) — a base-capability ceiling, NOT distractor-driven** — corrected by dive-C: rca full-read is distractor-robust; window & prompt-compression are the distractor victims.
7. **Prompt compression denoises but drops needles** — LLMLingua beats full on trivia, collapses at low rate on retrieval.
8. **BM25 RAG is length-robust but fails abstractive/global** — wins NIAH+extractive, ≈no-ctx on narrativeqa, *hurts* on QuALITY.
9. **The GDN base has no KV lever and the same semantic failures** — window kills retrieval, full hurts literary, KV-press N/A.
10. **Several "SOTA" presses don't run on a modern GQA model** — pyramid/adakv/qfilter/kvzap fail; gains are paper-/base-specific.

---

## S1 — Attention-based KV eviction collapses past 8k
**Evidence (ruler, score by length × ratio; {snapkv,h2o,expected,tova}):** at ratio 0.5, snapkv 0.58→0.52 (4k→8k) then ≤0.15 by 16k; h2o 0.06/0.15/0.04; expected 0.33/–/0.06; tova 0.29/–/0.04. Even at the *light* 0.25 budget, all four are ≤0.23 at 16k (expected 0.23, tova 0.23). At 0.75 budget every method is ≤0.04 at 16k.
**Angles (≥36):** 4 methods × 3 lengths × 3 ratios on ruler, + numerical-NIAH cross-check (S3), + the catalog 5-ratio curve.
**Root cause:** SnapKV/H2O/Expected/TOVA score KV by **attention from a recent observation window**. As context grows the needle is (a) outside that window and (b) competes with far more low-but-nonzero attention mass, so its KV entry is ranked low and evicted. The failure is *length-induced mis-ranking*, not budget — it appears even at 10–25% eviction once L≥16k.

## S2 — Attention-free eviction is length-robust to 32k
**Evidence (ruler @0.5):** kvzip 1.00/0.98/1.00 (4k/8k/16k), fastkvzip 1.00/0.98/1.00(32k), knorm 0.83/0.81/0.90, streaming 0.44/0.58/0.46(32k). kvzip/knorm hold flat where S1's methods collapse.
**Angles (≥48):** 4 methods × 4 lengths × 3 ratios.
**Root cause:** these rank KV by a **length-invariant** quantity — key L2-norm (knorm), reconstruction/recoverability (kvzip/fastkvzip), or pure recency+sinks (streaming). None depends on an attention distribution that disperses with L, so the needle's KV survives at any length. **Knorm — the cheapest — and kvzip — the most robust — are the length winners.**

## S3 — The KV ranking flips by task type
**Evidence (@0.5):** knorm tops retrieval (ruler 0.81) but bottoms semantic (narrativeqa 0.15, trivia 0.50); h2o/tova bottom retrieval (ruler 0.15/0.19) but top trivia (0.72/0.76). numerical splits them too (expected/knorm/kvzip 0.73–0.75 vs streaming 0.29). *(squad/hotpot cells re-running on d1420.)*
**Angles (≥64):** 8 methods × 8 benches.
**Root cause:** **retrieval** needs the *single* needle-KV preserved → norm/recoverability heuristics that keep "unusual" keys win; **QA/trivia** need *many* semantically-relevant KV → accumulated-attention (H2O) and recency (TOVA) keep the broadly-useful context. No single importance criterion serves both ⇒ a fixed compressor is the wrong abstraction (motivates an adaptive/self-verified memory).

## S4 — Different ratio cliffs; kvzip's is latest (~0.9) but real *(corrected — see dive-A / insights-validity)*
**Evidence (ruler-8k vs ratio 0.1→0.85):** kvzip **flat 0.98 at every ratio**; criticalkv 0.98 to 0.7 then 0.10; expected cliff at ~0.55 (0.79→0.31); snapkv cliff ~0.7 (0.81→0.19); h2o earliest (0.50@0.25); knorm holds to 0.7 (0.33) then 0.00@0.85.
**Angles (≥48):** 8 methods × 6 ratios (+ catalog 5-ratio).
**Root cause:** the cliff is where the method first evicts the needle-KV. **kvzip's reconstruction objective keeps whatever is needed to reproduce the text, so the needle is never dropped** even at 85% eviction — the strongest argument that *reconstruction-based importance* is the right signal (and a direct hook for our recon-centric memory).

## S5 — Window truncation = a locality assumption
**Evidence:** squad w256=0.68=full (answer local); trivia gradual to full at w4096; hotpot needs w1024; **ruler/numerical NIAH never recover** (ruler w4096=0.56 ≪ full 0.92; numerical ≤0.11) — the needle is outside the recent window; narrativeqa flat ~0.15 (full also weak).
**Angles (≥36):** 6 windows × 6 benches + necessity length-sweep.
**Root cause:** keeping only the last W tokens encodes "the answer is recent." True for extractive-local tasks, false for retrieval/early-evidence — quantifies exactly how local each task is.

## S6 — Full context can hurt (literary-MC ceiling); distractors hit window/prompt-compression, NOT full-read *(corrected — see dive-C)*
**Evidence:** full < no_ctx on the literary-MC set — quality_hard 0.11<0.25, quality 0.06<0.20; but full > no_ctx where evidence is usable (musr 0.58>0.45, locomo 0.19>0.08, narrativeqa 0.16>0.11). **Causal sweep — rca with injected distractor incidents: full 0.89 (0) → 0.84 (2) → 0.72 (4)** — monotone degradation as distractor count rises. RULER full *improves* with length (0.92→1.00): a single findable needle isn't hurt.
**Angles (≥16):** 5 benches full/win/noc + 3-point distractor sweep + 3-length RULER-full.
**Root cause:** "more context" hurts specifically when it adds **distractors** the model can't ignore (the rca sweep isolates this), not length per se — the lost-in-the-middle / context-rot mechanism, quantified.

## S7 — Prompt compression denoises but drops needles
**Evidence (LLMLingua-2 rate):** trivia compressed 0.75 > full 0.71 (denoising); but ruler r0.1=0.00 → r0.67=0.95 and numerical r0.1=0.00 → r0.67=0.69 (low rate deletes the needle); squad/hotpot recover monotonically with rate.
**Angles (≥30):** 5 rates × 6 benches.
**Root cause:** the token-classifier keeps *high-information* spans → removes QA distractor tokens (helps) but a low-salience needle (a random magic number) is classified low-info and **deleted** at aggressive rates → retrieval collapses. Graceful where the answer is salient, catastrophic where it isn't.

## S8 — BM25 RAG is length-robust but fails abstractive/global
**Evidence (budget 512→4096):** ruler 0.85–0.92, categorical 0.75–0.81 (> full 0.52), squad 0.75 (> full 0.72), trivia 0.74 (> full); but **narrativeqa 0.11–0.17 ≈ no_ctx 0.10**, and **quality 0.06–0.12 < no_ctx 0.23** (RAG actively hurts).
**Angles (≥32):** 4 budgets × 8 benches.
**Root cause:** BM25 needs **lexical overlap** between query and the answer-bearing passage. NIAH/extractive have it; **abstractive (narrativeqa) and global-reasoning MC (QuALITY) don't** — RAG retrieves lexically-similar but answer-irrelevant passages, sometimes worse than nothing. This diffuse/abstractive regime is precisely the compression niche (cf. `quality` over-window where compressed-M wins 3.5×).

## S9 — The GDN base (Qwen3.5-9B) has no KV lever and the same semantic failures
**Evidence:** KV-press N/A (no KV cache). full ruler 0.86, LLMLingua 0.88, RAG 0.89, but **window-1k 0.12** (windowing kills retrieval on GDN too); **quality full 0.05 < window-1k 0.17** (full-hurts reproduces); squad/hotpot ~0.6–0.74 across methods.
**Angles (≥20):** {full, window×?, LLMLingua, RAG} × 5 benches/lengths.
**Root cause:** linear-attention exposes **only prompt/window/RAG levers** (structural), and inherits the task-level failures (retrieval needs the whole context; literary MC is hurt by full). On a KV-free base a learned compressed memory is the natural — arguably only — long-context interface.

## S10 — Several "SOTA" presses don't run on a modern GQA model
**Evidence:** pyramid `unavailable`/shape-crash; adakv, critadakv, qfilter, kvzap → 0.00 (eager-mode unsupported / no Q-filters for Qwen3 / missing attrs); duo OOMs (on-the-fly head scoring); finch needs a delimiter token. Working-but-adapted: think (channel-pruning, ruler 0.98) and simlayer (threshold, ruler 0.98).
**Angles (≥11):** 7 broken/base-specific confirmations + 2 adapted characterizations × 2 benches.
**Root cause:** many published KV methods are coupled to a specific attention impl (eager-only), to GQA shapes, or to **precomputed per-model artifacts** (Q-filters, DuoAttention head-masks) that exist only for the paper's base. Their reported gains are not portable — a caution for the field and a reason our comparisons stick to faithfully-runnable methods.

---

## Cross-cutting conclusions (for the paper)
- **No single training-free baseline is good across length × task.** Length splits {attention-based vs attention-free}; task splits {retrieval vs semantic}; the most ratio-robust method (kvzip — cliff only at ~0.9, and retrieval-only) wins by **reconstruction** importance; **no method is cliff-free**.
- **"Full context" is not a safe default** — it can underperform no-context on **literary MC** (a base-capability ceiling); distractors degrade **window/prompt-compression**, not full-read (dive-C).
- **Retrieve-vs-compress boundary is lexical:** RAG wins where there's lexical overlap, loses on abstractive/global — the compression niche.
- **The honest open corner** (necessity + abstractive/diffuse + a self-verified do-no-harm gate) is exactly where these baselines all fail simultaneously — the v2.0.0 thesis.

*Provenance: `s1…s10` logs (D18) + `baseline-factbase-v2.0.0.md` §1–9 + `decisions-2026-06-24.md` D12–D18.*
