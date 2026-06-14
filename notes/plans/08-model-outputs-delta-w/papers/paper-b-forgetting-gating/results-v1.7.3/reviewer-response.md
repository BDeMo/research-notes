# v1.7.3 — Reviewer-response plan (simulated critical review → resolutions)

The thesis reframe (**robustness layer, not a better compressor** — see [`robustness-plan.md`](robustness-plan.md)) *disarms* the biggest attacks (R1/R2/R6/R8 no longer rest on "compression wins"). The rest map to concrete experiments. Legend: 🟢 running · 🟡 queued · 🔵 analysis-only (no GPU) · 🛠️ needs code · ⚪ later.

| # | Reviewer's skeptical claim | Resolution | Status |
|---|---|---|---|
| **R1** | "You sell *long-context* compression but corr(ctx,margin)=−0.4 — you fail exactly there." | **Reframe** to robustness layer; turn the capacity limit into a **contribution** ("when does soft-prompt compression work?"). **K-sweep** maps the accuracy×ratio frontier — does more K rescue long ctx? | 🟢 K-sweep running; reframed |
| **R2** | "Saves ≈0 compute (often negative)." | Reframe: value = **do-no-harm robustness**, not raw saving. Report **accuracy × compression-ratio × FLOPs** frontier; state the regime honestly. | 🟢 K-sweep → 🔵 cost analysis |
| **R3** | "`tool_acc` = function **name** only; ignores **args** — your wins may be hollow." | **AST/args-aware re-scoring**: keep BFCL ground-truth args in the loader + arg-match scorer; re-score. Could erase small wins → must know. | 🛠️ **next code batch (high priority)** |
| **R4** | "Weak baselines. Need LLMLingua / KV-compression / **trivial** (truncate-to-K, mean-pool, random-K) / **retrieval**." | (a) **matched-budget** Cartridge/Gist (fairness), (b) **trivial-compression** baselines (necessity), (c) **LLMLingua** (E7), (d) **retrieval** (BM25/embed top-fn). | 🟢 matched running · 🛠️ trivial/retrieval · ⚪ LLMLingua |
| **R5** | "Is the method necessary? Drop LoRA / encoder / AE." | LoRA0 (✅ +0.02), decoder/ndec (✅), conditioning/dev/distill (clean re-run), **no-encoder / mean-pool** baseline (does an N-layer encoder beat pooling?). | 🔵 have most · 🛠️ pooling |
| **R6** | "Gate ≈ chance on 8/10 benches ⇒ contribution unsupported." | Reframe: robustness via **conservative fallback** holds even at chance (gАcc=full); value-add only where AUROC>0.5. **E1** detection-AUROC across compressors; **E5** better detector (multi-signal/supervised vs trivial perplexity). | 🟢 E1 running · 🔵 E5 |
| **R7** | "Single seed; config gaps (0.36–0.38) within ±0.04; gАcc optimistic." | **Multi-seed CIs** on headline configs + a proper **train/cal/test** split for the gate threshold (vs 5-fold). | 🟡 **queuing now** |
| **R8** | "Novelty vs Gisting/ICAE/AutoCompressor/LLMLingua." | Position the **gate/do-no-harm** as the novelty; add an **ICAE/AutoCompressor-style** soft-prompt baseline to show our compressor is at least competitive (or honestly not). | ⚪ positioning + baseline |
| **R9** | "`full` has no chat template — artificial ceiling." | Re-run `full`/`no_ctx` **with the chat template**; report whether the gap (headroom) changes. | 🛠️ harness flag |

## Execution order (GPU-aware)
1. **Running now (49 jobs):** matched-budget baselines (R4), K-sweep (R1/R2), Phase-2 transfer, E1 compressor-agnostic gate (R6).
2. **Queue now:** **multi-seed CIs** (R7) on the value-add benches (combo, seeds 43/44).
3. **Next code batch (small, high-value):** **R3 args-aware scoring** (could change the verdict — do first) → **R4 trivial baselines** (truncate/mean-pool/random-K) → **R9 chat-template full** → **R5 no-encoder/pooling**.
4. **Analysis-only (no GPU, after data lands):** R2 cost frontier, R6/E5 detector combination, E3 harm-prevented, E4 calibration.
5. **Later / bigger:** **R4/E7 LLMLingua**, **R4 retrieval**, **R8 ICAE/AutoCompressor** baselines.

## The two failure modes for the paper, and which experiments decide them
- **If R3 (args-aware) tanks the tool numbers** AND the K-sweep shows no long-ctx rescue ⇒ the honest paper is **"diagnosis + a do-no-harm robustness layer"** (R1/R2/R5/R6 evidence carries it; we do NOT claim SOTA compression).
- **If the K-sweep finds a moderate-ratio regime that beats LLMLingua** with real saving ⇒ a **"robust compression"** paper (needs R4 LLMLingua + R3 args to hold).
The next code batch (R3, R4-trivial, R9) is what tells us which paper we have — so it is the priority after the current queue.
