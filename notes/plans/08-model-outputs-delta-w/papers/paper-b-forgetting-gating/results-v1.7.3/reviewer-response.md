# v1.7.3 — Reviewer-response plan (simulated critical review → resolutions)

The thesis reframe (**robustness layer, not a better compressor** — see [`robustness-plan.md`](robustness-plan.md)) *disarms* the biggest attacks (R1/R2/R6/R8 no longer rest on "compression wins"). The rest map to concrete experiments. Legend: 🟢 running · 🟡 queued · 🔵 analysis-only (no GPU) · 🛠️ needs code · ⚪ later.

| # | Reviewer's skeptical claim | Resolution | Status |
|---|---|---|---|
| **R1** | "You sell *long-context* compression but corr(ctx,margin)=−0.4 — you fail exactly there." | **Reframe** to robustness layer; turn the capacity limit into a **contribution** ("when does soft-prompt compression work?"). **K-sweep** maps the accuracy×ratio frontier — does more K rescue long ctx? | 🟢 K-sweep running; reframed |
| **R2** | "Saves ≈0 compute (often negative)." | Reframe: value = **do-no-harm robustness**, not raw saving. Report **accuracy × compression-ratio × FLOPs** frontier; state the regime honestly. | 🟢 K-sweep → 🔵 cost analysis |
| **R3** | "`tool_acc` = function **name** only; ignores **args** — your wins may be hollow." | **AST/args-aware re-scoring**: keep BFCL ground-truth args in the loader + arg-match scorer; re-score. Could erase small wins → must know. | 🛠️ **next code batch (high priority)** |
| **R4** | "Weak baselines. Need LLMLingua / KV-compression / **trivial** (truncate-to-K, mean-pool, random-K) / **retrieval**." | (a) **matched-budget** Cartridge/Gist (fairness), (b) **trivial-compression** baselines (necessity), (c) **LLMLingua** (E7), (d) **retrieval** (BM25/embed top-fn). | ✅ matched done (GCM≥Cart≫Gist; [§2.4a](results-v1.7.3.md#24-matched-budget-baselines--the-compressor-agnostic-gate-e1)) · ✅ trivial done (**trunc=full≥GCM on short ctx**; [§2.4c](results-v1.7.3.md)) · ⚪ LLMLingua/retrieval |
| **R5** | "Is the method necessary? Drop LoRA / encoder / AE." | LoRA0 (✅ +0.02), decoder/ndec (✅), conditioning/dev/distill (clean re-run), **no-encoder / mean-pool** baseline (does an N-layer encoder beat pooling?). | ✅ **trivial answers it**: encoder beats `trunc`/`meanpool` **only on bfcl_live_multiple** (ctx≫K); elsewhere trivial wins → necessity is **capacity-bound** |
| **R6** | "Gate ≈ chance on 8/10 benches ⇒ contribution unsupported." | Reframe: robustness via **conservative fallback** holds even at chance (gАcc=full); value-add only where AUROC>0.5. **E1** detection-AUROC across compressors; **E5** better detector (multi-signal/supervised vs trivial perplexity). | 🟢 E1 running · 🔵 E5 |
| **R7** | "Single seed; config gaps (0.36–0.38) within ±0.04; gАcc optimistic." | **Multi-seed CIs** on headline configs + a proper **train/cal/test** split for the gate threshold (vs 5-fold). | 🟡 **queuing now** |
| **R8** | "Novelty vs Gisting/ICAE/AutoCompressor/LLMLingua." | Position the **gate/do-no-harm** as the novelty; add an **ICAE/AutoCompressor-style** soft-prompt baseline to show our compressor is at least competitive (or honestly not). | ⚪ positioning + baseline |
| **R9** | "`full` has no chat template — artificial ceiling." | Re-run `full`/`no_ctx` **with the chat template**; report whether the gap (headroom) changes. | 🛠️ harness flag |

## Execution order (GPU-aware)
1. ✅ **Done:** matched-budget baselines (R4 — GCM≥Cart≫Gist), **trivial baselines (R4/R5 — trunc=full≥GCM on short ctx)**, E1 compressor-agnostic gate on Cartridge (R6).
2. 🔄 **Running now:** **R3 args-aware scoring** (verdict-decider, 4/6 done), **K-sweep** capacity frontier (R1/R2), **multi-seed CIs** (R7, seeds 43/44), **Phase-2 transfer** matrix.
3. **Analysis-only (no GPU, after data lands):** R2 cost frontier, R6/E5 detector combination, E3 harm-prevented, E4 calibration.
4. **Later / bigger:** **R4/E7 LLMLingua**, **R4 retrieval**, **R8 ICAE/AutoCompressor**, **R9 chat-template full**.

## The two failure modes for the paper, and which experiments decide them
- **If R3 (args-aware) tanks the tool numbers** AND the K-sweep shows no long-ctx rescue ⇒ the honest paper is **"diagnosis + a do-no-harm robustness layer"** (R1/R2/R5/R6 evidence carries it; we do NOT claim SOTA compression). **The trivial-baseline result already leans this way:** truncation matches full on short ctx, so the compressor's value is narrow and capacity-bound — the *gate* is the durable contribution.
- **If the K-sweep finds a moderate-ratio regime that beats LLMLingua** with real saving ⇒ a **"robust compression"** paper (needs R4 LLMLingua + R3 args to hold).
The remaining verdict-deciders are **R3 (args-aware)** and the **K-sweep** — both running now.
