# Paper B v2.0.0 — method brainstorm (transform the method for long-context + domain generalization)

> Generative brainstorm for the **method transformation** behind v2.0.0 (long-context *necessity* + *domain generalization*, not efficiency).
> Format (per [`09/ideas-brainstorm`](../../../09-intrinsic-site-protection/ideas-brainstorm-2026-06-04.md)): each idea = **Motivation (which fact) · Method · Expected insight · Frontier · Cost/Risk**. 💡 = unbuilt.
> Grounded in our **verified** v1.8.0 facts + the lit DB [`references-longcontext.md`](references-longcontext.md). Companion to [`v2.0.0-plan.md`](v2.0.0-plan.md). Cull later; this is the wide net.

## The facts we build on (verified this cycle)
- **F1 — semantic ≫ retrieval split (per-item verified).** Over-window compress **wins** on diffuse/semantic long-ctx (quality_hard 0.48 vs trunc-full 0.13; right where full is wrong) and **fails** on exact-needle retrieval (RULER 0.00; full 0.47). *→ M preserves **meaning**, destroys **exact identifiers**.*
- **F2 — exact-string failure + gen collapse.** On RULER/bfcl_multiple the compressed memory can't reproduce a rare verbatim string; free-gen **collapses into repetition / echoes the query key** ("k1544573"→"154457"). *→ need a verbatim/copy channel.*
- **F3 — gate is a safety-net, not free accuracy (held-out CV).** gated_cv −0.02 vs *strong* full, **+0.52 when full collapses** (Qwen3.5-4B). *→ value = rescue when the feasible read is degraded; "≥ full" only holds out-of-sample with calibration.*
- **F4 — raw window beats compression when evidence is recent / fits.** TXL/StreamingLLM(W=1024) ≈ full ≫ ours on apibank/squad/hotpot. *→ compression only earns its keep over-window + spread.*
- **F5 — over-window memory wall.** recurrent carry grows ⌈L/chunk⌉·K → single 93GB H100 tops ~8–12k at K=256; 16k OOMs. *→ scaling needs a sub-linear memory, not naive carry.*
- **F6 — the base is a Gated-DeltaNet hybrid (Qwen3.5).** KV-cache-free linear-attn ⇒ no per-layer KV to inject/evict ⇒ **soft-prompt is the only interface**, and M competes with the base's *own* recurrent state S_t. *→ a native interface: write into S_t.*
- **F7 — baselines.** SFT-LoRA wins accuracy at full-ctx cost + forgets; Gist≈ours; Cartridge wins only on *amortizable* ctx. *→ our niche is per-input, weight-frozen, over-window.*
- **F8 — transfer ∝ hidden stats, not surface similarity** (lit #101/#104). *→ domain-generalization must be designed, and is *intrinsically* predictable.*
- **F9 — method is HP-robust; K128 mild-best; the gate (not the compressor) was v1.8.0's contribution.** *→ v2 must add a NEW method axis, not re-tune K.*

---

## §1 · Method-idea grid (regime to win × mechanism)
Rows = the regime we must win; cols = mechanism family. ✪ = most promising given F1–F9.

| win-regime ↓ \ mechanism → | raw+compressed **hybrid** | **copy/verbatim** channel | **retrieval** into memory | **domain-invariant** M | **native-state** write (F6) |
|---|---|---|---|---|---|
| spread/semantic (quality) | H1 window+global-M ✪ | — | H2 chunk-retrieve→M | H3 invariance loss ✪ | H4 state-init from M |
| exact retrieval (RULER) | H5 window covers needle | **H6 copy-slots / pointer-M** ✪ | H7 query-routed chunk fetch ✪ | — | — |
| over-window scaling (F5) | H8 sink+window+M | — | H9 hierarchical mem tree | — | H10 fixed-size S_t carry ✪ |
| cross-domain (F8) | — | — | — | **H11 adversarial-invariant + H12 predict-transfer** ✪ | — |
| do-no-harm (F3) | H13 multi-path router ✪ | — | — | — | — |

→ **The unifying v2.0.0 method (M0): "Hybrid Evidence Memory + conformal router."** Keep a **bounded raw recent window** (exact/local detail) + a **compressed global memory M** (spread/semantic) + an optional **copy channel** for high-surprise identifiers; a **conformally-calibrated router** picks the cheapest sufficient combination per query; M is trained with a **domain-invariance** objective so it transfers. Everything below is an ablation axis of M0.

---

## §2 · Ideas (depth)
**H1 — Window + global-M hybrid (headline candidate).** *Motiv:* F1+F4 — recent-window fixes local/exact, M fixes spread. *Method:* reader prefix = `[ctx[-W:] ; M(all ctx) ; q]`; train M-only (Variant A) or M+read-LoRA (Variant B). *Insight:* does hybrid beat **both** TXL(W) and compress-alone on quality AND recover on bfcl_multiple/RULER? *Frontier:* method. *Cost:* low (the `GCM_COMBINE` we sketched). ✪

**H6 — Copy-slots / pointer memory.** *Motiv:* F2 — soft M can't emit rare verbatim strings. *Method:* augment M with a few **discrete copy slots** = pointers to (or low-rank keys of) high-surprise ctx tokens (identifiers, numbers); at decode, a copy-head can emit the pointed token verbatim (à la pointer-networks / CopyNet in the soft-prompt era). *Insight:* closes the RULER/identifier gap without paying full-ctx KV. *Frontier:* method (novel for soft compression). *Cost:* med. ✪

**H2/H7 — Compress-then-retrieve (memory as a tiny index).** *Motiv:* F1 (retrieval), F5 (scaling). *Method:* encode ctx into per-chunk memories; given q, **retrieve top-k chunk-memories** (cheap dot-product) → assemble M. Bridges RAG and compression; bounded memory regardless of L. *Insight:* recovers needle/retrieval tasks; scales past the F5 wall. *Frontier:* method. *Cost:* med. ✪

**H3/H11/H12 — Domain-invariant memory (the generalization pillar).** *Motiv:* F8. *Method:* train the encoder with (a) a **domain-adversarial** head on M (gradient-reversal so M can't predict its source domain) or (b) an **invariance/IRM** penalty; (c) use the SAE/STS predictor (#104) to *forecast* transfer and select source mixes. *Insight:* does invariance lift cross-domain compress at fixed in-domain? does the intrinsic predictor (our §K probes) correlate? *Frontier:* both — this is the v2 *science*. *Cost:* med. ✪

**H4/H10 — Native state-write into the Gated-DeltaNet recurrence.** *Motiv:* F6 — the base already carries a fixed-size state S_t; a soft prefix is a clumsy proxy. *Method:* learn a small map (compressed evidence → an **initial/edited recurrent state S_0**) injected before the query, so the base "remembers" the whole ctx through its *own* mechanism; fixed-size ⇒ no F5 wall. *Insight:* is writing S_t better than soft-prefix M on a linear-attn base? (architecture-native, novel). *Frontier:* method (high-risk/high-reward). *Cost:* high. ✪✪

**H8 — Sink + window + M.** *Motiv:* F4 (StreamingLLM sinks matter) + F1. *Method:* `[sinks ; ctx[-W:] ; M ; q]`. *Insight:* do attention sinks stabilize the hybrid? *Frontier:* method. *Cost:* low.

**H9 — Hierarchical memory tree.** *Motiv:* F5. *Method:* recursively compress chunk-memories into a tree (log-depth) → sub-linear carried state; query descends the tree. *Insight:* breaks the ⌈L/chunk⌉·K growth. *Frontier:* method. *Cost:* high.

**H13 — Multi-path conformal router (gate → v2).** *Motiv:* F3. *Method:* router over {base, window, M, window+M, full} chosen by **conformal-calibrated** signals with a coverage guarantee; report cost–accuracy Pareto. *Insight:* provable do-no-harm out-of-sample + cost story. *Frontier:* method+theory. *Cost:* low-med (eval-only-ish). ✪

**H14 — Adaptive content-aware K.** *Motiv:* capacity. *Method:* allocate memory tokens by local information density (more to identifier-/entity-dense spans). *Frontier:* method. *Cost:* med.

**H15 — Verifiable memory (strong recon + self-check).** *Motiv:* F3 (neg_recon gate). *Method:* push reconstruction so M is provably faithful; the reader self-verifies M against a cheap probe before trusting it. *Frontier:* method. *Cost:* med.

**H16 — Iterative/agentic compression (test-time compute).** *Motiv:* long-ctx reasoning (LongBench-v2 needs *reasoning* not retrieval). *Method:* multi-pass: model reads M, asks "what's missing?", re-compresses targeted spans. *Frontier:* blue-sky. *Cost:* high.

---

## §3 · What to evolve into the v2.0.0 plan (down-select)
**Tier-1 (build first, cheap, high-signal):** H1 (window+global-M hybrid) · H13 (conformal router) · H3 (domain-invariant M + the §K intrinsic probes). These three = the three v2 claims (necessity-via-hybrid, do-no-harm-proved, generalization).
**Tier-2 (the differentiators):** H6 (copy-slots → fix exact-string) · H2/H7 (compress-then-retrieve → fix RULER + scale).
**Tier-3 (high-risk, high-reward / next paper):** H4/H10 (native state-write on Gated-DeltaNet) · H9 (hierarchical tree).

**Open questions to resolve before the plan:**
1. Is the headline **hybrid (H1)** or **pure compression** the method? (hybrid is honest given F4, but blurs "compression" — decide the framing.)
2. Does domain-invariance (H3) actually lift cross-domain, or just trade off in-domain? (the make-or-break for pillar 2.)
3. Native-state-write (H4) — worth the risk for v2, or hold for v3?

**Next step:** turn Tier-1 + the chosen framing into `v2.0.0-method-plan.md` (concrete configs, ablations, benches = the long-ctx semantic suite + transfer matrix). This file stays the idea backlog.
