# Does IMP work? — an honest, evidence-backed assessment (2026-07-09)

> ⚠️ **`full` = TRUNCATED to MAXCTX (16k)**, not the whole document. Any ∞Bench IMP−RAG gap here is pre-fix (truncation artifact, F44); full-doc retrieval closes/reverses it. `full` = *true full* only for docs ≤ 16k (all benches except ∞Bench). See [`correctness-audit-2026-07-13.md`](correctness-audit-2026-07-13.md).

**Method:** `IMP-v2.1.0` (span-level, Mode A, training-free; [`imp-method-and-implementation.md`](imp-method-and-implementation.md)).
**Scope of evidence:** full-test main table ([`main-table-fulltest.md`](main-table-fulltest.md), F27), cross-architecture generality ([`generality-model-matrix.md`](generality-model-matrix.md), F28), keep-rate ablation ([`keep-ablation-results.md`](keep-ablation-results.md), F29), token-vs-span ablation ([`baseline-factbase-v2.0.0.md`](baseline-factbase-v2.0.0.md) §12, F24/F25). All numbers ×100.

---

## Verdict (one paragraph, honest)
**IMP works *decisively in one regime* — needle/synthetic retrieval — and its win there is *unique and architecture-invariant*; on real long-document QA, extractive QA, and literary-MC it is *mid-pack to weak*, and it does NOT beat strong baselines (RAG / LLMLingua-2 / kvzip / full).** So IMP is **not** a "best-accuracy-everywhere" method. Its defensible contribution is being **training-free, architecture-agnostic (works on cache-free linear models where KV-eviction can't run), and needle-lossless at low budget** — not raw QA accuracy. Making IMP competitive on QA needs **Mode-B** (a small trained router / side-cache), which is not yet built.

---

## 1. Where IMP works (the case FOR)

**(a) Synthetic retrieval — decisive, and the distinctive win.** RULER-NIAH@16k ([main table](main-table-fulltest.md)): **IMP 96.8 ≈ full 99.0**, beating RAG 94.2, while **ToMe → 0.0 and window → 6.0 collapse** (F24/F27). IMP is the only *input-side selection* method that keeps a needle near-losslessly at keep-0.5. kvzip also ties (99) — but see (c).

**(b) Architecture-invariant — works where KV-methods can't.** On **Qwen3.5-9B (linear/GDN, no KV cache)**, kvzip/knorm/snapkv literally cannot run; **IMP still keeps the needle (79 vs full 91)** while ToMe→0, window→4 (F28, [generality](generality-model-matrix.md)). This is IMP's clearest structural advantage: an *input-side* router transfers to any frozen base.

**(c) Budget-efficient at low keep.** Keep-ablation ([keep-ablation-results.md](keep-ablation-results.md), F29): on RULER, **IMP holds 70→98→99 at keep 0.25/0.5/0.75** — near-full from a quarter budget — where window (25→44→74), ll2 (43→95), knorm (19→89) are budget-hungry and ToMe stays 0. IMP's squad curve is **cleanly monotone** (34/33/43/53), the textbook budget↔accuracy trade-off.

**(d) Span-level fixed the token-level failure.** Token-level IMP-v2.0 shredded QA (squad 0.15 < blind); span-level lifted it (F24→F25). The design ablation is clean and load-bearing.

## 2. Where IMP is mid-pack or weak (the case AGAINST — do not overclaim)

**(a) Extractive QA — clearly beaten.** squad_v2 (full N): **IMP 38.9 ≪ full 69.0, RAG 71.5, window 69.0, kvzip 64.9** ([main table](main-table-fulltest.md)). IMP recovers only ~half of full. *(Note: the earlier N=48 snapshot showed span-IMP ~46, F25; at full N=500 it is 38.9 — the "QA rescued" claim is real vs token-level but modest in absolute terms and well below RAG/full.)*

**(b) Real long-doc QA — below RAG/full.** lb_hotpotqa IMP 17.4 < full 22.6 < RAG 24.8; lb_2wikimqa 11.2 < full 22.8 < knorm 22.4; lb_qasper 21.0 < full 27.6; hotpot_qa 43.9 < full 56.2 ≈ kvzip 56.2; trivia 62.3 < full 69.4 < ll2 71.4. IMP is consistently **not** the best and usually **below full**.

**(c) Literary-MC — not IMP's regime.** QuALITY-hard: IMP 11.7 > full 9.2 (full hurts) **but ≪ blind 19.6 and ≪ kvzip 29.9 / knorm 29.6**. Here the winner is KV-mass pruning, not IMP.

**(d) Ultra-long MC — below retrieval baselines.** ∞Bench: IMP 51.5 ≈ full 53.7 but ≪ RAG 64.2, ll2 60.7.

**(e) Hardest MC is a ceiling for everyone.** LongBench-v2: IMP 32.8 ≈ full 33.6 ≈ blind 33.4 — no method (incl. IMP) extracts signal; not evidence for or against IMP.

## 3. What this means for the paper (honest positioning)
- **Do NOT claim** "IMP ≥ best baseline on accuracy." The full-test data refutes it on QA (§2).
- **DO claim** what the evidence supports: IMP is a **training-free, architecture-agnostic, needle-preserving input-side router** that (i) matches `full` on retrieval where merge/truncate collapse (F24/F27), (ii) is the *only* such method that works on **cache-free linear bases** (F28), (iii) is **budget-efficient at low keep** (F29). Its accuracy on QA is a **Mode-A (training-free) lower bound**.
- **The gap to close (Mode B):** a small distilled router + verbatim side-cache should lift QA toward RAG/full while keeping the retrieval + architecture-generality wins. Until then, IMP's QA numbers are honestly reported as mid-pack.

## 4. Falsifiable next steps (to make the "works" claim stronger)
1. **Mode-B on QA:** train the tiny router/side-cache; target squad/hotpot ≥ RAG. Falsifier: if Mode-B can't beat training-free RAG on QA, IMP's QA value is limited to efficiency, not accuracy.
2. **Prefill/efficiency table:** IMP's real selling point may be *cost*, not accuracy — add a tokens-read / latency column so "≈full accuracy at X% prefill" becomes the headline where true.
3. **Cross-family generality (running):** confirm §1(a–c) hold on Qwen2.5 / GLM-4 / Ministral / gpt-oss (quadratic) and Qwen3.5-4B / Moonlight (linear).
4. **Small-scorer variant:** replace the surprisal forward with a draft LM to actually realize the prefill saving on quadratic bases ([`imp-method-and-implementation.md`](imp-method-and-implementation.md) §7).

---
*Bottom line: IMP **works as a training-free, architecture-agnostic needle-router** (strong, unique, defensible), **not** as a SOTA-accuracy QA compressor (the full-test data says it is mid-pack there). The paper's honest headline is generality + needle-preservation + efficiency, with QA accuracy as a Mode-A baseline to be lifted by Mode-B.*
