# Keep the Needle, Drop the Rest: Lightweight Importance Routing for Long-Context on Frozen LLMs

> **Paper B working draft (v0.1).** Content-complete skeleton with real numbers (July-w01 grid snapshot). Numbers cite `baseline-factbase-v2.0.0.md` / `matrix-facts.md`; method detail in `imp-method-and-implementation.md`. Draft — to be tightened; some cells are the in-progress grand-grid snapshot.
> **Method version:** all `imp` numbers here = **`IMP-v2.1.0`** (span-level, Mode A). The FULL-test headline numbers now live in [`main-table-fulltest.md`](main-table-fulltest.md); this draft's inline cells are the earlier N=48 snapshot (superseded — prefer the full-test table).

---

## Abstract

Long-context language models degrade well before their advertised window, and the many training-free coping methods — KV-cache eviction, prompt compression, token merging, retrieval — each fail in a different, characterizable regime. We first map these failures under one harness and one frozen base, establishing that **no single method is universal**, that **naive token-merging destroys the answer**, and that **cheap $O(L)$ signals (query-relevance, surprisal) already localize the answer token** (AUROC 0.95 / 0.84). Guided by these observations we propose **IMP**, a lightweight *importance-routing prefilter* that attaches to a **frozen** base of any architecture: it scores each context token with the base's own cheap signals, **keeps the top-$p$ most-important spans verbatim**, drops the redundant rest, and lets the frozen base read the shortened sequence — deployable *plug-and-play* (no training) or with a *tiny learned router*. IMP matches full-context accuracy on retrieval at every length (incl. 16k, where token-merging scores 0), **rescues coherent QA that token-level selection shreds** (SQuAD 0.15→0.46, HotpotQA 0.21→0.42), and on **real-length LongBench (32k, untruncated)** compression frequently **beats full context** by denoising. We give an honest account of where IMP does *not* yet win (short-context extractive QA vs. RAG) and identify the three experiments that make or break the contribution.

---

## 1. Introduction

Modern LLMs accept 32k–1M token inputs, but "long context" is not solved: (i) full context is expensive ($O(L^2)$ prefill, linearly growing KV cache) and can *hurt* accuracy (distractors, lost-in-the-middle); (ii) training-free reduction methods are brittle; (iii) different tasks need different evidence-preservation. Practitioners reach for KV-eviction, prompt compression, token merging, or retrieval — but there is no apples-to-apples map of *when each works*, and no single method spans the regimes.

We take an **observe-then-design** approach. We first establish, under one harness / one frozen base / one scoring protocol, a set of failure facts (§3). The load-bearing one: **the answer token is cheaply localizable** — an attention-free, $O(L)$ combination of query-relevance and surprisal ranks the needle above filler at AUROC up to 0.95, length-invariantly. This makes a **training-free importance router** feasible.

We then propose **IMP** (§4): score → **keep important spans verbatim** → drop redundancy → frozen base reads the short sequence. IMP is an *input-side* structure, so it applies to **cache-free linear/SSM bases** where KV-eviction does not exist, and it shrinks the read (and, with a small scorer, the prefill). 

**Contributions.** (1) A controlled failure map of training-free long-context methods (§3). (2) IMP, a plug-and-play / lightly-trained importance-routing structure for a frozen base (§4). (3) A main table across ~15 methods × ~19 benchmarks incl. real-length LongBench, plus token-vs-span / keep-rate / span-size / signal ablations (§5–6). (4) An honest analysis of IMP's boundaries and the decisive open experiments (§7).

---

## 2. Related work

- **KV-cache eviction** keeps the cache but shrinks it: recency+sinks (StreamingLLM), attention mass (H2O, SnapKV, TOVA, ExpectedAttention), key-norm (Knorm), reconstructability (KVzip, FastKVzip). *Only applies to models with a KV cache; attention-based scorers collapse at ≥16k.*
- **Prompt compression** prunes low-information tokens from the text: LLMLingua / LongLLMLingua (perplexity, small LM), LLMLingua-2 (trained classifier), Selective-Context (self-information). *Architecture-agnostic; our closest neighbors.*
- **Token merging** (ToMe) merges similar tokens layer-internally; our input-side merge is the naive baseline that destroys needles.
- **Retrieval (RAG)** retrieves relevant passages (BM25 / dense). *Strong when the answer is lexically anchored; fails on abstractive inputs.*
- **Linear attention / SSM / hybrids** (Mamba, Gated-DeltaNet, RWKV) replace the KV cache with a fixed recurrent state — the setting where IMP is uniquely applicable and R-MeeTo-style merge+retrain is relevant.
- **Long-context evaluation**: NIAH, RULER/RULERv2, LongBench(-v2), ∞Bench, HELMET, BABILong, NoLiMa.
- **Information theory**: MINE-style token importance; IMP's signals are cheap surrogates for I(token; answer/query).

IMP differs by (a) protecting the un-mergeable needle's *span* (vs ToMe), (b) being input-side and architecture-agnostic incl. cache-free models (vs KV-eviction), (c) using the base's own *semantic* importance (vs lexical RAG), and (d) combining multiple cheap signals (vs single-LM perplexity).

---

## 3. Observations (the failure map that motivates IMP)

*Frozen Qwen3-8B, public benchmarks, length-normalized log-likelihood for MC, EM/F1/ROUGE for gen; `full` = whole input, `no_ctx` = question only. Full numbers: `baseline-factbase-v2.0.0.md`.*

- **O1 — No universal baseline (F18).** At 16k / 50% budget the winner is regime-specific: retrieval favors kvzip/knorm/RAG/LLMLingua (0.8–1.0); **SnapKV collapses (0.17), ToMe scores 0.00**; abstractive (NarrativeQA): nothing beats full (~0.14); literary-MC (QuALITY): full *hurts* (0.08 < no_ctx).
- **O2 — Naive merging kills the needle (F14/F21).** ToMe = 0.00 on retrieval at every length; it only helps redundant content. ⇒ never merge the important token.
- **O3 — Cheap $O(L)$ signals localize the needle (F20).** Query-relevance AUROC **0.95** (word needle), surprisal **0.84** (numeric), **neither wins both**, norm-signals ≤0.34, length-invariant (8k≈16k).
- **O4 — Attention-free is length-robust; attention-KV collapses ≥16k (F1/F2).**

These say: decide *per input* what to keep, using an attention-free combination of cheap signals, and keep *spans* not tokens.

---

## 4. Method — IMP (Importance-routing Prefilter)

Given context tokens $c_{1..L}$, query $q$, keep fraction $p$, frozen base $F$:

**Score.** For each token $t$: query-relevance $\text{qrel}(t)=\cos(e(c_t),\bar e_q)$ (embedding-only, prefill-free) and surprisal $\text{surp}(t)=-\log p_F(c_t\mid c_{<t})$ (one forward). Combine z-scored: $s(t)=z(\text{qrel})(t)+z(\text{surp})(t)$ (extensible with reconstructability / redundancy).

**Route (span-level).** Split $c$ into contiguous $w$-token spans ($w\approx32$); score each span by its max token score; keep the top $\lfloor pL/w\rfloor$ spans **whole**, in reading order. (Token-level $w{=}1$ is the retrieval-only ablation that shreds coherent QA.)

**Read.** The frozen base reads $\text{embed}(c_S)$ in place of the full context.

**Deployment modes.** *Mode A — plug-and-play:* use the base's own signals directly (no training). *Mode B — light-weight:* train only a tiny distilled router head (combine signals) and, for linear/GDN bases, a verbatim side-cache (+brief LoRA) to counter state-collapse; the base stays frozen.

**Cost / honest caveat.** query-relevance is prefill-free; surprisal needs one forward — on a *quadratic* base this equals the prefill IMP claims to save, so the prefill win requires a **small draft-LM scorer** (score cheap, read big) or the query-only variant; on **linear** bases the forward is $O(L)$ so no conflict. Full algorithm + code: `imp-method-and-implementation.md`.

---

## 5. Experimental setup

- **Base:** Qwen3-8B (dense; a linear/GDN base is the key next setting). **Budget:** keep/ratio 0.5 unless swept. **N** = 48 (16 for 32k LongBench).
- **Benchmarks (~19):** synthetic retrieval — RULER 4k/8k/16k/32k, numerical/categorical/coding NIAH; realistic QA — SQuAD, HotpotQA, TriviaQA, MS-MARCO; long-doc — NarrativeQA, QuALITY, QuALITY-hard, MuSR, LoCoMo; **real-length LongBench (32k, untruncated)** — 2WikiMQA, HotpotQA, MuSiQue, MultiFieldQA, Qasper, NarrativeQA.
- **Baselines (~15):** window truncation; KV-eviction (kvzip, knorm, snapkv, h2o, streaming, expected, tova, criticalkv); prompt compression (LLMLingua-2, LLMLingua-orig, LongLLMLingua, Selective-Context); ToMe; BM25-RAG; references full / no_ctx. Faithfulness catalog in `baseline-catalog-faithfulness.md`.
- **Metrics:** each benchmark's native metric, scored its own way; report vs full (ceiling) and no_ctx (floor).

---

## 6. Results

*Snapshot of the 12h/6-GPU grand grid; full table in `baseline-factbase-v2.0.0.md §12`.*

**Main table (keep 0.5), selected rows (method acc):**

| bench | full | no_ctx | window | LLMLingua-2 | RAG | ToMe | kvzip | **IMP** |
|---|---|---|---|---|---|---|---|---|
| RULER-8k | 0.98 | 0.00 | 0.15 | 0.96 | 0.94 | **0.00** | 0.98 | 0.96 |
| numerical NIAH | 0.75 | 0.00 | 0.06 | 0.75 | 0.88 | 0.00 | 0.75 | 0.71 |
| categorical NIAH | 0.88 | 0.85 | 0.44 | 0.92 | **1.00** | 0.56 | 0.85 | 0.98 |
| SQuAD | 0.72 | 0.19 | 0.72 | 0.58 | **0.75** | 0.23 | 0.75 | 0.46 |
| HotpotQA | 0.61 | 0.27 | 0.60 | 0.52 | 0.59 | 0.40 | 0.58 | 0.42 |
| TriviaQA | 0.72 | 0.59 | 0.68 | **0.76** | 0.74 | 0.67 | — | 0.75 |
| NarrativeQA | 0.17 | 0.10 | 0.19 | 0.16 | 0.11 | 0.12 | — | 0.11 |

**Token vs span (the key ablation).** Token-level matches full on retrieval (RULER 1.0) but collapses coherent QA (SQuAD 0.15 < no_ctx). **Span-level rescues QA** (SQuAD 0.15→0.46, HotpotQA 0.21→0.42) while keeping retrieval (RULER 0.96–0.98), and beats full on TriviaQA (0.75>0.72) and categorical (0.98>0.88).

**Keep-rate curve.** Monotone: SQuAD 0.33 / 0.46 / 0.59 at keep 0.25 / 0.5 / 0.75. **Span size** 16–64 insensitive (~32 default).

**Real-length LongBench (32k, no truncation).** The frozen base scores low on real long docs; compression frequently **beats full**: HotpotQA IMP **0.27 > 0.17**, NarrativeQA(33k) IMP **0.13 > 0.12**, 2WikiMQA LLMLingua **0.33 > 0.19**. This is the necessity regime and is stronger evidence than synthetic NIAH.

**Signal ablation.** query-relevance wins lexical QA (SQuAD 0.48), surprisal wins numeric (0.79), **and equal-weight "both" is dragged down by the weak signal** (numerical 0.71 < 0.79) → motivates a *learned* router (Mode B).

---

## 7. Analysis & limitations (honest)

- **Where IMP wins:** retrieval = full at all lengths (vs ToMe 0.00, SnapKV collapse); denoising (beats full on trivia/categorical); real-length LongBench.
- **Where it does not (yet):** on **short-context extractive QA, IMP does not beat RAG / LLMLingua** (SQuAD 0.46 vs 0.75 / 0.58); window is a strong trivial baseline when answers are local; NarrativeQA/MuSR are at the base's ceiling (IMP can't fix a weak base).
- **IMP's real case** is therefore *not* raw short-context accuracy but **generality (architecture-agnostic incl. cache-free linear), efficiency (with a small scorer), denoising, and the necessity regime** — which must be *demonstrated*.
- **Threats:** prefill-saving is conditional (small-scorer needed on quadratic bases); equal-weight signal sub-optimal; not yet run on a linear base.
- **Critical question — "is IMP just a worse RAG?"** Defensible only if (a) it works on cache-free linear models, (b) it wins when the input overruns the window, (c) semantic importance helps where lexical BM25-RAG fails. These are the make-or-break experiments (§8).

---

## 8. Conclusion & next

IMP turns a cheap, observable fact — the answer token is $O(L)$-localizable — into a lightweight, frozen-base, architecture-agnostic long-context method that keeps the needle's span and sheds redundancy. It matches full on retrieval, rescues coherent QA, and wins by denoising on real long documents, while honestly not yet beating RAG on short extractive QA. Next, in priority: **(1) a linear/GDN base** (IMP's unique territory), **(2) the necessity regime** (LongBench/∞Bench >100k, full impossible), **(3) semantic > lexical** (abstractive inputs where RAG fails); then a small draft-LM scorer (real prefill saving) and a learned router (Mode B).

*Provenance: `matrix-facts.md` (F1–F26), `baseline-factbase-v2.0.0.md` (§10–§12.3), `imp-method-and-implementation.md`, `OVERVIEW-both-papers-and-facts.md`; grand-grid `gt_*` logs.*
