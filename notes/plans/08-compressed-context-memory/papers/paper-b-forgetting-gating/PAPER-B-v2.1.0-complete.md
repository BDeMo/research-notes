# Paper B (v2.1.0) — *Observe long-context failures, then bolt a lightweight importance-routing structure onto a frozen base*

> Complete working spec: outline · main method (IMP) · experiment logic · insights · related work · difference claims.
> **Paper shape (per directive):** two parts — **(I) observe** the problems of long-context, **(II) a model-structure design method** that **extends an existing frozen base** in one of two deployment modes: **plug-and-play (no training)** or **light-weight training** (a tiny module; the base is never retrained).
> Emphasis: **generality + performance**, **impactful** via **efficient · simple · insightful · fast-reproducible checkpoint**, compute/data-frugal, covering **linear + quadratic** AR and **long-prefill**. **No gate** (that's Paper A). Facts by ID from `matrix-facts.md`.

## 0. One-line thesis
Long-context accuracy is bottlenecked by *finding the important information*, and every existing training-free baseline fails in a characterizable regime (**Part I: observation**). So we propose **IMP — a lightweight importance-routing structure that attaches to an existing frozen base** (**Part II: method**), deployable **plug-and-play (no training)** or via **light-weight training of a tiny module** (never retraining the base), that **keeps the un-reconstructable "needle" verbatim and merges redundant spans before the expensive read** — lifting long-context performance across **linear & quadratic** bases and shipping as a **fast-reproducible checkpoint**.

## 1. Motivation — the "which actually works" diagnosis (our headline evidence)
Head-to-head at **16k, 50% budget** (loglik-scored; `hh_*` logs, `matrix-facts` F18):
| regime (bench) | works | fails | why |
|---|---|---|---|
| retrieval (ruler/numerical) | kvzip, knorm, RAG, LLMLingua (0.8–1.0) | **snapkv 0.17, ToMe 0.00** | attention-KV collapses at 16k (F1); naive merge kills the needle (F14) |
| multi-hop (hotpot) | window, RAG, LLMLingua-2 (~0.5–0.6) | KV weaker | distributed evidence needs breadth |
| abstractive (narrativeqa) | **nothing** (all ≈ full ≈ 0.14) | all | no lexical anchor; base ceiling low |
| literary-MC (quality) | compression ≈ no-ctx (0.27–0.31) **> full (0.08)** | **full hurts** (F6) | long distractor MC degrades the base |
**⇒** No method is universal; the win is **regime-specific**, and the shared failure corner (long + abstractive + distractor-mixed) is the target. The method must **identify importance per input**, not apply a fixed policy.

## 2. Main method — IMP (Importance-routing structure on a frozen base)
> **Current version = `IMP-v2.1.0`** (span-level, span=32, keep=0.5, signals={query-relevance, surprisal}). All results in this paper's tables are this version. `IMP-v2.0` = token-level (retrieval-only, superseded, F24).

A lightweight **structural extension** inserted around a **frozen** base — not a text pre-processor and not a base retrain. Same structure, **two deployment modes**:

**Mode A — plug-and-play (no training).** Use the base's OWN cheap signals directly:
long ctx → **(a) cheap O(L) importance scoring** (attention-free, length-robust: query-relevance + surprisal + reconstructability) → **(b) keep the top-p most-important SPANS verbatim** (whole ~32-tok spans, not isolated tokens) + merge/drop redundant spans → **(c) frozen base reads the short sequence** (linear or quadratic).
- **Span-level is essential (validated ablation, §12.1):** token-level top-p matches `full` on retrieval (ruler 1.0 @16k, vs ToMe 0.00) but **shatters coherent QA** (squad 0.15, hotpot 0.21 — *below* no_ctx); **span-level rescues it** (squad 0.15→0.49, hotpot 0.21→0.43) **while keeping retrieval = full** — because the answer lives in a coherent span, not scattered tokens.

**Mode B — light-weight training (tiny module, base frozen).** Add and train ONLY:
- a **distilled importance-router head** (learns to combine the signals; trained on a small unlabeled corpus by forward-passes, à la FastKVzip's <1-H100-hr distill), and/or
- a **verbatim side-cache module** for **linear/GDN** bases (re-injects the very-top tokens to beat state-collapse, F10), with a **brief LoRA** to rebuild state key-knowledge (naive merge alone fails — F14 / R-MeeTo).

- **Importance signals (learned combination; E2/F20-confirmed):** **query-relevance** `query_dot` (AUROC 0.95 word-needles) + **surprisal** (0.84 numeric needles) + reconstructability (KVzip, F4) + redundancy (F16); *learned* combine because **no single signal wins across tasks** (F3/F20).

**Why it fits the directive:** attaches to an **existing frozen base** (plug-and-play OR light-train, never retrain) · O(L) prune before O(L²)/state read (**solves long-prefill**) · architecture-agnostic router + linear side-cache (**linear + quadratic**) · ship base + tiny module (**fast-reproducible checkpoint**) · one-sentence rule (**simple/insightful**).

## 3. Experiment logic (what proves what → main table)
| exp | proves | status |
|---|---|---|
| E1 diagnosis (which fails where) | motivates per-input importance; no universal baseline | ✅ (head-to-head + fact-base F1–F17) |
| E2 signal identification (X-C8, `run_probe.py`) | ✅ **query_dot AUROC 0.95 (word-needle), surprisal 0.84 (numeric), neither wins both, length-invariant (F20)** → router inputs = {query_dot, surprisal}+reconstructability, learned combine | ✅ |
| E3 training-free IMP (Mode A) | ✅ **retrieval = full at every length (ruler 1.0 @16k, vs ToMe 0.00); span-level rescues QA (squad 0.15→0.49, hotpot 0.21→0.43); token-vs-span is the key ablation (§12/§12.1, F24)** | ✅ |
| E4 distilled router > heuristic + transfers across bases | the checkpoint + generality claim | ⏳ |
| E5 linear side-cache + R-MeeTo closes the GDN retrieval gap | linear arm | ⏳ |
| E6 5×5 domain transfer (X-C1) | the compressed importance generalizes across domains | ⏳ (aggregate rows only) |
**Main table (FULL test sets — live in [`main-table-fulltest.md`](main-table-fulltest.md), F27):** 9 method cols (no_ctx, full, window, RAG, LLMLingua-2, ToMe, **IMP**, kvzip, knorm) × 16 benches. Long-context headline benches on the **FULL split** (incl. the two hardest: **LongBench-v2** 503, **∞Bench** ~229); short-context sanity on a disclosed N=500 subset (config: [`experiment-config-and-sampling.md`](experiment-config-and-sampling.md)). Headline reads at keep-0.5:
- **"more context hurts":** QuALITY `full` 7.2 < blind 17.9; **LongBench-v2 (hardest) `full` 33.6 ≈ blind 33.4** (~0 gain over guessing).
- **selection beats `full`:** ∞Bench RAG **64.2 > 53.7**, squad 71.5 > 69.0, trivia ll2 71.4 > 69.4, lb_hotpotqa RAG 24.8 > 22.6.
- **IMP (ours) uniquely lossless on retrieval:** **RULER-16k 96.8 ≈ full 99.0** while **ToMe→0.0, window→6.0** collapse; mid-pack on real long-doc QA (⇒ Mode-B light training).

Remaining axes to add: bases (Qwen3.5/3.6 GDN, 1.7B/14B/Qwen2.5/Llama for generality) and a prefill-tokens column. Claim = **IMP ≥ best feasible baseline on accuracy at a fraction of the prefill, across architectures.**

## 4. Insights (facts this rests on — `matrix-facts.md`)
F1 attn-KV collapses ≥16k · F2 attn-free length-robust · F3 ranking flips by task · F4 kvzip=reconstruction-importance, retrieval-only · F6 full-hurts on literary-MC · F9 RAG fails abstractive (lexical) · F11 failures size/family-invariant (⇒ transferable router) · F12 compressor is commodity · F14 naive merge kills needle · F18 head-to-head which-works.

## 5. Related work
- **Token merging:** ToMe (Bolya) — layer-internal bipartite merge; our input-side merge is the naive baseline that fails on needles (F14).
- **KV eviction:** SnapKV/H2O/StreamingLLM (query-aware/recency), **KVzip** (query-agnostic, reconstruction-importance), **FastKVzip** (distilled sink-attention gate).
- **Prompt compression:** LLMLingua/-2/Long — token pruning by (small-LM) informativeness.
- **Retrieval:** RAG/DPR/BM25 — lexical/dense retrieve-then-read.
- **Linear/SSM + reduction:** Mamba, Gated-DeltaNet (Qwen3.5/3.6), **R-MeeTo** (merge+re-train recovers Mamba's key-knowledge).
- **Information theory:** MINE (mutual-information estimation) — our token-importance = I(token;answer)/I(token;query).
- **Long-context eval:** RULER, ∞Bench, NoCha, LongBench-v2 (effective vs advertised length).

## 6. Difference / novelty (sharp)
1. **vs ToMe:** we don't merge blindly — we **protect the un-reconstructable needle** (F14/F17 show naive merge fails); merging is only for redundancy.
2. **vs KVzip/FastKVzip:** they evict KV *inside* one model's cache; **IMP is an input-side, architecture-agnostic prefilter** that works on **linear models too** (no KV cache) and **shrinks the prefill** — plus it **combines signals** (they use one). We *reuse* their reconstruction-importance as one signal.
3. **vs RAG:** IMP uses the **base's own semantic importance** (not lexical overlap) → works on **abstractive** inputs where BM25-RAG fails (F9).
4. **vs prompt compression:** learned multi-signal importance vs single-LM perplexity → keeps low-salience needles LLMLingua drops (F8).
5. **The impactful bundle:** a **transferable, tiny, distilled router** giving long-context gains across **architectures and sizes** (F11) from **little data**, as a **fast-reproducible checkpoint** — none of the above ships that.

## 7. Immediate next (continue unfinished)
1. **E2 — signal harvest + MINE (X-C8/X-C2):** `run_probe.py` dumps per-token {reconstructability, key-norm, query-dot, surprisal, redundancy} + AUROC vs answer-token; MINE I(token;answer). → pick the router's signals. **← launching next.**
2. **E3 — training-free IMP** (reconstruction-guided keep) vs ToMe/KV/RAG on necessity benches.
3. **E4/E5 — distilled router + GDN side-cache/R-MeeTo.**
4. **E6 — full 5×5 transfer (X-C1).**

*Provenance: `matrix-facts.md` (F1–F18), `v2.1.0-paperB-method-designs.md`, `baseline-diagnosis-report.md`, head-to-head `hh_*` + MC-fix `mcfix_*` logs; KVzip/FastKVzip/ToMe/R-MeeTo/MINE refs.*
