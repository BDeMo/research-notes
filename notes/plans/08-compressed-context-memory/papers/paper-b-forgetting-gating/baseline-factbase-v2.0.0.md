# A Black-Box Fact-Base for Long-Context Baselines

*Self-contained empirical reference for the v2.0.0 paper. Everything needed to read this file is defined below — no external context required.*

---

## Why this matters (motivation)

When an input is **too long to fit** (or too expensive to attend to in full), practitioners reach for a *long-context coping mechanism*: drop part of the KV cache, compress the prompt, or truncate to a window. These methods are deployed as **drop-in, training-free wrappers around a frozen model** — i.e. they treat the LLM as a **black box**. The field has produced a dozen such methods, each reporting wins on its own favorite benchmark, but there is **no apples-to-apples map of when each actually works**: at what compression budget, at what context length, and on what *kind* of task.

This document is that map. We run the major training-free baselines on **public benchmarks** under one harness, one frozen base model, and one scoring protocol, and report the raw behavior. The goal is not to crown a winner but to **establish the empirical facts** that motivate our method:

1. **No method is universally good.** The ranking *flips* depending on context length and task type (§Results 2–3).
2. **"More context" is not always better** — on a hard multiple-choice set, the full context actively *hurts* accuracy (§Results 3).
3. **The cheapest method (key-norm eviction) is the most length-robust on retrieval**, while the "smart" attention-based methods collapse at long context (§Results 2).

These facts are the foundation for the dive-in study and directly motivate an **adaptive / content-aware memory** rather than any single fixed compressor.

---

## Setup (how to read every number)

- **Base model (the black box):** `Qwen/Qwen3-8B` (dense attention, has a KV cache). A second base, `Qwen/Qwen3.5-9B`, uses **Gated DeltaNet (GDN)** linear-attention layers and therefore **has no KV cache** — KV-eviction methods are *not applicable* to it; only prompt/window compression are. This itself is a structural fact (§Results 6).
- **Training-free / eval-only.** None of these baselines train any weights. They are applied at inference on the frozen base. (Our own trained method is reported elsewhere; this file is the *baseline* reference.)
- **Public benchmarks only** (per the project data policy — internal CMG data is never used for paper numbers).
- **Sample size:** N = 64–96 validation items per cell (KV cells N=64; prompt/window cells N=96).
- **Total:** 201 cells across two sweeps (wave-1 = 102, wave-2 = 99).

**The three scores in every table:**
| symbol | meaning |
|---|---|
| **`method`** | accuracy of the baseline being tested (the number of interest) |
| **`full`** | uncompressed **full-context** reference — the frozen base reading the entire input. This is the ceiling a compressor tries to preserve. |
| **`no_ctx`** | **no-context floor** — the base answering with the question only, no document. Tells us how much the task needs the context at all. |

**Compression ratio** = fraction of the KV cache *evicted* (higher = more aggressive; `r0.9` keeps only 10%). For LLMLingua-2 the rate is the *retained* prompt fraction. **Context length** is set by the number of filler chunks; the RULER mapping is: `nc11 ≈ 2k`, `nc22 ≈ 4k`, `nc44 ≈ 8k`, `nc88 ≈ 16k`, `nc176 ≈ 32k` tokens.

---

## The baselines (definitions + references)

All KV methods are run via NVIDIA's [**kvpress**](https://github.com/NVIDIA/kvpress) library through its official pipeline. "attn?" = whether the method needs the attention matrix (these run on eager attention; the rest run on fast SDPA).

| baseline | family | one-line definition | ref |
|---|---|---|---|
| **SnapKV** | KV-eviction (attn) | Keeps the KV entries most attended-to by the last "observation window" of query tokens. | [Li et al. 2024](https://arxiv.org/abs/2404.14469) |
| **H2O** (`ObservedAttention`) | KV-eviction (attn) | "Heavy-Hitter Oracle" — keeps tokens with the highest accumulated attention mass. | [Zhang et al. 2023](https://arxiv.org/abs/2306.14048) |
| **ExpectedAttention** | KV-eviction (attn) | Estimates each token's *expected future* attention from query statistics and evicts the lowest. | [kvpress (NVIDIA)](https://github.com/NVIDIA/kvpress) |
| **TOVA** | KV-eviction (attn) | "Token Omission Via Attention" — at each step drop the single lowest-attention token (multi-state-RNN view). | [Oren et al. 2024](https://arxiv.org/abs/2401.06104) |
| **StreamingLLM** | window + sinks (no attn) | Keep only the first few "attention-sink" tokens + a recent sliding window; evict the middle. | [Xiao et al. 2023](https://arxiv.org/abs/2309.17453) |
| **Knorm** | norm-based (no attn) | Evict KV pairs with the **largest key L2-norm** (cheap, attention-free heuristic). | [Devoto et al. 2024](https://arxiv.org/abs/2406.11430) |
| **LLMLingua-2** | prompt compression | A trained token-classifier *deletes low-information tokens from the prompt text* before the model reads it (model-agnostic). | [Pan et al. 2024](https://arxiv.org/abs/2403.12968) |
| **Window truncation** | input truncation | Simply feed only the **last W tokens** of the input. The trivial baseline; bounds how "local" each task is. | (StreamingLLM-style, no sinks) |
| **PyramidKV** *(excluded)* | KV-eviction | Allocates a per-layer pyramidal KV budget. **Crashed on Qwen3** (shape mismatch) — see §Excluded. | [Cai et al. 2024](https://arxiv.org/abs/2406.02069) |
| **AdaKV** *(excluded)* | KV-eviction | Adaptively reallocates a base press's budget across heads. **All-zero on our setup** — see §Excluded. | [Feng et al. 2024](https://arxiv.org/abs/2407.11550) |

---

## The benchmarks (definitions + references)

| bench | type | what it measures | ref |
|---|---|---|---|
| **RULER / NIAH** | synthetic retrieval | "Needle-in-a-Haystack": find a planted fact in a long filler context. Length-controllable → our length-scaling axis. | [Hsieh et al. 2024](https://arxiv.org/abs/2404.06654) |
| **SQuAD v2** | extractive QA | Answer (or abstain) from a single paragraph. Answer is **local**. | [Rajpurkar et al. 2018](https://arxiv.org/abs/1806.03822) |
| **HotpotQA** | multi-hop QA | Combine facts spread across multiple paragraphs. Answer is **distributed**. | [Yang et al. 2018](https://arxiv.org/abs/1809.09600) |
| **NarrativeQA** | long-doc QA | Answer questions about long stories/scripts (abstractive; intrinsically hard — `full` is low). | [Kočiský et al. 2018](https://arxiv.org/abs/1712.07040) |
| **QuALITY (`quality_hard`)** | long-doc multiple-choice | Hard subset of multiple-choice reading comprehension over long passages. | [Pang et al. 2022](https://arxiv.org/abs/2112.08608) |
| **TriviaQA** | open QA + evidence | Trivia questions with (noisy) evidence documents. | [Joshi et al. 2017](https://arxiv.org/abs/1705.03551) |
| **multi_needle / numerical NIAH** | synthetic retrieval | Multi-fact / numeric variants of NIAH. *(multi_needle scoring is currently broken — see §Excluded.)* | (NIAH variants) |

---

## Results

> Reading guide: compare **`method`** against its row/column **`full`** (ceiling) and **`no_ctx`** (floor). A method "preserves" full context when `method ≈ full`; it has "collapsed" when `method → no_ctx` or 0.

### 1. KV-compression budget curve — RULER-8k retrieval (full = 1.00)

How accuracy decays as we evict more of the KV cache. `r0.1` keeps 90% of the cache; `r0.9` keeps 10%.

| method | attn? | r0.1 | r0.25 | r0.5 | r0.75 | r0.9 |
|---|---|---|---|---|---|---|
| **snapkv** | Y | 0.92 | 0.81 | 0.55 | 0.20 | 0.03 |
| **h2o** | Y | 0.86 | 0.47 | 0.11 | 0.05 | — |
| **streaming** | N | 0.89 | 0.78 | 0.56 | 0.30 | 0.09 |
| **expected** | Y | 0.96 | 0.93 | 0.40 | 0.02 | 0.00 |
| **tova** | Y | 0.92 | 0.69 | 0.25 | 0.04 | 0.02 |
| **knorm** | N | 0.96 | 0.95 | 0.81 | 0.23 | 0.01 |

*Reading:* all methods degrade monotonically with eviction. **Knorm and Expected hold best at light budgets; H2O degrades fastest.** Past ~50% eviction everything falls apart on retrieval — the needle's KV entry gets dropped.

### 2. Length-scaling crossover — RULER @ fixed ratio 0.5 (full = 1.00 at every length)

**The headline fact.** Hold the compression budget fixed and grow the context.

| method | attn? | 2k | 4k | 8k | 16k | 32k |
|---|---|---|---|---|---|---|
| **knorm** | N | 0.74 | 0.85 | 0.81 | 0.89 | 0.64 |
| **streaming** | N | 0.49 | 0.47 | 0.55 | 0.45 | 0.47 |
| **snapkv** | Y | 0.53 | 0.52 | 0.56 | **0.15** | — |
| **expected** | Y | 0.15 | 0.35 | 0.40 | **0.06** | — |
| **tova** | Y | 0.28 | 0.32 | 0.25 | — | — |
| **h2o** | Y | 0.15 | 0.14 | 0.11 | — | — |
| *full (ref)* | — | 1.00 | 1.00 | 1.00 | 1.00 | — |

*Reading:* **attention-based methods (SnapKV, Expected) match the cheap ones up to 8k, then collapse at 16k** (SnapKV 0.56 → 0.15). The **attention-free methods (Knorm, StreamingLLM) are length-robust out to 32k.** Knorm — the *cheapest* method (just key norms, no attention) — is the strongest at long retrieval. The full-context base, by contrast, scores a perfect 1.00 at every length up to 16k: the failure is the *compressor's*, not the model's.

### 3. KV method × task-type @ ratio 0.5 (the ranking flips)

Same methods, same budget, different task families.

| method | RULER-8k | squad_v2 | hotpot_qa | narrativeqa | quality_hard | trivia_qa |
|---|---|---|---|---|---|---|
| **snapkv** | 0.55 | 0.30 | 0.45 | 0.17 | 0.12 | 0.72 |
| **h2o** | 0.11 | 0.63 | 0.54 | 0.17 | 0.14 | 0.74 |
| **streaming** | 0.56 | 0.41 | 0.44 | 0.16 | 0.12 | 0.63 |
| **expected** | 0.40 | 0.61 | 0.49 | 0.18 | 0.11 | — |
| **tova** | 0.25 | 0.50 | 0.44 | 0.18 | 0.09 | — |
| **knorm** | **0.81** | 0.20 | 0.20 | 0.15 | 0.20 | 0.47 |
| *full (ref)* | 1.00 | 0.66 | 0.53 | 0.15 | 0.09 | — |

*Reading:* the ranking **inverts by task**. **Knorm wins retrieval (0.81) but is *worst* on extractive/multi-hop QA (squad 0.20, hotpot 0.20)**; the attention-based methods (H2O/Expected) are the reverse. No single KV method dominates → a fixed compressor is the wrong abstraction. **Note `quality_hard`: `full` (0.09) is *below* `no_ctx` (0.23, see §6) — full context HURTS** on this hard MC set.

### 4. Prompt compression — LLMLingua-2 (compare `method` to `full`/`no_ctx`)

Rate = fraction of prompt tokens retained.

| bench | full | no_ctx | r0.1 | r0.25 | r0.33 | r0.5 | r0.67 |
|---|---|---|---|---|---|---|---|
| **squad_v2** | 0.66 | 0.15 | 0.24 | 0.37 | 0.42 | 0.53 | 0.59 |
| **hotpot_qa** | 0.53 | 0.19 | 0.28 | 0.35 | 0.36 | 0.48 | 0.51 |
| **narrativeqa** | 0.15 | 0.11 | 0.17 | 0.16 | 0.17 | 0.16 | 0.16 |
| **quality_hard** | 0.09 | 0.23 | 0.17 | 0.17 | 0.14 | 0.15 | 0.12 |
| **ruler_niah** | 1.00 | 0.00 | 0.12 | 0.54 | 0.80 | 0.98 | 1.00 |
| **trivia_qa** | 0.70 | 0.49 | 0.67 | 0.73 | — | 0.73 | 0.73 |

*Reading:* prompt compression degrades **gracefully** on retrieval (RULER r0.67 → 1.00, far gentler than KV eviction) because it keeps whole informative spans. On **trivia it *beats* full context** (r0.25 → 0.73 > 0.70): removing distractor tokens **denoises** the prompt.

### 5. Window truncation (last W tokens) — the locality probe

| bench | full | w128 | w256 | w512 | w1024 | w2048 | w4096 |
|---|---|---|---|---|---|---|---|
| **squad_v2** | 0.66 | 0.45 | 0.66 | 0.66 | 0.66 | 0.66 | 0.66 |
| **hotpot_qa** | 0.53 | 0.26 | 0.29 | 0.42 | 0.52 | 0.53 | 0.53 |
| **narrativeqa** | 0.15 | 0.14 | 0.14 | 0.14 | 0.13 | 0.14 | 0.14 |
| **ruler_niah** | 1.00 | 0.07 | — | — | — | — | 1.00 |

*Reading:* the trivial "keep the last W tokens" baseline is **lossless on SQuAD at W=256** (the answer is local) but **fails on retrieval** (RULER w128 = 0.07 — the needle is usually outside the window) and needs the full W on multi-hop. This **quantifies the locality assumption** each task makes.

### 6. Qwen3.5-9B (GDN linear attention — no KV cache → KV-press N/A)

KV-eviction baselines cannot run here; only prompt/window compression apply.

| bench | full | no_ctx | LLMLingua r0.25 | r0.5 | window w512 | w1024 |
|---|---|---|---|---|---|---|
| **squad_v2** | 0.64 | 0.24 | 0.43 | 0.62 | 0.64 | 0.64 |
| **hotpot_qa** | 0.58 | 0.24 | 0.43 | 0.55 | 0.41 | 0.54 |
| **ruler_niah** | 0.96 | — | 0.62 | 0.97 | — | — |
| **narrativeqa** | 0.15 | 0.11 | 0.16 | 0.17 | — | — |
| **quality_hard** | 0.09 | 0.20 | 0.14 | 0.09 | — | — |

*Reading:* the GDN base reads full 8k context well (RULER 0.96), and the **same `quality_hard` "context-hurts" effect reproduces** (full 0.09 < no_ctx 0.20) — it is a property of the *task*, not the architecture. Structurally, GDN exposes **only the prompt/window levers**, not the KV lever.

---

## Excluded / broken (not real results — do not cite)

- **PyramidKV** — throws a tensor-shape mismatch on every Qwen3 item (a kvpress/transformers incompatibility); all-zero output. Dropped.
- **AdaKV** — all-zero on our setup (it wraps SnapKV and inherits the same failure). Dropped.
- **`multi_needle_niah`** — `full` itself scores 0.0 everywhere → the benchmark's gold-matching scoring is broken, not the methods. Needs a scoring fix before any cell is trustworthy.

### Collapse / validity audit (Jul-w01, over 784 `RECIPE_EVAL` cells) — is `full` a sane ceiling?
The single diagnostic: **`full` (uncompressed) must be a sane ceiling**; if `full` is itself at the floor on a retrieval bench, the *scoring* collapsed, not the method. Result:

| bench | `full` range | verdict |
|---|---|---|
| ruler_niah | 0.55–1.00 | ✅ valid (full is the ceiling) |
| numerical_niah | 0.00–**0.81** | ✅ valid **when configured**; the `0.00` cells are old bad-config (drop) |
| categorical_niah | 0.00–**0.78** | ✅ valid when configured; drop the `0.00` cells |
| squad_v2 / hotpot_qa / trivia_qa | 0.51–0.74 / 0.53–0.63 / 0.71–0.72 | ✅ valid |
| musr_mm | 0.58 | ✅ valid |
| **narrativeqa** | 0.12–0.17 (> no_ctx 0.10) | ✅ **NOT collapse, but UNDER-scored** — gen-level check (Jul-w01): the base *does* produce real answers (e.g. Q"Abigail is what to Barabas?" gold"Daughter" → pred"Abigail is Barabas's daughter" = correct), but scores are deflated by (a) `rouge_l` vs a *short* reference penalizing the verbose/rambly raw base, (b) a **case-sensitive, gold-only substring fallback** (misses "Daughter"≠"daughter"), and (c) occasional real base failures (HTML-continuation loops from document markup). Scoring is uniform across methods → **relative comparison fair**; absolute ceiling is pessimistic. Report as *abstractive floor*, not head-to-head. **Fixable:** case-insensitive contains-any-reference / token-F1 would raise the absolute (optional; would require a consistent re-run). |
| **quality / quality_hard** | 0.05–0.20 / 0.09–0.11 (< no_ctx) | ✅ **NOT collapse** — valid loglik-MC; `full < no_ctx` (even below chance) is the real **F6 "full hurts"** finding. |
| locomo | 0.17–0.19 (> no_ctx 0.08) | ✅ genuine hard (full ≈ 2× no_ctx); low ceiling. |
| **coding_niah** | 0.00 (old) → **0.75 (Jul-w01, nc44/8k)** | ✅ **valid when configured** (corrected). The old `full=0.00` were bad-config cells; at nc44/8k `full=0.75`, LLMLingua 0.94, RAG 0.90 — discriminative. Drop only the old `0.00` cells. |
| old `lc_*`, `smoke_*`, `smk_*` | 0.00 / 0.06 | ❌ stale smoke/bad-config cells — ignore. |

**Rule for the paper:** cite only cells whose `full` is a sane ceiling. Excluded from any comparison table: `multi_needle_niah` and any `*_niah` cell with `full=0.00` (old bad-config). `narrativeqa` / `locomo` / `ms_marco` are kept but framed as *low-ceiling / abstractive-floor*; `quality*` framed as *"full-hurts"* (F6). `coding_niah`/`categorical_niah`/`numerical_niah` are valid at nc44/8k.

## 11. Expanded benchmark suite (Jul-w01) — 6 new benches, KV-free family (keep 0.5)

*Adds open-QA, reasoning, dialogue-memory, and code/category retrieval. `full`/`no_ctx` verified as sane ceilings (collapse audit above). N=48.*

| bench (type) | no_ctx | full | LLMLingua-2 | BM25-RAG | ToMe |
|---|---|---|---|---|---|
| **trivia_qa** (open QA) | 0.59 | 0.72 | **0.76** | 0.74 | 0.67 |
| **ms_marco** (passage QA) | 0.21 | 0.26 | **0.29** | 0.27 | 0.23 |
| **musr_mm** (multi-step MC) | 0.48 | 0.60 | 0.58 | — | 0.52 |
| **locomo** (long dialogue mem) | 0.08 | 0.20 | 0.17 | **0.22** | — |
| **coding_niah** (code retrieval) | 0.00 | 0.75 | **0.94** | 0.90 | — |
| **categorical_niah** (cat. retrieval) | 0.85 | 0.88 | 0.92 | **1.00** | — |

**Reading:** (i) on the **retrieval-style** benches (coding/categorical NIAH) prompt-compression and RAG **beat `full`** (0.90–1.0 > 0.75–0.88) — removing distractors *denoises* the needle; (ii) LLMLingua-2 **beats `full` on trivia (0.76>0.72) and ms_marco (0.29>0.26)** — same denoising effect on lexical QA; (iii) **locomo/ms_marco are low-ceiling** (full 0.20/0.26) — genuinely hard, kept as floor evidence; (iv) ToMe consistently trails (0.67/0.23/0.52) — merging still hurts.

---

## Contributions of this fact-base (for the dive-in)

1. **Length × method crossover.** Attention-based KV eviction (SnapKV/Expected/H2O/TOVA) collapses at ≥16k; attention-free heuristics (Knorm/StreamingLLM) stay robust to 32k. The smartest method is not the most scalable.
2. **Task-dependent ranking flip.** Knorm tops retrieval but bottoms QA; attention-based methods do the reverse. **No fixed compressor is universally good → motivates an adaptive, content-aware memory.**
3. **"More context can hurt."** On `quality_hard`, full context underperforms no context on both bases — this is a **literary-MC base-capability ceiling**, **not** distractor volume (dive-C: rca full-read is distractor-robust; corrected from the earlier distractor reading).
4. **Locality is task-specific and measurable.** Window truncation pins down exactly how local each task is (SQuAD local; retrieval/multi-hop not).
5. **Compression family matters.** Prompt compression (LLMLingua-2) degrades far more gracefully than KV eviction on retrieval and can *denoise* (beats full on trivia); KV eviction is cheaper but brittle.
6. **The available levers depend on architecture.** Linear-attention/GDN bases have no KV cache, so KV-eviction methods simply do not exist for them.

*Provenance: generated from the raw `RECIPE_EVAL` log lines of the wave-1/wave-2 sweeps; see `decisions-2026-06-24.md` entries D12 and D13 for the exact experiment matrices and the harness fixes.*


---

## 8. Full method catalog (177-cell uniform sweep, NVAL=48 KV / 64 refs)

*Apples-to-apples re-run of all reproduced methods on one harness. Faithfulness annotations in `baseline-catalog-faithfulness.md`.*


### 8.1 Master budget curve — RULER-8k retrieval, all 20 methods × ratio (full=1.00)

| method | r0.1 | r0.25 | r0.5 | r0.75 | r0.9 |
|---|---|---|---|---|---|
| **snapkv** | 0.94 | 0.81 | 0.52 | 0.19 | 0.02 |
| **h2o** | 0.90 | 0.50 | 0.15 | 0.06 | 0.00 |
| **streaming** | 0.92 | 0.81 | 0.58 | 0.29 | 0.10 |
| **expected** | 0.98 | 0.98 | 0.40 | 0.00 | 0.00 |
| **tova** | 0.90 | 0.71 | 0.19 | 0.04 | 0.02 |
| **knorm** | 0.98 | 0.98 | 0.81 | 0.19 | 0.00 |
| **criticalkv** | 0.98 | 0.98 | 0.98 | 0.56 | 0.06 |
| **cur** | 0.98 | 0.98 | 0.98 | 0.12 | 0.00 |
| **kvzip** | 0.98 | 0.98 | 0.98 | 0.98 | 0.83 |
| **fastkvzip** | 0.98 | 0.98 | 0.98 | 0.98 | 0.65 |
| **lagkv** | 0.98 | 0.98 | 0.98 | 0.88 | 0.10 |
| **leverage** | 0.98 | 0.98 | 0.98 | 0.90 | 0.19 |
| **keydiff** | 0.98 | 0.96 | 0.90 | 0.21 | 0.04 |
| **compactor** | 0.98 | 0.94 | 0.96 | 0.69 | 0.02 |
| **noncausal** | 0.96 | 0.88 | 0.50 | 0.06 | 0.00 |
| **random** *(control)* | 0.98 | 0.94 | 0.46 | 0.00 | 0.00 |
| **rerotate** | 0.94 | 0.83 | 0.54 | 0.23 | 0.02 |
| **think** *(channel-pruning axis)* | 0.98 | 0.98 | 0.98 | 0.19 | 0.00 |
| **block** | 0.98 | 0.98 | 0.81 | 0.19 | 0.00 |
| **simlayer** *(threshold, ratio N/A)* | — | — | 0.98 | — | — |

### 8.2 Method × benchmark @ ratio 0.5

| method | narrativeqa | quality_hard | numerical_niah |
|---|---|---|---|
| **snapkv** | 0.12 | 0.12 | 0.62 |
| **h2o** | 0.13 | 0.17 | 0.12 |
| **streaming** | 0.17 | 0.12 | 0.29 |
| **expected** | 0.14 | 0.15 | 0.73 |
| **tova** | 0.13 | 0.17 | 0.67 |
| **knorm** | 0.16 | 0.19 | 0.75 |
| **criticalkv** | 0.14 | 0.12 | 0.75 |
| **cur** | 0.13 | 0.15 | 0.73 |
| **kvzip** | 0.13 | 0.12 | 0.75 |
| **fastkvzip** | 0.13 | 0.12 | 0.75 |
| **lagkv** | 0.15 | 0.12 | 0.75 |
| **leverage** | 0.15 | 0.12 | 0.75 |
| **keydiff** | 0.07 | 0.19 | 0.21 |
| **compactor** | 0.15 | 0.15 | 0.75 |
| **noncausal** | 0.15 | 0.17 | 0.75 |
| **random** | 0.13 | 0.19 | 0.35 |
| **rerotate** | 0.13 | 0.12 | 0.60 |
| **think** | 0.13 | 0.06 | 0.75 |
| **block** | 0.16 | 0.19 | 0.75 |
| **simlayer** | 0.13 | 0.12 | 0.75 |

### 8.3 Reference baselines on expanded benches (full / LLMLingua-2 r0.5 / window w1024)

| bench | full | LLMLingua r0.5 | window w1024 |
|---|---|---|---|
| **numerical_niah** | 0.00 | 0.00 | 0.00 |
| **categorical_niah** | 0.00 | 0.00 | 0.00 |
| **coding_niah** | 0.00 | 0.00 | 0.00 |
| **narrativeqa** | 0.16 | 0.17 | 0.14 |
| **quality** | 0.06 | 0.14 | 0.11 |
| **musr_mm** | 0.58 | 0.56 | 0.53 |
| **locomo** | 0.17 | 0.15 | 0.10 |

*All 177 cells completed (no failures). Excluded methods (pyramidkv/adakv/qfilter/duo/finch/chunk) per §catalog faithfulness doc.*



---

## 9. Necessity length-sweep + RAG (retrieve-vs-compress) — method-agnostic

*Training-free on Qwen3-8B. RULER lengths nc11/22/44/88/176 = 2k/4k/8k/16k/32k.*


### 9.1 Feasible baselines vs length (ruler_niah)

| baseline | 2k | 4k | 8k | 16k | 32k |
|---|---|---|---|---|---|
| **full (untrunc)** | 0.86 | 0.95 | 0.84 | 0.98 |  —  |
| **trunc-8k** | 0.86 | 0.84 | 0.92 | 0.55 | 0.84 |
| **window-1k** | 0.44 | 0.59 | 0.16 | 0.03 | 0.59 |
| **LLMLingua-0.5** | 0.81 | 0.89 | 0.92 | 0.88 |  —  |

*window-1k collapses with length (0.44->0.03); trunc-8k drops at 16k (0.92->0.55); full + LLMLingua hold; 32k-untrunc OOMs.*


### 9.2 BM25 RAG vs length (budget 2048)

| bench | 2k | 4k | 8k | 16k |
|---|---|---|---|---|
| **ruler NIAH** | 0.77 | 0.98 | 0.85 | 0.94 |
| **numerical NIAH** | 0.75 | 0.77 |  —  |  —  |
| **categorical NIAH** | 0.52 | 0.73 | 0.79 | 0.81 |

*RAG is length-robust (BM25 finds the needle regardless of length) — opposite of window/trunc.*


### 9.3 RAG vs full on real benches (budget 2048)

| bench | RAG | full | no_ctx |
|---|---|---|---|
| **squad_v2** | 0.75 | 0.72 | 0.19 |
| **hotpot_qa** | 0.59 | 0.61 | 0.27 |
| **trivia_qa** | 0.74 | 0.72 | 0.56 |
| **narrativeqa** | 0.11 | 0.16 | 0.10 |

*RAG matches/beats full on lexical/extractive (squad 0.75>0.72, trivia 0.74>0.72; denoising) but collapses on narrativeqa (0.11~=no_ctx): abstractive answers have nothing to lexically retrieve -> that diffuse/abstractive regime is the compression niche (cf. quality over-window 3.5x).*


---

## 10. KV-free compression family (Jul-w01) — the direct competitors to a soft-token memory

*KV-eviction edits a cache and does not exist on linear/GDN bases; the **KV-free** methods reduce the input/representation and are the real peer set for our method (and the only levers on GDN). All on Qwen3-8B, keep/budget 0.5, `run_baseline.py`; 39/42 cells (3 hardest failed: 16k perplexity-LLMLingua OOM on the Llama-2-7b compressor, and Selective-Context×narrativeqa timed out — resource limits, not method failures).*

### 10.1 Task axis (nc8 ≈ 4k; method accuracy)

| method | family | squad_v2 | hotpot_qa | narrativeqa | quality |
|---|---|---|---|---|---|
| **LLMLingua-2** | prompt (classifier) | 0.58 | 0.52 | 0.16 | 0.19 |
| **LLMLingua (orig)** | prompt (perplexity) | 0.62 | 0.59 | 0.16 | 0.15 |
| **LongLLMLingua** | prompt (q-aware ppl) | 0.35 | 0.59 | 0.16 | 0.15 |
| **Selective-Context** | prompt (self-info) | 0.49 | 0.46 | — | 0.16 |
| **ToMe** | input merge | 0.23 | 0.40 | 0.12 | 0.21 |
| **BM25-RAG** | retrieval | **0.75** | **0.59** | 0.11 | 0.08 |
| *full (ref)* | — | 0.72 | 0.61 | 0.17 | 0.12 |
| *no_ctx (ref)* | — | 0.19 | 0.27 | 0.10 | 0.23 |

### 10.2 Length axis (ruler_niah retrieval, keep 0.5)

| method | 4k | 8k | 16k |
|---|---|---|---|
| **LLMLingua-2** | 0.94 | 0.96 | 0.96 |
| **LLMLingua (orig)** | 1.00 | 0.98 | (OOM) |
| **LongLLMLingua** | 1.00 | 0.98 | (OOM) |
| **Selective-Context** | 1.00 | 1.00 | 1.00 |
| **ToMe** | 0.00 | 0.00 | 0.00 |
| **BM25-RAG** | 0.98 | 0.94 | 0.94 |

### 10.3 Reading (KV-free family facts)
1. **Prompt-compression is length-robust on retrieval (0.94–1.00 across 4k–16k)** — the opposite of attention-KV eviction, which collapses at 16k (§2). Selective-Context is perfectly length-flat (1.00). This is the KV-free family's structural advantage.
2. **ToMe fails retrieval outright (0.00 at every length)** — input-side merging destroys the needle (confirms F14/F17); it only helps redundant/semantic content (squad 0.23, quality 0.21).
3. **RAG leads extractive** (squad 0.75 > full 0.72; hotpot 0.59) but **collapses on abstractive** (narrativeqa 0.11 ≈ no_ctx) — no lexical anchor.
4. **Abstractive (narrativeqa): every KV-free method ≈ full ≈ 0.16** — nothing helps; the base ceiling is the wall.
5. **Literary-MC (quality): full *hurts* (0.12 < no_ctx 0.23); all compressors sit near no_ctx** (ToMe 0.21 highest via denoising).
6. **`LongLLMLingua` underperforms plain `LLMLingua-orig` on squad (0.35 vs 0.62)** — the question-aware reranking can over-prune a short local answer span.

*Provenance: `kvf_*` logs on d1525; faithfulness per `baseline-catalog-faithfulness.md` (all EXACT except ToMe=input-side adapted, RAG=generic). Selective-Context newly reproduced this week (authors' pkg + spaCy `en_core_web_sm` + gpt2 self-info).*

## 12. IMP (our method, Paper B) — Mode A token-level, keep 0.5

*Plug-and-play importance-routing prefilter (frozen base, no training): score each ctx token by `z(query-relevance)+z(surprisal)` (F20), keep top-p verbatim, drop the rest. `imp_*` logs on d1525. This is the **token-level** variant (Mode A); the **span-level** variant (Mode A.2) is under test.*

| bench (type) | no_ctx | full | **IMP (ours)** | ToMe | vs full |
|---|---|---|---|---|---|
| **ruler_niah 4k** | 0.00 | 1.00 | **1.00** | 0.00 | ✅ = full |
| **ruler_niah 8k** | 0.00 | 0.98 | **0.98** | 0.00 | ✅ = full |
| **ruler_niah 16k** | 0.00 | 1.00 | **1.00** | 0.00 | ✅ = full (length-robust) |
| **numerical_niah** | 0.00 | 0.75 | **0.75** | (0.00) | ✅ = full |
| **coding_niah** | 0.00 | 0.75 | 0.73 | — | ✅ ≈ full |
| **categorical_niah** | 0.85 | 0.88 | 0.75 | — | ↓ slightly |
| **squad_v2** | 0.19 | 0.72 | **0.15** | 0.23 | ❌ < no_ctx |
| **hotpot_qa** | 0.27 | 0.61 | **0.21** | 0.40 | ❌ < no_ctx |
| **trivia_qa** | 0.59 | 0.72 | 0.58 | 0.67 | ❌ ≈ no_ctx |
| **narrativeqa** | 0.10 | 0.17 | 0.11 | 0.12 | ❌ ≈ no_ctx |
| **quality** (MC) | 0.23 | 0.12 | 0.21 | 0.21 | (full hurts; IMP≈no_ctx) |

**Reading (a clean token-vs-span ablation story):**
1. **IMP dominates retrieval** — ruler 1.0 at *every* length incl. 16k, numerical/coding ≈ full, **where ToMe = 0.00**. Keeping the needle verbatim works and is length-robust (contrast F1 attention-KV collapse). This is the headline positive.
2. **IMP (token-level) collapses on coherent QA** — squad/hotpot/trivia/narrativeqa fall to ≈ no_ctx or below. **Diagnosis:** per-token top-p selection **shatters sentence structure** — the answer lives in a coherent span, so scattered tokens (minus connectives/local context) *distract* the base. Contrast RAG (squad 0.75) / LLMLingua (0.58) which keep coherent spans.
3. ⇒ **Token-level IMP = a strong retrieval-only baseline; a general method must operate at span level** (keep the needle's *span*, merge redundant spans) — the Mode A.2 fix, now validated below.

### 12.1 Token-level vs span-level IMP (Mode A.2) — the fix works

*Span-level = same per-token scores, but keep whole top-scoring 32-token spans (preserves local coherence). keep 0.5. Token-level N=48, span-level N=32 (trend ≫ noise).*

| bench | no_ctx | full | IMP token-level | **IMP span-level (32)** | Δ (span − token) |
|---|---|---|---|---|---|
| **ruler_niah 8k** | 0.00 | 0.97 | 0.98 | **0.97** | ≈ 0 (retrieval preserved) |
| **squad_v2** | 0.19 | 0.73 | 0.15 | **0.49** | **+0.34** |
| **hotpot_qa** | 0.25 | 0.63 | 0.21 | **0.43** | **+0.22** |
| **narrativeqa** | 0.09 | 0.18 | 0.11 | 0.12 | +0.01 (base ceiling, F18) |

**Reading:** span-level **rescues coherent QA** (squad 0.15→0.49, hotpot 0.21→0.43 — from *below* no_ctx to *well above* it, approaching full) **while retrieval stays = full** (ruler 0.97). Confirms the diagnosis: token-level shreds the answer sentence; keeping the whole span restores coherence without losing the needle. → **F24**.

## 12.2 IMP full sweep (span-level, N=48) — main table, budget curve, span-size

*`spf_*`/`spk*`/`sp16_*`/`sp64_*` on Qwen3-8B. Fig 13.*

### A. token- vs span-level (keep 0.5) — the main table
| bench | no_ctx | full | IMP token | **IMP span (32)** |
|---|---|---|---|---|
| ruler_niah 4k | 0.00 | 1.00 | 1.00 | 0.98 |
| ruler_niah 8k | 0.00 | 0.98 | 0.98 | 0.96 |
| ruler_niah 16k | 0.00 | 1.00 | 1.00 | 0.98 |
| numerical_niah | 0.00 | 0.75 | 0.75 | 0.71 |
| coding_niah | 0.00 | 0.75 | 0.73 | 0.75 |
| categorical_niah | 0.85 | 0.88 | 0.75 | **0.98** |
| squad_v2 | 0.19 | 0.72 | 0.15 | **0.46** |
| hotpot_qa | 0.27 | 0.61 | 0.21 | **0.42** |
| trivia_qa | 0.59 | 0.72 | 0.58 | **0.75** |
| narrativeqa | 0.10 | 0.17 | 0.11 | 0.11 |
| quality (MC) | 0.23 | 0.12 | 0.21 | 0.15 |

**Reading:** span-level keeps retrieval ≈ full (ruler 0.96–0.98 incl. 16k), **rescues QA** (squad/hotpot/trivia), and **beats full on trivia (0.75>0.72) & categorical (0.98>0.88)** by denoising. narrativeqa is at its ceiling (full 0.17). quality: full hurts (F6), IMP ≈ no_ctx.

### B. keep-rate budget curve (span 32; method acc)
| bench | 0.25 | 0.50 | 0.75 | full |
|---|---|---|---|---|
| ruler_niah 8k | 0.69 | 0.96 | 0.98 | 0.98 |
| numerical_niah | 0.58 | 0.71 | 0.75 | 0.75 |
| squad_v2 | 0.33 | 0.46 | 0.59 | 0.72 |
| hotpot_qa | 0.43 | 0.42 | 0.57 | 0.61 |

Monotone in budget (a clean accuracy–cost trade-off); by keep 0.75, retrieval is saturated and QA is within ~0.1 of full at 25% token savings.

### C. span-size ablation (keep 0.5; method acc)
| bench | span 16 | span 32 | span 64 | full |
|---|---|---|---|---|
| squad_v2 | 0.40 | 0.46 | 0.44 | 0.72 |
| hotpot_qa | 0.50 | 0.42 | 0.53 | 0.61 |

Span size is **insensitive** (0.40–0.53); ~32 is a fine default (multi-hop hotpot slightly prefers larger spans — distributed evidence). → **F25**.

## 12.3 Paper A blocker — learned compressor can't compress (squad, `cr_*`)
| recipe | compress | full | no_ctx |
|---|---|---|---|
| base (K256, 3k steps) | 0.195 | 0.69 | 0.17 |
| long (6k steps) | 0.193 | 0.69 | 0.17 |
| enc-LoRA (train encoder) | 0.132 | 0.69 | 0.17 |
| high-LR / no-KL | 0.160 | 0.69 | 0.17 |

**All ≈ no_ctx ≪ full** — the learned soft-memory carries almost no usable information on extractive QA, and longer/enc-LoRA/higher-LR make it *worse*. Training-free **IMP (span, squad 0.46) beats every learned-compressor recipe (≤0.20)**. This is Paper A's real blocker. → **F26**.

