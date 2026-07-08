# Baseline Catalog & Faithfulness Notes (v2.0.0)

*Companion to `baseline-factbase-v2.0.0.md`. This file documents **every baseline we attempted to reproduce** from the literature-review knowledge base (`references-longcontext.md`), exactly **how faithfully** we ran it, and **what differs** from the original paper. Read this before quoting any baseline number.*

**Reproduction principle (per request):** methods must be **faithful**; hyper-parameters and the base model are **tunable** *only when the method is not tied to a specific base architecture*. When a method **requires a specific base** (special architecture, or precomputed per-model artifacts), we (a) describe it and skip, or (b) run an adapted version and **annotate the differences explicitly**.

**Shared setup:** frozen `Qwen/Qwen3-8B` (dense, has KV cache) unless noted; `Qwen/Qwen3.5-9B` (Gated-DeltaNet, **no KV cache**) for the architecture-contrast block. Training-free / eval-only. Public benches only. KV methods run via NVIDIA [`kvpress`](https://github.com/NVIDIA/kvpress) through its official pipeline. Compression ratio = fraction of KV **evicted** (tunable knob).

Legend: ✅ **faithful** (method as published, base-agnostic, ratio-tunable) · 🟡 **adapted** (runs, but differs from paper — see notes) · 🔴 **excluded** (cannot run faithfully on this base — described, not scored).

---

## 1. KV-cache eviction / selection (run on dense Qwen3-8B)

### ✅ Faithful (20 methods — base-agnostic, swept over compression_ratio)

| method (our key) | paper | family / mechanism | smoke@0.5 RULER-4k |
|---|---|---|---|
| **snapkv** | [SnapKV (Li 2024)](https://arxiv.org/abs/2404.14469) | observation-window attention votes select prefill KV | ✓ |
| **h2o** | [H2O (Zhang 2023)](https://arxiv.org/abs/2306.14048) | evict by cumulative ("heavy-hitter") attention | ✓ |
| **streaming** | [StreamingLLM (Xiao 2023)](https://arxiv.org/abs/2309.17453) | attention sinks + recent window | ✓ |
| **expected** | [ExpectedAttention (kvpress)](https://github.com/NVIDIA/kvpress) | evict by estimated *future* attention from query stats | ✓ |
| **tova** | [TOVA (Oren 2024)](https://arxiv.org/abs/2401.06104) | drop lowest-attention token each step (multi-state-RNN) | 0.42 |
| **knorm** | [Knorm / L2 (Devoto 2024)](https://arxiv.org/abs/2406.11430) | evict largest key-L2-norm (attention-free) | 0.92 |
| **criticalkv** | [CriticalKV (Wang 2025)](https://arxiv.org/abs/2502.03805) | two-stage "critical" KV selection (wraps SnapKV) | 1.00 |
| **cur** | CUR-decomposition press (kvpress) | low-rank CUR leverage-score selection | 1.00 |
| **kvzip** | [KVzip (Kim 2025)](https://arxiv.org/abs/2505.23416) | query-agnostic importance via reconstruction | 1.00 |
| **fastkvzip** | FastKVzip (kvpress) | layerwise fast variant of KVzip | 1.00 |
| **lagkv** | [LagKV (Liu 2025)](https://arxiv.org/abs/2504.04704) | lag-relative information ratio scoring | 1.00 |
| **leverage** | LeverageScore press (kvpress) | statistical leverage-score selection | 1.00 |
| **keydiff** | [KeyDiff (Park 2025)](https://arxiv.org/abs/2504.15364) | evict by key-vector diversity/similarity | 0.92 |
| **compactor** | Compactor press (kvpress) | leverage + non-causal sketch blending | 0.92 |
| **noncausal** | NonCausalAttn press (kvpress) | chunked non-causal attention scoring | 0.83 |
| **rerotate** | KeyRerotation press (kvpress) | re-rotate RoPE keys after eviction (wraps SnapKV) | 0.83 |
| **block** | BlockPress (kvpress) | block-contiguous selection (wraps Knorm scorer) | 0.92 |
| **random** | RandomPress (kvpress) | **control**: evict uniformly at random | 0.58 |

> All 18 above + the 2 already-validated (we also re-run snapkv/h2o/streaming/expected/tova/knorm uniformly) form the **20-method master budget table** in the fact-base. `random` is included deliberately as the lower-bound control — any method below it is worse than chance eviction.

### 🟡 Adapted (run, but differs from the original)

| method | paper | what we ran | difference vs paper (annotate) |
|---|---|---|---|
| **think** | [ThinK (Xu 2024)](https://arxiv.org/abs/2407.21018) | `key_channel_compression_ratio = ratio` | ThinK prunes KV **channels (the head dim), not tokens** — a *different compression axis*. Our "ratio" column for ThinK means channel-pruning fraction, **not** token-eviction fraction. Not directly comparable to token-eviction rows; reported in its own note. |
| **simlayer** | [SimLayerKV (Zhang 2024)](https://arxiv.org/abs/2410.13846) | paper defaults (`lazy_threshold` etc.) | SimLayerKV is **threshold-based** ("lazy" layer identification), **not** a compression_ratio method. We run it once at default settings; the ratio axis is **N/A** (tagged `rNA`). |

---

## 2. KV methods attempted but EXCLUDED (cannot run faithfully here)

| method | paper | why excluded (described, not scored) |
|---|---|---|
| 🔴 **pyramidkv** | [PyramidKV (Cai 2024)](https://arxiv.org/abs/2406.02069) | Throws a tensor-shape mismatch on every Qwen3 item (kvpress/transformers GQA incompatibility). All-zero ⇒ invalid. |
| 🔴 **adakv** | [Ada-KV (Feng 2024)](https://arxiv.org/abs/2407.11550) | Requires a non-eager attention path ("eager mode not supported"); our scoring needs eager ⇒ all-zero. Head-adaptive budget needs flash-attn integration we don't have. |
| 🔴 **critadakv** | CriticalAdaKV (kvpress) | Same "eager mode not supported" as Ada-KV. |
| 🔴 **qfilter** | [Q-Filters (Godey 2025)](https://arxiv.org/abs/2503.02812) | **Base-specific**: needs precomputed Q-filter vectors per model; **none published for Qwen3-8B** ("Could not load Q-filters for Qwen3-8B"). Faithful run impossible without training the filters ourselves. |
| 🔴 **duoattention** | [DuoAttention (Xiao 2024)](https://arxiv.org/abs/2410.10819) | **Base-specific**: needs offline-optimized retrieval-vs-streaming **head masks** per model. On-the-fly scoring (our only option) **OOMs** (duplicate KV, 40 GB alloc). Would need to train head masks for Qwen3-8B. |
| 🔴 **finch** | [Finch (Corallo 2024)](https://arxiv.org/abs/2410.01532) | Needs a prompt **delimiter token id** (document/question separator) that our harness doesn't set; designed as a prompt-conditioned compressor. Excluded pending delimiter wiring. |
| 🔴 **kvzap** | KVzap (kvpress) | Constructor needs internal `model_type` attributes not present for Qwen3; AttributeError. |
| 🔴 **chunk / chunkkv** | [ChunkKV (Liu 2025)](https://arxiv.org/abs/2502.00299) | Runs without error (wrapping a Knorm scorer, sdpa) but scores **0.0** on RULER at `chunk_length=20` — untuned/degenerate. `block` (same contiguous-region family) works (0.92) and is used as the representative; ChunkKV needs chunk-length tuning before it's trustworthy. |

---

## 3. Prompt / soft compression & memory (non-KV)

| method | paper | faithfulness | notes |
|---|---|---|---|
| ✅ **llmlingua2** | [LLMLingua-2 (Pan 2024)](https://arxiv.org/abs/2403.12968) | **EXACT** | official `llmlingua` pkg + the authors' `llmlingua-2-xlm-roberta-large-meetingbank` classifier; only the retain-rate is swept. |
| ✅ **rag (BM25)** | [RAG (Lewis 2020)](https://arxiv.org/abs/2005.11401) + BM25 retriever | **EXACT algorithm**, lexical-retriever variant | standard [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) retrieve-then-read. ⚠ canonical RAG uses a **DPR dense** retriever; ours is the **BM25 (lexical)** variant — cite as "**BM25-RAG**", not DPR-RAG. |
| ✅ **full / no_ctx** | — | reference | uncompressed ceiling / no-context floor. |
| 🟢 **sft-lora** | generic ("why not just LoRA the reader on full ctx?") | faithful (generic baseline) | not a specific paper's system; the standard SFT-LoRA-on-full-context baseline. |
| 🟠 **window (sliding-read)** | (generic truncation) | **generic — NOT StreamingLLM** | last-W tokens + 4 sink tokens fed to the frozen base = input truncation. This is **not** published [StreamingLLM](https://arxiv.org/abs/2309.17453) (which manages the *decode* KV-cache). **The exact StreamingLLM is the `streaming` kvpress press in §1.** Report this only as a generic sliding-window read. |
| 🔴 **gist-lite** | [Gisting (Mu 2023)](https://arxiv.org/abs/2304.08467) | **NOT faithful** — do NOT cite as Gisting | reuses **our** `compressor.mem` K-tokens + our read-LoRA → the *idea* reimplemented in our framework, not the authors' gist-token training. Exact repro needs the [authors' code](https://github.com/jayelm/gisting). |
| 🔴 **cartridge-lite** | [Cartridges (Eyuboglu 2025)](https://arxiv.org/abs/2506.06266) | **NOT faithful** — do NOT cite as Cartridges | reuses **our** `compressor.mem` as a fixed prefix → not the authors' self-study KV-cartridge training. Exact repro needs the authors' code. |
| 🟡 **LongLLMLingua / LLMLingua(orig) / Selective-Context** | [#37](https://arxiv.org/abs/2310.06839) / [#36](https://arxiv.org/abs/2310.05736) / [#29](https://arxiv.org/abs/2304.12102) | **being added now (EXACT, official pkgs)** | LongLLMLingua + original perplexity-LLMLingua = official `llmlingua` pkg (`use_llmlingua2=False`, question-aware); Selective-Context = official `selective-context` pkg. See D21. |
| 🔴 **ICAE / AutoCompressor / 500xCompressor / xRAG / Nugget** | (#22–25,30,42) | excluded (need trained encoders) | related work only; our own method is the trained-encoder comparison point. |
| 🔴 **RMT / Infini-attention / Titans** | (#43,45,47,150) | excluded (need arch changes) | related work only. |

> **Faithfulness verdict (per your requirement — public methods must be EXACT, not our-infra approximations):**
> - **EXACT / reportable as the published method:** the ~20 kvpress KV methods (§1) + **LLMLingua-2** + (being added) LongLLMLingua / LLMLingua-orig / Selective-Context. `think`/`simlayer` are exact algorithms but on a non-token-eviction axis (annotated §1).
> - **Generic baselines (cite the concept, not a specific system):** `full`, `no_ctx`, `sft-lora`, `window (sliding-read)`, `BM25-RAG`.
> - **NOT faithful — must NOT be presented as the published method:** `gist-lite` (Gisting), `cart-lite` (Cartridges). They are our-framework approximations; to use them as baselines we must run the **authors' official code**, else drop them from the public-baseline comparison and keep only as internal ablations.

---

## 4. Architecture-family baselines (special bases — described, contrast only)

These are **tied to a specific base architecture**, so they cannot be run on Qwen3-8B faithfully. We represent the linear-attention/SSM family by evaluating on our **Qwen3.5-9B (Gated-DeltaNet)** base, where **KV-eviction methods are structurally N/A** (no KV cache) — only prompt/window compression apply. This *is* the faithful statement of the architectural fact.

| family | papers | how represented |
|---|---|---|
| Mamba / Mamba-2 / RWKV / RetNet / GLA / DeltaNet | refs #82–89 | Not run (different models). Represented by Qwen3.5-GDN as the shipped hybrid-linear instance. |
| Gated-DeltaNet (our 2nd base) | [GDN (Yang 2024)](https://arxiv.org/abs/2412.06464) | **Run** as `Qwen3.5-9B`: full-context + LLMLingua + window only; KV-press N/A (documented in fact-base §6). |
| YaRN / LongRoPE / Self-Extend (position) | refs #76–79 | Not separate baselines — our base already uses YaRN to reach its window; position-extension is the enabler, not a compressor. |

---

## 5. Benchmarks swept (all public)

Expanded from the wave-1/2 set. Definitions + refs are in `baseline-factbase-v2.0.0.md §benchmarks`.

| bench | type | length knob | in this sweep |
|---|---|---|---|
| ruler_niah | synthetic single-needle retrieval | `GCM_NCHUNKS` (2k–32k) | master budget table + length sweep |
| numerical_niah | numeric needle | nc | method×bench + refs |
| categorical_niah | categorical needle | nc | refs |
| coding_niah | code needle | nc | refs |
| multi_needle_niah | multi-fact needle | nc | 🔴 **scoring broken** (`full`=0.0) — excluded until fixed |
| narrativeqa | long-doc abstractive QA | — | method×bench + refs |
| quality / quality_hard | long-doc multiple-choice (MC) | — | method×bench (hard) + refs (full) |
| musr_mm | multi-step reasoning MC | — | refs |
| locomo | long dialogue memory QA | — | refs |
| squad_v2 / hotpot_qa / trivia_qa | extractive / multi-hop / open QA | — | (wave-1/2 fact-base) |

---

## 6. Summary counts

- **KV methods reproduced faithfully:** 18 + 2 adapted (think/simlayer) = **20** in the master table.
- **KV methods excluded with reasons:** 8 (pyramidkv, adakv, critadakv, qfilter, duoattention, finch, kvzap, chunk/chunkkv).
- **Non-KV baselines:** LLMLingua-2, window, full, no_ctx (faithful) + gist-lite/cartridge-lite (adapted, trained).
- **Benchmarks:** 12 public (1 — multi_needle — excluded for a scoring bug).

*Provenance: smoke tests `sm2_*`/`sm3_*` (faithfulness gate), catalog sweep `c_*` (the numbers), launched 2026-06-25; see `decisions-2026-06-24.md` D14.*
