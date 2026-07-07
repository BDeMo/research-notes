# Background: linear-attention architectures & the KV-cache (for newcomers)

> Split out of `OVERVIEW-both-papers-and-facts.md` (§ "extra background questions"). Standalone background a general AI
> researcher needs to follow why our second base (a Gated-DeltaNet model) has no KV-cache and why that shapes the method.

## (1) What does "GDN" stand for?
**GDN = Gated DeltaNet** ([Yang et al., ICLR 2025, arXiv:2412.06464](https://arxiv.org/abs/2412.06464)), "Gated Delta Networks: Improving Mamba2 with Delta Rule." It is a **linear-attention / recurrent** token-mixer that keeps a **fixed-size memory** updated by two mechanisms: a **gate** (how much old memory to *erase/decay*, à la Mamba-2) and a **delta rule** (a targeted *write* that corrects the current key's stored value, à la [DeltaNet](https://arxiv.org/abs/2406.06484)). It is the mixer in the Qwen3.5 family we use as a second base.

## (2) How many recent flagship models are GDN-style (linear-complexity)?
As of mid-2026 this went from niche to **mainstream for frontier models**. Public examples of linear/hybrid-linear flagships:
- **Qwen3-Next 80B-A3B** (2025) — 3:1 Gated-DeltaNet : Gated-Attention.
- **Qwen3.5** (2026, our base family) — Gated-DeltaNet-based.
- **Kimi Linear 48B-A3B** ([arXiv:2510.26692](https://arxiv.org/abs/2510.26692)) — *Kimi Delta Attention* (GDN + channel-wise gating) + MLA, 3:1; reportedly **beats full attention** with ~75% less KV-cache.
- **MiniMax-01** (2025) — lightning/linear attention at ~456B, ~6–7:1.
- **Nemotron-3 Nano 30B-A3B / Super 120B-A12B**, **Ling 2.5 1T**, **Olmo Hybrid** (2026) — all hybrid-linear.
- Method frontier: **Gated DeltaNet-2** ([arXiv:2605.22791](https://arxiv.org/abs/2605.22791)), **Mamba-3**, RWKV-7.

So: **most 2025–2026 efficiency-oriented frontier releases are hybrid-linear**; a good survey is [*A Systematic Analysis of Hybrid Linear Attention* (2507.06457)](https://arxiv.org/abs/2507.06457) and [Raschka's hybrid-attention gallery](https://sebastianraschka.com/llm-architecture-gallery/hybrid-attention/). Pure-quadratic full-attention is now the *exception* at the largest scales.

## (3) Linear + quadratic **hybrid** combinations, and who does what
The dominant recipe: **interleave many cheap linear layers with a few full-attention layers**, because "long-context recall rises steeply once a few full-attention blocks are present, then plateaus" — the linear:full ratio mainly controls **recall**.

| hybrid | mix / ratio | linear part does… | quadratic part does… |
|---|---|---|---|
| [RecurrentGemma/Griffin](https://arxiv.org/abs/2402.19427) | 2:1 recurrence + local attn | bulk sequence mixing | local exactness (sliding window) |
| [Jamba](https://arxiv.org/abs/2403.19887) | 7:1 Mamba:Transformer (+MoE) | long-range, cheap | periodic exact retrieval |
| [Samba](https://arxiv.org/abs/2406.07522) | Mamba + sliding-window attn | unbounded history (linear) | local window (bounded) |
| [Hymba](https://arxiv.org/abs/2411.13676) | **head-wise** in one layer | some heads = SSM (global gist) | other heads = softmax (precise recall) |
| [YOCO](https://arxiv.org/abs/2405.05254) | **prefill/decode split** | \(O(L)\) prefill compresses keys | decode reuses one shared cache |
| Qwen3-Next / Qwen3.5 | 3:1 GDN:(Gated)Attention | long-context state carry | exact content lookup |
| Kimi Linear | 3:1 KDA:MLA | fine-gated linear memory | low-rank exact attention (MLA) |
| Gated DeltaNet-2 | GDN-2 + SWA | constant-size compressed history | local shifts/compares (window) |

**General division of labor:** *linear/recurrent layers = compress the whole history into a bounded state (cheap, scalable); sparse full-attention (or MLA / sliding-window) layers = do the exact, content-addressed retrieval the recurrent state is bad at.* The known failure of the linear part is **"state collapse"** — a finite state can't hold arbitrarily many exact facts, so **multi-key retrieval degrades** (exactly the weakness our fact-base sees on the GDN base).

## (3.5) MiniMax-M3 vs MiniMax-Text-01/M1 vs Qwen3.5/3.6

MiniMax needs a careful split: **MiniMax-Text-01 / MiniMax-M1** belong to the
hybrid-linear family, but **MiniMax-M3 does not**. M3 uses a new sparse-attention
operator, not Lightning/linear attention.

| model family | main long-context mechanism | hybrid scheme (what is mixed?) | is it linear attention / SSM? | how long context is implemented | KV-cache status | can KV-cache methods apply? | read for our paper |
|---|---|---|---|---|---|---|---|
| **Qwen3.5 / Qwen3.6** | Gated-DeltaNet + gated attention hybrid | **Layer-level hybrid:** mostly Gated-DeltaNet recurrent/linear layers, with periodic gated-attention layers (Qwen3.6 reported as KV only on 16/64 layers). GDN layers carry compressed history; attention layers repair exact lookup. | **Yes, mostly linear/recurrent** (GDN) with periodic attention layers | GDN layers carry history in a bounded recurrent state; gated-attention layers provide exact lookup periodically | **Partial / layer-dependent.** GDN layers have no standard KV cache; attention layers do | **Only on the attention subset.** SnapKV/H2O/KVzip-style eviction is structurally N/A for GDN layers | This is the clean "KV lever mostly disappears" case; input-side memory, side-cache, or merge+retrain methods matter |
| **MiniMax-Text-01 / MiniMax-M1** | Lightning Attention + softmax attention + MoE | **Block-level hybrid:** repeating pattern of **7 Lightning/linear-attention blocks + 1 softmax-attention block**, plus MoE FFNs. Lightning blocks provide cheap long-range state; softmax blocks provide periodic exact token interactions. | **Yes, hybrid linear attention** | Repeating pattern: many Lightning/linear attention blocks plus periodic softmax attention blocks (reported as roughly 7:1 linear:softmax) | **Partial.** Lightning layers use linear/recurrent-style state; softmax layers still use KV cache | **Partially.** KV eviction can target softmax layers, but not Lightning layers | Similar to Qwen3.5 in spirit, but with Lightning Attention rather than GDN; keep under "hybrid-linear" |
| **MiniMax-M3** | MiniMax Sparse Attention (MSA) + MoE / multimodal backbone | **Sparse/dense attention hybrid:** per-layer type selects either full attention or `minimax_m3_sparse`; sparse layers use an indexer to choose top key blocks plus local blocks, then run exact attention only on those blocks. This is not recurrent-state mixing. | **No.** It is block-sparse softmax attention, not SSM/linear attention | An indexer scores/selects key blocks per query; attention runs over selected blocks plus local blocks | **Yes.** It still has K/V blocks; access is sparse rather than dense | **Yes, but with caveats.** KV methods can apply to cached K/V blocks, but M3 already has a learned sparse-selection mechanism, so it is not a plain dense-attention baseline | Classify as sparse-attention Transformer, not SSM. It belongs in the "sparse attention / learned block selection" comparison, not the linear/SSM arm |

**Practical rule:** if a layer stores per-token K/V and later attends to selected
tokens, KV-cache methods can operate on that layer. If a layer compresses history
into a recurrent / linear state, KV-cache eviction is not the right lever for
that layer. This is why Qwen3.5/3.6 and MiniMax-Text-01/M1 are **partial-KV**
hybrids, while MiniMax-M3 is **KV-backed but sparse**.

## (4) KV-cache transformer vs no-KV-cache vs linear — what actually differs
| property | **standard Transformer + KV-cache** | **standard Transformer, no KV-cache** | **linear-attention / SSM (e.g. GDN)** |
|---|---|---|---|
| what's stored across steps | **all** past K,V (grows with \(L\)) | nothing — recompute each step | a **fixed-size** recurrent state (independent of \(L\)) |
| decode compute / step | \(O(L)\) (read whole cache) | \(O(L^2)\) (recompute everything) | **\(O(1)\)** (update state) |
| decode memory | **\(O(L)\)** and unbounded | \(O(1)\) working but \(O(L^2)\) compute | **\(O(1)\)**, bounded |
| prefill | \(O(L^2)\) attention | \(O(L^2)\) | **\(O(L)\)** (chunked scan) |
| exact recall of far tokens | **excellent** (every token addressable) | excellent (same math, just slow) | **limited** — bounded state → *state collapse* on many keys |
| the "compression knob" | evict/quantize/share the cache (StreamingLLM/H2O/KIVI/MLA) | n/a (nothing cached) | none at the cache level — you only have prompt/window/RAG or a learned memory |
| our two bases | Qwen3-8B (dense attention) | — | Qwen3.5-9B (GDN) |

**Intuition:** a **KV-cache** is *perfect but heavy memory* — it remembers every token exactly, at linear cost. **No cache** is the same memory recomputed (correct but quadratically slow — nobody serves this way). A **linear/SSM model** is *cheap but lossy memory* — a constant-size state that must *summarize* history, so it's fast and scalable but can't recall many exact needles (state collapse). This is precisely why, **on a KV-free linear base, a learned/compressed soft-memory is the natural — arguably only — long-context lever**, and why the "which knobs even exist" question depends on the architecture (a core motivation for Paper B).
