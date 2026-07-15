# Paper C (v3.0) — references: linear attention & its variants

Reference collection for the next paper line, **focused on linear attention and variants**. Categorized; arXiv IDs. Source-of-truth for recency: `fla-org/flash-linear-attention` catalog (checked 2026-07) + our own fact-base (Qwen3.5-GDN experiments). Mark 🌟 = likely core-cite for Paper C.

> Live web search was unavailable at collection time; IDs below are from the fla catalog (reliable) + memory. **Verify each arXiv ID before final citation.**

## 1. Foundations — linearized / kernel attention
- 🌟 **Linear Transformers** — Katharopoulos et al., "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention" (2006.16236). The φ(q)·(φ(k)ᵀv) associativity → O(L) recurrence.
- **Performer** — Choromanski et al., FAVOR+ random features (2009.14794).
- **cosФormer / Linear attention kernels** — kernel-feature-map line.
- **ABC** — Attention with Bounded-memory Control (2110.02488).

## 2. Gated linear attention & linear RNNs
- 🌟 **RetNet** — Retentive Network (2307.08621); decay-based retention, parallel/recurrent/chunkwise.
- 🌟 **GLA** — Gated Linear Attention w/ hardware-efficient training (2312.06635); data-dependent gates + chunkwise kernel.
- **HGRN / HGRN2** — Hierarchically Gated RNN (P1TCHxJwLB) / state expansion (2404.07904).
- **LightNet** (2405.21022); **GSA** — Gated Slot Attention (2409.07146); **Rodimus\*** (2410.06577).

## 3. State-space models (SSM)
- 🌟 **Mamba** — Selective SSM (2312.00752, Gu & Dao).
- 🌟 **Mamba-2** — SSD, Transformers-are-SSMs duality (2405.21060).
- 🌟 **Mamba-3** — improved SSM principles (2603.15569, 2026).
- S4/S5 lineage (structured state spaces).

## 4. Delta-rule family (test-time-regression / associative memory) — **central to Paper C**
- 🌟 **DeltaNet** — Parallelizing Linear Transformers with the Delta Rule over sequence length (2406.06484); orig delta rule (2102.11174).
- 🌟 **Gated DeltaNet (GDN)** — Improving Mamba2 with the delta rule (2412.06464). **← the module in Qwen3.5/3.6 (our base).**
- 🌟 **Gated DeltaNet-2 (GDN-2)** — decoupling erase & write in linear attention (2605.22791, 2026).
- **DeltaProduct** — better state-tracking via Householder products (2502.10297).
- **Comba** — bilinear RNN w/ closed-loop control (2506.02475).
- **MesaNet** — locally optimal test-time training (2506.05233).
- **DeltaFormer** — Transformer as associative memory (2505.19488).
- 🌟 **KDA / Kimi Linear** — Kimi Delta Attention, expressive+efficient (2510.26692, 2025).
- **Parallax** — parameterized local linear attention (2605.29157, 2026).
- **PaTH** — position encoding via Householder accumulation (2505.16381).

## 5. RWKV line
- **RWKV-4/5/6 "Eagle & Finch"** — matrix-valued states, dynamic recurrence (2404.05892).
- 🌟 **RWKV-7 "Goose"** — expressive dynamic state evolution (2503.14456).

## 6. Recall / capacity of linear attention
- 🌟 **Based** — simple linear attention balances the recall–throughput tradeoff (2402.18668).
- **ReBased** — learnable kernels for in-context (2402.10644).
- 🌟 **Zoology** — measuring & closing the associative-recall gap (Arora et al., 2312.04927).
- **MoM** — Mixture-of-Memories: multiple states to raise capacity (2502.13685).
- **Log-Linear Attention** — log-sized state, between linear & full (2506.04761).

## 7. Hybrid architectures (linear + periodic full/sparse attention) — **our base's design**
- 🌟 **Samba** — Mamba + sliding-window attention (2406.07522).
- 🌟 **Qwen3.5 / 3.6** — GDN linear : full-attn **3:1 hybrid**, gated attention, MTP (HF model cards; our fact-base `linear-attention-and-kvcache-background.md`).
- 🌟 **MiniMax-01** — Lightning Attention hybrid at scale (2501.08313).
- **Jamba** (Mamba-Transformer MoE, 2403.19887); **Zamba**; **Griffin/Hawk** (gated linear recurrences + local attn, 2402.19427).
- **YOCO** — decoder-decoder, cache once (2405.05254).

## 8. Sparse / block attention (efficient-attention peers, for baselines)
- 🌟 **NSA** — Native Sparse Attention, hardware-aligned trainable sparse (2502.11089).
- 🌟 **MoBA** — Mixture of Block Attention for long-context (2502.13189).
- **FoX** — Forgetting Transformer: softmax attn + forget gate (2503.02130).
- **MLA** — Multi-head Latent Attention (DeepSeek-V2, 2405.04434) — KV-compression, not linear but a cache-shrink peer.

## 9. Test-time training / memory-as-learning (conceptual neighbors of the delta rule)
- 🌟 **TTT** — Learning to (Learn at Test Time): RNNs with expressive hidden states (2407.04620).
- 🌟 **Titans** — Learning to Memorize at Test Time (2501.00663, Google) + follow-ups (Atlas).

## 9b. GDN forgetting / memory-editing (the erase–write axis) — **Paper C core**
- 🌟 **Gated DeltaNet-2 (GDN-2)** — Hatamizadeh et al. (NVIDIA), 2605.22791 (2026). Decouples the delta edit into a **channel-wise ERASE gate `b_t` (key axis)** + **channel-wise WRITE gate `w_t` (value axis)** + channel-wise decay (from KDA). Generalizes GDN (all scalar) & KDA (channel decay, scalar erase/write). **Ablation: the erase gate accounts for most of the gain** → *selective forgetting of stale key-side associations* is the lever. Strongest on RULER multi-key retrieval. Code: NVlabs/GatedDeltaNet-2.
- 🌟 **The Delta Rule Dominates: A Factorial Analysis of Decay in Linear Attention** — OpenReview 2026. Controlled 2×2×2 cube: **granularity {scalar, channel-wise} × conditioning {data-dependent, data-independent} × {delta, no-delta}** (8 variants). Findings: (1) **delta rule is the strongest factor**; (2) **granularity×delta interaction** — channel-wise helps *only with* delta, hurts without (interference); (3) channel-wise aids long-range persistence but can **hinder precise recall** (scalar+delta > channel+delta at some ranges). → the ablation template for Paper C.
- 🌟 **KDA / Kimi Linear** — 2510.26692. Channel-wise (per-dimension) data-dependent decay `diag(α_t)` = multi-timescale memory per head. Note: reported to have **convergence/stability issues during distillation at scale** (per HALO).
- **HALO / HyPE (Lightning-Attention hybrid)** — Chen et al. 2026. **Data-independent fixed scalar decay generalizes to length better** than data-dependent gates ("data-dependent forget gates may hurt length generalization"). A counter-point on forgetting design.
- 🌟 **Linear Attention Architectures: Mechanisms, Trade-offs, and Cross-Layer Routing** — 2607.07953 (2026). Survey of the delta-rule family skeleton (DeltaNet→GDN→KDA→GDN-2) + cross-layer routing; how variants balance selectivity/forgetting/granularity.

## 10. Theory / limitations — **the "why it fails" backbone for a diagnostic paper**
- 🌟 **The Illusion of State in State-Space Models** — Merrill, Petty, Sabharwal (2404.08819): SSMs/linear RNNs are in TC⁰-like classes → cannot do inherently sequential state tracking.
- **DeltaProduct / state-tracking** (2502.10297) — Householder products extend the tractable state-tracking class.
- **Repeat-after-me / associative-recall** capacity analyses (Zoology 2312.04927).
- **"Illusion of Progress"?** long-context eval critiques (RULER, HELMET, LongBench-v2, ∞Bench, BABILong).

## 11. Long-context evaluation (shared with Paper B; used to probe linear-model recall)
- RULER (2404.06654), ∞Bench (2402.13718), LongBench / LongBench-v2, BABILong (2406.10149), HELMET (2410.02694), NIAH variants, NoLiMa.

## 12. Our own fact-base (bridge to Paper C)
- `linear-attention-and-kvcache-background.md` — GDN vs full-attn, KV-cache differences, Qwen3.5/3.6 hybrid.
- F10/F28/F43 (`matrix-facts.md`): long-context compression findings reproduce on GDN linear; **KV-eviction cannot run on linear (no KV cache)**; input-side IMP/RAG can; **IMP's RULER gap is WIDER on GDN → state-collapse/recall limit** — the empirical hook for Paper C.

---
**Next:** verify arXiv IDs; pull 3–5 core papers (GDN, GDN-2, Mamba-3, KDA/Kimi-Linear, Based/Zoology, Illusion-of-State) in full for the related-work + positioning in `PAPER-C-research-line.md`.
