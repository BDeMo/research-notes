# Plan 08 v2 — Related Work

> Snapshot taken 2026-06-01, after a literature pass triggered by the
> v2 design discussion. Coverage: last ~24 months of latent / continuous
> reasoning, memory tokens, KV-cache compression, and persistent
> conversational memory.
>
> **Setting / provenance**: this is external-resource provenance for
> [`P08-S4`](../settings/settings.md#p08-s4--v2-design-setting), not a verified experiment.
> Any future v2 result table must cite a separate result setting in
> `settings.md`.
>
> The space is **crowded** — at least 13 published / preprint methods
> overlap with the v2 idea of "embedding memory written during AR
> generation and reused later." Differentiation will rely on the
> A + B angles in `v2-plan.md` (cross-session persistence + frozen
> base + bit-capacity diagnostic).

---

## Two surveys to read first

| paper | arxiv | what it gives |
|---|---|---|
| **A Survey on Latent Reasoning** | [2507.06203](https://arxiv.org/abs/2507.06203) | Taxonomy: vertical recurrence (loop layers) vs horizontal recurrence (propagate hidden states across time). v2 is horizontal. ~1 h read. |
| **Latent Chain-of-Thought Reasoning** | [2505.16782](https://arxiv.org/abs/2505.16782) | Second survey, slightly different categorization. Lists Coconut / CODI / CoLaR / CCoT / KaVa / R-Caps / LatentSeek / Pythia Arch and others. |

---

## Closest neighbours (must-cite, head-to-head compare)

| # | paper | year/m | id / venue | similarity | one-line difference vs v2 |
|---|---|---|---|---|---|
| 1 | **Coconut: Training LLMs to Reason in Continuous Latent Space** (Hao et al., Meta + UCSD) | 2024-12 | [2412.06769](https://arxiv.org/abs/2412.06769) · NeurIPS '24 | very high | feeds last hidden state back as next input embedding; **requires full base FT**, no cross-session story |
| 2 | **TokMem: One-Token Procedural Memory for LLMs** (MANGA-UO) | 2025-10 | [2510.00444](https://arxiv.org/abs/2510.00444) | very high | learnable memory tokens, frozen base, invoked / chained during generation — but **supervised by procedure–response pairs, no cross-session persistence, no capacity story** |
| 3 | **LightThinker: Thinking Step-by-Step Compression** (Liu et al., ZJU + Ant) | 2025-02 | [2502.15589](https://arxiv.org/abs/2502.15589) · EMNLP '25 | very high | compress generated thoughts → gist tokens → **discard original** within one context; v2 keeps and re-uses them across sessions |
| 4 | **CoLaR: Think Silently, Think Fast — Dynamic Latent Compression** | 2025-05 | [2505.16552](https://arxiv.org/abs/2505.16552) | high | auto-regressive next-compressed-embedding head + RL on compression factor |
| 5 | **R3Mem: Memory Retention/Retrieval via Reversible Compression** (Tan et al.) | ACL '25 | [findings-acl.235](https://aclanthology.org/2025.findings-acl.235.pdf) | high | explicit **read / write token pairs**, hierarchical compression, propagate across context windows; tunes the encoder, not a strictly frozen base |
| 6 | **NextMem: Latent Factual Memory for LLM-based Agents** | 2026-03 | [2603.15634](https://arxiv.org/abs/2603.15634) | high | autoregressive autoencoder for latent factual memory, two-stage train + quantization. Closest v1 cousin. |
| 7 | **Memory Tokens: LLMs Can Generate Reversible Sentence Embeddings** (Suruguay et al.) | 2025-06 | [2506.15001](https://arxiv.org/abs/2506.15001) | high | frozen LLM, **one memory token can losslessly reconstruct ≤240 tokens** — directly relevant capacity reference |
| 8 | **Cramming 1568 Tokens into a Single Vector** (Kuratov et al.) | 2025 | [2505.21189](https://arxiv.org/abs/2505.21189) | high | one memory token can cram 1568 tokens; embedding-space capacity floor |

---

## Adjacent (must mention, not head-to-head)

| # | paper | year/m | id | how it relates |
|---|---|---|---|---|
| 9 | **SoftCoT / SoftCoT++** (Xu et al., NTU) | 2025-02 / 05 | [2502.12134](https://arxiv.org/abs/2502.12134) ACL '25 / [2505.11484](https://arxiv.org/abs/2505.11484) | external assistant generates soft thoughts, projected into frozen LLM — the "frozen base + external encoder" pattern v2 uses |
| 10 | **CODI: Continuous CoT via Self-Distillation** (Shen et al.) | 2025 | (in survey) | hidden state → projection → re-inject |
| 11 | **Latent Thought Models with Variational Bayes** (Hu et al.) | 2025-02 | [2502.01567](https://arxiv.org/abs/2502.01567) | explicit prior on latent thoughts + dual-rate VB |
| 12 | **LTPO: Latent Thought Policy Optimization** | 2025-10 → 2026-01 v4 | [2510.04182](https://arxiv.org/abs/2510.04182) | **test-time** RL on latent thoughts; conceptually mirrors v1's RL phase |
| 13 | **Soft Thinking** (Zhang et al.) | 2025-05 | [2505.15778](https://arxiv.org/abs/2505.15778) NeurIPS '25 | training-free; probability-weighted token-embedding mixture as soft concept token per step |
| 14 | **System-1.5 Reasoning** | NeurIPS '25 | [poster](https://neurips.cc/virtual/2025/poster/118475) | dynamic shortcuts in latent space; reuse hidden states across decode steps; 20× faster |
| 15 | **Latent-SFT: Vocabulary-Space Superposition** | ICLR '26 sub | [openreview ciiKoeM206](https://openreview.net/forum?id=ciiKoeM206) | constrain latent to vocab column space; new SoTA on GSM8k |
| 16 | **CCoT: Compressed Chain-of-Thought** (Cheng & Van Durme) | 2024 | (cited in SoftCoT++) | content-rich continuous contemplation tokens |
| 17 | **Pause Tokens** (Goyal et al., Google) | 2024 | [2310.02226](https://arxiv.org/abs/2310.02226) ICLR '24 | learnable pause tokens give the model extra hidden vectors before answering — the "give the model more latent computation" ancestor |
| 18 | **Recurrent Memory Transformer (RMT) / ARMT** (Bulatov et al.) | 2022 → 2024 | NeurIPS '22, AAAI '24 | segment-level memory token recurrence; v1 already cites |
| 19 | **KaVa: KV-cache Distillation** (Kuzina et al.) | 2025 | (in survey) | distill KV cache into latent — orthogonal compression line |
| 20 | **KVzip: Query-Agnostic KV Cache Compression** (Kim et al.) | 2025 | [poster](https://neurips.cc/virtual/2025/poster/118741) NeurIPS '25 oral | 3–4× KV reduction via context-reconstruction importance |
| 21 | **Multi-Head Latent Attention (MLA)** | DeepSeek-V2/V3 | (DeepSeek papers) | architectural KV compression via low-rank latent projection |

---

## Persistent / conversational memory (the angle we lean on)

| # | paper | year | id | what |
|---|---|---|---|---|
| 22 | **Mem0: Production-Ready AI Agents with Scalable Long-Term Memory** (Chhikara et al.) | 2025-04 | [2504.19413](https://arxiv.org/abs/2504.19413) | textual + graph memory, not latent; LOCOMO benchmark king. **v2 contrasts as the fully-latent point on the Pareto curve.** |
| 23 | **LongMemEval / LOCOMO benchmarks** | 2024-25 | various | the eval datasets v2 needs to run on for credibility |

---

## How v2 sits in the survey taxonomy

Both surveys split latent reasoning into:

* **Vertical recurrence**: re-process activations through layers
  (depth-axis). Includes: looped transformers, AlBERT-style sharing.
  **v2 is NOT here.**
* **Horizontal recurrence**: propagate hidden state through time.
  Includes: Coconut, CODI, LightThinker, CoLaR, R3Mem, TokMem,
  NextMem, KaVa, RMT.
  **v2 is here.**

Inside horizontal recurrence, v2 occupies the **(frozen base) ×
(cross-session) × (suffix write at generation time)** cell, which is
the empty intersection of the existing methods.

---

## Open / live links to watch

* GitHub `LatentCoT-Horizon` — the survey author's repo, auto-tracks
  new latent reasoning papers.
* `xuyige/SoftCoT` — active codebase for SoftCoT / SoftCoT++.
* `zjunlp/LightThinker` — official LightThinker release; useful for
  borrowing the gist-token training-data construction recipe.
* `mem0ai/mem0` — production Mem0 codebase; relevant when we build the
  LOCOMO comparison harness.

---

## Implications for v2 paper

1. **Cannot pitch "writing latent memory during generation" as novel** —
   that's been done four times in 12 months. Must frame as either:
   (a) *first to make it cross-session and frozen-base simultaneously*,
   or (b) *first to characterize the bit-capacity wall of generation-
   time memory*.
2. **Must run head-to-head against TokMem, LightThinker, and Mem0** to
   be credible.
3. **Must show the bit-capacity curve transfers** from v1 (encode-time)
   to v2 (write-time) — if it does, the "read-side bottleneck" thesis
   becomes very strong; if it does not, we have a different paper
   (write-side capacity helps).
4. **Should not over-claim against Coconut** — Coconut tunes the base;
   v2 can't beat it on math reasoning. Stay in the "frozen-base
   regime" lane.
