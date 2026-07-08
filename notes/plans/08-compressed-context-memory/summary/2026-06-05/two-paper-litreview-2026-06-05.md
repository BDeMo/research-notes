# Two-Paper Split — Literature Reviews (2026-06-05)

We split the work into two papers along its two real contributions:

- **Paper A — Latent Memory** (Plan 08): a recurrent **soft-prompt memory wrapper** that
  compresses long context into K tokens on a **frozen base**; the empirical
  **bit-capacity wall**.
- **Paper B — Forgetting & Gating**: **do-no-harm adaptation** of a pretrained model =
  the **read-time** input-conditioned **gate** (Plan 08 v1.5: use the memory only when
  safe) + the **write-time** **site/gradient protection** (Plan 09 / Janus: protect
  long-context-important sites during SFT). Both fight **catastrophic forgetting** with
  intrinsic, data-agnostic signals.

> Connection: both papers are about **not degrading a frozen/pretrained model when
> adapting it**. Paper A = a removable memory module (never touch weights); Paper B =
> when you *must* touch behavior, gate (read) or protect (write) so you don't forget.
> Detailed source docs: [`../../v2-related-work.md`](../../history/v2-related-work.md),
> [`../../v0-opd-on-latent-memory-survey-2026-06-01.md`](../../history/v0-opd-on-latent-memory-survey-2026-06-01.md),
> [`v1.5-related-work-2026-06-05.md`](v1.5-related-work-2026-06-05.md),
> [`../../../09-intrinsic-site-protection/references.md`](../../../09-intrinsic-site-protection/references.md).

=================================================================
## PAPER A — LATENT MEMORY
=================================================================

### A1 · Context compression into soft tokens (the wrapper's direct family — must-cite)
| paper | id | role |
|---|---|---|
| **Gist Tokens** (Mu et al.) | 2023 NeurIPS | the canonical "compress prompt → K soft tokens", frozen-ish base — our explicit baseline |
| **AutoCompressor** (Chevalier et al.) | 2023 | summary tokens accumulated across segments |
| **ICAE** (Ge et al.) | 2024 ICLR | in-context auto-encoder → memory slots (reconstruction) |
| **CCM — Compressed Context Memory** (Kim et al.) | [2312.03414](https://arxiv.org/abs/2312.03414) | conditional-LoRA online compression, **frozen base**, KV/soft-prompt memory |
| **PCC — Pretraining Context Compressor** | ACL'25 | frozen LLM + compressor → memory slots; 256× loses too much |
| **xRAG / PISCO** | 2024-25 | single-token / soft compression for RAG |

### A2 · Latent reasoning / continuous-thought (adjacent — cross-ref `v2-related-work.md`)
Coconut, TokMem, LightThinker, CoLaR, R3Mem, NextMem, SoftCoT/++, Pause Tokens, CCoT,
RMT/ARMT, Memory Tokens, Cramming-1568. (Full table + the "horizontal recurrence"
taxonomy in `v2-related-work.md`; OPD/RL recipe neighbors in the v0 OPD survey.)

### A3 · The PROBLEMS of latent memory (Paper A's empirical core)
| problem | paper | finding = our result |
|---|---|---|
| lossy → **exact-recall collapse** | **"A Silver Bullet or a Compromise?"** [2412.17483](https://arxiv.org/abs/2412.17483) ACL'25 | gist compression fails on **synthetic recall**; 3 patterns (lost-by-boundary / if-surprise / along-the-way) = **our RULER-NIAH→0** |
| compressed memory **degrades the base** | **"Rethinking Soft Compression in RAG" (SeleCom)** [2602.15856](https://arxiv.org/abs/2602.15856) | full soft-compression infeasible; embeds "**blind the LLM**" = **our negative transfer** |
| limited capacity, drops entities/numbers | **Info-Preservation in Prompt Compression** EMNLP'25 findings | soft prompts can't encode cardinals/names; hybrid soft-hard |
| embedding-space capacity floor | Cramming-1568 [2505.21189](https://arxiv.org/abs/2505.21189), Memory Tokens [2506.15001](https://arxiv.org/abs/2506.15001) | how much one vector holds |

### A4 · Positioning + honest novelty (Paper A)
- **Don't** claim the capacity wall / exact-recall failure as novel — the gist "Silver
  Bullet" study + Info-Preservation document it. **Do** claim: (i) the wall is a
  **read-interface** property (matched-K Gist hits it at the same place), (ii) the
  recurrent **frozen-base** wrapper + the OPD/RL recipe (v0 survey), (iii) the precise
  bit-entropy curve. **Baselines a reviewer demands:** Gist, ICAE, CCM, AutoCompressor;
  evaluate on the synthetic-recall axis the Silver-Bullet study used.

=================================================================
## PAPER B — FORGETTING & GATING (read-time gate ⊕ write-time protection)
=================================================================

### B1 · Catastrophic forgetting in LLM fine-tuning (the problem we prevent)
| paper | id | finding |
|---|---|---|
| **Empirical Study of CF in LLMs** (Luo et al.) | [2308.08747](https://arxiv.org/abs/2308.08747) | CF is general (1–7B), **worsens with scale**; general instruction tuning mitigates — the canonical motivation |
| **Mechanistic Analysis of CF** | [2601.18699](https://arxiv.org/abs/2601.18699) | 3 mechanisms: **gradient interference in attention**, representational drift, loss-landscape flattening; **gradient-alignment predicts forgetting (r=0.87)**; 15–23% lower-layer heads disrupted. **THE comparison for Plan 09** (their ΔW-damaged set ≈ our read-side long-ctx-important set). |
| **CF origins: RL vs SFT** | 2605.28860 | differential circuit vulnerability |

### B2 · Continual-learning mitigation families (the textbook taxonomy)
- **Regularization**: EWC (Kirkpatrick 2017), L2-SP, LwF (Li & Hoiem).
- **Replay/rehearsal**: experience replay; **Self-Synthesized Rehearsal** (ACL'24); RandSel/KMeansSel (1% replay preserves ability, Scialom 2022).
- **Optimization/gradient**: gradient projection to avoid harming old tasks.
- **Parameter isolation + routing**: adapters; **wise-ft / model soups** (Wortsman 2022).

### B3 · "Which parameters to protect" criteria (Plan 09's direct competitors)
| baseline | criterion | vs Plan 09 |
|---|---|---|
| **MoFO / MIGU** | momentum / gradient magnitude | ours: **intrinsic long-ctx importance** (data-agnostic) |
| **OPLoRA** (AAAI'26) | orthogonal to top **singular** subspace | ours: orthogonal to **long-ctx-important** subspace |
| **ESFT / DES-MoE / DAS** | expert task/domain affinity | ours: **super-expert / sink-induction** (intrinsic) |
| **SAE-FD** ([2605.25525](https://arxiv.org/abs/2605.25525)) | SAE-feature distillation | feature-space incumbent; needs a trained dictionary |

### B4 · Intrinsic sites (the read-side substrate, shared by both halves)
Retrieval heads ([2404.15574](https://arxiv.org/abs/2404.15574), ICLR'25; sparse <5%,
intrinsic, causal for NIAH) · dynamic retrieval heads (2602.11162) · attention sinks
(StreamingLLM, Xiao 2023) · massive activations (Sun 2024) · super-experts (2507.23279) ·
DuoAttention (2410.10819, retrieval-vs-streaming split). (All in `09/references.md`.)

### B5 · Gating / conditional use of augmentation (the **gating** half = v1.5)
| paper | id | what = our gate |
|---|---|---|
| **TARG — Training-free Adaptive Retrieval Gating** | [2511.09803](https://arxiv.org/abs/2511.09803) | gate the augmentation from the model's **own prefix logits** (entropy / top-1-2 margin / variance); "using it every time hurts" → **= our routing gate**, but on retrieval w/ logit signals; we use **memory-write dynamics** that transfer cross-model |
| **Self-RAG** / **FLARE** / **CRAG** / **SeaKR** | 2023-25 | reflection tokens / confidence / passage-quality / uncertainty to decide when to use evidence |
| **When Do LLMs Need RA?** | [2402.11457](https://arxiv.org/abs/2402.11457) | confidence-gated retrieval (the "do-no-harm by gating" statement) |
| **LLaMA-Adapter** | [2303.16199](https://arxiv.org/abs/2303.16199) ICLR'24 | **per-layer zero-init gated injection** into a frozen base (topmost L layers) — = our **multi-depth injection** (probe_deep); + Prefix-Tuning ([2101.00190](https://arxiv.org/abs/2101.00190)) / P-Tuning v2 ([2110.07602](https://arxiv.org/abs/2110.07602)) |

### B6 · Positioning + honest novelty (Paper B)
**One frame, two sides:** *do-no-harm adaptation via intrinsic, data-agnostic signals.*
- **read-time (gate)**: per-input, decide whether to apply the memory/augmentation; fall back to the base (v1.5).
- **write-time (protect)**: per-site, protect long-ctx-important weights during SFT so general ability isn't overwritten (Plan 09).
- The unifying claim: the **same intrinsic read-side criterion** (long-context importance / memory-write dynamics) tells you both *when to open the gate* and *what to freeze* — and it **transfers across model families** (our H1/H2).

**Honest:** the gate ≈ adaptive retrieval (TARG/Self-RAG); the protection ≈ MoFO/MIGU/ESFT/Mechanistic-Forgetting head-freezing; multi-depth injection ≈ LLaMA-Adapter. Novelty = unifying read+write do-no-harm under one **cross-model-general intrinsic signal**, not any single mechanism. **Baselines:** read — TARG logit-margin gate, LLaMA-Adapter; write — EWC, MoFO/MIGU, ESFT, Mechanistic-Forgetting head-freezing.

---

## Split summary (one line each)
- **Paper A:** *Bit-capacity limits of a frozen-base soft-prompt memory wrapper* — characterize what K soft tokens can/can't encode (vs Gist/ICAE/CCM; the Silver-Bullet failure axis).
- **Paper B:** *Do-no-harm adaptation* — one intrinsic, cross-model signal that both **gates** a memory/augmentation at read-time (vs TARG/Self-RAG) and **protects** sites at write-time (vs MoFO/ESFT/Mechanistic-Forgetting), preventing catastrophic forgetting.
