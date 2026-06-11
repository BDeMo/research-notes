# Paper B References (citation registry)

> Citation registry for Paper B / v1.7. Pairs with [`related-work.md`](related-work.md) (prose + OUR DELTA).
> Grouped by topic; each row has a bibkey, title, id, and a one-line. A copy-pasteable BibTeX block for the
> v1.7-critical entries is at the bottom.
>
> **VERIFY BEFORE CITING.** arXiv ids below marked (web) were collected via web search on 2026-06-10 and
> several are very recent preprints; confirm each id, title, authors, and venue on arXiv / Google Scholar
> before it enters the LaTeX. Do not cite anything unverified (no hallucinated references). House rule: no em-dashes.

## A. Must-cite for v1.7 (closest neighbors; the novelty squeeze)
| bibkey | title | id (verify) | one-line / why it matters |
|---|---|---|---|
| `ctxdistill-lmm` | Context Distillation as Latent Memory Management | arXiv 2605.28889 (web) | per-context LoRA memory + retrieval + Self-Gating, fall back to base on context-agnostic queries. Our Track A twin (LoRA memory). |
| `latent-ctx-compilation` | Latent Context Compilation: Distilling Long Context into Compact Portable Memory | arXiv 2602.21221 (web) | disposable-LoRA compiler to buffer tokens; self-aligned = reconstruction (fidelity) + context-agnostic regularizer (do-no-harm). Design twin of our gate-robust compressor. |
| `slt` | Selective Latent Thinking: Adaptive Compression of LLM Reasoning Chains | arXiv 2605.25745 (web) | reliability-aware feature decoder + confidence gate + latent compressor, jointly trained; fall back to explicit CoT. Same mechanism, reasoning regime. |
| `pipo` | Pair-In, Pair-Out: Latent Multi-Token Prediction for Efficient LLMs | arXiv 2605.27255 (web) | lightweight confidence head trained on rejection-sampling acceptance prob (free in-distribution label); safe fallback to single-token decoding. |
| `six-arch-memory` | Trained Persistent Memory for Frozen Encoder-Decoder LLMs: Six Architectural Methods | arXiv 2603.16413 (web) | Prefix/KV-Ext/XAttn/Hebbian/Slot/Gated on frozen base; prefix collapses, XAttn best, Gated negative, capacity wall. Reproduces our architecture ablation -> cede it. |
| `poc-compression` | PoC: Performance-oriented Context Compression via Performance Prediction | arXiv 2603.19733 (web) | predictor picks the most aggressive ratio meeting a user "performance floor". Our compress-suffices prediction at the ratio level. |
| `lclm-e2e` | End-to-End Context Compression at Scale (Latent Context LMs) | arXiv 2606.09659 (web) | encoder-decoder soft-token compressors; goal = a general compressor that preserves base capabilities. |
| `gmemllm` | G-MemLLM: Gated Latent Memory Augmentation | arXiv 2602.00015 (web) | frozen backbone + trainable latent memory bank + GRU-style gate. |
| `lycheememory` | LycheeMemory | arXiv 2602.08382 (web) | Compressor + Gate (separate classifier) + Reasoner; frozen base + LoRA compressor; joint RL. |

## B. Compressor robustness by design (reconstruction / calibration / verifier co-training)
| bibkey | title | id (verify) | one-line |
|---|---|---|---|
| `compactor-kv` | Compactor: Calibrated Query-Agnostic KV Cache Compression | arXiv 2507.08143 (web) | context-calibrated: infer per-context how much can be evicted without quality loss. |
| `structural-confidence` | Trust in One Round: Confidence Estimation via Structural Signals | arXiv 2602.00977 (web) | single-pass confidence from structural stability of a frozen-encoder hidden trajectory. |
| `verithinker` | VeriThinker: Learning to Verify Makes Reasoning Model Efficient | arXiv 2505.17941 (web) | auxiliary verification fine-tune so the model knows when to self-check (CoT compression). |
| `rl-tango` | RL Tango: Reinforcing Generator and Verifier Together | arXiv 2505.15034 (web) | co-train generator + verifier; keep the verifier in-distribution with the evolving generator. |
| `v1-pairrl` | V1: Unifying Generation and Self-Verification for Parallel Reasoners | id unknown (web; project page) | jointly train one model as generator + pairwise self-verifier; co-evolving in-distribution verification. |
| `recon-ae-maha` | Improving Reconstruction Autoencoder OOD Detection with Mahalanobis Distance | arXiv 1812.02765 | reconstruction error as novelty; AEs can also reconstruct OOD -> add latent Mahalanobis distance. |
| `rethink-recon-ood` | Rethinking Reconstruction Autoencoder-Based OOD Detection | CVPR 2022 | maximally compress latent while preserving reconstructive power; layer-wise semantic reconstruction. |
| `diffusion-layerwise-ood` | Diffusion-based Layer-wise Semantic Reconstruction for Unsupervised OOD | arXiv 2411.10701 | feature-level reconstruction error for OOD; compact ID latent. |

## C. Gated / risk-adaptive KV-cache + text-level quality gating
| bibkey | title | id (verify) | one-line |
|---|---|---|---|
| `compilerkv` | CompilerKV: Risk-Adaptive KV Cache Compression | arXiv 2602.08686 (web) | risk-adaptive threshold gate; conservative retention for high-risk prompts. |
| `runtime-certified-kv` | Runtime-Certified Bounded-Error Quantized Attention | arXiv 2605.20868 (web) | online error bounds + four-rung fallback ladder to exact dense attention. |
| `racc-kv` | RACC: retrieval-augmented KV cache | OpenReview F7kDkYjBVa (web) | retrieve KV evicted by compression to recover long-output accuracy. |
| `compress-keep-commit` | Compress the Context, Keep the Commitments | arXiv 2605.17304 (web) | verifiable compression; conservative fallback for low-confidence semantic atoms. |
| `taac` | TAAC: quality-gated text compression | arXiv 2602.15843 (verify) | text-level quality gate, fall back to raw prompt. |
| `contextpilot` | ContextPilot: Fast Long-Context Inference via Context Reuse | MLSys 2026 (verify) | middleware: compress + quality score + safe fallback to original. |
| `characterizing-prompt-compr` | Characterizing Prompt Compression Methods for Long Context Inference | OpenReview vs6CCDuK7l (web) | query-aware abstractive compression > query-agnostic; survey of prompt-compression. |

## D. Adaptive reasoning compression / self-assessed confidence (mechanism cousins, reasoning regime)
| bibkey | title | id (verify) | one-line |
|---|---|---|---|
| `think-just-enough` | Think Just Enough: Self-Assessed Confidence for Adaptive Reasoning | EACL 2026 Findings (verify) | training-free self-assessed confidence to early-stop reasoning. |
| `conmax` | ConMax: Confidence-Maximizing Compression for CoT | arXiv 2601.04973 (web) | RL compresses reasoning traces by maximizing answer + thinking confidence. |
| `certainty-robustness` | Certainty Robustness Benchmark | arXiv 2603.03330 (web) | two-turn stability under self-challenging prompts; certainty vs accuracy. |

## E. Established baselines already in related-work (carried; ids mostly stable)
| bibkey | title | id |
|---|---|---|
| `gist` | Learning to Compress Prompts with Gist Tokens (Mu et al. 2023) | arXiv 2304.08467 / NeurIPS 2023 |
| `icae` | In-context Autoencoder (Ge et al.) | arXiv 2307.06945 / ICLR 2024 |
| `autocompressor` | Adapting LMs to Compress Contexts (Chevalier et al. 2023) | arXiv 2305.14788 |
| `cartridges` | Cartridges: Lightweight Long-Context Reps via Self-Study | arXiv 2506.06266 |
| `acon` | ACON: context compression for long-horizon agents | arXiv 2510.00615 |
| `selective-context` | Compressing Context to Enhance Inference Efficiency | EMNLP 2023 (2023.emnlp-main.391) |
| `snapkv` | SnapKV | arXiv 2404.14469 |
| `pyramidkv` | PyramidKV | arXiv 2406.02069 |
| `h2o` | H2O: heavy-hitter KV eviction | arXiv 2306.14048 |
| `streamingllm` | StreamingLLM (attention sinks) | arXiv 2309.17453 |
| `targ` | TARG: training-free retrieval gate | arXiv 2511.09803 |
| `self-rag` | Self-RAG | arXiv 2310.11511 |
| `flare` | FLARE | arXiv 2305.06983 |
| `adaptive-rag` | Adaptive-RAG | arXiv 2403.14403 |
| `llama-adapter` | LLaMA-Adapter (zero-init gated injection) | arXiv 2303.16199 |
| `prefix-tuning` | Prefix-Tuning | arXiv 2101.00190 |
| `llm-cascade-abstention` | Cost-Saving LLM Cascades with Early Abstention | arXiv 2502.09054 |
| `luo-cf` | An Empirical Study of Catastrophic Forgetting in LLMs (Luo et al.) | arXiv 2308.08747 |
| `pecft-survey` | Parameter-Efficient Continual Fine-Tuning survey | arXiv 2504.13822 |

---

## BibTeX (v1.7-critical; verify eprint ids before use)
```bibtex
@misc{ctxdistill-lmm,
  title  = {Context Distillation as Latent Memory Management},
  note   = {arXiv:2605.28889 (verify on arXiv before citing)}, year = {2026}
}
@misc{latent-ctx-compilation,
  title  = {Latent Context Compilation: Distilling Long Context into Compact Portable Memory},
  note   = {arXiv:2602.21221 (verify)}, year = {2026}
}
@misc{slt,
  title  = {Selective Latent Thinking: Adaptive Compression of LLM Reasoning Chains},
  note   = {arXiv:2605.25745 (verify)}, year = {2026}
}
@misc{pipo,
  title  = {Pair-In, Pair-Out: Latent Multi-Token Prediction for Efficient LLMs},
  note   = {arXiv:2605.27255 (verify)}, year = {2026}
}
@misc{six-arch-memory,
  title  = {Trained Persistent Memory for Frozen Encoder-Decoder LLMs: Six Architectural Methods},
  note   = {arXiv:2603.16413 (verify)}, year = {2026}
}
@misc{poc-compression,
  title  = {PoC: Performance-oriented Context Compression for LLMs via Performance Prediction},
  note   = {arXiv:2603.19733 (verify)}, year = {2026}
}
@misc{lclm-e2e,
  title  = {End-to-End Context Compression at Scale},
  note   = {arXiv:2606.09659 (verify)}, year = {2026}
}
@misc{compactor-kv,
  title  = {Compactor: Calibrated Query-Agnostic KV Cache Compression with Approximate Leverage Scores},
  note   = {arXiv:2507.08143 (verify)}, year = {2025}
}
@misc{recon-ae-maha,
  title  = {Improving Reconstruction Autoencoder Out-of-distribution Detection with Mahalanobis Distance},
  note   = {arXiv:1812.02765}, year = {2018}
}
```
