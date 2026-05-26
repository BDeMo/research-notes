# Plan 03 — Budget

## TL;DR

| Phase | Wall-clock | GPU-hours | Cloud $ (H100 @ $3/hr) | Decision point |
|---|---|---|---|---|
| **0. Sanity prototype** | 1 day | ~5 | ~$15 | Does WBoN-Noise show ≥ 70% of XBoN? |
| **1. Noise/library sweep** | 1 week | ~80 | ~$240 | What's the best blind baseline? |
| **2. Hypernet training** | 2–4 weeks | ~1000 | ~$3000 | Hypernet beats Stage 1 best? |
| **3. Headline experiments** | 2 weeks | ~300 | ~$900 | Hybrid + paper-scale |
| **4. Ablations** | 2 weeks | ~200 | ~$600 | Robustness checks |
| **Total (paper-track)** | ~2–3 months | **≈ 1600** | **≈ $4800** | |

## Phase 0 — sanity prototype

- 100 MATH-L5 problems.
- 4 LoRAs (Gaussian noise) × 3 σ values + 4 X-BoN samples.
- Total: 100 × (12 + 4) = 1600 inferences at ~5s/inference (with CoT) on 1×H100.
- **~3 GPU-hours**. Add a half hour for the verifier loop = ~5 GPU-hours.

Decision: continue iff WBoN ≥ 0.7 × XBoN at any σ.

## Phase 1 — noise/library sweep

- ~1000 MATH problems × 8 LoRAs × 4 hyper-config combos = ~32k inferences.
- ~50–80 GPU-hours total.
- For LoRA-Library: train 8 sub-domain LoRAs (each ~4 GPU-hours).

## Phase 2 — hypernet training

Largest budget item. Modeled on Doc-to-LoRA: 5 days × 8 H200 = ~960 GPU-hours for Sakana's run.

We can be cheaper:
- Smaller hypernet (~100M params instead of 300M)
- Smaller target LLM (Qwen2.5-7B instead of larger)
- Shorter meta-training (target ~100–300 GPU-hours for v0; ~1000 for paper version)

Stages:
- 3a — distillation init: ~50 GPU-hours.
- 3b — RL with verifier: ~500–1000 GPU-hours.

## Phase 3 — headline experiments

- WBoN vs XBoN at $N \in \{1, 2, 4, 8, 16, 32\}$ on MATH, HumanEval+, MMLU-Pro.
- Per benchmark: ~500 problems × 32 samples = 16k inferences.
- Three benchmarks → ~50k inferences. At ~3s per inference: ~40 GPU-hours.
- Multiply by hyperparameter sweep (LoRA inject site, verifier type) → ~250–300 GPU-hours.

## Phase 4 — ablations

- Diversity-vs-quality decomposition: ~50 GPU-hours.
- OOD probe (math→code): ~50 GPU-hours.
- Verifier swap (PRM): ~50 GPU-hours.
- Hybrid WBoN+XBoN search: ~50 GPU-hours.
- **Total**: ~200 GPU-hours.

## Headcount

- 1 strong engineer can do Stages 0–1 in a week.
- Stages 2–4 benefit from a second engineer for the RL pipeline.
- A theorist friend is nice-to-have for the diversity / coverage analysis.

## $\$$ table

| Phase | Cost |
|---|---|
| 0 sanity | $15 |
| 1 sweep | $240 |
| 2 hypernet | $3000 |
| 3 headline | $900 |
| 4 ablations | $600 |
| **Total** | **$4800** |

If we want to skip the trained-hypernet path entirely (stop at Phase 1), the project still has a story: **"$W$-space test-time scaling exists, even without training."** That cuts cost to ~$1800.

## Hardware notes

- Hypernet training fits comfortably on a single 8×H100 node.
- WBoN inference benchmarks: use vLLM with [LoRA hot-swap](https://docs.vllm.ai/en/latest/features/lora.html) to batch heterogeneous LoRAs.
- Reserve a single A100-80GB instance for verifier-heavy stages (math/code verification doesn't need a GPU but co-locating with inference simplifies orchestration).

## When to kill (budget-wise)

- After Phase 0: if WBoN-Noise < 0.5 × XBoN even at the best σ → premise is wrong, kill.
- After Phase 1: if best WBoN-Library < XBoN at matched FLOPs → kill or replan around the **disjoint-coverage** angle only.
- After Phase 2 (mid-training): if RL reward doesn't improve over Stage 1 best after 200 GPU-hours of training → kill the RL approach, fall back to distillation-only.
