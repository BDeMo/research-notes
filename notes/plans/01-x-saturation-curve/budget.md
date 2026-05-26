# Plan 01 — Budget

## TL;DR

| Phase | Wall-clock | GPU-hours | Cloud $ (H100 @ $3/hr) | Decision point |
|---|---|---|---|---|
| **0. Single-day pilot** | 1 day | ≈ 20 | ≈ $60 | Does the curve have signal at all? |
| **1. Full curve sweep** | 1–2 weeks | ≈ 200 | ≈ $600 | Is $\mathcal{D}_W$ < 30% of pool? |
| **2. SFT comparison** | 1–2 weeks | ≈ 150 (4 runs × ~35) | ≈ $450 | Does $\mathcal{D}_W$-SFT beat baselines? |
| **3. Ablations + transfer** | 2–4 weeks | ≈ 300 | ≈ $900 | Do ablations hold? Cross-model transfer? |
| **4. Paper-scale runs (70B)** | 2–4 weeks | ≈ 1500 | ≈ $4500 | Camera-ready experiments |
| **Total (paper-track)** | ~2–3 months | **≈ 2200** | **≈ $6500** | |

## Phase 0 — single-day pilot

**Goal**: prove the curve has signal in <24 hours.

- Take 500 problems from GSM8K train.
- Base: Qwen2.5-7B-Base.
- Sweep 5 CoT budgets × 3 seeds = 15 inferences per problem.
- ~7,500 inferences. At ~1.5s per CoT-1024 inference on 1×H100: ~3 GPU-hours raw.
- Plus 2× SFT runs at 5 GPU-hours each = 10 GPU-hours.
- **Total ≈ 15–20 GPU-hours.**

Decision: if the curve is monotone-ish for ≥ 50% of problems and the $X$-saturated-wrong set is ≥ 5% of the 500 → proceed to Phase 1.

## Phase 1 — full curve sweep

**Goal**: build the curated $\mathcal{D}_W$ on the full candidate pool.

- 50k–100k problems × 5 budgets × 3 seeds → 750k–1.5M inferences.
- With proper batching + vLLM (~200 prob/s on H100 for short outputs, ~20 prob/s for CoT-4096): **~50–150 GPU-hours**.
- Output: JSONL with per-problem curves. ~1 GB.

Stretch: sweep 200k problems for stronger statistical power.

**Decision**: $\mathcal{D}_W$ should be 10–30% of pool. If < 5% or > 60%, re-tune $(\epsilon, \tau)$.

## Phase 2 — SFT comparisons

**Goal**: the headline experiment.

Four 7B SFT runs (A: random / B: ours / C: loss-mining / D: full):
- Train size: 5k–20k examples (B) up to ~100k (D).
- 3 epochs, bf16, LoRA-r=64 to save compute (sanity full-FT one run).
- ~30–50 GPU-hours per run on 8×H100 node.
- 3 seeds → 9 medium runs + 3 full-data large runs.
- **Total: ~150–250 GPU-hours.**

## Phase 3 — ablations + transfer

Across:
- Budget-axis ablation (sample-count vs CoT-length)
- Granularity ablation (3 vs 5 budgets)
- Saturation-definition ablation (slope vs probabilistic)
- Cross-model transfer (curate with Qwen-7B, train Llama-8B)
- Cross-task transfer (curate on MATH, evaluate on GSM8K)

≈ 12 additional runs × ~25 GPU-hours = ~300 GPU-hours.

## Phase 4 — paper-scale (70B)

Final headline numbers on Llama-3.1-70B-Base or Qwen2.5-72B-Base:
- Curve sweep: 50k problems × 3 budgets (skip top to save) × 1 seed = 150k inferences. ~200 GPU-hours on 8×H100.
- 4 SFT runs × ~300 GPU-hours each (LoRA) = ~1200 GPU-hours.

## Headcount

- **Solo-feasible** through Phase 3 with strong engineering (1 person, ~3 months part-time).
- Phase 4 benefits from 1 collaborator for parallel runs + 1 for paper writing.

## Tooling pre-reqs

- **vLLM** or **SGLang** for fast batched inference.
- **TRL** / **Axolotl** / **LLaMA-Factory** for SFT.
- **Weights & Biases** for tracking curves + SFT loss.
- **HuggingFace Datasets** for stable splits.

## $\$$ table (rough)

Assuming on-demand H100 @ $3/GPU-hr or 8×H100 node @ $24/hr:

| Phase | Estimated cost |
|---|---|
| 0 pilot | $60 |
| 1 curve sweep | $450 |
| 2 SFT comparisons | $600 |
| 3 ablations | $900 |
| 4 paper-scale | $4500 |
| **Total** | **$6500** |

With spot pricing / academic credits / NCSA-style allocations the effective $ can be 30–70% lower.

## Risk-adjusted budget

If we hit one major risk (e.g., curve too noisy → need more seeds), add ~30%. Plan for ~$8500 worst-case.

## When to kill the project (budget-wise)

If after Phase 0+1 (≈ 200 GPU-hours, ≈ $600) the W-residual set is < 5% or > 60% of pool *across reasonable $(\epsilon, \tau)$ settings*, **stop**. The curve does not have actionable signal at this scale.
