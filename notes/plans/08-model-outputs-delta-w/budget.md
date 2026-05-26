# Plan 08 — Budget

This is the most expensive of the three plans. Budget honestly.

## TL;DR

| Phase | Wall-clock | GPU-hours | Cloud $ (H100 @ $3/hr) | Decision point |
|---|---|---|---|---|
| **0. Toy proof (Gemma-2-2b, synthetic stream)** | 2–3 weeks | ~200 | ~$600 | Closed loop trainable at all? |
| **1. Phase-1 main experiments (continual learning)** | 1–2 months | ~2000 | ~$6000 | Benefit/harm rates positive? |
| **2. Phase-2 (personalization)** | 1–2 months | ~3000 | ~$9000 | Beats RAG by ≥ 5% absolute? |
| **3. Phase-3 (agents)** | 2–4 months | ~3000 | ~$9000 | Within-task learning observable? |
| **Total** | **8–12 months** | **~8200** | **~$24500** | |

This is a PhD-thesis-level project. Don't pretend it's a quick paper.

## Phase 0 — toy proof

**Goal**: verify the basic architecture trains end-to-end.
- Base: Gemma-2-2B-IT (matches D2L's setup, ~10× cheaper than 7B).
- Hypernet from Sakana checkpoint, retrofitted for emit-on-reasoning input.
- Synthetic 100-turn stream of (passage, question) pairs.
- Two seeds, Stage 1a (warmup) only — no RL yet.
- **~150–200 GPU-hours**.

Pass condition: hypernet loss decreases, post-application probe accuracy increases.

## Phase 1 — feasibility (main)

**Goal**: prove benefit-rate > 2 × harm-rate on synthetic streams.

Breakdown:
- Stage 1a (supervised warmup): ~500 GPU-hours.
- Stage 1b (RL): ~1000 GPU-hours (RL is more expensive due to rollouts).
- Ablations (verifier type, $\alpha$ schedule, capacity constraint): ~500 GPU-hours.

Total: **~2000 GPU-hours**. About 4 months on a single 8×H100 node.

Decision: if RL stage doesn't beat supervised warmup after 500 GPU-hours of RL training, abandon RL — ship a paper on the supervised-only version.

## Phase 2 — personalization utility

**Goal**: beat strong baselines on LaMP, PerLTQA, CLAM.

- Migrate to 7B base (Qwen2.5).
- Retrain hypernet on personalization streams: ~1500 GPU-hours.
- Baseline runs (frozen, RAG, D2L-as-memory, TTT): ~500 GPU-hours.
- Ablations across 3 benchmarks: ~1000 GPU-hours.

Total: **~3000 GPU-hours**. ~2 months calendar.

## Phase 3 — agents

**Goal**: within-task improvement on SWE-Bench / WebArena.

- Agent rollouts are expensive due to length: ~10× per-task compute vs Phase 2.
- ~500 problems × 3 baseline configs × ~5 GPU-hours per rollout: ~7500 GPU-rollout-hours but parallelizable.
- Realistically: ~3000 GPU-hours (a lot of debugging).

Phase 3 is where things get risky budget-wise. Reserve early; pause if cumulative budget exceeds $20K.

## Headcount

- 1 strong engineer + 1 strong researcher (12 months together).
- Optional: 1 systems person for serving infra (essential for Phase 3 if we want real-time deployment).
- Optional: 1 theory collaborator for the capacity / forgetting analysis.

## Hardware

- 8×H100 dedicated node for Phase 1–2.
- Burst capacity for Phase 3 (long agent rollouts benefit from parallelism).
- Disk: ~5 TB for checkpoints, logs, and trace data.

## Cost ladder (cumulative)

| Milestone | Cumulative $ | Decision |
|---|---|---|
| End Phase 0 | $600 | Continue iff loop trains |
| End Phase 1a (warmup) | $2100 | Continue iff probe acc moves |
| End Phase 1b (RL) | $5100 | Continue iff benefit/harm > 2 |
| End Phase 1 ablations | $6600 | Write paper if compelling; continue to Phase 2 |
| End Phase 2 | $15600 | Major paper if Phase 2 wins; otherwise wrap |
| End Phase 3 | $24500 | Thesis-scale result |

## Off-ramps

- **At end of Phase 0**: if it doesn't train at all, salvage as "hypernet for personalization without `<learn>` token" — pivot to a simpler D2L-extension paper.
- **At end of Phase 1**: if benefit/harm doesn't hit 2:1, salvage as "negative results on inference-time learning via emitted ΔW" — still publishable.
- **At end of Phase 2**: if we beat RAG by < 5% but match D2L-as-memory, refocus the paper on "model-emitted ΔW vs context-conditioned ΔW" comparison.

## Resource alternatives

- **Academic credits** (NCSA, ACCESS, NeurIPS volunteer compute): could cover ~30–50% of Phase 1.
- **Lambda Labs / RunPod spot** instances: save ~50% vs on-demand.
- **Smaller base model** (Gemma-2-2B all the way through): cuts budget by ~5–8× but limits the agentic scope.

## When to absolutely kill

- Phase 1: benefit < harm (catastrophic). Kill, write negative-result paper.
- Phase 2: no benchmark advantage at all over D2L-as-memory at matched compute. Kill, the `<learn>`-emit machinery is not the bottleneck.
- Anywhere: if AlphaEdit-style constrained edit *plus* a simple "summarize the last turn into the prompt" beats us on every metric. Then prompts win, weights lose — for now.
