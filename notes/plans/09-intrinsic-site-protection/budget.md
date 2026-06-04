# Plan 09 — Budget

## TL;DR

| Phase | Wall-clock | GPU-hours | Cloud $ (H100 @ $3/hr) | Decision gate |
|---|---|---|---|---|
| **0. Tooling** | 3–5 days | ≈ 40 | ≈ $120 | detectors reproduce known sinks / massive-acts / super-experts |
| **1. Observation study** | 2–3 weeks | ≈ 400 | ≈ $1.2K | **H1 + H2**: ρ(long-ctx, forgetting) ≥ 0.4 on ≥2 models/domains? |
| **2. Method design + causal test** | 2–4 weeks | ≈ 500 | ≈ $1.5K | **H3**: protection cuts forgetting ≥30%, beats best baseline ≥5%? |
| **3. Paper eval (cross-X, 4-seed)** | 4–6 weeks | ≈ 2200 | ≈ $6.6K | headline tables hold across models/domains |
| **Total (paper-track)** | ~3–4 months | **≈ 3140** | **≈ $9.4K** | |

The **Phase-1 gate is the cheap de-risk** (~$1.3K including tooling): it can confirm/kill the unifying thesis before any method work.

## Phase 0 — tooling (~40 GPU-h)
- Implement detectors (sink/massive-act/retrieval-head/induction/super-expert/SVD) + drift meters (‖ΔW‖, Fisher, CKA, KL-via-lens, routing drift) + intervention hooks.
- Validate on Qwen3-8B + Qwen3-30B-A3B against published phenomena. Mostly inference + small SFT for the drift meters.

## Phase 1 — observation study (~400 GPU-h)
- Site detection on generic text: cheap inference, a few GPU-h per model.
- **Forgetting probes**: SFT each model on {RCA, code, math, proxy} × {light, heavy dose} → ~8 short SFT runs/model × 3 models ≈ 24 runs × ~10 GPU-h ≈ 240 GPU-h.
- **Long-ctx probes**: RULER/NIAH sweeps × lengths × models ≈ 100 GPU-h.
- **Coupling + causal-preview interventions**: a handful of protected-SFT runs ≈ 60 GPU-h.
- Output: coupling maps + the headline scatter figure. **Gate on H2.**

## Phase 2 — method + causal test (~500 GPU-h)
- Implement the variant Phase 1 selects (V1–V5).
- Causal comparison {full SFT, best baseline, ours} × 3 models × 4 seeds, plus k-sweep ≈ 500 GPU-h.
- **Gate on H3.**

## Phase 3 — paper eval (~2200 GPU-h)
- Cross-domain forgetting matrix ({RCA,code,math,proxy}², 4 seeds) + cross-task continual sequences + cross-model (4 sizes + 2 families + dense/MoE) + all baselines on the decisive cells.
- Long-ctx (RULER/HELMET/∞Bench) + capability (lm-eval-harness/EvalPlus) + TRACE.
- This is the bulk; parallelizable.

## Headcount
- **Solo-feasible** through Phase 2 (1 person, ~6 weeks, strong infra).
- Phase 3 benefits from 1 collaborator for parallel baseline runs + 1 for writing.

## Tooling pre-reqs
- HF Transformers (hooks), **PEFT/TRL/LLaMA-Factory** (SFT + protection wrapper), **vLLM/SGLang** (long-ctx inference), **lm-evaluation-harness**, **RULER** + **HELMET** + **EvalPlus** + **TRACE**, **W&B**.
- 1× 8×H100 node (or 4×H100 for Phase 0–1) covers everything; MoE-30B fits with ZeRO-3 / offload.

## $ table (rough)
| Phase | $ |
|---|---|
| 0 tooling | $120 |
| 1 observation (de-risk) | $1.2K |
| 2 method + causal | $1.5K |
| 3 paper eval | $6.6K |
| **Total** | **≈ $9.4K** |

Spot pricing / academic credits → 30–70% lower. Risk-adjusted worst case (more seeds, bigger MoE) ≈ $13K.

## When to kill (budget-wise)
- After Phase 0+1 (≈ $1.3K): if H2's ρ is near zero across models/domains, **stop** — the unifying thesis is false. Salvage value: a short "long-ctx and forgetting are decoupled" negative-result note.
- After Phase 2: if H3 fails despite H2 holding, pivot to a **pure measurement/observation paper** (no method claim) — still publishable, much cheaper than pushing a dead method to Phase 3.
