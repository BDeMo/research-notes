# Plan 03 — Validation

## Validation hypothesis

> *W-diversity at inference time provides coverage that is orthogonal to X-diversity (temperature sampling).
> At matched FLOPs, combining the two yields a strictly better compute-vs-quality Pareto front than either alone.*

## Experimental progression

### Stage 0 — Sanity prototype (1 day)

- Pick 100 MATH-level-5 problems.
- Base model: Qwen2.5-7B-Instruct.
- Generate 4 LoRAs $\{\Delta W_i\}$ by Gaussian noise on `down_proj` of MLP layers (matching D2L's target sites). $\sigma$ swept in $\{10^{-3}, 10^{-2}, 10^{-1}\}$.
- For each problem, generate one answer per LoRA + verifier-select.
- Compare to 4-sample temperature-BoN at $T = 0.7$.
- **Pass condition**: WBoN-Noise reaches at least 70% of $X$-BoN at the best $\sigma$. If yes, hypernet path is justified.

### Stage 1 — WBoN-Noise calibration

- Sweep $\sigma$ on finer grid + sweep where the LoRA is injected (MLP `down_proj` only / attention `q_proj` `v_proj` only / all linear).
- Sweep rank $r \in \{4, 8, 16, 32\}$.
- Goal: find the best blind-noise configuration. This is the bar that smarter methods must beat.

### Stage 2 — WBoN-Library (pre-trained experts)

- Train 8 LoRAs on disjoint sub-distributions of math (algebra / geometry / calculus / probability / number theory / combinatorics / proof / word-problems).
- Inference: try all 8, verifier picks.
- Compare to:
  - WBoN-Noise (same N)
  - $X$-BoN (same N)
  - $X$-BoN at $N = 8 \times$ inference FLOPs of WBoN-Library.
- Hypothesis: pre-trained experts win on the diversity-coverage front but not on hard problems outside the union.

### Stage 3 — WBoN-Hypernet (the main result)

Design choices:
- Initialize the hypernet from Doc-to-LoRA architecture (Perceiver, ~300M params).
- Input to hypernet: tokenized query + a noise seed $z$.
- Output: $\Delta W$ for LoRA-r=8 on `down_proj`.
- Training objective: maximize verifier reward over the verifier-best answer in a batch of $N$ generated. Adds an explicit diversity loss (entropy on the $\Delta W$ ensemble or InfoNCE between LoRAs).

Train via:
- **Stage 3a — distillation**: clone the best results of Stage 2 into the hypernet to bootstrap.
- **Stage 3b — RL**: reward = verifier-selected best of the $N$ generated; gradient flows through to the hypernet via PPO/REINFORCE. (LoRA application is differentiable; sampling $z$ is not — Gumbel or just stochastic policy on $z$.)

### Stage 4 — Hybrid WBoN + XBoN

Use the budget for *both* axes:
- Allocate $\sqrt{N}$ to each axis, generating $\sqrt{N}^2 = N$ answers from $\sqrt{N}$ LoRAs × $\sqrt{N}$ X-samples each.
- Or: an adaptive allocator that decides per-query.

## Required ablations

| Ablation | Tests |
|---|---|
| **$\sigma = 0$ control** | No noise → trivially identical samples → verifier picks any |
| **Verifier swap** | Replace gold verifier with a learned reward model. Does WBoN still beat $X$-BoN? Important for generalization beyond rule-based domains. |
| **N-curve** | Plot accuracy vs $N \in \{1, 2, 4, 8, 16, 32, 64\}$. Identify saturation point per axis. |
| **Cost-matched** | Fix total inference FLOPs. Compare WBoN-N=K vs XBoN-N=K' at matched FLOPs. (LoRA inference may be slightly slower per sample.) |
| **Hypernet capacity** | Hypernet param count vs WBoN gain. |
| **Diversity diagnostic** | Pairwise output similarity (BLEU / embedding cos-sim) between samples — confirms WBoN samples are more diverse. |
| **Out-of-distribution probe** | Take a model trained for math, test WBoN on out-of-domain (code). Does the hypernet generalize or overfit to its training tasks? |

## Metrics

- **Primary**: pass@1 with verifier-selection at fixed $N$ on MATH / HumanEval+.
- **Secondary**:
  - **Coverage gain** = (problems WBoN solves but XBoN doesn't) / total solved.
  - **Pareto frontier**: accuracy vs total FLOPs.
  - **Diversity**: mean pairwise output dissimilarity.
  - **Verifier-agreement**: does the verifier pick a *correct* answer when one is in the candidate pool?

## Risks + mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Verifier picks a confidently-wrong sample more often under WBoN (LoRAs make model overconfident) | medium | Calibrate verifier; report verifier-success-given-candidate-exists |
| LoRA noise breaks instruction-following → garbage outputs | high at large $\sigma$ | $\sigma$ sweep; reject samples that fail a basic format check |
| Hypernet collapses to one LoRA | high | Diversity loss; multiple noise seeds; train-time entropy bonus |
| WBoN looks good only because of compute leak (more total FLOPs) | medium | Strict FLOP matching in the headline comparison |
| All wins are on easy problems already at 100% with XBoN | medium | Stratify by problem difficulty, report stratum-level gains |

## Statistical practice

- Compare at fixed $N$ AND at fixed total FLOPs (two paired plots).
- Bootstrap CIs on accuracy difference per problem set.
- Pre-register the primary comparison: WBoN-Hypernet $N=8$ vs $X$-BoN $N=8$ on MATH-level-5.
