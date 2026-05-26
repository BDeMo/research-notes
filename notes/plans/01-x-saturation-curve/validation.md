# Plan 01 — Validation

## Validation hypothesis

> *If the X-saturation curve carries signal about $W$-demand, then training on the X-saturated-residual subset will produce greater held-out accuracy gains per training FLOP than training on the full set or any equal-sized heuristic subset.*

This is the one sentence the entire project is trying to prove or refute.

## Experimental protocol

### Step 1 — Build the curve sweep tool

Given $(\text{model}, \text{problem set}, \text{X-budget grid})$, output for every problem a curve $\alpha(b)$.

X-budget axes to try (each is an experiment):
- **CoT length**: cap reasoning at $\{0, 64, 256, 1024, 4096\}$ tokens (truncate or stop on length).
- **Sample count**: Best-of-N with $N \in \{1, 4, 16, 64\}$, take pass@1 with verifier.
- **Search depth**: tree search (MCTS-style) with depth $\in \{0, 2, 4, 8\}$.
- **Retrieval depth**: top-$k$ for $k \in \{0, 1, 4, 16\}$.

Start with CoT length and sample count — cheap, no extra infra.

Output schema per problem:
```json
{
  "id": "...",
  "query": "...",
  "gold": "...",
  "curve": [{"budget": 1, "metric": 0.0}, {"budget": 8, "metric": 0.3}, ...],
  "saturated": true,
  "final_metric": 0.4,
  "in_W_residual": true
}
```

### Step 2 — Define and tune $(\epsilon, \tau)$

Two hyper-parameters:
- $\epsilon$ = plateau threshold (last two budget steps yield < $\epsilon$ marginal gain → saturated)
- $\tau$ = correctness target (saturated AND $\alpha < \tau$ → W-residual)

Sweep on a 1k-sample development set. Pick the pair that maximizes downstream SFT gain in a small pilot.

### Step 3 — Train comparisons (matched compute)

Train 4 SFT runs at *matched* training compute (≈ 5–10 GPU-hours each):

| Run | Training data | Size | Notes |
|---|---|---|---|
| **A** (baseline) | random subset of full | $|\mathcal{D}_W|$ | matched size to W-residual |
| **B** (ours) | $\mathcal{D}_W$ (X-saturated AND wrong) | $|\mathcal{D}_W|$ | the curated set |
| **C** (loss-mining) | top-loss subset | $|\mathcal{D}_W|$ | strong heuristic baseline |
| **D** (full) | full $\mathcal{D}_{\text{cand}}$ | full | sanity upper-bound |

Each run produces a held-out accuracy. Compare with paired bootstrap CIs.

### Step 4 — Held-out evaluation

Held-out sets must be **separate from the candidate pool**.
- For GSM8K / MATH: use official train / test splits; never sweep on test.
- For HumanEval+: pure held-out (no candidate examples from there).
- For MMLU-Pro: held-out subjects.

Report:
- Primary: pass@1 / EM on test
- Secondary: token efficiency (test accuracy at fixed CoT budget)
- Tertiary: "$X$-saturation curve shift" — does the trained model now require *less* $X$ to reach same accuracy? (This is the cleanest demonstration that we shifted demand from $X$ to $W$.)

## Ablations

The ablations that, if they fail, kill the paper:

1. **Curve granularity** — does coarser sweep (3 budgets vs 5) still work? If yes, big practical win.
2. **Budget axis** — CoT-length vs sample-count vs retrieval. Are they interchangeable? Or task-specific?
3. **Saturation definition** — slope-based vs probabilistic ("$P(\alpha\text{ improves at next budget}) < 0.1$"). Robustness check.
4. **Random equal-size subset matched on length / difficulty** — kills "the curated set is just shorter / easier".
5. **Curriculum** — does training in increasing $W$-residual order beat random order over the same set?
6. **Cross-model transfer** — does the curve from Model A transfer to selecting data for Model B?

## Baselines we must report against

- Random subsampling at matched size
- Loss-based hard-negative mining ([DEITA](https://arxiv.org/abs/2312.15685), [LESS](https://arxiv.org/abs/2402.04333))
- Reward-variance selection (RL-style)
- Self-consistency-disagreement selection (similar to "X-saturated AND uncertain")
- The full dataset (upper bound under no selection)

## Statistical practice

- Each held-out metric run with 3 random seeds for SFT.
- Report mean and 95% bootstrap CI.
- For pairwise claims, use the paired-bootstrap difference CI (not separate CIs).
- Pre-register the primary metric and decision rule before final runs.

## Failure modes to monitor

- **Verifier failure**: rule-based verifier gives false negatives on alternative-but-correct answers → curve underestimates $\alpha$. Mitigation: spot-check 100 samples.
- **Curve noise**: same $(q, b)$ pair has high variance across runs. Mitigation: average $\alpha$ over $\geq 3$ seeds in the sweep.
- **Distribution mismatch**: $\mathcal{D}_W$ becomes biased toward problem types where the verifier is harsh. Mitigation: include problem-type as a stratification variable in subset construction.
- **"X-saturation" is just "easy enough"**: the residual set is actually just easy problems the model already gets. Mitigation: enforce $\alpha(b_{\max}) < \tau$ filter explicitly.
