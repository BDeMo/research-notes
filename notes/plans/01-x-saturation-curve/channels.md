# Plan 01 — Validation channels

## Primary channels (cheap, rule-based verifiers)

### GSM8K
- **Why first**: tiny, fast, rule-based scoring, well-understood difficulty distribution.
- **Train pool**: 7,473 train problems.
- **Test**: 1,319 held-out.
- **Verifier**: exact-match on final numeric answer (strip commas, dollar signs).
- **Note**: many 7B+ models are near saturation. Choose a base model that gives ~50–70% baseline (e.g., Llama-3-8B-Base, *not* instruct).

### MATH
- **Why**: harder, broader, clearer $X$-saturation signal (more room to improve with more CoT).
- **Train pool**: 7,500 train problems.
- **Test**: 5,000 held-out (use level-5 stratified subset for stability).
- **Verifier**: SymPy-based or LLM-judge fallback for unparseable answers (rate-limit LLM-judge calls).

### HumanEval+ / MBPP+
- **Why**: tests transfer to a different modality (code) with execution-based scoring.
- **Train pool**: use a code-instruction dataset like Magicoder or OpenCodeInstruct.
- **Test**: HumanEval+ (164) and MBPP+ (378).
- **Verifier**: code execution with EvalPlus extended test suites.

## Secondary channels

### MMLU-Pro
- **Why**: knowledge-intensive, less amenable to extra CoT — should have many $W$-residual examples even at large CoT budgets.
- **Use**: as a *transfer test* after SFT on math.

### LongBench / RULER
- **Why**: long-context $X$-saturation. Different shape of curve (information *availability*, not *processing*).
- **Use**: optional ablation. Mainly relevant if we extend the budget axis to retrieval depth.

### AGIEval, BIG-Bench-Hard
- **Why**: distribution check. Should not regress on either after $\mathcal{D}_W$-SFT.

## Verifier infrastructure

- **Rule-based scoring** for GSM8K, MATH (via SymPy), code (execution).
- **Soft-judge** (Qwen-2.5-72B or Claude) only for tasks where rule-based fails; budget 1–5% of total inference cost.
- Avoid LLM-judge for the *primary* curve metric — too noisy and expensive at the scales here.

## Base models to sweep

In priority order:

| Model | Size | Why |
|---|---|---|
| **Qwen2.5-7B-Base** | 7B | Strong baseline, base-not-instruct gives larger headroom for SFT |
| **Llama-3.1-8B-Base** | 8B | Industry standard, easy comparison |
| **Qwen2.5-3B-Base** | 3B | Cheaper pilot iteration |
| **Llama-3.1-70B-Base** | 70B | Final result; only run after pilot signals stable |

Avoid instruct variants for the main results: they bias the $X$-saturation curve because the model is already "saturated" at low budgets due to RLHF.

## Datasets for the candidate pool $\mathcal{D}_{\text{cand}}$

To stress-test the curator:

- **MetaMathQA** (~395k math problems) — large pool with mixed difficulty.
- **OpenMathInstruct-2** — diverse difficulty levels.
- **CodeFeedback** / **Magicoder-Evol-Instruct-110k** — code candidate pool.
- **WizardMath / WizardCoder** — synthetic but useful for cross-distribution checks.

Filter for high-quality answers first (gold-verified or oracle-distilled). Don't waste curve compute on noisy gold.

## Comparable published baselines (data-selection)

These are the methods we need to beat (or match at matched compute):

- **DEITA** (Liu et al., 2024) — complexity + quality + diversity selection.
- **LESS** (Xia et al., 2024) — influence-function-based.
- **Self-Filtering** (Wang et al., 2023) — model-based filtering.
- **Reward-Variance Curriculum** (Razin et al., 2024) — RL-relevant baseline.

Report numbers against the strongest one that has a public implementation.

## Reproducibility deliverables

- Curve sweep code (LMK / vLLM-based, ~300 LoC).
- Per-problem curve JSONLs released for GSM8K + MATH train.
- SFT scripts (HF Trainer / TRL).
- Hyper-parameters and seeds.
- Trained checkpoints for runs A/B/C/D.
