# Inference-time compute

> Scaling test-time compute along the **$X$-axis**: more reasoning tokens, more samples, deeper search, more retrieval. The dominant 2024–2025 paradigm for getting better LLM outputs without retraining. Sibling to [`inference-time-training`](../inference-time-training/), which scales along the *weights* axis.

## Structure

**Contained by**: (top-level — this folder represents the "$X$-axis" sibling to inference-time-training).

**Contains**:
- **Chain-of-Thought (CoT) scaling** — longer reasoning sequences (o1, R1).
- **Sampling / Best-of-N** — generate $N$ candidates, verifier or self-consistency picks. (See also plan 03 for the W-axis analog.)
- **Tree / search-based** — Tree-of-Thoughts, MCTS, ReST-MCTS\*, RAP.
- **RL on reasoning traces** — GRPO / PPO with verifier-shaped rewards (R1).
- **Self-correction / self-refine** — model iterates on its own output.
- **Long-context exploitation** — using long inputs as a form of compute (overlaps [`long-context`](../long-context/)).

## Nearest neighbors

- [`inference-time-training`](../inference-time-training/) — the sibling axis. Same goal, different lever.
- [`long-context`](../long-context/) — long inputs are one form of $X$-scaling; their limits are why people now look at the $W$-axis.

## Key concepts

- **$X$-saturation** — for any given query and model, accuracy plateaus as $X$-budget increases. The plateau height defines $W$-demand (the residual). See plan 01.
- **Verifier-based selection** — programmatic (math/code) or learned (PRM, ORM) verifier picks the best of $N$ samples. Quality of verifier dominates the gain from larger $N$.
- **Process Reward Model (PRM)** — verifies *each step* of a CoT, enabling tree search and step-level filtering.
- **Outcome Reward Model (ORM)** — verifies only the final answer. Cheaper but less informative.
- **RL on traces** — STaR-style loop: generate CoT, filter by correctness, train. Modern RL methods (GRPO) skip the explicit filter and use the gradient directly.
- **Exchange rate** — how much $X$-FLOPs buys what accuracy. Empirically follows soft scaling laws (Snell 2024).
- **Inference-time compute vs train-time compute** — Snell 2024: for a fixed budget, optimal allocation depends on problem difficulty distribution.

## Foundational references

- OpenAI (2024). **Introducing o1.** Defined the "long CoT" inference-time-compute regime.
- [r1] DeepSeek-AI (2025). **DeepSeek-R1.** [arXiv:2501.12948](https://arxiv.org/abs/2501.12948). Open-weights RL-on-reasoning replication.
- [scaling-tt] Snell, C., Lee, J., Xu, K., Kumar, A. (2024). **Scaling LLM Test-Time Compute Optimally.** [arXiv:2408.03314](https://arxiv.org/abs/2408.03314). The canonical scaling-law paper for this axis.
- Wang, X., Wei, J. et al. (2022). **Self-Consistency.** [arXiv:2203.11171](https://arxiv.org/abs/2203.11171).
- Lightman, H. et al. (2023). **Let's Verify Step by Step (PRM800k).** [arXiv:2305.20050](https://arxiv.org/abs/2305.20050).
- Yao, S. et al. (2023). **Tree of Thoughts.** [arXiv:2305.10601](https://arxiv.org/abs/2305.10601).
- Chen, Z. et al. (2024). **ReST-MCTS\*.**
- Zelikman, E. et al. (NeurIPS 2022). **STaR: Bootstrapping Reasoning with Reasoning.**
- Singh, A. et al. (2023). **ReST^EM.**
- Madaan, A. et al. (NeurIPS 2023). **Self-Refine.**
- Shinn, N. et al. (NeurIPS 2023). **Reflexion.**

## Open questions / live debates

- **Diminishing returns** — does CoT scaling top out, and if so where?
- **Reasoning vs memorization** — does long CoT actually generalize, or just memorize traces?
- **Combined X+W scaling** — given a fixed compute budget, is it better to do more $X$ or to do some $W$-update (e.g., a small TTT step)? Plan 01's framing operationalizes this.
- **Verifier hacking** — long-CoT RL eventually games the verifier; how to prevent without slowing iteration.
