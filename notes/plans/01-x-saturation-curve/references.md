# Plan 01 — References

Closest related work, with our take.

## Data selection / curriculum

- **DEITA** — Liu, W., Zeng, W., et al. (2024). *What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning.* [arXiv:2312.15685](https://arxiv.org/abs/2312.15685)
  - Their selection criterion = complexity × quality × diversity. Ours = "model can't solve it with more X." Orthogonal axis.
- **LESS** — Xia, M., Malladi, S., et al. (2024). *LESS: Selecting Influential Data for Targeted Instruction Tuning.* [arXiv:2402.04333](https://arxiv.org/abs/2402.04333)
  - Influence-function based. We share the goal (pick data that matters) but use a different signal source.
- **Self-Filter** — Wang, Y. et al. (2023). *Self-Instruct: Aligning Language Models with Self-Generated Instructions.*
  - Uses base model to filter own training data. Predecessor in spirit.
- **Reward-variance curriculum** — Razin, N. et al. (2024). *Vanishing Gradients in Reinforcement Finetuning of Language Models.*
  - Argues for data with informative reward gradients. Adjacent.

## Inference-time compute / test-time scaling

- **Scaling Test-Time Compute** — Snell, C., Lee, J., Xu, K., Kumar, A. (2024). [arXiv:2408.03314](https://arxiv.org/abs/2408.03314).
  - Quantifies the exchange between inference compute and model size. Closest ancestor of the X-saturation framing.
- **OpenAI o1 / DeepSeek-R1** — popularized "long CoT" as the dominant X-budget axis.

## Self-improvement / verifier-driven training

- **STaR** — Zelikman, E. et al. (2022). *STaR: Bootstrapping Reasoning With Reasoning.* NeurIPS.
  - Loop: generate CoT, keep correct ones, SFT. Closely related — but does *not* differentiate X-residual from W-residual.
- **V-STaR / Quiet-STaR** — extensions.
- **ReST^EM** — Singh et al. (2023).
- **Self-Rewarding LMs** — Yuan et al. (2024).

## Difficulty / failure analysis

- **CIRCUS / Difficulty estimation in NLP** — various papers; relevant for the $(\epsilon, \tau)$ tuning.
- **The Calibration of Language Models** — Kadavath et al. (2022). Models can sometimes self-estimate whether more X will help.

## Verifier infrastructure

- **MATH official scorer**, **SymPy-based grading scripts** in [PRM800k](https://github.com/openai/prm800k).
- **EvalPlus** — Liu, J., Xia, C., Wang, Y., Zhang, L. (2023). For code verification.

## Benchmarks

- **GSM8K** — Cobbe, K. et al. (2021). [arXiv:2110.14168](https://arxiv.org/abs/2110.14168).
- **MATH** — Hendrycks, D. et al. (2021). [arXiv:2103.03874](https://arxiv.org/abs/2103.03874).
- **HumanEval+** — EvalPlus, [arXiv:2305.01210](https://arxiv.org/abs/2305.01210).
- **MMLU-Pro** — Wang, Y. et al. (2024).
- **LongBench** — Bai et al. (2023).
- **RULER** — Hsieh et al. (2024).

## Theory of inference-time learning

- **Information-theoretic bounds on prompt-length / model-size trade-off** — see Snell 2024 above.
- **Implicit Bayesian inference under ICL** — Xie, S. et al. (2022). [arXiv:2111.02080](https://arxiv.org/abs/2111.02080).
- **Power-law of test-time compute** — see Kaplan et al. extensions; informal but useful for budget planning.

## Open implementations to fork

- [TRL](https://github.com/huggingface/trl), [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for SFT.
- [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang) for fast inference during sweep.
- [PRM800k repo](https://github.com/openai/prm800k) for math verifier.
