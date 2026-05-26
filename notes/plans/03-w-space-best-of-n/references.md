# Plan 03 — References

## Test-time compute / X-axis search (the baselines)

- **Self-Consistency** — Wang, X., Wei, J. et al. (2022). *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* [arXiv:2203.11171](https://arxiv.org/abs/2203.11171). The original Best-of-N along $X$.
- **Best-of-N with PRMs** — Lightman, H. et al. (2023). *Let's Verify Step by Step.* [arXiv:2305.20050](https://arxiv.org/abs/2305.20050). Verifier-selected BoN; the prior art that WBoN must beat.
- **Tree-of-Thoughts** — Yao, S. et al. (2023). [arXiv:2305.10601](https://arxiv.org/abs/2305.10601).
- **ReST-MCTS\*** — Chen, Z. et al. (2024). Tree search + reward model.
- **Scaling Test-Time Compute** — Snell, C. et al. (2024). [arXiv:2408.03314](https://arxiv.org/abs/2408.03314). Quantifies the FLOPs vs accuracy curve for X-axis methods.
- **AlphaCode** — Li, Y. et al. (2022). Massive sampling + filtering for code generation.

## W-axis methods (closest cousins)

- **LoRA-Hub** — Huang et al. (2023). [arXiv:2307.13269](https://arxiv.org/abs/2307.13269). Training-time LoRA combination. Closest spiritual cousin; we adopt the modular idea but apply it at inference.
- **Doc-to-LoRA** — Charakorn et al. (2026). [arXiv:2602.15902](https://arxiv.org/abs/2602.15902). Provides the hypernet starting point.
- **Text-to-LoRA** — Charakorn et al., ICML 2025. Hypernet conditioned on task descriptions — could be the $\Delta W_i$ generator if we adapt it.
- **HyperLoRA** — Lv et al., ACL 2024. Cross-task hypernet.
- **MoLE** (Mixture of LoRA Experts) — Wu et al., 2024. Routing among LoRAs at inference. We push toward sampling and Best-of-N rather than routing.
- **Bayesian LoRA / SWAG-style ensembling** — adapt classical deep-ensemble methods to LoRA-space.

## Hypernet machinery

- **Hypernetworks** — Ha, D., Dai, A., Le, Q. (2016). [arXiv:1609.09106](https://arxiv.org/abs/1609.09106). The original.
- **HyperTuning** — Phang, J. et al., ICML 2023. Hypernet without backprop at test time.
- **HINT** — Ivison, H. et al., ACL 2023. Hypernet for instruction tuning.
- **MEND (Meta-Demonstration Distillation)** — Li, Y. et al., ICLR 2024. Hypernet that compresses in-context examples.

## Verifiers / reward models

- **PRM800k** — [github.com/openai/prm800k](https://github.com/openai/prm800k). Math step-level rewards.
- **Math-Shepherd** — Wang, Y. et al. (2024). Process reward model for math.
- **OmegaPRM** — Luo et al. (2024). Improved PRM via Monte Carlo.

## Diversity & exploration (theory we lean on)

- **Implicit Reparameterization Gradients** — Figurnov et al. (2018). For training with sampled $z$ → $\Delta W$.
- **Wasserstein loss for diversity** — for the entropy bonus in Stage 3b.
- **Determinantal Point Processes** — Kulesza & Taskar (2012). One option for diversity-aware selection.

## Systems papers (for serving)

- **S-LoRA** — Sheng, Y. et al., MLSys 2024. Serving thousands of LoRA adapters concurrently.
- **Punica** — Chen, L. et al., 2023. Multi-tenant LoRA serving.

## Open-source code

- [vLLM](https://github.com/vllm-project/vllm) — supports LoRA hot-swap.
- [S-LoRA](https://github.com/S-LoRA/S-LoRA).
- [Sakana doc-to-lora](https://github.com/SakanaAI/doc-to-lora).
- [TRL](https://github.com/huggingface/trl) — PPO / GRPO for the hypernet RL stage.
- [PEFT](https://github.com/huggingface/peft).

## Datasets

- [Lots-of-LoRAs](https://huggingface.co/datasets/Lots-of-LoRAs/Lots-of-LoRAs) — for the WBoN-Library stage; ~479 task-specific LoRAs ready to use.
- [PRM800k](https://github.com/openai/prm800k) — for PRM-style verifier.
- [MATH](https://github.com/hendrycks/math) — primary benchmark.
- [EvalPlus](https://github.com/evalplus/evalplus) — code verifier.
