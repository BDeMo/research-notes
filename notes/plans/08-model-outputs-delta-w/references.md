# Plan 08 — References

## Hypernet → weight delta (direct ancestors)

- **Doc-to-LoRA** — Charakorn et al., Sakana AI (2026). [arXiv:2602.15902](https://arxiv.org/abs/2602.15902). Provides the hypernet architecture and proves single-pass LoRA generation works.
- **Text-to-LoRA** — Charakorn et al., ICML 2025. Task description → LoRA. Foundational for the `<learn>` text conditioning.
- **Generative Adapter** — Chen, T. et al., ICLR 2025. Closest peer to D2L using NTP loss. Read carefully — its training procedure has lessons.
- **HyperTuning** — Phang, J. et al., ICML 2023. Hypernet without backprop at inference.
- **HINT** — Ivison, H. et al., ACL 2023.
- **MEND** — Li, Y. et al., ICLR 2024.

## Model editing (the closest weight-update prior)

- **MEMIT** — Meng, K. et al., 2023. Mass model editing.
- **ROME** — Meng, K. et al., 2022. Locate-then-edit.
- **AlphaEdit** — Fang, J. et al., ICLR 2025. **Critical reference**: null-space constrained editing that preserves capability. Our $\Delta W$ pipeline should adopt this constraint or risk catastrophic forgetting.
- **MEND (Editing)** — Mitchell, E. et al., ICLR 2022. Hypernet for *editing*.
- **Knowledge editing surveys** — Yao, Y. et al. (2023); Wang, S. et al. (2024).

## Continual learning / sparse updates

- **Sparse Memory Finetuning** — Lin, J. et al., 2025. [arXiv:2510.15103](https://arxiv.org/abs/2510.15103). Update only top-K MLP rows — directly applicable as the W-parameterization here.
- **Continual learning with hypernetworks** — von Oswald, J. et al., ICLR 2020.
- **Continual LM survey** — Shi, H. et al., ACM Comput. Surv. 2025.

## Self-improvement / RL on reasoning

- **STaR** — Zelikman, E. et al., NeurIPS 2022. The grandfather paper for "model generates training data for itself."
- **V-STaR** — Hosseini, A. et al., 2024.
- **ReST^EM** — Singh, A. et al., 2023.
- **Self-Refine** — Madaan, A. et al., NeurIPS 2023.
- **Reflexion** — Shinn, N. et al., NeurIPS 2023. Prompt-side analog of what we propose to do in weights.
- **DPO** — Rafailov, R. et al., NeurIPS 2023. For the `<learn>`-emit gating signal.
- **Self-Rewarding LMs** — Yuan, W. et al., 2024.

## Test-time training

- **TTT** — Sun, Y. et al., ICML 2020. Original.
- **TTT-Layers** — Sun, Y., Li, X., et al., 2024. Modern revival.
- **TENT** — Wang, D. et al., ICLR 2021.
- **MEMO** — Zhang, M., Levine, S., Finn, C., NeurIPS 2022.
- **Inference-Time Training for Long Text Generation** — Wang, Y., Ma, D., Cai, D., COLM 2024.

## Context distillation / sleep-time compute

- **Cartridges** — Eyuboglu, S. et al. (2025). [arXiv:2506.06266](https://arxiv.org/abs/2506.06266). Sleep-time CD via prefix-tuning.
- **Sleep-time compute** — Lin, K. et al. (2025).
- **Context distillation classics** — Askell et al. (2021); Snell, Klein, Zhong (2023).
- **Prompt baking** — Bhargava et al. (2024).
- **In-context editing** — Qi, S. et al., ICLR 2025.

## On-policy distillation on latent memory / reasoning

> Survey doc: `v0-opd-on-latent-memory-survey-2026-06-01.md` — written 2026-06-01.
> **Headline**: the *closest* prior art to the v1 OPD-on-wrapper pipeline is OPCD (Microsoft, Feb 2026) and LatentMem-LMPO (Tencent et al., Feb 2026); neither does the exact thing we do (recurrent latent memory wrapper on a strictly frozen base + three-stage SFT→OPD→REINFORCE).

- **OPCD — On-Policy Context Distillation for Language Models** — Ye et al., MSR (Furu Wei). [arXiv:2602.12275](https://arxiv.org/abs/2602.12275). Bridges OPD with context distillation; **internalizes context into model weights**, not into a wrapper.
- **Privileged Information Distillation (π-Distill / OPSD)** — [arXiv:2602.04942](https://arxiv.org/abs/2602.04942). Teacher sees CoT/trace; student doesn't; OPD with reverse KL on PI-conditioned teacher.
- **On-Policy Prefix Distillation** — Zhang et al., Optum AI (2026-02). [arXiv:2602.15260](https://arxiv.org/abs/2602.15260). Cheap OPD by supervising only the prefix of student rollouts.
- **A Survey of On-Policy Distillation for LLMs** — (2026). [arXiv:2604.00626](https://arxiv.org/abs/2604.00626).
- **LatentMem / LMPO** — Wang et al. (2026-02). [arXiv:2602.03036](https://arxiv.org/abs/2602.03036). **GRPO-variant backprop through latent memories** in a multi-agent system with frozen backbones — *closest cousin* to our RL phase, just multi-agent not single-base.
- **MEM1** — Zhou et al., NeurIPS 2025. End-to-end RL trains agent to consolidate context into a hidden state; trains the base, not a wrapper.
- **MemRL** — Wang et al. (2026-01). [arXiv:2601.03192](https://arxiv.org/abs/2601.03192). Non-parametric PPO over an episodic text-bank memory with frozen LLM — frame & MDP formulation we should adopt.
- **Memory-R1** — Yan et al. (2025-08). [arXiv:2508.19828](https://arxiv.org/abs/2508.19828). GRPO over add/update/delete ops on a **text** memory bank.
- **MemFactory** — (2026-03). Unified framework for GRPO on text-mediated memory ops.
- **Latent RL on Coconut** — *Reinforcement Learning for Latent-Space Thinking in LLMs* (2025-12). [arXiv:2512.11816](https://arxiv.org/abs/2512.11816). Value head fixes Coconut's "indirect supervision to latent steps" problem — directly applicable to our m_T.
- **Shadow Mask Distillation** — (2026-05). [arXiv:2605.06850](https://arxiv.org/abs/2605.06850). On-policy alignment for KV cache compression under RL; the trick that ensures rollout and gradient-time policies match — sanity check for our RLTrainer.
- **Deep Context Distillation (DCD)** — Caccia et al. (2025-03). [arXiv:2503.08727](https://arxiv.org/abs/2503.08727). Per-document LoRA via hidden-state + output KL — supervised, not on-policy.
- **Context Distillation as Latent Memory Management** — (2026-05). [arXiv:2605.28889](https://arxiv.org/abs/2605.28889). Per-context LoRA + cache-sharing self-gating; off-policy KL.
- **MEMO** — modular MEMORY model + frozen executive (2026-05).
- **MemoryPrompt** — Pannitto et al., LREC 2024. Light recurrent wrapper writing 5 soft vectors to a frozen base — **architecture is the closest direct ancestor of ours**, but trained with standard NTP CE only.

## Agentic memory & long-horizon agents

- **MemGPT** — Packer, C. et al. (2023).
- **MemoryBank** — Zhong, W. et al. (2023).
- **A-MEM** — Xu, F. et al. (2025). Agent memory architectures.
- **Voyager** — Wang, G. et al. (2023). Open-world Minecraft agent with skill library — a *prompt-side* analog of self-modifying weights.
- **SWE-Agent** — Yang, J. et al. (2024).
- **Aider** — community project, code agent.

## Personalization

- **LaMP** — Salemi, A. et al. (2024).
- **PerLTQA** — Du, Y. et al. (2024).
- **MSC (Multi-Session Chat)** — Xu, J. et al., NAACL 2022.
- **Personalization of LLMs: A Survey** — Zhang, Z. et al., TMLR 2025.

## Interpretability of weight changes (helpful, not core)

- **A Toy Model of Interference Weights** — Olah, C., Turner, N. L., Conerly, T. (2025). transformer-circuits.pub. Useful for capacity analysis.
- **Weight-space interpretability survey** — emerging area.

## Datasets / benchmarks

- LaMP — [github.com/LaMP-Benchmark/LaMP](https://github.com/LaMP-Benchmark/LaMP)
- PerLTQA
- SWE-Bench-Verified — [github.com/princeton-nlp/SWE-bench](https://github.com/princeton-nlp/SWE-bench)
- WebArena — [github.com/web-arena-x/webarena](https://github.com/web-arena-x/webarena)
- OSWorld — [github.com/xlang-ai/OSWorld](https://github.com/xlang-ai/OSWorld)
- MLE-Bench — [github.com/openai/mle-bench](https://github.com/openai/mle-bench)

## Open-source code to fork

- [SakanaAI/doc-to-lora](https://github.com/SakanaAI/doc-to-lora) — hypernet scaffold.
- [SakanaAI/text-to-lora](https://github.com/SakanaAI/text-to-lora) — task-conditioned hypernet.
- [SWE-Agent](https://github.com/princeton-nlp/SWE-agent) — agent loop for Phase 3.
- [PEFT](https://github.com/huggingface/peft), [TRL](https://github.com/huggingface/trl), [vLLM](https://github.com/vllm-project/vllm).
