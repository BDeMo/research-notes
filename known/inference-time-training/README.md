# Inference-time training

> Updating a model's **weights** at inference time, rather than only at training time. Covers both backprop-at-test-time and amortized (hypernet) approaches that produce a per-query weight delta without backprop.

## Structure

**Contained by**: (top-level — no parent category yet in this repo).
Conceptually adjacent to *meta-learning* and *online learning*, but treated as a separate frontier here because the practical pressure (LLM scale, hypernets) is different.

**Contains**:
- **Backprop-at-test-time** — real gradient descent per instance / stream (TTT, online RFT, Cartridges, Sparse Memory Finetuning).
- **Amortized weight generation** — single forward pass yields a weight delta (Doc-to-LoRA, Text-to-LoRA, Generative Adapter, HINT, MEND).
- **Test-time weight search** — generate / sample / select among weight variants without learning anew (W-space Best-of-N, our plan 03).
- **Self-modifying generation** — model outputs a weight delta as part of its response (our plan 08; speculative).
- **Inference-time RL** — per-conversation reward-driven weight updates.

## Nearest neighbors

See [the main graph](../README.md#nearness-graph). Closest:
- [`hypernetworks`](../hypernetworks/) — the machinery used by amortized approaches.
- [`context-distillation`](../context-distillation/) — D2L's training objective; "internalize context as weights".
- [`test-time-training`](../test-time-training/) — the backprop-based predecessor.
- [`inference-time-compute`](../inference-time-compute/) — the sibling axis (X-side compute).

## Key concepts

- **$X$-$W$ exchange** — the central framing in this repo. Inference output $y = f(X; W)$ has two tunable inputs at inference: the context $X$ and (now) the weights $W$. Most "test-time compute" work scales $X$; this category scales $W$.
- **Weight delta $\Delta W$** — a low-rank or sparse update to base weights, typically realized as a LoRA. The interesting question is *who generates it and when*.
- **Amortization vs backprop** — Doc-to-LoRA pays a meta-training cost once so each inference is a single forward pass. TTT pays a small backprop cost per inference. Trade-off in latency vs flexibility.
- **Per-request scoping** — whether the $\Delta W$ is discarded after the request (privacy-safe, no cross-contamination) or kept (continual learning, drift risk).
- **Verifier-gated application** — when $\Delta W$ is risky, only apply it if a verifier scores it as beneficial. Central to plan 08.

## Foundational references

(IDs reference [`docs/matrix/knowledge-sources.md`](../../docs/matrix/knowledge-sources.md).)

- [d2l] **Doc-to-LoRA** (Sakana AI, 2026) — amortized context distillation via a hypernet that outputs LoRA.
- [t2l] **Text-to-LoRA** (ICML 2025) — task-description → LoRA.
- [cartridges] **Cartridges** (Eyuboglu et al., 2025) — sleep-time CD via prefix-tuning.
- [ttt-layers] **TTT Layers** (Sun, Li et al., 2024) — modern revival of TTT as architectural primitive.
- [ttt-2020] **Test-Time Training** (Sun et al., ICML 2020) — original TTT formulation.
- [tent] **Tent** (Wang et al., ICLR 2021) — entropy-minimization variant.
- [memo] **MEMO** (Zhang, Levine, Finn, NeurIPS 2022).
- [sparse-mem] **Sparse Memory Finetuning** (Lin et al., 2025) — minimal-disruption TTT.
- [genadapter] **Generative Adapter** (Chen et al., ICLR 2025) — D2L's closest peer using NTP loss.
- [hypertune] **HyperTuning** (Phang et al., ICML 2023) — hypernet adaptation without backprop.
- Wang, Y., Ma, D., Cai, D. (COLM 2024) **Inference-Time Training Helps Long Text Generation**.

## Open questions / live debates

- **Where on the X-W trade-off curve does each task sit?** Some problems are X-bounded (more compute helps); others are W-bounded (need to change the model). No clean characterization yet — see [`known/inference-time-compute`](../inference-time-compute/) for the X-side parallel.
- **Stability** — repeated inference-time updates risk catastrophic forgetting. [alphaedit] and [sparse-mem] are partial answers; none scale to thousands of updates yet.
- **Privacy** — per-request weight updates in a multi-tenant serving environment can leak between users if not carefully scoped. Unresolved.
- **Theoretical equivalence** — when does TTT = implicit Bayesian inference = ICL with appropriate priors? See [icl-bayes] (Xie et al., 2022).
- **Memory-in-weights vs memory-in-prompt** — the cluster around [memgpt] / [voyager] / [a-mem] keeps memory in the prompt-side. When is moving it into weights (plan 08) actually better?

## In this repo

- Brainstorm: [`notes/ideas/inference-time-training.md`](../../notes/ideas/inference-time-training.md) (50 ideas, $X$-$W$ framing).
- Plans:
  - [01 X-saturation curve](../../notes/plans/01-x-saturation-curve/) — data curation by inference-time difficulty.
  - [03 W-space Best-of-N](../../notes/plans/03-w-space-best-of-n/) — search along the W-axis.
  - [08 Model outputs ΔW](../../notes/plans/08-model-outputs-delta-w/) — self-modifying LLMs.
- Session: [`docs/matrix/2026-05-26-inference-time-training.md`](../../docs/matrix/2026-05-26-inference-time-training.md).
