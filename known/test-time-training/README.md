# Test-time training (TTT)

> The original meaning (Sun et al., ICML 2020): at test time, the model takes a few **backprop steps** on a self-supervised auxiliary loss computed from the test input, then uses the adapted weights to make its prediction. A predecessor of all "inference-time training" methods.

## Structure

**Contained by**: *inference-time-training* (this folder is the classical / backprop-based sub-area of inference-time-training).

**Contains**:
- **Auxiliary-loss TTT** — self-supervised aux task at test time (rotation prediction, masking, reconstruction). The Sun-2020 family.
- **Test-time adaptation (TTA)** — entropy minimization on the test prediction (TENT). No labels needed.
- **Test-time augmentation + adaptation** — MEMO: marginal entropy on augmented copies.
- **TTT as an architectural layer** — Sun, Li et al. 2024: replace attention with a TTT layer for long-sequence modeling.
- **TTT for long-text generation** — Wang, Ma, Cai (COLM 2024): per-segment TTT during generation.

## Nearest neighbors

- [`inference-time-training`](../inference-time-training/) — parent / containing category.
- [`model-editing`](../model-editing/) — both apply weight updates at test time, but TTT is unsupervised / aux-loss-driven, while editing targets specific known facts.

## Key concepts

- **Auxiliary self-supervised loss** — a task that can be computed from input alone (no labels), e.g., rotation prediction in vision, reconstruction in NLP. Updates flow back to shared encoder.
- **One-pass-per-sample backprop** — TTT typically updates for K=1 to K=10 gradient steps per test instance.
- **Continual vs reset** — does the model carry updates across test samples (continual TTT) or reset between them? Continual gives more signal but introduces drift; reset is safer but wastes information.
- **Calibration / entropy as a signal** — TENT minimizes prediction entropy, a strong signal under distribution shift but a degenerate one under uniform inputs.
- **TTT-as-a-layer** — recent reframing: a TTT update is itself a differentiable operation, so we can stack TTT layers and train them end-to-end as an architectural primitive (Sun, Li et al. 2024).

## Foundational references

- Sun, Y., Wang, X., Liu, Z., Miller, J., Efros, A., Hardt, M. (ICML 2020). **Test-Time Training with Self-Supervision for Generalization under Distribution Shifts.**
- Wang, D., Shelhamer, E., Liu, S., Olshausen, B., Darrell, T. (ICLR 2021). **Tent: Fully Test-time Adaptation by Entropy Minimization.**
- Zhang, M., Levine, S., Finn, C. (NeurIPS 2022). **MEMO: Test time robustness via adaptation and augmentation.**
- [ttt-layers] Sun, Y., Li, X., Dalal, K., et al. (2024). **Learning to (Learn at Test Time): RNNs with Expressive Hidden States.** [arXiv:2407.04620](https://arxiv.org/abs/2407.04620).
- Wang, Y., Ma, D., Cai, D. (COLM 2024). **With Greater Text Comes Greater Necessity: Inference-Time Training Helps Long Text Generation.**
- Liang, J., Hu, D., Feng, J. (2023). **Test-time adaptation survey.**

## Open questions / live debates

- **How does classical TTT (a few aux-loss steps) interact with modern LLM scale?** Most TTT work is on CV / small NLP; the LLM-scale version is the modern "inference-time training" frontier we work on.
- **Is the aux loss actually needed?** For LLMs, the task itself (next-token, with a verifier) might be a better signal than an aux task.
- **Stability across many test samples** — TTT was originally per-sample. Long agent rollouts need stability over thousands of updates, an unsolved problem.

## In this repo

- See plan 08 ([`notes/plans/08-model-outputs-delta-w/`](../../notes/plans/08-model-outputs-delta-w/)) for the most ambitious TTT-descendant work.
- Plan 03's WBoN-Noise is *not* TTT — no backprop. But the discussion of "$\sigma$ swept" calibration mirrors TTT's "step-size" hyper-parameter sweeps.
