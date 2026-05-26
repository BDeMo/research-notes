# Hypernetworks

> A neural network $H_\phi$ whose **output is the parameters** of another (target) network. Used to *generate* model weights conditioned on some input (task description, context, demonstration, etc.) rather than learning them via backprop on a single fixed objective.

## Structure

**Contained by**: *meta-learning* (broader; not yet a folder here — hypernets are a specific recipe for meta-learning).

**Contains**:
- **Static hypernets** — output a single set of weights (Ha et al. 2016).
- **Conditional hypernets** — output weights conditioned on a task/context input.
- **Hypernet-generated LoRA** — output is a low-rank adapter (HyperLoRA, Text-to-LoRA, Doc-to-LoRA).
- **Hypernet-generated prefix / soft prompts** — output is prefix tokens (Gisting, MEND).
- **Hypernet-based model editors** — MEND (Mitchell et al.) edits weights conditioned on a (query, new fact) pair.
- **Continual-learning hypernets** — von Oswald et al., one hypernet conditioned on task ID across a stream.

## Nearest neighbors

- [`inference-time-training`](../inference-time-training/) — hypernets are the dominant "amortized" mechanism for inference-time weight updates.
- [`context-distillation`](../context-distillation/) — D2L's hypernet learns to *be* the context-distillation procedure.
- [`lora-peft`](../lora-peft/) — the output format is almost always LoRA in current LLM work.
- [`model-editing`](../model-editing/) — MEND family overlaps both.

## Key concepts

- **Amortization** — instead of training a new model per task at deployment, train one *generator* offline that produces a deployment-ready model per task in a forward pass.
- **Perceiver architecture for variable-length input** — D2L uses Perceiver cross-attention so the hypernet can ingest variable-length context and emit fixed-shape weights.
- **Chunking for compositionality** — split long inputs into chunks, generate per-chunk weight deltas, concat along the rank dimension. Lets hypernets scale beyond their training-length window.
- **Output parameterization choice** — LoRA matrices vs prefix tokens vs full-weight delta. Choice constrains what the hypernet can express and how it composes.
- **Meta-training data scarcity** — hypernets need a *distribution* of (input → weights) pairs. Generating that distribution is often the bottleneck (context-distillation provides one source of supervision).

## Foundational references

- [d2l] **Doc-to-LoRA** (Sakana AI, 2026) — context → LoRA.
- [t2l] **Text-to-LoRA** (ICML 2025) — task-description → LoRA.
- Ha, D., Dai, A., Le, Q. (2016). **HyperNetworks.** [arXiv:1609.09106](https://arxiv.org/abs/1609.09106). The original.
- [genadapter] **Generative Adapter** (Chen et al., ICLR 2025) — context → LoRA with NTP loss; closest peer to D2L.
- Phang, J., Mao, Y., He, P., Chen, W. (ICML 2023). **HyperTuning** — task-conditioned hypernet without backprop at test time.
- Ivison, H. et al. (ACL 2023). **HINT: Hypernetwork instruction tuning.**
- Lv, C. et al. (ACL Findings 2024). **HyperLoRA.**
- Mitchell, E. et al. (ICLR 2022). **MEND** — hypernet for *model editing*.
- Li, Y. et al. (ICLR 2024). **MEND (demo distillation)** — separate paper, same family, compresses few-shot examples.
- Mu, J., Li, X., Goodman, N. (NeurIPS 2024). **Gisting** — learn to compress prompts into gist tokens via a hypernet-like mechanism.
- von Oswald, J. et al. (ICLR 2020). **Continual learning with hypernetworks.**
- Zhao, D., Kobayashi, S., Sacramento, J., von Oswald, J. (2020). **Meta-learning via hypernetworks.**

## Open questions / live debates

- **What architecture beats Perceiver as a hypernet trunk?** D2L found 8 cross-attn blocks ≈ 309M params is enough for Gemma-2-2b; scale to larger targets is open.
- **One hypernet, many targets** — currently a hypernet must be retrained for each base LLM. Foundation-hypernet that conditions on the target as well would change deployment.
- **Output-space prior** — is there a structural prior on LoRA matrices that hypernets should respect (orthogonality, sparsity, null-space)?
- **Backprop hybrid** — D2L outputs are 82% of the oracle CD upper bound. The remaining 18% likely needs a few backprop steps; how to combine cheaply?
