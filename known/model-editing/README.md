# Model editing

> Surgical updates to a pre-trained model's weights to **insert, modify, or remove specific knowledge** without retraining. Distinct from generic fine-tuning by its locality, targetedness, and goal of *not disturbing* other capabilities.

## Structure

**Contained by**: *continual-learning* + *knowledge-distillation* (both broader; not yet folders here).

**Contains**:
- **Locate-then-edit** — find the parameters that store a fact, then directly overwrite them (ROME, MEMIT).
- **Hypernet-based editing** — learn a hypernet that maps `(query, new fact)` to a weight update (MEND).
- **Optimization-based editing** — fine-tune a small adapter / patch under a constraint (e.g., KL to base on unrelated queries).
- **Null-space / capacity-preserving editing** — constrain the edit to lie in the null-space of other knowledge (AlphaEdit).
- **Memory-style editing** — insert facts into an external store rather than the weights (memory-augmented models; technically not "editing" but often grouped here).

## Nearest neighbors

- [`hypernetworks`](../hypernetworks/) — MEND family is a hypernet for editing.
- [`lora-peft`](../lora-peft/) — most modern editors store the edit as a LoRA / small adapter.
- [`test-time-training`](../test-time-training/) — both apply weight updates at test time; editing is supervised on a specific fact, TTT is unsupervised.

## Key concepts

- **Locality** — an edit should change behavior only on the target fact (and its paraphrases), not on unrelated queries. Hard to verify rigorously.
- **Generalization** — does editing "Paris is the capital of France" → "Paris is in Italy" also flip "What is the capital of France?" → "Paris (but in Italy)"? Most editors fail at one or both.
- **Catastrophic forgetting under sequential edits** — applying many edits in sequence eventually breaks base performance.
- **Capacity-preserving / null-space constraint** — project the edit onto the null-space of other knowledge representations, so it has zero effect on them in expectation (AlphaEdit).
- **Edit vs un-edit** — *unlearning* is the dual problem. The toolkits look similar but the verification is harder.

## Foundational references

- Meng, K. et al. (NeurIPS 2022). **Locating and Editing Factual Associations in GPT (ROME).**
- Meng, K. et al. (2023). **Mass-Editing Memory in a Transformer (MEMIT).**
- Mitchell, E., Lin, C., Bosselut, A., Finn, C., Manning, C. (ICLR 2022). **Fast Model Editing at Scale (MEND).**
- [alphaedit] Fang, J. et al. (ICLR 2025). **AlphaEdit: Null-space Constrained Editing.**
- Yao, Y. et al. (2023). **Editing Large Language Models: Problems, Methods, and Opportunities.** Survey.
- Wang, S. et al. (2024). **Knowledge editing for LLMs.** Newer survey.
- Hartvigsen, T. et al. (NeurIPS 2023). **GRACE: Continual editing via adaptive codebook.**

## Open questions / live debates

- **Is locality a real constraint or wishful thinking?** Many edits ripple through the model in unpredictable ways; clean locality may be impossible.
- **Editing vs RAG** — for fast-changing knowledge, is editing worth the engineering when RAG is simpler? The answer depends on use case (privacy, offline access, latency).
- **Editing for *style* or *behavior*** — current editing is mostly factual. Editing alignment / style / refusal behaviors is much more delicate.
- **Theoretical bound on edit capacity** — how many edits can a network of size $N$ absorb before its base distribution collapses? Open.

## In this repo

- Plan 08 ([`notes/plans/08-model-outputs-delta-w/`](../../notes/plans/08-model-outputs-delta-w/)) inherits the *stability* problem from editing — solving it likely requires an AlphaEdit-style constraint on the emitted $\Delta W$.
