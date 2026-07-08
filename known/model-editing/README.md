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
- [`catastrophic-forgetting`](../catastrophic-forgetting/) — sequential edits forget; the null-space/capacity-preserving idea is shared.
- [`transformer-internals`](../transformer-internals/) — locate-then-edit localizes facts to weights, akin to localizing capability to intrinsic sites.

## Key concepts

- **Locality** — an edit should change behavior only on the target fact (and its paraphrases), not on unrelated queries. Hard to verify rigorously.
- **Generalization** — does editing "Paris is the capital of France" → "Paris is in Italy" also flip "What is the capital of France?" → "Paris (but in Italy)"? Most editors fail at one or both.
- **Catastrophic forgetting under sequential edits** — applying many edits in sequence eventually breaks base performance.
- **Capacity-preserving / null-space constraint** — project the edit onto the null-space of other knowledge representations, so it has zero effect on them in expectation (AlphaEdit).
- **Edit vs un-edit** — *unlearning* is the dual problem. The toolkits look similar but the verification is harder.

## Foundational references

(IDs reference [`docs/matrix/knowledge-sources.md`](../../docs/matrix/knowledge-sources.md).)

- [rome] **ROME** (Meng et al., NeurIPS 2022) — locate-then-edit.
- [memit] **MEMIT** (Meng et al., 2023) — mass-editing many facts at once.
- [mend-edit] **MEND** (Mitchell et al., ICLR 2022) — hypernet that learns to edit.
- [alphaedit] **AlphaEdit** (Fang et al., ICLR 2025) — null-space constrained editing.
- [grace] **GRACE** (Hartvigsen et al., NeurIPS 2023) — continual editing via adaptive codebook.
- Yao, Y. et al. (2023). **Editing Large Language Models: Problems, Methods, and Opportunities.** Survey.
- Wang, S. et al. (2024). **Knowledge editing for LLMs.** Newer survey.

## Open questions / live debates

- **Is locality a real constraint or wishful thinking?** Many edits ripple through the model in unpredictable ways; clean locality may be impossible.
- **Editing vs RAG** — for fast-changing knowledge, is editing worth the engineering when RAG is simpler? The answer depends on use case (privacy, offline access, latency).
- **Editing for *style* or *behavior*** — current editing is mostly factual. Editing alignment / style / refusal behaviors is much more delicate.
- **Theoretical bound on edit capacity** — how many edits can a network of size $N$ absorb before its base distribution collapses? Open.

## In this repo

- **Plan 08** ([`notes/plans/08-compressed-context-memory/`](../../notes/plans/08-compressed-context-memory/)) inherits the *stability* problem from editing — solving it likely requires an [alphaedit]-style constraint on the emitted $\Delta W$. [memit]'s many-edits regime is the deployment scenario.
- **Plan 03** ([`notes/plans/03-w-space-best-of-n/`](../../notes/plans/03-w-space-best-of-n/)) borrows the *locality* discipline: each per-request LoRA in WBoN must not corrupt base behavior across requests.
