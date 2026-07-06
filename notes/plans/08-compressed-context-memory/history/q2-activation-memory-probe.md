# Plan 08 Q2 — Which Activations Carry Memory?

> **Status**: parked research question (independent of v1 / v2
> shipping cadence)
> **Created**: 2026-06-01
> **Owner**: Mingjia
> **Parent**: Plan 08 v2 (`v2-plan.md` §4 Q2)
> **One-liner**: Probe the base LM's activations to identify which
> dimensions / channels / neurons are heavily used by ordinary language
> versus which are sparse / unused. Use the unused ones as the
> *write target* for the latent memory wrapper. This is its own
> methodology paper.

---

## 1 · Why this question exists

The v1 / v2 wrapper currently writes into the **full hidden dimension
$d = 4096$ of Qwen3-8B**. Every write thus has the potential to step
on signal the base LM is already using for plain language generation.
Q6 in `v2-plan.md` will measure that interference directly (MMLU /
HumanEval / GSM8K with empty memory should be unchanged).

But there is a much stronger version of the question:

> *Not all dimensions of `h ∈ ℝ^d` are equally used. Some channels
> fire on almost every token (high-information backbone). Others
> are sparse — activated only on rare contexts, or apparently
> dormant. If we limit the wrapper's writes to the **sparse /
> dormant** channels, we get (a) zero interference with ordinary
> generation by construction, (b) a principled, base-aware capacity
> budget, and (c) a paper-worthy methodology.*

This generalizes the AlphaEdit / null-space-constrained editing line
(weight-update space) to the **activation space** at inference time.

---

## 2 · Research questions

* **Q2.1** Across a representative corpus, what is the activation
  density of each of the $d = 4096$ channels at each of the $N=36$
  layers in Qwen3-8B? Concretely: per (layer, channel), distribution
  of $|h_{l,i}|$ across tokens.
* **Q2.2** Is there a stable "backbone" (high-density channels) vs
  "free" (sparse / dormant) subspace? How much of the dimension is
  free (e.g. 5 %, 30 %, 50 %)?
* **Q2.3** Does the backbone / free partition transfer across domains
  (code, math, dialog, scientific)? Or is it domain-specific?
* **Q2.4** If we project the memory wrapper's writes into the free
  subspace only, does v1 / v2 task performance drop? By how much?
  (Capacity-vs-interference Pareto curve.)
* **Q2.5** Does writing into the free subspace eliminate the base
  perplexity drift on empty-memory MMLU (Q6 above)?

---

## 3 · Connections to existing literature

* **Knowledge Neurons** (Dai et al. 2022) — identify which MLP
  neurons encode specific factual associations. Same probing
  primitive, different downstream use.
* **Sparse autoencoder analysis** (Anthropic, OpenAI 2024+) — the
  current SoTA approach to decomposing activation space into
  interpretable sparse features. Provides the toolchain.
* **AlphaEdit / null-space-constrained editing** (2024) — applies
  the same intuition in **weight** space; we'd apply it in
  **activation** space.
* **DARE / Model Merging via Drop-and-Rescale** (Yu et al. 2024) —
  shows you can remove a large fraction of fine-tune deltas without
  hurting performance; suggests the "free subspace" is real and
  large.
* **Pruning literature** (Wanda, SparseGPT, LLM-Pruner) — provides
  importance scores per neuron / channel, reusable for the partition.
* **LoRA / Adapter rank analysis** — shows task-specific updates
  concentrate in low-rank subspaces; consistent with the hypothesis
  that a frozen-base wrapper has a small effective write subspace
  available.

(Add a thorough lit pass when this becomes the active line.)

---

## 4 · Methodology sketch

### Phase A — Profile

1. Sample 100 K tokens of diverse text (Pile mix or C4 + code +
   math + dialog).
2. Run forward passes through frozen Qwen3-8B. At each layer $l$ and
   channel $i$, log $|h_{l,t,i}|$ for every token $t$.
3. Per (l, i) compute: mean magnitude, fraction-of-tokens-active
   (above a threshold), 99th percentile, kurtosis.
4. Cluster channels into **backbone** (top-K by mean magnitude) vs
   **free** (bottom-K by mean × activation fraction).

Cost: ~2 hours on 1× H100.

### Phase B — Validate "free" really is free

1. For each (l, i) in the free set, zero out that channel during
   forward pass on MMLU / HumanEval / GSM8K.
2. Measure delta accuracy.
3. Keep only channels whose zero-out is < $\epsilon$ accuracy drop.
   That's the truly free subspace.

Cost: ~6 hours of evals.

### Phase C — Constrained wrapper

1. Modify the v1 / v2 wrapper to project writes through
   $P_\text{free} \in \mathbb{R}^{d \times d}$ that zeroes out backbone
   channels.
2. Retrain wrapper from scratch with this constraint.
3. Compare against unconstrained baseline on:
   * Capacity wall (bit-capacity at K = 32, 64, 128)
   * Interference (MMLU with empty memory)
   * Real-text benchmarks (LongMemEval / LOCOMO)

### Phase D — Layer choice from the same data

The probing data also answers `v2-plan.md` §2.1: pick the layer with
the **best memory-bearing signal**. Heuristic: layer where the
fraction-of-tokens-active distribution is closest to uniform across
channels (= layer with the most disentangled representation).

---

## 5 · What a published paper would claim

Working title: **"Free Channels: A Frozen-Base LLM Has a Spare
Subspace, and That's Where Soft Memory Should Live."**

Headline claims (target):

1. ~25 % of Qwen3-8B's hidden dimension is **measurably unused** by
   ordinary generation (across domains).
2. Restricting a memory wrapper's writes to that subspace **eliminates
   the empty-memory perplexity drift** observed in v1 / v2.
3. The bit-capacity wall **shifts later** (more bits before
   collapse) when writes are constrained, because the wrapper no
   longer competes with the base for the same channels.
4. The free-subspace partition **transfers across base model sizes**
   (1B → 7B → 70B) up to a known fraction, giving a recipe for
   memory wrapper design across the Qwen / Llama families.

If even (1) and (2) hold, that's already a venue-worthy contribution
(ACL / EMNLP).

---

## 6 · Why this is a separate paper from v1 / v2

* v1 = encode-time memory + bit-capacity wall (the diagnostic).
* v2 = generation-time memory + cross-session persistence (the
  extension of the diagnostic).
* Q2 = a **methodology** that improves the *write* side of any
  wrapper architecture, independently of when/where the memory is
  built.

Q2 is the answer to "okay you found the read-side bottleneck — what
do we do about the *write* side?" It's the natural follow-up question
after v1 and v2 land.

---

## 7 · Compute estimate

* Phase A (profile): 1× H100 × 2 h
* Phase B (validate): 4× H100 × 6 h = 24 GPU-h
* Phase C (constrained training): same scale as v1 main sweep ≈
  150 GPU-h
* Phase D (layer choice): free, falls out of A

**Total: ~180 GPU-h** for the methodology paper. Cheap once v1 / v2
are off the runway.

---

## 8 · Open subquestions

* Does the free subspace differ between attention residual stream
  and MLP residual stream?
* Does it correlate with which heads are "retrieval heads"
  (Wu et al. 2024)?
* Is there an unsupervised way to extract the free subspace at
  inference time per-prompt, rather than as a global partition?
* Can we use SAE features directly as the write basis (write into
  feature-space, not channel-space)?
* If we tie the free-subspace projection across layers, does that
  hurt or help capacity?

---

## 9 · Status

Parked. Will revisit after v1 paper is in submission and v2
prototype is running. Estimated start: 8–10 weeks from today.
