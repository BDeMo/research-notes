# v1 method-design improvements — companion to critical review

Created 2026-06-02 22:32 UTC, in parallel with
`critical-review-2026-06-02-v1.md` in `mem-embedding/summary/`.

This note collects **method-design** improvements the critical review
identified. Each is a concrete change to `mem_embedding/wrapper.py` (or
a new module) plus a one-paragraph hypothesis and a one-sentence
falsifier. None are scheduled for v1; all are candidates for v2 or for
the v1 camera-ready if Phase T leaves time.

The ordering is by **value-per-effort** as judged from current results.

---

## I-1. Fixed-K vs growing-K plot at multiple `n_chunks`

**Status**: very high value, low effort.

**Change**: add a `--scan-K` mode to `train_smoke.py` that re-runs the
eval loop for `K ∈ {8, 16, 32, 64, 128}` on the trained wrapper, and a
Gist counterpart that sweeps `gist_k_per_chunk ∈ {1,2,4,6,8}` (memory
growing as `k·N` chunks). Plot `em(K)` and `em(k·N)` on the same axis.

**Hypothesis**: OURS's `em(K)` saturates somewhere between `K=32` and
`K=64`; Gist's `em(k·N)` continues to grow but at a strictly worse
slope per unit memory. The crossover defines the regime where
recurrent fixed-K wins on a budget.

**Falsifier**: if Gist's `em(k·N)` is monotonically above OURS's
`em(K)` at every matched-memory point, the recurrent claim is dead and
v2 should pivot to a different architectural axis.

**Implementation**: 100 lines of Python, 1 GPU-hr per cell.

---

## I-2. Multi-needle stress as the v1 headline task

**Status**: very high value, medium effort.

**Change**: replace `categorical_niah` with `multi_needle_niah_k=3` as
the headline cell in `main.tex` Table 1, once the in-flight Phase L
filler `data500_multi_k3` lands tonight.

**Hypothesis**: Gist is per-chunk independent. On a 3-needle task where
the answer requires combining facts from chunks $i, j, k$ with $i < j < k$,
Gist must store the three needles in three separate per-chunk gists; the
final read must softmax-attend over all of them simultaneously. The
recurrent wrapper carries the cross-chunk binding in `m_t`.

**Falsifier**: if `multi_needle_k=3` shows no \OURS{}-vs-\GIST{} gap at
matched K, the per-chunk-independence argument is wrong and Gist is
operationally equivalent to OURS for multi-needle.

**Implementation**: the data is already generated; just need to
re-bind the main table to a different dataset. Maybe 1 hour of LaTeX,
1 GPU-hr for the matched Gist cell.

---

## I-3. Sparse-write gate (chunks without needles should not update memory)

**Status**: medium value, medium effort.

**Change**: add a learned scalar `gate(chunk) ∈ [0,1]` after the update
block; multiply `Δm_t` by `gate(c_t)`. Optionally `top-k` over a batch
to enforce sparsity at training. This is the wrapper-equivalent of
sparse attention: most chunks contribute nothing, a few contribute a
lot.

**Hypothesis**: when only ~1 in 6 chunks contains a needle, the
current dense wrapper wastes 5/6 of its update capacity. A learned
gate should improve `em` by ~5-10pp at fixed `K`, and should reduce
`drift_mt` (the cumulative `||Δm_t||`) by ~6×.

**Falsifier**: if the gate's mean activation converges to ~1.0 (uniform
write) or its addition does not change `em`, the dense-write hypothesis
was correct and we should not add complexity.

**Implementation**: ~50 lines in `wrapper.py`, no architecture changes
elsewhere. Free with the `lambda_div` regulariser; could simply replace it.

---

## I-4. Dual-rail memory (hot prefix + cold xattn)

**Status**: high value, high effort.

**Change**: split memory into two streams:
- **Hot stream**: small (`K_hot = 8`), prepended as soft prefix every
  forward pass. Cheap, always visible to base.
- **Cold stream**: large (`K_cold = 128`), accessed via cross-attention
  only on a learned `should-i-look` gate. Expensive but high-capacity.

The wrapper writes both; the base reads both. Inspired by the
small-fast-cache / large-slow-cache pattern in hardware and in some
recent attention papers (the closest in spirit is the
`memorising-transformer` (Wu et al. 2022)).

**Hypothesis**: at long context (`n_chunks > 24`), the single-K
wrapper saturates and `em` drops. Dual-rail saturates later because
the cold stream absorbs the long-tail facts that the hot stream cannot
hold.

**Falsifier**: at `n_chunks = 6` (paper's headline length), dual-rail
matches single-rail. The story is a v2 long-context claim, not a v1
result.

**Implementation**: ~200 lines, requires a new `combine_mode` and a
non-trivial gating sub-network. Maybe 2-3 days of engineering plus a
sweep budget.

---

## I-5. Memory as test-time online learning

**Status**: speculative but interesting.

**Change**: at inference, after each user query+answer pair, do one
SGD step on the wrapper's memory state (not the wrapper parameters)
using a reconstruction loss between the wrapper's predicted answer and
the actual answer that the base produced under full context. This is
in-context learning, but in the memory state instead of the prompt.

**Hypothesis**: a wrapper exposed to a session of $T$ queries will,
after a few queries, develop a memory state that is implicitly tuned
to the session topic; subsequent queries will be answered faster and
more accurately than the cold-start memory.

**Falsifier**: if the per-query SGD step does not improve subsequent
`em` on held-out queries from the same session, the memory state is
not the right object for online learning and we should look at LoRA
adapters or similar.

**Implementation**: ~150 lines. Requires a benchmark with multi-turn
sessions (QuALITY has them; MMLU does not). This is a v2 storyline.

---

## I-6. Adversarial chunk ordering (read-side stress)

**Status**: medium value, low effort.

**Change**: at eval, randomly permute the chunk order before encoding.
Hypothesis: the wrapper's `m_t` is order-sensitive (it must be —
recurrence!). The Gist baseline is order-invariant (concatenation
commutes). Which is more brittle in practice?

**Implementation**: a `--shuffle-chunks` flag in eval, 30 lines. One
GPU-hr per cell.

---

## I-7. Joint training across multiple datasets (cross-task transfer)

**Status**: low value for v1, high value for v2.

**Change**: round-robin training on `{categorical, numerical_d1,
coding, multi_k3}` to test whether one wrapper learns to handle all
four task families.

**Implementation**: needs `--eval-dataset` and `--train-datasets`
flags (v1-followups item 1). 1 day of engineering, 2 GPU-hr per cell.

---

## I-8. Soft prefix at the base's `inputs_embeds` vs. at the first layer

**Status**: low value, low effort, mostly a diagnostic.

**Change**: instead of prepending memory at `inputs_embeds`, inject it
as a delta to the K/V of the base's first attention layer (the
`prefix-tuning` style). Test whether the prepend choice matters at
all.

**Implementation**: 100 lines, requires a base-model hook. Maybe 1
GPU-hr per cell.

---

## Triage for v1 vs v2

| improvement | effort | value | v1 if T leaves time | v2 candidate |
|---|---|---|---|---|
| I-1 fixed-K plot | low | high | ✓ ('Phase U') | – |
| I-2 multi-needle as headline | medium | high | ✓ (only LaTeX swap) | – |
| I-3 sparse-write gate | medium | medium | – | ✓ |
| I-4 dual-rail | high | high | – | ✓ (lead method change) |
| I-5 online learning | high | speculative | – | ✓ |
| I-6 chunk-order stress | low | medium | – | ✓ (small section) |
| I-7 joint training | medium | low for v1 | – | ✓ |
| I-8 prefix vs. K/V | low | low (diagnostic) | – | ✓ |

The v1 line items (I-1 and I-2) are both about *insight delivery* —
they don't change the method, they make the method look stronger by
choosing the right axis to display it on. Both are achievable in the
remaining ~16 hours of GPU time before noon tomorrow if Phase T
finishes early enough.

---

## Action items now

- [ ] **Phase U (optional v1)**: define a 5-cell scan
  `K ∈ {8,16,32,64,128}` on `categorical_niah` for OURS and the
  corresponding 5-cell scan for Gist's `k_per_chunk`. Total ~5
  GPU-hr. Queue **only if** Phase T lands before 04:00 PT 6/3.
- [ ] **`main.tex` Table 1 re-bind to multi-needle** (I-2) once the
  in-flight filler `data500_multi_k3` lands (ETA ~22:40 UTC). Even
  without the matched Gist cell, the table's headline shifts from
  *one* 4-way categorical task to *one* multi-needle task that better
  motivates the wrapper's recurrence.
- [ ] **v2 lead method (I-4 dual-rail)** as the v2 first-draft topic.
  Write a one-page sketch in a new `v2-method-sketch.md` for after
  v1 ships.

---

## Cross-references

- Critical review: `mem-embedding/summary/critical-review-2026-06-02-v1.md`
- v1 deferrals: `notes/plans/08-compressed-context-memory/history/v1-followups-2026-06-02.md`
- Phase T design: `mem-embedding/k8s/sweep/build_sweep.py` (PHASE_T block)
- Phase T scorecard (lands tomorrow): `runs/sweep-phaseT-*/report.md`
