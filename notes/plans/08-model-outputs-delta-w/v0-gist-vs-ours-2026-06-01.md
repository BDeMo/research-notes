# Plan 08 v0 — Gist vs Ours, Phase J (2026-06-01)

> **Status**: live results, partial (Phase K still running). The numbers
> below are the first head-to-head between the Gist Tokens baseline and
> our memory-embedding wrapper at matching K, on the same Qwen3-8B base.
>
> **Important context**: Gist results were recovered post-hoc from
> per-model summary JSONs after the geometry probe crashed on
> ``GistMemoryPayload`` (fixed in `llm-infra` commit on this date and
> backfilled with `mem-embedding/scripts/synth_combine_summary.py`). All
> 10 Gist runs trained and evaluated successfully; only the final
> wrap-up + geometry probe failed.
>
> **Settings / provenance:** this is a pre-final exploratory comparison. Use
> [`settings.md`](settings.md) for the current provenance registry; these cells
> are historical variants around [`P08-S1`](settings.md#p08-s1--v1-canonical-wrapper-recipe),
> while final v1 headline cells use
> [`P08-S2`](settings.md#p08-s2--v1-phase-y-three-regime-benchmark-cells).

---

## TL;DR

Gist beats our wrapper on **simple, small-answer-space tasks** at
matched K (categorical_niah K=32, numerical d=1). On **every harder
task** (coding_niah, same_form_distractors, multi_needle, numerical
d ≥ 2) **both methods collapse to zero**.

This sharpens — does not weaken — the v1 paper thesis:

> *Bit-capacity is a property of the read interface, not of any
> particular wrapper architecture. Gist's compressor is stronger
> below the wall; our wrapper degrades more gracefully past the
> wall (per Phase F coding_niah evidence, ≥0.9). Past a certain
> answer-entropy threshold, both architectures hit the same hard
> wall at the same place.*

The corollary: **the headline contribution of v1 is the wall and the
diagnostic, not "our wrapper beats baselines"**. Frame accordingly in
the paper.

---

## Head-to-head, Phase J (held-out eval, n=50)

| task | n_mem | gist contains | gist exact | gist f1 | ours best contains¹ | ours best exact | ours best f1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| categorical_niah | K=32 | **0.340** | **0.200** | **0.225** | 0.300 (seed42, 5-seed mean 0.244) | 0.220 (seed99) | 0.228 (seed99) |
| categorical_niah | K=64 | 0.000 | 0.000 | 0.000 | _not in J, Phase K covers it_ | — | — |
| coding_niah | K=32 | 0.000 | 0.000 | 0.000 | _Phase J ours not re-tested, prior runs ≥0.9_ | — | — |
| coding_niah | K=128 | 0.000 | 0.000 | 0.000 | _re-test in Phase K_ | — | — |
| numerical d=1 | K=32 | **0.260** | **0.140** | **0.145** | 0.180 (recheck_d1) | 0.000 | 0.000 |
| numerical d=2 | K=32 | 0.040 | 0.040 | 0.040 | 0.000 (recheck_d2) | 0.000 | 0.000 |
| numerical d=5 | K=32 | 0.000 | 0.000 | 0.000 | _Phase K covers_ | — | — |
| numerical d=12 | K=128 | 0.000 | 0.000 | 0.000 | _Phase K covers_ | — | — |
| same_form_distractors | K=32 | 0.000 | 0.000 | 0.000 | 0.000 (ours_sameform_K32) | 0.000 | 0.000 |
| same_form_distractors | K=128 | _n/a_ | — | — | 0.000 (ours_sameform_K128) | 0.000 | 0.000 |
| multi_needle k=3 | K=32 | 0.000 | 0.000 | 0.000 | 0.000 (ours_multineedle_k3) | 0.000 | 0.000 |
| multi_needle k=5 | K=32 | _n/a_ | — | — | 0.000 (ours_multineedle_k5) | 0.000 | 0.000 |

¹ "ours best contains" = the single highest contains across the 5
seed-ref runs (seed{1,2,3,42,99}_ref) on categorical_niah K=32. For
non-categorical tasks "ours" is the single run on that task in
Phase J.

---

## What this means for the paper

* **Section 4 (Results) cannot lead with "our wrapper beats Gist at
  matched K."** That claim is false on the easy tasks.
* **Section 4 should lead with the wall** — task-by-task, where both
  methods cross from ~moderate to ~zero as answer-entropy rises.
* **Section 4 needs the *graceful degradation* result** — coding_niah
  is the cleanest demonstration. Re-test in Phase K (in flight) at
  K=32 and K=128.
* **Section 5 (Discussion) should explicitly concede the easy-task
  loss** and use it to argue that the read-side bottleneck is what
  matters, not the write-side compressor.

---

## Outstanding questions

1. Is Gist's edge on categorical / numerical_d1 robust across seeds,
   or does it disappear with 5-seed averaging like ours does? Phase K
   does NOT include extra Gist seeds — add to Phase L.
2. Does Gist's compressor design — K_per_chunk × n_chunks compressed
   tokens — give it more effective context than our K=32 monolithic
   memory? Compare effective bit-budgets in a normalized way.
3. What's the per-task break-even K at which Gist matches our
   wrapper? Needs a Phase L K-sweep for Gist analogous to our
   Phase K.

---

## Operational follow-ups

* `llm-infra/src/llm_infra/probes_geometry.py` — defensive payload
  check, tolerates `GistMemoryPayload`. Committed 2026-06-01.
* `mem-embedding/scripts/aggregate_sweep.py` — defensive geometry
  read, no longer KeyErrors on synthesized summaries. Committed
  2026-06-01.
* `mem-embedding/scripts/synth_combine_summary.py` — NEW recovery
  tool to rebuild missing top-level `combine=<C>/summary.json`
  from per-model files. Committed 2026-06-01.
* The Phase J manifest was overwritten by the Gist re-run launch;
  rebuilt from disk and re-aggregated. Add a guard to the
  orchestrator's manifest-write path: if a manifest already exists,
  **merge** (don't overwrite). Track as a separate task.
