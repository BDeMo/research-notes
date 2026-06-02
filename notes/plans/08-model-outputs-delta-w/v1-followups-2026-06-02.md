# v1 follow-ups → v2 backlog (2026-06-02)

> Generated 2026-06-02 21:55 UTC, alongside the Phase S launch.
> Catalogs every "we wanted this in v1 but couldn't fit it in the
> 20 h paper-deadline window" item. Each entry has a concrete
> intervention sketch, an estimated cost, and which paper section
> it would strengthen.
>
> **Why v1 ships without these.** The submission deadline is the
> hard constraint. The Phase L→S chain that's running now lands
> at ~11:00 PT Wed (~20 h from the snapshot below); after that
> we're aggregating, replotting, and writing. Items in this file
> are *strict additions* — they would not change any v1 conclusion,
> they would only widen the experimental envelope.

---

## 1 · Code-level changes deferred

### 1.1 `--eval-dataset` flag in `train_smoke.py` &nbsp; (cost: ~30 min code + ~6 h re-run)

**What.** Decouple `--dataset` (training distribution) from a new
`--eval-dataset` (held-out evaluation distribution). Today's harness
ties them together, so we cannot run "train on `categorical_niah`,
eval on `quality`" without retraining for every (train, eval) pair.

**Impact on v1.** Would let us claim **TRUE cross-domain transfer**:
the wrapper trained on synthetic NIAH still works on real-text
QuALITY (and vice versa). Currently the cross-domain story in
`main.tex` §4.3 leans on Phase R, which trains directly on QuALITY.

**Paper section it would strengthen.** `main.tex` §4.3 (could add a
"synthetic → real transfer" sub-table). `details.tex` §6 (comparison
table — moves us into a more decisive cell vs Gist Tokens on
"distribution shift robustness").

**v2 intervention sketch.** Add `parser.add_argument("--eval-dataset",
default=None)` and route `eval_dataset = args.eval_dataset or
args.dataset` into the eval loop. The wrapper checkpoint is already
in memory at end-of-training, so we just call the existing
`evaluate(wrapper, base, eval_dataset)` once more with the held-out
dataset before exiting. Adds ~30 s per job for the additional eval
pass. Phase S would be rerun as a "Phase S' " with the flag enabled
on every cell.

### 1.2 Llama-3.1-8B base mount &nbsp; (cost: ~6 h re-train + ~2 h pod ops)

**What.** Pull Llama-3.1-8B-Instruct weights onto `sam-dev` at
`/home/devuser/models/Llama-3.1-8B-Instruct` and verify the
`encoder-infra` / `mem-embedding` adapters load it (tokenizer,
embedding dim 4096 vs Qwen3-8B's 3584).

**Impact on v1.** Would let us claim **true model-agnostic** rather
than the weaker "scale-agnostic within the Qwen family". Currently
the cross-family claim is queued as v2 work.

**Paper section it would strengthen.** `main.tex` §4.3 (would let us
add a Llama-3.1-8B row to Table 1) and §1 abstract (could write
"two different model families" instead of "two scales of one family").

**v2 intervention sketch.** Phase S structure is base-agnostic
(everything routes through `--base`); just need to add 16 jobs
(4 configs × 4 datasets × Llama-3.1-8B base). Total cost = pod ops
(2 h) + 16 jobs × ~30 min on 4 GPUs = 4 h compute + 2 h ops = 6 h.

### 1.3 RULER-NIAH-13 + ∞Bench-LongQA dataset adapters &nbsp; (cost: ~4 h per adapter)

**What.** Add `llm_infra.datasets.ruler_adapter` and
`llm_infra.datasets.infinitebench_longqa_adapter`. Both are standard
long-context benchmarks; their absence from the v1 main table is
purely an onboarding gap.

**Impact on v1.** Phase R already provides QuALITY as the real-text
benchmark. RULER and ∞Bench would broaden the "benchmark column"
of Table 1 from 1 to 3 entries.

**Paper section it would strengthen.** `main.tex` Table 1 (more
benchmark columns). `details.tex` §4 (more held-out benchmark
examples).

**v2 intervention sketch.** Mirror the `quality_adapter` pattern:
`load_dataset(...)` → 4-way MC or QA pairs → chunked iterator
yielding `LongContextItem` tuples. Each adapter ~4 h including
end-to-end smoke test.

### 1.4 Mid-training wrapper checkpointing &nbsp; (cost: ~2 h)

**What.** `train_smoke.py` currently throws away the wrapper at end
of training. Saving a `wrapper.pt` would enable (a) cross-domain
eval-only passes (depends on 1.1), (b) post-hoc seed-pooled probes,
and (c) external reproducibility.

**Impact on v1.** None directly — but unlocks everything in §1.1
and §1.2 above, plus a "the wrapper checkpoint is on Zenodo" link
in the camera-ready.

**v2 intervention sketch.** `torch.save({"wrapper": wrapper.state_dict(),
"args": vars(args), "metrics": final_metrics}, args.out /
"wrapper.pt")` at the bottom of `_main_smoke`. Add `--resume-from`
that loads the checkpoint and skips training.

---

## 2 · Experiment-level extensions deferred

### 2.1 Multi-needle k ∈ {7, 10} stress test

**Why deferred.** Phase S's S6 covers k=5 (one notch above the
Phase N k=3 grid). k=7 and k=10 would map the "how many needles
before the wrapper falls off the cliff" curve more completely, but
the qualitative claim ("needle count is a stress axis the wrapper
handles up to k=5") is already supported by S6.

### 2.2 Capacity wall at K ∈ {256, 512} on Qwen3-14B

**Why deferred.** Phase S's S4 covers K=32 / 128 (depending on d).
Adding K=256 and K=512 cells × 4 d-values would add 8 more jobs
(~3 h). Would tighten "more memory cannot climb the wall" by an
order of magnitude. Defer to v2 if v1 reviewers ask.

### 2.3 Soft-prompt ablations: replace cross-attn readout with prefix-only

**Why deferred.** Phase R's `quality_prefix_a0` job (1 cell) is the
v1 stub for this. A full sweep (5 prefix variants × 4 datasets)
would add 20 jobs (~10 h) and is more of a "why xattn" story than
a "the wrapper works" story.

---

## 3 · Writing-level deferrals

### 3.1 Direct head-to-head with TokMem (Liu et al. 2024)

**Why deferred.** TokMem requires re-implementing their *write
operator* (single-token continuous summary per chunk, no recurrence)
which doesn't fit our `Wrapper` protocol cleanly. The Related-Work
table in `details.tex` §6 lists it under the appropriate row but
without a numerical comparison. v2 would add a "TokMem-style baseline"
job to the matrix.

### 3.2 Ablation: what does the answer head actually learn?

**Why deferred.** The Phase Q answer-head experiments will produce
the numerical evidence. A diagnostic — linear probe of `m_T` against
the gold class — exists (`scripts/run_mem_probes.py`) but the v1
deadline doesn't have room for a full appendix subsection on
"what's in the memory". Logged for v2 as
`research-notes/.../v2-memory-interpretability-plan.md`.

---

## 4 · Tracking

| follow-up | blocked-by | who picks up | when |
|---|---|---|---|
| 1.1 `--eval-dataset` flag | nothing | v1 deadline post | within 1 d of submission |
| 1.2 Llama mount | pod ops | v2 kickoff | first task of v2 |
| 1.3 RULER + ∞Bench adapters | nothing | v2 first sprint | week of v2 kickoff |
| 1.4 wrapper checkpointing | nothing | bundled with 1.1 | within 1 d of submission |
| 2.1 multi-needle k=7,10 | 1.4 | v2 main sweep | after 1.4 |
| 2.2 K=256,512 on 14B | nothing | v2 second sprint | week 2 of v2 |
| 2.3 prefix sweep | 1.4 | v2 main sweep | after 1.4 |
| 3.1 TokMem baseline | new wrapper class | v2 baselines | week 3 of v2 |
| 3.2 head interpretability | Phase Q + 1.4 | v2 paper | after v2 main sweep |
