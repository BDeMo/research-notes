# Plan 08 v0 — Budget

> Budget for the practical v0: an 8B frozen base model plus a trainable memory
> wrapper. This is separate from the full self-modifying-LLM budget in
> `budget.md`.

## Assumptions

- Base model: open-source 7B/8B class model, kept frozen.
- Trainable component: memory wrapper only.
- Hardware cost model: H100 at `$3 / GPU-hour`, matching the main Plan 08 budget.
- Main deliverables: research-paper prototype, RCA-Code public model, technical
  report, and demo.
- Private RCA / Nokia DTS data is not part of public release cost.

## TL;DR

| Scope | Wall-clock | GPU-hours | H100 cloud cost | What it buys |
|---|---:|---:|---:|---|
| Lean MVP | 4-7 weeks | 1,420-2,950 | $4.3k-$8.9k | One 8B base, one wrapper, smoke + RCA-Code demo. |
| Paper-quality v0 | 7-13 weeks | 2,820-5,750 | $8.5k-$17.3k | General long-context benchmarks, RCA-Code, RCA domain eval, ablations. |
| Strong paper / model release | 10-16 weeks | 5,000-9,000 | $15k-$27k | Multiple seeds, larger sweeps, polished model card and report. |

On a dedicated 8xH100 node, the paper-quality v0 GPU budget is roughly 15-30
days of raw GPU occupancy. Calendar time is longer because data curation,
debugging, evaluation, and report writing do not parallelize perfectly.

## Phase Breakdown

| Phase | Wall-clock | GPU-hours | Cost | Decision gate |
|---|---:|---:|---:|---|
| 0. Scope + smoke | 3-5 days | 120-250 | $360-$750 | Wrapper trains on synthetic / NIAH smoke. |
| 1. Base long-context track | 2-3 weeks | 600-1,200 | $1.8k-$3.6k | Wrapper beats summary on LoCoMo / LongBench / RULER at matched budget. |
| 2. RCA-Code data + wrapper | 2-4 weeks | 900-1,800 | $2.7k-$5.4k | Public debug traces produce useful direct RCA predictions. |
| 3. RCA domain transfer + demo | 1-2 weeks | 400-900 | $1.2k-$2.7k | RCA compression works on Nezha / OpenRCA / RCAEval / lincyaw. |
| 4. Ablations + release polish | 1-2 weeks | 800-1,600 | $2.4k-$4.8k | Memory length, chunk order, summary/retrieval baselines, model card. |
| Total paper-quality v0 | 7-13 weeks | 2,820-5,750 | $8.5k-$17.3k | Decide whether to write the paper and release RCA-Code. |

## Phase Details

### Phase 0 — Scope + smoke

Goal: prove the implementation path works before spending on real benchmarks.

Includes:

- frozen 8B base setup;
- memory-wrapper forward pass;
- NIAH or synthetic planted-evidence smoke set;
- one short teacher-student training run.

Budget: `120-250 GPU-hours`.

Pass if learned memory beats a same-length naive summary on the smoke task.

### Phase 1 — Base long-context track

Goal: validate the wrapper before adding RCA-specific complexity.

Datasets:

- LoCoMo;
- LongBench;
- RULER;
- Needle-in-a-Haystack variants;
- optional Qasper / NarrativeQA / 2WikiMultihopQA.

Includes:

- full-context teacher generation;
- full-context / summary / retrieval / wrapper evaluation;
- wrapper training on memory-state next prediction;
- memory length sweep, e.g. 16 / 32 / 64 / 128 memory tokens.

Budget: `600-1,200 GPU-hours`.

Pass if wrapper memory is within 5-10 points of full context and beats summary
at matched token budget on at least two benchmark families.

### Phase 2 — RCA-Code data + wrapper

Goal: build the public coding/debug model path.

Datasets:

- SWE-bench / SWE-bench Verified;
- BugsInPy;
- Defects4J;
- QuixBugs;
- optionally curated CodeNet / APPS wrong-submission traces.

Includes:

- converting public traces into RCA-Code long-context items;
- direct prediction evaluation;
- agentic trace collection for a small subset;
- wrapper continued training on trace / test-output / code-context chunks.

Budget: `900-1,800 GPU-hours`.

Pass if the wrapper improves direct root-cause / fix recommendation quality over
summary and retrieval baselines, without updating the frozen base model.

### Phase 3 — RCA domain transfer + demo

Goal: connect the research method to RCA domain data.

Datasets:

- Nezha;
- OpenRCA-500;
- RCAEval;
- lincyaw/rca;
- private Liang/Nokia DTS only as internal qualitative probe.

Includes:

- long-context RCA benchmark generation;
- prompt strategy comparison;
- demo incident;
- initial technical report figures/tables.

Budget: `400-900 GPU-hours`.

Pass if learned memory preserves evidence grounding and RCA output format while
reducing prompt/KV cost by at least 4x.

### Phase 4 — Ablations + release polish

Goal: turn the prototype into a defensible paper/model release.

Includes:

- memory size ablation;
- chunk order ablation;
- early/middle/late evidence retention;
- direct vs agentic RCA-Code comparison;
- model card;
- technical report;
- public-data-only release package.

Budget: `800-1,600 GPU-hours`.

## Budget Formula

At `$3 / H100 GPU-hour`:

```text
cost = GPU_hours * 3

Lean MVP:          1,420-2,950 GPUh  => $4,260-$8,850
Paper-quality v0: 2,820-5,750 GPUh  => $8,460-$17,250
Strong release:   5,000-9,000 GPUh  => $15,000-$27,000
```

## Why This Is Cheaper Than Full Plan 08

The full Plan 08 budget assumes model-emitted weight deltas, verifier-gated
application, RL, rollback, and agent integration. V0 avoids most of that:

- no base-model finetuning;
- no LoRA/weight output in the first version;
- no RL in the first version;
- wrapper training is cheaper than full model training;
- RCA-Code release can be built from public debug data and public RCA datasets.

## Off-Ramps

- Stop after Phase 0 if learned memory cannot beat summary on smoke tasks.
- Stop after Phase 1 if LoCoMo / LongBench / RULER show no advantage over
  retrieval or summary at matched token budget.
- Stop after Phase 2 if RCA-Code direct prediction does not improve over summary
  / retrieval baselines.
- Do not escalate to LoRA/weight memory until explicit memory tokens show a
  clear advantage.

## Headcount

Minimum:

- 1 researcher/engineer for wrapper architecture, training, and reporting.
- 1 engineer for dataset conversion, evaluation plumbing, and demos.

Helpful but optional:

- systems support for faster batched inference and wrapper serving;
- RCA/domain reviewer for qualitative failure analysis.
