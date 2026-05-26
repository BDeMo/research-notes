# Plans index

> Catalog of all project plans in this repo with brief meta.
> Each plan folder follows the template defined in [`docs/workflow.md §2`](../../docs/workflow.md).
> Notation defined in [`memory/symbols.md`](../../memory/symbols.md).

```
Legend
S: ? seed · R read · F formalized · D drafted · P piloted · X executing
   W writing · S shipped · M maintenance · B blocked · K killed · Z archived
★: ★ notable · ★★ strong · ★★★ top
φ: 1/N chain · ↪#NN spawned NN · ←#NN continues NN · ≈#NN parallel
M: exp · proto · paper · thesis · prod · side  +  solo · collab · team · open
```

---

## Index

| # | Plan | Parent | ★ | S | φ | M | Cost | Time | Headline question |
|---|---|---|---|---|---|---|---|---|---|
| 01 | [X-Saturation Curve + Dataset Curator](01-x-saturation-curve/) | [I1](../ideas/README.md) | ★★★ | D | 1/? | paper,solo | ~$6.5K | 3 mo | Does training on the $X$-saturated residual beat baselines per FLOP? |
| 03 | [W-Space Best-of-N](03-w-space-best-of-n/) | [D1](../ideas/README.md) | ★★ | D | | paper,solo | ~$4.8K | 3 mo | Does test-time search along the *weights* axis beat (or complement) search along the X-axis? |
| 08 | [Model Outputs ΔW as Part of Generation](08-model-outputs-delta-w/) | [H6](../ideas/README.md) | ★★★ | D | 1/3 | thesis,solo | ~$24K | 12 mo | Can an LLM produce its own weight updates as a first-class output and improve within a session? |

---

## Per-plan meta

### Plan 01 — X-Saturation Curve + Dataset Curator
- **Parent idea**: I1 · `S = D` · `φ = 1/?` · `M = paper,solo`
- **One-liner**: Sweep $X$-budget per example; train on the residual where extra inference can't help.
- **Validation hypothesis**: SFT on the $X$-saturated residual beats matched-size random / loss-mined subsets per FLOP.
- **Primary channels**: GSM8K · MATH · HumanEval+
- **Budget tiers**: Phase-0 pilot ≤ 20 GPU-h ≈ $60 · Full paper ≈ 2200 GPU-h ≈ $6.5K
- **Kill if**: residual set < 5% or > 60% of pool across reasonable thresholds.
- **Sequels in queue**: idea I3 (X-then-W curriculum) marked `←#01` — natural follow-up if plan 01 ships.

### Plan 03 — W-Space Best-of-N
- **Parent idea**: D1 · `S = D` · `φ = standalone` · `M = paper,solo`
- **One-liner**: Sample $N$ weight perturbations, verifier picks best — Best-of-N along the weights axis.
- **Validation hypothesis**: $W$-diversity gives orthogonal coverage to temperature sampling at matched FLOPs.
- **Primary channels**: MATH (L4–5) · HumanEval+ · AIME
- **Budget tiers**: Sanity ≈ 5 GPU-h · Full paper ≈ 1600 GPU-h ≈ $4.8K
- **Kill if**: WBoN-Noise < 0.5 × XBoN even at the best $\sigma$.
- **Sequels in queue**: idea D2 (MCMC over W) marked `←#03`.

### Plan 08 — Model Outputs ΔW as Part of Generation
- **Parent idea**: H6 · `S = D` · `φ = 1/3` · `M = thesis,solo`
- **One-liner**: Model emits both an answer and a weight delta; verifier gates application. Self-modifying LLM.
- **Validation hypothesis**: Within-session benefit rate > 2 × harm rate, beating frozen base by ≥ 5% on continual / personalization tasks.
- **Primary channels**: LaMP · PerLTQA · SWE-Bench-Verified (Phase 3)
- **Budget tiers**: Phase-0 toy ≈ 200 GPU-h ≈ $600 · Full PhD project ≈ 8200 GPU-h ≈ $24K
- **Kill if**: benefit rate < harm rate at end of Phase 1.
- **Sequels in queue**: ideas H3 (reasoning→ΔW), H7 (`<learn>` tool call), E7 (GUI agent online learning), F1 (TTT serving infra), G4 (capacity-preserving updates), G5 (privacy) all marked `←#08` or `≈#08` — they form the natural Phase 2 / sibling work around plan 08.

---

## Plan folder template

```
NN-<slug>/
├── README.md       # problem, idea, success criteria, file index
├── validation.md   # protocol, baselines, metrics, ablations
├── channels.md     # benchmarks, datasets, verifiers, base models
├── budget.md       # GPU-hours, $, headcount, decision points
└── references.md   # closest prior work
```

Numbering matches the parent idea ID in [`../ideas/README.md`](../ideas/README.md). Two-digit padded.

When a plan is killed, add `postmortem.md` to the folder and set `S = K` in the table above. Do not delete.
