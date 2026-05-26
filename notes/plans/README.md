# Plans

> Detailed project plans for the highest-priority ideas in this repo.
> Each plan folder follows the template defined in [`docs/workflow.md §2`](../../docs/workflow.md).

| # | Plan | Parent idea | Status | Cost estimate | Headline question |
|---|---|---|---|---|---|
| 01 | [X-Saturation Curve + Dataset Curator](01-x-saturation-curve/) | I1 (X-W exchange) | drafting | ~$6.5K · 3 mo | Does training on the $X$-saturated residual beat baselines per FLOP? |
| 03 | [W-Space Best-of-N](03-w-space-best-of-n/) | D1 | drafting | ~$4.8K · 3 mo | Does test-time search along the *weights* axis beat (or complement) search along the X-axis? |
| 08 | [Model Outputs ΔW as Part of Generation](08-model-outputs-delta-w/) | H6 | drafting | ~$24K · 12 mo | Can an LLM produce its own weight updates as a first-class output and improve within a session? |

Each plan folder contains:

```
NN-<slug>/
├── README.md       # problem, idea, success criteria
├── validation.md   # protocol, baselines, metrics, ablations
├── channels.md     # benchmarks, datasets, verifiers, base models
├── budget.md       # GPU-hours, $, headcount, decision points
└── references.md   # closest prior work
```

Numbering matches the brainstorm idea ID in [`../inference-time-training-ideas.md`](../inference-time-training-ideas.md).
