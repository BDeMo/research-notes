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
| 09 | [Intrinsic-Site Protection (long-ctx ↔ forgetting)](09-intrinsic-site-protection/) | [P0](../ideas/rca-transformer-intrinsic-2026-06-03.md) | ★★★ | P | ≈#08 | paper,solo | ~$9.4K | 3–4 mo | Are the transformer sites that carry long-context behavior the *same* sites whose perturbation causes forgetting — and does protecting them fix both? |

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

### Plan 09 — Intrinsic-Site Protection (long-ctx ↔ forgetting)
- **Parent**: P0 thesis in [`../ideas/rca-transformer-intrinsic-2026-06-03.md`](../ideas/rca-transformer-intrinsic-2026-06-03.md) (design rules §0) · `S = P` · `φ = ≈#08` · `M = paper,solo`
- **One-liner**: Measure-first — are the intrinsic sites that carry long-context the *same* sites perturbed by SFT? If yes, a data-agnostic protection rule fixes both long-ctx retention and forgetting (zero task-specific params).
- **Pilot (2026-06-04, 8 models / 7 H100s)**: sink≠retrieval robust across 0.6→14B; ρ(retrieval, SFT-drift) > ρ(sink, drift) in ~all models (H2 *direction* ✅, magnitude clears 0.4 only on smallest 2 → partial); grad-mask protection works; NIAH probe + forgetting setup need hardening before H3. See `phase1-results-2026-06-04.md`.
- **Validation hypothesis**: Spearman ρ(long-ctx-importance, forgetting-disruption) ≥ 0.4 (H2); protecting top sites cuts forgetting ≥ 30% and beats MoFO/OPLoRA/ESFT by ≥ 5% retention without long-ctx regression (H3).
- **Primary channels**: RULER/HELMET (long-ctx) · GSM8K/HumanEval+/MMLU (forgetting) · TRACE (continual) · RCA (showcase) · Qwen3-8B + Llama-3.1-8B + Qwen3-30B-A3B.
- **Budget tiers**: Phase-0+1 de-risk ≈ 440 GPU-h ≈ $1.3K · Full paper ≈ 3140 GPU-h ≈ $9.4K.
- **Kill if**: H2 ρ ≈ 0 across models/domains after Phase 1 (→ negative-result note); or H3 fails despite H2 (→ pivot to pure-measurement paper).
- **General, not RCA-specific**: the method targets long-ctx + forgetting; RCA is the application where both coincide. Embodies design rules DR1–DR15.

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

If a plan adds child folders such as `slides/`, `grids-*`, `runs/`, `misc/`, or
`figures/`, the plan's top-level `README.md` must include a folder organization
section: what each child folder is for, whether it has its own `README.md`, and
how files should be appended or archived. Nontrivial child folders should carry
their own local `README.md`.

When a plan is killed, add `postmortem.md` to the folder and set `S = K` in the table above. Do not delete.

---

## Maintenance

Full policy: [`docs/maintenance.md`](../../docs/maintenance.md). Local rules for this index:

- **This file is T1** — read every session. Soft cap **100 lines**, hard cap **150 lines**.
- **Plan files (`NN-<slug>/*.md`) are T2** — read only when working on that plan. No per-file cap; budget lives in the parent idea + workflow template.
- **Add a plan**: pick `NN` matching the parent idea's promotion order; create folder from template; add row + meta block; bump parent idea's `S` to `D` and add Plan link.
- **Phase relations**: when a sequel emerges, mark its idea with `←#NN` or `↪#NN` in [`../ideas/README.md`](../ideas/README.md) — do *not* duplicate the sequel content here. Only list "Sequels in queue" with idea IDs.
- **Kill a plan**: add `postmortem.md`, set `S = K`, keep the folder; if the meta block grows verbose, condense it to one line "killed YYYY-MM-DD — see postmortem".
- **Archive a killed plan**: only if the row + meta block contribute > 20 lines of stale content. `git mv NN-<slug>/ _archive/NN-<slug>/`, replace row with `archived YYYY-MM-DD → _archive/NN-<slug>/`.
- **Meta-block size**: each per-plan meta block ≤ 10 lines. If a plan needs more, that content belongs in the plan's own `README.md`, not here.
