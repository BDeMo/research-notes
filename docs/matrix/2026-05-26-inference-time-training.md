# 2026-05-26 — Inference-time training (X-W framing)

> First session of the repo. Long brainstorm + paper reading + 3 plans.

## Activities

- **Framing** — defined `y = f(X; W)`. $X$ is everything tunable at inference (prompt, CoT, retrieval, tool use, test-time compute). $W$ is weights. Most research focuses on the $X$ axis; the open frontier is the $X \leftrightarrow W$ exchange. Specifically: *exhaust $X$ first, then look for the best $W$*.
- **Brainstorm 1** — 35 ideas across 8 dimensions: supervision signal (A), parameterization (B), trigger (C), search (D), application (E), systems (F), theory (G), wild (H).
- **Reading** — Sakana AI's **Doc-to-LoRA** (arXiv:2602.15902) and **Text-to-LoRA** (ICML 2025). Meta-learns context distillation into a hypernet → sub-second LoRA from a document.
- **Brainstorm 2** — reframed under "$X$ exhausted, find best $W$". 11 more ideas (I-series), including the three that became plans below.
- **Plans drafted** — three highest-priority ideas:
  - **Plan 01**: X-saturation curve + curriculum (data curation by inference-time difficulty)
  - **Plan 03**: W-space Best-of-N (test-time compute scaling along the weights axis)
  - **Plan 08**: Model outputs ΔW as part of generation (self-modifying LLMs)

## Decisions

- **Adopted** the $X$-$W$ exchange as the central framing for this whole sub-area. All future ideas should be locate-able on this axis.
- **Demoted** "TTT as compression" (originally H4) — Doc-to-LoRA already covers it. We're not chasing.
- **Promoted** ideas that connect inference-time compute to weight updates. The 2024–25 narrative was "more X tokens"; we expect the 2026+ narrative to be "more W bits at inference."
- **Naming convention** for plans: zero-padded 2-digit index matching the matched brainstorm idea ID (so plan 01 = I1, plan 03 = D1, plan 08 = H6).

## Output artifacts

- [`notes/inference-time-training-ideas.md`](../../notes/inference-time-training-ideas.md) — 50-idea brainstorm
- [`notes/plans/01-x-saturation-curve/`](../../notes/plans/01-x-saturation-curve/) — X-saturation curve plan
- [`notes/plans/03-w-space-best-of-n/`](../../notes/plans/03-w-space-best-of-n/) — W-space Best-of-N plan
- [`notes/plans/08-model-outputs-delta-w/`](../../notes/plans/08-model-outputs-delta-w/) — self-modifying LLM plan
- [`docs/workflow.md`](../workflow.md) — research methodology
- [`docs/matrix/knowledge-sources.md`](knowledge-sources.md) — references gathered today

## Knowledge sources used

See `knowledge-sources.md` entries tagged `#inference-time-training`, `#hypernetwork`, `#context-distillation`, `#ttt`.

Primary new sources:
- Doc-to-LoRA (Sakana AI, 2026)
- Text-to-LoRA (Sakana AI, 2025)
- Cartridges (Eyuboglu et al., 2025)
- Sleep-time compute (Lin et al., 2025)
- AlphaEdit (Fang et al., 2025) — null-space constrained editing
- Sparse memory finetuning (Lin et al., 2025)

## Late addition: `memory/` folder convention

After the plans were committed, the user added a standing rule:
> "所有要你记住的、和本repo相关的东西，写到本repo，都维护在memory文件夹里面"

This created `memory/{README,instructions,conventions,context}.md` as the new top-priority read for every session. The first three persistent rules were recorded in `memory/instructions.md`:
1. Maintain a `memory/` folder for everything to remember.
2. Plans must include validation, channels, and budget — concretely.
3. Every session ends with a matrix entry + knowledge-sources update.

Root README and `docs/workflow.md` updated to point at `memory/` as the first read.

## Next steps

In priority order:

1. **Pilot Plan 01** — single-day experiment on GSM8K with Qwen2.5-7B. Sweep CoT-budget {1, 8, 32, 128, 512} and plot accuracy curve. Estimate hits/sample size for $X$-saturated residual set. Budget: ≤20 GPU-hours.
2. **Pilot Plan 03** — pick 200 MATH problems, hand-craft 8 LoRA experts (or sample Gaussian noise), measure pass@1 with verifier-selection vs temperature-sampling-BoN at matched compute. Budget: ≤10 GPU-hours.
3. **Plan 08 prep** — read Generative Adapter (Chen et al., 2025) and HINT (Ivison et al., 2023) more carefully. Sketch the simplest possible RL formulation.

## Open questions raised this session

- What is the *intrinsic $W$-demand* of a task — measurable in bits?
- Is the $X{\to}W$ exchange rate stable across tasks, or task-specific?
- Does W-diversity provide *different* coverage than temperature-sampling diversity in BoN?
- Privacy: per-request LoRA in batched serving — can it leak?
