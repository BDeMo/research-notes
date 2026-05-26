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

- [`notes/ideas/inference-time-training.md`](../../notes/ideas/inference-time-training.md) — 50-idea brainstorm (catalog at [`notes/ideas/README.md`](../../notes/ideas/README.md))
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

## Late additions

### `memory/` folder convention

After the plans were committed, the user added a standing rule:
> "所有要你记住的、和本repo相关的东西，写到本repo，都维护在memory文件夹里面"

This created `memory/{README,instructions,conventions,context}.md` as the new top-priority read for every session. The first three persistent rules were recorded in `memory/instructions.md`:
1. Maintain a `memory/` folder for everything to remember.
2. Plans must include validation, channels, and budget — concretely.
3. Every session ends with a matrix entry + knowledge-sources update.

Root README and `docs/workflow.md` updated to point at `memory/` as the first read.

### Mirrored `ideas/` and `plans/` folders with directory READMEs

User wanted both folders to have a TOC file with one-row meta per item. Moved `notes/inference-time-training-ideas.md` → `notes/ideas/inference-time-training.md`; added `notes/ideas/README.md` as the ideas catalog (50 rows) and rewrote `notes/plans/README.md` to match.

### Symbol system for status / priority / phase / mode

User asked for a notation set to record validation progress, project progress, phase/sequel relationships, and development mode. Created `memory/symbols.md` with four axes:
- **S** — lifecycle: `? R F D P X W S M B K Z`
- **★** — priority: `★ ★★ ★★★`
- **φ** — phase/chain: `1/N`, `↪#NN`, `←#NN`, `≈#NN`
- **M** — mode: `exp / proto / paper / thesis / prod / side` × `solo / collab / team / open`

Applied across both `notes/ideas/README.md` (all 50 idea rows) and `notes/plans/README.md` (3 plan rows + per-plan meta blocks). Linked sequels-in-queue from each plan.

### Public knowledge base `known/` initialized

User added the structural rule:
> "维护一个公共知识库，作为一个文件夹，known，其需要按类别分类进不同的知识库… 然后里面要维护包含关系，以及最相近的知识库。维护这个'相近'关系的是known本身的目录"

Created `known/` with 8 initial categories, all drawn from this session's topics:
- `inference-time-training/` (central)
- `hypernetworks/` · `context-distillation/` · `test-time-training/` (close cluster around the central topic)
- `lora-peft/` · `model-editing/` (W-axis machinery)
- `inference-time-compute/` (the X-axis sibling)
- `long-context/` (the problem the W-axis addresses)

`known/README.md` maintains the **nearness graph** (Mermaid + adjacency list). Each category's own `README.md` maintains **containment** (Contained-by / Contains), key concepts, and foundational references that cross-reference `[id]`s in `knowledge-sources.md`. The standing rule is recorded as the fifth instruction in `memory/instructions.md`.

### Maintenance & context-budget plan

User added the sixth standing rule:
> "在关键的地方写好这套维护方法，为了避免累积造成context过大，我们必须做好这个知识库、idea库的管理文件以及方案"

Created [`docs/maintenance.md`](../maintenance.md) as the central, authoritative policy with:
- **Tier system** — T0 (`memory/*` + matrix README + latest matrix entry), T1 (the three TOC files), T2 (on-demand), T3 (archive). Combined T0+T1 budget ≤ ~1200 lines ≈ 15K tokens.
- **Per-file size caps** (soft / hard) for every T0/T1 file.
- **Pruning rules** — summarize-don't-delete; collapse 90-day-old matrix entries; archive dormant ideas (`S=?` >12 mo), killed plans (`S=K`), and unvisited `known/<cat>/`.
- **Factoring rules** — when a single doc carries multiple sub-topics, split into a folder.
- **End-of-session hygiene checklist** (8 items).
- **Anti-bloat principles** — one source of truth, stable IDs, indexes over prose, defer specifics.
- **Escalation** — when even pruning fails, spin a topic into its own repo.

Wired local **§ Maintenance** sections into all T1 entry points so they're discoverable from where you'd actually add new content:
- `notes/ideas/README.md`
- `notes/plans/README.md`
- `known/README.md`
- `docs/matrix/README.md`
- `memory/conventions.md`

Updated `memory/README.md` read-order section to make the T0/T1 tier explicit. Added the maintenance rule as instruction #6.

Initial audit at this point:
- T0 total: `memory/*` 443 + matrix README 83 + this entry 127 ≈ **653 lines**.
- T1 total: ideas 154 + plans 86 + known 141 ≈ **381 lines**.
- **Combined ≈ 1034 lines ≈ 13K tokens** — within the 1200-line budget. Every individual file under its hard cap. No pruning needed yet.

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
