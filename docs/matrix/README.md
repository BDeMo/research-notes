# Matrix — history & knowledge sources

> The **matrix** is the long-term memory of this repo.
> Two things it tracks:
> 1. **Sessions** — chronological log of what we did, what we read, what we decided.
> 2. **Knowledge mother nest (`knowledge-sources.md`)** — running index of papers, blogs, and conversations that seed our ideas.
>
> Re-read the matrix before any new session. The point is to *not* repeat yourself.

---

## How to use

### When starting a session
1. Open `docs/matrix/README.md` (this file).
2. Skim the latest session entries.
3. Check `knowledge-sources.md` for any tagged sources relevant to today's topic.

### When ending a session
1. Add a new session entry: `docs/matrix/YYYY-MM-DD-<topic>.md`.
2. If we read new papers or used new sources, add them to `knowledge-sources.md`.
3. Update the index table below.

### Session entry template
```
# YYYY-MM-DD — <topic>

## Activities
- Brainstorm on …
- Read paper …
- Wrote plan …

## Decisions
- Decided to …
- Killed idea …

## Output artifacts
- `notes/<topic>-ideas.md`
- `notes/plans/NN-<slug>/`

## Knowledge sources used
- [paper-id-1] in `knowledge-sources.md`

## Next steps
- …
```

---

## Session index

| Date | Topic | Headline output | Files |
|---|---|---|---|
| 2026-05-26 | Inference-time training (X-W framing) | Brainstorm of 50 ideas + 3 detailed plans + Doc-to-LoRA reading | [`2026-05-26-inference-time-training.md`](2026-05-26-inference-time-training.md) |
| 2026-05-28 | Methods reading + new plan brainstorm | Read 4 methods (Cartridges / Activation Beacon / Gisting / Generative Adapter) for plan 08 v0; 5 J-series ideas + 3 existing ★★+ candidates | [`2026-05-28-methods-reading-and-new-plans.md`](2026-05-28-methods-reading-and-new-plans.md) |
| 2026-06-03 | mem-X v1 results harvest (plan 08 v0) | Extracted facts from 7-day Phase A→Y burst; **3-regime law** (A wins / B at-chance / C collapse) is the v1 paper claim; OPD+RL stages dropped; J5 empirically answered | [`2026-06-03-mem-x-v1-harvest.md`](2026-06-03-mem-x-v1-harvest.md) |
| 2026-06-03 | Long-ctx↔forgetting brainstorm + audit + Plan 09 | 2 rounds → R1–R12 + M1–M9; **2022-2026 audit: none novel as standalone mechanism**; re-scoped to a *general* long-ctx + forgetting method (RCA = application); distilled **design rules DR1–DR15**; drafted **Plan 09** (measure-first coupling study → intrinsic-site anti-forgetting) | [`2026-06-03-rca-transformer-intrinsic-brainstorm.md`](2026-06-03-rca-transformer-intrinsic-brainstorm.md) |
| 2026-06-04 | Plan 08 setting management | Added centralized settings/provenance registry and linked v1/v1.5/v2 result cells, slides, PDFs, and summaries to stable setting IDs | [`2026-06-04-plan08-setting-management.md`](2026-06-04-plan08-setting-management.md) |
| 2026-06-04 | Plan 08 slide organization | Pulled latest main; restored previous weekly slide input, appended new week, removed repeated weekly covers, and updated README folder-organization rules | [`2026-06-04-plan08-slide-organization.md`](2026-06-04-plan08-slide-organization.md) |
| 2026-06-04 | Plan 08 high-level slides | Added goal/significance framing, comparison slide, TikZ architecture diagram, and evidence chain for wrapper training → forced-memory risk → gate | [`2026-06-04-plan08-high-level-slides.md`](2026-06-04-plan08-high-level-slides.md) |
| 2026-07-08 | Plan 08 IMP + long-context benchmarks | **Pivot to Paper B (IMP)**: span-level importance routing (retrieval=full, QA rescued, beats full by denoising); Paper A reconstruction-gate rejected (random-M control) + learned compressor can't compress; LongBench integrated at real length; 12h/6-GPU grand grid; Paper B draft; ResearchMeta methodology repo | [`2026-07-08-plan08-imp-and-longcontext-benchmarks.md`](2026-07-08-plan08-imp-and-longcontext-benchmarks.md) |

---

## Active threads

Topics that span multiple sessions and are still alive:

- **Plan 08 / Paper B — IMP (importance-routing on a frozen base)** — last touched 2026-07-08. **Primary line.** Span-level IMP: retrieval = full, QA rescued, beats full by denoising; not yet > RAG on short QA. Make-or-break next: linear base · necessity regime · semantic > lexical. Draft: `notes/plans/08-compressed-context-memory/papers/paper-b-forgetting-gating/PAPER-B-draft.md`.
- **Plan 08 / Paper A — do-no-harm gate** — blocked: learned compressor can't compress extractive QA (reconstruction-gate rejected); shipping gate = confidence. Secondary until a better compressor exists.
- **Inference-time training** — last touched 2026-06-03. Lead ideas: X-saturation curriculum (plan 01), W-space BoN (plan 03), model outputs ΔW (plan 08).
- **Plan 08 v0 — mem-X soft-prompt wrapper (`mem-test/mem-embedding`)** — **v1 paper submitting (characterization)** as of 2026-06-03. 25 phases A→Y on 8×H100. **3-regime law** is the v1 headline (A: wins on QuALITY · B: at-chance on MuSR · C: collapses on RULER-NIAH). v2 = suffix memory blocked until v1 freeze 2026-06-15. Full harvest: [`../../notes/plans/08-compressed-context-memory/history/v1-results-2026-06-03.md`](../../notes/plans/08-compressed-context-memory/history/v1-results-2026-06-03.md). Settings registry: [`../../notes/plans/08-compressed-context-memory/settings.md`](../../notes/plans/08-compressed-context-memory/settings/settings.md). Slide updates append weekly inputs in `slides/main.tex`; do not delete previous week inputs.

## Archived threads

(empty)

---

## Maintenance

Full policy: [`../maintenance.md`](../maintenance.md). Local rules for the matrix:

- **This file is T0** — read every session. Soft cap **80 lines**, hard cap **120 lines**.
- **Latest session entry is T0** — read every session. Soft cap **200 lines**, hard cap **300 lines**.
- **Older session entries are T2** — read only when researching past decisions.
- **knowledge-sources.md is T2** — read only when looking up a citation by `[id]`.
- **Stable IDs**: source IDs in `knowledge-sources.md` are forever. Never renumber.
- **Aging policy**:
  - After **30 days**, a session entry should be reviewed: drop "Next steps" if completed, keep "Activities / Decisions / Output".
  - After **90 days**, eligible for compression: collapse "Activities" to one line, keep "Decisions" + "Output" full.
  - After **12 months**, eligible for archive: `git mv` to `_archive/YYYY-MM/`; in the session index above, replace its row with a compact `archived 2027-MM-DD → [link]` line, or move under an "Archived sessions" collapsible block.
- **Active threads** list: cap at **10 rows**. Anything inactive > 6 months moves to Archived threads.
