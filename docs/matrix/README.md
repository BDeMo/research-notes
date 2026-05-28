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

---

## Active threads

Topics that span multiple sessions and are still alive:

- **Inference-time training** — last touched 2026-05-28. Lead ideas: X-saturation curriculum (plan 01), W-space BoN (plan 03), model outputs ΔW (plan 08).
- **Plan 08 v0 — Learned Memory Wrapper** — actively in execution (30+ commits since 2026-05-26). Frozen 8B base + trainable wrapper, RCA-demo motivated. Reading [genadapter] / [cartridges] / [act-beacon] / [gisting] for architecture decisions and baselines. New J-series ideas (2026-05-28) propose v0.1 architecture ablations + the bridge to plan 08 north star.

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
