# Symbol system

> Notation for recording **validation progress**, **project status**, **phase/sequel relation**, and **development mode** across ideas and plans.
> Designed to be scannable in a single table row.

The four axes:

| Axis | Column header | Question it answers |
|---|---|---|
| Status | `S` | Where in the lifecycle is this (from raw idea to shipped)? |
| Priority | `★` | How much do I rate this? |
| Phase | `φ` | Is this standalone, part of a chain, a spinoff, or a continuation? |
| Mode | `M` | How is it being developed (ambition + commitment)? |

---

## 1. Status — `S`

Single letter. Lifecycle of an idea/plan from seed to shipped (or killed).

| Code | Phase | Meaning |
|---|---|---|
| `?` | seed | Raw idea, no scrutiny beyond inception. |
| `R` | read | Adjacent literature digested; idea contextualized. |
| `F` | formalized | Problem statement + validation hypothesis written. |
| `D` | drafted | Plan folder exists (validation + channels + budget). |
| `P` | piloted | Small pilot experiment completed; signal observed. |
| `X` | executing | Main experiments running. |
| `W` | writing | Paper / report drafting in progress. |
| `S` | shipped | Paper submitted / artifact released / merged. |
| `M` | maintenance | Shipped and receiving ongoing care. |
| `B` | blocked | Waiting on resource / decision / external dependency. |
| `K` | killed | Validation failed; post-mortem written. |
| `Z` | archived | Paused indefinitely, not killed. |

**Reading rule**: monotonic forward `? → R → F → D → P → X → W → S → M`. Side branches: `B` (temporary), `K` and `Z` (terminal). Skipping forward is fine; rewinding requires a comment in the matrix.

## 2. Priority — `★`

Empty / `★` / `★★` / `★★★`. Pure subjective ranking from `novelty × feasibility × impact`.

- `★` notable — interesting but not actively prioritized
- `★★` strong — would happily work on this next
- `★★★` top — should already have a plan or be drafting one

## 3. Phase — `φ`

Relationships between work items. Empty for the common case (standalone).

| Code | Meaning |
|---|---|
| *(blank)* | Standalone; no planned follow-up. |
| `1/N` | Part 1 of N planned parts. Use `?` for unknown N. |
| `n/N` | Part n of N. |
| `↪#NN` | This item *spawned* item NN as a spinoff/follow-up. |
| `←#NN` | This item *continues* / *depends on* item NN. |
| `≈#NN` | Closely entangled with item NN (parallel work). |

Examples:
- `1/?` — first phase, more planned but count unknown.
- `2/2` — the planned continuation.
- `↪#04` — spawned plan 04 as a follow-up.
- `←#01` — continues from plan 01.

## 4. Mode — `M`

Development mode = ambition × commitment level. Use comma-separated tags.

**Ambition / pace**:

| Code | Meaning |
|---|---|
| `exp` | Exploratory — low-commitment, just trying things. |
| `proto` | Prototype — proof-of-concept, throwaway code OK. |
| `paper` | Paper-track — committed to a single-paper outcome. |
| `thesis` | Thesis-track — chapter-scale or multi-paper. |
| `prod` | Production — must be deployable / maintained. |
| `side` | Side-project — low priority, no deadline. |

**Collaboration**:

| Code | Meaning |
|---|---|
| `solo` | Just me. *(Default; omit if obvious.)* |
| `collab` | 1–2 named collaborators. |
| `team` | 3+ people. |
| `open` | Open-source / public from day one. |

**Combine with commas**: e.g. `paper,collab` · `thesis,solo` · `proto,exp` · `prod,team`.

---

## Compact row format for TOCs

Standard table columns (in order):

```
| ID | Title | ★ | S | φ | M | Link | Meta |
```

- `Link` = path to the plan folder if drafted, otherwise `–`.
- `Meta` = one-line description, ≤ ~80 chars.

Example rows:

```
| I1 | X-saturation curve as dataset curator        | ★★★ | D | 1/? | paper        | [01](../plans/01-…) | Train on residual where X can't help |
| D1 | W-space Best-of-N                            | ★★  | D |     | paper        | [03](../plans/03-…) | Sample N weight perturbations; verifier picks |
| H6 | Model outputs ΔW as part of generation       | ★★★ | D | 1/3 | thesis,solo  | [08](../plans/08-…) | Self-modifying LLM; emit (y, ΔW) per turn |
| A1 | Self-consistency as loss                     | ★   | ? |     |              | –                   | Distill majority vote into weights |
```

## Quick reference (legend block for any TOC)

Place at the top of any TOC document for self-containment:

```
S: ? seed · R read · F formalized · D drafted · P piloted · X executing
   W writing · S shipped · M maintenance · B blocked · K killed · Z archived
★: ★ notable · ★★ strong · ★★★ top
φ: 1/N part of chain · ↪#NN spawned NN · ←#NN continues NN · ≈#NN parallel
M: exp · proto · paper · thesis · prod · side  +  solo · collab · team · open
```

---

## Maintenance rules

- **One source of truth**: the symbol values live in `notes/ideas/README.md` and `notes/plans/README.md`. Detail files (per-plan `README.md`, brainstorm sections) link back to the TOC; they don't independently maintain status.
- **Update Status on every meaningful transition**, even if no work has happened — explicitly marking `B` (blocked) is more informative than letting a `D` row look stale.
- **Phase mark `↪#NN` / `←#NN` is bi-directional**: if I add `↪#04` on idea X, I should also add `←#X` on plan 04 (whichever direction makes more sense in the row).
- **Promotion from idea → plan**: at moment of promotion, set the idea's Status to `D` (or higher), populate Link, and add a Phase mark if there's a chain.
- **Killing**: set `S = K` and write a one-paragraph post-mortem in the plan folder (`postmortem.md`) before changing status.
