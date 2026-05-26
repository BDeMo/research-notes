# Maintenance & Context-Budget Policy

> How to keep this repo readable at session start as it accumulates content.
> All four pillars (`memory/` · `docs/` · `known/` · `notes/`) will grow. Naive "read everything" exceeds any LLM context window. This document defines what to read when, what to archive, and what to prune.

This is the single source of truth for maintenance rules. Other files reference this one.

---

## 0. The context budget

A session-start read should fit in **≤ ~15K tokens** (≈ 1200 markdown lines) so the model has room to think.

### Tier system

| Tier | What | Read when | Total budget |
|---|---|---|---|
| **T0** | `memory/*` + `docs/matrix/README.md` + latest matrix entry | Every session | ≤ 600 lines |
| **T1** | TOC files (`known/README.md`, `notes/ideas/README.md`, `notes/plans/README.md`) | Every session | ≤ 600 lines |
| **T2** | Specific `known/<cat>/`, idea brainstorm, plan detail files | On demand | unbounded per-file |
| **T3** | Archive (`*/_archive/...`) | Rare (postmortems, history searches) | unbounded |

**Combined T0+T1 ≤ 1200 lines ≈ 15K tokens.** Everything else is loaded only when needed.

---

## 1. T0 — must-read files & size caps

| File | Soft cap | Hard cap | If over hard cap |
|---|---|---|---|
| `memory/README.md` | 50 | 80 | Trim narrative; rules belong in `instructions.md` |
| `memory/instructions.md` | 120 | 200 | Archive instructions over 12 months old to `instructions-archive.md` |
| `memory/context.md` | 80 | 120 | Move archived threads out; keep only active |
| `memory/conventions.md` | 150 | 200 | Factor sub-areas into `conventions/<topic>.md` |
| `memory/symbols.md` | 150 | 200 | Mostly stable; rarely should grow |
| `docs/matrix/README.md` | 80 | 120 | Archive sessions > 12 mo old (see §3) |
| Latest matrix entry | 200 | 300 | Summarize older parts; keep "Decisions" + "Output" full |

If a file is over **hard cap**, treat it as a blocker for the next session and prune first.

## 2. T1 — TOC files & size caps

| File | Soft cap | Hard cap | If over hard cap |
|---|---|---|---|
| `notes/ideas/README.md` | 200 | 300 | Collapse non-★ rows to an appendix; archive dormant ideas |
| `notes/plans/README.md` | 100 | 150 | Move killed plans to `_archive/`; keep index row only |
| `known/README.md` | 200 | 300 | Split categories into clusters (e.g., a `known/_meta-index.md` per cluster) |

## 3. Pruning & archival

### Triggers

- A T0 or T1 file exceeds its **hard cap**.
- An idea has been `S = ?` for > 12 months → dormant.
- A plan is `S = K` (killed) → archive.
- A `known/<cat>/` has not been touched in 12 months and has no inbound link from active work → dormant.
- A session is > 90 days old → eligible for summarization.
- The same content is repeated in 3+ places → consolidate to one, link from others.

### How to prune (priority order)

1. **Summarize, don't delete.** Replace verbose content with a 5-line TL;DR + a link to the archived original. Always preserve enough to find what was removed.
2. **Collapse old sessions**. After 90 days, reduce a matrix entry to: *Activities (1 line) · Decisions (full) · Output artifacts (full) · Next steps (drop)*.
3. **Move to `_archive/`**. Place archived files at `<original-parent-dir>/_archive/YYYY-MM/<original-name>.md`. Replace the original with a stub or remove the TOC row.
4. **Factor / split**. When a single doc carries multiple sub-topics, split (see §4).

### Archive directory layout

```
docs/matrix/_archive/YYYY-MM/<session-file>.md      # old session summaries
notes/ideas/_archive/<topic>.md                     # dormant brainstorms
notes/plans/_archive/<NN>-<slug>/                   # killed plans with postmortem.md
known/_archive/<category>/                          # dormant categories
```

The `_` prefix sorts archives at the top of `ls`, visually flags them as system content, and excludes them from normal grep paths by convention.

### Archive workflow

1. `git mv` original → archive path (preserves history).
2. Write a 1-paragraph postmortem / dormancy note at the top of the archived file.
3. Update the relevant TOC: remove the row or replace with `archived 2026-MM-DD → [link]`.
4. Log the archival in the *current* session's matrix entry under a `## Pruning` heading.

## 4. Factoring rules (split when too large)

A single document carrying multiple sub-topics should become a folder.

### When to split

- `notes/ideas/<topic>.md` > 500 lines → factor.
- `known/<cat>/README.md` > 200 lines → factor.
- Any file > 300 lines that has 3+ independent sub-topics → factor.

### How to split

Convert the file into a directory with the same name, write a slim `README.md` index, and put the body in topical sub-files.

```
# Before:                      # After:
notes/ideas/X.md (700 lines)   notes/ideas/X/
                                ├── README.md      # framing + top picks + TOC (≤ 200 lines)
                                ├── core.md        # central content
                                └── full-list.md   # detail
```

The `README.md` of the new folder is what enters T1; the sub-files stay T2.

## 5. Hygiene checklist (per session)

Run this at the **end** of every session before closing:

- [ ] Logged this session in `docs/matrix/YYYY-MM-DD-*.md`.
- [ ] Updated TOC rows for any idea/plan that changed status `S` or phase `φ`.
- [ ] Added new sources to `docs/matrix/knowledge-sources.md` with a stable `[id]`.
- [ ] Any source that's now central to a `known/<cat>/` is cross-referenced there.
- [ ] No T0 file is over its hard cap. (Check with `wc -l memory/*.md docs/matrix/README.md`.)
- [ ] No TOC over its hard cap.
- [ ] Any idea/plan now eligible for archival? If so, move it.
- [ ] If any new convention emerged, recorded it in `memory/conventions.md` or `memory/symbols.md`.

If **two or more** items are unchecked, spend 5–10 minutes on housekeeping before closing.

## 6. Anti-bloat principles

These guide all the rules above.

- **One source of truth.** Status lives in the TOC. References live in `knowledge-sources.md`. Conventions live in `memory/`. Don't repeat any of these in detail files.
- **Indexes > prose for navigation.** Tables, adjacency lists, TOCs. Reserve prose for irreducible explanation.
- **Stable IDs.** Once a `[id]` in `knowledge-sources.md`, an idea letter code, or a plan `NN` is assigned, never change it. Renames break referential integrity across the repo.
- **Defer specifics.** If a detail isn't needed to *navigate*, push it into a leaf file (T2).
- **Resist new top-level folders.** Add a subfolder to an existing pillar before creating a fifth pillar.
- **Reciprocate cross-references.** If file A links to file B, B should (when relevant) link back. Especially `←#NN` / `↪#NN` phase relations and `known/` neighbor lists.
- **Date when in doubt.** Every accumulating list (instructions, sessions, archives) gets `YYYY-MM-DD` stamps to enable later pruning by age.

## 7. Escalation — when pruning fails

If a single area is consistently over budget despite pruning:

1. Spin out a separate repo for that area.
2. Replace its content here with a stub README + link to the new repo.
3. Maintain only the *interface* (status, last-touched date, one-liner) in the parent TOC.

Triggers for escalation:
- A `known/<cat>/` has accumulated 10+ T2 files and is regularly read in full.
- A topic has 3+ plans all in `running` or `executing` simultaneously.
- A single session entry is being amended for 30+ days (means it's a project, not a session).

## 8. Pointers from other files

The following T0/T1 files each carry a short **Maintenance** section pointing back here:

- `memory/conventions.md` § Maintenance
- `notes/ideas/README.md` § Maintenance
- `notes/plans/README.md` § Maintenance
- `known/README.md` § Maintenance
- `docs/matrix/README.md` § Maintenance

Those sections state only the *local* rule (size cap, archive trigger) and link here for the full policy.
