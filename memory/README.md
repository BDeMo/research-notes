# Memory

> **Standing instructions, preferences, and conventions for this repo.**
> Read this folder first at the start of every session.

---

## What goes here

Anything the user has explicitly asked to be remembered, *and* anything that should persist across sessions about how this repo is run.

In contrast:
- [`docs/workflow.md`](../docs/workflow.md) = methodology (*how* we work).
- [`docs/matrix/`](../docs/matrix/) = history (*what* we did, chronologically).
- [`known/`](../known/) = curated knowledge by category (*what we know about a topic*).
- [`memory/`](.) = standing rules (*what to remember*, durably).

If a session needs a rule that should apply to all future sessions, it goes here.

## Files

| File | Purpose |
|---|---|
| [`instructions.md`](instructions.md) | Explicit user instructions ("from now on, always …") |
| [`conventions.md`](conventions.md) | Repo conventions: file layouts, naming, formatting |
| [`symbols.md`](symbols.md) | Notation for status / priority / phase / mode in TOCs |
| [`context.md`](context.md) | Stable context about the user and active threads |

---

## How to maintain

- When the user says "记住 …" or "from now on …" → add it to [`instructions.md`](instructions.md) with the date.
- When we settle on a new convention (numbering, template, etc.) → record it in [`conventions.md`](conventions.md).
- When important stable context appears (active topics, ongoing collaborators, in-flight projects) → update [`context.md`](context.md).
- Never delete; supersede with dated entries and mark the old one `~~struck~~ (superseded YYYY-MM-DD)`.

## Read order at the start of a session (the **T0 + T1 hot set**)

Read everything below, and only this, before starting new work. Combined budget ≤ ~1200 lines ≈ 15K tokens. Anything else is loaded on demand.

**T0 — always**
1. `memory/README.md` (this file) — orientation
2. `memory/instructions.md` — standing rules
3. `memory/context.md` — who/what/where
4. `memory/conventions.md` — formatting + maintenance summary
5. `memory/symbols.md` — notation
6. `docs/matrix/README.md` — session index + latest activity
7. **latest** `docs/matrix/YYYY-MM-DD-*.md` — current working context

**T1 — always (navigation)**

8. `notes/ideas/README.md` — ideas TOC
9. `notes/plans/README.md` — plans TOC
10. `known/README.md` — knowledge-base TOC + nearness graph

**T2 — on demand**: any specific `known/<cat>/`, idea brainstorm, or plan file when relevant to the current task.

**T3 — rare**: anything under `*/_archive/...`.

Full tier definitions, size caps, pruning rules: [`docs/maintenance.md`](../docs/maintenance.md).
