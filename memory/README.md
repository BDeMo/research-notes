# Memory

> **Standing instructions, preferences, and conventions for this repo.**
> Read this folder first at the start of every session.

---

## What goes here

Anything the user has explicitly asked to be remembered, *and* anything that should persist across sessions about how this repo is run.

In contrast:
- [`docs/workflow.md`](../docs/workflow.md) = methodology (*how* we work).
- [`docs/matrix/`](../docs/matrix/) = history (*what* we did, chronologically).
- [`memory/`](.) = standing rules (*what to remember*, durably).

If a session needs a rule that should apply to all future sessions, it goes here.

## Files

| File | Purpose |
|---|---|
| [`instructions.md`](instructions.md) | Explicit user instructions ("from now on, always …") |
| [`conventions.md`](conventions.md) | Repo conventions: file layouts, naming, formatting |
| [`context.md`](context.md) | Stable context about the user and active threads |

---

## How to maintain

- When the user says "记住 …" or "from now on …" → add it to [`instructions.md`](instructions.md) with the date.
- When we settle on a new convention (numbering, template, etc.) → record it in [`conventions.md`](conventions.md).
- When important stable context appears (active topics, ongoing collaborators, in-flight projects) → update [`context.md`](context.md).
- Never delete; supersede with dated entries and mark the old one `~~struck~~ (superseded YYYY-MM-DD)`.

## Read order at the start of a session

1. `memory/README.md` (this file) — orientation
2. `memory/instructions.md` — what the user told me to remember
3. `memory/context.md` — who, what, where we are now
4. `memory/conventions.md` — how to format / where to put things
5. `docs/matrix/README.md` — what we've recently done

Only after these five should we start any new work.
