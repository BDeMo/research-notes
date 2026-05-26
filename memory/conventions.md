# Repo conventions

> Naming, formatting, and structural rules. Stable across sessions.

---

## Directory layout

```
research-notes/
├── memory/        # standing instructions (read first)
├── docs/
│   ├── workflow.md
│   └── matrix/    # session log + knowledge-sources.md
└── notes/
    ├── <topic>-ideas.md       # brainstorm dumps
    └── plans/<NN>-<slug>/     # detailed project plans
```

## Naming

- **Plan folders**: `NN-<slug>/` where `NN` is a zero-padded 2-digit index that **matches the brainstorm idea ID** (e.g., plan `01` = idea `I1`; plan `08` = idea `H6`).
- **Session entries**: `docs/matrix/YYYY-MM-DD-<topic-slug>.md`.
- **Topic slugs**: lowercase, hyphenated, no spaces. e.g., `inference-time-training`, not `Inference Time Training`.
- **Idea IDs in brainstorms**: `<LetterCategory><Number>`, e.g., `A1`, `I3`, `H6`. Categories: A supervision · B parameterization · C trigger · D search · E applications · F systems · G theory · H wild · I framing-derived. Use consistent letters across topics where possible.

## Plan folder template

Every plan folder must contain at least these five files (template defined in [`docs/workflow.md §2`](../docs/workflow.md)):

| File | Required content |
|---|---|
| `README.md` | problem, core idea (with equations where applicable), success criteria, kill criteria, file index |
| `validation.md` | validation hypothesis (one sentence), experimental protocol, ablations, baselines, statistical practice, risk register |
| `channels.md` | specific benchmarks (with versions), datasets, verifiers, base models, comparable published methods, code/data to fork |
| `budget.md` | TL;DR table (phase/wall-clock/GPU-hours/$/decision), per-phase breakdown, headcount, $-ladder with kill points |
| `references.md` | grouped by sub-topic with short annotations explaining *why each ref matters here*, not just citations |

## Markdown formatting

- Math: inline `$...$`, block `$$...$$`.
- Code references to existing repo files: relative links `[label](relative/path)`.
- External papers: prefer arXiv links; include the year.
- Tables for budgets / comparisons / TL;DRs. Plain prose for problem statements.
- Heading levels: `#` for file title only; `##` for major sections; `###` for sub-sections; rarely below.
- No emojis unless explicitly requested.

## Brainstorm document structure

The canonical brainstorm doc (e.g., `notes/<topic>-ideas.md`) has these sections in order:
1. TL;DR
2. Framing (the bilevel / axes / trade-off being explored)
3. Ideas, grouped by dimension (A, B, C, …), marked with ★ for top picks
4. Reading notes on key adjacent papers (with explicit "limitations" / "idea gaps" sub-section)
5. Top picks (the 3–5 to actually plan)
6. Open questions
7. References (grouped, annotated)

## Knowledge sources (`docs/matrix/knowledge-sources.md`)

- Every entry has a short ID in `[brackets]` so it can be referenced from any note (e.g., "see [d2l]").
- Entry format: `### [id] Short title — Author, Year` + Type/Link/Why it matters/Used in/Tags.
- Tags use `#kebab-case`.

## Commit messages

- Imperative present tense ("Add", "Update", "Refactor"), no period at end of title.
- One-line summary, then optional body wrapped at ~72 cols.
- Title scope: state what *changed* (file/folder), not what was thought.

## Language / communication

- Author writes/thinks in Chinese; English in code, headings, file names, and technical writing.
- Markdown content can be either language but should be consistent within a file.

## When to deviate

- Deviations from this file are allowed for one-off cases. **Do not change conventions silently.** Update this file with a dated note explaining the change.
