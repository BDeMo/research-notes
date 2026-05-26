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
├── known/         # public knowledge base by category
│   ├── README.md  # nearness graph between categories
│   └── <cat>/     # one folder per category
│       └── README.md  # definition + containment + key concepts + refs
└── notes/
    ├── ideas/                  # idea catalog
    │   ├── README.md           # TOC with meta per idea (the directory file)
    │   └── <topic>.md          # full brainstorm per topic
    └── plans/                  # detailed project plans
        ├── README.md           # TOC with meta per plan (the directory file)
        └── <NN>-<slug>/        # one folder per plan
```

`ideas/` and `plans/` mirror each other: both have a `README.md` acting as the directory/TOC.
`known/` mirrors this on the *knowledge* side: top-level `README.md` is the directory + nearness graph; each subfolder is one category.

## Naming

- **Plan folders**: `NN-<slug>/` where `NN` is a zero-padded 2-digit index that **matches the brainstorm idea ID** (e.g., plan `01` = idea `I1`; plan `08` = idea `H6`).
- **Idea brainstorm files**: `notes/ideas/<topic-slug>.md`, one file per major topic.
- **Session entries**: `docs/matrix/YYYY-MM-DD-<topic-slug>.md`.
- **Topic slugs**: lowercase, hyphenated, no spaces. e.g., `inference-time-training`, not `Inference Time Training`.
- **Idea IDs in brainstorms**: `<LetterCategory><Number>`, e.g., `A1`, `I3`, `H6`. Categories: A supervision · B parameterization · C trigger · D search · E applications · F systems · G theory · H wild · I framing-derived. Use consistent letters across topics where possible.

## Directory/TOC files (`README.md` in `notes/ideas/` and `notes/plans/`)

Each of these acts as a *directory* of items. Required content:
- A short header explaining what's in the folder.
- The four-axis legend block at the top (see [`memory/symbols.md`](symbols.md)).
- One row per item with columns: **ID · Title · ★ · S · φ · M · Plan · Meta** (for ideas) or **# · Plan · Parent · ★ · S · φ · M · Cost · Time · Headline** (for plans).
- For plans: also a per-plan meta block with one-liner / validation hypothesis / primary channels / budget tier / kill criterion / sequels-in-queue.

All status / priority / phase / mode tokens are defined in [`memory/symbols.md`](symbols.md). Do not invent new codes silently.

## Public knowledge base (`known/`)

`known/` is the **curated, paradigm-level** knowledge organized by topic. Distinct from `docs/matrix/knowledge-sources.md` which is the *chronological intake log*.

- `known/README.md` maintains the **nearness graph** between categories (which categories are conceptually close), as both a Mermaid graph and a per-category adjacency list.
- Each `known/<category>/README.md` is a category-specific knowledge entry containing: definition · **Contained by** (parents) · **Contains** (sub-topics) · key concepts · foundational references · open questions.
- A reference (paper/blog/etc) is logged in `knowledge-sources.md` once with a stable `[id]`. Any number of `known/<cat>/README.md` files may then cite that `[id]`.
- Adding a category requires updating both the new folder's README *and* the top-level `known/README.md` (categories table + Mermaid graph + adjacency list + reciprocal entries in any affected neighbor's README).

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
