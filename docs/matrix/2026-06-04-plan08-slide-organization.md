# 2026-06-04 — Plan 08 slide organization

## Activities

- Pulled and merged latest `research-notes/main` from GitHub.
- Restored the previous weekly slide input in `slides/main.tex` and appended the new week below it.
- Moved repeated weekly `\titlepage` usage to a single deck-level cover in `slides/main.tex`.
- Updated Plan 08 and slides README files with parent/child folder organization rules.
- Recompiled `slides/main.pdf` after including both weekly inputs.

## Decisions

- `slides/main.tex` keeps historical weekly inputs and appends new weekly inputs in chronological order.
- Weekly slide files should start with a `\section{...}` and a compact reporting-period frame, not their own deck cover.
- Parent README files should describe child folders, link child README files, and explain how files are organized/appended.

## Output artifacts

- `memory/instructions.md`
- `notes/plans/README.md`
- `notes/plans/08-model-outputs-delta-w/README.md`
- `notes/plans/08-model-outputs-delta-w/slides/README.md`
- `notes/plans/08-model-outputs-delta-w/slides/main.tex`
- `notes/plans/08-model-outputs-delta-w/slides/main.pdf`
- `notes/plans/08-model-outputs-delta-w/slides/weekly/2026-05-w04.tex`
- `notes/plans/08-model-outputs-delta-w/slides/weekly/2026-06-w01.tex`

## Knowledge sources used

- No new external sources.

## Next steps

- For future weekly slide updates, append a new `\input{weekly/YYYY-MM-wNN.tex}` line under the existing weekly input block.
- If a child folder grows new artifact types, update both the parent README and the child folder README.
