# Plan 08 v0 Slides

Weekly progress slides for Plan 08 v0: learned memory wrapper plus RCA-Code.

## Files

- `main.tex`: Beamer wrapper. It owns the preamble and uses `\input{...}` to
  include the active combined weekly report.
- `main_brief_research.tex`: high-level research delivery deck. It should only
  explain the research question, architecture, hypothesis, evaluation, and paper
  path.
- `main_brief_project.tex`: high-level project delivery deck. It should only
  explain RCA-Code, product value, public datasets, demo, model release, and
  execution plan.
- `weekly/`: combined weekly reports.
- `brief/research/`: research-only brief content.
- `brief/project/`: project-only brief content.

## Deck Separation Rule

Maintain three decks separately:

- **Combined weekly progress** (`main.tex`): useful for internal weekly status.
  It can mention both research and project, but should stay outcome-first.
- **Research brief** (`main_brief_research.tex`): do not mix in project delivery
  details except as motivation. Focus on the research contribution.
- **Project brief** (`main_brief_project.tex`): do not spend time on paper-level
  novelty except as rationale. Focus on RCA-Code delivery and adoption.

When updating content, edit the corresponding `weekly/`, `brief/research/`, or
`brief/project/` file, not the main wrapper unless switching active week.

## Weekly File Rule

Maintain one `.tex` file per reporting week under `weekly/`.

Naming format:

```text
weekly/YYYY-MM-wNN.tex
```

Rules:

- The month is determined by the Monday of the reporting week.
- The week number is the ordinal Monday inside that month.
- If a week crosses a month boundary, still use the month of that week's Monday.
- `main.tex` should not contain weekly slide content directly. It should include
  exactly one weekly file with `\input{weekly/YYYY-MM-wNN.tex}`.
- `main_brief_research.tex` should include exactly one file from
  `brief/research/YYYY-MM-wNN.tex`.
- `main_brief_project.tex` should include exactly one file from
  `brief/project/YYYY-MM-wNN.tex`.
- Each weekly file should show the reporting month and week on the title slide.

Examples:

- Monday 2026-05-04 -> `weekly/2026-05-w01.tex`
- Monday 2026-05-11 -> `weekly/2026-05-w02.tex`
- Monday 2026-05-18 -> `weekly/2026-05-w03.tex`
- Monday 2026-05-25 -> `weekly/2026-05-w04.tex`
- Monday 2026-06-01 -> `weekly/2026-06-w01.tex`

## Compile

On Overleaf:

1. Upload this `slides/` folder.
2. Set `main.tex` as the main file.
3. Compile with pdfLaTeX.

Local compile, if LaTeX is installed:

```bash
pdflatex main.tex
```

## Maintenance Notes

- For a new week, copy the previous weekly file, rename it using the rule above,
  update the title slide, then update the `\input{...}` line in `main.tex`.
- Use an industry-style progress-report structure:
  - If the project/report is continuing from a previous week, start with a short
    recap: last week's goal, decision, open risk, and what changed this week.
  - If the project/report is starting a new topic, start with background and
    significance before the technical plan.
  - Prefer outcome first, then evidence, then risks / asks / next decisions.
  - Each slide should answer one management question: why it matters, what was
    decided, what was done, what changed, what is blocked, or what decision is
    needed.
- Keep this deck high-level. Put detailed calculations in `../v0-budget.md`.
- Update the budget slide only after changing `../v0-budget.md`.
- Keep private RCA/Nokia DTS details out of the slides unless the audience is internal.
- Add result slides after the first wrapper smoke run lands.
