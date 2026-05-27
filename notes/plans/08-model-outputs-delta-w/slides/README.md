# Plan 08 v0 Slides

Project progress slides for Plan 08 v0: learned memory wrapper plus RCA-Code.

## Files

- `main.tex`: Beamer wrapper. It owns the preamble and uses `\input{...}` to
  include the active weekly report.
- `weekly/`: one report file per week.

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
- Keep this deck high-level. Put detailed calculations in `../v0-budget.md`.
- Update the budget slide only after changing `../v0-budget.md`.
- Keep private RCA/Nokia DTS details out of the slides unless the audience is internal.
- Add result slides after the first wrapper smoke run lands.
