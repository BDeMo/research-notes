# 2026-06-04 — Plan 08 setting management

## Activities

- Recorded the standing rule that every result cell must trace to a setting.
- Created a centralized Plan 08 settings/provenance registry.
- Linked v1 result cells, v1.5 signal-grid cells, v2 design claims, slides, weekly summaries, and matrix benchmark summaries back to stable setting IDs.
- Recompiled the v1.5 technical note PDF and weekly slides PDF so PDF deliverables also expose the setting links.

## Decisions

- Use `settings.md` as the source of truth for Plan 08 model/data/hyperparameter provenance.
- Use `P08-S1` for the v1 canonical wrapper recipe, `P08-S2` for v1 Phase Y three-regime benchmark cells, `P08-S3` for v1.5 intrinsic signal probes, and `P08-S4` for v2 design claims.
- Short reports and slides should cite setting IDs instead of repeating full hyperparameters.

## Output artifacts

- `memory/instructions.md`
- `notes/plans/08-model-outputs-delta-w/settings.md`
- `notes/plans/08-model-outputs-delta-w/README.md`
- `notes/plans/08-model-outputs-delta-w/v1-results-2026-06-03.md`
- `notes/plans/08-model-outputs-delta-w/v1.5-intrinsic-gating-study-2026-06-04.tex`
- `notes/plans/08-model-outputs-delta-w/v1.5-intrinsic-gating-study-2026-06-04.pdf`
- `notes/plans/08-model-outputs-delta-w/slides/weekly/2026-06-w01.tex`
- `notes/plans/08-model-outputs-delta-w/slides/main.pdf`

## Knowledge sources used

- No new external papers added. This session reorganized provenance for existing Plan 08 artifacts.

## Next steps

- For any new Plan 08 result, add/update a setting entry before publishing the table.
- If Plan 09 / Janus result tables are promoted into weekly reporting, create a matching Plan 09 settings registry before citing numbers in slides.
