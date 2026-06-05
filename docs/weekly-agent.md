# Weekly Agent State

> Weekly structured state for agents. Latest week first.
> Purpose: resume context quickly without reading every plan file.

---

## Week of 2026-06-01

### Active Threads

| Thread | State | Agent action |
|---|---|---|
| Plan 08 / `mem-embedding` v1 | Characterization framing locked | Treat v1 as lossy-compression regime study; cite setting `P08-S2` for result cells |
| Plan 08 v1.5 | Intrinsic do-no-harm gating in progress | Use confidence + divergence as first gate features; cite setting `P08-S3` |
| Plan 09 / Janus | Pilot / measurement line active | Prioritize H3 causal protection test |
| RCA application | Motivation / showcase | Connect methods to preserving base ability during RCA adaptation |

### Plan 08 State

- v1 result: learned soft-prompt memory is a narrow lossy compressor.
- Settings registry: [`settings.md`](../notes/plans/08-model-outputs-delta-w/settings/settings.md).
- Regime A: gist/compression tasks can benefit.
- Regime B: reasoning-transfer tasks are near chance.
- Regime C: exact retrieval collapses; full context remains strong.
- v1 headline cells use `P08-S2`.
- v1.5 pivot: do-no-harm gate that opens only when compression is safe.
- v1.5 signal-grid cells use `P08-S3`.
- Current usable gate features:
  - positive: wrapper confidence, sequence log probability;
  - negative / warning: wrapper-to-base divergence.
- Do not headline single-seed layer-specific signals unless multi-seed confirms.

### Plan 09 State

- Thesis: long-context read-side sites may predict fine-tuning forgetting sites.
- Current evidence: cross-model coupling is positive for selected LC × CF metric
  pairs, especially retrieval / attention-distance / previous-token metrics
  against gradient magnitude / activation drift / Fisher.
- Sink heads and retrieval heads are largely disjoint; do not frame sink-only as
  the main protection target.
- Next gate: H3 causal test. Protect LC-coupled heads during SFT and measure:
  forgetting reduction, long-context retention, and new-domain learning.

### Human-Facing Message

The high-level story is controlled adaptation: improve RCA / long-context
capability while preserving the frozen base model's general ability. Plan 08
does this through gated memory compression; Plan 09 tests whether intrinsic
sites can be protected during fine-tuning.

### Files To Check First

- Human weekly summary: [`docs/weekly-human.md`](weekly-human.md)
- Plan 08 main: [`notes/plans/08-model-outputs-delta-w/README.md`](../notes/plans/08-model-outputs-delta-w/README.md)
- Plan 08 settings registry: [`notes/plans/08-model-outputs-delta-w/settings/settings.md`](../notes/plans/08-model-outputs-delta-w/settings/settings.md)
- Plan 08 v1 result: [`notes/plans/08-model-outputs-delta-w/v1-results-2026-06-03.md`](../notes/plans/08-model-outputs-delta-w/v1-results-2026-06-03.md)
- Plan 08 v1.5 note: [`notes/plans/08-model-outputs-delta-w/summary/2026-06-04/v1.5-intrinsic-gating-study-2026-06-04.tex`](../notes/plans/08-model-outputs-delta-w/summary/2026-06-04/v1.5-intrinsic-gating-study-2026-06-04.tex)
- Plan 09 main: [`notes/plans/09-intrinsic-site-protection/README.md`](../notes/plans/09-intrinsic-site-protection/README.md)
- Plan 09 matrix: [`notes/plans/09-intrinsic-site-protection/matrix.md`](../notes/plans/09-intrinsic-site-protection/matrix.md)
