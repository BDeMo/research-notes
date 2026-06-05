# Plan 08 — raw result data

Result data (CSVs + figures) for plan 08, by experiment. Written analysis lives in
[`../summary/`](../summary/); exact recipes in [`../settings/settings.md`](../settings/settings.md);
raw per-item probe dumps / checkpoints stay in `mem-test/mem-embedding/` (pods).

## [`grids-2026-06-04/`](grids-2026-06-04/) — single-model signal grid (Qwen3-8B, setting `P08-S3`)
- `corr_long.csv`, `grid_auroc_correct.csv`, `grid_auroc_noharm.csv`, `grid_spearman_cont.csv` — signal↔usefulness correlations.
- `rank_help.png`, `rank_interesting.png` — ranking figures.
- Analysis: [`../summary/2026-06-04/v1.5-metric-candidates-2026-06-04.md`](../summary/2026-06-04/v1.5-metric-candidates-2026-06-04.md).

## [`grids-xmodel-2026-06-05/`](grids-xmodel-2026-06-05/) — cross-model + gate + boundary (settings `P08-S6/S7/S8`)
- `xmodel_consistency_7models.csv` (+`.png`) — per-signal AUROC × 7 model families.
- `gate_ceiling_cv.csv`, `gate_transfer_lomo.csv`, `gate_route_main.csv` — gate ceiling / LOMO transfer / offline routing.
- `transfer_matrix.csv`, `transfer_long.csv`, `transfer_heatmap.png` — train×test capability-boundary grid.
- Analysis: [`../summary/2026-06-05/v1.5-glossary.md`](../summary/2026-06-05/v1.5-glossary.md).
