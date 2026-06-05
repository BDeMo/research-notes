# Plan 08 — reporting summary (by date)

This folder collects the **delivered reports** for plan 08 (v1.5 do-no-harm gating
study). Each subfolder is one work-block, named by generation date, and is
**self-contained** (docs + that day's grids + figures live together).

> Principle: only the **main logic** lives here. **Details refer out** —
> raw probe data, training checkpoints, and scripts stay in
> `mem-test/mem-embedding/` and are linked, not copied.

## Bundles

### [`2026-06-04/`](2026-06-04/) — single-model intrinsic-signal study
- [`v1.5-metric-candidates-2026-06-04.md`](2026-06-04/v1.5-metric-candidates-2026-06-04.md) — signal codebook + 4-axis screening (the curated metric list).
- [`v1.5-intrinsic-gating-study-2026-06-04.tex`](2026-06-04/v1.5-intrinsic-gating-study-2026-06-04.tex) / [`.pdf`](2026-06-04/v1.5-intrinsic-gating-study-2026-06-04.pdf) — paper-level note: settings, metric definitions, ranking, non-obvious filtering.
- [`grids-2026-06-04/`](2026-06-04/grids-2026-06-04/) — Qwen3-8B correlation CSVs + ranking figures.

### [`2026-06-05/`](2026-06-05/) — cross-model + gate + capability boundary
- [`v1.5-glossary.md`](2026-06-05/v1.5-glossary.md) — **start here**: wiki for every term, route, experiment (H1–H4), signal, setting; how to read the numbers; claims↔evidence.
- [`v1.5-gate-sweep-2026-06-05.md`](2026-06-05/v1.5-gate-sweep-2026-06-05.md) — the 9-config soft-gate ablation (negative result → motivates hard routing), with formulas.
- [`architecture-v1.5.tex`](2026-06-05/architecture-v1.5.tex) / [`architecture-v1.5.png`](2026-06-05/architecture-v1.5.png) — gated recurrent-memory diagram.
- [`grids-xmodel-2026-06-05/`](2026-06-05/grids-xmodel-2026-06-05/) — 7-family consistency, gate ceiling/LOMO, routing-gate, and train×test transfer CSVs + figures.

## Pointers (details, kept out of this folder)
- **Exact recipes / settings:** [`../settings.md`](../settings.md) (`P08-S0…S8`).
- **Live results ledger:** `mem-test/mem-embedding/summary/matrix.md`.
- **Code:** `mem-test/mem-embedding/scripts/` (`probe_v3.py`, `gate_train.py`, `gate_route.py`, `grid_transfer.py`, `signal_corr.py`, `xmodel_consistency.py`).
- **Slides (for reporting):** [`../slides/v1.5-gating.tex`](../slides/v1.5-gating.tex).
