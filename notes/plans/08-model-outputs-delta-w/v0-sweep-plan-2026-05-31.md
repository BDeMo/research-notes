# Plan 08 v0 — 35-hour Hyperparameter Sweep Plan

> Companion to [`v0-results-2026-05-30.md`](v0-results-2026-05-30.md) and
> [`v0-datasets-plan.md`](v0-datasets-plan.md). Locks in the design
> space the overnight sweep covers, the rationale per phase, and the
> aggregation/reporting plan.

## Reference point

The reference recipe entering this sweep is:

| dim | value |
|---|---|
| combine | `xattn` |
| K (memory tokens) | 32 |
| dataset | `categorical_niah`, 100 train + 50 held-out eval, 6 chunks × 256 tok |
| recipe | `sft,opd,rl` (800 + 600 + 400 steps) |
| RL | N=4 rollouts, temp=1.5, top_p=0.95, λ_KL=0.05, reward=`contains` |
| reg | λ_div=0.1, grad-clip=1.0 |
| lr | 5e-5 |
| base | Qwen3-8B, bfloat16 |

This was the best-known configuration at sweep start: `sft_opd_rl`
hit **eval contains=0.340** in earlier runs (vs `sft_only` 0.240),
albeit with RL contributing zero gradient (N=2). Baselines:
full_context = 0.72, retrieval = 0.90.

## Goal

Produce, by morning:

1. **Best configuration** with its eval contains / f1, plus a 5-seed
   stability number.
2. **Ablation matrix** along 6 axes (recipe, RL HP, OPD HP, architecture,
   regularization, data scale) with a clear "which knob matters" verdict.
3. **Aggregate markdown table** sorted by held-out contains, with
   train→eval gap, drift, and Δm-norm side-by-side.

## Scope (~28-30 GPU-hours of work, 5-7 hr buffer)

All 4 H100s run in parallel via a shared-queue orchestrator
(`mem-embedding/scripts/run_sweep.py`). Each job is ~30-50 min;
total wall time ≈ (jobs × 40 min) / 4 GPUs.

### Phase A — Recipe & Stage Ablation (~14 jobs, ~2.5 hr)

What it answers: does each stage actually contribute, and what's
the optimal stage budget?

| ID | recipe | sft / opd / rl steps | notes |
|---|---|---|---|
| A1 | `sft` | 1800 / — / — | reference; should overfit |
| A2 | `opd` | — / 1800 / — | confirms OPD-alone collapses |
| A3 | `rl` | — / — / 1800 | confirms cold-start RL fails |
| A4 | `sft,opd` | 1000 / 800 / — | does OPD generalize without RL anchor? |
| A5 | `sft,rl` | 1000 / — / 800 | RL with proper N=4 + temp 1.5 |
| A6 | `sft,opd,rl` | 800 / 600 / 400 | current reference |
| A7 | `sft,opd,rl` | 600 / 800 / 400 | more OPD |
| A8 | `sft,opd,rl` | 400 / 1000 / 400 | even more OPD |
| A9 | `sft,opd,rl` | 800 / 400 / 600 | less OPD, more RL |
| A10 | `sft,opd,rl` | 600 / 400 / 800 | minimum OPD, max RL |
| A11 | `sft,opd,rl` | 1200 / 400 / 200 | more SFT |
| A12 | `sft,opd,rl` | 400 / 400 / 1000 | RL-heavy |
| A13 | `opd,rl` | — / 600 / 600 | skip SFT (OPD cold start) |
| A14 | `sft,rl,opd` | 800 / 400 / 600 | reorder: SFT → RL → OPD |

### Phase B — RL hyperparameters (~12 jobs, ~2.5 hr)

Holds recipe at `sft,opd,rl` 800+600+400, varies RL knobs.

| ID | N rollouts | sample temp | λ_KL | top_p | reward |
|---|---|---|---|---|---|
| B1 | **2** | 1.5 | 0.05 | 0.95 | contains |
| B2 | 4 | 1.5 | 0.05 | 0.95 | contains (ref) |
| B3 | **8** | 1.5 | 0.05 | 0.95 | contains |
| B4 | **16** | 1.5 | 0.05 | 0.95 | contains |
| B5 | 4 | **1.0** | 0.05 | 0.95 | contains |
| B6 | 4 | **2.0** | 0.05 | 0.95 | contains |
| B7 | 4 | 1.5 | **0.0** | 0.95 | contains |
| B8 | 4 | 1.5 | **0.2** | 0.95 | contains |
| B9 | 4 | 1.5 | **0.5** | 0.95 | contains |
| B10 | 4 | 1.5 | 0.05 | **0.8** | contains |
| B11 | 4 | 1.5 | 0.05 | 0.95 | **f1** |
| B12 | 4 | 1.5 | 0.05 | 0.95 | **exact** |

### Phase C — Architecture (~12 jobs, ~2.5 hr)

| ID | combine | K | n_heads | n_layers | ffn_mult |
|---|---|---|---|---|---|
| C1 | xattn | **16** | 4 | 1 | 2 |
| C2 | xattn | 32 | 4 | 1 | 2 (ref) |
| C3 | xattn | **64** | 4 | 1 | 2 |
| C4 | xattn | **128** | 4 | 1 | 2 |
| C5 | **hybrid** | 32 | 4 | 1 | 2 |
| C6 | **interleave** | 32 | 4 | 1 | 2 |
| C7 | **residual** | 32 | 4 | 1 | 2 |
| C8 | xattn | 32 | **8** | 1 | 2 |
| C9 | xattn | 32 | 4 | **2** | 2 |
| C10 | xattn | 32 | 4 | **3** | 2 |
| C11 | xattn | 32 | 4 | 1 | **4** |
| C12 | xattn | 64 | 8 | 2 | 4 (big) |

### Phase D — Regularization & learning rate (~12 jobs, ~2.5 hr)

| ID | λ_div | dropout | lr | grad_clip |
|---|---|---|---|---|
| D1 | **0** | 0 | 5e-5 | 1.0 |
| D2 | 0.1 | 0 | 5e-5 | 1.0 (ref) |
| D3 | **0.3** | 0 | 5e-5 | 1.0 |
| D4 | **1.0** | 0 | 5e-5 | 1.0 |
| D5 | 0.1 | **0.1** | 5e-5 | 1.0 |
| D6 | 0.1 | **0.2** | 5e-5 | 1.0 |
| D7 | 0.1 | **0.3** | 5e-5 | 1.0 |
| D8 | 0.1 | 0 | **1e-5** | 1.0 |
| D9 | 0.1 | 0 | **3e-5** | 1.0 |
| D10 | 0.1 | 0 | **1e-4** | 1.0 |
| D11 | 0.1 | 0 | **3e-4** | 1.0 |
| D12 | 0.1 | 0.1 | 1e-4 | **0.5** |

### Phase E — Data scale (~9 jobs, ~3.5 hr)

Tests whether the wrapper gains scale with more diverse data (longer
runs proportional to data size to hold roughly-constant epochs).

| ID | n_items | n_eval_items | total steps |
|---|---|---|---|
| E1 | 100 | 50 | 1800 (ref) |
| E2 | **200** | 100 | 1800 |
| E3 | **200** | 100 | **2700** (matched epochs) |
| E4 | **400** | 200 | 1800 |
| E5 | **400** | 200 | **3600** (matched epochs) |
| E6 | **100** | 50 | **3600** (longer training on same data → overfit?) |
| E7 | 100 | 50 | 1800, **n_chunks=12** |
| E8 | 100 | 50 | 1800, **n_chunks=24** |
| E9 | **400** | 200 | 3600, n_chunks=12 |

### Phase F — Seed stability on top variants (~15 jobs, ~3 hr)

After Phases A-E, pick top 3 by held-out contains. Re-run each with
5 seeds (`seed ∈ {12345, 23456, 34567, 45678, 56789}`).

Output: mean ± std per metric per variant. This is the number we'll
quote in the v1 results.

### Phase G — Buffer / stretch (4-8 hr unused budget)

If buffer remains:

- **G1 long-context capacity** — n_chunks ∈ {32, 64} × K ∈ {32, 128}
  on the winning variant. Tests whether the wrapper actually
  compresses *long* context, the regime where it matters.
- **G2 coding_niah revisit** — re-run the winner on the original
  40-bit task to see if any of the discoveries help on the hard
  setting we abandoned.
- **G3 add multi_needle_niah** (per the datasets plan) and run the
  winner on it.

## Sweep infrastructure

```
mem-embedding/
├── scripts/
│   ├── run_sweep.py          # 4-GPU orchestrator, shared queue
│   └── aggregate_sweep.py    # walks sweep dir → markdown table
└── k8s/
    └── sweep/
        ├── sweep-master.json # ~74 jobs (Phases A-F)
        └── run-sweep.sh      # k8s entrypoint
```

**`run_sweep.py`**: reads the master JSON, spawns 4 workers (one per
GPU), each pops jobs from the shared queue and invokes
`train_smoke.py`. Writes a `runs_manifest.json` (name → hyperparams)
and a `status.jsonl` (job × completion timestamp × exit code) so the
aggregator can join and sort.

**`aggregate_sweep.py`**: at sweep end (or any time mid-sweep),
walks `runs_manifest.json` + per-run `summary.json`, builds a
markdown report with:

- Sorted leaderboard (eval contains, train contains, gap)
- Ablation tables (one per phase, with deltas vs reference)
- Geometry side-table (‖Δm‖ p50, drift, cos(Δ,c))
- Failed-run section (rc ≠ 0)

## Tomorrow's deliverable

`v0-results-2026-05-30.md` will get a new section
"Sweep results (overnight 2026-05-31)" with:

1. **Headline winner** + 5-seed mean±std + gap to baseline.
2. **Per-phase ablation tables** (A-E).
3. **What we learned** (3-5 bullets) — likely candidates:
   - "OPD requires RL anchor; sft+opd alone collapses f1."
   - "RL with N≥4 + temp≥1.5 is necessary for non-zero advantage."
   - "K=32 vs 64 vs 128 — does memory size matter?"
   - "Combine: is xattn really better than hybrid?"
   - "λ_div=0.1 is the right magnitude / off doesn't matter."
4. **Open questions for v1**.

## Budget tracking

| phase | jobs | per-GPU mins | total GPU-min | wall hr (4 GPUs) |
|---|---|---|---|---|
| A | 14 | 45 | 630 | 2.6 |
| B | 12 | 45 | 540 | 2.3 |
| C | 12 | 50 | 600 | 2.5 |
| D | 12 | 45 | 540 | 2.3 |
| E | 9 | 75 | 675 | 2.8 |
| F | 15 | 45 | 675 | 2.8 |
| **subtotal** | **74** | — | **3660** | **~15** |
| G (stretch) | up to 30 | 50 | 1500 | 6 |
| **total budget** | — | — | — | **~21** wall hr |

Generous buffer to 35 hr accommodates: (a) per-job startup overhead
not in the estimate, (b) longer runs on n_items=400, (c) re-runs of
failed jobs, (d) Phase G stretch goals.
