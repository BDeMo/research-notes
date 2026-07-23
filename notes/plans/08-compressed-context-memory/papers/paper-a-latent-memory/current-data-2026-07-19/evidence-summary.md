# Paper A evidence summary — current data

> Updated 2026-07-21. Main grid: 88/88 complete. Scores below use full public validation splits. Technical repeats of the
> same tag are averaged within seed, never counted as extra seeds.

## What the completed data supports

### 1. BFCL is a repeatable compression result, but not an accuracy win over raw context

| model | bounded raw | full SFT | LLMLingua-2 | Compressor (w/o gate), 3 seeds |
|---|---:|---:|---:|---:|
| Qwen3-8B | 92.4 | 95.4 ± 0.3 | 70.3 | **72.3 ± 0.5** |
| Qwen3.5-9B | 84.5 | 94.9 ± 1.0 | 60.8 | **72.0 ± 0.8** |

GCM carries a stable amount of tool-use information across the two main models. It is close to or better
than the matched text compressor, especially on Qwen3.5, but remains well below raw context and full SFT.
The valid claim is **stable compressed competence**, not peak accuracy.

### 2. QuALITY is positive on Qwen3-8B but not model-general

| model | bounded raw | SFT@8k | SFT@all input | Compressor (w/o gate) |
|---|---:|---:|---:|---:|
| Qwen3-8B | 7.2 | 64.5 ± 27.5 | 81.7 ± 1.7 | **54.4 ± 0.2** |
| Qwen3.5-9B | 7.1 | 84.7 ± 1.6 | 85.0 ± 0.4 | **51.5 ± 1.7** |

The six-cell full SFT reaudit is complete. Bounded SFT remains unstable: seed 42 collapses to 32.8 while
seeds 43/44 reach 79.4/81.3. True-input SFT is stable at 80.6--83.7. The GCM main-grid seeds are
54.3/54.6/54.2, but independent fixed-config and replicate runs give 44.2±11.2 and 48.7±15.1. The
QuALITY result is therefore positive but run-sensitive, and it does not beat the reaudited SFT mean.
Qwen3.5 SFT is also much stronger than GCM.

### 3. Extractive and multi-hop QA remain negative boundaries

| model | task | bounded raw | full SFT | raw window | LLMLingua-2 | Compressor (w/o gate) |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-8B | SQuAD-v2 | 65.5 | 93.1 ± 0.3 | 49.6 | 53.4 | 26.5 ± 0.4 |
| Qwen3.5-9B | SQuAD-v2 | 66.8 | 93.8 ± 0.3 | 49.6 | 58.8 | 26.9 ± 0.6 |
| Qwen3-8B | HotpotQA | 53.7 | 68.8 ± 0.6 | 26.2 | 22.1 | 28.9 ± 0.2 |
| Qwen3.5-9B | HotpotQA | 53.9 | 71.7 ± 0.6 | 24.8 | 28.9 | 30.5 ± 0.3 |

On SQuAD, GCM is below both raw windows and LLMLingua-2. On HotpotQA, GCM is slightly better than the
matched text/window baselines that have completed, but it remains far below bounded raw context and SFT.
These rows define where continuous memory should not replace raw evidence.

### 4. Shared-backbone fallback recovers much of the compressor gap

The compressor with the gate improves the without-gate score by 8.5--37.8 points on BFCL, SQuAD, and HotpotQA. The
fallback detector is strongest on BFCL (AUROC 82.8/84.1), where it raises the compressor from 72.3→88.5 on Qwen3 and
72.0→80.5 on Qwen3.5. On SQuAD, the routed score approaches raw but fallback is used on about 94% of items,
so little compression benefit remains. The two paths share one backbone; fallback disables the memory-reader
adapter and reads bounded raw context.

The compressor with the empirical gate remains 2.0--3.8 points below raw on the strong-raw tasks. The formal test certifies
0/24 groups and uses all-raw.

## Baseline readout

- **LLMLingua-2:** strongest matched text baseline on SQuAD and competitive on BFCL; weaker on HotpotQA.
- **LongLLMLingua / original LLMLingua:** current rows are excluded. The installed environment is
  incompatible with the official compressor's legacy KV-cache API; many items silently used the fallback
  text path. These methods require an isolated official-version rerun.
- **Raw window:** strong on SQuAD, moderate on HotpotQA, weak on QuALITY.
- **Official latent-memory baselines:** no paper-grade scores yet. Repository audits are complete, but native
  environments/checkpoints and task-compatible wrappers remain pending.

## Evidence that is ready now

- Qwen3-8B GCM: all four tasks, three seeds, full validation.
- Qwen3.5 GCM: BFCL, SQuAD, and HotpotQA, three seeds, full validation.
- Qwen3-8B SFT: all four tasks, three seeds; QuALITY uses the auditable six-cell reaudit.
- Qwen3.5 SFT: all four tasks, three seeds.
- True raw QuALITY: both main bases.
- One-seed LLMLingua-2 and window controls on all rows.

## Negative findings and evidence still pending

- LongLLMLingua and original LLMLingua: 10 completed cells invalidated by compressor exceptions and silent
  fallback; rerun under the official pinned environment with fallback disabled.
- Held-out routing is complete. On BFCL/SQuAD/Hotpot it remains 2.0–3.8 points below bounded raw; on
  QuALITY it uses memory on almost every item because memory is much stronger than frozen raw.
- The corrected formal test certified 0/24 groups at \(\epsilon=.02,\delta=.10\); every formally tested gated policy is
  all-raw. Use **held-out empirical routing** only and report risk control as a negative result.
- Source-adapter and long-context stages are underway. Replicate and feature reruns are complete.
  Seven-model, budget, ablation, cost, and official-baseline stages remain pending.

## Technical-replicate audit

Eight tags were executed on both pods with different scores. They are treated as technical repeats within
one seed and averaged before the three-seed mean. They are not extra samples. The largest observed same-tag
spread is 0.95 percentage points. This spread is small for most QA/BFCL rows but must remain visible because
the earlier QuALITY pipeline showed much larger nondeterminism.

## Files

- [`main-table.md`](main-table.md): current headline table.
- [`aggregates.csv`](aggregates.csv): one row per model/task/method/variant.
- [`snapshot.json`](snapshot.json): per-cell provenance, technical repeats, conflicts, and missing cells.
