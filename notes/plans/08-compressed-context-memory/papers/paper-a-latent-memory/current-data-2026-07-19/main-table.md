# Paper A current data snapshot

Scores are native metrics ×100; parentheses give completed seed count. `†` means at least one artifact
does not contain the full public validation split. `‡` marks a seed with duplicate full-length technical
replicates; those repeats are averaged within seed and their spread is retained in `snapshot.json`.
`§` marks the six-cell full SFT reaudit with saved outputs.
This snapshot excludes removed tags, archived
Qwen3.5 backends, smoke runs, and internal Gist replicas.

## Panel A — Compressor quality

GCM and the matched-state controls use a short context state during answering. The SFT column is a
full-cost adaptation reference: it reads raw context and is not a same-cost compression competitor.
In the paper table, raw and SFT references share the same gray column shading.

| model | task | raw | SFT | window | LLMLingua | Compressor (w/o gate) |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-8B | QuALITY | 7.2 | 81.7±1.7§ | 15.7 | 14.3 | **54.4±0.2** |
| Qwen3-8B | BFCL | 92.4 | 95.4±0.3‡ | 55.7 | 70.3 | **72.3±0.5** |
| Qwen3-8B | SQuAD-v2 | 65.5 | 93.1±0.3‡ | **49.6** | **53.4** | 26.5±0.4‡ |
| Qwen3-8B | HotpotQA | 53.7 | 68.8±0.6‡ | 26.2 | 22.1 | **28.9±0.2‡** |
| Qwen3.5-9B | QuALITY | 7.1 | 85.0±0.4‡ | 16.7 | 20.3 | **51.5±1.7** |
| Qwen3.5-9B | BFCL | 84.5 | 94.9±1.0‡ | 52.8 | 60.8 | **72.0±0.8** |
| Qwen3.5-9B | SQuAD-v2 | 66.8 | 93.8±0.3 | 49.6 | **58.8** | 26.9±0.6‡ |
| Qwen3.5-9B | HotpotQA | 53.9 | 71.7±0.6 | 24.8 | 28.9 | **30.5±0.3** |

## Panel B — GCM compressor with and without the gate

The memory and raw paths use the same pretrained model. The memory path turns on the read adapter and reads
GCM vectors; the fallback path turns the adapter off and reads bounded raw tokens. No second model copy is
needed.

| model | task | Compressor (w/o gate) | Compressor (w/ gate) | gain | FB AUC | FB rate | Δ raw |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | QuALITY | 54.4 | **54.6** | +0.2 | 57.2 | 0.2% | +47.4 |
| Qwen3-8B | BFCL | 72.3 | **88.5** | +16.2 | 82.8 | 46.6% | -3.5 |
| Qwen3-8B | SQuAD-v2 | 26.5 | **62.8** | +36.3 | 66.6 | 93.4% | -2.0 |
| Qwen3-8B | HotpotQA | 28.9 | **50.9** | +22.0 | 63.9 | 68.3% | -2.4 |
| Qwen3.5-9B | QuALITY | **51.5** | 51.4 | -0.1 | 54.9 | 0.3% | +44.4 |
| Qwen3.5-9B | BFCL | 72.0 | **80.5** | +8.5 | 84.1 | 22.0% | -3.8 |
| Qwen3.5-9B | SQuAD-v2 | 26.9 | **64.7** | +37.8 | 62.4 | 94.2% | -2.1 |
| Qwen3.5-9B | HotpotQA | 30.5 | **51.7** | +21.2 | 67.4 | 52.1% | -2.2 |

Fallback AUROC uses low memory confidence to rank examples where bounded raw scores higher than the compressor without the gate.
Fallback rate is measured on held-out empirical routing. The formal test certifies 0/24 groups and therefore
uses 100% fallback.

## Evidence status

- Canonical manifest: 88 cells.
- Current states: `{"done": 88, "pending": 0}`.
- Canonical artifacts found: 88.
- Duplicate-tag conflicts requiring review: 8.
- Completed cells excluded as invalid: 10.
- Done cells without an artifact on either pod: 0.

Use `snapshot.json` for per-cell provenance and `aggregates.csv` for analysis.
