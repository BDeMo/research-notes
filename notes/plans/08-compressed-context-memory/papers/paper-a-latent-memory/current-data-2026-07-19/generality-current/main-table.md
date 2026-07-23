# Paper A current data snapshot

Scores are native metrics ×100; parentheses give completed seed count. `†` means at least one artifact
does not contain the full public validation split. `‡` marks a seed with duplicate full-length technical
replicates; those repeats are averaged within seed and their spread is retained in `snapshot.json`.
This snapshot excludes removed tags, archived
Qwen3.5 backends, smoke runs, and internal Gist replicas.

## Fair-main results available now

| model | task | no-ctx | bounded raw | raw all | SFT@B | SFT@all | window | LL2 | LongLL | LL orig | Compressor (w/o gate) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q3_8b | quality | 17.8 | 7.2 | — | — | — | — | — | — | — | 44.2±11.2 (3) |
| q3_8b | bfcl_live_multiple | 1.3 | 92.1 | — | — | — | — | — | — | — | 71.2±1.1 (3) |
| q3_8b | squad_v2 | — | — | — | — | — | — | — | — | — | — |
| q3_8b | hotpot_qa | — | — | — | — | — | — | — | — | — | — |
| q35_9b | quality | — | — | — | — | — | — | — | — | — | — |
| q35_9b | bfcl_live_multiple | — | — | — | — | — | — | — | — | — | — |
| q35_9b | squad_v2 | — | — | — | — | — | — | — | — | — | — |
| q35_9b | hotpot_qa | — | — | — | — | — | — | — | — | — | — |

## Evidence status

- Canonical manifest: 42 cells.
- Current states: `{"done": 12, "pending": 27, "running": 3}`.
- Canonical artifacts found: 12.
- Duplicate-tag conflicts requiring review: 0.
- Completed cells excluded as invalid: 0.
- Done cells without an artifact on either pod: 0.

Use `snapshot.json` for per-cell provenance and `aggregates.csv` for analysis.
