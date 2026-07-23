# Paper A current data snapshot

> [!WARNING]
> All QuALITY rows below are invalid because this harvest used the stale zero-based-label loader.
> BFCL rows are unaffected. The automatic `invalid` count is therefore not the scientific validity count.

Scores are native metrics ×100; parentheses give completed seed count. `†` means at least one artifact
does not contain the full public validation split. `‡` marks a seed with duplicate full-length technical
replicates; those repeats are averaged within seed and their spread is retained in `snapshot.json`.
This snapshot excludes removed tags, archived
Qwen3.5 backends, smoke runs, and internal Gist replicas.

## Fair-main results available now

| model | task | no-ctx | bounded raw | raw all | SFT@B | SFT@all | window | LL2 | LongLL | LL orig | GCM |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q3_8b | quality | 17.8 | 7.2 | — | — | — | — | — | — | — | 44.2±11.2 (3) |
| q3_8b | bfcl_live_multiple | 1.3 | 92.1 | — | — | — | — | — | — | — | 71.2±1.1 (3) |
| q3_8b | squad_v2 | — | — | — | — | — | — | — | — | — | — |
| q3_8b | hotpot_qa | — | — | — | — | — | — | — | — | — | — |
| q35_9b | quality | 22.0 | 7.1 | — | — | — | — | — | — | — | 46.3±11.4 (3) |
| q35_9b | bfcl_live_multiple | 1.3 | 84.5 | — | — | — | — | — | — | — | 71.4±1.9 (3) |
| q35_9b | squad_v2 | — | — | — | — | — | — | — | — | — | — |
| q35_9b | hotpot_qa | — | — | — | — | — | — | — | — | — | — |

## Evidence status

- Canonical manifest: 42 cells.
- Current states: `{"done": 26, "failed": 8, "pending": 6, "running": 2}`.
- Canonical artifacts found: 26.
- Duplicate-tag conflicts requiring review: 0.
- Completed cells excluded automatically as invalid: 0; 13 completed QuALITY cells are manually quarantined.
- Done cells without an artifact on either pod: 0.

Use `snapshot.json` for per-cell provenance and `aggregates.csv` for analysis.
