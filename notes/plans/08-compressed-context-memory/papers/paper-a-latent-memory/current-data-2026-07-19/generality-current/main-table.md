# Paper A current data snapshot

> [!WARNING]
> QuALITY generality rows from the old harvest remain invalid and are
> excluded from the paper table. This snapshot reports the unaffected BFCL
> generality evidence only.

Scores are native metrics ×100; parentheses give completed seed count. `†` means at least one artifact
does not contain the full public validation split. `‡` marks a seed with duplicate full-length technical
replicates; those repeats are averaged within seed and their spread is retained in `snapshot.json`.
This snapshot excludes removed tags, archived
Qwen3.5 backends, smoke runs, and internal Gist replicas.

## Fixed K=128 BFCL generality

| model | bounded raw | Compressor (w/o gate) | completed runs |
|---|---:|---:|---:|
| Qwen3-8B | 92.1 | 71.2±1.1 | 3 |
| Qwen3.5-9B | 84.5 | 71.4±1.9 | 3 |
| Qwen3.5-4B | 11.1 | 71.5±1.7 | 3 |
| GLM-4-9B | 84.8 | 72.9±0.7 | 2 |
| Ministral-8B | 87.0 | 72.0±0.7 | 2 |
| xLAM-8B | 88.6 | 73.1±0.9 | 2 |
| ToolACE-8B | 88.9 | 72.2±0.4 | 2 |

## Evidence status

- Paper-critical manifest: 21 BFCL cells.
- Current states: `{"done": 17, "technical_failure": 4}`.
- Canonical artifacts found: 17.
- Duplicate-tag conflicts requiring review: 0.
- QuALITY generality is outside the current paper table.
- Done cells without an artifact on either pod: 0.

Use `snapshot.json` for per-cell provenance and `aggregates.csv` for analysis.
