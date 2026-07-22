# Corrected held-out gate results

> 24 model/task/seed groups · 20 exact-context-disjoint repeated holdouts per group · primary signal fixed
> to compressed-path confidence. Values below average test results over seeds and splits.

| model | task | mem. use | FB rate | FB AUC | Compressor (w/ gate) | Δ raw |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-8B | QuALITY | 99.8% | 0.2% | 57.2 | 54.6 | +47.4 |
| Qwen3-8B | BFCL | 53.4% | 46.6% | 82.8 | 88.5 | -3.5 |
| Qwen3-8B | SQuAD-v2 | 6.6% | 93.4% | 66.6 | 62.8 | -2.0 |
| Qwen3-8B | HotpotQA | 31.7% | 68.3% | 63.9 | 50.9 | -2.4 |
| Qwen3.5-9B | QuALITY | 99.7% | 0.3% | 54.9 | 51.4 | +44.4 |
| Qwen3.5-9B | BFCL | 78.0% | 22.0% | 84.1 | 80.5 | -3.8 |
| Qwen3.5-9B | SQuAD-v2 | 5.8% | 94.2% | 62.4 | 64.7 | -2.1 |
| Qwen3.5-9B | HotpotQA | 47.9% | 52.1% | 67.4 | 51.7 | -2.2 |

Fallback AUROC treats `raw score > memory score` as the positive label and uses negative compressor confidence as
the fallback score. It is computed over the full validation records; thresholded gated-policy values use held-out
calibration/test splits.

## Interpretation

- QuALITY is an easy routing regime because compressed memory is much better than the bounded frozen raw
  path; the empirical gate uses memory on almost every item.
- On BFCL, SQuAD, and HotpotQA, held-out threshold selection does not preserve the strong raw score. The
  routed policy is about 2.0–3.8 points below raw.
- SQuAD obtains only about 6% compressed coverage, so the gate offers little reader-state saving.
- Confidence is therefore an analysis signal, not a safe deployment rule on the strong-raw tasks.

## Formal test

The corrected document-level, fixed-threshold-family test certified **0 of 24** model/task/seed groups at
\(\epsilon=0.02\), family-wise \(\delta=0.10\). Every formally tested gated policy returns all-raw.

Paper wording:

- use **held-out empirical routing**;
- do not use **finite-sample risk-controlled compressor with gate**;
- report the formal all-raw result as a negative finding;
- keep the title and contribution centered on mapping the reliability boundary, not proving a safe gate.

Full machine-readable report: [`gate-current.json`](gate-current.json).
