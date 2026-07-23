# Paper A paper-grade experiment protocol (2026-07-15)

> Status: running. This protocol supersedes the in-sample gate and ambiguous `full` terminology in the original
> v1.8 table. Audit: [`README.md`](README.md). Runner:
> `/Users/s1shi/workspace/gcm/experiments/paper_a_grid.py`.

## Claim under test

At a fixed read/KV budget, recurrent GCM scans the available input and condenses it into K memory tokens.
A pre-registered confidence gate may use that compressed path when calibrated safe and otherwise use a
feasible bounded raw path.

This protocol does **not** assume that GCM beats true full context, that gating is universally harmless, or
that QuALITY normally exceeds an 8k context.

## Main grid

### Models and seeds

- Primary: Qwen3-8B and Qwen3.5-9B.
- Independent training seeds: 42, 43, 44.
- Fixed method configuration: **K=128 per 4,096-token chunk** (final memory length = S×K), read-LoRA rank 64, half-depth encoder, hard normalization,
  joint task loss, distillation 0.5, reconstruction 0.5, recurrent 4,096-token chunks.
- Exact train/validation duplicate removal is enabled.

### Tasks

- QuALITY: full train=1,899; validation=1,595; MC letter log-likelihood.
- BFCL-live-multiple: train=737 minus one duplicate; validation=316; tool accuracy.
- SQuAD-v2: first 2,000 train items; full validation=5,928; native token-F1.
- HotpotQA: first 2,000 train items; full validation=7,405; native token-F1.

The two 2,000-item training caps are disclosed and identical across trainable methods.

### Compared paths

1. `no_ctx`: frozen query-only reference.
2. `feasible_raw`: frozen raw path at 8,192 tokens for QuALITY and 4,096 otherwise.
3. `raw_full`: true raw QuALITY path at 16,384 cap; all audited QuALITY items fit.
4. `full+SFT`: task-trained LoRA over the raw path, matching training data, rank, steps, and seeds.
5. `faithful_gist`: LoRA-trained LM under the gist attention mask with K=128.
6. `LLMLingua-2@matched-memory`: exact public compressor; 256 kept tokens on QuALITY (median 2 chunks), 128 otherwise.
7. `window@matched-memory`: the corresponding raw-token budget control.
8. `GCM@128/chunk`: proposed compressed path; QuALITY usually yields about 256 final memory tokens.
9. `GCM-gate`: GCM or `feasible_raw`, with a threshold calibrated without test labels.

Official compression-baseline stage: authors' released implementations/checkpoints for AutoCompressor, ICAE,
CCM, Activation Beacon, Cartridges, Gist, xRAG, LLMLingua-2/LongLLMLingua, 500xCompressor, and mean pooling.
If an official method does not support Qwen, it is evaluated on its released base and normalized to that
base's own no-context/raw references. Internal `baselines2025.py` replicas are engineering smoke only and
are excluded from paper claims. See
[`../../../paper-a-latent-memory/baseline-selection-2026-07-16.md`](../../../paper-a-latent-memory/baseline-selection-2026-07-16.md).

The main QuALITY GCM configuration always uses `ENC_MAXCTX=16,384`, so it sees every audited article.
An early manifest accidentally emitted both `default` and `truefull` tags with this identical configuration;
the `truefull`-tagged GCM cells are technical replicates, not a separate method, and are excluded from the main table.

## Gate protocol

- Primary signal is fixed before evaluation: compressed-path `conf`.
- TARG and margin are baselines; neither may replace `conf` after seeing test results. Direct follow-up
  baselines are a Belikova-style joint query–memory overflow probe, PoC-style performance predictor,
  latent-memory first-token entropy, and a Self-Route-style sufficiency judgment.
- Twenty repeated document-disjoint 25/75 calibration/test splits.
- Empirical baseline: maximize coverage subject to calibration-set signed policy excess ≤2 points.
- Formal route: Bonferroni Learn-then-Test over a fixed threshold family, controlling accepted-set positive
  compression harm at \(\epsilon=0.02\), family-wise \(\delta=0.10\). No certified threshold means all-raw.
- Test readout: native score, signed policy excess, accepted-set positive harm, accepted harm/benefit rates,
  four raw/memory outcome types, fire/fallback, random-at-matched-coverage, always-full, always-compress,
  and oracle.
- A risk-control claim requires nonzero LTT-certified coverage; practical headline status additionally
  requires mean coverage ≥20%.

## Additional grids

- Real long-context transfer (all seven bases, one fixed run configuration):
  - QuALITY → LongBench-v2 and ∞Bench-choice;
  - SQuAD → LongBench MultiFieldQA and Qasper;
  - HotpotQA → LongBench HotpotQA, 2WikiMQA, and MuSiQue;
  - NarrativeQA → LongBench NarrativeQA;
  - HotpotQA → BABILong QA1–QA3 at 16k;
  - all seven bases run source-trained GCM plus no-context and feasible raw;
  - the two primary bases (Qwen3-8B, Qwen3.5-9B) additionally run source-trained full+SFT on every target;
  - ∞Bench additionally runs K=32/chunk to bound the length-adaptive final memory.
- Fixed-config generality: all seven bases on QuALITY and BFCL, three runs, no per-model K tuning.
- Budget curve: QuALITY and RULER-NIAH; GCM K∈{64,128,256,512};
  raw window∈{256,512,1024,2048,4096,8192}.
- Mechanism: joint-task off, distillation off, reconstruction off, recurrence off,
  K=64, K=256; Qwen3-8B × {QuALITY, BFCL}, three seeds.
- Cost: same H100, separating encoder scan, compressed read, feasible raw read, fallback,
  CUDA latency, incremental peak memory, and actual tokens.

## Kill/reframe rules

- If true `raw_full` or matched `full+SFT` removes the QuALITY accuracy advantage, report GCM only on
  a cost-quality Pareto; do not claim accuracy superiority.
- If no nonzero-coverage threshold receives LTT certification, remove the risk-control claim; if empirical
  coverage is below 20%, demote the gate to an analysis tool.
- If one fixed K fails on more than two of eight model families, replace “model-agnostic” with
  “base-conditioned.”
- If GCM fails RULER while raw selection succeeds, report the exact-retrieval boundary prominently.

## Artifact locations

- Pod root: `/mnt/persist/paper_a/`.
- Per-cell logs/status/artifacts: `logs/`, `status/`, `artifacts/`.
- Gate analysis: `/Users/s1shi/workspace/gcm/experiments/paper_a_gate_analysis.py`.
- Cost profiler: `/Users/s1shi/workspace/gcm/experiments/paper_a_cost.py`.
- Harvester: `/Users/s1shi/workspace/gcm/experiments/paper_a_harvest.py`.
