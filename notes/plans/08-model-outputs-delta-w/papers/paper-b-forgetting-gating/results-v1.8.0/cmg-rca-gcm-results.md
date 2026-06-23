# cmg-rca-gcm — main results (module-ID via GCM compression)

**Task.** Identify the root-cause module of a Nokia CMG fault from telemetry, via a GCM compressor that
encodes the (masked) evidence into a 256-token memory `M`. Eval = held-out `cmg_rca` multiple-choice
(12-case validation split; module names masked in the evidence). Base = Qwen3-8B, K=256, recurrent prefix
compression, read-LoRA. Metric = module-ID accuracy from three sources: **no-context** / **truncated raw
context (8k cap)** / **compressed M (full evidence, no truncation)**.

## Main table (module-ID accuracy, 12-case cmg_rca val)

| cell | train data | K / LoRA / extra | no-ctx | truncated(8k) | **compress M** |
|------|------------|------------------|:------:|:-------------:|:--------------:|
| **mt_all_distill1** | distill+aug+rca | K256 L64 distill=1.0 | 0.08 | 0.33 | **0.42** |
| **mc_augrca_lg** | aug+rca | K256 L64, 3.5k steps | 0.17 | 0.42 | **0.42** |
| mc_aug | aug | K256 L64 | 0.00 | 0.25 | 0.33 |
| mc_aug_k128 | aug | K128 L64 | 0.08 | 0.33 | 0.33 |
| mc_aug_long | aug | K256 L64, 3.5k | 0.08 | 0.42 | 0.33 |
| mc_allthree | distill+aug+rca | K256 L64 | 0.08 | 0.25 | 0.33 |
| mt_all_lora128 | distill+aug+rca | K256 L128 | 0.08 | 0.25 | 0.33 |
| mt_all_norec | distill+aug+rca | K256 recon=0 | 0.25 | 0.25 | 0.25 |
| mt_all_long2 | distill+aug+rca | K256, 4.5k | 0.17 | 0.33 | 0.25 |
| mc_all_final | distill+aug+rca | K256, 5k | 0.00 | 0.33 | 0.25 |
| mc_augrca | aug+rca | K256 L64 | 0.08 | 0.25 | 0.17 |
| mc_distillaug | distill+aug | K256 L64 | 0.17 | 0.33 | 0.17 |
| mt_all_k128 | distill+aug+rca | K128 L64 | 0.17 | 0.33 | 0.17 |
| **mt_distillonly** | **distill only** | K256 L64 | 0.08 | 0.33 | **0.00** |

## Findings
1. **Compression carries the module signal.** Best `compress` ≈ **0.42**, vs **no-context ≈ 0.08** (≈5×).
   The compressor encodes enough of the masked telemetry into 256 tokens to identify the module.
2. **Module-classification training data is required.** `mt_distillonly` (report-generation data only) →
   `compress = 0.00`: the read path collapses to a constant module on MC. Adding `cmg_aug` / `rca_openrca`
   (balanced module classification) is what makes `M` discriminative. Best = all three + `distill=1.0`.
3. **Compress ≈ truncated, at ~25× fewer tokens.** `compress` matches or beats the truncated-8k raw-context
   baseline while using a 256-token memory vs up to 8,192 raw tokens — and it ingests the *full* evidence
   (no truncation) via recurrent compression.
4. **Generation ≠ classification.** Free-form RCA *report generation* mode-collapses to a constant module
   (~0.21 = majority share), invariant to K/recon/contrast/LoRA/steps/data-mix (see prior sweep). Reliable
   module-ID comes from the MC scoring of `M`, not the generated prose. Reports are coherent + collapse-free
   (rep_penalty 1.15 + no_repeat_ngram 3) but partly templated.

## Caveats
- 12-case val split → each case ≈ 0.083; differences of 1–2 cases are within noise. The robust signal is the
  *ordering* (compress ≫ no-ctx; all-data+classification ≫ distill-only) not exact points.
- "truncated(8k)" uses the **frozen base** on raw evidence (no adapter); "compress" uses the trained read-LoRA.
  A strictly-fair `full+adapter` / untruncated-full baseline is the next run (`MAXCTX=16384`).

## Demo
Live interactive site `cmg-rca-gcm` (model mounted on the d1525 pod): per-case compression, clickable
per-source live generation, context-window swimlane, and a chatbot over the compressed memory.
Best demo adapter: `mc_allthree` (0.33); `mt_all_distill1` (0.42) is the stronger candidate to swap in.
