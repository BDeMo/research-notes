# Weekly Human Update

> Weekly high-level progress note for human readers. Latest week first.
> Keep this file concise: direction, progress, good news, lessons, next steps.

---

## Week of 2026-06-01

### TL;DR

This week clarified the main research direction around **controlled model
adaptation**: improve RCA / long-context capability while preserving the frozen
base model's general ability. Plan 08 now has a clear v1 characterization and a
natural v1.5 do-no-harm extension; Plan 09 / Janus opened a broader path toward
intrinsic-site protection.

### Main Progress

- **Plan 08 / memory wrapper.** The `mem-embedding` experiments now support a
  clear story: the learned soft-prompt wrapper is useful as a lossy context
  compressor when gist-level information is enough, but it should not be used as
  a universal replacement for long context. Setting/provenance:
  [`P08-S2`](../notes/plans/08-model-outputs-delta-w/settings/settings.md#p08-s2--v1-phase-y-three-regime-benchmark-cells).
- **v1.5 do-no-harm gating.** The next step is no longer vague. The wrapper
  should activate when it is likely to help and stay closed when the frozen base
  or full context is safer. Multi-seed probing found practical signals:
  confidence / sequence log probability as positive indicators, and
  wrapper-to-base divergence as a warning sign. Setting/provenance:
  [`P08-S3`](../notes/plans/08-model-outputs-delta-w/settings/settings.md#p08-s3--v15-intrinsic-signal-probe).
- **Plan 09 / Janus.** A new higher-upside hypothesis is taking shape:
  long-context degradation and fine-tuning forgetting may be coupled through
  intrinsic transformer sites. Early cross-model analysis found positive
  coupling between long-context read-side metrics and fine-tuning disruption
  metrics.

### Good News

- The memory-wrapper line produced a positive paper framing: characterize where
  frozen-base learned memory works, then improve it with a gate.
- The gating direction is evidence-driven rather than purely architectural.
- Janus gives a broader method target: detect important sites before SFT and
  protect them during adaptation, instead of reacting to forgetting afterward.
- The new GPU setup is already supporting larger loops: multi-seed probing and
  cross-model checks.

### Lessons

- Single-seed observations are useful for discovery, but paper claims should
  wait for multi-seed checks.
- A soft-prompt memory wrapper should be framed as a controlled compression
  module, not a universal long-context solution.
- For RCA model training, the important objective is controlled adaptation:
  improve RCA / long-context behavior without erasing the base model's original
  capabilities.
- Attention sinks alone are not the right protection target; retrieval /
  attention-distance / previous-token behavior looks more relevant.

### Next Steps

- Implement the first do-no-harm gated wrapper using confidence and divergence
  signals.
- Run the Janus causal test: protect long-context-coupled heads during SFT and
  measure both forgetting reduction and long-context retention.
- Keep Plan 08 and Plan 09 synced with only the strongest multi-seed /
  cross-model evidence.

### Pointers

- Plan 08: [`notes/plans/08-model-outputs-delta-w/`](../notes/plans/08-model-outputs-delta-w/)
- Plan 08 settings registry: [`settings.md`](../notes/plans/08-model-outputs-delta-w/settings/settings.md)
- Plan 09 / Janus: [`notes/plans/09-intrinsic-site-protection/`](../notes/plans/09-intrinsic-site-protection/)
