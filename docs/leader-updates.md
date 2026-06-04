# Leader Updates

> Short daily progress notes for leader reporting. Latest entry first.
> Keep this file high level: progress, good news, lessons, and next steps.

---

## 2026-06-04

### TL;DR

Today clarified the main research direction around **controlled model adaptation**:
improve RCA / long-context capability while preserving the frozen base model's
general ability. The memory-wrapper line now has a clear v1 result and a natural
v1.5 extension; the new Janus line produced early evidence for a broader
intrinsic-site protection method.

### Key Progress

- **Memory wrapper: from result boundary to method direction.** The
  `mem-embedding` experiments now support a clear characterization: the learned
  soft-prompt wrapper is useful as a lossy context compressor when the answer can
  be recovered from gist-level information, but it should not be trusted for
  exact retrieval or detail-preserving tasks. This is a stronger story than a
  generic success/failure result because it identifies the useful regime.
- **v1.5 do-no-harm gating is now well motivated.** Since the base model is
  frozen, the wrapper should help when it can and stay out of the way otherwise.
  Multi-seed signal analysis found practical gate inputs: wrapper confidence /
  sequence log probability can indicate likely correctness, while large
  wrapper-to-base divergence is a warning sign that the wrapper may hurt.
- **Janus opened a higher-upside research path.** Instead of treating
  long-context degradation and fine-tuning forgetting as separate issues, Janus
  tests whether they are connected through the same intrinsic transformer sites.
  Early cross-model analysis found positive coupling between long-context
  read-side metrics and fine-tuning disruption metrics, giving a concrete target
  for a protection-based method.

### Good News

- We now have a positive paper framing for the memory-wrapper work: characterize
  where frozen-base learned memory works, then improve it with a do-no-harm gate.
- The v1.5 gating idea is evidence-driven, not just architectural speculation.
- The Janus direction has a clear novelty angle: detect important model sites
  before SFT and protect them during adaptation, rather than reacting to
  forgetting after it happens.
- The new GPU environment is already enabling larger research loops, including
  multi-seed probing and cross-model checks.

### Lessons

- Single-seed signals are useful for discovery but should not become headline
  claims. Multi-seed checks already corrected several attractive but unstable
  observations.
- A soft-prompt memory wrapper should be framed as a controlled compression
  module, not a universal replacement for long context.
- For future RCA model training, the goal should be controlled adaptation:
  improve RCA / long-context behavior without erasing the base model's original
  capabilities.
- Attention sinks alone are not the right target for long-context protection;
  retrieval / attention-distance / previous-token behavior looks more relevant.

### Next Steps

- Implement the first do-no-harm gated wrapper using confidence and divergence
  signals.
- Run the Janus causal test: protect long-context-coupled heads during SFT and
  measure both forgetting reduction and long-context retention.
- Keep Plan 08 and Plan 09 documentation synced with only the strongest
  multi-seed / cross-model evidence.

### Pointers

- Plan 08: [`notes/plans/08-model-outputs-delta-w/`](../notes/plans/08-model-outputs-delta-w/)
- Plan 09 / Janus: [`notes/plans/09-intrinsic-site-protection/`](../notes/plans/09-intrinsic-site-protection/)
