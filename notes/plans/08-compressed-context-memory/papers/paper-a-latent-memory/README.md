# Paper A — When to Trust Soft Memory

**Status:** 🟢 **active writing** (draft v0.1; fair v1.8 rerun in progress).

**Working title:** *When to Trust Soft Memory: Mapping the Reliability Boundary of Long-Context Compression in Frozen Language Models.*
The name collides with G-MemLLM/CCM terminology; “risk-controlled” is retained only if LTT certifies nonzero coverage.

**Draft:** [`PAPER-A-draft-v0.1.md`](PAPER-A-draft-v0.1.md)
**Literature review:** [`literature-review-2026-07-16.md`](literature-review-2026-07-16.md)
**Experiment receipt:** [`experiment-receipt-2026-07-16.md`](experiment-receipt-2026-07-16.md)
**Qwen3.5 run repair:** [`qwen35-run-repair-2026-07-17.md`](qwen35-run-repair-2026-07-17.md)
**Current data:** [`current-data-2026-07-19/evidence-summary.md`](current-data-2026-07-19/evidence-summary.md)
**Final baseline plan:** [`paper-baselines-v2-2026-07-19.md`](paper-baselines-v2-2026-07-19.md)
**Training/resource runbook:** [`paper-a-training-and-resource-checklist-2026-07-19.md`](paper-a-training-and-resource-checklist-2026-07-19.md)
**Output/truncation audit:** [`integrity-audit-2026-07-19/README.md`](integrity-audit-2026-07-19/README.md)
**Counts/durations:** [`paper-a-duration-matrix-2026-07-20.md`](paper-a-duration-matrix-2026-07-20.md)
**Terminology/style:** [`paper-terminology-style-guide.md`](paper-terminology-style-guide.md)

## Thesis (one line)
A frozen LLM can encode context into a short recurrent latent memory, but that memory is selectively lossy;
use a held-out risk-controlled gate to trust it only when safe and otherwise fall back to a feasible raw path.

## Core claims
- **Compressible-regime signal:** BFCL compressed accuracy is approximately 0.65–0.75 across seven bases.
- **Held-out boundary:** routing is within 0.3–2.3 points of a strong raw path and rescues a degraded
  Qwen3.5-4B path by 52.3 points in the existing pilot.
- **Capacity wall:** exact/extractive detail remains a hard boundary; raw context and full-cost SFT dominate
  on several QA tasks.
- **Cost/risk framing:** the paper reports risk–coverage and measured cost rather than an in-sample
  “gated ≥ full” construction.

## Honest novelty / required baselines
We do not claim the first latent compressor or the first confidence fallback. The contribution is the
decision-quality treatment of a learned latent memory on a frozen shared base, with fair adaptation controls,
held-out routing, cost/risk curves, and an explicit competence boundary.

Required controls: true raw, bounded raw, full+SFT-LoRA, faithful Gisting, LLMLingua-2, raw window, and RULER.

## Links
- evidence audit: [`../paper-b-forgetting-gating/results-v1.8.0/paper-a-audit/`](../paper-b-forgetting-gating/results-v1.8.0/paper-a-audit/)
- scope + lit: [`../../summary/2026-06-05/two-paper-litreview-2026-06-05.md`](../../summary/2026-06-05/two-paper-litreview-2026-06-05.md) (§Paper A)
- related work: [`../../v2-related-work.md`](../../history/v2-related-work.md) · [`../../v0-opd-on-latent-memory-survey-2026-06-01.md`](../../history/v0-opd-on-latent-memory-survey-2026-06-01.md)
- evidence: `mem-test/mem-embedding/summary/matrix.md` §7 (ceiling), §8 (boundary + mix-train)
