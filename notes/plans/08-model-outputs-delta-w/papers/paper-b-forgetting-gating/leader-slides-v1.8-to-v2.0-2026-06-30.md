# CMG-RCA / Long-Context GCM Update - Leader Slides

Audience: leaders / VP review  
Goal: summarize what we did, what we learned, and what decision is needed for the next phase.

---

## Slide 1 - Executive Message

**Title:** From CMG-RCA demo to long-context foundation-model research

**Message:**
- We packaged a deployable CMG-RCA foundation-model demo.
- We validated that long RCA context is a real deployment problem, not just a modeling preference.
- We learned that fixed long-context reduction methods fail differently by task, length, and model architecture.
- The next paper should focus on **adaptive knowledge extraction + reliability-gated fallback**.

**Decision needed:**
- Do we invest in the next phase: larger compute, more data, and public long-context paper experiments?

---

## Slide 2 - What We Built This Cycle

**CMG-RCA foundation repo**
- Repo: `https://gitlabe2.ext.net.nokia.com/s1shi/cmg-rca-foundation`
- Demo/deploy: `deploy.sh`, `server.py`, `static/index.html`
- Model/code: `gcm/`, `adapters/mt_all_distill1_adapters.pt`
- Data/results: `data/cmg_rca_cases.jsonl`, `overall.json`, `SAMPLES.md`, `README.md`
- Generated labels: `data/generated_cmg_rca_analysis/`

**Outcome:**
- The work is now reviewable and runnable, not only a set of experiment scripts.
- The demo shows how CMG-RCA can use extracted knowledge instead of raw full context.

---

## Slide 3 - Why This Problem Matters

**CMG-RCA is long-context RCA.**
- A CMG-RCA case can include logs, telemetry, traces, tech-support files, previous reports, and retrieved evidence.
- Full raw evidence can be close to or beyond the practical serving window.

**Deployment issue:**
- Full-context inference can hit GPU memory limits.
- Truncation loses evidence.
- No-context has little CMG-specific prior.

**Resulting problem:**
- We need a deployable way to extract useful RCA knowledge from long evidence.

---

## Slide 4 - v1.8.0: What We Learned

**Current method components**
- Knowledge extractor: turns long evidence into a compact model-readable representation.
- Gate: reliability signal that decides whether to trust extracted knowledge or fall back.

**Internal CMG-RCA result**

| Setting | CMG module-ID accuracy |
|---|---:|
| No context | 0.079 |
| Truncated full context | 0.105 |
| Extracted knowledge | 0.263 |

**Interpretation:**
- Absolute accuracy is early-stage.
- The important result is the ordering:
  `extracted knowledge > truncated full context > no context`.
- This shows the extracted-knowledge path is useful under a real memory constraint.

---

## Slide 5 - v1.8.0 Outcome and Caveat

**Outcome:**
- We have a deployable CMG-RCA foundation-model artifact.
- We have an early positive signal that knowledge extraction beats truncation.
- We have a concrete demo and generated analysis labels for handoff.

**Caveats:**
- Internal CMG is used as demo / motivation, not as the main paper benchmark.
- Some early CMG numbers may be optimistic if trained before module-name masking.
- External generalization is not solved yet.
- The current result is small-scale and mostly single-turn.

---

## Slide 6 - Public Long-Context Baseline Fact-Base

To prepare the next paper, we shifted from internal CMG numbers to **public benchmark facts**.

**Question:**
Which long-context coping method works, when, and why?

**Baseline families tested:**
- Window / truncation
- KV eviction
- Prompt compression
- RAG / retrieval
- Full-context and no-context references

**Key change:**
- The next paper should be grounded in public benchmarks, with CMG as the motivation/demo.

---

## Slide 7 - Key Long-Context Facts

**No single baseline wins everywhere.**

Key findings from the fact-base:
- Attention-based KV eviction can collapse past 8k-16k.
- Attention-free or reconstruction-style eviction is more length-robust on retrieval.
- Ranking flips by task type: retrieval, extractive QA, global reasoning, and literary MC behave differently.
- Full context can hurt when distractors dominate.
- RAG is strong when lexical overlap exists, but weak for abstractive/global tasks.
- Linear-attention / GDN-style models do not expose the KV lever at all.

**Implication:**
- A fixed context-reduction method is not enough.

---

## Slide 8 - v2.0.0 Paper Direction

**Proposed high-level thesis:**

Long-context RCA and agentic reasoning need **adaptive knowledge extraction**, not one fixed compression or retrieval rule.

**Why:**
- Full context can be too expensive or misleading.
- Truncation only works when the answer is local.
- RAG only works when lexical overlap is available.
- KV eviction is model-architecture dependent and can fail at long length.

**Next method framing:**
- Extract useful knowledge from long evidence.
- Verify whether that extracted knowledge is reliable.
- Fall back when it is not reliable.

---

## Slide 9 - v2.0.0 Ideas

**Idea 1: Self-verified extracted memory**
- Train the memory to preserve evidence.
- Use the same preservation/reconstruction signal as the reliability check.

**Idea 2: Reliability-gated fallback**
- If extracted knowledge is reliable: use it.
- If unreliable: fall back to raw context, retrieval, or a higher-cost path.

**Idea 3: Adaptive budget**
- Easy contexts use fewer memory tokens.
- Dense or ambiguous contexts get more budget.

**Idea 4: Public benchmark story**
- Use public long-context tasks for paper claims.
- Keep CMG as the motivating internal deployment demo.

---

## Slide 10 - Current Resource Bottlenecks

**Compute**
- Current sub-10B experiments already require multiple GPUs.
- Practical training/evaluation is around 4 GPUs.
- Larger models / longer windows likely need 8-32 H100s or more.

**Data**
- Current curated CMG distillation data is about 3K examples:
  - `cmg_distill`: 1,860 train + 1,178 test
  - location: `sam-dev-d1525-gpu4:/mnt/persist/datasets/cmg_distill/`
- Built with Opus distillation plus human filtering.
- Useful, but expensive to scale.

**Open question**
- Will real RCA reports plus distillation/human filtering remain effective at scale?

---

## Slide 11 - What We Need Leaders to Decide

**Option A: Demo / applied path**
- Focus on CMG-RCA deployability.
- Improve the demo and internal data pipeline.
- Goal: usable internal RCA assistant prototype.

**Option B: Paper / research path**
- Focus on public long-context benchmark story.
- Build v2.0.0 around adaptive knowledge extraction and reliability-gated fallback.
- Goal: next paper with strong public benchmark evidence.

**Option C: Both, with scoped resources**
- Keep CMG as applied demo.
- Use public benchmarks for paper claims.
- Requires more GPU budget and data-generation budget.

---

## Slide 12 - Recommended Next Step

**Recommendation: Option C, but with clear scope.**

Next 2-4 weeks:
- Finish CMG gating result for the demo.
- Keep CMG-RCA foundation repo updated as applied deliverable.
- Complete public long-context fact-base and figures.
- Define v2.0.0 method around self-verified extracted memory.
- Run the minimum experiments needed to decide whether the next paper is viable.

**Decision ask:**
- Approve GPU budget for public long-context sweeps and method experiments.
- Approve data budget / plan for scaling RCA supervision beyond current 3K curated examples.
