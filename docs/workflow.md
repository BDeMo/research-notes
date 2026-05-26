# Research Workflow

> Methodology and operating principles for this notebook.
> Distilled from how we actually work, not aspirational.
>
> **Before reading this file, read [`../memory/README.md`](../memory/README.md) first.** That folder holds the standing rules; this file holds the *how*.

---

## 0. Operating principles

1. **Frame first, brainstorm second.** A bad framing produces 50 shallow ideas. A good framing produces 5 sharp ones. Always spend the first 10 minutes finding the right axes before generating ideas.
2. **Rank by `novelty × feasibility × impact`.** Cheap, but disciplined. If an idea scores low on any of the three, demote it. If it scores high on all three, write the plan now.
3. **Adjacent papers are the cheapest signal.** Before extending an idea, read 1–2 papers that occupy the closest neighborhood. Look for their *limitations* section — that's where your idea lives.
4. **Plans must be falsifiable.** Every plan needs (a) a metric, (b) a baseline, (c) a budget. If you can't write these in three sentences, the plan is not ready.
5. **Track the source of every idea.** Every idea has a parent: a paper, a conversation, a prior project. Record it. This becomes the *knowledge mother nest* (`docs/matrix/knowledge-sources.md`).

---

## 1. Brainstorm workflow

A four-phase loop. Each phase has an artifact.

```
[Framing]  →  [Brainstorm]  →  [Read]  →  [Plan]  ─┐
   ↑                                                │
   └────────────  reframe if blocked  ──────────────┘
```

### Phase A — Framing
- Goal: turn a vague question ("inference-time training") into a *bilevel* or *trade-off* structure with named axes.
- Tools: variable substitution ($X$, $W$, supervision, parameterization, …), 2×2 matrices, exchange-rate framings, information-theoretic budgets.
- Artifact: `notes/<topic>/framing.md` or a "framing" section at the top of the idea list.

### Phase B — Structured brainstorm
- Break into **dimensions** (signal / parameterization / trigger / search / domain / systems / theory / wild). Each dimension forces a different kind of idea.
- For each dimension, generate 3–7 ideas; mark top picks with ★.
- Do *not* expand ideas yet. Brainstorm is wide, not deep.
- Artifact: `notes/<topic>-ideas.md` with a flat numbered/ID'd list (e.g. `A1`, `B3`, `I1`).

### Phase C — Read adjacent work
- For each top pick, find the 1–3 closest papers. Read the **abstract, method, limitations, and one ablation**. Skip the rest.
- Key extraction:
  - **TL;DR** (1 sentence)
  - **Method** (key equations + architecture)
  - **Results** (the one number that matters)
  - **Limitations** ← *this is where your gap lives*
  - **What it doesn't cover** ← list 3 unexplored directions
- Artifact: `notes/<topic>-reading.md` (one paper per section).

### Phase D — Plan
- For the 2–4 highest-priority ideas, write a plan folder with:
  - `README.md` — problem statement, idea recap, success criteria
  - `validation.md` — how you'd verify it works, baselines, metrics
  - `channels.md` — concrete benchmarks, datasets, evaluation suites
  - `budget.md` — GPU-hours, $ (cloud), wall-clock, headcount
  - `references.md` — papers, code, datasets
- Artifact: `notes/plans/NN-<idea-slug>/`.

---

## 2. Plan structure (mandatory fields)

Each plan must answer these questions. If you can't, don't write the plan yet.

| Section | Required content |
|---|---|
| **Problem** | What's broken / missing in the world today? In one paragraph. |
| **Idea** | What you'd do, in 3–5 sentences. Include the equation if there is one. |
| **Validation hypothesis** | The single sentence "If our idea works, we should observe X *measurably differently from baseline Y*." |
| **Channels** | The exact benchmarks (with versions), datasets, scorers. Be specific. |
| **Baselines** | At least 3: trivial, strong existing method, oracle / upper bound. |
| **Metrics** | Primary metric + 2 secondary. Include compute cost as a secondary metric. |
| **Budget** | GPU-hours by phase + $ estimate + wall-clock + people. |
| **Milestones** | At least 3 checkpoints with a measurable "ship / kill" criterion. |
| **Risks** | The 3 most likely reasons this fails. For each, a mitigation. |
| **References** | Closest 5–10 papers, with one-line annotations. |

---

## 3. Validation channels — preferred list

When choosing benchmarks, prefer (in order):

1. **Cheap, fast, programmatic verifiers** — GSM8K, MATH (rule-based), HumanEval+, MBPP, NIAH variants.
2. **Established multi-task suites** — MMLU-Pro, BIG-bench-Hard, AGIEval.
3. **Long-context / agentic** — LongBench, RULER, SWE-Bench, WebArena, OSWorld.
4. **Reward-model-graded** — Arena-Hard, MT-Bench (use only after the above).
5. **Human eval** — last resort; only when the question is alignment / open-ended.

**Avoid**:
- Benchmarks the base model has memorized (e.g., old MMLU on recent LLMs).
- Single-task benchmarks for general-claims work.
- Anything where the "verifier" is the same family as the model under test.

---

## 4. Budget estimation rules of thumb

| Activity | Rough cost (single H100) | Notes |
|---|---|---|
| Inference (7B, 1 problem, 2K-token CoT) | ~1–3 s | ×8 if Best-of-8 |
| LoRA training (7B, 10K samples, 3 epochs) | ~2–4 GPU-hours | Use PEFT + bf16 |
| Full SFT (7B, 100K samples, 3 epochs) | ~30–80 GPU-hours | 8 GPU run in 5–10 h wall-clock |
| RL (7B, GRPO-style, 1K iter) | ~100–300 GPU-hours | Verifier-bound for math/code |
| Hypernet training (D2L-scale, 300M params) | ~960 GPU-hours | Sakana spec |
| Curve sweep (10K problems × 6 budgets, 7B) | ~15–30 GPU-hours | Embarrassingly parallel |

**Cloud price benchmarks** (2026, approximate):
- H100 on-demand: $2–4/hr
- A100-80GB on-demand: $1.5–2.5/hr
- 8×H100 node: $20–30/hr

**Default budget categories for a paper-scale project**:
- Exploratory: 50–200 GPU-hours, ~$500
- Single-paper experiment: 500–2000 GPU-hours, ~$2–10K
- Full PhD-thesis-level project: 5K–20K GPU-hours, ~$20–100K

---

## 5. Idea-to-publication conversion checklist

Before committing serious resources to a plan, check:

- [ ] Can the headline result be stated in one sentence with one number?
- [ ] Is there a clear ablation that, if it fails, would kill the paper?
- [ ] Are there 3+ existing methods to compare against, and is data available for all of them?
- [ ] Is the cheapest version of this experiment feasible in <100 GPU-hours? (If not, prototype first.)
- [ ] Does the closest related work either (a) miss this idea or (b) report it as a limitation?
- [ ] Will the result still matter in 12 months? (Model improvements move fast.)
- [ ] Who else is most likely to scoop this in 6 months? Are they already working on it?

If 5+ are ✓, write the plan. Otherwise, iterate on framing.

---

## 6. Notes hygiene

- Each note links to its parents in `knowledge-sources.md`.
- Every new conversation / session gets an entry in `docs/matrix/`.
- Every plan has a status: `drafting | running | shipped | killed`.
- When killing a plan, write a one-paragraph postmortem in its folder. Don't delete.
- Re-read the matrix before starting any new session. The point of the matrix is to *not* repeat yourself.

---

## 7. When to switch from research to engineering

Heuristic: switch when the *bottleneck* changes.

- **Research bottleneck**: "What experiment do I run?" → stay in `notes/`.
- **Engineering bottleneck**: "How do I run this at scale?" → spin up a separate `src/` repo or fork an open-source baseline.

Do not mix the two. Notes are for thinking. Code is for shipping.
