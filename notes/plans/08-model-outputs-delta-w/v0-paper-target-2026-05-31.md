# Plan 08 v0 — Paper-Level Target & Roadmap

> Goal reframing. The 35-hr sweep finishes by tomorrow morning and
> tells us a recipe. That is **necessary but not sufficient** for a
> paper. This doc lays out (a) what "paper-level" requires concretely,
> (b) the gap between us and that bar, (c) a defensible paper framing
> that uses our existing results as a feature not a bug, and (d) a
> roadmap from "engineering checkpoint" to "draft."
>
> **Decisions locked in (2026-05-30):**
>
> - Framing: **Candidate A — Bit-Capacity Limits of Soft-Prompt
>   Memory** (analysis paper).
> - Venue: **Top venue (ACL / EMNLP / NeurIPS / ICLR main)**.
>   Realistic targets given today's date:
>   - **ICLR 2027** (primary; deadline ~Sep-Oct 2026, ≈ 4 months)
>   - **ACL 2027** (backup; deadline ~Feb 2027, ≈ 9 months)
>   - NeurIPS 2026 workshops (Sep-Oct deadlines) for early preview /
>     feedback if we want to test the framing publicly.
> - Compute ceiling: 4× H100 for 8-12 weeks ≈ 5000-7000 GPU-hr.
>   Candidate A's full top-venue matrix uses ≈1500-2500 GPU-hr —
>   leaves room for re-runs and reviewer additions.

## 1 — What a paper-level long-context-memory paper requires

Distilled from the recent long-context-memory literature
(ICAE, AutoCompressor, Gist Tokens, RMT, CompAct, KV-cache surveys):

| Requirement | Minimum bar | Our current state |
|---|---|---|
| **Benchmark coverage** | ≥2 standard suites (LongBench / RULER / LongMemEval / SCROLLS) | 0 (only synthetic NIAH variants) |
| **Comparable baselines** | ≥3 (no-memory, full-context, ≥1 published memory method) | 2 (full-context, retrieval); 0 published memory methods |
| **Long context regime** | ≥8K tokens, ideally 16K-128K | 1.5K (6 chunks × 256) |
| **Model scale** | ≥1 model + 2-3 sizes for ablation, or 1 size + cost analysis | 1 model (Qwen3-8B), 1 size |
| **Statistical rigor** | ≥3 seeds, error bars on headline numbers | 1 seed (in flight: 5 seeds Phase F) |
| **Positive contribution** | Either win on something, or characterize *why* failure modes happen with new insights | Currently neither |
| **Cost / inference analysis** | FLOPs or latency or memory tradeoff curve | None |
| **Reproducibility** | code + configs + checkpoints | code only, no checkpoints |

We have a long way to go.

## 2 — What we have that's actually paper-worthy

Even though headline numbers are below baselines, three findings
are publishable as nuggets:

1. **Bit-capacity wall**: Wrapper goes from `eval=0` on 40-bit task
   to `eval=0.34` on 2-bit task with the same architecture. This is
   a clean, reproducible failure-to-success transition by changing
   only the answer entropy. Information-theoretically interpretable.
2. **OPD-alone collapse**: Pure on-policy distillation collapses
   f1 (0.04 vs 0.24 on contains) and explodes memory drift (6× the
   SFT baseline). SFT+OPD+RL recovers it. This is a methods finding
   about *how to train* learned memory.
3. **xattn read interface**: Adding apply-time cross-attention from
   recurrent memory to chunk hidden states moves the needle (0.24 →
   ~0.34) without changing the write-side recurrence. Suggests the
   bottleneck is on read, not on compression.

None of these alone is a paper. Together with the right framing,
they could be.

## 3 — Proposed paper framing (decision needed)

### Candidate A — "Bit-Capacity Limits of Soft-Prompt Memory" (recommended)

**Thesis**: there's a hard ceiling on what a fixed-size soft-prompt
memory can encode, set by `K × d_model` bits at the theoretical
extreme and much less in practice. We characterize this ceiling
empirically across answer entropy, sequence length, and memory
size, and identify the read interface (not write capacity) as the
operational bottleneck for current designs.

**Why this works for us**:

* Centerpiece is a *failure-mode characterization*, which doesn't
  require us to beat retrieval — only to explain when soft memory
  works and when it doesn't, with predictive accuracy.
* Our existing 2-bit vs 40-bit gap is a 2-point capacity curve
  start. We need 4-6 more points to publish, but each is cheap.
* The training-recipe contribution (SFT+OPD+RL) becomes
  *secondary* — useful method but not the headline. That's fine
  because it's a small effect on its own.
* Comparable to recent work like "How Much Context Does My Attention
  Need?" — diagnostic papers are accepted at top venues.

**Required experiments**:

* Capacity curve: K ∈ {16, 32, 64, 128, 256, 512} × entropy ∈
  {2-bit, 5-bit, 10-bit, 17-bit, 30-bit, 40-bit} = 36 points
* Sequence length scan: n_chunks ∈ {4, 8, 16, 32, 64} at fixed K
* Real-text validation: QuALITY (~4-way ≈ 2 bit), QASPER (~10-bit),
  show the same shape transfers
* Read-interface ablation: prefix-only vs xattn vs interleave
  across capacity curve
* **Total**: ~80-120 runs at 1-3 hr each = 200-400 GPU-hours.

**Venue**: ACL/EMNLP main, ICLR (analysis track), TMLR.

### Candidate B — "Multi-Stage Training for Learned Memory"

**Thesis**: SFT-only training of learned soft memory overfits;
OPD-alone diverges; SFT→OPD→RL is the right recipe. Reduces train-
eval gap from −0.68 to −0.12 (eval > train).

**Why this is weaker**: we'd need to apply the recipe to ≥2 other
memory architectures (RMT, Gist) to show generality. The recipe is
not novel-enough on its own — it's known RL fine-tuning theory.

**Required experiments**: ~150 runs (3 recipes × 3 architectures ×
3 datasets × 5 seeds + ablations).

### Candidate C — "Soft Memory + Retrieval Hybrid"

**Thesis**: pair the wrapper with cheap retrieval. The wrapper
captures the "compressed gist" while retrieval surfaces specific
facts. Compete on a Pareto frontier of cost vs accuracy.

**Why this is intriguing but premature**: requires us to first
*have* a wrapper that adds value over retrieval alone, which we
don't yet. Re-visit after Candidate A clarifies what soft memory
adds.

### → Recommendation

Go with **Candidate A**. It's the framing where our actual results
(modest positive on easy, negative on hard) become a *feature*
rather than a weakness. It also has the cleanest experimental
matrix (one big sweep + a few real-text validators).

## 4 — Roadmap (top-venue scope, 8-12 weeks)

Assumes Candidate A + top venue. All wall-clock estimates assume
4 H100s, single-user. Each "week" is calendar-week.

### Week 1 — sweep aggregation + capacity curve foundation

| day | deliverable | infra need |
|---|---|---|
| Sun (tomorrow) | aggregate 35-hr / 106-job sweep; lock recipe & best HP; write `v0-sweep-results.md` | none |
| Mon | extend `datasets.py`: `multi_needle_niah` (k=3, 5 selective-recall stress), `same_form_distractors` (semantic stress) | code |
| Mon-Tue | refresh capacity curve: add Phase J runs at entropy points missing from Phase I (numerical_niah at 5-seed × 4 anchor points) | sweep extension |
| Wed-Thu | implement **Gist Tokens** baseline (Mu et al. 2023) — published comparator with the cleanest training recipe. Train on same data using our orchestrator. | code |
| Fri | add QuALITY adapter (HF: `emozilla/quality`). First real-text eval point. | code |
| Sat | mid-week internal write-up: capacity curve v1 + Gist comparison on synthetic. |

### Week 2 — published baselines + real-text benchmarks

| day | deliverable |
|---|---|
| Mon-Tue | RULER-NIAH subset (HF: `simonjegou/ruler`) at 4K and 16K context. The standard long-context NIAH protocol. |
| Wed | LongMemEval-mini (50-100 items hand-curated; full eval if time). The most paper-relevant *memory* benchmark. |
| Thu-Fri | second published baseline: **AutoCompressor** (Chevalier et al. 2023) — recurrent soft-token compression, directly comparable to our recurrence. |
| Sat | results table v0: 3 benchmarks × 4 methods × 3 seeds. |

### Week 3 — multi-model scaling (biggest compute)

| day | deliverable | est. GPU-hr |
|---|---|---|
| Mon-Tue | Qwen3-1.7B re-runs across the capacity curve | ~80 |
| Wed-Thu | Qwen3-4B re-runs across the capacity curve | ~250 |
| Fri | optional Qwen3-14B at headline points only (3-5 configs) | ~300 |
| Sat | scaling curves: eval-c × model size; bit-capacity vs theoretical bound |

Multi-model is the single biggest compute block: ~600 GPU-hr → ~6
wall-days on 4 H100s. Front-load it Week 3.

### Week 4 — third baseline + cost analysis + 5-seed bars

| day | deliverable |
|---|---|
| Mon-Tue | third published baseline: **RMT** (Bulatov et al. 2022) or **ICAE** (Ge et al. 2024). Pick whichever has cleaner code. |
| Wed | cost analysis: per-token FLOPs and inference latency for each method, on each benchmark, at each model size. |
| Thu-Fri | seed-replicate top headline numbers (5 seeds) so error bars are publishable. |
| Sat | analysis-only sweep: drift geometry, attention patterns, ablation of the read interface. |

### Weeks 5-6 — writing + iteration

| week | deliverable |
|---|---|
| 5 | full draft v0: intro / related / method / experiments / discussion. Figures: capacity heatmap, per-method radar, scaling curves, geometric diagnostics. |
| 6 | full draft v1 with internal feedback incorporated. Re-run any experiments revealed as missing. |

### Weeks 7-8 — final experiments, polish, submission

| week | deliverable |
|---|---|
| 7 | any reviewer-anticipating experiments (alternative metric definitions, sensitivity analyses, code-release prep). |
| 8 | submission package: PDF + supplementary + code release + reproducibility checklist. |

### Stretch (Weeks 9-12, if reviewer feedback / extension cycle)

* Qwen3-32B headline points (~600 GPU-hr alone).
* SCROLLS or InfiniteBench as third real-text benchmark.
* Compress-then-retrieve hybrid (Candidate C lite) as an extension.

**Total to first submission**: ~8 weeks from today (≈ late July 2026
for ICLR 2027 submission).

**Compute budget check**:
- Phases A-I (this week's 35-hr sweep): ~70 GPU-hr (sunk).
- Weeks 1-2 (real benchmarks + 1st baseline): ~150 GPU-hr.
- Week 3 (multi-model): ~600 GPU-hr.
- Week 4+ (more baselines + 5-seed + analyses): ~400 GPU-hr.
- Reserved buffer: ~500 GPU-hr.
- **Total**: ~1700 GPU-hr; ≈18 wall-days on 4× H100. Comfortable.

## 5 — What changed tonight (already queued)

1. **Kept the 35-hr sweep running** — Phase 0 of Week 1.
2. **Added Phase H** (10 jobs): `coding_niah_K{16, 256, 512}`,
   `categorical_K{8, 256}`, length-axis (chunks 8/64), and longer-
   context coding_niah at K=128. Pre-stages capacity-curve anchors.
3. **Shipped `numerical_niah` adapter** in `llm_infra.datasets`,
   exposing 6 entropy levels (digits=1, 2, 3, 5, 8, 12 → 3.3 / 6.6
   / 10 / 16.6 / 26.6 / 39.9 bits). Registered as
   `numerical_niah_d{N}` choices in `train_smoke.py`.
4. **Added Phase I** (12 jobs): explicit K × entropy capacity
   sub-matrix using `numerical_niah_d{N}` at K∈{16, 32, 128, 256}.
   This is the actual centerpiece data for the paper.

**Current sweep:** 106 jobs, ~18 wall-hr on 4× H100, picked up by
the orchestrator when it transitions out of `WAIT_IDLE` (i.e. once
the 4 in-flight stages runs finish).

## 6 — Open questions still pending

Resolved (2026-05-30): framing = A, venue = top.

Still pending:

1. **Specialization timing.** Do we still want coding-RCA / telecom-
   RCA specialization in this paper, or save for a v1 follow-up?
   *Recommendation:* save for v1. Candidate A stands alone, and
   adding specialization here dilutes the analysis message and
   doubles experiment count.
2. **Model family commitment.** Stick with Qwen3 throughout, or add
   a single point from a different family (e.g. Llama-3-8B) to show
   the findings aren't Qwen-specific? *Recommendation:* add Llama-
   3.1-8B headline points only (3-5 configs) in Week 4 as a
   robustness check.
3. **Code release timing.** Release on submission, on acceptance,
   or on preprint? Top venues typically expect submission-time
   release. Need to decide before Week 4.

## 7 — Honest assessment

Even with the above, this is a **moderate-confidence** paper. The
risk is that capacity-curve papers can land as "tells us something
expected" (information theory predicts capacity ≤ K·d bits, we
empirically confirm this). The novel angle is the *practical*
ceiling — how far below the theoretical bound we actually sit,
and why (read interface). That's a more interesting result if it
shows a 100×+ gap, less so if it shows 2-3×.

We'll know after Phase I runs (~tomorrow) whether the story is
compelling enough to push for a top venue, or whether we should
descope to TMLR / workshop.

**Go / no-go decision points:**

- **Tomorrow (Sun)**: do Phase I results show a clear bit-capacity
  cliff that doesn't go away with larger K? If yes → proceed
  Week 1 as planned. If no (wrapper plateaus everywhere) → revisit
  framing: maybe Candidate B is safer.
- **End of Week 1**: does the Gist Tokens baseline outperform us
  by >>20 points on QuALITY? If yes → we have a real problem;
  reconsider whether to keep our architecture as the headline.
  If no → continue.
- **End of Week 3 (multi-model)**: do the capacity findings hold
  across model sizes? If yes → strong paper. If no → the
  framing needs a "Qwen-specific" caveat that weakens the story.

---

## Addendum (2026-06-01) — v2 framing decided

After a literature pass on "embedding memory written during AR
generation" (see `v2-related-work.md`), the v2 paper direction is
locked: **cross-session latent memory (A) + read/write under a
strictly frozen base with the same bit-capacity diagnostic (B)**.

This makes v1 (this paper) and v2 a single line of work:

* v1 = "encode-time soft memory has a read-side bit-capacity wall"
* v2 = "the wall persists for generation-time, cross-session memory
  too, and here is the wrapper that lives at the empty intersection
  of the existing methods"

Therefore: do **not** over-claim v1 as the last word on soft memory.
Frame v1 explicitly as the diagnostic foundation that v2 extends.
See `v2-plan.md` for the full v2 design and `q2-activation-memory-probe.md`
for the spun-off methodology question on which activations to write into.
