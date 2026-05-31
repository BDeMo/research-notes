# Plan 08 v0 — Paper-Level Target & Roadmap

> Goal reframing. The 35-hr sweep finishes by tomorrow morning and
> tells us a recipe. That is **necessary but not sufficient** for a
> paper. This doc lays out (a) what "paper-level" requires concretely,
> (b) the gap between us and that bar, (c) a defensible paper framing
> that uses our existing results as a feature not a bug, and (d) a
> 3-4 week roadmap from "engineering checkpoint" to "draft."

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

## 4 — Three-week roadmap

Assumes we go with Candidate A. All wall-clock estimates assume
4 H100s, single-user.

### Week 1 (now → next Sat, ~7 days)

| day | deliverable |
|---|---|
| Sun (tomorrow) | aggregate 35-hr sweep, lock recipe + best HP. ½-day. |
| Mon | extend `datasets.py`: `numerical_niah` (3/5/10/15-digit), `multi_needle_niah`, longer-context coding_niah. 1 day. |
| Mon-Tue | run the capacity curve (K × entropy) on the winning recipe. ~40 runs, ~30 GPU-hr. |
| Wed | implement Gist Tokens baseline (cleanest published comparator). Train on same data. 1 day. |
| Thu-Fri | add QuALITY adapter (HF: `emozilla/quality`). Eval winner + Gist on QuALITY. ½ day infra + 4-6 hr runtime. |
| Sat | first internal write-up: capacity curve + recipe + Gist comparison. |

### Week 2

| day | deliverable |
|---|---|
| Mon-Tue | RULER-NIAH subset (HF: `simonjegou/ruler`) at 4K and 16K context. Adapter ½ day; eval matrix 1 day. |
| Wed | LongMemEval-mini (50-100 items hand-curated). The most paper-relevant memory benchmark. |
| Thu-Fri | second baseline: AutoCompressor or RMT (whichever has cleaner code). 1-2 days. |
| Sat | results table v0: 3 benchmarks × 3 methods × 3-5 seeds. |

### Week 3

| day | deliverable |
|---|---|
| Mon-Wed | multi-model: re-run winning recipe on Qwen3-1.7B and Qwen3-4B. Generates a 2-3 point scaling curve. |
| Thu | cost analysis: FLOPs per token × eval cost curves. ~½ day. |
| Fri | analysis figures: capacity heatmap, train-eval gap by recipe, drift geometry over training. |
| Sat | full results table + figures finalized. |

### Week 4 (write)

| day | deliverable |
|---|---|
| Mon-Tue | draft intro + method. |
| Wed-Thu | draft experiments + analysis. |
| Fri | draft related work + discussion. |
| Sat-Sun | full pass + figures + submission target check. |

**Total**: ~4 weeks from today. Probably realistic for a workshop
draft (NeurIPS workshops late summer) or a TMLR submission.

## 5 — What changes immediately tonight

1. **Keep the 35-hr sweep running** — it's exactly Phase 0 of Week 1.
2. **Extend Phase G of the sweep** with two cheap additions that
   start populating the Week-1 capacity curve:
   * `coding_niah_K{64,128,256}` — see if larger K rescues the
     40-bit failure (one more entropy point).
   * `categorical_niah_n_chunks{16,32,64}` — extends the length
     axis we'll need.
   These cost a few GPU-hours and pre-stage Week-1 data.
3. **Add Phase H (numerical_niah)** — but only after the dataset
   adapter ships; deferred to Monday.

The Phase G additions are queued via `sweep-master-extras.json`
that the orchestrator picks up next time we re-launch (or
hand-queued tomorrow morning when the master sweep finishes).

## 6 — Open questions for the user

Before Monday I need a decision on:

1. **Paper framing**: Candidate A (capacity), B (training), or C
   (hybrid)? Or a fourth I'm not seeing?
2. **Target venue**: ACL/EMNLP/ICLR (longer cycle, higher bar) vs
   TMLR (rolling, methods-friendly) vs workshop (faster, smaller
   contribution acceptable)? Drives experiment scope.
3. **Compute ceiling**: roughly 4 H100s for ~4 weeks gives us
   ~2700 GPU-hours total. Candidate A's full matrix uses
   ~400 — comfortably within budget. Anything bigger (multi-model
   at scale, e.g. Qwen3-32B) starts to bite.
4. **Specialization timing**: do we still want coding-RCA /
   telecom-RCA specialization in the paper, or save for v1?
   Recommend: save for v1 — Candidate A is general enough to stand
   alone, and adding specialization here dilutes the message.

## 7 — Honest assessment

Even with the above, this is a **moderate-confidence** paper. The
risk is that capacity-curve papers can land as "tells us something
expected" (information theory predicts capacity ≤ K·d bits, we
empirically confirm this). The novel angle is the *practical*
ceiling — how far below the theoretical bound we actually sit,
and why (read interface). That's a more interesting result if it
shows a 100×+ gap, less so if it shows 2-3×.

We'll know after the capacity curve runs whether the story is
compelling enough to push, or whether we should pivot to one of the
weaker candidates (B or C) or scope down to a workshop draft.
