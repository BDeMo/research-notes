# If the soft-prompt wrapper doesn't work — brainstorm + pivot menu

**Date:** 2026-06-03 PT
**Settings / provenance:** result cells in this memo cite
[`settings.md`](settings/settings.md). The Phase Y / RULER evidence uses
[`P08-S2`](settings/settings.md#p08-s2--v1-phase-y-three-regime-benchmark-cells);
v2 design directions use [`P08-S4`](settings/settings.md#p08-s4--v2-design-setting).
**Trigger:** Phase Y multi-seed bands (16/24 done) + Phase X early
RULER-NIAH single seed have crystallised the three-regime
characterisation (Regime A: matches Gist on QuALITY; Regime B:
at-chance on MuSR; Regime C: 0.000 on RULER while
full_context = 0.995). The honest reading: the wrapper is **a
lossy compressor with a narrow utility window**, not a
general-purpose memory.

This document lays out (a) what "doesn't work" means concretely,
(b) ten directions we could pivot to, (c) priority ranking
across "testable today / next-week / v2 paper" timeframes, (d)
the immediate decision: do we **scope down v1 to the
characterisation paper** (the honest read), or do we **scope up
to a hybrid system** that combines the wrapper's strengths
with retrieval's verbatim fidelity?

## 1. What "doesn't work" means

| failure mode | setting | evidence | conclusion |
|---|---|---|---|
| Doesn't beat the matched Gist baseline | [`P08-S2`](settings/settings.md#p08-s2--v1-phase-y-three-regime-benchmark-cells) | Phase Y QuALITY 4-seed: OURS 0.193 ± 0.037 vs GIST 0.180 ± 0.051; Δ < σ | The single-seed +12 pp claim was an artefact. With variance bands the two architectures are statistically tied. |
| Doesn't beat full-context on tasks where the base can actually use the context | [`P08-S2`](settings/settings.md#p08-s2--v1-phase-y-three-regime-benchmark-cells) | MuSR-mm: OURS 0.492 ± 0.010 vs full_context 0.551 | The compression is throwing away information the base can use. |
| Catastrophically fails on verbatim-retrieval tasks | [`P08-S2`](settings/settings.md#p08-s2--v1-phase-y-three-regime-benchmark-cells) | RULER-NIAH single seed: OURS 0.000 vs full_context 0.995 vs retrieval 0.995 | Lossy compression destroys the exact string the answer requires. The wrapper has no path to "preserve verbatim". |
| Doesn't transfer across task families | [`P08-S1`](settings/settings.md#p08-s1--v1-canonical-wrapper-recipe) | Phase~P cross-task: cat_niah → numerical_niah is a hard failure (bit-capacity wall, both wrappers at em ≈ 0) | Single-task SFT does not yield a general-purpose memory. |
| Has bimodal seed behaviour at the held-out cat_niah cell | [`P08-S1`](settings/settings.md#p08-s1--v1-canonical-wrapper-recipe) | abstract finding (ii): 4-seed mean 0.27 ± 0.29, bimodal | Even within-distribution behaviour is fragile to initialisation. |

**Synthesis.** What we built is a small *lossy compressor*
trained on one task. It works as a compressor when the task is
amenable to lossy compression (Regime A, QuALITY) and fails
when the task isn't (Regime C, RULER) or when single-task SFT
doesn't transfer (Regime B, MuSR; cross-task numerical_niah).
This is a real but **narrow** contribution.

## 2. The brainstorm — ten directions

Ordered loosely by distance-from-current-method (closest first).

### 2.1 Direction A — Accept characterisation, ship the negative-result paper

**Idea.** The 3-regime taxonomy of §5 of the v1 paper IS the
contribution. Rewrite to emphasise: "we tried hard, here is the
capacity ceiling, here is the regime where you can use this
class of method, here is the regime where you cannot".

**Pros.** Honest. Defensible. Already mostly written.
Connects to the bit-capacity-wall finding cleanly.

**Cons.** Reads as a "wrapper-doesn't-work" paper, which is
hard at a main venue. Likely workshop fit (NeurIPS workshop on
efficient inference, ICLR workshop on representations).

**Cost.** ~zero — already most of the way there.

**Verdict.** Floor — always available as fallback.

### 2.2 Direction B — Hybrid wrapper + retrieval (the easiest pivot)

**Idea.** Under [`P08-S2`](settings/settings.md#p08-s2--v1-phase-y-three-regime-benchmark-cells),
retrieval baseline gets 0.995 on RULER and 0.493 on
QuALITY. Wrapper gets 0.000 on RULER and 0.193 on QuALITY.
**They are complementary.** Build a 2-stage router that:
1. Decides whether the query needs verbatim retrieval or lossy
   gist (a tiny classifier or always-both with merging).
2. Wrapper provides compressed reasoning context, retrieval
   provides verbatim spans, both fed to the base.

**Why it might work.** We already have both outputs measured.
The wrapper's job becomes "compress for reasoning"; retrieval's
job becomes "preserve verbatim". The combined system has
strictly better coverage than either alone.

**Pros.** Testable TODAY in ~1 hour (we have all the predictions).
Real engineering contribution. Frames the wrapper as one tool in
a RAG stack, not a replacement. Sells.

**Cons.** RAG papers are crowded. Need a careful novelty story
(probably: "learned compression of context for the synthesis
stage, retrieval for the verbatim stage, end-to-end fairness").

**Cost.** 1-2 hours of code + 1 sweep cell to confirm. The
predictions JSONLs already exist.

**Verdict.** STRONG candidate for v1.5 / immediate next step.

### 2.3 Direction C — Suffix memory (the v2 plan)

**Idea.** Memory tokens appended **after** the autoregressive
generation begins, not prefix. Allows iterative refinement: as
the model decodes, the wrapper updates memory based on what's
being generated, and the updated memory steers later decoding.
This is the v2 the user already planned.

**Why it might work.** Aligns with how the base actually
processes context (left-to-right with growing KV cache).
The base's "future" attention can see both source chunks AND
the current draft; the wrapper's memory becomes a working-memory
register that's read+written during decoding.

**Pros.** Genuinely novel mechanism. Plays well with
speculative decoding. Could fix Regime B by giving the wrapper
a chance to "think" mid-decode.

**Cons.** Major architecture change. Training instability risk.
At least 2 weeks of work to land a single result.

**Cost.** ~2 weeks of dev + sweeps.

**Verdict.** v2 paper. Document the design now, ship in v2.

### 2.4 Direction D — Train with infilling, not next-token

**Idea.** Current training: SFT on (context → answer). The
gradient pressure on the memory is indirect (must back-prop
through frozen base). Alternative: train the memory to
**reconstruct masked spans of the context** (BERT-style infilling).
This is a stronger, more local supervision signal that forces
the memory to be a "compressed reconstruction key".

**Why it might work.** If the memory can reconstruct spans, it
can answer verbatim-retrieval questions (Regime C). Currently
the memory has zero training signal for verbatim preservation.

**Pros.** Same wrapper architecture, just a different loss.
Testable in 1-2 hours (we already have the auxiliary head
machinery). Connects to ICAE / autocompressor literature.

**Cons.** Might trade Regime A performance for Regime C
performance (the compression is now optimised differently).

**Cost.** 2-4 hours of dev + 1 multi-seed sweep.

**Verdict.** STRONG candidate. Can run alongside Direction B as
a v1.5 ablation.

### 2.5 Direction E — Multi-resolution memory

**Idea.** Current: 32 tokens at one resolution. Alternative:
hierarchical — e.g. `[4 global summary tokens] + [16
medium-grain tokens] + [12 retrieval-anchor tokens]`. The
retrieval anchors are vectors that **point back to specific
source chunks** that can be fetched at decode time.

**Why it might work.** Different regimes need different
information densities. Global summary handles Regime A,
retrieval anchors handle Regime C.

**Pros.** Generalises both Gist (purely compressive) and pure
retrieval (purely indexed). Real novelty.

**Cons.** Combinatorial design space; lots of hyperparams.

**Cost.** 1-2 days dev + bigger sweep.

**Verdict.** v1.5 or v2 candidate.

### 2.6 Direction F — KV-cache compression (skip soft-prompt entirely)

**Idea.** Don't use a soft prompt. Instead, the "wrapper"
learns to compress the FROZEN BASE'S OWN KV cache. Train it to
output a low-rank approximation of the K and V tensors for each
chunk that the base would have produced. Then at decode the
base attends over the compressed KV directly.

**Why it might work.** Connects to H2O, StreamingLLM, KV-cache
pruning. The compression operates INSIDE the attention
machinery rather than as a prefix, which is more bandwidth-
preserving.

**Pros.** Cleaner separation from soft-prompt literature.
Could give us a different bit-capacity ceiling.

**Cons.** Lower-level kernel work; requires understanding the
base's exact KV layout (rotary positional encoding, head
splits, etc.). Likely a different paper entirely.

**Cost.** 1-2 weeks of dev.

**Verdict.** v2 or v3 (independent paper).

### 2.7 Direction G — Learned retrieval index (vector DB compression)

**Idea.** The wrapper learns to compress each chunk into a
vector that's used as a retrieval key. At inference: query →
retrieval over the compressed vectors → original (or also
compressed) chunk → prefix-attention to the base.

**Why it might work.** Solves Regime C explicitly (retrieval
recovers verbatim spans). Solves Regime A by training the
compression to be task-relevant. Generalises both Gist and DPR.

**Pros.** Joint training of compression + retrieval is genuinely
novel. Plays well with vector DB infrastructure (FAISS, Qdrant).

**Cons.** Two-stage system; harder to evaluate end-to-end.

**Cost.** 3-4 days dev.

**Verdict.** STRONG v1.5 candidate.

### 2.8 Direction H — Test-time generation of LoRA deltas (the original "plan 08")

**Idea.** The wrapper outputs LoRA-style delta weights ΔW that
adapt the base for this specific context. At decode: base+ΔW(c)
where ΔW is generated by the wrapper conditioned on c.

**Why it might work.** Much more capacity than 32 soft tokens —
ΔW can have millions of parameters and modify the base's
behaviour deeply. Connects to drag-and-drop LLM, HyperNet,
TextGrad-style approaches.

**Pros.** This was the original ambition of "08-model-outputs-
delta-w". Soft-prompt was the constrained v0.

**Cons.** High variance; hard to train; risk of hypernet
collapse (the wrapper finds a degenerate ΔW that hard-codes
common answers).

**Cost.** 2-3 weeks dev.

**Verdict.** v2 or v3 (this is the original delta-W plan
revisited).

### 2.9 Direction I — Unfreeze the last 1-2 layers

**Idea.** Currently 100% frozen. Try unfreezing the last
attention block + LM head (~150 M params, similar to the
wrapper itself). Wrapper supplies the memory; the last layers
learn to USE the memory.

**Why it might work.** The current wall might be that the
frozen base layers can't actually USE the soft-prompt signal
well. Adapting the read-out helps.

**Pros.** Single-knob ablation. Easy to test.

**Cons.** Loses the "no base finetuning" purity. Becomes more
like a LoRA / adapter paper.

**Cost.** 1-2 hours dev + 1 sweep.

**Verdict.** STRONG candidate for an honest ablation row,
even if we don't pivot the whole paper.

### 2.10 Direction J — Application pivot (telecom / coding RCA)

**Idea.** Drop the academic public-benchmark target. Pivot to
the original use case (telecom RCA, coding RCA) where the data
distribution is narrower, the answer format is more
structured, and a single-task wrapper has a real chance.

**Why it might work.** RCA tasks have repetitive structure that
the wrapper can learn. No need to compete with FullContext on
QuALITY or NIAH on RULER.

**Pros.** Mat ches user's original goal. Less competitive
benchmark space.

**Cons.** Loses the "general" framing. Reviewers prefer general
methods.

**Cost.** 1-2 weeks for new evaluation suite.

**Verdict.** v3 — application paper after v1/v2 main paper.

## 3. Priority ranking

By time-to-result × probability-of-payoff:

| rank | direction | testable today? | upside | downside |
|---|---|---|---|---|
| 1 | **B. Hybrid wrapper + retrieval** | ✅ 1 hour | clean v1.5 story | RAG crowded |
| 2 | **D. Infilling training** | ✅ 2-4 hours | might fix Regime C | risks Regime A |
| 3 | **I. Unfreeze last layer ablation** | ✅ 2 hours | clean ablation row | loses purity |
| 4 | **A. Ship characterisation paper** | (writing only) | always safe | "negative" optics |
| 5 | **G. Learned retrieval index** | ❌ 3-4 days | most novel | engineering load |
| 6 | **E. Multi-resolution memory** | ❌ 1-2 days | generalises Gist + retrieval | hyperparam explosion |
| 7 | **C. Suffix memory (v2 plan)** | ❌ 2 weeks | next paper | major rebuild |
| 8 | **H. LoRA-delta hypernet** | ❌ 2-3 weeks | original ambition | training instability |
| 9 | **F. KV-cache compression** | ❌ 1-2 weeks | new mechanism | independent paper |
| 10 | **J. Application pivot** | ❌ 1-2 weeks | matches user goal | less general |

## 4. Recommendation — three concrete next steps for THIS WEEK

### Step 1 (today, ~1 hr) — Hybrid wrapper + retrieval merge

Take the existing Phase Y predictions.jsonl files. For each
QuALITY / MuSR / RULER cell, build the merged prediction:
* If the question is verbatim-style (e.g. "what is X?"), use
  retrieval's answer.
* Otherwise, use wrapper's answer.
* Fallback / both: prefer wrapper if both agree, retrieval if
  they disagree on a verbatim cue.

Compute the merged metric per benchmark. If the merged score
> max(wrapper, retrieval) on ≥ 2/3 benchmarks, **this is the
v1.5 pivot**.

### Step 2 (tonight, ~3 hr) — Infilling training pilot

Add a `--train-modes infill` option to train_smoke.py: mask 15%
of context tokens, force the wrapper's memory to predict them
back via a separate decoder head. Run 1 multi-seed cell on RULER
with this objective. If RULER OURS jumps from 0.000 to > 0.5,
**this fixes Regime C and v1 paper is back on**.

### Step 3 (tomorrow, ~4 hr) — Unfreeze ablation row

Run 1 multi-seed cell with last-layer + LM head unfrozen.
Report alongside the canonical row. Either we add 1 row to
Table 1 and move on, or this becomes the v1.5 main result.

## 5. Bottom line for the v1 paper

Given the current 3-regime evidence, the v1 paper has three
viable framings, in increasing risk:

* **(low risk)** Direction A — characterisation paper. Ship as
  is, mostly written. Workshop venue likely.
* **(medium risk)** Direction A + Direction B as a v1.5
  extension — adds "and we show a simple hybrid with retrieval
  closes the Regime C gap". Main venue viable.
* **(high risk)** Direction A + Directions B+D+I together — a
  fuller pivot that re-establishes "OURS competitive on all 3
  regimes". Main venue strong, but requires the experiments
  above to actually land favourably.

User's call which risk level we commit to. Default: pursue
medium risk path; run B+D+I in parallel over the next 24 hours
and re-assess.

## 6. What's already covered in earlier documents

* `v0-make-it-work-2026-06-01.md` — earlier list of "make the
  method work" approaches (data scale, answer head, etc.) —
  superseded by this doc because the multi-seed data has now
  told us those didn't fully succeed at delivering a
  superiority claim.
* `v1-method-design-improvements-2026-06-02.md` — sketches
  bigger-budget memory, dual-rail memory, sparse-write — those
  remain valid but are subsumed under Directions E + H here.
* `v2-plan.md` — the suffix-memory v2 plan; Direction C above
  formalises this and adds priority context.

## 7. Connections to external literature

* RAG: hybrid (B) connects to `Adaptive-RAG`, `Self-RAG`,
  `Retrieval-Augmented Generation` general lineage.
* Infilling (D): ICAE (`icae_2024`), Autocompressor
  (`autocompressor2023`), and Marrying SoftPrompt with KD
  literature.
* KV compression (F): H2O, StreamingLLM, FastGen.
* HyperNet / drag-and-drop (H): drag-and-drop LLM 2025,
  HyperNetworks 2017, T-Few.
* Multi-resolution memory (E): connects to RMT
  (`bulatov2022rmt`), ARMT (`bulatov2024armt`), and
  next-mem 2026 (`nextmem2026`).

End of brainstorm. Next action: get user buy-in on risk level
(low / medium / high), then queue B + D + I cells if medium
or above.
