# Plan 08 v0 — Datasets & Benchmarks Plan

> Companion to [`v0-results-2026-05-30.md`](v0-results-2026-05-30.md).
> Catalogs the datasets currently wired into ``llm_infra.datasets``
> and proposes the next set to add. Priorities follow what we
> learned: the wrapper has limited bits-of-information capacity at
> K=32, but it *does* learn on solvable tasks; we need a mix of
> synthetic-with-knobs and real-text benchmarks to characterize
> where the wrapper helps and where it doesn't.
>
> **Settings / provenance:** dataset/result references here predate the final
> v1 registry. Use [`settings.md`](settings.md) for current setting IDs; final
> v1 benchmark cells use
> [`P08-S2`](settings.md#p08-s2--v1-phase-y-three-regime-benchmark-cells).

## What we have

| ID | Type | Answer entropy | Length | Status | Where |
|---|---|---|---|---|---|
| `coding_niah` | synthetic NIAH, 1 needle | ~40 bits (8-char random string) | 6–8 chunks × 256 tok | shipped | `llm_infra.datasets.generate_coding_niah` |
| `categorical_niah` | synthetic NIAH, 1 needle | 2 bits (4-way label) | 6 chunks × 256 tok | shipped | `llm_infra.datasets.generate_categorical_niah` |

Both are 1-needle haystacks with the same Python-stub distractors;
the only difference is the answer-space cardinality. Results so far:
the wrapper plateaus at `eval_contains = 0.000` on coding_niah and
~`0.420` on categorical_niah (`xattn` combine, 2000 SFT steps).

## Design dimensions worth varying

Beyond the answer-entropy axis we already have, the next round
should vary:

1. **Number of needles** (1 → 3 → 5). One-needle = locate + recall.
   k-needle = locate + select-and-recall, which stresses the
   wrapper's memory-as-index property more.
2. **Distractor distribution.** Plain code-style distractors are
   trivially separable from the needle by surface form. Same-form
   distractors (functions that look like the needle but compute
   something else) force the wrapper to use semantic features.
3. **Sequence length.** 6 × 256 = ~1.5 K tokens currently. The
   interesting wrapper regime is 16K–128K, where soft-prompt
   compression actually saves cost vs full context.
4. **Answer mode.** Free-form generation (current) vs multiple-choice
   logit comparison vs span-extraction. Each tests a different
   slice of the wrapper's expressive power.
5. **Reasoning depth.** Single-step retrieval vs multi-hop
   aggregation. Multi-hop is where soft compressed memory should
   shine, because the compression *is* the aggregation.

## Tier 1 — add immediately (cheap, high info, single-repo work)

The first three additions are all 1-day jobs and cover the most
informative gaps.

### T1.1 — Multi-needle NIAH (synthetic, controlled k)

* **What it tests**: selective recall under interference. With
  k=5 planted facts and a query that names one, the wrapper has
  to remember which value goes with which key, not just "the"
  value.
* **Integration**: extend `_needle_chunk` to plant k needles and
  the query to specify which to retrieve. Probably 60 LOC in
  `datasets.py`. No external dependencies.
* **Why now**: directly tests the failure mode the
  categorical_niah results suggested — the wrapper carries
  *enough* bits for one fact but smears them into a default;
  multi-needle forces the wrapper to keep facts *distinguishable*.

### T1.2 — Numerical NIAH (synthetic, intermediate entropy)

* **What it tests**: answer-entropy interpolation between 2 bits
  (categorical_niah) and 40 bits (coding_niah). Plant a 3-digit
  number → ~10 bits, a 5-digit number → ~17 bits, etc.
* **Integration**: trivial, ~20 LOC. Reuses the same haystack
  scaffolding.
* **Why now**: lets us plot `eval_contains(K, entropy)` and see
  whether the wrapper has a smooth capacity curve or a hard
  cliff. Either result is informative.

### T1.3 — Same-form distractors (synthetic, semantic stress)

* **What it tests**: whether the wrapper relies on surface form
  (`def foo_xyz():`) or semantic content. If the haystack contains
  10 functions named `foo_*` returning different values and the
  query names one of them, the wrapper has to use the *name* to
  pick.
* **Integration**: ~40 LOC, reuses `_categorical_needle_chunk` but
  fills the distractors with same-shape functions.
* **Why now**: if the wrapper collapses here even at k=1, then
  it's not using attention over chunk content correctly — strong
  diagnostic.

## Tier 2 — real text (1–2 days each, requires HF download)

### T2.1 — QASPER (scientific paper QA)

* **HF**: `allenai/qasper`
* **Format**: paper + question + answer (free-form or yes/no).
* **Length**: ~5K tokens per paper (manageable).
* **Why**: closest real analog to coding_niah — single document,
  fact-extraction question. Lets us see if the wrapper helps on
  real text the way it (partially) does on synthetic.

### T2.2 — QuALITY (long-doc multiple choice)

* **HF**: `emozilla/quality` (mirror)
* **Format**: ~5K-token Gutenberg passage + 4-way multiple choice.
* **Length**: ~5K tokens.
* **Why**: matches categorical_niah's 4-way structure on real
  text. Direct comparability — if wrapper hits 0.42 on synth-4-way
  and X on real-4-way, the gap tells us about distribution shift.

### T2.3 — RepoBench-C (line completion)

* **HF**: `tianyang/repobench-c-8k` (or the 16K variant)
* **Format**: repo files as chunks + last lines as context +
  predict the next line.
* **Length**: 8K or 16K tokens.
* **Why**: matches the plan's stated coding-first focus
  ([v0 plan §model-assumption](v0-learned-memory-wrapper.md)).
  This is what the wrapper would actually need to handle for the
  coding-RCA specialization.

## Tier 3 — comprehensive suites (after tier 2 lands)

### T3.1 — RULER

* **HF**: `simonjegou/ruler` (or generate locally from upstream).
* **What**: 13 controlled long-context subtasks at chosen
  sequence length (4K → 128K). Includes single-needle,
  multi-needle, multi-key, multi-value, multi-query, common-words
  extraction, variable-tracking, and frequent-words tracking.
* **Why**: drop-in replacement for our hand-rolled NIAH variants
  with an established eval protocol. Lets us publish numbers
  comparable to other long-context papers.
* **Cost**: model-side eval at 32K+ tokens needs gradient
  checkpointing or chunked attention; current full-forward is OK
  to 8K-16K.

### T3.2 — LongBench (v1 first)

* **HF**: `THUDM/LongBench`
* **What**: 6 task categories, 21 datasets, 4K-30K context.
  Includes single-doc QA (NarrativeQA, Qasper, MultiFieldQA),
  multi-doc QA (HotpotQA, 2WikiMQA, MuSiQue), summarization,
  few-shot, synthetic, code (LCC, RepoBench-P).
* **Why**: standard reference numbers. Helps position the wrapper
  against published baselines (e.g. KV-cache compression methods,
  retrieval-augmented inference).
* **Cost**: 21 datasets is a lot — start with the 3-4 most
  relevant (NarrativeQA, MuSiQue, RepoBench-P, MultiFieldQA-en)
  to keep eval-time manageable.

### T3.3 — HELMET (2024)

* **GitHub**: princeton-nlp/HELMET
* **What**: newer comprehensive long-context eval, controls many
  axes that LongBench v1 conflates (length, task type, instruction
  vs in-context learning).
* **Why**: forward-looking; if the wrapper publishes, this is
  the eval reviewers will ask for.

## Tier 4 — memory-specific (where the wrapper framing really shines)

These are the benchmarks that *specifically* test what the wrapper
is supposed to do — maintain a compact updateable memory across
sessions, not just compress a single long context.

### T4.1 — LongMemEval (2024)

* **GitHub**: `xiaowu0162/LongMemEval` (or HF mirror)
* **What**: multi-session conversational memory eval. 5 question
  types: single-session-user, single-session-assistant, multi-
  session-temporal, knowledge-update, abstention.
* **Why**: directly matches the wrapper's positioning. Lets us
  claim "this is a wrapper for long-term memory" with evidence.
* **Cost**: medium — need to set up multi-session evaluation
  harness; the wrapper.update interface fits this naturally.

### T4.2 — MSC (Multi-Session Chat, Xu et al. 2022)

* **HF**: `facebook/personality_chat` or via ParlAI mirror.
* **What**: 5-session personalization. Each session adds new
  facts about the user; the model needs to remember across.
* **Why**: simpler than LongMemEval; good starter.

### T4.3 — PerLTQA (Chinese long-term QA)

* **HF**: `RUC-NLPIR/PerLTQA`
* **What**: long-term personal QA across multiple sessions.
* **Why**: complementary to LongMemEval; Chinese-language coverage
  if relevant to downstream telecom-RCA users.

## Tier 5 — coding-specific (plan's stated specialization)

These align with the "later specialize to coding RCA / telecom RCA"
direction in the v0 plan.

### T5.1 — CrossCodeEval

* **GitHub**: `nyu-mll/CrossCodeEval`
* **What**: function-completion that requires looking up symbols
  defined in *other* files of the same repo.
* **Why**: harder than RepoBench-C because the relevant context
  is genuinely distributed across files — exactly the setting
  where compressed memory could save cost.

### T5.2 — SWE-bench Lite (or Verified)

* **HF**: `princeton-nlp/SWE-bench`
* **What**: real GitHub issues + repo, judged by passing tests.
* **Why**: the ultimate downstream — if the wrapper helps here,
  it's actually useful, not just a benchmark exercise.
* **Cost**: high. Probably skip until after tier 4. Mentioned for
  completeness so the path is visible.

## Concrete next-step ordering

After the current ``stages-…`` run finishes:

1. **Tier 1 in one batch** (multi-needle + numerical +
   same-form-distractors). All synthetic, ~½-day, one diff.
2. **One sweep on those three** to characterize how SFT / OPD /
   RL recipes generalize across difficulty.
3. **T2.2 QuALITY** as the first real-text test — directly
   comparable to categorical_niah.
4. **T3.1 RULER** at 4K and 16K — gets us into "real long
   context" territory with a published protocol.
5. **T4.1 LongMemEval** — pivot to the multi-session story that
   actually matches the wrapper framing.
6. **T5.1 CrossCodeEval** — pivot to the coding-RCA story.

Tiers 3.2 / 3.3 / 5.2 are reserved for if/when there's something
publishable to compare.

## Implementation notes

* `LongContextItem` is the universal contract — every adapter
  emits one. New adapters live next to `generate_coding_niah` in
  `llm_infra.datasets` or as separate modules
  (`llm_infra.datasets_real`?) if HF deps grow.
* `train_smoke.py` only needs the `--dataset` choices extended;
  the rest of the harness is dataset-agnostic.
* For real-text datasets that ship as JSONL: extend
  `iter_items_jsonl` so we can checkpoint generated splits to
  disk and not regenerate.
* For HF datasets: cache via `datasets` library defaults, mount
  the cache dir on `sam-dev` (already there at
  `~/.cache/huggingface`).
