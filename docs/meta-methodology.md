# Empirical Research Meta-Methodology

> Execution discipline for *running* an empirical ML study to a paper-grade
> result. Distilled from how we actually work (plans 08 / 09), not aspirational.
>
> Companion to [`workflow.md`](workflow.md). Division of labor:
> **`workflow.md` = how to choose what to work on** (framing → brainstorm →
> read → plan). **This file = how to run it honestly once it is running**
> (measure → metric → robustness → report). Read `workflow.md` first.

---

## 0. The one loop everything reduces to

```
measure (broad, intrinsic, native)
   → find the load-bearing / trustworthy subset
      → act only on that subset
         → verify it holds vs baselines, across seeds AND model families
            → report the whole grid, not the cherry-pick
```

Every principle below is a way of not cheating at one of these five steps.

---

## 1. Honesty first — claim only what survives the strongest fair test

- **Reframe to the strongest *true* claim; never inflate to the desired one.**
  When OURS and Gist came out statistically tied on QuALITY (paired *t*-test on
  strict Phase-Y data), the headline became the **3-regime characterization**,
  not "we beat the baseline." A correct smaller claim outranks a wrong big one.
- **Fairness of settings is part of the result.** Matched compute, matched data,
  matched eval protocol, matched token budget. An unfair win is a bug, not a win.
- **A "verifier" must not be the same family as the thing under test** (carried
  from `workflow.md` §3). Self-graded numbers are not evidence.

## 2. Measure before you build (measure-first)

- Run the **observation study before designing the method**, and **gate the
  method on what the observation finds.** Plan 09 is structured this way (H1/H2
  measured before any protection mechanism); plan 08 v1.5 ran the intrinsic
  **signal study before** building the gate.
- This is cheap insurance: if the coupling/signal is weak, you learn it in the
  cheap phase instead of after building a method on a false premise.

## 3. Native metrics & faithful evaluation

- **Use each benchmark's own metric and its own eval protocol — read the
  benchmark paper.** QuALITY/MuSR are likelihood-scored multiple-choice (not
  free-gen + regex); SQuAD-style uses EM/F1 with the unanswerable case;
  NarrativeQA/MS-MARCO use ROUGE-L/BLEU over references.
- **Watch for evaluation artifacts.** A "signal" that is really a parsing
  artifact (free-gen letter-matching) must be quarantined and labeled as such,
  not reported as a finding.

## 4. Multi-granularity, multi-angle measurement

- Measure the *same* phenomenon at several **granularities**: first-token,
  answer-span, per-layer, layer-curve derivatives (early/mid/late band, slope),
  and module-level (memory slot / head / expert).
- Measure from several **angles**: confidence/uncertainty, divergence-from-base,
  representation geometry/drift, agreement, relevance, causal influence (ablate
  and re-measure). Correlate broadly first, then prune — you cannot find a
  non-obvious correlate you never computed.

## 5. Filter to the non-obvious (soundness × interestingness)

- Discard correlates that are **trivial or near-tautological** with the target
  (e.g. raw output confidence ≈ correctness). Promote the **non-obvious** ones
  that look unrelated at a glance but survive analysis (mid–late-layer
  wrapper↔base divergence and residual-stream drift as *inverse* predictors of
  help). Rate every candidate on a soundness × interestingness grid; keep
  KEEP/SUPPORT, drop FILTER.

## 6. Rank, visualize, tabulate — show the whole grid

- Rank correlates high→low by **effect size** (AUROC vs a binary label,
  Spearman vs a continuous lift), and **show the distribution** (ranked bar
  chart), not just a top-3 list.
- **Export the big grids as CSV** (tidy long form + wide pivots). The reader
  gets the full angle × metric × benchmark × model grid, so they can audit the
  cherry-pick that you didn't make.

## 7. Explain every metric (a codebook, not a name)

- For each metric record: **the exact setting it was computed under**, its
  definition/formula, its range, and a **rationale for why it should matter**
  plus how to read it. A number with no setting and no mechanism is noise.

## 8. Robustness = multi-seed AND cross-model reproduction

- A correlation is not real until it survives **multiple seeds** (single-seed
  KL "signals" turned out to be artifacts once pooled across 9 seeds) **and**
  **replicates across model families.** Reproduce on a different family
  (Qwen → GLM) and a newer/heavier arch (Qwen3.5 MoE); keep only the signals
  that are **consistent** (same direction, meaningful magnitude in all).
- Compare only **architecture-comparable** quantities across models: scalars and
  fractional-depth layer-curve features travel; raw absolute-layer-index
  features do not (depths differ, e.g. 36 / 40 / 32 layers).

## 9. Anchor every result to baselines, controls, and a ceiling

- A number means nothing alone. Always report against the **trivial baselines**
  (no-op / unmodified / random / no-context) and, where definable, an **oracle
  ceiling** — the contribution is the *measured gap* you close between them, never
  an assumed one.
- **Ablate each added component** against a no-op control so its effect is
  isolated, and keep the control in the reported grid.

> *Project note (not a meta principle):* "do-no-harm / frozen-base gating" is a
> plan-08/09 **research thesis** (a response to catastrophic forgetting), not a
> general methodology rule. It lives in §12 (worked instances) and the plan docs,
> not here.

## 10. Compute discipline

- **Use all available accelerators**; parallelize across seeds/benchmarks
  (one seed per GPU). **Isolate fragile environments** (a separate venv /
  transformers-`main` + torch upgrade for a bleeding-edge arch) so you never
  break the working env or the runs in flight.
- When a fast-path kernel won't cooperate, a **correct slow fallback beats a
  fast wrong path** — validate correctness first, optimize second. Manage
  memory explicitly (detach/clone, free, periodic cache clears); kill by PID,
  never by a pattern that also matches the launcher.

## 11. Paper-grade reporting + a living ledger

- Write the **paper-level report as you go** (settings, metric definitions,
  interpretation, results) — not at the end. Keep a **living matrix/ledger**
  with status + results + queue.
- **Archive superseded versions with dates**; one source of truth per fact
  (status in the TOC, citations in `knowledge-sources.md`, settings in a
  settings registry). Date every accumulating list.

---

## 12. Worked instances — the same loop, two axes

The methodology is real because two separate plans instantiate the *identical*
loop on opposite sides of the project's $X \leftrightarrow W$ thesis:

| | **Plan 08 v1.5** — read-time gating | **Plan 09 (Janus)** — write-time protection |
|---|---|---|
| Axis | $X$: adapt the *inputs* (soft-prompt memory) | $W$: constrain the *weight updates* (SFT) |
| Threat to do-no-harm | wrapper degrades the frozen base on OOD inputs | SFT forgets pretrained code/math/general |
| Intrinsic signal | wrapper↔base KL, confidence, drift, mem-geometry, causal influence | retrieval-head / sink-mass / super-expert score, SFT hidden-drift |
| Load-bearing object | per-**input** trust (when to open the gate) | per-**site** importance (what to freeze/mask) |
| Action | gate the wrapper per input | protect the sites during SFT |
| Verify | keep Regime-A gains, no harm on B/C, across seeds + models | retention up at matched new-domain acc, across domains + models |

They are **siblings, not duplicates**: same discipline, different object and
mechanism. Cross-reference their cohort measurements; do not merge them.

---

## 13. Documentation, reporting & evidence hygiene

Distilled from how we present results (plan 08 v1.5). The goal: a reader or
listener can follow **every claim to its evidence** in one or two clicks.

1. **One wiki page decodes every term.** Every label on a table/slide/figure
   (route, signal, setting ID, experiment ID, metric) is defined **once, with a
   link**, in a glossary. If a term isn't in the glossary, it doesn't go on a slide.
2. **A settings registry is the single source for exact recipes.** Result cells
   cite a **stable setting ID** (e.g. `P08-S7`); hyperparameters live there and are
   never repeated locally. New result ⇒ new/updated setting entry.
3. **Refer details OUT of slides.** A slide is for a *human*: capture the **core
   logic** — *why it matters → why this approach → expected effect → real result*
   — with ≤ a few bullets and one figure/number per slide; end each with
   "→ details: \<doc/setting\>". Never dump tables a person can't read in 10 s.
4. **Annotate every claim with its basis.** Keep a **claim ↔ evidence** table:
   claim → the number → the file → the setting. Same for motivations. No claim
   ships without a pointer to the data that supports it.
5. **Explain metrics high-level, defer the math.** State how to *read* each metric
   (what higher/lower means, what chance is, why it supports the claim) in the
   wiki; put formulas / ranges / derivations in the settings/codebook.
6. **Readable assets index.** Group artifacts by **purpose** (start-here /
   headline results / figures / docs / scripts / data / archives) with a "what it
   gives you" per row — not a flat path dump.
7. **Result-logging convention.** Every run ⇒ flip the status, persist CSV+figure
   to the dated grids dir, add it to the assets index. Raw per-item data stays on
   the pods; pulled summaries live in-repo.
8. **Negative results are first-class.** Keep failed ablations *with the reason*;
   they are what justify the design choices.

## Execution checklist (run at the start and end of every empirical study)

- [ ] Is the metric the benchmark's **native** metric, scored its own way?
- [ ] Are baselines **fairly matched** (compute / data / tokens / eval)?
- [ ] Did I **measure before building** the method?
- [ ] Did I sweep **multiple granularities × angles** before pruning?
- [ ] Did I **filter trivial** correlates and surface the **non-obvious** ones?
- [ ] Is every reported metric in a **codebook** (setting + definition + why)?
- [ ] Does each finding survive **multi-seed** and **cross-model** reproduction?
- [ ] Did I report against **trivial baselines** (+ an **oracle ceiling** where definable) and **ablate** each added component?
- [ ] Did I export the **full grid** (CSV + ranked chart), not a cherry-pick?
- [ ] Did I log the session in the **matrix** and archive superseded versions with dates?
