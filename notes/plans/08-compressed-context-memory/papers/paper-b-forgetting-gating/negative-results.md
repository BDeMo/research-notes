# Negative results, null findings, and limitations (2026-06-09)

> Every below-expectation / null / scoping result in the paper, with the experiment, the
> expectation, the measured numbers, why it is negative, and a one-line **honest framing** that the
> writing agent can paste into the limitations or discussion. House rule: no em-dashes.
> Companion canvas: `~/.cursor/projects/Users-s1shi-workspace/canvases/negative-results.canvas.tsx`.
> Sources: `critical-review-2026-06-08.md`, `critical-review-v2-priority-2026-06-09.md`, `matrix.md`.

## The central negative (it reshaped the whole paper)

**Per-input harm is not predictable from internal signals.** This forced the pivot from "we built an
accurate gate that knows when compression helps" to "the architectural do-no-harm floor is what you
rely on; the gate is only a cheap conservative safety net." Every gate / signal / routing result
below is a facet of this one finding. The honest contribution is **safety and generality, not gate
accuracy**.

---

## 1. The gate is real but weak

- **Setup.** `gate_policy.py`, Track A (use compressed vs fall back to no-context), four policies
  (never / always / learned-gate / oracle), 5-fold CV, macro over cells. `recovered%` is the share
  of the never-to-oracle headroom captured.
- **Expected.** A learned gate, especially with the rich logit-lens and hidden-geometry signal menu,
  discriminates per input and recovers most of the oracle headroom.
- **Measured.**

  | compressor (cells) | never | always | gate | oracle | recovered% |
  |---|---|---|---|---|---|
  | Gist, glm+q8 (14) | 0.233 | 0.227 | 0.245 | 0.310 | 15% |
  | Cartridge-lite, glm+q8 (14) | 0.233 | 0.249 | 0.264 | 0.320 | 36% |
  | Gist, q8 (7) | 0.208 | 0.216 | 0.219 | 0.264 | 19% |

  - Rich signals do not help (the "P0-B negative"): the full menu does not beat the cheap 8-feature
    base-uncertainty set (Gist `+rich` 15% vs `+comp` 19%, with fewer do-no-harm cells).
  - Precision / recall / F1 is balanced but weak: gist 0.34 / 0.34 / 0.30, prefix 0.46 / 0.49 / 0.38.
  - 8 to 10 of 24 cells collapse to always-base on low-help tasks; realized cell-level do-no-harm was
    17/24, not 24/24.
  - On compressors that mostly help, the gate loses to always-compress (prefix-q8: always 70% vs gate
    68% of headroom).
- **Honest framing.** The gate is a safety mechanism, not an accuracy booster: it recovers only 15 to
  51% of the oracle headroom and wins by being conservative, not by discriminating items. We state
  this openly because it is exactly what motivates the architectural floor.

## 2. Breakage detection is low and flat (E1 + signal study)

- **Setup.** `early_curve.py` measures detection AUROC at increasing read budgets (prefill, first
  1/2/4/8 tokens, full decode); `signal_corr.py` ranks roughly 300 signals.
- **Expected.** Reading more of the compressed response sharpens detection, and the rich signal menu
  beats simple uncertainty.
- **Measured.** "Broke vs full" AUROC is approximately 0.57 and flat (prefill 0.583, full 0.567).
  "Broke vs base" is approximately 0.50 (chance). The best single QA signal is approximately 0.57.
  Logit-lens signals are model-specific and do not transfer across families; only `delta_last`, the
  memory-geometry family, and `mem_influence_span` generalize.
- **Honest framing.** Per-input breakage is close to undetectable from internal state on QA, and
  decoding further buys nothing. The one positive implication: reading early suffices, so a cheap
  prefill-time read is as good as an expensive full decode. Tool-use is the exception where detection
  works (BFCL `conf_w` AUROC approximately 0.78).

## 3. "Compression hurts" is not universal (rejection risk)

- **Setup / expected.** The motivating claim is "compression hurts, therefore you need a gate,"
  expected to hold broadly across compressors.
- **Measured (Mistral, TriviaQA).** no-context 0.296, our wrapper 0.003 (collapses), Gist 0.481
  (helps), Cartridge-lite 0.562 (helps), full context 0.671. Gist always-compress hurts only about
  9% on average; on many benign cells always-compress already helps.
- **Why negative.** The headline is largely driven by our own wrapper's fragility; the two cheaper
  compressors do not collapse Mistral. A reviewer will say "fix your wrapper instead of gating it."
- **Honest framing.** Reframe from "compression always hurts" to "compression hurts sometimes, in a
  compressor- and model-specific way, and a single cheap gate makes any compressor do-no-harm." The
  selling point is generality of the gate, not universality of the harm.

## 4. Knowledge-MoE routing: source selection at chance

- **Setup / expected.** Generalize the gate to a do-no-harm router over {null, source_1..k} and
  expect it to pick the correct knowledge source.
- **Measured.** Source-selection is approximately chance (~0.25 of 4). On GLM, concat-all-sources
  hurts about 53% (trivia 0.459 to 0.263); the gate correctly falls back to the null expert (+24%)
  but cannot rank which source is right. Gist drops to about 0 on exact-retrieval (RULER dropped).
- **Honest framing.** The contribution is the guaranteed null expert (plain MoE has none), not routing
  accuracy. This is the same lesson as the binary gate: per-input utility is hard, the floor is the asset.

## 5. BFCL tool-use caveats

- "Compression helps abstention" on the irrelevance subset is partly trivial: withholding the
  irrelevant tool causes abstention, the right answer for the wrong reason (information withholding,
  not reasoning). Frame carefully.
- High compressor-training variance: the same config (q8, gist) gave `bfcl_multiple` 0.30 vs 0.70
  across two runs on the approximately 200-item training set.
- In-domain `bfcl_multiple` train/eval overlap is not a clean held-out split; the OOD `live_multiple`
  and OOD-general runs are the clean evals.
- **Honest framing.** Report the abstention result with the trivial-baseline caveat, lead the
  tool-selection claim with the OOD `live` numbers, and report variance as a band.

## 6. RCA: the hardest, lowest-ceiling regime

- **Setup / expected.** Root-cause analysis from incident telemetry as an agentic long-context
  compression task; expected the frozen base to localize the root-cause service.
- **Measured.** Free-form RCA is beyond the frozen base (full-context primary-service match about 0.0
  to 0.15; it names wrong-system services or narrates). Even as multiple-choice selection, competing
  distractor incidents drown the signal:

  | # distractor incidents | mc_acc_full | vs chance (0.167) |
  |---|---|---|
  | 0 (single incident) | 0.575 | strong signal |
  | 2 (locked default) | 0.425 | clear signal, some length |
  | 6 | 0.067 | below chance, signal drowned |

- **Honest framing.** RCA is viable only after two redesigns: reframe to root-cause-service selection
  (MC, scored by log-likelihood) and cap distractors at 2. Even then the ceiling is modest (0.425 vs
  BFCL 0.92), so present RCA as a deliberately hard, low-ceiling regime and lean on the OOD-general
  runs as the clean test.

## 7. Method ablations that failed

- E5 soft-gate sweep fails: a per-token soft gate gets stuck at approximately 0.5 and its residual
  corrupts generation. Kept only as an ablation; the hard offline gate is what works.
- Thin multi-layer-injection gains and a bit-capacity wall: a fixed K=64 latent budget limits how
  much context can be carried, independent of injection depth.
- **Honest framing.** Report the soft-gate failure as an ablation that justifies the hard gate, and
  the capacity wall as the reason for the compress-or-defer (Track B) design.

## 8. Caught artifacts (NOT real negatives, do not report as findings)

- Qwen3-14B BFCL 0.41 was a generation-length artifact (reasoning preamble plus a 16-token cap clipped
  the function name); at 48 tokens it recovered to 0.55.
- MC full-context approximately 0.08 on quality/musr was a generation-scoring red herring; using
  `mc_acc` (log-likelihood) resolved it.
- `live_irrelevance` harm may be slightly inflated by the pre-fix loose substring scorer; re-run with
  the hardened word-boundary scorer is pending.

---

## The counterweight: the one strongly-positive keystone

The forgetting contrast is what makes the do-no-harm floor non-trivial and survives the central
negative. LoRA forgets held-out tasks by 0.066 to 0.083 mean (worst cells 0.46 to 0.62); fine-tuning
on a new format (MC-quality) is catastrophic across all models at 0.18 to 0.22. You cannot predict
which fine-tune task forgets, so the architectural guarantee (ours is +0.000 by construction) is the
safe choice. Lead with this result; the gate's weakness is the honest characterization that follows it.
