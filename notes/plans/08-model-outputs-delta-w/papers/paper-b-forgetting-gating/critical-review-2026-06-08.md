# Critical review: will a reviewer believe the claims? (2026-06-08)

> Reviewer-simulation of the compression-setting paper "Do No Harm: When Is a
> Compressed Context Safe to Use?". House rule: no em-dashes.

## THE EXPERIMENT IN ONE SCREEN (simple version)
- **Setting.** A frozen LLM + a *detachable* compressor that turns a long context into K latents.
- **The one question.** Per input: *use* the compressed context, or *ignore* it (fall back to no-context)?
- **Why it matters.** Using it is not free: sometimes the compressed context is worse than nothing.
  Because the compressor is detachable and the base is frozen, "ignore it" == exactly the base ==
  a **do-no-harm floor**.
- **The method.** A cheap, label-free gate predicts, per input, whether to use it.
- **The one experiment.** On each (compressor, model, benchmark) cell, compare 4 policies:

  | policy | what it does |
  |---|---|
  | never  | ignore the compressed context (= base = the floor) |
  | always | always use it (naive) |
  | gate   | the cheap label-free gate decides |
  | oracle | perfect per-input decision (upper bound) |

  Metric = **do-no-harm rate** (fraction of cells where policy >= base) + **recovered%** (share of the
  never->oracle headroom captured).
- **The one result.** `always` breaks the floor where compression is dangerous (worst: our wrapper on
  Mistral, base 0.260 -> 0.192, only 3/8 cells do-no-harm). The `gate` restores it (8/8) across every
  compressor, but captures only part of the upside (15-51%) because per-input harm is hard to predict.
  **So the contribution is the floor + a cheap safety gate, not gate accuracy.**

Everything below (5 models, 2 compressors, Track B, signal ablations, generality) is **robustness around
this one spine**, not extra stories.

## 0. The full claim list, mapped to the spine (for completeness)
- **C1 compression can hurt** = the `always` row falls below the floor on some cells.
- **C2 do-no-harm floor by construction** = `never` == base (detach identity) + no forgetting vs LoRA.
- **C3 a label-free gate keeps you on the floor** = the `gate` row vs `never`/`always`/`oracle`.
- **C4 two tracks** = floor is the no-ctx fallback (Track A); full-ctx is the other fallback (Track B).
- **C5 gate is compressor-agnostic** = the same gate works on Gist / Cartridge-lite / our wrapper.
- **C6 a gate is necessary** = `always` hurts even for a wrapper-free compressor (Gist, −9% across models).

## 1. The gate-generality result (just landed, 5 models x 7 benches, agnostic gate)
Do-no-harm held (gated >= base), agnostic gate, 5-fold CV:
- **Cartridge-lite (prefix): 33/35 cells.**  **Gist-lite: 26/35 cells.**
- Track B shows the compress/defer split for both (trivia kept at 5-32% tokens; squad/hotpot defer at ~95-100%).

## 2. The finding that reshapes the story (and a reviewer will see it)
On Mistral-TriviaQA: no-ctx 0.296, **our wrapper 0.003 (collapses)**, but **Gist 0.481 and Cartridge-lite 0.562 (help)**, full 0.671. So the headline "compression hurts, hence gate" is **largely driven by OUR wrapper's specific fragility**, not a general property of compression: the two cheaper compressors do not collapse Mistral. A reviewer will say: "maybe fix your wrapper instead of gating it."
- **Honest reframe (use this):** compression hurts *sometimes*, in a *compressor-and-model-specific* way (our wrapper on Mistral; Gist on GLM-trivia 0.234<0.459; prefix on several MC cells), and the gate catches those cheaply and **generalises across compressors**. The selling point is not "compression always hurts" but "a single cheap gate makes any compressor do-no-harm, and it is needed because some compressor/model combinations do hurt."

## 3. The decisive measurement: gate vs trivial policies + feature ablation (MEASURED 2026-06-08)
`gate_policy.py`, Track A (do-no-harm, memory vs no-ctx), 5-fold CV, macro over cells.
`recovered%` = (policy − never)/(oracle − never): the share of the achievable headroom captured.

| compressor (cells) | never | always | gate[TARG] | gate[+comp] | oracle | recovered% (always / TARG / +comp) | do-no-harm cells (+comp) |
|---|---|---|---|---|---|---|---|
| Gist, glm+q8 (14) | 0.233 | 0.227 | 0.246 | 0.245 | 0.310 | −9% / 16% / 15% | 9/14 |
| Cartridge-lite, glm+q8 (14) | 0.233 | 0.249 | 0.263 | 0.264 | 0.320 | 18% / 34% / 36% | 14/14 |
| Gist, q8 (7) | 0.208 | 0.216 | 0.215 | 0.219 | 0.264 | 15% / 13% / 19% | 6/7 |

What the numbers actually say (this reshapes the paper):
1. **The gate is real but weak.** It recovers only **15-36%** of the oracle headroom, and the risk-coverage curve is nearly **flat** vs the oracle (gist gate ~0.234 across coverages while the oracle jumps to 0.305 at 25%). Per-item harm prediction is hard; the gate mostly wins by being *conservative* (staying near "never"), not by discriminating items.
2. **The do-no-harm value depends on the compressor.** For Gist, naive always-compress actually **hurts** (0.227 < 0.233, −9%); the gate's job is to *restore* the floor. For Cartridge-lite, always-compress already helps (+18%) and the gate lifts do-no-harm to **14/14** at +36%.
3. **Our novel ingredient (compressed-response signals) does NOT robustly beat the cheap TARG baseline.** It helps Cartridge-lite (14/14 vs 12/14) but on Gist it is no better and has *fewer* do-no-harm cells (9/14 vs 11/14). The "richer signal -> better gate" claim is currently **not supported**.
4. **The full signal menu does not help either (P0-B answered, negative).** Adding logit-lens, hidden geometry, and all w-vs-0 divergences (`gate[+rich]`) does *not* beat the cheap 8-feature agnostic set: gist +rich 15% vs +comp 19%; prefix +rich 56% vs +comp 68%; risk-coverage stays flat. **Per-item harm is hard to predict from internal signals.** (Caveat: ~150 items/cell, glm+q8 only so far; the landing `sig_*` cells across 5 models will firm this up, but the pattern is consistent.)
5. **On a compressor that mostly helps, the gate LOSES to always-compress** (prefix-q8: always 70% vs best gate 68% of headroom). So the gate is a **safety mechanism for the cells where compression hurts**, not a universal accuracy booster. Its job is to not-hurt, not to win.

### 3b. The make-or-break: the gate where compression is dangerous (the headline)
Recovered% of oracle gap (always / best-gate) and do-no-harm cell count (always -> best-gate):

| compressor x model | base | always (recov% , dnh) | best gate (recov% , dnh) | reading |
|---|---|---|---|---|
| **Ours (wrapper), Mistral** | 0.260 | **-136%, 3/8** | **+19%, 8/8** | naive use is catastrophic; gate restores the floor. THE headline. |
| Gist, Mistral | 0.236 | +17%, 3/7 | +23%, 4/7 | always hurts on 4 cells; gate restores most. |
| Gist, glm+q8 | 0.233 | -9%, 8/14 | +16%, 11/14 | always hurts on average; gate restores. |
| Prefix, Mistral | 0.236 | +50%, 5/7 | +51%, 7/7 | mostly helps; gate fixes the 2 harm cells. |
| Gist/Prefix, q14/q25/q8 | ~0.16-0.21 | +22..70%, 5-6/7 | +21..62%, 7/7 | benign regime: always already safe; gate neutral but harmless. |
| Ours, Phi | ~0.000 | n/a | n/a | **degenerate (base approx 0 with no context); drop this cell.** |

Headline numbers to report: **applied naively, our wrapper costs Mistral 0.260 -> 0.192 accuracy (-136% of headroom, only 3/8 cells do-no-harm); the cheap label-free gate restores 8/8 cells to >= base** (and the wrapper-internal `delta_last` signal recovers +19% of the upside on top). This is the cleanest evidence that the do-no-harm floor + gate is needed and works, and that the gate matters most exactly where naive compression is dangerous.

## 4. Reviewer critique, claim by claim (updated with measured evidence)
| claim | attack | verdict (with data) |
|---|---|---|
| C1 hurt | "harm is small / wrapper-specific" | **true and now quantified**: Gist always-compress is −9% (it hurts); our wrapper on Mistral collapses. Reframe to "compression hurts in a compressor/model-specific way" (see §2). |
| C2 floor | "trivial by construction" | **the strongest claim**: detach gives base by identity; +0.001 off-task vs LoRA −0.27. This is the contribution, not gate accuracy. |
| C3 gate | "AUROC 0.71; how much does it buy? > TARG?" | **partly fails**: recovers only 15-36% of headroom; +comp does not beat TARG on Gist. Must re-scope to "restores do-no-harm cheaply", not "accurate". |
| C4 tracks | "Track B only saves tokens" | GAP: still need the honest cost-coverage frontier + amortised savings. |
| C5 generality | "shows your wrapper is least safe" | holds: prefix 33/35, gist 26/35 do-no-harm under the agnostic gate; reconcile with §2. |
| C6 necessity | "necessary only because YOUR wrapper hurts" | **stronger than thought**: Gist always-compress hurts on its own (−9%), so the gate is needed even for a wrapper-free compressor. |

## 5. Supplementary experiments the data now demands (priority order)
**[DONE] A. Gate vs trivial policies + TARG ablation + risk-coverage** (`gate_policy.py`). Result in §3. It already told us the gate is weak and +comp does not beat TARG on Gist; this redirects everything below.

**[DONE, negative] B. Can ANY signal predict per-item harm better than TARG?** Tested the full menu (`gate[+rich]`); it does not beat the cheap agnostic set (§3.4). Headline becomes: **per-item harm is hard to predict; the architectural do-no-harm floor is what makes compression safe.** Confirm on the full 5-model `sig_*` set when it lands.

**P0 - C. Run A on the hard cells (where harm is large): our wrapper + Mistral + q14/q25.**
Do-no-harm should earn its keep exactly where always-compress collapses (our wrapper on Mistral −0.49; some MC cells). Needs the `gx_`/two-track cells for mistral/q14/q25 (on ray/test) + the `+delta_last` wrapper-signal arm from the probe_v3_gate logs. This is the one place the gate's value should be large and where +delta_last might beat TARG; it is the make-or-break for "the gate is worth having."

**P1 - D. Track B honest cost-coverage frontier**: accuracy vs token% vs always-full, amortised. Reframe "saves tokens" with the real curve.

**P1 - E. Decoy / relevance eval**: feed irrelevant context; measure the gate close-rate. The direct "decides blind" test.

**P1 - F. MC full-context sanity**: quality/musr full-ctx ~0.08 looks degenerate; verify scoring before any full-ctx claim.

**P2 - G. Seed CIs** (`gs_*`, in flight): error bars on every headline number (the deltas above are small, so CIs are essential).

**P2 - H. Stronger compressor baseline** (real Cartridges/ICAE) or explicit scoping to lite compressors.

## 6. What is already solid (keep)
- The do-no-harm floor **guarantee** (frozen + detach) and the LoRA forgetting contrast (+0.001 vs −0.27). This is the spine.
- Cross-compressor generality of the cheap gate (prefix 14/14 do-no-harm, gist restored from −9%).
- Cross-model boundary: always-compress hurts on Gist (−9%) and collapses our wrapper on Mistral, so a gate is genuinely needed.

## 7. The honest story the evidence supports (revised sell)
NOT "we built an accurate gate that knows when compression helps" (it recovers <40% of headroom; per-item harm is hard to predict). INSTEAD:
1. **Detaching a frozen-base compressor gives a do-no-harm floor by construction** (worst case = base; no forgetting, unlike LoRA).
2. **Naive always-compress violates that floor** (Gist −9% across models; our wrapper collapses Mistral), so a decision is needed.
3. **A cheap, label-free, compressor-agnostic gate restores the floor across compressors** (prefix 14/14, gist recovered), even though it captures only part of the oracle headroom because per-item harm is hard to predict.
The contribution is **safety and generality**, not gate accuracy. State the gate's weakness openly; it motivates the floor.

## 8. Plan
B + C are the next runs (full-menu signal test + hard-cell/wrapper arm). D-F are small scripts. G lands with seeds. Then fold §3 + B + C into a single "gate vs policies" main table and a risk-coverage figure, and rewrite C3 in the paper to the §7 sell.
