# Critical review v2: the most direct reviewer doubt + a priority-ranked logical plan (2026-06-09)

> Goal: answer the reviewer's MOST DIRECT question first, run the LOGICAL foundation before
> robustness/significance, and rank everything by importance. House rule: no em-dashes.

## THE single most direct reviewer doubt (the one that gets us rejected)
> "You detach a frozen-base compressor to get a 'do-no-harm floor' and add a gate to decide when
> to use it. But (a) the floor is trivial (of course detaching recovers the base), and (b) your OWN
> results show the gate is weak (recovers only 15-51% of headroom; rich signals do not beat cheap
> base-uncertainty; MoE source-selection is at chance). So this is a weak gate on a trivial mechanism.
> Why is it a paper?"

If we cannot answer this crisply, nothing else matters. Everything below is ordered by how directly
it answers this.

## The answer (what makes the work non-trivial), and the experiment that proves each
1. **The floor is the property that the standard alternative LACKS.** The natural way to inject
   knowledge/adapt a model is to fine-tune it (LoRA), which causes **catastrophic forgetting** of
   held-out capabilities. Compression into a *detachable* module is the only way to add context
   knowledge with a **guaranteed do-no-harm floor on everything else**. So "do-no-harm floor" is not
   trivial; it is the dividing line between modular compression and weight editing.
   -> PROOF = the **forgetting contrast** (LoRA forgets held-out tasks; our module is base-identical
      on held-out by construction). **This is the logical keystone and it is currently under-built.**
2. **Even granting the floor, naive use violates it**, in a compressor/model-specific way
   (Gist −9% cross-model, GLM-MoE concat −53%, our wrapper on Mistral −136%), so a deployment-time
   decision is genuinely needed. -> PROOF = the cross-compressor/model **harm map** + `gate_policy`
   (mostly DONE).
3. **Compression buys efficiency** (K=64 latents vs full context), and the gate preserves it.
   -> PROOF = **Track B honest cost-coverage** (NOT done; a direct "why not always full context?" Q).
4. **The honest scientific finding**: per-input harm is NOT predictable from internal signals
   (gate approx TARG; rich signals do not help; MoE source-select at chance). The practical recipe is
   therefore the architectural floor + a cheap conservative gate, NOT a clever predictor. This is a
   careful characterization with a clear takeaway. -> PROOF = the signal study + `gate_policy` + MoE
   (DONE). Frame the gate's weakness as a finding, not a failure.

Net reframed contribution: **a do-no-harm recipe for latent context compression that, unlike weight
editing, never forgets; a map of when compression hurts across compressors/models; and the honest
result that per-input harm is hard to predict, so the floor (not gate accuracy) is what to rely on.**

## CRITICAL CHECK (2026-06-09 pm): does the gate collapse? is it aggressive or conservative?
A reviewer's sharpest attack: if the gate degenerately always falls back (always-base or always-full), the
do-no-harm result is vacuous. Measured with `gate_policy.py` (now logs use-rate + precision/recall/F1):
- **Use-rate (frac routed to compressed) tracks help-rate**, so NOT globally collapsed: gist 28% use vs 27%
  help; prefix 44% use vs 32% help; spread min 0% / median 15-62% / max ~98% (high use on trivia, ~0% on
  quality). BUT **8-10/24 cells DO collapse** to always-base on low-help tasks (do-no-harm trivial there).
- **Aggressive vs conservative**: decision "use compressed" vs truth "compression helped (nw>n0)".
  gist precision/recall/F1 = 0.34/0.34/0.30 (balanced); prefix 0.46/0.49/0.38 (slightly aggressive,
  use>help). **Balanced but WEAK** P/R confirms per-input discrimination is poor (matches E1 AUROC ~0.57).
- **Rules now enforced:** (a) always report do-no-harm next to its use-rate (non-trivial only where use>0);
  (b) report use-rate vs dataset distance (in-domain high, OOD defers) via the `act_*` runs;
  (c) measure the realized below-base rate (the floor is conditional on the gate detaching; cell-level
  do-no-harm was 17/24, not 24/24); (d) report help/broke base-rate beside every do-no-harm number.
- **Scale fix (no subsampling):** big-N do-no-harm at n_eval=2000 (`big_*` runs) replaces the n=120-150 cells.

## PRIORITY-RANKED PLAN (run logical first; no seed-repeats yet)
**P0 RESULT [MEASURED 2026-06-09] (the keystone holds, answers the most direct doubt):**
LoRA trained on task A, evaluated no-context on held-out B != A (`forget_table.py`), vs ours (frozen+detach = +0.000 by construction):

| model | LoRA held-out forget Δ | worst cell | ours |
|---|---|---|---|
| q8 | −0.083 | −0.463 | +0.000 |
| glm | −0.076 | −0.577 | +0.000 |
| q25 | −0.069 | −0.473 | +0.000 |
| mistral | −0.066 | −0.623 | +0.000 |
| q14 | +0.077 (only trivia/squad trained) | −0.036 | +0.000 |

**Sharp finding:** forgetting is *task/format dependent*. Adapting to a new FORMAT (MC-quality) is
catastrophic for held-out generation: **−0.18 to −0.22 across ALL models (q8 −0.199, glm −0.189,
q25 −0.184, mistral −0.220), worst cells −0.46 to −0.62.** Adapting to a similar format (QA) forgets
mildly (−0.01 to −0.08), sometimes 0. **You cannot predict in advance whether your fine-tune task will
be catastrophic** (quality looks innocuous but wrecks the model), so the architectural do-no-harm
guarantee (ours = +0.000 always) is the safe choice. This makes the floor non-trivial: it is exactly
what weight-editing cannot give. (q14 +0.077 = it was only trained on the mild QA tasks; quality cell queued.)

**P0 (logical keystone): forgetting contrast, made solid.**
- `mix_sft.py --train-dataset A --eval-benches ALL8 --signals` trains LoRA on A and logs held-out
  no-ctx `native_sft0` (adapted) vs `native_0` (base) = forgetting on each B != A. Currently only
  glm/q8 x ~3 tasks. **Complete the forgetting matrix**: A in {trivia, hotpot, squad, narrativeqa,
  ms_marco, quality} x models {q8, glm, mistral, q25}. Each run = one matrix ROW (train A, eval all 8).
- Pair with our side (zero forgetting by construction): the transfer grid `xf_*` (train A eval all,
  module attached) already shows held-out approx base; the gate detaches on held-out. Analysis only.
- Deliverable: "LoRA forgets held-out by X (e.g. −0.05..−0.27); our module by 0.000 (construction)."

**P0.5 (reviewer will demand): forgetting-aware LoRA (O-LoRA/OPLoRA).** We must beat the *forgetting-aware*
baseline, not just vanilla LoRA. Needs an orthogonal-subspace reg added to `mix_sft.py` (code change).
Queue after P0; if time-boxed, at least cite + run vanilla LoRA solidly.

**P1 (logical): Track B honest cost-coverage / efficiency. [MEASURED 2026-06-09]**
`gate3_route` Track B on gx_gist_q8: the gate keeps the cheap memory where it suffices and re-reads full
context where it does not, so:
- compressible tasks: **match or beat full context at a fraction of the tokens** -- trivia 0.265 @ 24%
  tokens (full 0.204), quality-MC 0.273 @ 7% (full 0.200), musr 0.533 @ 9% (full 0.513). Memory BEATS
  full here (full context distracts; the compressed summary helps).
- extractive tasks: gate **defers to full** (squad 0.353 @ 100%, hotpot @ 86-100%, ms_marco @ 80%),
  never below full accuracy.
Answer to "why not always full context": on compressible tasks you save 64-93% of tokens at equal-or-
better accuracy; on the rest you pay full cost only when needed. This is arguably a stronger sell than
Track A. TODO: amortised cost (latents prebuilt once, reused), cross-model, into one frontier figure.

**P2 (logical): consolidate the harm map + the OURS main-table row** (analysis on `cmb_xattn_*` +
`gx_*` across 7 models). DONE-ish; assemble into one table.

**P3 (significance, LATER per user): seed CIs.** Deferred.

**P4 (robustness, LATER): stronger compressor (real Cartridges/ICAE), online gate, decoy/relevance.**

## Immediate actions
1. Queue P0 forgetting matrix at TOP priority (jumps ahead of the queued ablations).
2. P1 Track B cost-coverage: analysis now (no GPU).
3. P0.5 O-LoRA: implement next.
Order of execution = P0 -> P1 -> P0.5 -> P2 -> (P3/P4 later).
