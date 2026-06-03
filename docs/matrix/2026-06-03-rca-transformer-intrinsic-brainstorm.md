# 2026-06-03 — RCA model building brainstorm (transformer-intrinsic adaptation)

> Session goal: capture the user-posed RCA problem + two rounds of brainstorm into research-notes ideas. Same day as the mem-X v1 harvest; this is the *forward-looking* counterpart (what to build next) to that *backward-looking* harvest.

## Activities

- Reframed the two RCA challenges (long-context inference + training-time catastrophic forgetting) under one lens: "where is information allowed to flow in a transformer".
- Round 1: 5 angles (sink-anchored adaptation, layer-band LoRA + orthogonal constraint, per-head temperature, read-side KV reweighter, massive-activation-aware FT).
- Round 2: widened to a phenomenon→problem coverage map + 12 scored angles (R1–R12) + 3 composition recipes (Light/Medium/Heavy) + RCA-specific free toppings + anti-patterns + decision questions.
- Used mem-X v1's 3-regime law as a constraint prior: RCA ≈ Regime C (verbatim needle) → favor overlay-existing-KV/sinks over add-new-soft-tokens.

## Decisions

- Captured as a **new ideas topic** (distinct from X-W inference-time-training), source file `notes/ideas/rca-transformer-intrinsic-2026-06-03.md`, IDs **R1–R12** (R-prefix to avoid collision with A–J and the tentative K1–K3 proposals).
- No recipe locked yet — 5 open decision questions recorded for the user (mechanism scope, verbatim hard-req, experiment scope, relation to mem-X v1, Nokia release boundary).
- Anchor recommendation logged: Medium recipe, "single-mechanism unified paper" around **sink-anchored adaptation (R1)** + induction-head freeze (R2) + per-head temperature (R4).
- **AR → dLLM extension** added as source §8 (future room, not now): anchor narrative on the architecture-agnostic principle, abstract "site" as a `site_selector`, favor architecture-agnostic levers (R4/R8/R7), use infilling aux loss in AR version (= native dLLM objective). dLLM is natively better at Regime C + has a free denoising-steps compute knob. Nudges recipe away from over-relying on AR-specific R2. Added decision question #6.

## Prior-work audit (added 2026-06-03, after a 2022-2026 lit review)

User asked for "no similar work / not hand-wavy" → ran a 4-year arXiv/Scholar review across all 12 R-angles. **Verdict: NONE of R1-R12 is novel as a standalone mechanism; the 2025-2026 literature is saturated.** Kill list (closest prior in `knowledge-sources.md` §"RCA prior-work audit"):

- **R8** = `[oplora]` OPLoRA (AAAI 2026) verbatim · **R10** = `[smf]` Sparse Memory Finetuning (Meta 2025) verbatim · **R4** = `[ssa]`/`[ssmax]`/`[focal-attn]` · **R7** = `[steer-reason]`/`[instruct-steer]`/`[cache-steer]` · **R3** = `[reasoncache]`/`[kv-packet]`/`[rlkv]` · **R5** = `[selfaug]`/`[tmkl]`/`[logit-lens-loss]` · **R6** = `[mofo]`/`[migu]` · **R2** = `[mech-forget]`/`[ft-no-forget-icl]`/`[abft]` · **R11** = `[sae-ft]`/`[sae-fd]` · **R9/R12** = CALM/LayerSkip area · **R1** = `[persist-mem-dec]`/`[kvm]`/`[reasoncache]` (only a thin unclaimed core).

**Revised decision**: drop the "novel mechanism" framing entirely. Two honestly-open paths only:
1. **Unifying-observation paper** — same intrinsic sites solve long-ctx AND forgetting, RCA+code+math co-eval (main threat: `[mech-forget]` already localizes forgetting to attention heads → differentiate via the *joint* framing + RCA testbed).
2. **Verified-new-phenomenon** — sink **key-bias-only** tuning (`[sink-emerge]` frames sink as key bias; not directly found in review; needs a dedicated search before trusting).

Logged 28 prior-work IDs to `knowledge-sources.md`.

## Output artifacts

- `notes/ideas/rca-transformer-intrinsic-2026-06-03.md` (**NEW** — full 2-round brainstorm + §10 prior-work audit, T2)
- `notes/ideas/README.md` — new Sources row + "RCA model building — 12 ideas" section + audit verdict banner
- `docs/matrix/knowledge-sources.md` — new "RCA prior-work audit" section (~28 IDs)
- This entry + matrix index row

## Knowledge sources to chase (not yet in knowledge-sources.md)

Xiao 2023 (attention sinks / StreamingLLM) · Sun 2024 (massive activations) · Olsson 2022 (induction heads) · Liu 2024 (lost-in-the-middle) · Arditi 2024 (refusal direction) · Tuned Lens (Belrose et al.) · LISA. `[revive]` already in knowledge-sources.md.

## Next steps

1. **User decides among the 3 honest paths** (source §10.4): (A) unifying-observation paper, (B) verified-new-phenomenon (sink key-bias), (C) RCA application paper. The "novel single mechanism" path is closed by the audit.
2. If (B): dedicated lit search on "sink key-bias / key-bias-only fine-tuning" before committing.
3. If (A): sharpen the joint long-ctx↔forgetting observation vs `[mech-forget]`; design RCA+code+math co-eval; pick known levers as components (OPLoRA-style W-side + persistent-memory X-side).
4. On lock: draft `notes/plans/09-<slug>/` (or plan-08 v3 sub-folder) with validation / channels / budget.
