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

## MoE-specific angles + audit (added 2026-06-03, §11)

User: "再看看针对 MoE 模型有什么可以利用的点." Motivated — plausible RCA bases are MoE (Qwen3-30B-A3B / 235B-A22B, DeepSeek-R1, Llama-4). Brainstormed 9 MoE angles (M1-M9) + same-day novelty search (6 queries).

- **Honest scoping**: MoE sparsity is in the FFN → MoE is a *forgetting* lever, NOT a long-context lever (long-ctx bottleneck is attention/KV, shared across experts).
- **Forgetting side is saturated**: M3=`[loramoe]` (ACL 2024), M4=`[esft]` (EMNLP 2024), M8=`[same-moe]` (2026), multi-domain=`[des-moe]` (EMNLP 2025), M2/M6 standard, M1 examined-and-inferior. **None novel.**
- **The one genuine find — Super-Expert ↔ attention-sink identity**: `[super-experts]` (2507.23279) shows a *tiny* set of experts **induce the sinks**; pruning them → sink decay >90% → model collapse. `[sink-native-moe]` (2602.01203): sink weight = implicit gating → heads are a native MoE. **This is the cleanest empirical anchor for the §10.3-1 unifying-observation paper**: long-context load-bearers = forgetting-vulnerable sites. Super-expert detection is task-free / HF-native (Qwen3-MoE).
- **Surviving angle `M★`** (narrow gap, must verify): select the *protected* expert set by the **super-expert / sink-induction criterion** (not task-affinity like ESFT/DES-MoE) for fine-tuning-without-forgetting. `[super-experts]` studies compression only → fine-tuning use is unclaimed. Needs a dedicated "super-expert + fine-tuning + forgetting" search before committing.
- **Subsumption win**: dense base → protect sinks/massive-act; MoE base → protect super experts (which *are* the sink inducers). One principle, two instantiations.

Logged 8 IDs to `knowledge-sources.md` §"MoE prior-work audit".

## Prioritization under refined constraints (added 2026-06-03, §12)

User refined: method should be **data-agnostic + transformers-intrinsic + light training OK but ideally NOT task-specific training**. Applied as a filter to all R/M angles:

- The constraint moves the *novel contribution* from "train an RCA adapter" → "a data-agnostic intrinsic rule that *constrains/protects* training". Every trained-task-specific-module angle (R1/R2/R3/R8/R10) demotes to **baseline** (and was preempted anyway).
- **Program collapses onto ONE P0 thesis**: *the intrinsic sites carrying long-context behavior (sinks/massive-act dense; super-experts MoE) are the same sites whose perturbation causes forgetting → detect them on generic text (data-agnostic, training-free) and protect them (freeze/grad-mask) during any SFT → long-ctx retention + no code/math forgetting, zero task-specific params added.*
- Satisfies all three constraints; only-surviving novelty = (1) site-selection criterion (sink-induction, not task-affinity / not raw grad-magnitude) + (2) the joint long-ctx↔forgetting claim with dual eval. Main threat `[mech-forget]`.
- **Next actions (cheap-first)**: (1) **P0c de-risk experiment — NO RCA data needed**: on Qwen3-8B + Qwen3-30B-A3B, detect sites on generic text, run small proxy-domain SFT, correlate site-shift with ΔGSM8K/ΔHumanEval (forget) and ΔRULER (long-ctx). (2) targeted "super-expert + fine-tuning + forgetting" search. (3) if positive → draft `notes/plans/09-intrinsic-site-protection/`.
- **Honest caveat**: under no-task-training, the *novel* long-context leg is thin (training-free sink-KV ≈ StreamingLLM); forgetting + the unifying observation carry the paper.

## Reorganization (2026-06-03, conclusion-first)

The source file had grown into a chronological pile (brainstorm → dLLM → §10 dense audit → §11 MoE audit → §12 prioritization), with the conclusion buried at the bottom under now-superseded optimistic sections. Rewrote it **conclusion-first**, no facts lost:
- §1 TL;DR (verdict + P0 thesis box + priorities + next action + caveat) — read-first.
- §2 problem (+ refined D/I/T constraints) · §3 unifying lens.
- §4 brainstorm: R1–R12 merged into ONE table (score + closest prior + verdict), M1–M9 table.
- §5 audit summary + narrow survivors + Super-Expert↔sink bridge + two open paths.
- §6 P0 in depth (constraint fit, super-expert instantiation, prioritization table, next actions, long-ctx caveat).
- §7 pre-audit material clearly banner-marked **superseded** (recipes / paper-story / Nokia bridge).
- §8 dLLM · §9 refreshed decision questions · §10 references.

## Output artifacts

- `notes/ideas/rca-transformer-intrinsic-2026-06-03.md` (**REORGANIZED conclusion-first** — full 2-round brainstorm + audit + prioritization, T2)
- `notes/ideas/README.md` — new Sources row + "RCA model building — 12 ideas" section + audit verdict banner + MoE-audit banner
- `docs/matrix/knowledge-sources.md` — new "RCA prior-work audit" (~28 IDs) + "MoE prior-work audit" (8 IDs) sections
- This entry + matrix index row

## Knowledge sources to chase (not yet in knowledge-sources.md)

Xiao 2023 (attention sinks / StreamingLLM) · Sun 2024 (massive activations) · Olsson 2022 (induction heads) · Liu 2024 (lost-in-the-middle) · Arditi 2024 (refusal direction) · Tuned Lens (Belrose et al.) · LISA. `[revive]` already in knowledge-sources.md.

## Next steps

1. **User decides among the 3 honest paths** (source §10.4): (A) unifying-observation paper, (B) verified-new-phenomenon (sink key-bias), (C) RCA application paper. The "novel single mechanism" path is closed by the audit.
2. If (B): dedicated lit search on "sink key-bias / key-bias-only fine-tuning" before committing.
3. If (A): sharpen the joint long-ctx↔forgetting observation vs `[mech-forget]`; design RCA+code+math co-eval; pick known levers as components (OPLoRA-style W-side + persistent-memory X-side).
4. On lock: draft `notes/plans/09-<slug>/` (or plan-08 v3 sub-folder) with validation / channels / budget.
