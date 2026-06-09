# Experiment: gate generality across compressors (bridges the orthogonality gap)

> Motivation: our contribution (the do-no-harm gate) is orthogonal to the compressor.
> A reviewer will ask: you showed the gate on YOUR compressor; does the gating signal
> generalize to OTHER compressors? This experiment answers yes by running the
> identical gate on multiple, distinct compression methods. Designed + launched 2026-06-08.

## Hypothesis
A do-no-harm gate built on a **compressor-agnostic** signal set achieves do-no-harm
(Track A) and good compress-vs-full routing (Track B) for **multiple distinct
compressors**, on multiple base models. If so, the gate is a property of the
frozen-base setting, not of our specific compressor.

## The compressor-agnostic signal set (the crux)
`delta_last` (our wrapper's write magnitude) is wrapper-specific, so it cannot be the
generality signal. Instead we gate on signals every compressor can produce:
- **base uncertainty** (no-context draft; TARG-style): `conf_0, margin_0, entropy_0`.
- **compressed-response** (how the frozen base reacts to the compressed tokens):
  `conf_w, margin_w, entropy_w` (first-token uncertainty WITH the compressed context).
- **w-vs-0 divergence**: `kl_w0` (shift the compressed context induces), `first_tok_agree`.

These read only the base's logits given `[compressed tokens ; query]`, so they are
identical in form for Gist, Cartridge-lite, and our wrapper. `FEATS_GEN` in
`gate3_route.py` is exactly this set; `--general` uses it for BOTH tracks.

## Design
- **Compressors (3 families):** Gist-lite (`mem_baselines --mode gist`), Cartridge-lite
  (`--mode prefix`), and our recurrent wrapper (`probe_v3_gate`, xattn/hybrid).
- **Bases (5):** Qwen3-8B, GLM-4-9B (sam-dev); Qwen3-14B, Qwen2.5-7B (ray); Mistral-7B (test).
  Mistral is the key cell: does the gate do-no-harm even on the base where compression hurts?
- **Benches (7):** trivia_qa, squad_v2, hotpot_qa, narrativeqa, ms_marco, quality, musr_mm.
- **Per item, all compressors log:** `native_0/w/full` + `FEATS_GEN` signals + token costs.
- **Gate:** `gate3_route.py --general` per (compressor, model) -> Track A (do-no-harm: gated
  >= base) + Track B (acc @ tok%), using the identical agnostic gate everywhere.

## What we changed (implemented 2026-06-08)
- `mem_baselines.py`: added `comp_unc(mem,qid)` -> logs `conf_w/margin_w/entropy_w/kl_w0/
  first_tok_agree` under `--gate` (baselines previously logged only base-uncertainty).
  Our wrapper already logs all of these.
- `gate3_route.py`: added `FEATS_GEN` + `--general` (agnostic feature set for both tracks).

## Runs (launched; ids `gx_<mode>_<model>_<bench>`, 70 cells)
gist+prefix x {q8,glm} (sam-dev, 28) ; x {q14,q25} (ray, 28) ; x {mistral} (test, 14).
"Ours" reuses existing `cmb_*` outputs (already carry FEATS_GEN). These `gx_*` runs also
serve as the proper two-track baseline cells the lit review asked for.

## Analysis (when they land)
For each compressor: `gate3_route.py --glob 'runq_out/gx_<mode>_<model>_*' --general`.
Build a table: rows = compressor x model, cols = Track-A do-no-harm rate (cells gated>=base),
Track-B acc, Track-B tok%. Compare to the same gate on `cmb_*` (ours).

**Success criterion (the bridge):** the same agnostic gate holds do-no-harm and recovers
most harm for Gist, Cartridge-lite, AND ours, across bases. Then: "the do-no-harm gate is
orthogonal to the compressor; we demonstrate it on three compression families." Secondary:
`delta_last` (ours-only) adds a margin over `FEATS_GEN` for our wrapper (`--general` off vs on).

## Honest risks
- If the agnostic gate works on all compressors, it partly de-novelties `delta_last` (the
  wrapper-write signal): the generality comes from the agnostic signals, not our signal.
  Frame `delta_last` as an optional improvement for our compressor, not the load-bearing part.
- Track B uses cheap base-uncertainty for honest cost; `conf_w` needs one forward over the
  compressed tokens (cheap, amortized) -> note this in the cost accounting.
