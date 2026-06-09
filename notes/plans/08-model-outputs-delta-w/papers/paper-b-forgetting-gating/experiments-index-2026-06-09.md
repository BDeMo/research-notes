# Experiments index (new version): the logical directory (2026-06-09)

> The canonical map of every experiment in the do-no-harm latent-compression paper, organised by the
> LOGICAL argument (not by script). Each entry: what it is, the script/cells, its logical role, status,
> and the headline result. Status: [DONE] [RUN]ning [ANALYSIS]. House rule: no em-dashes.

## The spine (one question)
Framing = **v1.6** (`v1.6-framing-2026-06-09.md`): robust context compression for agentic tool-use;
detect a broken compression EARLY from intermediate states and fall back to full context (harness
circuit breaker). Scenario (1) known knowledge location (LoRA/MoE); Scenario (2) unknown (frozen-base
lightweight compressor + relevance read). The parts below are the experiments under this framing.
Question: per input, *use* the compressed context or *fall back* (full context, or base = the do-no-harm floor)?
Argument flow:  I setting -> II problem (compression can hurt) -> III keystone (why the floor is
non-trivial) -> IV mechanism (the gate) -> V generality -> VI efficiency -> VII ablations -> VIII honesty.

---

## Part I -- The setting (what we study)
- **I.1 Compression baselines: Gist-lite + Cartridge-lite.** `mem_baselines.py --mode {gist,prefix}`.
  Cells `gx_*`, `mb_*`. ROLE: the literature compressors we gate. STATUS [DONE] (6 models x 7 benches).
- **I.2 Our learned compressor: mem-X wrapper.** `probe_v3.py --combine xattn` (K=64 latents).
  Cells `cmb_xattn_*`, `ours_*`. ROLE: our compressor; also the OURS row of the main table. [DONE] (7 models).

## Part II -- The problem (compression can hurt)
- **II.1 Cross-model x cross-compressor HARM MAP.** From `gx_*` + `cmb_*`; analysed by `gate_policy.py`.
  ROLE: show compression hurts in a compressor/model-specific way. [DONE].
  RESULT: Gist always-compress −9% across models; GLM-MoE concat −53%; our wrapper on Mistral −136%.
- **II.2 When/how much it hurts: capacity + combine.** `gk_*` (gist K-sweep), `cmb_{mode}_*` (read modes),
  `kc_/kabl_/cfg_*`. ROLE: harm vs K latents and vs injection style. [RUN] (gk_) / [DONE] (cmb_).

## Part III -- The KEYSTONE (why the do-no-harm floor is non-trivial)
- **III.1 Forgetting contrast.** `mix_sft.py --signals` (LoRA on task A, eval held-out B!=A no-ctx);
  cells `fgt_*`; tool `forget_table.py`. ROLE: the floor is what weight-editing LACKS. [DONE] (20 cells).
  RESULT: LoRA held-out forgetting **q8 −0.083 / glm −0.076 / q25 −0.069 / mistral −0.066** (worst cell
  −0.62; MC-format training catastrophic −0.18..−0.22). **Ours = +0.000 by construction.** This answers
  the single most direct reviewer doubt ("isn't the floor trivial?"): no, it is the dividing line vs
  fine-tuning. See `critical-review-v2-priority-2026-06-09.md`.

## Part IV -- The MECHANISM (the do-no-harm gate)
- **IV.1 Gate vs trivial policies + feature ablation.** `gate_policy.py` on `gx_/sig_/cmb_`.
  Compares never(=base) / always / gate{TARG, +compressed-response, +rich} / oracle, 5-fold CV + risk-cov.
  ROLE: is the gate better than trivial; do our signals beat cheap base-uncertainty. [DONE].
  RESULT: gate restores do-no-harm but recovers only 15-51% of oracle headroom; rich signals do NOT beat
  cheap TARG; risk-coverage near-flat. The gate is a SAFETY mechanism, not an accurate predictor.
- **IV.2 Signal-correlation study (can any signal predict per-input harm?).** `sig_*` (full base-readable
  menu via `base_signals.py`), `signal_corr.py`. ROLE: rigorously test predictability. [RUN]/[DONE-partial].
  RESULT (so far): per-input harm is hard to predict; cheapest base-uncertainty is as good as the full menu.
- **IV.3 Two-track gate.** `gate3_route.py`: Track A (memory<->no-ctx = do-no-harm), Track B (memory<->full
  = efficiency). 5-fold CV, LOMO transfer. ROLE: the two deployment modes. [DONE]. LOMO AUROC 0.71.

## Part V -- GENERALITY (the gate is portable)
- **V.1 Cross-compressor generality.** `gx_*` + `gate3_route --general` (compressor-agnostic features).
  ROLE: the SAME cheap gate works across Gist/Cartridge-lite/ours. [DONE].
  RESULT: do-no-harm held under the agnostic gate (Cartridge-lite 33/35, Gist 26/35 cells).
- **V.2 Knowledge-MoE routing.** `mem_baselines.py --moe` + `moe_route.py`: route over {null, source_1..k},
  one memory per chunk. ROLE: generalise the gate to a do-no-harm MoE router with a guaranteed null expert.
  [DONE] (5 models). RESULT: do-no-harm holds 4/4; GLM null-rescue (concat −53% -> gate +24%); source-
  selection at chance (~0.25/4). Reaffirms: the value is the null expert, not routing accuracy.

## Part VI -- EFFICIENCY (why compress at all)
- **VI.1 Track B cost-coverage.** `gate3_route.py` Track B (mem_tokens vs full_tokens). ROLE: answer
  "why not always full context." [MEASURED]. RESULT: compressible tasks match/beat full at 7-36% of tokens
  (trivia 0.265@24%, quality 0.273@7%, musr 0.533@9%); extractive tasks defer to full (~100%), no loss.

## Part VII -- ABLATIONS (design justification)
- **VII.1 Multi-layer injection.** `probe_deep.py`: depth `n-inject {1,2,3,4,6}`, placement `top-N`,
  gate-mode {none,scalar,mlp}, `alpha`, `d-inj`. Cells `dl_/dlt_/dlg_/dla_/dld_*`. [RUN] (~118 cells;
  Phi excluded: DynamicCache bug).
- **VII.2 Combine-mode (read side).** `cmb_{xattn,residual,hybrid,interleave,prefix}_*`. [DONE] (7 models).
- **VII.3 Capacity (K latents).** `gk_* / kc_ / kabl_ / cfg_*`. [RUN]/[DONE].
- **VII.4 Transfer / capability-boundary.** `xf_*` + `grid_transfer.py` (train-dataset x eval-all). ROLE:
  where the wrapper helps (in-dist) vs hurts (OOD); also the empirical "ours preserves held-out" side of III.1.
  [RUN] (~45 cells).

## Part VIII -- HONESTY / limits (a feature, not a bug)
The gate is weak (recovers <51% of headroom; rich signals do not help; MoE source-select at chance). We
state this openly: **per-input harm is hard to predict, so the contribution is the architectural floor +
a cheap conservative gate + the cross-compressor/MoE generality + the no-forgetting advantage, NOT gate
accuracy.** P3 seed-CIs (significance) and P4 (stronger compressor / online gate / decoy) are deferred.

---

## Dependency / reading order for the paper
1. III.1 forgetting (why the floor matters) -> 2. II.1 harm map (why a decision is needed) ->
3. IV.1-IV.3 the gate (how we decide) -> 4. V generality + VI efficiency (it ports + it pays) ->
5. VII ablations -> 6. VIII honesty. Analysis tools: `gate_policy.py`, `gate3_route.py`, `moe_route.py`,
`forget_table.py`, `signal_corr.py`, `grid_transfer.py`. Review docs: `critical-review-2026-06-08.md`,
`critical-review-v2-priority-2026-06-09.md`, `experiment-matrix-2026-06-08.md`.
