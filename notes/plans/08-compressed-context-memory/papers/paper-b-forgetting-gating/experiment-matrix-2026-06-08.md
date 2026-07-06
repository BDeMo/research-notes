# Experiment matrix: everything to run + every reviewer question (2026-06-08)

> The full board for the compression + do-no-harm-gate paper. One spine (see
> `critical-review-2026-06-08.md`): per input, *use* the compressed context or *ignore* it
> (= base = floor); a cheap gate decides; contribution = floor + safety, not gate accuracy.
> House rule: no em-dashes. Status: [R]unning, [Q]ueued, [A]nalysis-only (no GPU), [B]uild-needed, [D]esign.

## 0. The spine experiment (DONE / consolidating)
- [A] **gate vs trivial policies** (`gate_policy.py`): never / always / gate{TARG,+comp,+rich} / oracle, 5-fold CV,
  + risk-coverage. Done on q8/glm/q14/q25/mistral x {gist,prefix} + our wrapper (cmb). Headline: ours-Mistral
  always −136% (3/8 do-no-harm) -> gate 8/8. Consolidate into ONE main table + risk-coverage figure.

## 1. Generality / breadth (fills the 7-model grid)
- [R] **Phi-3.5-mini** gate cells (gist+prefix x 7 benches) -> completes the 7th model (queued on ray, 2026-06-08).
- [R] **sig_** signal-correlation cells (full base-readable menu) across gist/prefix/lora/ours (sam-dev).
- [R] **gs_** seed-CI cells (270+ landed) -> error bars on every headline number.
- [Q] more seeds (7,11) for the gate cells on q8/glm/mistral -> CIs on the gate-policy table.

## 2. The MoE-knowledge angle (NEW; user request 2026-06-08)
**Idea.** Our gate is a 2-expert MoE router over {null (base), memory}. Generalise to N knowledge sources:
route over {null, source_1..k} where each source is a chunk/passage compressed into its own memory. The
**null is a guaranteed do-no-harm expert** (standard MoE has none). This reframes the gate as a do-no-harm
MoE router over knowledge inputs.
- [DONE] **knowledge-MoE routing** (`mem_baselines.py --moe` + `moe_route.py`, gist): per item one memory
  per chunk + null; route over {null, sources}; vs {null=base, concat-all=standard-gist, random, oracle-src}.
  **RESULT (5 models x {hotpot,ms_marco,trivia}; ruler dropped = gist destroys exact-retrieval, all 0):**

  | model | null | concat=gist | gate | oracle-src | do-no-harm | src-select |
  |---|---|---|---|---|---|---|
  | q8 | 0.122 | 0.142 | 0.131 | 0.164 | 3/4 | 0.25 |
  | **glm** | **0.198** | **0.153 (-53%)** | **0.218 (+24%)** | 0.217 | 4/4 | 0.23 |
  | q14 | 0.070 | 0.147 | 0.146 | 0.170 | 4/4 | 0.28 |
  | q25 | 0.105 | 0.164 | 0.165 | 0.228 | 4/4 | 0.23 |
  | mistral | 0.166 | 0.266 | 0.263 | 0.283 | 4/4 | 0.23 |

  **Reading (reaffirms the thesis, gives a 2nd headline):** (1) do-no-harm holds (4/4 on 4 of 5 models);
  (2) on **GLM** concat-gist HURTS (-53%, glm-trivia 0.459->0.263) and the MoE gate FALLS BACK TO NULL and
  recovers (+24%) = the null expert rescues a harmful compressor, like ours-Mistral; (3) elsewhere gate
  approx concat (routing adds nothing when concat is fine); (4) **source-selection is at chance** (~0.25 for
  4 sources) = the gate cannot pick the right knowledge source. Net: the contribution is the **guaranteed
  do-no-harm null expert**, not routing accuracy. Same lesson as the binary gate (per-input utility is hard).
- [D] **MoE base model robustness** (no MoE checkpoint local): optionally fetch OLMoE-1B-7B or Qwen3-30B-A3B
  and run the floor+gate pipeline on it -> shows do-no-harm generalises to MoE *architectures*, not just
  our 2-expert framing. Network-dependent; lower priority.

## 3. Track B (compression / inference-accel) honesty
- [A] **cost-coverage frontier**: accuracy vs token% vs always-full, amortised (mem prebuilt once, reused;
  fallback re-reads full). Extend `gate3_route` to sweep eps and emit the frontier. Reviewer: "why not
  always full?" -> answer with the curve + real savings.

## 4. Robustness / reviewer-anticipated baselines
- [B] **stronger compressor**: gist with K in {128,256} + longer training; a summarisation/LLMLingua-style
  prompt-compression baseline. Reviewer: "your compressors are lite; does harm + the gate survive on a
  strong one?" -> queue gist-K128 cells now (existing script); LLMLingua needs a small wrapper.
- [B] **forgetting-aware LoRA (O-LoRA / OPLoRA)**: we only beat vanilla LoRA. Add an orthogonal-subspace LoRA
  arm to `mix_sft.py` for the forgetting contrast. (lit-review gap; code change needed.)
- [A] **cross-compressor gate transfer**: train the gate on Gist, test on Cartridge-lite (and vice versa),
  on existing `gx_` data. Shows the agnostic gate is portable across compressors (C5).
- [Q] **K-capacity ablation** of the compressor: how harm/help vs K latents (existing `probe_v3`/`mem_baselines`).

## 5. Sanity / correctness (must-fix before claims)
- [DONE] **MC full-ctx sanity**: RESOLVED. For MC benches the real metric is `mc_acc_*` (option
  likelihood), which `gate_policy`/`gate3_route` already use; the `native_full approx 0` was the
  meaningless generation-score for MC (model emits free text, not a clean letter). RULE: never report
  `native_*` for MC; use `mc_acc_*`. Side-finding: on GLM-QuALITY `mc_acc_full` 0.247 approx `mc_acc_0`
  0.253, i.e. full context does not help MC quality (real result), so MC benches inform Track A
  (do-no-harm), not Track B.
- [A] **drop degenerate cells**: `cmb_hybrid_phi` (base approx 0 with no context).

## 6. Mechanism / extra (reviewer might ask, lower priority)
- [D] **online gate**: apply-or-skip during generation (current gate is offline routing on logged signals).
- [D] **selective-prediction / abstention** tie-in: do-no-harm gate vs confidence abstention baselines.
- [A] **per-bench harm taxonomy**: which task types compression hurts (MC vs exQA vs absQA) -> when to gate.

## Priorities (next)
P0: consolidate the gate-policy main table + risk-coverage (A) ; smoke + queue knowledge-MoE (2) ;
    seeds for CIs (1). P1: cost-coverage (3) ; stronger compressor + cross-compressor transfer (4) ;
    MC sanity (5). P2: O-LoRA arm ; MoE base ; online gate.
