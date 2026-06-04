# Janus — Phase-0 exploratory run (Qwen3-8B, 2026-06-03)

> First real artifact for Plan 09. **Forward-only** site detection on the *actual target model* (Qwen3-8B, eager attn, bf16, GPU0 shared with v1). Code: [`runs/phase0_detect.py`](runs/phase0_detect.py). Raw: [`runs/phase0_qwen3-8b_2026-06-03.json`](runs/phase0_qwen3-8b_2026-06-03.json). ~35 s wall, ~16 GB, did not disturb v1.

## Phase-0 gate: **PASS** — all three intrinsic-site phenomena reproduce

| phenomenon | published expectation | Qwen3-8B (this run) | verdict |
|---|---|---|---|
| **BOS attention sink** | a chunk of heads dump ~all mass on pos 0 | 684 head-positions with sink-mass > 0.5; **L13H12 = 1.00**, L7H6/L9H28/30/31 ≈ 0.99 | ✓ |
| **Massive activations** (Sun 2024: ~3–5 channels) | a handful of residual channels ≫ rest | **exactly 5** channels ≥ 6× median; **ch 2276 = 12096 (≈410× the 29.5 median)**, ch 233 = 2352, ch 1838 = 992 | ✓ clean |
| **Retrieval heads** (Wu 2024: sparse <5%, mid-late) | a few heads spike on the needle value | sparse ✓; **L23H4 = 0.63**, L29H15 = 0.57, L21H23 = 0.56; cluster in **L17–31**, layer 23 is a hub (H4/H10/H12/H13) | ✓ |

Detectors are trustworthy → Phase-1 can proceed on this instrumentation.

## The site inventory for Qwen3-8B (what to protect, by granularity)

- **Sink / streaming heads** → **early-mid layers L7–L13** (L13H12 is a perfect sink).
- **Retrieval heads** (the long-context **read-side** site) → **mid-late layers L17–L31**, hub at **L23**.
- **Massive-activation channels** (residual-dim granularity, a separate axis) → **{2276, 233, 1838}** dominate; ch 2276 alone is ~410× median.

## The bonus finding (cheap, no training) — sink heads ≠ retrieval heads

H2-precondition probe: overlap of the top-15 sink heads vs top-15 retrieval heads.

- **Jaccard = 0.0** (completely disjoint sets).
- **Spearman(sink-mass, retrieval-score) over all 1152 heads = 0.09** (≈ no relation).
- Sink heads live in L7–13; retrieval heads live in L17–31 — **different layers, different heads**.

→ **This confirms DuoAttention's dichotomy on Qwen3-8B** and has a direct consequence for the method: **the long-context-load-bearing site is the *retrieval heads*, NOT the sink.** The sink is a separate functional object (attention stabilization / streaming). So:

- The **read-side protection criterion = retrieval-head score** (not sink mass). Sink-only tuning (the dead Path B, brainstorm §5.5) was never going to carry long-context anyway — this is empirical confirmation.
- The brainstorm's "sink ↔ super-expert ↔ massive-activation" axis is the **stabilization** site; the "retrieval-head" axis is the **content-routing** site. Janus's coupling claim (H2) should be tested **primarily on retrieval heads** (+ massive-act channels as the dense analog of super-experts).

## Refined exploration map (what to run next, in cost order)

The decisive next test needs SFT (base→tuned drift); everything above was free. Ordered by cost / information:

1. **[needs 1 small SFT] H2 core — the predictive coupling.** Take Qwen3-8B, record retrieval-head scores + massive-act channels (done). LoRA/full-SFT on one domain (RCA or code), then measure per-head **ΔW / activation-drift / Fisher**. Test: **does the read-side retrieval-head ranking predict the write-side disruption ranking?** (Spearman ρ). This is the H2 de-risk and the claim that beats `[mech-forget]` (their set is post-hoc ΔW; ours is a-priori read-side). *Gate: ρ ≥ 0.4 → continue.*
2. **[needs the SFT'd ckpt] capability drop attribution.** Measure ΔGSM8K/ΔMMLU after SFT and correlate the drop with disruption of the *retrieval* heads specifically (ablation: post-hoc restore the top-disrupted retrieval heads → how much capability returns? mirrors `[mech-forget]`'s 47% but keyed on the read-side set).
3. **[1 protected-SFT run] H3 preview.** Freeze the top-k retrieval heads (criterion = read-side score) during SFT → measure Δforgetting **and** Δlong-ctx vs unprotected SFT and vs the `[mech-forget]` ΔW-set baseline. *Gate: ≥30% less forgetting, no long-ctx regression.*
4. **[MoE, the headline] super-expert version.** Repeat 1–3 on **Qwen3-30B-A3B**: detect super experts (sink-induction), protect them during SFT. This is the least-crowded novelty (§5.6a) — make it the headline instantiation.
5. **Detector hardening** (per `[ret-dyn]`): retrieval heads are context-dynamic → average the retrieval score over ≥10 needle variants + positions before trusting the ranking; report stability.

## Compute note
All forward-only; ran alongside v1 on GPU0 with ~75 GB free. Steps 1–4 need short SFT runs (~10 GPU-h each) — schedule **after** v1 frees the cluster, or on the H200 once the IN/LoCoMo chain drains.
