# Scaling: critical review + minimize-scale investigation (2026-06-15)

> Triggered by: "if scaling is useless, can we minimize scale? and critically review the scaling code/design."
> Rule: no dogmatic conclusions — every hypothesis gets an experiment + a critical review that must pass.

## 0. The claim under review
v1.7.4 §8 concluded **"the compressor does not scale"** (avg compress over 12 benches flat ~0.26 across 24×
FLOPs / 8× data / 7× depth). Before trusting this, audit the experiment.

## 1. Critical review of the scaling experiment (`queue_mixscale.py` + `analyze_scale.py` + `_mix`)
| # | issue | severity | how tested |
|---|---|---|---|
| C1 | **Bench-average masking**: the flat 12-bench avg can hide per-bench trends. | high | per-bench re-analysis (done, §2) |
| C2 | **Capacity axis untested**: the grid varied depth $N$ and data $D$ but **fixed $K{=}64$**. Capacity is the most likely scaling lever and was never swept. | **high** | K-sweep single-bench (HP sweep) + N×K grid |
| C3 | **Curriculum data-axis confound**: $D$ takes the easy→hard *prefix*, so "more data" = *harder* data; flat $D$-scaling could be gains cancelled by harder examples. | med | fixed-difficulty single-bench data-scaling |
| C4 | **Mixture confound**: ONE compressor trained on 12 heterogeneous domains is mediocre everywhere regardless of scale. | med | single-bench (per-domain) scaling vs mix |
| C5 | FLOPs formula (6ND, per-layer≈211M) | low | affects x-axis only, not the flatness |

## 2. Per-bench re-analysis (C1) — evidence
**Depth (D=384):** scaling is FLAT or **negative** — `N=4` (smallest) is best/tied almost everywhere
(bfcl_live_multiple 0.42→0.21 as N: 4→16; glaive 0.75→0.56). **Data (N=16):** only `bfcl_live_multiple`
scales (0.15→0.27 over 8× data); floored benches (toolace 0.0–0.04, rca ~0.10, narrativeqa ~0.13) never move.
**Noise floor** at n=48, p≈0.2 is ±~0.07; the depth gap (Δ0.21) and the live_multiple data trend (Δ0.12)
exceed it → real; the flat/floored ones are flat within noise → real. *(Critical review: PASS; the minimize
grid uses n=96 to tighten.)*

## 3. Hypotheses (each: test + critical review)
- **H1 — "no scaling" is an averaging artifact.** *Test:* §2 per-bench. *Result:* PARTLY true — avg hid
  depth-scales-DOWN and one bench's data-scaling, but floored benches genuinely don't scale. *Review:* PASS (above).
- **H2 — capacity $K$ is the real scaling axis (untested).** *Test:* HP sweep K∈{16..256} single-bench + the
  N×K grid. *Review to pass:* must be on a **learnable** bench (ctx>K, headroom) and single-domain (no mix/curriculum);
  if compress rises toward full with K ⇒ scalable-via-capacity (then do NOT minimize K); if flat ⇒ truly bounded. *Status: running.*
- **H3 — mixture is the confound.** *Test:* single-bench K/N scaling vs the mix grid on the same bench.
  *Review:* if single-bench ≫ mix ⇒ mixture confound; if both flat ⇒ not it. *Status: running.*
- **H4 (the cure) — minimize scale: smallest $(N,K)$ is Pareto-best.** *Test:* N×K grid (N∈{2,4}, K∈{8..64}) on
  learnable benches; smallest config retaining plateau acc, at lower cost. *Review:* already supported for depth
  (N=4 best); K-min depends on H2 (can only minimize K if K doesn't scale). *Status: running (`ab_min_*`).*
- **H5 — the ceiling is OOD-induced, not capacity.** *Test:* HP sweep `m-manifold-temp`/`m-norm-match` (the A1/A2
  on-manifold fixes) vs base. *Review to pass:* if the OOD fix lifts compress substantially ⇒ the ceiling is an
  OOD/training issue, NOT fundamental capacity (would *weaken* the capacity-bound claim — report honestly); if no
  lift ⇒ OOD is a symptom, capacity-bound stands. *Status: running.*

## 4. Experiments running (5 GPUs, ~10h)
- **HP sweep** (`aa_hp_*`, free+test): K/N/lr/LoRA/OOD-fixes/losses single-bench → H2, H5, and the best HP.
- **Minimize-scale N×K grid** (`ab_min_*`, free): N∈{2,4}×K∈{8,16,32,64} on bfcl_live_multiple + bfcl_multiple → H4.
- Per-bench re-analysis (`analyze_scale.py` extended) → H1 (done).

## 5b. Preliminary results (2026-06-15 10:25; sweep 49/80, search benches only)
- **H2 (capacity K scales?) — REJECTED.** On bfcl_live_multiple (the bench with headroom), compress vs K:
  K16 0.688 / K32 0.677 / K64 0.688 / K128 0.646 / K256 0.635 — flat / slightly **down**, never up (Δ within
  noise). So the untested capacity axis also does NOT scale. *Critical review: PASS (n=96; monotone non-increase).*
- **H4 (minimize scale) — SUPPORTED.** `enc4` 0.708 ≥ `enc16` 0.688 ≥ `enc28` 0.667; `K16` = `K64`. The
  smallest config (N≈4, K≈16) matches the big one at ~10× less compute → minimize-scale is the cure.
- **Bonus:** λ_distill is load-bearing — `distill0` collapses to 0.021 (vs base 0.69) on live_multiple.
- **H5 (OOD-induced ceiling) — PENDING** (manif/mnorm configs not yet run; the potential overturner).
- **`ab_min` N×K grid (extreme min N2/K8) — PENDING** (runs after the sweep).
> Net so far: "capacity-bound" survives (depth AND capacity flat); the cure is minimize-scale. Hold for H5.

## 5. What the 10h report will answer
1. Does capacity ($K$) scale on a learnable bench (H2)? — decides "minimize K" vs "scale K".
2. Is the ceiling OOD-induced (H5)? — decides whether "capacity-bound" survives or becomes "fix-the-OOD".
3. The **minimal Pareto config** (smallest N,K at plateau acc) — the "minimize scale" cure if scaling is flat.
4. Best HP overall (for the v1.7.5 headline).
