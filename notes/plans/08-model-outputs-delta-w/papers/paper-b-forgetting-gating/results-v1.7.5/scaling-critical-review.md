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

## 5b. Results (2026-06-15 13:10; sweep 69/80, search benches bfcl_live_multiple + bfcl_multiple, avg over both)
- **H2 (capacity K scales?) — REJECTED.** compress vs K (live_multiple): K16 0.688 / K32 0.677 / K64 0.688 /
  K128 0.646 / K256 0.635 — flat / slightly **down**, never up. The untested capacity axis also does NOT scale.
  *Critical review: PASS (n=96; monotone non-increase).*
- **H4 (minimize scale) — SUPPORTED.** `enc4` (avg 0.429) ≥ `enc16`/base (0.371); `K16` = `K64`. The smallest
  config (N≈4, K≈16) matches the big one at ~10× less compute → minimize-scale is the cure.
- **H5 (OOD-induced ceiling) — preliminary REJECTED.** `manif01` avg 0.401 ≈ base 0.371 (on live_multiple it is
  even *lower*, 0.635 vs 0.688); the on-manifold fix does **not** break the ceiling. *manif10 / mnorm landing —
  will confirm.* So "capacity-bound" survives; OOD is a symptom, not the fixable cause.
- **★ Best HP = lr 3e-4** (avg over the 2 search benches **0.450** vs base **0.411** on the *same* 2 benches —
  base lr1e-4 was a touch low). lr5e5 0.423 / lr2e4 0.414 also beat base. Next non-lr: `ch16a` 0.430, `enc4`
  0.429, `distill10` 0.421. *Critical review: on 2 benches, n=96; live_multiple 0.75 vs base 0.69 is ~1σ —
  suggestive, not decisive; comparison must use the **common** benches (the analyzer's raw `avg` mixes bench-counts
  — `base` was over 6 benches incl. the floored hotpot/narrativeqa, so its raw 0.371 is NOT comparable). Confirm lr3e4 on more benches.*
- **λ_distill is load-bearing** — `distill0` collapses to 0.021 (vs base 0.69) on live_multiple.
- **`ab_min` N×K extreme-min grid — PENDING** (runs after the sweep, ~2h).
> Net: "capacity-bound" **survives** (depth, capacity, AND the OOD-fix all fail to scale/rescue). The wins are
> **tuning** (lr↑ to 3e-4) and **minimize-scale** (N=4/K=16), not more scale. Hold for manif10/mnorm + ab_min.

## 5c. HP sweep data table (compress; sweep 69/80; in-task on the 2 search benches)
`avg` = mean over the two benches all configs share (bfcl_live_multiple, bfcl_multiple); each cell n=96. Sorted by `avg`.
| config (Δ vs base) | live_multiple | multiple | **avg** | gate AUROC |
|---|---|---|---|---|
| **lr3e4** (lr 3e-4) | 0.750 | 0.150 | **0.450** | 0.65 |
| ch16a (len-adaptive) | 0.760 | 0.100 | 0.430 | 0.53 |
| enc4 (N=4) | 0.708 | 0.150 | 0.429 | 0.61 |
| lr5e5 (lr 5e-5) | 0.729 | 0.117 | 0.423 | 0.54 |
| distill10 (λ_dist 1.0) | 0.708 | 0.133 | 0.421 | 0.59 |
| ch32a | 0.698 | 0.133 | 0.416 | 0.59 |
| lr2e4 (lr 2e-4) | 0.677 | 0.150 | 0.414 | 0.68 |
| **base** (enc16/K64/lr1e-4/lora32) | 0.688 | 0.133 | 0.411 | — |
| dev20 / ga4 / lora64 / lora16 / enc8 | ~0.66–0.71 | ~0.10–0.15 | 0.403–0.408 | 0.58–0.67 |
| **manif01** (A2 OOD-fix, H5) | 0.635 | 0.167 | 0.401 | — |
| K256 / K32 / initrand / K16 / enc28 | 0.64–0.69 | 0.08–0.15 | 0.384–0.393 | 0.63 |
| dev0 / ga16 / lora0 / K128 | 0.65–0.69 | 0.08–0.12 | 0.365–0.382 | — |
| **distill0** (no distill) | **0.021** | 0.117 | **0.069** | — |

Reading: **lr3e4 tops** (0.450 vs base 0.411); the **OOD-fix (manif01) sits mid-pack at base level** (H5 reject);
**bigger K is worse** (K128/K256 near the bottom); **distill0 collapses**. (manif10/mnorm/rec/the other-domain
columns + the `ab_min` extreme-min grid still landing — table fills out then.)

## 5. What the 10h report will answer
1. Does capacity ($K$) scale on a learnable bench (H2)? — decides "minimize K" vs "scale K".
2. Is the ceiling OOD-induced (H5)? — decides whether "capacity-bound" survives or becomes "fix-the-OOD".
3. The **minimal Pareto config** (smallest N,K at plateau acc) — the "minimize scale" cure if scaling is flat.
4. Best HP overall (for the v1.7.5 headline).
