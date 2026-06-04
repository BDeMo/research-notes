# Plan 09 — Validation: observation study + method design

## Validation hypothesis (one sentence)

> *If the same intrinsic transformer sites carry long-context behavior and absorb the most damaging fine-tuning perturbations, then a ranking of sites by long-context importance will positively correlate with a ranking by forgetting-disruption, and protecting the top sites during SFT will reduce forgetting without hurting long-context ability.*

The whole plan tries to confirm or refute this. Phase 1 measures the correlation (H2); Phase 2 tests the causal protection (H3).

---

# Phase 0 — tooling (no task data)

Build three reusable instruments, validated against known results before any study:

1. **Site detectors** (data-agnostic, run on generic text — C4 / FineWeb / Pile slices):
   - `sink_detector`: per-(layer, head) attention mass on positions {0, 1, …, k} and on delimiter tokens; "sink score".
   - `massive_activation_detector`: per-(layer, channel) activation magnitude; flag outlier channels (≥ N× median) à la Sun 2024.
   - `retrieval_head_detector`: per-head copy/retrieval score on synthetic needle inputs (Wu et al. retrieval heads).
   - `induction_head_detector`: per-head prefix-matching score on repeated-token sequences (Olsson 2022).
   - `super_expert_detector` (MoE): per-expert sink-induction / activation-outlier score; ablate expert → attention-sink decay rate (per `[super-experts]`).
   - `singular_subspace`: per-weight-matrix top-k SVD subspace (for the W-side criterion).
2. **Drift meters** (compare base vs SFT'd checkpoint on a held-out generic set):
   - per-site ‖ΔW‖, per-site gradient accumulation, Fisher information, activation-CKA / cosine drift, KL(base‖SFT) attributed per layer via logit lens, singular-subspace rotation, routing-distribution shift (MoE), sink-mass change, massive-activation magnitude change.
3. **Intervention hooks**: freeze / grad-mask / orthogonal-project / KL-anchor a chosen site set during SFT; ablate (zero) a site at inference.

**Phase-0 gate**: detectors must reproduce published phenomena (BOS sink, ~3–5 massive-activation channels, a handful of retrieval heads, MoE super experts) on Qwen3-8B + Qwen3-30B-A3B. If not, fix instrumentation before proceeding.

---

# Phase 1 — the observation study (the scientific core)

> Goal: a **coupling map** — for each model, the set of sites that are *both* long-context-load-bearing *and* forgetting-vulnerable — plus the correlation/causal evidence (H1, H2). **Be broad**: many sites, many granularities, many metrics, many conditions. Coverage is the point.

## 1.1 Where we look — the SITE axis (levels × granularity)

| Level | Granularity | Candidate "site" |
|---|---|---|
| Token position | per-position | sink positions (BOS/first-k), delimiter/newline tokens |
| Attention head | per-(layer,head) | sink heads, retrieval heads, induction heads |
| Layer | per-layer / layer-band | early / middle-third / late; the inter-layer U-curve |
| Channel / neuron | per-channel (residual dim) | massive-activation channels, outlier dims |
| Residual direction | per-direction | capability / task directions (diff-in-means, PCA) |
| Expert (MoE) | per-expert | super experts, shared expert, routed experts |
| Weight subspace | per-matrix top-k SVD | dominant singular subspace (general-ability carrier) |
| Module | per-(Q,K,V,O,gate,up,down) | which projection concentrates the coupling |

Every metric below is computed **at every applicable granularity** so we can see *which granularity* the coupling lives at (head? channel? expert? subspace?). That answer dictates the Phase-2 mechanism.

## 1.2 What we measure — the METRIC axis

### A. Long-context observables (read-time, inference on long inputs)
- Attention-sink mass per head/layer; **attention entropy** per head (sharpness).
- Attention distance / span; attention-rollout information flow from needle→answer position.
- **Retrieval-head score** per head; induction-head score.
- **Lost-in-the-middle curve**: accuracy vs needle depth (position grid).
- **Effective context length**: longest length before NIAH accuracy hits chance.
- KV-norm / KV-cache magnitude per position; per-channel KV outliers.
- RULER / NIAH accuracy vs length (4k→64k).
- Per-layer "needle survival": does the needle representation persist to the readout (logit/tuned lens)?

### B. Forgetting observables (write-time, base vs SFT'd model on held-out generic + code/math)
- Per-site **‖ΔW‖** (parameter-update magnitude) at every granularity.
- Per-site **gradient magnitude / accumulation** during SFT (where the big grads land).
- **Fisher information** per parameter (importance to base distribution).
- **Activation drift**: CKA / cosine between base and SFT'd activations per layer/head/channel on held-out general data.
- **KL(base ‖ SFT)** on a general corpus, attributed per layer via logit lens; logit-margin collapse.
- **Capability drop**: ΔGSM8K / ΔMATH / ΔHumanEval / ΔMMLU after SFT.
- **Singular-subspace rotation**: angle between base and SFT'd top-k singular vectors per matrix.
- **Sink erosion**: change in sink mass after SFT; **massive-activation shift**: do outlier channels shrink/move?
- **Routing drift** (MoE): change in expert-routing distribution on general data; which experts move most.

### C. Coupling metrics (the shared observation — H2/H3)
- **Rank correlation**: Spearman ρ between the per-site *long-ctx-importance* ranking (A) and the per-site *forgetting-disruption* ranking (B). Computed per granularity, per domain, per model.
- **Top-k overlap**: Jaccard between "top sites by long-ctx" and "top sites by forgetting".
- **Scatter**: site-level scatter (long-ctx importance) vs (SFT disruption); the headline Phase-1 figure — look for a diagonal.
- **Causal intervention (H3 preview)**: protect site-set S chosen by the *long-ctx* criterion during SFT → measure Δforgetting *and* Δlong-ctx vs unprotected SFT.
- **Reverse intervention**: protect sites chosen by the *forgetting* criterion → measure Δlong-ctx (symmetry).
- **Necessity / ablation**: zero a candidate site at inference → does it break **both** long-ctx **and** a base capability? (extends the `[super-experts]` sink-decay test to capability).
- **Dose-response**: sweep k (#protected sites) → trade-off curves for both axes.

## 1.3 Probing tools
Logit lens / tuned lens · activation patching / causal tracing · attention visualization & rollout · CKA · Fisher information · gradient logging · ablation/zeroing hooks · (optional) SAE feature attribution.

## 1.4 Conditions — the FACTOR axis (run the matrix broadly)
- **SFT domains**: RCA · code · math · + a neutral proxy (e.g. legal/medical text) to separate "domain shift" from "code/math-specific".
- **Models**: dense **Qwen3-8B**, **Llama-3.1-8B** (cross-family); MoE **Qwen3-30B-A3B** (+ stretch: **OLMoE-1B-7B** or **DeepSeek-V2-Lite** for cross-MoE). Sizes: Qwen3-1.7B/4B/8B/14B for a scaling slice.
- **Context lengths**: 4k / 8k / 16k / 32k / 64k.
- **Training dose**: light (1 epoch, low lr) → heavy (multi-epoch, overfit) to trace the forgetting dose-response.

## 1.5 Phase-1 deliverables
- A **coupling map** per model: ranked shared sites + the granularity at which coupling is strongest.
- The **headline scatter/correlation figure** (long-ctx importance vs forgetting disruption) — the de-risk gate.
- The **causal-intervention table** (protect long-ctx sites → Δforgetting) — preview of H3.
- A short decision memo: confirmed / weak / refuted, and *which granularity* Phase 2 should target.

---

# Phase 2 — method design (contingent on Phase 1)

The exact mechanism is **chosen by which granularity carries the coupling** (DR9: criterion = intrinsic importance, never task affinity). Candidate variants:

| Variant | If coupling lives at … | Mechanism |
|---|---|---|
| **V1 — site freeze** | heads / experts / channels | hard-freeze the top-k intrinsic sites during SFT (forward params fixed) |
| **V2 — gradient mask/scale** | channels / params | scale gradients ∝ (1 − long-ctx-importance); protect massive-act channels (MoFO-like but importance = long-ctx criterion, **not** momentum) |
| **V3 — orthogonal update** | weight subspace | project LoRA/SFT updates off the long-ctx-important subspace (OPLoRA-like but subspace chosen by long-ctx importance, **not** top singular value) |
| **V4 — local KL/repr anchor** | layers / directions | KL or representation anchor applied *only* at the coupled sites (cheaper than global KL; sharper than SelfAug) |
| **V5 — super-expert protection** (MoE) | experts | freeze super experts, adapt routed experts/router |

All variants: **frozen public base + tiny/zero trainable** (DR5), **detector runs on generic text** (DR3), **no task-specific module** (DR6). Optional trainable piece (if any) uses a **task-agnostic self-supervised infilling** objective (DR6, and the AR→dLLM bridge DR15).

**Causal test (H3)**: for the chosen variant, compare {full SFT, capability-agnostic baseline, ours} at matched new-domain accuracy; ours must reduce forgetting by ≥ 30% and beat the best baseline by ≥ 5% retention, without long-context regression.

---

# Ablations (the ones that, if they fail, kill the paper)

1. **Criterion ablation** — long-ctx-importance site set vs (a) random sites, (b) task-affinity sites (ESFT-style), (c) raw-gradient-magnitude sites (MoFO-style), (d) top-singular-subspace (OPLoRA-style). Ours must win or tie-cheaper; this isolates DR9 as the contribution.
2. **Granularity ablation** — head vs channel vs expert vs subspace protection; which is necessary/sufficient.
3. **k sweep (dose-response)** — protection strength vs (forgetting, long-ctx, new-domain) trade-off.
4. **Data-agnostic transfer** — detect sites on generic text only, then protect against RCA/code/math SFT; must match sites detected on in-domain data (proves DR3).
5. **Cross-model transfer of the criterion** — site set / recipe ported Qwen3→Llama, dense→MoE.
6. **Coupling necessity** — show that when H2 ρ is low (if any model/domain shows it), protection gives no benefit (negative control linking H2→H3).
7. **Long-context-only control** — does protection alone (no new SFT) change long-ctx? Should be ≈0 (protection is about *preserving*, not adding).

# Statistical practice
- **4 seeds** per headline number (DR14); report mean ± 95% bootstrap CI; paired-bootstrap for pairwise claims.
- Pre-register H2's ρ threshold and H3's decision rule before the final runs.
- Per-site metrics averaged over ≥ 3 generic-text batches to denoise.

# Failure modes to monitor
- **Detector noise** — sink/retrieval-head scores vary across inputs → average over batches, report stability.
- **Confound: "important sites just have big weights"** — control with Fisher-normalized and random-equal-norm site sets (ablation 1c).
- **Long-ctx vs forgetting measured on different inputs** — fix a shared probe corpus so the two rankings are comparable.
- **MoE routing instability at long ctx** — separate router drift from expert drift (per `[same-moe]`).
