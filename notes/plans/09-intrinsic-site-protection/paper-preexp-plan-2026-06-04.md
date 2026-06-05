# Janus — pre-experiments needed for a paper (+ the mechanism reckoning)

> Created 2026-06-04 (evening). Written after the **mechanism/confound controls**
> revealed the v1 headline coupling is largely confounded — so the paper plan is
> now about *establishing the real, defensible claim*, not adding coverage.

## ⚠️ The reckoning (why this plan exists)
Partial-correlation + within-layer controls on the existing 41-cell grid
([`analyze_mechanism.py`], run 2026-06-04 18:20) showed the **v1 "uniformly positive
LC×CF coupling" does NOT survive per-model scrutiny**:
- Pooling cells inflated it; **per model**, most LC↔CF couplings are weak/≈0 or **negative** (qwen2.5 `v_norm~fisher` −0.28, `retrieval~fisher` −0.17).
- The robust survivor `attn_distance~dW_drift` (glm4 +0.43, qwen3-8b +0.52; survives the activation-norm control) has **within-layer ρ ≈ 0/neg** → a **layer/depth effect, not head-level**.
- `v_norm~dW_drift` is mostly an **activation-magnitude confound** (qwen3-8b raw 0.34 → partial 0.07).
- **Fisher (a-priori) couplings weak** → "predict-before-training" leg is shaky.
- qwen3.5's 0.8–0.95 are **hybrid zero-inflation artifacts** (24/32 layers no attention) — exclude from head-level claims.

**Revised defensible claim (to be confirmed):** *long-context-heavy **layers** are
disproportionately perturbed by SFT (esp. `attn_distance`→`dW_drift`), partly mediated
by activation magnitude.* The strong head-level "same-site" claim is **not supported**.

## Pre-experiments for the paper

### Tier 1 — establish the real claim (runnable now, no SFT recipe needed) 🟢
1. **Rigorous a-priori mechanism across the full-attention ladder** (0.6→14B): per-model raw / partial(|out_norm) / within-layer ρ + bootstrap CI, on standard-arch models (drop hybrid). → `mechanism.py` (**running on the 4 GPUs**). Answers: is there ANY clean head-level a-priori coupling, and how does it scale?
2. **Layer-level coupling formalized**: aggregate per-layer LC (mean attn_distance / retrieval) vs per-layer drift/Fisher; report the layer-level ρ + CI. This is likely the surviving result — quantify it cleanly.
3. **Activation-magnitude mediation**: full mediation analysis (grad ≈ error × activation); how much of the coupling is explained by `out_norm`/`v_norm`. Decide "trivial vs functional."

### Tier 2 — the causal method (needs the SFT-recipe fix) ⏳
4. **Valid forgetting recipe**: LoRA-all-modules / full-FT that *learns* the task (Δtarget>0) AND forgets (ΔMMLU<0). (3 setups failed: gentle math, gsm8k-gentle, attn-only-heavy.)
5. **Protection test at the granularity the mechanism supports** (likely **layer-level** or the `attn_distance` criterion): does protecting the coupled *layers* reduce forgetting vs random / ΔW-oracle / EWC / MoFO, at matched new-domain gain & no long-ctx regression?
6. **Negative control**: a decoupled setting (e.g., RL, or a domain with low layer-coupling) → protection gives no benefit (links correlation→causation).

### Tier 3 — robustness / breadth (mostly have, or cheap)
7. Cross-family (have: 4 families) + **scaling** (Tier-1 #1).
8. Data-agnostic transfer: detect on generic text → protect against any SFT domain (the protected set is domain-invariant).
9. Long-context degradation at **long** length (32k+) — does any SFT actually erode NIAH? (so far NIAH robust on GLM-4).

## Paper framing options (pick after Tier 1)
- **(A) Measurement paper:** "long-context-heavy layers are the catastrophic-forgetting hotspots" — honest, layer-level, with the mechanism (activation-mediated) + scaling + cross-family. Solid even without a method.
- **(B) Measurement + method:** A + a layer-level protection method that beats baselines (needs Tier 2).
- **(C) Negative/cautionary:** "the intuitive head-level coupling is a confound; here's the controlled truth" — valuable if Tier 1 kills even the layer-level claim.

## Status
- Tier-1 #1 **running** (mechanism ladder, 4×H100). #2–#3 are analysis on its output + existing grid.
- Tier 2 blocked on the trainer fix (LoRA). Tier 3 partly done.
