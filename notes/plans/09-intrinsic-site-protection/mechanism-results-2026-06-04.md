# Janus — mechanism results: the head-level coupling does NOT survive controls

> **Pivotal negative result, 2026-06-04 eve.** Rigorous a-priori coupling across the
> full-attention Qwen3 ladder (0.6→14B), with partial-correlation (activation control),
> within-layer (depth control), and bootstrap CIs. Code: `runs/mechanism.py`. Raw:
> `runs/mechanism_*.json`. This **overturns the v1 headline** (`grid-metrics-2026-06-04.md`).

## Setup
Per model: per-head LC metrics (retrieval, v_norm, attn_distance, sink) on generic
text + per-head **a-priori** gradient/Fisher on GSM8K pairs (forward+backward, **no
optimizer step** → no model destruction; measures *where SFT gradients would land*).
For each (LC, CF) pair: **raw** ρ, **partial ρ | out_norm** (removes activation-
magnitude), **within-layer** ρ (removes depth/layer confound), 95% bootstrap CI.

## Result — a-priori grad_mag vs LC (within-layer = the real head-level test)
| LC | 0.6B | 1.7B | 4B | 8B | 14B | verdict |
|---|---|---|---|---|---|---|
| **attn_distance** within-layer | −0.50 | −0.58 | −0.60 | −0.57 | −0.55 | **robustly NEGATIVE** |
| **sink** within-layer | −0.47 | −0.52 | −0.57 | −0.52 | −0.52 | **robustly NEGATIVE** |
| **v_norm** within-layer | +0.12 | +0.14 | +0.10 | +0.08 | +0.10 | weak +, but partial|out_norm ≪0 (raw = activation artifact) |
| **retrieval** within-layer | −0.02 | −0.09 | −0.03 | −0.04 | +0.08 | ≈ 0 |

(Fisher mirrors grad_mag; CIs are tight, e.g. 8B attn_distance within-layer ⊂ [−0.38,−0.27] raw / consistently negative.)

## Interpretation
1. **No robust head-level long-context↔forgetting coupling.** The a-priori gradient
   does **not** preferentially land on long-context heads.
2. **The opposite, clean fact:** `attn_distance`/`sink` heads are **gradient-AVOIDED**
   a-priori — robustly within-layer, at every scale. The structural long-context heads
   are *intrinsically protected* from the SFT gradient, not vulnerable.
3. **`v_norm`'s apparent coupling is an activation-magnitude confound** (partial|out_norm
   collapses or reverses it).
4. The v1 "+0.17…0.56 uniformly positive" matrix was inflated by **cell-pooling +
   activation-magnitude + depth confounds + qwen3.5 hybrid zero-inflation**. The
   post-hoc `dW_drift` positives were **depth effects** (within-layer ≈ 0).

## What this means for the paper
The strong **"one site, two frontiers (head-level)"** thesis is **falsified** on this
evidence. Honest options:

- **(A) Cautionary / negative-result paper.** "The intuitive long-context↔forgetting
  *head* coupling is a measurement artifact; under partial/within-layer/scaled controls
  it vanishes or reverses." Contribution = the rigorous methodology + the controlled
  truth (incl. the clean **gradient-avoidance** of structural heads). Publishable,
  saves the field a trap. We already have the multi-model × multi-dataset × scaled study.
- **(B) Reframe to the real fact.** "Long-context structural heads (sink/long-reach)
  are intrinsically shielded from fine-tuning gradients" — a positive mechanistic claim,
  opposite direction. Implies forgetting does *not* hit long-context (consistent with our
  NIAH-robust observations) → no protection method needed; the story is *why* they're
  shielded (attention-sink/massive-activation stability).
- **(C) Keep searching** for a coupling at a *different* target: not gsm8k-gradient but
  specific capability-forgetting (code/math retention) vs specific sites; or layer-level
  only; or a different frontier pairing. Higher risk.

## A+B quantified — shielding mechanism + layer-level + scaling (7 models)
Extended a-priori run (`runs/mechanism.py` v2; raw arrays in `runs/mechanism_*.npz`).

**Shielding — within-layer ρ(a-priori grad_mag, explainer):**
| model | attn_entropy | sink | attn_distance | out_norm | retrieval |
|---|---|---|---|---|---|
| qwen3-0.6b | +0.23 | −0.47 | −0.50 | +0.56 | −0.02 |
| qwen3-1.7b | +0.29 | −0.52 | −0.58 | +0.64 | −0.09 |
| qwen3-4b | +0.33 | −0.57 | −0.60 | +0.63 | −0.03 |
| qwen3-8b | +0.35 | −0.52 | −0.57 | +0.61 | −0.04 |
| qwen3-14b | +0.35 | −0.52 | −0.55 | +0.58 | +0.08 |
| glm4-9b | +0.05 | −0.27 | −0.29 | +0.35 | −0.03 |
| qwen2.5-7b-inst | +0.06 | −0.04 | −0.47 | +0.50 | −0.23 |

→ **Gradient magnitude is driven by activation magnitude** (`out_norm` +0.35…+0.64, every model) and **suppressed for long-reach/sink heads** (`attn_distance` −0.29…−0.60, every model). On Qwen3, low-attention-entropy (peaked/saturated) heads get less gradient (+0.23…+0.35); weaker on GLM-4/Qwen2.5, so the **universal** explainer is activation magnitude, with long-context structural heads being lower-activation / lower-gradient.

**Layer-level coupling ρ(LC_layer, a-priori grad_layer):**
| model | attn_dist~grad | sink~grad | retrieval~grad |
|---|---|---|---|
| qwen3-0.6b | −0.72 | −0.71 | −0.55 |
| qwen3-1.7b | −0.75 | −0.73 | −0.59 |
| qwen3-4b | −0.33 | −0.29 | −0.21 |
| qwen3-8b | −0.14 | −0.13 | −0.04 |
| qwen3-14b | +0.02 | +0.01 | +0.35 |
| glm4-9b | −0.23 | −0.23 | −0.11 |
| qwen2.5-7b-inst | −0.32 | −0.33 | −0.18 |

→ At the **layer level** too, long-context-heavy layers receive **less** a-priori gradient; the shielding is **strongest in small models and fades with scale** (0.6–1.7B ≈ −0.75 → 8–14B ≈ 0).

**Defensible claims for the paper:**
1. (debunk) No positive head- or layer-level long-context↔forgetting coupling under controls; the naive intuition is a confound.
2. (fact) **Long-context structural heads/layers are gradient-shielded** during fine-tuning — robust within-layer, cross-family, primarily via lower activation magnitude (+ low attention entropy on Qwen3).
3. (scaling) The shielding **weakens with model scale** (a new, clean scaling result).
4. (consequence) This explains why **NIAH stays robust under SFT** (H3 v1/v2) — long-context isn't where forgetting lands; no head-protection method is needed for long-context.

## Recommendation
Lead with **(A)+(B) combined**: a rigorous controlled study that (i) debunks the naive
head-level coupling and (ii) establishes the surviving fact (structural long-context
heads are gradient-shielded, scale-robust). Drop the protection-method framing unless
(C) finds a real coupling. Next concrete steps in `paper-preexp-plan-2026-06-04.md`
Tier-1 #2–#3 (formalize layer-level + activation mediation) are still worth doing to
fully characterize the (null) head-level + the (real) gradient-avoidance.
