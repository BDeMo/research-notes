# Faithful representation-alignment ablation (v1.7.5) — can the alignment loss be contrastive?

> Ablation **within v1.7.5** (not a version bump). Question: the GCM memory lives in the base's
> **input-embedding space**, so it is OOD (T7: mean-cos ≈ 0.10 to the vocabulary) — a frozen base
> reads a memory it can't fully trust. A **faithful-alignment loss** pulls the `[M;q]` hidden at a
> shared base layer (`align_layer=18`) toward the `[ctx;q]` hidden. We test three forms and ask:
> **(1) does a contrastive (InfoNCE) form help, (2) across all settings, (3) vs / combined with the
> adversarial form?** Code: `svc/method.py` (lam_contrast / lam_align / lam_adv), `analyze_contrast.py`.

## The three alignment forms (all align `hM=[M;q]` → `hF=[ctx;q]` at layer 18)
- **contrast** (`lam_contrast`): InfoNCE — pull `(hM_i, hF_i)` together, push `hM_i` from *other*
  samples' `hF_j` (teacher hF detached; SimCLR-style teacher-negative bank).
- **align** (`lam_align`): pure positive cosine pull, **no negatives** (`1 − cos(hM, hF)`).
- **adv** (`lam_adv`): a discriminator can't tell M-induced from full-induced hidden (min-max).

> **Bug fixed before testing (important):** the InfoNCE negative pool included each sample's **own
> positive** `hF_i` (self-contradiction → pushes a sample away from its own target). Fixed to
> exclude self (`cbank_idx`). Any earlier contrastive attempt was invalid.

## Recipe (aligned to the v1.7.5 main study)
Qwen3-8B, frozen; GCM enc-16 / K64 / dec-2 / base-LoRA-32 / lam-distill 0.5 / **lr 3e-4** /
**grad-accum 8 (eff batch 8)** / **3000 steps** / **384 train items** / 96 eval; `--signals`.
In-task on **bfcl_live_multiple** (headroom + a real gate signal). Metric: compress acc (`gcm`),
gap to `full`, and best agnostic-gate AUROC (failure detection).

## Main table — λ sweep + adv + combo (3000 steps, bfcl_live_multiple, n=96)
| variant | compress | full | gap | gate AUROC | Δcompress vs base |
|---|---|---|---|---|---|
| base (no align) | 0.667 | 0.917 | 0.250 | 0.751 | — |
| ctr λ=0.2 | 0.656 | 0.917 | 0.260 | 0.735 | −0.011 |
| **ctr λ=0.5** | **0.719** | 0.917 | **0.198** | 0.773 | **+0.052** |
| ctr λ=1.0 | 0.698 | 0.917 | 0.219 | 0.771 | +0.031 |
| adv 0.5 | 0.594 | 0.917 | 0.323 | 0.742 | −0.073 |
| ctr+adv (0.5/0.5) | 0.615 | 0.917 | 0.302 | **0.851** | −0.052 |

## Findings
1. **Contrastive alignment helps — but it is λ-dependent (inverted-U, peak at 0.5).** λ=0.2 is a
   wash (≈ base); **λ=0.5 is the sweet spot (+5.2pp compress, +0.022 gate)**; λ=1.0 still helps
   (+3.1pp) but past the peak. So **not all settings提点** — the gain is robust for λ∈[0.5,1.0],
   not for tiny λ. (Answers Q2: no, there's a threshold/sweet spot.)
2. **Contrast ↑ vs adversarial ↓ — same goal, opposite outcome.** adv *hurts* compress at every
   budget (−10pp @1200 steps, −7.3pp @3000) — the min-max only matches the marginal, is unstable
   on a frozen base, and the compressor sacrifices task acc to fool D. Contrast gives a stable
   per-sample supervised target → it's the *right* way to do faithful alignment. (Answers Q1/Q3.)
3. **The combo is antagonistic on compression but best for the gate.** ctr+adv drags compress
   *down* (0.719 → 0.615, toward adv's level) — adv destabilizes the contrastive target — **but**
   yields the **highest gate AUROC (0.851)**. So: contrast-alone for compression; the combo only
   if detection matters more than compression. (Small-n caveat on 0.851.)
4. **Convergence justified the alignment.** At 1200 steps the gain was only +2.5pp because
   *contrast was still improving* (val loss monotone-decreasing to step 1080) while base/align/adv
   had converged (base overfits past step ~600). At the aligned 3000 steps the gain **widened to
   +5.2pp** and ctr0.5 (0.719) reaches the main-table GCM level (~0.72). Contrastive alignment
   behaves like a regularizer: keeps generalizing where base overfits.
5. **It does NOT break the capacity ceiling** (0.719 ≪ full 0.917) — consistent with v1.7.5's
   capacity-bound diagnosis. Faithful contrastive alignment is a **refinement/ablation**, not a
   headline fix.

## 1200-step preview (n=120, pre-alignment recipe — kept for the convergence story)
| variant | compress | gate AUROC |
|---|---|---|
| base | 0.583 | 0.734 |
| contrast 0.5 | 0.608 | 0.756 |
| align 0.5 | 0.592 | 0.746 |
| adv 0.5 | 0.483 | 0.712 |

## Generalization (2026-06-16) — does the +5.2pp hold? **No, it does not robustly generalize.**
| axis | setting | base → ctr0.5 | Δ |
|---|---|---|---|
| (headline) | Qwen3-8B, bfcl_live_multiple | 0.667 → 0.719 | **+5.2** |
| base/family | GLM-4-9B, bfcl_live_multiple | 0.490 → 0.510 | +2.0 |
| training-source | hermes (Qwen3-8B) | 0.510 → 0.531 | +2.1 |
| training-source | glaive (Qwen3-8B) | 0.802 → 0.771 | **−3.1** |
| benchmark (eval) | hermes | 0.510 → 0.510 | 0 |
| benchmark (eval) | bfcl_multiple (floored) | 0.133 → 0.133 | 0 |

**Verdict:** the contrastive gain is **small, high-variance, and occasionally negative** (helps
bfcl_live_multiple/GLM-4/hermes-source, null on hermes-eval/bfcl_multiple, **hurts glaive**). The
+5.2pp headline is **not representative**. → report contrastive alignment as an **honest ablation
with weak generalization**, not a headline. Cross-domain gate: contrast does **not** rescue it
(ctr neg_recon 0.832 ≈ base 0.826). The paper's positive story remains the **in-task gate net-win**
(ours `neg_recon` > TARG `margin`: 95% of full quality at 34% fewer tokens — see
[`review-and-backlog-2026-06-15.md`](review-and-backlog-2026-06-15.md) §C2).

## ⚠️ Data-quality issues found (eval harness)
- **Qwen3 thinking-mode:** the downloaded HF Qwen3/Qwen3.5 models *think* by default; the harness
  decodes only `max_new_tokens=16` with no thinking-off → full-ctx eval = reasoning preamble
  (`"Okay, let's see..."`), scores garbage (Qwen3-14B full=0.156 < gcm). The **local Qwen3-8B
  checkpoint and GLM-4/Qwen2.5 do not think**. Qwen3 scale-ladder cross-base **deferred** to a
  proper thinking-fix (chat-template + `enable_thinking=False`, preserving the `[memory;query]` prefix).
- **Qwen2.5-7B: gcm = 0.000** (full works) — the compressed-memory path is broken for Qwen2.5 (arch
  incompat). **Dropped from the model set.**

## Cross-model main table (running 2026-06-16, `out/maintable/` + cross-model) — drop Qwen2.5
Spanning diverse non-thinking model types (Qwen3 dense, GLM-4, Llama, Mistral, tool-models, …) ×
representative benches; fills the many-model main table. *(table lands as cells complete.)*

## Caveats
Single-seed main table (seeds running); n=96 eval (gate AUROC, esp. the 0.851 combo, is small-n);
one in-task bench so far (cross-bench running); 3000-step single budget. The *relative* ordering
(contrast>base>adv; peak λ0.5) is consistent across the 1200- and 3000-step runs.
