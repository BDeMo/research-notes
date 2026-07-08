# v1.8.x Method-Elegance Plan — prune the carrier, sharpen the contribution

> Goal (user): make the **whole method more elegant** without losing what works. Grounded in v1.8.0's own ablations
> (`results-v1.8.0.md`, `paper-logic-v1.8.0.md`). This is refinement of the *existing* method, not the v2.0.0 transformation.

---

## 0. Diagnosis — where the inelegance is (from our own ablations)

The method has accreted a **loss zoo** (7 objectives) and a **signal zoo** (6 gate signals) + 4 normalization modes + optional encoder-LoRA. Our ablations say **most of it is not load-bearing**:

| component | ablation finding | verdict |
|---|---|---|
| joint task-CE | pivotal (without it, low compress) | **keep** |
| distillation (self, full-ctx) | load-bearing | **keep** |
| reconstruction (M0→ctx) | load-bearing **and** powers the `neg_recon` gate | **keep (spine)** |
| deviation ‖Mq−M0‖² | ±0, droppable | **cut** |
| contrastive (InfoNCE) | situational, *can hurt* | **cut** |
| VAE (μ,logσ,KL) | neutral, KL must be tiny | **cut** |
| encoder-LoRA | does not beat frozen-enc + joint-task | **cut** |
| normalization: off/hard/learn/manifold (4) | hard-manifold = stable default | **collapse to 1** |
| gate signals: margin/conf/neg_entropy/neg_recon/dlogit/targ (6) | best is **bench-dependent** | **collapse to 1 + conformal** |

Two hard facts that should *drive* the redesign:
- **The compressor is a commodity.** Gist-lite ≈ ours (+~0.05); flip-one moves `compress` by only **±0.03**. ⇒ Polishing the compressor is wasted elegance budget.
- **The headline gate claim was tautological.** "gated ≥ full by construction" can't drop below full even for a noise signal ⇒ the contribution must be re-grounded as *selective prediction with a real, out-of-sample guarantee*, not a construction.

**Thesis of the elegant version:** *Stop polishing the carrier. Make the memory faithful by **one** objective (reconstruction), and use **that same** reconstruction as the **one** signal that decides when to trust it — calibrated with a real coverage guarantee.*

---

## 1. The elegant core — one base, four roles, two losses, one gate, one guarantee

Everything is the **same frozen base** + tiny adapters + the base's **own LM head**. No external teacher, no external scorer, no extra models.

```
            ┌──────────────── ONE frozen base (Qwen3-8B / Qwen3.5-GDN) ────────────────┐
 encode →   K memory tokens + proj-MLP   (recurrent, varlen → length-agnostic)          │  = M
 read   →   read-LoRA prefix on M        → answer                                       │
 teach  →   the base reading FULL ctx     → self-distillation target (no external LM)    │
 verify →   the base's tied LM head reconstructs ctx from M  → faithfulness score s(M)   │
            └────────────────────────────────────────────────────────────────────────┘
```

- **Two training objectives only:**
  1. **Answer** = joint task-CE + self-distillation KL (M must let the reader reproduce the full-context behavior).
  2. **Preserve** = self-reconstruction (M must let the base reconstruct the evidence).
  Drop deviation, contrastive, VAE, encoder-LoRA.
- **One gate signal = the reconstruction score itself.** At inference, compute the *same* reconstruction quality `s(M)` on this input. High `s` ⇒ M faithfully holds the evidence ⇒ trust the compressed path; low `s` ⇒ fall back. **The training regularizer and the inference gate are literally the same mechanism.**
- **One guarantee:** calibrate the threshold on a held-out split with **conformal** selective prediction → a *stated* coverage / risk bound, replacing the tautological "≥ full" and the per-bench τ picking.

One-paragraph statement (paper-ready):
> *Train a small memory to (i) answer and (ii) reconstruct the context, both supervised by the same frozen base. Because a memory that can reconstruct the evidence is a memory that retains it, the reconstruction score is a self-verification signal: at inference we trust the compressed memory exactly when it verifiably preserves the input, and conformal calibration turns that into a distribution-free quality guarantee. The compressor is deliberately a commodity; the contribution is the self-verified, cost-ascending gate.*

---

## 2. Concrete elegance moves (each with the ablation that licenses it)

| # | move | why elegant | licensed by | risk |
|---|---|---|---|---|
| **M1** | **Prune losses to {answer (task+distill), preserve (recon)}** | 7→2 objectives; one knob each | dev ±0, contrastive can-hurt, VAE neutral | none (ablation says ±0.03) |
| **M2** | **Gate = the reconstruction score** (train-loss ≡ inference-signal) | 6 signals → 1; removes the bench-dependent signal pick | `neg_recon` is the gated-acc maximizer + recon is already load-bearing | keep `conf` as a logged secondary, not in the method |
| **M3** | **Conformal threshold** (held-out coverage) | replaces ad-hoc τ + the tautological claim; gives a real bound | "≥ full by construction was tautological" | needs a calibration split (cheap) |
| **M4** | **3-way as one signal at two thresholds** (base→compress→full) | the whole cost–coverage curve from a single monotone signal | 3-way adaptive already exists | — |
| **M5** | **Frozen encoder + single normalization (hard-manifold)** | drops encoder-LoRA + 3 norm modes | encoder-LoRA doesn't beat frozen+joint; hard-manifold stable | — |
| **M6** | **Varlen recurrent encode as the default** (not an ablation knob) | one length-agnostic model, no per-length retrain/OOD | recurrent+varlen already the long-ctx fix | verify no length OOD |
| **M7** | **"Self-everything" framing** (encoder/reader/teacher/judge = one base) | removes any external model from the story | already true | narrative only |

Net: **7 losses → 2 · 6 gate signals → 1 · 4 norm modes → 1 · 2 trained adapters (mem+proj, read-LoRA), encoder frozen.** Same accuracy (ablation), far smaller surface.

---

## 3. One genuinely-new elegance idea (optional, beyond pruning)

**Reconstruction-budgeted memory (ties to the kvzip fact-base finding).** Our sweep showed *reconstruction-based* KV importance (kvzip) is the most compression-robust baseline. Mirror it in the soft memory: let the **per-input memory length scale with reconstruction difficulty** — easy/redundant contexts get fewer M tokens, hard/dense ones get more, decided by the same recon score. This makes adaptive-M *principled* (driven by the verify signal) instead of a fixed `M ∝ ctx` heuristic, and unifies "how much to compress" with "can we trust it" under one quantity. *(Defer to v2; listed for completeness.)*

---

## 4. Validation plan (cheap, eval-mostly)

1. **Pruned-vs-full A/B:** retrain the 2-loss core (M1+M5) vs the current 7-loss method on both bases; confirm `compress` within ±0.03 and `gated` unchanged (ablation predicts yes). 1 train per base.
2. **Single-signal gate:** re-score the existing eval dumps with recon-only gate + conformal τ; report gated-acc + coverage + risk vs the 6-signal best. Eval-only (reuse saved runs).
3. **Necessity regrounding:** show the elegant method still wins the only defensible regime — `quality`/over-window where compress ≫ truncated-full (the 3.5× money result) — now with a conformal-backed gate. Eval-only.
4. **Length-agnostic check (M6):** varlen-trained encoder evaluated at 2k/8k/16k/32k → flat, no OOD dip.

If (1) holds (very likely), the paper's method section shrinks from a module catalog to: **one base · two losses · one self-verified gate · one guarantee** — and every cut is justified by an ablation we already ran.

---

## 5. What this buys the paper
- **Smaller, defensible method:** every component earns its place via ablation; nothing to attack as "unmotivated."
- **A non-tautological contribution:** self-verification + conformal coverage is a real selective-prediction result, not "≥ full by construction."
- **One mechanism, two uses (recon = regularizer = gate):** the kind of unification reviewers reward.
- **Clean bridge to v2.0.0:** the self-verified gate is exactly the "do-no-harm, out-of-sample" claim (C3); the commodity-compressor framing is honest about C1/C2.

*Provenance: derived from `paper-logic-v1.8.0.md` §3–§5 and `results-v1.8.0.md` §0b/§0c ablations (flip-one ±0.03, gist-lite≈ours, recon load-bearing, tautological-gate note) + the v2.0.0 necessity/kvzip fact-base.*
