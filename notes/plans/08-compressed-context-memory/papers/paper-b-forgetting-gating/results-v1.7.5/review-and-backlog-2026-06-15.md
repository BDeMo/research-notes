# Paper B / v1.7.5 — review, critical review & task backlog (2026-06-15)

> Holistic pass: what's left to run, where the paper stands, and a tough-reviewer critique.
> Complements the existing self-critiques ([`scaling-critical-review.md`](scaling-critical-review.md),
> README §"Honest novelty"). Paper B = *"Do No Harm: a portable, model-agnostic memory module that
> knows when not to fire."*

## A. Task backlog (ranked by paper impact)

**Tier 1 — decisive (gate whether this is a main-conference paper):**
1. **7-family gate head-to-head: ours (`delta_last`/intrinsic) vs TARG vs output-conf vs oracle vs ours⊕TARG**, leave-one-model-out. The README calls this *the decisive next run*; §7d already shows **TARG ≥ ours on 3 families**, so the gate contribution lives or dies here. **Not run** (no `targ/headto/generality` dir). 
2. **Cost–quality Pareto with a net win.** Compression never beats full (T1/T2) and full is always the fallback ⇒ the only win is **tokens/latency saved on the compressible fraction at matched quality**. Show it vs always-full and vs TARG-gated. **Missing — this is the positive headline the paper currently lacks.**
3. **Relevance / agent eval** — a mixed helpful/irrelevant-augmentation setting to earn the "do-no-harm for agents" framing. **Not run.**

**Tier 2 — firm up:**
4. **Contrast ablation follow-ups** — cross-bench + seeds. *Running* (2/8; ETA ~02:00). Early: cross-bench gain is **headroom-dependent** (see §B).
5. **Qwen3.5-9B cross-family confirm** (hybrid linear-attn = the embedding-space rationale): do lr3e-4 / minimize-scale / gate / contrast hold on a hybrid base? **Not run.**
6. **p2_gcm (T8c) analysis** — 52/52 done; turn into the do-no-harm-*transfers* table (clean in/cross-task/cross-domain + per-relation gate). **Data ready, analysis pending.**

**Tier 3 — nice-to-have:**
7. Does contrastive-trained memory yield a **more transferable gate** (fold contrast into the T6/T8 gate-transfer matrix)?
8. `ab_min` N×K Pareto figure (minimize-scale).

**Done since the last plan:** scaling K-sweep + N×K min grid; **H5 confirm (manif10 0.083 / mnormhard 0.094 ≤ base ⇒ OOD-fix doesn't lift ⇒ capacity-bound stands)**; faithful 2025 baselines (T2); p2_gcm transfer (52/52); the faithful-contrastive ablation ([`faithful-contrastive-ablation.md`](faithful-contrastive-ablation.md)).

## B. Review — state of v1.7.5 (honest)
Clean (post-leak-fix) **diagnosis-led** paper. Established: necessity (T1); all faithful 2025 baselines ≤ GCM ≤ full (T2); length-adaptive ≈ fixed-K (T3); **no scaling** on data/depth/compute/**K** (T4/T5, critical-review PASS); **lr 3e-4 best + minimize-scale (N4/K16) matches big at ~10× less compute** (T5); gate composes with any compressor + the **agnostic gate transfers** (T6); memory is **OOD** (T7); do-no-harm **extends to cross-domain + non-tool**, with *real* detection (AUROC>0.6) on **3 benches**, conservative-fallback elsewhere (T8). New: faithful **contrastive** alignment helps **+5.2pp** (λ-sweet-spot 0.5, inverted-U), **adv hurts**, combo antagonistic-on-compress / best-gate. **Cross-bench caveat (fresh): on the floored bfcl_multiple contrast = base (0.133=0.133)** → the contrastive gain is **headroom-dependent**, not universal.

## C. Critical review (as a committed ICLR 2027 reviewer)
The team is commendably honest (TARG-competitive flagged, capacity-bound stress-tested). A reviewer still presses:

1. **What is the actual win?** Compression *never* beats full, and full is always kept for fallback. So the system is "a worse-than-full compressor + a safety net." The paper **needs the cost–quality Pareto net-win** (Tier-1 #2) or the contribution collapses to "minimize-scale + a gate." Right now the positive deliverable is thin.
2. **The gate is matched by a training-free baseline.** TARG ≥ ours on 3/7 families (§7d). If the 7-family head-to-head confirms parity, "our signal" is not a contribution; the paper must lead **purely** on framing + do-no-harm-by-construction for a *learned* module — a narrower claim. The head-to-head (Tier-1 #1) is do-or-die.
3. **Detection is thin / "do-no-harm" is trivially gameable.** Real discrimination on only 3 benches; elsewhere high-F1 = always-compress = "just use full." A reviewer notes do-no-harm is trivially satisfied by never compressing; the *interesting* claim (know when to compress) holds on a minority.
4. **Negative-result-heavy.** Headlines are mostly negative (no scaling, capacity-bound, OOD, length-adaptive null, adv hurts, contrast null on floored benches). Valuable, but needs a crisp positive result to clear a top venue (the Pareto, or a relevance-eval win).
5. **Single base model.** Most results Qwen3-8B; Qwen3.5 (the hybrid base that *motivates* embedding-space) is unconfirmed. Cross-family is asserted via `delta_last` AUROC, not via the head-to-head.
6. **Capacity-bound vs the Beacon/ICAE/Cartridge camp.** Those claim near-lossless; reviewers there will say "K=64 / 384 items / 3000 steps is undertrained/small." The scaling-critical-review defends depth/data/K up to K256 — keep that front-and-center, and pre-empt "scale further."
7. **Contrast ablation is secondary + fragile.** +5.2pp is single-seed, one bench, and **null on the floored bench** — it's an ablation, not a headline; don't oversell.

**Verdict:** As framed (do-no-harm gate, complementary to Cartridges, extending TARG), it is a **borderline empirical/diagnosis paper (~5/6)**. To reach a clear accept it needs **(i)** the cost–quality net-win Pareto, **(ii)** the 7-family ours-vs-TARG-vs-combo head-to-head (or an honest pivot to framing-only), and **(iii)** a relevance/agent eval. Without those it is a strong **findings/workshop** contribution.

## C2. Cost–quality Pareto + ours-vs-TARG (no-GPU analysis, `pareto_gate.py`, 2026-06-15 late)
Addresses critical-review #1/#2 directly. To hold **95% of full quality**:
- **In-task (ctr0.5, n=96):** **ours `neg_recon` AUROC 0.773 → 48% fallback → 34% token saving** vs full;
  beats **TARG `margin` (0.659 → 80% fb → 14%)**, conf (15%), neg_entropy (25%). Oracle = 43%.
  → **the net-win positive headline exists in-task, and ours > TARG here.**
- **Cross/transfer (p2_gcm, n=4712):** every signal ≈ chance (margin/conf 0.50–0.53; ours
  neg_recon **0.36**, below chance — intrinsic collapses cross-domain) → ~93% fallback → **~0–8%
  saving**. Oracle could save 63% ⇒ headroom exists but no signal captures it cross-domain.
- **Verdict:** real efficiency win **in-task (ours>TARG)**, **gate collapses cross-domain** — the
  honest boundary for the paper. The cross-domain gate is the true open problem (Tier-1 #1/#3).

## D. Recommendation
Run **Tier 1** next (head-to-head + Pareto + relevance eval) — they decide the venue. The contrast/alignment work (incl. v1.8.0) is a good secondary ablation, not the headline. Keep the capacity-bound diagnosis as the rigorous backbone; find one **positive** deliverable (net-win Pareto or relevance-eval win) to anchor the story.
