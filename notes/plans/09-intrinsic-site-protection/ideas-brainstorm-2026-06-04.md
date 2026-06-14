# Janus — idea list (brainstorm from the facts)

> Generative brainstorm: every coupling cell → a candidate method, plus blue-sky
> directions. Each idea = **Motivation (which fact) · Method · Expected insight ·
> Frontier · Cost/Risk**. Archived 2026-06-04. Grounded in
> [`grid-metrics-2026-06-04.md`](grid-metrics-2026-06-04.md) + [`metrics-reference.md`](metrics-reference.md).
> Status of every idea: 💡 unbuilt unless tagged. Cull later; this is the wide net.

## The facts we build on (recap)
- **F1 (coupling).** Long-context head metrics (v_norm, attn_distance, retrieval, sink, prev_token, kv_norm) are **uniformly positively correlated** with forgetting metrics (dW_drift, act_drift, fisher, grad_mag, grad_noise); 0.17–0.56, all 41 cells. → *the LC sites are the CF sites.*
- **F2 (centres).** `v_norm` and `attn_distance` sit at the centre of both clusters → best single predictors.
- **F3 (disjoint).** sink ≠ retrieval heads (Jaccard 0) → two distinct protectable families.
- **F4 (copy family).** induction ~ retrieval (0.61) → a unified copy/retrieval circuit.
- **F5 (collapse axis).** late layers: eff_rank↓, anisotropy↑, massive-acts↑, down-proj spectrum co-moves, prediction crystallizes (lens depth). Cross-family.
- **F6 (a-priori).** Fisher/grad/grad-noise predict drift *before* SFT → "predict-before-training."
- **F7 (universality).** holds across Qwen3/Qwen2.5/GLM-4 + the hybrid linear-attn Qwen3.5.

---

## §1 · The coupling-matrix method grid (one method per cell)
Rows = LC site to protect; cols = CF mechanism to defend against. Each cell = a concrete protection method. (✪ = strongest empirical cells from F1.)

| protect ↓ \ defend → | dW_drift (weight move) | act_drift (function move) | fisher (curvature) | grad_mag (gradient) | grad_noise (instability) |
|---|---|---|---|---|---|
| **retrieval heads** | M1 freeze retrieval-head W ✪ | M2 activation-anchor retrieval outputs ✪ | M3 Fisher-cap on retrieval heads | M4 grad-scale↓ on retrieval heads | M5 EMA-smooth retrieval-head grads |
| **attn_distance (long-reach)** | M6 freeze long-reach heads ✪ | M7 KL-anchor long-reach attn maps ✪ | M8 natural-grad downweight by reach | M9 grad-mask top-reach heads | M10 reach-weighted grad clip |
| **v_norm (high-content)** | M11 freeze high-V proj ✪ | M12 value-cache anchor ✪ | M13 Fisher∝v_norm prior | M14 lr∝1/v_norm | M15 variance-floor on V grads |
| **sink heads** | M16 freeze sink K-bias ✪ | M17 sink-mass anchor (keep sink alive) ✪ | M18 protect sink curvature | M19 zero sink grads | M20 sink-logit stabilizer |
| **prev_token / induction** | M21 freeze induction circuit | M22 anchor induction attn | M23 — | M24 grad-mask prev-token ✪ | M25 — |
| **kv_norm** | M26 freeze KV proj ✪ | M27 KV-cache distill anchor | M28 — | M29 — | M30 — |

→ **The unifying method (M0):** *Intrinsic-Site Protection* — detect the top-k heads by a combined LC score (z(v_norm)+z(attn_distance)+z(retrieval)) on **generic text**, then **freeze/grad-mask** them during any SFT. Everything above is an ablation axis of M0 (which site criterion × which protection operator). **This is the headline method; H3 tests it.**

---

## §2 · Protection methods (depth)

**M0 — Intrinsic-Site Protection (ISP).** *Motivation:* F1+F2 — protect the heads that are both LC-load-bearing and CF-vulnerable. *Method:* data-agnostic detect → grad-mask top-k. *Expected insight:* forgetting↓ with no long-ctx loss at fixed new-domain gain; a-priori criterion ≈ post-hoc ΔW oracle. *Frontier:* both. *Cost:* low (H3 running).

**M31 — Dual-criterion protection.** *Motivation:* F3 (sink≠retrieval are distinct). *Method:* protect sinks (stability) + retrieval (content) with *different* operators — freeze sinks (they're near-static), KL-anchor retrieval (they must stay selective). *Insight:* do the two families need different treatment? *Frontier:* both.

**M32 — Soft protection / elastic anchor.** *Motivation:* hard freeze blocks learning. *Method:* per-head L2-SP / EWC penalty with strength ∝ LC-coupling score (not Fisher). *Insight:* is the *LC criterion* a better anchor weight than Fisher (EWC)? Directly beats EWC's own weighting. *Frontier:* CF.

**M33 — Orthogonal-subspace SFT keyed on LC.** *Motivation:* OPLoRA orthogonalizes off top singular subspace; we off the **LC-important subspace**. *Method:* project SFT updates off the span of the high-v_norm/retrieval heads' weight subspace. *Insight:* subspace vs head granularity — which localizes the coupling? *Frontier:* both.

**M34 — Curriculum of protection (anneal k).** *Motivation:* dose-response. *Method:* protect many heads early (when grads are wild, F6), release as training stabilizes. *Insight:* the forgetting happens in early epochs (mech-forget) — does early-only protection suffice? *Frontier:* CF.

---

## §3 · Detection / criterion methods (how to choose the set)

**M35 — Generic-text-only detector.** *Motivation:* F7 universality. *Method:* compute the LC score on C4/wikitext, never on the SFT domain. *Insight:* prove data-agnosticism — the protected set transfers across all SFT domains. *Frontier:* both.

**M36 — One-shot a-priori protection.** *Motivation:* F6. *Method:* a single forward+backward on generic text → Fisher/grad → pick set → protect; zero extra training cost. *Insight:* how cheap can the criterion be (1 batch?) and still work? *Frontier:* CF.

**M37 — Cross-model transferable head map.** *Motivation:* F7. *Method:* detect on a small model, port the (layer-fraction, head-role) map to a big one. *Insight:* is the coupling map a *family* property (transferable) or per-checkpoint? *Frontier:* both.

**M38 — Coupling-strength as a layer selector.** *Motivation:* F5 collapse axis. *Method:* protect whole *layers* where local ρ(LC,CF) is highest instead of heads. *Insight:* head vs layer granularity of the coupling. *Frontier:* both.

**M39 — Learn the criterion (meta-gate).** *Motivation:* hand-picked z-score is crude. *Method:* a tiny per-head gate (logistic on the 8 LC metrics) trained to predict drift on held-out SFT, then frozen. *Insight:* which linear combo of LC metrics best predicts CF? (interpretable coefficients). *Frontier:* both.

---

## §4 · Inference-time methods (use the coupling at test time)

**M40 — Coupling-aware KV compression.** *Motivation:* F1 + DuoAttention. *Method:* keep full KV for high-coupling (retrieval/high-v_norm) heads, compress the rest. *Insight:* the *same* head set that we protect in training is the set to preserve in KV-compression → one map serves both. *Frontier:* LC (inference efficiency).

**M41 — Drift-aware merging / model soup.** *Motivation:* F1 (LC heads drift most). *Method:* when merging base+finetuned, **keep base weights for the high-coupling heads**, take finetuned elsewhere. *Insight:* surgical merge that recovers long-ctx for free. *Frontier:* both.

**M42 — Test-time sink/retrieval restoration.** *Motivation:* F3, sink-mass anchor. *Method:* at inference on a forgotten model, re-inject base sink K-bias / retrieval attention bias. *Insight:* can long-ctx be *restored* post-hoc by patching the coupled heads? (zero retraining). *Frontier:* LC.

**M43 — Coupling as a forgetting early-warning.** *Motivation:* F6. *Method:* monitor grad landing on LC heads during SFT; alarm/auto-throttle lr when it spikes. *Insight:* an online "forgetting meter." *Frontier:* CF (ops).

---

## §5 · Architectural methods

**M44 — Protected-head adapter split.** *Motivation:* F1. *Method:* route new-domain learning into *parallel* adapters on the non-coupled heads; coupled heads stay frozen + get a tiny residual adapter only. *Insight:* capacity for new tasks without touching the load-bearing set. *Frontier:* both.

**M45 — Sink/retrieval as explicit modules.** *Motivation:* F3+F4. *Method:* factor the sink and copy/retrieval circuits into named, freezable sub-modules at pretrain/continued-pretrain. *Insight:* if these are universal, make them first-class & protectable by design. *Frontier:* both (arch).

**M46 — MoE super-expert protection (the headline instantiation).** *Motivation:* §5.6a audit — super experts induce sinks; nobody protects them by the intrinsic criterion. *Method:* on Qwen3-30B-A3B, freeze super experts (sink-inducing) during SFT, adapt routed experts. *Insight:* does the dense coupling carry to MoE via super-experts? cleanest novelty. *Frontier:* both.

**M47 — Value-norm regularized attention.** *Motivation:* F2 (v_norm central). *Method:* a light arch tweak normalizing/​capping V-norm so no single head becomes a coupling bottleneck. *Insight:* can we *reduce* the coupling (distribute load) so forgetting is less catastrophic? (opposite of protect — *dilute*). *Frontier:* both.

---

## §6 · Training-dynamics / optimizer methods

**M48 — Coupling-preconditioned optimizer.** *Motivation:* F1+F6. *Method:* AdamW with per-head lr ∝ 1/(1+LC-coupling): the more LC-coupled, the slower it moves. *Insight:* a drop-in optimizer that protects without freezing — beats MoFO (momentum) using the LC criterion. *Frontier:* CF.

**M49 — Gradient surgery against the LC subspace.** *Motivation:* F1. *Method:* project the SFT gradient to be orthogonal to the base LC-head activation subspace (PCGrad-style). *Insight:* learn new domain while leaving long-ctx function invariant. *Frontier:* both.

**M50 — Replay-free LC self-distillation.** *Motivation:* SelfAug neighbour. *Method:* during SFT, add a KL term that keeps the **LC heads' attention maps** close to base on the *new-domain* inputs (no old data). *Insight:* anchor function, not weights; cheaper than full KL. *Frontier:* both.

**M51 — RL-instead-of-SFT for the coupled heads.** *Motivation:* `[rl-circuits]` (RL preserves circuits > SFT). *Method:* update coupled heads only via a KL-regularized RL objective, rest via SFT. *Insight:* does the RL-preservation effect localize to the coupled heads? *Frontier:* CF.

---

## §7 · Theory / measurement / negative-result papers

**M52 — Why are LC heads CF-vulnerable? (mechanism).** *Motivation:* F1 is a correlation. *Method:* causal mediation — do high-v_norm heads receive big grads *because* they have big activations (chain rule), or because the loss needs them? Decompose grad = activation × error. *Insight:* is the coupling a trivial chain-rule effect (big act→big grad) or functional? **Critical for the paper's depth.** *Frontier:* both.

**M53 — The collapse-axis theory.** *Motivation:* F5. *Method:* formalize late-layer representation collapse ↔ prediction crystallization ↔ massive activations as one phenomenon; tie to the coupling. *Insight:* a single-axis account of where computation "finishes." *Frontier:* ST.

**M54 — Scaling law of the coupling.** *Motivation:* F7. *Method:* measure ρ(LC,CF) vs model size 0.6→32B. *Insight:* does the coupling strengthen or weaken with scale? (predicts whether big models forget more surgically). *Frontier:* both.

**M55 — Decoupling control (negative result).** *Motivation:* falsifiability. *Method:* find/construct a setting where LC and CF are *decoupled* (e.g., RL, or a specific domain) → protection gives no benefit. *Insight:* the negative control that makes the positive claim credible. *Frontier:* both.

---

## §8 · Blue-sky / 天马行空

**M56 — "Two-faced" pretraining objective.** Co-train so that long-ctx-load and update-robustness are explicitly traded off per head → models born forgetting-resistant.

**M57 — Forgetting as compression.** Frame CF as the model overwriting its highest-MDL (most informative = LC) heads first; protection = an MDL prior. Connects to the surprisal/lens metrics.

**M58 — Head "insurance" market.** Allocate a fixed protection budget across heads by an auction on predicted (LC-value × CF-risk); learn the prices. A principled k-selection.

**M59 — Cross-task interference predictor.** Use the per-head coupling fingerprint to *predict which task pairs will interfere* before training (forgetting matrix from intrinsic signatures only).

**M60 — Long-context ⇒ continual-learning bridge.** If the same heads serve both, a model good at long-context should be intrinsically more continual-learnable. Test: does long-ctx ability correlate with retention across public models?

**M61 — Sink as a "global write" channel.** Treat the sink as a low-rank global memory; protect/edit it to inject knowledge without touching content heads (knowledge editing via the sink).

**M62 — Activation-drift steering vectors.** The act_drift directions of coupled heads = the "forgetting direction"; subtract it at inference to recover base ability (training-free un-forgetting).

---

## §9 · Priority (what to try first)
1. **M0/H3** (running) — the core claim. Then **M32 (soft anchor) vs EWC** and **M48 (preconditioned optimizer) vs MoFO** — the criterion-vs-baseline ablations that isolate DR9.
2. **M52** (mechanism: chain-rule vs functional) — decides whether the paper is "observation" or "mechanism."
3. **M46** (MoE super-expert) — the least-crowded novelty.
4. **M54** (scaling) + **M35** (data-agnostic transfer) — universality.
5. **M40/M41** (KV-compression / surgical merge) — cheap, high-impact spin-offs of the same head map.

## §10 · Cull notes
Likely dominated/risky: M19 (zero sink grads — sinks are load-bearing, may break), M47/M56 (arch changes — expensive, out of scope for v1), M58 (auction — cute, unclear payoff). Keep as colour.
