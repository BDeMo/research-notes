# Ideas evaluation snapshot — 2026-05-28

> Score every idea + plan + J-series candidate against 6 axes after a 2026 prior-work landscape scan. Snapshot only; status will drift, re-score quarterly.

## Scoring legend

| Column | Meaning |
|---|---|
| **Sig** | Significance if it works (1=marginal, 5=field-defining) |
| **Uniq** | Differentiation from 2026 prior work (5=blank slate, 1=fully preempted) |
| **TA** | Technical-approach concreteness (5=clear algorithm, 1=speculation) |
| **Res** | Data / code / benchmark accessibility (5=off-shelf, 1=needs new infra) |
| **$** | Budget tier — P (Pilot ≤$2K), S (Single paper $3-10K), L (Long paper $10-20K), T (Thesis $20K+) |
| **Closest 2026 prior** | The single most-preempting prior work (`[id]` in `knowledge-sources.md`) |
| **Verdict** | ★★★ GO (top), ★★ GO (queue), PIVOT (reframe), WATCH (defer), DROP (preempted/out-of-scope) |

**Composite intuition** (no formula — judgment):
Score ≈ Sig × Uniq × TA × Res / $-burden. Anything ≤ Uniq 1 demotes to PIVOT/DROP unless Res×Sig override. Anything ≤ TA 1 demotes to WATCH.

---

## Top picks at a glance (post-2026 refresh)

| Rank | ID | Title | Why | Verdict |
|---|---|---|---|---|
| 1 | **I1·Plan 01** | X-saturation curve + dataset curator | Drafted. Framing is *the* X-W operationalization. [scaling-tt] is the only adjacency; [ttc-rl] is online-not-offline. | ★★★ GO |
| 2 | **I3** | X-then-W curriculum | `S=F`. Sequel of I1 already at formalized status; no direct 2026 preemption. | ★★★ GO |
| 3 | **H6·Plan 08 v0** | Model outputs ΔW (Learned Memory Wrapper as v0) | Active engineering. Must differentiate from [in-place-ttt] + [shine] + [tempo]. v0 wrapper still distinctive. | ★★★ GO |
| 4 | **J1** | Wrapper → LoRA distillation | Two-stage bridge v0 → north star. Less crowded than direct hypernet-LoRA work. | ★★★ GO |
| 5 | **D1·Plan 03** | W-Space Best-of-N | Drafted. [tempo] is search-on-traces not weights — orthogonal to D1. | ★★★ GO |
| 6 | **E1** | Codebase-adaptive coding agent | High product value, SWE-Bench available, no direct preemption. | ★★ GO |
| 7 | **I6** | X-W inference-time compute allocator | Novel framing; gains theoretical backing from [w-a-equiv]. | ★★ GO |
| 8 | **G5** | Privacy of TTT in multi-tenant serving | Critical for production, under-explored, distinguishes plan 08 paper. | ★★ GO |
| 9 | **C4** | Checkpoint & rollback for inference-time updates | Safety-critical for plan 08 deployment, novel systems angle. | ★★ GO |
| 10 | **J3** | Hybrid X+W chunk router | Novel; [w-a-equiv] provides theoretical handle. | ★★ GO |

---

## Tier 1 — GO ★★★ (draft full plans now)

| ID | Title | Sig | Uniq | TA | Res | $ | Closest 2026 prior | Tech issue → approach | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| **I1 · Plan 01** | X-saturation curve | 5 | 5 | 4 | 5 | S | [scaling-tt], [ttc-rl] (online variant) | Per-example X-budget sweep, $\epsilon$-plateau detection, threshold-$\tau$ → W-residual subset | ★★★ |
| **D1 · Plan 03** | W-Space Best-of-N | 5 | 4 | 4 | 5 | S | [self-cons] (X-BoN), [tempo] (RL not BoN) | $\Delta W_i \sim \mathcal{N}$ / library / [shine]-style hypernet; verifier selects | ★★★ |
| **H6 · Plan 08 (+v0)** | Model outputs ΔW | 5 | 3 | 3 | 4 | T | [in-place-ttt], [shine], [tempo], [genadapter] | Model emits `<learn>` → $\mathcal{G}$ outputs $\Delta W$ → verifier $\alpha_t$ gates. v0 = frozen + memory wrapper. North star = + verifier + RL | ★★★ |
| **I3** | X-then-W curriculum | 5 | 4 | 3 | 4 | S | [restem], plan 01 (parent) | Iterate: saturate X → distill X-residual into W → reset X | ★★★ |
| **J1** | Wrapper → LoRA distillation | 5 | 4 | 3 | 4 | L | [shine], [genadapter] | Phase A: train v0 memory wrapper (tokens). Phase B: hypernet $H: m_t \to \Delta W_t$ distillation. Two-stage decoupling tames training. | ★★★ |

## Tier 2 — GO ★★ (queue for next plan-drafting cycle)

| ID | Title | Sig | Uniq | TA | Res | $ | Closest 2026 prior | Tech issue → approach | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| **E1** | Codebase-adaptive coding agent | 5 | 4 | 3 | 4 | S | [genadapter] (StreamingQA), Cursor / Aider | Repo → adapter; eval on SWE-Bench-Verified; per-repo $\Delta W$ scoped to session | ★★ |
| **F1** | TTT serving infra | 5 | 4 | 3 | 4 | S | [s-lora], [punica] (trained LoRAs only) | Runtime $\Delta W$ in batched serving; rollback path; p99 latency budget; verifier in inference loop | ★★ |
| **G5** | Privacy of TTT | 5 | 5 | 2 | 3 | S | (Open) | Threat model first; differential-privacy LoRA delta; membership inference on per-request updates | ★★ |
| **G6** | X-irreducible problems | 4 | 5 | 3 | 4 | S | (Open) | Characterize tasks where no X-budget suffices using plan 01 dataset; classifier on (task, model) → X/W-budget split | ★★ |
| **C4** | Checkpoint & rollback | 5 | 4 | 3 | 4 | S | [tempo] (recalibration but no rollback) | Probe set ⇒ snapshot ⇒ rollback if metric degrades; systems paper of plan 08's safety guard | ★★ |
| **I2** | X as virtual training data | 4 | 4 | 3 | 4 | S | [restem] (filter then SFT, doesn't reframe X) | Generate long-X traces with strong model → contrastive supervision on short-X student | ★★ |
| **I6** | X-W compute allocator | 5 | 5 | 3 | 3 | S | [w-a-equiv] (theoretical handle) | Tiny router per token / chunk picks "X more" vs "W update"; trained on (Δ acc, Δ FLOP) | ★★ |
| **B3** | KV-as-weight | 4 | 5 | 2 | 4 | S | (Genuinely open) | SVD-compress KV cache → LoRA update; revisit at long-context | ★★ |
| **A7** | Contrastive gradient between X-levels | 4 | 5 | 3 | 4 | S | (Open) | $\nabla_W [\log p(y\|x_\text{simple}) - \log p(y\|x_\text{CoT})]$ as loss | ★★ |
| **H3** | Reasoning trace → ΔW | 5 | 4 | 2 | 3 | L | [in-place-ttt] (NTP fast weights, not trace-conditioned) | "Compile" CoT into a weight delta rather than rendering as tokens. Strong if a clean compilation scheme exists | ★★ |
| **H7** | Inference-time ΔW as tool call | 4 | 5 | 3 | 3 | S | (Open) | Model emits `<learn>...</learn>` → D2L-like submodule → temp LoRA. Verifier gates. Cleaner agent angle | ★★ |
| **J3** | Hybrid X+W chunk router | 4 | 5 | 3 | 3 | S | [w-a-equiv] (theoretical) | Per-chunk decide soft-token (X) vs LoRA-delta (W) compression; learned router | ★★ |
| **C1** | Uncertainty-gated TTT | 4 | 4 | 3 | 4 | S | [tempo] (reward-gated, not uncertainty-gated) | Fire only when entropy / verifier-disagreement exceeds threshold | ★★ |

## Tier 3 — PIVOT (reframe in light of 2026 prior work)

| ID | Title | Why pivot | Suggested new angle |
|---|---|---|---|
| **A1** | Self-consistency as loss | [tempo] does verifier-gated TTT for reasoning; [star] family covers offline | Must specify single-session scope + non-reasoning domain (e.g. dialogue persona) |
| **A2** | Reasoning-trace bootstrap | Same — [vstar], [restem], [tempo] | Position as plan 01 sub-component, not standalone |
| **B1** | Per-request LoRA | [genadapter] + [s-lora] cover the mechanism | Push to F1 (systems) or J2 angle |
| **B2** | Steering vectors at hidden states | RepE field is now a saturated subfield — [repe-survey], [odesteer], [austeer], [w-a-equiv] | Only valuable if it brings RepE *into* X-W exchange framework (sibling of W-axis) |
| **B4** | Trainable memory tokens | [gisting] + [act-beacon] preempt | Survive only as v0 ablation, not as standalone idea |
| **B5** | Layer-selective TTT | [in-place-ttt] picks MLP final proj | Pivot to "*which* layer set for *which* task" — empirical mapping |
| **C3** | Streaming TTT | [in-place-ttt] + [genadapter] recurrent state cover | Pivot to verifier-gated streaming with rollback (overlap with C4) |
| **E2** | Long-document QA via D2L | Literally [d2l] | Drop as research; survives as application demo for plan 08 v0 |
| **G4** | Stability / catastrophic forgetting | [mose] [crispedit] [rlsedit] [stable-edit] [beta-edit] [revive] all attack this | Pick ONE differentiator (e.g., "stability *for verifier-gated streams*") or drop in favor of using their tools |
| **J2** | Verifier-gated Generative Adapter | [tempo] does verifier-gated TTT, [shine] is closer to genadapter | Pivot to "single-pass version of TEMPO" — emphasize the no-EM-loop case |
| **J4** | SVD-normalized memory wrapper | [genadapter] uses SVD norm; [revive] does dominant-subspace; [ouroboros] uses SVD-init bases | Absorb into v0 as ablation; not its own idea |
| **J5** | Beacon KV memory in v0.1 | [act-beacon] paper *is* this; [in-place-ttt] also touches | Absorb into v0 Phase-1 ablation; cite Beacon as baseline |

## Tier 4 — WATCH (defer; might mature with more evidence)

| ID | Title | Wait for what |
|---|---|---|
| **A4** | Contrastive on retrieved docs | Wait for plan 01 to surface where retrieval helps vs hurts |
| **A5** | User implicit feedback | Need realistic feedback data; piggyback on E3 if it lands |
| **A6** | Tool-call outcome reward | Wait for plan 08 v0 to ship; then a natural extension |
| **C2** | Error-driven TTT | Subsumed by C1+C4 |
| **D2** | MCMC over W | Compute-prohibitive without plan 03 first |
| **D3** | Joint (X, W) search | Combinatorial; wait for I6 router |
| **E3** | Personalization on-device | Needs data + ethical scaffolding |
| **E4** | Safety/jailbreak defense | Separate field; partner with alignment researcher if pursued |
| **E7** | GUI agent online learning | Wait for plan 08 v0 ship + agent infra |
| **F2** | Reusable TTT cache | Subsumed by F1; spin out if F1 lands |
| **F3** | Diff-quantized weight deltas | Engineering optimization; deferred |
| **F4** | Shared optimizer state proxy | Niche; deferred |
| **G1** | ICL ↔ TTT equivalence | [icl-bayes] + [w-a-equiv] starting it; theory needs experimental anchor |
| **G2** | X+W info budget | Defer until plan 01 data exists |
| **G3** | Sample complexity of TTT | Defer until plan 03 + plan 08 v0 give numbers |
| **H1** | Inference-time RL alignment | [tempo] partial; needs alignment-specific framing |
| **H2** | TTT × World Model | Too speculative for current resources |
| **H5** | Multi-model collaborative TTT | Open but research-budget heavy |
| **I4** | Inverse problem on X-trajectories | Theoretically interesting; defer until I1 lands |
| **I5** | Prompt ↔ LoRA equivalence map | [w-a-equiv] just established direction; build on top later |

## Tier 5 — DROP (preempted, out of scope, or low-value)

| ID | Title | Reason |
|---|---|---|
| **A3** | Reconstruction loss on context | [in-place-ttt] explicitly rejected reconstruction in favor of NTP — *and shows that's the right call* |
| **B6** | Sparse memory finetuning | Is literally [sparse-mem] (Lin 2025) |
| **E5** | Robotics sim-to-real | Outside LLM scope |
| **E6** | Diffusion test-time tuning | Outside LLM scope (separate diffusion track) |
| **H4** | TTT as compression | [cartridges] / [in-place-ttt] / [act-beacon] cover this fully |

---

## 2026 hot themes — landscape map

The five clusters that emerged from this scan. Idea uniqueness depends on which cluster they sit near.

### A. TTT for LLMs (red-hot, ICLR/NeurIPS 2026)
Core papers: [in-place-ttt] (ICLR 2026 Oral) · [tempo] · [lact] · [ttc-rl] · [ttt-2020] (origin) · [ttt-layers]
Key signals: TTT moved from "vision auxiliary loss" → "LLM fast weights with NTP objective + chunked update + 128K context support". The architectural recipe is settling.

### B. Sequential / lifelong editing stability (now a subfield)
Core papers: [mose] · [crispedit] · [rlsedit] · [stable-edit] · [ultra-edit] · [beta-edit] · [revive] · [alphaedit] (older).
Key signals: 5+ stability frameworks just dropped — multiplicative orthogonal, curvature-projected, recursive LS, LN-normalized, null-space history-aware, dominant-subspace-protected. Million-scale edit streams now demonstrated. **G4 is no longer a research opportunity; it's a toolbox we should use.**

### C. Hypernet → LoRA (mature but still expanding)
Core papers: [d2l] · [t2l] · [genadapter] · [shine] · [ouroboros] · [hint] · [hypertune] · [hyperlora]
Key signals: SHINE's M2P transformer + bidirectional self-attention is the state-of-the-art for "context → LoRA in one pass". OUROBOROS shows freezing the LoRA bases + training only a modulator is viable. Plan 08 north star at first order = [shine] + verifier gating.

### D. LoRA merging interference
Core papers: [pico] · [iso-c] · [lorahub] · [mole] · [ties] (older) · [task-arith] (older)
Key signals: 2026 progress is mostly calibration tricks (Pico) + spectral methods (iso-c). Plan 03's WBoN-Library could benefit but isn't blocked.

### E. Representation engineering (matured into field)
Core papers: [repe-survey] · [odesteer] · [w-a-equiv] · [austeer] · ACTIVATIONREASONING · COLD-Steer · PSR · SteerCLR
Key signals: The 2026 breakthrough is [w-a-equiv] — first-order equivalence between weight-space updates and activation-space interventions, with post-block output identified as the most expressive intervention point. **This directly enables I5 / I6 / J3 theory.** Activation steering itself is saturated for off-the-shelf moves; novelty is in *integration* with X-W axis (B2 pivot).

---

## Suggested new ideas inspired by this scan (not yet in TOC)

Three angles where the 2026 prior work points to a clean opening; can be added to `ideas/README.md` as K-series on user approval.

| Proposed ID | Title | Why now |
|---|---|---|
| **K1** | Multiplicative ΔW for plan 08 north star ([mose]-style) | Replace additive $\theta_{t+1} = \theta_t + \alpha_t \Delta W_t$ with $\theta_{t+1} = R_t \cdot \theta_t$, $R_t^\top R_t = I$. Inherits [mose]'s lifelong stability guarantees. The cleanest answer to plan 08's "thousands of updates without drift" problem. |
| **K2** | Joint weight+activation adaptation ([w-a-equiv]-inspired) | Per chunk decide: emit a $\Delta W$ (post-block weight delta) or an activation intervention $v$. Theory ground in [w-a-equiv]; gives I6 / J3 a principled substrate. |
| **K3** | Critic-recalibrated v0 ([tempo]-style E-step) | Add an E-step that periodically recalibrates the wrapper using a small labeled probe set, alternating with the M-step wrapper training. Direct [tempo] analog at the wrapper-level. Stabilizes long-running v0 deployment. |

---

## How to use this snapshot

- **Each session start**: glance at "Top picks at a glance" only (10 rows). Don't reread the full table.
- **When promoting an idea to a plan**: re-check its row to refresh against current prior work.
- **When a new important paper drops**: log its ID in `knowledge-sources.md`, then re-rate any idea row where the new paper is now the closest prior. Mark the row date.
- **Quarterly**: re-score the whole table. Snapshot to dated file like this one and archive the previous snapshot to `_archive/`.
- **Trigger for archive**: when ≥ 50% of rows have shifted verdict.
