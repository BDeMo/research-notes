# Ideas index

> Catalog of all ideas in this repo with brief meta.
> For full descriptions, see the source brainstorm file linked per topic.
> Notation defined in [`memory/symbols.md`](../../memory/symbols.md).

```
Legend
S: ? seed · R read · F formalized · D drafted · P piloted · X executing
   W writing · S shipped · M maintenance · B blocked · K killed · Z archived
★: ★ notable · ★★ strong · ★★★ top
φ: 1/N chain · ↪#NN spawned NN · ←#NN continues NN · ≈#NN parallel
M: exp · proto · paper · thesis · prod · side  +  solo · collab · team · open
```

---

## Sources

| Topic | Source | Date | # ideas |
|---|---|---|---|
| Inference-time training (X-W framing) | [`inference-time-training.md`](inference-time-training.md) | 2026-05-26 | 50 |
| RCA model building — transformer-intrinsic adaptation | [`rca-transformer-intrinsic-2026-06-03.md`](rca-transformer-intrinsic-2026-06-03.md) | 2026-06-03 | 12 |

---

## Inference-time training — 50 ideas

### A. Supervision signal — what to use as the loss for W

| ID | Title | ★ | S | φ | M | Plan | Meta |
|---|---|---|---|---|---|---|---|
| A1 | Self-consistency as loss | ★ | ? | | | – | Distill the model's own majority vote into its weights |
| A2 | Reasoning-trace bootstrap | ★ | ? | | | – | Verifier-validated CoT becomes online RFT data |
| A3 | Reconstruction loss on context | | ? | | | – | Classic TTT self-supervised aux loss for LLM inputs |
| A4 | Contrastive on retrieved docs | | ? | | | – | InfoNCE on relevant vs irrelevant retrieved chunks |
| A5 | User implicit feedback | | ? | | | – | Accept/reject/rewrite → DPO pair, single-session update |
| A6 | Tool-call outcome reward | | ? | | | – | Environment feedback → one online RL step |
| A7 | Contrastive gradient between X-levels | ★ | ? | ≈#01 | | – | $\nabla_W[\log p(y\|x_{\text{simple}}) - \log p(y\|x_{\text{CoT}})]$ |

### B. Parameterization — what part of W to update

| ID | Title | ★ | S | φ | M | Plan | Meta |
|---|---|---|---|---|---|---|---|
| B1 | Per-request LoRA | ★ | ? | ≈#03 | | – | Tiny LoRA, lifetime = single request, discarded on completion |
| B2 | Steering vectors at hidden states | | ? | | | – | No weight update; learn a residual vector $v$ |
| B3 | KV-as-weight | | ? | | | – | SVD-compress KV cache into low-rank weight update |
| B4 | Trainable memory tokens | | ? | | | – | Update only $M$ learnable input tokens |
| B5 | Layer-selective TTT | | ? | | | – | Update only top-K layers / heads chosen by Fisher info |
| B6 | Sparse memory finetuning | | ? | | | – | Update only small subset of MLP rows (Lin 2025) |

### C. Trigger — when to update

| ID | Title | ★ | S | φ | M | Plan | Meta |
|---|---|---|---|---|---|---|---|
| C1 | Uncertainty-gated TTT | ★ | ? | | | – | Fire only when entropy / verifier disagreement crosses threshold |
| C2 | Error-driven TTT | | ? | | | – | Self-critique detects error → contrastive update |
| C3 | Streaming TTT | | ? | | | – | Continuous in-place updates across long dialog, with forgetting |
| C4 | Checkpoint & rollback | | ? | | | – | Probe-based gating to undo regressions |

### D. Search / exploration

| ID | Title | ★ | S | φ | M | Plan | Meta |
|---|---|---|---|---|---|---|---|
| D1 | W-space Best-of-N | ★★ | D | | paper | [03](../plans/03-w-space-best-of-n/) | Sample N weight deltas, verifier picks best — Best-of-N along W |
| D2 | MCMC over W | | ? | ←#03 | | – | Langevin-style sampler in weight space, gated by verifier |
| D3 | Joint (X, W) search | | ? | ≈#01 | | – | Alternating outer prompt search + inner weight update |

### E. Application domains

| ID | Title | ★ | S | φ | M | Plan | Meta |
|---|---|---|---|---|---|---|---|
| E1 | Codebase-adaptive coding agent | ★ | R | | | – | Entering new repo → TTT on its source for API/style adaptation |
| E2 | Long-document QA via Doc-to-LoRA | ★ | R | | | – | Internalize 100K-token doc into 50MB adapter |
| E3 | Personalization on-device | | ? | | | – | Nightly TTT on user history; weights stay on device |
| E4 | Safety / jailbreak defense | | ? | | | – | Adversarial pattern → session-scoped contrastive update |
| E5 | Robotics sim-to-real | | ? | | | – | Self-supervised dynamics loss on deployment |
| E6 | Diffusion test-time tuning | | ? | | | – | CLIP / reward model backprop into U-Net per sample |
| E7 | GUI agent online learning | | ? | ←#08 | | – | Verified-success rollouts as online RFT examples |

### F. Systems / serving

| ID | Title | ★ | S | φ | M | Plan | Meta |
|---|---|---|---|---|---|---|---|
| F1 | TTT serving infra | ★★ | ? | ←#08 | | – | Per-request weight delta in batched serving; runtime *writes* |
| F2 | Reusable TTT cache | | ? | | | – | Cluster per-request LoRAs by task/user → dynamic expert pool |
| F3 | Diff-quantized weight deltas | | ? | | | – | Sparse + quantized for cheap storage/transfer |
| F4 | Shared optimizer state proxy | | ? | | | – | Avoid per-request Adam state allocation |

### G. Theory / science

| ID | Title | ★ | S | φ | M | Plan | Meta |
|---|---|---|---|---|---|---|---|
| G1 | ICL ↔ TTT equivalence boundary | | ? | | | – | When does in-context = implicit gradient descent? |
| G2 | Information-theoretic X+W budget | ★ | ? | ≈#01 | | – | $I_{\text{needed}}$ vs $I(X) + I(W)$ — quantify bit transfer |
| G3 | Sample complexity of TTT | | ? | | | – | How many TTT steps = E epochs offline? |
| G4 | Stability / catastrophic forgetting | | ? | ≈#08 | | – | Capacity-preserving update directions (e.g., null-space) |
| G5 | Privacy of TTT | | ? | ≈#08 | | – | Does per-instance update leak into subsequent requests? |
| G6 | X-irreducible problems | ★ | ? | ≈#01 | | – | Characterize tasks where no X suffices; W must change |

### H. Cross-paradigm / wild

| ID | Title | ★ | S | φ | M | Plan | Meta |
|---|---|---|---|---|---|---|---|
| H1 | Inference-time RL alignment | ★ | ? | | | – | Per-conversation lightweight RLHF using a small reward model |
| H2 | TTT × World Model | | ? | | | – | Agent updates world model online → re-plan → update again |
| H3 | Reasoning trace → weight delta | ★ | ? | ←#08 | | – | Compile thinking into weights instead of token stream |
| H4 | TTT as compression | | ? | | | – | Internalize retrieved prompts into LoRA over time |
| H5 | Multi-model collaborative TTT | | ? | | | – | Small model generates supervision; reverse distillation |
| H6 | Model outputs ΔW as part of generation | ★★★ | D | 1/3 | thesis,solo | [08](../plans/08-model-outputs-delta-w/) | Output $(y, \Delta W)$ per turn; verifier gates apply |
| H7 | Inference-time ΔW as a tool call | ★ | ? | ←#08 | | – | Model emits `<learn>…</learn>` → D2L-like submodule → temp LoRA |

### I. X-W exchange framing (latest brainstorm)

| ID | Title | ★ | S | φ | M | Plan | Meta |
|---|---|---|---|---|---|---|---|
| I1 | X-saturation curve as dataset curator | ★★★ | D | 1/? | paper | [01](../plans/01-x-saturation-curve/) | Train on the X-saturated residual where W *must* change |
| I2 | X as virtual training data | ★★ | ? | ←#01 | | – | Convert "long-X-can-do" into "short-X-can-do" at population scale |
| I3 | X-then-W curriculum | ★★★ | F | ←#01 | | – | Train X-saturating, then distill X-residual into W, repeat |
| I4 | Inverse problem on X-trajectories | ★★ | ? | ←#01 | | – | $\min_W \|f(\text{short-X};W) - f(\text{long-X};W_0)\|$ |
| I5 | Prompt ↔ LoRA equivalence map | | ? | | | – | Reverse map: LoRA→prompt for interpretability + jailbreak defense |
| I6 | Inference-time compute allocator | | ? | ≈#J3 | | – | Tiny controller: spend FLOP on more X or on ΔW? |

### J. Architecture from method-reading (added 2026-05-28)

Born from reading Cartridges / Activation Beacon / Gisting / Generative Adapter against plan 08 v0. Source notes: [`docs/matrix/2026-05-28-methods-reading-and-new-plans.md`](../../docs/matrix/2026-05-28-methods-reading-and-new-plans.md).

| ID | Title | ★ | S | φ | M | Plan | Meta |
|---|---|---|---|---|---|---|---|
| J1 | Wrapper → LoRA distillation | ★★★ | F | ←#08, ↪? | thesis,solo | – | Train v0 memory wrapper first; learn hypernet `m_t → ΔW_t` to bridge v0 to plan 08 north star. Two-stage |
| J2 | Verifier-gated Generative Adapter | ★★ | F | ←#08 | paper | – | Add $\alpha_t = V(C_t, \Delta_t)$ to [genadapter] streaming update. The simplest plan-08-north-star variant testable today |
| J3 | Hybrid X+W chunk router | ★★ | F | ≈#I6 | paper | – | Per chunk decide soft-token (X) vs LoRA-delta (W) compression. Operationalizes I6 |
| J4 | SVD-normalized memory wrapper | ★ | F | ←#08, v0.1 | – | – | Adopt [genadapter]'s SVD normalization in v0 wrapper architecture |
| J5 | Beacon-style KV memory in v0.1 | ★★ | F | ←#08, v0.1 | – | – | v0 default (soft tokens) is weakest long-context choice per [act-beacon]. Add KV-activation memory ablation in v0 Phase-1 |

---

## RCA model building — 12 ideas (added 2026-06-03)

> **Problem** (user-posed): build Nokia's RCA model solving two challenges — long context at **inference** + catastrophic forgetting at **training** — with an *insightful, easy-to-adapt, lightweight, model-agnostic, task-free* method derived from **transformer-intrinsic phenomena**. Paper: strong on RCA *and* preserves code/math. Project: in-domain generalize to other RCA datasets/abilities → Nokia model release.
> **Unifying lens**: both challenges = "where is information allowed to flow"; transformers already have privileged sites (attention sinks, massive activations, induction heads). Confine read (context) + write (adaptation) to those natural sites.
> **Constraint prior**: mem-X v1 3-regime law → RCA is likely **Regime C** (verbatim needle) so favor *overlay existing KV/sinks*, not *add new soft tokens*.
> **Future room (not now)**: method should be extensible AR → **dLLM** — anchor the paper on the architecture-agnostic *principle* (not AR sinks specifically), abstract "site" as a `site_selector`, and favor architecture-agnostic levers (R4/R8/R7). dLLM is natively better at Regime C (masked reconstruction = infilling) and has a free adaptive-compute knob (denoising steps). See source §8.
> Full brainstorm (2 rounds + recipes + anti-patterns + decision questions + dLLM extension): [`rca-transformer-intrinsic-2026-06-03.md`](rca-transformer-intrinsic-2026-06-03.md).

| ID | Title | ★ | S | φ | M | Plan | Meta |
|---|---|---|---|---|---|---|---|
| R1 | Sink-anchored writable memory | ★★★ | F | | paper | – | Reuse attention-sink positions as a writable memory register; train ~500K sink-shaper, base frozen. Main lever for both problems |
| R2 | Induction-head freezing + adapter elsewhere | ★★★ | F | ≈#R1 | paper | – | Freeze induction heads (ICL carriers) during SFT → few-shot/ICL preserved; adapt elsewhere |
| R3 | Read-side KV reweighter | ★★ | F | ≈#R1 | paper | – | Tiny head rewrites *how* KV is read, not the KV; base frozen → verbatim preserved (fixes Regime C) |
| R4 | Per-head attention temperature $\tau_{l,h}$ | ★★ | F | ≈#R1 | paper | – | ~1K learnable scalars in softmax; per-task entropy knob; lightest possible lever |
| R5 | Logit-lens consistency aux loss | ★ | ? | | | – | Penalize layer-wise readout drift during SFT (tuned lens) → anti-forgetting |
| R6 | Massive-activation gradient mask | ★ | F | ≈#R1 | paper | – | Grad-mask massive-activation channels (Sun 2024) → protect global regulators = protect base ability |
| R7 | "Task-direction" steering | ★★ | ? | | | – | Refusal-direction analog for *adding* ability; find RCA-mode direction from contrast pairs, add at inference → zero train, zero forgetting |
| R8 | Layer-band LoRA + orthogonal singular constraint | ★★ | F | ≈#R1 | paper | – | LoRA only on middle 1/3 layers, projected to orthogonal complement of top-k singular subspace (REVIVE) |
| R9 | Cosine-similarity layer skipping (Cal-Skip) | ★ | ? | | | – | Skip redundant layers via inter-layer U-curve → saves long-ctx FLOPs |
| R10 | Sparse-write fine-tune (parameter-level) | ★ | ? | | | – | Update only ~5% of params that actually move in fine-tune |
| R11 | SAE-feature gated routing | ★ | ? | | | – | Route/freeze task-relevant SAE features; very high differentiation, needs instrumentation |
| R12 | Tuned-lens-aware early exit | ★ | ? | | | – | Adaptive early exit via tuned lens → saves long-ctx FLOPs |

**Recipes** (full in source): Light = R1+R6 · Medium = R1+R2+R8+R4 · Heavy = R1+R3+R7+R6+R5.

> ⚠️ **Prior-work audit (2026-06-03, source §10)**: a 2022-2026 lit review found **none of R1-R12 is novel as a standalone mechanism** — the space is saturated. Exact hits: **R8 = OPLoRA (AAAI 2026)**, **R10 = Sparse Memory Finetuning (Meta 2025)**, **R4 = SSA/SSMax**, **R3 = ReasonCache/KV Packet**, **R5 = Logit-Lens-Loss/SelfAug**, **R6 = MoFO/MIGU**, **R2 = Mechanistic-Forgetting-2026/ABFT**, **R7 = trained steering vectors**, **R11 = SAE-FT/SAE-FD**. R1 (sink-as-writable-memory) has only a thin unclaimed core. **Revised recommendation**: drop the "novel mechanism" framing; pursue either (A) a *unifying-observation* paper (same intrinsic sites solve long-ctx AND forgetting, RCA+code+math co-eval) or (B) a *verified-new-phenomenon* (sink key-bias-only tuning, unverified). See source §10.3-10.4.

---

## Top picks at a glance

Last refreshed 2026-05-28 after the 2026 prior-work landscape scan. **Full scored table**: [`evaluation-2026-05-28.md`](evaluation-2026-05-28.md).

| ID | Status | Verdict | Why it tops the list |
|---|---|---|---|
| **I1** | `D` | ★★★ | Plan 01; cleanest X-W operationalization, no direct 2026 preemption |
| **I3** | `F` | ★★★ | Sequel of I1; iterative X-then-W curriculum |
| **H6** | `D` | ★★★ | Plan 08 + v0; must differentiate from [in-place-ttt], [shine], [tempo] |
| **D1** | `D` | ★★★ | Plan 03; W-axis BoN, orthogonal to [tempo]'s RL-on-traces |
| **J1** | `F` | ★★★ | Wrapper → LoRA distillation; cleanest v0 → north-star bridge |
| **E1** | `R` | ★★ | Codebase-adaptive coding agent; high product value, SWE-Bench-Verified |
| **I6** | `?` | ★★ | X-W compute allocator; backed by [w-a-equiv] theory |
| **G5** | `?` | ★★ | Privacy of TTT; under-explored, important for deployment |
| **C4** | `?` | ★★ | Checkpoint & rollback; safety guard for plan 08 |
| **J3** | `F` | ★★ | Hybrid X+W chunk router; [w-a-equiv] grounded |

**Pivoted from earlier top-pick list**: J2 (preempted by [tempo]), J5 (preempted by [act-beacon]/[in-place-ttt] — absorb as v0 ablation), G2 (defer until I1 produces data). See evaluation file for full reasoning.

## How to use this index

- The **table** is the directory. Rows link out where deeper docs exist.
- The **source** column on top of this file points to the brainstorm document with full motivation, equations, risks.
- The **Plan** column links to the project plan folder if one has been drafted.
- When promoting an idea to a plan: bump `S` to `D` (or higher), populate Plan link, and add a `φ` mark if it's part of a chain.

---

## Maintenance

Full policy: [`docs/maintenance.md`](../../docs/maintenance.md). Local rules for this index:

- **This file is T1** — read every session. Soft cap **200 lines**, hard cap **300 lines**.
- **Add an idea**: pick an unused letter+number (e.g. `J1`); add one row; if `★≥★★` and you'd actually pursue it, link a plan placeholder.
- **Promote to plan**: `S` → `D`, add `Plan` link, add `φ` if part of a chain.
- **Dormancy**: any idea with `S = ?` for **> 12 months** is eligible for archive.
- **Archive**: `git mv` the source file to `notes/ideas/_archive/<topic>.md`, replace the source-table row with `archived <date>`, drop the per-idea rows or move them under a collapsed "Archive" section. Log the move in today's matrix entry.
- **Split**: if a single brainstorm source file > 500 lines, factor into a folder per §4 of `maintenance.md`.
- **Symbol legend** stays terse — full definitions live in [`memory/symbols.md`](../../memory/symbols.md), not here.
- **Quarterly evaluation snapshot**: re-score all rows against current prior work; save as `evaluation-YYYY-MM-DD.md` (T2). Archive previous snapshot to `_archive/` when ≥50% rows shift verdict. Current snapshot: [`evaluation-2026-05-28.md`](evaluation-2026-05-28.md).
