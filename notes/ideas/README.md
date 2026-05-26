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
| I6 | Inference-time compute allocator | | ? | | | – | Tiny controller: spend FLOP on more X or on ΔW? |

---

## Top picks at a glance

| ID | Status | Why it tops the list |
|---|---|---|
| **I1** | `D` | First plan drafted; the framing's most concrete instance |
| **I3** | `F` | Natural continuation of I1; PhD-thread material |
| **H6** | `D` | Most ambitious; longest-horizon plan (08) |
| **D1** | `D` | Sharpest single-paper opportunity (plan 03) |
| **G2** | `?` | Theory backbone for the X-W framing — needed before scaling |

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
