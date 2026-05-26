# Ideas index

> Catalog of all ideas in this repo with brief meta.
> For full descriptions, see the source brainstorm file linked per topic.

Statuses: `idea` (raw) · `drafted` (plan exists) · `running` (pilot in progress) · `shipped` (paper / public artifact) · `killed`.
Priority: `★` notable · `★★` strong · `★★★` top.

---

## Sources

| Topic | Source | Date | # ideas |
|---|---|---|---|
| Inference-time training (X-W framing) | [`inference-time-training.md`](inference-time-training.md) | 2026-05-26 | 50 |

---

## Inference-time training — 50 ideas

### A. Supervision signal — what to use as the loss for W

| ID | Title | ★ | Status | Plan | Meta |
|---|---|---|---|---|---|
| A1 | Self-consistency as loss | ★ | idea | – | Distill the model's own majority-vote into its weights |
| A2 | Reasoning-trace bootstrap | ★ | idea | – | Verifier-validated CoT becomes online RFT data |
| A3 | Reconstruction loss on context | | idea | – | Classic TTT self-supervised aux loss for LLM inputs |
| A4 | Contrastive on retrieved docs | | idea | – | InfoNCE on relevant vs irrelevant retrieved chunks |
| A5 | User implicit feedback | | idea | – | Accept/reject/rewrite → DPO pair, single-session update |
| A6 | Tool-call outcome reward | | idea | – | Environment feedback → one online RL step |
| A7 | Contrastive gradient between X-levels | ★ | idea | – | $\nabla_W[\log p(y\|x_{\text{simple}}) - \log p(y\|x_{\text{CoT}})]$ |

### B. Parameterization — what part of W to update

| ID | Title | ★ | Status | Plan | Meta |
|---|---|---|---|---|---|
| B1 | Per-request LoRA | ★ | idea | – | Tiny LoRA, lifetime = single request, discarded on completion |
| B2 | Steering vectors at hidden states | | idea | – | No weight update; learn a residual vector $v$ |
| B3 | KV-as-weight | | idea | – | SVD-compress KV cache into low-rank weight update |
| B4 | Trainable memory tokens | | idea | – | Update only $M$ learnable input tokens |
| B5 | Layer-selective TTT | | idea | – | Update only top-K layers / heads chosen by Fisher info |
| B6 | Sparse memory finetuning | | idea | – | Update only small subset of MLP rows (Lin 2025) |

### C. Trigger — when to update

| ID | Title | ★ | Status | Plan | Meta |
|---|---|---|---|---|---|
| C1 | Uncertainty-gated TTT | ★ | idea | – | Fire only when entropy / verifier disagreement crosses threshold |
| C2 | Error-driven TTT | | idea | – | Self-critique detects error → contrastive update |
| C3 | Streaming TTT | | idea | – | Continuous in-place updates across long dialog, with forgetting |
| C4 | Checkpoint & rollback | | idea | – | Probe-based gating to undo regressions |

### D. Search / exploration

| ID | Title | ★ | Status | Plan | Meta |
|---|---|---|---|---|---|
| D1 | W-space Best-of-N | ★★ | drafted | [03](../plans/03-w-space-best-of-n/) | Sample N weight deltas, verifier picks best — Best-of-N along W |
| D2 | MCMC over W | | idea | – | Langevin-style sampler in weight space, gated by verifier |
| D3 | Joint (X, W) search | | idea | – | Alternating outer prompt search + inner weight update |

### E. Application domains

| ID | Title | ★ | Status | Plan | Meta |
|---|---|---|---|---|---|
| E1 | Codebase-adaptive coding agent | ★ | idea | – | Entering new repo → TTT on its source for API/style adaptation |
| E2 | Long-document QA via Doc-to-LoRA | ★ | idea | – | Internalize 100K-token doc into 50MB adapter |
| E3 | Personalization on-device | | idea | – | Nightly TTT on user history; weights stay on device |
| E4 | Safety / jailbreak defense | | idea | – | Adversarial pattern → session-scoped contrastive update |
| E5 | Robotics sim-to-real | | idea | – | Self-supervised dynamics loss on deployment |
| E6 | Diffusion test-time tuning | | idea | – | CLIP / reward model backprop into U-Net per sample |
| E7 | GUI agent online learning | | idea | – | Verified-success rollouts as online RFT examples |

### F. Systems / serving

| ID | Title | ★ | Status | Plan | Meta |
|---|---|---|---|---|---|
| F1 | TTT serving infra | ★★ | idea | – | Per-request weight delta in batched serving; runtime *writes* |
| F2 | Reusable TTT cache | | idea | – | Cluster per-request LoRAs by task/user → dynamic expert pool |
| F3 | Diff-quantized weight deltas | | idea | – | Sparse + quantized for cheap storage/transfer |
| F4 | Shared optimizer state proxy | | idea | – | Avoid per-request Adam state allocation |

### G. Theory / science

| ID | Title | ★ | Status | Plan | Meta |
|---|---|---|---|---|---|
| G1 | ICL ↔ TTT equivalence boundary | | idea | – | When does in-context = implicit gradient descent? |
| G2 | Information-theoretic X+W budget | ★ | idea | – | $I_{\text{needed}}$ vs $I(X) + I(W)$ — quantify bit transfer |
| G3 | Sample complexity of TTT | | idea | – | How many TTT steps = E epochs offline? |
| G4 | Stability / catastrophic forgetting | | idea | – | Capacity-preserving update directions (e.g., null-space) |
| G5 | Privacy of TTT | | idea | – | Does per-instance update leak into subsequent requests? |
| G6 | X-irreducible problems | ★ | idea | – | Characterize tasks where no X suffices; W must change |

### H. Cross-paradigm / wild

| ID | Title | ★ | Status | Plan | Meta |
|---|---|---|---|---|---|
| H1 | Inference-time RL alignment | ★ | idea | – | Per-conversation lightweight RLHF using a small reward model |
| H2 | TTT × World Model | | idea | – | Agent updates world model online → re-plan → update again |
| H3 | Reasoning trace → weight delta | ★ | idea | – | Compile thinking into weights instead of token stream |
| H4 | TTT as compression | | idea | – | Internalize retrieved prompts into LoRA over time |
| H5 | Multi-model collaborative TTT | | idea | – | Small model generates supervision; reverse distillation |
| H6 | Model outputs ΔW as part of generation | ★★★ | drafted | [08](../plans/08-model-outputs-delta-w/) | Output $(y, \Delta W)$ per turn; verifier gates apply |
| H7 | Inference-time ΔW as a tool call | ★ | idea | – | Model emits `<learn>…</learn>` → D2L-like submodule → temp LoRA |

### I. X-W exchange framing (latest brainstorm)

| ID | Title | ★ | Status | Plan | Meta |
|---|---|---|---|---|---|
| I1 | X-saturation curve as dataset curator | ★★★ | drafted | [01](../plans/01-x-saturation-curve/) | Train on the X-saturated residual where W *must* change |
| I2 | X as virtual training data | ★★ | idea | – | Convert "long-X-can-do" into "short-X-can-do" at population scale |
| I3 | X-then-W curriculum | ★★★ | idea | – | Train X-saturating, then distill X-residual into W, repeat |
| I4 | Inverse problem on X-trajectories | ★★ | idea | – | $\min_W \|f(\text{short-X};W) - f(\text{long-X};W_0)\|$ |
| I5 | Prompt ↔ LoRA equivalence map | | idea | – | Reverse map: LoRA→prompt for interpretability + jailbreak defense |
| I6 | Inference-time compute allocator | | idea | – | Tiny controller: spend FLOP on more X or on ΔW? |

---

## How to use this index

- The **table** is the directory. Rows link out where deeper docs exist.
- The **source** column on top of this file points to the brainstorm document that has the full motivation, equations, and risk discussion.
- The **Plan** column links to the project plan folder if one has been drafted.
- When promoting an idea: change **Status** to `drafted` and add the plan link.
