# LoRA / PEFT

> **Parameter-Efficient Fine-Tuning (PEFT)** is the family of techniques that adapt a large pre-trained model by training only a tiny number of new parameters, leaving the bulk frozen. **LoRA** is the dominant member: insert low-rank update matrices $\Delta W = BA$ on selected linear layers.

## Structure

**Contained by**: *fine-tuning* (broader; not a folder here yet).

**Contains**:
- **LoRA** — low-rank $\Delta W = BA$ on linear layers. Hu et al. 2022.
- **DoRA** — decoupled magnitude and direction in LoRA. Liu et al. 2024.
- **QLoRA** — LoRA on a 4-bit quantized base. Dettmers et al. 2023.
- **Prefix / prompt tuning** — train continuous prompt tokens; weights frozen. Li & Liang 2021.
- **Adapters** — small bottleneck MLPs inserted between transformer blocks (the original "adapter" family pre-dating LoRA).
- **LoRA ensembling / mixing** — LoRA-Hub, MoLE, etc.
- **Hypernet-generated LoRA** — see [`../hypernetworks/`](../hypernetworks/).
- **Serving for many LoRAs** — S-LoRA, Punica.

## Nearest neighbors

- [`hypernetworks`](../hypernetworks/) — hypernets in this repo's context almost always emit LoRA.
- [`model-editing`](../model-editing/) — modern editors store edits as LoRAs.
- [`context-distillation`](../context-distillation/) — D2L outputs a LoRA per document.

## Key concepts

- **Rank $r$** — the bottleneck of the low-rank update. Smaller $r$ = fewer parameters but less expressive. Typical: 4–64.
- **Target modules** — which weight matrices receive a LoRA. Common: `q_proj` and `v_proj` of attention; `down_proj` of MLP. D2L targets MLP `down_proj`; HyperLoRA targets `q,v`.
- **Merging** — at inference, $W + \alpha BA$ can be merged into the base weight for zero-overhead. But this commits to one adapter and loses modularity.
- **Composition** — sum, average, or weighted combination of multiple LoRAs. Interference is real and not well-understood.
- **Serving infrastructure** — S-LoRA / Punica enable batched serving of thousands of LoRAs concurrently. Critical for any plan that wants per-request or per-user LoRAs.

## Foundational references

(IDs reference [`docs/matrix/knowledge-sources.md`](../../docs/matrix/knowledge-sources.md).)

- [lora] **LoRA** (Hu et al., ICLR 2022) — the original.
- Dettmers, T. et al. (NeurIPS 2023). **QLoRA.**
- Liu, S. et al. (2024). **DoRA.**
- Li, X. L., Liang, P. (ACL 2021). **Prefix-Tuning.**
- Pfeiffer, J. et al. (EACL 2021). **AdapterFusion.**
- Houlsby, N. et al. (ICML 2019). **Parameter-Efficient Transfer Learning (Adapter).**
- [lorahub] **LoRAHub** (Huang et al., 2023) — dynamic LoRA composition at train time.
- [mole] **MoLE** (Wu et al., 2024) — Mixture of LoRA Experts at inference.
- [s-lora] **S-LoRA** (Sheng et al., MLSys 2024) — serving thousands of LoRAs concurrently.
- [punica] **Punica** (Chen et al., 2023) — multi-tenant LoRA serving (companion to S-LoRA).
- [lots-of-loras] **Lots-of-LoRAs** (HuggingFace dataset) — ~479 task-specific LoRAs ready to use.

## Open questions / live debates

- **Right rank for a given task** — practical guidance exists but no theory predicts the optimal $r$.
- **LoRA arithmetic** — adding / subtracting / negating LoRAs is empirically often meaningful (e.g., for unlearning) but theoretically poorly understood.
- **LoRA vs full fine-tuning gap** — closes for some tasks, persists for others. Cause unclear.
- **Serving overhead at scale** — even with S-LoRA, per-request LoRAs imply non-trivial memory bandwidth. Future-of-inference systems will need to budget for this.
