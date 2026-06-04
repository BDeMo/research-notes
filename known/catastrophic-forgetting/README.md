# Catastrophic forgetting

> When fine-tuning a pretrained model on a new domain erodes previously acquired ability (code, math, general knowledge, in-context learning). The **write-side** problem: where weight updates land and which prior-capability carriers they disturb. Central to [plan 09](../../notes/plans/09-intrinsic-site-protection/) (the anti-forgetting half).

## Structure

**Contained by**: (top-level).

**Contains**:
- **Regularization-based** — penalize moving important params: EWC (Fisher), L2-SP, MAS; distillation: LwF, SelfAug ([selfaug]), TMKL ([tmkl]), logit/lens-consistency ([logit-lens-loss], [distill-lens]).
- **Selective / sparse update** — update only a subset of params: MoFO ([mofo], momentum-filtered), MIGU ([migu], magnitude-gated), Sparse Memory Finetuning ([smf]), LISA.
- **Subspace-constrained** — keep updates off the general-ability subspace: OPLoRA ([oplora]), CLoRA ([clora]).
- **Replay / rehearsal** — mix in old-distribution data (rehearsal, experience replay, pseudo-replay).
- **Model averaging** — interpolate tuned and base weights: wise-ft, model soups.
- **MoE-specific** — confine the write to a few experts/router: ESFT ([esft]), DES-MoE ([des-moe]), LoRAMoE ([loramoe]), Same ([same-moe]), Lifelong-MoE ([lifelong-moe]).
- **Mechanistic accounts** — *where* forgetting happens: attention-head localization ([mech-forget]), value-matrix preserves ICL ([ft-no-forget-icl]), SAE-feature preservation ([sae-ft], [sae-fd]).

## Nearest neighbors

See [the main graph](../README.md#nearness-graph). Closest:
- [`transformer-internals`](../transformer-internals/) — **the key edge**: the sites perturbed during forgetting (sinks / massive activations / super experts) are the intrinsic load-bearers. Plan-09 thesis.
- [`lora-peft`](../lora-peft/) — the adaptation mechanism whose updates we constrain.
- [`model-editing`](../model-editing/) — sibling: both ask "which weights carry capability, and how to change them surgically."
- [`inference-time-training`](../inference-time-training/) — TTT/online updates risk forgetting too (plan 08 G4 "capacity-preserving updates").

## Key concepts

- **Stability–plasticity trade-off** — retaining old ability (stability) vs learning the new domain (plasticity); methods live on this frontier.
- **Backward Transfer (BWT)** — change in old-task accuracy after learning a new task; the standard forgetting metric.
- **Selection criterion** — *which* params to protect. Existing criteria: Fisher importance (EWC), gradient/momentum magnitude (MoFO/MIGU), task-affinity (ESFT), top singular subspace (OPLoRA). **Plan-09's bet: an intrinsic long-context-importance criterion (DR9).**
- **Continual / sequential FT** — forgetting compounds over a domain stream; measured by TRACE-style suites.

## Foundational references

- [mech-forget] Mechanistic analysis: freezing attention ↓ forgetting 64%. The nearest threat to our framing.
- [oplora] OPLoRA · [mofo] MoFO · [migu] MIGU · [smf] Sparse Memory Finetuning — selective/subspace-constrained updates.
- [esft] ESFT · [des-moe] DES-MoE · [loramoe] LoRAMoE · [same-moe] Same · [lifelong-moe] Lifelong-MoE — MoE continual learning.
- [selfaug] SelfAug · [tmkl] TMKL · [sae-ft] SAE-FT — distillation / feature-preservation.
- Classics (to log): EWC (Kirkpatrick 2017) · L2-SP (Xuhong 2018) · LwF (Li & Hoiem 2017) · wise-ft / model soups (Wortsman 2022).

## Open questions / live debates

- Is forgetting better fought at **selection** (which params) or **regularization** (how hard to hold them)? Plan-09 is a selection bet.
- Does the **same** protected set defend against *any* new domain (data-agnostic), or is it domain-specific?
- **Coupling with long context**: are forgetting-vulnerable sites the same as long-context load-bearers? ([transformer-internals](../transformer-internals/) + plan-09 H2.)
