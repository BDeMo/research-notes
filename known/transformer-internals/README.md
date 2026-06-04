# Transformer internals (intrinsic phenomena)

> Empirically observed, architecture-universal structures inside trained transformers — a few token positions, channels, heads, or experts that act as **load-bearing sites**. The **read-side substrate** of [plan 09](../../notes/plans/09-intrinsic-site-protection/): these sites carry long-context behavior, and the plan's thesis is that they are *also* the sites whose perturbation causes forgetting.

## Structure

**Contained by**: (top-level). Overlaps mechanistic interpretability.

**Contains**:
- **Attention sinks** — a few initial / delimiter positions absorb disproportionate attention; stabilize streaming + long context. [sink-streaming], [sink-emerge] (sink = key bias), [sink-first-token], [sink-ctr].
- **Massive activations** — a handful of residual-stream channels with 100–1000× magnitude; act as implicit bias, deleting them breaks the model. [massive-act].
- **Super experts (MoE)** — a tiny set of experts that *induce* the attention sinks; pruning them collapses the model (sink-decay >90%). [super-experts], [sink-native-moe].
- **Induction / retrieval heads** — heads implementing in-context copy (induction) and needle retrieval; carriers of ICL and long-context retrieval. Olsson 2022, Wu 2024 (to log).
- **Residual-stream directions** — linear "feature"/task directions (refusal direction, steering vectors). [steer-reason], [instruct-steer].
- **Readout lenses** — logit lens / tuned lens: project hidden states to vocab to inspect per-layer computation. (to log: Belrose tuned lens.)
- **SAE features** — sparse-autoencoder-decoded interpretable features. [sae-ft], [sae-fd].

## Nearest neighbors

See [the main graph](../README.md#nearness-graph). Closest:
- [`catastrophic-forgetting`](../catastrophic-forgetting/) — **the key edge**: these intrinsic sites are the hypothesized forgetting-vulnerable substrate (plan-09 thesis).
- [`long-context`](../long-context/) — attention sinks were discovered for streaming long context; retrieval heads govern effective context.
- [`model-editing`](../model-editing/) — both localize capability to specific weights/activations (knowledge neurons ≈ massive activations).

## Key concepts

- **Load-bearing site** — a small set of positions/channels/heads/experts whose ablation disproportionately damages the model.
- **Sink ↔ super-expert identity** — in MoE, specific experts produce the sinks; the dense (sinks/massive-act) and MoE (super-experts) substrates are the same phenomenon at different granularities.
- **Read-time overload vs write-time perturbation** — the *same* sites can fail two ways: overwhelmed by long context (read), or disturbed by fine-tuning (write). The unifying lens of plan 09 (DR8).
- **`site_selector` abstraction** — treat "site" as a first-class, swappable criterion (AR sinks ↔ dLLM mask/BOS) for architecture-agnostic methods (DR15).

## Foundational references

- [sink-streaming] StreamingLLM / attention sinks · [sink-emerge] sink = key bias · [massive-act] Massive Activations.
- [super-experts] Super Experts induce sinks · [sink-native-moe] sink weight = implicit gating.
- [gated-attn-sinkfree] sink-free gated attention (+10 RULER) — counterpoint that sinks are removable.
- To log: Olsson 2022 (induction heads) · Wu 2024 (retrieval heads) · Belrose (tuned lens).

## Open questions / live debates

- Are sinks a **bug** (softmax artifact, removable per [gated-attn-sinkfree]) or a **feature** (necessary global regulator)? Plan-09 treats them as load-bearing regardless.
- Do these sites **coincide** across the long-context and forgetting axes? (plan-09 H2 — the whole bet.)
- Does the dense↔MoE substrate identity hold across families, and does it port to **diffusion LLMs** (entirely unstudied)?
