# Plan 09 — References

Closest prior, grouped by role. Full entries (with arXiv IDs + our take) live in [`../../../docs/matrix/knowledge-sources.md`](../../../docs/matrix/knowledge-sources.md) under "RCA prior-work audit" + "MoE prior-work audit". The brainstorm §5 audit is the reasoning.

## Intrinsic-site phenomena (the substrate we measure)
- `[sink-streaming]` Xiao 2023 — StreamingLLM / attention sinks (origin + training-free long-ctx baseline).
- `[sink-emerge]` When Attention Sink Emerges (2410.10781) — sink = key bias; basis for the key-bias variant.
- `[massive-act]` Sun 2024 — Massive Activations; deleting them breaks the model → forgetting-vulnerability prior.
- `[super-experts]` Unveiling Super Experts (2507.23279) — MoE experts that *induce* sinks; the dense↔MoE bridge.
- `[sink-native-moe]` Attention Sink Forges Native MoE (2602.01203) — sink weight = implicit gating.
- Retrieval heads (Wu et al. 2024) · Induction heads (Olsson 2022) — long-ctx head criteria. *(to log)*

## Nearest threats (must differentiate — DR12) — refreshed 2026-06-04 audit (brainstorm §5.6)
- `[mech-forget]` Mechanistic Analysis of Catastrophic Forgetting (2601.18699) — forgetting localizes to heads (freezing ↓64%; ablating the 20% **ΔW-disrupted** heads restores 47%). **THE comparison.** Our edge: their damaged-set is **post-hoc by ΔW (write-side)**; ours is **a-priori by long-context importance (read-side, data-agnostic)** — H2 shows the two rankings coincide → predict-before-training protection.
- `[lccp-dyn]` Learning Dynamics of Long-Context Continual Pre-training (2604.02650) — retrieval heads stable (>93%), refined during long-ctx CPT; touches forgetting. **Edge**: they study continual *pre-training* dynamics (descriptive); we study downstream *SFT* forgetting + give a protection method keyed on the read-side criterion.
- `[rl-circuits]` Mechanistic origins of forgetting: RL vs SFT (2605.28860) — differential circuit vulnerability at head level. Adjacent; no long-ctx coupling, no method.
- `[duo-attn]` DuoAttention (2410.10819) — clean **retrieval-head vs streaming/sink-head** split; uses it for **KV compression (inference)**. We reuse the split as the **site oracle** but for **training-time protection**.
- `[focusft]` (2605.09932) + `[sink-forget]` (2410.05648) — the two **single-leg** sink fixes (long-ctx-at-training / forgetting). **Must beat their stack** (the P0c bar).
- `[ft-no-forget-icl]` (2602.23197) — value-matrix updates preserve ICL; adjacent.

## Read-side site definitions (the long-ctx-importance criterion)
- `[retrieval-head]` Retrieval Head Mechanistically Explains Long-Context Factuality (2404.15574, ICLR'25) — retrieval heads: universal, sparse (<5%), **intrinsic** (exist at pretrain, stable after continual pretrain), causal for NIAH. The canonical read-side site.
- `[ret-dyn]` Retrieval Heads are Dynamic (2602.11162) — retrieval heads vary per-context, irreplaceable, compensation insufficient → detector must average over inputs (informs Phase-0 stability).

## Criterion competitors (baselines that pick a protected set differently)
- `[oplora]` OPLoRA (AAAI 2026) — orthogonal to top **singular** subspace. Ours: orthogonal to **long-ctx-important** subspace.
- `[mofo]` MoFO / `[migu]` MIGU — select by **momentum / gradient magnitude**. Ours: select by **intrinsic long-ctx importance**.
- `[esft]` ESFT / `[des-moe]` DES-MoE / `[das-moe]` DAS (zBgjWTWgCh) — select experts by **task/domain affinity** + show forgetting mitigation. Ours: by **super-expert / sink-induction** (intrinsic, data-agnostic). These are THE MoE criterion competitors.
- `[expert-condenser]` ExpertCondenser/MoECondenser (2604.23036) — preserve **long-tail** experts via always-on gated condensers (opposite end from super-experts); MoE-SFT baseline.
- `[loramoe]` LoRAMoE — add experts. `[same-moe]` Same — router/expert drift.

## Feature-space (analysis lens / optional baseline, NOT the main mechanism — §5.6c)
- `[sae-fd]` SAE-FD (2605.25525) — SAE-feature distillation for continual learning (anchor active features, cosine+magnitude loss). The feature-space anti-forgetting incumbent; use as a baseline if we touch the feature route. Adding an SAE = trained task dictionary → violates the zero-task-param/data-agnostic appeal, so SAE stays a *lens*, not the lever.

## Classic anti-forgetting baselines
- EWC (Kirkpatrick 2017) · L2-SP · LwF (Li & Hoiem) · experience replay · wise-ft / model soups (Wortsman 2022). *(to log)*

## Long-context training-free baselines
- `[sink-streaming]` StreamingLLM · `[h2o]` H2O — read-side references (orthogonal to our write-side method).

## Benchmarks
- `[ruler]` RULER · HELMET · ∞Bench · LongBench-v2 · `[quality]` QuALITY · `[narrativeqa]` NarrativeQA (long-ctx).
- `[quality]`/`[musr]` + GSM8K · MATH · HumanEval+/MBPP+ (EvalPlus) · MMLU-Pro · BBH · IFEval (capability).
- TRACE (continual-learning) · LogHub-derived RCA (showcase). *(to log)*

## To add to knowledge-sources.md when Phase 1 starts
Retrieval-heads (Wu 2024) · EWC · L2-SP · LwF · wise-ft · TRACE · HELMET · ∞Bench · LongBench-v2 · LogHub.
