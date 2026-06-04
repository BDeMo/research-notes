# Plan 09 — References

Closest prior, grouped by role. Full entries (with arXiv IDs + our take) live in [`../../../docs/matrix/knowledge-sources.md`](../../../docs/matrix/knowledge-sources.md) under "RCA prior-work audit" + "MoE prior-work audit". The brainstorm §5 audit is the reasoning.

## Intrinsic-site phenomena (the substrate we measure)
- `[sink-streaming]` Xiao 2023 — StreamingLLM / attention sinks (origin + training-free long-ctx baseline).
- `[sink-emerge]` When Attention Sink Emerges (2410.10781) — sink = key bias; basis for the key-bias variant.
- `[massive-act]` Sun 2024 — Massive Activations; deleting them breaks the model → forgetting-vulnerability prior.
- `[super-experts]` Unveiling Super Experts (2507.23279) — MoE experts that *induce* sinks; the dense↔MoE bridge.
- `[sink-native-moe]` Attention Sink Forges Native MoE (2602.01203) — sink weight = implicit gating.
- Retrieval heads (Wu et al. 2024) · Induction heads (Olsson 2022) — long-ctx head criteria. *(to log)*

## Nearest threat (must differentiate — DR12)
- `[mech-forget]` Mechanistic Analysis of Catastrophic Forgetting (2601.18699) — localizes forgetting to attention heads (freezing ↓64%). **Our edge**: the long-ctx↔forgetting *coupling* + the intrinsic site-selection criterion + the dual eval. This paper is the single most important comparison.
- `[ft-no-forget-icl]` (2602.23197) — value-matrix updates preserve ICL; adjacent.

## Criterion competitors (baselines that pick a protected set differently)
- `[oplora]` OPLoRA (AAAI 2026) — orthogonal to top **singular** subspace. Ours: orthogonal to **long-ctx-important** subspace.
- `[mofo]` MoFO / `[migu]` MIGU — select by **momentum / gradient magnitude**. Ours: select by **intrinsic long-ctx importance**.
- `[esft]` ESFT / `[des-moe]` DES-MoE — select experts by **task affinity**. Ours: by **super-expert / sink-induction**.
- `[loramoe]` LoRAMoE — add experts. `[same-moe]` Same — router/expert drift.

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
