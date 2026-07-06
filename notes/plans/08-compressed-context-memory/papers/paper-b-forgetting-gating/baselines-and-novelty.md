# Paper B — Baselines & Novelty Defense (lit review, 2026-06-05)

> Focused review for the locked spine (do-no-harm gate for a detachable, model-agnostic,
> per-distribution memory; read-only scope). Builds on
> [`../../summary/2026-06-05/two-paper-litreview-2026-06-05.md`](../../summary/2026-06-05/two-paper-litreview-2026-06-05.md)
> + [`../../summary/2026-06-05/v1.5-related-work-2026-06-05.md`](../../summary/2026-06-05/v1.5-related-work-2026-06-05.md).

## TL;DR — the honest squeeze (read this first)
Two **very recent (both ICLR'26)** neighbors bracket our pitch:
- **Cartridges — self-study** ([2506.06266](https://arxiv.org/abs/2506.06266), Stanford/Hazy): train a *small KV cache offline per corpus* on a **frozen base**; pluggable like a prefix (no LoRA infra), **composable at inference**, **matches ICL at 38.6× less memory**. This *is* our SP1/SP2 ("detachable, lightweight, fast-ingested per-distribution memory") — and arguably done better than our wrapper.
- **TARG** ([2511.09803](https://arxiv.org/abs/2511.09803)): a **training-free, model-agnostic** gate that decides when to use augmentation from the base's **own prefix logits** (top-1/top-2 **margin** is the robust default). This *is* our §7d read gate — and §7d shows **TARG ≥ our `delta_last`**.

⇒ **Paper B's white space is NOT "a new memory module" nor "a new gating signal."** It is the
**systematic do-no-harm treatment of a pluggable memory**: (i) the **capability-boundary law**
(§8 — *where* a per-distribution module helps vs hurts), (ii) **do-no-harm by construction** for a
*learned* module (g→0 ⇒ exact frozen base), (iii) a **cross-model** study of which intrinsic signal
gates it (§2/§7d), (iv) the honest **negatives** (soft-gate & multi-depth collapse, capacity wall).
**Position: complementary to Cartridges (we decide *when* to fire a pluggable memory) and an
extension of TARG from retrieval to *learned memory modules*.**

## 1. Baselines — organized by the claim each tests
### A. Memory-module family (FORM/FUNCTION — SP1/SP2)
| baseline | id | role | status |
|---|---|---|---|
| no-context | — | floor | ✅ have |
| full-context (ICL) | — | ceiling | ✅ §7 |
| **SFT / LoRA on the data** | — | adapt-by-weights ⇒ *forgetting* baseline | ✅ §7c |
| Gist tokens | 2023 NeurIPS | prompt→K soft tokens | cite/compare |
| ICAE · AutoCompressor · CCM | 23–24 | compression memory (frozen-ish base) | cite |
| **Cartridges (self-study)** | [2506.06266](https://arxiv.org/abs/2506.06266) | **strong neighbor**: offline KV-cache memory, frozen base, composable, ICL-matching | ✅ **must-cite; ideally 1 quantitative point** |
### B. Gating / do-no-harm family (SAFETY — SP3)
| baseline | id | role | status |
|---|---|---|---|
| **TARG** (margin / entropy / small-N variance) | [2511.09803](https://arxiv.org/abs/2511.09803) | training-free prefix-logit gate, model-agnostic | ✅ done §7d (**TARG ≥ ours**) |
| Self-RAG | 2310.11511 | reflection tokens (IsRel/IsSup/IsUse) | cite/compare |
| FLARE · SeaKR · SUGAR | 23–25 | confidence-triggered retrieval | cite |
| Adaptive-RAG | 2403.14403 | tiered complexity routing | cite |
| When-do-LLMs-need-RA | [2402.11457](https://arxiv.org/abs/2402.11457) | confidence-gated retrieval (the "do-no-harm by gating" statement) | cite |
| confidence-threshold · **oracle** gate | — | our internal floor/ceiling on gating | ✅ §7b |
### C. Multi-layer injection family (the ablation — §7c/§7e)
| baseline | id | role | status |
|---|---|---|---|
| **LLaMA-Adapter (V1/V2)** | [2303.16199](https://arxiv.org/abs/2303.16199) | zero-init **gated deep** prompt injection at top layers | ✅ our `scalar` gate = this; the ablation covers it |
| Prefix-Tuning · P-Tuning v2 | [2101.00190](https://arxiv.org/abs/2101.00190) / [2110.07602](https://arxiv.org/abs/2110.07602) | deep prefixes at every layer | cite |
| input-only (our original wrapper) | — | the reference line (§8: trivia in-dist **+0.086**) | ✅ §8 |
### D. Continual-learning / forgetting (motivation only — read-only scope)
| baseline | id | finding we use |
|---|---|---|
| Empirical CF in LLMs (Luo) | 2308.08747 | CF general, worsens with scale |
| **PEFT *still* forgets** (PECFT survey; update-subspace geometry) | [2504.13822](https://arxiv.org/pdf/2504.13822), 2603.09684 | even **frozen-backbone LoRA** forgets via representation drift ⇒ **supports §7c** & motivates *never-touch-weights* |
| InfLoRA · **JANUS-LoRA** · **PLATE** · BiLoRA | 24–26 ([2605.28495](https://arxiv.org/html/2605.28495v1), [2602.03846](https://arxiv.org/html/2602.03846v1)) | orthogonal-subspace / plasticity-tunable LoRA-CL (the *write-side* competitors → Plan 09, parked here) |
| EWC · LwF | 2017 | classic regularization (cite) |

## 2. Novelty defense (reviewer objection → response)
- **R1 "Cartridges already does pluggable frozen-base memory — what's new?"** → We don't claim the
  module; we add **when to use it**. Cartridges is **always-on** and assumes its corpus is relevant;
  we characterize **when a per-distribution module helps vs hurts** (§8) and **gate** it
  (do-no-harm). Complementary — our gate could wrap a Cartridge.
- **R2 "TARG already gates augmentation, training-free & model-agnostic."** → TARG gates **retrieval**;
  we gate a **learned memory module**, **do-no-harm by construction** (g→0 ⇒ exact base), with a
  **cross-model signal study** + the **boundary**. *Honest:* TARG's base-uncertainty signal is
  competitive (§7d) → we report it as the baseline our gate must match; our value-add is the
  **systematic do-no-harm treatment**, not a SOTA scalar.
- **R3 "It's an empirical study, not a new method."** → Own it: contribution = **the
  capability-boundary law + do-no-harm-by-construction + cross-model gate transfer + honest negatives**.
  A method claim only if the 7-family head-to-head shows memory-write ⊕ TARG beats TARG alone.
- **R4 "Single model / narrow."** → 7 families for the signal study (§2); honest 3-family head-to-head
  limit (ray/test pods gone). In-domain target + no-harm floor is the *declared* scope, not a bug.

## 3. Recommended positioning
**Headline = a do-no-harm gate + capability-boundary for pluggable memory** — *complementary to
Cartridges, extending TARG to learned modules.* Lead with §8 (boundary) + §7/§7b (gate hierarchy) +
§7c (forgetting motivation). **De-emphasize** "we built a great memory module" (Cartridges wins that).

## 4. Must-run baseline checklist (de-risk)
1. ✅ no-ctx / full-ctx / SFT(LoRA) — floor / ceiling / forgetting (§7, §7c).
2. ✅ TARG / confidence / oracle gate (§7b, §7d).
3. ⬜ **Cartridges (or Gist as cheaper proxy)** — at minimum cite+position; ideally one quantitative memory-module point.
4. ⬜ **LLaMA-Adapter** ≈ our `scalar` multi-layer — covered by the running ablation (§7e).
5. ✅ input-only wrapper reference line (§8).
