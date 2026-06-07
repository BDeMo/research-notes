# Paper B — Related Work (consolidated, paper-ready · 2026-06-07)

> Full Related-Work for *adaptive latent memory for LLM agents: a detachable, gated, do-no-harm memory*.
> Consolidates + supersedes the scattered notes ([`baselines-and-novelty.md`](baselines-and-novelty.md),
> [`scope-and-compression.md`](scope-and-compression.md), [`../../summary/2026-06-05/v1.5-related-work-2026-06-05.md`](../../summary/2026-06-05/v1.5-related-work-2026-06-05.md),
> [`../../summary/2026-06-05/two-paper-litreview-2026-06-05.md`](../../summary/2026-06-05/two-paper-litreview-2026-06-05.md)).
> Six families; each ends with **OUR DELTA**. Honest one-liner: *Cartridges (the module) + TARG (the
> gate) bracket us; our white space is the **do-no-harm treatment** of a learned latent memory —
> selective use, cross-model, bounded — framed via selective prediction.*

## 1. Context compression into latent representations
**(a) Soft-prompt / learned memory.** **Gist** (Mu 2023), **AutoCompressor** (Chevalier 2023), **ICAE**
(Ge, ICLR'24), **CCM** (2312.03414, conditional-LoRA, frozen base), **500xCompressor**, **PCC** (ACL'25),
**Cartridges/self-study** ([2506.06266](https://arxiv.org/abs/2506.06266), ICLR'26 — offline KV-cache per corpus,
composable, ICL-matching), **ACON** (2510.00615, context compression for long-horizon agents).
**(b) KV-cache compression (training-free, inference-accel).** **StreamingLLM** (2309.17453, attention
sinks), **H2O** (2306.14048, heavy-hitter eviction), **SnapKV** (2404.14469, NeurIPS'24, cluster salient
KV; 3.6× speed / 8.2× mem), **PyramidKV** (2406.02069, pyramidal budget; 12% cache ≈ full), **FastGen**,
**FastKV** (2502.01068, prefill+decode).
> **OUR DELTA:** these compress **uniformly / always-on**. We **gate** a *learned latent* memory per
> input and **fall back to full context** when it won't suffice (Track B) — and characterize the
> **capacity wall** (RULER→0) that bounds *all* latent compression. KV-pruning is a complementary,
> training-free point on the cost axis (a baseline), not a learned, gated module.

## 2. Adaptive / selective use of augmentation (when to retrieve/compress)
**TARG** ([2511.09803](https://arxiv.org/abs/2511.09803), ICLR'26 — training-free gate from prefix-logit
**margin/entropy/variance**; the robust default), **Self-RAG** (2310.11511, reflection tokens),
**FLARE** (2305.06983, confidence-triggered), **Adaptive-RAG** (2403.14403, tiered), **CRAG**, **SeaKR**,
**When-do-LLMs-need-RA** ([2402.11457](https://arxiv.org/abs/2402.11457)).
> **OUR DELTA:** TARG/Self-RAG gate **retrieval** (token-level external context); we gate a **learned
> latent memory module** (parametric/latent form), **do-no-harm by construction** (g→0 ⇒ exact frozen
> base), with a **cross-model** signal and **two fallbacks** (no-ctx / full-ctx). *Honest: §7d shows
> TARG's base-uncertainty signal is competitive → we report it as a baseline, not claim signal supremacy.*

## 3. Selective prediction & abstention — the do-no-harm framing + metrics
**Know Your Limits** (TACL survey, abstention taxonomy + calibration), **SelectLLM** (ICLR'26, calibrate
coverage/risk), **CWSA** (2505.18622, confidence-weighted selective metrics), **Abstention-aware reasoning**
(2602.14189: "selectively withholding low-confidence predictions sharply reduces risk, stable across model
families"), calibrated-abstention practice. Metrics: **risk–coverage curve / AURC**, ECE, Abstain-ECE.
> **OUR DELTA / why this matters:** our gate **is selective prediction applied to a memory module** —
> "use the compressed memory only when reliable, else abstain to base (A) or full context (B)." We
> **adopt the risk–coverage view**: Track A = accuracy vs do-no-harm (abstain-to-base); Track B = accuracy
> vs **token cost** (a cost–coverage frontier). The survey's finding that *abstention reliability is more
> stable across model families than accuracy* echoes our **cross-model signal** result — a clean framing
> the paper should use. (Novel angle: selective prediction over *augmentation modules*, not answers.)

## 4. Memory-augmented LLMs & agentic memory (the scope)
**Backbone memory:** Memory Networks (Weston'15), NTM/DNC (Graves), **Memorizing Transformers** (Wu'22),
**Recurrent Memory Transformer** (Bulatov'22) — explicit memory layers (closest to our latent memory);
**retrieval-free injection via adapters** (Modarressi'24, adjacent to our wrapper). **External/agentic:**
**MemGPT** (2310.08560, OS-paging), **A-Mem** (2502.12110), and the 2026 RL-trained memory agents
**AgeMem** (2601.01885), **Memory-R1**, **Mem-α**, **Mem-T** (2601.23014), **RecMem** (2605.16045 —
consolidate *only when beneficial*, a do-no-harm-ish trigger), **UMA** (2602.18493, end-to-end). Surveys:
**Memory in the Age of AI Agents** (2512.13564), **Foundation-Agent Memory** (2602.06052), **Autonomous-Agent
Memory** (2603.07670); **ICLR'26 MemAgents** workshop. Taxonomy: memory forms = **token / parametric / latent**.
> **OUR DELTA:** the agentic-memory wave is mostly **token-level external memory** (RL-trained read/write/
> retrieve policies) or **memory-as-backbone-layers**. Ours is a **detachable latent/parametric module on a
> frozen base** with a **do-no-harm read-gate** — the *reliability/safety* axis (when to trust the memory)
> that the management-focused agent-memory work under-studies. RecMem's "consolidate only when beneficial"
> is the nearest spirit, but at the write/consolidation level, not read-time gating.

## 5. Continual learning / catastrophic forgetting (the problem we avoid)
**Empirical CF in LLMs** (Luo, [2308.08747](https://arxiv.org/abs/2308.08747); worsens with scale),
**EWC** (Kirkpatrick'17), **LwF**; **PECFT survey** (2504.13822) + **JANUS-LoRA** (2605.28495), **PLATE**
(2602.03846), update-subspace study (2603.09684): **even frozen-backbone LoRA forgets** (representation
drift). Which-params-to-protect: **MoFO/MIGU**, **ESFT**, mechanistic head-freezing (write-side / Plan 09).
> **OUR DELTA:** we *sidestep* forgetting — the module is **detachable, weights never touched** (do-no-harm
> by construction), and §7c shows SFT/LoRA forgets where our module doesn't. (Write-side protection = the
> parked Plan-09 follow-up.)

## 6. Multi-layer / deep prompt injection (an ablation, not a contribution)
**LLaMA-Adapter** (2303.16199, zero-init gated injection at top layers), **Prefix-Tuning** (2101.00190),
**P-Tuning v2** (2110.07602).
> **OUR DELTA:** our multi-layer ablation (zero-init scalar gate, §7c/§7e) **collapses** generation when
> active → we *justify input-level + hard-routing* over deep residual injection (a clean negative).

---

## Positioning summary (the white space)
We do **not** claim a new compressor (Cartridges), a new gating signal (TARG), or new abstention metrics.
**Our contribution = the systematic *do-no-harm treatment* of a learned latent agentic memory:** a
detachable frozen-base module + a **cross-model, data-agnostic gate** that (A) preserves the base
(selective-prediction / abstention, risk–coverage) and (B) recovers full-context accuracy at a fraction of
the tokens (compression, cost–coverage), with an **honest competence map** (helps in-domain, neutral/hurts
across families — e.g. Mistral, capacity wall) that the always-on compression and management-focused
agent-memory literatures don't provide. **Reviewer-mandatory baselines:** Cartridges/Gist (module) ·
TARG/Self-RAG (gate) · SnapKV/PyramidKV (KV-compression cost point) · LLaMA-Adapter (multi-layer) ·
SFT/LoRA (forgetting). **Adopt risk–coverage / AURC** as the gate's principled metric.
