# Paper B — Related Work (consolidated, paper-ready · 2026-06-07)

> Full Related-Work for *adaptive latent memory for LLM agents: a detachable, gated, do-no-harm memory*.
> Consolidates + supersedes the scattered notes ([`baselines-and-novelty.md`](baselines-and-novelty.md),
> `scope-and-compression.md`, [`../../summary/2026-06-05/v1.5-related-work-2026-06-05.md`](../../summary/2026-06-05/v1.5-related-work-2026-06-05.md),
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

## 7. Gated / co-designed compression with selective fallback (closest v1.7 neighbors, 2026-06-10)
> The "selective/gated compression with do-no-harm fallback" idea is now crowded in 2026. Verify every id on
> arXiv before citing (see [`references.md`](references.md)); several are very recent preprints.
- **Context Distillation as Latent Memory Management** (2605.28889): per-context **LoRA** memory bank +
  retrieval + **Self-Gating** (high prefix entropy ⇒ query is context-agnostic ⇒ deactivate LoRA, fall back
  to base). This is our Track A almost exactly, but the memory is a LoRA adapter and the gate fires on
  "context-agnostic query", not on "this compression lost the needed info".
- **SLT: Selective Latent Thinking** (2605.25745): backbone + **reliability-aware feature decoder** +
  **confidence gate** + latent compressor, jointly trained; compress reliable reasoning spans, fall back to
  explicit CoT when uncertain. Same mechanism as ours but the **reasoning-chain** regime, not input context.
- **PIPO: Pair-In, Pair-Out** (2605.27255): a lightweight **confidence head** trained with the
  rejection-sampling acceptance probability as a **free in-distribution label**; if confidence < tau, **safe
  fallback** to single-token decoding. Decoding regime; the "free, in-distribution gate label" trick.
- **G-MemLLM** (2602.00015): frozen backbone + trainable latent memory bank + GRU-style gate.
- **LycheeMemory** (2602.08382): Compressor + Gate (trained separately as a classifier) + Reasoner; frozen
  base + LoRA compressor; joint RL.
- **Trained Persistent Memory for Frozen Encoder-Decoder LLMs: Six Architectural Methods** (2603.16413):
  Prefix / KV-Ext / **XAttn / Hebbian / Slot / Gated** on a frozen base. Finds **prefix collapses, XAttn best
  (+6.76%), the Gated method has NEGATIVE net benefit, a capacity wall, and that training is necessary**.
  This reproduces a large part of our v1.5 architecture ablation; we must cite it and cede that ground.
- **PoC: Performance-oriented Context Compression** (2603.19733): a lightweight performance predictor picks
  the most aggressive compression ratio that meets a user "performance floor" (predicts whether compression
  is safe, at the ratio level, on off-the-shelf text compressors).
- **KV-cache risk gates / fallback:** **CompilerKV** (2602.08686, risk-adaptive threshold gate, conservative
  retention for risky prompts), **Runtime-Certified Bounded-Error Quantized Attention** (2605.20868, online
  error bounds + a four-rung fallback ladder to exact dense attention), **RACC** (retrieve evicted KV).
- **Text-level quality-gated compression (engineering):** **TAAC**, **ContextPilot**, **Compress the
  Context, Keep the Commitments** (2605.17304, conservative fallback for low-confidence semantic atoms).
> **OUR DELTA:** most of these gate "is context needed at all" (context-agnostic detection) or live at the
> text / KV / reasoning level. We gate "did the **latent** compression of THIS context lose what THIS query
> needs" on a **fully frozen** base (g→0 = exact base, which the LoRA-memory neighbors cannot give), frame it
> as a **compress-decision confusion matrix** (precision / recall / F1, cost-coverage) at the **compressor
> level**, with a **structurally-fused** gate and **two** fallbacks (base and full-context). Since the Six
> Architectural Methods paper overlaps our architecture ablation, we **cede the architecture-comparison
> framing** and lead with the decision-quality treatment + the agentic tool-use / RCA application that this
> literature does not target.

## 8. Compressor robustness BY DESIGN: reconstruction, calibration, verifier co-training (2026-06-10)
> Directly relevant to "a compressor structure that stays robust under a gate". This is where the prior art
> for a gate-friendly compressor lives.
- **Latent Context Compilation** (2602.21221): a disposable-LoRA "compiler" distills context into Buffer
  Tokens via **self-aligned optimization = a Context-Reconstruction task (force the buffer to encode the
  exact content, for fidelity) + a context-agnostic random-query regularizer (preserve the general manifold,
  i.e. do-no-harm)**. This is the closest design twin to our reconstruction-coverage + do-no-harm gate.
- **End-to-End Context Compression at Scale (LCLM)** (2606.09659): encoder-decoder soft-token compressors
  with an architectural search; explicit goal = "a general compressor that **preserves** the base's
  capabilities" rather than degrading them or needing domain-specific training.
- **Compactor** (2507.08143): **context-calibrated** compression that infers, per context, how much can be
  evicted without quality loss (different contexts tolerate very different ratios). The compressor estimates
  its own safe operating point.
- **Reconstruction-error OOD/novelty detection** (1812.02765 + CVPR'22 "Rethinking Reconstruction-AE OOD" +
  2411.10701): reconstruction error as a novelty score; **key caveat** = autoencoders can also reconstruct
  some OOD inputs, so reconstruction error alone is a weak signal and is best **fused with a latent-distance
  term (Mahalanobis)**; "maximally compress the latent while preserving reconstructive power" + layer-wise
  semantic reconstruction. This is the theory (and the warning) behind using reconstruction as a gate signal.
- **Structural Confidence / Trust in One Round** (2602.00977): single-pass confidence from the structural
  stability of a frozen-encoder hidden-state trajectory (no logits, no sampling). A cousin of our geometry
  feature family.
- **Generator-Verifier co-evolution:** **VeriThinker** (2505.17941, an auxiliary verification fine-tune so
  the model knows when to self-check), **RL Tango** (2505.15034) and **V1-PairRL** (co-train the verifier
  IN-DISTRIBUTION with the evolving generator so it stays calibrated). The deep lesson for a gate-robust
  compressor: co-train the gate with the compressor on free, in-distribution labels so it does not go stale.
> **IMPLICATION for our compressor design (the gate-robust structure):** (1) train the compressor with a
> **reconstruction objective** for fidelity + a **context-agnostic regularizer** for do-no-harm (Latent
> Context Compilation); (2) reconstruction error alone is a weak failure detector, so **fuse it with a
> latent-distance / geometry term** (Mahalanobis-AE lesson, matches our geometry features); (3) **co-train
> the gate with the compressor on free in-distribution labels** so it stays calibrated as the compressor
> changes (PIPO / Tango / V1); (4) **calibrate the safe operating point per input** (Compactor / PoC).

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

### 2026-06-10 update (honest re-read after §7/§8)
The mechanism (gated/selective compression with do-no-harm fallback on a frozen base) is **no longer novel**:
§7 lists multiple 2026 preprints that are very close (Context-Distillation-as-LMM and Latent-Context-
Compilation on the do-no-harm + reconstruction axes; SLT/PIPO on the co-trained confidence-gate + fallback
axis; Six-Architectural-Methods reproduces our architecture ablation). We should therefore **stop selling the
mechanism** and narrow the claim to the parts that survive:
1. **Decision-quality framing:** treat compress-vs-fallback as a selective-prediction classifier and report
   the **confusion matrix + precision/recall/F1 + cost/risk-coverage + oracle** at the **compressor level**,
   across in-task / cross-task-in-domain / cross-task-cross-domain. The neighbors report end-task accuracy,
   not this.
2. **Structurally-fused, single-forward gate** (reconstruction-coverage + query-coverage + geometry), no
   extra base pass and no separately-trained classifier; co-trained in-distribution with the compressor.
3. **Fully frozen base + two fallbacks** (base and full-context) unified by one gate; g→0 = exact base, a
   by-construction property the LoRA-memory neighbors do not have.
4. **Agentic tool-use / RCA application** (BFCL, API-Bank, ToolACE, OpenRCA, RCAEval), which this literature
   does not target.
**Newly mandatory citations / baselines:** Context-Distillation-as-LMM (2605.28889), Latent-Context-
Compilation (2602.21221), SLT (2605.25745), PIPO (2605.27255), PoC (2603.19733), and the Six-Architectural-
Methods paper (2603.16413, cede the architecture-ablation framing to it).
