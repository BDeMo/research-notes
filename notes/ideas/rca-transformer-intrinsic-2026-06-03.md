# RCA model building via transformer-intrinsic adaptation — brainstorm

> **Date**: 2026-06-03
> **Topic**: How to build Nokia's RCA model solving two challenges — (1) long context at **inference**, (2) catastrophic forgetting at **training** — with a method that is *insightful, easy to adapt, lightweight, model-agnostic, task-free*, derived from **transformer-intrinsic phenomena** (sinks, massive activations, induction heads, etc.).
> **Status**: brainstorm captured (two rounds) + **prior-work audit done (§10)** + **MoE angles audited (§11)**. ⚠️ **Audit verdict: none of R1-R12 (dense) nor M1-M9 (MoE) is novel as a standalone mechanism** — both the dense and MoE 2025-2026 literatures are saturated. Honestly-open paths: a *unifying-observation* paper (§10.3-1, best empirical anchor = the **Super-Expert ↔ attention-sink** identity, §11.2) or a *verified-new-phenomenon* (sink key-bias §10.2 / super-expert-anchored adaptation §11.3). See §10.4 + §11.4 before drafting any plan.
> **Relation to other work**: complements / contrasts Plan 08 v0 (mem-X soft-prompt wrapper). The mem-X v1 **3-regime law** is used here as a constraint prior (see [`../plans/08-model-outputs-delta-w/v1-results-2026-06-03.md`](../plans/08-model-outputs-delta-w/v1-results-2026-06-03.md)).

---

## 0. The problem (as posed by user, 2026-06-03)

Build the RCA model with two challenges:
- **Inference**: long context (long logs / stack traces / system state).
- **Training**: catastrophic forgetting (SFT on RCA drifts away pretrained code/math ability).

Overall the method must be **insightful + easy to adapt + lightweight**.

**Paper goals (do first):**
1. The model is strong on **RCA**.
2. The model is *also* strong on **code & math** (i.e. capability preserved).
- Method must use techniques the big-model / agent world is **not yet using**, but that the **open-source community accepts** — HuggingFace-basic, or AG2-specific-but-still-basic.
- Method should be **easy to adapt, lightweight, ideally model-agnostic and task-free** — derived from properties / phenomena **common to transformers themselves** (e.g. attention behavior, intrinsic structure).

**Project goals (after paper):**
- The model generalizes **in-domain** to *other* RCA datasets and *other* RCA abilities.
- Final deliverable: **a Nokia-owned RCA model release**.

**Release constraint** (derived): the cleanest IP story is *frozen public base + small Nokia-trained module*. This pushes the method toward **frozen-base + tiny-trainable** designs.

---

## 1. Unifying lens — both challenges are "where is information allowed to flow"

| Challenge | Surface symptom | Underlying issue |
|---|---|---|
| Long context | attention entropy explosion, lost-in-the-middle, huge KV | no one tells the model *which positions* should carry global state |
| Catastrophic forgetting | SFT drifts pretrained code/math ability | weight updates spread to *all* sites; the carriers of original capability are unprotected |

**One mechanism could address both**: transformers already have *privileged sites* — **attention sinks** (Xiao 2023, StreamingLLM) and **massive activations** (Sun 2024). The first few token positions + a few channels act as the model's global regulators. Confining read (context absorption) and write (adaptation) to these natural sites compresses both problems into one lever.

This is the **inverse of the mem-X v1 lesson**: mem-X *added* 32 new soft tokens and crushed the verbatim needle (Regime C). Re-using the base's *existing* sink positions = zero token overhead, positional encoding untouched, base already "knows how to read" those sites.

---

## 2. Round 1 — 5 candidate angles

| # | Angle | Phenomenon | Long ctx | Forget | Verdict |
|---|---|---|---|---|---|
| A | **Sink-Anchored Adaptation** | attention sinks + massive activations | ✓ keep only sink-KV per chunk | ✓ train only a "sink-shaper", base frozen | ★★★★★ |
| B | **Layer-band LoRA + orthogonal singular constraint** | inter-layer functional U-curve; main singular subspace = general ability (REVIVE 2026) | — | ✓ LoRA only on middle 1/3 layers, projected to orthogonal complement of top-k singular vectors | ★★★★ |
| C | **Per-head attention temperature** $\tau_{l,h}$ | per-head optimal entropy differs by task | ✓ small $\tau$ → sharper focus vs entropy explosion | ✓ per-task $\tau$ set, base frozen | ★★★★ (best as supporting lever) |
| D | **Read-side KV reweighting head** $R_\phi(K,V,q)$ | KV cache is the read-target; rewrite *how* it's read, not the KV | ✓ learns to upweight needle positions | ✓ base never changes → zero drift; re-uses base KV → verbatim preserved | ★★★★ |
| E | **Massive-activation-aware fine-tuning** | deleting massive activations breaks the model (Sun 2024) | — | ✓ gradient-mask those channels → protect global regulators | ★★★ (great story) |

Round-1 recommendation was **A + C** (sink-anchored memory + per-head temperature router).

---

## 3. Round 2 — phenomenon map + 12 angles

### 3.1 Phenomenon → problem coverage map

| Phenomenon family | Specific phenomenon | Long ctx? | Forget? | Big-model world using it? |
|---|---|---|---|---|
| Attention geometry | Attention sinks (Xiao 2023) | ✓ reuse as memory | ✓ frozen base, train sink-shaper | StreamingLLM uses, none as *writable* memory |
| | Massive activations (Sun 2024) | — | ✓ protect channels → protect capability | none |
| | Lost-in-the-middle (Liu 2024) | ✓ positional bias compensation | — | RAG sidesteps, none fixes inside attention |
| | Induction heads (Olsson 2022) | ✓ they do in-context copy | ✓ freeze them → protect ICL ability | none freeze them in fine-tuning |
| | Top-k attention sparsity | ✓ train a top-k selector | — | FastGen uses, none as learnable adapter |
| Residual stream | Refusal direction (Anthropic) | — | ✓ "RCA direction" steering, zero SFT | safety uses it, capability adaptation none |
| | Tuned lens (nostalgebraist) | ✓ adaptive early exit | ✓ lens consistency aux loss | academic, engineering none |
| | Inter-layer cosine U-curve | — | ✓ skip / only-train non-redundant layers | none |
| | SAE / polysemantic features | ✓ route task-relevant features | ✓ feature-level freeze | Anthropic interp, engineering none |
| Weight space | Main singular subspace = general ability (REVIVE 2026) | — | ✓ LoRA → orthogonal complement | editing uses, fine-tune none |
| | ~5% of params actually move in fine-tune | — | ✓ sparse-write fine-tune | LISA does layer-level, none param-level |
| | Middle 1/3 layers most plastic | — | ✓ Layer-band LoRA | LISA close, none public/commercial |
| Training dynamics | Massive-activation gradients are also massive | — | ✓ grad-mask them | none |
| | Logit margin collapses during SFT | — | ✓ KL anchor to base | academic, mainstream none |

**Only 4 phenomena hit both problems**: sinks, induction heads, tuned lens, SAE features. First two are most HF-friendly; last two more insightful but need instrumentation.

### 3.2 The 12 angles (scored 1–5; Δ = differentiation vs big-model/agent status quo)

| ID | Angle | Phenomenon | Train params | Long ctx | Forget | HF form | Δ | Score |
|---|---|---|---|---|---|---|---|---|
| R1 | **Sink-anchored writable memory** | sinks + massive act | ~500K | ✓★★★ | ✓★★★★ | forward hook | high | ★★★★★ |
| R2 | **Induction-head freezing + adapter elsewhere** | induction heads | LoRA | ✓★★ | ✓★★★★★ | grad mask | high | ★★★★★ |
| R3 | **Read-side KV reweighter** (change reading, not KV) | top-k attn + sinks | ~500K | ✓★★★★ | ✓★★★★ | custom attn | mid | ★★★★ |
| R4 | **Per-head attention temperature** ($\tau_{l,h}$, ~1K scalars) | head entropy differences | ~1K | ✓★★ | ✓★★ | attn.forward patch | mid | ★★★★ |
| R5 | **Logit-lens consistency aux loss** | tuned lens | — | — | ✓★★★★ | trainer hook | high | ★★★ |
| R6 | **Massive-activation gradient mask** | massive activations | — | — | ✓★★★★ | grad hook | high | ★★★ |
| R7 | **"Task-direction" steering** (refusal-direction analog for *adding* ability) | refusal direction | ~0 (find vector once) | ✓★ | ✓★★★★★ (zero SFT) | activation hook | very high | ★★★★ |
| R8 | **Layer-band LoRA + orthogonal singular constraint** | mid-layer plasticity + REVIVE | LoRA | — | ✓★★★★★ | PEFT 1-line | mid | ★★★★ |
| R9 | **Cosine-similarity layer skipping (Cal-Skip)** | inter-layer U-curve | — | ✓★★ (saves ctx flops) | — | forward hook | high | ★★★ |
| R10 | **Sparse-write fine-tune (parameter-level)** | 5% rule | sparse | — | ✓★★★★ | optimizer wrap | mid | ★★★ |
| R11 | **SAE-feature gated routing** | superposition | SAE | — | ✓★★★★ | activation hook | very high | ★★★ |
| R12 | **Tuned-lens-aware early exit** | tuned lens | lens | ✓★★★ (saves flops) | — | forward hook | high | ★★★ |

**Under-explored flags** (no one in the big-model world does these):
- **R7**: refusal-direction steering is SOTA in *safety*; nobody uses the same mechanism to *add capability* (e.g. find an "RCA-mode direction" from contrast pairs, add at inference → zero training → zero forgetting).
- **R2**: induction heads do in-context copy (the basis of ICL). Freezing them during SFT → ICL / few-shot preserved. Never done explicitly.
- **R6**: Sun 2024 showed deleting massive activations breaks the model; therefore protecting their gradients during training should protect base ability. One grad hook.

### 3.3 Composition recipes (by risk / ambition)

**Light** (workshop / short paper, ~2 wk): `R1 (sink memory)` + `R6 (massive-act grad mask)` — two fully independent levers (one forward, one backward), clean ablation, ≤500K params.

**Medium** (main short / EMNLP, ~4 wk): `R1` + `R2 (induction freeze)` + `R8 (layer-band+orth LoRA)` + `R4 (per-head temp)` — R1 main body, R2+R8 double-insurance against forgetting, R4 task-mode knob, ~1M params. Eval: RCA + RULER + GSM8K + HumanEval.

**Heavy** (main venue, ~8 wk): `R1` + `R3 (read-side reweighter)` + `R7 (task-direction steering, zero train)` + `R6 (grad mask)` + `R5 (lens consistency aux loss)` — full frozen base, unifying principle = "keep adaptation off the load-bearing parts of the transformer".

### 3.4 RCA-specific "free toppings" (inference-time, not in the method → keeps task-free claim)

- Structured input anchors `[STACK] [LOG] [STATE] [QUERY]` for the sink-shaper to locate the needle.
- Outlier-line amplification (entropy filter: 99% of logs are boilerplate).
- Causal-DAG-aware chunking (chunk by timestamp / causal order so sink memory accumulates in time order).
- `<verbatim>` wrapping of code snippets / stack lines to force retrieval mode.

### 3.5 Anti-patterns (NOT to do)

Full SFT or all-layer LoRA (breaks lightweight + forgetting + not insightful) · custom new architecture (breaks model-agnostic + HF) · RCA-specific special tokens *in the method* (breaks task-free) · per-task MoE expert (not lightweight; agent world already does it, no Δ) · data-synthesis/curriculum as the main contribution (not a method paper) · pure RAG (not transformer-intrinsic) · pure prompt-engineering (not a model contribution) · DPO/RLHF on Nokia data (heavy, no novelty) · distill to small model (doesn't solve forgetting).

### 3.6 mem-X v1 3-regime constraint

RCA almost certainly lands in **Regime C** (needs verbatim error code / stack line) → any *compress-to-soft-prompt* route (mem-X style) collapses. Therefore favor `R1 + R3`: both **overlay** the existing KV / sink positions rather than replacing them, so verbatim info survives. mem-X v1 is the negative teacher: "adding new tokens" fails; "reusing base's already-computed KV / sink positions" is the fix.

---

## 4. Paper story arc (one sentence)

> *"Transformers naturally develop attention sinks (Xiao 2023) and massive activations (Sun 2024) at a handful of token positions and channels. We show that all the adaptation a frozen base LLM needs — for long-context comprehension AND for new-domain fine-tuning without catastrophic forgetting — can be localized to these natural sites. We train a <1M-param sink-shaper + per-head temperature router, evaluated on RCA + RULER (long context) + GSM8K/HumanEval (preservation)."*

---

## 5. Project bridge → Nokia RCA release

```
Customer side:
  base = Qwen3-{8B,32B}  (open, frozen, downloaded once)
  + nokia_rca_sink_shaper.bin   (~500K params, Nokia-trained on internal RCA)
  + nokia_rca_temp_table.json   (~1K scalars per task domain)
  + [optional] nokia_rca_direction.vec  (R7 task-direction)

Inference:
  forward(input):
      hook: sink_shaper(chunk_encodings) -> override KV at sink positions
      attn: apply temperature_table[task] inside softmax
      [optional] add task-direction vector to residual stream
```

Advantages vs full-SFT / LoRA route: customer never re-fine-tunes the base (compliance / compute / IP aligned); Nokia releases only its own small trained module; cross-base portability (Qwen3 → Llama 4: sink mechanism exists everywhere, just retrain the sink-shaper).

---

## 6. Open decision questions (to lock before drafting a plan)

1. **Mechanism scope**: R1 only / R1+R4 / R1+R2+R8 (forgetting double-insurance) / heavy 4-lever set?
2. **Verbatim preservation a hard requirement?** (RCA almost certainly yes → R1 must pair with R3 or preserve original KV path.)
3. **Experiment scope**: RCA + RULER + code/math all at once, or validate on RULER + GSM8K first and defer RCA to a second paper? (method-first is easier to land at a top venue.)
4. **Relation to mem-X v1**: baseline ("v1 collapsed in Regime C, hence sink-anchored") or follow-up ("extends v1, reuses the same training infra")?
5. **Nokia release boundary**: sink-shaper + temperature table + (optional) task-direction vector — all releasable, or sink-shaper only?
6. **AR → dLLM extensibility** (future room, not now): how hard a constraint? If "must keep the door open", the recipe should center architecture-agnostic levers + a `site_selector` abstraction (see §8.5) and use an infilling aux loss in the AR version. If "nice-to-have", we can lean on AR-specific R2 more freely.

**Personal recommendation (anchor)**: Medium recipe, positioned as "single-mechanism unified paper" — core **sink-anchored adaptation (R1)** + two supporting levers (**induction-head freeze R2** + **per-head temperature R4**); RCA side uses §3.4 free toppings; Nokia release = sink-shaper + temperature table + optional task-direction vector, all Nokia IP, base fully public.

---

## 8. AR → dLLM extension path (future space, NOT now)

> Added 2026-06-03 per user: the method should *ideally* be extensible from AR (what we build now) to **diffusion LLMs (dLLM)**, and have dLLM-specific upside — but this is **leaving room for the future, not current scope**. The point is to not paint ourselves into an AR-only corner.

### 8.1 Design implication for the AR paper *now*

To keep the dLLM door open, the AR paper should:
- **Anchor the narrative on the *principle*, not the AR mechanism**: "confine adaptation to the transformer's load-bearing intrinsic sites, and *reuse existing representations instead of adding tokens*". This principle is architecture-agnostic; the dLLM version just swaps the *site* definition.
- **Abstract "site" as a first-class concept** in R1/R3 (a site = a set of privileged positions). AR site = attention sinks; dLLM site = mask-token / BOS positions under bidirectional attention. The sink-shaper API takes a `site_selector`, so dLLM support is a new selector, not a rewrite.
- **Prefer architecture-agnostic levers in the locked recipe** so the core result ports for free (see 8.3).
- **Use an infilling / reconstruction aux loss in the AR version** (mem-X Direction D) — because that *is* the native dLLM training objective, giving a continuous story AR → dLLM.

### 8.2 AR vs dLLM differences that matter to our two challenges

| axis | AR (now) | dLLM (future) | consequence |
|---|---|---|---|
| attention | causal mask | **bidirectional** | sinks are partly a causal artifact → dLLM privileged site = mask-token / BOS |
| generation | left→right token | **iterative denoising** (parallel blocks) | mem-X "chunk recurrence" → dLLM "denoising-step recurrence" — more natural |
| training objective | next-token | **masked reconstruction** | dLLM is *natively* infilling = mem-X Direction D = the route that may fix Regime C |
| compute knob | fixed token count | **variable denoising steps** | native task-free adaptive compute: hard RCA → more steps |
| KV cache | incremental | block-wise / different | R3 (KV reweighter) must become a block-diffusion variant |

### 8.3 Per-angle dLLM fate

- **Architecture-agnostic, port directly** (favor these to keep the door open): R4 (per-head temperature), R6 (massive-act grad mask), R7 (task-direction steering), R8 (layer-band + orthogonal LoRA), R10 (sparse-write), R11 (SAE routing).
- **Needs adaptation, but clean dLLM analog**: R1 (chunk recurrence → denoising-step recurrence; site = mask/BOS — *natural fit for dLLM*), R3 (operate on block-diffusion KV).
- **AR-specific, risky to port**: R2 (induction-head freezing — induction heads are an AR/ICL phenomenon; dLLM ICL mechanism under-studied, needs its own science first).

### 8.4 dLLM-native upside (where dLLM is *better* for our two problems — future paper material)

1. **Verbatim / Regime C natively easier**: dLLM's masked-reconstruction objective *is* infilling, which is exactly the mem-X Direction D that might preserve verbatim needles. AR needs an extra aux loss; dLLM gets it for free → continuous AR→dLLM story.
2. **Adaptive compute for hard RCA cases**: number of denoising steps is a free, task-free inference-compute dial. No training needed.
3. **Non-left-to-right dependencies**: bidirectional block attention naturally handles "a later log line explains an earlier one" — common in RCA logs.
4. **Iterative memory refinement**: the denoising trajectory is a natural place to insert/refine a memory state across steps (the dLLM analog of mem-X's recurrence and the v2 suffix-memory idea).

### 8.5 Net effect on angle selection (now)

The dLLM-extensibility requirement nudges the locked recipe toward **R1 (with a `site_selector` abstraction) + R4 + R8**, all of which either port directly or have a natural dLLM analog, and away from over-relying on **R2** (AR-specific). R7 is a strong optional both because it is zero-training *and* architecture-agnostic.

## 9. References to chase (not yet in knowledge-sources.md)

- Xiao et al. 2023 — StreamingLLM / attention sinks
- Sun et al. 2024 — Massive Activations in LLMs
- Olsson et al. 2022 — In-context Learning and Induction Heads
- Liu et al. 2024 — Lost in the Middle
- Arditi et al. 2024 — Refusal direction (single-direction ablation)
- nostalgebraist / Belrose et al. — Tuned Lens
- REVIVE 2026 — dominant singular subspace protection (already `[revive]` in knowledge-sources.md)
- LISA — layerwise importance sampling fine-tune
- **dLLM (for §8 extension)**: LLaDA (large language diffusion model) · Mercury Coder / Dream · Block Diffusion · dKV-Cache / dLLM-Cache (KV caching for diffusion LMs) · any "attention sinks in diffusion LMs" follow-up

## 10. Prior-work audit — 2022-2026 lit review (2026-06-03)

> Triggered by user: "最好是没有相似的工作或者过于拍脑袋的办法。先看四年内所有相关文献 (scholar + arxiv)." **Verdict up front: the 2025-2026 literature exploded; NONE of R1-R12 is novel as a standalone mechanism.** Each row below names the closest preempting work. This audit is the single most important content in this file.

### 10.1 Kill table

| ID | angle | closest 2024-2026 prior work | verdict |
|---|---|---|---|
| **R8** | Layer-band LoRA + orthogonal singular constraint | **OPLoRA** (AAAI 2026): SVD-decompose frozen weights, LoRA → orthogonal complement of top-k singular subspace, tested on math+code+commonsense, LLaMA-2-7B + Qwen2.5-7B. **CLoRA** (ACL 2025): null-space orthogonal reg. | ☠️ verbatim hit — DROP |
| **R10** | Sparse-write fine-tune (parameter-level) | **Sparse Memory Finetuning** (Meta, 2510.15103) + 2 follow-ups (2605.03229, 2604.05248): update only memory slots high-activated by new data; NQ F1 drops 89% full-FT / 71% LoRA / **11% SMF**. (= our own B6.) | ☠️ verbatim hit — DROP |
| **R4** | Per-head attention temperature | **SSA** (Selective Self-Attention, 2411.12892, <0.5% params, fine-tunable on existing LLMs); **Focal Attention** (2511.06818); **SSMax** (2501.19399, per-layer/head learnable scalar). | ☠️ done — DROP |
| **R7** | Task-direction steering (add capability) | trained steering vectors to unlock reasoning (OpenReview URrDgCHA1i); instruction-following steering (2410.12877); **cache steering** (2507.08799); CAST. | ☠️ done — DROP |
| **R3** | Read-side KV reweighter (frozen base) | **ReasonCache** (2602.02366, prefix-KV frozen); **KV Packet** (2604.13226, trainable soft-token KV adapter, frozen, *explicitly handles sink boundary artifacts*); **RLKV** (2510.08525); DynamicKV. | ☠️ done — DROP |
| **R5** | Logit-lens consistency aux loss | **Logit Lens Loss** (2602.01530); **DistillLens** (2602.13567); **SelfAug** (EMNLP 2025 findings.763, input-logit self-alignment vs forgetting); **TMKL** (2605.29498). | ☠️ done — DROP |
| **R6** | Massive-activation gradient mask | **MoFO** (2407.20999, momentum-filtered selective update); **MIGU** (2406.17245, magnitude-based gradient update, task-label-free). Magnitude/momentum-gated selective update vs forgetting is a sub-field. | ☠️ strongly preempted — DROP |
| **R2** | Induction-head freezing | **Mechanistic Analysis of Catastrophic Forgetting** (2601.18699): freezing attention → forgetting ↓64%; **Fine-Tuning Without Forgetting ICL** (2602.23197): restrict updates to value matrix preserves ICL; **ABFT** (Attention Behavior Fine-Tuning) manipulates induction heads directly. | ☠️ strongly preempted — DROP |
| **R11** | SAE-feature gating / freeze | **SAE-FT** (2605.15961, constrain updates to frozen-SAE feature span); **SAE-FD** (2605.25525, SAE-feature distillation for continual learning); **SAE-Tuning** (OpenReview vUrZaERt8b). | ☠️ done — DROP |
| **R9 / R12** | Cosine layer-skip / tuned-lens early exit | CALM, LayerSkip, and the broader adaptive-depth / early-exit literature. Also only addresses FLOPs, not our two core problems. | ☠️ mature area — DROP |
| **R1** | Sink-anchored writable memory | **Trained Persistent Memory for Frozen Decoder-Only LLMs** (2603.22329: prefix / KV-extension / Hebbian / slot-write memory adapters on frozen base); **KVM** (2605.09877, block-recurrent compressed memory); **LCIRC** (2502.06139) / **MemCom** (2510.16092) recurrent compression into frozen LLM; **ReasonCache** prefix-KV. Plus **SinkTrack** (anchors to BOS without lossy compression). | ⚠️ narrow gap survives (10.2) |

### 10.2 What narrowly survives — and why it's thin

- **R1's only unclaimed core**: nobody *specifically* says "the model already allocates sink positions as a no-op register; hijack *those exact positions* as the writable memory". But it sits one inch from prefix-tuning at the sink position (ReasonCache) and from slot-write persistent memory (2603.22329). A reviewer will ask "how is this different from prefix tuning placed at the sink?" → **cannot carry a paper as a standalone mechanism**.
- **Sink key-bias as the trainable knob** (NEW, less-searched): "When Attention Sink Emerges" (2410.10781) frames the sink as a **key bias storing extra attention scores**. "Fine-tune *only* the sink's key-bias behavior" was **not** directly found in this review. Narrower and cleaner than R1, but unverified — needs its own search before trusting.

### 10.3 The two honestly-open paths (both thinner than hoped)

1. **Unifying *observation* paper** (not a new mechanism). No single paper was found that uses **one intrinsic-site mechanism to solve BOTH** long-context inference AND training-time forgetting, evaluated on **RCA + code + math**. Defensible claim: *"the intrinsic sites that absorb long-range context (sinks / massive activations) are the same sites most disrupted by fine-tuning; protecting/reusing them addresses both problems at once."* Contribution grade = empirical/scientific observation, not new method. Hardest to preempt because it's about the *relationship* between the two phenomena. The 2601.18699 mechanistic-forgetting paper is the closest threat (it already localizes forgetting to attention heads) — our differentiation must be the *long-context-AND-forgetting joint* framing + the RCA testbed.
2. **Find a genuinely new phenomenon.** Everything searched (sinks / massive-act / induction / lens / SAE / temperature / sparse-write / orthogonal-LoRA) is saturated. Candidate not-yet-crowded leads to verify next: (a) sink **key-bias**-only tuning (§10.2); (b) the *joint dynamics* of context-absorption vs capability-disruption at the same site.

### 10.4 Revised recommendation (supersedes §6 anchor + §8.5)

The original "sink-anchored adaptation as a novel method" framing is **not viable** as a method-novelty paper — the space is too crowded. Honest options, in order:

- **(A) Reframe to the unifying-observation paper** (§10.3-1): lead with the science ("same sites, both problems"), use the known levers (OPLoRA-style orthogonal LoRA for the W-side, sink/persistent-memory for the X-side) as *components*, and make **RCA + code + math co-evaluation** the empirical centerpiece. Workshop-safe; main-venue needs a sharp observation.
- **(B) Chase the sink key-bias gap** (§10.2) with a dedicated lit search before committing.
- **(C) Lean on RCA as the contribution** (application paper) — but this sacrifices the "task-free / model-agnostic method novelty" goal.

Decision deferred to user. The brutally honest take: as a *method* paper with a *novel mechanism*, none of R1-R12 qualifies after this review; the realistic path is the unifying-observation framing (A) or a verified-new-phenomenon (B).

### 10.5 Sources logged

All closest-prior IDs added to `docs/matrix/knowledge-sources.md` under "RCA / transformer-intrinsic prior-work audit (2026-06-03)".

---

## 11. MoE-specific angles (added 2026-06-03)

> User: "再看看针对 MoE 模型有什么可以利用的点，来满足我们的需求." Motivation is real — most plausible RCA bases are **MoE**: Qwen3-30B-A3B / 235B-A22B, DeepSeek-R1 (1 shared + 8 routed of 256), Llama-4 Scout/Maverick (1 shared + 1 routed). Same brainstorm-then-audit discipline as §10.

### 11.0 Honest scoping — MoE is a *forgetting* lever, NOT a long-context lever

MoE sparsity lives in the **FFN**; the long-context bottleneck lives in **attention / KV** (shared across experts). So MoE *per se* does **not** address our inference-time long-context challenge — that stays an attention-side problem (sinks, §10). What MoE buys is a **natural discrete decomposition of capability across experts**, which is a clean substrate for the *catastrophic-forgetting* challenge: "confine the write to a few experts / the router / the shared expert" is far more natural than dense layer-band LoRA. **Net: MoE strengthens the forgetting half, leaves the long-context half to the attention-side levers.**

### 11.1 The 9 MoE angles + inline prior-work verdict

| ID | angle | mechanism | challenge | closest prior (2023-2026) | verdict |
|---|---|---|---|---|---|
| **M1** | Shared-expert-only adaptation (freeze routed, tune only always-on shared expert) | the shared expert is the FFN analog of the attention sink — the one always-on privileged compute path | forget | **ESFT** (`[esft]`, EMNLP 2024) §6.3 studies train-shared vs train-non-shared variants; finds non-shared is key. DeepSeekMoE shared-expert design. | ☠️ examined & found inferior — DROP |
| **M2** | Router-only fine-tuning (freeze ALL experts, train only gating) | re-compose existing expert skills for RCA; zero expert drift | forget | textbook recipe ("fine-tune gating network only"); LoRA-on-router; DES-MoE adaptive router (`[des-moe]`). | ☠️ standard — DROP |
| **M3** | Expert expansion / add LoRA experts (freeze pretrained, add new RCA experts + widen router) | new knowledge isolated in new experts → zero forgetting by construction | forget | **LoRAMoE** (`[loramoe]`, ACL 2024) verbatim; **Lifelong-MoE** (`[lifelong-moe]`, ICML 2023) expand+freeze+reg. | ☠️ verbatim hit — DROP |
| **M4** | Importance-guided expert freezing (probe routing freq on code/math, freeze high-affinity, train rest) | MoE-native analog of induction-head freeze | forget | **ESFT** (`[esft]`) verbatim — tune only top-affinity experts, freeze rest; **DES-MoE** (`[des-moe]`) domain-correlation expert isolation. | ☠️ verbatim hit — DROP |
| **M5** | Cold/dead-expert recycling (repurpose under-utilized experts for RCA) | low-activation experts contribute little → minimal forgetting | forget | expert-pruning + ESFT-complement; routing-collapse literature. | ☠️ derivative — DROP |
| **M6** | Router temperature / z-loss control for long ctx (stabilize routing as sequence grows) | router logits blow up / late tokens dropped at long ctx | long-ctx (weak) | **router z-loss** (ST-MoE), similarity-preserving balancing (`omi2025`), MaxScore capacity (`dong2025`). | ☠️ standard — DROP |
| **M7** | Input-adaptive conditional compute for logs (route boilerplate→cheap expert, needle→full) | 99% of logs are boilerplate; MoE already does conditional compute | efficiency only | expert-choice routing; capacity-aware inference. Doesn't touch attention/KV → not a capability fix. | ☠️ efficiency-only — DROP |
| **M8** | Routing-drift regularizer / diagnostic (penalize change in routing dist on general data during RCA SFT) | forgetting ⇔ general tokens re-routed away from their experts | forget | **Same** (`[same-moe]`, 2026): explicitly models *router drift* + *expert drift*, spectral-aware routing. | ☠️ done — DROP |
| **M9** | Load-balance-free RCA specialization (relax aux loss so RCA tokens concentrate on a few new experts) | let RCA form its own specialist experts without forcing spread onto general experts | forget | implied by ESFT routing-concentration finding + LoRAMoE localized balancing. | ⚠️ thin, derivative |

**Conclusion (forgetting side): the MoE continual-learning / PEFT space is as crowded as the dense one.** ESFT + DES-MoE + LoRAMoE + Same + Lifelong-MoE collectively preempt M1-M6, M8. None is a novel mechanism.

### 11.2 The one genuinely interesting find — **Super Experts ↔ attention sinks** (a true MoE-intrinsic bridge)

Two 2025-2026 papers connect MoE experts to the *exact* attention-sink phenomenon the §10 dense brainstorm was built on:

- **Super Experts** (`[super-experts]`, arXiv:2507.23279): a *tiny* number of experts in an MoE **induce the attention sinks**. Pruning these "super experts" drives the **attention-sink decay rate >90%** and collapses the model to near-zero on some tasks (measured on Qwen3-30B-A3B). They are the MoE analog of **massive activations** (`[massive-act]`) — the load-bearing site.
- **Attention Sink Forges Native MoE** (`[sink-native-moe]`, arXiv:2602.01203): the sink's attention weight *is* an implicit gating factor → attention heads already form a "native MoE"; designating top-m heads as frozen "shared heads" + routing the rest fixes head collapse.

**Why this matters for us**: it unifies the dense brainstorm and the MoE brainstorm into one site. In an MoE base, the privileged load-bearing site = **super experts** (which induce the sinks). They are simultaneously (a) the long-context stabilizers (induce sinks essential for long ctx) and (b) the most dangerous parameters to update during SFT (pruning/perturbing them collapses capability). That is **exactly the §10.3-1 unifying observation, now with an empirical MoE anchor**: *the sites that carry long-context ability are the same sites whose disruption causes forgetting.*

### 11.3 The surviving angle — **Super-Expert-anchored unified adaptation** (`M★`)

> All of `[super-experts]` / `[esft]` / `[des-moe]` / `[same-moe]` select protected experts by **task/domain affinity** or study super experts for **compression/pruning**. **Nobody (as of this search) selects the protected set by the super-expert / attention-sink-induction criterion *for the purpose of fine-tuning without forgetting*.** That is the gap.

Recipe sketch:
1. **Identify** super experts via the `[super-experts]` criterion (activation-outlier / sink-induction score) — task-free, one forward pass on generic data.
2. **Protect** them: freeze (or grad-mask, MoE analog of R6) the super experts during RCA SFT → preserves both long-context stability *and* code/math capability.
3. **Adapt** elsewhere: train RCA into the non-super experts (or the router), where disruption is cheap.
4. **Claim**: one task-free, model-agnostic, HF-basic criterion (super-expert detection) simultaneously protects long-context behavior and prevents forgetting — validated on RCA + RULER + GSM8K/HumanEval, on Qwen3-MoE.

**Novelty verdict**: ⚠️ **narrow gap, must verify** (same discipline as R1 / sink-key-bias). Threats: (a) ESFT/DES-MoE already do criterion-based expert freezing — differentiation is the *selection criterion* (sink-induction, not task-affinity) + the *joint long-ctx↔forgetting framing*; (b) `[super-experts]` could spawn a fine-tuning follow-up any day. **Needs a dedicated "super expert + fine-tuning + forgetting" search before committing.**

### 11.4 Net effect on the program

- MoE does **not** rescue the "novel mechanism" goal — the forgetting-MoE space is saturated (M1-M9 mostly dead).
- MoE **does** give the **strongest empirical anchor for the §10.3-1 unifying-observation paper**: the **Super-Expert ↔ sink** identity is the cleanest evidence that "long-context load-bearers = forgetting-vulnerable sites." If we pursue path (A), this is the centerpiece figure, and Qwen3-MoE is HF-native (easy to adapt, model-agnostic across MoE family).
- This also **subsumes** the dense §10 story: dense base → protect sinks/massive-activations; MoE base → protect super experts (which *are* the sink inducers). One principle, two architectural instantiations → a stronger, more general paper.
- dLLM note: super experts / sink-induction in **diffusion MoE** is entirely unstudied → another future-room lever consistent with §8.

### 11.5 Sources logged

New IDs in `knowledge-sources.md` (§"MoE prior-work audit"): `[super-experts]`, `[sink-native-moe]`, `[esft]`, `[des-moe]`, `[loramoe]`, `[same-moe]`, `[lifelong-moe]`, `[gated-attn-sinkfree]`.
