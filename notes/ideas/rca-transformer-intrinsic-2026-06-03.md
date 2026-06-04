# RCA model building via transformer-intrinsic adaptation — brainstorm

> **Date**: 2026-06-03 (reorganized conclusion-first same day)
> **Topic**: build Nokia's RCA model solving two challenges — (1) long context at **inference**, (2) catastrophic forgetting at **training** — with a method that is *data-agnostic, transformers-intrinsic, lightweight, ideally non-task-specific*.
> **Status**: 2 brainstorm rounds (dense R1–R12, MoE M1–M9) + full 2022-2026 prior-work audit + prioritization under refined constraints. **Conclusion is in §1 — read that first.** §4 = the raw brainstorm, §5 = the audit, §6 = the surviving direction, §7 = pre-audit material kept for the record.
> **Relation to other work**: complements / contrasts Plan 08 v0 (mem-X soft-prompt wrapper). The mem-X v1 **3-regime law** is a constraint prior (see [`../plans/08-model-outputs-delta-w/v1-results-2026-06-03.md`](../plans/08-model-outputs-delta-w/v1-results-2026-06-03.md)).

---

## 1. TL;DR — current conclusion (read this first)

**Verdict after the audit (§5)**: none of the 12 dense angles (R1–R12) nor the 9 MoE angles (M1–M9) is novel **as a standalone mechanism** — the 2025-2026 dense *and* MoE literatures are both saturated (OPLoRA, Sparse-Memory-FT, ESFT, DES-MoE, Same, MoFO/MIGU, SAE-FT, …).

**Under the refined constraints** — **(D) data-agnostic · (I) transformers-intrinsic · (T) light training OK but ideally NOT task-specific** — the whole space collapses onto a single defensible program:

> **P0 thesis.** The transformer-intrinsic sites that carry long-context behavior — **attention sinks / massive activations** (dense), the **super experts that induce those sinks** (MoE) — are the *same* sites whose perturbation during fine-tuning causes catastrophic forgetting. So one **data-agnostic, training-free** rule — *detect these sites on generic text, then protect them (freeze / gradient-mask) during any SFT* — improves long-context retention **and** prevents code/math forgetting, **adding zero task-specific trainable parameters**.

**Priorities** (full table §6.3): **P0c** de-risk measurement experiment (needs **no RCA data**) → **P0a** dense + **P0b** MoE (super-expert) instantiation of the protection rule. Everything that *trains a task-specific module* (R1/R2/R3/R8/R10) is demoted to a **baseline**; R9/R12 dropped; R6/R7/sink-key-bias kept as components.

**Next action** (§6.4): run the P0c experiment on Qwen3-8B + Qwen3-30B-A3B, then a targeted novelty search on "super-expert + fine-tuning + forgetting", then draft `notes/plans/09-intrinsic-site-protection/`.

**Honest caveat** (§6.5): under (T), the *novel* long-context leg is thin (data-agnostic long-ctx ≈ training-free sink-KV = StreamingLLM). The **forgetting leg + the unifying observation carry the paper**; long-context rides along as the second half of the "same sites" story.

---

## 2. The problem (as posed by user)

Build the RCA model with two challenges:
- **Inference**: long context (long logs / stack traces / system state).
- **Training**: catastrophic forgetting (SFT on RCA drifts away pretrained code/math ability).

**Paper goals (do first)**: model strong on **RCA** *and* still strong on **code & math** (capability preserved). Method should use techniques the big-model/agent world is **not yet using** but the OSS community accepts (HuggingFace-basic, or AG2-basic), be **easy to adapt + lightweight**, ideally **model-agnostic + task-free**, derived from **transformer-intrinsic phenomena**.

**Project goals (after paper)**: in-domain generalization to other RCA datasets/abilities; final deliverable = **a Nokia-owned RCA model release**.

**Derived release constraint**: cleanest IP story = *frozen public base + small Nokia-owned module* → pushes toward **frozen-base + tiny-trainable (or zero-trainable) designs**.

**Refined method constraints (locked 2026-06-03)**: **(D) data-agnostic · (I) transformers-intrinsic · (T) light training OK, ideally non-task-specific.** These are the filter used in §6.

---

## 3. Unifying lens — both challenges are "where is information allowed to flow"

| Challenge | Surface symptom | Underlying issue |
|---|---|---|
| Long context | attention entropy explosion, lost-in-the-middle, huge KV | no one tells the model *which positions* carry global state |
| Catastrophic forgetting | SFT drifts pretrained code/math ability | weight updates spread to *all* sites; the carriers of original capability are unprotected |

**One mechanism could address both**: transformers already have *privileged sites* — **attention sinks** (Xiao 2023) and **massive activations** (Sun 2024); a few token positions + channels act as global regulators. Confining read (context absorption) and write (adaptation) to these natural sites compresses both problems into one lever. This is the **inverse of the mem-X v1 lesson**: mem-X *added* 32 new soft tokens and crushed the verbatim needle (Regime C); re-using the base's *existing* sink positions = zero token overhead, base already "knows how to read" them.

---

## 4. The brainstorm — candidate angles

> Raw idea generation. Each angle now carries its audit verdict (§5) inline so the tables are self-consistent. The verdicts, not the original ★ scores, are authoritative.

### 4.1 Round 1 — 5 seed angles

| # | Angle | Phenomenon | Became |
|---|---|---|---|
| A | Sink-Anchored Adaptation | attention sinks + massive activations | → R1 / R3 |
| B | Layer-band LoRA + orthogonal singular constraint | inter-layer U-curve; main singular subspace = general ability | → R8 |
| C | Per-head attention temperature $\tau_{l,h}$ | per-head optimal entropy differs | → R4 |
| D | Read-side KV reweighting head | KV cache is the read-target | → R3 |
| E | Massive-activation-aware fine-tuning | deleting massive activations breaks the model | → R6 |

### 4.2 Phenomenon → problem coverage map

| Phenomenon | Long ctx? | Forget? | Big-model world using it? |
|---|---|---|---|
| Attention sinks (Xiao 2023) | ✓ reuse as memory | ✓ frozen base, train sink-shaper | StreamingLLM uses; none as *writable* memory |
| Massive activations (Sun 2024) | — | ✓ protect channels → protect capability | none |
| Lost-in-the-middle (Liu 2024) | ✓ positional-bias compensation | — | RAG sidesteps; none fixes inside attention |
| Induction heads (Olsson 2022) | ✓ in-context copy | ✓ freeze them → protect ICL | none freeze them in FT |
| Top-k attention sparsity | ✓ learnable selector | — | FastGen uses; none as adapter |
| Refusal direction (Anthropic) | — | ✓ "RCA direction" steering, zero SFT | safety uses it; capability none |
| Tuned lens | ✓ adaptive early exit | ✓ lens-consistency aux loss | academic only |
| Inter-layer cosine U-curve | — | ✓ only-train non-redundant layers | none |
| SAE / polysemantic features | ✓ route task features | ✓ feature-level freeze | interp only |
| Main singular subspace = general ability | — | ✓ LoRA → orthogonal complement | editing uses; FT none |
| ~5% of params move in FT | sparse | — | LISA (layer-level) |
| Middle 1/3 layers most plastic | — | ✓ layer-band LoRA | LISA close |
| Massive-activation grads are also massive | — | ✓ grad-mask them | none |

**Only 4 phenomena hit both problems**: sinks, induction heads, tuned lens, SAE features.

### 4.3 The 12 dense angles (R1–R12) — score + closest prior + verdict

| ID | Angle (phenomenon) | LC | Forget | Train type | Closest 2024-2026 prior | Verdict |
|---|---|---|---|---|---|---|
| R1 | Sink-anchored writable memory (sinks+massive-act) | ★★★ | ★★★★ | ~500K task-ish | `[persist-mem-dec]` 2603.22329 · `[kvm]` · `[reasoncache]` prefix-KV | ⚠️ thin gap (§5.2) |
| R2 | Induction-head freezing + adapter elsewhere | ★★ | ★★★★★ | LoRA, task-specific | `[mech-forget]` (attn-freeze ↓64% forget) · `[ft-no-forget-icl]` · ABFT | ☠️ strongly preempted |
| R3 | Read-side KV reweighter (change reading, not KV) | ★★★★ | ★★★★ | ~500K task-ish | `[reasoncache]` · `[kv-packet]` (handles sink boundary!) · `[rlkv]` | ☠️ preempted |
| R4 | Per-head attention temperature ($\tau_{l,h}$, ~1K) | ★★ | ★★ | ~1K | `[ssa]` · `[ssmax]` · `[focal-attn]` | ☠️ done |
| R5 | Logit-lens consistency aux loss | — | ★★★★ | aux loss | `[logit-lens-loss]` · `[selfaug]` · `[tmkl]` · `[distill-lens]` | ☠️ done |
| R6 | Massive-activation gradient mask | — | ★★★★ | adds none | `[mofo]` · `[migu]` | ☠️ preempted (→ component) |
| R7 | "Task-direction" steering (add capability) | ★ | ★★★★★ | ~0 (find vector) | `[steer-reason]` · `[instruct-steer]` · `[cache-steer]` | ☠️ done (→ baseline) |
| R8 | Layer-band LoRA + orthogonal singular constraint | — | ★★★★★ | LoRA, task-specific | `[oplora]` (AAAI 2026) · `[clora]` | ☠️ verbatim hit |
| R9 | Cosine-similarity layer skipping | ★★ (flops) | — | none | CALM / LayerSkip | ☠️ mature → drop |
| R10 | Sparse-write fine-tune (param-level) | sparse | ★★★★ | task-specific | `[smf]` (Meta 2025) | ☠️ verbatim hit |
| R11 | SAE-feature gated routing | ✓ | ★★★★ | SAE | `[sae-ft]` · `[sae-fd]` · `[sae-tuning]` | ☠️ done |
| R12 | Tuned-lens-aware early exit | ★★★ (flops) | — | lens | CALM / early-exit lit | ☠️ mature → drop |

### 4.4 The 9 MoE angles (M1–M9)

**Honest scoping**: MoE sparsity lives in the **FFN**; the long-context bottleneck lives in **attention/KV** (shared across experts). So **MoE is a *forgetting* lever, NOT a long-context lever.** Most plausible RCA bases *are* MoE (Qwen3-30B-A3B / 235B-A22B, DeepSeek-R1, Llama-4), so it's worth the pass.

| ID | Angle | Challenge | Closest prior | Verdict |
|---|---|---|---|---|
| M1 | Shared-expert-only adaptation (tune always-on shared expert) | forget | `[esft]` §6.3 (non-shared is key) | ☠️ examined & inferior |
| M2 | Router-only fine-tuning (freeze all experts) | forget | textbook; `[des-moe]` adaptive router | ☠️ standard |
| M3 | Expert expansion / add LoRA experts | forget | `[loramoe]` (ACL 2024) · `[lifelong-moe]` | ☠️ verbatim hit |
| M4 | Importance-guided expert freezing (task-affinity) | forget | `[esft]` verbatim · `[des-moe]` | ☠️ verbatim hit |
| M5 | Cold/dead-expert recycling | forget | expert-pruning + ESFT-complement | ☠️ derivative |
| M6 | Router temperature / z-loss for long ctx | long-ctx (weak) | router z-loss; `omi2025`; `dong2025` | ☠️ standard |
| M7 | Input-adaptive conditional compute for logs | efficiency only | expert-choice routing | ☠️ efficiency-only |
| M8 | Routing-drift regularizer / diagnostic | forget | `[same-moe]` (router+expert drift) | ☠️ done |
| M9 | Load-balance-free RCA specialization | forget | implied by ESFT + LoRAMoE | ⚠️ thin, derivative |

**Conclusion**: ESFT + DES-MoE + LoRAMoE + Same + Lifelong-MoE collectively preempt M1–M8. None is a novel mechanism.

---

## 5. Prior-work audit — why nothing is novel (2022-2026)

> Triggered by user: "最好是没有相似的工作或者过于拍脑袋的办法 — 先看四年内文献." Full closest-prior IDs in `docs/matrix/knowledge-sources.md` (§"RCA prior-work audit" + §"MoE prior-work audit"). The verdict columns in §4.3/§4.4 are the per-angle results.

### 5.1 Verdict

Both the dense continual-learning/PEFT space and the MoE continual-learning space exploded in 2025-2026. Every individual mechanism we brainstormed is preempted, most verbatim. **As a *method* paper with a *novel mechanism*, nothing in R1–R12 / M1–M9 qualifies.**

### 5.2 The narrow dense survivors (thin)

- **R1's only unclaimed core**: "the model already allocates sink positions as a no-op register; hijack *those exact positions* as writable memory" — but one inch from prefix-tuning at the sink (`[reasoncache]`) and slot-write persistent memory (`[persist-mem-dec]`). Can't carry a paper alone.
- **Sink key-bias-only tuning** (less-searched): `[sink-emerge]` frames the sink as a *key bias storing extra attention scores*. "Fine-tune *only* the sink's key-bias" was **not** directly found — narrower/cleaner than R1, but **unverified**; needs its own search.

### 5.3 The MoE find — **Super Experts ↔ attention sinks** (the bridge)

Two papers connect MoE experts to the exact sink phenomenon the dense brainstorm is built on:
- **`[super-experts]`** (2507.23279): a *tiny* set of experts **induce the attention sinks**; pruning them → sink-decay-rate >90% → model collapses to near-zero (Qwen3-30B-A3B). MoE analog of `[massive-act]` — the load-bearing site. **Studies compression/pruning, NOT fine-tuning protection.**
- **`[sink-native-moe]`** (2602.01203): the sink's attention weight *is* an implicit gating factor → heads form a "native MoE"; freezing top-m "shared heads" fixes head collapse.

This **unifies dense and MoE into one site**: in an MoE base, the privileged load-bearing site = super experts (which induce the sinks); they are simultaneously the long-context stabilizers **and** the most dangerous params to update during SFT. That is the §1 P0 thesis with an empirical MoE anchor.

### 5.4 The two honestly-open paths

1. **Unifying-observation paper** (not a new mechanism). No paper uses **one intrinsic site to solve BOTH** long-context inference AND training-time forgetting, on **RCA + code + math**. Hardest to preempt because it's about the *relationship*. Main threat `[mech-forget]` (localizes forgetting to attention heads) → differentiate via the long-ctx coupling + sink-induction criterion + RCA testbed. **→ this is P0.**
2. **Verified-new-phenomenon**: sink **key-bias-only** tuning (§5.2) or the *joint dynamics* of context-absorption vs capability-disruption at the same site. Needs a dedicated search first.

---

## 6. The surviving direction — P0 in depth

### 6.1 Why P0 satisfies all three constraints

- **(D) data-agnostic**: sites detected by a forward pass on *any* generic text (sink position / activation magnitude / super-expert score). No RCA labels in the *method*.
- **(I) intrinsic**: sinks (`[sink-streaming]`, `[sink-emerge]`), massive activations (`[massive-act]`), super experts (`[super-experts]`) are universal transformer phenomena.
- **(T) no task-specific training added**: the method is a *constraint on* the unavoidable RCA SFT — a freeze / grad mask — not a new task-trained module. (Optional component trainable with a *task-agnostic* self-supervised infilling objective.)

**Novelty position (honest)**: the *mechanism* (freeze/grad-mask by importance) is known (MoFO/MIGU/ESFT/DES-MoE). The **unclaimed contribution is two-fold**: (1) the **site-selection criterion** = *long-context-load-bearing / sink-induction* (NOT task-affinity, NOT raw gradient magnitude); (2) the **joint claim** that the *same* sites govern long-context AND forgetting, proven by a dual eval.

### 6.2 Super-expert-anchored protection (the sharp MoE instantiation, `M★`)

> ESFT/DES-MoE/Same select protected experts by **task/domain affinity**; `[super-experts]` studies super experts for **compression**. **Nobody selects the protected set by the super-expert / sink-induction criterion *for fine-tuning without forgetting*.** That is the gap (⚠️ verify before committing).

Recipe sketch: **(1) identify** super experts via the `[super-experts]` criterion (one generic forward pass, task-free); **(2) protect** them (freeze / grad-mask) during RCA SFT → preserves long-context stability *and* code/math; **(3) adapt** the non-super experts (or router), where disruption is cheap. HF-native on Qwen3-MoE.

### 6.3 Prioritization (the filter applied to everything)

| candidate | D | I | train type | novelty open? | priority |
|---|---|---|---|---|---|
| **P0a — Unifying-observation + intrinsic protection** (dense: sinks/massive-act) | ✓ | ✓ | **adds none** | ⚠️ narrow (joint framing + criterion) | **P0** |
| **P0b — Super-expert-anchored protection** (MoE) | ✓ | ✓ | **adds none** | ⚠️ narrow (verify §6.2) | **P0** |
| **P0c — De-risk measurement study** (no RCA data) | ✓ | ✓ | none | enabling | **P0 — do first** |
| sink **key-bias-only** tuning | ✓ | ✓ | tiny, task-ish | ⚠️ unverified | P2 — verify → maybe component |
| R6 massive-act grad-mask | ✓ | ✓ | adds none | ☠️ MoFO/MIGU | P2 — component of P0a |
| R7 task-direction steering | ~ (contrast pairs) | ✓ | ~zero | ☠️ preempted | P2 — baseline / inference topping |
| R1 sink-anchored memory (trained) | ~ | ✓ | task-ish | ⚠️ thin | P3 — component / ablation |
| R8 / R10 / R2 / R3 | ✗ task-specific module | ✓ | task-specific | ☠️ preempted | **baselines only** |
| R9 / R12 | ✓ | ✓ | none | ☠️ mature | drop |

### 6.4 Next actions (cheap-first)

1. **P0c de-risk experiment — needs NO RCA data, ~1–2 days.** On Qwen3-8B (dense) + Qwen3-30B-A3B (MoE): (a) detect sites on generic text (sink positions, top massive-act channels, super experts via sink-decay); (b) run a *small proxy-domain* SFT (non-code/math corpus), measure which sites shift, correlate site-shift with forgetting (ΔGSM8K/ΔHumanEval) and long-ctx (ΔRULER). **Goal: confirm/kill the P0 thesis before any RCA work.**
2. **Targeted novelty search** "super-expert / sink-induction + fine-tuning + forgetting" — gates P0b (10 min, blocking).
3. **If P0c positive + gap holds** → draft `notes/plans/09-intrinsic-site-protection/` (validation = dual eval; channels = HF Qwen3 dense+MoE; budget = small, mostly inference + light SFT). RCA data enters only as the *application* eval.
4. **Defer** all trained-adapter angles (R1/R2/R3/R8/R10) to "baselines" in that plan.

### 6.5 Honest caveat on the long-context leg

Under (T), the *novel* long-context contribution is thin: the strongest data-agnostic long-context lever is training-free sink-based KV handling ≈ `[sink-streaming]` (known). So the long-context win in P0 is **"protection preserves the base's existing long-context machinery"**, not a new long-context method. The **forgetting leg + the unifying observation carry the paper**. Don't over-claim a long-context method in the plan.

---

## 7. Pre-audit material (superseded — kept for the record)

> The following was written **before** the §5 audit, when "sink-anchored adaptation as a novel method" looked viable. It is retained for provenance; per §6 the recipes/bridge are now mostly **demoted to baselines/components**, and the original paper-story over-claims method novelty.

### 7.1 Composition recipes (original)
- **Light** (~2 wk): R1 + R6.
- **Medium** (~4 wk): R1 + R2 + R8 + R4 (~1M params; eval RCA + RULER + GSM8K + HumanEval).
- **Heavy** (~8 wk): R1 + R3 + R7 + R6 + R5 (full frozen base).

### 7.2 RCA-specific "free toppings" (inference-time, keep task-free claim — still usable)
Structured input anchors `[STACK] [LOG] [STATE] [QUERY]` · outlier-line amplification (entropy filter) · causal-DAG-aware chunking (timestamp/causal order) · `<verbatim>` wrapping of code/stack lines.

### 7.3 Anti-patterns (NOT to do — still valid)
Full SFT / all-layer LoRA · custom new architecture · RCA-specific special tokens *in the method* · per-task MoE expert · data-synthesis/curriculum as the main contribution · pure RAG · pure prompt-engineering · DPO/RLHF on Nokia data · distill to small model.

### 7.4 mem-X v1 3-regime constraint (still valid)
RCA ≈ **Regime C** (verbatim error code / stack line) → any *compress-to-soft-prompt* route collapses. Favor **overlay** of existing KV / sink positions over adding new tokens, so verbatim info survives. mem-X v1 is the negative teacher.

### 7.5 Original paper story arc (OVER-CLAIMS — see §6.5)
> *"…all the adaptation a frozen base LLM needs … can be localized to these natural sites. We train a <1M-param sink-shaper + per-head temperature router…"* — the "train a sink-shaper as the novel method" framing is preempted; the surviving framing is the §1 protection-rule observation.

### 7.6 Nokia release bridge (mechanism demoted, IP story intact)
```
base = Qwen3-{8B,32B} or Qwen3-MoE  (open, frozen)
+ nokia_rca_<protection-mask|module>  (small, Nokia-owned)
+ [optional] nokia_rca_direction.vec  (R7 task-direction)
```
IP advantage holds regardless of which surviving mechanism wins: customer never re-fine-tunes the base; Nokia releases only its small module; cross-base portability (sink/super-expert phenomena exist everywhere).

---

## 8. AR → dLLM extension path (future room, NOT now)

> Keep the door open from AR (now) to diffusion LLMs; not current scope.

- **Design now**: anchor the narrative on the *principle* ("confine adaptation to load-bearing intrinsic sites; reuse existing representations, don't add tokens") — architecture-agnostic. Abstract "site" as a `site_selector` (AR site = attention sinks; dLLM site = mask-token/BOS under bidirectional attention). Prefer architecture-agnostic levers; use an infilling/reconstruction aux loss (= native dLLM objective).
- **AR vs dLLM**: causal→bidirectional attention (sinks partly a causal artifact → dLLM site = mask/BOS); left-to-right→iterative denoising (recurrence → denoising-step recurrence); next-token→masked reconstruction (native infilling = mem-X Direction D, may fix Regime C); fixed→variable denoising steps (free task-free adaptive compute).
- **dLLM-native upside (future paper)**: verbatim/Regime C natively easier (infilling for free); adaptive compute dial; bidirectional handles "later log explains earlier"; denoising trajectory = natural memory-refinement site.
- **Note**: super experts / sink-induction in **diffusion MoE** is entirely unstudied → another future-room lever.

---

## 9. Open decision questions (refreshed post-audit)

1. **Path**: commit to P0 (unifying-observation + protection rule)? Or also keep (B) verified-new-phenomenon (sink key-bias) alive as a parallel bet?
2. **Base(s)**: dense (Qwen3-8B) + MoE (Qwen3-30B-A3B) both, or MoE-first (super-expert anchor is the sharpest story)?
3. **Experiment scope**: validate the thesis on proxy-domain + code/math/RULER first (method-first, easier to land), and bring RCA in as the application eval second?
4. **Nokia release boundary**: protection-mask only, or also ship optional task-direction vector / light module?
5. **dLLM extensibility**: hard constraint (center `site_selector` + architecture-agnostic levers) or nice-to-have?

**Recommendation**: commit to **P0**, MoE-first (super-expert anchor), method-first eval, keep the dLLM door open via the `site_selector` abstraction. Run P0c before locking anything.

---

## 10. References & sources

Logged in `docs/matrix/knowledge-sources.md`:
- **RCA prior-work audit** section: `[oplora]` `[clora]` `[smf]` `[mofo]` `[migu]` `[mech-forget]` `[ft-no-forget-icl]` `[abft]` `[selfaug]` `[tmkl]` `[logit-lens-loss]` `[distill-lens]` `[sae-ft]` `[sae-fd]` `[sae-tuning]` `[ssa]` `[ssmax]` `[focal-attn]` `[steer-reason]` `[instruct-steer]` `[cache-steer]` `[reasoncache]` `[kv-packet]` `[rlkv]` `[persist-mem-dec]` `[kvm]` `[lcirc]` `[memcom]` `[sink-streaming]` `[sink-emerge]` `[sink-first-token]` `[sink-ctr]` `[massive-act]`.
- **MoE prior-work audit** section: `[super-experts]` `[sink-native-moe]` `[esft]` `[des-moe]` `[loramoe]` `[same-moe]` `[lifelong-moe]` `[gated-attn-sinkfree]`.

Still to chase (not yet logged): Olsson 2022 (induction heads) · Liu 2024 (lost-in-the-middle) · Arditi 2024 (refusal direction) · Tuned Lens (Belrose et al.) · LISA · dLLM set (LLaDA · Mercury/Dream · Block Diffusion · dKV-Cache).
