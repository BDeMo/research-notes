# Long-context inference + catastrophic forgetting via transformer-intrinsic site protection — brainstorm

> **Date**: 2026-06-03 (reorganized conclusion-first; **re-scoped to a general method 2026-06-03**)
> **Topic**: a **general, data-agnostic, transformers-intrinsic** method targeting **two universal transformer problems** — (1) **long-context inference** and (2) **catastrophic forgetting** at training. The method is the contribution; it is *not* RCA-specific.
> **Why RCA appears here**: RCA (long logs + SFT on a new domain) is a downstream application where **both** pains coincide, so it is a natural *motivating + evaluation* case — and the Nokia RCA release is the eventual project payoff. **RCA is an application/eval, not the target of the method.** (File kept under its original `rca-...` name for link stability; the scope is now general.)
> **Status**: 2 brainstorm rounds (dense R1–R12, MoE M1–M9) + full 2022-2026 prior-work audit + prioritization under refined constraints. **Conclusion is in §1 — read that first.** §4 = raw brainstorm, §5 = audit, §6 = surviving direction, §7 = pre-audit material kept for the record.
> **Relation to other work**: complements / contrasts Plan 08 v0 (mem-X soft-prompt wrapper). The mem-X v1 **3-regime law** is a constraint prior (see [`../plans/08-compressed-context-memory/history/v1-results-2026-06-03.md`](../plans/08-compressed-context-memory/history/v1-results-2026-06-03.md)).
> **Promoted to a plan**: [`../plans/09-intrinsic-site-protection/`](../plans/09-intrinsic-site-protection/) — Phase-1 observation study of the long-ctx↔forgetting coupling, then a targeted anti-forgetting method.

---

## 0. Design rules (distilled — the durable constraints for this line of work)

> Standing principles distilled from the user's requirements (**U**) and the good points that survived the brainstorm/audit (**E** = emerged). **Any idea or plan under this topic must comply.** These are the constitution; §1 is the current conclusion; the rest is derivation.

**Goal**
- **DR1 (U)** Solve **two general transformer problems**: long-context inference **and** catastrophic forgetting. The *method* is the contribution and must stand on these two alone.
- **DR2 (U)** **RCA is the application where both coincide** (showcase eval + Nokia release payoff), never the method's target. Code/math are the preserved-capability probes.

**Method constraints (the filter)**
- **DR3 (U)** **Data-agnostic** — the method's core decisions come from a forward pass on *generic* text; no task labels inside the method.
- **DR4 (U)** **Transformers-intrinsic** — built on phenomena universal to transformers (attention sinks, massive activations, induction/retrieval heads, super experts), not architecture-specific tricks.
- **DR5 (U)** **Lightweight + model-agnostic** — frozen public base + small/zero trainable module; portable across model families & sizes.
- **DR6 (U)** **Light training OK, but not task-specific** — prefer a *constraint on* training (freeze / grad-mask / regularize) over a new task-trained adapter. If something is trained, use a task-agnostic objective (e.g. self-supervised infilling).
- **DR7 (U)** **OSS-acceptable, not big-lab-exotic** — implementable with HF-basic (forward hooks, grad masks, PEFT one-liners) or AG2-basic; no bespoke kernels.

**Scientific principle (the thesis)**
- **DR8 (E)** **Shared-substrate hypothesis** — the intrinsic load-bearing sites are *both* the long-context carriers *and* the forgetting-vulnerable sites. Two failure modes = **read-time overload** (long ctx) + **write-time perturbation** (forgetting) of one substrate.
- **DR9 (E)** **Site-selection by an intrinsic criterion**, *not* by task affinity (ESFT/DES-MoE) or raw gradient magnitude (MoFO/MIGU). The criterion (sink-induction / massive-activation / super-expert) is the unclaimed novelty.
- **DR10 (E)** **Overlay, don't add** (mem-X 3-regime law) — reuse the base's existing KV / sink positions; never add new soft tokens that destroy verbatim needles (Regime C).

**Novelty discipline**
- **DR11 (U+E)** **Brainstorm → audit → only then build.** No angle is "ours" until a 4-year lit check clears it. The honest contribution is the *unifying observation + the criterion*, not a new single mechanism (all single mechanisms are preempted — §5).
- **DR12 (E)** **Differentiate from the nearest threat explicitly** — here `[mech-forget]` already localizes forgetting to attention heads; our edge is the long-ctx↔forgetting *coupling* + the intrinsic site criterion.

**Evaluation**
- **DR13 (E)** **General-first eval** — long-context benchmark (read side) + continual-SFT forgetting benchmark (write side); RCA last as the both-at-once showcase. Cross-domain / cross-task / cross-model required.
- **DR14 (E)** **4-seed gate** for every headline number (mem-X v1 lesson: single-seed "wins" evaporated).

**Future room**
- **DR15 (U)** **AR → dLLM extensible** — anchor on the architecture-agnostic *principle*; abstract "site" as a `site_selector`; favor the DR8 substrate framing over AR-only induction heads.

---

## 1. TL;DR — current conclusion (read this first)

**Scope**: a **general method for two universal transformer problems — long-context inference + catastrophic forgetting**. RCA is an application where both coincide (the eval/motivation), *not* the target.

**Verdict after the audit (§5)**: none of the 12 dense angles (R1–R12) nor the 9 MoE angles (M1–M9) is novel **as a standalone mechanism** — the 2025-2026 dense *and* MoE literatures are both saturated (OPLoRA, Sparse-Memory-FT, ESFT, DES-MoE, Same, MoFO/MIGU, SAE-FT, …).

**Under the refined constraints** — **(D) data-agnostic · (I) transformers-intrinsic · (T) light training OK but ideally NOT task-specific** — the whole space collapses onto a single defensible, *general* program:

> **P0 thesis (a general claim about transformers).** The transformer-intrinsic sites that carry long-context behavior — **attention sinks / massive activations** (dense), the **super experts that induce those sinks** (MoE) — are the *same* sites whose perturbation during fine-tuning causes catastrophic forgetting. The two failure modes are **read-time overload** (long context) and **write-time perturbation** (forgetting) of one shared substrate. So one **data-agnostic, training-free** rule — *detect these sites on generic text, then regulate/protect them (freeze / gradient-mask) during any SFT* — improves long-context retention **and** prevents capability forgetting, **adding zero task-specific trainable parameters**. This holds for *any* model and *any* fine-tuning domain; RCA is simply the application where both pains are acute at once.

**Priorities** (full table §6.3): **P0c** de-risk measurement experiment (model-agnostic, **no task data**) → **P0a** dense + **P0b** MoE (super-expert) instantiation of the protection rule. Everything that *trains a task-specific module* (R1/R2/R3/R8/R10) is demoted to a **baseline**; R9/R12 dropped; R6/R7/sink-key-bias kept as components.

**Next action** (§6.4): run the P0c experiment on Qwen3-8B + Qwen3-30B-A3B, then a targeted novelty search on "super-expert + fine-tuning + forgetting", then draft `notes/plans/09-intrinsic-site-protection/`.

**Honest caveat** (§6.5, sharpened by the 2026-06-04 search §5.5): **both legs are now individually preempted** — `[sink-forget]` (2410.05648) already publishes "sink → forgetting + fix"; `[focusft]` (2605.09932) already publishes "sink dilutes long-ctx at training + fix". The **ONLY** surviving contribution is the **joint claim**: one site-selection criterion, the *same* sites, governing BOTH legs, co-evaluated on a decoder LLM (+ MoE super-expert instantiation). P0c must beat stacking the two single-leg fixes or there is no paper. Path B (sink-key-bias-only tuning) is **closed**.

---

## 2. The problem — two general transformer challenges (RCA is downstream)

The **method targets two universal problems**, independent of any task:
- **Long-context inference**: as context grows, attention entropy explodes / lost-in-the-middle / KV blows up.
- **Catastrophic forgetting**: SFT on *any* new domain drifts away pretrained ability (e.g. code/math).

**Paper goals (the method, do first)**: a method that **improves long-context behavior AND prevents forgetting**, using techniques the big-model/agent world is **not yet using** but the OSS community accepts (HuggingFace-basic, or AG2-basic), **easy to adapt + lightweight**, ideally **model-agnostic + task-free**, derived from **transformer-intrinsic phenomena**. Eval = a *general* suite (long-context benchmark + a continual-SFT forgetting benchmark), with code/math as the preserved-capability probe.

**Why RCA (the application)**: RCA = long logs/stack traces (long context) + SFT on a new domain (forgetting) → the **one application where both challenges fire simultaneously**, so it is the natural showcase eval. It is *not* the method's target and the method must stand on the two general phenomena alone.

**Project goals (after the paper)**: apply the general method to build a Nokia RCA model — in-domain generalization to other RCA datasets/abilities; final deliverable = **a Nokia-owned RCA model release**. (Downstream payoff, not the paper's contribution.)

**Derived release constraint**: cleanest IP story = *frozen public base + small Nokia-owned module* → pushes toward **frozen-base + tiny-trainable (or zero-trainable) designs** — which also happens to be exactly what the general data-agnostic protection rule is.

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

1. **Unifying-observation paper** (not a new mechanism). No paper uses **one intrinsic site to solve BOTH** long-context inference AND training-time forgetting. Hardest to preempt because it's about the *relationship* between the two phenomena — a general claim about transformers, not a task method. Main threat `[mech-forget]` (localizes forgetting to attention heads) → differentiate via the long-ctx coupling + the sink-induction site-selection criterion. RCA is the application where both fire at once (showcase eval). **→ this is P0.**
2. **Verified-new-phenomenon**: sink **key-bias-only** tuning (§5.2) or the *joint dynamics* of context-absorption vs capability-disruption at the same site. Needs a dedicated search first.

### 5.5 Path-B verification search (2026-06-04) — sink-key-bias is CLOSED; two new threats to P0; the joint wedge survives

> Triggered by user 2026-06-04 ("先把 (B) 的 sink key-bias gap 搜清楚再决定框架"). Dedicated 4-search pass on "tune only the attention-sink key-bias / sink positions to fix long-context and/or forgetting". **Verdict: Path B (sink-key-bias-only tuning as a *new mechanism*) is closed. Worse — both legs of P0 now have direct single-leg prior work. The ONLY surviving wedge is the *joint* claim (same site → both, one lever, co-eval).**

| claim under test | closest prior | status |
|---|---|---|
| sink = a *key bias* (the framing) | `[sink-emerge]` (2410.10781, ICLR'25): sink = implicit key bias; proposes learnable K-biases k* | ☠️ framing owned (as pre-train arch choice) |
| tune the sink as a *single knob* | **ZeroTuning** (`[zerotuning]`, 2026): scalar bias on first-token logit, inference-time, redistributes whole map | ☠️ inference knob done |
| training-free sink calibration | **ACT** (2406.15765): per-head offline sink calibration at inference | ☠️ done |
| **PEFT on only the special/sink-token reps** | **PASTA** (EACL'23, `[pasta]`): train only special-token ([CLS]/[SEP]) hidden vectors per layer, frozen base, **0.029% params** | ☠️ mechanism owned (BERT/NLU) |
| sink-token attention inside long-ctx PEFT | **SinkLoRA** | ☠️ done |
| **attention sink → catastrophic forgetting** | **RoBERTa-CL** (2410.05648, `[sink-forget]`): sink → over-smoothing → forgetting; mitigate via learned **pre-scaling** layer (probe→FT) | ⚠️ **NEW threat to P0 forgetting-leg** |
| **sink dilutes long-ctx at *training* time** | **FocuSFT** (2605.09932, `[focusft]`): bilevel opt, fast-weights pull attention off sinks; +14pp BABILong, sink mass ÷529 | ⚠️ **NEW threat to P0 long-ctx-leg** |
| the area as a whole | **Attention-Sink Survey** (2604.10098, 2026, `[sink-survey]`): taxonomizes gated-attn / mod-softmax / learnable-bias / registers | ⚠️ space fully mapped |

**Two consequences for the plan:**

1. **Drop Path B / demote sink-key-bias to "dead component," not "P2-verify."** PASTA (PEFT on only special-token reps, 0.029%) + ZeroTuning (single-scalar sink knob) + `[sink-emerge]` (key-bias framing) jointly close "tune only the sink key-bias." It is not a paper.
2. **P0's *two legs are individually preempted*** — this is the important update. `[sink-forget]` already publishes "attention sink → forgetting" *with a fix*; `[focusft]` already publishes "sink dilutes long-context at training time *with a fix*." So neither the forgetting-leg nor the long-ctx-leg is novel **alone**.

**What still survives (and is now evidence-backed, not hopeful):** **nobody connects the two legs.** `[sink-forget]` is encoder/RoBERTa, continual-learning NLU, forgetting-only. `[focusft]` is decoder, long-context-only. **No paper shows the *same* intrinsic site is the shared root of BOTH long-context-at-inference AND forgetting-at-training, regulated by *one* site-selection criterion, co-evaluated on a decoder LLM (+ the super-expert/MoE instantiation of §6.2).** That is exactly the §5.4-1 / §6.1-(2) **joint claim** — the unifying observation. The search *narrows* P0 to its only defensible core and kills the fallbacks around it.

**Bar this raises for P0c (§6.4):** the de-risk experiment must now produce a result that is *surprising given* `[sink-forget]` + `[focusft]` existing separately — e.g. (a) the *same* detected site set predicts BOTH ΔRULER (long-ctx) and ΔGSM8K/ΔHumanEval (forgetting) with one criterion, and (b) one protection lever Pareto-dominates applying the two separate single-leg fixes. If P0c can't beat "just stack `[focusft]` + `[sink-forget]`," there is no paper. Add both as mandatory baselines.

**To log in `knowledge-sources.md`:** `[zerotuning]` (ZeroTuning, sink scalar knob) · `[act-sink]` (ACT, 2406.15765) · `[pasta]` (PASTA special-token PEFT, EACL'23) · `[sinklora]` · `[sink-forget]` (2410.05648, sink→CL forgetting + pre-scaling) · `[focusft]` (2605.09932, sink-dilution long-ctx SFT) · `[sink-survey]` (2604.10098).

### 5.6 Joint-claim + MoE-gate + feature-leg search (2026-06-04, round 2) — P0b survives; the coupling wedge is real but *narrow and predictive*

> Triggered by user 2026-06-04 ("做剩下的 search,也做一遍 related work,保证 novelty;feature 层面有没有能同时解决的"). Three searches: (a) super-expert + FT + forgetting (the P0b gate), (b) the JOINT long-ctx↔forgetting coupling, (c) feature/SAE joint mechanism.

**(a) P0b gate — super-expert protection for FT-without-forgetting: SURVIVES (this is now the sharpest novelty).**

| who freezes/selects experts to stop forgetting | selection criterion | verdict vs us |
|---|---|---|
| **DAS** (OpenReview zBgjWTWgCh) two-stage, freeze all but top-k | **domain affinity** | task-specific — we differ (intrinsic) |
| **ESFT** `[esft]` / **DES-MoE** `[des-moe]` | task/domain affinity | baselines (criterion competitors) |
| **Same** (stabilized MoE) | task-relevant + router stab | task-specific |
| **ExpertCondenser / MoECondenser** (2604.23036, DxbLY3Fctc) | preserve **long-tail** experts via gated condensers | *opposite* end (long-tail, not super-experts) |
| `[super-experts]` (2507.23279) | super-expert / sink-induction (**intrinsic, data-agnostic, post-training-stable**) | **only does compression — NOT FT-protection** |

→ **Gap confirmed**: nobody protects the **super-expert / sink-induction set** (intrinsic criterion) *during downstream SFT to prevent forgetting*. `[super-experts]` even reports SEs are **data-agnostic and consistent after post-training** — exactly the property a data-agnostic protection rule needs. This is the cleanest, least-crowded instantiation → make MoE-super-expert the **headline**.

**(b) The JOINT coupling — narrow wedge survives, but it must be stated as a *predictive* claim.** New near-threats:

| paper | what it shows | why it's NOT us |
|---|---|---|
| `[mech-forget]` (2601.18699) | forgetting localizes to heads; 15–23% reorganized; ablating **ΔW-disrupted** heads restores 47% | defines the damaged set **post-hoc by ΔW** (write-side, after training). We **predict it a priori by read-side long-ctx importance** — the coincidence of the two rankings is the novel claim. |
| **LCCP-dynamics** (2604.02650, `[lccp-dyn]`) | retrieval heads stable (>93%), refined during long-ctx **continual pre-training**; mentions forgetting | continual **pre-training** for long-ctx, not downstream-**SFT** forgetting; descriptive, no protection method keyed on the criterion |
| **RL-vs-SFT circuits** (2605.28860, `[rl-circuits]`) | differential circuit vulnerability; RL preserves circuits > SFT | head-level forgetting, no long-ctx coupling, no method |
| **DuoAttention** (2410.10819, `[duo-attn]`) | clean split: **retrieval heads** vs **streaming/sink heads** | uses the taxonomy for **KV compression (inference)**, not training-time protection |
| `[retrieval-head]` (2404.15574, ICLR'25) + **Retrieval-Heads-are-Dynamic** (2602.11162, `[ret-dyn]`) | retrieval heads: universal, sparse(<5%), **intrinsic**, causal; dynamic & irreplaceable | establishes the read-side site; nobody links it to **SFT forgetting + protection** |

→ **Surviving novelty (precise)**: *the read-side, data-agnostic intrinsic-importance ranking (retrieval-head / sink / super-expert score) **predicts** the write-side forgetting-disruption ranking, so one can protect the load-bearing sites **before** SFT with a **task-agnostic** criterion.* No paper makes this **predictive** link or turns it into a downstream-SFT protection rule keyed on the **long-context** criterion. `[mech-forget]` is the thing to beat: our criterion is *a priori & read-side*, theirs is *post-hoc & ΔW*.

**(c) Feature/SAE leg — owned on the forgetting half; weak edge.** **SAE-FD** (2605.25525, `[sae-fd]`): SAE-feature distillation for continual learning (anchor active features, cosine+magnitude loss) — already the feature-space anti-forgetting method. SAFE (2025.findings-emnlp.496) = SAE features for hallucination/steering. SAE-retrieval (2603.13277) = SAE features as retrieval units. → Nobody does SAE features for the **joint** long-ctx+forgetting, but (i) SAE-FD is a strong incumbent on the forgetting half, (ii) an SAE adds a trained dictionary = violates the "zero-task-param / data-agnostic" appeal, (iii) the feature route's edge over the head/expert route is unclear. **Verdict**: keep SAE as an *analysis lens / optional baseline* (`[sae-fd]` becomes a feature-space baseline), **not** the main mechanism. The head/expert/super-expert site is the cleaner story.

**Empirical check (2026-06-03 Phase-0 run on Qwen3-8B, forward-only):** detectors reproduce all three phenomena (BOS sink L13H12=1.0; **5 massive-act channels**, ch2276≈410× median; retrieval heads sparse, hub at L23). **Bonus**: top sink heads (L7–13) vs top retrieval heads (L17–31) are **disjoint** (Jaccard 0, ρ=0.09) — confirms DuoAttention and means **the long-ctx site to protect is the *retrieval heads*, not the sink** (further kills sink-only Path B). Full note: `plans/09-intrinsic-site-protection/phase0-results-2026-06-03.md`.

**Net effect on the plan:**
1. **Headline instantiation = MoE super-expert protection** (P0b) — least crowded, sharpest differentiation (intrinsic vs task-affinity).
2. **The scientific core = the *predictive* coupling** (H2 restated): read-side importance ranking predicts write-side disruption ranking. Beating `[mech-forget]` means showing our *a-priori read-side* criterion matches/beats their *post-hoc ΔW* criterion for choosing what to protect.
3. **New mandatory baselines/threats** to add to `channels.md`: `[mech-forget]` (ΔW-disrupted-head protection), DAS, ExpertCondenser, `[duo-attn]` (retrieval/streaming split as the site oracle), `[sae-fd]` (feature-space anti-forgetting), `[focusft]`+`[sink-forget]` (the two single-leg fixes — must beat their stack).
4. **New `knowledge-sources.md` IDs**: `[das-moe]` (zBgjWTWgCh) · `[expert-condenser]` (2604.23036) · `[lccp-dyn]` (2604.02650) · `[rl-circuits]` (2605.28860) · `[duo-attn]` (2410.10819) · `[ret-dyn]` (2602.11162) · `[sae-fd]` (2605.25525) · `[retrieval-head]` (2404.15574).

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
| sink **key-bias-only** tuning | ✓ | ✓ | tiny, task-ish | ☠️ **closed** (PASTA + ZeroTuning + `[sink-emerge]`, §5.5) | dropped — not a paper, not a component |
| R6 massive-act grad-mask | ✓ | ✓ | adds none | ☠️ MoFO/MIGU | P2 — component of P0a |
| R7 task-direction steering | ~ (contrast pairs) | ✓ | ~zero | ☠️ preempted | P2 — baseline / inference topping |
| R1 sink-anchored memory (trained) | ~ | ✓ | task-ish | ⚠️ thin | P3 — component / ablation |
| R8 / R10 / R2 / R3 | ✗ task-specific module | ✓ | task-specific | ☠️ preempted | **baselines only** |
| R9 / R12 | ✓ | ✓ | none | ☠️ mature | drop |

### 6.4 Next actions (cheap-first)

1. **P0c de-risk experiment — needs NO RCA data, ~1–2 days.** On Qwen3-8B (dense) + Qwen3-30B-A3B (MoE): (a) detect sites on generic text (sink positions, top massive-act channels, super experts via sink-decay); (b) run a *small proxy-domain* SFT (non-code/math corpus), measure which sites shift, correlate site-shift with forgetting (ΔGSM8K/ΔHumanEval) and long-ctx (ΔRULER). **Goal: confirm/kill the P0 thesis before any RCA work.**
2. ~~**Targeted novelty search** "super-expert / sink-induction + fine-tuning + forgetting" — gates P0b.~~ **DONE 2026-06-04 (§5.6): gate PASSES.** All MoE-forgetting work selects experts by task/domain affinity (DAS/ESFT/DES-MoE/Same) or long-tail (ExpertCondenser); nobody uses the intrinsic super-expert/sink-induction criterion for FT-protection. P0b is the headline.
3. **If P0c positive + gap holds** → draft `notes/plans/09-intrinsic-site-protection/`. **Eval is general-first** (the method is the contribution): long-context benchmark (RULER / long-doc QA) for the read side; a continual-SFT forgetting benchmark (SFT on an arbitrary domain, measure retention on GSM8K / HumanEval / MMLU) for the write side; **RCA used last as the "both pains at once" showcase**, not as the method's definition. Channels = HF Qwen3 dense+MoE; budget = small (mostly inference + light SFT).
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

### 7.2 RCA "free toppings" (application layer only — NOT part of the general method or its eval)
For the *downstream Nokia application*, not the paper: structured input anchors `[STACK] [LOG] [STATE] [QUERY]` · outlier-line amplification (entropy filter) · causal-DAG-aware chunking (timestamp/causal order) · `<verbatim>` wrapping of code/stack lines. These keep the general method task-free; they live in the RCA deployment, not in the method or the general benchmark.

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

1. **Path**: ~~commit to P0, or also keep (B) sink-key-bias alive?~~ **Resolved 2026-06-04 (§5.5): Path B is closed** (PASTA + ZeroTuning + `[sink-emerge]`). P0 (unifying joint observation) is the only survivor — and even its two legs are individually preempted (`[sink-forget]`, `[focusft]`), so the joint claim must carry everything. Remaining open: does the §6.2 super-expert-protection gap survive its own search (§6.4-action-2, still TODO)?
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
