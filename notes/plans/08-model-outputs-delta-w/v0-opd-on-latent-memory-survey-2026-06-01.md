# OPD on Latent Memory / Reasoning — Literature Survey

> Generated 2026-06-01 while Phases K/L/M/N/O/P were running on the pod.
> User prompt: "has anyone done on-policy distillation on latent memory /
> reasoning?" — survey the last two years, then write down the gap and
> the actionable consequences for our work.

---

## TL;DR

Three-sentence answer:

1. **No one has shipped exactly our recipe** — *on-policy distillation
   (OPD) into a learned recurrent latent-memory wrapper on a frozen
   base*. The closest, OPCD (Microsoft, Feb 2026), internalizes
   context into model **weights**, not into a wrapper, and does not
   keep the base frozen.
2. **Each piece exists separately** — OPD into model weights (OPCD,
   MiniLLM, MOPD, GLM-5 cross-stage), RL/GRPO into latent memory in
   multi-agent settings (LatentMem-LMPO), RL into latent reasoning
   itself (Latent-RL on Coconut), KV-cache RL alignment (Shadow Mask
   Distillation), and SFT-only soft-prompt memory wrappers
   (MemoryPrompt, TokMem, AutoCompressor, Gist) — but no single
   paper combines (a) OPD + (b) recurrent latent memory + (c) frozen
   base + (d) compression-bounded soft tokens.
3. **The most threatening prior art is OPCD** (Microsoft, Furu Wei
   team) and **LatentMem-LMPO** (Tencent et al., Feb 2026). We need
   to differentiate from both explicitly in the paper and in the
   abstract.

---

## What "OPD on latent memory/reasoning" means — three readings

We should be explicit about which one(s) of these we're doing,
because the literature splits along these axes:

| reading | what's "latent" | what's "OPD" | who's done it |
|---|---|---|---|
| **R1: latent = soft memory tokens** | a wrapper writes K soft tokens summarizing a long input; the base reads them via prefix/xattn/etc. | the wrapper's parameters are trained by sampling student rollouts, scored by a full-context teacher | **us** — and the closest neighbor is OPCD with weights-not-wrapper |
| **R2: latent = continuous-thought reasoning state** | the model thinks by feeding hidden states back as next-token embeddings (à la Coconut) | RL/distillation gives explicit supervision to the latent thoughts | **Latent RL on Coconut** (Dec 2025), **HRPO** / **SofT-GRPO** (2026) |
| **R3: latent = compressed KV cache** | the cache is pruned during rollout; the policy must adapt | learner is aligned to the rollout-time sparse policy on-policy | **Shadow Mask Distillation** (May 2026), **RLKV** (Oct 2025), **KVP** (Feb 2026) |

Our work is squarely in **R1**, with some methodological overlap
with **R2** (the wrapper's m_T is functionally a "continuous
thought" of the long context).

---

## The map of prior art (2024 – mid-2026)

### A · On-policy distillation into model weights

This is OPD as a post-training stage; the student is the base model
itself (or a smaller version of it), not a wrapper. **Doesn't touch
latent memory**, but defines the canon we're applying.

| year | paper | key idea | how it differs from us |
|---|---|---|---|
| 2024 ICLR | **Agarwal et al., On-Policy Distillation of LMs** | foundational: sample from student, teacher scores each token, KL | distills into base weights, not into a memory module |
| 2024 | **MiniLLM** (Gu et al.) | reverse-KL OPD for long-form generation | same — model-weight distillation |
| 2026-02 | **OPCD** (Ye et al., MSR) — *closest neighbor in R1* | bridges OPD with **context distillation**: distill the behavior of a context-conditioned teacher into a student's weights using student-sampled rollouts | **trains weights, not a wrapper; base not frozen; one prompt at a time** — we train one wrapper that works across many prompts via a recurrent memory |
| 2026-02 | **Privileged Info Distill (π-Distill / OPSD)** | teacher sees privileged info (CoT/trace); student doesn't; OPD with reverse-KL | privileged info = text, not latent memory; full-weight student |
| 2026-02 | **OP prefix distillation** (Optum AI, Zhang et al.) | truncate rollouts, distill on prefix only | math reasoning; no memory wrapper |
| 2025-26 | **MOPD** (MiMo-V2), **GLM-5 cross-stage OPD**, **Qwen3 strong-to-weak**, **HY-MT1.5 SFT+OPD+RL**, **Baichuan-M3 task-RL+OPD**, **Nemotron-Cascade 2** | industrial three-stage recipes (SFT → OPD → RL) | very similar pipeline to ours; but into base weights and at frontier-model scale |
| 2026 | **A Survey of OPD for LLMs** (arXiv:2604.00626v3) | formalizes OPD as f-divergence min over student-sampled trajectories | the survey; doesn't itself overlap |

### B · RL / OPD into latent memory or memory policies

The most relevant cluster. Each one occupies a slightly different
corner of the same space.

| year | paper | latent thing | training signal | how it differs from us |
|---|---|---|---|---|
| 2024 (LREC) | **MemoryPrompt** (Pannitto et al.) | recurrent MLP writes 5 soft vectors prepended to base each turn | **standard next-token CE only** | almost the same architecture but only SFT; we're adding OPD + RL |
| 2024 ICLR | **AutoCompressor** (Chevalier et al.) | summary tokens accumulated across segments | SFT next-token loss | older, SFT only |
| 2023 NeurIPS | **Gist Tokens** (Mu et al.) | k tokens per chunk via prefix-tuning | SFT only | our explicit baseline; SFT-only architecture |
| 2025 | **ICAE** (Ge et al.) | context auto-encoder into memory slots | reconstruction loss | encoder-decoder, no on-policy |
| 2025-03 | **DCD: Deep Context Distillation** (Caccia et al.) | one LoRA module per document | hidden-state + output KL on context-vs-no-context | **per-document LoRA**, supervised KL not on-policy |
| 2026-05 | **Context Distill as Latent Memory Management** (arXiv:2605.28889) | per-context LoRA + cache-sharing self-gating | off-policy KL distillation | one adapter per doc; no on-policy sampling |
| 2026-05 | **MEMO** (modular memory model) | small dedicated LM stores knowledge, frozen executive LLM queries | SFT on answer tokens | text-mediated; no shared wrapper across queries |
| 2026 ICLR | **TokMem** (one-token procedural memory) | one trainable token per recurring procedure | standard NTP loss only | procedural lookup; not long-context compression; no OPD |
| 2026-03 | **Trained Persistent Memory for Frozen Enc-Dec LLMs** (arXiv:2603.16413) | 6 architectures, adapter + persistent bank | SFT proof-of-concept | small scale, SFT only; doesn't claim a method, just feasibility |
| 2025-08 | **Memory-R1** (Yan et al.) | GRPO over **add/update/delete** ops on a text memory bank | GRPO with verifiable QA reward | **text bank**, not latent; RL over memory ops not over latent state |
| 2025 NeurIPS | **MEM1** (Zhou et al.) | RL trains agent to consolidate context into a hidden state each turn | end-to-end GRPO | **trains the base**; we keep it frozen |
| 2026-01 | **MemRL** (Wang et al.) | non-parametric PPO over an episodic Intent-Experience-Utility memory bank with frozen LLM | RL over retrieval policy | **text bank**, retrieval-policy not latent compression |
| 2026-02 | **LatentMem / LMPO** (Wang et al.) — *closest neighbor in R1+B* | memory **composer** distills multi-agent trajectories into latent memories; **GRPO variant (LMPO) backprops through the latent memories** | **frozen agent backbones + RL on latent memory representations** | this is the *only* "RL on latent memory with frozen base" paper, but for multi-agent trajectory composition, not single-base recurrent long-context |
| 2026-03 | **MemFactory** | unified framework: GRPO on memory add/update/retrieve ops | GRPO | framework not method; text-mediated memory |
| 2026 | **ELMUR** | layer-local external memory with LRU updates | end-to-end loss | trains base; not OPD |

### C · RL / distillation on latent **reasoning** (Coconut family) — R2

| year | paper | what differs from us |
|---|---|---|
| 2024-12 | **Coconut** (Hao et al., Meta) — continuous-thought reasoning | **trains the base model**, not a wrapper; staged SFT only; for CoT not for long context |
| 2024 | **Implicit CoT** (Deng et al.) | distill CoT into latent | base-weight training |
| 2024 | **Pause Tokens** (Goyal et al.) | thinking tokens inserted | base-weight training |
| 2024 | **Quiet-STaR** (Zelikman et al.) | generate rationales then REINFORCE-style | base-weight + text rationales, not latent memory |
| 2025-12 | **Latent RL on Coconut** (arXiv:2512.11816) | value-head RL gives explicit supervision to latent thoughts; "Coconut SFT is unstable" | trains base + value head; for CoT/math not long context |
| 2026 | **HRPO** / **SofT-GRPO** | hybrid gating / Gumbel-Softmax for latent RL | base-weight training; for CoT |

### D · KV-cache RL alignment (R3)

Different problem (eviction policy), shares the on-policy alignment idea.

| year | paper | how it differs from us |
|---|---|---|
| 2025-10 | **RLKV** | RL probes which heads matter; doesn't add a memory module |
| 2026-02 | **KVP** | per-head RL agent decides eviction order; doesn't add new memory tokens |
| 2026-05 | **Shadow Mask Distillation (SMD)** | aligns RLHF policy with sparse-rollout KV; **on-policy alignment trick** we should borrow |

### E · Other context-compression with RL (text-level, not latent)

| year | paper | how it differs from us |
|---|---|---|
| 2025-10 | **Acon** | RL on natural-language compression **guidelines**; distills into a small compressor | text compression, not latent |
| 2026-05 | **ZipRL** | GRPO with hindsight response replay for context compression | text compression |
| 2026-05 | **NVIDIA Polar** | rollout framework for GRPO across agent harnesses | infra not method |

---

## Where we sit — the gap map

Plotting the literature on two axes:

```
              ┌─────────────────────────────────────────────────────────┐
              │                                                         │
   LATENT     │   LatentMem-LMPO ◄── (closest, but multi-agent only) ── │
   memory     │                                                          │
   trained    │   ● US (mem-embedding wrapper + SFT+OPD+RL on frozen)    │
   on-policy  │                                                          │
              │   ◇ MEM1 (RL on consolidated state, but base not frozen) │
              │   ◇ Latent RL on Coconut (R2, base not frozen)           │
              │                                                          │
              │   ◊ Memory-R1, MemRL, MemFactory                         │
              │     (RL on TEXT memory ops, not latent)                  │
              │                                                          │
              ├─────────────────────────────────────────────────────────┤
              │                                                          │
   TEXT or    │   ◆ OPCD (closest, but distills into WEIGHTS)            │
   WEIGHTS    │   ◆ DCD, MiniLLM, π-Distill, OP-prefix-distill           │
   trained    │   ◆ MOPD, GLM-5, Qwen3, HY-MT1.5, Baichuan-M3            │
   on-policy  │     (industrial OPD pipelines into weights)              │
              │                                                          │
              └─────────────────────────────────────────────────────────┘
                  frozen base?              base trainable?
```

The **white space** we occupy:

> **OPD into a recurrent latent-memory wrapper on a strictly frozen
> base, where the memory is K bounded soft tokens produced by an
> additive residual update $m_t = m_{t-1} + \alpha \cdot \Delta
> m_t$, and the wrapper is trained end-to-end SFT → OPD → REINFORCE
> against a full-context teacher.**

Nobody hits all four highlighted properties at once.

---

## The two papers we have to differentiate from in the paper intro

### OPCD (Microsoft, Furu Wei team, Feb 2026)

* **What they do**: OPD where the student samples without context,
  teacher scores with context; minimize RKL on student trajectories;
  internalize into student weights.
* **What's the same as us**: the OPD-with-context-conditioned-teacher
  setup; reverse KL on student rollouts.
* **What's different**:
  1. They distill into model **weights** (full fine-tuning or LoRA);
     we distill into a **wrapper attached to a frozen base**.
  2. They are **per-context** (one specific prompt or document is
     distilled into one model); we train **one wrapper that works
     across many contexts** because the recurrence accepts any
     chunk sequence.
  3. They have no capacity bound; we have an explicit
     $K$-soft-token bit-capacity wall, which is the empirical
     phenomenon our paper studies.
* **How to position**: "OPCD-with-soft-memory-and-frozen-base" — we
  inherit their distillation objective but specialize it to a
  parameter-efficient, deployable memory module. Cite OPCD as the
  immediate ancestor.

### LatentMem / LMPO (Tencent et al., Feb 2026)

* **What they do**: memory composer in a multi-agent system; GRPO
  variant (LMPO) backprops task reward through latent memories.
* **What's the same as us**: RL/GRPO over learned **latent
  memories**, with frozen backbones.
* **What's different**:
  1. They work in a **multi-agent trajectory** setting (compose
     memories from prior MAS rollouts); we work on **single-base
     long-context compression**.
  2. Their latent memories are **per-rollout**; ours are a
     **streaming recurrent state**.
  3. They use GRPO with relative advantages over multi-agent
     rollouts; we use SFT-warm + OPD + REINFORCE-LOO single-agent.
* **How to position**: same family of methods (RL on latent
  memory under a frozen base) but a different problem instance.

---

## Brainstorm — five things this survey makes us want to try

### B1 — Add a value head to m_T (steal from Latent RL on Coconut)

The Latent-RL paper (Dec 2025) explicitly says **"Coconut SFT
provides no direct supervision for latent steps; training is
unstable; a value head sharing parameters with the policy fixes
this"**. We have the same problem: the m_T tokens get *very*
indirect supervision through the base model's logits. A value head
$V_\phi: m_T \to \mathbb{R}$ trained on the eventual reward could
provide direct supervision to the memory state itself.

* Cost: small Linear head; one extra loss term in BPTTTrainer.
* Risk: small (purely additive).
* Expected lift: medium — addresses the same "indirect supervision"
  pathology that has plagued our wrapper.
* **Adds straight onto Phase Q after N/O/P finishes.**

### B2 — Replicate OPCD as a baseline on our datasets

If we beat OPCD on `categorical_niah` / `numerical_niah` /
`coding_niah` with our wrapper, that's a publishable lift over the
direct ancestor.

* Cost: OPCD code is on `aka.ms/opcd-code`; needs adapting to our
  harness.
* Risk: OPCD might just win at low capacities. That's a useful
  finding too.
* **Adds straight to the §4 ablation table.**

### B3 — On-policy sampling of m_T trajectories (not just student tokens)

Currently OPD samples student tokens conditioned on a single m_T
trajectory (the encoder-driven one). We could also sample memory
trajectories — e.g., dropout in `_CrossAttnBlock`, multi-seed
update — and average / select. This is "on-policy" in the
*memory-update* policy, not just the token-output policy.

* Cost: a few lines in BPTTTrainer.
* Risk: gradient variance explodes.
* Expected lift: small but novel — nobody does this.

### B4 — Cross-session m_T persistence (toward v2)

Train the wrapper to produce m_T that can be SERIALIZED + RELOADED
later in a different session, with no loss on a held-out query
about the original session. This is exactly the v2 axis we already
locked in (`v2-plan.md` / `CONVENTIONS.md` v2 extensions). The
survey confirms NO ONE has shipped this for soft-token memory.

### B5 — Borrow Shadow-Mask-Distillation's on-policy alignment trick

For our RL stage, the wrapper computes m_T at rollout time; if we
then evaluate the wrapper at gradient-update time we MUST use the
same m_T (otherwise the rollout's logprobs and the gradient-time
logprobs disagree → off-policy bias). We probably already do this
correctly because BPTTTrainer recomputes m_T fresh per step, but
worth a single-line check in `RLTrainer`. SMD's framing is the
clean way to write this in the paper.

---

## Five things this survey reveals as WEAKER claims than we thought

1. **"Three-stage SFT + OPD + RL is novel" — false.** Every Chinese
   industrial frontier model (Qwen3, GLM-5, MiMo-V2, HY-MT1.5,
   Baichuan-M3, Nemotron-Cascade-2) uses essentially this exact
   pipeline. Our novelty has to be on the MEMORY-MODULE side, not
   the recipe.
2. **"OPD on latent reasoning is unexplored" — false.** Latent RL
   on Coconut (Dec 2025) does exactly this for CoT. We need to be
   specific: "OPD on latent **memory** for long-context
   compression on a **frozen base**" — that's still ours.
3. **"RL on latent memory is unexplored" — false.** LatentMem-LMPO
   (Feb 2026) is the same idea in multi-agent. We need to argue the
   single-base, recurrent, K-bounded setting matters.
4. **"Memory wrapper on frozen base is unexplored" — half-false.**
   MemoryPrompt (LREC 2024) has the same architecture, just SFT
   only. We need to credit them and show OPD + RL is the missing
   piece.
5. **"Soft memory tokens for long context is unexplored" — false.**
   Gist (Mu 2023), AutoCompressor (Chevalier 2023), ICAE (Ge 2024),
   MemoryPrompt (Pannitto 2024), TokMem (2026) all do variants.
   Our angle has to be **(a) the bit-capacity wall as a phenomenon
   we measure, and (b) the OPD/RL training recipe that lifts that
   wall**.

---

## Recommended paper positioning (one paragraph)

> We introduce a parameter-efficient memory wrapper that compresses
> arbitrary-length chunked input into $K$ soft tokens on a
> **strictly frozen** base LLM. Unlike prior context-distillation
> approaches that internalize information into model weights
> (OPCD, DCD, CD-as-Latent-Memory) or per-document LoRA modules,
> our wrapper is a single recurrent module that handles any chunk
> stream at inference time. Unlike concurrent RL-on-latent-memory
> work in multi-agent (LatentMem-LMPO) and continuous-thought
> reasoning (Latent-RL on Coconut), we target single-base
> long-context compression and train the wrapper with the
> three-stage recipe now standard in industrial LLM distillation
> (SFT → OPD → REINFORCE), adapted to a recurrence whose
> supervision flows through a frozen base. We then characterize the
> empirical **bit-capacity wall** of $K$-bounded soft memory under
> increasing answer entropy — an observation that is, to our
> knowledge, undocumented in the literature.

---

## What to add to the BibTeX

The following entries need to land in
`latent-mem-paper/iclr2026_conference.bib` for the related-work
section to cite cleanly. Pulled from this survey:

| key | citation |
|---|---|
| `opcd2026` | Ye et al., *On-Policy Context Distillation for Language Models*, arXiv:2602.12275, 2026 |
| `pi-distill2026` | *Privileged Information Distillation for Language Models* (π-Distill / OPSD), arXiv:2602.04942, 2026 |
| `op-prefix-distill2026` | Zhang et al., *Fast and Effective On-Policy Distillation from Reasoning Prefixes*, arXiv:2602.15260, 2026 (Optum AI) |
| `opd-survey2026` | *A Survey of On-Policy Distillation for LLMs*, arXiv:2604.00626v3, 2026 |
| `latentmem-lmpo2026` | Wang et al., *LatentMem: Customizing Latent Memory for Multi-Agent Systems*, arXiv:2602.03036, 2026 |
| `memrl2026` | Wang et al., *MemRL: Self-Evolving Agents via Runtime RL on Episodic Memory*, arXiv:2601.03192, 2026 |
| `memory-r12025` | Yan et al., *Memory-R1*, arXiv:2508.19828, 2025 |
| `mem1neurips2025` | Zhou et al., *MEM1: Memory-Efficient Reasoning Agents via RL with Consolidated Memory*, NeurIPS 2025 |
| `latent-rl-coconut2025` | *Reinforcement Learning for Latent-Space Thinking in LLMs*, arXiv:2512.11816, 2025 |
| `coconut2024` | Hao et al., *Training LLMs to Reason in a Continuous Latent Space (Coconut)*, arXiv:2412.06769, 2024 |
| `smd2026` | *Shadow Mask Distillation*, arXiv:2605.06850, 2026 |
| `rlkv2025` | *RLKV: RL-Guided KV Cache Compression*, arXiv:2510.08525, 2025 |
| `kvp2026` | *Learning to Evict from Key-Value Cache (KVP)*, arXiv:2602.10238, 2026 |
| `cd-latent-mem-mgmt2026` | *Context Distillation as Latent Memory Management*, arXiv:2605.28889, 2026 |
| `dcd2025` | Caccia et al., *Training Plug-and-Play Knowledge Modules with Deep Context Distillation*, arXiv:2503.08727, 2025 |
| `memo2026` | *MEMO: A Modular Framework for Training a Dedicated Memory Model*, 2026 |
| `memoryprompt2024` | Pannitto et al., *MemoryPrompt: A Light Wrapper to Improve Context Tracking*, LREC 2024 |
| `tokmem2026` | *TokMem: Tokenized Procedural Memory for LLMs*, ICLR 2026 |
| `acon2025` | *Acon: Optimizing Context Compression for Long-Horizon LLM Agents*, arXiv:2510.00615, 2025 |
| `ziprl2026` | *ZipRL: Adaptive Multi-Turn Context Compression with Hindsight Response Replay*, arXiv:2605.28069, 2026 |
| `gck-modules2026` | *Training Persistent Memory for Frozen Encoder-Decoder LLMs*, arXiv:2603.16413, 2026 |
| `mopd-mimov2-2026` | *MiMo-V2-Flash with MOPD*, arXiv:2601.02780, 2026 |
| `glm5-2026` | *GLM-5: On-Policy Cross-Stage Distillation*, arXiv:2602.15763, 2026 |
| `agarwal-opd-2024` | Agarwal et al., *On-Policy Distillation of Language Models*, ICLR 2024 |
| `minillm-2024` | Gu et al., *MiniLLM: Knowledge Distillation of Large Language Models*, ICLR 2024 |

---

## Action items (queue for after Phase P)

1. ✅ This survey doc itself.
2. ⏳ Cite OPCD and LatentMem in the paper intro + related-work
   immediately (both are pre-arXiv-2026 and impossible to miss).
3. ⏳ Add value-head supervision for m_T (B1) as a candidate for
   Phase Q.
4. ⏳ Plan an OPCD-as-baseline experiment (B2) for the §4 ablation.
5. ⏳ Update `v2-related-work.md` to incorporate the v2-adjacent
   findings (LatentMem, MEM1, MemoryPrompt persistence).
6. ⏳ Append the BibTeX entries to
   `latent-mem-paper/iclr2026_conference.bib`.

---

## One-sentence summary for the parent project status

> Literature 2024–mid-2026: nobody combines OPD + recurrent latent
> memory + frozen base; closest neighbors are OPCD (weights, not
> wrapper), LatentMem-LMPO (multi-agent, not single-base), and
> MemoryPrompt (same architecture but SFT only). Our differentiation
> is real but narrow; we have to be careful in the related-work
> section.
