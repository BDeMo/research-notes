# Paper B — Framing brainstorm (agent-oriented, "do-no-harm", popular)

> Goal: pivot the headline from narrow *catastrophic forgetting* to a **popular application
> scenario** an agent reviewer cares about, while staying **honest** about what we can prove.
> Companion to [`README.md`](README.md) / [`outline.md`](outline.md).

## 0. Why "catastrophic forgetting" alone is too narrow
- CF is a crowded, incremental-feeling sub-area; reviewers see many CF papers.
- Our **read-side isn't even weight-forgetting** — it's *inference-time negative transfer* (an
  irrelevant augmentation makes the model worse). CF-only framing both undersells and mislabels.
- ✅ Keep CF as **motivation/measured-problem** (the §7c SFT-degradation result is a great hook),
  not the headline.

## 1. The agent failure mode we actually fix (the hook)
Agents bolt augmentations onto a base model at inference time — retrieved context, a memory
module, a tool's output. The failure mode: **the agent over-trusts an augmentation that is
irrelevant / out of its competence, and does *worse* than the bare model.** We measure exactly
this (negative transfer §8; SFT-forgetting §7c).

**Our asset, in agent words:** a *pluggable, inference-time memory module that carries a built-in,
model-agnostic "should I fire here?" gate* and **falls back to the frozen base when not** —
**do-no-harm by construction** (gate→0 ⇒ exact base output).

## 2. Candidate framings (popular) + honest reviewer verdict

### F1 — "Know when to use it": competence-aware / self-gating augmentation  ★ RECOMMENDED
- **Pitch:** an augmentation that knows its competence boundary and abstains (falls back to base)
  outside it — preventing the confident-but-wrong failure of irrelevant context/memory.
- **Why popular:** abstention / calibration / "knowing what you don't know" + adaptive-RAG +
  agent reliability — all hot.
- **Evidence in hand:** competence boundary §8 (the module *has* a sharp boundary), gate §7/§7b,
  cross-model §2, the forgetting/negative-transfer motivation §7c/§8.
- **Closest prior + our delta:** TARG / Self-RAG / CRAG gate *retrieval* by output logits; ours
  gates a **learned memory module** by intrinsic signals, **do-no-harm by construction**, and
  **transfers across model families with zero tuning**.
- **Reviewer verdict:** buys the motivation; **demands** TARG/Self-RAG baselines (✅ started) and an
  agent/realistic eval. Risk: "incremental over adaptive-RAG" → mitigate via the *module* + the
  *by-construction* property + *cross-model* transfer.

### F2 — Do-no-harm tool use for agents (composable skills that self-abstain)
- **Pitch:** every agent tool/skill should self-report competence; we give a memory-skill a
  built-in, transferable competence gate so adding it never hurts the base agent.
- **Popular:** tool-use, modular/plug-and-play skills.
- **Reviewer verdict:** appealing, but "tool gating" reviewers expect **multi-tool routing + an
  agent benchmark** we don't have yet → the framing would write a check the experiments don't cash.
  Use as a secondary angle unless we add an agentic eval.

### F3 — Inference-time / test-time adaptation with a reliability gate
- **Pitch:** the wrapper = amortized inference-time context adaptation; the gate decides when the
  adaptation is trustworthy.
- **Popular:** TTT / test-time adaptation is very hot.
- **⚠️ Honesty caveat:** our wrapper is **pre-trained** (not gradient TTT at inference). Claiming
  "test-time *training*" would be inaccurate and **rejected by TTT-literate reviewers**. Only safe
  as "inference-time context compression," which is less novel. **High over-claim risk.**

### F4 — Portable reliability controller (deploy any augmentation on any base)
- Deployment/robustness angle (cross-model, zero-tuning). Strong as a **section/pillar**, weak as a
  *popular* headline. Keep as the robustness contribution.

### Other popular angles (one-liners)
- **Continual / lifelong agents** — keep adding memory without degrading (agent-flavored CF).
- **Personalization without forgetting** — per-user memory module that never hurts the base.
- **Reliable RAG** — "retrieval that abstains when the passage is bad" (very searchable).

## 3. The honest constraint from today's baseline run (`raw/baselines-2026-06-05/gate_read_baselines.csv`)
Head-to-head read-side gates, 3 families (Qwen3-8B, GLM-4-9B, Qwen3.5-MoE), 7.2k items, honest CV:

| gate family | lead signal | pooled AUROC(help) | **LOMO (cross-model)** |
|---|---|--:|--:|
| **TARG** (base own-uncertainty) | `margin_0` | **0.624** | **0.604** |
| output-confidence (Self-RAG-ish) | `conf_w` | 0.587 | 0.586 |
| **ours** (memory-write dynamics) | `delta_last` | 0.584 | 0.557 |

**On these 3 families, a trivial TARG base-uncertainty gate ≥ our signal** (the matrix's
`delta_last` LOMO 0.71 came from the 7-family grid, strongest on Phi/Mistral). **Implication for
framing:** the paper **cannot** lead with *"our signal is the best gate."* The defensible
contributions are:
1. the **application/framing** (do-no-harm augmentation for agents),
2. **do-no-harm by construction** for a *learned memory module* (a property retrieval-gating lacks),
3. **cross-model robustness** of intrinsic gating *in general* (and possibly **ours + TARG combined**),
position our signal honestly as *one* intrinsic gate, complementary to TARG — not a new SOTA scalar.

This is a feature, not a bug: a paper about a **use-case + a guaranteed property** is harder to
refute than "my AUROC > yours."

## 4. Recommended headline (pick a title)
> *Do No Harm: a pluggable, inference-time memory module for frozen LLMs that **knows when not to
> fire** — an intrinsic, model-agnostic gate makes augmentation do-no-harm, recovering most negative
> transfer with zero per-model tuning.*

Title options: **"Know When to Use It"** · **"Do No Harm: Competence-Aware Augmentation for Frozen LLMs"** · **"Memory That Knows Its Limits."**

## 5. What this implies for experiments (so the framing isn't vaporware)
- **MUST — 7-family head-to-head** ours vs TARG vs output-conf vs **ours+TARG combined** (settle §3 honestly; Phi/Mistral are where ours was strongest). ← decisive next read-side run.
- **MUST — a realistic "augmentation-sometimes-irrelevant" eval** (mixed relevant/irrelevant context) to earn the agent framing; an agentic/tool eval to earn F2.
- **SHOULD — online gate** (apply-or-skip live in generation, not post-hoc routing).
- **Novelty bar (honest):** with TARG competitive, the contribution = framing + by-construction
  do-no-harm + cross-model robustness (and/or signal **complementarity**), not a SOTA signal.

## 6. Bottom line — will reviewers buy it?
- **CF-only:** limited (agreed).
- **F1 do-no-harm augmentation, with TARG/Self-RAG baselines + cross-model + an agent/relevance
  eval:** plausible **workshop accept**; **top-venue** needs the agent eval **and** a clean win —
  where the "win" may be *do-no-harm-by-construction for learned modules* or *intrinsic-gate
  complementarity/portability*, not raw AUROC. Freshest, least-contested claims = **cross-model
  zero-tuning transfer** + **learned-module gating** (TARG is retrieval-only). Biggest risk =
  "incremental over adaptive-RAG" → beat it with the module + property + agent use-case.
