# Paper B — Scope + Compression/Gate/Fallback Lit Review (2026-06-06)

> Answers the questions: (a) does "compress input → a few latent embeddings for inference speedup,
> with a gate, and fallback to full context" exist? (b) what scope (agentic AI / memory-compression
> tool)? Decides a **popular, defensible scope** and the honest novelty line. Pairs with
> [`baselines-and-novelty.md`](baselines-and-novelty.md).

## 1. Your question, answered precisely
**"Compress input into a few latent embeddings for inference acceleration"** — **YES, established line:**
- **Gist** (2023), **ICAE** (2024), **AutoCompressor**, **CCM** (2312.03414), **500xCompressor**, **PCC**,
  **Cartridges/self-study** (2506.06266, ICLR'26), **ACON** (2510.00615, context compression for long-horizon agents).
- **Effect:** Cartridges **matches in-context learning at 38.6× less KV memory, 26× throughput**; Gist ~26× prompt compression. **But lossy on exact recall** (the "Silver Bullet" study, 2412.17483) — = our capacity wall (RULER→0).

**"Did they use a gate?"** — the *latent* compressors are **always-on** (no gate). Gating-with-fallback **does exist, but in two adjacent places, not for learned latent input memory:**
- **Text/prompt level (engineering):** **TAAC** (2602.15843, "quality-gated" compression, stops/relaxes when predicted quality drops), **ContextPilot**, **token-compressor** — compress the *text*, validate (embedding cosine / predicted score), **fall back to the raw prompt** if it fails.
- **Reasoning level:** **SLT** (2605.25745) & **SeLaR** (2604.08299) — a **confidence/entropy gate** compresses reliable reasoning spans into latents and **falls back to explicit CoT** when uncertain.

**"Fallback to full context when the gate closes?"** — present at the **text** level (TAAC/ContextPilot → raw prompt) and **reasoning** level (SLT → explicit CoT). **For a *learned latent* input memory on a frozen base, this is a gap.**

**⇒ Is the selling point reasonable?** **Partly — be honest.** The *gate+fallback idea* is **not novel** (TAAC text-level; SLT/SeLaR reasoning-level). What's open: applying **selective compression + fallback-to-full-context to a *learned latent* (parametric) memory**, governed by a **cross-model intrinsic signal**, with **do-no-harm by construction**. Pitch the **combination + the signal + the systematic treatment**, *not* the mechanism.

## 2. Scope (decided): **adaptive latent memory for LLM agents**
The 2026 "agentic memory" surveys (2512.13564; 2602.06052; ICLR'26 **MemAgents** workshop) taxonomize memory by **form: token-level · parametric · latent**. Our module = a **trained (parametric) compressor that emits latent memory tokens** on a **frozen base** → sits at the **parametric⊕latent** intersection.

**Scope statement (popular + tight):**
> *Adaptive latent memory for LLM agents — a detachable, model-agnostic context compressor on a
> frozen base that is **fast** (few latent tokens), **safe** (a cross-model do-no-harm gate), and
> **accurate when needed** (falls back to full context when the memory is insufficient).*

- **Community / venue:** **agentic memory** (ICLR'26 MemAgents workshop is an ideal fit) — *not* a generic "prompt-compression tool" (crowded engineering space: TAAC/ContextPilot). The *agent* framing + *cross-model do-no-harm* + *frozen-base detachable* is the differentiator.
- This **refines** (doesn't replace) the do-no-harm framing: do-no-harm is the *safety property*; adaptive compression+fallback is the *efficiency mechanism*; agentic memory is the *application*.

## 3. The selling-point experiment (to make it real) — **3-way adaptive gate**
Implement + run a gate that routes **per input** among:
`compressed latent memory (fast)` ↔ `full context (accurate, slow)` ↔ `no context (base)`.
- **Metric:** accuracy vs **token/compute cost** frontier (the inference-acceleration story) + do-no-harm (≥ base).
- **Baselines:** always-memory (Cartridges-style) · always-full-context (ICL ceiling) · TAAC-style text gate · oracle 3-way.
- **Claim if it works:** "recover most of full-context accuracy at a fraction of the tokens, *and* never underperform the base" — the efficiency+safety pitch, honestly bracketed.
- Status: **needs code** (`gate3_eval.py`); queued after the running baselines. *(This is the experiment that earns the compression/fallback selling point.)*

## 4. Honest novelty defense (compression angle)
| reviewer objection | response |
|---|---|
| "Latent context compression is solved (Gist/Cartridges)" | yes — we **don't** claim the compressor; we add **when to fire it + fallback** for a *learned latent* memory, cross-model. |
| "TAAC/ContextPilot already gate compression with fallback" | those are **text-level** middleware (compress text → validate → raw fallback); ours is a **learned latent** memory on a **frozen base** with an **intrinsic, model-agnostic** signal — and a *systematic* do-no-harm study (boundary, by-construction). |
| "SLT/SeLaR already do gated latent + fallback" | those compress **reasoning/CoT**; we compress **input context** (agentic memory), different regime + the capacity-wall analysis. |
| "Why not just use full context?" | the **cost frontier** (§3) + KV/throughput; full context is the *ceiling we fall back to*. |

**Bottom line:** scope = **adaptive latent agentic memory**; novelty = **learned-latent + cross-model do-no-harm gate + fallback + the systematic boundary/by-construction treatment** — *not* the compress-or-gate mechanisms (cite Cartridges, TAAC, SLT honestly).
