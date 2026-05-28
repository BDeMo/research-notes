# 2026-05-28 — Methods reading + new plan brainstorm

> Session goal: deep-read 4 methods most relevant to plan 08 v0 (learned memory wrapper) and brainstorm next plans. Two days after session 2026-05-26.

## State refresh (since last session)

- Plan 08 spawned a **v0 "Learned Memory Wrapper"** sub-project: frozen 8B base + trainable wrapper that maintains a compact memory state `m_t = G_φ(m_{t-1}, c_t)`. The full self-modifying-LLM remains the north star; v0 proves the smaller claim that a wrapper can compress long context into an updateable memory state.
- v0 default is **explicit memory tokens** (Gisting-style). Hidden-state features and LoRA/weight deltas are v0.1 and v1.
- Empirical motivation comes from RCA-demo findings: SFT teaches RCA behavior but is fragile (catastrophic forgetting, format brittleness, evidence vs parseability gap). Hence "freeze base + train wrapper" for v0.
- 30+ plan08 commits since 2026-05-26 (mostly slides + RCA demo Overleaf). No new matrix entries logged.

## Activities

- Reread plan 08 / v0 docs, slides README, budget.
- Web-fetched + read 4 methods relevant to v0's "memory wrapper architecture" decision:
  - [cartridges] **Cartridges** — self-study CD into trained KV cache (long-context).
  - **Activation Beacon** (Zhang et al., 2024) — chunked progressive KV compression via beacon tokens. **New**, promoted to `[act-beacon]`.
  - [gisting] **Gisting** — gist tokens learned via attention masking; 26x prompt compression.
  - [genadapter] **Generative Adapter** — bi-linear hypernet generating LoRA per chunk, with recurrent state (constant memory).
- Brainstormed candidate next plans (3 from existing ★★+ ideas + 3 new J-series).

## Reading notes

### Cartridges [cartridges] — Eyuboglu et al. (2025)

**Claim**: For a large corpus that's repeatedly queried, training a *small KV cache offline* via self-study beats full ICL on cost, ties on quality.

**Mechanism**:
- Train a "Cartridge" (a small KV cache) per corpus.
- Naive NTP on the corpus does *not* match ICL — that's the negative result.
- "Self-study" = synthesize Q&A about the corpus with a teacher model, then train the Cartridge with **context distillation** (KL between teacher-with-context and student-with-Cartridge).
- At inference: load Cartridge as KV cache, decode.

**Key result**:
- Matches ICL accuracy with **38.6× less memory**, **26.4× throughput**.
- Extends effective context from 128K → 484K tokens on MTOB.
- Surprising: Cartridges compose at inference without retraining.

**Gap relative to v0**:
- Cartridges train *per corpus*, not a *learned wrapper* that generalizes. Closer to plan 01 mining than to v0.
- v0 wants amortization across corpora; Cartridges amortizes across *queries on one corpus*.

**Relevance**:
- The **self-study** recipe (teacher Q&A generation + KL distillation) is directly useful for v0's "full-context teacher → compressed-memory student" training objective. v0 should literally adopt the self-study Q&A generation pipeline.
- The **composability** observation says corpus-specific compressed states *add*. Suggests v0's `m_t` updates may compose well across chunks even without explicit gating.

---

### Activation Beacon [act-beacon] — Zhang et al. (2024) — arXiv:2401.03462

**Claim**: Soft-token compression (Gisting / ICAE / AutoCompressor) is a **bottleneck** for long context. Compress into KV activations at every layer instead.

**Mechanism**:
- Add a special `<b>` "beacon" token to vocab.
- Partition context into chunks (size `w`, e.g. 1024). Each chunk further split into units of size α (compression ratio). Interleave one `<b>` after each unit.
- LLM encodes one chunk at a time. After encoding, **discard raw-token activations, keep `<b>` activations**. These accumulate and serve as the context proxy for the next chunk.
- Training: compression-based auto-regression on plain text. **Sample random compression ratio per chunk** — model learns to handle arbitrary compression at inference.

**Key result**:
- 8× KV cache reduction, 2× inference acceleration.
- Llama-2 / Qwen-2 backbones. 1B-token training corpus + 30K fine-tune samples.
- Handles 128K-token inputs trained at max 20K. Soft-prompt methods can't.

**Gap relative to v0**:
- v0 default = soft prompt tokens (`inputs_embeds`). Activation Beacon directly argues this is the wrong choice for long context.
- Activation Beacon keeps the base model touched (attention modified), not frozen. v0 wants frozen base.

**Relevance**:
- **Strong evidence v0 should add a KV-activation-memory ablation early**, not defer it to v0.1.
- The "random compression ratio at training" trick → directly transferable to v0's memory-length sweep (16/32/64/128 tokens).
- The "discard raw activations, keep beacon activations" workflow is essentially what v0 wants for `m_t = G_φ(m_{t-1}, c_t)` — except v0 plans soft tokens. Two architectures to compare in v0's Phase-1.

---

### Gisting [gisting] — Mu, Li, Goodman (NeurIPS 2024)

**Claim**: An LM can be its own gist predictor. Train it via *attention masking* — no separate hypernet needed.

**Mechanism**:
- Insert `k` special gist tokens `g_1, …, g_k` between prompt `t` and input `x`.
- Modify attention mask: post-gist tokens cannot attend to pre-gist tokens. Forces information into gist activations.
- Train with standard instruction tuning. **No extra training cost**.
- Equivalent to amortized context distillation across a task distribution (Eq. 2 in paper).

**Key result**:
- Up to **26× prompt compression** on Alpaca+, LLaMA-7B / FLAN-T5-XXL.
- 40% FLOPs reduction, 4.2% wall time, minimal quality loss (human eval).
- Gist tokens **cached and reused** across calls.

**Gap relative to v0**:
- Gisting compresses *short* prompts (≤ ~30 tokens). Not designed for long context.
- Single-shot compression (gist all at once), not chunked recurrent like v0.
- Doesn't extend context window — gist is *of* the prompt, not a memory of more context.

**Relevance**:
- The **"LM is its own G"** insight is structural: v0 doesn't strictly need a separate wrapper module. Could train base + attention mask jointly and avoid the architectural cost. But this contradicts v0's "freeze base" rule.
- Gisting's attention-mask trick is a **baseline v0 must beat**. If v0 wrapper doesn't outperform a Gisting-style mask-only approach (at matched parameter budget), the wrapper isn't earning its keep.
- The 26× compression ceiling is a *single-shot ceiling*. v0's chunked recurrent approach is designed to push past it.

---

### Generative Adapter [genadapter] — Chen et al. (ICLR 2025) — arXiv:2411.05877

**Claim**: A trained **bi-linear generator** can map a context chunk's hidden states to a **LoRA delta** in a single forward pass, with **constant memory** across stream length.

**Mechanism**:
- Frozen base. Trainable generator $\mathcal{G}^{(l)}$ per Transformer layer.
- Bi-linear formula:
  $W_\Delta = A_1 \mathrm{norm}(S_t) B_2$ where $S_t = S_{t-1} + A_2 H_t^\top H_t B_1$.
- $S_t \in \mathbb{R}^{d_r \times d_r}$ with $d_r \ll d_h$. **Recurrent state, constant in stream length.**
- **SVD normalization** for stability (singular values reset to 1 within a low-rank approximation, naturally producing a LoRA structure).
- Trained with two self-supervised tasks: **reconstruction** (predict input tokens given context-adapted model) + **completion** (NTP on continuation).
- 500M generator → 32M LoRA delta on Mistral-7B / Llama2-7B. 1B SlimPajama tokens, chunk size 1024.

**Key result**:
- StreamingQA: F1 19.5 (SFT) → **31.5 (genadapter)** at 32K context. +63.5% relative.
- MetaICL across 26 tasks: 44.9% avg accuracy, beats base.
- MSC personalization: matches full-conversation prompting at **4× lower compute / memory**.

**Gap relative to v0**:
- Genadapter is essentially **plan 08's north star already implemented**, *minus the verifier gating*. Updates are applied unconditionally.
- No mechanism for "model decides what to learn" — generator is a passive function of context.
- Effective at <8K but doesn't claim dominance at 32K+ (where prompting + retrieval still wins).

**Relevance**:
- **v0 should treat genadapter as the LoRA-side comparison point**, not Doc-to-LoRA alone. Genadapter is closer to v0's setup (no oracle context distillation needed).
- The **bi-linear $S_t$ recurrence** is more parameter-efficient than what v0 currently sketches (per-chunk hypernet call). Worth adopting even for the soft-token wrapper variant.
- **SVD normalization** is a free win for any v0 variant that touches weight space.
- Plan 08 north star = genadapter + a verifier $V(\cdot)$ that scores `(C_t, Δ_t)` and gates `α_t = V(...)`. The mechanism gap to plan 08 is now precisely identifiable.

---

## Cross-paper synthesis

| Method | Compression target | Stream support | Backprop at inference? | Best for |
|---|---|---|---|---|
| Gisting | gist tokens (input embeds) | no (single-shot) | no | short prompts (≤ 30 tokens) |
| Activation Beacon | KV activations | **yes** (chunked) | no | long context (up to 128K), high ratio |
| Cartridges | KV cache (per corpus, offline) | n/a (offline) | training only | repeated queries on a corpus |
| Generative Adapter | LoRA delta (per chunk) | **yes** (recurrent $S_t$) | no | streaming knowledge / personalization |
| v0 (current default) | memory tokens (input embeds) | yes (chunked) | no | TBD — needs to beat at least one above |

**Key takeaway for v0**: the soft-token default is the *least competitive* compression target on the long-context axis (Activation Beacon evidence). v0 should add at minimum one ablation against KV-activation memory (beacon-style) and one against LoRA-delta memory (genadapter-style). This sharpens the v0 paper's positioning beyond "yet another soft-prompt wrapper".

---

## Candidate next plans (for picking)

### From existing ★★+ ideas not yet planned

| # | Parent | Why now | Headline question |
|---|---|---|---|
| 02 | **I3** X-then-W curriculum (★★★, S=F) | Natural sequel to plan 01. Already formalized. The "iterative" version of plan 01's data curation. | Does iterating "saturate X → distill X-residual into W → reset X" outperform plain plan 01 within the same compute? |
| 04 | **F1** TTT serving infra (★★, ←#08) | Plan 08 deployment requires per-request weight delta serving. Existing systems (S-LoRA) target many *trained* LoRAs, not *runtime-generated* ones. | What's the serving cost of per-request runtime LoRA writes (genadapter-style) at batch size 64, p99 < 100ms? |
| 05 | **G2** Information-theoretic X+W budget (★, ≈#01) | Theory backbone listed in top picks. Quantify $I_\text{needed}$ vs $I(X) + I(W)$ to predict X-vs-W allocation. | For a task class, what's the minimum bits of W-update needed beyond a fixed X-budget? |

### New (J-series, X-W extension — to add to ideas/README.md)

| ID | Title | ★ | Headline |
|---|---|---|---|
| J1 | Wrapper → LoRA distillation | ★★★ | Train v0 memory wrapper first; then learn a hypernet that maps `m_t → Δ W_t` to bridge v0 to plan 08 north star. Two-stage path. |
| J2 | Verifier-gated Generative Adapter | ★★ | Genadapter applies $\Delta_t$ unconditionally; add a verifier $\alpha_t = V(C_t, \Delta_t)$. The simplest plan 08-north-star variant testable today. |
| J3 | Hybrid X+W chunk router | ★★ | Per chunk, predict whether to compress into memory tokens (X-side) or LoRA delta (W-side). Some chunks have facts (W-friendly), others style/dialogue (token-friendly). Operationalizes I6. |
| J4 | SVD-normalized memory wrapper (v0.1) | ★ | Adopt genadapter's SVD normalization in v0's wrapper architecture. Pure architectural improvement. |
| J5 | Beacon-style KV memory (v0.1) | ★★ | v0's soft-prompt default is the *weakest* long-context compression choice (Activation Beacon paper). Add a KV-activation memory variant in v0 Phase-1. |

## Output artifacts (this session)

- This entry: `docs/matrix/2026-05-28-methods-reading-and-new-plans.md`
- Updated `docs/matrix/knowledge-sources.md` — promote Activation Beacon to `[act-beacon]` ID; tighten genadapter / cartridges / gisting entries.
- Updated `notes/ideas/README.md` — add J1–J5 under new section `J. Architecture from method-reading (2026-05-28)`.
- Updated `known/`:
  - `inference-time-training/` — add cross-paper synthesis table reference.
  - `context-distillation/` — note v0 lineage.
  - `long-context/` — promote `[act-beacon]`.

## Knowledge sources to add / update

- **NEW** `[act-beacon]` — Activation Beacon (Zhang et al., 2024). arXiv:2401.03462. Already mentioned but not given stable ID.
- **Updated** `[genadapter]` — clarify bi-linear $S_t$ recurrence and SVD normalization.
- **Updated** `[cartridges]` — clarify "self-study" recipe (synthetic Q&A + KL).
- **Updated** `[gisting]` — clarify attention-mask trick; note 26× ceiling for short prompts.

## Next steps

1. **Pick candidate plans to draft in full** — user to choose from the table above. My recommendation:
   - **J5** (Beacon KV memory in v0.1) — cheapest, immediately useful for plan 08 v0 paper positioning.
   - **J1** (Wrapper → LoRA distillation) — strategic; bridges v0 → north star.
   - **02 / I3** (X-then-W curriculum) — high $\star$, already at `S=F`, natural extension of plan 01.
2. **For plan 08 v0**: add genadapter + Activation Beacon as baselines in `validation.md`. Update v0 Phase-1 to include KV-activation memory variant.
3. **Theory work** (G2 / J3) — defer until at least one of v0 / plan 01 has produced numbers.

## Open questions raised this session

- Genadapter's bi-linear $S_t = S_{t-1} + A_2 H_t^\top H_t B_1$ is a Hebbian-style associative memory. Is plan 08 north star fundamentally just "genadapter + verifier"? If yes, the contribution surface shrinks; if no, the gap is what we're really claiming.
- Cartridges trains *per corpus*, but composes at inference. Does v0's `m_t` compose across corpora? If yes, that's a unique angle vs. genadapter (which streams sequentially without compositionality claims).
- Why does soft-prompt compression hit a ceiling at ~26× (Gisting) while KV-activation compression hits ~38× (Cartridges) or higher? Is there a structural reason, or is it engineering?
