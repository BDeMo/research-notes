# v1.5 Method — Gated, Do-No-Harm Latent Memory on a Frozen Base

> **Canonical method description for Paper B / mem-X v1.5.** Maintained doc: architecture + all combine
> modes (math), the do-no-harm gate, the two-track design, training, signals, settings, and the *logic*
> (why each piece exists). Pairs with [`logic.md`](logic.md) (decisions), [`summary-matrix.md`](summary-matrix.md)
> (evidence), [`related-work.md`](related-work.md). Code: `mem-test/mem-embedding/src/mem_embedding/wrapper.py`.
>
> **2026-06-10 reframe (under review):** §4, §5, §7 below are superseded by the **Gated Compressor Module**
> design in [`v1.7-gated-compressor-module-2026-06-10.md`](v1.7-gated-compressor-module-2026-06-10.md): the
> gate becomes intrinsic to OUR compressor (one module emits memory + a calibrated reliability score from its
> own internals), baselines stay vanilla, and we compare at the compressor level (drop "compressor as a
> setting"). Read that doc first; the sections here remain as the v1.5 record.

## 0. What v1.5 is (one paragraph)
A **recurrent soft-prompt memory wrapper** compresses a long context into **K latent tokens** on a
**frozen** base LLM, then a **do-no-harm gate** — an input-conditioned signal — decides *per input* whether
to apply that memory or **fall back** (to the bare base, or to full context). The base weights are never
touched; removing/closing the wrapper returns the exact base output. Goal: **add competence in-domain
without ever degrading the base out-of-domain**, portably across model families.

## 1. Notation
Chunks $c_1,\dots,c_N$; encoder + projection gives $h_t = W_{\text{in}}\,\mathrm{enc}(c_t)\in\mathbb{R}^{L\times d}$.
Memory $m\in\mathbb{R}^{K\times d}$ ($K{=}64$ slots, $d{=}4096$). $E_q$ = query token embeddings.
$\mathrm{MHA}(Q,K,V)$ = multi-head attention; $\mathrm{LN}$ = LayerNorm; $\alpha_\cdot$ = learnable scalars.

## 2. Encoder + recurrence (how memory $m_N$ is produced)
The base itself encodes each chunk (`BaseAsEncoderAdapter`) → $h_t$. Memory is built by an **additive
Perceiver-IO recurrence** (memory = queries, chunk tokens = keys/values), one chunk at a time:

$$m' = m + \mathrm{MHA}\big(\mathrm{LN}(m),\mathrm{LN}(h_t),\mathrm{LN}(h_t)\big),\qquad \Phi(m,h_t)=m'+\mathrm{FFN}(\mathrm{LN}(m'))$$

$$\Delta m_t = \mathrm{LN}\big(\Phi(m_{t-1},h_t)-m_{t-1}\big),\qquad m_t = m_{t-1}+\alpha_u\,\Delta m_t,\quad m_0=M_{\text{init}}$$

$M_{\text{init}}$ is learned. $\lVert\Delta m_t\rVert$ at the last step = the gate signal **`delta_last`** (the wrapper's "write magnitude").

## 3. Combine modes (how $m_N$ reaches the frozen base) — all six
Let $\mathcal{R}(e,m)=\mathrm{LN}\big(\mathrm{MHA}(\mathrm{LN}(e),\mathrm{LN}(m),\mathrm{LN}(m))\big)$ be a per-token read.

| mode | delivery | seq length |
|---|---|---|
| **prefix** | $x'=[\,m_N\,;E_q\,]$ (prepend K tokens) | $+K$ |
| **xattn** (default) | readout then prefix: $\tilde m=m_N+\alpha_r\,\mathrm{LN}(\mathrm{MHA}(m_N,H,H))$, $H=[h_1;\dots;h_N]$; $x'=[\,\tilde m\,;E_q\,]$ | $+K$ |
| **interleave** | insert $m_N$ at position $p=\mathrm{round}(T\cdot\rho)$ inside $E_q$ | $+K$ |
| **residual** | per token: $e'_t=e_t+\alpha\,\mathcal{R}(e_t,m_N)$; no prefix | unchanged |
| **hybrid** | residual **+** prefix: $x'=[\,m_N\,;\,e'\,]$ with $e'_t=e_t+\alpha\,\mathcal{R}(e_t,m_N)$ | $+K$ |
| **gated** (do-no-harm) | $e'_t=e_t+g\cdot\alpha\,\mathcal{R}(e_t,m_N)$, $g\in(0,1)$ | unchanged |

`xattn` = recurrence + one apply-time cross-attention over the **full chunk pool** + prefix (gives the
memory a second pass at detail the recurrence dropped). **Finding (structure sweep):** best combine is
**model/bench-dependent** — `hybrid`/`residual` beat `xattn` on Q8-trivia, xattn needs `h4`; report combine
as a tuned hyperparameter, not a universal default.

## 4. The do-no-harm gate (v1.5 core)
**(a) Gate signal `g` — `_GateHead`.** An MLP on **memory-write features** (cross-model-general, §2 study):
mean-pooled memory $\bar m$, mean/max slot-norm, drift $\lVert m_N-m_0\rVert$, and $\lVert\Delta m_{\text{last}}\rVert$
(optionally + query-relevance $\cos(\bar q,\bar m)$ for the query-conditioned variant):

$$g=\sigma\!\big(\mathrm{MLP}([\,\bar m,\ \overline{\lVert s\rVert},\ \max\lVert s\rVert,\ \mathrm{drift},\ \lVert\Delta m_{\text{last}}\rVert\,])\big)\in(0,1)$$

**(b) Gated combine — do-no-harm *by construction*:**

$$e'_t = e_t + g\cdot\alpha\,\mathcal{R}(e_t,m_N)$$

$g\to 0$ ⇒ $e'_t=e_t$ ⇒ **exact frozen base** (provably no harm); $g\to 1$ ⇒ full wrapper.

**(c) Deployed gate = offline *routing* (the working form).** The end-to-end soft gate (b) **failed** (gate
sticks ~0.5, per-token residual corrupts generation — kept as ablation H4). The **working** gate is a
**routing** classifier: a logistic model on the intrinsic signal predicts $P(\text{useful})$ per input and
**routes** between two outputs (no encode-into-weights, honest 5-fold CV):

$$\text{output}(x)=\begin{cases}\text{wrapper}(x) & \text{if } P_\theta(\text{useful}\mid \phi(x))\ge\tau\\ \text{fallback}(x) & \text{otherwise}\end{cases}$$

where $\phi(x)$ = the cheap signal(s) (`delta_last` and/or base-uncertainty `margin_0`).

## 5. Two tracks (one signal, two fallbacks)
- **Track A — `×no-ctx` (do-no-harm / selective prediction):** fallback = bare base. Target $\text{useful}{:=}(n_w>n_0)$. Metric: do-no-harm (gated $\ge$ base) + accuracy (risk–coverage / AURC).
- **Track B — `×full-ctx` (compression / inference-accel):** fallback = re-read full context. Target $\text{suffices}{:=}(n_w\ge n_{\text{full}}-\varepsilon)$; gate uses **cheap base-uncertainty only** (no encode) for honest cost. Metric: accuracy **at token cost** (cost–coverage). Cost (amortized agentic-memory setting) $\approx K{+}|q|$ tokens when routed to memory vs full-context tokens on fallback.

$n_0,n_w,n_{\text{full}}$ = native score with no-ctx / wrapper-memory / full-context. **Result:** Track B
recovers ~99% of full-context accuracy at **38% tokens** (trivia), beats full at **8%** (narrativeqa),
defers to full where compression is insufficient.

## 6. Training
**Wrapper** (BPTT through the frozen base; only wrapper params train): per item, teacher-forced **CE** on
the gold answer + **KL** to the no-context base on the answer span (do-no-harm regulariser) + a memory
**diversity** term; cat-NIAH adds an answer-head aux loss. *Setting:* lr 3e-5, $\lambda_{kl}{=}1$,
$\lambda_{div}{=}0.1$, 1800 steps, K=64, xattn. **Gate (soft, ablation):** real items (gate-open target) +
**decoy** items (foreign context → gate-close target) + the do-no-harm KL on decoys. **Gate (routing,
deployed):** logistic on logged signals, 5-fold CV, threshold $\tau$ picked on train folds.

## 7. Intrinsic signals (what the gate reads)
**General (cross-model, kept):** `delta_last` (write magnitude) + memory-geometry (slot-norm mean/max,
drift, `delta_mean/trend`) + `mem_influence_span` (causal). **Base-uncertainty (TARG-style, cheap, Track-B):**
`margin_0` (top-1−top-2), `conf_0`, `entropy_0` from a no-context draft. **Model-specific (dropped):**
logit-lens confidence/margin (≈chance off-Qwen). §2/§2b/§7d in the matrix.

## 8. Settings + best config (where it works)
Canonical: K=64, n_heads=8, ffn_mult=2, **n_layers=1** (single recurrence block; "multi-layer" = the deep-
injection ablation, which collapses), combine=xattn, 1800 steps, frozen base. *Structure sweep (Q8-trivia):*
`hybrid/K64` best (mem 0.551, +0.092; Track B ≈full at 24% tokens), `residual/K128` beats full at 31%;
default `xattn,K64,h8` underperforms (0.420) → xattn needs `h4`. K optimal ∈ 16–64, non-monotonic. Recipe:
`settings.md` P08-S6/S7/S8.

## 9. Scope + honest limits
- **In-domain only:** helps when test matches the training distribution (QA +2..+9pt); gain vanishes at the same-task line; harmful cross-domain → the gate closes (§8 boundary).
- **Cross-model heterogeneous:** helps Qwen3-8B/14B/GLM, mixed Qwen2.5, neutral Phi, **HURTS Mistral** (−0.488 trivia) → the gate is *necessary*, not optional.
- **Capacity wall:** can't hold exact detail — RULER-NIAH→0, extractive ≪ full-context (→ ceded to Paper A).
- **Gate is offline/routing** (not online); do-no-harm is "recovers most" (~85–95%), strict only by construction (g→0).

## 10. The logic (why each piece, in order)
frozen-base premise → **negative transfer** (v1 wrapper hurts OOD) → **do-no-harm gate** idea → need a
**signal** (§2: `delta_last` general; TARG competitive) + a **boundary** (§8: in-dist only) → the **gate**
(§7: routing recovers most harm, transfers; soft-gate & multi-depth fail → hard routing) → **forgetting
contrast** (§7c: SFT forgets, frozen module doesn't) → **cross-model** main table (Mistral surprise ⇒ gate
necessary) → **two-track reframe** (A do-no-harm / B compression) → **fair baselines** (Cartridge/Gist same
tracks) → **ablations** (K, combine/structure, multi-layer, significance) → **scope** = adaptive latent
agentic memory, framed via selective prediction.

## 11. TODO / to add
- **Relevance/decoy eval** — feed irrelevant context, confirm gate closes (earns the agent framing).
- **Online gate** — apply-or-skip live in generation (current is post-hoc routing).
- **Baseline Track-B recompute** vs shared full-ctx.
- **Capacity-wall curve** (K∈{4,256}, RULER lengths) for the Paper-A boundary.
- **Cross-model best-config** (combine sweep on GLM/Q14/Q2.5/Mistral — running).
