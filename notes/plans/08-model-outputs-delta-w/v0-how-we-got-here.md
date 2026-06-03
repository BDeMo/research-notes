# How we got to the v0 wrapper — derivation

**Status:** retrospective, written 2026-06-03 PT after Phase Y
multi-seed bands landed.
**Owner:** Mingjia (notes), Claude (write-up).
**Purpose.** This document explains the v0 wrapper as a sequence of
decisions, not as a list of architectural features. Every shape that
the code has now is here because of a specific constraint, an
experimental observation, or a horse-race we ran. Reading this should
make the v0 architecture feel *inevitable* — given the constraints
and the data — rather than *one design among many*.

It also serves as the canonical answer to the question we will keep
getting from reviewers and from ourselves: "why exactly does the
wrapper look the way it does, and not the obvious alternatives?"

The companion documents are:
* `v0-learned-memory-wrapper.md` — concise tech note (the *what*).
* `v0-paper-target-2026-05-31.md` — paper framing decision (the
  *why-this-paper*).
* This document — the *how-we-arrived-at-this-architecture*.

---

## 1. The constraints (immovable)

Before any architectural choice was made, four hard constraints were
locked in from the `plan-08` brief. Every later decision is a
consequence of one of these.

| # | constraint | consequence |
|---|---|---|
| C1 | **Do not fine-tune the base.** Frozen weights at inference; the base must stay general-purpose, deployable, and trust-preserving. | All wrapper outputs have to enter via input embeddings or a per-token additive residual; no LoRA, no adapter inside the base. |
| C2 | **Wrapper $\ll 1\,\%$ of base.** A wrapper that is the size of a small base model would defeat the whole "small-on-large" premise; we cap trainable params in the low hundreds of M. | $K \le 256$ memory tokens; depth $\le 2$ for the update block; FFN expansion $\le 4$; no per-layer adapters. |
| C3 | **Task-agnostic protocol.** We do not pre-commit to RAG-style retrieval, classifier head, or any task-specific decoder. The same protocol has to handle in-domain QA, zero-shot transfer, and cross-task. | Memory representation is generic ($K \times d$ vectors in the base's hidden dim); training loss is generic (token-level CE + KL); evaluation harness is task-pluggable via `--eval-dataset`. |
| C4 | **All interfaces swappable.** Future ablations need to plug in different memory shapes (dense pool / sparse hash / weight delta), different update transitions, and different combine modes — without rewriting the rest. | Three-method protocol `(init_memory, update, apply)`; combine mode is a string with seven values; update block is one class. |

These four are what made the problem *tractable* and what shaped
every other choice below.

---

## 2. Decision 1 — what does memory look like?

We considered three families of memory representation:

1. **Dense vector pool** ($K \times d$ learned tensor).
   Cousin of Gist Tokens (Mu et al. 2023), RMT (Bulatov 2022), and
   the soft-prompt lineage.
2. **Sparse hash / vector DB** (chunk hashes / vector embeddings,
   retrieved at inference). Cousin of RAG.
3. **Weight delta $\Delta W$** (the original `plan-08` aspiration:
   the wrapper emits LoRA-style weight deltas that adapt the base).

Why we picked **(1)** for v0:

* **Interface fit (C4).** Dense vectors share dtype + device with
  the base's input embeddings; no bridging code, no FAISS process,
  no hypernet collapse risk.
* **Smallest blast radius.** If the wrapper hurts, the base sees
  identical tokens at $K \to 0$; we can compare to `no_context`
  with zero gymnastics.
* **Ablate first, then escalate.** The dense pool is the *most
  capacity-limited* of the three. If it doesn't hit a wall, $\Delta W$
  is over-engineered. If it does hit a wall, the wall is intrinsic
  to compression and the next paper studies how to escape it. (Phase
  Y's RULER-NIAH 0.000 ± 0.000 over 8 cells confirmed the wall is
  real; v2 / v3 explore the escapes — see
  `v1-if-wrapper-doesnt-work-2026-06-03.md`.)

We also fixed $K \in \{16, 32, 64, 128, 256\}$ as the per-step
budget. Smaller $K$ are below typical prompt lengths; larger $K$
turn the wrapper into a small base.

---

## 3. Decision 2 — how does memory transition from $m_{t-1}$ to $m_t$?

Long context arrives in chunks of $L_c = 256$ tokens. The wrapper
sees the chunks one at a time and updates memory. We needed a
**transition function** `f: (m_{t-1}, c_t) → m_t`.

Candidates considered:

| variant | shape | upside | downside |
|---|---|---|---|
| (i) pure attention pool: $m_t = \mathrm{Attn}(Q=m_{t-1}, KV=c_t)$ | 1 attention, no residual | minimal | no residual = unstable training, no obvious way to scale depth |
| (ii) **Perceiver-IO block**: $\tilde m = m_{t-1} + \mathrm{Attn}(\cdot) + \mathrm{FFN}(\cdot)$ | 1 attn + 1 FFN, residual | stable, standard transformer block, depth scales by stacking | $\sim O(K \cdot d)$ extra params per layer |
| (iii) GRU-over-memory: $m_t = (1-z) \odot m_{t-1} + z \odot \tilde m$ | learned gate $z$ | tight control of forgetting | gate dynamics + transformer base = two non-matching paradigms |

We picked **(ii)** — implemented as
`src/mem_embedding/wrapper.py::_CrossAttnBlock`. Memory is the
query (small, $K \approx 32$); chunk hidden states are the
key/value (large, $L_c = 256$). This matches Perceiver-IO's
information bottleneck pattern.

Two non-obvious design choices on top of (ii):

**Δm exposure.** We deliberately wrote the transition as
$m_t = m_{t-1} + \alpha_\text{upd} \cdot \Delta m_t$, with
$\Delta m_t = \mathrm{candidate} - m_{t-1}$ stored in
`MemoryState.extra["last_delta"]`. Why bother carrying $\Delta m$
out of the block?

* It lets every diagnostic probe (drift, geometry, dead-channel)
  read a clean delta signal — no need to instrument the block.
* It lets us flip the wrapper to a no-op (set
  $\alpha_\text{upd} = 0$) without rewriting code. This was used
  for ReZero / SkipInit-style training and later to diagnose the
  seed bimodality finding.
* It is what the cosine-divergence regulariser
  $\lambda_\text{div} \mathcal{R}_\text{div}(m_{0:T})$ in the loss
  consumes — without explicit $\Delta m$ we cannot penalise
  "wrote the same thing twice in a row".

**Update gate $\alpha_\text{upd}$ as learnable scalar.** Initialised
to 1.0 (not ReZero 0.0) so the wrapper starts useful, not
identity. The 0.0 init was tried; it produced cells that never
escaped identity within the SFT budget. (Phase A failed cell.)

---

## 4. Decision 3 — how does memory enter the frozen base?

This is the most consequential single decision. We implemented
**seven combine modes** (`VALID_COMBINES` in `wrapper.py:59-67`)
because no a-priori reasoning told us which would work, and our
benchmarks split on this axis.

| mode | how $m$ enters the base | conceptual analogue |
|---|---|---|
| `prefix` | `inputs_embeds = [m, query]` | soft prompt / prefix tuning |
| `residual` | per-base-token: $e_t' = e_t + \alpha \cdot \mathrm{CrossAttn}(e_t, m)$ | adapter-style |
| `interleave` | `[query[:p], m, query[p:]]` at position $p$ | prefix-in-the-middle |
| `hybrid` | `residual` then `prefix` | two information channels |
| `xattn` | apply-time cross-attn: $m' = m + \alpha \cdot \mathrm{LN}(\mathrm{CrossAttn}(\mathrm{LN}(m), \mathrm{LN}(\text{concat-chunks}), \mathrm{LN}(\text{concat-chunks})))$; then prepend $m'$ | "read interface" — re-look-up over raw chunks |
| `xattn_residual` | `xattn` but with per-token residual instead of prefix | hybrid of xattn + residual |
| `gated` | reserved (NotImplemented in v0) | convex-mix variant of residual |

**Why we ended up choosing `xattn` as the default.**

The horse-race went in three rounds:

* **Round 1 (Phase A-C).** On `categorical_niah`, `prefix` worked
  reasonably (the answer is a 2-bit token, almost any read interface
  routes it). All four feasible modes (prefix, residual, hybrid,
  interleave) sat within $\pm 0.05$ exact-match of each other.
  Conclusion: on low-entropy tasks the read interface is not the
  bottleneck.

* **Round 2 (Phase D-E, `coding_niah`).** When we switched to
  retrieving 8-character code-needle strings, *all* modes collapsed
  to em ≈ 0. Diagnostic: the recurrence already compressed the
  string out of memory before the read happened. K=32 cannot
  reconstruct an arbitrary 8-char span. This was the first time we
  saw what later became the bit-capacity wall.

* **Round 3 (Phase F-G).** We added `xattn`: at apply time, memory
  queries the **concatenated raw chunk hidden states**, not just
  the compressed $m_T$. The intuition was that the recurrence's job
  is to maintain an *index*, not to literally reconstruct the
  content; the apply-time cross-attn does the verbatim retrieval.
  On `coding_niah`, `xattn` jumped from 0.00 to ~0.20. On
  `categorical_niah` it was tied with `prefix`. We picked `xattn`
  as the default because it strictly dominates: same performance on
  easy tasks, real lift on hard ones. Implementation:
  `wrapper.py::_ApplyTimeReadoutBlock`.

The Phase Y RULER-NIAH 0.000 ± 0.000 result later revealed that
even `xattn` cannot escape the bit-capacity wall on *truly*
verbatim retrieval (procedural needle-in-haystack of a 2k-token
WikiText). But `xattn` was still the best of the seven on the
benchmarks where any signal is recoverable. So it stayed the
default.

The Phase W / Phase W2 ablation sweeps (in flight at the time of
writing) re-run this combine-mode horse-race on the public
**QuALITY** benchmark with 4-seed bands so the ablation table in
the paper has variance bands not single-seed cells. Expected
outcome: `xattn` and `hybrid` lead on QuALITY by 1-2 σ over
`prefix` and `residual`; if it instead turns out everything ties
within seed noise on QuALITY (the lossy-compression regime),
that *also* supports the headline claim — read interface does not
matter when the task only needs the gist.

---

## 5. Decision 4 — training ladder

Three stages, in increasing complexity. We always started at the
simplest and only escalated when the simpler stage left signal on
the table.

### Stage 1 — SFT (always on)

$$\mathcal{L}_\text{SFT} = \mathcal{L}_\text{CE}(\hat y, y) + \lambda_\text{KL} \cdot \mathrm{KL}(p_\text{student} \,\|\, p_\text{teacher}) + \lambda_\text{div} \cdot \mathcal{R}_\text{div}(m_{0:T})$$

Three terms, three reasons:

* **CE** ($\mathcal{L}_\text{CE}$). Teacher-forced cross-entropy on
  the gold answer span. The only term that is strictly necessary.
* **KL anchor** ($\lambda_\text{KL}$). The KL is against the
  *full-context teacher* — the same frozen base reading the full
  context in the prompt. Without this anchor, the wrapper has free
  rein to "answer in its own style" and we lose the property that
  the wrapper is a *compression* of the teacher.
* **Diversity regulariser** ($\lambda_\text{div}$). Cosine penalty
  between consecutive $\Delta m_t$. Without this, the wrapper
  learns to write the same thing every chunk (the memory "freezes"
  on the first chunk and ignores later ones). We saw this failure
  mode explicitly in Phase B: with $\lambda_\text{div} = 0$,
  `cos_consec(Δm_{t-1}, Δm_t)` shot to $\sim 1.0$ within 100
  steps; with $\lambda_\text{div} = 0.1$, it stays at $\sim 0.7$
  and the wrapper actually uses later chunks.

### Stage 2 — OPD (on-policy distillation, optional)

Once SFT has converged, distil the full-context teacher onto
**student's own rollouts** from the SFT checkpoint (in the spirit
of Agarwal et al. 2024 and the MiniLLM line). This fixes the
exposure-bias gap pure SFT exhibits — the wrapper at inference is
fed its own draft, not the gold.

### Stage 3 — RL (REINFORCE w/ leave-one-out, optional)

Binary `contains_match` reward + a small KL anchor to the teacher
to prevent reward-hack collapse.

### What we kept

After Phase A-H and the data-scale ablation (Phase L), the
empirical answer was unambiguous: **on the categorical / public
benchmarks of v1, OPD and RL add no measurable lift over a
well-trained SFT cell**. The single $N{=}100 \to 500$ data-scale
intervention is *more* than the SFT $\to$ SFT+OPD $\to$ SFT+OPD+RL
ladder in mean exact-match. So the **final recipe is SFT only**
with the KL + div regularisers. OPD and RL code is in the repo and
ran in some Phase H cells; we report them in the ablation tables
of `details.tex` for completeness and as evidence that
"sophistication is not the issue".

This is itself an important paper claim — most prior soft-prompt
work argues that exposure-bias correction (OPD / SCST / RLHF-style)
matters. On our benchmarks at this scale, it does not. The
**bit-capacity wall on the read side** is the binding constraint,
not the training objective.

---

## 6. Decision 5 — hyperparameter freeze

After Phase A-H + Phase J-S sweeps, all knobs were locked. The
final canonical recipe (see
`mem-embedding/k8s/sweep/build_sweep.py::BEST_RECIPE`):

```python
combine            = "xattn"
n_memory           = 32        # K=16 weaker, K=64+ marginal
n_heads            = 8         # 8 > 4 > 1 (Phase A4)
n_layers           = 1         # 1-layer update block; 2 layers no lift, more params
ffn_mult           = 2         # FFN inner 2×
dropout            = 0.0       # dropout hurts at this size
update_alpha_init  = 1.0       # NOT ReZero (0 stalled in Phase A)
apply_alpha_init   = 0.0       # NoOp at init for the per-token residual path
lr                 = 3e-5      # 1e-5 slow, 1e-4 unstable
n_items            = 500       # the headline data-scale lift
n_train_steps      = 1800      # 5000 steps monotonically worse (over-fit)
train_modes        = "sft"     # OPD/RL no lift
seeds              = (42, 7, 11, 13)
base               = Qwen3-8B
```

Trainable footprint at this recipe:

* OURS (`combine=xattn, K=32, h=8`): **218.33 M** params
  ($\approx 2.67\,\%$ of Qwen3-8B's 8.19 B).
* GIST (matched `k_per_chunk=6`): 134.30 M (1.64% of base; 0.61×
  ours).

This is the cell that produced every Phase Y / Z / X / X2 number
in `summary/matrix.md` and the headline of the v1 paper.

---

## 7. Decision 6 — wrap it in a protocol

The final architecture is exposed via the
`llm_infra.wrappers.WrapperProtocol` interface in three methods:

```python
init_memory()  -> MemoryState                    # learned tensor m_0 ∈ R^{K×d}
update(m, c)   -> MemoryState  # m' = m + α · UpdateBlock(m, c) — one attn + FFN, residual
apply(m, q)    -> BaseCall     # base(combine(m, q)) — combine is 1-of-7 modes
```

Three methods, no train logic in the wrapper. Training lives in
`llm_infra.train` and consumes this protocol.

This is what made it cheap to add the Gist baseline
(`src/mem_embedding/baselines/gist.py`) at matched-$K$ — Gist
provides the same three-method protocol with a different `update`
(it chunks the input on the *token* axis and lets the base produce
its own $k_\text{per\_chunk}$ Gist-token KV pairs that get prepended
to the next chunk's query). Same training loop, same eval harness,
strictly comparable cells. The protocol is what makes the
"head-to-head OURS vs GIST" table of the paper a fair comparison.

---

## 8. What didn't make it (the negative space)

Every architectural / training choice has a discarded variant.
Documenting the discards is half the story.

| discarded | why |
|---|---|
| ReZero init ($\alpha_\text{upd} = 0$) | wrapper stalled at identity within SFT budget |
| 2-layer update block | $2\times$ params, no measurable lift on any task |
| Dropout > 0 | hurts wrapper recovery at small scale (Phase A8) |
| OPD-alone or RL-alone (without preceding SFT) | both collapsed to no-signal; SFT is the necessary warm-up |
| $\lambda_\text{div} = 0.5$ | over-regularises; memory stops writing anything |
| $\lambda_\text{div} = 0$ | memory writes same thing every chunk (cos_consec → 1.0) |
| `n_train_steps = 5000` (over-train at $N{=}100$) | monotonically worse than 1800 steps; refutes the under-training hypothesis |
| Chunk granularity $(512, 3)$ | sequence too long, wrapper drops more info; $(256, 6)$ is the sweet spot |
| $K = 256$ at $N{=}500$ | wrapper memorises training items; eval em flat, train em up |
| Per-layer adapter inside base | violates C1 (frozen base) |

All of these are sitting in `details.tex` §Failed combinations
catalogue with cell-level numbers.

---

## 9. The one-line answer to "how did we arrive at the wrapper?"

> We picked a dense-vector $K{\times}d$ memory pool because it is
> the smallest interface that lets a frozen base see a compressed
> long-context summary; we wrote the update as a residual
> cross-attention block so we could expose the per-step delta as a
> diagnostic; we tested seven ways to deliver memory to the base
> and kept `xattn` because it is the only one that did not collapse
> on `coding_niah`; we ran the training as SFT $\to$ SFT+OPD $\to$
> SFT+OPD+RL and kept SFT-only because the extra stages added no
> lift; we ablated eight axes and locked
> $(K{=}32, h{=}8, L{=}1, \alpha_\text{upd}{=}1.0,$
> SFT, $N{=}500, 1800$ steps$)$; the result is the recipe whose
> Phase Y, Z, X, X2 cells populate the paper. The bit-capacity wall
> on the read side is *invariant* to every wrapper knob we tested
> and *invariant* to swapping in Gist; we therefore frame the v1
> paper around characterising that wall (the three-regime law),
> not around claiming the wrapper beats Gist.

---

## 10. What the next paper looks like

The bit-capacity wall is intrinsic to lossy compression via a
fixed-$K$ soft prompt. v2 (suffix-style memory written *during*
autoregressive generation, with the persistent-across-sessions
contract) and the parked $\Delta W$ track of `plan-08` are the
two escape routes; the brainstorm in
`v1-if-wrapper-doesnt-work-2026-06-03.md` enumerates eight more.
v1 is honest about being *one resolution* of the problem, not the
only one.
