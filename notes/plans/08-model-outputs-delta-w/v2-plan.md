# Plan 08 — v2: Cross-Session Latent Memory with Read/Write Tokens

> **Status**: design (v1 still running; v2 is the next paper after v1 ships)
> **Created**: 2026-06-01
> **Owner**: Mingjia
> **Parent**: Plan 08 v0 / v1 (`v0-learned-memory-wrapper.md`,
> `v0-paper-target-2026-05-31.md`)
> **One-liner**: A lightweight wrapper on a *frozen* base LM that writes
> latent memory tokens **during** autoregressive generation and reads them
> back across **sessions**, with the wrapper's bit-capacity wall as the
> central diagnostic story.

---

## 0 · TL;DR

v1 (paper now) characterized **what a fixed-K soft-prompt memory built at
encode time can hold** — and where the bit-capacity wall sits.

v2 (paper next) asks: **what if the memory is appended during the
autoregressive output and persists across sessions?** Specifically, the
wrapper writes a small set of latent "suffix" tokens at the end of each
turn, those tokens are serialized as the session state, and a future
session can pick up by re-loading them as a soft prefix.

After a thorough literature pass (see `v2-related-work.md`), the
"suffix latent memory during generation" idea is **heavily crowded**
(Coconut, TokMem, LightThinker, CoLaR, R3Mem, NextMem, ...). The
differentiation we land on combines two angles those papers do not
jointly cover:

* **A — cross-session persistence**: existing work is mostly intra-task
  (compress one CoT, or one context window). Truly persistent latent
  memory across conversational sessions is almost empty space.
* **B — read/write under a *fully* frozen base** with a quantified
  bit-capacity wall: TokMem and R3Mem tune the base or part of it;
  Coconut / CODI are full fine-tunes. Doing it with a frozen base and
  inheriting v1's bit-capacity diagnostic ties v2 to v1 as a single line.

The two angles align around one thesis (v1 + v2 together):

> *The bottleneck of soft-prompt memory is the read interface, not the
> write capacity. Under a frozen base, this bottleneck holds whether the
> memory is built at encode time (v1) or written suffix-style during
> generation (v2), and persists across sessions when the latent state
> is serialized and re-attached.*

---

## 1 · The v2 system, at a sketch

```
                ┌─── BASE LM (Qwen3-8B, FROZEN) ──────────────────┐
                │                                                 │
   user msg ─▶ │ tokens → embed → frozen decoder → output tokens │
                │                                  │              │
                │                                  ▼              │
                │                               WRITE             │
                │                               head              │  Wrapper params:
                │                                  │              │   - write head
                │                                  ▼              │     (cross-attn from
                │                            <Δm_t suffix>        │      hidden states
                │                                  │              │      → K latent tokens)
                │                                  ▼              │   - read head
                │   (next turn)         m_t = m_{t-1} + α·Δm_t   │     (reuses v1 apply
                │                                  │              │      modes: prefix /
                │                                  ▼              │      xattn / residual)
                │                            session.save(m_t)   │   - α gates
                │                                                 │   - LayerNorms
                │   (reopen later)     m_t' = session.load(...)   │
                │   user msg' ─▶ READ(m_t', tokens) → output      │
                └─────────────────────────────────────────────────┘
```

Crucially, the wrapper is **strictly additive to the base LM** — same
frozen Qwen3-8B (or Llama 3.1 8B for the headline) — and the same
`Wrapper` protocol from v1 carries over, with one extension:

```python
class Wrapper(Protocol):
    def init_memory(B, device) -> MemoryState: ...
    def update(m, ChunkEncoding) -> MemoryState: ...           # v1: read inputs
    def apply(BaseCall, m) -> BaseCall: ...                    # v1: read into BaseCall
    # NEW in v2:
    def update_from_generation(m, GenStep) -> MemoryState: ... # write during AR
    def serialize(m) -> bytes: ...                             # session persistence
    def load(bytes) -> MemoryState: ...
```

`GenStep` carries the latest generated token's hidden states (or a
sliding window of them), and `update_from_generation` is invoked at
end-of-turn (default), or at sentence boundaries / EOS / explicit
`<memwrite>` markers (configurable).

---

## 2 · Wrapper-to-model integration: design space

This is the open question the conversation pinned. Six dimensions:

### 2.1 **Where to read base hidden states from?**

The base is frozen; we hook in via forward hooks. Options:

| option | description | pro | con |
|---|---|---|---|
| 1. last layer only | hook `model.layers[-1]` output | matches v1; cleanest | last layer is already linguistic-output-shaped, may discard memory-bearing earlier signal |
| 2. one chosen mid layer | e.g. `model.layers[20]` for 36-layer Qwen3-8B | concentrated memory-bearing signal (per Q2 probing) | which layer? data-driven choice |
| 3. multi-layer concat / pool | several layers projected to common dim | maximum signal | param + compute blow-up |
| 4. embed layer only | input embeddings before any block | minimal disturbance | almost certainly loses meaning |

**v2 default**: option 2, layer chosen by **Q2 activation-memory probe**
(see `q2-activation-memory-probe.md`). Fallback: option 1 (v1 behaviour).

### 2.2 **Where to inject the read?**

Symmetric design space. Options:

| option | description |
|---|---|
| 1. soft prefix on tokenized input | v1's `prefix` apply mode — simplest, no base changes |
| 2. soft prefix on `inputs_embeds` after embed lookup | same effect, slightly cleaner shape |
| 3. residual injection at one layer | hook on `model.layers[i]` adds `α · CrossAttn(h, m)` to residual |
| 4. KV-cache stitching | prepend latent KV directly into self-attention KV cache (à la Memorizing Transformer) |

**v2 default**: option 1 for the first run (zero base-model changes),
option 3 as the experimental track. Both are wrapper-only changes; no
base weights touched.

### 2.3 **When to write during generation?**

| option | description | pro | con |
|---|---|---|---|
| 1. end of turn | one write per response | cheapest, cleanest semantics | low temporal resolution |
| 2. sentence boundaries | write after each `. ! ?` or newline | natural granularity | tokenizer-dependent |
| 3. every $N$ tokens | fixed interval | simple | arbitrary |
| 4. every token | Coconut-style continuous feedback | maximum info | most expensive, hardest to train |
| 5. model-decided via `<memwrite>` markers | learn when to compress | TokMem / LightThinker style | needs supervised markers or RL signal |

**v2 default**: option 1 for the first run; option 5 as the secondary
ablation (lift from LightThinker's training data construction).

### 2.4 **Memory as suffix vs. side-bank**

| option | description |
|---|---|
| 1. suffix on the active context (TokMem-style) | memory consumes context-window slots; auto-fits into the standard KV cache |
| 2. side-bank (RMT-style) | memory lives off-context; read via dedicated cross-attn; doesn't compete for context |

**v2 default**: option 1, because it requires no base-model surgery and
the bit-capacity story is cleanest when memory shares the same attention
machinery as ordinary tokens. Option 2 is in scope only if option 1
hits an attention-routing problem during scaling.

### 2.5 **What gets serialized for cross-session?**

`MemoryState.payload` is a `(B, K, d)` bfloat16 tensor — ~256 KiB per
session at `K=32, d=4096`. That's the entire serialized state. No KV
cache, no transcript needed.

```python
session_blob = wrapper.serialize(m)              # bytes, ~256 KiB
storage.put(session_id, session_blob)            # any KV store
# ...days later...
m_resume = wrapper.load(storage.get(session_id))
m_resume = wrapper.update(m_resume, encode(new_message))   # if you want to re-prime
output = base_lm.generate(wrapper.apply(call, m_resume))
```

### 2.6 **Trainable parameters under frozen base**

| component | params (estimate at K=32, d=4096, 8 heads) |
|---|---|
| write head (cross-attn Q,K,V proj + out proj) | 4·d² = 67 M |
| update gate α + LayerNorm | <1 K |
| read head (mirror) | 4·d² = 67 M |
| apply gate α + LayerNorm | <1 K |
| (optional) FFN in update block | 2·d² = 33 M |

So **~70–170 M trainable params** for a wrapper around a 7–8 B base.
~1–2 % of base parameters. "Lightweight" claim is honest.

If we share write/read heads (tied projections) we drop to ~70 M.

---

## 3 · Where this fits in the published landscape

See `v2-related-work.md` for the full table. The two near-neighbours
that v2 must explicitly compare against, head-to-head:

| competitor | how v2 differs |
|---|---|
| **TokMem** (Oct 2025) | TokMem writes one memory token per *procedure*, supervised by procedure–response pairs. v2 writes per-turn, unsupervised, and (Δ1) does it under a strictly frozen base, (Δ2) persists across sessions, (Δ3) characterizes bit-capacity. |
| **LightThinker** (EMNLP 2025) | LightThinker compresses generated *thoughts* into gist tokens then *discards* them within the same context. v2 (Δ1) does not discard — keeps them as persistent latent state, (Δ2) bridges sessions, (Δ3) does it without supervised compression markers. |
| **R3Mem** (ACL 2025) | R3Mem has read/write token pairs but fine-tunes the encoder; v2 keeps the base frozen and quantifies the read-side bottleneck. |
| **Coconut** (NeurIPS 2024) | Coconut requires full base fine-tune for the language↔latent switching; v2 is wrapper-only. |
| **Mem0** (2025) | Mem0 is textual memory + graph store, not latent. v2 contrasts as the "fully latent" point on the same Pareto frontier on LongMemEval / LOCOMO. |

---

## 4 · Open research questions (parked for after v1 ships)

* **Q2 — Which activations are memory-bearing?**
  Documented separately in `q2-activation-memory-probe.md`. Methodology
  insight: if some channels / neurons are heavily used by ordinary
  language and others are sparsely activated, the wrapper should write
  *only* into the sparse channels — minimum disturbance, maximum
  capacity. This becomes its own methodology contribution.

* **Q3 — Is the bit-capacity wall the *same* curve at write time?**
  v1 measures the wall at encode time. v2 will measure it at write
  time. If the wall is at the same answer-entropy regardless of when
  memory is built → strong evidence the bottleneck is purely
  read-side.

* **Q4 — How does the wall move with read interface (prefix vs xattn
  vs KV stitching)?** Touched in v1's Phase C; needs a dedicated curve
  in v2.

* **Q5 — Does conversational drift accumulate across sessions?**
  Need a "100-session-back" eval where session N answers depend on
  session N-100 facts.

* **Q6 — Does the wrapper interfere with the base LM's ordinary
  competence?** Run MMLU / HumanEval / GSM8K with the wrapper attached
  but memory empty; perplexity / accuracy should be unchanged.

---

## 5 · Concrete next steps (sequence)

Assumes v1 paper is submitted within ~4 weeks.

| step | what | depends on | wall-time |
|---|---|---|---|
| 1 | Land v1 paper draft (Bit-Capacity Limits) | — | week 1–4 |
| 2 | Implement `update_from_generation` + `serialize`/`load` in `mem-embedding` | step 1 design freeze | 3–5 days |
| 3 | Smoke-test on a synthetic 2-session NIAH (encode facts in session 1, query in session 2) | step 2 | 2 days |
| 4 | Q2 activation-memory probe (separate, can run in parallel) | independent | 1 week |
| 5 | First v2 capacity curve on numerical-NIAH (encode-time vs write-time wall comparison) | step 3 | 3 days |
| 6 | LongMemEval / LOCOMO real-text benchmark integration | step 3 | 1 week |
| 7 | Head-to-head against TokMem (reimpl on top of `mem-embedding`) | step 6 | 1 week |
| 8 | v2 paper draft (target: ICLR 2027 + NeurIPS 2027 backup) | steps 5–7 | 4–6 weeks |

---

## 6 · How v1 and v2 share infra

`mem-embedding`, `llm-infra`, `encoder-infra` all already speak the
right protocols. v2 needs:

* **`mem-embedding`**: add `update_from_generation` to wrapper protocol
  and the `MemoryEmbeddingWrapper` impl. Add `serialize` / `load`
  helpers. Both are pure additions — v1 code path unaffected.
* **`llm-infra`**: add a generation hook that calls
  `wrapper.update_from_generation(m, h_t)` at chosen trigger points;
  drop into `BPTTTrainer`, `OnPolicyDistillTrainer`, `RLTrainer` as
  optional. Add a `MultiSessionDataset` that yields
  `(session_blob_in, msg, expected_out)` triples.
* **`encoder-infra`**: no change.
* **`mem-test`** (top-level): `CONVENTIONS.md` update — note that wrappers
  may now expose `update_from_generation` (optional protocol method).

No breaking changes anywhere.
