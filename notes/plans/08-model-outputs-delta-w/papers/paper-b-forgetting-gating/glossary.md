# Glossary — Paper B (gated parametric memory)

Definitions for the technical / internal terms used across the results docs. Link to a term as
`[GCM](../glossary.md#gcm)` (from a `results-v1.7.x/` doc) so terms jump here, wiki-style.

---

## Method & memory

### GCM
**Gated Context Memory** — our method. A learned **encoder** compresses a context into **K soft tokens (M)** that are
injected into a **frozen base** model in place of the context; a **do-no-harm gate** decides per query whether to trust the
compressed memory or **fall back to full context**. Contribution = the *reliable gate*, not a bigger compressor.

### M (memory / soft tokens)
The **K** compressed vectors the encoder produces from a context. Injected into the base (see [injection](#soft-prompt--prefix-injection)).
`M0` = unconditional memory (query masked during compression); `Mq` = query-conditioned memory.

### compress (path)
Running the base on `[M ; query]` — the compressed memory replaces the real context. The cheap path.

### full (full_ctx)
Running the base on the **real full context** `[ctx ; query]`. The quality **ceiling**.

### no_ctx
Running the base on the **query alone** (no context). The **necessity floor** — if `no_ctx ≈ full`, context isn't needed.

### trunc
Keep only the **first-K context tokens** (no training). A **trivial floor**, not a method — matches `full` only when ctx ≤ K.

---

## Injection (how M enters the base)

### soft-prompt / prefix injection
M = **K input-embedding vectors prepended to the query** (`--inject prefix`). Our default. Cost ≈ K extra input tokens.

### KV-cache injection
M = **per-layer Key/Value pairs loaded into every attention layer's cache** (`--inject kv`); the query attends to them at
each of the L layers. More expressive but **~L× the memory/compute** of a soft prompt. The Cartridge / Beacon / 500x family.

---

## Encoder & recipe knobs

### encoder
The trainable module that produces M. Default = a **trainable copy** of the base's first N layers (`enc-init copy`).

### encoder depth (1 / half / full)
N = number of encoder layers relative to base depth L: `1`, `half`=⌊L/2⌋, `full`=L (`--enc-layers full|half|N`).

### shared-lora encoder
*(option)* Encoder as a **LoRA on the shared frozen base** instead of a separate copy ⇒ store the base once (~½ memory),
makes `depth=full` cheap. Trades capacity for memory — pending ablation vs the copy encoder.

### K (memory budget)
Number of soft tokens in M (`--n-memory`). v1.7.6 sweeps fixed {4, 8, 16}.

### adaptive K (chunk-size)
K slots **per chunk** of `chunk-size` context tokens (`--chunk-size`); the budget **scales with length** at a fixed
compression ratio = chunk/K. v1.7.6 uses 2× / 4× / 8× compression.

### base-read LoRA
A LoRA on the **frozen base** so it learns to read M. **Disabled on the fallback path** ⇒ the exact original base on full
context (the do-no-harm guarantee).

### distillation (lam_distill)
Distill the **full-context teacher's distribution over the gold answer** into the M-conditioned student (one forward in v1.7.6).
Load-bearing (removing it collapses GCM).

### reconstruction (lam_rec)
A small decoder reconstructs the context **from M** — the lossless (information-content) signal.

### deviation (lam_dev) · task loss (lam_task)
`dev` keeps `Mq` close to `M0`; `task` = gold-answer cross-entropy (forces M to carry the answer).

### patience
**Early-stop** for "train until converge": a held-out **val loss** is checked every ~steps/10; training stops after
`patience` consecutive checks without improvement (>1e-4). An **epoch cap** bounds the worst case.

### epochs vs steps
`--epochs` = passes over the train set (1 epoch = every item once); internally `steps = epochs × |train set|`.
`--steps` is the legacy fallback (used only when `--epochs=0`).

---

## Gate

### gate (do-no-harm gate)
A **verifier** that decides, per query, whether to trust `compress` or **fall back to `full`**. "Do-no-harm" = the gated
system never scores below `full`. The reliability of this gate is Paper B's core claim.

### gate signals
Label-free per-item scores the gate thresholds on: `conf` (top-1 prob), `margin` (=[TARG](#targ)), `neg_entropy`,
`neg_recon` (intrinsic, reconstruction-based), `judge` (=[LLM-judge](#judge)).

### TARG
**Prefix-logit-margin** gate signal: the top-1 − top-2 probability margin on the compressed run's first answer token.

### judge
**LLM-judge** (D-Mem style): ask the base, reading M, "can you answer from this context?" and use **P(Yes) − P(No)**.

### net-win
The payoff: **tokens saved at matched quality** by gating (compress when safe, full when not). Measured on a cost↔quality Pareto.

### AUROC / recall@P95 / AURC
Gate-quality metrics. **AUROC** = how well a signal separates compress-success from failure. **recall@P95** = fraction of
true-good compressions kept at 95% precision. **AURC** = area under the risk–coverage curve (lower = better).

---

## Relations & regimes (the 3 main tables)

### in-task
Train **==** eval (same task **and** domain). Main Table 1.

### cross-task
Train task **≠** eval task, **same domain** (e.g. one tool bench → another tool bench). Main Table 2.

### cross-domain
Train domain **≠** eval domain (e.g. tool → wiki). Main Table 3.

### anchor
The single dataset a cell is **trained** on (one of tool / wiki / literary / ops).

### mix
Training on a **mixture of all anchors**, then evaluating all benchmarks.

### ctx>K
Items whose context length **exceeds the memory budget K** — the only items where compression does real work; the
**fair-comparison filter** for "no compressor matches full."

---

## Baselines (faithful 2025 context-compressors)

### Cartridge
Trained **KV-cache** per corpus (self-study + context distillation), frozen base. [KV-injection](#kv-cache-injection).

### Gist
Gist tokens — soft-prompt compression learned by masking. [soft-prompt](#soft-prompt--prefix-injection).

### ICAE
In-Context Auto-Encoder: encoder LoRA + memory slots, reconstruction objective. Soft-prompt.

### AOC
Attention-Only Compressor: a full-depth attention-only encoder. Soft-prompt.

### Beacon
Activation-beacon: per-layer **KV** beacons with interleaved alpha + context reconstruction. [KV-injection](#kv-cache-injection).

### 500x (x500)
500xCompressor: KV-conditioning compression. [KV-injection](#kv-cache-injection).

### ComprExIT
Compress-then-extract via cross-layer transmission (softmax-weighted anchor hidden states). Soft-prompt.

### LCC
Long-context compiler: full-context compilation into memory. Soft-prompt.
