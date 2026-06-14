# GCM + non-merged LoRA — mechanism (v1.7)   ⛔️ ARCHIVED

> ⛔️ **ARCHIVED (leakage-era).** The mechanism is still valid, but the numbers cited here are from the
> pre-leakage-fix v1.7 runs (retracted). Current mechanism + clean numbers + the robustness reframe:
> **[`../results-v1.7.3/gcm-lora-mechanism.md`](../results-v1.7.3/gcm-lora-mechanism.md)** and
> [`../results-v1.7.3/results-v1.7.3.md`](../results-v1.7.3/results-v1.7.3.md).

Single-page, self-contained explanation of how the Gated Compressor Module (GCM) with a non-merged base LoRA
works: components, how the K memory tokens are produced, the training losses, and the compress/fallback inference
path. Numbers and the transfer matrix live in [`solutions/master-plan-180gpuh.md`](solutions/master-plan-180gpuh.md).

## Code map (where it lives)
Core: **`mem-test/svc/src/svc/`** · Eval/runtime: **`mem-test/mem-embedding/src/mem_embedding/gcm/`**

| file | role |
|---|---|
| `svc/compressor.py` | **Encoder** = trainable copy of base's first N blocks; **K learnable memory slots**; `m_proj`; **decoder** (reconstruction) |
| `svc/method.py` | GCM training loop, the 5 losses, gate signals `signals()`, LoRA wiring |
| `svc/lora.py` | **non-merged** LoRA on `q_proj`/`v_proj` + `set_lora_enabled` / `lora_disabled` toggle |
| `mem_embedding/gcm/runtime.py` | frozen-base runtime (`generate`, `mc_loglik`, embeddings) |
| `mem_embedding/gcm/harness.py` | eval dispatch; `_set_base_lora` (compress=on, full/fallback=off) |
| `svc/queue_lora.py`, `svc/queue_matrix.py` | winning config: `base_lora_rank=16, n_items=384, steps=1600, enc=16, K=128, n_dec=2, lam_distill=0.5` |

## How the K memory tokens are produced (NOT autoregressive)
Encoder input is `[ctx ; query ; K learnable slots]`, run through the N-layer encoder in **one forward pass**. The
final hidden states at the K slot positions → `m_proj` → **M** (K soft tokens in the base's input-embedding space).
A causal + query-mask (`compressor.py::_enc_mask`) decides what the compression "saw":
- query **masked** → **M0** (unconditional memory)
- query **unmasked** → **Mq** (conditional memory)

## Training (per item) — what is trainable
Trainable: **Encoder (N blocks) + K memory slots + `m_proj` + Decoder + base-LoRA (A,B)**. **Base weights frozen.**

```mermaid
flowchart TD
  CTX["ctx + query"] --> ENC["ENCODER<br/>trainable copy of base's first N blocks<br/>input [ctx ; query ; K slots]"]
  ENC -->|"query NOT masked"| MQ["Mq (K soft tokens)"]
  ENC -->|"query masked"| M0["M0 (K soft tokens)"]
  MQ --> LT["L_task: [Mq ; query ; gold] -> base(+LoRA) -> CE(gold)<br/>forces M to carry the answer"]
  MQ --> LD["L_distill: [Mq ; query ; R] -> base(+LoRA)<br/>match full-context teacher distribution over R"]
  M0 --> LR["L_rec (L_uncond): M0 -> DECODER -> reconstruct ctx (CE)<br/>lossless signal"]
  MQ --> LV["L_dev: ||Mq_raw - M0_raw||^2<br/>anchor conditional to unconditional"]
  MQ --> LA["L_adv (optional): discriminator on base layer-l hidden<br/>cannot tell M-induced vs full-induced"]
  LT --> OPT["AdamW over: Encoder + slots + m_proj + Decoder + LoRA(A,B)<br/>(base frozen)"]
  LD --> OPT
  LR --> OPT
  LV --> OPT
  LA --> OPT
```

**LoRA's role:** the base is frozen, but the non-merged LoRA on `q_proj`/`v_proj` lets the base *learn to read M*,
trained jointly with the encoder. This is the unlock — with a frozen base, more compression data does not help
(0.354 stays flat); with LoRA, 96→384 items lifts bfcl compress 0.354→0.531–0.583. (See master-plan LoRA×data 2×2.)

## Inference (per item) — compress vs fallback, gated [HIGH-LEVEL]
```mermaid
flowchart TD
  IN["ctx + query"] --> E["ENCODER -> M"]
  E --> DEC{"GATE: is this compression trustworthy?<br/>(two-stage; expanded in 'Gate detail' below)"}
  DEC -->|"yes"| COMP["COMPRESS path<br/>[M ; query] -> base + LoRA ON<br/>K tokens replace the whole context (cheap)"]
  DEC -->|"no"| FB["FALLBACK path<br/>[ctx ; query] -> base, LoRA OFF<br/>= exact original base (do-no-harm)"]
  COMP --> ANS["answer"]
  FB --> ANS
```

- **compress path** runs with LoRA **ON** (base reads M); **fallback path** runs with LoRA **OFF**
  (`harness.py::_set_base_lora`), recovering the **exact original base** on full context — the do-no-harm guarantee.
  `full_ctx` is scored LoRA-off too, so the "full" baseline is never contaminated.
- The single `GATE` node above hides a two-stage mechanism — **expanded next**.

## Gate detail — "is compression trustworthy?" [zoom-in of the GATE node above]
**Stage 1 — signals** (`method.py::signals`, `@torch.no_grad`): run the encoder **twice** (M0 = query-masked, Mq =
query-unmasked) and read several scalar trustworthiness signals from the memory + the base's behaviour.
**Stage 2 — decision** (`disc_gate.py::gated_acc`): threshold one signal — `compress iff signal ≥ τ`, else fall back.
The two leaves below **are the COMPRESS / FALLBACK boxes of the [HIGH-LEVEL] inference figure above**.

```mermaid
flowchart TD
  ENC["ENCODER run TWICE on (ctx, query)"] --> M0["M0 = encode(query MASKED)<br/>unconditional memory"]
  ENC --> MQ["Mq = encode(query unmasked)<br/>conditional memory"]
  M0 --> REC["DECODER reconstructs ctx FROM M0 -> CE"]
  REC --> S1["neg_recon = -CE   (the BEST signal)<br/>high => M re-encodes ctx => faithful => trust"]
  MQ --> BR["base forward on [Mq ; query]<br/>-> first answer-token dist pq"]
  BR --> S2["conf / margin / neg_entropy<br/>high => base confident reading M"]
  M0 --> DV["compare M0 vs Mq"]
  MQ --> DV
  DV --> S3["dcode=||Mq_raw-M0_raw||, dlogit=KL(pq||p0)<br/>query-adaptation magnitude"]
  MQ --> DSC["(optional) discriminator D:<br/>base layer-18 hidden -> MLP -> p"]
  DSC --> S4["disc_p (low p = fooled D = lossless)<br/>disc_conf=|p-.5|, disc_negent"]
  S1 --> TAU{"signal >= tau ?<br/>best single signal = neg_recon<br/>(or supervised combo / disc strategy)"}
  S2 --> TAU
  S3 --> TAU
  S4 --> TAU
  TAU -->|"yes => trustworthy"| GC["==> COMPRESS path (HIGH-LEVEL fig above)"]
  TAU -->|"no"| GF["==> FALLBACK path (HIGH-LEVEL fig above)"]
```

### The signals (oriented so higher = safer-to-compress)
| signal | source | trustworthy reading |
|---|---|---|
| **`neg_recon`** (best) | decoder CE of reconstructing ctx from M0 | high = no info lost = trust (the self-verifying core) |
| `conf` / `margin` / `neg_entropy` | base first-token dist on `[Mq;q]` | high = base sure reading M (TARG-style) |
| `dlogit` / `dcode` | KL(Mq‖M0) on first token / ‖Mq_raw−M0_raw‖ | conditional adaptation magnitude |
| `mnorm` | ‖Mq‖ | OOD / geometry check |
| `disc_p` (if adversarial-trained) | MLP on base layer-18 hidden of `[Mq;q]` | low p = M fooled D = behaviorally lossless |

### Decision + the three ways to set the gate
- **Threshold** (`disc_gate.py::gated_acc`): sweep τ; `gAcc = mean(compress-score if signal≥τ else full-score)`,
  with coverage `gcov` and realized fallback rate `1−gcov`. Deploy fixes τ at a target precision (do-no-harm
  operating point); higher τ = more conservative (compress only the very-confident items).
- **(1) intrinsic** — threshold `neg_recon` (no training; best). **(2) adversarial** — the learned discriminator p,
  strategies S_low/S_conf/S_ent/S_high (`disc_gate.py:44-57`). **(3) supervised** — 5-fold logistic regression over
  all signals (`disc_gate.py::fit_gate`).
- **Empirical**: bfcl `neg_recon` AUROC 0.88, gAcc 1.0 (held-out = in-sample ⇒ realizable). rca/squad AUROC≈0.5
  (signals can't tell) ⇒ the gate correctly **always falls back ⇒ gAcc = full** (do-no-harm). Note `gAcc` = model +
  fallback; `best(compress,full)` = oracle-gate upper bound; `always-compress` = no fallback.

## Two-regime result (see master-plan for the full transfer matrix)
- **tool-use (bfcl): works** — in-task compress 0.531 ≫ no_ctx 0.010, beats faithful Cartridge (0.375) / Gist (0.104).
- **rca / hotpot / apibank: compression ≈ no_ctx (fails)** — gate-critical, fallback carries the do-no-harm floor.
- **transfer is weak** — encoder+LoRA are trained per-corpus (like Cartridge), so cross-corpus compression collapses.
