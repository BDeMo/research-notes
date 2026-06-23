# Paper B — v1.8.0 logic + module map (what is the PAPER; what is new)

> Companion to [`../logic.md`](../logic.md) (the locked v1 spine). This file maps the **v1.8.0 method modules** to the
> paper's claims and flags **what is new this cycle**. **Scope guard:** **CMG-RCA is the *project* (a long-context RCA
> foundation-model demo), NOT a paper claim.** It supplies the long-context *motivation* + a case study; the paper's
> numbers live on the public benches (bfcl / QA / RCA) + cross-model. CMG's absolute accuracy is near the base-model
> ceiling and is reported only as a demo, never as a headline result.

## 1. The one-line logic (unchanged)
Adapting a frozen LLM with a compressed-context memory is **capacity-bound** (the memory is lossy/OOD), so the durable
contribution is **not a better compressor** but a **do-no-harm gate**: per input, trust the cheap compressed memory when
a robust intrinsic signal says it is safe, else fall back to full context — so quality is **never below full** (by
construction) while paying the compressed cost when safe. *The compressor is the carrier; the gate is the contribution.*

## 2. Method = three roles on ONE frozen base (adapters only)
`encoder` (context → memory M) · `reader` (answer from M) · `judge/gate` (trust M or fall back). No base weights change;
only K memory tokens + a projection MLP + read-LoRA (optionally encoder-LoRA) are trained.

## 3. Module map (role · what it does · NEW in v1.8.0? · which claim · ablation finding)
| module | role | what it does | new? | claim | finding (v1.8.0) |
|---|---|---|---|---|---|
| K memory tokens + proj-MLP | enc | the compressed memory M | — | C-carrier | K=256 mild-best; not fragile |
| read-LoRA (soft-prompt prefix) | reader | reads M to answer | — | C-carrier | rank 64 enough |
| **joint-task** (answer-CE trains encoder too) | enc+reader | M learns to *carry* the answer | fix | enabler | pivotal (without it, low compress) |
| distillation (KL to full-ctx teacher) | enc+reader | match full-context behavior | — | C-carrier | load-bearing |
| **reconstruction** (M0 → ctx, tied head) | enc | M must preserve the context | ✅ | C-carrier **+ powers `neg_recon` gate** | load-bearing (gate dies without it) |
| deviation (‖Mq−M0‖²) | enc | query shouldn't distort M | ✅ | minor | ±0 (droppable) |
| **contrastive** (InfoNCE: M-hidden ↔ full-hidden) | enc | make M discriminative | ✅ | minor | situational (helps discrimination; can hurt) |
| VAE memory (μ,logσ + KL prior) | enc | probabilistic/regularized M | ✅ | minor | KL must be tiny; ≈neutral |
| encoder-LoRA | enc | should the *encoder* be trained? | ✅ | ablation | does not beat frozen-enc + joint-task |
| normalization (off/hard/learn/manifold) | enc | put M on the embedding manifold | ✅ | stability | hard-manifold = stable default |
| **adaptive-M** (chunked, M length ∝ ctx) | enc | budget scales with length | ✅ | scaling | (running) |
| **recurrent over-window encode** | enc | fold arbitrarily long ctx into M (detached chunk-carry) | ✅ | long-ctx | length-agnostic; fixed-length training → OOD at other lengths (fix below) |
| **encode-length robustness (varlen)** | enc | train on random #chunks | ✅ | long-ctx fix | (running) |

## 4. The gate (the contribution) — modules
| gate module | what it is | new? |
|---|---|---|
| signals: `margin`, `conf`, `neg_entropy` | compress-path first-token uncertainty | — |
| `neg_recon`, `dlogit` (KL(Mq‖M0)) | verifiability / query-shift (the v1.7.5 winners) | — |
| **`targ`** (base-uncertainty, no-ctx margin) | the TARG training-free baseline gate (honest: ≥ ours on some families) | ✅ |
| **2-way gate** (compress ↔ full) | do-no-harm: trust M iff signal ≥ τ else full | — |
| **3-way adaptive gate** (base → compress → full, cost-ascending) | use the *cheapest* path the gate trusts | ✅ |
| metrics: gated-acc, **F1 / precision / recall / fire / fallback**, AUROC | report the gate, not just accuracy | ✅ (F1/prec/recall/fallback this cycle) |

## 5. What is NEW in v1.8.0 (delta vs v1.7.5) — the headline additions
1. **Full gate evaluation:** gate F1 / precision / recall / fire-rate / fallback-rate (was AUROC-only) + a **TARG baseline gate** for honest comparison.
2. **3-way adaptive gate** (base/compress/full, cost-ascending) — the cost–coverage story.
3. **Reconstruction + deviation + contrastive + VAE + encoder-LoRA + normalization-mode** as first-class, env-gated **ablation modules** → the flip-one ablation (the method is robust; the gate carries the win).
4. **Recurrent over-window encode** (+ the varlen length-robustness fix) — length-agnostic compression.
5. **Per-base HP tuning, eff-batch ≥ 8, shuffle + domain-curriculum + per-domain warmup** — training hygiene; finding: **K256 > the old rdK64 default**.
6. **Same-hardware ranking protocol** — eval all checkpoints on one GPU to remove fp-rounding noise on near-tied tasks.

## 6. Claims → evidence (paper-level; public benches, not CMG)
- **C-do-no-harm:** gated-acc ≥ full on every bench; on bfcl the gate **beats** full (gated 0.86–0.92, F1 0.87). ✅
- **C-robust-method:** flip-one ablation moves compress by only ±0.03 on both bases ⇒ no single knob is load-bearing; the gate is the contribution. ✅
- **C-capacity-bound:** compress ≪ full on most benches (the lossy memory) ⇒ the gate, not a bigger compressor, is the answer. ✅
- **C-cross-model:** the diagnosis + gate reproduce across families (v1.7.5: 5 families; extending on v1.8.0).
- **Honest negatives:** TARG ≥ ours on some families (signal is an ablation, not the headline); CMG is near-random for the base (project demo, not a paper number).

## 7. CMG (the PROJECT) — where it belongs
CMG-RCA is the **deliverable/demo** (foundation-model serving + interactive case study) and the **motivation source**
(real long-evidence RCA, ≤28k tokens, 37% over the H100 8k window). In the paper it appears only as: (a) a *motivation*
for long-context compression, and (b) an optional *case study*. **Its accuracy is not a paper claim** (base-model
ceiling ~0.21). All quantitative paper claims use the public benches + cross-model study.
