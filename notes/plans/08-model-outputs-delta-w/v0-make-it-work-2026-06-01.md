# Plan 08 v0 — Make-It-Work: Prioritized Backlog

> Generated 2026-06-01 after Phase J Gist baseline showed we lose at
> matched-$K$ on `categorical_niah` ($K{=}32$: Gist 0.34 vs ours 0.24
> 5-seed mean). Goal: get our wrapper above Gist on at least one
> training task × benchmark pair, then propagate.
>
> Format: every approach has **lift** (expected accuracy gain if it
> works), **cost** (engineer-hours + GPU-hours), and **info value**
> (what we learn even if it fails). Tiers ordered by
> $\frac{\text{lift} \times \text{info value}}{\text{cost}}$.

---

## Status snapshot (the gap to close)

| metric | Ours best | Gist | gap |
|---|---:|---:|---:|
| categorical_niah K=32 contains | 0.30 (seed42) | 0.34 | −0.04 |
| categorical_niah K=32 5-seed mean | 0.244 | n/a | — |
| categorical_niah K=32 exact_match | 0.22 (seed99) | 0.20 | +0.02 ✓ |
| numerical_niah d=1 K=32 contains | 0.18 | 0.26 | −0.08 |
| coding_niah K=32 contains | $\sim$0.95 (single 2026-05-30 run) | 0.00 | +0.95 ✓ |
| every other task | 0 | 0 | — |

Notes:
* We already beat Gist on `coding_niah` (40-bit answers); needs
  replication under current code path.
* `exact_match` is already roughly even on categorical; the loss
  is on `contains_match` because Gist makes shorter, more focused
  outputs that hit `contains` even when wrong about the exact class.
* The headline number to move is `categorical_niah K=32 contains`
  from 0.244 → above 0.34.

---

## Tier S — running NOW (Phase L, launches when Phase K ends ~17:15 PT)

These are the highest-leverage cheap interventions and they're
already queued. Listed here for completeness.

| # | name | lever | expected lift | cost | info value |
|---|---|---|---|---|---|
| S1 | `phaseL__apply_alpha_1` | `--apply-alpha-init 1.0` (current default 0.0) | **HIGH** — if α stays at 0 the read-side gate never opens; this is the single most likely structural bug | 25 min on 1 GPU | tells us if SFT-only can lift α; if yes the whole sft_only line works |
| S2 | `phaseL__lambda_div_05` | `--lambda-div 0.5` (was 0.1) | medium — geometry showed consec-cos 0.99+; stronger regularizer should break the plateau | 25 min on 1 GPU | tells us if the collapse is the bottleneck |
| S3 | `phaseL__steps_5000` | 5000 steps not 1800 | medium — 2026-05-30 winner used 2000 | 70 min on 1 GPU | tells us if it's a budget problem |
| S4 | `phaseL__steps_5000_alpha1_div05` | stack S1+S2+S3 | HIGH if any of S1–S3 helps | 70 min on 1 GPU | identifies whether they're additive |
| S5 | `phaseL__data_500` | n_items 100 → 500 | medium — overfit signal from 2026-05-30 (train_c=0.92 / eval_c=0.42 on interleave) | 90 min on 1 GPU | tells us if generalization is the bottleneck |
| S6 | `phaseL__interleave_div05` | interleave + α=1 + λ_div=0.5 | medium — interleave had the highest train_c, just collapsed on eval | 25 min on 1 GPU | tells us if read mode matters more than write |
| S7 | `phaseL__3stage_2000` | reproduce the 2026-05-30 SFT+OPD+RL@2000 single-seed 0.42 winner | reproducibility check | 30 min on 1 GPU | falsifies or confirms the original lucky number |

---

## Tier A — next batch (Phase M), cheap follow-ups + diagnostics

Run after Phase L lands; pick based on Phase L's pointers. All
single-seed for screening; promote winners to 5-seed in Phase N.

| # | approach | one-line | lift | cost | info value |
|---|---|---|---|---|---|
| A1 | **Log α_apply over training** | instrument `BPTTTrainer` to dump α every 50 steps; CSV + plot | — | 1 engineer-hr, 0 GPU | **critical** — confirms or refutes the α-stuck hypothesis Phase L is testing |
| A2 | **Toy 2-class K=2 test** | reduce categorical to 2 classes (sum / product), K=2 → does the wrapper learn at all? | — | 2 engineer-hr, 0.5 GPU-hr | **critical** — if even this fails the architecture has a bug |
| A3 | **Probe $m_T$ with a frozen linear classifier** | freeze wrapper, train 4-way classifier on $m_T$ → if AUC > 0.9 the memory IS encoding the class; the problem is the apply | — | 4 engineer-hr, 1 GPU-hr | **critical** — splits "write fails" from "read fails" |
| A4 | **Beam search at eval (k=4)** | currently greedy; categorical has 4 labels, beam=4 forces commit to each | small | 1 engineer-hr, 0 GPU-hr | tells us if first-token bias is masking a correct ranking |
| A5 | **Eval temperature → 0** | sample_temperature is 1.5 throughout; sample greedy=False at eval is a likely bug | small–medium | 0.5 engineer-hr, 0 GPU-hr | clean eval signal |
| A6 | **First-token-only forced eval** | constrain decoding to `{sum, product, max, concat}` only — measure pure classification accuracy | small | 2 engineer-hr, 0.5 GPU-hr | strips the natural-text confound |
| A7 | **K=128 big-arch with α=1 + λ=0.5** | take Phase L winner's recipe + push K | medium | 75 min on 1 GPU | does scaling K close the gap once the recipe is right? |
| A8 | **Multi-task joint training** | one wrapper on cat + num_d1 + cod jointly | medium | 70 min on 1 GPU | shared inductive bias may help each task |
| A9 | **Eval set 50 → 200** | tighter CIs — current 50-item eval has ±7pp noise | 0 (eval only) | 1 engineer-hr, 0.5 GPU-hr | reveals real-vs-noise lift in Phase L |
| A10 | **`residual` mode with α_apply=1.0** | 2026-05-30 collapse was caused by α stuck at 0; not yet retested with α=1 | medium | 25 min on 1 GPU | rules in/out the second apply mode |
| A11 | **Curriculum: easy → hard** | train on cat 2-class first 1000 steps, then cat 4-class, then numerical_d1 | medium | 80 min on 1 GPU | low-risk inductive-bias warm-up |
| A12 | **Confusion matrix per class** | for the best Phase L config, dump per-class precision/recall — is "sum" always confused with "max"? | — | 1 engineer-hr | targets the next architectural fix |

---

## Tier B — architectural changes (Phase N), needs a code PR

These require touching `mem_embedding.wrappers.MemoryEmbeddingWrapper`
or `llm_infra.train.BPTTTrainer`. Pick at most 3, prioritized by
what Phase L + A1–A3 diagnostics point at.

| # | approach | one-line | lift | cost | info value |
|---|---|---|---|---|---|
| B1 | **Per-channel learnable α_apply** | scalar α is a single bottleneck; per-channel = $K \cdot d$ params, still tiny | medium–high | 4 engineer-hr, 25 min on 1 GPU | does a fine-grained gate help? |
| B2 | **Learnable $m_0$ init** (vs zeros) | a learnable bias vector seeds the recurrence; gives the first chunk something to subtract from | low–medium | 2 engineer-hr, 25 min on 1 GPU | does init matter, or is the optimizer escaping zeros fine? |
| B3 | **Init $m_0$ from a sentinel token's embedding** | seed with the embedding of `<mem>` or `[CLS]` — borrows base's prior | medium | 3 engineer-hr, 25 min on 1 GPU | gives memory a sensible starting prior |
| B4 | **Auxiliary reconstruction loss** | extra head: decode each chunk from $m_t$; forces $m$ to actually carry chunk info | high if write is broken | 8 engineer-hr, 40 min on 1 GPU | regularizes write side; partially covered by A3 |
| B5 | **Contrastive loss across items** | $m_T$ from different inputs should differ; $m_T$ from same input + paraphrase should match | medium | 6 engineer-hr, 40 min on 1 GPU | targets the collapse mode directly |
| B6 | **Self-distill against full-context teacher's full distribution** | currently we KL on first-token only; KL over the full answer span carries more signal | medium | 4 engineer-hr, 30 min on 1 GPU | richer supervision |
| B7 | **GLU/SwiGLU in update FFN** | replace the 2× FFN with SwiGLU | low–medium | 2 engineer-hr, 25 min on 1 GPU | known better non-linearity |
| B8 | **RoPE on memory tokens** | memory tokens currently are position-less; RoPE gives them a slot identity | medium | 4 engineer-hr, 25 min on 1 GPU | might prevent token-permutation degeneracy |
| B9 | **Insert apply at an early layer** instead of only at last | currently last-layer-only; early layer leaves more layers to "mix in" | medium–high | 4 engineer-hr, 30 min on 1 GPU | classic Memorizing-Transformers finding |
| B10 | **Multi-layer apply** (last 4 layers, residual-add) | the memory is read at every of the last 4 layers — borrows the Memorizing Transformers recipe | high | 6 engineer-hr, 30 min on 1 GPU | strong precedent in literature |
| B11 | **Pretrain wrapper on plain LM loss** before SFT | 1-epoch LM loss over a generic corpus warms the wrapper up to "produce useful soft tokens" | medium | 8 engineer-hr, 4 GPU-hr | gives the wrapper a sensible prior |
| B12 | **GRPO instead of REINFORCE** | better variance reduction for the RL stage | small–medium | 6 engineer-hr, 1 GPU-hr | improves the RL leg if RL ends up mattering |
| B13 | **DPO from collected rollouts** | offline preference learning on (good, bad) pairs from sampled rollouts | medium | 10 engineer-hr, 2 GPU-hr | stable alternative to RL |
| B14 | **Hierarchical memory** (`K_global` + `K_chunk_local`) | split K into a slow channel and a fast per-chunk channel | medium–high | 12 engineer-hr, 30 min on 1 GPU | bigger arch change; only if simpler wins plateau |

---

## Tier C — bigger swings, only if Tiers S+A+B plateau

| # | approach | one-line | lift | cost | info value |
|---|---|---|---|---|---|
| C1 | **Q2 free-channel methodology** | profile base activation density, restrict wrapper writes to dormant channels | very high if the hypothesis holds | 2 weeks engineer-time, 180 GPU-h | own paper; structurally fixes the read-side bottleneck |
| C2 | **Hybrid wrapper: mem-embedding + mem-feature** | combine soft-prompt with hidden-state-hook intervention; both update the same $m$ | medium–high | 1 week engineer-time, 1 GPU-day | unlocks the mem-feature line we parked |
| C3 | **TokMem-style supervised pretraining** | pretrain wrapper on (chunk, answer) pairs from a 10k-item synthetic corpus before task SFT | high if undertraining is the issue | 3 days engineer-time, 1 GPU-day | follows the published recipe that works |
| C4 | **Hypernet → LoRA delta (mem-weight)** | switch to weight-space updates; the only way to genuinely "rewrite the base" | high but risky | 2 weeks engineer-time, 1 GPU-day | unparks mem-weight; bridges to Plan 08 full scope |
| C5 | **Bigger base** (Qwen3-14B / 32B) | maybe the wall sits at $\sim$10 bits because 8B can't read more from $K$ slots | medium–high | a few days engineer-time, 4 GPU-days | tests the cross-scale hypothesis directly |
| C6 | **Encoder swap**: BGE-large instead of base-as-encoder | maybe the base-as-encoder loses too much chunk detail | low–medium | 2 days engineer-time, 1 GPU-day | tests the encoder bottleneck |
| C7 | **External retrieval fallback** for hard cells | wrapper for easy, RAG for needles — concedes the wall but ships a useful product | n/a (concession) | 3 days engineer-time, 1 GPU-day | not a method win, but a deployable system |

---

## Tier D — explicitly NOT doing

* Sweep more hyperparameters without a diagnostic. We did this in
  Phases A–H and produced single-seed picks (sft_only=0.48,
  lr=3e-5=0.64) that did not replicate. Until Tier-A diagnostics
  point somewhere, more hyperparameter sweeping is negative-value.
* Switching to a totally different wrapper repo before we
  understand what's broken about this one.
* Adding more regularizers without first checking if the suspected
  collapse mode is still present at all under Phase L's α=1 fix.
* Training longer than 10000 steps without curve evidence that the
  loss is still going down.

---

## Decision flow

```
Phase L lands (8 jobs, ~3 wall-hr from K-finish)
   │
   ├─ Any job beats Gist (≥0.34 contains AND ≥0.40)?
   │     └─ YES → promote to BEST_RECIPE_L; 5-seed in Phase M;
   │             move to filling main table; spawn Tier A in parallel
   │             for further +10pp.
   │
   └─ All jobs <0.34?
         │
         ├─ Run Tier A diagnostics A1, A2, A3 (probe + toy + α-log)
         │     in parallel; total ~6 GPU-hr.
         │
         ├─ A3 says memory encodes class info (AUC > 0.9)?
         │     └─ YES → problem is read (apply) side → Tier B B9/B10
         │            (multi-layer / early-layer apply).
         │     └─ NO  → problem is write → Tier B B4/B5/B6
         │            (reconstruction / contrastive / better KL).
         │
         └─ A2 toy K=2 fails?
               └─ Stop. There's a fundamental bug in the wrapper
                  or trainer. Open a separate debug session;
                  don't run more sweeps until fixed.
```

---

## Tier-by-tier compute budget

| tier | jobs | wall-hr on 4×H100 | GPU-hr |
|---|---:|---:|---:|
| S (Phase L) | 8 | 3 | 12 |
| A (Phase M screening) | ~10 | 5 | 20 |
| A (diagnostics A1-A3) | n/a | 1 (engineer) | 6 |
| B (Phase N PRs, screening) | ~6 winners | 3 | 12 |
| C (each, separately) | varies | days | 30-180 each |

Sequential through Tiers S+A+B: ~12 wall-hr + 1 engineer-day.
Headline pass should fit in 24 hr.

---

## Companion files

* `v0-gist-vs-ours-2026-06-01.md` — the head-to-head finding that
  triggered this list.
* `v0-results-2026-05-30.md` — original results, including the
  geometry-panel observations (α stuck, consec-cos = 0.99) that
  motivate S1, S2, A1.
* `../../mem-test/mem-embedding/summary/matrix.md` — the live
  results matrix Phase L will fill in.
* `../../mem-test/mem-embedding/k8s/sweep/build_sweep.py` —
  `PHASE_L` definition for Tier S.

---

## Addendum — Phase M shipped 2026-06-01 evening

Discovered while implementing Phase L that the `xattn` readout
`_ApplyTimeReadoutBlock` hard-coded `alpha_init=1.0` with no CLI
knob, so the Phase L jobs that paired `--apply-alpha-init 1.0`
with `combines=xattn` were no-ops on the read side. Phase M
ships the missing code (configurable read-side α, per-channel α,
new `xattn_residual` combine, reconstruction-loss head, no-
context baseline, toy K=2 dataset, m_T probe script,
alpha-logging CSV) and 12 jobs that actually exercise those
levers.

Phase M waiter polls for Phase L's run dir → drains it → launches
Phase M with the auto-aggregator. The waiter is resilient to
"Phase L hasn't started yet" — it polls every 60 s for the dir
to appear.

Code shipped today, summary:

| file | what |
|---|---|
| `mem-embedding/src/mem_embedding/wrapper.py` | `xattn_residual` combine; readout α CLI-configurable + per-channel; optional reconstruction head |
| `llm-infra/src/llm_infra/strategies.py` | `NoContextStrategy` (the literal "no wrapper" lower bound) |
| `llm-infra/src/llm_infra/datasets.py` | `generate_categorical_niah_k2` (1-bit toy A2 sanity) |
| `llm-infra/src/llm_infra/train.py` | reconstruction loss in `BPTTTrainer`; α-log CSV |
| `mem-embedding/scripts/train_smoke.py` | new CLI flags wired through; `readout_alpha_after_train` in summary |
| `mem-embedding/scripts/probe_memory.py` | new — m_T linear probe (A3) |
| `mem-embedding/scripts/run_sweep.py` | boolean-flag handling in `_build_cmd` |
| `mem-embedding/k8s/sweep/build_sweep.py` | `PHASE_M` (12 jobs) + `--only-phase-m` |
| `mem-embedding/k8s/run-phase-m-waiter.sh` | poll-for-Phase-L-dir + drain + launch |

Smoke-tested on the pod (CPU + tiny LM) — `xattn_residual`
trains, `no_context` baseline appears in the summary, alpha-log
CSV is written, reconstruction loss is added.

Decision rule update: Phase M jobs must beat `no_context_contains
+ 0.05` (the lower bound) before being considered "the wrapper
is doing something." Above that, must beat the Phase J Gist
numbers (0.34 contains / 0.20 exact) to be promoted to BEST.
