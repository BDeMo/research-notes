# v1.7 master experiment plan — ~180 GPU-hours (20 h × 9 GPU), verify one lever at a time

> ⛔️ **SUPERSEDED (2026-06-13):** this plan and its result numbers predate the train/eval **leakage fix**. The
> current clean plan + results are **[`../../results-v1.7.3/results-v1.7.3.md`](../../results-v1.7.3/results-v1.7.3.md)**,
> [`../../results-v1.7.3/robustness-plan.md`](../../results-v1.7.3/robustness-plan.md), and
> [`../../results-v1.7.3/reviewer-response.md`](../../results-v1.7.3/reviewer-response.md). Kept for history.

**Two north-star metrics** (every stage reports both):
- **M1 = compression effectiveness**: `comp` (compress-only acc) and the **gap to full** (`full − comp`); "don't lose too
  many points" = shrink this gap. Track vs `no_ctx` (must clear), the **faithful baselines**, and `full`.
- **M2 = gating quality**: **held-out AUROC + best-F1** of the compress-vs-fallback gate (intrinsic signals AND the
  trained discriminator), plus the **realized gated-acc vs always-full** and the do-no-harm precision.

**Method invariants** (don't change): frozen base is the *principle* but we MAY add a small read-adapter; encoder→memory
→inject→frozen base→answer; per-input gate→fallback-to-full. **Free to vary** (per stage): encoder (depth/width/arch),
init, training HPs (lr/sched/steps/data), our design HPs (K, lam-weights, condition, min-dev), **injection strategy**
(input-embed / per-layer KV-cache / +LoRA-read), and **gating design** (signals/discriminator arch/layer/pooling/sup-vs-adv).

**Protocol.** Sweep model = **Qwen3-8B** (fast, standard attn ⇒ KV-cache works); confirm winners on **Qwen3.5-9B**.
Tasks = in-task **squad_v2 + hotpot_qa** (full ≫ no_ctx ⇒ real headroom); best configs also get cross-task. Single seed
for sweeps, **3 seeds** for chosen configs. Run cost ≈ 0.25 GPU-h (Qwen3-8B, 96 items/800 steps) → 0.6 (1500 steps + signals).

---

## DOMAIN PIVOT (2026-06-12, per user): validate on **tool-use + RCA first**, other domains later
- **Primary benches**: `bfcl_simple` (tool, full≈0.99 ≫ no_ctx≈0.01 = max headroom) + `rca_openrca` (ops/MC, full≈0.43).
  (toolace/apibank as extensions.) **Why**: the agentic use case + the biggest compress-vs-full gap in the matrix.
- **Data location**: all on **sam-dev** — RCA = local file `/home/devuser/datasets/openrca_built/cases.jsonl` (`MEM_RCA_OPENRCA`);
  bfcl/toolace/apibank in `/root/.cache/huggingface/hub`. ray/test lack it ⇒ copy RCA jsonl + bfcl dir to use 9 GPUs.
- **Model**: Qwen3-8B (standard attn ⇒ faithful Cartridge KV-cache works) on all 3 pods.
- **Re-target**: faithful baselines + the HP/inject/encoder/gating sweeps all run on bfcl+rca first; squad/hotpot become
  the "other domain" follow-up.

## KEY CALIBRATION: faithful baselines ALSO fail on span-QA (justifies the tool/RCA pivot)
Faithful Cartridge (per-layer KV-cache) + faithful Gist (LoRA+gist-mask), Qwen3-8B, 96 items / 1500 steps, in-task:

| bench | method | full | **compress** | no_ctx | compress beats no_ctx? |
|---|---|---|---|---|---|
| squad_v2 | faithful Cartridge | 0.662 | **0.168** | 0.180 | ✗ (below) |
| squad_v2 | faithful Gist      | 0.795 | **0.077** | 0.205 | ✗ (well below) |
| hotpot_qa| faithful Cartridge | 0.494 | **0.221** | 0.233 | ✗ (≈ no_ctx) |
| hotpot_qa| faithful Gist      | 0.552 | **0.131** | 0.255 | ✗ (below) |

→ It is **not just our GCM** — published compressors re-implemented faithfully *also* cannot beat no-context on
span-extraction QA at this data/step budget. Span-QA needs verbatim spans the K-token bottleneck destroys.
This is exactly why we pivot: tool-use/RCA reward *semantic* recall (which function / which service), the regime
where compression is supposed to win (bfcl full≈0.99 vs no_ctx≈0.01). Targets now come from tool/RCA, not QA.

## PIVOTAL RESULT: compression WORKS on tool-use, FAILS on RCA (faithful baselines, Qwen3-8B, in-task)
| bench | method | full | **compress** | no_ctx | verdict |
|---|---|---|---|---|---|
| bfcl_simple | faithful Cartridge | 0.990 | **0.365** | 0.010 | ✅ compress ≫ no_ctx (36×) |
| bfcl_simple | faithful Gist      | 0.990 | **0.333** | 0.062 | ✅ compress ≫ no_ctx |
| rca_openrca | faithful Cartridge | 0.469 | **0.146** | 0.198 | ❌ compress < no_ctx (memory misleads) |
| rca_openrca | faithful Gist      | 0.344 | **0.135** | 0.167 | ❌ compress < no_ctx |

**Narrative (== user's two axes):**
- **tool-use = the "stronger compressor" regime.** Compression recovers ~36% of the full−no_ctx gap and is
  far above no-context. Levers (GCM+LoRA, encoder, K) push `compress` toward 0.99 ⇒ *fewer real fallbacks needed*.
- **RCA = the "detection" regime.** Even faithful baselines compress BELOW no-context (the K-token bottleneck on
  long logs produces misleading memory). Here the **gate must detect failure and fall back to full** (0.47); a good
  gate makes `gated_acc ≈ full` while still compressing the easy items. This is where fallback-precision earns its keep.
- Net: the robustness selling point = (compress when it helps on tool) + (detect-and-fallback when it hurts on RCA).

## RUNNING (2026-06-12, all 9 GPUs on tool/RCA, Qwen3-8B)
- sam-dev (4): faithful baselines `faith_{cart,gist}_{bfcl,rca}` (in-task).
- ray (4): GCM HP sweep (19 cfg) on `bfcl_simple` → `out/hp_bfcl`.
- test (1): GCM HP sweep (19 cfg) on `rca_openrca` → `out/hp_rca`.
- Data: RCA `cases.jsonl` copied to ray/test `/mnt/persist/datasets/openrca_built/`; bfcl pulled via `HF_HUB_OFFLINE=0`.

> **Mechanism + flowcharts**: see [`../gcm-lora-mechanism.md`](../gcm-lora-mechanism.md) (encoder, K memory tokens, 5 losses, compress/fallback gate).

## E0 RESULTS (2026-06-12, Qwen3-8B, in-task) — the two-regime story, corrected

### tool-use (bfcl_simple) — compression WORKS + gate WORKS (the win)
| method | compress | gate AUROC | gAcc | note |
|---|---|---|---|---|
| faithful Cartridge | 0.365 | — | — | published baseline |
| faithful Gist | 0.333 | — | — | published baseline |
| vanilla GCM (best HP = hp_s2400) | 0.354 | 0.820 | — | no-LoRA ceiling ≈ faithful |
| **GCM + LoRA + data(384)** | **0.583** | **0.880** | **1.000** | **beats all; best(compress,full)=1.000** |

- `compress` 0.583 ≫ no_ctx 0.010, and **beats faithful Cartridge/Gist**. The decisive lever was **more data**
  (96→384 items: 0.354→0.583), confirming the overfit diagnosis. LoRA rank r8≈r16≈r32 (~0.35–0.39); `data` ≫ `rank`.
  `lam_dev=0` (devoff) HURT (0.219) ⇒ the full-base deviation anchor helps.
- **Gate**: reconstruction signal `neg_recon` → AUROC 0.880, F1 0.862, **gАcc 1.000** (= oracle). Both axes land.

### RCA (rca_openrca) — compression does NOT help, but do-no-harm holds
| method | compress | full | best(compress,full) | gate AUROC | gAcc |
|---|---|---|---|---|---|
| faithful Cartridge | 0.146 | 0.469 | — | — | — |
| faithful Gist | 0.135 | 0.344 | — | — | — |
| GCM+LoRA+data | 0.219 | 0.469 | **0.469 (= full)** | ~0.46–0.58 (random) | **0.469 (= full)** |

- `best(compress,full) = full exactly` ⇒ **compression solves ZERO items full misses** (zero complementarity);
  even an oracle gate = full. Gate signals AUROC ≈ 0.5 (unlearnable here). The realizable gate correctly
  **always falls back → gAcc = full = 0.469** (never worse than base). RCA's value is *robustness* (do-no-harm),
  not compression savings. Likely cause: RCA answers need verbatim log lines the K-token bottleneck destroys.

### KEY ABLATION: is *data* the lever, or *LoRA×data*? (bfcl compress) — they are SYNERGISTIC
| base | 96 items | 384 items (matched 1600 steps) |
|---|---|---|
| **frozen** | ~0.20–0.35 | **0.240** (got WORSE with more data) |
| **+ non-merged LoRA** | 0.354 | **0.583** |

- Matched pair (384 items / 1600 steps, only LoRA differs): **frozen 0.240 vs LoRA 0.583**.
- **Frozen base: data is NOT key** — 96→384 items did not help (slightly hurt). The bottleneck is the base's ability
  to *read* M (fixed when frozen), not the encoder's ability to *produce* M, so more compression data can't move it.
- **Data only pays off with LoRA**: once the base can adapt to read M, 96→384 lifts 0.354→0.583. LoRA opens the
  read channel; data fills it. ⇒ This retro-justifies the user's "non-merged LoRA" design as the actual unlock
  (the old frozen-base principle was the ceiling). (`hp_data1000` frozen running to confirm no-scaling.)

### Net narrative (== user's two axes)
- **Axis-1 stronger compressor**: achieved on tool-use (0.583, beats baselines via LoRA+data). RCA: compression
  can't be made to help in this setup ⇒ E1 RCA levers = bigger K / hierarchical / log-selective memory, else accept
  RCA as a fallback-dominated (robustness) regime.
- **Axis-2 detection**: achieved on tool-use (AUROC 0.88, gAcc 1.0). On RCA signals are random but the
  fallback floor guarantees gAcc = full ⇒ the do-no-harm guarantee carries the hard regime.

## PER-METHOD CROSS-TASK/DOMAIN TRANSFER (Qwen3-8B, compress; * in-task ~ cross-task/in-domain blank cross-domain)
GCM (5/6 anchors; toolace pending), Cartridge/Gist (bfcl/rca anchors done, rest filling in). Built by
`mem-test/v17-results-backup/build_per_method.py` from the local backup.

In-task diagonal compress (no_ctx / full in parens):
GCM+LoRA:    bfcl 0.531* | rca 0.135* | squad 0.267* | hotpot 0.236* | apibank 0.000* | toolace (rerun)
Cartridge:   bfcl 0.375* | rca 0.146* | squad 0.191* | hotpot 0.222* | apibank 0.000* | toolace 0.062*   (COMPLETE 6/6)
Gist:        bfcl 0.104* | rca 0.104* | squad 0.115* | hotpot 0.160*   (apibank/toolace pending)

Relation summary — avg compress (lift over no_ctx)  [Cartridge now complete 6/6; GCM 5/6; Gist 4/6]:
| method | in-task | cross-task/in-domain | cross-task/cross-domain |
|---|---|---|---|
| GCM+LoRA  | 0.234 (+0.120) | 0.090 (+0.015) | 0.088 (−0.011) |
| Cartridge | 0.166 (+0.071) | 0.057 (−0.001) | 0.095 (−0.014) |
| Gist      | 0.121 (−0.019) | 0.048 (−0.022) | 0.052 (−0.028) |

**Verdict (all 3 methods agree): compression lifts ONLY in-task; cross-task ≈ 0; cross-domain NET-NEGATIVE.**
These are per-corpus methods — the compressor learns a corpus-specific code that does not generalize. On comparable
anchors (bfcl/rca) GCM > Cartridge > Gist in-task (bfcl 0.531 / 0.375 / 0.104). ⇒ a deployed system needs per-corpus
training + the gate/fallback, not one universal compressor. (Averages firm up once baseline + toolace rows complete.)

## REMAINING NECESSARY EXPERIMENTS (plan 2026-06-12; queue after the current ablation/HP sweep drains)

**Currently draining (no action needed):** tool 7×7 (GCM/Cart/Gist), N/K/LoRA/init/conditioning/decoder/loss ablations,
HP lr×steps sweep, data-scaling (DONE — pooling *dilutes*, peaks n=500@0.479 then drops; per-corpus, not data-bound).

### P0 — Gate & fallback evaluation (THE core contribution — currently the weakest-covered)
- **G1 Gate-signal table** *(analysis, 0 runs)*: per tool bench (simple/multiple/live_multiple/rca), AUROC / gAcc /
  fallback-rate / F1 for each signal (neg_recon, conf, margin, dlogit, dcode, mnorm) + the supervised combiner. Which
  signal is the operative gate, and how realizable (held-out) is gAcc.
- **G2 Adversarial gate** *(~6 runs)*: GCM `lam_adv ∈ {0.25,0.5,1.0}` on bfcl_simple + bfcl_multiple; does the learned
  discriminator gate (S_low/S_conf/S_high) beat intrinsic `neg_recon`? (We have lam_adv=0.5 on simple queued; extend.)
- **G3 Coverage–accuracy (efficiency) curve** *(analysis)*: sweep gate threshold → (compression-rate, gАcc) per bench;
  mark the do-no-harm operating point + break-even fallback rate. This is the "fewer real fallbacks / not too
  conservative" axis the project is about.

### P1 — Best-config consolidation
- **B1** *(~3 runs)*: assemble the ablation winners (e.g. K=32 + best LoRA-rank + best loss-weights) into one config →
  run in-task on the 3 *selection* benches (simple/multiple/live_multiple) + its gate → the headline "best GCM".

### P2 — Agentic robustness (the selling point — needs a little code)
- **M1 Multi-turn / iterative-memory eval** *(code + ~4 runs)*: reuse/refresh M across turns (the agentic long-context
  use case). Harness is single-turn today → small code task to thread memory across turns, then eval on a multi-turn
  tool bench. Necessary if we claim "agentic tool-use".
- **S1 Multi-seed (×2)** *(~6 runs)*: on the headline cells (best-config bfcl_simple/multiple compress + gАcc) for
  significance (single-seed variance ≈ ±0.04).

### P3 — Cost / efficiency (analysis, 0 runs)
- **C1**: compression-ratio (K tokens vs full ctx) + FLOPs + the realized saving = ratio × (1−fallback_rate); the
  break-even fallback rate vs the faithful baselines. Ties the gate's fallback-rate to actual compute saved.

**Priority order: P0 (gate is the contribution) → P1 → P3 → P2.** P0/P3 are mostly analysis on existing records;
G2/B1/S1 are the new runs (~15, ~6 GPU-h on the 5-GPU fleet). M1 needs the multi-turn code first.

## Live findings (执行中更新 / data-driven from training logs)
- **[2026-06-12] Overfitting confirmed from loss dynamics.** Vanilla GCM **gold-CE** task loss `2.69→0.28→0.04→0.11`
  on 96 items (memorizes the train golds) while eval `comp`=0.11–0.17 → **severe train/eval overfit**. Gist+gold-CE
  same (`ce→0.0006`). **Cartridge+distillation** stays healthy (`distillKL≈0.10`, no collapse). ⇒ **E2 reprioritized:
  the fix is (a) more data + (b) distillation objective + (c) regularization — NOT just lr/steps.** Added anti-overfit
  HP configs (n_items {384,1000} + distill-heavy lam_distill1/lam_task0.3). Distill-from-STRONG-teacher (LoRA-full≈0.88)
  is the headline E2 idea.

## Sequenced stages (proceed to the next only if the current lever helps M1/M2)

### E0 — Foundations & fair targets   [~15 GPU-h]  (RUNNING)
- **faithful baselines**: Cartridge (per-layer trainable KV-cache) + Gist (LoRA + gist-mask), squad/hotpot, 1500 steps.
- **tuned vanilla GCM**: HP sweep (15 cfg: lr/steps/lam_rec/lam_dev/lam_distill/recon-m-only + N/K/n_chunks).
- **Gate:** know the real targets; is a *tuned* vanilla GCM ≥ no_ctx? Do faithful baselines ≈ full?

### E1 — Injection strategy (the biggest structural lever)   [~30 GPU-h]
- **GCM-KV**: encoder reads out per-layer **KV** (reuse the new `runtime.kv_logits` path) instead of a layer-0 soft prefix.
- grid: inject ∈ {input-embed (ctrl), KV-all-layers, KV-half, input-embed+LoRA-read} × K ∈ {32,64,128} × init {prefill, random}.
- ~24 runs. **Gate:** pick the injection that most lifts `comp`. (Expectation: KV-cache ≫ input-embed, per faithful Cartridge.)

### E2 — Training quality (you flagged this as critical)   [~45 GPU-h]
- **objective**: gold-CE · **full-distribution KL distillation** · lossless reconstruction (M-only decoder) · combos, with a
  weight sweep (lam_task/lam_distill/lam_rec/lam_dev). 
- **teacher**: distill from a **strong teacher** (LoRA-adapted full ≈ 0.88) instead of the weak frozen base — likely a big win.
- **schedule/data**: steps {800,1600,3200} · lr + **warmup/cosine** · n_items {96,384,1000} · grad-accum/batch · EMA.
- ~50 runs. **Gate:** best recipe for `comp` (combined with E1's best injection).

### E3 — Encoder & initialization   [~25 GPU-h]
- N (depth) {4,8,12,16,24,full} · readout {parallel last-K, autoregressive, **cross-attention pooling**} · init {copy,
  random, **reconstruction-pretrained**} · condition {Mq, M0} · min-dev {0, .05} · m_proj/normalization.
- ~30 runs. **Gate:** best encoder/init for `comp` (stacked on E1+E2 winners).

### E4 — Gating design (M2; only once comp clears no_ctx)   [~35 GPU-h]
- signals: intrinsic (neg_recon, ΔCode, ΔLogit, conf/margin/entropy, mnorm) · **trained discriminator** (layer-ℓ sweep
  {6,12,18,24}, arch {MLP, attn-pool}, pooling) · **supervised multi-feature gate** (CV logreg/MLP) · adversarial-gate.
- calibration; precision-first do-no-harm; **gated-acc vs always-full**; cost-coverage curves.
- ~40 runs. **Gate:** best **held-out AUROC/F1** + realized gated-acc > always-full.

### E5 — Scale, seeds, cross-task, model-confirm   [~25 GPU-h]
- best end-to-end config: **3–5 seeds** (CIs), **cross-task/domain** matrix, **Qwen3.5-9B confirm** (if KV-cache maps; else
  Qwen3-14B), cost/FLOPs-coverage frontier.
- ~30 runs. **Gate:** robust headline numbers for the paper.

### Buffer / re-runs / failed-cell retries   [~5 GPU-h]

**Budget total ≈ 15+30+45+25+35+25+5 = 180 GPU-h.**

---

## Decision logic (the "one-by-one" gates)
- After **E0+E1**: if KV-injection (or tuned vanilla) makes `comp ≥ no_ctx` and within ~0.1 of a faithful baseline →
  compression is viable; continue to E2/E3 to close the gap to `full`.
- If E1 still ≈ no_ctx → the learned compressor is the wall; pivot weight to the **faithful baselines + gated-LoRA (P1)**
  story (compression as the boundary case) and spend the remaining budget on E4 (gating) over the faithful Cartridge.
- E4 only starts once *some* method has `comp` meaningfully > no_ctx (else the gate is unlearnable, per B1).

## Runners / how it executes
- `queue_hp.py` (HP), `queue_sol.py` (solutions), faithful baselines (`out/faithful`), + a new `queue_inject.py` for E1
  (injection grid) and reuse for E2/E3. One worker/GPU, GPU-free-aware, resumable; results auto-fill the tables here.
- Models: Qwen3-8B on all 3 pods for the KV/injection work; Qwen3.5-9B reserved for confirmation.
