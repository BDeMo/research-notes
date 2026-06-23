# v1.8.0 — 160 GPU-hour experiment + ablation plan (8 GPU × 20 h)

> **Goal:** finish the paper's evidence base. Budget = **8 experiment GPUs** (d1525×4, d1530×2, d1420×2); **test pod = demo only**.
> Cost model: bfcl/short ≈ 1 GPU-h, mid ≈ 2 GPU-h, long-ctx (over-window) ≈ 3 GPU-h. ~64 cells fit 160 GPU-h.
> **Thesis alignment (README/summary-matrix):** the contribution is the **do-no-harm GATE** (cross-model, boundary, by-construction), *not* the compressor. So the plan front-loads (a) the main table OURS+gated, (b) the reviewer-mandatory **baselines** the v1.8.0 harness still lacks, (c) the **3-way adaptive gate** selling point.

## Priorities (P0 = must-have for submission; P3 = nice-to-have)

### P0 — Main table: OURS (compress) + gated, per base × bench  (~35 GPU-h)
- Bases: **Qwen3-8B, Qwen3.5-9B** (primary). Config = per-base HP-tune winner (Phase-1, running).
- Benches (7): bfcl_live_multiple, toolace, squad_v2, hotpot_qa, narrativeqa, quality, rca_openrca. In-task (train b → eval b).
- Columns: no_ctx / full / **compress** / **gated-acc** (+ gate AUROC/F1).
- **Long/over-long benches (quality, narrativeqa, hotpot)** use the **recurrent over-window** (`GCM_ENC_MAXCTX` ≫ `GCM_MAXCTX`) — fixes the truncation-degeneracy that made quality ≈ guessing.
- 14 cells.

### P1 — Reviewer-mandatory BASELINES (the main-table columns v1.8.0 lacks)  (~50 GPU-h)
1. **SFT-LoRA (forgetting)** — standard LoRA fine-tune on each bench, eval no_ctx+full → show every QA drops below no-ctx (C1). 2 bases × 5 QA = 10 cells. **[needs code: SFT mode]**
2. **TARG gate baseline** — training-free full-ctx prefix-logit margin gate; compare gated-by-ours vs gated-by-TARG vs ours⊕TARG. **Eval-only add-on, ~free.** **[needs code: TARG signal in eval]**
3. **Cartridge-lite (prefix-tuning)** + **Gist-lite (gist tokens)** — frozen-base memory baselines. 2 bases × 4 benches × 2 methods = 16 cells. **[needs code: prefix/gist memory modes]**

### P2 — The GATE selling point (paper thesis) + cross-model  (~27 GPU-h)
1. **3-way adaptive gate (mem ↔ full ↔ base)** + **multi-level over-window fallback** (回退一格 ladder) — the compression+fallback contribution. Eval on trained OURS + a **level-dropout**-trained variant. ~4 train + eval. **[needs code: gate3/level eval + level-dropout train]**
2. **Cross-model gate transfer** — default config + gate on **Ministral-8B, Qwen2.5-7B** (+ the 2 primary) × {bfcl, squad, rca_openrca} → "gate transfers across families" (C3/C4). ~6 cells.

### P3 — Ablations (v1.8.0 axes) — strengthen "compressor is capacity-bound; gate carries it"  (~45 GPU-h)
- **Module flip-one (Qwen3.5 anchor=bfcl):** norm{off,hard,learn,manifold}, recon{0,0.5,1}, dev{0,0.05}, distill{0,0.5,1}, K{16,64,256}, lora{0,32,64}, depth{1,half,full}, **recurrent{on,off}**, contrast{0,0.5}. ~12 cells.
- **Over-window recurrent ablation:** `GCM_ENC_MAXCTX ∈ {off, 16k, 32k}` × {quality, narrativeqa, rca_openrca} → quantify the long-context fix. ~9 cells.
- **Curriculum + eff-batch** ablation (shuffle vs +curriculum vs +per-domain-warmup) on the all-mix. ~3 cells.

## Coding needed (build before launch; review after)
| # | piece | for | size | priority |
|---|---|---|---|---|
| 1 | **TARG gate signal** in eval (full-ctx prefix-logit margin) + report gated-by-TARG / ours⊕TARG | P1.2 baseline | S | **P1** |
| 2 | **3-way + multi-level fallback eval** (`mc_loglik_adaptive`: mem→full→base ladder by gate signal) | P2.1 selling point | M | **P1** |
| 3 | **SFT-LoRA mode** (`GCM_SFT=1`: train base-LoRA, no compressor; eval no_ctx/full) | P1.1 forgetting | M | **P1** |
| 4 | **level-dropout** training (random raw-suffix boundary) | P2.1 | M | P2 |
| 5 | **Cartridge-lite / Gist-lite** memory modes | P1.3 | M-L | P2 |

## Execution waves (each ~ fills 8 GPUs)
- **Wave A (now, running):** Phase-1 HP-tune (P0 prereq). → pick config/base.
- **Wave B:** P0 main table (14 cells) + start P1.1 SFT (uses freed GPUs).
- **Wave C:** P1.3 Cartridge/Gist-lite + P2.2 cross-model.
- **Wave D:** P3 ablations + P2.1 gate3/level (eval-heavy, cheap) folded in on spare GPUs.
- Eval-only add-ons (TARG, gate3) ride on every cell's eval (≈ free).

## Risks / guards
- **OOM on long-ctx**: over-window encode is memory-bounded (detached chunks); full-baseline truncates to MAXCTX. Keep recon=0 on 80 GB cards.
- **eff-batch ≥ 8** everywhere (DDP or accum); shuffle on; curriculum for mixes.
- **Archive**: every cell writes `out/<base>/<TAG>.json` + adapter; consolidate to repo `archive/` after each wave.
- **All in tmux/setsid**; per-GPU autonomous queues so 20 h runs unattended.
