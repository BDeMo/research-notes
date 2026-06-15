# Results v1.7.3 (Qwen3-8B) — clean re-run — 2026-06-13

**This supersedes [`../results-v1.7/`](../results-v1.7/results-v1.7.md) (ARCHIVED).** v1.7 had a **train/eval data-leakage bug** that inflated every tool result; v1.7.3 is the leak-free redo. Methodology: [`experimental-setup.md`](experimental-setup.md) · mechanism + gate: [`gcm-lora-mechanism.md`](gcm-lora-mechanism.md) · one-page brief: [`../summary-matrix-v1.7.3.md`](../summary-matrix-v1.7.3.md).

> 🔬 **v1.7.4/v1.7.5 superseded this for the long-context + scaling work — see [`../results-v1.7.5/results-v1.7.5.md`](../results-v1.7.5/results-v1.7.5.md).**
> Verdicts (negative, but clean): (1) a **length-adaptive (chunked) memory** does NOT rescue long context (ties trivial
> `trunc`, never beats `full`); (2) **no scaling** — avg-acc flat across 24× FLOPs / data / depth, and capacity K does
> not scale either; (3) the soft memory is **OOD** (nearest-token cos ≈ 0.10). Together these **confirm compression is
> capacity-bound** → contribution = the **do-no-harm gate** + minimize-scale. Also: **faithful** 2025 baselines
> (ICAE/AOC/Beacon/500x/ComprExIT/LCC) + **stable training** (grad-accum/warmup/cosine/val). Paper: [`../paper/main_v1.7.5.tex`](../paper/main_v1.7.5.tex).

---
## 0. What changed from v1.7 (why a new version)

A code review found that the harness sampled **eval** items with `seed+1` while the per-corpus 70/30 split was applied *after* a **seed-dependent shuffle** — so "first 70%" (train) and "last 30%" (eval) overlapped. Measured leakage: **bfcl_simple 71%, bfcl_multiple 75%, glaive 58%, hermes 39%, apibank ~100%, musr 96%** (toolace 5%; QA benches 0% — they use HF-native splits). The inflation was severe:

| bfcl_simple config | v1.7 reported (all 96) | seen (leaked) | **UNSEEN (honest)** |
|---|---|---|---|
| lr1e-4/3000 "best" | 0.760 | 1.000 | **0.179** |
| LoRA32 | 0.635 | 0.838 | **0.143** |
| LoRA0 | 0.396 | 0.456 | **0.250** |

So the v1.7 "compression works on tool-use" headline and the "LoRA is the unlock" claim were **mostly memorization of leaked items** and are **retracted**. (On unseen data LoRA32 0.14 < LoRA0 0.25.)

**Fixes applied (all in code, synced to pods):**
1. **Leakage** — `llm_infra/datasets.py::_disjoint_split`: a **seed-independent** 70/30 partition (fixed partition seed; run seed only orders *within* a split) on all 6 manual-split generators (bfcl/apibank/toolace/hermes/glaive/rca); Glaive query-dedup; and a content-hash guard in `gcm/data.py::load_items` for single-split benches (musr/ruler/categorical_niah). **Verified 0–1% overlap after the fix.**
2. **Gate optimism** — `svc/disc_gate.py::gated_acc_cv`: the reported `gAcc` is now **5-fold cross-validated** (threshold fit on train folds, applied to the held-out fold). The old in-sample `gated_acc` is kept only as an optimistic ceiling.
3. Latent `sp`-undefined crash in `generate_apibank` fixed.

**Code-review PASSes (unchanged, verified correct):** `full_ctx`/`no_ctx` are scored with base-LoRA **OFF** (exact base — uncontaminated fallback); `tool_acc` uses **word-boundary** name matching; gate `signals()` read only (context, query, memory) — **no gold/label peeking**; `decision._arrays` builds `nw`/`nfull`/`y` correctly and `acc_best` is the labeled oracle.

---
## 1. Experiment plan (clean) — tune in-task first, THEN transfer

> **Directive (scope of the v1.7.3 redo):** *Re-run the experiments. Do **all** tool benchmarks — **not just BFCL**. Then do **in-task, cross-task, and cross-domain**. Tune the **in-task hyperparameters first**.*

We follow it literally: **(a)** all tool benchmarks (13 BFCL categories + API-Bank + ToolACE + Hermes + Glaive) + RCA (ops), not just BFCL; **(b)** **Phase 1 tunes the in-task hyperparameters per bench first**, then **(c)** **Phase 2** builds the full **in-task / cross-task / cross-domain** transfer matrix using each bench's best in-task config.

All **tool benchmarks** (not just BFCL), across **5 teams**: BFCL (Berkeley), API-Bank (Alibaba), ToolACE (Huawei), Hermes (Nous), Glaive (Glaive AI) + RCA (ops). Two phases:

- **Phase 1 — in-task HP tuning** (per bench, train==eval, leak-free): sweep the levers that mattered — base-LoRA rank, lr×steps, K — and pick the best in-task config **per bench**. Grid per bench (≈5 configs): `base (L16/K128/lr5e-5/1600)`, `+steps (lr1e-4/3000)`, `+lora (L32)`, `+both (L32/lr1e-4/3000/K64)`, `L0 (control)`.
- **Phase 2 — transfer matrix** (using each anchor's best config): **in_task** (train==eval), **cross_task_in_domain** (tool→other tool), **cross_task_cross_domain** (tool→QA/ops). Baselines Cartridge + Gist on the same matrix.

Every cell reports: `no_ctx` · `full` · `compress` · **`+fallback gAcc` (held-out CV)** · **`fallback%`** · `best(oracle)` · gate `AUROC / F1 / P / R`.

**Compute:** ~92 jobs queued on `free` (4×H100) + `test` (1 GPU); per-GPU `tmux` runners drain the file-queue. ETA ≈ 15–20 GPU-h.

---
## 2. Results — IN-TASK (Phase 1, CLEAN)

Qwen3-8B, in-task (train==eval, leak-free). We sweep a 5-config HP grid per bench (`base`=L16/K128/lr5e-5/1600 · `steps`=L16/K128/lr1e-4/3000 · `lora32`=L32/K128/lr5e-5/1600 · `combo`=L32/K64/lr1e-4/3000 · `lora0`=no-LoRA), pick **one recommended config** by best average compress on the value-add benches (§2.3 ⇒ **`combo`**, though all configs are within noise), and report it in §2.1. `gAcc_cv` = held-out 5-fold gated accuracy; `fallback%` at the gАcc-optimal point; gate `F1/precision/recall` = in-sample best-F1 (optimistic — read **AUROC** as the honest gate quality); `y=1[compress≥full]`; signal `neg_recon`. Records: `out/clean/p1/<bench>__<cfg>`.

### 2.1 In-task summary — recommended config `combo` (LoRA32 / K64 / lr1e-4 / 3000 steps)
| bench | N | no_ctx | full | compress | gAcc_cv | fallback% | gate F1 | precision | recall | AUROC | best |
|---|---|---|---|---|---|---|---|---|---|---|---|
| glaive¹ | 96 | 0.333 | 0.990 | 0.771 | 0.958 | 85% | 0.888 | 0.798 | 1.000 | 0.588 | 0.990 |
| bfcl_live_multiple | 96 | 0.010 | 0.906 | 0.719 | 0.885 | 78% | 0.882 | 0.816 | 0.959 | 0.749 | 0.948 |
| bfcl_live_simple | 78 | 0.013 | 0.910 | 0.526 | 0.923 | 85% | 0.808 | 0.689 | 0.977 | 0.715 | 0.974 |
| hermes | 96 | 0.333 | 0.938 | 0.500 | 0.917 | 96% | 0.708 | 0.548 | 1.000 | 0.532 | 0.969 |
| bfcl_parallel | 60 | 0.100 | 0.950 | 0.300 | 0.933 | 95% | 0.548 | 0.377 | 1.000 | 0.642 | 0.967 |
| bfcl_simple | 96 | 0.021 | 0.990 | 0.167 | 0.990 | 99% | 0.529 | 0.529 | 0.529 | 0.710 | 0.990 |
| bfcl_multiple | 60 | 0.017 | 0.883 | 0.150 | 0.867 | 95% | 0.533 | 0.533 | 0.533 | 0.656 | 0.900 |
| toolace | 96 | 0.010 | 0.948 | 0.135 | 0.938 | 99% | 0.349 | 0.244 | 0.611 | 0.528 | 0.948 |
| bfcl_parallel_multiple | 60 | 0.067 | 0.767 | 0.117 | 0.750 | 97% | 0.538 | 0.368 | 1.000 | 0.490 | 0.767 |
| rca_openrca³ | 96 | 0.115 | 0.458 | 0.104 | 0.458 | 96% | 0.782 | 0.642 | 1.000 | 0.504 | 0.469 |
| apibank⁴ | 96 | 0.000 | 0.156 | 0.010 | 0.146 | 79% | 0.915 | 0.844 | 1.000 | 0.602 | 0.156 |

¹ glaive is **single-function** (1 tool def/item) so 0.77 is low-difficulty recall, not long-context compression; AUROC 0.59 modest. ² hermes is now scored on-disk with the fixed scorer (gate metrics included). ³ rca is **MC (option-LL) + distractor sampling**, full not bit-stable across configs (0.40–0.51), and under `combo` compress (0.104) ≈ no_ctx (0.115) ⇒ **compression adds nothing**. ⁴ apibank ran the **`base`** config (online on `test`; combo not run) — `full` is only 0.156 (apibank is **low-headroom**, hard even with context), compress 0.010 ≈ no_ctx ⇒ fails. The 6 BFCL benches + glaive have **bit-stable full across all configs** ⇒ exact.

**Honest read (clean, leak-free):**
1. **Compression never beats `full` on any tool bench** (with `combo`: glaive 0.77 [single-fn, trivial], bfcl_live_multiple 0.72, live_simple 0.53, hermes 0.50; rest ≤0.30). The leaked v1.7 "tool compression wins / 0.53–0.78" headline is **gone**. **But GCM is the best of the three compressors** — it beats faithful Cartridge and Gist on every bench (§2.4). *(per-bench-best configs reach a bit higher — live_multiple combo 0.72, live_simple lora32 0.59, glaive lora32 0.81 — see §2.3, still < full.)*
2. **Live / conversational benches compress far better than synthetic single-call ones** (live_multiple/live_simple/hermes ≫ simple/multiple/parallel/toolace) — richer/longer context = more to compress.
3. **The gate ≈ do-no-harm, but a held-out threshold isn't free.** `gAcc_cv` lands **−0.05 … +0.03 of full**: live_simple **+0.03** (compression genuinely helps), most within −0.02, bfcl_parallel **−0.05** (weak signal compresses a few harmful items). The strict do-no-harm floor (always-fall-back = full) is always available; the tuned-threshold gate trades a little for compute where the signal is weak.
4. **LoRA is NOT a universal unlock** (v1.7 claim retracted) — see the §2.3 ablation: over the value-add benches LoRA (base, R16) adds only **+0.03 avg compress over no-LoRA**; on bfcl_simple `lora0` (0.271) ≥ `lora32` (0.167).
5. **Gate AUROC** is good only where compression is real (live_simple 0.88, live_multiple 0.75) and ≈chance elsewhere (0.45–0.54) — held-out `gAcc_cv` ≈ full anyway (the gate abstains when its signal is uninformative).

→ **v1.7.3 honest headline:** *a gated compressor that stays **≈ full (held-out gAcc within −0.05…+0.03)** while saving compute where compression is competitive (live/conversational tool-use); it **does not beat full**, compression is weak on synthetic single-call benches and RCA, and a deployed threshold can dip slightly below full — the strict do-no-harm floor is the always-fall-back option (= full).*

### 2.2 Where GCM beats no_ctx (does compression carry information?)
GCM compress vs the bare model (no context), `combo` config — margin = compress − no_ctx. GCM > no_ctx on **every tool bench** (the memory carries real info beyond the parametric model); the *useful* margin concentrates on the live/conversational benches; **rca is the exception**.

| bench | no_ctx | GCM compress | **margin** | full |
|---|---|---|---|---|
| bfcl_live_multiple | 0.010 | 0.719 | **+0.709** | 0.906 |
| bfcl_live_simple | 0.013 | 0.526 | **+0.513** | 0.910 |
| glaive¹ | 0.333 | 0.771 | +0.438 | 0.990 |
| bfcl_parallel | 0.100 | 0.300 | +0.200 | 0.950 |
| hermes | 0.333 | 0.500 | +0.167 | 0.938 |
| bfcl_simple | 0.021 | 0.167 | +0.146 | 0.990 |
| bfcl_multiple | 0.017 | 0.150 | +0.133 | 0.883 |
| toolace | 0.010 | 0.135 | +0.125 | 0.948 |
| bfcl_parallel_multiple | 0.067 | 0.117 | +0.050 | 0.767 |
| rca_openrca | 0.115 | 0.104 | **−0.011 (none)** | 0.458 |

→ Compression **adds real information over no-context** on tool benches (margin +0.05…+0.71) but **never closes the gap to `full`** — the memory keeps the gist, loses the detail. **rca: compression adds ≈ nothing** (margin ≈ 0, slightly negative).

### 2.3 Config ablation — which HP config to use
Per-config **average compress** over the 9 value-add benches (§2.2, all 5 configs done; rca excluded — noisy MC):

| config (LoRA-rank / K / lr / steps) | avg compress | note |
|---|---|---|
| **`combo`** (32 / 64 / 1e-4 / 3000) | **0.376** | **best avg ⇒ recommended** |
| `steps` (16 / 128 / 1e-4 / 3000) | 0.372 | ≈ tied |
| `base` (16 / 128 / 5e-5 / 1600) | 0.371 | ≈ tied |
| `lora32` (32 / 128 / 5e-5 / 1600) | 0.367 | bigger LoRA doesn't help |
| `lora0` (no LoRA) | 0.356 | **LoRA adds only +0.02** |

→ **Configs barely differ on average (0.36–0.38).** combo ≈ steps ≈ base; LoRA buys only ~**+0.02** avg over no-LoRA (NOT the "unlock" v1.7 claimed); more steps / bigger K / bigger LoRA don't materially help. **Per-bench best varies** (glaive lora32 0.812; live_multiple combo 0.719; live_simple lora32 0.590) — so a **per-bench-tuned** deployment beats the single combo config by a few points. Full per-(bench × config) compress sweep (best per row bold):

| bench | base | steps | lora32 | combo | lora0 |
|---|---|---|---|---|---|
| bfcl_live_multiple | 0.573 | 0.708 | 0.635 | **0.719** | 0.552 |
| bfcl_live_simple | 0.513 | 0.551 | **0.590** | 0.526 | 0.474 |
| hermes | 0.490 | 0.500 | 0.521 | 0.500 | **0.542** |
| bfcl_parallel | 0.300 | 0.283 | 0.250 | 0.300 | **0.317** |
| bfcl_simple | 0.260 | 0.188 | 0.167 | 0.167 | **0.271** |
| bfcl_parallel_multiple | **0.183** | 0.133 | 0.117 | 0.117 | 0.050 |
| toolace | 0.146 | 0.115 | 0.125 | 0.135 | **0.177** |
| bfcl_multiple | 0.117 | 0.100 | 0.083 | **0.150** | 0.067 |
| rca_openrca³ | **0.135** | 0.115 | 0.104 | 0.104 | 0.115 |
| glaive | 0.760 | 0.771 | **0.812** | 0.771 | 0.750 |

*(gAcc_cv / fallback% / gate F1·P·R / AUROC for every config are in `out/clean/p1/<bench>__<cfg>`; the recommended `combo` row's full metrics are in §2.1.)*

### 2.4 Matched-budget baselines + ★ the compressor-agnostic gate (E1)
All three at the **same budget** (n_items **384**, steps **3000**, K **64**; each method's own lr) — the fair comparison. *(Cartridge + Gist complete.)*

**(a) Compress, matched budget** (GCM = combo):
| bench | no_ctx | full | GCM | Cartridge | Gist |
|---|---|---|---|---|---|
| bfcl_live_multiple | 0.010 | 0.906 | **0.719** | 0.510 | 0.000 |
| bfcl_live_simple | 0.013 | 0.910 | **0.526** | 0.308 | 0.000 |
| hermes | 0.333 | ~0.92 | **0.500** | 0.455 | 0.094 |
| glaive | 0.333 | 0.990 | 0.771 | 0.760 | 0.104 |
| bfcl_parallel | 0.100 | 0.950 | **0.300** | 0.283 | 0.200 |
| bfcl_simple | 0.021 | 0.990 | **0.167** | 0.146 | 0.062 |
| bfcl_multiple | 0.017 | 0.883 | **0.150** | 0.117 | 0.017 |
| bfcl_parallel_multiple | 0.067 | 0.767 | **0.117** | 0.117 | 0.033 |
| rca_openrca | ~0.10 | ~0.42 | 0.104 | 0.125 | 0.135 |

→ **At matched budget GCM ≥ Cartridge ≫ Gist** — GCM's edge holds on the live/conversational benches (live_multiple +0.21, live_simple +0.22) but is a **tie** on short/synthetic (glaive, parallel_multiple) or slightly behind on rca; **Gist is near-zero**. More budget helped Cartridge a lot (live_multiple 0.30→0.51), so the earlier unfair table overstated GCM. **None beats full** (and §2.4c shows trivial truncation is a tougher baseline than any of them on short ctx).

**(b) ★ The gate is compressor-agnostic** — apply the same confidence signal to **Cartridge**: it detects Cartridge's failures and recovers ≈full (`harm` = full−compress = what always-compress would lose; the gate prevents it):
| Cartridge bench | compress | harm if always-compress | conf-gate AUROC | gAcc_cv (gated) |
|---|---|---|---|---|
| bfcl_parallel | 0.283 | +0.67 | **0.81** | 0.933 |
| bfcl_live_multiple | 0.510 | +0.40 | **0.73** | 0.896 |
| bfcl_live_simple | 0.308 | +0.60 | **0.70** | 0.897 |
| hermes | 0.455 | +0.45 | 0.69 | 0.886 |
| glaive | 0.760 | +0.23 | 0.62 | 0.979 |
| bfcl_simple | 0.146 | +0.84 | 0.61 | 0.979 |

→ **The robustness layer is NOT tied to our encoder.** The same confidence gate, applied to **Cartridge**, gives **failure-detection AUROC 0.6–0.8** and **gАcc ≈ full** — it catches when *Cartridge* dropped what the query needed and falls back. That is the core "compressor-agnostic do-no-harm" claim, demonstrated on a *different* compressor. (On **Gist** the gate is weaker/degenerate because Gist's compress ≈ 0 ⇒ the gate just always-falls-back.)

**(c) ★ Necessity vs trivial truncation (NO training)** — the toughest baseline. `trunc` = keep the first K=64 context-token embeddings; `meanpool` = K segment-means; `randk` = K random vectors (floor):
| bench | full | GCM compress | **trunc** | meanpool | randk |
|---|---|---|---|---|---|
| bfcl_live_simple | 0.910 | 0.526 | **0.910** | 0.872 | 0.026 |
| glaive | 0.990 | 0.771 | **0.990** | 0.990 | 0.385 |
| bfcl_simple | 0.990 | 0.167 | **0.990** | 0.990 | 0.021 |
| bfcl_parallel | 0.950 | 0.300 | **0.950** | 0.950 | 0.117 |
| hermes | 0.938 | 0.500 | **0.938** | 0.625 | 0.365 |
| **bfcl_live_multiple** | 0.906 | **0.719** | 0.458 | 0.031 | 0.000 |
| bfcl_multiple | 0.883 | 0.150 | 0.600 | 0.250 | 0.017 |
| bfcl_parallel_multiple | 0.767 | 0.183 | 0.617 | 0.383 | 0.083 |
| toolace | 0.948 | 0.140 | 0.010 | 0.010 | 0.010 |

→ **On short-context benches `trunc` (first-K, no training) MATCHES full and BEATS GCM** (live_simple/glaive/simple/parallel/hermes: trunc = full ≈ 0.9–1.0; GCM 0.17–0.77) — because the context is ~K tokens, **there is nothing to compress**. **GCM beats trivial truncation on exactly one bench, `bfcl_live_multiple`** (0.72 vs 0.46; ctx ≈ 2.8×K), and both fail on the long bench (toolace, ctx 9×K). **This is the necessity boundary: a learned compressor only earns its keep when ctx ≫ K** — which sharpens the thesis that the deployable contribution is the *gate* (do-no-harm regardless of compressor), not the compressor. (Always report `trunc` as a baseline.)

### 2.5 Model-generality sweep (does the pattern hold beyond Qwen3-8B?)
GCM in-task on bfcl_simple/multiple, base-ish config, across 12 base models (dense + MoE).
| base model | bench | no_ctx | full | compress | gAcc_cv | AUROC |
|---|---|---|---|---|---|---|
| **Qwen3.5-9B** | bfcl_simple | 0.021 | 0.990 | **0.396** | 0.969 | 0.552 |
| GLM-4-9B-0414 | bfcl_simple | 0.021 | 0.979 | 0.292 | 0.969 | 0.604 |
| Ministral-8B-2410 | bfcl_simple | 0.052 | 0.990 | 0.156 | 0.979 | 0.614 |
| Qwen3-4B-2507 | bfcl_simple | 0.021 | 0.885 | 0.156 | 0.885 | 0.622 |
| ToolACE-2-8B | bfcl_simple | 0.000 | 0.781 | 0.146 | 0.760 | 0.527 |
| xLAM-2-8b-fc-r | bfcl_simple | 0.021 | 1.000 | 0.135 | 0.990 | 0.642 |
| Qwen3-14B† | bfcl_simple | 0.000 | 0.177 | 0.115 | 0.188 | 0.654 |
| Qwen3.5-4B | bfcl_simple | 0.010 | 0.375 | 0.094 | 0.354 | 0.601 |
| Qwen3.5-9B | bfcl_multiple | 0.000 | 0.950 | 0.117 | 0.933 | 0.638 |
| Llama-xLAM-2-8b | bfcl_multiple | 0.017 | 0.967 | 0.117 | 0.967 | 0.580 |
| GLM-4-9B-0414 | bfcl_multiple | 0.000 | 0.900 | 0.083 | 0.900 | 0.570 |

→ **Pattern holds across families:** compress < full on every model; **Qwen3.5-9B compresses best** (0.40 on simple, vs Qwen3-8B 0.27) — a stronger/newer base helps a bit. **Agent/tool-specialized bases (xLAM, ToolACE) are NOT better** at being compressed (0.135/0.146). † Qwen3-14B / Qwen3.5-4B show low `full` (chat-template/eval quirk on those bases — flagged, not a compression result). MoE (Moonlight / Qwen3-30B-A3B / gpt-oss-20b) ran but parse pending.

### 2.6 ★ Capacity & efficiency analysis — why compression is weak (and the wrong way round)
Per bench: context length, compression ratio (ctx ÷ K=64), value-add margin, and the **realized compute saving** at the gate's operating point.
| bench | ctx tok | ratio (ctx/K) | margin | compute saving |
|---|---|---|---|---|
| bfcl_live_multiple | 182 | 2.8× | **+0.71** | 11% |
| bfcl_live_simple | 43 | 0.7× | +0.51 | −3% |
| glaive | 29 | 0.5× | +0.44 | −6% |
| hermes | 84 | 1.3× | +0.17 | 0% |
| bfcl_parallel | 37 | 0.6× | +0.20 | −1% |
| bfcl_simple | 36 | 0.6× | +0.15 | 0% |
| bfcl_multiple | 97 | 1.5× | +0.13 | 1% |
| toolace | 580 | **9.1×** | +0.13 | 1% |
| rca_openrca | 1019 | **15.9×** | **−0.01** | 4% |

**corr(context length, margin) = −0.4 — compression helps on SHORT context and FAILS on LONG.** This is the **opposite** of a "long-context compression" story. Root cause = **K-token capacity**: compression works at ratio ≤ ~3× (live_multiple 2.8× → +0.71) and collapses at ≥9× (toolace 9× → +0.13, rca 16× → −0.01). **The realized compute saving is ≈ 0 (often negative):** where compression "works" the contexts are *shorter than K=64* (so M costs more than the context), and where saving would matter (toolace/rca, 9–16×) compression fails ⇒ the gate falls back ⇒ no saving. **⇒ Next experiment (QUEUED): a K-sweep** — does more K rescue long-context compression (toolace K∈{128,256,512}, rca K∈{256,512}) and how low can K go on the working bench (live_multiple K∈{16,32,128,256})? This maps the capacity / compression-ratio frontier.

---
## 3. Results — TRANSFER (Phase 2): in / cross-task / cross-domain — QUEUED

Using each bench's best in-task config, train on each tool anchor and eval across all tools + QA/ops (in_task / cross_task_in_domain / cross_task_cross_domain), GCM + Cartridge + Gist. *(Queued after Phase-1 completes; expectation from the per-corpus design: cross-task/domain compress ≈ 0, gate falls back ⇒ gAcc ≈ full.)*

| anchor (domain) | in_task | cross_task_in_domain | cross_task_cross_domain |
|---|---|---|---|
| … | GCM/full/no_ctx | … | … |

---
## 4. Status (2026-06-13, evening)

- **Code fixes done + verified:** leakage (all gens 0–1%), held-out gate (CV), hermes/glaive scoring routing, apibank/glaive offline-load.
- **Phase 1 (in-task) COMPLETE:** all 11 tool/ops benches × 5 configs clean (§2.1–2.3); apibank done (online on `test`).
- **Matched-budget baselines** (§2.4, Cartridge done / Gist landing): at equal budget GCM's edge **shrinks but holds on live benches**, ties elsewhere; none beats full.
- **★ E1 — the gate is compressor-agnostic** (§2.4b): the confidence signal applied to **Cartridge** gives failure-detection AUROC **0.6–0.8** + gАcc ≈ full. **This is the headline for the reframed thesis** (robustness layer, not a compressor).
- **Model-generality sweep DONE** (§2.5): pattern holds across 12 bases; Qwen3.5-9B best (0.40).
- **★ Capacity/efficiency analysis** (§2.6): compression is **capacity-bound** — helps on SHORT ctx (ratio ≤3×), fails on LONG (≥9×); corr(ctx,margin)=−0.4; compute-saving ≈ 0.
- **Reviewer-response experiments queued** ([`reviewer-response.md`](reviewer-response.md)): **R3 args-aware** (the verdict-decider, 6 jobs), **R4 trivial baselines** (trunc/mean-pool/random-K, 10 jobs), **R7 multi-seed CIs** (6 jobs) — code in, meta/additive (running jobs safe).
- **RUNNING (67 jobs, 4 cards):** matched baselines (e1, 7/20) → trivial (f) → args-aware (g) → K-sweep (k) → multi-seed (m) → Phase-2 transfer (p2, 4/24). ETA ~overnight.
