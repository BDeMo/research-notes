# v1.7.3 Experimental setup — models, datasets, recipe, protocol

Canonical record of everything we run in v1.7.3 (the **leak-free** redo). Results: [`results-v1.7.3.md`](results-v1.7.3.md) · mechanism + gate flowcharts: [`gcm-lora-mechanism.md`](gcm-lora-mechanism.md). v1.7.3 differs from the archived [v1.7](../results-v1.7/results-v1.7.md) by **(i)** a seed-independent train/eval split (fixes a leakage bug), **(ii)** a held-out (cross-validated) gate, **(iii)** word-boundary `tool_acc`, and **(iv)** full tool coverage (5 teams) with a two-phase plan (tune in-task → transfer).

## 1. Base models
GCM copies the base's first N blocks as the encoder, so the base must be a **dense** transformer (huge frontier MoEs don't fit 80 GB). All bf16. **Headline base = Qwen3-8B.** The rest are a queued base-model sweep (in-task bfcl) for generality.

| model | HF repo | arch | ~params | role |
|---|---|---|---|---|
| **Qwen3-8B** | `Qwen/Qwen3-8B` | Qwen3 (dense) | 8B | **default / headline** |
| Qwen3-4B-Instruct-2507 | `Qwen/Qwen3-4B-Instruct-2507` | Qwen3 (dense) | 4B | size-sweep (small) |
| Qwen3-14B | `Qwen/Qwen3-14B` | Qwen3 (dense) | 14B | size-sweep (large) |
| Qwen3.5-9B / Qwen3.5-4B | `Qwen/Qwen3.5-9B`, `Qwen/Qwen3.5-4B` | Qwen3.5 (dense) | 9B / 4B | latest Qwen |
| GLM-4-9B-0414 | `zai-org/GLM-4-9B-0414` | Glm4 (dense) | 9B | family generality (GLM) |
| xLAM-2-8b-fc-r | `Salesforce/Llama-xLAM-2-8b-fc-r` | Llama (dense) | 8B | agent-specialized base |
| ToolACE-2-8B | `Team-ACE/ToolACE-2-8B` | Llama (dense) | 8B | tool-specialized base |
| Ministral-8B-2410 | `mistralai/Ministral-8B-Instruct-2410` | Mistral (dense) | 8B | family generality (Mistral) |

**Small MoE (experimental; smaller encoder N to fit 80 GB):** Qwen3-30B-A3B-Instruct-2507 (N=4), gpt-oss-20b (N=8), Moonlight-16B-A3B (N=8). Weights cached at `/mnt/persist/checkpoints/<name>`.

## 2. Datasets / benchmarks
Full table (task · metric · source · paper) in [`results-v1.7.md` §Benchmarks](../results-v1.7/results-v1.7.md). Summary:
- **Tool-use (focus) — 5 independent teams** (so the claim isn't one benchmark's quirk):
  BFCL categories `simple, multiple, parallel, parallel_multiple, irrelevance, live_simple, live_multiple,
  live_parallel, live_irrelevance, java, javascript, rest, sql` (`gorilla-llm/BFCL`, **UC Berkeley/Gorilla**);
  `apibank` (`liminghao1630/API-Bank`, **Alibaba DAMO**); `toolace` (`Team-ACE/ToolACE`, **Huawei Noah's Ark**);
  `hermes` (`NousResearch/hermes-function-calling-v1`, single-turn multi-tool, **Nous Research**);
  `glaive` (`glaiveai/glaive-function-calling-v2`, single-fn + decline, **Glaive AI**); `bfcl_pooled` (ours).
  Metric `tool_acc`. Loaders: `llm_infra/datasets.py::generate_{bfcl,apibank,toolace,hermes,glaive}`.
  Deferred: xLAM/APIGen (Salesforce, HF-gated), ComplexFuncBench (Zhipu/THUDM), NexusRaven NFCL (Nexusflow).
- **Ops:** `rca_openrca` (OpenRCA, root-cause MC, `primary_service_match`), local `cases.jsonl`.
- **QA / reasoning (cross-domain):** `squad_v2`, `hotpot_qa`, `trivia_qa`, `narrativeqa`, `musr_mm`, `quality`.

## 3. GCM architecture (the compressor)
| component | setting | knob |
|---|---|---|
| **Encoder** | trainable copy of base's first **N** blocks (default 16), `init=copy` | `--enc-layers`, `--enc-init` |
| **Memory** | **K** learnable slots (64 / 128); produced in ONE forward, not autoregressive; M0=query-masked, Mq=query-unmasked | `--n-memory` |
| `m_proj` | linear into base input-embed space | — |
| **Decoder** | trainable copy of base's first **2** blocks (reconstruction `L_uncond` → the `neg_recon` gate signal) | `--n-dec-layers` |
| **base LoRA** | rank R (0 / 16 / 32) on `q_proj`/`v_proj`, **non-merged** (on=compress, off=fallback) | `--base-lora-rank` |
| discriminator | MLP on base layer-18 hidden (only if `--lam-adv>0`) | `--lam-adv`, `--adv-layer` |

Trainable = encoder + K slots + `m_proj` + decoder + LoRA(A,B). **Base weights frozen.**

## 4. Training recipe
- **Losses:** `L_task` (gold-CE on `[Mq;q;gold]`) + `L_distill` (0.5, match full-ctx teacher) + `L_rec`/L_uncond (1.0, decoder reconstructs ctx from M0) + `L_dev` (0.05, ‖Mq_raw−M0_raw‖²) + `L_adv` (0, off). `--lam-task 1 --lam-distill 0.5 --lam-rec 1 --lam-dev 0.05 --lam-adv 0`.
- **Optimizer:** AdamW, grad-clip 1.0, **bf16** (no fp32 master copy). `--n-items 384`, `--n-chunks 1` (tool), `--seed 42`.
- **Phase-1 in-task HP grid (5 configs per bench)** — this is what §2.1 of the results sweeps:

| config | base-LoRA rank | K | lr | steps |
|---|---|---|---|---|
| `base` | 16 | 128 | 5e-5 | 1600 |
| `steps` | 16 | 128 | 1e-4 | 3000 |
| `lora32` | 32 | 128 | 5e-5 | 1600 |
| `combo` | 32 | 64 | 1e-4 | 3000 |
| `lora0` | 0 (no LoRA) | 128 | 5e-5 | 1600 |

- **Phase 2 (transfer):** each anchor trained with its best Phase-1 config, evaluated across all tools + QA/ops; baselines Cartridge (`--n-items 96 --steps 1500 --lr 1e-3 --n-memory 64`) and Gist (`--lr 1e-4`, else same) on the same matrix.

## 5. Inference
- **Compress path:** `[M ; query]` → base **+ LoRA on**; the K memory tokens replace the whole context.
- **Fallback path:** `[ctx ; query]` → base **LoRA off** = exact original base (do-no-harm). `full_ctx`/`no_ctx` are always scored **LoRA-off** (`harness.py::_set_base_lora`) so the baselines are never contaminated — verified: `full`/`no_ctx` are bit-stable across all configs of a bench.
- **Gate:** per-item signal (best = `neg_recon` = −CE of reconstructing ctx from M0; also conf/margin/dlogit/dcode/mnorm/disc_p), threshold `compress iff signal ≥ τ` else fall back.
- **Decoding:** greedy ≤16 new tokens; `--max-ctx-tokens` cap = 1536 (1024 for long benches: toolace/apibank/rca).

## 6. Eval protocol & data hygiene (the v1.7.3 corrections)
- **N = 96 eval items/bench** (`--n-eval 96`, `--eval-n-chunks 1`). Train seed 42, **eval seed 43** (harness uses `seed+1`).
- **Leak-free split (the v1.7 bug fix).** All manual-split tool/ops generators (bfcl, apibank, toolace, hermes, glaive, rca) partition train/val with a **seed-INDEPENDENT** 70/30 split (`llm_infra/datasets.py::_disjoint_split`, fixed partition seed; the run seed only orders *within* a split). Glaive additionally dedups templated queries. Single-split benches (musr/ruler/categorical_niah) get a **content-hash** train/val guard in `gcm/data.py::load_items`. QA benches use HF-native disjoint splits. **Verified train∩eval overlap = 0–1%** (was 71–96% in v1.7).
- **Scoring.** `tool_acc` = **word-boundary** match of the gold function name in the answer-line (not substring; `benchmark_metrics.py`); MC (rca/musr/quality) = per-option **log-likelihood**; generation = first-answer-line then bench metric. Same scorer for every method.
- **Gate metric (honest).** Reported `gAcc_cv` = **held-out 5-fold cross-validated** gated accuracy (`svc/disc_gate.py::gated_acc_cv`: threshold fit on train folds, applied to the held-out fold). The in-sample best-F1 `F1/precision/recall` are also reported but are **optimistic** (degenerate to always-compress on low-AUROC benches); **AUROC** is the threshold-free gate quality. `best` = per-item oracle `max(compress, full)`.
- **Single seed (42/43)** ⇒ run-to-run noise ≈ ±0.04; MC benches (rca) additionally have ±0.04 option-LL variance (not bit-stable across configs). Trust large gaps, not small ones.

## 7. Compute / environment
- **Pods (k8s aird-ray-dev):** `sam-dev-free` (4×H100-80GB, the queue + runners), `sam-dev-test` (1 GPU; runs the streaming/online benches e.g. apibank). ray cleared.
- **Env (matched):** torch 2.11.0+cu128, transformers 5.10.1, datasets 4.8.5 — conda env `llm` at `/mnt/persist/miniforge3/envs/llm`. Everything persistent on `/mnt/persist`. **`HF_HUB_OFFLINE=1`** on the runners (cached datasets only; apibank/glaive needed special handling — glaive cached non-streaming, apibank run online on test).
- **Runner:** per-GPU tmux workers draining a file-queue (`svc/runner.py`, claims lowest-sorted `*.job`, appends `--device cuda:<gpu>`, logs to `queue/log/`); recovery via `/mnt/persist/start_runners.sh`.
