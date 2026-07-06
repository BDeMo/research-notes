# v1.7 Experimental setup — models, datasets, recipe

Canonical record of everything we run: base models, datasets, and the training/inference/architecture parameters.
Mechanism + flowcharts: [`gcm-lora-mechanism.md`](gcm-lora-mechanism.md) · results: [`results-v1.7.md`](results-v1.7.md) ·
execution/plan: [`solutions/master-plan-180gpuh.md`](solutions/master-plan-180gpuh.md).

## 1. Base models (the base-model sweep)
GCM requires a **dense** transformer (it copies the base's first N blocks as the encoder), so frontier MoE models
(GLM-4.6/5, Kimi-K2.6, Llama-4, DeepSeek-V4, MiniMax) are out — they don't fit 80 GB and aren't dense. All bf16.
Default base for the headline experiments = **Qwen3-8B**.

| model | HF repo | arch | ~params | role |
|---|---|---|---|---|
| Qwen3-8B | `Qwen/Qwen3-8B` | Qwen3 (dense) | 8B | **default / headline** |
| Qwen3-4B-Instruct-2507 | `Qwen/Qwen3-4B-Instruct-2507` | Qwen3 (dense) | 4B | size-sweep (small) |
| Qwen3-14B | `Qwen/Qwen3-14B` | Qwen3 (dense) | 14B | size-sweep (large) |
| Qwen3.5-9B | `Qwen/Qwen3.5-9B` | Qwen3.5 (dense) | 9B | latest Qwen |
| Qwen3.5-4B | `Qwen/Qwen3.5-4B` | Qwen3.5 (dense) | 4B | latest Qwen (small) |
| GLM-4-9B-0414 | `zai-org/GLM-4-9B-0414` | Glm4ForCausalLM (dense) | 9B | family generality (GLM) |
| xLAM-2-8b-fc-r | `Salesforce/Llama-xLAM-2-8b-fc-r` | Llama (dense) | 8B | **agent-specialized** base |
| ToolACE-2-8B | `Team-ACE/ToolACE-2-8B` | Llama (dense) | 8B | tool-specialized base |
| Ministral-8B-2410 | `mistralai/Ministral-8B-Instruct-2410` | Mistral (dense) | 8B | family generality (Mistral) |

**Small MoE (experimental — GCM copies MoE blocks, so smaller encoder N to fit 80 GB):**
| model | HF repo | arch | total/active | N |
|---|---|---|---|---|
| Qwen3-30B-A3B-Instruct-2507 | `Qwen/Qwen3-30B-A3B-Instruct-2507` | Qwen3Moe (agentic) | 30B/3B | 4 |
| gpt-oss-20b | `openai/gpt-oss-20b` | GptOss (OpenAI) | 21B/3.6B | 8 |
| Moonlight-16B-A3B | `moonshotai/Moonlight-16B-A3B-Instruct` | DeepseekV3/MLA (Kimi family) | 16B/3B | 8 |

Weights cached on the pod under `/mnt/persist/checkpoints/<name>`. (Llama/Mistral/GLM are new for GCM — verified
on Qwen; runs confirm encoder-copy + eager-mask compatibility, else marked `.fail`.)

## 2. Datasets / benchmarks
Full table (task · metric · source · paper) in [`results-v1.7.md` §Benchmarks](results-v1.7.md). Summary:
- **Tool-use (focus, ~17) — 5 independent teams** (so the claim isn't one benchmark's quirk):
  BFCL categories `simple, multiple, parallel, parallel_multiple, irrelevance, live_simple, live_multiple,
  live_parallel, live_irrelevance, java, javascript, rest, sql` (`gorilla-llm/BFCL`, **UC Berkeley/Gorilla**);
  `apibank` (`liminghao1630/API-Bank`, **Alibaba DAMO**); `toolace` (`Team-ACE/ToolACE`, **Huawei Noah's Ark**);
  **`hermes`** (`NousResearch/hermes-function-calling-v1`, single-turn multi-tool, **Nous Research**);
  **`glaive`** (`glaiveai/glaive-function-calling-v2`, single-fn + decline, **Glaive AI**); `bfcl_pooled` (ours).
  Metric `tool_acc` (AST/name match). Loaders: `llm_infra/datasets.py::generate_{bfcl,apibank,toolace,hermes,glaive}`.
  Deferred candidates: xLAM/APIGen (Salesforce, HF-gated), ComplexFuncBench (Zhipu/THUDM), NexusRaven NFCL (Nexusflow).
- **Ops:** `rca_openrca` (OpenRCA, root-cause MC, `primary_service_match`), local `cases.jsonl`.
- **QA / reasoning (cross-domain):** `squad_v2`, `hotpot_qa`, `trivia_qa`, `narrativeqa`, `musr_mm`, `quality`.

## 3. GCM architecture (the compressor)
| component | setting | knob |
|---|---|---|
| **Encoder** | trainable copy of base's first **N=16** blocks, `init=copy` (from base) | `--enc-layers`, `--enc-init` |
| **Memory** | **K=128** learnable slots (K=32 found near-best); produced in ONE forward, not autoregressive | `--n-memory` |
| `m_proj` | linear into base input-embed space; M0=query-masked, Mq=query-unmasked | — |
| **Decoder** | trainable copy of base's first **2** blocks (reconstruction = L_uncond) | `--n-dec-layers` |
| **base LoRA** | rank **16** on `q_proj`/`v_proj`, **non-merged** (on=compress, off=fallback) | `--base-lora-rank` |
| discriminator | MLP on base layer-**18** hidden (only if `lam_adv>0`); doubles as learned gate | `--lam-adv`, `--adv-layer` |

## 4. Training recipe
- **Losses:** `L_task` (gold-CE on `[Mq;q;gold]`) + `L_distill` (0.5, match full-ctx teacher over R) +
  `L_rec`/L_uncond (1.0, decoder reconstructs ctx from M0) + `L_dev` (0.05, ‖Mq_raw−M0_raw‖²) + `L_adv` (0, optional).
  Weights: `--lam-task 1 --lam-distill 0.5 --lam-rec 1 --lam-dev 0.05 --lam-adv 0`.
- **Optimizer:** AdamW, **lr 5e-5**, **steps 1600** (sweeps to 3000), grad-clip 1.0, **bf16** (no fp32 master copy).
- **Data:** `--n-items 384` (capped to the bench pool, e.g. ~280 for bfcl_simple), `--n-chunks 5` (1 for single-item
  tool benches), `--seed 42`. Trainable = encoder + slots + m_proj + decoder + LoRA(A,B); **base frozen**.
- **Default winning config** (tool): enc16, K128 (or K32), n_dec2, LoRA16, distill0.5, lr5e-5, 1600–2400 steps.

## 5. Inference
- **Compress path:** `[M ; query]` → base **+ LoRA on**; M (K tokens) replaces the whole context.
- **Fallback path:** `[ctx ; query]` → base **LoRA off** = exact original base (do-no-harm). `full_ctx` scored LoRA-off.
- **Gate:** per-item signal (`neg_recon` best; or conf/margin/dlogit/dcode/disc_p) thresholded → compress iff
  signal ≥ τ, else fall back. τ set for a target precision (the do-no-harm operating point).
- **Decoding:** greedy; `--max-ctx-tokens` cap (1536–2048, to bound the encoder's eager-attention on long tool defs);
  MC benches scored by per-option log-likelihood; gen benches by `tool_acc`/`squad_f1`/`rouge_l`.

## 6. Compute / environment
- **Pods (k8s aird-ray-dev):** `sam-dev-free` (4×H100-80GB), `sam-dev-test` (1 GPU). (ray cleared.)
- **Env (matched):** torch 2.11.0+cu128, transformers 5.10.1, datasets 4.8.5, accelerate 1.13.0 — conda env `llm` at
  `/mnt/persist/miniforge3/envs/llm` (test uses tf 4.56.2, code-verified). Everything persistent on `/mnt/persist`.
- **Runner:** per-GPU tmux workers draining a file-queue (`runner.py`); recovery via `/mnt/persist/start_runners.sh`.
