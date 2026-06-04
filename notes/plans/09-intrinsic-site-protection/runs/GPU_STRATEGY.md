# Janus GPU usage strategy (reorganized 2026-06-04)

**Principle**: pre-experiments = **fact-finding**. Forward-heavy fact jobs are cheap → fill every idle GPU with broad metric coverage; reserve the expensive backward (Fisher/SFT) work for where data lives.

## Nodes (8× H100)
| node | GPUs | role | why |
|---|---|---|---|
| **sam-dev** | 0,1,2,3 | data-dependent + cross-family | has math/MMLU/trivia cached + GLM-4 downloaded; runs intrinsic-coupling (needs SFT) + GLM family + facts for Qwen2.5/GLM |
| **sam-dev-ray** | 0,1,2,3 | Qwen3 scale ladder, data-light | fast; runs facts + intrinsic across 0.6→14B (trivia only — no math/mmlu cached) |

## Job tiers (priority order)
1. **FACTS (forward-only, broad)** — per-layer metrics across representation / information-theory / parameter-spectra angles + cross-link to per-head coupling. Cheap, fills all GPUs. **Current priority.**
2. **INTRINSIC (per-head coupling, needs backward+SFT)** — done for the Qwen ladder + Qwen2.5 + GLM-4 (9B & 32B). Big models use grad-checkpoint + SGD + per-batch Fisher reduction (memory-safe to 32B).
3. **NIAH / capability** — long-context + forgetting outcome probes (overnight batch, done).

## Memory rules (learned)
- ≥20B models: enable `gradient_checkpointing` + `enable_input_require_grads`; Fisher via **per-batch per-head reduction** (never store full-weight accumulators); drift SFT via **SGD on q/o-proj only** (no Adam states). `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Forward-only facts/detect fit any size on one 80–95GB H100.

## Qwen3.5 — SOLVED (2026-06-04)
Isolated venv on ray: `/root/q35iso` = **torch 2.11.0+cu128 + transformers 5.10 + datasets + matplotlib** (driver 570/CUDA12.8 supports cu128). `AutoModelForCausalLM` loads `Qwen3_5ForCausalLM` (text backbone, no multimodal handling needed). Hybrid 3:1 attention handled: `_iter_attn_maps` remaps the length-8 `out.attentions` to real layer indices [3,7,…,31]; linear-attn layers guarded everywhere. `fla`/`causal-conv1d` not installed → slow torch path (fine). Run Qwen3.5 (and the whole cohort, for env consistency) via `/root/q35iso/bin/python` on ray.

## Current focus: 7–9B cohort (controlled scale)
Per user 2026-06-04: standardize on sub-10B (**Qwen3.5-9B, GLM-4-9B, Qwen3-8B, Qwen2.5-7B-Instruct**), find facts, **scale later**. All run in the iso venv on ray, native bf16, no quant.

## Model coverage (facts + intrinsic)
Qwen3: 0.6B,1.7B,4B,8B,14B · Qwen2.5: 1.5B,1.5B-Instruct,7B-Instruct · GLM-4: 9B,32B (cross-vendor).
