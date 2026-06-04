# Plan 09 — Channels: benchmarks, metrics, baselines, cross-X settings

> Phase-3 paper-grade evaluation. General-first (DR13): a **long-context** axis (read side) + a **forgetting/continual** axis (write side); **RCA enters last** as the both-pains-at-once showcase. Everything is reported **cross-domain / cross-task / cross-model**, 4-seed (DR14).

---

## 1. Benchmarks

### 1a. Long-context (read side)
| Benchmark | What | Why |
|---|---|---|
| **RULER** | synthetic NIAH + multi-hop + aggregation, 4k–128k | primary length-controlled probe; gives accuracy@length |
| **NIAH** (vanilla + multi-needle) | needle retrieval vs depth/length | the verbatim "Regime C" case (DR10) |
| **LongBench-v2** | realistic long-doc QA / summarization | ecological validity |
| **HELMET** | application-centric long-context suite | broad, recent, less saturated |
| **∞Bench** | >100k extreme length | stress / effective-context-length |
| **QuALITY**, **NarrativeQA** | long-doc multiple-choice / QA | mem-X reuse; Regime-A neighbour |

### 1b. Capability / forgetting probes (write side — measured before vs after SFT)
| Domain | Benchmarks |
|---|---|
| Math | **GSM8K**, **MATH** (level-5 subset) |
| Code | **HumanEval(+)**, **MBPP(+)** (EvalPlus) |
| General/knowledge | **MMLU(-Pro)**, **ARC-C**, **HellaSwag**, **TriviaQA**, **BBH** |
| Instruction | **IFEval** |

### 1c. Continual / forgetting-specific
| Benchmark | What |
|---|---|
| **TRACE** | LLM continual-learning suite; standard BWT/forgetting reporting |
| **SuperNI / Flan continual** | sequential instruction-domain stream (cross-task order effects) |
| (our own) **domain sequence** | RCA → code → math fine-tune chain, measure retention along the way |

### 1d. RCA showcase (last — the application, not the method)
- Public: **LogHub**-derived log QA / anomaly, incident-RCA datasets (e.g. log-anomaly + RCA QA), any open network-ops trace set.
- Internal: **Nokia RCA** dataset (the eventual product eval).
- Used to demonstrate "both pains at once": long logs (read) + new-domain SFT (write).

---

## 2. Metrics

### Forgetting / retention
- **BWT** (Backward Transfer) — avg change in old-task accuracy after learning the new domain.
- **Retention ratio** — acc_after / acc_before per capability probe.
- **Forgetting measure F** — max−final accuracy across the continual stream.
- **Average accuracy (ACC)** — mean over all tasks at the end.
- **ΔKL(base‖tuned)** on a general corpus — distributional drift, fine-grained.

### Long-context
- **Accuracy@length** curve; **effective context length** (length at which acc → chance).
- **NIAH exact-match** (verbatim) vs depth.

### Plasticity (did we actually learn the new domain)
- New-domain accuracy gain vs frozen base (must stay high — protection shouldn't block learning).

### Joint / headline
- **Retention–plasticity Pareto** (forgetting vs new-domain gain) — ours should dominate the frontier.
- **Harmonic mean** of (new-domain gain, capability retention, long-ctx retention) — single number for tables.

---

## 3. Baselines (must beat or match-cheaper)

| Group | Methods | Role |
|---|---|---|
| Naïve | frozen base (no SFT) · **full SFT** · **LoRA SFT** | lower/upper anchors |
| Anti-forgetting (dense) | **EWC** · **L2-SP** · **LwF** (distillation) · **experience replay/rehearsal** · **wise-ft / model soup** · **OPLoRA** (`[oplora]`) · **MoFO** (`[mofo]`) · **MIGU** (`[migu]`) · **SelfAug** (`[selfaug]`) | the criterion competitors (DR9) |
| Anti-forgetting (MoE) | **ESFT** (`[esft]`) · **DES-MoE** (`[des-moe]`) · **LoRAMoE** (`[loramoe]`) | MoE competitors |
| Long-ctx training-free | **StreamingLLM** (`[sink-streaming]`) · **H2O** (`[h2o]`) | read-side reference (orthogonal) |
| **Ours** | intrinsic-site protection (V1–V5, the variant Phase-1 selects) | the method |

The decisive comparison is **ours vs {OPLoRA, MoFO, ESFT}** at matched new-domain accuracy: same *mechanism family* (freeze/mask/orthogonalize), different **site-selection criterion** — isolates DR9.

---

## 4. Cross-X settings (detailed)

### Cross-domain (the forgetting matrix)
- SFT on domain A, measure capability on B, C. Full matrix over {RCA, code, math, neutral-proxy}.
- Headline cell: **SFT on RCA → retention on code/math/MMLU** (the product-relevant case).
- Report the A×B forgetting matrix (heatmap).

### Cross-task (continual / order effects)
- Sequence the domains: RCA→code→math and a reversed order; measure BWT / F along the stream (TRACE-style).
- Tests that protection survives *sequential* fine-tuning, not just single-step.

### Cross-model (model-agnostic, DR5)
- **Family**: Qwen3-8B vs Llama-3.1-8B (same recipe, no per-model tuning).
- **Dense ↔ MoE**: Qwen3-8B (dense) vs Qwen3-30B-A3B (MoE, super-expert variant V5).
- **Scale**: Qwen3-1.7B / 4B / 8B / 14B — show the criterion and benefit persist with scale.
- (stretch) second MoE family (OLMoE / DeepSeek-V2-Lite) to prove the super-expert criterion is not Qwen-specific.

### Training / eval protocol (settings)
- **SFT**: LoRA-r ∈ {16, 64} (+ one full-FT sanity); lr ∈ {1e-5, 2e-5}; 1–3 epochs; seq-len matched to the long-ctx eval; AdamW, cosine, warmup 3%. Protected-set size **k swept** (dose-response).
- **Long-ctx eval**: lengths {4k,8k,16k,32k,64k}; ≥ 50 needles/length; position grid {0,25,50,75,100}%.
- **Seeds**: 4 (DR14). Report mean ± 95% bootstrap CI.
- **Eval harness**: `lm-evaluation-harness` (capability), **RULER official** + **HELMET** (long-ctx), **EvalPlus** (code), TRACE scripts (continual).
- **Decode**: greedy for capability/NIAH; report temperature sensitivity once.

### Detection protocol (data-agnostic, DR3)
- Site detectors run on a fixed **generic** corpus slice (C4/FineWeb), **never** on the SFT domain — this is the data-agnostic claim; an ablation re-detects on in-domain data to show the site set is the same.

---

## 5. Reproducibility deliverables
- Detector + drift-meter + intervention-hook library (HF forward/backward hooks, ~600 LoC).
- Per-model **coupling maps** + scatter figures (Phase 1).
- Protection wrapper (PEFT-compatible), trained checkpoints for ours + baselines.
- Full eval configs, seeds, hyper-params; RULER/HELMET/TRACE run scripts.
