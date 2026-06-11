# Matrix — Janus v2: candidate-method testing (long-context ↔ forgetting)

> **New line, created 2026-06-04 ~23:10 UTC.** The previous matrix (the v1
> **measurement** ledger: novelty audit → Phase-0/1 → facts → LC×CF grid → metric
> codebook/curation → idea lists → H3 v1) is **archived** to
> [`matrix-archive-2026-06-04.md`](matrix-archive-2026-06-04.md).
> This file tracks the **method-testing line**: turn the confirmed coupling into a
> working anti-forgetting method.
>
> **Where v1 landed.** The LC×CF coupling is **confirmed** (every long-context head
> metric positively predicts every forgetting metric, 41 cells, cross-family — see
> archive §1 / `grid-metrics-2026-06-04.md`). The causal test **H3 v1 was
> inconclusive** only because the GSM8K-SFT setup *didn't induce long-context
> forgetting* (GLM-4 NIAH 1.0→1.0). So v2 = (a) a **forgetting-inducing** setup,
> (b) the full **candidate-method library**, (c) the decisive protection sweep.
>
> **Library.** `janus-methods/` (on both pods): every method = **criterion × operator**
> (17 × 6) + standalone (`predict_i1`, `steer_i7`, `surgical_merge`). Config-driven.

---

## 0 · GATE · ACTIVE LINE (2026-06-05): Paper B — "Know When to Fall Back"
> **Paper B = the v1.5 wrapper/gating line (Plan 08 v1.5), NOT Janus/Plan 09.** Docs filed here
> for now; to be relocated to the wrapper line. Janus (this plan) = Paper A.
> Spec: [`gate-pipeline-2026-06-05.md`](gate-pipeline-2026-06-05.md) · scope+litreview+novelty:
> [`gate-scope-litreview-2026-06-05.md`](gate-scope-litreview-2026-06-05.md) · ICLR self-review:
> [`gate-critical-review-2026-06-05.md`](gate-critical-review-2026-06-05.md).

**Scope (locked 2026-06-05): parametric / latent-space *agentic* memory** (MemoryLLM/Memory³/ICAE
substrate, not text-RAG). **Idea.** Compress context → latent memory (wrapper); a **gate** decides
per-query if memory suffices, else **fall back to full context**. Fallback ⇒ correctness floor ⇒
the whole game is **gate reliability** ⇒ study the **gating signal**, not the architecture.

**Novelty wedge (after 2026 audit).** Gated escalation is now crowded in *non-parametric* agent
memory (D-Mem/Oblivion/RF-Mem/HyMem, all 2026) and latent-memory models (MemoryLLM/M+/Memory³)
have *no* reliability gate. We fill the empty cell: **latent memory × read-path reliability gate ×
full-context fallback**, + the first **systematic signal study** (D1–D6, cross-X, calibration) +
**latent-specific signals** (counterfactual-KL, retrieval-head routing onto slots) that beat
LLM-judge/entropy cheaper. Defense: [`gate-scope-litreview-2026-06-05.md`](gate-scope-litreview-2026-06-05.md) §3.

**Why it's a paper (and the risk).** Surviving seam after the 2026 audit (D-Mem / Entropy-Gate /
SLT / RazorAttention all do gate-or-fallback in pieces): make the contribution a **cheap,
mechanism-grounded, calibrated signal** that **matches the LLM-judge at ≫10× lower cost** + a
**cross-shift guarantee**. Reviewer sim verdict: current scope = reject (~4.7, "confidence-signal
benchmark"); +baselines+oracle+trained-compressor → accept (~6.7); +conformal guarantee+deployment
→ spotlight (~7.5). Contribution, not coverage, is the gate.

**"Good signal" = D1–D6** (falsifiable): in-domain AUROC≥0.7 · sig·boot-CI · universality
coverage@0.7≥0.8 + sign-consistent · cross-domain/task/dataset transfer (ΔAUROC≤0.05, ECE≤0.1) ·
cheap (≤10% overhead, no full-ctx) · Pareto-dominates random/entropy/LLM-judge.

**Code.** `gate_study.py` (collect labels+13 signals) · `gate_analyze.py` (scorecard:
AUROC/universality/LODO/LOMO). **Fix applied:** Qwen3 **thinking-mode off** + strip `<think>` +
concise prompt → qwen3-8b/squad full_acc **0.30→1.0** (and de-poisons confidence signals).

**Status (running).** ray 4×H100, **8 models × 8 datasets = 64 cells** (glm4-32b dropped — not
cached offline; qwen3.5-4b added), n=60. Now driven by the **`gpu-runq` monitor** (tmux daemon on
ray), not the orch scripts.

**Clean scorecard (post `<think>`-fix; 13 bases / 663 items — see
[`gate-scope-litreview-2026-06-05.md`](gate-scope-litreview-2026-06-05.md) §3.1b):** generation
confidence wins — `ans_entropy` median AUROC **0.88**, coverage@0.7 **1.0**, transfers LODO/LOMO
**~0.87**. The latent-specific proxies (`kl_mem_vs_q`, `attn_to_mem`) are at the bottom (~0.60) →
**N3 not yet supported**; needs the real retrieval-head-routing signal on a trained wrapper.

**TODO-B:** (1) full clean fanout → scorecard. (2) D-Mem-judge + Entropy-Gate + oracle baselines
in-harness + Pareto. (3) trained-wrapper transfer (Activation-Beacon/ICAE). (4) conformal fallback
guarantee + latency table. (5) LoCoMo head-to-head vs D-Mem.

---

## 0 · Status (one screen)

| # | item | state |
|---|---|---|
| Lib | `janus-methods` built + deployed (sam-dev + ray), caches symlinked, end-to-end smoke ✅ | ✅ |
| Setup | forgetting-inducing dose: GLM-4-9B, heavy narrow SFT (gsm8k, 800 steps, lr 1e-4), NIAH @2k/8k + MMLU + gsm8k | 🟢 |
| Sweep v1 | protection matrix (12 configs) on ray 4×H100 | ◐ done, **null (broken recipe)** |
| Decide | — | blocked on a valid recipe |
| **Fix** | attn-proj-only SFT @lr1e-4/800steps **destroys** the model (Δgsm8k<0 even for `none`); switch trainer to **LoRA-all-modules / full-FT** that *learns* (Δgsm8k>0) then forgets | ⏳ next |

**Sweep v1 verdict (2026-06-04 17:30).** Heavier dose finally moved MMLU (−0.11) but
the SFT **degraded GSM8K itself in every variant** (incl. `none`, Δ −0.108) → it's
global destabilization, not learn-while-forget. No protection variant beats `none`;
NIAH unchanged (GLM-4 robust). **Precondition failed: the SFT must learn the new task.**
Root cause: attention-proj-only training can't learn GSM8K reasoning + lr 1e-4 wrecks
attention. → fix the trainer (LoRA on all modules / full-FT, moderate lr; protection
masks protected-head deltas), re-run. (3rd time the *setup*, not the method, is the blocker.)

> ⛔ **THESIS FALSIFIED at head-level (2026-06-04 eve).** Rigorous a-priori coupling
> across the 0.6→14B full-attn ladder, with partial(|out_norm)+within-layer+CI controls
> ([`mechanism-results-2026-06-04.md`](mechanism-results-2026-06-04.md)): **no robust
> head-level long-context↔forgetting coupling.** Within-layer, `attn_distance`/`sink`
> are **robustly NEGATIVE** (−0.45…−0.60, all sizes) → structural long-ctx heads are
> *gradient-AVOIDED*, not vulnerable. v1's "+coupling" = pooling + activation + depth +
> qwen3.5 hybrid artifacts. **The "same heads, two frontiers" claim does not hold.**
> Surviving fact: long-context heads are intrinsically *shielded* from SFT gradients.
> → paper pivot needed (options below). The v1 measurement infra/metrics are still sound.

**Decisive question (revised):** at the granularity that survives controls (**layer-level**, `attn_distance`), is there a clean coupling that scales — and can protecting those *layers* reduce forgetting beyond an activation-magnitude / depth baseline?

**Compute:** ray 4×H100 = the sweep. sam-dev 4×H100 = v1 mem-test (not ours, leave).

---

## 1 · The sweep (running)

Configs (`janus-methods/configs/`), distributed across ray GPU 0–3; `baseline_none`
runs first (its per-head ΔW seeds the `oracle_deltaw` criterion):

| lane | configs |
|---|---|
| gpu0 | **baseline_none** → **oracle_deltaw** (ΔW post-hoc oracle) |
| gpu1 | **m0_lc_freeze** (headline) → m0_lc_l2sp → m0_lc_orthogonal |
| gpu2 | site_retrieval_freeze → site_attn_distance_freeze → fisher_freeze |
| gpu3 | control_random → site_v_norm_freeze → trainedness_freeze → m0_lc_grad_scale |

Each cell logs before/after **NIAH (2k,8k) · MMLU · GSM8K**, per-head ΔW drift, and
the protected-set size. Results → `janus-methods/out/protect_*.json`.

**Read-out plan:** retention–plasticity Pareto (Δretention vs Δgsm8k) per variant;
the criterion ablation (lc/retrieval/attn_distance/v_norm vs random vs fisher vs
ΔW-oracle) isolates *which site criterion* matters; the operator ablation
(freeze/l2sp/orthogonal/grad_scale) isolates *how* to protect.

---

## 1.5 · PAPER A locked (2026-06-05): "Long-context heads are gradient-shielded during fine-tuning"
Plan: [`paperA-plan-2026-06-05.md`](paperA-plan-2026-06-05.md). Claim: long-ctx heads receive
*smaller* SFT gradients (shielded), task-objective-specific, activation-mediated, scale-fading;
explains instruction-SFT preserving long-context. Main tables (scale/family + objective) ✅ have
data. Pre-exps in flight: A1 carrier disentangle (offline), A2 a-priori→actual drift (LoRA),
M2 method payoff under LM-continued-pretraining (where the shield is absent). Methodological
cautionary (phantom couplings) folded in as rigor backbone.

## 2 · TODO queue
1. **Run the sweep** (🟢) → fill the Pareto + ablation tables.
2. **If a protection winner emerges:** 4-seed band + add Qwen3-8B & Qwen3.5-9B (cross-family) + code/long-ctx retention probes.
3. **If no forgetting even at 800 steps:** raise dose / switch to a long-ctx-eroding domain (short extractive SFT) / eval NIAH at 32k.
4. **Standalone methods:** `predict_i1` (zero-backward forgetting predictor) · `steer_i7` (act-drift un-forgetting) · `surgical_merge`.
5. **MoE instantiation** (super-expert protection, Qwen3-30B-A3B) — the headline novelty.

## 3 · Pointers
- Library: `janus-methods/` (README, `janus/{criteria,operators,train,eval,inference,predict,run,core}.py`, `configs/`).
- v1 results (archived line): `grid-metrics-2026-06-04.md` · `metrics-reference.md` · `metrics-curation-2026-06-04.md` · `ideas-brainstorm-2026-06-04.md` · `paper-draft-2026-06-04.md` · `matrix-archive-2026-06-04.md`.
- H3 v1 raw: `runs/h3/`.
