# 2026-06-04 — Cross-repo harvest (summarize all repos → research-notes)

> Session goal (per user): summarize everything across all my repos, pull the useful facts into research-notes' right places, then append a weekly-deck input for the update (leaving prior weeks archived). This is a *status roll-up* across the ~6 active research repos as of 2026-06-04.

## Repo inventory (research-relevant, under `~/workspace/`)

| repo | role | status (2026-06-04) |
|---|---|---|
| `research-notes` | the notebook (this repo) | active; plans 01/03/08/09 + known/ + memory/ |
| `janus`, `janus-methods` | Plan 09 implementation (long-ctx ↔ forgetting) | active; measurement done, method-line blocked on SFT recipe |
| `latent-mem-paper` | Plan 08 paper (soft-prompt memory) | paper drafting → **ICLR 2027** |
| `mem-test` (`mem-embedding`) | Plan 08 experiments | v1 characterization complete |
| `test_env/EMNLP-dllm-BoN` | dLLM Best-of-N verifier paper | **EMNLP** draft (review template) |
| `rca-demo-qwen` (+ older `rca-demo`) | RCA application (Nokia) | curriculum LoRA adapters trained on Qwen2.5/Qwen3 |
| `intern-project` | Nokia intern cohort | multi-intern: RCA, agent-skills, MoECompiler, telecom-FM |

## Per-repo harvest

### 1. Janus = Plan 09 (`janus`, `janus-methods`) — already deeply harvested into `notes/plans/09-intrinsic-site-protection/`
The big update vs the 2026-06-03 plan draft is **execution + an honest pivot**:
- Built a **39-metric / 12-angle** intrinsic instrument (5-pass minimal schedule); ran a **41-cell grid** (4 families × 12 benchmarks): Qwen3-8B, Qwen2.5-7B-Instruct, GLM-4-9B, Qwen3.5-9B (hybrid).
- **v1 measurement** looked like a uniformly-positive per-head long-ctx × forgetting coupling (ρ 0.17–0.56) — see `paper-draft-2026-06-04.md`.
- ⛔ **Falsified under controls** (partial-out activation magnitude + within-layer + per-model + CIs): no robust head-level coupling; the "same heads, two frontiers" thesis fails at head level. (`mechanism-results-2026-06-04.md`)
- ✅ **Surviving findings** (`facts-2026-06-04.md`):
  - **F-shield** — long-context structural heads are gradient-**shielded** during SFT (within-layer ρ(grad, attn_distance) ≈ −0.5…−0.6; gradient avoids long-reach/sink heads, mediated by their lower activation magnitude).
  - **F-scale** — shielding is strong at 0.6–14B, **neutralizes to ≈0 by 30B/32B** (no flip to vulnerability); the activation-magnitude driver also fades at 30B+.
  - **F-domain** — it's a **task/instruction (Q→A) objective** effect: holds for hotpot/narrativeqa, **vanishes for plain LM (wikitext)** → instruction-SFT shields long-context; continued LM-pretraining would not.
  - **F-consequence** — explains why **NIAH stays robust under instruction-SFT** (long-context isn't where forgetting lands).
  - **F-collapse** — a vendor-independent representation-collapse axis (eff-rank↓ / massive-acts↑ / prediction-crystallizes) is robust but *structural*, not the LC↔CF bridge.
- **H3 causal test inconclusive** — GSM8K-SFT didn't induce long-ctx forgetting (GLM-4 NIAH 1.0→1.0). `janus-methods` (criterion×operator library, 17×6) blocked on a valid forgetting-inducing recipe (attn-proj-only SFT @ lr1e-4 destroys the model).
- **Paper pivot**: from "same heads, two frontiers" → "long-context heads are intrinsically shielded from instruction-SFT gradients (why NIAH survives); a small/mid-model effect that neutralizes by 30B." Infra/metrics sound.

### 2. Plan 08 paper (`latent-mem-paper`) — reframed as "Bit-Capacity Limits of Soft-Prompt Memory"
- **Title**: *Bit-Capacity Limits of Soft-Prompt Memory: Characterising What a Learned Memory Wrapper Can and Cannot Encode about Long Context.* Target **ICLR 2027** (ICLR-2026 template). `main.tex` = conference story; `details.tex` = full metrics/recipes.
- **Method**: recurrent residual memory $m_t = m_{t-1} + \alpha\,\Delta m_t$ on a **frozen** base, read via `apply` interfaces (prefix / residual cross-attn / interleave / hybrid / `xattn` = memory-queries-chunks). Frozen Qwen3-8B & 14B.
- **Two findings**: **(i) a sharp bit-capacity wall** — exact recall ~1.0 at ≤2-bit answer entropy → 0 by ~3 bits; **Wrap and matched-K Gist hit the wall at the same place** → a property of the soft-prompt *read* interface, not the write architecture. **(ii) a 3-regime zero-shot transfer law** — Phase-Y 4-seed bands (frozen Qwen3-8B): QuALITY OURS **0.193±0.032** ≈ Gist 0.180±0.044 (regime A, both beat no-ctx 0.073), MuSR-mm 0.492 (B, at chance), RULER-NIAH **0.000** both (C, collapse) vs FullCtx 0.995.
- **Below the wall it works but not robustly**: categorical-NIAH-held is bimodal across 4 seeds {0.61, 0.43, 0.06, 0.00} → mean **0.27±0.29** (vs matched Gist 0.07); data-scale 100→500 lifts em 0.10→0.43 (seed 42). Wrapper = **~218 M trainable (~2.67% of base)** → "small model, not lightweight adapter".
- **Status**: draft, **workshop-track lean** (main-venue deferred per `v1-submission-checklist`); Fig-2 (3-regime bar chart) still missing; many `\Pend` cells. Polished, generalized version of the 2026-06-03 "3-regime" harvest.

### 3. dLLM Best-of-N (`test_env/EMNLP-dllm-BoN`) — verifier-readout paper
- **Title**: *Where Should a Selection Verifier Read for Best-of-N Sampling in Diffusion Language Model Reasoning?* **SUBMITTED to EMNLP 2026** (2026-05-26, ~2 min before deadline); rebuttal staged.
- **Thesis**: AR decoder-only verifiers read the final position because of **"confluence"** (causal mask → that position's receptive field is the whole prompt+solution). **dLLMs break confluence** (bidirectional masked-canvas fill → the tail is often EOS/mask/padding). So read the **output-span mean** (`dORM_mean`), not the inherited last position. (Note: confluence ≠ attention sink — a clean distinction.)
- **Result** (N=4, T=0.7, frozen base, V-STaR labels): on **LLaDA-8B-Instruct** / GSM8K, switching readout `last`→`mean(R_out)` lifts the linear head **75.4 → 83.7** (Selector-Gap 14.6→6.3). On **LLaDA2.0-mini** (16.26B MoE, ~1.43B active) the boundary **flips** (`last` wins among linear heads; only the MLP separability diagnostic recovers signal). No universal rule — match readout to generation geometry, check linear separability, separate verifier error from candidate-set limits.
- **Relation to us**: connects to Plan 03 (W-space BoN) and the AR→dLLM extension (DR15); the confluence/sink distinction ties to `known/transformer-internals/`.

### 4. RCA application (`rca-demo`, `rca-demo-qwen`) — the Nokia RCA model line
- **Matrix**: 16 checkpoints × 4 RCA datasets (**Nezha** [FSE'23], **OpenRCA-500**, **RCAEval**, **LincYaw**) + 5 general benchmarks (ifeval, gsm8k, bbh, truthfulqa, mmlu). Bases: **Qwen2.5-7B-Instruct, Qwen3-8B**. Adapters: `q{25,3}_curr_{1..4}` (curriculum), `q25_curr_v2_{1..4}`, `q{25,3}_mix`.
- **Headline numbers** (`recommendation_f1`): `q25_mix` strongest complete ckpt — 0.034 / 0.305 / 0.733 / 0.242 (nezha / openrca500 / rcaeval / lincyaw), `json_parseable`=1.0; general-cap erosion bounded (MMLU 0.749–0.769; worst drop q25_mix GSM8K 0.795→0.715).
- ⚠️ **Anti-forget v2 quick-win FAILED** (2026-05-27, 0/1 criteria): oversample + 5% anchor pushed Qwen2.5 to "valid JSON, empty fields" (nezha evidence_overlap 0.578→0.000); v2a/v2b ablation queued. **Qwen3 collapses under cumulative SFT** (json_parseable falls, output length blows up — a *format* failure).
- **Lesson recorded**: the base-mismatch bug (`resolve_model_path` priority inversion → "7B" runs were really 1.5B) invalidated all pre-2026-05-20 SFT numbers — the project's most consequential silent bug.
- **Infra shipped 2026-06-04**: `rca-demo-qwen/train_console/` — a FastAPI + browser control plane that schedules LoRA/QLoRA/DoRA/rsLoRA/full-FT runs across pods (ssh+tmux, SQLite runs DB, SSE loss curves) with published-baseline presets. Plus the 2-slot Gradio chat (14 adapters, live attention heatmaps, on Nokia VPN).
- **New today**: "compressed-memory long-context RCA v0" line filed in `rca-demo` (Mingjia, = Plan-08 v0 applied to RCA) — datasets planned LoCoMo / LongBench / RULER / NIAH / SWE-bench / BugsInPy / Defects4J / QuixBugs + the 4 RCA sets; numbers TBD.

### 5. Intern cohort (`intern-project`) — Nokia summer-2026 interns (mentor: Liang)
Weekly leadership review is AI-summarized from this repo, so intern folders *are* the report substrate.
- **Xueqi** — "Small-Model Distillation + Self-Evolving Learning for Merlin's Agentic Reasoning": distill **MiniMax-M2.5 (230B MoE)** into role-specialist students (planner/tool-caller/reflector) + teacher-fallback cascade. `empirical-results-2026-06-01` (Phase A+B; 3 teachers × 3 students × 3 MAS topologies CrewAI/AutoGen/MetaGPT): **internal-structure-matters strongly supported** (Spearman 0.93 between Phase-A self-match and compression resilience); **H1.A "high-K → more headroom" REFUTED** (MetaGPT K=5 most damaged); specialization gap >0.05 on all 3 topologies; **verdict PROCEED**.
- **Mingjia** (user) — "Building NokiaFM-RCASkill" (beat internal Qwen2.5-1.5B+curriculum baseline 0.264 on Liang's held-out) + owns the rca-demo compressed-memory v0 (Plan 08). This is the *Nokia-intern* face of the same RCA + long-ctx work tracked in plans 08/09.
- **Jiamu** — `MoECompiler`: training-free expert pruning of MiniMax-M2.5 (256→192/128 experts). **192e: 8/8 PASS Merlin smoke @ iso-throughput, −18% HBM**; 128e 7.5/8 @ −45% HBM; layer-prune 0/8 (negative result). Relevant to the MoE / super-expert thread.
- **Tianze** — "NokiaFM-MM": 17-task multimodal telecom FM (Qwen2.5-1.5B + LoRA + TaskID + reprogramming + VQ + gated cross-attn).
- **Paimon** — "AutoSkill for Merlin": extract versioned `SKILL.md` from Merlin traces (SkillBank + retrieval, no fine-tune). **Feng** — cluster on-ramp (Ray/kube). **Liang** — `FM-Telecom` time-series FMs (TimesFM, Time-LLM, Moirai, Toto).

## What was harvested where (this session)

- **This entry** = the cross-repo roll-up (the "summarize all repos" deliverable).
- `memory/context.md` — Janus active thread corrected to executed/falsified/shielding status (done 2026-06-03 edit); added compact "other active repos" pointers.
- `docs/matrix/README.md` — session index row + active-threads refresh (Janus executed, Plan-08 bit-capacity/ICLR-2027, dLLM-BoN EMNLP, RCA-app).
- `notes/plans/08-.../slides/weekly/2026-06-w02.tex` — new weekly-deck input (this week's update), appended to `slides/main.tex`; w04/w01 left unchanged (archived).
- Janus IDs (retrieval-head, duo-attn, focusft, sink-forget, das-moe, …) already in `knowledge-sources.md` (logged by the Janus line).

## Decisions
- **Janus is a measurement paper with an honest pivot**, not (yet) a causal "two frontiers" paper. The shielding finding (F-shield/scale/domain) is the real, defensible result.
- **Plan-08 headline = bit-capacity wall** (supersedes "3-regime law" as the paper framing).
- No new plan folders for RCA-app / dLLM-BoN — they live in their own repos; tracked here as active threads with pointers (avoids notebook scope-creep).

## Next steps
- Janus: fix the forgetting-inducing SFT recipe (LoRA-all-modules/full-FT that learns-then-forgets); re-run protection sweep; expert-level metrics for the MoE super-expert probe.
- Plan 08: bit-capacity paper polish for ICLR 2027; v1.5 do-no-harm gate; v2 cross-session latent memory.
- dLLM-BoN: EMNLP submission polish.
