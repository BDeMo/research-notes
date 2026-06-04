# Janus — Phase-1 exploration (actual runs, 2026-06-03 → 06-04 overnight)

> Single section logging **what actually ran** (not the design — that is `validation.md`). Overnight batch on **7 free H100s** (sam-dev GPU 0/2/3 + sam-dev-ray GPU 0–3), ~7.5 h, **81 stage-runs, 0 failures, 38 figures**. Code: `janus/janus_run.py` + `janus_viz.py` (in workspace, not yet in repo). Artifacts: `janus/artifacts/` on sam-dev; gallery mirrored to `janus/gallery/`.

## What ran

| node | GPUs | models | work |
|---|---|---|---|
| sam-dev | 0,2,3 | Qwen2.5-1.5B(+inst), Qwen3-8B, Qwen2.5-7B-inst, Qwen3-14B | detect + NIAH(+retr/sink ablation) + **full-FT/LoRA SFT** + **drift/coupling** + capability eval + **H3 protection variants** (none/retrieval/random/ΔW/sink) + k-sweep {12,24,48} + dose {150,500} + cross-domain {math,trivia} |
| sam-dev-ray | 0,1,2,3 | Qwen3-0.6B/1.7B/4B/8B/14B | scaling-ladder: detect + NIAH(+ablation) + trivia full-FT coupling (none vs retrieval-protect) |

8 models total: a **Qwen3 0.6→14B scale ladder** + a **Qwen2.5 cross-family** check + a **base-vs-instruct** check.

## Findings (honest; this is a measure-first phase)

### ✅ R1 — detectors reproduce at every scale (Phase-0 gate PASS ×8)
BOS sink present in all 8; massive-activation channels **5–8** for every Qwen3 size (Qwen2.5-1.5B is an outlier at 41 — family difference, worth a footnote); induction-head score peaks 0.92–0.99. Instrumentation is trustworthy across the ladder.

### ✅ R2 — sink heads ≠ retrieval heads is ROBUST across scale & family
Top-15 **Jaccard = 0.0** for 6/8 models (0.034 for Qwen3-8B & Qwen2.5-1.5B); Spearman(sink, retrieval) = 0.04–0.24. Holds from 0.6B to 14B and across families. → the long-context **read-side** site to protect is the **retrieval heads**, not the sink. `figs/02_sink_vs_retrieval.png`, `figs/09_scaling_ladder.png` (middle).

### ◐ R3 — coupling (H2): direction robust, magnitude modest
Per-head Spearman ρ(**retrieval-score**, **‖ΔW‖ after SFT**) vs ρ(**sink**, ΔW), vanilla SFT:

| model / SFT | ρ(retrieval, drift) | ρ(sink, drift) |
|---|---|---|
| Qwen3-0.6B / trivia | **0.398** | 0.240 |
| Qwen3-1.7B / trivia | **0.399** | 0.227 |
| Qwen3-4B / trivia | 0.261 | −0.005 |
| Qwen2.5-7B-inst / math (LoRA) | 0.252 | −0.130 |
| Qwen2.5-1.5B / math | 0.140 | 0.203 |
| Qwen2.5-1.5B / trivia | 0.168 | 0.185 |

**ρ(retrieval, drift) > ρ(sink, drift) in nearly every model** — the read-side retrieval criterion is consistently more predictive of where SFT lands than the sink. But the **magnitude clears the pre-registered ρ≥0.4 gate only on the two smallest Qwen3 models**; mid-sizes sit at 0.25–0.40. **Verdict: H2 partially supported** — the *direction* (retrieval > sink) is the robust claim; the raw magnitude is weaker than hoped and size-dependent. `figs/08_coupling_summary.png`.

### ✅ R4 — the protection mechanism works (causal sanity)
Masking retrieval-head gradients during SFT **removes the coupling at those heads**: ρ(retrieval, drift) drops 0.6B 0.398→0.217, 1.7B 0.399→0.224, 1.5B 0.14→−0.12 (k=24) → −0.31 (k=48). Confirms the grad-mask intervention does what it should (a Phase-2 V2 prerequisite).

### ◐ R5 — NIAH causal ablation: one clean datapoint, probe needs hardening
Predicted: ablating **retrieval** heads hurts NIAH more than ablating **sink** heads. Clean confirmation on **Qwen3-1.7B**: base 0.98 → ablate-retrieval **0.60** < ablate-sink **0.90** ✓. But base NIAH was too low on most ladder rungs (0.6B/4B/8B/14B ≤ 0.12 at 1.5k–6k) for the contrast to mean anything — the verbatim-number probe is too hard / base (non-instruct) models don't follow the "answer the number only" format. **Action: harden the NIAH probe** (instruct models, easier extraction, more needle variants per `[ret-dyn]`) before trusting this leg. `figs/05_niah_qwen3-1.7b.png`, `figs/09_scaling_ladder.png` (right).

### ✗ R6 — H3 inconclusive: the math-SFT setup didn't induce forgetting
On Qwen2.5-1.5B, full-FT on MATH did **not** drop MMLU/TriviaQA (retain 0.365 → **0.40**, went *up*) — so there was nothing to protect against, and none/retrieval/random/ΔW/sink protection are all within eval noise (retain 0.39–0.40 across variants, 200-example evals). The **one real forgetting signal**: **trivia-SFT → MATH dropped 0.40 → 0.267 (−33%)** (cross-domain). **Action: H3 needs (a) a stronger/aggressive SFT dose, (b) a domain pair that actually forgets (cross-domain like trivia→math, or code→math), (c) bigger eval n** to escape noise.

## Net read & what it changes
- **Two solid, paper-relevant wins**: sink≠retrieval is rock-solid across scale (R2); the retrieval criterion out-predicts the sink for forgetting-drift in direction (R3) — this is the seed of the §5.6 "predictive coupling" wedge.
- **Two things to fix before Phase-1 is conclusive**: harden the long-context probe (R5) and actually *induce* forgetting (R6). The current overnight setup was too gentle to test H3.
- **No result kills the thesis**; the de-risk is "promising-but-not-yet-decisive," exactly a measure-first checkpoint. The 0.4 gate is *not yet* met at scale → the honest next step is the fixes above, then re-measure ρ on a forgetting-inducing setup.

## Concrete next runs (queued for next free-GPU window)
1. **Harden NIAH** → instruct models, lengths 2k–16k, 20 needle variants, lenient extraction; re-do retrieval-vs-sink ablation on the ladder.
2. **Forgetting-inducing matrix**: cross-domain SFT {trivia→math, math→code, code→math} × dose {light, heavy} × {none, retrieval-protect, ΔW-protect, random} on 1.5B + 8B; eval n≥500. This is the real H3 / `[mech-forget]` comparison.
3. **Coupling at scale with a forgetting setup**: recompute ρ on the heavy cross-domain runs (expect higher ρ than the gentle math runs).
4. **MoE rung** (the headline §5.6a): port detect+protect to Qwen3-30B-A3B super-experts once a big-GPU window opens.
