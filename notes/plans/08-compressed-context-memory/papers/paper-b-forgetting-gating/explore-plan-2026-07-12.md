# Method-improvement exploration — 10h × 6 GPU (2026-07-12)

Goal: turn the diagnosis (F37 crossover, F39 ∞Bench gap) into a **method that beats RAG on average across regimes**, not just ties per-regime. Grid tag `ax_*` on `grid_impredesign` (shared disk). 6 GPUs: free1 GPUs 0–3 (physically distinct from the auto-baseline on d1525=free1 GPUs 4–7) + free3 GPUs 0–1. Concurrent auto-baseline (`au_q8_*`, peak-tau3, all 20 benches) keeps running on d1525×3+d1530×2.

## Hypotheses
- **H1 (routing):** a per-input router that sends lexically-anchored inputs to `chunk` and diffuse/reasoning inputs to `span` recovers best-of-both (F37). *Which router?* → sweep.
- **H2 (budget, targets F39):** IMP loses ∞Bench-choice (50 vs RAG 70) because keep=0.5 over-256-chunks retains too many distractors for MC log-likelihood; **tightening the budget when a lexical anchor localizes the answer** should lift it toward RAG.
- **H3 (generality):** the crossover + the winning router hold on **linear/GDN (Qwen3.5-9B)** too.

## Design (`ax_launch.sh`, code archived below)
**New method knobs** (patched into `_imp`, backward-compatible defaults):
- `GCM_IMP_AUTO_ROUTER ∈ {peak, mc, qover}` — peak = BM25 max/mean ≥ τ; mc = options-present→span else chunk; qover = query∩context token-overlap ≥ τ.
- `GCM_IMP_ADAPT_BUDGET=1` — when routed to chunk, tighten keep→0.25·L (isolate answer passage).

**Phase 1 — router study** (Qwen3-8B, keep 0.5): routers {peak τ2, peak τ4, mc, qover 0.5} × 8 crossover benches {squad, hotpot, quality_hard, ∞Bench-choice, musr, ruler, babilong_qa3, lb_hotpotqa}. (peak τ3 already covered by the `au_*` baseline.) → 32 cells.

**Phase 2 — adaptive budget** (Qwen3-8B): 6 F39-relevant benches × {chunk k0.1, k0.15, k0.25, auto+adaptbudget, rag b512, rag b1024}. → 36 cells. Tests whether tighter budget closes ∞Bench and where the accuracy/budget knee is.

**Phase 3 — ALL-family generality** (`fam_*`, N=100, ctx≤8k): the improved method must be validated on **every family by default, not one model.** 9 families/sizes × 5 crossover benches × {auto(peak τ3), chunk, span, rag} = **180 cells** on free3×2. Models: Qwen3-{1.7B,4B,14B} (dense size-scaling), Qwen2.5-7B (prev-gen), Qwen3.5-{4B,9B} (linear/GDN), GLM-4-9B, Ministral-8B, Llama-xLAM-8B. (Qwen3-8B already fully covered by `rf_*`/`au_*`; gpt-oss-20B & Moonlight-16B excluded — harness-broken per F31.) Verifies the chunk↔span crossover + auto's best-of-both across architectures & families.

Total: free1 P1+P2 = 68 (4 GPU); free3 `fam_*` = 180 (2 GPU); + auto-baseline `au_*` (20 benches, Qwen3-8B) on d1525×3+d1530×2. ETA ~6–8h.

## Decision rules (what "improvement" means)
- **Best router** = the one whose per-bench score ≈ max(chunk, span) on the *most* benches (best-of-both).
- **Win vs RAG** = `auto`(best router) ≥ RAG on hard-MC/reasoning (quality_hard, musr, babilong) *while* ≈ RAG on extractive QA (squad, hotpot) → then **auto beats RAG on the regime-average**, which is the paper's method claim.
- **F39 closed** if chunk/auto+AB reaches ≥ 60 on ∞Bench-choice (RAG 70.3, full 52.8).

## Provenance
Router/budget code snapshot: [`configs/`](configs/) (to archive after run). Harvest → this file + `imp-redesign-results.md`. Facts → `matrix-facts.md`.
