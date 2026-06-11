# Paper B — Summary Matrix (writing-agent handoff)

> **2026-06-10: SUPERSEDED for the current program by the v1.7 reframe.** The active brief is
> [`summary-matrix-v1.7-2026-06-10.md`](summary-matrix-v1.7-2026-06-10.md) (gated compressor module; the
> compress-decision confusion matrix / F1 objective; compressor-level comparison with vanilla Cartridge/Gist).
> This file stays as the **v1.5/v1.6 archive**; its still-valid evidence (forgetting, capacity wall, cross-model
> signal, TARG) is carried into v1.7 §7 and must be re-validated under the v1.7 protocol.
>
> **Purpose:** single-source brief for the **writing agent**. Everything needed to draft Paper B —
> thesis, claims↔evidence (with exact numbers + source files), the main table, baselines, novelty
> defense, the do-not-overclaim list, section plan, asset index, status. **Read this first**; deep-dive
> via the links. Companion docs: [`README.md`](README.md) · [`method.md`](method.md) (v1.5 architecture + math) · [`logic.md`](logic.md) ·
> [`framing.md`](framing.md) · [`baselines-and-novelty.md`](baselines-and-novelty.md) · [`outline.md`](outline.md).
>
> **Two repos:** evidence ledger = `mem-embedding` repo `summary/matrix.md` (sections cited as §N);
> raw data/figures = `research-notes` repo `notes/plans/08-model-outputs-delta-w/raw/` (cited `raw/…`).
> **Base model for all numbers = Qwen3-8B unless noted.** **HONESTY FIRST: do not exceed the numbers here.**

---

## 0. Abstract seed (draft, ~120 words)
Adapting a pretrained LLM to new data hurts what it already knows: fine-tuning the weights causes
*catastrophic forgetting*, and bolting on an always-on augmentation causes *negative transfer* when the
augmentation is irrelevant. We study **do-no-harm adaptation**: a **detachable, model-agnostic memory
module** on a **frozen base** that is trained on a target data distribution, **gated by a robust
intrinsic signal** so it is applied only when it helps and otherwise falls back to the exact base
(do-no-harm *by construction*). Across 8 benchmarks and 7 model families we show the module *adds*
in-distribution competence (QA +2–9 pts) where weight fine-tuning *forgets* (every QA task drops below
no-context), map the **sharp capability boundary** where the gain vanishes, and show a simple
intrinsic gate recovers most negative transfer with **zero per-model tuning**. We are honest about the
limits: a capacity wall on exact recall, and that a trivial base-uncertainty gate is competitive.

---

## 1. Thesis + positioning
- **Thesis:** *do-no-harm adaptation* — a detachable, model-agnostic, per-distribution memory module on
  a frozen base; **helps in-domain, never harms** (gate ⇒ falls back to exact base).
- **Headline framing:** "a pluggable memory that **knows when not to fire**" (agent/reliability angle, not narrow CF). See [`framing.md`](framing.md).
- **Scope (decided, popular):** **adaptive latent memory for LLM agents** — fast (few latent tokens), safe (cross-model do-no-harm gate), accurate when needed (**falls back to full context**). Community/venue = **agentic memory** (ICLR'26 MemAgents workshop), *not* generic prompt-compression. Full answer to "compress→latent + gate + fallback, novel?" + lit (Gist/Cartridges/ACON; TAAC/ContextPilot; SLT/SeLaR) in [`scope-and-compression.md`](scope-and-compression.md). **Honest:** gate+fallback exists at text-level (TAAC) & reasoning-level (SLT) — our novelty = applying it to a *learned latent* memory + cross-model do-no-harm signal, not the mechanism.
- **Scope (declared, not a weakness):** target **in-domain** lift; floor = **in-task / in-dataset no-harm**; cross-domain → gate closes. **Not** fact storage (capacity wall → Paper A).
- **⚠️ Novelty squeeze (must respect):** two ICLR'26 neighbors — **Cartridges** (frozen-base pluggable KV-cache memory = the *module*) and **TARG** (training-free base-uncertainty gate = the *gate*; ≥ ours, §7d). **Our white space = the do-no-harm *treatment*** (boundary + do-no-harm-by-construction for a *learned* module + cross-model gate + honest negatives). **Complementary to Cartridges; extends TARG from retrieval to learned modules.** Full defense in [`baselines-and-novelty.md`](baselines-and-novelty.md).

---

## 2. Contributions (status + the one number each rests on)
| # | contribution | headline number | status |
|---|---|---|---|
| **C1** | forgetting is real on **two axes** | SFT-LoRA: trivia no-ctx 0.475→**0.250**; ungated OOD mean Δ **−0.075** (cross) | ✅ solid |
| **C2** | **in-distribution competence + sharp boundary** | QA +2..+9 pt in-dist; mean Δ +0.017 (same-dataset) → −0.075 (cross) | ✅ solid |
| **C3** | **do-no-harm gate**, cross-model, zero-tuning | recovers ~85–95% of negative transfer; LOMO AUROC 0.71 (7-fam); by construction g→0=base | ✅ solid (offline) |
| **C4** | **cross-model intrinsic signal** study | `delta_last` dir-consistent in **all 7** families, AUROC 0.59–0.80 | ⚠️ TARG competitive (§7d) |
| **C5** | **honest negatives** (sharpen the design) | soft-gate & multi-depth both collapse generation; capacity wall RULER 0→.995 (full-ctx) | ✅ solid |

---

## 3. MAIN TABLE (Qwen3-8B, module trained in-distribution) — `raw/main-table-2026-06-05/main_table.csv`
| bench | task | no-ctx (floor) | SFT-LoRA¹ | Cartridge-lite² | Gist-lite² | **OURS (in-dist)** | Δ | full-ctx (ceiling) |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| trivia_qa | exQA | 0.475 | 0.250 | pending | pending | **0.571** | **+0.086** | 0.635 |
| squad_v2 | exQA | 0.170 | 0.149 | pending | pending | **0.231** | **+0.056** | 0.656 |
| hotpot_qa | exQA | 0.228 | 0.226 | pending | pending | 0.247 | +0.003 | 0.623 |
| narrativeqa | absQA | 0.139 | 0.118 | pending | pending | **0.194** | **+0.035** | 0.150 |
| ms_marco | absQA | 0.214 | 0.198 | pending | pending | **0.236** | **+0.023** | 0.282 |
| quality | MC | 0.240 | 0.233 | pending | pending | 0.200 | −0.060 | 0.200 |
| musr_mm | MC | 0.490 | 0.493 | pending | pending | 0.480 | −0.010 | 0.495 |
| ruler_niah | needle | 0.000 | 0.000 | pending | pending | 0.000 | +0.000 | 0.995 |

**Takeaways the prose should make:** (a) OURS *adds* QA competence in-dist while **SFT forgets** (every QA below no-ctx); (b) compression **beats full-ctx on narrativeqa** (0.194>0.150); (c) **capacity wall**: full-ctx ≫ OURS on extractive/needle; (d) MC/needle don't benefit even in-dist (honest).
¹ SFT-LoRA = **mix/multi-task** SFT (not per-bench) — matched per-bench-SFT control is TODO. ² Cartridge-lite (prefix-tuning) & Gist-lite from `membase_q` queue (filling).

---

## 4. Claim → evidence map (cite these exact numbers; do not round up)
| claim (for the prose) | exact number | source (repo) | confidence |
|---|---|---|---|
| SFT fine-tuning forgets | trivia no-ctx 0.475→**0.250**, full-ctx 0.635→0.301; squad 0.656→0.386; hotpot 0.623→0.342 | `raw/sft-2026-06-05/mix_sft_qwen3_8b.csv`; matrix §7c | high (caveat: 400-step LoRA on 500 items, overfit; magnitude likely exaggerated, direction canonical) |
| in-dist competence | trivia +0.086, squad +0.056, narrativeqa +0.035, ms_marco +0.023, hotpot +0.003 | `raw/grids-xmodel-2026-06-05/transfer_matrix.csv`; matrix §8 | high |
| sharp boundary | mean Δ +0.017 (same-dataset) → −0.013 (same-task) → −0.034 (same-domain) → −0.075 (cross) | `…/transfer_long.csv`; matrix §8 | high |
| mix-train doesn't broaden | trivia +0.086→**−0.150** when trained on all-mixed | `raw/mix-2026-06-05/mix_results.csv`; matrix §8 | med (high seed variance) |
| gate recovers most harm | ungated 16/35 → learned 17/35 cells ≥ no-ctx; **oracle 35/35**; trivia 0.230→0.448 (no-ctx 0.462) | `raw/baselines-2026-06-05/gate_baselines.csv`; matrix §7b | high (honest 5-fold CV) |
| gate transfers cross-model | `delta_last` LOMO AUROC **0.71** (unseen family, 7-fam) | `…/gate_transfer_lomo.csv`; matrix §2b | high |
| do-no-harm by construction | gated combine g→0 ⇒ exact base (E5) | `src/mem_embedding/wrapper.py`; matrix §0 | high (exact) |
| cross-model signal | `delta_last` dir-consistent all 7, AUROC 0.59–0.80; logit-lens model-specific | `…/xmodel_consistency_7models.csv`; matrix §2/§2b | high |
| **TARG ≥ our signal (3-fam)** | TARG margin_0 AUROC 0.624 / LOMO 0.604 vs ours 0.584 / 0.557 | `raw/baselines-2026-06-05/gate_read_baselines.csv`; matrix §7d | high — **report honestly** |
| soft-gate fails | gate stuck ~0.5, per-token residual corrupts gen | matrix §7/H4 | high (negative) |
| multi-depth fails (v1) | single-layer gate→0 (no effect); multi-layer gates open, trivia 0.463→0.00/0.005 | `raw/deep-2026-06-05/multidepth_trivia.csv`; matrix §7c | high (negative; v2 zero-init ablation running) |
| capacity wall (ceiling) | RULER 0→**0.995**, squad .17→.656 with full-ctx; ours ~0 on needle | `raw/fullctx-2026-06-05/qwen3_8b_ceiling.csv`; matrix §7 | high (→ Paper A) |

---

## 5. Baselines (for experiments + related work)
- **Memory module:** no-ctx (floor) · full-ctx/ICL (ceiling) · **SFT/LoRA** (forgetting) · **Cartridges** ([2506.06266], cite + Cartridge-lite/Gist-lite proxies running) · Gist/ICAE/CCM/AutoCompressor (cite).
- **Gate:** **TARG** ([2511.09803], ✅ §7d) · Self-RAG · FLARE · Adaptive-RAG · confidence-threshold · **oracle** (✅ §7b).
- **Multi-layer (ablation):** **LLaMA-Adapter** ([2303.16199], = our scalar-gate) · prefix/P-tuning-v2 · input-only (ours, §8).
- **Forgetting (motivation):** Luo CF (2308.08747) · PECFT survey (2504.13822) + JANUS-LoRA/PLATE (PEFT-still-forgets) · EWC/LwF.
Full table + IDs: [`baselines-and-novelty.md`](baselines-and-novelty.md) §1. **Consolidated paper-ready Related Work (6 families + our delta): [`related-work.md`](related-work.md)** — adds KV-cache compression (SnapKV/PyramidKV), **selective-prediction/abstention** (the do-no-harm framing + risk–coverage/AURC metric), and memory-augmented/agentic-memory (RMT/MemGPT/AgeMem/RecMem).

---

## 6. Novelty defense (for related-work + rebuttal)
| reviewer objection | response |
|---|---|
| "Cartridges already does pluggable frozen-base memory" | we don't claim the module; we add **when to use it** (boundary + gate). Cartridges is always-on; complementary. |
| "TARG already gates augmentation, model-agnostic" | TARG gates **retrieval**; we gate a **learned module**, do-no-harm **by construction**, + cross-model study + boundary. Honest: TARG competitive → reported as baseline. |
| "Just an empirical study" | own it: contribution = boundary law + by-construction do-no-harm + cross-model gate transfer + honest negatives. |
| "Single model / narrow" | 7 families (signal §2); declared in-domain scope; honest 3-family head-to-head limit. |

---

## 7. ⛔ Do-NOT-overclaim list (hard constraints for the writing agent)
1. **Our signal is NOT the best gate** — a trivial TARG base-uncertainty gate ≥ ours on the 3 families tested (§7d). Frame the *gating approach* + cross-model robustness, not signal supremacy.
2. **Gate is offline/routing**, not online/live — say "post-hoc routing on logged signals."
3. **do-no-harm is "recovers most" (~85–95%), NOT absolute** — learned gate 17/35 cells ≥ no-ctx, oracle 35/35. Strict do-no-harm only by construction (g→0), not empirically guaranteed by the learned gate.
4. **SFT row = mix/multi-task**, not per-bench matched control (TODO).
5. **No fact storage / parametric memory claim** — capacity wall (RULER→0). That's Paper A.
6. **7-family head-to-head not done** (ray/test pods deleted) — `delta_last` LOMO 0.71 is from a *prior* 7-family run, the head-to-head vs TARG is 3-family.
7. **In-dist only helps QA**, not MC (quality/musr) or needle (ruler).
8. **Multi-layer v1 = collapse** (negative); the zero-init v2 ablation result is pending — don't claim multi-layer works until §7e lands.

---

## 8. Section plan (where each piece goes)
1. **Intro** — two forgetting modes (read: negative transfer; write: CF); thesis. Hook = SFT-forgets row (§3/C1).
2. **Related work** — §5 / [`baselines-and-novelty.md`](baselines-and-novelty.md): Cartridges, TARG, LLaMA-Adapter, PECFT.
3. **Method** — frozen-base detachable wrapper (architecture fig: `raw/`/`summary/2026-06-05/wrapper-connect.png`, `architecture-v1.5.png`); the gate (do-no-harm by construction).
4. **The forgetting problem** — C1 (SFT §7c + ungated §8).
5. **In-distribution competence + boundary** — C2, **Main Table (§3)** + heatmap (`raw/grids-xmodel-2026-06-05/transfer_heatmap.png`).
6. **Do-no-harm gate** — C3, §7b hierarchy + §7d TARG baseline (honest) + the intrinsic signal §2.
7. **Multi-layer ablation** — C5 / §7e (pending) + LLaMA-Adapter framing.
8. **Limitations** — §7 do-not-overclaim list; capacity wall → Paper A.

---

## 9. Asset index (paths; research-notes repo unless noted)
| asset | path |
|---|---|
| **main table** | `raw/main-table-2026-06-05/main_table.csv` · builder `mem-embedding:scripts/main_table_build.py` |
| ceiling (floor/full-ctx) | `raw/fullctx-2026-06-05/qwen3_8b_ceiling.csv` |
| SFT forgetting | `raw/sft-2026-06-05/mix_sft_qwen3_8b.csv` |
| capability boundary | `raw/grids-xmodel-2026-06-05/transfer_{matrix,long}.csv` · `transfer_heatmap.png` |
| gate hierarchy + locking | `raw/baselines-2026-06-05/gate_baselines.csv` · `locking_by_distance.csv` |
| read-gate baselines (TARG) | `raw/baselines-2026-06-05/gate_read_baselines.csv` |
| cross-model signal | `raw/grids-xmodel-2026-06-05/xmodel_consistency_7models.csv` (+`.png`) · `gate_transfer_lomo.csv` |
| multi-depth (v1 negative) | `raw/deep-2026-06-05/multidepth_trivia.csv` |
| mix-train | `raw/mix-2026-06-05/mix_results.csv` |
| architecture figures | `summary/2026-06-05/architecture-v1.5.png` · `wrapper-connect.png` (+ `.tex`) |
| settings / recipes | `settings/settings.md` (P08-S6 probe · S7 grid · S8 gate) |
| glossary | `summary/2026-06-05/v1.5-glossary.md` |
| evidence ledger | `mem-embedding:summary/matrix.md` (§★★ main · §2 signal · §7/§7b/§7d gate · §7c/§7e ablation · §8 boundary) |

---

## 10. Status (2026-06-06) — scheduler = gpu-runq worker; next-week focus = draft this paper
**All runs now go through the `gpu-runq` worker** (`research-notes/tools/gpu-runq/`, daemon on sam-dev, 4 GPUs). Inspect: `runq.py --root /home/devuser/runq-root ls`. Submit/stop/prioritize via files — the agent does **not** poll GPUs.
| item | state |
|---|---|
| Main table (core columns) | ✅ done & persisted |
| Multi-layer ablation (18 configs) → §7e | ✅ DONE (folding pending: depth profile + placement×count×gate) |
| 5-seed significance CIs | ✅ DONE 38/38 (folding pending) |
| Cartridge-lite / Gist-lite columns (all 8 benches) | 🔄 worker (prefix+gist × 8) → fills main table |
| per-bench SFT matched control + off-diagonal forgetting panel | 🔄 worker (SFT × 5 QA, eval all 8) |
| **3-way adaptive gate** (mem ↔ full-ctx ↔ base) — the compression/fallback selling point | ⬜ needs `gate3_eval.py` → then queue ([`scope-and-compression.md`](scope-and-compression.md) §3) |
| 7-family read head-to-head | ⛔ blocked (ray/test pods deleted); 3-family done. (+Qwen3-14B = cheap follow-up) |
| relevance/agent eval · online gate | ⬜ T2/T3 future |

**Re-fold on completion:** `main_table_build.py --membase /…/paper_week` (fills Cartridge/Gist) · add SFT-grid forgetting panel · ablation depth-profile → matrix §7e · significance CIs. This doc's §3/§4 update with the new numbers.
