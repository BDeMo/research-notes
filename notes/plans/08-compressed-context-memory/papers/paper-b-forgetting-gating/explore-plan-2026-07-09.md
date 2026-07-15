# Overnight exploration plan — 2026-07-09 (check tomorrow)

**Goal (design-phase, per directive):** gather *facts* — is the importance-routing direction worth it, where's the boundary — not force IMP. Central open question from F30: **IMP is dominated by free RAG on accuracy; its only edge is generality/efficiency/query-agnostic.** These waves test (1) how universal the *diagnosis* is across models, and (2) whether IMP has any design headroom.

All waves: base frozen, keep 0.5, **N=100** (disclosed exploration subset), patch-free benches (ruler_niah@16k, hotpot_qa, squad_v2, quality_hard, trivia_qa) so they run on every pod disk. Methods = window/rag/ll2/tome/**imp** (+kvzip/knorm for quadratic). Skip-if-`RECIPE_EVAL`. Launchers archived in [`configs/`](configs/) (`explore_gen.sh`, `explore_imp_abl.sh`).

## Waves launched (~00:45)
| wave | pod · GPUs | what | models / grid | cells | out dir | question |
|---|---|---|---|---|---|---|
| **A1 size-scaling** | free3 · 0-3 | generality | Qwen3-**1.7B / 4B / 14B** (quad) | ~105 | `grid_generality/g_q3_*` | do the facts (no-universal / full-hurts / IMP-retrieval-only) hold across size? (F11) |
| **A2 family+linear** | d1530 · 0-1 | generality | **Qwen2.5-7B** (quad), **Qwen3.5-4B** (linear) | ~60 | `grid_generality/g_q25_*, g_q35_4b_*` | cross-family + a 2nd linear point (F28) |
| **B exotic family** | free1 · 5,7 (shared) | generality | **GLM-4-9B, Ministral-8B, gpt-oss-20B** (quad), **Moonlight-16B-A3B** (linear) | ~130 | `grid_generality/g_glm4_*, g_ministral_*, g_gptoss_*, g_moonlight_*` | do facts hold across *vendors* & MoE? (⚠ load-risk: may need trust_remote_code) |
| **C IMP design-ceiling** | free2 · 0-1 | IMP ablation | Qwen3-8B, **signal{query,surprisal,both} × span{8,16,32,64,128}** × 4 benches | 60 | `grid_impabl/ia_*` | can a better signal/span combo close IMP's gap to RAG? (F30) |
| (finishing) main re-runs | d1525 · 0,1,3 | fill 3 blanks | imp_lb_multifieldqa, kvzip_quality, ll2_longbench_v2 | 3 | `grid_fulltable/ft_*` | complete the full main table |

**Totals:** ~360 exploration cells across ~10 GPUs on 4 pods.

## ETA (rough)
- free3 (1.7B/4B fast, 14B slower): ~4–6 h.
- d1530 (Qwen2.5-7B fast; Qwen3.5-4B linear torch-fallback slow): ~5–7 h.
- free1 exotic (loads are the risk; 8–20B): ~5–8 h if they load.
- free2 IMP ablation (60 cells, 2 GPUs): ~4–6 h.
- → all should finish within the ~10 h window.

## How to check tomorrow (per pod, own disk)
```
# per pod: RECIPE_EVAL count + fails
free3/d1530/free1:  cd /mnt/persist/grid_generality; grep -alE RECIPE_EVAL g_*.log | wc -l ; grep -alE CELL_FAIL g_*.log | wc -l
free2:              cd /mnt/persist/grid_impabl;   grep -alE RECIPE_EVAL ia_*.log | wc -l
```
Harvest scripts: `_harvest_maintable.py` (main), `_dive_imp.py` (gap analysis) — adapt dir. Masters: `master_free3.log`, `master_d1530.log`, `master_exotic.log`, `master_free2.log`.

## What each result will decide
- **A1+A2+B (generality):** if the *diagnosis* (no-universal, full-hurts, IMP=retrieval-only, RAG-dominates-QA) reproduces across size/family/vendor/MoE/linear → **Paper B's contribution = the diagnosis fact-base** is robust (option A in [`dive-in-imp-weakness-and-baselines.md`](dive-in-imp-weakness-and-baselines.md)).
- **C (IMP ceiling):** if no signal/span combo lifts IMP near RAG on QA → confirms F30 (importance-routing has ~0 accuracy headroom); if some combo does → a method lead worth pursuing.
- **NOT yet covered (needs code, next):** the decisive **query-agnostic multi-query IMP-vs-RAG** falsifier and the **cost/prefill column** — to be built after this batch (they need harness changes, too risky to leave unattended tonight).

*Provenance: launched 2026-07-09 ~00:45; masters + `g_*`/`ia_*`/`ft_*` logs on the respective pod disks.*
