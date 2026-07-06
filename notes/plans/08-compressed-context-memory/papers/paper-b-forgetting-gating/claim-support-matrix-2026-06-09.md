# Claim -> experiment support matrix (main_v1.6.tex), 2026-06-09

> Every claim in `main_v1.6.tex`, big to small, mapped to the experiment that backs it and its
> status. DONE = data exists; PARTIAL = some cells exist, consolidate/extend; QUEUED = running now;
> GAP = needs a new run. All training is on PUBLIC train splits (categorical_niah removed).

## Headline contributions (abstract + intro (i)-(v))

| # | claim | experiment | status |
|---|---|---|---|
| (i) | detect-and-fallback read from intermediate state, distinct from input-time + best-of-N | E0 frontier (`bon_frontier` on `ec_*` bon logs) | PARTIAL (ec_ bon data exists; analyze) |
| (ii) | do-no-harm floor by construction; fine-tuning does not match it (forgetting) | Table forget (`fgt_*`/`mix_sft --signals`, 12 done) | DONE |
| (iii) | map WHEN compression breaks (cross-compressor x cross-model) | harm map (`cmb_*` wrapper 90 + gist/prefix `big_*`/`xdom_*`) | PARTIAL (wrapper QA done; gist/prefix cross-domain QUEUED) |
| (iv) | measure HOW EARLY breakage detectable (read-budget sweep) | E1 (`early_curve` on `ec_*` curve, 30 done) | PARTIAL (QA done; agentic via `bfcl_*_curve`) |
| (v) | per-input breakage hard to predict -> circuit breaker + floor | `signal_corr` + `gate_policy` | DONE (analysis) |

## Section-level claims

| section | claim (numbers in paper) | experiment | status |
|---|---|---|---|
| 3 floor | detach = base exactly; LoRA held-out -0.07/-0.08 mean, -0.6 worst, MC-format -0.18..-0.22; ours +0.000 | `fgt_*` LoRA forgetting (4 bases) | **DONE** (Table forget filled) |
| 5.1 harm | always-compress -9% (Gist); wrapper collapses Mistral; helps other bases | `gate_policy` + `cmb_*`/gist/prefix harm map | PARTIAL: have headline; **full cross-compressor x cross-model x bench map = consolidate `cmb_/gk_/gx_` + new `xdom_`** |
| 5.2 early (E1) | detection AUROC vs read budget; prefill approx full (~0.57 flat) | `early_curve` on `ec_*` | PARTIAL: QA `ec_` done (~0.57); **agentic `bfcl_*_curve` landing; consolidate figure** |
| 5.3 E0 vs BoN | breaker matches full at fraction cost, beats BoN at 1/N | `bon_frontier` on `ec_*` bon logs | PARTIAL: bon logs exist; **run analyzer + figure** |
| 5.4 efficiency | trivia 0.265@24%, quality 0.273@7%, musr 0.533@9%; extractive defer to full | `gate3_route` Track B | DONE (numbers); **cross-model amortised table = consolidate** |
| 5.5 MoE (Tab) | do-no-harm 4/5 bases; GLM concat -53% -> gate null +24%; source-select ~chance | `moe_route` (8 done) | **DONE** (table filled). E2a/E2b signal AUROC = **GAP** |
| 5.5 E2a/E2b | OLMoE router-signal AUROC; LoRA-branch activation AUROC | OLMoE `--moe --gate` + `mix_sft` branch-norm log | **GAP (queue)** |
| 5.6 scatter | compressor preserves cross-domain (generalises, no forget); LoRA/MoE below diagonal | `xf_` (compressor transfer) + `fgt_` (LoRA) + OLMoE | PARTIAL: QA `xf_` done; **new-dataset cross-domain `xdom_*` QUEUED**; assemble scatter |
| 5.6 generality | gate compressor-agnostic; Cartridge 33/35, Gist 26/35 do-no-harm | `gate3_route --general` | DONE (old); **re-confirm on clean split** |
| 5.7 signal | no signal beats cheap base-uncertainty; gate 15-51% headroom; use-rate tracks help (28/27, 44/32); P/R balanced-low (0.34/0.34, 0.46/0.49) | `signal_corr` + `gate_policy` | DONE (analysis); **AUROC-per-read-budget ranking = consolidate from `ec_` curve** |

## Main table = in-domain + cross-domain (the user's ask)

- **In-domain (diagonal)**: train compressor on D, eval held-out D. Cells: QA `big_*` (done/queued), BFCL/RCA/apibank/toolace in-domain (done/queued). gist+prefix x {q8,glm}.
- **Cross-domain (off-diagonal)**: train D1, eval D2. The old `xf_` grid is QA+synthetic (categorical-trained, stale). **NEW clean `xdom_{gist,prefix}_q8_{trivia,hotpot,quality,bfcl_multiple,apibank,toolace,rca_openrca}` queued (prio 8), each evals 9 benches** -> fills in+cross for the full new dataset set. GLM cross-domain = next.
- Columns reported per cell: no-ctx / compressed / full; metric = tool_acc (BFCL/apibank/toolace), mc_acc (RCA/quality/musr), F1/ROUGE (QA).

## Remaining GAPS to queue (besides analysis/consolidation)
1. **`xdom_` GLM** cross-domain grid (q8 queued; add glm).
2. **E2b OLMoE router-signal** + **E2a LoRA-branch activation** AUROC (5.5 placeholder).
3. **E1/E0 on agentic** (apibank/toolace/rca `--curve --bon`) to extend the read-budget + BoN beyond QA.
4. Clean re-run of the QA harm map (the `xdom_` QA rows cover this with clean split).
