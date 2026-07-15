# Compressed-context memory — two-paper workspace (index)

> **CURRENT STATE (2026-07-08).**
> - **Paper B — ACTIVE (the paper we write).** *Observe long-context failures, then bolt a lightweight
>   importance-routing structure (**IMP-v2.1.0**, span-level, training-free) onto a frozen base.* FULL-test main
>   table + cross-architecture generality (Qwen3.5 GDN + others) in progress.
> - **Paper A — BLOCKED/ARCHIVED.** The learned soft-memory compressor cannot compress extractive QA (F26) and the
>   reconstruction gate is a non-effect (F22); kept for the confidence-gate result but not the current focus.
>
> **Method version registry:** `IMP-v2.1.0` (span-level, Mode A) is behind ALL current results — see
> [`decisions-2026-06-24.md`](decisions-2026-06-24.md) **D28**. Do not mix with token-level `IMP-v2.0` (superseded).

**Start here:** [`OVERVIEW-both-papers-and-facts.md`](OVERVIEW-both-papers-and-facts.md) — both papers, the fact base, and the chronology in one place.

---

## 📄 Paper B (IMP) — CURRENT
| doc | what |
|---|---|
| [`PAPER-B-v2.1.0-complete.md`](PAPER-B-v2.1.0-complete.md) | **the spec** — thesis, method, experiment logic, related work, novelty |
| [`imp-method-and-implementation.md`](imp-method-and-implementation.md) | **method detail** — signals, span selection, modes, cost caveat, exact code path |
| [`hyperparameters.md`](hyperparameters.md) | **all major hyperparameters explained against the framework** (score→select→read) + keep→knob mapping |
| [`keep-ablation-results.md`](keep-ablation-results.md) | **keep-rate ablation, all 7 methods** × 4 benches × 4 keeps (F29) — the curves invert on literary-MC |
| [`imp-does-it-work-analysis.md`](imp-does-it-work-analysis.md) | **honest "does IMP work?" assessment** — wins on retrieval + linear; mid-pack on QA (evidence-backed) |
| [`dive-in-imp-weakness-and-baselines.md`](dive-in-imp-weakness-and-baselines.md) | **dive-in (F30): IMP is dominated by free RAG on accuracy** — where/why, baseline characterization, is-the-direction-worth-it |
| [`explore-plan-2026-07-09.md`](explore-plan-2026-07-09.md) | **overnight exploration plan** — cross-size/family/vendor generality + IMP design-ceiling ablation (~360 cells, 4 pods) |
| [`v2.1.0-paperB-method-designs.md`](v2.1.0-paperB-method-designs.md) | method design space (signals → pluggable compressor) |
| [`PAPER-B-draft.md`](PAPER-B-draft.md) | prose working draft (v0.1) |

## 📊 Paper B — results & reproducibility
| doc | what |
|---|---|
| [`main-table-fulltest.md`](main-table-fulltest.md) | **headline main table — FULL test sets** (Qwen3-8B, 9 methods × 16 benches) |
| [`generality-model-matrix.md`](generality-model-matrix.md) | cross-architecture / family / size table (Qwen3.5 GDN + others) |
| [`experiment-config-and-sampling.md`](experiment-config-and-sampling.md) | **settings, train/test config, per-bench N & sampling disclosure** |
| [`configs/`](configs/) | **exact code + launch scripts** (IMP-v2.1.0 code, grid launchers, patches, env) |
| [`figures/`](figures/) | figures (Fig 8–13) + `make_julw01_figs.py` |

## 🧱 Fact base & tracking matrices
| doc | what |
|---|---|
| [`facts-and-insights-summary.md`](facts-and-insights-summary.md) | **consolidated summary** — 27 facts → 7 clusters → 6 meta-insights |
| [`matrix-facts.md`](matrix-facts.md) | **confirmed facts F1–F27** (status · evidence · scope) |
| [`baseline-factbase-v2.0.0.md`](baseline-factbase-v2.0.0.md) | self-contained empirical reference (baselines §1–§12) |
| [`matrix-experiments.md`](matrix-experiments.md) | experiment status / backlog |
| [`matrix-paper-design.md`](matrix-paper-design.md) | claims ↔ methods ↔ evidence ↔ gaps, both papers |

## 🔬 Analysis · diagnosis · negative results
| **[`results-summary-and-insights-2026-07-11.md`](results-summary-and-insights-2026-07-11.md) — START HERE: grounded results + reliability review + multi-level insights + validation plan** · [`intrinsic-probe-results.md`](intrinsic-probe-results.md) · [`insights-longcontext-validity.md`](insights-longcontext-validity.md) · [`baseline-diagnosis-report.md`](baseline-diagnosis-report.md) · [`baseline-diagnosis-campaign.md`](baseline-diagnosis-campaign.md) · [`negative-results.md`](negative-results.md) |
|---|

## ✅ Baselines & faithfulness
| [`baseline-catalog-faithfulness.md`](baseline-catalog-faithfulness.md) (EXACT/GENERIC/ADAPTED per method) · [`baselines-and-novelty.md`](baselines-and-novelty.md) · [`faithfulness-and-traceability-audit-2026-07-06.md`](faithfulness-and-traceability-audit-2026-07-06.md) |
|---|

## 🅰️ Paper A (archived / blocked)
| [`PAPER-A-v1.8.1-complete.md`](PAPER-A-v1.8.1-complete.md) · [`method-elegance-plan-v1.8.x.md`](method-elegance-plan-v1.8.x.md) · [`paper-A-archive.md`](paper-A-archive.md) |
|---|

## 🧭 Decisions & background
| [`decisions-2026-06-24.md`](decisions-2026-06-24.md) **(current, D12–D28)** · [`decisions-2026-06-08.md`](decisions-2026-06-08.md) · [`linear-attention-and-kvcache-background.md`](linear-attention-and-kvcache-background.md) · [`related-work.md`](related-work.md) · [`references-longcontext.md`](references-longcontext.md) · [`references.md`](references.md) · [`glossary.md`](glossary.md) |
|---|

## 🗂️ Planning & direction (historical context)
[`v2.1.0-direction-and-method-plan.md`](v2.1.0-direction-and-method-plan.md) · [`v2.0.0-method-plan.md`](v2.0.0-method-plan.md) · [`v2.0.0-plan.md`](v2.0.0-plan.md) · [`ideas-brainstorm-v2.0.0-2026-06-24.md`](ideas-brainstorm-v2.0.0-2026-06-24.md) · [`framing.md`](framing.md) · [`logic.md`](logic.md) · [`outline.md`](outline.md) · [`method.md`](method.md)

## 📚 Reviews (historical, dated)
[`critical-review-v2-priority-2026-06-09.md`](critical-review-v2-priority-2026-06-09.md) · [`critical-review-v1.7-2026-06-10.md`](critical-review-v1.7-2026-06-10.md) · [`critical-review-2026-06-08.md`](critical-review-2026-06-08.md) · [`claim-support-matrix-2026-06-09.md`](claim-support-matrix-2026-06-09.md) · [`litreview-claimcheck-2026-06-08.md`](litreview-claimcheck-2026-06-08.md) · [`experiment-matrix-2026-06-08.md`](experiment-matrix-2026-06-08.md) · [`experiments-index-2026-06-09.md`](experiments-index-2026-06-09.md) · [`exp-gate-generality.md`](exp-gate-generality.md)

## 🗄️ v1.7-era archive — **do NOT cite these numbers** (train/eval leakage era)
`summary-matrix.md` · `summary-matrix-v1.7.md` · `summary-matrix-v1.7.3.md` · `results-v1.7/` · `results-v1.7.3/` · `results-v1.7.5/` · `results-v1.7.6/` · `results-v1.8.0/` · `results-v2.0.0/`

---
*Note: files are grouped here by role rather than moved into subfolders, to keep the dense cross-links between them intact. The one physical grouping is [`configs/`](configs/) (code+launchers) and [`figures/`](figures/).*
