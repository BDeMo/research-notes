# Matrix 2 — EXPERIMENTS (done / running / to-iterate)

> Status: ✅ done · 🏃 running · ⏳ todo · 🔴 blocked. "cells" = eval cells (logs in `results-v2.0.0/logs/`). Last synced 2026-07-06.

## A. Done (in-repo logs)
| id | experiment | cells | log |
|---|---|---|---|
| X-cat | Catalog master budget table (20 KV methods × ratio + method×bench + refs) | 177 | `catalog-177.txt` |
| X-nec | Necessity length-sweep (trunc/window/full/LLMLingua × 2k–32k) | 72 | `necessity-72.txt` |
| X-rag | BM25-RAG × length/budget/bench | 18 | `rag-18.txt` |
| X-diag | 10-story diagnosis campaign (`s1…s10`) | 329 | `diagnosis-329.txt` |
| X-dive | Dive-ins: extreme-ratio, cliff-length, distractor-causal, RAG-hurts (`da/db/dc/dd`) | 100 | `dive-100.txt` |
| X-prec | Precision wave: pin 16k cliff, kvzip cliff, distractor×KV (`pw`) | 30 | `precision-30.txt` |
| X-scale | Scaling/cross-family: Qwen3 1.7B→14B, Qwen2.5, Llama, GDN-4B (`x1–x4`) | 86 | `expansion-86.txt` |
| X-t1 | Method A/B (pruned vs kitchen-sink × 2 bases) + transfer **rows** (train-on-A → mixed eval) | 13/14 | `t1_*` (1 q35 cell crashed) |

## B. Running
| id | experiment | status | where |
|---|---|---|---|
| X-fl | **Faithful LongLLMLingua + orig-LLMLingua** (2 methods × 3 rates × 5 benches), authors' Llama-2-7b compressor | 🏃 ~16/30 | d1525 |
| X-tm | **ToMe token-merge ablation** (ratio × {embed vs random} × benches) | 🏃 ~5/34 | d1530 |

## C. To-iterate (NOT done) — the backlog
| id | experiment | for | priority | note |
|---|---|---|---|---|
| X-C1 | **Full 5×5 domain-transfer matrix** (per-bench, not aggregate rows) × 2 bases | Paper B (C2) | high | T1 gave only mixed-eval rows; need per-target breakdown |
| X-C2 | **MINE information experiments**: I(token;answer), I(compressed-M;full-ctx), I(token;query) | Paper B dive-in | high | feeds importance-guided merging; ref MINE (Belghazi 2018) |
| X-C3 | **Conformal / held-out gate validation** (coverage + risk; gated ≥ feasible OUT-of-sample) | Paper A + B (C3) | high | designed only; **F13 depends on this** |
| X-C4 | **Importance-guided merging** (protect high-importance tokens via MINE/retrieval-head/key-norm/reconstructability, merge redundant) | Paper B method | high | the actual Paper-B method beyond ToMe |
| X-C5 | **SSM merge + brief re-train** (R-MeeTo style on GDN base) to rebuild key-knowledge | Paper B (linear arm) | med | needs short LoRA retrain on Qwen3.5-GDN |
| X-C6 | **Gate generality across compressors** (one gate over KV-evict / prompt / merge / RAG) | Paper A headline | high | intrinsic-signal vs trained-gate; robustness metrics |
| X-C7 | **Intrinsic probes (§K, ~20)**: retrieval-head, attention-entropy, CKA(M,full), OOD score, logit-lens, ECE | Paper B / A | med | down-select 4–6 that predict transfer/failure |
| X-C8 | ✅ **DONE (signal probe, `run_probe.py`)**: query_dot AUROC 0.95 (word-needle), surprisal 0.84 (numeric), norm-signals useless, length-invariant → **F20**. `IMP-v2.1.0` router inputs decided (method version registry: decisions D28) | Paper B insight | ✅ | remaining 16-metric harvest optional; core signals identified |
| X-C9 | **Selective-Context (exact)** | faithful baseline | low | 🔴 install-blocked (spacy build fails); needs fix or skip |
| X-C10 | **gist / cartridge EXACT reproduction** (authors' code) OR drop as public baselines | faithful baselines | med | current lite versions are NOT faithful (D21) |
| X-C11 | **multi_needle_niah scoring fix** | bench hygiene | low | full=0.0 bug; fix then re-enable |
| X-C12 | **Long-doc/global benches** (∞Bench, NoCha, LongBench-v2 subset) | Paper B eval | med | current suite lacks a leakage-safe global-reasoning bench |
| X-A-kvzip | **Gate-over-KVzip validation** (per-item {kvzip_ok, full_ok, reconstruction-fidelity r(x)}; conformal gate → gated≥full + coverage + AUROC) + importance-weighted preserve | Paper A (kvzip combine) | high | `run_gate_probe.py` (to build); design in `PAPER-A-v1.8.1-complete.md §7` |
| X-C13 | **Cross-model gate** (does the gate transfer across bases/sizes?) | Paper A robustness | med | reuse the scaling/cross-family checkpoints |

## D. Blocked / phantom
- 12 "missing" campaign cells were **phantom** (dedup/never-created, redundant) — not re-run.
- Selective-Context (X-C9) blocked on `spacy` build.
- 1 T1 cell (`t1_mx_narr_q35`) crashed — non-critical (row covered by others).

*Provenance: `results-v2.0.0/logs/`, `decisions-2026-06-24.md` D12–D21, `v2.0.0-plan.md`, `v2.1.0-direction-and-method-plan.md`.*
