# Paper A — all-benchmark manuscript tables (skeleton)

> Snapshot 2026-07-22 20:34 PT. Research only (no CMG-RCA / OpenRCA — those are company work).
> Rule: completed cells carry numbers; unfinished cells are `TBD` placeholders (fill when the run lands,
> never estimate). Scores are native metrics ×100. Raw and SFT are **references**, not compressed-path
> competitors. `GCM` names the full framework. `Compressor (w/o gate)` is the compressed path;
> `Compressor (w/ gate)` adds the held-out gate and bounded-raw fallback. SQuAD-v2 is a diagnostic column
> only (excluded from headline claims).

---

## Table 1 — Core (complete)

Benchmarks = columns, methods = rows.

### Qwen3-8B
| method | QuALITY | BFCL | HotpotQA | SQuAD-v2 (diag) |
|---|---:|---:|---:|---:|
| Raw (bounded) · ref | 7.2 | 92.4 | 53.7 | 65.5 |
| SFT · ref | 81.7±1.7 | 95.4±0.3 | 68.8±0.6 | 93.1±0.3 |
| Window · control | 15.7 | 55.7 | 26.2 | 49.6 |
| LLMLingua-2 · baseline | 14.3 | 70.3 | 22.1 | 53.4 |
| **Compressor (w/o gate) · ours** | **54.4±0.2** | **72.3±0.5** | 28.9±0.2 | 26.5±0.4 |
| **Compressor (w/ gate) · ours** | **54.6** | **88.5** | **50.9** | 62.8 |

### Qwen3.5-9B
| method | QuALITY | BFCL | HotpotQA | SQuAD-v2 (diag) |
|---|---:|---:|---:|---:|
| Raw (bounded) · ref | 7.1 | 84.5 | 53.9 | 66.8 |
| SFT · ref | 85.0±0.4 | 94.9±1.0 | 71.7±0.6 | 93.8±0.3 |
| Window · control | 16.7 | 52.8 | 24.8 | 49.6 |
| LLMLingua-2 · baseline | 20.3 | 60.8 | 28.9 | 58.8 |
| **Compressor (w/o gate) · ours** | **51.5±1.7** | **72.0±0.8** | **30.5±0.3** | 26.9±0.6 |
| **Compressor (w/ gate) · ours** | 51.4 | **80.5** | **51.7** | 64.7 |

## Table 1b — Routing / reliability boundary (complete)

`Gain` = with-gate − without-gate; `Δ raw` = with-gate − bounded raw; `FB AUC` = fallback ranking AUROC; `FB rate` = fraction routed to raw. Formal fixed-threshold test certifies **0/24** groups (→ all-raw); numbers below are **held-out empirical routing**.

| base | metric | QuALITY | BFCL | HotpotQA | SQuAD-v2 |
|---|---|---:|---:|---:|---:|
| Qwen3-8B | Gain (pp) | +0.2 | +16.2 | +22.0 | +36.3 |
| Qwen3-8B | Δ raw (pp) | +47.4 | −3.5 | −2.4 | −2.0 |
| Qwen3-8B | FB AUC | 57.2 | 82.8 | 63.9 | 66.6 |
| Qwen3-8B | FB rate | 0.2% | 46.6% | 68.3% | 93.4% |
| Qwen3.5-9B | Gain (pp) | −0.1 | +8.5 | +21.2 | +37.8 |
| Qwen3.5-9B | Δ raw (pp) | +44.4 | −3.8 | −2.2 | −2.1 |
| Qwen3.5-9B | FB AUC | 54.9 | 84.1 | 67.4 | 62.4 |
| Qwen3.5-9B | FB rate | 0.3% | 22.0% | 52.1% | 94.2% |

---

## Table 2 — Real long-context transfer (source-trained adapter, zero-shot on target · seed 42)

Filled from the E4B harvest (44/44 cells for the two main bases at s42; single seed → error bars pending).
Window / LLMLingua-2 transfer rows were not run in E4B (n/a). BABILong = mean of QA1/QA2/QA3 @16k.

### Qwen3-8B
| method | LongBench-v2 | InfiniteBench-choice | RULER-NIAH (avg) | BABILong (QA1/2/3 avg) |
|---|---:|---:|---:|---:|
| no-context · ref | 33.4 | 41.9 | TBD | 2.0 |
| Raw (bounded) · ref | 31.2 | 52.8 | TBD | 18.1 |
| **Compressor (w/o gate) · ours** | 20.1 | 21.4 | TBD | 0.9 |
| **Compressor (w/ gate) · ours** | 31.4 | 52.8 | TBD | 18.1 |

### Qwen3.5-9B
| method | LongBench-v2 | InfiniteBench-choice | RULER-NIAH (avg) | BABILong (QA1/2/3 avg) |
|---|---:|---:|---:|---:|
| no-context · ref | 27.0 | 42.4 | TBD | 3.1 |
| Raw (bounded) · ref | 33.0 | 52.4 | TBD | 10.5 |
| **Compressor (w/o gate) · ours** | 22.5 | 31.9 | TBD | 2.2 |
| **Compressor (w/ gate) · ours** | 33.4 | 52.4 | TBD | 11.9 |

> RULER-NIAH still pending (Table 4). ∞Bench-choice K32 control repairs remain. BABILong: the compressor without the gate scores ~0 (recurrent
> synthetic; the compressed state cannot hold the planted facts) → the gated variant correctly falls back to raw.

---

## Table 3 — LongBench breakdown (source-trained adapter, zero-shot · seed 42)

Per-task detail (source adapter in parentheses). Single seed; error bars pending. F1/ROUGE ×100.

### Qwen3-8B
| method | MultiFieldQA (SQuAD) | Qasper (SQuAD) | HotpotQA (Hotpot) | 2WikiMQA (Hotpot) | MuSiQue (Hotpot) | NarrativeQA (Narr.) |
|---|---:|---:|---:|---:|---:|---:|
| no-context · ref | 21.1 | 9.6 | 8.0 | 10.0 | 4.7 | 5.3 |
| Raw (bounded) · ref | 43.3 | 27.1 | 19.1 | 12.0 | 10.4 | 13.5 |
| **Compressor (w/o gate) · ours** | 18.5 | 12.4 | **30.5** | **29.3** | **11.4** | 11.6 |
| **Compressor (w/ gate) · ours** | 43.3 | 27.1 | **36.0** | **29.5** | 10.5 | 13.5 |

### Qwen3.5-9B
| method | MultiFieldQA | Qasper | HotpotQA | 2WikiMQA | MuSiQue | NarrativeQA |
|---|---:|---:|---:|---:|---:|---:|
| no-context · ref | 21.8 | 10.0 | 7.7 | 11.7 | 3.9 | 9.6 |
| Raw (bounded) · ref | 41.5 | 17.0 | 3.9 | 2.9 | 0.9 | 16.6 |
| **Compressor (w/o gate) · ours** | 16.7 | 14.3 | **26.6** | **27.6** | **11.9** | 13.6 |
| **Compressor (w/ gate) · ours** | 41.5 | 17.0 | **31.0** | **28.0** | **13.5** | 16.6 |

> **Finding (real, s42):** on **multi-hop** long-context transfer (HotpotQA/2WikiMQA/MuSiQue) the compressor without the gate **beats bounded
> raw** (raw is truncated), and routing adds more (e.g. q3_8b HotpotQA raw 19.1 → w/o gate 30.5 → w/ gate 36.0;
> q35_9b HotpotQA raw 3.9 → w/o gate 26.6 → w/ gate 31.0). On **single-doc/extractive** (MultiFieldQA/Qasper/
> NarrativeQA) raw wins and the gated variant **falls back to raw** (do-no-harm). Net: **Compressor (w/ gate) ≥ Raw on every cell.**

---

## Table 4 — Controlled length boundary: RULER-NIAH (RUNNING)

Exact-retrieval length sweep (fixed K128 recipe, no source adapter). The budget stage is 2/23 complete;
the RULER K128 cell is currently running. Numerical values remain `TBD` until harvesting.

| base / method | 4k | 8k | 16k | 32k |
|---|---:|---:|---:|---:|
| Qwen3-8B · Raw (bounded) | TBD | TBD | TBD | TBD |
| Qwen3-8B · **Compressor (w/o gate; ours)** | TBD | TBD | TBD | TBD |
| Qwen3.5-9B · Raw (bounded) | TBD | TBD | TBD | TBD |
| Qwen3.5-9B · **Compressor (w/o gate; ours)** | TBD | TBD | TBD | TBD |

---

## Table 5 — Budget / capacity ablations (RUNNING / QUEUED)

GCM memory budget K sweep and mechanism ablation (Qwen3-8B × {QuALITY, BFCL}, 3 seeds). Budget/length is
2/23 complete with one running; mechanism ablation is 0/36 and queued.

| axis | variant | QuALITY | BFCL |
|---|---|---:|---:|
| K budget | K=64 | TBD | TBD |
| K budget | K=128 (main) | TBD | TBD |
| K budget | K=256 | TBD | TBD |
| K budget | K=512 | TBD | TBD |
| mechanism | joint0 (detach answer loss) | TBD | TBD |
| mechanism | distill0 (no teacher KL) | TBD | TBD |
| mechanism | recon0 (no slot recon) | TBD | TBD |
| mechanism | recur0 (independent chunks) | TBD | TBD |

---

## Fill map (where each TBD comes from)

| table | benchmark(s) | produced by | current state |
|---|---|---|---|
| 1, 1b | QuALITY, BFCL, HotpotQA, SQuAD-v2 | Core main grid (E1) + gate analysis (E2) | ✅ complete (88/88) |
| 2, 3 | LongBench-v2, ∞Bench-choice, BABILong, LongBench tasks | transfer adapters (E4A) → long-context eval (E4B) | 103/106 done; three K32 evaluations need technical repair |
| 4 | RULER-NIAH 4k/8k/16k/32k | budget/length (E6) | 2/23 budget cells done; one RULER cell running |
| 5 | K sweep + mechanism ablation | budget (E6) + ablation (E7) | budget 2/23 done, one running; ablation 0/36 queued |

**Blocked / not in tables:** NoLiMa (data access), HELMET, LongMemEval (loaders not implemented).
