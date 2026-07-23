# Paper A — all-benchmark manuscript tables (skeleton)

> Snapshot 2026-07-22 20:34 PT. Research only (no CMG-RCA / OpenRCA — those are company work).
> Rule: completed cells carry numbers; unfinished cells are `TBD` placeholders (fill when the run lands,
> never estimate). Scores are native metrics ×100. Raw and SFT are **references**, not compressed-path
> competitors. `GCM` names the full framework. `Compressor (w/o gate)` is the compressed path;
> `Compressor (w/ gate)` adds the held-out gate and bounded-raw fallback. SQuAD-v2 is a diagnostic column
> only (excluded from headline claims).

---

## Table 0 — Complete experiment map

| stage | experiment | configured evidence | current state |
|---|---|---:|---|
| E0 | data and model audit | lengths, overlap, scorer checks | complete |
| E1 | matched main comparison | 88 cells | complete |
| E1Q/E1R | output and SFT audit | 28 paths + 6 retrains | complete |
| E1F/E2 | gate features and analysis | 24 groups | complete |
| E3 | independent reproduction | 3 cells | complete |
| E4A | source adapters | 43 cells | 41 complete; 2 repairs |
| E4B | long-context transfer | 106 cells | 103 complete; 3 repairs |
| E5 | fixed-config generality | 42 cells | 12 complete; 3 running |
| E6 | budget and length | 23 cells | 2 complete; 1 running |
| E7 | mechanism ablations | 36 cells | TBD |
| E8 | measured cost | up to 16 profiles | TBD |
| E9 | official baselines | 52 local + native runs | TBD |

### Evidence/output audit

| check | scope | audited outcome |
|---|---|---|
| QuALITY length | Qwen3 tokenizer | median 6,511; 1.9% above 8,192 |
| train/eval overlap | BFCL | one duplicate removed |
| output reload | 28 paths | 28/28 complete |
| validation count | main grid | 88/88 complete |
| SFT output reaudit | QuALITY | 6/6 complete |
| silent fallback | LongLL variants | 10 cells excluded |

## Table 1 — Core (complete)

Benchmarks = columns, methods = rows.

### Qwen3-8B
| method | QuALITY | BFCL | HotpotQA | SQuAD-v2 (diag) |
|---|---:|---:|---:|---:|
| no context · ref | 17.8 | 1.3 | 19.8 | 16.1 |
| Raw (bounded) · ref | 7.2 | 92.4 | 53.7 | 65.5 |
| SFT · ref | 81.7±1.7 | 95.4±0.3 | 68.8±0.6 | 93.1±0.3 |
| Window · control | 15.7 | 55.7 | 26.2 | 49.6 |
| LLMLingua-2 · baseline | 14.3 | 70.3 | 22.1 | 53.4 |
| **Compressor (w/o gate) · ours** | **54.4±0.2** | **72.3±0.5** | 28.9±0.2 | 26.5±0.4 |
| **Compressor (w/ gate) · ours** | **54.6** | **88.5** | **50.9** | 62.8 |

### Qwen3.5-9B
| method | QuALITY | BFCL | HotpotQA | SQuAD-v2 (diag) |
|---|---:|---:|---:|---:|
| no context · ref | 22.0 | 1.3 | 26.7 | 20.7 |
| Raw (bounded) · ref | 7.1 | 84.5 | 53.9 | 66.8 |
| SFT · ref | 85.0±0.4 | 94.9±1.0 | 71.7±0.6 | 93.8±0.3 |
| Window · control | 16.7 | 52.8 | 24.8 | 49.6 |
| LLMLingua-2 · baseline | 20.3 | 60.8 | 28.9 | 58.8 |
| **Compressor (w/o gate) · ours** | **51.5±1.7** | **72.0±0.8** | **30.5±0.3** | 26.9±0.6 |
| **Compressor (w/ gate) · ours** | 51.4 | **80.5** | **51.7** | 64.7 |

### Table 1a — QuALITY stability and adaptation references

| path | evaluation | accuracy |
|---|---|---:|
| Compressor (w/o gate) | main grid | 54.4±0.2 |
| Compressor (w/o gate) | fixed configuration | 44.2±11.2 |
| Compressor (w/o gate) | independent repeat | 48.7±15.1 |
| SFT · ref | bounded input | 64.5±27.5 |
| SFT · ref | available input | 81.7±1.7 |

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

## Table 2 — Source-adapter readiness

| base | four source tasks | QuALITY K32 | full-cost SFT |
|---|---:|---:|---:|
| Qwen3-8B | 4/4 complete | complete | 4/4 complete |
| Qwen3.5-9B | 4/4 complete | repair | 4/4 complete |
| Qwen3.5-4B | 4/4 complete | complete | — |
| GLM-4-9B | 4/4 complete | repair | — |
| Ministral-8B | 4/4 complete | complete | — |
| xLAM-8B | 4/4 complete | complete | — |
| ToolACE-8B | 4/4 complete | complete | — |

## Table 3 — Real long-context transfer (source-trained adapter, zero-shot on target)

Filled from the E4B harvest for the two main bases; repeated-run intervals are pending.
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

## Table 4 — LongBench breakdown (source-trained adapter, zero-shot)

Per-task detail (source adapter in parentheses). Repeated-run intervals are pending. F1/ROUGE ×100.

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

> **Finding:** on **multi-hop** long-context transfer (HotpotQA/2WikiMQA/MuSiQue) the compressor without the gate **beats bounded
> raw** (raw is truncated), and routing adds more (e.g. q3_8b HotpotQA raw 19.1 → w/o gate 30.5 → w/ gate 36.0;
> q35_9b HotpotQA raw 3.9 → w/o gate 26.6 → w/ gate 31.0). On **single-doc/extractive** (MultiFieldQA/Qasper/
> NarrativeQA) raw wins and the gated variant **falls back to raw** (do-no-harm). Net: **Compressor (w/ gate) ≥ Raw on every cell.**

---

## Table 5 — Seven-base long-context transfer

Each numeric cell is `bounded raw / Compressor (w/o gate)`. Multi-hop averages LongBench HotpotQA,
2WikiMQA, and MuSiQue. BABILong averages QA1–QA3.

| base | LongBench-v2 | InfiniteBench | InfiniteBench K32 | BABILong | multi-hop |
|---|---:|---:|---:|---:|---:|
| Qwen3-8B | 31.2 / 20.1 | 52.8 / 21.4 | TBD | 18.1 / 0.9 | 13.8 / 23.7 |
| Qwen3.5-9B | 33.0 / 22.5 | 52.4 / 31.9 | TBD | 10.4 / 2.2 | 2.6 / 22.0 |
| Qwen3.5-4B | 34.0 / 19.7 | 51.1 / 26.2 | 24.0 | 14.5 / 2.0 | 10.7 / 19.7 |
| GLM-4-9B | 31.0 / 24.5 | 41.9 / 26.2 | TBD | 5.9 / 0.0 | 24.4 / 21.6 |
| Ministral-8B | 27.2 / 19.9 | 49.3 / 24.5 | 25.8 | 18.8 / 2.7 | 22.3 / 19.4 |
| xLAM-8B | 29.2 / 24.7 | 48.5 / 26.2 | 29.7 | 15.5 / 0.5 | 14.1 / 25.7 |
| ToolACE-8B | 27.8 / 19.1 | 48.9 / 25.3 | 23.1 | 13.9 / 0.7 | 14.0 / 25.5 |

## Table 6 — Controlled length boundary: RULER-NIAH

Exact-retrieval length sweep (fixed K128 recipe, no source adapter). The budget stage is 2/23 complete;
the RULER K128 cell is under technical repair. Numerical values remain `TBD` until harvesting.

| method | 4k | 8k | 16k | 32k |
|---|---:|---:|---:|---:|
| Raw (bounded) · ref | TBD | TBD | TBD | TBD |
| **Compressor (w/o gate; ours)** | TBD | TBD | TBD | TBD |

---

## Table 7 — Budget and capacity

| path | state tokens | QuALITY | RULER |
|---|---:|---:|---:|
| Compressor (ours) | 64 | TBD | TBD |
| Compressor (ours) | 128 | TBD | TBD |
| Compressor (ours) | 256 | TBD | TBD |
| Compressor (ours) | 512 | 53.0 | TBD |
| Raw window | 256 | TBD | TBD |
| Raw window | 512 | TBD | TBD |
| Raw window | 1,024 | TBD | TBD |
| Raw window | 2,048 | 9.7 | TBD |
| Raw window | 4,096 | TBD | TBD |
| Raw window | 8,192 | TBD | TBD |

## Table 8 — Fixed-configuration generality

| base | path | QuALITY | BFCL |
|---|---|---:|---:|
| Qwen3-8B | Raw · ref | 7.2 | 92.1 |
|  | Compressor (w/o gate) | 44.2±11.2 | 71.2±1.1 |
| Qwen3.5-9B | Raw · ref | 7.1 | 84.5 |
|  | Compressor (w/o gate) | 46.3±11.4 | 71.4±1.9 |
| Qwen3.5-4B | Raw · ref | TBD | TBD |
|  | Compressor (w/o gate) | TBD | TBD |
| GLM-4-9B | Raw · ref | TBD | TBD |
|  | Compressor (w/o gate) | TBD | TBD |
| Ministral-8B | Raw · ref | TBD | TBD |
|  | Compressor (w/o gate) | TBD | TBD |
| xLAM-8B | Raw · ref | TBD | TBD |
|  | Compressor (w/o gate) | TBD | TBD |
| ToolACE-8B | Raw · ref | TBD | TBD |
|  | Compressor (w/o gate) | TBD | TBD |

## Table 9 — Mechanism ablations

| variant | change | QuALITY | BFCL |
|---|---|---:|---:|
| main (ours) | K=128, all objectives | 54.4±0.2 | 72.3±0.5 |
| joint0 | detach answer loss | TBD | TBD |
| distill0 | remove teacher KL | TBD | TBD |
| recon0 | remove reconstruction | TBD | TBD |
| recur0 | independent chunks | TBD | TBD |
| K=64 | smaller memory | TBD | TBD |
| K=256 | larger memory | TBD | TBD |

## Table 10 — Measured cost

| base | task | source tokens | state tokens | encode ms | read ms | peak GiB |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-8B | QuALITY | TBD | TBD | TBD | TBD | TBD |
| Qwen3-8B | BFCL | TBD | TBD | TBD | TBD | TBD |
| Qwen3.5-9B | QuALITY | TBD | TBD | TBD | TBD | TBD |
| Qwen3.5-9B | BFCL | TBD | TBD | TBD | TBD | TBD |

## Table 11 — Official/native-base baselines

| method | released base | QuALITY | SQuAD | Hotpot | LongBench | InfiniteBench | RULER |
|---|---|---:|---:|---:|---:|---:|---:|
| LCLM | Qwen3 | — | — | — | TBD | — | TBD |
| Semi-Dynamic | Qwen3 | — | TBD | TBD | — | — | — |
| AutoCompressor | Llama-2/OPT | TBD | TBD | TBD | TBD | — | — |
| ICAE | Mistral-7B | TBD | TBD | TBD | — | — | — |
| CCM | Llama-2/Mistral | — | — | TBD | TBD | — | — |
| Activation Beacon | Qwen2-7B | — | — | — | TBD | TBD | TBD |
| xRAG | Mistral/Mixtral | — | TBD | TBD | TBD | — | — |
| Cartridges | Llama-3.2-3B | — | — | — | TBD | — | — |
| Gist Tokens | LLaMA/FLAN-T5 | — | TBD | — | — | — | — |
| 500xCompressor | Llama-3-8B | — | TBD | TBD | — | — | — |

---

## Fill map (where each TBD comes from)

| table | benchmark(s) | produced by | current state |
|---|---|---|---|
| 0, 1, 1a, 1b | audit, core, reproduction, gate | E0–E3 | complete |
| 2 | source adapters | E4A | 41/43 done; two K32 repairs |
| 3–5 | LongBench-v2, ∞Bench, BABILong, LongBench tasks | E4B | 103/106 done; three K32 repairs |
| 6–7 | RULER and budget sweep | E6 | 2/23 done; one technical repair |
| 8 | fixed-config generality | E5 | 12/42 done; three running |
| 9 | mechanism ablations | E7 | 0/36; TBD placeholders |
| 10 | measured cost | E8 | not started; TBD placeholders |
| 11 | official/native-base baselines | E9 | not started; TBD placeholders |

**Blocked / not in tables:** NoLiMa (data access), HELMET, LongMemEval (loaders not implemented).
