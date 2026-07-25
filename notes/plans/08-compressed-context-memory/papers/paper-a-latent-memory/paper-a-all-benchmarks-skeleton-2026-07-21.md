# Paper A — all-benchmark operations tables

> Snapshot 2026-07-25 01:32 PT. Research only (no CMG-RCA / OpenRCA — those are company work).
> Rule: completed cells carry numbers; unfinished cells are `TBD` placeholders (fill when the run lands,
> never estimate). Scores are native metrics ×100. Raw and SFT are **references**, not compressed-path
> competitors. `GCM` names the full framework. `Compressor (w/o gate)` is the compressed path;
> `Compressor (w/ gate)` adds the held-out gate and bounded-raw fallback. SQuAD-v2 is a diagnostic column
> only (excluded from headline claims).
>
> **Storage boundary:** this Markdown is the maintenance and experiment source of truth. Run status,
> invalidation history, rerun queues, stage IDs, and `TBD` cells stay here and must not be copied into the
> Overleaf repository. The paper receives only validated, claim-bearing results and submission text.

> [!WARNING]
> **Integrity quarantine (2026-07-22).** `⚠ INVALID` marks results affected by the QuALITY
> label-index bug: the upstream labels are zero-based, but the loader subtracted one, dropping
> all A-labelled examples and shifting the remaining targets. This invalidates every direct
> QuALITY score, every adapter trained on QuALITY, and downstream LongBench-v2 /
> InfiniteBench-choice results that read those adapters. `⚠ COLLAPSE` marks a separate observed
> failure on Hotpot→BABILong transfer; its data labels are not affected. The reported confidence
> gate on unaffected tasks remains valid; only the unreported entropy-signal baseline needs
> recomputation after correcting its sign.

---

## Table 0 — Complete experiment map

| stage | experiment | configured evidence | current state |
|---|---|---:|---|
| E0 | data and model audit | lengths, overlap, scorer checks | complete |
| E1 | matched main comparison | 88 original + 24 corrected QuALITY cells | 60 unaffected; 12 corrected done; 3 running; 9 queued |
| E1B | expanded matched-base baselines | 18 cells | configured: BM25, LongLLMLingua, mean pooling |
| E1Q/E1R | output and SFT audit | 28 paths + optional SFT retrains | output audit complete; SFT reruns do not block the paper |
| E1F/E2 | gate features and analysis | 24 groups | 18 valid; 6 QuALITY groups invalid |
| E3 | independent reproduction | optional diagnostic | removed from the paper-critical path |
| E4A | source adapters | 28 paper-critical cells | 7 QuALITY-source reruns remain |
| E4B | long-context transfer | 77 paper-critical cells | 14 QuALITY-source target reruns remain |
| E5 | fixed-config BFCL generality | 21 cells | 17 valid; four third-seed technical failures |
| E6 | state-token sensitivity | included in E7 | complete |
| E7 | mechanism ablations | 36 cells | 36/36 corrected and complete |
| E8 | measured cost | 4 paper-table profiles | TBD |
| E9 | official baselines | 8 compatible native packages | TBD |

### Evidence/output audit

| check | scope | audited outcome |
|---|---|---|
| QuALITY length | Qwen3 tokenizer | median 6,511; 1.9% above 8,192 |
| train/eval overlap | BFCL | one duplicate removed |
| output reload | 28 paths | 28/28 complete |
| validation count | main grid | QuALITY used 1,595 corrupted rows; corrected split has 2,086 |
| SFT output reaudit | QuALITY | 6/6 artifacts invalid |
| silent fallback | LongLL variants | 10 cells excluded |

## Table 1 — Core main comparison

Benchmarks = columns, methods = rows.

### Qwen3-8B
| method | QuALITY | BFCL | HotpotQA | SQuAD-v2 (diag) |
|---|---:|---:|---:|---:|
| no context · ref | 43.2 corrected | 1.3 | 19.8 | 16.1 |
| Raw (bounded) · ref | 73.2 corrected | 92.4 | 53.7 | 65.5 |
| SFT · ref | 79.1±1.1 corrected | 95.4±0.3 | 68.8±0.6 | 93.1±0.3 |
| Window · control | 48.5 corrected | 55.7 | 26.2 | 49.6 |
| BM25 retrieval · baseline | TBD | TBD | TBD | — |
| LLMLingua-2 · baseline | 45.4 corrected | 70.3 | 22.1 | 53.4 |
| LongLLMLingua · baseline | TBD | TBD | TBD | — |
| mean pooling · control | TBD | TBD | TBD | — |
| **Compressor (w/o gate) · ours** | 48.1±1.5 corrected | **72.3±0.5** | 28.9±0.2 | 26.5±0.4 |
| **Compressor (w/ gate) · ours** | TBD corrected | **88.5** | **50.9** | 62.8 |

### Qwen3.5-9B
| method | QuALITY | BFCL | HotpotQA | SQuAD-v2 (diag) |
|---|---:|---:|---:|---:|
| no context · ref | ~~22.0~~ ⚠ INVALID | 1.3 | 26.7 | 20.7 |
| Raw (bounded) · ref | ~~7.1~~ ⚠ INVALID | 84.5 | 53.9 | 66.8 |
| SFT · ref | ~~85.0±0.4~~ ⚠ INVALID | 94.9±1.0 | 71.7±0.6 | 93.8±0.3 |
| Window · control | ~~16.7~~ ⚠ INVALID | 52.8 | 24.8 | 49.6 |
| BM25 retrieval · baseline | TBD | TBD | TBD | — |
| LLMLingua-2 · baseline | ~~20.3~~ ⚠ INVALID | 60.8 | 28.9 | 58.8 |
| LongLLMLingua · baseline | TBD | TBD | TBD | — |
| mean pooling · control | TBD | TBD | TBD | — |
| **Compressor (w/o gate) · ours** | ~~51.5±1.7~~ ⚠ INVALID | **72.0±0.8** | **30.5±0.3** | 26.9±0.6 |
| **Compressor (w/ gate) · ours** | ~~51.4~~ ⚠ INVALID | **80.5** | **51.7** | 64.7 |

### Table 1a — QuALITY stability and adaptation references

| path | evaluation | accuracy |
|---|---|---:|
| Compressor (w/o gate) | main grid | ~~54.4±0.2~~ ⚠ INVALID |
| Compressor (w/o gate) | fixed configuration | ~~44.2±11.2~~ ⚠ INVALID |
| Compressor (w/o gate) | independent repeat | ~~48.7±15.1~~ ⚠ INVALID |
| SFT · ref | bounded input | ~~64.5±27.5~~ ⚠ INVALID |
| SFT · ref | available input | ~~81.7±1.7~~ ⚠ INVALID |

## Table 1b — Routing / reliability boundary (complete)

`Gain` = with-gate − without-gate; `Δ raw` = with-gate − bounded raw; `FB AUC` = fallback ranking AUROC; `FB rate` = fraction routed to raw. Formal fixed-threshold test certifies **0/24** groups (→ all-raw); numbers below are **held-out empirical routing**.

| base | metric | QuALITY | BFCL | HotpotQA | SQuAD-v2 |
|---|---|---:|---:|---:|---:|
| Qwen3-8B | Gain (pp) | ~~+0.2~~ ⚠ INVALID | +16.2 | +22.0 | +36.3 |
| Qwen3-8B | Δ raw (pp) | ~~+47.4~~ ⚠ INVALID | −3.5 | −2.4 | −2.0 |
| Qwen3-8B | FB AUC | ~~57.2~~ ⚠ INVALID | 82.8 | 63.9 | 66.6 |
| Qwen3-8B | FB rate | ~~0.2%~~ ⚠ INVALID | 46.6% | 68.3% | 93.4% |
| Qwen3.5-9B | Gain (pp) | ~~-0.1~~ ⚠ INVALID | +8.5 | +21.2 | +37.8 |
| Qwen3.5-9B | Δ raw (pp) | ~~+44.4~~ ⚠ INVALID | −3.8 | −2.2 | −2.1 |
| Qwen3.5-9B | FB AUC | ~~54.9~~ ⚠ INVALID | 84.1 | 67.4 | 62.4 |
| Qwen3.5-9B | FB rate | ~~0.3%~~ ⚠ INVALID | 22.0% | 52.1% | 94.2% |

---

## Table 2 — Source-adapter readiness

The paper-critical grid keeps one compressor source adapter per
base/source pair. K32 and transfer-SFT diagnostics are optional and no
longer block the paper. Seven corrected QuALITY source adapters remain.

## Table 3 — Real long-context transfer (source-trained adapter, zero-shot on target)

Filled from the E4B harvest for the two main bases; repeated-run intervals are pending.
Window / LLMLingua-2 transfer rows were not run in E4B (n/a). BABILong = mean of QA1/QA2/QA3 @16k.

### Qwen3-8B
| method | LongBench-v2 | InfiniteBench-choice | BABILong (QA1/2/3 avg) |
|---|---:|---:|---:|
| no-context · ref | 33.4 | 41.9 | 2.0 |
| Raw (bounded) · ref | 31.2 | 52.8 | 18.1 |
| **Compressor (w/o gate) · ours** | ~~20.1~~ ⚠ INVALID | ~~21.4~~ ⚠ INVALID | ~~0.9~~ ⚠ COLLAPSE |
| **Compressor (w/ gate) · ours** | ~~31.4~~ ⚠ INVALID | ~~52.8~~ ⚠ INVALID | 18.1 |

### Qwen3.5-9B
| method | LongBench-v2 | InfiniteBench-choice | BABILong (QA1/2/3 avg) |
|---|---:|---:|---:|
| no-context · ref | 27.0 | 42.4 | 3.1 |
| Raw (bounded) · ref | 33.0 | 52.4 | 10.5 |
| **Compressor (w/o gate) · ours** | ~~22.5~~ ⚠ INVALID | ~~31.9~~ ⚠ INVALID | ~~2.2~~ ⚠ COLLAPSE |
| **Compressor (w/ gate) · ours** | ~~33.4~~ ⚠ INVALID | ~~52.4~~ ⚠ INVALID | 11.9 |

> LongBench-v2 and InfiniteBench compressor values are quarantined because
> they use QuALITY-trained adapters. BABILong is a separate performance collapse: the source is
> HotpotQA and is not affected by the QuALITY label bug.

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

| base | LongBench-v2 | InfiniteBench | BABILong | multi-hop |
|---|---:|---:|---:|---:|
| Qwen3-8B | 31.2 / ~~20.1~~ ⚠ | 52.8 / ~~21.4~~ ⚠ | 18.1 / ~~0.9~~ collapse | 13.8 / 23.7 |
| Qwen3.5-9B | 33.0 / ~~22.5~~ ⚠ | 52.4 / ~~31.9~~ ⚠ | 10.4 / ~~2.2~~ collapse | 2.6 / 22.0 |
| Qwen3.5-4B | 34.0 / ~~19.7~~ ⚠ | 51.1 / ~~26.2~~ ⚠ | 14.5 / ~~2.0~~ collapse | 10.7 / 19.7 |
| GLM-4-9B | 31.0 / ~~24.5~~ ⚠ | 41.9 / ~~26.2~~ ⚠ | 5.9 / ~~0.0~~ collapse | 24.4 / 21.6 |
| Ministral-8B | 27.2 / ~~19.9~~ ⚠ | 49.3 / ~~24.5~~ ⚠ | 18.8 / ~~2.7~~ collapse | 22.3 / 19.4 |
| xLAM-8B | 29.2 / ~~24.7~~ ⚠ | 48.5 / ~~26.2~~ ⚠ | 15.5 / ~~0.5~~ collapse | 14.1 / 25.7 |
| ToolACE-8B | 27.8 / ~~19.1~~ ⚠ | 48.9 / ~~25.3~~ ⚠ | 13.9 / ~~0.7~~ collapse | 14.0 / 25.5 |

## Tables 6–7 — Capacity sensitivity

The separate RULER and budget tables are removed from the paper-critical
set. Capacity sensitivity is measured by the corrected K=64 and K=256
ablation rows in Table 9.

## Table 8 — Fixed-configuration BFCL generality

| base | path | BFCL |
|---|---|---:|
| Qwen3-8B | Raw · ref | 92.1 |
|  | Compressor (w/o gate) | 71.2±1.1 |
| Qwen3.5-9B | Raw · ref | 84.5 |
|  | Compressor (w/o gate) | 71.4±1.9 |
| Qwen3.5-4B | Raw · ref | 11.1 |
|  | Compressor (w/o gate) | 71.5±1.7 |
| GLM-4-9B | Raw · ref | 84.8 |
|  | Compressor (w/o gate) | 72.9±0.7 (2 runs) |
| Ministral-8B | Raw · ref | 87.0 |
|  | Compressor (w/o gate) | 72.0±0.7 (2 runs) |
| xLAM-8B | Raw · ref | 88.6 |
|  | Compressor (w/o gate) | 73.1±0.9 (2 runs) |
| ToolACE-8B | Raw · ref | 88.9 |
|  | Compressor (w/o gate) | 72.2±0.4 (2 runs) |

## Table 9 — Mechanism ablations

| variant | change | QuALITY | BFCL |
|---|---|---:|---:|
| main (ours) | K=128, all objectives | 48.1±1.5 | 72.3±0.5 |
| joint0 | detach answer loss | 48.0±1.5 | 71.4±2.9 |
| distill0 | remove teacher KL | 39.7±14.0 | 71.8±1.8 |
| recon0 | remove reconstruction | 39.2±13.2 | 72.2±2.0 |
| recur0 | independent chunks | 47.2±0.8 | 72.2±1.6 |
| K=64 | smaller memory | 46.0±3.0 | 69.6±1.9 |
| K=256 | larger memory | 46.9±4.5 | 70.9±1.5 |

## Table 10 — Measured cost

| base | task | source tokens | state tokens | encode ms | read ms | peak GiB |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-8B | QuALITY | TBD | TBD | TBD | TBD | TBD |
| Qwen3-8B | BFCL | TBD | TBD | TBD | TBD | TBD |
| Qwen3.5-9B | QuALITY | TBD | TBD | TBD | TBD | TBD |
| Qwen3.5-9B | BFCL | TBD | TBD | TBD | TBD | TBD |

## Table 11 — Official/native-base baselines

| method | released base | QuALITY | SQuAD | Hotpot | LongBench | InfiniteBench |
|---|---|---:|---:|---:|---:|---:|
| LCLM | Qwen3 | — | — | — | TBD | — |
| Semi-Dynamic | Qwen3 | — | TBD | TBD | — | — |
| AutoCompressor | Llama-2/OPT | TBD | TBD | TBD | TBD | — |
| ICAE | Mistral-7B | TBD | TBD | TBD | — | — |
| CCM | Llama-2/Mistral | — | — | TBD | TBD | — |
| Activation Beacon | Qwen2-7B | — | — | — | TBD | TBD |
| xRAG | Mistral/Mixtral | — | TBD | TBD | TBD | — |
| Cartridges | Llama-3.2-3B | — | — | — | TBD | — |
| Gist Tokens | LLaMA/FLAN-T5 | — | TBD | — | — | — |
| 500xCompressor | Llama-3-8B | — | TBD | TBD | — | — |

---

## Fill map (where each TBD comes from)

| table | benchmark(s) | produced by | current state |
|---|---|---|---|
| 0, 1, 1a, 1b | audit, core, reproduction, gate | E0–E3 | 12/24 corrected QuALITY cells complete; old cells remain quarantined |
| 2 | source adapters | E4A | 7 corrected QuALITY-source adapters remain |
| 3–5 | LongBench-v2, ∞Bench, BABILong, LongBench tasks | E4B | 14 target reruns remain; BABILong collapse |
| 6–7 | capacity sensitivity | E7 | folded into K=64/K=256 ablations |
| 8 | fixed-config generality | E5 | BFCL complete across seven bases |
| 9 | mechanism ablations | E7 | 36/36 corrected and complete |
| 10 | measured cost | E8 | not started; TBD placeholders |
| 11 | official/native-base baselines | E9 | not started; TBD placeholders |

**Blocked / not in tables:** NoLiMa (data access), HELMET, LongMemEval (loaders not implemented).
