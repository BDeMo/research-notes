# Paper A model, baseline, evaluation, and training duration matrix

> Updated 2026-07-22 20:34 PT. `Observed` values come from completed status JSONs and include both training and full
> evaluation for a cell unless marked eval-only. `Estimated` values are planning ranges on one H100 NVL.
> Four healthy experiment GPUs are available.

## 1. Base models

There are **7 base models** in the GCM experiment package.

| base | architecture / role | train cells | long eval cells | estimated remaining GPU hours | expected wall time in current schedule |
|---|---|---:|---:|---:|---:|
| Qwen3-8B | dense; main base | 98 total across all grids | 23 | 66–110 | 1–2 days, shared over 3 GPUs |
| Qwen3.5-9B | hybrid GDN/full; main base | 42 | 23 | 117–180 | **5–7 days on secondary single GPU** |
| Qwen3.5-4B | small hybrid | 11 | 12 | 38–75 | 12–24 h once primary grid starts |
| GLM-4-9B-0414 | dense vendor base | 11 | 12 | 26–58 | 10–20 h |
| Ministral-8B-Instruct | dense Mistral family | 11 | 12 | 21–52 | 8–18 h |
| Llama-xLAM-2-8B | dense tool-tuned Llama | 11 | 12 | 21–52 | 8–18 h |
| ToolACE-2-8B | dense tool-tuned Llama | 11 | 12 | 21–52 | 8–18 h |

The five non-primary bases share three primary-pod GPUs. Qwen3.5-9B is the current bottleneck because it uses
the only healthy secondary-pod GPU and the safe pure-PyTorch hybrid-attention path.

### Observed main-grid total

| base | completed main GPU hours | note |
|---|---:|---|
| Qwen3-8B | about 48 h | 44 cells, including training and eval-only baselines |
| Qwen3.5-9B | about 184–199 h | 44 cells; pure-PyTorch GDN is slower |
| total main | about 232–247 h | duplicate technical runs make the exact physical total slightly larger |

## 2. Observed per-cell time on the two main bases

### Qwen3-8B — one full train+evaluation cell

| task | GCM observed | SFT observed | LL2 eval-only | window eval-only |
|---|---:|---:|---:|---:|
| QuALITY | 1.13–1.16 h | 0.77–0.81 h | 0.24 h | 0.19 h |
| BFCL | 0.62–0.70 h | 0.09–0.10 h | 0.04 h | 0.03 h |
| SQuAD-v2 | 0.51–0.71 h | 0.28–0.82 h | 0.42 h | 0.17 h |
| HotpotQA | 0.87–1.13 h | 0.74–1.02 h | 0.49 h | 0.42 h |

### Qwen3.5-9B — one full train+evaluation cell

| task | GCM observed | SFT observed | LL2 eval-only | window/raw eval-only |
|---|---:|---:|---:|---:|
| QuALITY | 4.35–9.99 h | 7.38–7.94 h | 1.19 h | 1.04–1.46 h |
| BFCL | 1.91–4.73 h | 0.70–1.61 h | 0.31 h | about 0.3 h |
| SQuAD-v2 | 3.81–9.38 h | 2.05–4.95 h | 3.38 h | 0.30 h |
| HotpotQA | 4.99–12.02 h | 3.11–6.89 h | 4.53 h | estimated 0.5–1 h |

The wide GCM ranges include full evaluation and technical reruns. They are not pure training throughput.

## 3. Baseline counts

### Same-base controls: 9 families

| baseline family | number of main cells | training? | current/planned duration |
|---|---:|---|---|
| no context | emitted inside every cell | no | included in cell time |
| bounded raw | emitted inside every cell | no | included in cell time |
| true raw | 2 | no | 0.28 h Qwen3; 1.46 h Qwen3.5 |
| full SFT | 30 | yes | table above; 0.1–7.9 h/cell |
| raw window | 8 | no | 0.03–1.0 h/cell |
| LLMLingua-2 | 8 | no | 0.04–4.5 h/cell |
| LongLLMLingua | 8 | no | **current rows invalid; official-env rerun 1–6 h/cell** |
| original LLMLingua | 8 | no | **current rows invalid; official-env rerun 1–6 h/cell** |
| exact mean pooling | 8 planned | no | estimated 0.1–1.5 h/cell |

### Official soft/recurrent comparisons: 11 families

| method | first planned package | training by us? | estimated GPU time |
|---|---|---|---:|
| LCLM | RULER + LongBench, 4x/8x/16x | no | 24–48 h |
| Semi-Dynamic | SQuAD/Hotpot, auto +32x | no | 4–8 h |
| Activation Beacon | LongBench/InfiniteBench | no | 12–24 h |
| AutoCompressor | Llama2-6k + OPT-30k | no | 8–16 h |
| ICAE v2 | SQuAD/Hotpot/capped QuALITY | no | 4–8 h |
| CCM | streaming + QA | no; adapter download | 8–16 h |
| xRAG | Hotpot and compatible QA | no | 8–16 h |
| Cartridges | native LongHealth | no for released cartridge | 2–4 h |
| Gist Tokens | short-context appendix | no | 2–4 h |
| Cramming 1568 | one-vector capacity sample | per-sample optimization | 5–20 h |
| 500xCompressor | blocked | weights unavailable | 0 h until released |

Required runnable official methods: **8**. Appendix/blocked methods: **3**.

## 4. Evaluation sets

### Main evaluation: 4 datasets

| evaluation | public validation size | metric | typical eval-only time Qwen3 | typical eval-only time Qwen3.5 |
|---|---:|---|---:|---:|
| QuALITY | 1,595 | answer-letter accuracy | 0.2–0.4 h | 1–2 h |
| BFCL-live-multiple | 316 | tool accuracy | 0.03–0.1 h | 0.3–0.5 h |
| SQuAD-v2 | 5,928 | token F1 | 0.2–0.6 h | 1–4 h |
| HotpotQA | 7,405 | token F1 | 0.4–1 h | 1–5 h |

Method-side compression time can dominate LLMLingua and GCM eval, so the upper range is method-dependent.

### Real long-context evaluation: 12 benchmark groups

| evaluation | cells/models | estimated time per model/method | purpose |
|---|---|---:|---|
| LongBench-v2 | all 7 bases | 0.5–2 h | hard long-context MC |
| InfiniteBench-choice | all 7; GCM also K32 | 2–6 h | 100k-scale book MC |
| MultiFieldQA | all 7 | 0.5–2 h | long extractive QA |
| Qasper | all 7 | 0.5–2 h | paper QA |
| LongBench HotpotQA | all 7 | 0.5–2 h | long multi-hop |
| 2WikiMQA | all 7 | 0.5–2 h | long multi-hop |
| MuSiQue | all 7 | 0.5–2 h | compositional QA |
| NarrativeQA | all 7 | 1–3 h | long narrative QA |
| BABILong QA1 | all 7 | 0.5–1.5 h | recurrent recall |
| BABILong QA2 | all 7 | 0.5–1.5 h | recurrent recall |
| BABILong QA3 | all 7 | 0.5–1.5 h | recurrent recall |
| RULER 4k/8k/16k/32k | budget grid + official baselines | 2–6 h per method suite | exact-retrieval boundary |

E4B contains **106 evaluation cells**. The estimated total is 135–270 GPU hours.

### Additional evaluation passes

| pass | count | size | estimated time |
|---|---:|---:|---:|
| feature rerun | up to 24 GCM adapters | full validation | 20–50 GPU h |
| output/truncation audit | 48 paths | 256 items each | 8–20 GPU h |
| reproducibility | 3 GCM cells | full QuALITY | 8–18 GPU h |
| cost profiling | up to 16 profiles | 20–100 timed items | 10–25 GPU h |

## 5. Training sets

There are **6 training datasets/families** in the GCM grid.

| training set | examples used per cell | models/stages | observed Qwen3 training-cell time | estimated Qwen3.5 time |
|---|---:|---|---:|---:|
| QuALITY | 1,899 | main, source, generality, budget, ablation | GCM source 0.61 h; SFT source 0.66 h | GCM 7–9 h; SFT 6–8 h |
| BFCL-live-multiple | 736 after dedup | main, generality, ablation | GCM full cell 0.62–0.70 h; SFT 0.09–0.10 h | GCM 2–4 h; SFT 0.7–1.6 h |
| SQuAD-v2 | 2,000 cap | main, source | GCM source 0.18 h; SFT source 0.16 h | GCM 3–5 h; SFT 2–4 h |
| HotpotQA | 2,000 cap | main, source | GCM source 0.23 h; SFT source 0.16 h | GCM 4–6 h; SFT 3–5 h |
| NarrativeQA | 2,000 cap | source adapters | GCM source 0.78 h; SFT still running/observed soon | GCM 4–8 h; SFT 3–6 h |
| RULER-NIAH | 500 generated training items | budget/length | estimated 1–3 h | not in current Qwen3.5 grid |

The source-adapter stage has **43 training cells**:

- 35 GCM = 7 bases × (4 source tasks + QuALITY K32);
- 8 SFT = 2 primary bases × 4 source tasks.

## 6. Training-cell totals by base

| base | main train | source train | generality | budget | ablation | replicate | total train cells |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | 27 + 6 SFT reaudit | 9 | 6 | 11 | 36 | 3 | **98** |
| Qwen3.5-9B | 27 | 9 | 6 | 0 | 0 | 0 | **42** |
| each other base | 0 | 5 | 6 | 0 | 0 | 0 | **11** |

Total: **195 trainable cells**.

## 7. Current wall-clock ETA

Live snapshot: 255/399 manifest cells are complete. Four jobs are active: three Qwen3.5-4B generality
cells and one Qwen3-8B RULER budget cell.

| milestone | current state | ETA from 2026-07-22 20:34 PT |
|---|---|---:|
| core main, audit, SFT reaudit, gate, reproducibility | complete | done |
| source adapters | 41/43; two K32 technical repairs | 6–12 h after isolated rerun |
| real long-context evaluation | 103/106; three K32 repairs | repairs separate |
| seven-base fixed-config generality | 12/42; three running | 1–2 additional days |
| budget and RULER length | 2/23; one running | 1–2 additional days |
| mechanism ablation | 0/36 | 0.5–1 additional day |
| official baselines | 0/52 local cells; separate environments | 2–4 days after launch |
| cost profiling and final harvest | not started | 0.5–1 day |
| full Paper A package | includes repairs, official baselines, cost, and harvest | **5–7 days (July 27–29)** |

The immediate scheduled critical path is long-context → generality → budget → ablation. The full-package
critical path is the not-yet-launched official-baseline suite plus K32 resource-isolated repairs.
