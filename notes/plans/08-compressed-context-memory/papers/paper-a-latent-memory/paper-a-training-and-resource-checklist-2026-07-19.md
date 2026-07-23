# Paper A training and resource checklist

> Operational runbook · updated 2026-07-22 20:34 PT. GPU-hour ranges are planning estimates, not measured costs.
> Every final table value must point to a status JSON, result JSON, per-item records, adapter when trained,
> model/checkpoint ID, and scorer.

## 0. Current resource snapshot

| resource | available now | use |
|---|---:|---|
| healthy experiment GPUs | 4× NVIDIA H100 NVL, 95.8 GiB each | primary GPU 0–2; secondary GPU 0 |
| CMG demo GPU | 1× H100 NVL, ~69.5 GiB occupied | excluded from Paper A scheduling |
| unavailable GPU | secondary GPU 1 | disappeared from CUDA; never schedule |
| primary host RAM | 2.0 TiB, 1.9 TiB available | model loading, CPU caches |
| secondary host RAM | 2.2 TiB, 2.1 TiB available | model loading, compressor LMs |
| primary persistent disk | 2.0 TiB total, 1.2 TiB free | code, bases, artifacts |
| secondary persistent disk | 2.0 TiB total, 1.6 TiB free | code, bases, artifacts |
| main environment | `/mnt/persist/venv` | GCM, SFT, LLMLingua, evaluation |
| experiment root | `/mnt/persist/paper_a` | statuses, logs, artifacts, harvest |

Current stage:

- main: 88/88 complete;
- output audit: 28/28 reloadable paths complete; SFT reaudit 6/6 complete;
- feature rerun: 24/24 canonical adapters complete; corrected gate analysis complete;
- source adapters: 41/43 complete; two K32 source adapters need technical repair;
- long-context: 103/106 complete; three K32 evaluations need technical repair;
- generality: 12/42 complete, three running, 27 pending;
- budget/length: 2/23 complete, one RULER cell running; ablation: 0/36 queued;
- official-baseline local cells: 0/52, not yet launched.

## 1. Master experiment ledger

| ID | stage | cells | training cells | purpose | expected outcome | planned GPU hours | state |
|---|---|---:|---:|---|---|---:|---|
| E0 | evidence audit | — | 0 | verify lengths, splits, overlap, scorers, access | remove false/full-context claims | <1 CPU h | complete |
| E1 | fair main | 88 | 54 | compare GCM with matched adaptation and text/raw controls | stable BFCL; task/model-dependent QuALITY; QA boundary | 250–400 | complete |
| E1Q | output/truncation audit | 28×256 items | 0 | detect fixed outputs, empty outputs, hidden caps, compressor fallback | every reloadable path has inspectable predictions and lengths | 8–20 | complete |
| E1R | QuALITY SFT reaudit | 6 | 6 | reproduce score-only SFT cells whose adapters were not saved | full outputs, saved adapters, explain seed variance | 5–8 | complete |
| E1F | feature rerun | 24 | 0 | add document IDs and gate probe features | one complete per-item record set per GCM adapter | 20–50 | complete |
| E2 | empirical gate | 24 groups ×20 splits | 0 | test compress-or-raw routing without test-selected threshold | empirical risk–coverage curves; gate demoted on strong-raw tasks | <4 CPU h | complete |
| E2F | formal gate check | one fixed split/group | 0 | test cluster-valid nonzero coverage | 0/24 certified; all-raw | <2 CPU h | complete negative |
| E3 | reproducibility | 3 | 3 | rerun Qwen3 QuALITY seeds with new run IDs | quantify hardware/kernel spread | 8–18 | complete |
| E4A | transfer-source training | 43 | 43 | train each source adapter once, including K32 QuALITY | reusable adapter for each base/source/method/budget | 100–200 | 41 done; 2 failed K32 repairs |
| E4B | real long-context eval | 106 | 0 | test source-trained transfer without target tuning | find length/task boundary; exact retrieval likely fails | 135–270 | 103 done; 3 failed K32 repairs |
| E5 | fixed-config generality | 42 | 42 | test K128 recipe on all seven bases | BFCL remains useful on most; QuALITY may vary | 160–310 | 12 done; 3 running; 27 pending |
| E6 | budget/length | 23 | 11 | map memory/raw budgets and RULER length | Pareto curve; RULER exposes exact-retrieval limit | 40–90 | 2 done; 1 running; 20 pending |
| E7 | mechanism ablation | 36 | 36 | test five method components and K | joint loss required; other effects small/task-dependent | 120–220 | pending |
| E8 | measured cost | ≤16 profiles | 0 | measure encoder, memory read, raw read, fallback | honest end-to-end cost, not ratio alone | 10–25 | pending |
| E9 | official baselines | variable | mostly 0 | compare author checkpoints on native bases/tasks | native-base retention and long-context frontier | 80–200 | preparation |

Core planned trainable jobs: **195** = 54 main + 6 SFT reaudit + 3 replicate + 43 transfer-source +
42 generality + 11 budget + 36 ablation. Official Cartridges training is optional and excluded from this count.

## 2. E1 — fair main table

### Runs

| component | cells | models | tasks | seeds |
|---|---:|---|---|---|
| GCM | 24 | Qwen3-8B, Qwen3.5-9B | QuALITY, BFCL, SQuAD, Hotpot | 42/43/44 |
| full SFT | 24 | same | same | 42/43/44 |
| QuALITY true-input SFT | 6 | same | QuALITY | 42/43/44 |
| raw full | 2 | same | QuALITY | deterministic |
| window / LL2 / LongLL / original LL | 32 | same | four tasks | seed 42 / eval-only |

### Purpose

1. Separate compression quality from task adaptation.
2. Compare GCM to the same raw-token read budget.
3. Test whether QuALITY gains survive true raw and matched SFT.
4. Establish negative boundaries on extractive and multi-hop QA.

### Expected / decision rule

- BFCL: GCM around 70–75, close to LLMLingua-2, below raw/SFT.
- QuALITY: stable Qwen3 GCM; no assumption that it wins on Qwen3.5.
- SQuAD/Hotpot: raw and SFT should win; report this directly.
- If true raw/SFT removes the QuALITY advantage, use a cost–quality claim only.

### Required outputs

- `status/<tag>.json`;
- `logs/<tag>.log`;
- `artifacts/<tag>/<tag>.json`;
- `artifacts/<tag>/<tag>_adapters.pt` for trained methods;
- full per-item records and actual method-token count.

## 3. E1F/E2 — feature and gate experiments

### Feature rerun

- input: each completed main GCM adapter;
- output: document ID, item ID, raw/memory/no-context scores, confidence/margin/TARG, direct probe features;
- no retraining;
- success: full validation records exactly match the source adapter's aggregate score.

### Empirical routing

- 25% exact-context-disjoint calibration, 75% test;
- 20 repeated splits for sensitivity only;
- primary signal fixed to compressed first-token confidence;
- threshold: maximum coverage under calibration signed policy excess ≤.02;
- report raw-only, memory-only, both-correct, both-wrong, harm, benefit, coverage, and fallback.

Expected: close to raw on some cells; useful rescue when the raw path fails; not a guaranteed improvement.

### Formal routing

- one pre-registered split, not 20 selectable splits;
- fixed numeric confidence thresholds;
- independent document clusters;
- zero coverage has undefined conditional risk and is reported as all-raw;
- no “finite-sample risk control” wording unless nonzero coverage is actually certified.

Expected at current sample size: likely no nonzero certificate for ε=.02.

## 4. E3 — reproducibility

| item | setting |
|---|---|
| model/task | Qwen3-8B / QuALITY |
| method | canonical GCM |
| seeds | 42, 43, 44 |
| variant | `repeat1`, separate run IDs |
| purpose | measure same-config run spread after backend lock |
| success | mean remains close to main result; spread is reported, not hidden |
| failure rule | large spread becomes a main limitation and widens intervals |

## 5. E4A — source-adapter training

### GCM source adapters

`7 bases × {QuALITY, SQuAD, HotpotQA, NarrativeQA} = 28`, plus
`7 bases × QuALITY K32 = 7`.

### SFT source adapters

`2 primary bases × 4 source tasks = 8`

Every adapter:

- seed 42;
- trained once;
- evaluated on one source validation item only to confirm loading;
- saved before transfer;
- reused unchanged by every corresponding E4B target.

Purpose: prevent target-specific retraining from being mislabeled as transfer.

Expected: all 35 adapters load and emit a source prediction. Source score is not a paper result for this stage.

## 6. E4B — real long-context transfer

| source | targets | models/methods | expected |
|---|---|---|---|
| QuALITY | LongBench-v2, InfiniteBench-choice | 7 GCM; 2 SFT | model-dependent MC transfer |
| SQuAD | MultiFieldQA, Qasper | 7 GCM; 2 SFT | extractive transfer likely below raw |
| Hotpot | LB-Hotpot, 2Wiki, MuSiQue, BABILong QA1–3 | 7 GCM; 2 SFT | multi-hop degradation; BABILong tests recurrence |
| NarrativeQA | LB-NarrativeQA | 7 GCM; 2 SFT | summarizable long context may transfer |
| QuALITY | InfiniteBench-choice K32 control | 7 GCM | bound growth of final memory length |

Total: 106 eval-only cells.

Required measurements:

- actual source length and latent length;
- target score;
- no-context and bounded raw score;
- encoder/read latency;
- truncation/skip count;
- no target labels used for training.

## 7. E5 — all-base fixed-config validation

| setting | value |
|---|---|
| bases | Qwen3-8B, Qwen3.5-9B/4B, GLM-4-9B, Ministral-8B, xLAM-8B, ToolACE-8B |
| tasks | QuALITY, BFCL |
| seeds | 42/43/44 |
| configuration | K128/chunk, no model-specific tuning |
| total | 42 training cells |

Purpose: test the algorithm/configuration, not shared latent vectors.

Expected:

- BFCL compressed score remains useful on most bases;
- QuALITY is more variable;
- adapters and embedding spaces remain model-specific.

Decision rule: if fixed K fails on more than two families, write **base-conditioned**, not model-agnostic.

## 8. E6 — budget and length

### GCM budgets

- tasks: QuALITY and RULER;
- K: 64, 128, 256, 512;
- 8 cells.

### Raw windows

- tasks: QuALITY and RULER;
- tokens: 256, 512, 1,024, 2,048, 4,096, 8,192;
- 12 eval-only cells.

### RULER lengths

- 4k, 8k, 32k;
- K128/chunk;
- 3 cells.

Purpose: produce accuracy–reader-state–length curves.

Expected: QuALITY has an intermediate useful K; RULER favors raw token retention and exposes the exact-string limit.

## 9. E7 — mechanism ablation

| variant | change | question | expected |
|---|---|---|---|
| `joint0` | detach answer loss from encoder | does memory learn answer-bearing content? | largest negative effect |
| `distill0` | remove raw-teacher KL | does teacher behavior help? | small/moderate drop |
| `recon0` | remove slot reconstruction | does preserve loss help? | task-dependent, likely small on BFCL |
| `recur0` | encode chunks independently | does prior-memory recurrence matter? | larger effect on long QuALITY |
| `k64` | half memory budget | capacity sensitivity | lower state, possible score drop |
| `k256` | double memory budget | whether more state helps | small gain or dilution |

Scope: Qwen3-8B × {QuALITY, BFCL} × 3 seeds = 36.

Success: report effect sizes and intervals. Do not declare a component required from one seed.

## 10. E8 — measured cost

For up to 16 representative adapters on the same H100:

1. encoder scan latency and peak memory;
2. compressed-reader latency and peak memory;
3. bounded-raw latency and peak memory;
4. fallback latency;
5. actual source, memory, query, and generated tokens;
6. expected route cost at measured coverage.

Expected: reader-state savings are clear; end-to-end savings depend on reuse and fallback. Query-conditioned GCM
must not be described as automatically amortized across queries.

## 11. E9 — official baseline execution

### Inference-only, public checkpoints

| method | checkpoint / base | required runs | purpose / expected |
|---|---|---|---|
| LCLM | `latent-context/0.6b-4b-LCLM-{4,8,16}x` | RULER + LongBench at 4/8/16x | latest general soft-token frontier |
| Semi-Dynamic | `yuyijiong/qwen3-semi-dynamic-soft-context-compress` | SQuAD/Hotpot, auto + fixed32x | latest adaptive short-soft baseline |
| Activation Beacon | `namespace-Pt/beacon-qwen-2-7b-instruct` | LongBench + InfiniteBench | progressive long-context baseline |
| AutoCompressor | Llama-2-7B-6k; OPT-30k | QuALITY/QA + long | recurrent soft-summary ancestor |
| ICAE v2 | Mistral-7B official ICAE | SQuAD/Hotpot/QuALITY≤5120 | frozen-decoder soft memory |
| xRAG | `Hannibal046/xrag-7b` | Hotpot and compatible retrieval QA | one soft token per retrieved document |
| CCM | released Llama-2/Mistral adapters | streaming + chunked QA | 2/8-state recurrent cache |
| Cartridges | `hazyresearch/cartridge-wauoq23f` | native LongHealth | reusable corpus KV state |

### Appendix / blocked

| method | action |
|---|---|
| Gist Tokens | short-context row only; requires LLaMA-1 or FLAN-T5-XXL |
| Cramming 1568 | run one-vector reconstruction capacity only |
| 500xCompressor | mark blocked; weights and ArxivQA unavailable |
| new Cartridges | optional; requires per-corpus synthesis and training |

### Official baseline resource rule

Each method uses an isolated environment and writes canonical JSONL:

`{task, item_id, base, checkpoint, method, pred, gold, input_tokens, state_tokens, latency, peak_memory}`

Plus same-base no-context and raw references.

## 12. Model resources

| slug | path / checkpoint | role |
|---|---|---|
| q3_8b | `/mnt/persist/checkpoints/Qwen3-8B` | main dense base |
| q35_9b | `/mnt/persist/checkpoints/Qwen3.5-9B` | main hybrid base |
| q35_4b | `/mnt/persist/checkpoints/Qwen3.5-4B` | small hybrid stress |
| glm4_9b | `/mnt/persist/checkpoints/GLM-4-9B-0414` | vendor generality |
| ministral_8b | `/mnt/persist/checkpoints/Ministral-8B-Instruct-2410` | Mistral family |
| xlam_8b | `/mnt/persist/checkpoints/Llama-xLAM-2-8b-fc-r` | tool-tuned Llama |
| toolace_8b | `/mnt/persist/checkpoints/ToolACE-2-8B` | tool-tuned Llama |

No adapter or latent vector is shared between these models.

## 13. Dataset resources

### Main

| dataset | train used | validation | encoder/raw cap | metric |
|---|---:|---:|---|---|
| QuALITY | 1,899 | 1,595 | 16,384 / 8,192 | letter accuracy |
| BFCL-live-multiple | 736 after dedup | 316 | 4,096 / 4,096 | tool accuracy |
| SQuAD-v2 | 2,000 cap | 5,928 | 4,096 / 4,096 | token F1 |
| HotpotQA | 2,000 cap | 7,405 | 4,096 / 4,096 | token F1 |

### Long and controlled

- NarrativeQA;
- RULER-NIAH;
- LongBench-v2;
- InfiniteBench-choice;
- LongBench MultiFieldQA, Qasper, HotpotQA, 2WikiMQA, MuSiQue, NarrativeQA;
- BABILong QA1, QA2, QA3 at 16k.

Blocked or missing:

- NoLiMa: authenticated/local data needed;
- HELMET and LongMemEval: loaders not implemented.

## 14. Software environments

| environment | methods | key requirements |
|---|---|---|
| `/mnt/persist/venv` | GCM, SFT, LLMLingua-2, windows | current PyTorch/Transformers; pure-torch Qwen3.5 training path |
| `env-longll` | LongLLM / original LL | official Transformers 4.38-compatible cache API; fallback disabled |
| `env-lclm` | LCLM | repo `uv sync`, flash-attn, vLLM encode/decode split |
| `env-semidynamic` | Semi-Dynamic | recent Transformers with Qwen3, released HF checkpoint |
| `env-beacon` | Activation Beacon | Python 3.10, CUDA 12.1, flash-attn, long-llm data |
| `env-ac` | AutoCompressor | Transformers 4.34, flash-attn 2.3.5 |
| `env-icae-v2` | ICAE | Transformers ≥4.36.2, PEFT, safetensors, flash-attn |
| `env-ccm-mistral` | CCM | Transformers 4.37.2, PEFT, downloaded adapters |
| `env-xrag` | xRAG | Transformers 4.38, SFR-Embedding-Mistral |
| `env-cartridges` | Cartridges | Python 3.12, Transformers 4.49–4.55, FlexAttention |
| `env-gisting` | Gist appendix | old HF commit, DeepSpeed 0.8.3, LLaMA-1 access |

Do not install these mutually incompatible packages into the main environment.

## 15. Storage plan

Estimated additional reservation:

| item | reserve |
|---|---:|
| remaining GCM/SFT adapters and optimizer-free artifacts | 80–120 GiB |
| per-item JSON, feature records, gate/cost harvest | 10–30 GiB |
| LCLM 4/8/16x + vLLM staging | 40–80 GiB |
| Beacon/AutoCompressor/ICAE/CCM/xRAG checkpoints | 100–180 GiB |
| temporary caches and isolated envs | 80–150 GiB |
| recommended total free-space reserve | 350–550 GiB |

Both pods currently exceed this requirement.

## 16. Artifact and acceptance checklist

For every final row:

- [ ] current manifest tag or official baseline ID;
- [ ] official repository commit;
- [ ] checkpoint and base ID;
- [ ] exact prompt and scorer;
- [ ] train/eval split and duplicate guard;
- [ ] independent seed or clearly labeled technical repeat;
- [ ] full validation count or explicit truncation/skip count;
- [ ] no-context, raw, and compressed scores;
- [ ] source and realized state-token counts;
- [ ] latency and peak memory;
- [ ] per-item prediction/score records;
- [ ] failure and N/A reason preserved;
- [ ] no internal replica labeled as the published method.

## 17. Current execution order

1. finish E1 main repairs;
2. finish E1Q output/truncation audit and invalidate silent fallbacks;
3. finish E1R six-cell QuALITY SFT reaudit;
4. finish feature reruns and empirical gate;
5. E3 replicate;
6. E4A source adapters;
7. E4B long-context evaluation;
8. E5 fixed-config seven-base validation;
9. E6 budget/length;
10. E7 ablation;
11. E8 cost and final harvest;
12. official baselines run in parallel as isolated environments become ready.

Estimated wall-clock completion with four healthy experiment GPUs: **7–10 days**, assuming no additional
GPU loss. 500xCompressor remains blocked and is not on the critical path.
