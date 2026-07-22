# Paper A experiment receipt

> Snapshot: 2026-07-22 15:43 PT. This is the operational source of truth for what is configured, running,
> queued, blocked, or excluded. “Receipt” means exact recipe + evaluation contract + artifact path.

## 1. Live status

| stage | configured cells | done | running | failed / repair | state |
|---|---:|---:|---:|---:|---|
| E0 evidence audit | 2 tokenizer/model audits + overlap check | complete | 0 | 0 | locked |
| E1 fair main grid | **88** | **88** | 0 | 0 | complete |
| E1Q output/truncation audit | 28 reloadable paths | 28 | 0 | 0 | complete |
| E1R QuALITY SFT reaudit | 6 full retrains | 6 | 0 | 0 | complete |
| E1F feature-only rerun | 24 canonical GCM adapters | 24 | 0 | 0 | complete |
| E2 gate/risk analysis | 24 groups × 20 splits | 24 | 0 | 0/24 formal certified | complete negative |
| E3 reproducibility | 3 duplicate-seed GCM cells | 3 | 0 | 0 | complete |
| E4A transfer-source adapters | **48** | 46 | 0 | 2 failed / 0 pending | complete except K32 repairs |
| E4B real-long evaluation | **118** | 111 | 1 | 3 failed / 3 pending | running |
| E5 fixed-config generality | **48** | 12 | 0 | 36 pending | waiting for long-context stage |
| E6 budget/length | **23** | 2 | 1 | 20 pending | running |
| E7 mechanism ablation | **36** | 0 | 0 | 0 | queued |
| E8 measured cost | up to 16 adapter profiles | 0 | 0 | 0 | finalization |
| E9 official baselines | 52 local cells + native-base runs | 0 | 0 | 52 pending; LCLM/Semi-Dynamic cloned | separate envs |

Current live workers: one ToolACE InfiniteBench long-context job on primary GPU 1 and one Qwen3-8B RULER budget job on
secondary GPU 0. Primary GPUs 0 and 2 are temporarily idle because their deterministic long-context shards
finished before shard 1; the sequencer advances after the remaining shard exits. Primary GPU 3 is reserved
outside Paper A, and secondary GPU 1 is unavailable.

The five failed cells are technical, not scientific negatives. GLM-4 K32 source training ran out of GPU
memory while another process occupied most of the device; Qwen3.5-9B K32 source training was killed during
training; the Qwen3/Qwen3.5/GLM K32 InfiniteBench cells then lacked the required adapter on the evaluating
pod. These cells require resource-isolated retraining or adapter transfer before rerun.

The earlier `74/112` count was invalid: it summed duplicate cross-pod cells and stale tags. The corrected
manifest removes the internal Gist smoke baseline. All old Qwen3.5 cells are archived and rerun because the
training backend changed from a mixed FLA/Triton path to one tested pure-PyTorch path. All 88 main cells
are complete. Eight duplicate tags are retained only as technical repeats and
averaged within seed.

## 2. GCM core recipe

| item | exact setting |
|---|---|
| base | frozen; one independent adapter per model |
| memory queries | K=128 **per 4,096-token chunk** |
| effective memory length | `S×K`, `S=ceil(context_tokens/4096)` |
| encoder depth | half of base layers |
| encoder weights | frozen base; no encoder-LoRA |
| encoder input | `[prior projected M; current chunk; query; K memory queries]` |
| position | native model positions; each chunk re-indexed from 0 |
| recurrent prefix | all previous projected chunk memories, detached between chunks |
| variable-length training | on |
| projection | `d → 2d → d`, GELU |
| normalization | hard norm match to mean input-embedding norm |
| read interface | soft prefix `[M; query]` |
| read adapter | LoRA rank 64; attention/GDN/MLP targets |
| fallback | adapter off; frozen feasible raw path |
| task objective | teacher-forced answer CE; gold ends with newline |
| self-distillation | weight 0.5; feasible-raw teacher; top-64 logits |
| reconstruction | weight 0.5; slot reconstruction; first 512 context tokens |
| optimizer | AdamW |
| learning rate | 3e-4; warmup + cosine |
| batch | 1 × gradient accumulation 8 |
| steps | 2,000 micro-steps |
| seeds | 42, 43, 44 |
| exact-overlap guard | on |
| generation score fallback | off |

Trainable parameters for Qwen3-8B default: approximately 242.2M, including the broad rank-64 read-LoRA.

## 3. Primary models

| slug | checkpoint | architecture | hidden d | tokenizer group | role |
|---|---|---|---:|---|---|
| q3_8b | Qwen3-8B | dense quadratic | 4096 | Qwen2-compatible BPE | primary |
| q35_9b | Qwen3.5-9B | hybrid GDN/full | 4096 | Qwen3.5 248k BPE | primary linear/hybrid |
| q35_4b | Qwen3.5-4B | hybrid GDN/full | 2560 | same Qwen3.5 tokenizer | small stress |
| q25_7b | Qwen2.5-7B-Instruct | dense quadratic | 3584 | token IDs align with Qwen3 core | family |
| glm4_9b | GLM-4-9B-0414 | dense quadratic | 4096 | GLM BPE | vendor |
| ministral_8b | Ministral-8B-Instruct | dense quadratic | 4096 | Mistral BPE | vendor |
| xlam_8b | Llama-xLAM-2-8B | dense quadratic | 4096 | Llama-3 BPE | tool-tuned |
| toolace_8b | ToolACE-2-8B | dense quadratic | 4096 | same xLAM tokenizer/near-identical embeddings | tool-tuned |

No memory, projection, or LoRA parameters are shared across these bases.

## 4. Development-task settings

| bench | train N | validation N | feasible raw | encoder access | median context | metric | max generation |
|---|---:|---:|---:|---:|---:|---|---:|
| QuALITY | 1,899 | 1,595 | 8,192 | 16,384 | 6,511 (Q3) | letter log-likelihood accuracy | 8 |
| BFCL-live-multiple | 736 after dedup | 316 | 4,096 | 4,096 | 167 | tool accuracy | 32 |
| SQuAD-v2 | 2,000 cap | 5,928 | 4,096 | 4,096 | 168 | native token-F1 | 16 |
| HotpotQA | 2,000 cap | 7,405 | 4,096 | 4,096 | 1,075 | native token-F1 | 16 |

QuALITY `default` and early `truefull` GCM tags used the same 16,384-token encoder setting. The duplicate
tags are technical replicates, not separate methods.

## 5. E1 fair-main methods

| method/path | training | information access | reader-state target | status |
|---|---|---|---|---|
| no context | none | query only | 0 context tokens | emitted by every cell |
| feasible raw | none | first B raw tokens | B | emitted by every cell |
| true raw | none | all audited QuALITY context | up to 16,384 | QuALITY only |
| full+SFT | rank-64 LoRA, matched data/steps/seeds | raw path | B / true-raw variant | main complete; Q3 QuALITY reaudit complete |
| raw window | none | sink + recent tail | matched realized memory | complete |
| LLMLingua-2 | official `llmlingua==0.2.2` | full source → token classifier | explicit target token | complete |
| LongLLMLingua | official Llama-2 compressor LM | list of source chunks + raw question | explicit target token | current rows invalid; isolated-env rerun |
| original LLMLingua | official Llama-2 compressor LM | list of source chunks | explicit target token | current rows invalid; isolated-env rerun |
| Compressor (w/o gate) | per-model recipe above | up to encoder cap | actual S×K | complete |
| Compressor (w/ empirical gate) | no new model training | compressed memory or feasible raw | variable | complete; held-out empirical result |
| Compressor (w/ formal gate) | calibration only | compressed memory or feasible raw | variable / all-raw | complete negative; 0/24 certified |

The internal Gist cell was removed from E1. It is an **adapted smoke**, not the published Gist baseline.
The official integration below is the only Gist result allowed in paper claims.

## 6. Gate and routing receipt

| item | setting |
|---|---|
| primary score | compressed-path first-token confidence |
| simple baselines | margin, entropy, TARG |
| learned baselines | Belikova-style joint query–memory probe; PoC-style gap regressor |
| split | document-disjoint 25% calibration / 75% test |
| repeats | 20 |
| empirical gate | maximum coverage under calibration signed policy excess ≤0.02 |
| finite-sample gate | cluster-level fixed-family analysis under correction; no current positive claim |
| controlled loss | accepted-set positive compression harm |
| epsilon | 0.02 |
| family-wise delta | 0.10 |
| no certified threshold | all-raw fallback; conditional harm is undefined at zero coverage |
| required outcomes | both correct; raw-only; memory-only; both wrong |
| required rates | accepted harm, accepted benefit, coverage, fallback |
| required cost | encoder, compressed read, raw fallback, expected total |

## 7. Real long-context transfer

All eight bases run GCM at discovery seed 42. Qwen3-8B and Qwen3.5-9B additionally run source-trained SFT.
E4A trains one adapter per `(base, source task, method)` and saves it before evaluation. Every E4B target
loads that exact adapter; target rows never retrain a nominally identical source adapter.

| source adapter | target evaluation |
|---|---|
| QuALITY | LongBench-v2; ∞Bench-choice |
| SQuAD | LongBench MultiFieldQA; Qasper |
| HotpotQA | LongBench HotpotQA; 2WikiMQA; MuSiQue; BABILong QA1–QA3 |
| NarrativeQA | LongBench NarrativeQA |

Additional controls:

- ∞Bench K=32/chunk on every base to limit final memory growth.
- RULER 4k/8k/16k/32k.
- NoLiMa is blocked by dataset access.
- HELMET and LongMemEval require new loaders.

## 8. Generality, budget, and ablation

| grid | exact scope |
|---|---|
| fixed-config generality | 8 bases × {QuALITY, BFCL} × 3 seeds = 48 |
| reproducibility | Qwen3-8B QuALITY × seeds 42/43/44 duplicate runs = 3 |
| GCM budget | K∈{64,128,256,512} on QuALITY and RULER |
| raw budget | 256/512/1024/2048/4096/8192 tokens |
| RULER length | 4k/8k/16k/32k |
| mechanism | joint off; distill off; reconstruction off; recurrence off; K64; K256 |
| mechanism models/tasks | Qwen3-8B × {QuALITY, BFCL} × 3 seeds = 36 |

## 9. Official faithful baselines

Only authors' code/checkpoints count as paper baselines.

| baseline | pinned upstream | official base/checkpoint | native compatible tasks | state |
|---|---|---|---|---|
| LCLM | `LeonLixyz/LCLM@e04ceb9` | Qwen3 0.6B encoder + 4B decoder; 4/8/16x | RULER/LongBench/LongHealth | cloned; public weights; required |
| Semi-Dynamic | `yuyijiong/semi-dynamic@07fae01` | Qwen3; adaptive 2/4/8/16/32x | SQuAD/Hotpot/short QA | cloned; public weights; required |
| Gist Tokens | `jayelm/gisting@3be0d06` | LLaMA-1-7B diff / FLAN-T5-XXL | short prompts only | LLaMA access blocked; short-context row only |
| AutoCompressor | `princeton-nlp/AutoCompressors@80352a4` | Llama-2-7B-6k; OPT 6k/30k | QuALITY/QA; long via 30k OPT | public checkpoints; highest-priority env |
| ICAE | `getao/icae@469a468` | Mistral-7B ICAE, max 5,120 | QA within native cap | public v2 checkpoint; isolated env pending |
| CCM | `snu-mllab/context-memory@a89dd08` | Llama/Llama-2 adapters | recurrent chat/ICL; custom wrappers for our tasks | cloned; Google Drive adapters |
| Activation Beacon | `FlagEmbedding@7ed43d6` | official Qwen2-7B-Instruct Beacon | native LongBench/∞Bench/passkey | public checkpoint; long-context priority |
| xRAG | `Hannibal046/xRAG@121fa41` | public Mistral-7B / Mixtral + SFR | Hotpot/QA retrieval rows | public; use gold/chunk retrieval, not a 4-TB index |
| 500xCompressor | `ZongqianLi/500x@ff454a1` | Llama-3-8B, 500-token spans | SQuAD; truncated Hotpot if weights obtained | weights/data not public; blocked |
| Cartridges | `HazyResearch/cartridges@ef34ba9` | public LongHealth cartridge on Llama-3.2-3B | corpus-conditioned long QA | other tasks require per-corpus self-study |
| Cramming 1568 | `yurakuratov/hidden_capacity@c371c3f` | one optimized vector on Llama-3.1-8B | PG19 reconstruction | capacity appendix; not online compression |
| LLMLingua family | `microsoft/LLMLingua@e0e9d99` | official compressor LMs | all 8 readers | integrated; faithful repair running |
| mean pooling | exact local operation | each reader's embeddings | all bases | queued |

If a method uses a different official base, report within-base retention:

$$
\mathrm{retention}
=
\frac{\mathrm{compressed}-\mathrm{noctx}}
{\mathrm{raw}-\mathrm{noctx}}.
$$

Internal reimplementations are labeled **ADAPTED REPLICA** and excluded from final claims.

## 10. Artifact paths

| artifact | path |
|---|---|
| audit | `paper-b-forgetting-gating/results-v1.8.0/paper-a-audit/` |
| per-pod experiment root | `/mnt/persist/paper_a/` |
| cell logs | `/mnt/persist/paper_a/logs/` |
| cell status | `/mnt/persist/paper_a/status/` |
| isolated artifacts | `/mnt/persist/paper_a/artifacts/` |
| feature reruns | `/mnt/persist/paper_a/feature_artifacts/` |
| gate output | `/mnt/persist/paper_a/harvest/gate.json` |
| cost profiles | `/mnt/persist/paper_a/harvest/cost/` |
| GCM runner | `/Users/s1shi/workspace/gcm/experiments/paper_a_grid.py` |
| faithful baseline policy | [`baseline-selection-2026-07-16.md`](baseline-selection-2026-07-16.md) |
| official worktrees | `/Users/s1shi/workspace/paper-a-official-baselines/` |
