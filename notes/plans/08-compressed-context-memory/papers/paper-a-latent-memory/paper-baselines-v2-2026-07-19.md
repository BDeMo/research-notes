# Paper A baseline plan v2

> Clear execution plan · 2026-07-19  
> Rule: a published method is named as a baseline only when we use the authors' code and released
> checkpoint. Methods on different base models are compared by within-base retention.

## 1. Final selection

| priority | method | type | compressed state | official base | native scope | decision |
|---:|---|---|---:|---|---|---|
| 1 | [LCLM](https://arxiv.org/abs/2606.09659) | soft tokens | `L/4`, `L/8`, `L/16` | Qwen3 0.6B encoder + 4B decoder | RULER, LongBench, LongHealth | **must run** |
| 2 | [Semi-Dynamic Context Compression](https://arxiv.org/abs/2603.25926) | adaptive soft tokens | selected 2x–32x | Qwen3 0.6B / 4B | SQuAD, Hotpot, short QA | **must run** |
| 3 | [Activation Beacon](https://openreview.net/forum?id=45zwTmts0G) | condensed activations | adaptive | Qwen2-7B-Instruct | LongBench, InfiniteBench | **must run** |
| 4 | [AutoCompressor](https://aclanthology.org/2023.emnlp-main.232/) | recurrent soft summary | 50 tokens/segment | Llama-2-7B / OPT | 6k / 30k | **must run** |
| 5 | [ICAE](https://openreview.net/forum?id=uREj4ZuGJE) | soft memory slots | 128/segment, ≤512 | Mistral-7B-Instruct | QA ≤5,120 tokens | **must run** |
| 6 | [CCM](https://openreview.net/forum?id=64kSvC4iPg) | recurrent KV memory | 2 or fixed 8 | Llama-2 / Mistral | streaming, chat, ICL | **must run** |
| 7 | [xRAG](https://openreview.net/forum?id=6pTlXqrO0p) | retrieval soft token | 1/document | Mistral-7B / Mixtral | retrieval QA | run compatible rows |
| 8 | [Cartridges](https://openreview.net/forum?id=0k5w8O0SNg) | reusable KV prefix | corpus-specific | Llama-3.2-3B | LongHealth | native task first |
| 9 | [Gist Tokens](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3d77c6dcc7f143aa2154e7f4d5e22d68-Abstract.html) | gist KV token | 1 | LLaMA-1 / FLAN-T5 | short instructions | appendix only |
| 10 | [Cramming 1568](https://aclanthology.org/2025.acl-long.948/) | optimized input vector | 1 | Llama-3.1-8B | reconstruction | capacity appendix |
| blocked | [500xCompressor](https://aclanthology.org/2025.acl-long.1219/) | layerwise KV slots | 1 / 4 / 16 | Llama-3-8B | 500-token spans | weights unavailable |

## 2. Latest methods added

### 2.1 LCLM — latest general long-context soft-token baseline

| item | value |
|---|---|
| paper | [End-to-End Context Compression at Scale](https://arxiv.org/abs/2606.09659) |
| code | [LeonLixyz/LCLM](https://github.com/LeonLixyz/LCLM) |
| checkpoints | [latent-context models](https://huggingface.co/latent-context) |
| exact models | `latent-context/0.6b-4b-LCLM-{4,8,16}x` |
| evaluation data | [latent-context/lclm-eval](https://huggingface.co/datasets/latent-context/lclm-eval) |
| local commit | `paper-a-official-baselines/lclm@e04ceb9` |
| encoder / decoder | Qwen3-Embedding-0.6B / Qwen3-4B-Instruct-2507 |
| latent length | `ceil(source_tokens / {4,8,16})` |
| encoder window | 1,024 |
| required runs | RULER + LongBench at 4x, 8x, 16x |
| reference | same decoder, no context and uncompressed raw context |

**Important:** LCLM-16x is not shorter than GCM. At 16k input, LCLM-16x uses about 1,024 latent tokens;
GCM K128/chunk uses 512. LCLM is required because it is the newest public general long-context soft-token
system, not because it has the smallest state.

### 2.2 Semi-Dynamic — latest adaptive shorter-soft baseline

| item | value |
|---|---|
| paper | [Density-aware Soft Context Compression with Semi-Dynamic Compression Ratio](https://arxiv.org/abs/2603.25926) |
| code | [yuyijiong/semi-dynamic-context-compress](https://github.com/yuyijiong/semi-dynamic-context-compress) |
| checkpoint | [qwen3-semi-dynamic-soft-context-compress](https://huggingface.co/yuyijiong/qwen3-semi-dynamic-soft-context-compress) |
| training data | [context_qa_sum_qwen3_synthetic](https://huggingface.co/datasets/yuyijiong/context_qa_sum_qwen3_synthetic) |
| local commit | `paper-a-official-baselines/semi-dynamic-context-compress@07fae01` |
| base | Qwen3 0.6B and 4B variants |
| candidate ratios | 2x, 4x, 8x, 16x, 32x |
| training context | 128–1,300 tokens |
| required runs | auto ratio + fixed 32x on SQuAD and Hotpot |
| reference | same Qwen3 base, raw and no-context |

This is the direct comparison for “use fewer soft embeddings when the input is easier.” It is not used as
a 16k–100k long-context result because the released training range is short.

## 3. Shortest-state comparison

| method | state size | runnable | what it actually does | use in paper |
|---|---:|---|---|---|
| [xRAG](https://github.com/Hannibal046/xRAG) | 1 token/retrieved document | yes | projects a retriever embedding | retrieval QA rows |
| [Gist](https://github.com/jayelm/gisting) | 1 gist KV token | partly | compresses short instructions | short appendix |
| [CCM](https://github.com/snu-mllab/context-memory) | 2 or 8 KV positions | yes after download | recurrent online cache | streaming table |
| [500xCompressor](https://github.com/ZongqianLi/500xCompressor) | 1 / 4 / 16 per 500 tokens | no weights | learned extreme KV state | blocked row |
| [Cramming 1568](https://github.com/yurakuratov/hidden_capacity) | 1 vector | yes | optimizes a new vector per text | capacity only |
| [Semi-Dynamic](https://github.com/yuyijiong/semi-dynamic-context-compress) | input-dependent, ≤32x | yes | predicts one of five ratios | adaptive table |
| GCM | 128 per 4,096 tokens | yes | recurrent soft memory | proposed method |
| [LCLM](https://github.com/LeonLixyz/LCLM) | 1 per 4/8/16 tokens | yes | general encoder–decoder | long table |

**Conclusion:** no public method currently compresses arbitrary 16k–100k input into one or a few soft
embeddings and supports ordinary long-document QA. The very short methods have a different interface,
require per-sample optimization, support short prompts, or lack weights.

## 4. Same-base controls

These methods run on our Qwen3-8B and Qwen3.5-9B bases.

| method | purpose | operating point | status |
|---|---|---|---|
| no context | query-only floor | 0 context tokens | complete |
| bounded raw | feasible raw reference | 8k QuALITY; 4k otherwise | complete |
| true raw | all available QuALITY input | up to 16k | one repair |
| full SFT-LoRA | matched task adaptation | rank 64; 3 seeds | nearly complete |
| raw window | matched token-state control | 128; 256 on QuALITY | two repairs |
| [LLMLingua-2](https://github.com/microsoft/LLMLingua) | hard token compression | matched tokens | complete |
| [LongLLMLingua](https://github.com/microsoft/LLMLingua) | question-aware compression | matched tokens | current rows invalid; isolated-env rerun |
| original [LLMLingua](https://github.com/microsoft/LLMLingua) | perplexity compression | matched tokens | current rows invalid; isolated-env rerun |
| exact mean pooling | simple latent control | same base/state budget | queued |
| GCM | proposed method | K128/chunk | one final cell |

## 5. Official soft/recurrent methods

| method | paper | code | checkpoint/assets | native limit | planned tasks |
|---|---|---|---|---|---|
| LCLM | [arXiv](https://arxiv.org/abs/2606.09659) | [GitHub](https://github.com/LeonLixyz/LCLM) | [HF](https://huggingface.co/latent-context) | long, official eval to very long context | RULER, LongBench |
| Semi-Dynamic | [arXiv](https://arxiv.org/abs/2603.25926) | [GitHub](https://github.com/yuyijiong/semi-dynamic-context-compress) | [HF](https://huggingface.co/yuyijiong/qwen3-semi-dynamic-soft-context-compress) | trained to 1,300 | SQuAD, Hotpot |
| Activation Beacon | [OpenReview](https://openreview.net/forum?id=45zwTmts0G) | [GitHub](https://github.com/FlagOpen/FlagEmbedding/tree/master/research/Long_LLM/activation_beacon) | [HF](https://huggingface.co/namespace-Pt/beacon-qwen-2-7b-instruct) | LongBench/InfiniteBench | long table |
| AutoCompressor | [ACL](https://aclanthology.org/2023.emnlp-main.232/) | [GitHub](https://github.com/princeton-nlp/AutoCompressors) | [Llama2-6k](https://huggingface.co/princeton-nlp/AutoCompressor-Llama-2-7b-6k) | 6k / 30k | QuALITY, QA, long |
| ICAE | [OpenReview](https://openreview.net/forum?id=uREj4ZuGJE) | [GitHub](https://github.com/getao/icae) | [HF](https://huggingface.co/sggetao/icae) | 5,120 | SQuAD, Hotpot, capped QuALITY |
| CCM | [OpenReview](https://openreview.net/forum?id=64kSvC4iPg) | [GitHub](https://github.com/snu-mllab/context-memory) | repo download script / Google Drive | short chunks, 16k streaming demo | streaming + QA |
| xRAG | [OpenReview](https://openreview.net/forum?id=6pTlXqrO0p) | [GitHub](https://github.com/Hannibal046/xRAG) | [xrag-7b](https://huggingface.co/Hannibal046/xrag-7b) | retrieval documents | Hotpot, Qasper, MultiFieldQA |
| Cartridges | [OpenReview](https://openreview.net/forum?id=0k5w8O0SNg) | [GitHub](https://github.com/HazyResearch/cartridges) | [LongHealth cartridge](https://huggingface.co/hazyresearch/cartridge-wauoq23f) | corpus-specific | native LongHealth first |

## 6. Paper table layout

### Table 1 — same-base main result

| rows | columns |
|---|---|
| QuALITY, BFCL, SQuAD, Hotpot × two main bases | no-context, bounded raw, true raw, SFT, window, LL2, GCM, empirical route |

Only directly comparable Qwen rows appear here.

### Table 2 — native-base soft-memory retention

| rows | operating points |
|---|---|
| LCLM | 4x, 8x, 16x |
| Semi-Dynamic | automatic, fixed 32x |
| AutoCompressor | Llama2-6k, OPT-30k |
| ICAE | native 4x / realized slots |
| CCM | concat-2, merge-8 |
| Activation Beacon | `adapt-1024` |
| xRAG | 1 token/document |
| GCM | K128/chunk, K32 long control |

Required columns: base, checkpoint, source tokens, realized state tokens, raw score, no-context score,
compressed score, retention, encoder latency, reader latency, peak memory.

### Table 3 — long-context result

| method | settings |
|---|---|
| raw / window | matched context and state budgets |
| LLMLingua-2 | matched reader tokens |
| LCLM | 4x, 8x, 16x |
| Activation Beacon | `adapt-1024` |
| AutoCompressor | 30k checkpoint |
| CCM | streaming state |
| GCM | K128/chunk and K32/chunk |

Tasks: RULER 4k/8k/16k/32k, LongBench-v2, InfiniteBench, NarrativeQA, Qasper, MultiFieldQA, HotpotQA,
2WikiMQA, MuSiQue, BABILong QA1–3.

### Appendix — extreme state

xRAG-1, Gist-1, CCM-2/8, 500x-1/4/16 (blocked), Cramming-1, and exact mean pooling.

## 7. Cross-base comparison

Use:

$$
\mathrm{retention}
=
\frac{\mathrm{compressed}-\mathrm{noctx}}
{\mathrm{raw}-\mathrm{noctx}}.
$$

Do not rank absolute scores across unrelated bases.

Every official row records:

1. paper, repository commit, and checkpoint hyperlink;
2. base model and tokenizer;
3. prompt and scorer;
4. source tokens and realized compressed-state tokens;
5. raw, no-context, and compressed scores;
6. encoder/read latency and peak memory;
7. skipped or truncated examples.

## 8. Execution order

| order | method | first run | reason |
|---:|---|---|---|
| 1 | LCLM | RULER + LongBench, 4/8/16x | latest general long baseline |
| 2 | Semi-Dynamic | SQuAD/Hotpot, auto +32x | latest adaptive short-soft baseline |
| 3 | Activation Beacon | LongBench/InfiniteBench | released long-context checkpoint |
| 4 | AutoCompressor | Llama2-6k + OPT-30k | recurrent ancestor |
| 5 | ICAE | SQuAD/Hotpot | close frozen-decoder method |
| 6 | CCM | streaming + QA | short recurrent state |
| 7 | xRAG | Hotpot + compatible QA | one-token retrieval control |
| 8 | Cartridges | native LongHealth | released task-compatible cartridge |
| 9 | Gist/Cramming/500x | appendix / blocked | not main-table equivalents |

## 9. Local pinned repositories

| method | local path | commit |
|---|---|---|
| LCLM | `paper-a-official-baselines/lclm` | `e04ceb9` |
| Semi-Dynamic | `paper-a-official-baselines/semi-dynamic-context-compress` | `07fae01` |
| Cramming | `paper-a-official-baselines/hidden_capacity` | `c371c3f` |
| AutoCompressor | `paper-a-official-baselines/autocompressors` | `80352a4` |
| ICAE | `paper-a-official-baselines/icae` | `469a468` |
| CCM | `paper-a-official-baselines/context-memory` | `a89dd08` |
| Activation Beacon | `paper-a-official-baselines/flagembedding` | `7ed43d6` |
| xRAG | `paper-a-official-baselines/xrag` | `121fa41` |
| Cartridges | `paper-a-official-baselines/cartridges` | `ef34ba9` |
| Gist | `paper-a-official-baselines/gisting` | `3be0d06` |
| 500x | `paper-a-official-baselines/500xcompressor` | `ff454a1` |
| LLMLingua | `paper-a-official-baselines/llmlingua` | `e0e9d99` |
