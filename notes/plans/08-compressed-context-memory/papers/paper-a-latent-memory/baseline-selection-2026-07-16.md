# Paper A baseline selection — faithful originals only

> Date: 2026-07-16. Selection criteria, in order: (1) direct relevance to learned latent context or
> compression-failure detection; (2) official code/checkpoint compatibility with our setting; (3) contemporary
> citation/venue impact. Approximate citation counts are an OpenAlex snapshot and are used only for relative
> ranking; duplicate arXiv/proceedings records make exact counts unstable.
>
> **2026-07-19 update:** the execution-ready inventory and table placement now live in
> [`paper-baselines-v2-2026-07-19.md`](paper-baselines-v2-2026-07-19.md). It adds LCLM and
> Semi-Dynamic Context Compression as required 2026 baselines, and places Cramming 1568 as a capacity
> control rather than a deployment baseline.

## 1. Final policy

1. A method is called a paper baseline only when run through the authors' official code or released checkpoint.
2. Internal reimplementations (`baselines2025.py`, `gist-lite`, `cart-lite`) remain engineering smoke controls
   and are never labeled as the published method.
3. If the official method does not support our Qwen base, evaluate its released model on the same dataset and
   report performance normalized to that model's own uncompressed/raw path.
4. Main-table methods must receive the same source information and disclose realized memory/read-state tokens.
5. Architecture-specific methods are marked N/A rather than approximated on an incompatible base.

## 2. Selected compression baselines

| priority | method | approximate citation signal | closeness | official implementation | setting compatibility | placement |
|---:|---|---:|---|---|---|---|
| 1 | AutoCompressor | ~39 | **closest recurrent soft-summary ancestor** | [`princeton-nlp/AutoCompressors`](https://github.com/princeton-nlp/AutoCompressors) | released Llama-2-7B/OPT checkpoints; 6k/30k | main, external-own-base |
| 2 | ICAE | ~7 | **closest frozen-decoder soft-memory architecture** | [`getao/icae`](https://github.com/getao/icae) | official Llama-2 and Mistral checkpoints | main, external-own-base |
| 3 | CCM | ~1 | **closest recurrent compressed-memory mechanism** | [`snu-mllab/context-memory`](https://github.com/snu-mllab/context-memory) | official Llama/Llama-2/Mistral | main, external-own-base |
| 4 | Activation Beacon | ~2 | strongest progressive long-context latent/KV compressor | [`FlagOpen/FlagEmbedding`](https://github.com/FlagOpen/FlagEmbedding/tree/master/research/Long_LLM/activation_beacon) | official Llama-2/Mistral; newer code supports Llama-3/Qwen-2 | main, official native-base evaluation |
| 5 | Cartridges | new (2025) | frozen-base reusable compressed KV, strong deployment match | [`HazyResearch/cartridges`](https://github.com/HazyResearch/cartridges) | official Qwen examples via PEFT; standard-KV bases | main, Qwen3-8B |
| 6 | Gist Tokens | ~23 | foundational soft-token interface | [`jayelm/gisting`](https://github.com/jayelm/gisting) | official LLaMA-7B/FLAN-T5; short-prompt original setting | main lineage; external-own-base |
| 7 | xRAG | ~15 | frozen bridge to one soft retrieval token; direct overflow substrate | [`Hannibal046/xRAG`](https://github.com/Hannibal046/xRAG) | official frozen retriever/LLM pipeline | main on retrieval/QA rows |
| 8 | LLMLingua-2 | ~37 | strongest widely used task-agnostic text compressor | [`microsoft/LLMLingua`](https://github.com/microsoft/LLMLingua) | architecture-agnostic; already exact in harness | main |
| 9 | LongLLMLingua | ~86 | highest-impact long-context prompt-compression baseline | [`microsoft/LLMLingua`](https://github.com/microsoft/LLMLingua) | architecture-agnostic; external compressor LM | main |
| 10 | 500xCompressor | ~5 | extreme latent/KV compression | [`ZongqianLi/500xCompressor`](https://github.com/ZongqianLi/500xCompressor) | official Llama-3-8B LoRA/checkpoints | appendix/main extreme-ratio row |
| 11 | No Mean Feat / mean pooling | new (2025) | strongest simple latent baseline | official implementation when released; exact mean pooling is trivial | all embedding models | mandatory trivial baseline |

### Citation context

The highest contemporary citation signals in the broader compression family are LLMLingua (~115 aggregated
title matches), LongLLMLingua (~86), AutoCompressor (~39), LLMLingua-2 (~37), Gist (~23), and xRAG (~15).
The most relevant recurrent latent-memory papers (CCM, Cartridges, Activation Beacon, ICAE) are newer or more
niche and therefore have lower counts; they remain mandatory because relevance outweighs raw citation count.

## 3. Selected failure-detection and routing baselines

| priority | baseline | exact role | faithful implementation |
|---:|---|---|---|
| 1 | Belikova overflow probe | joint query–compressed-memory classifier for `raw correct, memory wrong` | reproduce paper's probe features/training split; calibration-only fitting |
| 2 | PoC performance predictor | predict raw-minus-compressed performance and choose a safe path | official PoC code when released; lightweight matched linear predictor as transparent ablation |
| 3 | Context Distillation first-token entropy | latent-memory activation/deactivation score | exact entropy/margin calculation on each method's released latent path |
| 4 | TARG | query-only entropy/margin/sample variance | official score definitions; no compression features |
| 5 | Self-Route sufficiency judgment | model judges whether compressed evidence suffices | exact prompt/protocol from Self-Route |
| 6 | random at matched coverage | routing sanity control | exact |
| 7 | oracle | per-item best of memory/raw | exact upper bound |

## 4. Statistical routing baselines

- empirical held-out threshold;
- Learn-then-Test with stated \(\epsilon,\delta\);
- Linear Expectation Constraint / selection-conditioned control when its official implementation is available;
- always-memory and always-raw.

Every route reports signed policy excess, accepted-set positive harm, raw-only/memory-only correctness,
coverage, latency, and expected cost.

## 5. Agent-system methods

TierMem, Slipstream, Selective Latent Thinking, TrustMem, SelfCompact, and D-Mem are closest system-level
comparisons, but they are not interchangeable static-QA compressors. They enter:

- the Related Work and design-comparison table;
- the long-horizon agent recovery-policy experiment;
- not the static compression-accuracy main table.

## 6. Internal replicas to demote

The following existing methods remain useful for debugging but are labeled **ADAPTED REPLICA**, not the
published baseline:

- `mem_embedding.gcm.methods.Gist`;
- `mem_embedding.gcm.methods.Cartridge`;
- `baselines2025.py::{ICAE,AOC,Beacon,X500,ComprExIT,LCC}`.

They may be reported only in an implementation-ablation appendix after the official results.

## 7. Main-table core

To keep the headline table readable, the core baseline columns are:

1. no context;
2. same-input feasible raw;
3. full+SFT;
4. raw window;
5. LLMLingua-2;
6. LongLLMLingua;
7. official Gist or ICAE (soft-memory representative);
8. official Activation Beacon or CCM (long/recurrent representative);
9. official Cartridges (reusable KV representative);
10. Compressor (w/o gate);
11. Compressor (w/ empirical gate);
12. Compressor (w/ formal gate).

AutoCompressor, xRAG, 500xCompressor, mean pooling, and additional official methods appear in the expanded
table or the task-compatible rows.

## 8. Compute rule

When an original method uses a different base model, report:

$$
\mathrm{retention}
=
\frac{\mathrm{score}_{\mathrm{compressed}}-\mathrm{score}_{\mathrm{noctx}}}
{\mathrm{score}_{\mathrm{raw}}-\mathrm{score}_{\mathrm{noctx}}}.
$$

Absolute scores are not compared across unrelated bases without this within-base normalization.
