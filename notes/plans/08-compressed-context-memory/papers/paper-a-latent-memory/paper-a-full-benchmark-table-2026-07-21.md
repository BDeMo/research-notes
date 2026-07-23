# Paper A — full benchmark and baseline table

> Snapshot: 2026-07-22 20:46 PT
> Rule: completed values are reported numerically; incomplete experiments are labeled by status and are not
> filled with estimated results. SQuAD-v2 is retained only as an exact-text diagnostic and is excluded from
> the headline tables.

## 1. Headline compressor results

Benchmarks are columns and methods are rows. Raw and SFT are references, not compressed-path competitors.
Bold marks the best completed compressed path.

| base | method | role | QuALITY | BFCL | HotpotQA |
|---|---|---|---:|---:|---:|
| Qwen3-8B | Raw | reference | 7.2% | 92.4% | 53.7% |
|  | SFT | reference | 81.7 ± 1.7% | 95.4 ± 0.3% | 68.8 ± 0.6% |
|  | Window | control | 15.7% | 55.7% | 26.2% |
|  | LLMLingua | baseline | 14.3% | 70.3% | 22.1% |
|  | Compressor (w/o gate) | ours | **54.4 ± 0.2%** | **72.3 ± 0.5%** | **28.9 ± 0.2%** |
| Qwen3.5-9B | Raw | reference | 7.1% | 84.5% | 53.9% |
|  | SFT | reference | 85.0 ± 0.4% | 94.9 ± 1.0% | 71.7 ± 0.6% |
|  | Window | control | 16.7% | 52.8% | 24.8% |
|  | LLMLingua | baseline | 20.3% | 60.8% | 28.9% |
|  | Compressor (w/o gate) | ours | **51.5 ± 1.7%** | **72.0 ± 0.8%** | **30.5 ± 0.3%** |

`Window` is a budget-matched control. LLMLingua is currently the only completed published compression
baseline in the same-base table. Official soft-memory methods use different released backbones and therefore
belong in the native-base table below.

## 2. Shared-backbone routing

`Compressor (w/ gate)` combines the compressor with bounded raw fallback. `FB AUC` measures whether low compressor confidence ranks
raw-better examples. All values are percentages or percentage points.

| base | metric | QuALITY | BFCL | HotpotQA |
|---|---|---:|---:|---:|
| Qwen3-8B | Compressor (w/o gate) | 54.4% | 72.3% | 28.9% |
|  | Compressor (w/ gate) | **54.6%** | **88.5%** | **50.9%** |
|  | Gain | +0.2 pp | +16.2 pp | +22.0 pp |
|  | FB AUC | 57.2% | 82.8% | 63.9% |
|  | FB rate | 0.2% | 46.6% | 68.3% |
|  | Δ Raw | +47.4 pp | -3.5 pp | -2.4 pp |
| Qwen3.5-9B | Compressor (w/o gate) | **51.5%** | 72.0% | 30.5% |
|  | Compressor (w/ gate) | 51.4% | **80.5%** | **51.7%** |
|  | Gain | -0.1 pp | +8.5 pp | +21.2 pp |
|  | FB AUC | 54.9% | 84.1% | 67.4% |
|  | FB rate | 0.3% | 22.0% | 52.1% |
|  | Δ Raw | +44.4 pp | -3.8 pp | -2.2 pp |

## 3. Complete benchmark inventory

| panel | benchmark | task type | source adapter | paper role | current state |
|---|---|---|---|---|---|
| Core | QuALITY | long-document multiple choice | QuALITY | headline competence | complete |
| Core | BFCL-live-multiple | tool selection | BFCL | headline tool use | complete |
| Core | HotpotQA | multi-hop QA | HotpotQA | raw-evidence boundary | complete |
| Diagnostic | SQuAD-v2 | short-passage extractive QA | SQuAD | exact-text diagnostic only | complete; removed from headline |
| Long | LongBench-v2 | mixed long-context reasoning | QuALITY | main long-context transfer | partial |
| Long | InfiniteBench-choice | very-long multiple choice | QuALITY | main very-long transfer | partial; K32 repairs remain |
| Long | MultiFieldQA | long multi-document QA | SQuAD | extractive transfer | partial |
| Long | Qasper | long scientific-document QA | SQuAD | document QA transfer | partial |
| Long | LongBench HotpotQA | long multi-hop QA | HotpotQA | multi-hop transfer | partial |
| Long | 2WikiMQA | multi-hop QA | HotpotQA | multi-hop transfer | partial |
| Long | MuSiQue | compositional multi-hop QA | HotpotQA | multi-hop transfer | partial |
| Long | NarrativeQA | long narrative QA | NarrativeQA | summarizable-memory test | partial |
| Long | BABILong QA1 | recurrent synthetic QA | HotpotQA | recurrence test | partial |
| Long | BABILong QA2 | recurrent synthetic QA | HotpotQA | recurrence test | partial |
| Long | BABILong QA3 | recurrent synthetic QA | HotpotQA | recurrence test | partial |
| Controlled | RULER NIAH 4k | exact retrieval | none | length boundary | pending |
| Controlled | RULER NIAH 8k | exact retrieval | none | length boundary | pending |
| Controlled | RULER NIAH 16k | exact retrieval | none | length boundary | pending |
| Controlled | RULER NIAH 32k | exact retrieval | none | length boundary | pending |
| Blocked | NoLiMa | literal-match-resistant retrieval | none | robustness extension | dataset access blocked |
| Missing loader | HELMET | long-context suite | none | broader external validity | loader not implemented |
| Missing loader | LongMemEval | conversational memory | none | agent-memory extension | loader not implemented |

Current stage totals:

| stage | done | running | failed | pending | total |
|---|---:|---:|---:|---:|---:|
| Core main grid | 88 | 0 | 0 | 0 | 88 |
| Transfer adapters | 41 | 0 | 2 | 0 | 43 |
| Real long-context | 103 | 0 | 3 | 0 | 106 |
| Seven-model generality | 12 | 3 | 0 | 27 | 42 |
| Budget and length | 2 | 1 | 0 | 20 | 23 |
| Mechanism ablation | 0 | 0 | 0 | 36 | 36 |
| Reproducibility | 3 | 0 | 0 | 0 | 3 |
| SFT reaudit | 6 | 0 | 0 | 0 | 6 |
| Official-baseline local cells | 0 | 0 | 0 | 52 | 52 |

## 4. Full baseline inventory

| method | type | official base | benchmark scope | comparison rule | current state |
|---|---|---|---|---|---|
| Raw | full-token reference | each reader base | all compatible tasks | reference only | complete on core |
| SFT-LoRA | full-cost adaptation reference | each reader base | core + configured long targets | reference only | core complete |
| Window | hard token window | each reader base | core + RULER | matched reader budget | core complete |
| LLMLingua-2 | hard token compression | each reader base | core | matched reader budget | core complete |
| LongLLMLingua | question-aware hard compression | each reader base | core | matched reader budget | incompatible-cache rerun needed |
| Original LLMLingua | perplexity compression | each reader base | core | matched reader budget | incompatible-cache rerun needed |
| Mean pooling | latent control | each reader base | core | matched latent length | pending |
| LCLM | general soft tokens | Qwen3 encoder + decoder | RULER, LongBench, LongHealth | within-native-base retention | cloned; not executed |
| Semi-Dynamic | adaptive soft tokens | Qwen3 0.6B / 4B | Hotpot and short QA | within-native-base retention | cloned; not executed |
| Activation Beacon | condensed activations | Qwen2-7B-Instruct | LongBench, InfiniteBench | within-native-base retention | pending |
| AutoCompressor | recurrent soft summary | Llama-2 / OPT | QuALITY, QA, long context | within-native-base retention | pending |
| ICAE | soft memory slots | Mistral-7B | QA within native limit | within-native-base retention | pending |
| CCM | recurrent KV memory | Llama-2 / Mistral | streaming, chat, ICL | native protocol | pending |
| xRAG | retrieval soft token | Mistral / Mixtral | Hotpot, Qasper, MultiFieldQA | retrieval-specific row | pending |
| Cartridges | reusable KV prefix | Llama-3.2-3B | LongHealth | native corpus protocol | pending |
| Gist Tokens | gist KV token | LLaMA-1 / FLAN-T5 | short instructions | appendix only | access-limited |
| Cramming 1568 | optimized input vector | Llama-3.1-8B | reconstruction | capacity control only | appendix |
| 500xCompressor | layerwise KV slots | Llama-3-8B | short spans | unavailable row | weights blocked |

## 5. Manuscript table inventory

Every configured experiment now has a manuscript table. Missing values are explicit `TBD` placeholders;
unsupported native interfaces use `—`.

1. **Experiment map and audit:** E0–E9 coverage, data checks, output checks, and invalid-cell exclusions.
2. **Core and routing:** matched main comparison, independent reproduction, and held-out gate results.
3. **Source readiness:** per-base source adapters and K32 repair state.
4. **Long-context transfer:** two-main-base gated result, six-task LongBench breakdown, and seven-base transfer.
5. **Generality:** fixed K across all seven bases, with unfinished rows left as `TBD`.
6. **Budget and length:** memory/raw budget sweep and RULER 4k–32k boundary.
7. **Mechanism:** joint loss, distillation, reconstruction, recurrence, and memory-size ablations.
8. **Measured cost:** source/state tokens, encoder/read/fallback time, expected route cost, and peak memory.
9. **Official baselines:** native-base benchmark matrix with within-base references.
10. **SQuAD-v2 diagnostic:** full completed short-passage results in the appendix, outside headline claims.

The exact fill-ready values and placeholders are maintained in
[`paper-a-all-benchmarks-skeleton-2026-07-21.md`](paper-a-all-benchmarks-skeleton-2026-07-21.md).
