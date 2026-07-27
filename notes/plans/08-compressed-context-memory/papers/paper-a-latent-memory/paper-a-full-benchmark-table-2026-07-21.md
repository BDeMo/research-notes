# Paper A — full benchmark and baseline table

> Snapshot: 2026-07-26 17:05 PT
> Rule: completed values are reported numerically; incomplete experiments are labeled by status and are not
> filled with estimated results. SQuAD-v2 is retained only as an exact-text diagnostic and is excluded from
> the headline tables.
>
> **Storage boundary:** this file owns benchmark maintenance, incomplete cells, invalidation history, and
> rerun state. The Overleaf repository contains only validated paper results and submission text.

> [!WARNING]
> **Integrity quarantine (2026-07-22).** `⚠ INVALID` marks direct QuALITY results and
> results from QuALITY-trained adapters. The loader treated zero-based labels as one-based,
> dropped all A-labelled examples, and shifted the remaining targets. LongBench-v2 and
> InfiniteBench-choice compressed-path values are affected because their source adapter is
> trained on QuALITY. Hotpot→BABILong is separately marked as an observed transfer collapse.

## 1. Headline compressor results

Benchmarks are columns and methods are rows. Raw and SFT are references, not compressed-path competitors.
Bold marks the best completed compressed path.

| base | method | role | QuALITY | BFCL | HotpotQA |
|---|---|---|---:|---:|---:|
| Qwen3-8B | no context | reference | 43.2% corrected | 1.3% | 19.8% |
| Qwen3-8B | Raw | reference | 73.2% corrected | 92.4% | 53.7% |
|  | SFT | reference | 79.1 ± 1.1% corrected | 95.4 ± 0.3% | 68.8 ± 0.6% |
|  | Window | control | 48.5% corrected | 55.7% | 26.2% |
|  | BM25 retrieval | baseline | **54.7%** | 78.8% | **37.3%** |
|  | LLMLingua-2 | baseline | 45.3% | **82.6%** | 26.3% |
|  | LongLLMLingua | baseline | TBD | TBD | TBD |
|  | original LLMLingua | baseline | 45.7% | 60.1% | TBD |
|  | mean pooling | latent control | 32.4% | 36.1% | 14.0% |
|  | Compressor (w/o gate) | ours | 48.1 ± 1.5% corrected | 72.3 ± 0.5% | 28.9 ± 0.2% |
| Qwen3.5-9B | no context | reference | 44.4% corrected | 1.3% | 26.7% |
| Qwen3.5-9B | Raw | reference | 81.5% corrected | 84.5% | 53.9% |
|  | SFT | reference | 82.5 ± 0.7% corrected | 94.9 ± 1.0% | 71.7 ± 0.6% |
|  | Window | control | 50.6% corrected | 52.8% | 24.8% |
|  | BM25 retrieval | baseline | **56.6%** | **76.9%** | TBD |
|  | LLMLingua-2 | baseline | 45.9% | 73.4% | TBD |
|  | LongLLMLingua | baseline | TBD | TBD | TBD |
|  | original LLMLingua | baseline | TBD | TBD | TBD |
|  | mean pooling | latent control | 44.5% | 33.9% | 22.8% |
|  | Compressor (w/o gate) | ours | 45.8 ± 2.3% corrected | 72.0 ± 0.8% | **30.5 ± 0.3%** |

`Window` is a budget-matched control. The corrected same-base stage contains 30 cells:
LLMLingua-2, LongLLMLingua, original LLMLingua, BM25 retrieval, and mean pooling on both bases and all three headline tasks. Eighteen cells are complete, three are running, and nine are queued under the source-and-budget-v2 contract. Unfinished cells remain `TBD`. Official soft-memory methods use different released backbones and therefore
belong in the native-base table below.

## 2. Shared-backbone routing

`Compressor (w/ gate)` combines the compressor with bounded raw fallback. `FB AUC` measures whether low compressor confidence ranks
raw-better examples. All values are percentages or percentage points.

| base | metric | QuALITY | BFCL | HotpotQA |
|---|---|---:|---:|---:|
| Qwen3-8B | Compressor (w/o gate) | ~~54.4%~~ ⚠ INVALID | 72.3% | 28.9% |
|  | Compressor (w/ gate) | ~~54.6%~~ ⚠ INVALID | **88.5%** | **50.9%** |
|  | Gain | ~~+0.2 pp~~ ⚠ INVALID | +16.2 pp | +22.0 pp |
|  | FB AUC | ~~57.2%~~ ⚠ INVALID | 82.8% | 63.9% |
|  | FB rate | ~~0.2%~~ ⚠ INVALID | 46.6% | 68.3% |
|  | Δ Raw | ~~+47.4 pp~~ ⚠ INVALID | -3.5 pp | -2.4 pp |
| Qwen3.5-9B | Compressor (w/o gate) | ~~51.5%~~ ⚠ INVALID | 72.0% | 30.5% |
|  | Compressor (w/ gate) | ~~51.4%~~ ⚠ INVALID | **80.5%** | **51.7%** |
|  | Gain | ~~-0.1 pp~~ ⚠ INVALID | +8.5 pp | +21.2 pp |
|  | FB AUC | ~~54.9%~~ ⚠ INVALID | 84.1% | 67.4% |
|  | FB rate | ~~0.3%~~ ⚠ INVALID | 22.0% | 52.1% |
|  | Δ Raw | ~~+44.4 pp~~ ⚠ INVALID | -3.8 pp | -2.2 pp |

## 3. Complete benchmark inventory

| panel | benchmark | task type | source adapter | paper role | current state |
|---|---|---|---|---|---|
| Core | QuALITY | long-document multiple choice | QuALITY | headline competence | 12/24 corrected; old cells quarantined |
| Core | BFCL-live-multiple | tool selection | BFCL | headline tool use | complete |
| Core | HotpotQA | multi-hop QA | HotpotQA | raw-evidence boundary | complete |
| Diagnostic | SQuAD-v2 | short-passage extractive QA | SQuAD | exact-text diagnostic only | complete; removed from headline |
| Long | LongBench-v2 | mixed long-context reasoning | QuALITY | main long-context transfer | ⚠ compressed paths invalid |
| Long | InfiniteBench-choice | very-long multiple choice | QuALITY | main very-long transfer | ⚠ compressed paths invalid |
| Long | MultiFieldQA | long multi-document QA | SQuAD | extractive transfer | partial |
| Long | Qasper | long scientific-document QA | SQuAD | document QA transfer | partial |
| Long | LongBench HotpotQA | long multi-hop QA | HotpotQA | multi-hop transfer | partial |
| Long | 2WikiMQA | multi-hop QA | HotpotQA | multi-hop transfer | partial |
| Long | MuSiQue | compositional multi-hop QA | HotpotQA | multi-hop transfer | partial |
| Long | NarrativeQA | long narrative QA | NarrativeQA | summarizable-memory test | partial |
| Long | BABILong QA1 | recurrent synthetic QA | HotpotQA | recurrence test | ⚠ COLLAPSE; labels unaffected |
| Long | BABILong QA2 | recurrent synthetic QA | HotpotQA | recurrence test | ⚠ COLLAPSE; labels unaffected |
| Long | BABILong QA3 | recurrent synthetic QA | HotpotQA | recurrence test | ⚠ COLLAPSE; labels unaffected |
| Optional | RULER NIAH | exact retrieval | none | stress test only | excluded after repeated 96 GiB OOM |
| Blocked | NoLiMa | literal-match-resistant retrieval | none | robustness extension | dataset access blocked |
| Missing loader | HELMET | long-context suite | none | broader external validity | loader not implemented |
| Missing loader | LongMemEval | conversational memory | none | agent-memory extension | loader not implemented |

Current stage totals:

| stage | done | running | failed | pending | total |
|---|---:|---:|---:|---:|---:|
| Core main grid | 60 unaffected + 12 corrected | 3 corrected | 28 old invalid | 9 corrected queued | 112 incl. replacement |
| Transfer adapters | 21 valid | 0 | 7 QuALITY invalid | 7 corrected | 28 paper-critical |
| Real long-context | 42 usable | 0 | 14 invalid + 21 collapse | 14 corrected | 77 paper-critical |
| Seven-model BFCL generality | 17 valid | 0 | 4 technical | 0 | 21 |
| Capacity sensitivity | 36 ablation cells | 0 | 0 | 0 | folded into E7 |
| Mechanism ablation | 36 | 0 | 0 | 0 | 36 |
| Reproducibility | 0 valid | 0 | 3 invalid | 0 | 3 |
| SFT reaudit | 0 valid | 0 | 6 invalid | 0 | 6 |
| Official baseline packages | 0 | 0 | 0 | 8 | 8 |

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
6. **Capacity:** K=64/K=256 sensitivity is included in the mechanism table.
7. **Mechanism:** joint loss, distillation, reconstruction, recurrence, and memory-size ablations (complete).
8. **Measured cost:** source/state tokens, encoder/read/fallback time, expected route cost, and peak memory.
9. **Official baselines:** native-base benchmark matrix with within-base references.
10. **SQuAD-v2 diagnostic:** full completed short-passage results in the appendix, outside headline claims.

The exact fill-ready values and placeholders are maintained in
[`paper-a-all-benchmarks-skeleton-2026-07-21.md`](paper-a-all-benchmarks-skeleton-2026-07-21.md).
