# configs/ — exact code + run configs (reproducibility archive)

Everything needed to reproduce the Paper-B (`IMP-v2.1.0`) results, saved verbatim. The live code runs from
`gcm/experiments/` on the shared pod FS (`/mnt/persist/gcm`); these are byte-copies for traceability.

## Contents
| file | what |
|---|---|
| `IMP-v2.1.0.code.py` | **exact `_imp` scoring+selection code** (verbatim from `run_baseline.py`, `MODE==imp`) |
| `fulltable_launch.sh` | FULL-test main-table grid (112 cells, 7 methods × 16 benches) → `main-table-fulltest.md` |
| `fulltable_retry.sh` | OOM-repair pass (RULER→16k, `rag_lb_musique`) |
| `gen_launch.sh` | generality grid (per-model, 5 methods × 5 benches, N=100) → `generality-model-matrix.md` |
| `keep_ablation.sh` | keep-rate ablation, all 7 methods × 4 benches × keep{0.1,0.25,0.5,0.75}, N=100 → `keep-ablation-results.md` |
| `_patch_hardbench.py` | adds LongBench-v2 + ∞Bench loaders to the shared harness |
| `_patch_cmgrca.py` | re-registers the `cmg_rca` bench (demo) |
| `_count_fullN.py` | verifies the true FULL-split N per bench |

## Method config — `IMP-v2.1.0`
```
GCM_BASELINE=imp  GCM_IMP_SPAN=32  GCM_LL_RATE=0.5  GCM_IMP_SIGNAL=both
```
- span-level (SPAN=32), keep p=0.5, signals = z(query-relevance)+z(surprisal). Mode A (training-free, frozen base).
- token-level `IMP-v2.0` = `GCM_IMP_SPAN=1` (superseded, retrieval-only). Version registry: `decisions-2026-06-24.md` D28.

## Shared run environment (all cells)
```
Base            Qwen/Qwen3-8B (bf16, frozen)   # linear arm: Qwen/Qwen3.5-9B (GDN, torch fallback)
PYTHONPATH      /mnt/persist/gcm/src:/mnt/persist/v17/{svc,mem-embedding,llm-infra}/src
HF_HOME         /mnt/persist/hf-cache
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   TRITON_CACHE_DIR=/mnt/persist/.triton
scoring         MC=letter log-likelihood (mc_loglik) · gen=score_gen (EM/F1/ROUGE-L) + gold-substring fallback
```

## Baseline configs (same harness)
```
window (txl)   GCM_BASELINE=txl        GCM_TXL_WINDOW=1024
RAG (BM25)     GCM_BASELINE=rag        GCM_RAG_BUDGET=2048
LLMLingua-2    GCM_BASELINE=llmlingua  GCM_LL_RATE=0.5      (microsoft/llmlingua-2-xlm-roberta-large-meetingbank)
ToMe           GCM_BASELINE=tome       GCM_TOME_RATIO=0.5
kvzip / knorm  run_kvpress.py  GCM_KVPRESS=kvzip|knorm  GCM_KV_RATIO=0.5   (quadratic bases only)
```

## Benchmarks & N — see `../experiment-config-and-sampling.md`
Long-context headline = FULL split (LongBench-v2 503, ∞Bench-choice 229, LongBench-v1 150–200, QuALITY 1595,
QuALITY-hard 813, MuSR 90); RULER-NIAH 500 synthetic; short-context sanity (squad/hotpot/trivia/ms_marco) = **N=500 disclosed subset**.

## Provenance (result logs, on pod)
```
Full main table   /mnt/persist/grid_fulltable/ft_<method>_<bench>.log        (+ ruler16k retry)
Generality        /mnt/persist/grid_generality/g_<model>_<method>_<bench>.log
Exploratory       /mnt/persist/grid_main/{gt,mt,gtab}_*.log  (N=48, superseded by full table)
Per-cell JSON     /mnt/persist/gcm/out/<model>/<cell>.json
```
