# Experiment configuration & sampling disclosure

> Single source for **settings, train/test config, and how many test items each number uses.**
> **Paper rule:** the paper main table is run on the **FULL test set** of each benchmark. Any number computed on a **down-sampled subset must state the subset size `N`.** This file records both the current (exploratory, down-sampled) runs and the planned full-set main table.

## ⚠️ `full` is a TRUNCATED reference (authoritative definition)
In **every** result table across this project, the `full` column = `embed(ctx_ids[:, :MAXCTX])` — the context **truncated to the first MAXCTX tokens** (16k for long benches; 8k for QuALITY/MuSR; 4k for short QA). It is **NOT the entire document**; it is the uncompressed upper bound *within the budget*. Same MAXCTX bound applies to `window`, `tome`, `kvzip`, `knorm` (forward/KV methods) and IMP's `span` branch. **Only `rag` and our `auto`(chunk) scan the full untruncated document** (BM25 needs no forward).

**Which benches: truncated vs true-full** (does the raw doc exceed MAXCTX?):
| bench | MAXCTX | raw doc len | `full` = ? |
|---|--:|--:|---|
| **∞Bench-choice** | 16k | **median 131k** | **TRUNCATED — sees ~12%** |
| lb_hotpotqa / lb_* (LongBench-v1) | 16k | ~16k (median) | ≈ true full (≤1% overflow) |
| lb_triviaqa / lb_repobench | 12k | can exceed | **run @12k, mild truncation (disclosed)** |
| LongBench-v2 | 16k | ≤16k mostly | ≈ true full |
| RULER-NIAH@16k | 16k | =16k (synthetic) | true full |
| BABILong qa1-4k / qa*-16k | 4k/16k | fits | true full |
| QuALITY / QuALITY-hard / MuSR | 8k | ≤8k | true full |
| squad / hotpot / trivia / ms_marco | 4k | ≤4k | true full |

⇒ **Only `∞Bench-choice` has `full` materially truncated** (and lb_triviaqa/repobench mildly, at 12k). Everywhere else `full` = the whole document. This is why ∞Bench "retrieval ≫ full / IMP<RAG(pre-fix)" reflects **information access**, not compression quality (F39/F44). Result docs are labelled `full·trunc16k` and carry a banner pointing here.

---

## 1. Base model & method configs

| | value |
|---|---|
| **Base (frozen)** | `Qwen/Qwen3-8B` (dense, 32k window). Planned second base: a linear/GDN model (e.g. Qwen3.5-9B) — no KV cache. |
| **Attention impl.** | SDPA (fast) for attention-free methods; eager for attention-based KV-eviction (kvpress). |
| **Precision** | bf16. |
| **Scoring** | each benchmark's **native** metric: MC = length-normalized log-likelihood over option letters; extractive = EM/F1 over all references; abstractive/summ = ROUGE-L/BLEU over references; NIAH = gold-substring / exact-value. |

### IMP (Paper B, our method) — **version tag `IMP-v2.1.0`** (span-level, Mode A). All `imp`/`ft_*imp*`/`g_*imp*`/`spf_*` numbers are this version; token-level IMP-v2.0 is superseded (retrieval-only).
- **Mode A (evaluated): training-free.** No train config. Inference knobs: `GCM_IMP_SPAN` (1=token, 32=span default), `GCM_LL_RATE` (keep fraction, 0.5 default), `GCM_IMP_SIGNAL` (query/surprisal/both=default). Scorer = the base itself (query-relevance embedding-only; surprisal = one forward).
- **Mode B (planned): light-weight.** Train ONLY a tiny distilled router head (+ linear-base side-cache + brief LoRA); base frozen.

### Baselines (inference-time, no training unless noted)
- Window (`GCM_TXL_WINDOW=1024`) · KV-eviction via kvpress (`GCM_KV_RATIO=0.5`) · LLMLingua-2/orig/Long & Selective-Context (`GCM_LL_RATE=0.5`) · ToMe (`GCM_TOME_RATIO=0.5`) · BM25-RAG (`GCM_RAG_BUDGET=2048`).

### Learned soft-memory compressor (Paper A, `cr_*` wave)
K=256, depth=half, read-LoRA rank=32, VAE + KL=0.01, slot-recon λ=1.0, distill=0.5, 3000–6000 steps, LR 3e-4–5e-4, train benches = squad_v2. (Result: cannot compress extractive QA; see fact-base §12.3.)

---

## 2. Sampling disclosure — current runs are DOWN-SAMPLED

**All results reported so far (fact-base §8–§12, overview §3.1, matrix-facts F18–F26, slides 2026-07-w01) are on a DOWN-SAMPLED test subset:**

| wave | N (items/cell) | notes |
|---|---|---|
| KV fact-base (§8) | 48 (KV) / 64 (refs) | exploratory |
| KV-free family / expanded (§10–11) | 48 | exploratory |
| IMP main + ablations (§12) | 48 | exploratory |
| LongBench (real length, 32k) | **16** | 32k is slow; small subset |
| head-to-head (F18) | 64–96 | exploratory |

These are **for exploration and relative comparison**, not the paper's headline numbers.

---

## 3. Planned FULL-test main table (paper-grade)

**Rule: no down-sampling for the main table.** Each cell uses the complete public test/validation split; report the exact size.

| benchmark | full test size (to confirm at run time) | length regime |
|---|---|---|
| RULER (per length 4k/8k/16k/32k) | 500 synthetic / length (fixed, disclosed) | synthetic |
| numerical/categorical/coding NIAH | 500 synthetic each | synthetic |
| SQuAD v2 (val) | ~11,873 | short |
| HotpotQA (val, distractor) | ~7,405 | multi-hop |
| TriviaQA / MS-MARCO | full val | open QA |
| NarrativeQA / QuALITY(-hard) / MuSR / LoCoMo | full val | long-doc |
| **LongBench** (2wiki/hotpot/musique/multifield/qasper/narrative) | full (~150–200 each) | real long |
| **LongBench-v2** (hardest, MC, up to 2M words) | full (503) | real long, hard |
| **∞Bench** (retrieve/QA/math, >100k) | full per task | ultra-long |

- **Hardest-benchmark focus (per user):** LongBench-v2, ∞Bench, RULER-32k, (BABILong stretch).
- **Compute note:** full-set × ~10 methods × ~20 benchmarks is large; the main table uses a **core method set** (full, no_ctx, window, RAG, LLMLingua-2, kvzip, knorm, ToMe, IMP-span) at full N; **ablations may stay down-sampled with N disclosed.**

---

## 4. AS-LAUNCHED full main table (2026-07-08, running)

Launcher `gcm/experiments/fulltable_launch.sh`; base `Qwen/Qwen3-8B`; **free1 GPUs 0–3** (4×96 GB, idle — does **not** touch the exploratory grid on d1525/d1530); output `/mnt/persist/grid_fulltable/ft_<method>_<bench>.log`; **112 cells**.

**Method columns (9).** Each `run_baseline` cell also emits `no_ctx` (blind lower bound) and `full` (uncompressed upper bound), so those two columns come free:
| column | script | knob | family |
|---|---|---|---|
| no_ctx | (emitted by every cell) | — | blind lower bound |
| full | (emitted by every cell) | full context | upper bound |
| window | run_baseline | `GCM_TXL_WINDOW=1024` | truncate-to-window |
| rag | run_baseline | `GCM_RAG_BUDGET=2048` | BM25 retrieval |
| ll2 | run_baseline | LLMLingua-2 `GCM_LL_RATE=0.5` | prompt compression |
| tome | run_baseline | `GCM_TOME_RATIO=0.5` | token merging |
| **imp (ours)** | run_baseline | `GCM_IMP_SPAN=32 GCM_LL_RATE=0.5` | span importance-routing |
| kvzip | run_kvpress | `GCM_KV_RATIO=0.5` | KV eviction (reconstruction) |
| knorm | run_kvpress | `GCM_KV_RATIO=0.5` | KV eviction (key-norm) |

**Benchmarks & test config.** `nval=100000` ⇒ **FULL split** (generator caps to dataset size); scoring = native (MC=letter log-likelihood; extractive=EM/F1; abstractive=ROUGE-L; NIAH=exact-value):
| bench | N (test, verified) | n_chunks | max_ctx | metric | tier |
|---|---|---|---|---|---|
| **longbench_v2** | FULL = 503 | 16 | 32768 | MC-acc | hardest real |
| **infbench_choice** (∞Bench longbook_choice_eng) | FULL = 229 | 16 | 32768 | MC-acc | ultra-long (>100k) |
| lb_2wikimqa / hotpotqa / musique / qasper / narrativeqa | FULL = 200 each | 12 | 16384 | F1/ROUGE-L | real long |
| lb_multifieldqa | FULL = 150 | 12 | 16384 | F1 | real long |
| ruler_niah @16k (`ruler16k`) | 500 (fixed synthetic, disclosed) | 88 | 16384 | exact-value | synthetic retrieval |
| quality | FULL = 1595 | 8 | 8192 | MC-acc | literary MC |
| quality_hard | FULL = 813 | 8 | 8192 | MC-acc | literary MC (hard) |
| musr_mm | FULL = 90 | 8 | 8192 | MC-acc | reasoning MC |
| squad_v2 / hotpot_qa / trivia_qa / ms_marco | **500 = DISCLOSED subset** (full 5928 / 7405 / 16137 / 55578) | 8 | 4096 | F1/ROUGE-L | short sanity |

- **Full vs subset, stated explicitly:** every long-context headline bench is FULL; the four short-context sanity benches are a **disclosed N=500 subset** (they are not the paper's focus — long-context is).
- **Per-cell wall-clock cap:** 32k cells 4 h, 16k cells 2 h, others 1–1.5 h (so a slow ultra-long cell yields partial rather than blocking; re-runnable).
- **RULER note (finding, not a bug):** RULER is *generation*-scored, so at **32k the uncompressed `full` column OOMs even a 96 GB card** (single 32 GB attention alloc + model + KV) — a concrete instance of the long-context memory wall this paper is about. MC benches at 32k (LongBench-v2, ∞Bench) are single-forward log-likelihood and run fine at 32k. RULER is therefore reported at **16k** (`ruler16k`, retry on d1525 GPUs 0–3, same `grid_fulltable`). The 32k-full OOM is itself logged.

- **`lb_triviaqa` / `lb_repobench` context note (2026-07-11):** at 16 k the FULL-context reference forward OOMs a 96 GB card on *every* method (these two tasks have the longest single-item prefills; a 16 k forward ≈ 72 GB) — another instance of the memory wall. Both are therefore reported at **12 k** (`rf_q8_*_lb_triviaqa`, `rf_q8_*_lb_repobench`, re-run on free3 GPU 1, same `grid_impredesign` disk). All other LongBench tasks stay at 16 k. The 16 k-OOM is logged as `CELL_FAIL`.

## 5. Provenance
Exploratory numbers: `gt_*`/`mt_*` (grand grid, N=48/16), `kvf_*`/`be_*`/`spf_*`/`imp_*`. Full main table v1: `/mnt/persist/grid_fulltable/ft_*` (2026-07-08). **Reliable FULL-N redesign main comparison (2026-07-11): `/mnt/persist/grid_impredesign/rf_q8_*` — d1525×3 (15 core benches) + d1530×2 (10 new/short) + free3 (lb_triviaqa@12k), all 1-worker/GPU.** Keep-sweep: `rd_q8_*` (finishing on free1 GPUs 0–3). Each file states its own N.
