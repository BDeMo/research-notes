# v1.7.4 — Making GCM work on LONG context (length-adaptive memory)

> **Status: implemented + smoke-validated + 46 jobs queued (~46 GPU-h), 2026-06-14.** This is the improvement
> increment on top of [v1.7.3](../results-v1.7.3/results-v1.7.3.md). Settings: [experimental-setup.md](../results-v1.7.3/experimental-setup.md) ·
> mechanism: [gcm-lora-mechanism.md](../results-v1.7.3/gcm-lora-mechanism.md). Numbers fill in as the queue drains.

## 1. The problem (diagnosed in v1.7.3)
GCM (and every compressor we tried) is **capacity-bound**: it only beats trivial truncation when the context is
several times larger than the memory budget K, and it **fails on genuinely long context**:

| bench | ctx tok | ctx/K (K=64) | GCM compress | full | verdict |
|---|---|---|---|---|---|
| bfcl_live_multiple | ~182 | 2.8× | **0.72** | 0.91 | works (only win over `trunc`) |
| bfcl_multiple | >64 | ~3× | 0.15 | 0.88 | weak |
| bfcl_parallel_multiple | >64 | ~3× | 0.18 | 0.77 | weak |
| toolace | ~580 | **9×** | 0.14 | 0.95 | **fails** |
| rca_openrca | long | ≫ | 0.10 | ~0.45 | **fails** |

Root cause (from the code): the encoder is a **fixed-K, single-pass, Perceiver-style latent bottleneck** —
`[ctx ; query ; K mem]` in one causal pass, K learned slots pool the *entire* context. As ctx grows, K stays
fixed → the compression ratio explodes and a single global pool of 64 slots cannot represent a 580-token tool
list. **Naively raising K** (the K-sweep) trades away the compression and still pools globally.

## 2. The improvement (non-naive): length-adaptive, fixed-ratio chunked memory
Instead of one global pool, **split the context into chunks of `C` tokens and give each chunk its own `k`
memory slots**, then concatenate → `M` of length `S·k` where `S=⌈Lc/C⌉`.

- **Compression ratio is fixed** (`C/k`, e.g. 64/16 = 4×) but the **budget scales with length** → no global
  bottleneck. Short ctx (≤C) ⇒ identical to the current single-pass GCM (graceful degradation).
- **Per-chunk locality**: each chunk is encoded independently, so a tool signature in chunk 7 is not crushed
  by chunks 0–6 competing for the same 64 slots.
- **Query-relevance routing (optional, `keep=m`)**: score each chunk-memory against the query and keep the
  top-`m` → bounded budget (`m·k`) AND focused on the relevant region = **soft retrieval**, the right
  abstraction for tool selection (only the relevant tool matters, regardless of how many are listed).

This is distinct from "just raise K": the memory is **length-adaptive** and **local**, not a bigger global pool.

### What this adjudicates (4 hypotheses)
- **H1 capacity-bound** — `k256` (bigger fixed budget) vs `k64`: does more budget alone help?
- **H2 locality** — `ch16a` (~`S·16` tokens) vs `k256` (256 tokens, single pool) at ~matched budget: does
  *per-chunk* memory beat a *single big pool*? (Isolates locality from raw budget.)
- **H3 length-adaptive (headline)** — the **capacity frontier**: `compress` vs `n_ctx_tokens`. Prediction:
  `k64` declines with length, `ch16a` stays flat.
- **H4 retrieval** — `ch16r` (keep top-8 chunks) vs `ch16a` (keep all): does routing preserve accuracy at a
  bounded budget on the longest benches?

## 3. Implementation (code, all synced to the pod + smoke-validated)
| file | change |
|---|---|
| `svc/compressor.py` | `encode_chunked(ctx, query, chunk_size, keep)` — per-chunk encode + concat; refactored `_encode_one` / `_readout` helpers; query-relevance routing (cosine of chunk-mean vs query-mean). |
| `svc/method.py` | `GCM._enc()` dispatch (chunked when `chunk_size>0`); `mem_len(n_ctx)` for the chunk-aware **cost** column; **gradient checkpointing** auto-enabled for chunked training (S encoder-graphs fit one GPU); shape-guarded dev-anchor for routing. |
| `mem_embedding/gcm/{methods.py,harness.py}` | `--chunk-size`, `--chunk-keep` args → `TrainCfg`; cost uses `mem_len`. |

**Smoke (toolace, ctx 384 → 6 chunks, keep-4):** trains end-to-end, reconstruction loss 45→4.8 (memory is
learning), eval emits valid tool calls, cost = `keep·k + query` (chunk-aware). No OOM with checkpointing.

## 4. Experiment design (~73 jobs, ~70 GPU-h) — see `queue_longctx.py`
Base Qwen3-8B; combo recipe (enc-16, dec-2, LoRA-32, lr1e-4, 3000 steps, n=384); **enc-16 for ALL configs**
(same encoder capacity); `--methods no_ctx,full_ctx,trunc,gcm` in every run (baselines for free).

| config | memory | what it tests |
|---|---|---|
| `k64` | 64, single pass | baseline (current GCM) |
| `k256` | 256, single pass | H1 naive bigger budget |
| `ch16a` | 16/chunk, keep-all (`S·16`) | **ours** — length-adaptive (H2/H3) |
| `ch16r` | 16/chunk, keep-8 (≤128) | ours+routing (H4) |
| `ch32a`,`ch16k4` | ablation | capacity (k=32) / tighter routing (keep-4) |

**Domain coverage** (the improvement must be domain-general, not just tool/ops):
- **tool/ops (long):** hermes, bfcl_multiple, bfcl_parallel_multiple, bfcl_live_multiple, toolace, rca_openrca — × 4 configs = 24.
- **other domains (long):** narrativeqa, quality (literary), hotpot_qa (wiki multi-hop), musr_mm (synthetic reasoning) — × 4 configs = 16.
- **other domains (short QA, gate generality):** squad_v2, trivia_qa — × {k64, ch16a} = 4.
- **ruler_niah (synthetic NIAH, the cleanest frontier):** fix the task, sweep length via `n_chunks∈{4,12,24}` (max-ctx 2048) — k64 vs ch16a (+ k256 at the longest) = 7.
- **Multi-seed CIs:** {live_multiple, toolace, rca} × {k64, ch16a} × seeds {43,44,45} = 18.
- **Ablation:** {toolace, rca} × {ch32a, ch16k4} = 4.

→ The **headline figure** is now twofold: (a) compress vs natural ctx length *across* benches/domains, and (b) the
**controlled** `ruler_niah` length sweep (same task, only length changes) — the decisive test of H3.

## 5. What we will get (deliverables)
1. **Capacity-frontier figure** — `compress` vs `n_ctx_tokens`, one line per config → the headline "chunked stays flat, fixed-K declines" (H3).
2. **Long-context table** — per bench: no_ctx / full / trunc / k64 / k256 / ch16a / ch16r (+ gate AUROC, gАcc_cv, **fallback%**, **cost/compression-ratio**).
3. **Capacity-vs-locality** — k256 vs ch16a at matched budget (H2).
4. **Multi-seed CIs** — mean±std on the 3 headline benches (k64 vs ch16a) → is the gain real?
5. **Gate + cost** — does the do-no-harm gate still hold, and what is the realized compute saving at each length?

## 6. Results (updated 2026-06-14 22:30; long benches landed)
> ⚠️ Still partial (toolace-ch16a + ruler frontier + matched k64/k256 reruns running), but the **decisive long
> benches landed** and the verdict is clear: **chunked memory does NOT rescue long context** — it ties or loses
> to trivial `trunc` and stays far below `full`.

### 6.1 ★ v1.7.3 fixed-K vs v1.7.4 length-adaptive (the headline comparison)
`compress` accuracy. **fixed-K** = v1.7.3 GCM combo (K=64, single-pass; §2.1/§2.4a). **len-adapt** = v1.7.4 `ch16a`
(16 slots/chunk, budget = ⌈ctx/64⌉·16, keep-all). `full`/`trunc` for reference. Δ = len-adapt − fixed-K.
| bench | ctx | full | trunc | **v1.7.3 fixed-K** (K64) | **v1.7.4 len-adapt** (ch16a) | Δ |
|---|---|---|---|---|---|---|
| bfcl_parallel_multiple | 83 | 0.767 | 0.300 | 0.117 | 0.133 | +0.02 |
| hermes | 84 | 0.938 | 0.958 | 0.500 | 0.479 | −0.02 |
| bfcl_multiple | 97 | 0.883 | 0.333 | 0.150 | 0.133 | −0.02 |
| bfcl_live_multiple | 182 | 0.906 | 0.375 | **0.719** | 0.708 (3-seed 0.63±0.02) | −0.01 |
| toolace | 579 | 0.948 | 0.010 | 0.140 | *(running)* | — |
| rca_openrca | 1019 | 0.448 | 0.135 | 0.104 | 0.094 | −0.01 |
| hotpot_qa † | 150 | 0.320 | 0.229 | *(k64 rerun pending)* | 0.307 | — |
| musr_mm † | 1008 | 0.582 | 0.552 | *(pending)* | 0.552 | — |
| narrativeqa † | 1024 | 0.136 | 0.162 | *(pending)* | 0.129 | — |
| quality † | 1024 | 0.146 | 0.250 | *(pending)* | 0.219 | — |

† new domains added in v1.7.4 (no v1.7.3 fixed-K; same-eval k64 reruns queued). **Δ within ±0.02 on every shared bench.**

→ **Verdict: length-adaptive does NOT beat fixed-K.** Chunking gives the *same* compress accuracy as the v1.7.3
single-pass K64 (±0.02) — including on the long benches (rca 0.094 vs 0.104; toolace pending) where it was *supposed*
to help. And neither version beats `full`, nor (mostly) trivial `trunc`. Combined with §8 (no scaling) and §10 (OOD
probe), the length-adaptive memory **does not make compression work on long context** — compression is capacity-bound,
and the contribution stays the **do-no-harm gate**.

**Mechanism note (why ch16a is not yet ahead):** `ch16a` uses **16 slots/chunk**, so total budget is `S·16`
where `S=⌈ctx/64⌉`. For **ctx < 256** (`S≤4`) that is **fewer than k64's 64 slots** → ch16a is *under-provisioned*
on the medium benches above, exactly where it trails. The crossover is **ctx ≳ 256** (`S·16 > 64`): toolace
(580→9 chunks→144 slots), rca, narrativeqa, ruler-nc24. **Those are the benches still training** — the headline
test is pending, not yet decided.

### 6.2 The do-no-harm gate holds on the chunked method ✅
| ch16a bench | gate AUROC | gAcc_cv | fallback% |
|---|---|---|---|
| bfcl_live_multiple | 0.700 | 0.938 (≈full) | 99% |
| hermes | 0.702 | 0.938 (≈full) | 95% |
| bfcl_multiple | 0.646 | 0.883 (≈full) | 98% |
| bfcl_parallel_multiple | 0.603 | 0.750 (≈full) | 98% |
| hotpot_qa | 0.526 | 0.331 (≈full) | 52% |
→ the compressor-agnostic gate still recovers ≈full via fallback on the chunked memory (robustness layer is method-independent).

### 6.3 Status of related runs
- **args-aware (R3): INVALID, re-queued.** `--bfcl-full-call` changed the prompt but generation was capped at
  `max-new-tokens=16` — too short to emit a full `name(args...)` call, so strict `tool_call_acc` collapsed even
  for full_ctx (0.10–0.22). Re-queued 6 jobs with `max-new-tokens=48` (`zz_args_*`, run last). Not a verdict.
- K-sweep big-K on long ctx (`k_rca_K512`, `k_toolace_K512`) OOM'd — expected; the naive single-pool budget
  does not even fit at K=512 on long context (an argument *for* chunking).

## 7. Code review (2026-06-14) + added baselines
**Data hygiene (the v1.7-leakage class) — re-verified CLEAN.** Train(seed42/"train") vs eval(seed43/"validation")
content overlap on every long-ctx bench: hotpot_qa/narrativeqa/quality/ruler_niah/squad_v2/trivia_qa/toolace **0%**,
bfcl_multiple 2%, **musr_mm 0%**. (A first pass flagged musr 12%, but that was a *measurement artifact* — I keyed on
`item.context`, which is empty for musr; the real text is in `item.chunks`, the same field `_hash_bucket` uses.
Re-checked with the correct key → 0%.) The `_disjoint_split` + content-hash guard hold for the new benches.

**Chunked code — correct.** `encode_chunked` (per-chunk encode + concat), routing (top-keep by query cosine,
order-preserved), grad-checkpointing (chunked only; no-op under eval no_grad), chunk-aware `mem_len`/cost, and the
shape-guarded dev-anchor all verified. (Minor perf nit: `_readout` is computed twice in the routing score.)

**Added baseline — RETRIEVAL (the missing rival).** A train-free `retrieval` method: split ctx into 64-tok chunks,
keep the top-4 by query lexical overlap (BM25-style term presence), feed their **raw** embeddings (no compression).
This is the natural rival to chunked routing — *does a learned compressor beat just retrieving the relevant raw
chunk?* Smoke (toolace, 8 items): **retrieval 0.625** vs trunc 0.0 vs full 1.0, cost 391 (256 raw + query). Queued
on all long benches + the ruler frontier (13 jobs). The key contrast: **ch16r (128 *compressed* tokens) vs
retrieval (256 *raw* tokens)** — if compression matches retrieval at ~½ the tokens, that is the efficiency win.

**Added controls:** `k160m_toolace` (single-pass K=160 = ch16a's budget → isolates locality from budget, H2);
`c128a` on toolace/rca (chunk 128, ratio 8× → does a higher compression ratio still hold?).

→ Long-context study now **~85 jobs** (added retrieval ×13, matched-K ×1, chunk-size ×2; R3 re-queued ×6).

## 8. Mix-training SCALING study + more 2025+ baselines (2026-06-14)
**Question:** does our compressor *scale*? We train ONE compressor on a **mix** corpus (12 benches pooled) and
sweep three axes, reporting **average compress-acc over all benches**:
- **Data axis = curriculum learning:** mix is ordered easy(short ctx)→hard(long); `n_items` = how much of the
  easy→hard prefix is included (96/192/384/768). `steps = 8·n_items` (8 epochs) so compute tracks data. Verified
  the curriculum order (ctx-len ascending, sorted=True).
- **Model axis = depth:** encoder layers `N ∈ {4,8,16,28}` (more trainable params).
- **Compute axis = FLOPs:** `C = 6·P_trainable(N)·tokens` (6ND), reported by `analyze_scale.py` (acc-vs-FLOPs
  curve + data/depth slices). `P_trainable(N)=(N+2)·211M + LoRA` for Qwen3-8B.

**Grid (11 jobs, `aa_scale_*`, run first):** {N4,N8,N16}×{D96,D192,D384} + N16/D768 + N28/D384. Each trains GCM
(K=64, single-pass) on the mix and evals all 12 benches. The scaling figure tells us whether more
data/depth/compute lifts the average — i.e., whether compression has headroom or is fundamentally capacity-bound.

**★ RESULT (11/11 done): NO scaling.** Avg compress-acc over 12 benches is **FLAT at ~0.25–0.28** across the grid:
| axis | sweep | acc |
|---|---|---|
| **compute (FLOPs)** | 3.5e15 → **8.4e16** (24×) | 0.253 → 0.267 |
| **data** (curriculum, N=16) | D=96 → 768 (8×) | 0.247 → 0.267 |
| **depth** (D=384) | N=4 → 28 (7×) | **0.285 → 0.279** (N=4 is *best*) |

→ More data / depth / compute does **not** improve the compressor; the **smallest** depth (N=4) is among the best. The compressor is **capacity-bound — it does not scale.** Figure: `out/clean/scale/scaling.png`. This is the central negative that anchors the robustness-layer thesis.

**New 2025+ baselines (from related-work scan):**
- **`tokprune`** (queued, 10 benches) — query-AGNOSTIC hard-prompt pruning: keep the K highest-**self-information**
  (surprisal) ctx tokens (LLMLingua / Selective-Context family), train-free. Smoke: 0.83 on bfcl_multiple.
- **`retrieval`** (already queued) — query-AWARE token selection (SnapKV/BM25 spirit).
- Together these bracket "select important tokens" (agnostic vs query-aware) at budget K, train-free, offline.
- **Scoped follow-up:** faithful **KVzip** (NeurIPS'25, query-agnostic *KV-cache* eviction via reconstruction)
  — heavier (needs per-head KV eviction in the runtime); deferred. `tokprune` is the token-level stand-in for its
  query-agnostic importance idea.

Analysis: `analyze_scale.py` (acc vs FLOPs) · baselines fold into the §6 long-context table.

## 9. Baseline taxonomy fix + the 2025+ CONTEXT-COMPRESSION family (2026-06-14)
**Correction (per review):** `tokprune`/`retrieval` feed **raw embeddings of a selected token subset**
(`rt.embed(selected_ids)`) — that is **token pruning/selection on embeddings, NOT context compression**. We
must not claim **efficiency** wins vs token-pruning or runtime KV-prune/merge (KVzip/SnapKV/ToMe): those reduce
the sequence at inference, whereas we (and context compression) produce a **reusable, amortized** compressed
memory — a different efficiency model. So `tokprune` was dropped and KVzip de-scoped; selection methods stay
only as **necessity lower-bounds**, never as efficiency rivals.

**Implemented the 2025+ context-compression family** (faithful core re-impls, `baselines2025.py`), all compress
to K units the frozen base reads (budget-comparable to GCM, fair amortized efficiency), trained at the matched
budget (enc-16, K64, 384 items, 3000 steps) — queued `ac_b25_*` (6 methods × 6 benches = 36):
| method | cite | mechanism | inject |
|---|---|---|---|
| **ICAE** | ICLR'24 | encoder → K soft tokens, AE + LM | soft prefix |
| **AOC** | 2501.06730 (Jan'25) | ICAE w/ **attention-only** encoder (MLPs frozen) | soft prefix |
| **Beacon** | Activation Beacon (2025) | K **interleaved** compression beacons (one pass) | soft prefix |
| **X500** | 500xCompressor (2025) | compress into **K KV pairs** (per layer) | KV cache |
| **ComprExIT** | 2602.03784 (2026) | ICAE + explicit **transmission** aggregation head | soft prefix |
| **LCC** | 2602.21221 (2026) | **test-time** per-context buffer optimization (reconstruction) | soft prefix |
**Faithfulness — VERIFIED FAITHFUL** (audit: [`baselines-faithfulness.md`](baselines-faithfulness.md)). The
`ac_b25_*` jobs ran the **faithful** code (confirmed: deployed `baselines2025.py` + run logs show e.g.
`[icae] LoRA(q,v) rank=64 encoder + frozen decoder; K=64`). Exact mechanisms: **ICAE** = LoRA(q,v)-on-base
encoder + memory-token **output-hidden-state** slots + **frozen-base decoder** + **[AE]** (AE + task); **X500** =
same but **KV injection** ([BOS]-triggered); **AOC** = separate **MLP-removed (attention-only)** encoder; **Beacon**
= interleaved **beacon-KV** (single-pass at ≤1024); **ComprExIT/LCC** = faithful cores. Only the **training HPs**
are our matched budget (batch-1 training kept as-is, per scope). So the numbers below are the **faithful** results:

**b25 results — FAITHFUL, compress vs full:**
| bench | full | icae | aoc | beacon | x500 | comprexit | lcc |
|---|---|---|---|---|---|---|---|
| bfcl_live_multiple | 0.906 | 0.062 | 0.083 | 0.031 | 0.021 | 0.083 | 0.010 |
| bfcl_multiple | 0.883 | 0.050 | 0.117 | **0.333** | 0.183 | 0.117 | 0.017 |
| hermes | 0.938 | 0.510 | 0.406 | **0.615** | 0.365 | 0.469 | 0.354 |
| narrativeqa | 0.136 | 0.137 | 0.130 | 0.099 | 0.109 | 0.130 | 0.150 |
| rca_openrca | 0.406 | 0.083 | 0.104 | **0.250** | 0.083 | 0.094 | 0.125 |
| toolace | 0.948 | 0.104 | 0.073 | **0.375** | — | 0.073 | 0.010 |
→ all (faithful) baselines are far below `full`; **Beacon** (interleaved KV) is the strongest. None approaches full — consistent with the §6/§8 capacity-bound finding across *all* compressors. These join Gist + Cartridge as the §2.4 rivals; trunc/retrieval are necessity floors only.

## 10. Soft-memory → text probe (interpretability): the memory is OOD
Mapping each of the K memory vectors to its nearest vocabulary token (cosine in the base input-embedding space)
and decoding yields **gibberish** (`。 = ע ， paso lengths …`), not the relevant content (e.g. `rectangle.area`).
- **Mean nearest-token cosine = 0.096** → the memory is **near-orthogonal to the entire vocabulary** (OOD).
- **M0 (uncond) ≈ Mq (cond)** decode identically → query-conditioning barely moves it in token space (matches the small ΔCode).
- Slots **collapse to fixed directions** (slot-0 always `。，`, slot-1 always `=lengths`) regardless of input.
→ Classic soft-prompt OOD failure (motivates the A1/A2 fixes, OFF by default) and a clean reason the gate is needed: a frozen base reads an OOD memory it can't fully trust. Probe: `probe_decode.py`.

## 11. Training dynamics + stability fix (v1.7.5)
**The spiky loss curves are an artifact, not divergence.** Cause: (1) **batch size = 1** (per-example `backward+step`)
→ each logged point is one example's CE (a hard long tool example → CE 12, easy → 0.7); (2) only ~10 **raw** single-step
points logged per run, no smoothing. The runs converge underneath (e.g. `icae_toolace` task: min 0.66, final 1.18, max 12.3).
HP history: only **lr×steps** was swept (→ combo lr1e-4/3000) + multi-seed; **batch / warmup / schedule / fp32 were not tuned**;
and **no val/test loss was tracked during training** (only a final eval).
**Fix (now default):** gradient accumulation (`grad_accum=8` → effective batch>1), **LR warmup + cosine decay**,
**EMA-smoothed** loss logging, and **periodic held-out val loss** (real train/val dynamics). Smoke: `task_ema`
descends cleanly 1.6→1.33, `lr` cosine→0, `val` logged (2.30→2.29). Re-training headline + scaling with the stable
recipe on `test` (`ad_stab_*`, 7 jobs); will re-plot `task_ema` + `val` vs the spiky originals. Plotter: `plot_dynamics.py`.

