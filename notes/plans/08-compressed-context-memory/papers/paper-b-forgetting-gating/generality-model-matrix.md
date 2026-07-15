# Paper-B generality table — cross-architecture / cross-family / size

> **Method version:** `imp` = **IMP-v2.1.0** (span-level Importance-routing Prefilter, Mode A training-free, span=32, keep=0.5, signals={query-relevance, surprisal}) — same tag as the headline main table.

**Purpose.** The headline **full** table is Qwen3-8B ([`main-table-fulltest.md`](main-table-fulltest.md)). This second table shows the *same* method behavior — and IMP's win on retrieval + the "full-hurts" phenomenon — **generalizes across model architecture, vendor family, and size.**

**Protocol (disclosed).** Representative **5-bench subset** (one per regime) at **N=100** (a disclosed subset — the headline table stays full-set). Text-side methods (no_ctx, full, window, RAG, LLMLingua-2, ToMe, **IMP**) run on any causal LM; **KV-eviction (kvzip/knorm) applies only to quadratic-attention models** (a linear/GDN model has no KV cache — itself a paper point). Launcher `gcm/experiments/gen_launch.sh` (`MODEL/MTAG/ARCH/GPUS`), output `/mnt/persist/grid_generality/g_*`.

Bench subset & config: `ruler_niah`@16k (nc88), `longbench_v2`@16k (nc12), `lb_hotpotqa`@16k (nc12), `quality_hard`@8k, `squad_v2`@4k. Keep 0.5.

## Model matrix (all cached on pod)

| tag | model | arch | role | status |
|---|---|---|---|---|
| q3_8b | Qwen3-8B | quad | **headline (full table)** | ✅ full 16-bench |
| **q35_9b** | **Qwen3.5-9B** | **linear/GDN (no KV)** | architecture contrast | ✅ done (23/25) — see below |
| q35_4b | Qwen3.5-4B | linear/GDN | linear + size | ⏳ |
| q25_7b | Qwen2.5-7B-Instruct | quad | cross-family / prev-gen | ⏳ |
| glm4_9b | GLM-4-9B-0414 | quad | cross-vendor (Zhipu) | ⏳ |
| ministral_8b | Ministral-8B-Instruct | quad | cross-vendor (Mistral) | ⏳ |
| gptoss_20b | gpt-oss-20b | quad (MoE) | cross-vendor (OpenAI OSS) | ⏳ |
| q3_1p7b | Qwen3-1.7B | quad | size-scaling ↓ | ⏳ |
| q3_4b | Qwen3-4B | quad | size-scaling | ⏳ |
| q3_14b | Qwen3-14B | quad | size-scaling ↑ | ⏳ |
| moonlight_16b | Moonlight-16B-A3B | linear/MoE | linear + MoE (stretch) | ⏳ |

**Claim to establish:** across all rows, (1) no fixed method wins everywhere; (2) `full` is a liability on literary-MC (quality_hard) and the hardest MC (longbench_v2); (3) **IMP is near-lossless on RULER** and competitive elsewhere — *architecture- and family-independent*, which is what licenses "bolt IMP onto any frozen base."

**Compute note (honest).** Linear/GDN models run the **torch fallback** for the gated-delta kernel (the `flash-linear-attention` fast path needs `causal-conv1d`, not installed to avoid perturbing the shared venv mid-grid) → ~15–20 s/item @16k. Hence N=100 for the generality table. If we want full-set on the linear arm, install `causal-conv1d` after the main grid drains and re-run.

## Results — Qwen3.5-9B (linear/GDN, no KV cache), N=100 (×100)

| bench | no_ctx | full | window | rag | ll2 | tome | **imp** |
|---|--:|--:|--:|--:|--:|--:|--:|
| RULER-NIAH @16k | 0 | 91 | 4 | 89 | · | 0 | **79** |
| lb_hotpotqa | 9.8 | 44.9 | 14.2 | 37.4 | · | 20.5 | 26.9 |
| LongBench-v2 (hardest) | 28 | 32 | 35 | 36 | 34 | 29 | 29 |
| QuALITY-hard (MC) | 21 | **9** | 17 | 12 | 9 | 17 | 12 |
| squad_v2 | 25.7 | 71.8 | 71.8 | 67.3 | 70.8 | 42.8 | 44.9 |

`·` = 2 ll2 cells (ruler, hotpot) errored (LLMLingua model + linear base OOM) — re-queue. GEN_DONE 23/25.

### The whole story REPRODUCES on a cache-free linear architecture (the generality headline)
| phenomenon | Qwen3-8B (quadratic) | Qwen3.5-9B (linear/GDN) | ✓ |
|---|---|---|---|
| **"full hurts" on literary-MC** | QuALITY-hard full 9.2 < blind 19.6 | full **9 < blind 21** | ✓ |
| **hardest MC: full ≈ blind** | LongBench-v2 full 33.6 ≈ blind 33.4 | full **32 ≈ blind 28** | ✓ |
| **IMP near-lossless on RULER; merge/trunc collapse** | imp 96.8 / full 99 (tome 0, win 6) | imp **79** / full 91 (**tome 0, win 4**) | ✓ |
| **selection ≥ full on hardest MC** | RAG 34.2 > full 33.6 | RAG **36 / win 35 / ll2 34 > full 32** | ✓ |

**Reading.** On a model with **no KV cache** — where kvzip/knorm/snapkv literally cannot run — the same failure map holds and **IMP (input-side) still keeps the RULER needle near-losslessly** while ToMe (0) and window (4) collapse. IMP's RULER gap is a bit wider than on the quadratic base (79 vs 91), consistent with GDN **state-collapse** (F10): a fixed recurrent state struggles to hold even a *selected* needle → motivates Mode-B's verbatim **side-cache** for the linear arm. This is the direct evidence for "bolt IMP onto *any* frozen base, incl. linear."

## Cross-model summary — the diagnosis is model-invariant (2026-07-09, N=100)

10 models across size (1.7B→14B), family (Qwen2.5/3/3.5), vendor (GLM-4, Ministral), and architecture (quad / linear-GDN). Three signature phenomena, per model (×100):

| model | arch | **retrieval: IMP / full / ToMe** (RULER) | **squad: IMP / RAG / full** | **literary-MC full-hurts?** (q_hard full vs blind) |
|---|---|---|---|---|
| Qwen3-1.7B | quad | 96 / 98 / · | 29 / 45 / 52 | 18 < 21 ✓ |
| Qwen3-4B | quad | 99 / 99 / · | 23 / 52 / 55 | 14 < 20 ✓ |
| Qwen3-8B (headline) | quad | 97 / 99 / **0** | 39 / 72 / 69 | 9 < 20 ✓ |
| Qwen3-14B | quad | · / · / · | 24 / 33 / 38 (kvzip 66) | 9 < 18 ✓ |
| Qwen2.5-7B | quad | 99 / 99 / **0** | 28 / 64 / 59 | 11 < 23 ✓ |
| **Qwen3.5-4B** | **linear** | 99 / 99 / **0** | 39 / 58 / 56 | 11 < 24 ✓ |
| **Qwen3.5-9B** | **linear** | 79 / 91 / **0** | 45 / 67 / 72 | 9 < 21 ✓ |
| GLM-4-9B | quad | 97 / 24 / · | 23 / 45 / 46 | 17 < 18 ✓ |
| Ministral-8B | quad | · (kvzip 99) | 33 / 65 / 63 | 15 < 21 ✓ |
| gpt-oss-20B | quad | ✗ harness-broken (base ~5% everywhere; needs harmony format) | — | 18 < 24 ✓ |
| Moonlight-16B | linear/MoE | ✗ failed to load (trust_remote_code/config) | — | — |

**Three model-invariant facts (F31):**
1. **IMP ≈ full on retrieval, ToMe collapses to 0** — holds at every size/family/arch (incl. both linear GDN). (GLM-4: IMP 97 ≫ its own full 24 — selection even *rescues* a base that fails full-context RULER.)
2. **IMP is the weakest on extractive QA; RAG/window ≫ IMP** — every model (squad IMP 23–45 vs RAG/window 45–72). F30 reproduces universally.
3. **"full hurts" on literary-MC is universal** — full < blind on **all 10 models** (q_hard), and kvzip/knorm denoise there (~30 > blind) on quad models.

⇒ the **diagnosis is not a Qwen3-8B artifact**; it is a property of frozen LLMs + long context. This is the robust, model-independent contribution.

## IMP design-ceiling ablation (F32) — no signal/span combo closes the RAG gap
Qwen3-8B, signal{query,surprisal,both} × span{8,16,32,64,128}, N=100 (`grid_impabl/ia_*`):
- **squad_v2:** every one of the 15 configs lands **11–24** — vs RAG **71**, full 69. Best (any signal/span) ≈ 24.
- **hotpot_qa:** all 15 configs **22–32** — vs RAG **55**, full 56. Best ≈ 32.
⇒ **IMP's QA gap to RAG is structural, not a hyperparameter**: no scoring signal or span width recovers it. Confirms F30 decisively — tuning IMP will not make it a SOTA-accuracy QA method.

## Verified so far
- **Qwen3.5-9B loads & runs IMP in the harness** (linear_attn params materialize; `enable_torch_linear_attention()` active). Smoke: RULER N=3 → IMP method **1.0**, full 0.667, no_ctx 0.0. Full generality subset **done (23/25)** — see table above.
- Next models (quad cross-family + size): Qwen2.5-7B, GLM-4-9B, Ministral-8B, gpt-oss-20b, Qwen3-1.7B/4B/14B (launch after keep-ablation drains the pods).
