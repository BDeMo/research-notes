# Code-faithfulness, paper-fidelity & traceability audit (2026-07-06)

> Full re-check of the eval code so our observations match the papers, which facts are affected, per-dataset characteristics,
> traceability of every data point, and a fix/method brainstorm. Cross-ref: `matrix-facts.md`, `baseline-catalog-faithfulness.md`.

## 1. Code-faithfulness verdict (per method / path)
| item | verdict | evidence (code) | note |
|---|---|---|---|
| ~20 KV methods | ✅ EXACT | `run_kvpress.py:40-56` — official `kvpress` classes, `compression_ratio=RATIO`, **paper-default** window/kernel/n_sink | attention presses on eager, norm/window on sdpa (correct per kvpress) |
| LLMLingua-2 / LongLLMLingua / orig | ✅ EXACT | `run_baseline.py` — official `llmlingua` pkg; orig/Long use `use_llmlingua2=False` + authors' Llama-2-7b | question-aware config for Long |
| BM25-RAG | ✅ faithful (lexical) | `run_baseline.py` `_rag` — textbook BM25 | **not** DPR-RAG; cite as BM25-RAG |
| MC scoring, baselines | ✅ faithful | `model.py:128 mc_loglik` — **length-normalized loglik** of the option letter | standard MC-by-loglik |
| NIAH scoring | ✅ consistent | both paths: `score_gen` + **gold-substring fallback** | same fallback in `run_kvpress:103` and `run_baseline` `_sc` |
| **ToMe** | 🟡 ADAPTED | `run_baseline.py` `_tome` — **input-side** bipartite embedding merge | real ToMe merges **inside layers on keys**; ours is a one-shot input-side merge (annotate; it's our own merging baseline) |
| gist-lite / cart-lite | 🔴 NOT faithful | reuse our `compressor.mem` | already flagged (D21) — not Gisting/Cartridges |
| window (txl) | 🟠 generic | `_win` last-W + 4 sinks | not StreamingLLM (that = kvpress `streaming`) |

## 2. ⚠ THE key finding — MC scoring is INCONSISTENT across harnesses
- **`run_baseline` (non-KV baselines + refs):** MC via **`mc_loglik`** (loglik-argmax of the option letter). Reliable on a base with no chat template.
- **`run_kvpress` (all KV methods):** MC via **generate 4 tokens → extract first A–F letter** (`run_kvpress.py:92-96`). On a no-chat-template base this **often fails to emit a clean letter → systematically deflated/noisy**.
- **Consequence:** on the **MC benches — `quality`, `quality_hard`, `musr_mm`** — the **KV-method numbers are NOT comparable** to the loglik-scored baselines/refs. Any table mixing them (task-flip §F3 `quality_hard` column; head-to-head `quality` column; catalog MC cells for KV methods) is **scoring-biased**.
- **What is SAFE:** everything on **NIAH / gen benches** (ruler, numerical, categorical, coding, squad, hotpot, narrativeqa, trivia, locomo) — both paths use `score_gen` + the same substring fallback. And **F6 ("full context hurts" on quality/quality_hard)** is SAFE because `full` and `no_ctx` are **both** `run_baseline` loglik (same harness) — it's a within-harness comparison.

## 3. Which established facts are affected
| fact | status after audit |
|---|---|
| F1 (attn collapse ≥16k), F2 (attn-free robust), F4 (kvzip cliff/retrieval-only), F5 (window locality), F7 (distractors), F8 (LLMLingua), F9 (RAG), F10 (GDN), F11 (scaling), F12 (commodity), F14 (ToMe), F16, F17 | ✅ **unaffected** — all on NIAH/gen, consistent scoring |
| **F6** (full hurts on literary MC) | ✅ **holds** — full vs no_ctx both loglik (same harness) |
| **F3 task-flip** | ✅ story holds (retrieval vs QA on ruler/squad/hotpot/trivia); ⚠ the **`quality_hard` column for KV methods is generation-scored** → re-score before quoting |
| **head-to-head `quality` column** (running) | ⚠ KV (kvpress gen) vs others (loglik) — **not comparable**; re-score |
| any **KV-method MC number** (quality/quality_hard/musr) | ⚠ **re-run under loglik** before use |

**Net:** no *headline* fact breaks (necessity, length-crossover, task-flip-on-retrieval, full-hurts, commodity-compressor all stand); only the **KV-method numbers on the 3 MC benches** need re-scoring.

## 4. Traceability (source · config · log for every data point)
- **Logs:** all 812 `RECIPE_EVAL` lines in `results-v2.0.0/logs/` (per-wave files). ✅
- **Config (NEW this audit):** the exact per-cell env (method, ratio, NVAL, MAXCTX, NCHUNKS, GEN_MAX, model) lived only in the pod queue files → now **committed to `results-v2.0.0/configs/`** (`q_*.txt` + `runner_u.sh` for the COMMON defaults). Each `RECIPE_EVAL <tag>` ↔ the queue line with the same `GCM_TAG` ↔ the source `.log` on the pod. ✅
- **Gap remaining:** the full per-cell `.log` files (with model-load banner + per-item output) stay on the pods (large); we keep the RECIPE_EVAL summary + config. If needed, we can archive full logs for the headline cells.

## 5. Dataset characteristics (why each behaves as it does)
| bench | type | answer form | scoring | key gotcha |
|---|---|---|---|---|
| ruler_niah | synthetic single-needle | rare verbatim value | exact/substring | needle is 1 token-run → merge/prompt-drop **kills it**; length-controllable |
| numerical/categorical/coding_niah | NIAH variants | numeric/label/code | substring | same; `run_baseline` needed the fallback (was 0 before fix) |
| **multi_needle_niah** | multi-fact | multiple | 🔴 **broken (full=0)** | scoring bug — **excluded** |
| squad_v2 | extractive QA | local span (+abstain) | gen (EM/F1) | answer is **local** → window w256≈full |
| hotpot_qa | multi-hop QA | distributed | gen | needs several spans → window needs large W |
| narrativeqa | abstractive long-doc | **not lexically in text** | gen | RAG/merge **fail** (no lexical overlap); full ceiling low (~0.15) |
| trivia_qa | open QA + noisy evidence | short fact | gen | `no_ctx` already high (~0.5, parametric); compression can **denoise** |
| quality / quality_hard | long **literary MC** | letter | **MC-loglik (baseline) vs gen (KV) ⚠** | full can **hurt** (base ceiling); near-random; **the MC-scoring issue lives here** |
| musr_mm | multi-step reasoning MC | letter | same MC caveat | |
| locomo | long dialogue memory | short | gen | multi-session recall |

## 6. Fixes + brainstorm (feasible)
**Corrective (do now):**
1. **Unify MC scoring** — add a **loglik-MC path to `run_kvpress`**: prefill `[ctx]` under the press (`with press(model): out=model(ctx,use_cache=True)`), then score each option's logprob with the compressed `past_key_values`; argmax. Re-run the KV × {quality, quality_hard, musr_mm} cells. *(This makes MC cross-family comparable and closes the only real bias.)*
2. **Also compute `full`/`no_ctx` inside `run_kvpress`** (same harness) so a KV method's Δ-vs-full is self-consistent, not cross-harness.
3. **Annotate ToMe as input-side** (not layer-internal) wherever cited.
4. **Fix `multi_needle_niah` scoring** or keep it excluded.

**Method brainstorm (unchanged direction, now on verified facts):** the audit *strengthens* IMP (`v2.1.0-paperB-method-designs.md`): since the only scoring bias was MC (which is near-random anyway), the **retrieval/abstractive facts that IMP relies on are solid** — merge kills needles (F14/F17), attention-KV collapses at length (F1), reconstruction-importance is most robust (F4). The **feasible, faithful method** remains: *cheap length-robust importance scoring → keep-verbatim the un-reconstructable needle / merge redundant → short prefill; distilled from the base's own signals; + side-cache for GDN.* Add one audit-driven refinement: **evaluate the method's MC with loglik from the start** (avoid the generation-MC trap).

*Provenance: `run_kvpress.py`, `run_baseline.py`, `src/gcm/model.py:128`, `results-v2.0.0/{logs,configs}/`, `matrix-facts.md`.*
