# Decision & action log — 2026-06-24

> Per user directive ("你做了你决定的事情，都得记录在案"): every autonomous decision/action is recorded here.
> Format: **Decision · Why · Action/artifact · Status (reversible?)**. Newest concern first. Follows `decisions-2026-06-08.md`.

## 🔴 D0 — CMG-distill → foundation repo: **HELD, nothing pushed** (privacy/IP)
- **Decision:** do **NOT** add/commit/push `cmg_distill` / `cmg_aug` / `cmg_skill` to the external GitLab (`gitlabe2.ext.net.nokia.com/s1shi/cmg-rca-foundation`) **as-is**, despite the request to "把我们生成的/蒸馏的数据弄上去".
- **Why:** content audit found these "distilled/synthetic" sets still **embed real identifiers** — real DTS ticket numbers (distill 69+56 uniq; aug 662; skill up to 841), **customer names** (TMO/T-Mobile etc.: distill-train 240 hits, aug 94, skill 172), real timestamps (distill-test 1594), system/VM IDs (distill-test 357), even a stray `.msg`. Pushing to an *external* repo = customer-data/IP leak — exactly what "真实数据不用" intends to prevent.
- **Action/artifact:** audit only (read-only); **0 files added/committed/pushed.** Surfaced the risk + a sanitize-then-ship plan to the user. Also flagged: the **real** `data/cmg_rca_cases.jsonl` (38 raw DTS `.msg`) is **already committed + pushed + in git history** → recommend removal + history scrub (destructive → needs explicit go).
- **Status:** ~~reversible (nothing changed). Blocked on user decision.~~ → **UPDATE 16:03 — user override:** "push 吧,没什么大问题,之后再修复,现在是内部用." User (data owner) explicitly authorized after I flagged the leak twice. **Action taken:** added `data/generated/{cmg_distill,cmg_aug,cmg_skill}` (3038+643+846 rows) + a README documenting provenance/known-issues, committed + pushed to the **internal-use** foundation repo. Real `cmg_rca_cases.jsonl` left as-is (not asked to remove; flagged). **On record: real DTS IDs + some operational details are in the pushed data — accepted by owner for internal use.** Conformance fixes (verification-ending 74%, module-mask 93%→100%) deferred per user ("之后再修复"). → **UPDATE 16:10 — fixes DONE on request "fix":** `data/generated/_fix_conformance.py` masks the answer-module token in `context` (word-boundary + lte_/path/long-form aliases → `<module>`; cross-module mentions kept) and normalizes verification endings to PASS/FAIL/VERIFY (restate in-text verdict, else neutral VERIFY action — no fabricated pass/fail). **Result: module-leak 0, verification-end 980/980.** Propagated to all 3 pods' `/mnt/persist/datasets/cmg_distill` + committed/pushed (`31954ba`). Minor cosmetic over-mask noted (e.g. `gtp` case "GTPv2"→"`<module>`v2" — correct since the protocol name is the answer giveaway).

## 🟡 D1 — Correction logged: my first SkillMesh section-scan over-stated the gaps
- **Decision:** correct my own earlier claim that the distill reports were missing most SkillMesh sections.
- **Why:** my marker scan used exact header strings ("Executive Summary", "Verification"); the real reports use aliases (`DTS Summary`, `Fix Recommendation`, `Verification Plan`, `Failure Mechanism`). Root Cause / Impact / Fix / Verification **are present**. Genuinely missing = Execution Flow / Code Analysis / Stack Trace / Log Correlation / `file::fn:line`.
- **Why those are missing:** the source `evidence` is a **~900-char defect description with no stack/counters/logs/code** → deep sections can't be grounded → **deliberately omitted to avoid hallucination** (by design, not a bug). Fuller sections only exist in `cmg_distill_real` (real-report decomposition; but leaky + train-on-test).
- **Status:** documented; no data changed.

## Earlier autonomous actions this session (on record)
- **D2 — Honest CV gate (code change to shared repo).** Added `_gate_cv` (5-fold held-out τ+signal) to `gcm/experiments/run_recipe.py` (local + both pods). **Finding:** held-out `gated_cv` is **−0.02 vs strong full** (and +0.52 when full collapses) → the v1.8.0 "gated ≥ full / beats full" claim was an **in-sample artifact**; corrected in `results-v1.8.0.md`. *Reversible (git).*
- **D3 — eos-list harness fix.** Coerced list-valued `eos_token_id`→first id in `compressor.py`/`train.py` (both pods + local) → unlocked Llama-xLAM / ToolACE / GLM bases. *Reversible.*
- **D4 — Baselines implemented (new code).** `gcm/experiments/run_baseline.py` with `sft` / `gist` / `cart` / `txl` modes. *Additive.*
- **D5 — Long-context suite wired (data.py change).** Registered 6 benches (`multi_needle_niah`, `numerical_niah`, `categorical_niah`, `coding_niah`, `locomo`, `quality_hard`) + `GCM_NCHUNKS` length knob (local + pods). *Additive.*
- **D6 — Killed hung jobs.** `mt2_q35_rca`, `mt2_q35_quality`, `xtool_qwen354b_hermes` (Qwen3.5 deadlocks, ~1–5h no output). Killed to free GPUs; **numbers were redundant/already-had**. *Irreversible (lost partial runs), low cost.*
- **D7 — vLLM ops (d1420).** Shut down vLLM (freed GPU1); **installed `cuda-nvcc-12-8`** (apt) to fix the GDN-kernel JIT crash; made `--enable-prefix-caching` opt-in; created `start.sh`/`stop.sh`; symlinked resources into `fengluo`'s `~/workspace` + set log dir writable. *Mostly reversible; nvcc install persists.*
- **D8 — Default config picks (low-stakes, noted not asked):** K=128 as per-base sweep winner; TXL implemented as StreamingLLM-style bounded window (W=1024 + 4 sinks) since true TXL recurrence is N/A on a frozen base; 16k over-window cells deferred (OOM at K=256).
- **D9 — Docs/notes created (no code/data risk):** `references-longcontext.md` (163 refs + main-thread judgment), `v2.0.0-plan.md`, `ideas-brainstorm-v2.0.0-2026-06-24.md`, results-doc updates.

## ⚠️ D10 — NOTE: pre-mask cmg_distill adapters have inflated compress scores
- **Fact:** the module-mask fix (D0 update) changed `context` (what M compresses). **Any `cmg_distill`/`cmg_all` adapter trained BEFORE 2026-06-24 16:10 used the leaky (un-masked) context** → the answer module name was in the input → **their `compress` numbers are optimistic** (partly copy, not compress). Affects e.g. the demo's `mt_all_distill1` cmg numbers (0.263) and any cmg_distill cell.
- **Action when cmg numbers matter again:** retrain cmg_distill on the masked data (now on all pods) and re-report; treat old cmg compress as an upper bound. Not urgent (cmg is demo/motivation, not a paper headline per the scope guard), but **must caveat before quoting cmg compress as a clean result.**
- **Status:** recorded; no retrain launched yet.

## 📏 D11 — DATA POLICY: paper experiments use PUBLIC benches only (user 16:35)
- **Rule:** "都用公开数据集,不要在 internal 上做." All paper/baseline/method experiments run on **public** benches; **internal CMG stays demo/motivation only** (already the scope guard).
- **PUBLIC (use freely):** bfcl* · toolace · hermes · glaive · apibank · squad_v2 · hotpot_qa · trivia_qa · narrativeqa · quality · quality_hard · musr_mm · ms_marco · ruler_niah + niah family (multi_needle/numerical/categorical/coding) · locomo · rca_openrca · rca_rcaeval (OpenRCA/RCAEval are public RCA benchmarks).
- **INTERNAL — do NOT use for paper numbers:** cmg_rca · cmg_aug · cmg_skill · cmg_distill · cmg_all (Nokia CMG).
- **Status:** in effect. Current baselines (quality_hard, narrativeqa) are public ✓; v2 long-ctx suite + transfer matrix already specced on public benches.

## 🔬 D12 — 10h autonomous "black-box fact-base" sweep launched (user 19:00)
- **Goal (user):** "把 gpu 都用上,排10小时的任务。不断验证不同 baselines,我们要验证黑盒的 facts" → broad empirical fact-base on how each *black-box* (frozen-base) baseline behaves across budget / task-type / context-length, to seed a later dive-in for insights.
- **Throughput fix:** eager-attn KVPress on 8k×N96 was ~1h/cell (too slow for a matrix). Patched `run_kvpress.py` → eager only for attention-based presses (snapkv/h2o/pyramid); **window/norm presses (streaming/knorm) run on fast SDPA**. KV cells use NVAL=64, others NVAL=96.
- **Matrix = 102 cells, Qwen3-8B (+Qwen3.5-9B block), PUBLIC benches only:**
  - **A budget fact:** {snapkv,h2o,pyramid,streaming} × RULER-8k × ratio{0.1,0.25,0.5,0.75,0.9} (20)
  - **B task-type fact:** KV4 × {squad_v2,hotpot_qa,narrativeqa,quality_hard} @0.5 (16)
  - **C length fact:** snapkv{4k,8k}, streaming{4k,8k,16k} @0.5 (5)
  - **D prompt-compression:** LLMLingua-2 × rate{0.1,0.25,0.33,0.5,0.67} × {squad,hotpot,narrativeqa,quality_hard,ruler8k} (25)
  - **E window fact + Full ref:** StreamingLLM-window{256,512,1024,2048} + Full × {squad,hotpot,narrativeqa,quality_hard} (20)
  - **F retrieval-type:** {snapkv,streaming} × {multi_needle,numerical}-NIAH @0.5 (4)
  - **G GDN base (KV N/A):** Qwen3.5-9B × {LLMLingua, txl-window, full} on {quality_hard,narrativeqa,squad} (12)
- **Exec:** round-robin 17 cells/GPU → `q_fb_0..5`, drained by `runner_u.sh` (skips `.done`). All 6 GPUs confirmed live 19:04–19:05. Est ~4–6h to drain (eager-KV cells dominate); if it finishes early, extend with more lengths/ratios.
- **Status:** running. Results land as `fb_*.json` in `/mnt/persist/grid_main/`. Next: harvest → fact-base table (budget/task/length curves) → pick dive-in targets.

## 🔬 D13 — Wave-2 baseline sweep (99 cells, ~9h plan) launched (user 00:30 Jun 25)
- **Wave-1 (D12) drained:** all 102 cells done. Harvested as `RECIPE_EVAL <tag> {...}` lines in the `.log` files (NOT `.json` — results are printed; `.json.done` is just the runner marker).
- **Wave-1 headline facts:** (a) KV budget curve on RULER-8k — **SnapKV≈StreamingLLM dominate, H2O degrades fast**; (b) `quality_hard`: **full context HURTS** (full 0.094 < no_ctx 0.229, both bases) — strong motivation hook; (c) squad answer is local → window w256 ≈ full; (d) LLMLingua-2 gentlest on retrieval (RULER r0.67→1.0).
- **Bugs found:** **PyramidKV** throws tensor-shape mismatch on Qwen3 (all-0.0 = invalid, dropped). **AdaKV** also 0.0 (dropped). Smoke-confirmed working new presses: **knorm 0.92, tova 0.42, expected 0.25** (RULER-4k @0.5).
- **Wave-2 = 99 cells, public, NVAL=96:** length-scaling crossover (the headline: {snapkv,h2o,expected,tova}×{2k,4k,8k}+{streaming,knorm}×{2k…32k}+snapkv/expected@16k), new-method ratio curves + task-type (expected/tova/knorm), LLMLingua on trivia/multineedle/numerical, txl-window {128,4096}, Qwen3.5-GDN block, full-ref RULER length curve, multi-needle budget, trivia KV.
- **Exec:** round-robin 16–17/GPU → `q_fb2_0..5`, `runner_u.sh`. All 6 GPUs live 00:44–00:46. Est ~5–6h (eager-KV + 16k/32k dominate). Results → `fb2_*`/`ws_*` log lines. Next: harvest both waves → length-scaling curves + method ranking for the dive-in.

## 🔬 D14 — Baseline-catalog reproduction sweep (20 KV methods + refs, 177 cells) launched (user 09:00 Jun 25)
- **Goal (user):** reproduce *every* baseline from the lit review, faithfully; ≥20–30 methods; tune HP/base when not base-specific; for base-specific methods describe + skip, or run adapted + annotate diffs. Sweep more baselines AND benchmarks.
- **Method:** generalized `run_kvpress.py` to a builder over the full kvpress catalog (ratio-only presses, wrapper presses, special-arg presses). **Smoke-gated** all 21 new candidates (`sm2_*`/`sm3_*`, RULER-4k N=12) before committing.
- **Faithful & reproduced (20):** snapkv, h2o, streaming, expected, tova, knorm, criticalkv, cur, kvzip, fastkvzip, lagkv, leverage, keydiff, compactor, noncausal, rerotate, block, random(control) + adapted: think (channel-pruning axis), simlayer (threshold, ratio N/A).
- **Excluded w/ documented reasons (8):** pyramidkv (shape crash), adakv/critadakv (eager-mode unsupported), qfilter (Q-filters not published for Qwen3-8B = base-specific), duoattention (needs offline head masks; on-the-fly OOMs), finch (needs delimiter token), kvzap (attr error), chunk/chunkkv (run clean but 0.0 untuned; `block` is the working contiguous-region rep).
- **Benchmarks expanded:** + numerical/categorical/coding NIAH, quality(MC), musr_mm(MC), locomo. multi_needle stays excluded (scoring `full`=0.0 bug).
- **Matrix (177 cells, NVAL=48 KV / 64 refs):** A) master budget table 19 methods × 5 ratios on RULER-8k + simlayer×1; B) 20 methods × {narrativeqa, quality_hard, numerical_niah} @0.5; C) refs (full / LLMLingua-2 / window-1024) × 7 expanded benches.
- **Exec:** round-robin ~30/GPU → `q_cat_0..5`, `runner_u.sh`; all 6 GPUs live 09:18–09:19. Est ~7–8h. Results → `c_*` log lines.
- **Docs:** wrote `baseline-catalog-faithfulness.md` (full method-by-method faithfulness table w/ paper links + annotated diffs). Next: harvest `c_*` → append master budget table + method×bench to `baseline-factbase-v2.0.0.md`.

## 🔬 D15 — Catalog complete + necessity length-sweep (method-agnostic v2.0.0) (user 11:59 Jun 25)
- **Catalog sweep (D14): 177/177 complete** (the earlier "119" was mid-drain `.json.done` bookkeeping; all produced RECIPE_EVAL). Appended as **§8** of `baseline-factbase-v2.0.0.md` (master budget table 20 methods × 5 ratios; method×bench; refs on expanded benches).
- **New finding:** **kvzip / fastkvzip are the most compression-robust** — kvzip holds 0.83 on RULER-8k even at r0.9 (90% evicted) where snapkv/knorm collapse to ~0.01. Query-agnostic reconstruction-based eviction >> attention/norm-based at extreme budgets.
- **Bug found + FIXED:** `run_baseline.py` scored **0.00 on numerical/categorical/coding NIAH** (full included) — missing the gold-substring fallback that `run_kvpress` has (score_gen exact-match fails on these variants). Added `_sc()` fallback (count hit if gold string appears verbatim in generation). Compiled + copied to both pods.
- **Necessity length-sweep launched (78 cells, method-agnostic — NO GCM method/training):** feasible no-compression + prompt-compression baselines × length = {truncated-full@8k, recent-window w1024, untruncated-full, LLMLingua-2 r0.5} × {ruler/numerical/categorical NIAH} × {2k,4k,8k,16k,32k}, + best KV {knorm/streaming/snapkv}@0.5 × {numerical,categorical}×{4k,8k,16k}. This is the **C1 crossover figure** (where truncation/window lose as length grows). All 6 GPUs (d1525×4+d1530×2) live 12:09–12:10; relaunched once after the scoring fix. d1420 GPU0 free, GPU1 = the demo (full-report, compress 0.263).
- **Demo note:** the cmg demo regression (0.26→0.13) was diagnosed: reports up to 28.6k tok but `ENC_MAXCTX` defaulted to 8192 → compressor saw only first 8k. Restored by serving the **full-report instance (ENC_MAXCTX=32768)** → compress **0.263** > trunc 0.211 > no_ctx 0.158. Data unchanged (md5 identical across pods, untouched since Jun 21). Module name confirmed masked (0/38 in evidence; paths show `/MODULE/` placeholder) — correct for RCA.

## 🔬 D16 — More baselines: BM25 RAG + necessity length-sweep fixed (user 15:01 Jun 25)
- **Bug fixed:** `run_baseline.py` **hardcoded `n_chunks=8`** in `load_items` → the necessity length axis was dead (flat curves). Now reads `GCM_NCHUNKS`. Re-ran necessity sweep → length axis works.
- **Necessity result (the C1 evidence):** window-1k **collapses with length** (RULER 0.44→0.59→0.16→**0.03** at 2k/4k/8k/16k; numerical 0.34→0.05), trunc-8k **drops at 16k** (RULER 0.92→0.55; numerical 0.66→0.27→0.12 at 8k/16k/32k), while **full and LLMLingua-2 hold** (~0.8–0.98). 6 cells at 32k-untruncated **OOM'd** — itself necessity evidence ("full read doesn't fit at 32k").
- **NEW baseline implemented — BM25 RAG** (`GCM_BASELINE=rag`, dependency-free pure-python BM25; pods have no sentence-transformers/faiss/bm25). Retrieves top passages within a token budget (`GCM_RAG_BUDGET`), frozen base reads them. The key **retrieve-vs-compress** comparison (v2.0.0 plan explicitly wanted RAG).
- **RAG early result:** **holds across length** (RULER 2k/4k/8k = 0.77/0.98/0.85, no_ctx=0) because BM25 finds the needle regardless of total length — a strong necessity-regime competitor (sharpens the compress-vs-retrieve story).
- **RAG sweep launched (22 cells, 8 GPUs** d1525×4+d1530×2+d1420×2**):** RAG×length×{ruler/numerical/categorical}, RAG budget curve {512/1024/2048/4096} on 16k, RAG on real benches {narrativeqa/hotpot/squad/trivia}. Results → `rag_*` logs.
- Deferred (need a 2nd small LM / network): LongLLMLingua, original LLMLingua (perplexity), Selective-Context.

## 🔬 D17 — Paper training program T1 launched: method A/B + domain-transfer matrix (user 15:25 Jun 25)
- **Goal:** "complete all experiments the paper needs." Launched the method-related training spend (the eval/baseline side is done: catalog 177, necessity, RAG).
- **T1 = 14 training cells (run_recipe), 8 GPUs (d1525×4+d1530×2+d1420×2):**
  - **Method A/B (elegance validation, M1+M5):** {pruned 2-loss core (recon+distill+joint, dev/contrast/vae/adv/enc-lora=0, hard-norm, varlen) vs current kitchen-sink} × {Qwen3-8B, Qwen3.5-9B}.
  - **Domain-transfer matrix (C2):** train pruned M on 5 source domains {tool, wiki, narrative, ops, web} × 2 bases, eval each on 5 targets {bfcl, squad, narrativeqa, quality, rca_openrca} → 5×5 transfer matrix per base.
- **Two bugs fixed mid-launch:** (1) **Qwen3.5 training OOM** at MAXCTX=8192+recurrent encode → dropped train MAXCTX to **4096** (eval still on benches). (2) **contrastive/adv crash on GDN** (`hidden_states[adv_layer]` IndexError — GDN returns fewer states) → **clamped layer index** in `train.py` (deployed to 3 pods). *Note: the kitchen-sink config literally crashed on GDN while the pruned core didn't — concrete support for the elegance/pruning thesis.*
- **Status:** all 8 cells running, **0 errors** post-fix; q35 cells ~90 GB (tight but stable). Est ~1–2h. Results → `t1_*` logs; matrix row per `t1_mx_*` cell.
- **Also done this session:** BM25 RAG baseline (D16) + necessity length-sweep (D16) appended to `baseline-factbase-v2.0.0.md` §9; `run_baseline` NCHUNKS + NIAH-scoring fixes.
- **d1420 setup:** copied `runner_u.sh` + `run_recipe.py` there (was the vLLM pod, never grid-enabled).

## 🔬 D18 — Overnight 10-story baseline-diagnosis campaign (341 cells, ~10h, 8 GPUs) (user 23:06 Jun 25)
- **Goal:** find *why* each baseline fails; 10 stories × ≥20 angles; report next day. Free use of 8 GPUs.
- **T1 harvested first:** 13/14 (t1_mx_narr_q35 crashed). **Elegance A/B validated:** pruned compress ≈ current (q38 0.197 vs 0.207; q35 0.194 vs 0.189, Δ≈0.01) ⇒ dropping dev/contrast/vae/enc-lora costs ~nothing. Transfer-matrix rows captured (tool-trained M transfers best; per-bench in the json for the 5×5).
- **Campaign = 341 cells, stories S1–S10** (design in `baseline-diagnosis-campaign.md`): S1 attn-KV length-collapse, S2 attn-free length-robust, S3 task-flip, S4 ratio-cliff, S5 window-locality, S6 full-hurts (incl rca distractor-count), S7 LLMLingua denoise/drop, S8 RAG length-robust/abstractive-fail, S9 GDN-q35 no-KV-lever, S10 broken/base-specific presses. Tags `s1_…s10_`.
- **Exec:** `q_diag_0..7`, 8 GPUs (d1525×4+d1530×2+d1420×2), launched 23:11–23:13. Heaviest cells (16k-eager) ~92 GB — OOM risk on those; runner is per-cell resilient (no `.done` on fail → re-run at harvest).
- **Next:** when drained, compile the detailed report (per-story evidence table + root-cause) reusing catalog/necessity/RAG/T1 data for already-covered angles.

## 🔬 D19 — Diagnosis report + visualization + insights-validity + dive/precision waves (user Jun 26)
- **Campaign harvested** (310/341; d1420 was missing `run_kvpress.py`+`kvpress` → fixed, re-running the 62). **Dive-into wave (100/100 done).**
- **Report:** `baseline-diagnosis-report.md` (10 stories, root causes). **Figures:** `figures/fig1–5*.png` (matplotlib): length-collapse, ratio-cliff, task-flip heatmap, distractor-causal, RAG-vs-full. **Insights map:** `insights-longcontext-validity.md` (8 insights × precise scenario boundary).
- **Dive corrections (precision):** (1) **kvzip cliffs at ~0.9** (0.83→0.08→0.00 at 0.9/0.95/0.99), not "no cliff"; robustness is **retrieval-only** (narrativeqa 0.14 @0.9). (2) **Attention cliff is a sharp ~16k threshold** (snapkv flat to 14k=0.62 → 0.17 @16k). (3) **RAG-hurts-on-QuALITY is budget/chunk-invariant** (relevance problem). (4) **"Full hurts" is literary-MC-specific, not distractor-driven** — on rca full/RAG are flat vs 0→8 distractors while **window collapses (0.78→0.17)** and LLMLingua degrades.
- **Precision wave (30 cells) launched** (d1525+d1530): pin the 16k attention cliff (ratio-dependence + h2o/tova), attention-free beyond 32k (40k), kvzip cliff localization {0.86–0.95} + retrieval-specificity, distractor×KV.

## 🔬 D21 — Faithfulness audit of public baselines + exact (Long)LLMLingua added (user Jul 6)
- **Requirement:** every *public published* method must be reproduced **EXACTLY** (the original), not our-infra approximations.
- **Audit verdict (see `baseline-catalog-faithfulness.md`):**
  - **EXACT:** all ~20 KV methods (official NVIDIA `kvpress` classes, paper-default params) + **LLMLingua-2** (authors' model/lib). Reportable as the published methods.
  - **Generic (cite concept, not a system):** `full`, `no_ctx`, `sft-lora`, **`window`** (relabeled: sliding-window READ, **NOT StreamingLLM** — exact StreamingLLM = the `streaming` kvpress press), **`BM25-RAG`** (lexical-retriever RAG, not DPR-RAG).
  - **NOT faithful (flagged, must not be cited as the paper):** `gist-lite` (Gisting) and `cart-lite` (Cartridges) — they reuse OUR `compressor.mem`+LoRA (the *idea* in our framework). To use as baselines → run authors' code, else keep as internal ablations only.
  - Fixed the misleading `txl` code comment; corrected catalog labels + added a top "faithfulness verdict".
- **Continue (exact public methods added):** wired **LongLLMLingua** + **original perplexity-LLMLingua** via the official `llmlingua` pkg (`use_llmlingua2=False`, question-aware) with the authors' compressor LM **NousResearch/Llama-2-7b-hf** (downloaded). Smoke OK (squad@0.5: longllmlingua 0.41, llmlingua_orig 0.60, full 0.74). Launched a 30-cell faithful wave (2 methods × 3 rates × 5 benches) on d1525.
  - **Selective-Context:** install **blocked** (its `spacy` build fails in the venv) → deferred, noted honestly.

## 🔬 D22 — Faithfulness/traceability audit + MC-scoring fix (user Jul 6)
- **Full code re-check** → doc `faithfulness-and-traceability-audit-2026-07-06.md`. Verdict: KV methods (kvpress official + paper defaults), LLMLingua-2/Long/orig (official), BM25-RAG, `mc_loglik` (length-normalized) all **faithful**; ToMe is **input-side (adapted, not layer-internal)**; gist/cart NOT faithful; txl≠StreamingLLM (already flagged).
- **KEY BUG found:** MC scoring **inconsistent across harnesses** — `run_baseline` uses **loglik** (`model.py:128`), `run_kvpress` used **generate-4-tokens→extract-letter** → KV-method MC (quality/quality_hard/musr) **deflated & not comparable**. Confirmed: snapkv/quality **0.33 (loglik-fix) vs ~0.12 (old gen)**.
- **Fixed:** added `mc_loglik_kv` to `run_kvpress` (prefill ctx under the press → score option-letter loglik with the compressed KV). Deployed to pods. **Re-scoring wave** (8 KV methods × {quality, quality_hard, musr_mm}) launched on d1530; hh quality-KV `.done` cleared to re-score.
- **Affected facts:** only **KV-method MC numbers** need re-scoring (F3 `quality_hard` column, head-to-head `quality` column). **F6 (full-hurts) SAFE** (full vs no_ctx both loglik, same harness); all NIAH/gen facts (F1–F5,F7–F17) unaffected.
- **Traceability closed:** 128 per-cell config queue files + `runner_u.sh` committed to `results-v2.0.0/configs/` — every `RECIPE_EVAL <tag>` ↔ config ↔ log.

## 🧪 D23 — kvzip repeat-prompt export of soft memory → Paper A (user Jul 6)
- **Code:** added `GCM_RECON_MODE=repeat` → `model.recon_repeat()` (`model.py`): prompt the FROZEN base to REPEAT ctx from **M alone**, teacher-forced, one forward → returns `L_repeat` (mean NLL, trains M as a sufficient statistic, grad→mem+proj) **and per-token `r_t`** (= kvzip reconstruction-importance of ctx token t **w.r.t. our M**). Default **query-agnostic (M0)**, matching kvzip. Wired new label-free gate signals `neg_recon_repeat`, `neg_maxr` into `gate_signal` + `_val_metrics` (`gate/auroc_recon_repeat`) + `run_recipe` `SIGNALS`. Compiles, lint-clean.
- **Ran (`paperA_kvzip`, Qwen3-8B, K256, squad_v2+quality, 300 steps, repeat-recon):**
  - **Gate works:** honest 5-fold-CV `gated_acc_cv=0.469` vs `full=0.409` → **+6.0 pts do-no-harm**, routing per-item (quality→trust compress 0.362>full 0.125 [F6]; squad→fall back to full 0.693).
  - **Per-signal AUROC (predict compress-correct):** `conf 0.638` > `neg_recon/neg_recon_repeat 0.564` > `margin 0.436` > `neg_maxr 0.430` > `neg_entropy 0.318`. **CV picked `conf` in all 5 folds.**
  - **⚠️ Confound found:** with `GCM_RECON_MODE=repeat` set globally, `reconstruct()` delegates to `recon_repeat`, so **`neg_recon` (slot) == `neg_recon_repeat` (kvzip)** this run (identical 0.564) → this run **cannot** separate slot-vs-kvzip. Also **underpowered**: 300-step K256 M is weak (compress 0.22 ≈ no_ctx 0.19) → recon-fidelity signals uninformative.
- **Honest verdict:** kvzip-as-**gate-signal** is implemented & measurable but **does not beat `conf`** on this underpowered M. Its real Paper-A promise is **(a) importance-weighted reconstruction** (use `r_t` as per-token loss weights so M preserves the un-repeatable needle) and **(b) a well-trained M** where recon-fidelity varies meaningfully. **Next (await user):** clean A/B — slot-recon vs repeat-recon TRAINING (longer, K≤128) + importance-weighted recon + `r_t` needle-AUROC (does r_t localize needles when M is good).

## 🧪 D24 — clean A/B: repeat-prompt reconstruction as a composable loss (user Jul 6)
- **Code:** added composable `lam_repeat` (env `GCM_REPEAT`) in `train.py` — a SEPARATE additive term calling `recon_repeat`, so it stacks WITH slot-recon + VAE (fixes D23 confound: Arm A runs `GCM_RECON_MODE=slot`, so `neg_recon`=slot and `neg_recon_repeat`=kvzip are now distinct signals). Compiles, lint-clean.
- **Ran properly-trained A/B** (Qwen3-8B, K128, 2000 steps, squad_v2+hotpot+quality, VAE+slot+KL both arms; B adds `GCM_REPEAT=0.5`):
  | | compress | honest CV gate vs full | recon_repeat AUROC | conf AUROC |
  |---|---|---|---|---|
  | A: VAE+slot | 0.277 | **−1.1%** | 0.545 | 0.617 |
  | B: +repeat | 0.276 | **+0.6%** | **0.581** | 0.649 |
- **Findings:** (1) repeat loss is **accuracy-neutral** for the compressed path (0.277→0.276); (2) it **improves the do-no-harm gate** — the reconstruction signal's AUROC rises 0.545→0.581 and the honest held-out gate flips from below-full to above-full; (3) **`conf` is still the strongest single signal** (0.649) → combine conf+recon; (4) still moderately underpowered (compress 0.28 vs full 0.44) + small N → deltas suggestive, not conclusive.
- **Verdict for Paper A:** the repeat-prompt reconstruction earns its place as the **self-verification/gate** mechanism (its theoretical do-no-harm story + calibrated signal), not as an accuracy booster. Next: higher-K/longer train to de-risk the underpower, and a conf+recon combined gate.

## 🐞 D25 — fine-grained ablation, a deploy bug, and the fix (user Jul 6)
- **Goal:** fine-grained ablation to show the repeat-recon method is not tricky — loss-component add-one-in, λ dose-response {0,0.1,0.25,0.5,0.75,1.0,2.0}, prompt variants {default, word-for-word, empty, generic}, cond {M0,Mq}, K {64,128,256}, and a **random-M leak control** (`GCM_GATE_RANDM`: recompute recon with a Gaussian M matched to M's stats; if AUROC(rand)≈AUROC(real), the signal is ctx-intrinsic = trick).
- **🐞 DEPLOY BUG (found + owned):** I edited `model.py`/`train.py`/`run_recipe.py` but did **not re-sync to the pods** before launching. Result: **d1525 was fully stale** (no `recon_repeat`/`lam_repeat`) → the 11 finer cells trained with `GCM_REPEAT` **silently ignored** (no repeat loss) → **invalid**; **d1530 had the repeat loss but not the random-M code** → `randAUC` empty. The valid-at-the-time d1530 cells (A/B, λ0.25/1.0, Mq, prompt-word, K-sweep) did train repeat correctly.
- **Early (valid, d1530) finding:** repeat-recon gate AUROC is **noise-level and non-monotonic** across λ (λ0=0.545, λ0.25=0.455, λ0.5=0.581, λ1.0=0.564) at K128/2000-step; compress acc swings 0.19–0.31; **`conf` is the only stable gate (0.61–0.66)**. → at this budget the "repeat helps the gate" effect is within noise.
- **Fix:** redeployed all three files to BOTH pods (verified `recon_repeat`/`lam_repeat`/`rand` present), killed the stale d1525 workers, wiped the stale finer logs + lock, and **relaunched**: d1525 = 11 finer cells (correct code + random-M), d1530 = clean λ dose-response {0,0.25,0.5,1.0} (+random-M). 
- **Provisional verdict (to finalize on drain):** not "tricky" in the hidden-hack sense, but the gate benefit is **not robust at this training budget** — a **properly-trained (higher-K / longer) ablation is required before any paper claim**. Matches the user's standing "no underpowered results in the paper" directive.

### ❌ D25-FINAL — the random-M leak control kills the repeat-recon-as-gate claim
Corrected λ dose-response (K128, correct code, +random-M), the decisive numbers:
| λ | compress | gate-CV | repeat-recon AUROC | **random-M AUROC** | conf AUROC |
|---|---|---|---|---|---|
| 0.0 | 0.247 | 0.421 | 0.498 | 0.481 | 0.655 |
| 0.25 | 0.302 | 0.467 | 0.523 | 0.516 | 0.644 |
| 0.5 | 0.262 | 0.417 | 0.464 | 0.512 | 0.614 |
| 1.0 | 0.293 | 0.479 | 0.493 | 0.539 | 0.654 |
- **`repeat-recon AUROC ≈ random-M AUROC` at every λ, and both ≈ chance (0.5).** Component cells agree (answer/slot/VAE/repeat-only: real≈random≈0.48–0.53). ⇒ the repeat-recon signal is **NOT using M's content** — it reflects **context-intrinsic reconstructability**, not memory quality. The earlier D24 "+0.036 AUROC / +6pt" was **noise** (didn't survive the control or the dose-response).
- **Verdict:** at this budget the **repeat-prompt reconstruction is a non-effect as a do-no-harm gate signal** (fails the leak test) AND accuracy-neutral (D24). It must **not** be claimed as a Paper-A contribution as-is. `conf` remains the only real gate signal (~0.61–0.66).
- **Options:** (a) **drop** repeat-recon from the gate story; keep the confidence/margin gate for Paper A; (b) **re-test on a properly-trained high-K model** where M actually carries the answer, to see if real-M then separates from random-M (the leak control becomes the acceptance criterion). Recommend (b) as the decisive test before deciding; do NOT ship (a-)the claim meanwhile.
- **Full 15-cell confirmation (all done, 0 fails):** across the component add-one-in, λ∈{0,0.1,0.25,0.5,0.75,1.0,2.0}, prompt {default/word/empty/generic}, cond {M0/Mq} cells, `real-M − random-M` AUROC = {−0.031,−0.051,−0.056,−0.043,+0.053,+0.045,−0.016, +0.017,+0.007,−0.048,−0.046,…} → **bounces around 0 (mean ≈ −0.01), never a consistent positive gap**. Decisive: repeat-recon adds nothing over a random memory at this budget. Awaiting user decision on the high-K decisive re-test (option b) vs drop (option a).

## 🔬 D26 — high-K decisive re-test result (repeat-recon): mixed, leaning negative
Bracket (Qwen3-8B, VAE+slot+repeat, +random-M control), `real-M − random-M` gate AUROC for **+repeat** cells:
| cell | compress (full=0.44) | repAUC | randAUC | Δ real−rand |
|---|---|---|---|---|
| K256@4k +rep | 0.275 | 0.604 | 0.520 | **+0.084** |
| K256@6k +rep | 0.263 | 0.406 | 0.499 | **−0.093** |
| K512@4k +rep | 0.224 | 0.507 | 0.509 | **−0.002** |
- **CONFIRMED negative (all 3 done):** the +repeat `real−random` deltas **scatter around 0 (+0.084, −0.093, −0.002; mean ≈ −0.004)** → the lone K256@4k +0.084 was **noise**, not a reversal. **Confound:** compressed accuracy **never exceeds ~0.30 at any K** (M never compresses well) → real-M can't be memory-specific.
- **Final verdict:** repeat-recon is **not a usable do-no-harm gate**; the bottleneck is the **compressor training recipe, not K**. Ship the **confidence** gate for Paper A; reconstruction = hypothesis blocked on a stronger compressor. Overview §3.1(D) + Fig 12 + Paper A caveat finalized to this reading (I had briefly over-claimed the +0.084 mid-run; corrected).

## Standing rule (acknowledged)
From now on, **any decision I make autonomously (esp. code changes to shared repos, job kills, perms, data publication, anything destructive/sensitive) gets an entry here** before/when I do it. Anything destructive or that publishes data externally → I **ask first** and log the outcome.
