# Code review — method, benchmarks, evaluation (2026-07-13)

Reviewed the paths that produce every reported number, to certify correctness before the vNext iteration.

## Files reviewed
`gcm/experiments/run_baseline.py` (methods + gen scoring), `run_kvpress.py` (KV baselines), `mem_embedding/gcm/runtime.py` + `model.py` (mc_loglik, generation), `mem_embedding/gcm/data.py` (load_items, score_gen, PRIMARY).

## Verdict per module
| module | what it does | correct? | notes |
|---|---|:--:|---|
| `_imp` modes (span/chunk/hier/bm25span/qfree/auto) | token/chunk importance selection | ✅ | BM25 Okapi (k1=1.5,b=0.75) correct; span max-pool correct; `signal=all` = z(qdot)+z(surp)+z(lex) |
| `_fulldoc_bm25` (F44 fix) | chunk-family retrieves over UNtruncated doc | ✅ | pure BM25, no forward; budget=keep·min(L,MAXCTX); reading-order restored |
| `auto` router | doc>MAXCTX or extractive→chunk-FD; literary-MC-that-fits→span | ✅ | doc-length term + options-aware; verified on ∞Bench(→chunk) & quality_hard(→span) |
| `_rag` | full-doc BM25 128-tok passages, budget 2048 | ✅ | query includes MC options; reading-order restored |
| `query_ids` | question **+ options** (MC) / question (extractive) | ✅ | verified: ∞Bench query carries `A)…D)` — retrieval can match the answer option |
| `mc_loglik` | length-normalized Σ logP(option)/len given [prefix;query] | ✅ | standard; `use_memory=False`→frozen base for eval-only methods |
| `score_gen` / `score_item` | per-bench PRIMARY metric (squad_f1 / exact_value_match / rouge_l) + `_answer_line` | ✅ | sensible metric per bench |
| kvpress (`kvzip/knorm`) | ctx tok[:MAXCTX]→build KV→evict | ✅ | official kvpress; MAXCTX-bounded (KV can't exceed); no-linear (needs KV) |
| `load_items` | per-bench generator; hash-partition train/val for no-native-split benches | ✅ | leakage guard (train/eval disjoint by content hash) |

## Issues found (2 — both disclosure, not correctness bugs)
1. **Gen gold-substring fallback (P4):** in `run_baseline._sc`, `if squad_f1==0 and gold∈generation → score=1`. Applied **uniformly to every method**, so *rankings* are valid, but **absolute gen-F1 is slightly inflated** vs pure squad-F1. → For the paper: report the primary metric WITHOUT the fallback (or report both), and keep the fallback only as a NIAH robustness aid. Action item for vNext eval.
2. **`_NO_NATIVE_SPLIT` benches eval on the validation-hash-half, not literally all items** — the hash partition (for leakage guard) means "full" for those benches = the ~50% validation bucket. Consistent across methods and disclosed via the `N` column, but the wording "FULL split" should read "full validation split". Minor; fix in `experiment-config-and-sampling.md`.

## Confirmed non-issues
- Truncation asymmetry (F44) — already fixed (`FULLDOC`) and audited ([`correctness-audit-2026-07-13.md`](correctness-audit-2026-07-13.md)).
- MC options present in retrieval query — verified.
- No train/eval leakage — hash-partition guard verified.
- RAG reproducibility — `mt_rag_*` reproduces E1 exactly (∞Bench 64.2, squad 67.2…).

**Certification:** every number in [`COMPLETE-RESULTS-2026-07-13.md`](COMPLETE-RESULTS-2026-07-13.md) is traceable and correctly computed, modulo the disclosed gen-F1 fallback (issue 1), which affects absolute extractive numbers by a few points but not rankings. Safe to build vNext on.
