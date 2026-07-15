# Correctness audit — module I/O, naive issues, correct settings (2026-07-13)

Triggered by F44 (a truncation asymmetry that inverted a headline). Systematic check of every module's input/output, then re-run the main table with the corrected "most-correct" setting.

## 1. Module input/output audit
| module | input it sees | budget | forward pass? | naive issue? |
|---|---|---|---|---|
| `query_ids` (all methods) | question **+ options (MC) / question (extractive)** — verified: ∞Bench query includes `A) Everingham …` | — | no | **OK** — options ARE in the retrieval query (both IMP & RAG match against them) |
| RAG (`_rag`) | **FULL untruncated doc** → BM25 128-tok passages → keep 2048 | 2048 | no | fair (full-doc access is the point of retrieval) |
| IMP chunk/bm25span/hier | was truncated→**now FULL doc (F44 fix, `FULLDOC=1`)** | 0.5·min(L,MAXCTX) | no | **FIXED** |
| IMP span/qfree | first MAXCTX only | 0.5·MAXCTX | **yes** (surprisal) | inherent — a forward can't span 131k; router now avoids span when doc>MAXCTX |
| `full` (reference) | first MAXCTX only | MAXCTX | yes | legitimate cap (can't fit 131k); state "full is truncated, retrieval is not" |
| `no_ctx` | query only | 0 | yes | ok (lower bound) |
| window/txl | a contiguous MAXCTX-window | window | yes | truncation by design (F5) |
| ll2 / tome / kvzip / knorm | first MAXCTX (forward/KV bound) | keep 0.5 | yes | inherent; their ∞Bench numbers are prefix-only (their real property) |

**Key finding:** on docs longer than MAXCTX (only ∞Bench in our suite: median 131k), **only RAG and IMP-chunk-FD access the whole document**; every forward/KV-based method (full, window, tome, kvzip, knorm, IMP-span) is capped at the first MAXCTX. This is (a) the true F44 root cause and (b) a real, disclosable method property — not a bug for the forward-based methods, but it **must be stated** so ∞Bench comparisons aren't read as "same information".

## 2. Naive issues found
| issue | status |
|---|---|
| IMP truncated before selecting (chunk family) | **FIXED** — `GCM_IMP_FULLDOC=1` retrieves over full doc (F44) |
| `auto` mc-router sends huge-doc MC (∞Bench) → span → truncated → 52 | **FIXED** — router now: **doc>MAXCTX → chunk-FD** (span can't see it all); else options→span, else→chunk |
| query_ids missing options for MC | **not an issue** — options verified present |
| gen gold-substring fallback may inflate absolute F1 | known (P4) — applied to ALL methods uniformly → rankings valid; disclosed |
| scorer-limited LB gen tasks | known (F40) — excluded |

## 3. "Most correct" setting (used for the re-run)
- **IMP** = `auto`, router = {doc>MAXCTX → chunk full-doc; else options→span; else chunk}, `signal=all`, span 32 / chunk 256, keep 0.5, **FULLDOC=1**.
- **RAG** = BM25 full-doc, budget 2048.
- **full/no_ctx** = MAXCTX / none (unchanged references).
- **Scoring** = mc_loglik (MC, length-norm letter LL) / token-F1 (extractive, uniform gold-substring fallback) — unchanged.
- **N** = FULL for long benches, 2000 for short-QA (disclosed). MAXCTX 16k (8k for QuALITY/MuSR, 4k short-QA), triviaqa/repobench 12k.

## 4. Re-run & validity
- **Re-run:** `mt_*` grid (d1525×3 + d1530×2) = IMP(auto-FD) + RAG across 16 discriminative benches, full N. Refs {no_ctx,full} recomputed per cell.
- **Which prior numbers change:** only **∞Bench** materially (IMP ~50 → ~76; and the corrected auto now routes it right). All ctx≤MAXCTX benches (squad/hotpot/trivia/ms_marco/most LongBench/QuALITY/MuSR/RULER) are **unchanged** (verified: RAG-beyond-MAXCTX ≈ 0 there; squad chunk-FD 69 ≈ 66 sanity).
- **Baselines window/ll2/tome/kvzip/knorm:** unchanged (the fix is IMP-specific); their E1 full-test numbers stand. Their ∞Bench values remain prefix-only (their real property, now disclosed).

→ Corrected numbers fold into [`main-table-fulltest.md`](main-table-fulltest.md) on harvest; facts F30/F39 flagged 🔧 with pointers to F44.
