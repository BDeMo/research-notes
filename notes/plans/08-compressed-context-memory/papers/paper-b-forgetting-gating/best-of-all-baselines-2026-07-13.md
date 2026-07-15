# Best-of-ALL-baselines comparison (not just RAG) — 2026-07-13

> ⚠️ **`full` = TRUNCATED to MAXCTX (16k)**, not the whole document (`embed(ctx[:,:MAXCTX])`). It equals *true full* only when a doc ≤ 16k — true for every bench here **except ∞Bench** (131k docs → `full` sees ~12%). Only `rag` and our `auto`(chunk) read the whole doc; `full/window/tome/kvzip/knorm` are MAXCTX-bounded. See [`correctness-audit-2026-07-13.md`](correctness-audit-2026-07-13.md).

Per user: don't only compare to RAG — compare to the **single best baseline per bench** across {full, window, rag, ll2(LLMLingua-2), tome(ToMe), kvzip, knorm} vs **our best IMP variant** {span/chunk/hier/bm25span/auto}. Baseline numbers from the full-test main table (E1, `main-table-fulltest.md`, Qwen3-8B, keep 0.5, full N); our best = max over IMP variants (E1 span + E5 redesign chunk/auto). All ×100.

⚠️ **kvzip/knorm are KV-cache-eviction methods → they do NOT run on cache-free linear/GDN models. RAG/ll2/tome/window/IMP are architecture-agnostic.** So two "best" columns below: **best-of-all** and **best-arch-agnostic** (the fair comparison set for IMP, which is arch-agnostic).

| bench | best-of-ALL (method) | best arch-agnostic (method) | **our best IMP** (variant) | ours − best-all | ours − best-agnostic |
|---|--|--|--|--:|--:|
| RULER-NIAH | 99.0 (kvzip=full) | 95.0 (ll2) | **96.8** (span) | −2.2 | **+1.8 win** |
| lb_narrativeqa | 14.0 (full) | 13.7 (rag/ll2) | **15.1** (chunk) | **+1.1 win** | **+1.4 win** |
| lb_musique | 11.9 (full) | 10.2 (knorm†) | **11.4** (chunk) | −0.5 | **+1.2 win** |
| lb_hotpotqa | 24.8 (rag) | 24.8 (rag) | 24.1 (span) | −0.7 tie | −0.7 tie |
| lb_multifieldqa | 38.8 (rag) | 38.8 (rag) | 38.6 (chunk) | −0.2 tie | −0.2 tie |
| lb_qasper | 23.8 (ll2) | 23.8 (ll2) | 23.6 (chunk) | −0.2 tie | −0.2 tie |
| ms_marco | 30.9 (rag) | 30.9 (rag) | 29.6 (chunk) | −1.3 | −1.3 |
| squad_v2 | 71.5 (rag) | 71.5 (rag) | 69.0 (chunk-FD) | −2.5 | −2.5 |
| trivia_qa | 71.4 (ll2) | 71.4 (ll2) | 68.5 (chunk) | −2.9 | −2.9 |
| MuSR (MC) | 58.9 (full) | 56.7 (rag) | 52.2 (bm25span) | −6.7 | −4.5 |
| lb_2wikimqa | 22.8 (full) | **22.4 (knorm†)** →16.0(rag agn.) | 15.8 (bm25span) | −7.0 | −0.2 tie |
| hotpot_qa | 56.2 (full/kvzip) | 55.0 (rag) | 45.6 (chunk) | −10.6 | −9.4 |
| ∞Bench-choice | **chunk-FD 76.0 (ours)** | **chunk-FD 76.0 (ours)** | **76.0 (chunk full-doc, F44 fix)** | **+11.8 WIN** | **+11.8 WIN** |
| **QuALITY-hard (MC)** | **29.9 (kvzip)** | 14.8 (tome) | 12.8 (span) | **−17.1** | −2.0 |
| LongBench-v2 (MC) | 34.2 (rag) | 34.2 (rag) | 32.8 (tome-side) | −1.4 | −1.4 |

†knorm/kvzip on lb_2wikimqa/lb_musique are KV-methods; the arch-agnostic best there is rag/full.
*∞Bench −12.7 is inflated by the F44 truncation artifact (IMP saw first 16k of 131k); full-doc re-run pending.

## Corrections this forces (important, honest)
1. **My earlier "IMP beats RAG on hard-MC" was misleading.** True vs RAG (12.8 > 9.6), but **on QuALITY-hard the best baseline is kvzip/knorm ≈ 30 — IMP loses by −17.** KV-eviction *denoises* literary-MC far better than text-side selection. IMP only "wins" hard-MC within the arch-agnostic set (and even there `no_ctx` 19.6 tops all methods). → the honest claim is **"context hurts on literary-MC; KV-eviction denoises best; IMP is not the answer there."**
2. **IMP wins outright on 3 benches** (lb_narrativeqa; RULER-among-agnostic; **∞Bench chunk-FD 76 > RAG 64–70 after the F44 fix** — the fix flipped a −13 loss into a win, though it should be re-confirmed at full N vs the N=100 measurement) and **ties on ~5** (lb_hotpotqa/multifieldqa/qasper/2wiki-agnostic, squad-close). It **loses** on hard-MC (kvzip −17), multi-hop hotpot (−10), MuSR (−5).
3. **IMP's real, defensible niche = the intersection:** (a) **lossless needle retrieval** where ToMe→0 / window→6 collapse (RULER 96.8, best arch-agnostic); (b) **runs on cache-free linear** where kvzip/knorm (the hard-MC winners) cannot; (c) a **single router** that is never far from the best arch-agnostic baseline across regimes. It is *not* a SOTA-accuracy method.

## Update: ∞Bench flips after the F44 fix
With full-doc retrieval, **IMP-chunk-FD = 76.0 > RAG 69.0 > full 57** on ∞Bench (CONFIRMED same N=100, same items) — the −12.7 "loss" becomes a **+7 win** once the truncation artifact is removed. So on ∞Bench IMP is now **best-of-all**, not behind. This strengthens the niche claim but the router must be fixed to *use* chunk-FD there (mc currently mis-routes MC→span).

## Tie-time: what else does IMP offer when it ≈ RAG? (honest)
When IMP **ties** RAG (squad/multifield/qasper/hotpot-lb), it is because **IMP-`chunk` IS BM25 passage retrieval** (256-tok chunks vs RAG's 128) — same mechanism, so **no free-lunch efficiency edge**:
| axis | IMP-chunk | IMP-span/qfree | RAG |
|---|---|---|---|
| selection cost | O(L) BM25 (= RAG) | **O(L²) forward for surprisal — MORE expensive than RAG** | O(L) BM25 |
| compression (keep) | 0.5 (same) | 0.5 | 0.5 (budget 2048) |
| query-agnostic reuse | no | only `qfree` (weakest acc) | no |
| runs on linear/GDN | yes | yes | yes |
| lossless needle | good | **best (RULER 96.8, tome→0/window→6)** | good (94.2) |
**Honest conclusion:** there is **no compelling efficiency advantage of IMP over RAG at tie**; IMP-chunk≈RAG *because it is RAG-like*, and IMP-span is *more* expensive (extra forward) with no QA benefit. IMP's only genuine differentiators are **(a) span-mode lossless-needle where merge/truncate collapse, (b) the single unified router spanning regimes, (c) ∞Bench-chunk-FD now > RAG.**

## Net (revised, honest)
Against **best-of-all-baselines**, the paper cannot claim IMP is most accurate. The defensible contributions are: **the diagnosis (F27/F37/F39/F43, model-invariant)** + **IMP as the one arch-agnostic router that (i) never collapses like ToMe/window, (ii) covers the linear regime kvzip/knorm can't, (iii) stays within ~0–3 of the best arch-agnostic baseline on most QA while being lossless on needles.** Where a KV cache exists and the task is literary-MC, **kvzip/knorm are simply better** — state this plainly.
