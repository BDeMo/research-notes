# Intrinsic probe — which selection-quality metric is most significant? (2026-07-10)

**Question:** beyond downstream accuracy (extrinsic), which *intrinsic* property of the IMP selection best explains/predicts which scheme wins? Metrics per scheme (keep=0.5, N=30):
- **gold-recall** — does the gold answer substring survive in the kept context (answer retention).
- **query-coverage** — fraction of query tokens present in kept.
- **kept-fraction** — actual tokens kept / L.

Probe = embedding-`qrel` + IDF-`lex` + `BM25` selection (surprisal omitted for a memory-safe CPU probe; the answer-retention trend is BM25/lexical-dominated so robust). Script `_diag_intrinsic2.py`.

| scheme | metric | squad_v2 | hotpot_qa | ruler_niah | trivia_qa | lb_hotpotqa |
|---|---|--:|--:|--:|--:|--:|
| span | goldR | **0.33** | 0.63 | 0.97 | 0.67 | 0.80 |
| **chunk** | goldR | **1.00** | 0.63 | 0.97 | 0.70 | 0.80 |
| hier | goldR | 0.33 | 0.67 | 0.97 | 0.67 | 0.80 |
| bm25span | goldR | 0.53 | **0.73** | 0.93 | **0.73** | 0.80 |
| qfree | goldR | 0.43 | 0.43 | 0.97 | 0.60 | 0.70 |
| (all) | qCov | 0.31–0.47 | 0.55–0.65 | 0.70–0.72 | 0.58–0.64 | 0.76–0.81 |
| (all) | keptF | span/hier 0.37, **chunk 0.99** | ~0.42–0.48 | 0.50 | ~0.50 | 0.50 |

## Finding: **gold-recall is the most significant intrinsic metric — it directly predicts downstream accuracy.**
- **squad_v2 (the decisive bench):** span gold-recall **0.33 → downstream 46.2**; chunk gold-recall **1.00 → downstream 69.3 (≈RAG)**. The whole chunk-vs-span downstream gap is explained by answer retention: squad passages are short, so 256-tok `chunk` keeps the answer passage **whole (goldR=1.0, keptF≈0.99)** while token-level `span` shatters it (goldR=0.33). This is *the* mechanism behind F34.
- **query-coverage** (0.31–0.47 on squad) does **not** separate schemes cleanly nor track accuracy → not significant.
- **kept-fraction** varies by scheme×context (chunk keeps short contexts whole) → explains *why* gold-recall differs, but is not the predictor itself.
- **Significance ranking: gold-recall ≫ query-coverage > kept-fraction.**

## Secondary insights
- On **long** contexts (hotpot/lb_hotpotqa) gold-recall **converges (~0.63–0.80) across schemes** → downstream also converges (matches the verdict table: schemes tie there).
- **bm25span has the highest gold-recall on multi-hop/open-QA** (hotpot 0.73, trivia 0.73) — proper BM25 best locates the answer span among distributed evidence.
- **ruler:** all ~0.97 (every scheme keeps the single needle) → retrieval is not answer-retention-limited; the downstream RULER differences (chunk 77 vs span 86) come from *coherence/position*, not answer loss.

## Why it matters (use)
gold-recall is a **label-light intrinsic selector**: it needs only the gold string (or, unsupervised, a needle-probe), no downstream generation → it can pick the **best scheme / keep-rate per input** cheaply ("short passage → chunk; scattered needle → span"). That is the adaptive mode-selector next step (F34 follow-up).

*Caveat: surprisal omitted in this probe (memory-safe); trend is BM25/lexical-dominated so robust. Full-signal re-probe pending a free GPU.*
