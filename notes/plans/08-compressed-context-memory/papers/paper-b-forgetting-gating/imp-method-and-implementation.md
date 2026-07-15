# IMP — method spec & implementation details

> **Version: `IMP-v2.1.1`** (span-level, Mode A, span=32, keep=0.5, signals=**{query-relevance, surprisal, IDF-lexical}**, `GCM_IMP_SIGNAL=all`) — adds the IDF-weighted lexical query-match signal (F33) that closes most of the QA gap to RAG. `IMP-v2.1.0` = {query, surprisal} only; `IMP-v2.0` = token-level (retrieval-only), superseded.
>
> **IMP = Importance-routing Prefilter.** A lightweight, architecture-agnostic structure that attaches to a **frozen** base LLM: score every context token by cheap O(L) signals, **keep the most-important spans verbatim**, drop/merge the redundant rest, and let the frozen base read the shortened sequence. Two deployment modes: **plug-and-play (no training)** and **light-weight training** (a tiny module; base never retrained).
>
> This doc is the precise algorithm + the exact code path (`run_baseline.py`, mode `imp`). Results: fact-base §12 / §12.2, Fig 13, F24/F25.

---

## 1. Problem it solves
Long-context accuracy is bottlenecked by *finding the important information*. Every training-free baseline fails in a characterizable regime (F18): attention-KV eviction collapses at ≥16k (F1), naive token-merging (ToMe) destroys the needle (F14, 0.00 on retrieval), RAG needs lexical overlap (F9). IMP is the response: **decide, per input, which tokens to keep — using the base's own cheap signals — before the expensive read.**

## 2. Interface
- **Input:** context token ids `c[1..L]`, query ids `q`, keep fraction `p ∈ (0,1)`, frozen base `F`.
- **Output:** a shortened embedding sequence `M = embed(c[S])` where `S ⊆ {1..L}`, `|S| ≈ pL`, fed to `F` in place of the full context. No weights of `F` change.

## 3. Importance signals (cheap, O(L), attention-free where possible)
For each context token `t`:

**(a) Query-relevance** — cosine of the token embedding to the mean query embedding:
$$\text{qrel}(t) = \frac{e(c_t)}{\lVert e(c_t)\rVert}\cdot \frac{\bar e_q}{\lVert \bar e_q\rVert},\qquad \bar e_q=\tfrac1{|q|}\sum_j e(q_j)$$
Cost: embedding lookup + dot product, **O(L·d), no transformer forward** → truly prefill-free. Best on word/phrase needles (AUROC 0.95, F20).

**(b) Surprisal** — negative log-prob of the token given its prefix, from **one teacher-forced forward**:
$$\text{surp}(t) = -\log p_F(c_t \mid c_{<t})$$
Cost: one forward over the context (O(L²) on quadratic attn, O(L) on linear). Best on high-information/numeric needles (AUROC 0.84, F20). *(See §7 for the prefill caveat and the small-scorer fix.)*

**(c) Lexical query-match (IDF-weighted, added in `IMP-v2.1.1`)** — the ingredient that made RAG strong (F30/F33). For each token that also appears in the query, weight it by an **inverse-context-frequency** IDF so common boilerplate is down-weighted and rare, discriminative query terms are boosted:
$$\text{lex}(t) = \mathbb{1}[c_t \in q]\cdot \log\frac{L+1}{\text{tf}(c_t)+0.5}$$
This directly fixes the CMG/QA failure mode where `qrel` locked onto generic headers and `surp` onto high-entropy noise (diagnostic: [`dive-in-imp-weakness-and-baselines.md`](dive-in-imp-weakness-and-baselines.md)).

**(d) Combination** — z-score each, then sum (no single signal wins across tasks, F3/F20):
$$s(t) = z(\text{qrel}) + z(\text{surp}) + z(\text{lex})$$
Switch via `GCM_IMP_SIGNAL ∈ {query, surprisal, both, lex, qlex, all}`. **`all` (query+surprisal+lex) is the recommended default (v2.1.1)** — it closes most of the QA gap to RAG (lb_hotpotqa 17.4→24.1≈RAG; hotpot 42.5→49.6) **without harming retrieval** (ruler 95). `lex`-only is best on high-lexical extractive (squad 51.6) but hurts pure retrieval (ruler 92), so `all` is the robust choice. (F33)
*(Extensible: add reconstructability (KVzip-style, F4) and redundancy (F16) as extra z-scored terms.)*

## 4. Selection — token-level vs span-level
Budget: `keep = max(8, ⌊p·L⌋)` tokens.

- **Token-level (`SPAN=1`):** keep the top-`keep` tokens by `s(t)`, restore reading order.
  → Perfect on isolated-needle retrieval (ruler 1.0), **but shatters coherent QA** (squad 0.15 < no_ctx) because scattered tokens lose sentence structure (F24).

- **Span-level (`SPAN=w`, default w≈32) — the general method:** split the context into contiguous `w`-token spans; score each span by its **max** token score; keep the top `⌊keep/w⌋` spans **whole**, in reading order.
  → Keeps the needle's *span* intact (retrieval stays ≈ full) **and** preserves the answer sentence (QA rescued: squad 0.46, hotpot 0.42; even beats full on trivia 0.75>0.72 & categorical 0.98>0.88 by denoising). F25.

Pseudocode:
```
score s[1..L]                                   # §3
if SPAN <= 1:
    S = indices of top-(keep) of s              # sorted ascending (reading order)
else:
    B = ceil(L / SPAN) spans (contiguous)
    span_score[b] = max_{t in span b} s[t]       # max-pool
    keep_spans = top-(keep // SPAN) of span_score
    S = concat(all token indices of keep_spans)  # sorted → contiguous, order-preserving
return embed(c[S])
```

## 5. Two deployment modes
- **Mode A — plug-and-play (no training):** use the base's own signals directly (§3), frozen base reads `embed(c[S])`. *This is what's implemented & evaluated.*
- **Mode B — light-weight training (base frozen):** add and train ONLY a tiny module:
  - a **distilled importance-router head** that learns the signal combination (on a small unlabeled corpus, forward-passes only — FastKVzip-style <1 H100-hr);
  - for **linear/GDN bases**, a **verbatim side-cache** of the very-top tokens (+ a brief LoRA) to counter state-collapse (F10) where a fixed recurrent state can't hold the needle even if selected.

## 6. Why it fits the design constraints
Frozen base (pluggable) · training-free or tiny module (frugal) · O(L) prune before the O(L²)/state read (**cuts long-prefill**, with the §7 caveat) · input-side prefilter works on **linear + quadratic** (no KV cache needed) · ships base + tiny module (**fast-reproducible checkpoint**) · one-sentence rule (simple, insightful).

## 7. Cost & the prefill caveat (important, honest)
- `qrel` is prefill-free (embeddings only). `surp` needs **one forward over the full context** — on a *quadratic* base that is exactly the O(L²) prefill IMP claims to save, so **surprisal-based scoring does not save prefill on quadratic bases** if it uses the full base.
- **Fixes (decouple scorer from reader):** (i) score with a **small draft LM** (or the base's first few layers) — like LLMLingua's small-LM perplexity — then the big base reads only the short sequence; (ii) use **only `qrel`** for a strictly prefill-free variant; (iii) on **linear bases** the scoring forward is O(L) anyway, so no order-of-magnitude conflict — there IMP's win is shortening the read + the side-cache.
- Research vs deployment: current experiments use the **full base** for surprisal to measure the *signal's upper bound*; the deployable version swaps in a small scorer (next experiment: measure accuracy drop under a small-LM scorer).

## 8. Exact code path (`experiments/run_baseline.py`, `GCM_BASELINE=imp`)
*Verbatim snapshot + all run configs archived in [`configs/`](configs/) ([`configs/IMP-v2.1.0.code.py`](configs/IMP-v2.1.0.code.py), [`configs/README.md`](configs/README.md)).*

Per item (frozen base `base`, embedding `embed`):
```python
ids = tok(context, add_special_tokens=False, truncation=True, max_length=MAXCTX)   # NOTE: MAXCTX = window; set 32768 for long-context, no artificial 4k cut
E   = embed(ids)[0].float()
qe  = embed(query_ids)[0].float().mean(0, keepdim=True)
qdot = cos(E, qe)                                   # query-relevance (no transformer)
logits = base(inputs_embeds=embed(ids)).logits[0]   # ONE forward
surp[1:] = -log_softmax(logits)[:-1].gather(ids[1:])# surprisal
z = lambda t: (t - t.mean())/(t.std()+1e-6)
sig  = GCM_IMP_SIGNAL                                # query | surprisal | both
score = z(qdot) if sig=="query" else z(surp) if sig=="surprisal" else z(qdot)+z(surp)
keep = max(8, int(LL_RATE*L)); SPAN = GCM_IMP_SPAN
if SPAN<=1: idx = score.topk(keep).indices.sort().values
else:       # span-level: max-pool per SPAN block, keep top blocks whole, order-preserving
    bs = pad(score, -inf).view(nb, SPAN).max(1).values
    blocks = bs.topk(max(1, keep//SPAN)).indices.sort().values
    idx = concat([arange(b*SPAN, min(b*SPAN+SPAN, L)) for b in blocks])
mpref = embed(ids[:, idx])                          # short seq → frozen base reads it
```
Knobs: `GCM_IMP_SPAN` (1=token, 32=span default), `GCM_LL_RATE` (keep p), `GCM_IMP_SIGNAL` (query/surprisal/both), `GCM_MAXCTX` (window; **32768 for long-context, no truncation**).

## 9. Ablations that pin down the design (fact-base §12.2, Fig 13)
- **token vs span:** span rescues coherent QA while keeping retrieval → span is essential (F24/F25).
- **keep-rate curve** (0.25/0.5/0.75): monotone → clean budget–accuracy trade-off.
- **span size** (16/32/64): insensitive (0.40–0.53); ~32 default; multi-hop prefers larger.
- **signal** (query / surprisal / both): both ≥ each single across tasks (query wins word-needles, surprisal wins numeric) → the combination is load-bearing.

## 10. Open items / next
- Small-LM (draft) scorer → recover the prefill saving on quadratic bases (§7).
- Add reconstructability + redundancy signals; learn the router (Mode B).
- Linear/GDN arm: side-cache + brief LoRA (Mode B); cross-architecture generality.
- Long-context suites at real length (LongBench / ∞Bench) where full context overruns the window — the necessity regime.
