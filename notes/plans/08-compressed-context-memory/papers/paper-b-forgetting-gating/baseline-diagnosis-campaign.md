# Overnight Baseline-Diagnosis Campaign — 10 stories × ≥20 angles

> Goal (user 2026-06-25 23:06): use ~10h on 8 GPUs to find **why** each baseline has its own failure mode. Ten coherent
> stories, each validated from ≥20 angles (method × length × ratio × task × base × budget slices). Report follows the drain.
> Tags: `s1_…`–`s10_…`. Reuses existing catalog/necessity/RAG/T1 data where an angle is already covered.

## The 10 stories (hypotheses about baseline failure modes)

- **S1 — Attention-based KV eviction collapses at long context.** snapkv/h2o/expected/tova hold to 8k then fall at 16k+.
  *Why:* the observation-window attention can't localize the needle as context grows / scores diffuse.
  *Angles:* {snapkv,h2o,expected,tova} × len{4k,8k,16k} × ratio{0.25,0.5,0.75} on ruler = 36.
- **S2 — Attention-free eviction (knorm/kvzip/streaming) is length-robust.** Holds to 32k.
  *Why:* key-norm / reconstruction importance doesn't depend on attention scores that degrade with length.
  *Angles:* {knorm,streaming,kvzip,fastkvzip} × len{4k,8k,16k,32k} × ratio{0.25,0.5,0.75} = 48.
- **S3 — The method ranking flips by task type.** knorm wins retrieval, loses QA; attention-based the reverse.
  *Why:* retrieval needs sparse needle-KV; QA/multi-hop needs distributed semantic KV.
  *Angles:* 8 methods × 8 benches @0.5 = 64.
- **S4 — Each method has a compression-ratio cliff at a different budget.**
  *Why:* below a method-specific budget the needle KV gets evicted.
  *Angles:* 8 methods × 6 ratios on ruler-8k = 48.
- **S5 — Window truncation = a locality assumption.** Lossless when the answer is local (squad), fails on retrieval/multi-hop.
  *Why:* keeps only the recent window; early/diffuse evidence is dropped.
  *Angles:* window{128,256,512,1024,2048,4096} × 6 benches = 36.
- **S6 — Full context can HURT (distractor degradation).** quality_hard full < no_ctx.
  *Why:* long distractor context degrades the base (lost-in-the-middle / context-rot).
  *Angles:* full/window/no_ctx × 5 long benches + RULER full at 8k/16k/32k + rca distractor-count{0,2,4} ≈ 16.
- **S7 — LLMLingua-2 denoises but drops needles.** Beats full on trivia; collapses on low-rate retrieval.
  *Why:* removes low-information tokens (helps noisy QA) but can delete a low-salience needle.
  *Angles:* rate{0.1,0.25,0.33,0.5,0.67} × 6 benches = 30.
- **S8 — BM25 RAG is length-robust but fails abstractive.** Holds on NIAH; collapses on narrativeqa.
  *Why:* lexical retrieval needs lexical overlap; abstractive/global answers have none.
  *Angles:* budget{512,1024,2048,4096} × 8 benches = 32.
- **S9 — GDN base (Qwen3.5): no KV lever + state limits.** KV methods N/A; exact retrieval weak at long ctx.
  *Why:* linear-attn has no KV cache (only prompt/window/RAG apply) + finite-state capacity.
  *Angles:* {full,llmlingua×2,window×2,rag} × 5 benches/lengths = 30.
- **S10 — Several "SOTA" presses don't run / are base-specific.** pyramid/adakv crash; qfilter/duo need per-model artifacts.
  *Why:* implementation/architecture coupling (GQA shape, eager-mode, precomputed filters/head-masks).
  *Angles:* 7 broken/base-specific × confirm + {think,simlayer} adapted-characterization = 11.

Total ≈ 351 new cells (+ existing data). Each story gets ≥20 independent data points from its grid slices.

## Status
- Launched as `q_diag_0..7` across d1525×4 + d1530×2 + d1420×2; `runner_u.sh`; results → `s{1..10}_*` logs.
- Report (10 stories, per-story evidence table + root-cause read) to be compiled after the drain.
