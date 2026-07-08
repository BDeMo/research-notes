# 2026-07-08 — Plan 08: IMP method, long-context benchmarks, and the Paper-A/B pivot

## Activities
- **KV-free baseline family**: reproduced/expanded prompt-compression (LLMLingua-2/orig/Long, **Selective-Context newly unblocked**), ToMe, BM25-RAG on Qwen3-8B; head-to-head "which works" at 16k; cheap importance-signal probe (query-relevance 0.95 / surprisal 0.84).
- **Paper A — kvzip repeat-prompt reconstruction gate**: implemented; ablated (15 cells) + high-K bracket + **random-M leak control** → **rejected** (signal ≈ random). Root cause: learned soft-memory **can't compress extractive QA** (4 recipes ≈ no_ctx).
- **Paper B — IMP** (importance-routing prefilter): implemented token- and span-level; validated span-level rescues QA while keeping retrieval; keep-rate / span-size / signal ablations.
- **Long-context benchmarks**: integrated **LongBench** at *real length, no truncation* (via `data.zip`, MAXCTX 32768); surveyed the 2026 suite (RULER/v2, LongBench-v2, ∞Bench, BABILong, HELMET, NoLiMa).
- **Benchmark validity audit** over 784 cells: confirmed `full` is a sane ceiling; narrativeqa under-scored (not a collapse); excluded old bad-config NIAH / multi_needle.
- Launched a **12h / 6-GPU grand grid** (full method × benchmark main table + all ablations + KV×LongBench + KV ratio sweep).
- **Slides** rewritten (`slides/weekly/2026-07-w01.tex`): high-level logic + hyperlinks to detail docs.
- Built a portable, content-agnostic methodology repo **ResearchMeta** (`/Users/s1shi/workspace/research-methodology`, local, MIT).
- Drafted **Paper B** (`PAPER-B-draft.md`).

## Decisions
- **Pivot toward Paper B (IMP)** as the primary line; Paper A's shipping gate = confidence (reconstruction gate rejected; blocked on a better compressor).
- **Span-level is the general IMP**; token-level is retrieval-only (shreds coherent QA).
- Honest positioning: IMP matches full on retrieval, rescues QA, beats full by denoising, but **does not yet beat RAG/LLMLingua on short-context extractive QA** → the contribution rests on generality (linear bases) + efficiency + necessity regime.
- Three make-or-break experiments: linear/GDN base · necessity (input overruns window) · semantic > lexical.

## Output artifacts
- `papers/paper-b-forgetting-gating/PAPER-B-draft.md` (new) · `imp-method-and-implementation.md` (new)
- Updated `OVERVIEW-both-papers-and-facts.md` (§3.1 A–G), `baseline-factbase-v2.0.0.md` (§10–§12.3), `matrix-facts.md` (F18–F26), `decisions-2026-06-24.md` (D23–D26)
- Figures `figures/fig8–fig13*.png`
- `slides/weekly/2026-07-w01.tex` (+ appended in `slides/main.tex`)
- Code (in the `gcm` repo, not here): `run_baseline.py` (imp mode, span, signal), `model.py` (recon_repeat), `run_probe.py`, LongBench loader patch.

## Knowledge sources used
- LongBench, RULER/RULERv2, ∞Bench, LLMLingua/-2/Long, Selective-Context, KVzip/FastKVzip, ToMe, R-MeeTo (see `knowledge-sources.md`).

## Next steps
1. Linear/GDN base (IMP's unique territory). 2. Necessity regime (LongBench / ∞Bench >100k). 3. Semantic > lexical (abstractive where RAG fails). 4. Small draft-LM scorer (real prefill saving). 5. Learned router (Mode B). 6. Harvest the grand grid → full main table + figures.
