# Paper C (v3.0) — Linear Attention & its variants

Next paper line. Version tag **v3.0** (demo **v0.3.0**). Focus: **linear attention and variants** (delta-rule / gated-linear / SSM / hybrids) and their long-context recall limits + training-free input-side memory.

## Index
- [`PAPER-C-research-line.md`](PAPER-C-research-line.md) — the research thread, positioning, candidate contributions (C-1 diagnostic + C-2 training-free plug-in as the spine), open questions, first actions.
- [`references-linear-attention.md`](references-linear-attention.md) — categorized reference collection (foundations → gated-linear → SSM → delta-rule → RWKV → recall/capacity → hybrids → sparse peers → TTT → theory/limits → long-ctx eval), arXiv IDs, 🌟 core-cites.

## Bridge from Paper B
Builds on the validated fact-base that the long-context diagnosis is **model-invariant and reproduces on cache-free linear (Qwen3.5-GDN)**, that **KV-eviction can't run on linear**, and that **IMP-lite (input-side, training-free) can** — see Paper B `matrix-facts.md` F10/F28/F43 and `linear-attention-and-kvcache-background.md`.

## Status
Scaffolding only (2026-07-14): references collected + research line drafted. Next = verify arXiv IDs, read core papers, inventory runnable linear/hybrid checkpoints, run the recall-failure-map scoping experiment.
