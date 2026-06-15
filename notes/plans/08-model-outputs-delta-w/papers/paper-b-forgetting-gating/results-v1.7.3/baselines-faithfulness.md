# Baseline faithfulness audit (context-compression rivals) — 2026-06-14

> Verified each baseline against the **original paper** (not memory). Principle: the **MECHANISM/architecture
> matches the original exactly**; only the **training hyper-parameters** (steps / lr / data amount) are set to
> our matched/fair budget (enc-16 where applicable, K=64, 384 items, 3000 steps — same as Gist/Cartridge/GCM).
> Code: [`baselines2025.py`](#). These are **context compression** (learned compact reps), not token pruning.

## Summary table
| method | paper | original mechanism (verified) | our impl | faithful? |
|---|---|---|---|---|
| **ICAE** | Ge, ICLR'24 (2307.06945) | encoder = base **+ LoRA(q,v)** over `[ctx ; K mem toks]`; mem toks' **output hidden states** = slots; **frozen base** decoder; **[AE]** token; AE + LM/FT | LoRA(q,v) encoder → hidden slots; frozen-base decoder; learnable [AE]; AE + task CE | ✅ exact |
| **500x** | 2408.03094 | = ICAE but decoder conditions on the compressed tokens' **KV values** (per layer), `[BOS]`-triggered | same LoRA encoder, extract last-K **KV** per layer, inject via KV path; task CE | ✅ exact |
| **AOC** | 2501.06730 | ICAE with a separate **attention-only** encoder (MLP sublayers **removed**), trained fully | separate encoder copy, **MLP sublayers zeroed** (`_ZeroMLP`), trained fully; frozen-base decoder; AE+task | ✅ exact (MLP removed) |
| **Beacon** | 2401.03462 | interleave 1 beacon per α-token unit; compress each unit into **beacons' KV**; accumulate KV across chunks | LoRA-base, interleave 1 beacon/α toks, keep **beacon-position KV**, inject via KV path | 🟡 core (single-pass; cross-chunk **KV accumulation simplified** at our ≤1024 budget) |
| **ComprExIT** | 2602.03784 | LLM-as-encoder + explicit info **transmission into anchor tokens** across layers | attention-only encoder → anchor/memory tokens (AOC-style) | 🟡 core (cross-layer transmission approximated; needs authors' code for exact) |
| **LCC** | 2602.21221 | **test-time** disposable-LoRA "compiler"; reconstruction + context-agnostic regularizer → buffer tokens | **test-time** buffer optimized by reconstruction + on-manifold regularizer (per context) | 🟡 core (buffer instead of a disposable LoRA) |

## Notes on exactness
- **ICAE / 500x / AOC are exact** re-implementations of the published architecture (LoRA-on-base encoder + frozen
  decoder for ICAE/500x; attention-only separate encoder for AOC). The only changes are training HPs (our budget)
  and the task/LM objective using the benchmark's gold answer (the FT objective) + autoencoding — same objectives
  as the papers, our data.
- **Beacon**: the defining mechanism (interleaved beacons → beacon KV) is faithful; the paper's **streaming
  cross-chunk KV accumulation** (for >window contexts) is collapsed to a single pass because our eval contexts are
  capped at 1024 tokens (≤ the backbone window), so accumulation is unnecessary here. Flagged.
- **ComprExIT / LCC** (both 2026): implemented as faithful **cores** of the described mechanism; exact reproduction
  would need the authors' (unreleased) code. Flagged as 🟡 in tables.
- **Fairness:** every method is trained at the **same budget** as Gist/Cartridge/GCM and compresses to **K units**
  the frozen base reads → the §2.4 comparison and the (amortized) efficiency comparison are apples-to-apples.
- **Not baselines for efficiency:** token-pruning/selection (`trunc`, `retrieval`, dropped `tokprune`) and runtime
  KV-prune/merge (KVzip/SnapKV) — different mechanism class; reported only as necessity floors, never efficiency rivals.
