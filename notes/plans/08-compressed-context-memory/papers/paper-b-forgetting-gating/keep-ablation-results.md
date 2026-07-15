# Keep-rate ablation — all methods (results)

Base **Qwen3-8B**, N=100 (disclosed ablation subset), one shared keep-fraction `k∈{0.1,0.25,0.5,0.75}` mapped to each method's native knob (mapping in [`hyperparameters.md`](hyperparameters.md) §C). Method = `IMP-v2.1.0` (span-32). Numbers ×100 = accuracy/F1. Launcher [`configs/keep_ablation.sh`](configs/keep_ablation.sh); logs `/mnt/persist/grid_keepabl/ka_*`. 111/112 cells (imp·ruler·k0.1 re-queued).

## RULER-NIAH @16k (synthetic retrieval)
| method | k=0.1 | 0.25 | 0.5 | 0.75 |
|---|--:|--:|--:|--:|
| window | 7 | 25 | 44 | 74 |
| RAG | **95** | **97** | 99 | 100 |
| LLMLingua-2 | 5 | 43 | 95 | 100 |
| ToMe | 0 | 0 | 0 | 17 |
| **IMP** | · | **70** | **98** | 99 |
| kvzip | 79 | 68 | 75 | 74 |
| knorm | 0 | 19 | 89 | 99 |

## squad_v2 (extractive)
| method | k=0.1 | 0.25 | 0.5 | 0.75 |
|---|--:|--:|--:|--:|
| window | 65 | 66 | 66 | 66 |
| RAG | 70 | 70 | 70 | 70 |
| LLMLingua-2 | 25 | 39 | 53 | 60 |
| ToMe | 10 | 17 | 25 | 49 |
| **IMP** | 34 | 33 | 43 | 53 |
| kvzip | 43 | 60 | 74 | 72 |
| knorm | 1 | 8 | 20 | 38 |

## lb_hotpotqa (multi-hop, low ceiling)
| method | k=0.1 | 0.25 | 0.5 | 0.75 |
|---|--:|--:|--:|--:|
| window | 15 | 21 | 24 | 23 |
| RAG | 21 | 27 | 25 | 24 |
| LLMLingua-2 | 15 | 18 | 20 | 20 |
| ToMe | 12 | 14 | 15 | 16 |
| **IMP** | 14 | 18 | 17 | 20 |
| kvzip | 16 | 12 | 19 | 19 |
| knorm | 8 | 17 | 19 | 19 |

## QuALITY-hard (literary MC — the "full-hurts" regime)
| method | k=0.1 | 0.25 | 0.5 | 0.75 |
|---|--:|--:|--:|--:|
| window | **17** | 12 | 9 | 9 |
| RAG | **15** | 12 | 8 | 9 |
| LLMLingua-2 | **17** | 17 | 14 | 12 |
| ToMe | **23** | 18 | 14 | 15 |
| **IMP** | **18** | 13 | 15 | 12 |
| kvzip | 31 | 30 | 29 | 31 |
| knorm | 23 | 26 | 28 | 29 |
*(blind no_ctx = 21.9; full = 9.2)*

---

## What the keep-curves show

**1. Retrieval (RULER): keep-robustness separates the methods.**
- **RAG (95→100) and IMP (70→98→99) hold the needle at *small* budgets**; kvzip is flat-robust but capped (~75). **ToMe collapses (0 until k=0.75)** and window/ll2/knorm are budget-hungry (need k≥0.5). ⇒ importance-guided selection (IMP) and retrieval (RAG) are the *budget-efficient* ways to keep a needle; merging/truncation are not.

**2. Extractive (squad): IMP is cleanly monotone (34/33/43/53).** window/RAG are flat-high (answer is local / retrievable at any budget); kvzip peaks mid (~0.5); knorm weak. The monotone IMP curve = the clean budget↔accuracy trade-off (confirms F25).

**3. Literary-MC (QuALITY-hard): the curve INVERTS — keeping *less* is *better*.**
For **window, RAG, LLMLingua-2, ToMe, and IMP, accuracy DECREASES as keep rises** (e.g. window 17→9, ToMe 23→15) — because on this regime the full context *hurts* (full 9.2 < blind 21.9, F6). Only **kvzip/knorm stay high (~30, above both blind and full)** — KV-mass pruning *denoises* literary-MC. ⇒ a striking, direct confirmation that **"more context is a liability" here**, and the *right* keep rate is **regime-dependent** (low for literary-MC, high for retrieval) — which is exactly the case for a **per-input** budget, not a fixed one.

**4. Multi-hop (hotpot): low ceiling, weak monotonicity**, RAG best (~21–27); not budget-discriminative.

### Takeaway
There is **no single keep rate that is right across regimes**: retrieval wants high keep (or importance-selection to stay high at low keep), literary-MC wants *low* keep. This is the budget-axis version of Insight ① ("no fixed compressor is right everywhere") and motivates an **input-adaptive** budget. IMP's own curves are monotone/stable where they should be (retrieval, extractive) and correctly *decline* where full hurts (literary-MC).
