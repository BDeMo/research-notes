# Results v1.7.6 (Qwen3.5 main; full suite across all selected bases)

> **Supersedes [v1.7.5](../results-v1.7.5/results-v1.7.5.md)** (Qwen3-8B) — kept as **legacy**. v1.7.6 pivots the main
> model to **Qwen3.5** and re-runs the entire experiment logic across **6 bases**, with a cleaner recipe
> (epochs+convergence, full data, depth/K sweeps relative to each base).
> **Canonical settings (exact recipe / env / models / benches, reproducible):** [`settings/v1.7.6-settings.md`](settings/v1.7.6-settings.md).
> **Glossary (technical/internal terms → definitions):** [`../glossary.md`](../glossary.md). Terms below link there, e.g. [GCM](../glossary.md#gcm), [do-no-harm gate](../glossary.md#gate-do-no-harm-gate).

## Test-phase protocol vs formal run (2026-06-18 — locked)
**Core rule: TRAINING is identical in test and formal — never discounted.** The *only* test↔formal difference is **eval scale**.

| aspect | test phase | formal run |
|---|---|---|
| **training config** (batch, lr, distill, K, depth, epochs, grad_accum, **n_train**) | **identical to formal** — `n_train` always **maxed (full train set)**, no shortcut | same |
| **batch / effective batch** | **same** (fixed; e.g. parallel batch=8 → eff 8; AR DDP 4×1×accum2 → eff 8) | same |
| **eval scale** (`n_eval`) | **reduced OK** for fast iteration | **full split** |
| training regimes | **two only**: (a) per-dataset → one ckpt each (in-task); (b) **mix** = all train sets → one ckpt. **Mix deferred** — do in-task first. | same |

**What we're doing now = module/recipe ablation.** The single best config becomes the **default setting in the main table**.

**"Make it work" on in-task — 3 bars that must all hold (single anchor, eval = its in-task / cross-task / cross-domain group):**
1. **compress-only ≫ no_ctx** (significantly stronger, not just non-zero).
2. **gate AUC and F1 high** (the label-free do-no-harm gate is discriminative).
3. **gated-Acc ≈ full** (per-item: trust M when gate fires, else fall back → near the full-context ceiling).

**Then:** run the 2025 baselines; if our best config **beats baselines on most benchmarks**, it is adopted as the **formal default**.

**n_train cost note (full train splits):** rca 350 · bfcl_live_multiple 737 · toolace 6.5k · narrativeqa 33k · squad_v2 87k · hotpot_qa 90k. Literal-full on the large benches = 15–42 h/config; `n_train` policy still TBD (the compressor adapter is tiny so `patience` early-stop likely converges well before full). AR encode is ~25 s/item (sequential) → full-scale AR needs a batched-AR impl.

**Eval TODO:** current eval reports compress/full/no_ctx + gate AUROC; must add **gated-Acc** + **gate F1** to score bars (2) and (3) above.

## Models (selected bases)
Qwen3.5-9B (main) · Qwen3.5-4B · GLM-4-9B · Ministral-8B · Llama-xLAM-8B · ToolACE-8B.
*(Qwen3-8B and Qwen3-4B-2507 dropped from the selected set; Qwen3-8B results live in v1.7.5 as legacy.)*
Base depths (for `depth full/half`): GLM 40, Ministral 36, xLAM/ToolACE 32, Qwen3.5-9B/4B 32 (resolved at load).

## Recipe (the v1.7.6 "best/main" setting + the swept axes)
| knob | value(s) | notes |
|---|---|---|
| **K (memory)** | fixed **{4, 8, 16}** + **adaptive {2×, 4×, 8×}** | adaptive = `--chunk-size = ratio×K` ⇒ budget ≈ ctx/ratio |
| **encoder depth** | **{1, half, full}** | per base: full = all layers, half = ⌊L/2⌋ (`--enc-layers full|half|N`) |
| **compute** | **train until convergence** | early-stop (see *patience*) under a generous **epoch cap** |
| **data** | **all relevant data, in epochs** | 1 epoch = every item seen once (`--epochs`; steps derived) |
| **lr** | **fixed 3e-4** | not tuned in v1.7.6 |
| default (main/baseline cells) | K=16, depth=half, epochs≤6 + patience=3 | the scaling phase sweeps K & depth around this |

**`patience` definition:** during training a held-out **val loss** is measured every ~`steps/10` steps; if it fails to
improve (by >1e-4) for **`patience` consecutive checks**, training **stops early** ("converged"). `patience=3` ⇒ stop
after 3 straight non-improving val checks. This is the "train until converge" mechanism; the **epoch cap** bounds the worst case.

**Why both `--epochs` and `--steps`?** The training loop is **step-based** internally (val-check cadence, LR schedule all
count optimizer steps). `--epochs` is the **user-facing knob**: it's converted to `steps = epochs × |train set|` at launch
(1 epoch = one full pass). `--steps` is the **legacy fallback** used only when `--epochs=0`. In v1.7.6 we always pass
`--epochs` (+ `--patience`), so `--steps` is derived, not set by hand.

## Experiment design (binary ↔ fit; unary ≥3 options at best setting)
- **Regimes (per base):** single-dataset FT (eval covers **in-task / cross-task / cross-domain**) + **mix** (train mix, eval all).
- **Binary B1** train×test (the 3 main tables) · **B2** compressor×gate (from `--signals`).
- **Unary** (≥3 each): model (6), compressor (9), gate-signal (7), module, **K {4,8,16}+adaptive{2,4,8×}**, **depth {1,half,full}**.

## Orchestration
**Base-major**: each pod runs ONE base's full suite at a time (baselines → matrix → ablation → scaling). **free** runs
first; **test** is reserved (user is modifying it) and will take a different base when released. Code+env on `/mnt/persist`,
symlinked to `~/workspace` on both pods. Runners: `run_suite.sh` / `run_base_driver.sh`.

> **How to read the 3 main tables.** Each cell = **compress accuracy** ([ctx>K](../glossary.md#ctxk)) of that compressor under that relation,
> averaged over the relation's (train→eval) pairs; [`full`](../glossary.md#full-full_ctx) = ceiling, [`no`](../glossary.md#no_ctx) = no-context floor, **[GCM](../glossary.md#gcm) = ours**. `·` = not yet run.
> Methods left of ‖ are **[soft-prompt](../glossary.md#soft-prompt--prefix-injection)** (≈ our cost); right of ‖ are **[KV-injection](../glossary.md#kv-cache-injection)** (cost ≫, reported but not ranked against us).
> Relations: **[in-task](../glossary.md#in-task)** train==eval · **[cross-task](../glossary.md#cross-task)** train≠eval, same domain · **[cross-domain](../glossary.md#cross-domain)** train domain ≠ eval domain.
> **Note (Qwen3.5 rows):** the [KV-injection](../glossary.md#kv-cache-injection) baselines (Beacon / Cartridge / 500x) are **N/A** on Qwen3.5 — its **linear-attention** layers have no per-layer K/V cache to inject into. A point for soft-prompt/GCM: it runs where KV-compression can't.

## MAIN TABLE 1 — in-task (train == eval), compress acc @ ctx>K
| base | no | full | **GCM** | ICAE | AOC | ComprExIT | LCC | Gist ‖ | Beacon | Cartridge | 500x |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Qwen3.5-9B** (main) | · | · | · | · | · | · | · | · | · | · | · |
| Qwen3.5-4B | · | · | · | · | · | · | · | · | · | · | · |
| GLM-4-9B | · | · | · | · | · | · | · | · | · | · | · |
| Ministral-8B | · | · | · | · | · | · | · | · | · | · | · |
| Llama-xLAM-8B | · | · | · | · | · | · | · | · | · | · | · |
| ToolACE-8B | · | · | · | · | · | · | · | · | · | · | · |

## MAIN TABLE 2 — cross-task (train task → other task, SAME domain), compress acc @ ctx>K
| base | no | full | **GCM** | ICAE | AOC | ComprExIT | LCC | Gist ‖ | Beacon | Cartridge | 500x |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Qwen3.5-9B** (main) | · | · | · | · | · | · | · | · | · | · | · |
| Qwen3.5-4B | · | · | · | · | · | · | · | · | · | · | · |
| GLM-4-9B | · | · | · | · | · | · | · | · | · | · | · |
| Ministral-8B | · | · | · | · | · | · | · | · | · | · | · |
| Llama-xLAM-8B | · | · | · | · | · | · | · | · | · | · | · |
| ToolACE-8B | · | · | · | · | · | · | · | · | · | · | · |

## MAIN TABLE 3 — cross-domain (train domain → other domain), compress acc @ ctx>K
| base | no | full | **GCM** | ICAE | AOC | ComprExIT | LCC | Gist ‖ | Beacon | Cartridge | 500x |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Qwen3.5-9B** (main) | · | · | · | · | · | · | · | · | · | · | · |
| Qwen3.5-4B | · | · | · | · | · | · | · | · | · | · | · |
| GLM-4-9B | · | · | · | · | · | · | · | · | · | · | · |
| Ministral-8B | · | · | · | · | · | · | · | · | · | · | · |
| Llama-xLAM-8B | · | · | · | · | · | · | · | · | · | · | · |
| ToolACE-8B | · | · | · | · | · | · | · | · | · | · | · |

## Mix-trained (train = mix of all anchors → eval ALL benchmarks), GCM compress acc @ ctx>K
> ⚠ **GCM runs only eval 4 of these 7 benches** (recipe `$ALL=bfcl,toolace,squad_v2,narrativeqa`); hotpot/quality/rca were **not evaluated** by the free-pod GCM cells (`–`). Eval set is **tiny (~6 items on bfcl/toolace)** → high variance.
| base | bfcl_live_multiple | toolace | squad_v2 | hotpot_qa | narrativeqa | quality | rca_openrca |
|---|---|---|---|---|---|---|---|
| **Qwen3.5-9B** (main) | 0.00 | 0.00 | **0.48** | – | 0.12 | – | – |
| (others) | · | · | · | · | · | · | · |

> **Read:** on Qwen3.5-9B mix-trained, GCM compress works **only on squad_v2** (0.48, vs full 0.78 / no-ctx 0.47 → it actually *beats* the no-ctx floor here). On **bfcl & toolace it is 0.00** (full=0.83), and narrativeqa 0.12 (< no-ctx 0.16). **4-bench agg: GCM 0.15 < no-ctx 0.20 < full 0.70** — i.e. averaged, GCM currently underperforms giving no context, dragged down by the tool-calling benches.

## Gate table (B2) — compressor × gate-signal AUROC (does the gate compose with any compressor?)
Per base (main = Qwen3.5-9B), AUROC of each label-free signal at predicting compress-success. `compressor_gate.py`.
| compressor | conf | margin(TARG) | neg_entropy | neg_recon | judge |
|---|---|---|---|---|---|
| **GCM** | · | · | · | · | · |
| ICAE / AOC / … | · | · | · | · | · |

## Unary recipe sweeps (main = Qwen3.5-9B, bfcl_live_multiple, in-task) — GCM compress acc + gate AUROC
> 🛑 **Sweep is UNINFORMATIVE as configured.** All scaling cells eval **bfcl-only**, and GCM compress on bfcl is **0.00** (see above) → every K / depth setting reads 0.00, giving **zero discrimination**. To get a usable sweep, re-point the scaling cells' `GCM_EVAL` to **squad_v2** (the one bench with signal). `adapt 2×/4×/8×` (`–`) = **not in recipe** (`run_base.sh` has only fixed K4/K8/K16).

**K (memory):** fixed {4, 8, 16} + adaptive {2×, 4×, 8×}
| K=4 | K=8 | K=16 | adapt 2× | adapt 4× | adapt 8× |
|---|---|---|---|---|---|
| 0.00 | 0.00 | 0.00 | – | – | – |

**Encoder depth:** {1, half, full}
| depth=1 | depth=half | depth=full |
|---|---|---|
| 0.00 | 0.00 | 0.00 |

## Module ablation (main = Qwen3.5-9B, from full → drop one) — GCM compress acc / gate AUROC
> 🛑 **Also bfcl-in-task only → all 0.00, uninformative** (same root cause as the sweep). Recipe only produces `−distill` (=`abl-nodistill`), a norm-off cell (`abl-normoff`, not a named column here), and the scaling cells re-used as `−enc(1)`=`scale-depth1`, `−enc(full)`=`scale-depthfull`, `−K(4)`=`scale-K4`. The remaining columns (`init-rand`, `−lora`, `−rec`, `−dev`, `−task`, `inject-kv`) have **no cell in the recipe** (`·`).
| ab0_full | −enc(1) | −enc(full) | −K(4) | init-rand | −lora | −distill | −rec | −dev | −task | inject-kv |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.00 | 0.00 | 0.00 | · | · | 0.00 | · | · | · | · |

## Status (2026-06-18 04:10 UTC)
**Progress:** `free` finished **Qwen3.5-9B 13/13 cells** (~2 h, not 15–20 h — the convergence recipe at EPOCHS=3/NTRAIN=96
runs ~10–29 min/cell, far faster than the doc's earlier estimate). Now on **base-2 Qwen3.5-4B** (main-mix). `test` runs the
**baselines** (q35_9b/gist done on 5 anchors; aoc/comprexit/icae/lcc still queued). Both orchestrators are in **tmux** (`run:gcm-orch` on free, `run:baselines` on test) — survive disconnect; cells resume via `out/<base>/<TAG>.json` skip-if-exists.

**🚨 BLOCKER — Qwen3.5-9B numbers are NOT paper-ready (recipe needs a fix before bases 2–6 burn ~12 h reproducing it):**
1. **GCM compress = 0.00 on bfcl & toolace, even in-task** (`tr-bfcl` trains+evals bfcl → 0.00; full ceiling = 0.83).
   Only **squad_v2 shows real signal** (0.33–0.48, beats no-ctx floor 0.47, approaches full 0.78); narrativeqa weak (0.06–0.19).
2. **4-bench agg: GCM 0.15 < no-ctx 0.20 < full 0.70** — averaged, GCM currently *underperforms giving no context*,
   because the two tool-calling benches are 0.00. Headline risk: as-is this says "GCM hurts on tool-calling."
3. **Eval set tiny** (~6 items on bfcl/toolace → 1/6 granularity) ⇒ 0.00 could be 0/6 noise, not a stable estimate.
4. **Scaling + ablation sweeps are uninformative**: every K/depth/ablation cell evals **bfcl-only** (the 0.00 bench) →
   all read 0.00, no discrimination. **No adaptive-K cells** in recipe (doc's adapt 2×/4×/8× columns unfillable).
5. **GCM (4-bench, n≈6) and baselines (7-bench, n=96) are not comparable** → the 3 main tables can't be filled responsibly yet.

**Root cause (found in `run_recipe.py` eval, lines ~88–95) — two hard caps make the numbers under-measured, esp. on tool benches:**
1. **`seen[b] <= 6` → only 6 eval items/bench.** bfcl/toolace 0.00 = literally 0/6 (pure noise floor, not a stable estimate).
2. **`model._gen(..., 8)` → max_new_tokens = 8 for ALL gen paths.** squad_v2 = short extractive spans (fit in 8 tok → GCM works,
   0.48). **bfcl/toolace = structured function calls (`func(arg1=…, arg2=…)`, ≫8 tok) → truncated → scored 0.** Largely an
   **eval-config artifact**, though full-ctx still hits 0.83 on the same 6 items so GCM *also* loses call-precision under compression.
3. Free recipe **dumps no per-item records** (only test-pod baselines do) → can't inspect predictions without a re-run.

**Recommended recipe fixes before letting bases 2–6 finish (cheap, 1-number edits):**
(a) raise `model._gen(..., 8)` → **≥64 max-new-tokens** (so tool-call answers can be emitted); (b) raise `seen[b] <= 6` → **≥48**;
(c) re-point scaling/ablation `GCM_EVAL` → **squad_v2** so the sweep has signal; (d) dump per-item records for debuggability.
Bases 2–6 will otherwise reproduce the same 0.00-on-tool-benches pattern (~12 h of mostly-uninformative cells).

**Infra notes (kept):**
- **Qwen3.5 UNBLOCKED**: fla Triton gated-delta-rule backward buggy on Hopper+Triton≥3.4 (#640); tilelang fix crashes on
  import (tvm-ffi). Fix = **null the fla symbols → qwen3_5 modeling uses built-in pure-torch `torch_{chunk,recurrent}_gated_delta_rule`** (~6 s/step). Verified.
- **`run_recipe.py` crash fixed (this session):** the results-writer line used `json.dump` without `import json` → all 3
  initial main runs trained then died at write-time (0 cells saved for 7 h). Added `import json`; verified 13/13 now write.
- **GCM precompute fix**: distillation teacher target = full-ctx teacher distribution over the **gold answer in one forward**.
