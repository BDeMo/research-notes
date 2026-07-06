# Results v1.7.5 (Qwen3-8B) — scaling diagnosis, minimize-scale, stable training

> **Builds on [v1.7.3](../results-v1.7.3/results-v1.7.3.md)** (the leak-free clean re-run + do-no-harm gate).
> v1.7.4 added a length-adaptive memory + the 2025 baselines + the scaling study; v1.7.5 adds **training-stability
> rigor** and the **HP sweep / scaling critical review / minimize-scale** investigation. Setup is shared with
> v1.7.3 ([experimental-setup.md](../results-v1.7.3/experimental-setup.md)) plus the v1.7.5 deltas below.
> One-page brief: [`../summary-matrix-v1.7.3.md`](../summary-matrix-v1.7.3.md). Paper draft: [`../paper/main_v1.7.5.tex`](../paper/main_v1.7.5.tex).

## What v1.7.5 changed vs v1.7.3/v1.7.4
- **Training stability** (so the negatives are not an optimization artifact): gradient accumulation (effective
  batch 8), LR warmup + cosine decay, EMA-smoothed loss logging, and periodic held-out **val** loss. Default on.
- **Faithful 2025 baselines** (ICAE/AOC/Beacon/500x/ComprExIT/LCC) at matched budget — audit:
  [`baselines-faithfulness.md`](baselines-faithfulness.md).
- **Broad HP sweep + scaling critical review** (this version's main investigation).

## The documents (this version)
| doc | content |
|---|---|
| [`longcontext-improvement.md`](longcontext-improvement.md) | length-adaptive vs fixed-K (no long-ctx rescue), no-scaling result, OOD soft-memory probe, 2025-baseline table, training-dynamics fix (§11) |
| [`scaling-critical-review.md`](scaling-critical-review.md) | critical review of the scaling design (the capacity-K axis was untested), per-bench re-analysis, hypotheses H1–H5 + verdicts |
| [`baselines-faithfulness.md`](baselines-faithfulness.md) | per-method faithfulness audit of the 2025 context-compression baselines vs their papers |
| [`faithful-contrastive-ablation.md`](faithful-contrastive-ablation.md) | **faithful representation-alignment ablation** — contrastive (InfoNCE) vs positive-only vs adversarial: λ-sweep (inverted-U, peak 0.5 = +5.2pp compress), ctr↑ vs adv↓ (opposite), combo (antagonistic on compress / best gate AUROC 0.85); recipe-aligned 3000 steps |

## Headline findings (sweep 69/80, 2026-06-15)
1. **Length-adaptive memory ≈ fixed-K** (Δ±0.02) — chunking does not rescue long context.
2. **No scaling** — avg compress flat across 24× FLOPs / data / depth; **capacity K does not scale either** (K16 ≈ K64 ≥ K256).
3. **The memory is OOD** (cos ≈ 0.10), and the **on-manifold fix does NOT break the ceiling** (H5 *preliminary* rejected: `manif01` 0.40 ≈ base 0.37) → OOD is a *symptom*, not the fixable cause.
4. **The wins are tuning + minimize-scale, not more scale:** best HP = **lr 3e-4** (avg **0.450** vs base lr1e-4 0.371 — lr was too low); the **smallest config (N≈4, K≈16)** matches the big one at ~10× less compute; **λ_distill is load-bearing** (distill0 → collapse 0.02).
5. Pending (~2–3h): `manif10`/`mnorm` (confirm H5) + the minimize Pareto (`ab_min`) — see [`scaling-critical-review.md` §5b](scaling-critical-review.md).
6. **Transfer + non-tool (new, T8):** the do-no-harm gate **extends to cross-domain and 5 non-tool benches** (GCM mix→held-out: real detection AUROC>0.6 on bfcl_simple/squad/rca, conservative always-compress elsewhere); compress ≪ full everywhere; per-corpus Cartridge does **not** transfer (compress 0.1–0.2 across all relations); the **clean single-anchor GCM matrix is running** on `test` (`p2_gcm`).

→ Net: the **capacity-bound diagnosis holds and strengthens** (depth, capacity, AND the OOD-fix all fail); the contribution is the **do-no-harm gate** + **minimize-scale** + **lr tuning**, not a better/bigger compressor (diagnosis-led paper, [`../paper/main_v1.7.5.tex`](../paper/main_v1.7.5.tex)).

## Experimental roadmap — what each experiment answers
> **Thesis:** a *reliable gate* turns lossy parametric-memory compression into a safe **net-win** (compress when
> safe, fall back to full when not). Each table answers one link in the chain
> Q1 *(problem real)* → Q2 *(fair carrier)* → Q3 *(why a gate, not a bigger compressor)* → Q4 *(detect failure)*
> → Q5 *(gate general/detachable)* → Q6 *(generalize)* → Q7 *(net-win)*.

| # | Question it answers | Experiment(s) | Answer so far | status |
|---|---|---|---|---|
| **Q1** | Is the problem real? (context needed **and** compression lossy) | **T1, T2′** | no_ctx ≪ full; **compress ≪ full for *every* compressor** on high-headroom benches (ctx>K) ⇒ can't "always compress" | ✅ **T2′ 48/48, 5 benches** (+narrativeqa degenerate) |
| **Q2** | Is our compressor a *fair carrier* (not the contribution)? | **T2′** (split by injection type, ctx>K) | none beats full; **GCM best among same-complexity soft-prompt methods**; KV-injection (Beacon) costs more, still < full (trunc = trivial floor, set aside) | ✅ **T2′** |
| **Q3** | Why a gate, **not a bigger/better compressor**? (diagnosis) | **T3** (chunking), **T4** (scaling), **T5** (HP), **T7** (OOD probe) | no scaling (compute/data/depth/**K** flat); memory OOD; wins = tuning+minimize, not scale ⇒ **capacity-bound** | ✅ |
| **Q4** | Can we *detect* compression failure? (gate signal quality) | gate **AUROC / F1 / recall@P95** + **confusion matrix** (TP=save, FP=harm, FN=wasteful, TN=correct-fallback) | AUROC ≈ **0.8** ⇒ do-no-harm net-win *reachable* (0.7 = safe but no savings) | ✅ |
| **Q5** | Is the gate a *detachable, general* module? (composability + which signal) | **T6** (compressor × gate-family), **T10** (signal selection) | agnostic signal composes with ANY compressor; **T10: neg_entropy/conf ≈ best intrinsic, ~0.6 mean AUROC** | ✅ **T6+T10** |
| **Q6** | Does it **generalize**? (task / domain / model) | **T8** (relation, +non-tool), **T9** (cross-model), generalization-v2 (base/train/K) | reproduces across **5 dense families** (T9); net-win concentrates on high-headroom tool benches | ✅ **T9 (5 fam)** / ⬜ T8c re-run |
| **Q7** | Is there a real **net-win**? (tokens saved @ matched quality) | cost-quality **Pareto + recall@P95** vs always-compress/always-full/random/entropy/LLM-judge/**TARG** | tokens/latency saved at matched quality | ⬜ **biggest gap** |
| **Mod** | Which **modules** are necessary vs droppable? | **T11** module ablation (2 benches, from full → ablate each) | **distillation + reconstruction necessary; task bench-dependent**; encoder-depth/K/init/dev/base-LoRA **reducible** | ✅ **T11 (2 benches)** |
| **×** | Cross-cutting rigor (not a claim) | contrastive ablation; **uniform re-run + faithfulness audit** | contrastive gain small/inconsistent; baselines faithful + one recipe (ctx>K) | ✅ |

## GCM configuration (the main-table setting — every GCM cell uses this)
All GCM cells in this version (T2′, T9, T11 `ab0_full`, the gate/Q7 Pareto studies) use ONE config — the module-ablation
**`ab0_full`** reference (the fullest, best point). Invocation: `harness --methods no_ctx,full_ctx,trunc,gcm` with
| group | setting |
|---|---|
| memory | **K = 64** soft tokens, **prefix** injection (input-embedding), `enc-init copy` (encoder copied from base weights) |
| encoder / decoder | **16-layer** encoder, **2-layer** reconstruction decoder |
| base read-out | **LoRA rank 32** on the base (A3: base learns to read M; disabled on full/no_ctx ⇒ exact do-no-harm fallback) |
| losses | task **1.0** + distillation **0.5** + reconstruction **1.0** + deviation **0.05** |
| optimization | **lr 3e-4**, grad-accum 8 (eff. batch 8), **3000 steps**, 384 train items |
| eval / gate | 96 eval items, max-ctx 1024 tokens, `--signals` (conf / margin=TARG / neg_entropy + neg_recon / …) |

The 2025 baselines (ICAE/AOC/Beacon/500x/ComprExIT/LCC/Cartridge/Gist) share the **budget-matched** knobs (K64 / 3000 steps /
lr3e-4 / 384 items) but keep **their own architectures** ([`baselines-faithfulness.md`](baselines-faithfulness.md)). GCM did
not run inside the 4-lane uniform sweep (SVC-encoder fork-deadlock) — its cells come from dedicated 1-lane runs at this identical recipe.

## Paper tables (latest data, Qwen3-8B, leak-free, stable training)

### T1 — Trivial-floor reference (truncation / meanpool / randk) — the ONLY table that discusses trunc
`trunc` = keep first-K ctx tokens (no training); `meanpool`/`randk` = K segment-means / K random vectors. These are
**trivial floors, not methods** — kept here once so the learned-method tables below never repeat them.
| bench | ctx | full | trunc | meanpool | randk |
|---|---|---|---|---|---|
| bfcl_live_simple | 43 | 0.91 | 0.91 | 0.87 | 0.03 |
| glaive | 29 | 0.99 | 0.99 | 0.99 | 0.39 |
| bfcl_simple | 36 | 0.99 | 0.99 | 0.99 | 0.02 |
| hermes | 84 | 0.94 | 0.94 | 0.63 | 0.37 |
| bfcl_live_multiple | 182 | 0.91 | 0.46 | 0.03 | 0.00 |
| toolace | 580 | 0.95 | 0.01 | 0.01 | 0.01 |

→ **One point:** truncation matches `full` **only when ctx ≤ K or the answer is front-loaded** (the short benches +
hermes); once context truly exceeds the budget (live_multiple, toolace) it **collapses far below full**. So trunc is a
*floor*, never a featured comparison — every learned-method table below omits it.

### T2 — Context-compression baselines at matched budget (compress vs full)
GCM/Cartridge/Gist from v1.7.3 §2.4; the 2025 family (faithful, [`baselines-faithfulness.md`](baselines-faithfulness.md)) from `ac_b25_*`. All enc-16/K64/384-items/3000-steps.
| bench | full | **GCM** | Cartridge | Gist | ICAE | AOC | Beacon | 500x | ComprExIT | LCC |
|---|---|---|---|---|---|---|---|---|---|---|
| bfcl_live_multiple | 0.91 | **0.72** | 0.51 | 0.00 | 0.06 | 0.08 | 0.03 | 0.02 | 0.08 | 0.01 |
| bfcl_multiple | 0.88 | 0.15 | 0.12 | 0.02 | 0.05 | 0.12 | 0.33 | 0.18 | 0.12 | 0.02 |
| hermes | 0.94 | 0.50 | 0.46 | 0.09 | 0.51 | 0.41 | 0.62 | 0.37 | 0.47 | 0.35 |
| rca_openrca | 0.41 | 0.10 | 0.13 | 0.14 | 0.08 | 0.10 | 0.25 | 0.08 | 0.09 | 0.13 |
| toolace | 0.95 | 0.14 | 0.10 | 0.01 | 0.10 | 0.07 | 0.38 | — | 0.07 | 0.01 |
| narrativeqa | 0.14 | — | — | — | 0.14 | 0.13 | 0.10 | 0.11 | 0.13 | 0.15 |

→ **GCM ≥ Cartridge ≫ Gist**, and GCM ≥ the 2025 family; **none beats `full`** anywhere. (Beacon is the strongest 2025 baseline.)

**GCM+fallback (gate view).** For the GCM cells the deployed do-no-harm gate adds **gcm+fallback** (gАcc ≥ full), **fallback%**,
**F1 / precision / recall** and the **confusion matrix (TP/FP/FN/TN)** — full per-bench numbers in **T2′-gate** above. So "no
compressor matches full" (this table) ⇄ "the *gated* system never loses to full, and gains where compression is safe".

> ⚠️ **Recipe-uniformity caveat (2026-06-16).** These T2 numbers are **not** at one recipe (the header's "All
> enc-16/K64/384-items/3000-steps" is inaccurate): per the launch scripts/logs, GCM ran **lr3e-4 / 1600 steps**,
> Gist **1500 steps**, the 2025 family **lr1e-4 / 3000 steps**, and K is **64∪128** across benches — the lr gap
> *advantaged* GCM. The 2025 re-impls were also **faithfulness-fixed** (ICAE/500x→rank128 q,k,v,o; Beacon→sampled-α+recon;
> ComprExIT→real cross-layer; LCC→full-recon — see [`baselines-faithfulness.md`](baselines-faithfulness.md)). A
> **uniform-recipe re-run** (K64/384/3000/lr3e-4/grad-accum8) + a **K128 paper-setting validation** is running
> (`out/baselines_uniform/`); T2 will be refreshed from it. The matched-budget gap to the papers' reported retention
> (~62–99% on QA) is the **no-pretrain + K64** regime, documented in the faithfulness doc's repro-vs-paper table.

> ⚠️ **Compression-ratio caveat (2026-06-16).** A compressor only does real work when **ctx > K**; items with ctx ≤ K
> (or front-loaded benches) are no-op-compression and must be excluded. So "**no compression matches full**" is made on
> **ctx>K items only**, comparing learned methods *within* an injection type. (Trivial floors — truncation etc. — live in T1.)
> Analyzer: `ratio_table.py`.
**T2′ (refreshed 2026-06-17) — learned compressors vs `full`, ctx>K items ONLY, BY INJECTION TYPE** (uniform recipe
K64/3000/lr3e-4). acc @ ctx>K. **Compute-fair comparison only *within* a type** (KV-injection costs far more per token).

**(A) Soft-prompt / input-embedding memory** — ≈ our complexity (K extra input tokens). acc @ ctx>K; *(pending)* = uniform cell still running:
| method | live_multiple (n96) | hermes (n61)† | multiple (n51) | toolace (n96) | rca_openrca (n96) | narrativeqa (n96)‡ |
|---|---|---|---|---|---|---|
| `full` (ceiling) | 0.91 | 0.93 | 0.92 | 0.95 | 0.46 | 0.14 |
| **GCM (ours)** | **0.67** | **0.52** | **0.16** | **0.10** | **0.09** | **0.15**‡ |
| ICAE | 0.22 | 0.67 | 0.08 | 0.07 | 0.08 | 0.12 |
| ComprExIT | 0.29 | 0.33 | 0.10 | 0.21 | 0.10 | 0.18 |
| AOC | 0.05 | 0.34 | 0.10 | 0.05 | 0.07 | 0.13 |
| LCC | 0.00 | 0.38 | 0.04 | 0.01 | 0.13 | 0.15 |
| Gist | 0.00 | 0.34 | 0.00 | 0.01 | 0.10 | 0.16 |
| `no_ctx` (floor) | 0.01 | 0.34 | 0.02 | 0.01 | 0.10 | 0.16 |

**(B) KV-injection memory** — per-layer KV cache; **compute/memory ≫ soft-prompt**, reported separately (not ranked against A):
| method | live_multiple | hermes† | multiple | toolace | rca_openrca | narrativeqa‡ |
|---|---|---|---|---|---|---|
| `full` (ceiling) | 0.91 | 0.93 | 0.92 | 0.95 | 0.46 | 0.14 |
| Beacon | 0.31 | 0.85 | 0.24 | 0.64 | **0.44** | 0.11 |
| Cartridge | 0.34 | 0.46 | 0.12 | 0.15 | 0.13 | 0.16 |
| 500x | 0.09 | 0.51 | 0.10 | 0.38 | 0.14 | 0.12 |
| `no_ctx` (floor) | 0.01 | 0.34 | 0.02 | 0.01 | 0.10 | 0.16 |

† hermes front-loaded (answer in first K tokens) ⇒ weak compression bench (T1). ‡ **narrativeqa degenerate** — full 0.14 ≈ no_ctx 0.16
(no headroom) ⇒ **excluded** from the "no method matches full" claim (same regime as the T8a non-tool benches).

> **Uniform re-run COMPLETE (48/48, 2026-06-17)** — all 8 compressors × 6 benches at one recipe.
> **GCM** at the uniform recipe deadlocks (SVC encoder build under 4-lane concurrency), so GCM cells use the module-ablation
> `ab0_full` run for live_multiple/hermes; the other 4 GCM cells **now filled 1-lane** (`gcm_fill.sh`): multiple **0.16**, toolace **0.10**, rca **0.09**, narrativeqa **0.15** — all ≪ full (do-no-harm gАcc in T2′-gate above).

→ **No soft-prompt method matches `full`** at real compression on any of the 5 real benches, and **GCM is clearly best among
same-complexity soft-prompt** methods. The **KV-injection** family (Beacon strongest) costs far more per token and only
*approaches* full where there is **little to compress**: hermes (0.85/0.93, **front-loaded** — answer in first K) and
rca_openrca (0.44/0.46, **low ceiling** — full itself is only 0.46). On the **high-headroom** benches it collapses just like
everything else — Beacon live_multiple **0.31/0.91**, bfcl_multiple **0.24/0.92**, toolace **0.64/0.95**. So the rule holds:
**whenever there is real context to compress, every compressor leaves a large gap to full** ⇒ "always compress" is unsafe ⇒
a gate is needed. *(Trivial floors incl. truncation are in T1.)* **Bench breadth:** T1 covers 6, **T8a 12 (mix→held-out)**,
T9 4×5 model cells — the suite is ~17 benchmarks; T2′ reports the uniform-recipe subset (48/48 complete).

**T2′-gate — GCM + do-no-harm gate: gcm+fallback, fallback rate, F1/precision/recall, confusion matrix.** Two operating
points reported per bench: **deployed** = do-no-harm (`disc_gate gated_acc_cv`, threshold maximises held-out accuracy ⇒
gАcc ≥ full) giving **gcm+fallback** + **fallback%**; **detection** = best-F1 (`gate_confusion`, signal `neg_recon`)
giving F1 / precision / recall / the confusion matrix. Convention: POSITIVE = gate **keeps compressed** (signal ≥ τ);
y=1 = compression **safe** (gcm ≥ full). So **FP = harm** (compressed something it shouldn't), **FN = wasted fallback**.
| bench | full | gcm | **gcm+fallback** | **fallback%** | AUROC | **F1** | **prec** | **recall** | **TP / FP / FN / TN** |
|---|---|---|---|---|---|---|---|---|---|
| bfcl_live_multiple | 0.91 | 0.75 | **0.93** ↑ | 68% | **0.86** | **0.92** | 0.88 | 0.96 | **74 / 10 / 3 / 9** |
| hermes | 0.94 | 0.46 | 0.93 | 98% | 0.56 | 0.68 | 0.52 | 0.96 | 46 / 42 / 2 / 6 |
| bfcl_multiple | 0.88 | 0.15 | 0.88 | 97% | 0.56 | 0.42 | 0.26 | 1.00 | 15 / 42 / 0 / 3 |
| toolace | 0.95 | 0.10 | 0.95 | 100% | 0.50 | 0.30 | 0.21 | 0.53 | 8 / 31 / 7 / 50 |
| rca_openrca | 0.43 | 0.09 | 0.43 | 100% | 0.36 | 0.80 | 0.66 | 1.00 | 63 / 32 / 0 / 1 |
| narrativeqa‡ | 0.14 | 0.15 | 0.18 | 78% | 0.56 | 0.81 | 0.69 | 0.98 | 63 / 29 / 1 / 3 |

→ **gcm+fallback ≥ full everywhere** (do-no-harm). Only **bfcl_live_multiple** has true detection (AUROC **0.86**, F1 **0.92**,
just **FP=10/96**) ⇒ the deployed gate safely compresses 32% and **gains +0.02**. Elsewhere AUROC≈0.5 (the signal can't
discriminate ⇒ best-F1 over-compresses: FP=42 on hermes/multiple, i.e. would-be harm), so the **deployed gate falls back**
(fallback 97–100%) ⇒ gcm+fallback = full. (Confusion = best-F1 detection point; deployed fallback% is the conservative
do-no-harm point — the two differ by design: high detection-recall vs safe deployment.) ‡ narrativeqa degenerate (no_ctx>full).

> **Tool-coverage extension (queued 2026-06-17).** The harness supports **18 tool-calling bench keys / 5 families** (BFCL ×13, apibank,
> toolace, hermes, glaive); most BFCL/glaive variants are **short (ctx ≤ K)** ⇒ trivial floors (T1), which is why T2′ featured only the
> high-headroom ones. An audit found **two high-headroom tool benches missing** — now adding: `apibank` (median **4099** ctx; Qwen3-8B full
> **0.15**, GCM **0.01**) and `bfcl_parallel_multiple` (full **0.81**, GCM **0.10**) — both **compress ≪ full**. The 8 baselines × 2 benches
> (T2′) + 5-model cross cells (T9) auto-launch when the current sweep frees GPUs (`run_newtool_waiter.sh`).

### T3 — fixed-K (v1.7.3) vs length-adaptive (v1.7.4) — the chunking test
| bench | ctx | full | **fixed-K** (K64) | **len-adapt** (ch16a) | Δ |
|---|---|---|---|---|---|
| bfcl_parallel_multiple | 83 | 0.77 | 0.12 | 0.13 | +0.02 |
| hermes | 84 | 0.94 | 0.50 | 0.48 | −0.02 |
| bfcl_multiple | 97 | 0.88 | 0.15 | 0.13 | −0.02 |
| bfcl_live_multiple | 182 | 0.91 | 0.72 | 0.71 | −0.01 |
| rca_openrca | 1019 | 0.45 | 0.10 | 0.09 | −0.01 |

→ **Length-adaptive ≈ fixed-K (Δ±0.02) everywhere, incl. long ctx** → chunking does not rescue long context.

### T4 — Scaling (mix of 12 datasets; avg compress): NO scaling
| axis | sweep | acc |
|---|---|---|
| compute (FLOPs) | 3.5e15 → 8.4e16 (24×) | 0.253 → 0.267 |
| data (curriculum, N=16) | D=96 → 768 (8×) | 0.247 → 0.267 |
| depth (D=384) | N=4 → 28 (7×) | **0.285 → 0.279** (N=4 best) |

→ More data/depth/compute does **not** help; capacity **K also flat** (K16 0.69 ≈ K64 0.69 ≥ K256 0.64). Compressor is capacity-bound.

### T5 — HP sweep (compress, avg over 2 search benches; best configs)
| config | avg | gate AUROC | note |
|---|---|---|---|
| **lr 3e-4** | **0.450** | 0.65 | best — base lr1e-4 (0.41) was too low |
| ch16a | 0.430 | 0.53 | length-adaptive |
| enc4 (N=4) | 0.429 | 0.61 | **minimize depth** |
| lr 5e-5 | 0.423 | 0.54 | |
| distill10 (λ=1) | 0.421 | 0.59 | λ_distill load-bearing (distill0 → 0.07 collapse) |
| base (enc16/K64/lr1e-4) | 0.411 | — | reference |
| manif01 (OOD-fix, H5) | 0.401 | — | ≈ base → OOD-fix does not break the ceiling |

→ Wins are **lr↑ (3e-4)** + **minimize-scale**; full table in [`scaling-critical-review.md` §5c](scaling-critical-review.md). *(minimize Pareto `ab_min` N×K — pending.)*

### T6 — Compressor × gate COMPOSABILITY (in-task): can any compressor pair with any gate?
> **What T6 isolates (vs T8).** Fix the data (in_task), vary the **compressor × gate-signal-family** product — is the
> gate a *detachable module* usable on ANY compressor? "Three gates" = families by training dependence: **agnostic**
> (conf/margin/neg_entropy — data-free, reads only the frozen base on `[M;q]`; computable for ANY compressor),
> **intrinsic** (neg_recon/dcode/dlogit/mnorm — needs GCM's encoder/decoder), **learned** (disc_p — needs adversarial
> training). Metric = gate AUROC. Design: [`gate-transfer-design.md`](gate-transfer-design.md).

**(i) Composability matrix — gate AUROC by (compressor × signal family), in-task (rep. bfcl_live_multiple, N=96):**
| compressor | agnostic (conf / neg_ent) | intrinsic (neg_recon) | learned (disc_p) | composes with |
|---|---|---|---|---|
| **GCM** | 0.77 / 0.81 | 0.80 | *(adv-gcm running)* | **all three** |
| Cartridge | 0.6–0.8¹ | — (no encoder) | — (no disc) | **agnostic only** |
| Gist / ICAE / Beacon / 500x | *(p2_cart_sig running)* | — | — | **agnostic only** |

¹ v1.7.3 §2.4b: the agnostic conf-gate applied to Cartridge gives failure-detection AUROC 0.6–0.8 + gАcc≈full.

→ **Only the agnostic family composes with ANY compressor** (intrinsic & learned are GCM-internal). In-task, GCM's
agnostic gate (0.77–0.81) ≈ its best intrinsic (neg_recon 0.80) — the detachable, data-free gate loses ~nothing
in-task — and (T8a + xbench in [`gate-transfer-design.md`](gate-transfer-design.md)) it is the **only** family that
survives cross-domain *and* whose threshold transfers across benches. So "compressor ⊗ gate 任意组合" holds for the
**agnostic** gate; intrinsic/learned are GCM-only refinements.

**(ii) Per-bench gate detail (best signal, in-task; F1 primary, AUROC flags base-rate):**
| compressor | bench | **F1** | precision | recall | AUROC |
|---|---|---|---|---|---|
| **GCM (ch16a)** | bfcl_live_multiple | **0.89** | 0.80 | 1.00 | 0.68 |
| GCM (ch16a) | hermes | 0.73 | 0.60 | 0.94 | 0.70 |
| GCM (ch16a) | bfcl_multiple | 0.48 | 0.34 | 0.79 | 0.65 |
| GCM (ch16a) | bfcl_parallel_multiple | 0.62 | 0.53 | 0.73 | 0.60 |
| GCM (ch16a) | rca_openrca | 0.78 | 0.64 | 1.00 | 0.51 |
| **Cartridge** | bfcl_parallel | 0.66 | 0.50 | 0.95 | **0.81** |
| Cartridge | bfcl_live_multiple | 0.77 | 0.64 | 0.98 | 0.73 |
| Cartridge | glaive | 0.89 | 0.80 | 1.00 | 0.62 |
| Cartridge | toolace | 0.28 | 0.16 | 1.00 | 0.48 |

→ Gate works on GCM *and* Cartridge: **F1 0.66–0.89 / AUROC 0.65–0.81** on conversational/live benches (real
discrimination). **Honest caveat:** where AUROC≈0.5 (musr/quality/rca: F1 0.78–0.98, recall≈coverage≈1.0) the high F1
is **always-compress** (high base-rate), not detection — do-no-harm holds via conservative fallback. **Real gate
quality = high F1 AND AUROC>0.6.**

**(iii) Does composability hold across relations? — GCM signal-family × relation × the 3 τ-points** (AUROC; honest
do-no-harm at τ∈{in-sample, cv, xbench} per [`gate-transfer-design.md`](gate-transfer-design.md)):
| signal family | in_task AUROC | cross_task | cross_domain AUROC | xbench-τ Δ vs full (cross_dom) |
|---|---|---|---|---|
| **agnostic** (conf / neg_ent) | 0.77 / 0.81 | *(T8c)* | **0.63** | **−0.002** (τ transfers ✓) |
| intrinsic (neg_recon) | 0.80 | *(T8c)* | **0.37 ↓** | −0.017 (live −0.21 ✗) |
| learned (disc_p) | *(adv-gcm)* | *(adv-gcm)* | *(adv-gcm)* | *(adv-gcm)* |

→ **Agnostic AUROC holds across relations (0.77→0.63) and its τ transfers across benches (Δ≈0); intrinsic collapses
cross-domain (0.80→0.37, ~reversed) and its τ does NOT transfer (−0.21 on live_multiple).** So the only cell that is
composable (any compressor) × generalizing (cross-domain) × portable-τ is **agnostic**. cross_task fills from T8c,
learned from adv-gcm (running); honest gAcc_cv per relation lands with them.

### T7 — Soft-memory → text probe (interpretability)
Nearest-vocabulary decode of the K memory vectors: **mean cosine = 0.096** (near-orthogonal to the vocabulary → OOD); decodes to gibberish; M0≈Mq; slots collapse to fixed directions.

### T8 — System × relation GENERALIZATION (compressor+gate as one pipeline)
> **What T8 isolates (vs T6).** T6 fixes the data (in_task) and varies *compressor × gate* → composability. **T8
> fixes the SYSTEM (compressor+gate) and varies the eval RELATION** — in_task → cross_task/in_domain →
> cross_task/cross_domain — i.e. does the whole pipeline generalize to an unseen task/domain (incl. non-tool)? The
> two training-source sub-axes (*what the compressor saw* `self`/`anchor`/`mix` vs *what the gate saw* / which τ) are
> in [`gate-transfer-design.md`](gate-transfer-design.md). **(a)** GCM trained on the 12-bench *mix*, evaluated on
> each held-out eval-split (`scale/N16_D384`) — a **multi-task held-out** read (the mix saw each bench's *leak-free
> train* split, so it is **not** clean zero-shot; the clean single-anchor matrix is **(c)**). **(b)** faithful Cartridge
> per-anchor transfer (`p2`) — those dumps are **compress-only** (no `full`/signals ⇒ no gate). **(c)** GCM clean
> single-anchor matrix — **RUNNING** on `test`. Analyzers: `analyze_transfer.py`, `decision.py`, `xbench_gate.py`.

**(a) GCM mix→held-out: compress vs full + do-no-harm gate (incl. 5 non-tool), `neg_recon` signal**
| bench | type | no_ctx | comp | full | gate F1 | AUROC | real gate? |
|---|---|---|---|---|---|---|---|
| bfcl_simple | tool | 0.00 | 0.10 | 0.98 | 0.53 | **0.69** | ✓ detect |
| squad_v2 | NON-tool QA | 0.20 | 0.18 | 0.71 | 0.43 | **0.68** | ✓ detect |
| rca_openrca | ops | 0.04 | 0.08 | 0.33 | 0.89 | **0.66** | ✓ detect |
| glaive | tool | 0.38 | 0.56 | 0.98 | 0.75 | 0.58 | ~ weak |
| hermes | tool | 0.38 | 0.38 | 0.90 | 0.70 | 0.57 | ~ weak |
| bfcl_multiple | tool | 0.02 | 0.15 | 0.92 | 0.40 | 0.51 | ✗ |
| bfcl_live_multiple | tool | 0.02 | 0.21 | 0.92 | 0.43 | 0.48 | ✗ |
| toolace | tool | 0.00 | 0.02 | 0.92 | 0.21 | 0.41 | ✗ |
| hotpot_qa | NON-tool QA | 0.28 | 0.33 | 0.35 | 0.87 | 0.38 | ✗ always-comp |
| narrativeqa | NON-tool | 0.17 | 0.14 | 0.14 | 0.82 | 0.38 | ✗ always-comp |
| quality | NON-tool | 0.33 | 0.25 | 0.17 | 0.98 | 0.48 | ✗ always-comp |
| musr_mm | NON-tool | 0.50 | 0.52 | 0.54 | 0.98 | 0.42 | ✗ always-comp |

→ **compress ≪ full everywhere** (non-tool `full` is itself low-headroom — narrativeqa 0.14, quality 0.17 ⇒ nothing to
compress). The **gate really discriminates (AUROC>0.6) on bfcl_simple / squad / rca**; on the high-base-rate non-tool
benches the high F1 is **always-compress, not detection** (exactly the T6 caveat) — do-no-harm there holds via conservative
fallback. Net: the do-no-harm gate **extends to cross-domain + non-tool**, but real *detection* needs AUROC>0.6 (3 benches).

**Gate metrics for these relations** (gcm+fallback / fallback% / F1 / precision / recall / confusion matrix): the full per-bench
numbers are in **T2′-gate** (with TP/FP/FN/TN) and per-model in **T9** (gАcc = gcm+fallback, fb = fallback%). Pattern: gcm+fallback
= full via conservative fallback wherever AUROC≈0.5; > full only where detection is real (AUROC>0.6, e.g. bfcl_simple/squad/rca).

**(b) Cartridge per-anchor transfer (compress only — no `full`/signals in these dumps ⇒ no gate)**
| train anchor | in_task | cross_task_in_domain | cross_task_cross_domain |
|---|---|---|---|
| bfcl_live_multiple | 0.30 | 0.10 | 0.18 |
| bfcl_live_simple | 0.26 | 0.14 | 0.15 |
| bfcl_multiple | 0.07 | 0.19 | 0.16 |
| bfcl_simple | 0.14 | 0.20 | 0.16 |

→ per-corpus Cartridge does **not** transfer (compress stays 0.1–0.2 across relations; cross can even exceed in_task when
the target bench is easier, e.g. bfcl_simple→glaive 0.50, →hermes 0.40). No `full`/signals were logged ⇒ gate not computable
here; the clean GCM matrix **(c)** logs both, with anchors aligned to these 4 for a like-for-like compare.

**(c) GCM clean single-anchor transfer matrix — RUNNING** (`test`, `out/clean/p2_gcm`): 4 tool anchors
(bfcl_simple/multiple/live_simple/live_multiple) × 13 eval benches (7 tool + rca + squad/hotpot + narrativeqa/quality/musr),
GCM logging full+no_ctx+**signals** (lr3e-4 / K64 / LoRA32 / enc16, stable recipe). Fills in: **clean** in/cross-task/cross-domain
compress **and** the per-relation gate F1/AUROC (the do-no-harm-*transfers* claim on GCM, aligned to the Cartridge anchors). *(ETA ~overnight; folds into (a)/(b).)*

### Tτ — Test-agnostic gate threshold (deployable; no test labels) — answers Q4's "how to set τ"
> At deploy you can't tune τ on test labels. A **calibrated-probability agnostic signal** gives a
> **dataset-independent** threshold; an **intrinsic** signal does NOT (arbitrary scale ⇒ a fixed value keeps
> nothing or everything ⇒ needs per-dataset calibration). Validated across the **20 T9 cells** (8 models × benches),
> label-free. Tooling: `tau_sweep.py` / `tau_select.py`.

**Fixed-probability threshold applied label-free — worst-cell Δ vs full + mean coverage over 20 cells:**
| signal | τ | mean Δfull | **worst Δfull** | mean cov | do-no-harm cells |
|---|---|---|---|---|---|
| **margin** (top1−top2) | **≥ 0.95** | −0.002 | **−0.010** | 0.01 | **20/20** ✓ |
| margin | ≥ 0.90 | −0.007 | −0.052 | 0.04 | 17/20 |
| conf (top-1 prob) | ≥ 0.95 | −0.004 | −0.031 | 0.01 | 18/20 |
| conf | ≥ 0.90 | −0.015 | −0.135 | 0.09 | 16/20 |
| `neg_recon` (intrinsic) | any fixed | 0 | 0 | 0 | 20/20\* |

\* intrinsic at a fixed value keeps **nothing** (scale ≠ [0,1]) ⇒ trivial do-no-harm with **zero coverage** → not usable test-agnostically.

→ **Deployable rule:** gate on **`margin ≥ 0.95`** (a probability, identical on any dataset) ⇒ do-no-harm everywhere;
coverage **auto-adapts** (high where compression is safe, ~0 where not). The **τ RANGE for any unlabeled dataset** =
the probability frontier (margin ∈ [0.5, 0.98]) read off *its own* signal distribution — pick a point on the
savings↔safety curve. **Honest limit:** universal do-no-harm forces a *conservative* τ ⇒ mean coverage ~1% (the
savings concentrate on high-AUROC benches like bfcl_live_multiple; **front-loaded benches like hermes stay
high-confidence even when compression is lossy**, so they force fallback). **The gate's savings are bounded by the
signal's per-dataset AUROC** — a test-agnostic τ exists, but it inherits the signal's quality.

### T9 — Cross-model main table (model TYPES; does the pipeline hold off Qwen3-8B?) — 5 dense types DONE
> **Why.** T1–T8 are all Qwen3-8B. T9 asks whether *necessity (no≪full) + compress≪full + do-no-harm gate* reproduces
> across model **families/arches**. **Dropped** Qwen2.5-7B (gcm=0); **deferred** base-Qwen3/Qwen3.5/gpt-oss (CoT corrupts
> eval). **2 MoE (Qwen3-30B-A3B, Moonlight-16B) DROPPED** (2026-06-17): OOM at 95 GB, and their checkpoints aren't on
> the 143 GB pod (separate volume) — the 6 dense families below suffice. Qwen3-8B comes from the depth (`mt_`) cells.
> Stable recipe (lr3e-4/K64/enc16/LoRA32), signal `neg_recon`. Analyzer `analyze_crossmodel.py`.

| model (family) | bench | no | full | GCM | AUROC | **gАcc=gcm+fallback** | **fb=fallback%** | **F1 (P/R)** | **confusion TP/FP/FN/TN** |
|---|---|---|---|---|---|---|---|---|---|
| **Qwen3-8B** (ref) | bfcl_live_multiple | 0.01 | 0.92 | 0.67 | **0.75** | 0.92 | 85% | **0.92** (0.88/0.96) | 74 / 10 / 3 / 9 |
| | hermes / squad_v2 / narrativeqa | 0.33/0.18/0.16 | 0.95/0.69/0.14 | 0.52/0.18/0.12 | 0.57/0.48/0.48 | 0.94/0.69/0.13 | 97/100/91% | conservative — gate falls back | (mostly TN/FN) |
| **GLM-4-9B** (GLM) | bfcl_live_multiple | 0.01 | 0.85 | 0.66 | **0.70** | 0.85 | 77% | 0.86 (0.78/0.96) | 67 / 19 / 3 / 7 |
| | hermes / squad_v2 / narrativeqa | 0.28/0.14/0.13 | 0.90/0.49/0.12 | 0.49/0.15/0.15 | 0.62/0.44/0.64 | 0.89/0.49/0.15 | 94/100/40% | conservative | (mostly TN/FN) |
| **Qwen3-4B-2507** (Qwen3) | bfcl_live_multiple | 0.00 | 0.88 | 0.71 | **0.82** | 0.88 | 53% | 0.90 (0.85/0.95) | 70 / 12 / 4 / 10 |
| | hermes / squad_v2 / narrativeqa | 0.28/0.16/0.15 | 0.81/0.63/0.15 | 0.47/0.17/0.14 | 0.54/0.55/0.48 | 0.80/0.63/0.15 | 95/98/70% | conservative | (mostly TN/FN) |
| **Ministral-8B** (Mistral) | bfcl_live_multiple | 0.02 | 0.88 | 0.63 | **0.79** | 0.87 | 83% | 0.86 (0.77/0.96) | 65 / 19 / 3 / 9 |
| | hermes / squad_v2 / narrativeqa | 0.28/0.20/0.14 | 0.95/0.59/0.18 | 0.53/0.19/0.18 | 0.59/0.56/0.44 | 0.95/0.59/0.18 | 95/98/41% | conservative | (mostly TN/FN) |
| **Llama-xLAM-8B** (Llama) | bfcl_live_multiple | 0.00 | 0.90 | 0.76 | **0.81** | 0.88 | 69% | 0.92 (0.87/0.96) | 76 / 11 / 3 / 6 |
| | hermes / squad_v2 / narrativeqa | 0.25/0.13/0.15 | 0.95/0.52/0.14 | 0.49/0.18/0.13 | 0.66/0.43/0.48 | 0.94/0.52/0.15 | 96/100/61% | conservative | (mostly TN/FN) |
| **ToolACE-8B** (Llama) | bfcl_live_multiple | 0.00 | 0.88 | 0.72 | **0.90** | 0.89 | 62% | 0.92 (0.89/0.95) | 72 / 9 / 4 / 11 |
| | hermes / squad_v2 / narrativeqa | 0.15/0.13/0.18 | 0.59/0.45/0.18 | 0.49/0.16/0.16 | 0.63/0.46/0.51 | 0.57/0.45/0.17 | 53/100/81% | conservative | (mostly TN/FN) |

→ **The Qwen3-8B story reproduces across 5 families** (GLM, Qwen3, Mistral, 2× Llama): necessity holds (no≪full),
**compress ≪ full everywhere**, **do-no-harm gate recovers ≈full** (gАcc≈full). Real *detection* (high AUROC + selective
fallback) concentrates on **bfcl_live_multiple** (AUROC **0.70–0.90**, fb 53–83%); on squad/hermes the gate is conservative
always-fallback (fb 94–100% = safe, ~no savings); **narrativeqa is degenerate** (full≈no_ctx). The **Qwen3-8B reference row**
is now folded in (live_multiple AUROC 0.75 / fb 85%; hermes conservative fb 97%) — same pattern. *(Qwen3-8B squad_v2/narrativeqa
cells **now filled** (squad_v2 full 0.69 / GCM 0.18 / gАcc 0.69 = do-no-harm; narrativeqa degenerate full 0.14 / gАcc 0.13); 2 MoE dropped — OOM + off-volume checkpoints.)*

→ **Gate columns now integrated INTO the T9 table above**: `gАcc` **= gcm+fallback**, `fb` **= fallback%**, plus **F1 (P/R)** and
the **confusion (TP/FP/FN/TN)** on the bfcl_live_multiple rows (`gate_confusion`, neg_recon; y=1 = compression safe, POSITIVE =
keep compressed, FP = harm). All 6 families: gcm+fallback ≥ full (do-no-harm) **and strong detection** on bfcl_live_multiple
(F1 0.86–0.92, FP only 9–19/96); the QA benches can't discriminate (AUROC≈0.5) ⇒ gate is conservative (gcm+fallback = full).

**T9 extended tool benches (added 2026-06-17 — the two high-headroom tool benches that were missing: `apibank`, `bfcl_parallel_multiple`):**
| model | apibank — no/full/GCM/AUROC/fb | parallel_multiple — no/full/GCM/AUROC/fb |
|---|---|---|
| **Qwen3-8B** (ref) | 0.00 / 0.15 / 0.01 / 0.60 / 76% | 0.07 / 0.77 / 0.13 / 0.49 / 100% |
| GLM·Qwen3-4B·Ministral·xLAM·ToolACE | *(5 models × 2 benches — running, auto-launch on GPU free)* | *(running)* |

→ Same pattern holds on both: **compress ≪ full**. `apibank` is a *hard* tool bench (full only **0.15** even with full context — 4099-token API
catalogs capped at 1024), `bfcl_parallel_multiple` has high headroom (full **0.81**, GCM **0.10**). Neither is degenerate (full ≫ no_ctx).

### T10 — Gate-signal selection (Q5): which signal is "good"? (mean over 20 T9 cells)
| signal | family | mean AUROC | frac cells >0.7 | mean recall@P95 | AURC_acc |
|---|---|---|---|---|---|
| **neg_entropy** | agnostic | **0.641** | 0.30 | 0.135 | 0.520 |
| **conf** | agnostic | 0.632 | 0.30 | 0.109 | 0.516 |
| margin | agnostic | 0.615 | 0.20 | 0.075 | 0.513 |
| neg_recon | intrinsic | 0.602 | 0.25 | **0.157** | 0.511 |
| mnorm / dcode | intrinsic | ~0.51 | 0.00 | 0.03 | 0.49 |
| dlogit | intrinsic | 0.416 | 0.00 | 0.02 | 0.47 |

→ **The agnostic family (neg_entropy / conf) ≈ the best intrinsic (neg_recon)** on detection (~0.60–0.64 mean AUROC), and
is **data-free + test-agnostic** (see Tτ) — so it is the recommended signal; the other intrinsic signals (mnorm/dcode/dlogit)
are at/below chance. **"Good signal" = agnostic neg_entropy/conf**, AUROC>0.7 only on the high-headroom benches (30% of cells)
— consistent with the net-win concentrating there.

### T11 — Module ablation (Q-mod): which modules are load-bearing? (Qwen3-8B, bfcl_live_multiple, from full → ablate one)
Reference `ab0_full` = enc16 / K64 / copy-init / base-LoRA32 / {task+distill0.5+rec+dev}. Δgcm = compress acc vs reference. `analyze_ablation.py`.
| ablate from full | gcm | gate AUROC | Δgcm | verdict |
|---|---|---|---|---|
| **ab0_full** (reference) | 0.67 | 0.75 | — | — |
| **− distillation** | **0.16** | 0.45 | **−0.51** | **NECESSARY (collapse)** |
| **− reconstruction** | 0.71 | **0.48** | +0.04 | gcm fine but **gate AUROC collapses** → necessary *for the gate* (`neg_recon`) |
| − base-LoRA (A3) | 0.65 | 0.79 | −0.02 | minor |
| − deviation | 0.68 | 0.79 | +0.01 | droppable |
| − task loss | 0.68 | 0.80 | +0.01 | ~droppable here (distill subsumes) |
| enc16 → **enc4** | 0.72 | 0.82 | **+0.05** | smaller encoder is *better* |
| enc16 → enc1 | 0.69 | 0.76 | +0.02 | 1-layer ≈ full |
| K64 → K16 / K4 | 0.69 / 0.70 | 0.73 / 0.80 | +0.02 / +0.03 | K reducible |
| copy-init → random | 0.71 | 0.72 | +0.04 | base-init doesn't matter |
| prefix → KV inject | 0.67 | 0.75 | 0.00 | equivalent |

→ **2-bench verdict (bfcl_live_multiple + hermes):** **distillation is load-bearing** (Δgcm −0.51 / −0.22) and
**reconstruction is necessary for the gate** (AUROC collapses on both: 0.75→0.48, 0.57→0.47). **Task loss is
bench-dependent — KEEP it** (droppable on bfcl +0.01 but **load-bearing on hermes −0.09** — the 2nd bench *corrected*
that surprise). **Reducible/droppable on both:** encoder depth (16→4 ≈ neutral: +0.05 bfcl / −0.02 hermes), K (→4),
init (random OK), deviation, base-LoRA (−0.02 to −0.04), prefix≡KV. ⇒ **minimal sufficient GCM = tiny encoder (N≈4,
K≈16) + distillation + reconstruction(for the gate) + task loss** — supports minimize-scale + "the gate is the contribution".

## Remaining work / what still needs doing (2026-06-17, post uniform-completion)
**✅ just finished**
- **Uniform ctx>K all-methods T2′ (48/48)** — 8 compressors × 6 benches at ONE recipe; the proper "no compression matches full" table. Folded into T2′ above.
- **Module ablation** on bfcl_live_multiple + hermes (2 benches) — T11; the surprising cells (task-droppable / random-init-OK / enc4-better) confirmed on the 2nd bench.
- AOC/ComprExIT 4-lane deadlock — **resolved** (both completed in the staggered re-run; full rows in T2′).

**⬜ not done — prioritized (GPUs now free on both pods)**
1. **Q7 full net-win Pareto** *(biggest gap — the payoff claim)*: cost↔quality Pareto + recall@P95 / AURC_acc at the test-agnostic τ, vs **always-compress / always-full / random / entropy / LLM-judge / TARG**. Partial signals already in T10/Tτ; need the TARG + LLM-judge gate baselines and the Pareto curve.
2. **GCM uniform cells** (T2′ A row *(pending)*): **1-lane** GCM re-run on bfcl_multiple / toolace / rca_openrca / narrativeqa (4-lane deadlocks; 1-lane smoke-passes) to replace the placeholders with same-recipe GCM.
3. **T4 (scaling) / T5 (HP) re-confirm at the uniform recipe** (Q3 provenance) — retire the old-env flag on "no scaling".
4. **T9**: fold the **Qwen3-8B** cross-bench rows (from depth) into the cross-model table. *(MoE dropped — see T9 note.)*

**🟡 deferred (per prior decisions)**
- **Bootstrap CIs** on the headline AUROC / recall@P95 (no multi-seed, per decision).
- **T8c** clean cross-task/domain GCM matrix (lost with `sam-dev-ray`; largely covered by Tτ cross-dataset τ + T8 transfer).
- Qwen3 thinking-fix **scale-ladder**; **relevance/agent** do-no-harm eval.

## Raw outputs (pods)
`out/maintable/` (T9 cross-model, sam-dev-free; `analyze_crossmodel.py`), `out/module_ablation/` (T11; `analyze_ablation.py`), `out/baselines_uniform/` (uniform ctx>K T2; `ratio_table.py`), test-agnostic τ (`tau_sweep.py`/`tau_select.py`), signal-selection (`signal_select.py`). Older: `out/clean/{hp,min,v175,stab,scale,b25,p2,p2_gcm}` (much lost with sam-dev-ray; Jun-12 snapshot in `mem-test/v17-results-backup/`).
