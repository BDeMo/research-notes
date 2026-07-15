# Paper C — GDN forgetting: memory-state properties, variant taxonomy, ablations & module extensions (2026-07-14)

Exploration aimed at **the forgetting problem on Gated DeltaNet (GDN)** — the linear module in our base (Qwen3.5/3.6). Grounded in a literature pass (Google Scholar / web, 2026-07): GDN-2, KDA, the factorial-decay analysis, HALO, cross-layer-routing survey ([`references-linear-attention.md`](references-linear-attention.md) §9b).

## 1. What GDN's memory state IS (properties)
- **State = a fixed-size matrix `S ∈ R^{d_k×d_v}`** (per head), not a growing KV cache. It is an **associative memory** (fast weights): write key→value outer products, read with the query `o_t = Sₜᵀ φ(q_t)`.
- **Update (delta rule + gate):** `Sₜ = diag(αₜ)·Sₜ₋₁ + kₜ ⊗ (vₜ − Sₜ₋₁ᵀ k̂ₜ)·βₜ` — decay `αₜ` (global forgetting) + delta correction (subtract current prediction before writing → reduces interference). Keys ℓ2-normalized for stability.
- **Forgetting is lossy & entangled:** with a *scalar* `αₜ`/`βₜ` (GDN), one knob controls **both** how much old content to **erase** (key side) and how much new to **write** (value side). Fixed capacity ⇒ new writes **overwrite/interfere** with old associations → the **forgetting / recall bottleneck** on long context.
- **Our empirical hook (F28):** on Qwen3.5-9B (GDN 3:1 hybrid), IMP's RULER-needle gap is **wider than on quadratic** (79 vs 96.8) — consistent with state-collapse: early-context needles get overwritten by later writes. The **hybrid full-attn layers (1 in 4)** are what rescue global recall.

## 2. Variant taxonomy — the forgetting design space (from the factorial analysis)
Three binary axes × the delta rule:
| model | decay granularity | conditioning | erase/write | delta |
|---|---|---|---|:--:|
| Mamba-2 | scalar | data-dep | write only (scalar) | ✗ (outer-product) |
| Lightning (HALO) | scalar | **data-indep (fixed)** | write only | ✗ |
| GLA | (low-rank) channel | data-dep | write only | ✗ |
| **DeltaNet** | none/scalar | — | tied scalar | ✓ |
| **GDN (ours' base)** | **scalar** | data-dep | **tied scalar `βₜ`** | ✓ |
| **KDA** | **channel-wise** | data-dep | tied scalar `βₜ` | ✓ |
| **GDN-2** | channel-wise | data-dep | **decoupled channel-wise `bₜ`(erase)/`wₜ`(write)** | ✓ |

**Field findings we must respect:** (a) **delta rule is the dominant factor** for recall; (b) **channel-wise helps only *with* delta** (else interference); (c) channel-wise aids long-range persistence but can **hurt precise recall**; (d) **data-independent fixed decay generalizes to length better**; (e) **GDN-2's erase gate carries most of its gain** → *selective erasure of stale key-side associations* is the key forgetting lever; (f) channel-wise data-dependent gating can be **unstable at scale** (KDA distillation).

## 3. Our angle (frozen-base / light-training, consistent with Paper A/B philosophy)
We do **not** pretrain a new linear LM. Two tracks:

### Track I — DIAGNOSE GDN forgetting (no training) — the scoping spine
Characterize *when & why* the GDN state forgets, on our frozen Qwen3.5 base (and in-band Qwen3.6-27B later):
- **P1 Forgetting curve:** synthetic needle at controlled relative depth (0→100%) in a long context → recall vs depth. Hypothesis: monotone decay with depth on GDN layers, flat on the hybrid full-attn layers.
- **P2 Interference / capacity:** vary #keys (1→k needles) at fixed length → recall vs load (the recall–throughput tradeoff, Based/Zoology) — does GDN collapse past a key count?
- **P3 State probe (instrumented):** hook the GDN layers; track `‖Sₜ‖`, effective rank, and cosine(read, written-value) for an early needle as later tokens arrive → *measure overwrite directly*.
- **P4 Hybrid-ratio attribution:** ablate which layers carry recall by zeroing/passing context through only linear vs only full-attn layers (Qwen3.5 3:1) → how much recall is the 1-in-4 full-attn doing?

### Track II — FIX forgetting (module extensions, minimal training) — the payoff
Ordered by Occam (cheapest first):
- **X1 (training-free, input-side):** feed the fixed state a **shorter, denser context** via IMP-lite/RAG so it never overflows — "anti-forgetting by not asking the state to hold junk." *Already runs on linear (F10).* Test: does IMP-lite close the linear-vs-quadratic RULER gap (F28)?
- **X2 (light retrofit, erase-gate):** add a **channel-wise erase gate** (GDN-2 style `bₜ`) as a small LoRA-like adapter on the *frozen* GDN base, trained briefly — since the erase gate is the highest-leverage part (GDN-2 ablation), can a cheap retrofit recover GDN-2-like recall without full retraining?
- **X3 (side-memory):** a small external associative store (delta-rule / TTT-style) that the query can read, augmenting the fixed state's capacity — bridges to Titans/TTT.
- **X4 (data-independent decay ablation):** replace GDN's data-dependent scalar decay with a **fixed multi-timescale scalar decay** (Lightning/HALO) at inference-time reparam — does length generalization improve (per HALO) without retraining?

## 4. Ablations to run (design space, small models for scoping then in-band)
- **A1 forgetting-curve** (P1) across GDN vs quadratic (Qwen3.5-9B vs Qwen3-8B scoping; then Qwen3.6-27B vs Qwen3-14B in-band).
- **A2 load sweep** (P2): #needles ∈ {1,2,4,8,16} × length {4k,16k,64k}.
- **A3 hybrid-ratio** (P4): recall vs which-layers-see-context.
- **A4 input-side fix** (X1): IMP-lite/RAG on GDN — RULER gap closed?
- **A5 (stretch) erase-gate retrofit** (X2): brief LoRA erase-gate on frozen GDN.

## 5. Candidate core insight to test
> On GDN, long-context failure = **overwrite of stale key-side associations in a fixed state** (scalar erase can't protect them). Two remedies converge: **erase selectively (GDN-2's channel-wise `bₜ`)** or **never write the junk (input-side IMP-lite)**. If X1 (input-side) recovers most of what X2 (erase-gate) buys, the **training-free input-side fix is the Occam winner** and the paper's method — with the diagnosis (Track I) as the backbone.

## 6. Immediate next steps
1. Implement the **self-contained synthetic needle probe** (no external dataset → dodges the current HF-Hub block): generate filler + magic-number needle at controlled depth/count; measure recall on Qwen3.5-9B (GDN) vs Qwen3-8B (quad). → A1/A2.
2. Read in full: GDN-2 (2605.22791), factorial-decay (OpenReview), cross-layer-routing (2607.07953).
3. Secure Qwen3.6-27B (in-band hybrid) for the headline runs.
