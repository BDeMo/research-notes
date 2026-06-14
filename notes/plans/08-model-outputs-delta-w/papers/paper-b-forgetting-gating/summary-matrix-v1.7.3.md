# Paper B — Summary Matrix v1.7.3 (writing-agent handoff)

> **Single-source brief, current as of 2026-06-13.** Supersedes `summary-matrix.md` (v1.5/v1.6) and
> `summary-matrix-v1.7-2026-06-10.md` (leakage-era). Detailed results:
> [`results-v1.7.3/results-v1.7.3.md`](results-v1.7.3/results-v1.7.3.md) · thesis+plan:
> [`results-v1.7.3/robustness-plan.md`](results-v1.7.3/robustness-plan.md) · reviewer plan:
> [`results-v1.7.3/reviewer-response.md`](results-v1.7.3/reviewer-response.md) · setup:
> [`results-v1.7.3/experimental-setup.md`](results-v1.7.3/experimental-setup.md).
> **Base = Qwen3-8B.** **HONESTY FIRST.** All numbers are leak-free (seed-independent split), held-out
> (5-fold CV) gate, matched-budget baselines. Some rows are still landing — marked *(partial)*.

---
## 0. Abstract seed (draft)
Context compression for a frozen LLM is **unreliable**: a fixed-size soft-prompt memory is *capacity-bound*
(it helps only when the context is several times larger than the memory) and **never beats full context**;
on short contexts even **trivial truncation already matches full**. We therefore study a **compressor-agnostic
robustness layer** — a confidence / self-verification signal that, per input, **detects when compression is
unsafe and falls back to full context** (do-no-harm). The same gate, applied to our module *and* to a faithful
Cartridge baseline, recovers ≈ full accuracy while compressing where it is safe. We map *when* compression
helps (a capacity / compression-ratio frontier) and are explicit about the negatives: no method beats full,
trivial truncation is a strong baseline on short contexts, and the gate adds compute savings only in the
high-ratio regime.

---
## 1. Thesis + positioning
- **Thesis:** a **compressor-agnostic robustness layer** (detect bad compression → fall back = do-no-harm), **not** a better compressor.
- **Why (the honest pivot):** compression is capacity-bound and lossy; the deployable value is *knowing when not to trust it*.
- **Scope:** agentic tool-use + RCA (long-context); frozen base; the gate is the contribution, the compressor is interchangeable.
- **Novelty vs neighbors:** Cartridges (the *module*), TARG (a base-uncertainty *gate*), Gist/ICAE/AutoCompressor/LLMLingua (compressors). **White space = a compressor-agnostic do-no-harm layer + an honest "when does compression work" diagnosis (capacity frontier + trivial-truncation baseline).**

---
## 2. Contributions (status + the number each rests on)
| # | contribution | headline number | status |
|---|---|---|---|
| **C1** | **compressor-agnostic do-no-harm gate** | conf-gate on **Cartridge**: detection AUROC 0.6–0.8, held-out gАcc ≈ full | ✅ |
| **C2** | **compression is capacity-bound (diagnosis)** | corr(ctx-len, value-add) **= −0.4**; works at ratio ≤3×, fails ≥9× | ✅ |
| **C3** | among trained compressors, GCM ≥ Cartridge ≫ Gist (matched budget) | live_multiple GCM 0.72 > Cart 0.51 > Gist 0.00; ties/loses on short/synthetic | ✅ |
| **C4** | **trivial truncation is a strong baseline (necessity caveat)** | `trunc`(first-K) = full on simple/parallel/live_simple/glaive/hermes; GCM beats trunc **only** on live_multiple | ⚠️ sharp negative |
| **C5** | none of the compressors beats `full`; compute saving ≈ 0 except high-ratio | best compress 0.72 < full 0.91; realized saving ≈ 0 | ✅ honest |

---
## 3. Main results (Qwen3-8B, in-task, clean)
### 3a. In-task compress + gate (recommended config `combo` = LoRA32/K64/lr1e-4/3000)
| bench | no_ctx | full | GCM compress | gate gAcc (CV) | gate AUROC |
|---|---|---|---|---|---|
| bfcl_live_multiple | 0.01 | 0.91 | **0.72** | 0.89 | 0.75 |
| bfcl_live_simple | 0.01 | 0.91 | 0.53 | 0.92 | **0.88** |
| hermes | 0.33 | 0.94 | 0.50 | 0.92 | 0.53 |
| glaive (1-fn) | 0.33 | 0.99 | 0.77 | 0.96 | 0.59 |
| bfcl_simple | 0.02 | 0.99 | 0.17 | 0.99 | 0.52 |
| bfcl_multiple | 0.02 | 0.88 | 0.15 | 0.87 | 0.66 |
| toolace | 0.01 | 0.95 | 0.14 | 0.94 | 0.53 |
| rca_openrca | 0.12 | ~0.45 | 0.10 | ≈full | 0.47 |

### 3b. Matched-budget baselines (384/3000/K64) — compress
GCM ≥ **Cartridge** ≫ **Gist** on the live benches; ties/loses on short/synthetic; **none beats full.**
live_multiple: GCM 0.72 / Cart 0.51 / Gist 0.00. live_simple: 0.53 / 0.31 / 0.00. simple: 0.17 / 0.15 / 0.06.

### 3c. ★ Necessity vs trivial truncation (NO training)
| bench | full | GCM | **trunc (first-K)** | meanpool | randk |
|---|---|---|---|---|---|
| bfcl_live_simple | 0.91 | 0.53 | **0.91** | 0.87 | 0.03 |
| glaive | 0.99 | 0.77 | **0.99** | 0.99 | 0.39 |
| bfcl_simple | 0.99 | 0.17 | **0.99** | 0.99 | 0.02 |
| bfcl_parallel | 0.95 | 0.30 | **0.95** | 0.95 | 0.12 |
| hermes | 0.94 | 0.50 | **0.94** | 0.63 | 0.37 |
| **bfcl_live_multiple** | 0.91 | **0.72** | 0.46 | 0.03 | 0.00 |
→ On **short-context** benches (ctx ≈ K) **truncation = full and beats GCM** — there is nothing to compress. GCM beats trivial truncation **only on `bfcl_live_multiple`** (ctx ≈ 2.8×K). This is the necessity boundary: a learned compressor earns its keep only when ctx ≫ K.

### 3d. ★ The gate is compressor-agnostic
Same confidence gate on **Cartridge**: detection AUROC **0.6–0.8** (parallel 0.81, live_multiple 0.73, live_simple 0.70) + held-out gАcc ≈ full → the robustness layer is not tied to our encoder. (Weaker on Gist, whose compress ≈ 0 ⇒ gate always falls back.)

---
## 4. ⛔ Do-NOT-overclaim (hard constraints)
1. **No compressor beats `full`** on any tool/ops bench.
2. **Trivial truncation (first-K) matches full on short-context benches and beats GCM there.** GCM only adds over truncation on `bfcl_live_multiple` (ctx ≫ K). Always report `trunc` as a baseline.
3. **Compute saving ≈ 0** except in the high-ratio regime (short benches: K ≥ ctx ⇒ negative; long benches: compression fails ⇒ gate falls back).
4. **Gate AUROC ≈ chance on most benches**; do-no-harm there is via *conservative fallback* (gАcc = full), not detection. Real detection only on live benches (AUROC 0.75–0.88).
5. **LoRA adds only ~+0.02** avg (the v1.7 "LoRA is the unlock" claim is retracted — it was a leakage artifact).
6. **Single seed** so far (multi-seed CIs queued); config differences are within ±0.04.
7. **Name-only tool scoring** in §3a; **args-aware** (name+args) re-runs in progress (may lower the numbers).
8. **No cross-task/cross-domain transfer** claim yet (Phase-2 transfer matrix running).
9. **glaive is single-function** (trivial recall, not long-context compression).

---
## 5. Status / next (2026-06-13)
- ✅ leak-free redo (v1.7.3), held-out gate, matched baselines, model-generality sweep, capacity analysis, trivial baselines, compressor-agnostic gate (Cartridge).
- 🔄 running: args-aware verdict (R3), K-sweep capacity frontier, multi-seed CIs, Phase-2 transfer matrix.
- ⬜ next: LLMLingua gate (agnostic claim on a SOTA compressor), retrieval baseline, lock paper identity = **"diagnosis + do-no-harm robustness layer"**.
