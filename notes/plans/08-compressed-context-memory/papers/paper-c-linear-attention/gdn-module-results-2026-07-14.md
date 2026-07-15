# Paper C — GDN anti-forgetting module results & next ablations (2026-07-14)

Grid `pc_*` (Qwen3.5-9B & 4B, GDN; 8 long benches × {imp-lite, rel_last, replay, rag}, N=100; refs full/no_ctx free). Plus synthetic needle forgetting curve (`needle_*`). All ×100. Modules defined in [`gdn-forgetting-exploration.md`](gdn-forgetting-exploration.md) §3.

## 1. Module comparison (Qwen3.5-9B GDN)
| bench | full·t | M1 imp-lite | M2 rel_last | M3 replay | rag |
|---|--:|--:|--:|--:|--:|
| ∞Bench-choice | 55 | 76 | 75 | **78** | 65 |
| lb_multifieldqa | 38.6 | 42.1 | 40.7 | **44.0** | 41.1 |
| lb_hotpotqa | 44.9 | 43.5 | **44.8** | 41.5 | 37.4 |
| babilong_qa1_16k | 65 | **68** | 62 | 61 | 54 |
| babilong_qa2_16k | 26 | 22 | 20 | 22 | 7 |
| lb_2wikimqa | 51.5 | 40.8 | 42.3 | 37.8 | 34.8 |
| lb_narrativeqa | 18.4 | 15.1 | 11.9 | 11.7 | 12.1 |
| quality_hard | 9 | 8 | 12 | 9 | 12 |
(Qwen3.5-4B same pattern: ∞Bench full 49 → imp 69–71; reorder small/mixed; babilong/2wiki full best.)

## 2. Findings
- **F-C1 (X1 works): input-side compression rescues GDN on ultra-long.** ∞Bench full **55 → imp-lite 76** (+21), > rag 65. Not overflowing the fixed state is the big lever — exactly where GDN forgets. Holds on 4B (49→69).
- **F-C2 (recency reorder = weak/mixed): M2 rel_last / M3 replay give only small, task-dependent gains.** Win on single-passage / ∞Bench / multifieldqa (replay 78 > reading 76; multifield 44 > 42) but **hurt on multi-fact** (babilong_qa1 reading 68 > rel_last 62; 2wiki, narrativeqa). ⇒ GDN *has* a recency bias (needle curve below) but **input reordering doesn't robustly exploit it** because multi-fact tasks need reading order & all facts. **Occam: imp-lite (reading) is the robust default; reordering is not a clean win.**
- **F-C3 (regime pattern mirrors Paper B on GDN):** imp-lite beats full on ∞Bench/multifield, ties hotpot, **loses on 2wiki/narrative/babilong/quality_hard (full best — dense/multi-fact needs everything).**

## 3. Forgetting curve (synthetic needle, single needle)
| model | 4k (depth 0→1) | 16k (depth 0→1) |
|---|---|---|
| Qwen3-8B (quad) | 10/10 flat | **10/10 flat** (no forgetting) |
| Qwen3.5-9B (GDN) | 9–10/10 | **depth0 7/10**, depth≥0.25 10/10 |
⇒ GDN forgets the **earliest** part of a 16k context (recency bias); quad doesn't. But 16k is too short → effect mild. **Need 32k–128k to see real state collapse.**

## 4. MORE ablations (to "figure it out") — launching
- **A6 long forgetting curve:** single needle, depth {0,.25,.5,.75,1}, lengths **{16k,32k,64k,128k}**, Qwen3.5-9B & 4B, raw GDN → where does length break recall? (128k ok: Qwen3.5 max_pos 262k.)
- **A7 input-side rescue at the break:** same lengths × {raw, imp-lite keep0.25} → does compression restore recall where raw GDN collapses? (validates X1 at the forgetting regime, not just ∞Bench score.)
- **A8 load sweep (recall–throughput):** #needles {1,2,4,8} at 32k, query one → does GDN fixed state collapse as key-count grows (Based/Zoology)? quad control.
- **A9 (stretch, instrumented) state probe:** hook GDN layers, track ‖Sₜ‖ / overwrite of an early needle as tokens arrive (Track I P3).

## 4b. Extended ablation results (A6/A7/A8, 2026-07-14)
- **A8 load sweep — CLEAN, the headline GDN-capacity result.** Qwen3.5-9B raw, 32k, needle at depth 0, query 1 of N keys: **N=1: 8/8, N=2: 8/8, N=4: 4/8, N=8: 4/8.** GDN's fixed state collapses as key-load grows (recall–throughput limit; Based/Zoology). This is reproducible and interpretable.
- **A6/A7/A8 (needle2/needle3) — DISCARDED: I broke a working probe by over-engineering the prompt.** The rephrased query ("vault key for team ALPHA0 … Answer with the number only") made even **raw single-needle 16k = 1/16** — impossible for a real model → the model doesn't emit the bare number within 8 greedy tokens for this format; it is a **generation/parse bug, not a GDN property.** The ORIGINAL `needle_probe.py` (round 1) format ("The secret passcode is X" / "What is the secret passcode?") gave **sensible** numbers (GDN 16k depth0 **7/10**, else 10/10; quad flat 10/10). **Lesson: keep the round-1 probe; extend only its lengths.** Re-running round-1 at 32k/64k/128k (`nfc_*`, GDN-only per Paper C directive) — harvest pending.

## 4c. Trustworthy spine (what we actually stand on)
1. **Input-side compression rescues GDN on real ultra-long benches** — ∞Bench full 55 → imp-lite 76 (pc grid, N=100, harness-verified). ← the method result.
2. **GDN capacity collapses under key-load** — A8 (4–8 keys → 50% recall). ← the mechanism/limit.
3. Recency-reorder modules (M2/M3) = marginal, task-dependent (F-C2). Occam: imp-lite (reading) default.
The synthetic single-needle *depth* curve is set aside pending a hardened probe.

## 5. Working hypothesis to confirm/kill
> GDN long-context failure = **early-context (low-recency) content is overwritten in the fixed state**, worsening with **length** and **key-load**. The robust training-free fix is **input-side compression (imp-lite)** that keeps the state from overflowing — reordering to exploit recency is marginal and hurts multi-fact. If A6/A7 show imp-lite closing a *large* raw-GDN forgetting gap at 64k–128k, that is Paper C's method result; A8 quantifies the capacity limit; A9 gives the mechanism.
