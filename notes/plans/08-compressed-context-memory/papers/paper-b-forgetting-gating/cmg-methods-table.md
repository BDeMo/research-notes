# All methods on CMG (cmg_rca) — module-ID table

**Task:** Nokia CMG root-cause **module identification**, 6-way multiple choice (chance ≈ 16.7%).
**Set:** `cmg_rca` **validation split, N=12** (the eval set; ⚠ tiny — 1 case = 8.3 pts, so ranks are noise-band).
**Base:** Qwen3-8B frozen (training-free methods); GCM-M uses its `mc_allthree` LoRA adapter (different model — separate reference).
**Budget:** keep 0.5 (window 1024, RAG 2048, LLMLingua-2 0.5, ToMe 0.5, IMP span-32/0.5, kvzip/knorm ratio 0.5). Logs `/mnt/persist/grid_cmg/cmg_*`.

| method | trained? | train acc (N=26) | test acc (N=12) |
|---|:--:|--:|--:|
| no_ctx (blind) | — (ref) | 11.5 | 8.3 |
| **full** (raw telemetry, trunc 8k) | — (ref) | 7.7 | 25.0 |
| window-1024 | ✗ | 15.4 | 16.7 |
| **RAG-2048** (BM25) | ✗ | 23.1 | **41.7** |
| LLMLingua-2 | ✗ | 3.8 | 16.7 |
| ToMe | ✗ | 3.8 | 25.0 |
| **IMP-v2.1.0** (span-32, ours) | ✗ | 15.4 | 25.0 |
| kvzip | ✗ | 19.2 | 8.3 |
| knorm | ✗ | 11.5 | 16.7 |
| GCM soft-memory **M** (v1.8) | **✓** | 19.2 | 33.3 |

**`trained?`** = does the method learn weights. **Only GCM v1.8 is trained (✓)** — for it, `train` = cases seen in training (not a fair test), `test` = held-out. Training-free methods (✗) never train, so `train`/`test` are just two disjoint case subsets (no contamination). Training-free rows on frozen Qwen3-8B; GCM-M on its LoRA adapter. Splits: `GCM_EVAL_SPLIT=train|validation`, logs `grid_cmg/cmgtr_*` (train) + `cmg_*` (test). N=12/26 → noisy, illustrative.

**Reading (honest, N=12 is noisy):**
- **RAG (41.7) is the single standout**, above `full` (25) and every other method — consistent with the main finding (F30) that BM25 retrieval dominates on real QA/RCA.
- **IMP, ToMe, GCM-M, full all tie at 25.0**; window/ll2/knorm at 16.7 (≈ chance); kvzip 8.3.
- **The whole task is low-ceiling** (`full` only 25% — the frozen base is weak at CMG RCA), which is exactly why **CMG-demo accuracy is not used as a paper headline** (matrix-facts excluded item).
- `full` fluctuates 25–33 across runs at N=12 (greedy decode + tiny N) — treat ±1 case as noise.

**Note on setups:** training-free methods share `no_ctx=8.3 / full=25.0` (frozen base). GCM-M runs on the adapter model (its own `no_ctx=16.7 / full=25.0`), so its 25.0 is not strictly comparable to the frozen-base column.

*Provenance: `grid_cmg/cmg_{window,rag,ll2,tome,imp,kvzip,knorm}.log` (val-12); GCM-M aggregated from the live demo `/api/analyze` over the 12 val cases.*

---

## GCM (v1.8) vs IMP (v2.1) — train/val split + honest caveats (2026-07-09)
Demo loads all **38** cmg cases (26 train / 12 val). Per-case ✓/✗ for both versions is shown in the demo (GCM card = adapter; IMP card = frozen base, read-LoRA off).

| split | GCM-M v1.8 (adapter) | IMP v2.1 (frozen) |
|---|--:|--:|
| all-38 | 23.7% (9/38) | 7.9% (3/38) |
| **train (26, GCM trained on these)** | 19.2% | 7.7% |
| **val (12, held-out)** | 33.3% | 8.3% |

**Two honest caveats — do not over-read CMG:**
1. **GCM's all-38 lead is partly train memorization** — GCM-M was trained on the 26 train cases, so comparing GCM vs a training-free method on train is unfair. On held-out val the gap is what matters (and N=12 is tiny).
2. **The demo's IMP number ≠ the harness IMP number** (demo val IMP 8.3% vs harness val IMP 25%). Reason: the demo runs IMP *through the GCM-compressor model wrapper* (`normalize=True`, recur config) even with read-LoRA off, which changes `mc_loglik` scoring vs the clean `run_baseline` frozen base. **The authoritative CMG numbers are the harness table above** (RAG 41.7 best; full/IMP/GCM-M ≈ 25); the demo is for **per-case illustration**, not authoritative accuracy.
3. CMG is **low-ceiling + N=12 noisy** → illustrative only; **not a paper headline** (matrix-facts excluded item).
