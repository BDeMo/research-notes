# Results v1.7 (Qwen3-8B) — 2026-06-11

**What this is.** Every v1.7 result with the concrete data behind it. Each number links to its source: the
per-item JSONL records (scores + per-item gate signals) under [`evidence/`](evidence/), pulled from
`sam-dev-ray:/mnt/persist/v17/out` on 2026-06-11. Click any linked number to see the actual per-item data.

**Setup.** Frozen Qwen3-8B base; per-item decision metric (track B): for each input we either *compress* (use
the method's memory) or *fall back to full context*. `comp` = compressed accuracy, `full` = full-context
ceiling (single canonical baseline), `no_ctx` = bare base (do-no-harm floor), `best` = per-input
best(compress, full) = the perfect-gate ceiling. **Only GCM has an intrinsic fallback** (its per-item signal);
Cartridge/Gist are vanilla (no recourse). N=96 eval items/bench unless noted; **single seed → run-to-run
variance ≈ ±0.04**, so trust large gaps, not small ones.

Aggregates: [`evidence/phase2_summary.csv`](evidence/phase2_summary.csv) ·
[`evidence/sweep_summary.csv`](evidence/sweep_summary.csv). Worker logs:
[`evidence/phase2_w0.log`](evidence/phase2_w0.log) … `w3`, [`evidence/sweep_w0.log`](evidence/sweep_w0.log) …

---

## T1 — Main comparison, in-task, 6 domains (phase-2 grid; fixed config: GCM N24/K64, baselines s1500/lr1e-4/K64)
Each compressor number links to its job's per-item records (which also hold `no_ctx`/`full`/signals).

| domain | benchmark | no_ctx | Cartridge | Gist | **GCM** | full | best$_{GCM}$ |
|---|---|---|---|---|---|---|---|
| wiki | TriviaQA | 0.177 | [0.205](evidence/phase2/main_cart_trivia_qa/records_trivia_qa.jsonl) | [**0.370**](evidence/phase2/main_gist_trivia_qa/records_trivia_qa.jsonl) | [0.293](evidence/phase2/main_gcm_trivia_qa/records_trivia_qa.jsonl) | 0.218 | 0.372 |
| wiki | HotpotQA | 0.118 | [0.116](evidence/phase2/main_cart_hotpot_qa/records_hotpot_qa.jsonl) | [0.147](evidence/phase2/main_gist_hotpot_qa/records_hotpot_qa.jsonl) | [0.134](evidence/phase2/main_gcm_hotpot_qa/records_hotpot_qa.jsonl) | **0.201** | 0.249 |
| literary | QuALITY | 0.260 | [0.198](evidence/phase2/main_cart_quality/records_quality.jsonl) | [0.240](evidence/phase2/main_gist_quality/records_quality.jsonl) | [0.208](evidence/phase2/main_gcm_quality/records_quality.jsonl) | 0.146 | 0.229 |
| literary | NarrativeQA | 0.126 | [0.134](evidence/phase2/main_cart_narrativeqa/records_narrativeqa.jsonl) | [0.126](evidence/phase2/main_gist_narrativeqa/records_narrativeqa.jsonl) | [0.117](evidence/phase2/main_gcm_narrativeqa/records_narrativeqa.jsonl) | 0.116 | 0.149 |
| web | MS-MARCO | 0.205 | [0.202](evidence/phase2/main_cart_ms_marco/records_ms_marco.jsonl) | [0.159](evidence/phase2/main_gist_ms_marco/records_ms_marco.jsonl) | [0.156](evidence/phase2/main_gcm_ms_marco/records_ms_marco.jsonl) | **0.271** | 0.300 |
| synthetic | MuSR | 0.500 | [0.469](evidence/phase2/main_cart_musr_mm/records_musr_mm.jsonl) | [0.479](evidence/phase2/main_gist_musr_mm/records_musr_mm.jsonl) | [0.490](evidence/phase2/main_gcm_musr_mm/records_musr_mm.jsonl) | **0.531** | 0.604 |
| tool | BFCL (simple) | 0.010 | [0.219](evidence/phase2/tool_cart_bfcl_simple/records_bfcl_simple.jsonl) | [0.271](evidence/phase2/tool_gist_bfcl_simple/records_bfcl_simple.jsonl) | [0.208](evidence/phase2/tool_gcm_bfcl_simple/records_bfcl_simple.jsonl) | **0.990** | 1.000 |
| tool | API-Bank | 0.000 | [0.010](evidence/phase2/tool_cart_apibank/records_apibank.jsonl) | [0.000](evidence/phase2/tool_gist_apibank/records_apibank.jsonl) | [0.000](evidence/phase2/tool_gcm_apibank/records_apibank.jsonl) | **0.260** | 0.260 |
| tool | ToolACE | 0.010 | [0.042](evidence/phase2/tool_cart_toolace/records_toolace.jsonl) | [0.062](evidence/phase2/tool_gist_toolace/records_toolace.jsonl) | [0.062](evidence/phase2/tool_gcm_toolace/records_toolace.jsonl) | **0.979** | 0.979 |
| ops | OpenRCA | 0.135 | [0.135](evidence/phase2/tool_cart_rca_openrca/records_rca_openrca.jsonl) | [0.135](evidence/phase2/tool_gist_rca_openrca/records_rca_openrca.jsonl) | [0.198](evidence/phase2/tool_gcm_rca_openrca/records_rca_openrca.jsonl) | **0.438** | 0.490 |

**Read:** compression beats `full` only on TriviaQA (parametric QA); `full` is the ceiling everywhere else, by a
lot on structured/agentic (BFCL/ToolACE `full`≈0.98) and ops. Gist is the strongest *raw* compressor. → not a
"better compression" result.

---

## T2 — Per-dataset hyperparameter tuning (phase-1 sweep, QA; best config per method)
The QA numbers above use a fixed config; tuning per dataset (100-run sweep) gives the stronger baselines (this
is what the LaTeX main table uses for QA). Number links to the best-config job records.

| benchmark | Cartridge (best) | Gist (best) | GCM (best) | full |
|---|---|---|---|---|
| TriviaQA | [0.205](evidence/sweep/cartridge_trivia_qa_s1500_lr0.0001_K64/records_trivia_qa.jsonl) `s1500/lr1e-4/K64` | [**0.370**](evidence/sweep/gist_trivia_qa_s1500_lr0.0001_K64/records_trivia_qa.jsonl) `s1500/lr1e-4/K64` | [0.293](evidence/sweep/gcm_trivia_qa_N24_K64/records_trivia_qa.jsonl) `N24/K64` | 0.218 |
| HotpotQA | [0.107](evidence/sweep/cartridge_hotpot_qa_s1500_lr0.0003_K128/records_hotpot_qa.jsonl) `s1500/lr3e-4/K128` | [**0.260**](evidence/sweep/gist_hotpot_qa_s1500_lr0.0003_K128/records_hotpot_qa.jsonl) `s1500/lr3e-4/K128` | [0.165](evidence/sweep/gcm_hotpot_qa_N24_K64/records_hotpot_qa.jsonl) `N24/K64` | 0.201 |
| SQuAD-v2 | [0.136](evidence/sweep/cartridge_squad_v2_s600_lr0.0003_K64/records_squad_v2.jsonl) `s600/lr3e-4/K64` | [0.157](evidence/sweep/gist_squad_v2_s600_lr0.0001_K64/records_squad_v2.jsonl) `s600/lr1e-4/K64` | [0.136](evidence/sweep/gcm_squad_v2_N24_K64/records_squad_v2.jsonl) `N24/K64` | **0.367** |
| QuALITY | [0.208](evidence/sweep/cartridge_quality_s600_lr0.0001_K128/records_quality.jsonl) `s600/lr1e-4/K128` | [**0.312**](evidence/sweep/gist_quality_s600_lr0.0003_K128/records_quality.jsonl) `s600/lr3e-4/K128` | [0.260](evidence/sweep/gcm_quality_N12_K128/records_quality.jsonl) `N12/K128` | 0.146 |

---

## T3 — The intrinsic gate (GCM): per-item compress-vs-fallback signal
Oriented AUROC (the gate learns the sign) + best-F1 for the do-no-harm decision. Signal families: *verifiability*
(recon coverage, $\Delta$code/$\Delta$logit) vs *base uncertainty* (entropy/margin). Vanilla baselines expose
none of these. Each row links to the GCM job whose records carry the per-item `signals`.

| benchmark | best signal | family | AUROC | F1 | source |
|---|---|---|---|---|---|
| TriviaQA | $\Delta$code | verifiability | 0.67 | 0.76 | [records](evidence/phase2/main_gcm_trivia_qa/records_trivia_qa.jsonl) |
| HotpotQA | mem-norm | geometry | 0.65 | 0.75 | [records](evidence/phase2/main_gcm_hotpot_qa/records_hotpot_qa.jsonl) |
| QuALITY | $\Delta$code | verifiability | 0.73 | 0.77 | [records](evidence/phase2/main_gcm_quality/records_quality.jsonl) |
| NarrativeQA | $\Delta$logit | verifiability | 0.71 | 0.73 | [records](evidence/phase2/main_gcm_narrativeqa/records_narrativeqa.jsonl) |
| MS-MARCO | entropy | uncertainty | 0.66 | 0.73 | [records](evidence/phase2/main_gcm_ms_marco/records_ms_marco.jsonl) |
| MuSR | $\Delta$code | verifiability | 0.69 | 0.77 | [records](evidence/phase2/main_gcm_musr_mm/records_musr_mm.jsonl) |
| BFCL | recon | verifiability | 0.73 | 0.68 | [records](evidence/phase2/tool_gcm_bfcl_simple/records_bfcl_simple.jsonl) |
| API-Bank | $\Delta$code | verifiability | 0.65 | 0.76 | [records](evidence/phase2/tool_gcm_apibank/records_apibank.jsonl) |
| ToolACE | margin | uncertainty | 0.69 | 0.71 | [records](evidence/phase2/tool_gcm_toolace/records_toolace.jsonl) |
| OpenRCA | entropy | uncertainty | **0.83** | 0.74 | [records](evidence/phase2/tool_gcm_rca_openrca/records_rca_openrca.jsonl) |

**Mean AUROC 0.71**, single forward pass. Verifiability signals (Δcode/recon) win on structured/reconstructable
tasks; uncertainty on others — supports combining them (multi-signal gate, pending).

---

## T4 — Transfer / relation split (GCM, trivia-trained example anchors)
in-task / cross-task-in-domain / cross-task-cross-domain, comp vs full. Source = the GCM job's all-bench records.

| anchor | relation | GCM comp | full | no_ctx | best | source |
|---|---|---|---|---|---|---|
| TriviaQA | in_task | 0.293 | 0.218 | 0.177 | 0.372 | [records](evidence/phase2/main_gcm_trivia_qa/records_trivia_qa.jsonl) |
| TriviaQA | cross_task_in_domain | 0.142 | **0.284** | 0.126 | 0.327 | [records](evidence/phase2/main_gcm_trivia_qa/records_hotpot_qa.jsonl) |
| TriviaQA | cross_task_cross_domain | 0.254 | 0.266 | 0.273 | 0.353 | [records](evidence/phase2/main_gcm_trivia_qa/records_quality.jsonl) |

**Read:** an in-task win collapses to ≈no_ctx out-of-task while full stays high → the silent do-harm the gate
must catch (universal across all anchors; see [`phase2_summary.csv`](evidence/phase2_summary.csv) for every
anchor × relation).

---

## T5 — Capacity curve (GCM, TriviaQA in-task, K=64, bf16)
Compressed accuracy rises monotonically with encoder depth N (optimum ≈24; N=32 degrades — over-parameterized
at N=96). N=4 ≈ 0.16 (early v3 run, superseded).

| N | comp | full | source |
|---|---|---|---|
| 8 | [0.197](evidence/rel_N8/records_trivia_qa.jsonl) | 0.218 | rel_N8 |
| 12 | [0.208](evidence/rel_N12/records_trivia_qa.jsonl) | 0.218 | rel_N12 |
| 16 | [0.231](evidence/rel_N16bf16/records_trivia_qa.jsonl) | 0.218 | rel_N16bf16 |
| 24 | [**0.247**](evidence/rel_N24/records_trivia_qa.jsonl) | 0.218 | rel_N24 |

---

## T6 — Ablations (GCM, TriviaQA in-task unless noted; ±0.04 single-seed noise)
| ablation | variant | comp | baseline | source |
|---|---|---|---|---|
| **encoder init** | random | [0.195](evidence/phase2/abl_initrandom_trivia_qa/records_trivia_qa.jsonl) | copy-base 0.293 | strong: base init matters |
| **query conditioning** | agnostic M0 | [0.228](evidence/phase2/abl_agnostic_trivia_qa/records_trivia_qa.jsonl) | conditioned M_q 0.293 | conditioning helps |
| query conditioning (MC) | agnostic M0 | [0.208](evidence/phase2/abl_agnostic_quality/records_quality.jsonl) | conditioned 0.208 | no effect on QuALITY |
| **autoencoder** | no L_uncond+min-dev | [0.195](evidence/abl_nae/records_trivia_qa.jsonl) | with-AE 0.231 | AE adds ≈+0.036 |
| min-dev only off | rec=1,dev=0 | [0.177](evidence/sweep/gcm_trivia_qa_N16_K64_lr1_ld0/records_trivia_qa.jsonl) | with min-dev | min-dev matters |
| decoder depth | dec=1 / 2 / 4 | [0.218](evidence/sweep/gcm_trivia_qa_N16_K64_dec1/records_trivia_qa.jsonl) / [0.186](evidence/sweep/gcm_trivia_qa_N16_K64/records_trivia_qa.jsonl) / [0.268](evidence/sweep/gcm_trivia_qa_N16_K64_dec4/records_trivia_qa.jsonl) | — | noisy; deeper helps |
| memory width K | 32 | [0.215](evidence/phase2/abl_K32_trivia_qa/records_trivia_qa.jsonl) | K64 0.293 | wider helps (K256 run errored) |
| compression ratio (n_chunks) | 2 / 4 / 8 / 12 | [0.190](evidence/phase2/abl_ratio_nc2_trivia_qa/records_trivia_qa.jsonl) / [0.232](evidence/phase2/abl_ratio_nc4_trivia_qa/records_trivia_qa.jsonl) / [0.214](evidence/phase2/abl_ratio_nc8_trivia_qa/records_trivia_qa.jsonl) / [0.183](evidence/phase2/abl_ratio_nc12_trivia_qa/records_trivia_qa.jsonl) | — | capacity wall: more ctx → degrades |

---

## Sources / provenance
- Per-item records (scores + signals): [`evidence/phase2/`](evidence/phase2/) (main + tool + ablations),
  [`evidence/sweep/`](evidence/sweep/) (100-run tuning), [`evidence/rel_N*`](evidence/) (capacity).
- Aggregated CSVs: [`evidence/phase2_summary.csv`](evidence/phase2_summary.csv),
  [`evidence/sweep_summary.csv`](evidence/sweep_summary.csv).
- Runner logs: `evidence/phase2_w{0..3}.log`, `evidence/sweep_w{0..3}.log`.
- Reporter: `mem-test/svc/sweep_report.py` + `svc/decision.py`. Harness: `mem_embedding.gcm.harness`.
- Caveats: single seed (±0.04); QA baselines tuned (T2), other domains a single config (T1); BFCL/RCA etc. ran
  on sam-dev (different pod). API-Bank/RCA `full`/`no_ctx` are low — MC/extraction scoring, see records.
