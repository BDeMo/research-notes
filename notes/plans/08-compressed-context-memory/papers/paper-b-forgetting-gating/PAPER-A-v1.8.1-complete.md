# Paper A (v1.8.1) — *A verified do-no-harm gate for compressed context, unified by reconstruction*

> Complete working spec: outline · main method (now combining **KVzip**'s reconstruction-importance) · experiment logic ·
> insights · related work · difference claims. Supersedes the `paper-A-archive.md` freeze — **re-opened** to fold in kvzip.
> Facts referenced by ID from `matrix-facts.md`; experiments by ID from `matrix-experiments.md`.

## 0. One-line thesis
Any lossy long-context compressor (KV-eviction, prompt-compression, soft-memory) is unsafe on *some* inputs. We add a **query-agnostic, reconstruction-verified do-no-harm gate**: trust the cheap compressed path **iff the compressed state can reconstruct the evidence it needs**, else fall back to full — so quality is **never below the feasible baseline**, with an **out-of-sample coverage guarantee**. *The compressor is the carrier; the verified gate is the contribution; **reconstruction is the single mechanism that both scores importance and certifies safety**.*

## 1. The unifying idea (why reconstruction ties it together)
- **KVzip's insight (Kim 2025):** a KV pair's importance = its contribution to **reconstructing the original context** (max attention it receives during a teacher-forced "repeat the context" forward). Reconstruction reveals a *query-agnostic* sparse structure that generalizes across queries/tasks.
- **Our method already centers reconstruction:** the memory is trained to (i) answer and (ii) reconstruct; the **reconstruction score is the gate signal** (a memory that can reconstruct the evidence is one that retains it).
- **⇒ Combine:** use **reconstruction-importance to (a) decide *what* the compressed state must preserve and (b) *certify* that it did.** One quantity, three roles: train-regularizer · importance-scorer (à la KVzip) · safety-gate. This is the paper's conceptual unification.

## 2. Main method (v1.8.1 = elegant core + kvzip module)
**Frozen base, adapters only.** Four roles on one base: encode (K memory tokens + proj-MLP), read (read-LoRA), self-distill (full-ctx teacher), **verify (reconstruction)**.
- **Two training losses:** *answer* (task-CE + self-distillation KL) + *preserve* (self-reconstruction). (Ablations F12: nothing else is load-bearing.)
- **NEW — reconstruction-importance-weighted preserve (kvzip-combined):** weight the reconstruction loss by **KVzip-style importance** (max reconstruction-attention per context token) so the memory prioritizes preserving the *query-agnostically important* content — not uniform reconstruction. This is the concrete kvzip contribution to the method.
- **One gate signal — reconstruction fidelity `r(x)`:** at inference, measure how well the compressed state reconstructs the (importance-weighted) evidence. High `r` ⇒ trust compressed; low ⇒ fall back. **Query-agnostic** (computed from the compressed state + a repeat-prompt, no query needed) — aligning with KVzip's reusable-cache framing.
- **One guarantee — conformal threshold:** calibrate the `r`-threshold on a held-out split for a target coverage → distribution-free "gated ≥ feasible baseline out-of-sample" (fixes the tautology F13).
- **Compressor-agnostic:** the gate wraps *any* compressor — incl. **KVzip / FastKVzip / SnapKV / LLMLingua / our soft-memory** — turning each into a do-no-harm system. (Generality is a headline; see §4 C-A3.)

## 3. Experiment logic (what each experiment proves)
| exp | proves | status |
|---|---|---|
| **E1 — pruned ≈ kitchen-sink** (T1 A/B) | the compressor is a commodity (F12) → effort belongs on the gate | ✅ done (Δ0.01) |
| **E2 — reconstruction-importance-weighted preserve helps** | kvzip weighting raises compressed accuracy and/or gate quality vs uniform recon | ⏳ **X-A-kvzip (below)** |
| **E3 — conformal gate: gated ≥ feasible OUT-of-sample** | the do-no-harm contract is real, not tautological (F13) | ⏳ X-C3 |
| **E4 — gate generality across compressors** | one gate makes KVzip/FastKVzip/LLMLingua/soft-memory all "never worse than full" | ⏳ X-C6 |
| **E5 — signal ablation** | reconstruction-fidelity `r(x)` predicts unsafe inputs (AUROC) ≥ margin/conf/entropy | 🏃 partial (v1.8 signals) |
| **Main table** | per-bench: full · no_ctx · best-compressor · **gated (ours)** with coverage + Δ-vs-full ≥ 0 | ⏳ after E3/E4 |

**Main-table design:** rows = benches (RULER-sweep, QuALITY, NarrativeQA, squad, hotpot, trivia); cols = `no_ctx`, `full`, `KVzip@budget`, `soft-memory`, **`gated (ours)`** with **coverage%** and **risk** (fraction below full). The claim is not "highest accuracy" but **"never below full, at compressed cost on the covered fraction"** — a selective-prediction result.

## 4. Insights (the facts this paper rests on — see `matrix-facts.md`)
- F12 compressor-is-commodity → the gate is the contribution.
- F13 in-sample "gated≥full" is tautological → conformal is mandatory.
- F4/F14 no compressor is universally safe (kvzip cliffs at ~0.9, retrieval-only; merge kills needles) → a per-input gate is necessary.
- Reconstruction (KVzip) reveals query-agnostic importance that generalizes → the gate signal can be query-agnostic and reusable.

## 5. Related work (positioning)
- **KV eviction:** SnapKV, H2O, StreamingLLM (query-aware / recency) → degrade under multi-query reuse. **KVzip** (query-agnostic, reconstruction-importance) and **FastKVzip** (distilled sink-attention gate, <1 H100-hr, forward-only) are the SOTA and the closest to us.
- **Soft/learned memory:** Gisting, ICAE, AutoCompressor, 500xCompressor, Cartridges — compress context into tokens; lossy, no safety layer.
- **Selective prediction / calibration:** confidence thresholds (Geifman), **conformal prediction** (Angelopoulos & Bates) — our do-no-harm guarantee's home.
- **Cascades / speculative decoding:** cheap-then-verify — our cost-ascending gate is a within-model cascade with a verification (reconstruction) instead of a draft.

## 6. Difference / novelty claims (sharp, honest)
1. **vs KVzip / FastKVzip (closest):** they use reconstruction-importance to **decide what to compress/evict** (a *compression policy*). We use reconstruction-fidelity to **decide whether to trust the compressed path or fall back to full** (a *safety contract*). **We can run *on top of* FastKVzip** — our gate turns their (or any) compressor into a *never-worse-than-full* system with a **coverage guarantee they don't provide.**
2. **vs soft-memory papers:** they sell ratio/efficiency; we sell **provable do-no-harm** and are **compressor-agnostic**.
3. **The unification:** reconstruction as a *single* mechanism for importance (what to keep) **and** certification (when to trust) — neither the KV-eviction line (importance only) nor the calibration line (certification only) does both.
4. **Query-agnostic verified reuse:** the gate certifies a compressed state is safe **without the query** (reconstruction-based) → a reusable, verified compressed cache with a risk bound — a combination not in prior work.

## 7. The kvzip-combination validation experiment (X-A-kvzip) — design
**Goal:** show (E2) reconstruction-importance-weighting helps, and (E4-lite) the gate makes KVzip do-no-harm.
1. **Per-item dump:** on {RULER-16k, QuALITY(loglik), squad, hotpot, narrativeqa}, run **KVzip@{0.75,0.9}**, **full**, **no_ctx**, dumping per-item correctness + a **reconstruction-fidelity signal** `r(x)` (teacher-forced "repeat context" logprob under the compressed cache).
2. **E4-lite:** define gate = trust KVzip iff `r(x) ≥ τ` (τ conformal on a held split); report **gated-acc, coverage, risk (fraction < full)**, and **AUROC(r vs KVzip-fails-but-full-right)**. Success = gated ≥ full with useful coverage + AUROC ≫ 0.5.
3. **E2:** train the soft-memory with **uniform** vs **kvzip-importance-weighted** reconstruction; compare compressed accuracy + `r`-gate AUROC. Success = weighting improves one/both.
4. **If effective → main method + main table:** fold the importance-weighted preserve into the method and add a "gate over KVzip" row to the main table (do-no-harm over the SOTA compressor).
*(Needs a per-item eval harness that emits {kvzip_ok, full_ok, r(x)} — small new script `run_gate_probe.py`; eval-only, ~1–2 GPU-h.)*

*Provenance: KVzip ([arXiv:2505.23416](https://arxiv.org/abs/2505.23416), NeurIPS'25), FastKVzip ([arXiv:2601.17668](https://arxiv.org/abs/2601.17668)); `matrix-facts.md`, `method-elegance-plan-v1.8.x.md`, `paper-A-archive.md` (now re-opened).*
