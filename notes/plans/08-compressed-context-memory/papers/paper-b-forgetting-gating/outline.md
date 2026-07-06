# Paper B — Outline + Claim→Evidence Map

> Companion to [`README.md`](README.md). This is the working skeleton: section plan, a
> claim→evidence table (what's in hand vs gap), and the prioritized experiment queue.

## ⚠️ Open decisions (need a call before deep work)
1. **Scope** — full **read ⊕ write** (08 gate + 09 site-protection, the litreview frame) *or*
   **read-only first** (08 gate as a standalone forgetting/negative-transfer paper, 09 as a follow-up)?
   Read-side is mostly evidenced; write-side (C4) is the big remaining experiment.
2. **Target venue / deadline** — sets required depth & mandatory baselines (workshop vs ICLR/ACL full).
3. **Headline framing** — lead with *forgetting* (problem-first, CF audience) or *gating/do-no-harm*
   (method-first, RAG/adaptive-retrieval audience)?

## Section skeleton (draft)
1. **Intro** — adaptation forgets (read: negative transfer; write: CF); thesis = do-no-harm via one intrinsic cross-model signal.
2. **Related work** — CF in LLMs (B1) · CL mitigation taxonomy (B2) · which-params-to-protect (B3) · intrinsic sites (B4) · adaptive-retrieval gating (B5). [litreview](../../summary/2026-06-05/two-paper-litreview-2026-06-05.md).
3. **The forgetting problem (two axes)** — measure it: SFT degrades base (§7c); ungated augmentation hurts OOD (§8).
4. **The intrinsic signal** — `delta_last` / memory-write dynamics; cross-model consistency (§2/§2b); why it travels (mechanism), why logit-lens doesn't.
5. **Read-time gate** — routing gate, do-no-harm-by-construction; recovers most harm, LOMO transfer; honest temper (oracle headroom). Ablations: soft-gate & multi-depth fail (H4, §7c).
6. **Write-time protection (Plan 09)** — same importance map → freeze sites during SFT; vs EWC/MoFO/MIGU/ESFT. **← main open experiment.**
7. **Unification + analysis** — one signal, both knobs; cross-model.
8. **Limitations** — offline gate, magnitude caveats, MC tasks don't benefit, capacity wall (→ Paper A).

## Claim → evidence map
| # | claim | status | evidence / asset | matrix |
|---|---|---|---|---|
| C1a | SFT-the-base forgets (parametric + full-ctx QA) | ✅ have | `raw/sft-2026-06-05/mix_sft_qwen3_8b.csv` | §7c |
| C1b | ungated augmentation → negative transfer OOD | ✅ have | `raw/grids-xmodel-2026-06-05/transfer_*.csv` | §8 |
| C1c | mix-train can't hold all distributions (capacity interference) | ✅ have | `raw/mix-2026-06-05/mix_results.csv` | §8 |
| C2 | one intrinsic signal predicts help across 7 families | ✅ have | `…/xmodel_consistency_7models.csv` | §2/§2b |
| C3a | read gate recovers most negative transfer, zero per-model tuning | ✅ have (offline) | `gate_route_main.csv`, `gate_transfer_lomo.csv` | §7 |
| C3b | honest hierarchy floor/threshold/learned/oracle (oracle has headroom) | ✅ have | `raw/baselines-2026-06-05/gate_baselines.csv` | §7b |
| C3c | gate keeps in-dist gain AND reduces OOD harm (locking) | ✅ have (partial) | `raw/baselines-2026-06-05/locking_by_distance.csv` | §7b |
| C3d | do-no-harm **by construction** (g→0 ⇒ exact base) | ✅ have | gated combine in `wrapper.py` | §0 (E5) |
| A1 | soft residual gate **fails** (motivates hard routing) | ✅ have | gate-sweep doc | §7/H4 |
| A2 | multi-depth injection **fails** (motivates input-level routing) | ✅ have | `raw/deep-2026-06-05/multidepth_trivia.csv` | §7c |
| S1 | significance: per-bench 5-seed means ± CIs | 🔄 running | `abl` batch (23/38) | §0 |
| **C4** | **write-time site protection beats EWC/MoFO (same signal)** | ⏸ out of scope (read-only paper) | Plan 09 | — |
| B1 | read baseline: **TARG** (base-uncertainty) gate | ⚠️ **done (3-family) — TARG ≥ ours** | `raw/baselines-2026-06-05/gate_read_baselines.csv` | §7d |
| B2 | read baseline: output-confidence (Self-RAG-ish) | ✅ done (3-family) | same CSV | §7d |
| **B0** | **7-family head-to-head** ours vs TARG vs out-conf vs ours⊕TARG | ❌ **decisive gap** | pull Phi/Mistral/Q2.5/Q3-14B jsonls | — |
| B3 | write baselines: EWC, MoFO/MIGU, ESFT, head-freezing | ❌ gap | Plan 09 | — |
| O1 | **online** gate (current routing is offline/simulated) | ❌ gap | new run | — |

## Main table (built 2026-06-05) → `mem-embedding/summary/matrix.md` §★★ + `raw/main-table-2026-06-05/main_table.csv`
Qwen3-8B, module trained **in-distribution**: per bench — no-ctx (floor) · SFT-LoRA · Cartridge-lite/Gist-lite (pending) · **OURS** · full-ctx (ceiling).
**One-row thesis:** OURS *adds* QA competence in-dist (trivia +8.6, squad +5.6, narrativeqa +3.5) while **SFT-LoRA forgets** (every QA below no-ctx); compression even **beats full-ctx on narrativeqa** (0.194>0.150); capacity wall on extractive/needle (→ Paper A); MC/needle don't help (honest). do-no-harm gate under shift = §7b. *(SFT = mix/multi-task; per-bench-SFT control = TODO. Memory-module baselines fill from `membase_q`.)*

## Experiment queue (priority)
- **P0 — in hand:** C1a/b/c, C2, C3a–d, A1, A2 (above). Fold significance (S1) when `abl` lands.
- **P1 — read baselines (DONE, 3-family):** TARG (B1) + output-confidence (B2) vs ours → **honest finding: TARG ≥ ours on 3 families** (§7d). *This reshapes the thesis — see [`framing.md`](framing.md).*
- **P1.5 — head-to-head extend (B0):** ⚠️ **ray/test pods deleted** → full 7-family per-item data is gone. Cheap path: **re-run probe on Qwen3-14B** (model on sam-dev) → 4-family head-to-head ours vs TARG vs out-conf vs **ours⊕TARG**. (Phi/Mistral/Qwen2.5 would need re-download.) *Signal question is an ablation now (§7d), not the headline.*
- **P1.6 — multi-layer extension v2 (T2):** the §7c multi-depth collapse → retry with **true zero-init, slow-opening per-layer gate (LLaMA-Adapter style)**, top-layers-only. Either it lifts (positive T2 deliverable) or a rigorous negative (sharpens "input-level + detachable + gated"). *Needs GPU — queue after `abl`.*
- **P2 — relevance eval (T2):** a "context-sometimes-irrelevant" benchmark (mixed relevant/decoy context) to **earn the do-no-harm framing**; without it the agent framing is unbacked.
- **P2.5 — few-step adaptation curve (T2):** train-steps vs in-domain lift → back the "fast/lightweight ingestion" (SP2-as-C) claim.
- **P3 — online gate (O1, T2/T3):** wire apply-or-skip into generation (live, not post-hoc routing).
- **(parked, T3) fancy:** weight-fusion (LoRA-merge), parametric cartridge (crux B), agentic multi-tool routing.

## Target venue / timeline
**OPEN** (decision #2 above). Candidate framings: CF/continual-learning track (problem-first) or
RAG/adaptive-retrieval (method-first). Fill once scope is locked.
