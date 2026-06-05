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
| **C4** | **write-time site protection beats EWC/MoFO (same signal)** | ❌ **gap** | Plan 09 experiments | — |
| B1 | read baseline: **TARG** logit-margin gate | ❌ gap | buildable from existing signal jsonls | — |
| B2 | read baseline: Self-RAG / confidence gate | ❌ gap | — | — |
| B3 | write baselines: EWC, MoFO/MIGU, ESFT, head-freezing | ❌ gap | Plan 09 | — |
| O1 | **online** gate (current routing is offline/simulated) | ❌ gap | new run | — |

## Experiment queue (priority)
- **P0 — in hand:** C1a/b/c, C2, C3a–d, A1, A2 (above). Fold significance (S1) when `abl` lands.
- **P1 — read-side baselines (cheap, reuse signal jsonls):** TARG logit-margin gate (B1), confidence gate (B2) → compare AUROC/harm-recovery vs `delta_last` gate. *No new GPU for the offline versions.*
- **P2 — online gate (O1):** wire the routing decision into generation (apply memory vs skip per-input) and re-measure do-no-harm live, not via post-hoc routing.
- **P3 — write-side (C4, Plan 09):** build long-ctx-importance site map → freeze during SFT on the mix; compare to no-protect SFT (the §7c forgetting baseline), EWC, MoFO/MIGU, ESFT, head-freezing. *Decides whether Paper B = read⊕write or read-only.*

## Target venue / timeline
**OPEN** (decision #2 above). Candidate framings: CF/continual-learning track (problem-first) or
RAG/adaptive-retrieval (method-first). Fill once scope is locked.
