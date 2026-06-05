# Paper A — "Long-context heads are gradient-shielded during fine-tuning"

> **Decision (2026-06-05):** this is the first paper. Built on the surviving facts
> (`mechanism-results-2026-06-04.md`, `facts-2026-06-04.md`). The methodological
> cautionary (phantom couplings) is folded in as the rigor backbone, not a separate paper.

## Claim (one sentence)
Contrary to the intuition that long-context machinery is fragile, **the attention heads
that carry long-context behaviour (long-reach / sink) systematically receive *smaller*
fine-tuning gradients** — they are *shielded* — and this shielding is **specific to
task/instruction (Q→A) objectives**, **mediated by activation magnitude / attention
saturation**, and **fades with scale**; it explains why instruction-tuning preserves
long-context, and predicts that long-context erodes only where the shield is absent
(continued LM-pretraining / very large models).

## Headline figures/tables

### Main Table 1 — the phenomenon across scale & family
rows = {Qwen3-0.6/1.7/4/8/14B, GLM-4-9/32B, Qwen2.5-7B-Inst, Qwen3-30B-A3B};
cols = within-layer ρ(a-priori grad, **attn_distance**), ρ(grad, **sink**),
partial|out_norm, layer-level ρ, bootstrap 95% CI. **Have it** (mechanism_*.json).
→ message: shield −0.5…−0.6 @0.6–14B, neutralizes ≈0 @30B+, cross-family.

### Main Table 2 — objective specificity (the boundary that makes it mechanistic)
rows = SFT-gradient source {gsm8k, triviaqa, hotpotqa, narrativeqa | wikitext-LM};
cols = within-layer ρ(grad, attn_distance) for Qwen3-8B, GLM-4-9B. **Have it** (B1).
→ message: shield present for tasks (incl. long-ctx tasks), absent under LM.

### Main Figure — scaling curve
within-layer shield ρ vs params (0.6→32B), one line per family. **Have the points.**

## Ablations (the mechanism)
| # | ablation | question | status |
|---|---|---|---|
| A1 | **carrier disentangle**: partial-out each of {out_norm, attn_entropy, attn_distance, sink} in turn | which property *causally* carries the shield? | offline from npz — **do now** |
| A2 | **a-priori → actual drift**: valid LoRA SFT, measure per-head dW_drift, correlate with the a-priori shield + with NIAH retention | is the shield real (not just a-priori grad)? | needs LoRA trainer — **pre-exp** |
| A3 | **depth profile**: shield ρ per depth-band × scale | where does the shield live, how does it move | offline from npz |
| A4 | **head-type control**: long-ctx heads vs random vs induction vs high-out_norm matched | shield is specific to long-ctx heads, not generic | offline + small run |
| A5 | **NIAH consequence**: NIAH retention after SFT vs shield strength (across models/objectives) | shield → preserved long-context (the payoff) | needs SFT runs |
| A6 | **probe robustness**: seeds / #texts / needle variants → CIs | not seed-dependent | quick run |

## Candidate-method tests (the practical hook)
The shield means **protection is unnecessary under task-SFT** (long-context already
survives) but **should help where the shield is absent** — continued LM-pretraining.
Test the janus-methods candidates exactly there:
| # | test | setup | expected |
|---|---|---|---|
| M1 | protection under **task-SFT** | gsm8k LoRA × {none, attn_distance-freeze, lc_combined, random} → NIAH | no improvement (shield ⇒ NIAH already retained) — *negative-by-design* |
| M2 | protection under **LM-continued-pretraining** | wikitext LoRA (shield absent) × {none, attn_distance-protect, lc_combined, random, deltaw} → NIAH | **protecting long-ctx heads preserves NIAH > random/none** — the method payoff |
| M3 | operator ablation on the winner | freeze / l2sp / orthogonal / grad_scale | which operator best |

→ The paper's method contribution = "**use the shield**: skip protection for instruction-SFT; **apply it for LM-CPT**, where our long-context criterion beats baselines."

## Pre-experiment checklist (in order)
1. **A1 carrier disentangle** (offline, npz) — now.
2. **Fix trainer → LoRA-all-modules** (so SFT learns the task; the broken attn-only recipe motivated this).
3. **A2 a-priori→actual** (LoRA, gsm8k) — does shield predict real low drift + NIAH retention.
4. **M2 method payoff** (wikitext-LM LoRA) — the key test: does protecting long-ctx heads help when the shield is absent. First confirm wikitext-LM SFT *does* erode NIAH (precondition).
5. A3/A4/A6 (offline + quick) to fill the tables.

## Risks / honesty
- If M2 shows protection *doesn't* help even under LM-CPT → the paper is purely the
  measurement+mechanism (still solid; drop the method hook).
- Keep the phantom-coupling controls front-and-center (it's why the result is credible).
- Qwen3.5 hybrid excluded from head-level tables (artifact); mention in limitations.
