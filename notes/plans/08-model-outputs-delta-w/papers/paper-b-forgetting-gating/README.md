# Paper B — Do-No-Harm Adaptation (Forgetting & Gating)

**Status:** 🟢 **ACTIVE — the paper we write first** (started 2026-06-05).
**Working title:** *Do-No-Harm Adaptation: One Intrinsic, Cross-Model Signal to Gate (read) and Protect (write) a Pretrained LLM against Catastrophic Forgetting.*
**Scope source:** [two-paper litreview §Paper B](../../summary/2026-06-05/two-paper-litreview-2026-06-05.md).

---

## Thesis
Adapting a pretrained LLM hurts what it already knows. Two ways to adapt, two ways to forget:
- **Read-time** — bolting on an augmentation (retrieved context, a memory wrapper) **degrades** outputs when the augmentation is irrelevant (*negative transfer*).
- **Write-time** — fine-tuning (SFT/LoRA) on new data **overwrites** general ability (*catastrophic forgetting*).

We show **one intrinsic, data-agnostic, cross-model read-side signal** — the wrapper's own
**memory-write dynamics** (lead `delta_last`) — supports **do-no-harm adaptation on both sides**:
*open the augmentation only when it helps* (read gate), and *freeze the sites that signal
relies on* (write protection). The signal needs **no labels and no per-model tuning**, and
**transfers across 7 model families**.

## One frame, two sides
| side | mechanism | plan | claim |
|---|---|---|---|
| **read** | per-input **routing gate**: apply memory iff predicted-useful, else fall back to frozen base | 08 v1.5 | removes most negative transfer, do-no-harm by construction |
| **write** | per-site **protection**: shield long-context-important weights during SFT | 09 (Janus) | preserves general ability while learning the new task |

> Unifying claim: the **same** intrinsic read-side criterion tells you both *when to open the
> gate* (read) and *what to freeze* (write).

## Contributions
- **C1 — forgetting is real & measurable on both axes.** SFT degrades the base (trivia 0.475→0.250 no-ctx, 0.635→0.301 full-ctx; matrix §7c); ungated augmentation hurts OOD (cross Δ −0.075; matrix §8).
- **C2 — a cross-model intrinsic signal predicts when adaptation hurts.** `delta_last` is direction-consistent in **all 7** families, AUROC(help) 0.59–0.80 (matrix §2/§2b).
- **C3 — read-time gate.** A gate on that signal recovers ~85–95% of the negative transfer with **zero per-model tuning** (LOMO AUROC 0.71), do-no-harm by construction (g→0 ⇒ exact base) (matrix §7/§7b).
- **C4 — write-time protection (Plan 09).** The same read-side importance map flags sites to freeze during SFT, beating EWC/MoFO-style baselines. **← the main open experiment.**

Negative/ablation results that sharpen the story: **soft residual gate fails** (H4), **multi-depth injection fails** (matrix §7c) → both motivate *hard routing over soft/deep residual injection*.

## Honest novelty (do not over-claim)
read gate ≈ adaptive retrieval (**TARG**, Self-RAG, "When do LLMs need RA?"); write protection ≈
**MoFO/MIGU/ESFT** + Mechanistic-Forgetting head-freezing; multi-depth ≈ **LLaMA-Adapter**.
**Novelty = unifying read+write do-no-harm under ONE cross-model-general intrinsic signal**, not any
single mechanism. **Reviewer-mandatory baselines:** read — TARG logit-margin gate, LLaMA-Adapter;
write — EWC, MoFO/MIGU, ESFT, Mechanistic-Forgetting head-freezing.

## Where things stand → see [`outline.md`](outline.md)
Read-side (08) is **mostly evidenced**; write-side (09) is the **main gap**; reviewer baselines + an
**online** gate are the rest. Open scope/venue decisions are flagged at the top of `outline.md`.

## Links
- **outline + claim→evidence map:** [`outline.md`](outline.md)
- evidence ledger: `mem-test/mem-embedding/summary/matrix.md` (§2 signals · §7/§7b gate · §7c forgetting+multi-depth · §8 boundary)
- settings: [`../../settings/settings.md`](../../settings/settings.md) (P08-S6 probe · P08-S8 gate)
- raw data: [`../../raw/`](../../raw/) (sft-2026-06-05 · baselines-2026-06-05 · grids-xmodel-2026-06-05)
- related work: [`../../summary/2026-06-05/v1.5-related-work-2026-06-05.md`](../../summary/2026-06-05/v1.5-related-work-2026-06-05.md)
- write-side plan: [`../../../09-intrinsic-site-protection/`](../../../09-intrinsic-site-protection/)
