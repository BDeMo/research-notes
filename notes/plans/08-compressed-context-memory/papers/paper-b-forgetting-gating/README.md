# Paper B — Do-No-Harm Adaptation (Forgetting & Gating)

> 🟢 **CURRENT STATE (2026-06-15): v1.7.5.** The program ran through v1.7 (a gated compressor on tool-use / RCA);
> a code review then found a **train/eval leakage bug** that had inflated the tool results. Everything was re-run
> clean as **v1.7.3**, and the thesis was **reframed** from "a better compressor" to **a compressor-agnostic
> robustness layer** — a confidence / self-verification gate that *detects when compression is unsafe and falls
> back to full context* (do-no-harm). v1.7.4/v1.7.5 then showed (cleanly, under stable training) that compression is
> **capacity-bound** — length-adaptive memory and scaling don't help, the memory is OOD — so the cure is the gate +
> **minimize-scale**. **Start here:**
> latest [`results-v1.7.5/results-v1.7.5.md`](results-v1.7.5/results-v1.7.5.md) ·
> one-page brief [`summary-matrix-v1.7.3.md`](summary-matrix-v1.7.3.md) ·
> full v1.7.3 results [`results-v1.7.3/results-v1.7.3.md`](results-v1.7.3/results-v1.7.3.md) ·
> thesis+plan [`results-v1.7.3/robustness-plan.md`](results-v1.7.3/robustness-plan.md) ·
> reviewer-response [`results-v1.7.3/reviewer-response.md`](results-v1.7.3/reviewer-response.md) ·
> setup [`results-v1.7.3/experimental-setup.md`](results-v1.7.3/experimental-setup.md).
> The v1.5/v1.6 material below is **historical context**; v1.7 (`results-v1.7/`, `summary-matrix-v1.7-*`) is
> **archived (leakage-era — do not cite its numbers).**

> 📨 **Writing agent: start at [`summary-matrix.md`](summary-matrix.md)** — the single-source handoff
> brief (thesis · claims↔evidence with exact numbers + sources · main table · baselines · novelty
> defense · ⛔do-not-overclaim list · section plan · asset index · status).

**Status:** 🟢 **ACTIVE — the paper we write first** (started 2026-06-05).
**Working title:** *Do No Harm: a portable, model-agnostic memory module that knows when not to fire.*
**Scope source:** [two-paper litreview §Paper B](../../summary/2026-06-05/two-paper-litreview-2026-06-05.md). **Reasoned spine + locked decisions:** [`logic.md`](logic.md). **Framing:** [`framing.md`](framing.md).

**Decisions locked (2026-06-05):** read-only scope (write-side/Plan-09 parked); **memory = task/distribution adapter** (crux C — cede fact-storage to Paper A's capacity wall); **SP1 = original single-layer wrapper + multi-layer extension first, weight-fusion/parametric = v2**. De-risked **tiered** plan (T1 safe core ✅ evidenced / T2 strengthen 🔄 / T3 fancy ⏸) — see [`logic.md` §6](logic.md).

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

## Framing (popular / agent-oriented) → see [`framing.md`](framing.md)
We **do not** lead with narrow "catastrophic forgetting." Headline = **do-no-harm augmentation for
agents**: *a pluggable, inference-time memory module that knows when not to fire* (gate→base
fallback, do-no-harm by construction).
**⚠️ Honest constraint (read-baseline run §7d):** on 3 families a trivial **TARG** base-uncertainty
gate ≥ our `delta_last` signal (LOMO 0.60 vs 0.56). So the defensible contribution is the
**framing + do-no-harm-by-construction for a *learned* module + cross-model robustness of intrinsic
gating** — *not* "our signal beats all." The 7-family head-to-head (ours vs TARG vs ours⊕TARG) is the
decisive next run.

## Honest novelty (do not over-claim) → full review in [`baselines-and-novelty.md`](baselines-and-novelty.md)
**Two ICLR'26 neighbors bracket the pitch:** the *memory module* ≈ **Cartridges** (offline KV-cache on
a frozen base, composable, ICL-matching — [2506.06266](https://arxiv.org/abs/2506.06266)); the *gate* ≈
**TARG** (training-free prefix-logit margin gate — [2511.09803](https://arxiv.org/abs/2511.09803), and
§7d shows it ≥ ours). multi-depth ≈ **LLaMA-Adapter**. ⇒ **White space = the systematic *do-no-harm*
treatment of a pluggable memory** (capability boundary §8 + do-no-harm-by-construction + cross-model
gate), **complementary to Cartridges, extending TARG to learned modules** — *not* a new module or signal.
**Reviewer-mandatory baselines:** Cartridges/Gist (memory module) · TARG + output-conf + oracle (gate, ✅ §7b/§7d) · LLaMA-Adapter (≈ multi-depth ablation) · SFT/LoRA (forgetting, ✅ §7c).

## Where things stand → see [`outline.md`](outline.md) + [`framing.md`](framing.md)
Read-side (08) is **mostly evidenced**, but the read baselines show **TARG is competitive** → the
decisive gap is the **7-family head-to-head** + a **relevance/agent eval** to earn the framing.
Write-side (Plan 09) is **parked** (read-only scope).

## Links
- **v1.5 method (architecture + all combine-mode math + gate + two-track):** [`method.md`](method.md)
- **2026-06-08 decision log (scope/framing/experiment decisions):** [`decisions-2026-06-08.md`](decisions-2026-06-08.md)
- **lit-review + claim check (novelty, must-cite, gaps):** [`litreview-claimcheck-2026-06-08.md`](litreview-claimcheck-2026-06-08.md)
- **gate-generality experiment design:** [`exp-gate-generality.md`](exp-gate-generality.md)
- **outline + claim→evidence map:** [`outline.md`](outline.md)
- evidence ledger: `mem-test/mem-embedding/summary/matrix.md` (§2 signals · §7/§7b gate · §7c forgetting+multi-depth · §8 boundary)
- settings: [`../../settings/settings.md`](../../settings/settings.md) (P08-S6 probe · P08-S8 gate)
- raw data: [`../../raw/`](../../raw/) (sft-2026-06-05 · baselines-2026-06-05 · grids-xmodel-2026-06-05)
- related work: [`../../summary/2026-06-05/v1.5-related-work-2026-06-05.md`](../../summary/2026-06-05/v1.5-related-work-2026-06-05.md)
- write-side plan: [`../../../09-intrinsic-site-protection/`](../../../09-intrinsic-site-protection/)
