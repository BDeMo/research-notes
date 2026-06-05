# Matrix — history & knowledge sources

> The **matrix** is the long-term memory of this repo.
> Two things it tracks:
> 1. **Sessions** — chronological log of what we did, what we read, what we decided.
> 2. **Knowledge mother nest (`knowledge-sources.md`)** — running index of papers, blogs, and conversations that seed our ideas.
>
> Re-read the matrix before any new session. The point is to *not* repeat yourself.

---

## How to use

### When starting a session
1. Open `docs/matrix/README.md` (this file).
2. Skim the latest session entries.
3. Check `knowledge-sources.md` for any tagged sources relevant to today's topic.

### When ending a session
1. Add a new session entry: `docs/matrix/YYYY-MM-DD-<topic>.md`.
2. If we read new papers or used new sources, add them to `knowledge-sources.md`.
3. Update the index table below.

### Session entry template
```
# YYYY-MM-DD — <topic>

## Activities
- Brainstorm on …
- Read paper …
- Wrote plan …

## Decisions
- Decided to …
- Killed idea …

## Output artifacts
- `notes/<topic>-ideas.md`
- `notes/plans/NN-<slug>/`

## Knowledge sources used
- [paper-id-1] in `knowledge-sources.md`

## Next steps
- …
```

---

## Session index

| Date | Topic | Headline output | Files |
|---|---|---|---|
| 2026-05-26 | Inference-time training (X-W framing) | Brainstorm of 50 ideas + 3 detailed plans + Doc-to-LoRA reading | [`2026-05-26-inference-time-training.md`](2026-05-26-inference-time-training.md) |
| 2026-05-28 | Methods reading + new plan brainstorm | Read 4 methods (Cartridges / Activation Beacon / Gisting / Generative Adapter) for plan 08 v0; 5 J-series ideas + 3 existing ★★+ candidates | [`2026-05-28-methods-reading-and-new-plans.md`](2026-05-28-methods-reading-and-new-plans.md) |
| 2026-06-03 | mem-X v1 results harvest (plan 08 v0) | Extracted facts from 7-day Phase A→Y burst; **3-regime law** (A wins / B at-chance / C collapse) is the v1 paper claim; OPD+RL stages dropped; J5 empirically answered | [`2026-06-03-mem-x-v1-harvest.md`](2026-06-03-mem-x-v1-harvest.md) |
| 2026-06-03 | Long-ctx↔forgetting brainstorm + audit + Plan 09 | 2 rounds → R1–R12 + M1–M9; **2022-2026 audit: none novel as standalone mechanism**; re-scoped to a *general* long-ctx + forgetting method (RCA = application); distilled **design rules DR1–DR15**; drafted **Plan 09** (measure-first coupling study → intrinsic-site anti-forgetting) | [`2026-06-03-rca-transformer-intrinsic-brainstorm.md`](2026-06-03-rca-transformer-intrinsic-brainstorm.md) |
| 2026-06-04 | Cross-repo harvest (all repos → notebook) | Status roll-up of 6 repos: **Janus/Plan-09 executed** (head-coupling falsified → gradient-**shielding** finding); Plan-08 paper reframed **bit-capacity wall** (ICLR 2027); **dLLM-BoN** verifier-readout (EMNLP); RCA-app curriculum LoRA; intern cohort. New weekly-deck input appended | [`2026-06-04-cross-repo-harvest.md`](2026-06-04-cross-repo-harvest.md) |

---

## Active threads

Topics that span multiple sessions and are still alive:

- **Inference-time training** — last touched 2026-06-03. Lead ideas: X-saturation curriculum (plan 01), W-space BoN (plan 03), model outputs ΔW (plan 08).
- **Plan 08 — soft-prompt memory (`latent-mem-paper`, `mem-test/mem-embedding`)** — paper reframed as **"Bit-Capacity Limits of Soft-Prompt Memory"**, target **ICLR 2027**. Headline: a **bit-capacity wall** (exact recall collapses ≤2-bit→0 above ~10 bits; Wrap = matched-K Gist → a *read-interface* property). Supersedes the "3-regime law" framing. v1.5 = do-no-harm gate; v2 = cross-session latent memory. Harvest: [`../../notes/plans/08-model-outputs-delta-w/v1-results-2026-06-03.md`](../../notes/plans/08-model-outputs-delta-w/v1-results-2026-06-03.md).
- **Long-ctx ↔ forgetting = "Janus" / Plan 09 (`janus`, `janus-methods`)** — **executed 2026-06-04**. Head-level "same sites, two frontiers" coupling **falsified under controls**; surviving finding: long-context heads are **gradient-shielded** during instruction-SFT (strong ≤14B, neutralizes by 30B; an instruction-objective effect → why NIAH survives SFT). H3 causal test inconclusive; method-line blocked on SFT recipe. Detail: [`../../notes/plans/09-intrinsic-site-protection/facts-2026-06-04.md`](../../notes/plans/09-intrinsic-site-protection/facts-2026-06-04.md).
- **dLLM Best-of-N verifier readout (`test_env/EMNLP-dllm-BoN`)** — **submitted EMNLP 2026** (2026-05-26), rebuttal staged. AR verifiers read the final token via "confluence" (causal mask); dLLMs break it (bidirectional) → read the **output-span mean**. LLaDA-8B/GSM8K: `last`→`mean` 75.4→83.7 (linear); boundary flips on LLaDA2.0-mini (MoE). Ties to plan 03 + DR15.
- **RCA application (`rca-demo-qwen`)** — the Nokia RCA line. Curriculum LoRA adapters on Qwen2.5/Qwen3 over Nezha → OpenRCA-500 → RCAEval → LincYaw (+ mixed). The application end of long-ctx + forgetting.
- **Nokia intern cohort (`intern-project`)** — multi-intern: MoECompiler (jiamu), agent skills (paimon), empirical results (xueqi), telecom time-series FM (liang), RCA deliverables (mingjia). Full roll-up: [`2026-06-04-cross-repo-harvest.md`](2026-06-04-cross-repo-harvest.md).

## Archived threads

(empty)

---

## Maintenance

Full policy: [`../maintenance.md`](../maintenance.md). Local rules for the matrix:

- **This file is T0** — read every session. Soft cap **80 lines**, hard cap **120 lines**.
- **Latest session entry is T0** — read every session. Soft cap **200 lines**, hard cap **300 lines**.
- **Older session entries are T2** — read only when researching past decisions.
- **knowledge-sources.md is T2** — read only when looking up a citation by `[id]`.
- **Stable IDs**: source IDs in `knowledge-sources.md` are forever. Never renumber.
- **Aging policy**:
  - After **30 days**, a session entry should be reviewed: drop "Next steps" if completed, keep "Activities / Decisions / Output".
  - After **90 days**, eligible for compression: collapse "Activities" to one line, keep "Decisions" + "Output" full.
  - After **12 months**, eligible for archive: `git mv` to `_archive/YYYY-MM/`; in the session index above, replace its row with a compact `archived 2027-MM-DD → [link]` line, or move under an "Archived sessions" collapsible block.
- **Active threads** list: cap at **10 rows**. Anything inactive > 6 months moves to Archived threads.
