# 2026-06-03 — mem-X v1 results harvest (Plan 08 v0)

> Session goal: extract all facts from `mem-test/mem-embedding/` (mem-X axis of Plan 08 v0) into research-notes after Phase Y completion. The v1 paper (`latent-mem-paper`) is now submittable as a characterization paper. The 3-regime law is the headline.

## State refresh (since 2026-05-28)

- Plan 08 v0 → split into **3 sister wrapper axes**: mem-X (input embeddings, this paper), mem-H (hidden state, parked), mem-W (LoRA-delta via hypernet, parked).
- ~7 calendar days of intense execution: Phases A → Y, 25 phases, on a 4×H100 pod (`sam-dev`) that grew to 8×H100 on 2026-06-03 (added `sam-dev-ray`).
- Critical-review chain landed: v1.0 → v1.1 → v1.1.5 → v1.2 → v1.3, each ablating a previous headline.
- The **headline claim moved twice in 24 hours**:
  - v1.2 (~19:30 UTC 06-03): "OURS > Gist on QuALITY +12 pp" (seed 42 only).
  - v1.3 (~20:35 UTC 06-03): Phase Y 4-seed bands → that +12 pp was a seed artifact. Settled on the **3-regime law** characterization.

## Activities

- Read mem-embedding's `README.md`, `summary/matrix.md` (140K chars; T2), `summary/critical-review-{v1.1.5,v1.3}.md`, `summary/v1-submission-checklist-2026-06-03.md`, `docs/{project-status,architecture,v2-roadmap}.md`.
- Identified the canonical recipe, the 3-regime evidence, the superseded claims, the engineering findings, the operational lessons.
- Wrote `notes/plans/08-model-outputs-delta-w/v1-results-2026-06-03.md` — the fact-dense harvest doc (15 sections).
- Updated `notes/plans/08-model-outputs-delta-w/README.md`: status → "v1 paper submitting (characterization)"; added v1-status banner + pointer to harvest doc and pivot menu.
- Added 8 public-benchmark `[id]`s to `docs/matrix/knowledge-sources.md`: `[quality]`, `[musr]`, `[hotpotqa]`, `[narrativeqa]`, `[triviaqa]`, `[msmarco]`, `[squad2]`, `[wikitext103]`. Plus `[icae]`, `[autocompressor]`, `[h2o]`, `[streamingllm]`, `[tokmem]` (v2 baseline target).
- Updated `docs/matrix/README.md` session index and active-threads block.

## Decisions

- **v1 paper framing is characterization, not superiority.** The honest claim is the 3-regime law. Workshop floor; main-venue (ICLR 2027 / EMNLP 2026) is medium-risk under Direction B (hybrid retrieval) pivot.
- **OPD and RL stages dropped from the v1 final recipe.** No measurable lift over SFT-only at Phase Y. Reduces "method complexity" page in the paper.
- **4-seed gate is mandatory** for every headline number going forward. Single-seed cells in tables get an "indicative" label.
- **"Lightweight" wording dropped from abstract** — 218 M trainable params = small *model*, not small *adapter*.
- **K=32 is recipe-locked**; K=36 collapses to 0.000. The K-brittleness is now a documented finding, not a confound to hide.
- **mem-H and mem-W axes stay parked** until v1 ships. v2 is mem-X + suffix memory.
- **J5 (Beacon KV memory) idea is empirically answered**: lossy soft-token compression hits a Regime-C wall; KV-activation memory ([act-beacon]) is the recommended Direction F variant for v2/v3. Demoted from queue to "done — picked Direction F as carrier".

## Headline numbers (the 3-regime law)

| benchmark | setting | metric | OURS μ ± σ (n=4) | GIST μ ± σ (n=4) | best baseline | regime |
|---|---|---|---|---|---|---|
| **QuALITY-val** | [`P08-S2`](../../notes/plans/08-model-outputs-delta-w/settings/settings.md#p08-s2--v1-phase-y-three-regime-benchmark-cells) | `accuracy_letter` | **0.193 ± 0.032** | 0.180 ± 0.044 | no_context 0.141 | **A** — wrapper wins |
| **MuSR-mm** | [`P08-S2`](../../notes/plans/08-model-outputs-delta-w/settings/settings.md#p08-s2--v1-phase-y-three-regime-benchmark-cells) | `accuracy_letter` | 0.493 ± 0.008 | 0.501 ± 0.013 | full_context 0.551 | **B** — at-chance |
| **RULER-NIAH** | [`P08-S2`](../../notes/plans/08-model-outputs-delta-w/settings/settings.md#p08-s2--v1-phase-y-three-regime-benchmark-cells) | `exact_value_match` | **0.000 ± 0.000** | 0.000 ± 0.000 | full_context 0.995 | **C** — collapse |

Source and internal setting details:
[`P08-S2`](../../notes/plans/08-model-outputs-delta-w/settings/settings.md#p08-s2--v1-phase-y-three-regime-benchmark-cells).
Per-seed RULER-NIAH: every seed = exact zero on both arms. Cleanest possible negative result.

## Output artifacts (this session)

- `notes/plans/08-model-outputs-delta-w/v1-results-2026-06-03.md` (**NEW** — 15-section fact dump; T2)
- `notes/plans/08-model-outputs-delta-w/README.md` — banner + status update
- `docs/matrix/knowledge-sources.md` — +13 new `[id]`s (8 public benchmarks + 5 related-work entries)
- `docs/matrix/README.md` — session index + active-threads update
- This entry

## Knowledge sources used / added

New IDs (full entries in `knowledge-sources.md`): `[quality]`, `[musr]`, `[hotpotqa]`, `[narrativeqa]`, `[triviaqa]`, `[msmarco]`, `[squad2]`, `[wikitext103]`, `[icae]`, `[autocompressor]`, `[h2o]`, `[streamingllm]`, `[tokmem]`.

Used heavily already in our scan: `[gisting]` (matched-soft-prompt baseline), `[ruler]` (Regime C benchmark), `[act-beacon]` (J5 / Direction F), `[cartridges]` (Regime A neighbour), `[genadapter]` (LoRA-delta hypernet = north star).

## Open questions raised this session

- **Why does Gist also hit 0.000 on RULER-NIAH?** This was unexpected — Gist is K=48 vs OURS K=32, but both still die. Hypothesis: the bottleneck is "any lossy compression destroys the needle string", not "OURS-specific architectural issue". If true, it generalizes the Regime C claim to all soft-prompt compressors.
- **Does Direction D (infilling objective) actually fix Regime C?** Testable in ~3 hours. If yes, v1 paper is "back on" as a superiority paper; if no, the 3-regime characterization is a structural fact.
- **Bit-capacity wall position is invariant across Qwen3-8B and Qwen3-14B** — is this an X-saturation phenomenon (plan 01) on the *wrapper's K-budget* rather than on context length? Worth a sub-figure / sub-section bridge.
- **For v2**: does the wall hold at *write-time* (suffix memory, after generation) the same way as at *encode-time*? Key empirical question of the v2 paper.

## Next steps

1. **User decision** on v1 risk level: low (Direction A only, workshop) / medium (A + B = hybrid retrieval v1.5) / high (A + B + D + I in parallel over 24 hr).
2. If medium/high: run Directions B / D / I cells overnight (predictions for B already on disk; D needs new training objective; I needs unfreezing the last attention block).
3. Build Figure 2 (3-regime bar chart) — highest-impact figure.
4. Write critical-review v1.4 after RULER 4-seed bands are formally tabled in `main.tex`.
5. Mirror the canonical recipe block (BEST_RECIPE + BEST_ARCH from `build_sweep.py`) into `details.tex` §7.

## Pruning check (per maintenance.md §5)

- All T0 files within hard caps after this session (verified: `wc -l memory/*.md docs/matrix/README.md notes/ideas/README.md` post-edit).
- `notes/plans/08-.../v1-results-2026-06-03.md` is T2 (no cap).
- `knowledge-sources.md` grew to ~610 lines (T2, no cap).
- This session entry within 200-soft / 300-hard line cap.
