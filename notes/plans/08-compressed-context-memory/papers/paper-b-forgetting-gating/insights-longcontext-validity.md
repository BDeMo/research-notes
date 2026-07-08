# Long-Context Insights — when do they actually hold? (validity map)

> Which commonly-stated long-context "insights" are valid, and **precisely in which scenarios**. Grounded in the diagnosis
> campaign (D18) + dive-into wave (D19) + fact-base. Figures in `figures/`. Each insight gets: the claim · where it HOLDS ·
> where it FAILS · the controlling variable.

![length collapse](figures/fig1_length_collapse.png)
![ratio cliff](figures/fig2_ratio_cliff.png)
![task flip](figures/fig3_task_flip.png)
![distractor causal](figures/fig4_distractor_causal.png)
![rag vs full](figures/fig5_rag_vs_full.png)

---

## The validity map (8 insights × scenario boundary)

| # | Insight (as usually stated) | HOLDS when… | FAILS when… | controlling variable |
|---|---|---|---|---|
| I1 | "Longer context helps" | the evidence is a **findable item** (RULER full ↑ to 1.0 at 32k) | the task is **literary MC** (quality full 0.06 < no-ctx 0.20) | task type, not length |
| I2 | "KV compression is near-lossless" | **ratio ≤0.25 AND ≤8k AND retrieval** | **≥50% eviction OR ≥16k OR QA** | ratio × length × task jointly |
| I3 | "Attention tells you which KV to keep" | **≤14k** (snapkv flat 0.52→0.62 to 14k) | **a sharp threshold at ~16k** — snapkv 0.62→0.17, expected →0.06 (Dive B) | a hard length threshold, not gradual |
| I4 | "A recent window is enough" | the answer is **local/extractive** (squad w256 = full) | **retrieval/multi-hop/distractor** (RULER w4096=0.56≪full; rca-window collapses under distractors, Fig 4) | answer locality |
| I5 | "Prompt compression preserves the info" | **high keep-rate OR salient answers** (trivia 0.75>full) | **low rate on non-salient needles** (RULER r0.1=0.00) | answer salience × rate |
| I6 | "RAG beats stuffing the context" | **lexical overlap** exists (NIAH, extractive, trivia) | **abstractive/global** (narrativeqa ≈ no-ctx; QuALITY RAG < no-ctx, Fig 5) | query–evidence lexical overlap |
| I7 | "Distractors degrade long-context reading" | for **windowing & prompt-compression** (Fig 4: window 0.78→0.17, LLMLingua 0.81→0.47) | for **full-read & RAG**, which stay ~0.8 flat (refines the earlier claim) | the *method*, not the model |
| I8 | "Compression saves meaningful memory" | **≥128k** context (KV becomes the dominant term) | **≤8k** (KV ≈1 GB ≪ 16 GB weights → negligible) | context length |

## kvzip: the latest cliff, but real — and retrieval-specific (corrected by Dive A)
Earlier I read kvzip as "no cliff" because the ratio grid stopped at 0.85. **Dive A (0.9/0.95/0.99) shows it does cliff, just latest of all:** RULER 0.98 (≤0.85) → **0.83 @0.9 → 0.08 @0.95 → 0.00 @0.99** (Fig 2). And the robustness is **retrieval-only**: at 0.9, kvzip is squad 0.39 / hotpot 0.43 / **narrativeqa 0.14** — it preserves the *needle* well but not distributed semantic evidence. knorm/criticalkv already collapse to ~0 by 0.9. So the precise statement: **kvzip has the highest usable compression budget (~0.9) on retrieval, none of it on abstractive QA.** The reconstruction-importance idea is still the right hint for our memory — but it buys *retrieval* robustness, not semantic robustness.

## Refined / corrected reads (precision matters)
- **"Full context hurts" is literary-MC-specific, not distractor-driven.** On rca, full is *flat* against 0→8 distractors (Fig 4). Full underperforms no-context only on QuALITY/quality_hard (a base-capability ceiling on long literary MC), not because of distractor volume.
- **The distractor victim is the recent-window** (StreamingLLM-style), because added distractor incidents push the answer incident out of the window. RAG sidesteps this by *retrieving* the right incident.
- **"No single method is best" is the firmest insight**: the ranking flips on two independent axes — length (I3 vs attention-free) and task (I6/Fig 3) — so any fixed compressor is dominated somewhere.

## What this says for the method (v2.0.0)
The robust corner that **no** training-free baseline covers: **long + abstractive/global + distractor-mixed** evidence — where window collapses (I4/I7), RAG can't retrieve (I6), attention-KV collapses (I3), and full either doesn't fit or is capability-limited (I1). That intersection is the necessity regime a **self-verified compressed memory** targets; kvzip's reconstruction-importance result (Fig 2) is the empirical hint that a **reconstruction-trained** memory is the right carrier.

*Provenance: `baseline-diagnosis-report.md` (S1–S10) + dive-into `da/db/dc/dd_*` + `baseline-factbase-v2.0.0.md`. Figures auto-generated from the `s*`/`d*` result logs.*
