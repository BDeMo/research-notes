# Decision log — 2026-06-08 (Paper B / ICLR 2027)

> Faithful record of what we discussed and decided this session: scope, framing,
> lit-review conclusions, the new generality experiment, and the writing decisions.
> Maintained doc. Companion: `iclr2027-story.md` (spine), `litreview-claimcheck-2026-06-08.md`,
> `exp-gate-generality.md`, `method.md`. House rule: avoid em-dashes.

## 1. Scope decision: latent context compression (setting), do-no-harm gating (sell)
- We considered renaming the scope away from "latent memory" because the lit review
  showed that lane is crowded.
- **"Latent space consolidation" was proposed and rejected.** Two reasons, both
  evidenced: (a) "latent / KV consolidation" is already taken by FlashMem (frozen
  backbone + KV consolidator + uncertainty-triggered consolidation, very close to us),
  Bottlenecked Transformers, and Mela; (b) in continual learning "consolidation"
  specifically means integrating new knowledge **into weights** (EWC = Elastic Weight
  Consolidation; CLS sleep-phase), which is the opposite of our frozen-base method, so
  the term fights our own thesis.
- **"Latent context compression" adopted as the SETTING.** It is a more honest object
  name than "memory" (we compress this item's context into K=64 tokens per item; we do
  not maintain a persistent store), and it sets up the baselines (Gist/Cartridge-lite)
  and the cost axis (token %) cleanly.
- **Do-no-harm gating is the CONTRIBUTION/sell, not compression.** We do not claim a
  better compressor (Cartridges/500x compress more, and our compressed context is below
  full on extractive QA). We claim a *safer* one: a per-input gate that decides when the
  compressed context is safe to use. User confirmed: "we sell with do-no-harm gating."
- **Forgetting demoted** from headline to a supporting contrast (compress-into-a-module
  vs fine-tune-into-weights), since the setting is now compression, not lifelong adaptation.
- **Spine concept (kept):** the **do-no-harm floor** = frozen base + detachable compressed
  tokens ⇒ worst case is the bare base, by construction.

## 2. Lit-review conclusions (claim check)
Full detail in `litreview-claimcheck-2026-06-08.md`. Decisions taken from it:
- **How current latent memory works:** compression line (Gist, ICAE, AutoCompressor,
  500xCompressor, Cartridges) optimises *how much* a budget carries and uses the tokens
  unconditionally. None gate *when* to use them. Correct as we describe it.
- **Novelty tightened (mandatory):** *when-to-augment* gating is mature for **retrieval**
  (TARG = our cheap base-uncertainty gate; L-RAG = our Track B; CTRLA). We must not imply
  we invented it. Defensible delta = gate a **learned latent compression** (not retrieval),
  **frozen-base floor by construction** (not learned robustness like Astute RAG), on a
  **memory-write signal** `delta_last` that transfers (utility AUROC 0.71 vs retrieval
  scores 0.53 in "When Retrieval Hurts").
- **Must-cite added:** Cartridges, TARG, L-RAG, CTRLA, Astute RAG, "When Retrieval Hurts",
  FlashMem, OPLoRA (bib entries drafted, authors flagged TODO).
- **Baseline gap (open):** we only beat **vanilla** LoRA on forgetting; forgetting-aware
  LoRA exists (O-LoRA, OPLoRA, CLoRA). Decision: add an **O-LoRA/OPLoRA** baseline, or
  scope the claim to vanilla LoRA and cite the mitigation line.

## 3. The gate-generality experiment (the key new decision)
- **Why:** the gate is orthogonal to the compressor, so a reviewer asks whether the
  gating signal generalises beyond our compressor. We run the *same* gate across multiple
  compressors to show it does. This bridges the orthogonality gap.
- **Design:** gate on a **compressor-agnostic** signal set (base uncertainty + the base's
  response to the compressed tokens conf_w/margin_w/kl_w0 + first-token agreement), NO
  `delta_last`. Compressors = Gist-lite, Cartridge-lite, ours. Bases = q8/glm/q14/q25/mistral.
  Benches = 7. Analyse with `gate3_route --general`.
- **Implemented:** `mem_baselines.py` logs `comp_unc` under `--gate`; `gate3_route.py`
  gained `FEATS_GEN` + `--general`. Smoke-tested.
- **Running:** 70 `gx_*` cells (also serve as the proper two-track baseline cells). ETA ~1 h.
- **Honest implication (decided to state):** if the agnostic gate works everywhere, the
  generality comes from the agnostic signals, so `delta_last` is an optional ours-only gain,
  not the load-bearing signal. We say this rather than overclaim `delta_last`.

## 4. Writing decisions
- **Voice:** match the EMNLP-BoN house style (`notes/writing/emnlp-bon-style.md`): one coined
  spine concept, question-driven title/heads, "not X; it is Y" scoping, mechanism before
  assertion, main-vs-diagnostic discipline, reading guidance per table, boundary framing.
- **No em-dashes** (破折号): use comma/colon/semicolon/parentheses; `n/a` in empty table cells;
  keep en-dash `--` number ranges. Recorded in the style guide.
- **Title:** "Do No Harm: When Is a Compressed Context Safe to Use?" (do-no-harm spine +
  compression object + question).
- **Both `main_iclr2027.tex` and `details.tex`** rescoped to compression; `main_iclr2027.tex`
  deepened section by section + two pgfplots figures (forgetting bar, two-track scatter), 9 pp.
- **File layout decision:** the do-no-harm paper lives in `main_iclr2027.tex` (canonical,
  Overleaf-safe), separate from the Overleaf-managed legacy `main.tex`.

## 5. Open decisions / TODO
- **`main.tex` fate** (unresolved): restore to the legacy bit-capacity Paper A / delete /
  leave as duplicate. main_iclr2027.tex is canonical regardless.
- **O-LoRA / OPLoRA baseline** for the forgetting claim (close the gap from §2).
- **Decoy / relevance eval** (gate close-rate on irrelevant context) — needs a new script;
  the reviewer-critical "does the gate decide blind" test.
- **MC full-context sanity check** (quality/musr `full` looks degenerate).
- **CI bands** (from the gs_ seed runs) + fill the `\Pend` cells (generality tables, MC-LoRA).
- **Qwen3.5** stays dropped (transformers lacks `qwen3_5`).

## 6. Artifacts produced this session
- Repo `BDeMo/-ICLR-2027-Latent-Mem`: rescoped + deepened `main_iclr2027.tex` (+figures),
  rescoped `details.tex`, README/project-status/references updated, bib entries added.
- research-notes: `method.md`, `iclr2027-story.md`, `litreview-claimcheck-2026-06-08.md`,
  `exp-gate-generality.md`, this log, and `notes/writing/emnlp-bon-style.md`.
- Code: `mem_baselines.py` (`comp_unc`), `gate3_route.py` (`FEATS_GEN`/`--general`), synced to pods.
- Compute: 70 `gx_*` generality runs + CI seeds in flight on 9 GPUs (gpu-runq).
