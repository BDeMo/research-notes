# Critical review — Paper B ("Know When to Fall Back") as an ICLR 2027 reviewer

> Self-adversarial review. I play the ICLR program: 3 reviewers (incl. the notorious
> committed-to-reject R2) + an AC, score with the real ICLR-2026+ rubric, then iterate the
> paper through **3 revision rounds**, re-scoring each time. Goal: find every kill-shot now,
> while experiments are cheap to add. Pairs with [`gate-pipeline-2026-06-05.md`](gate-pipeline-2026-06-05.md).

## 0 · The rubric I'm grading against (ICLR 2026 reviewer guide, used as proxy for ICLR 2027)
- **Overall** ∈ {0,2,4,6,8,10}; **Soundness / Presentation / Contribution** ∈ {1–4}; + Confidence.
- Four questions: (1) problem? (2) well-motivated & **placed in literature**? (3) **claims supported** (correct, rigorous)? (4) **significance** (new knowledge — SOTA *not* required).
- Empirically: overall correlates **most with Contribution, then Soundness**. High-rated papers win on **novelty + elegant/efficient method + broad, realistic coverage + theory**. Low-rated papers die on **weak baselines, too few datasets/limited domain, missing ablations, reproducibility, overlap with prior work / unclear contribution / marginal gains**.
- Accept threshold ≈ 6. Spotlight ≈ 8.

## 1 · The submission under review (honest one-paragraph abstract)
*Context compression speeds LLM inference but silently loses verbatim facts. We argue that
because a fallback-to-full-context path guarantees a correctness floor, the only thing that
matters is the reliability of the gate that triggers fallback. We therefore study gating
**signals** rather than gate architectures: we define "reliable" via in-domain AUROC,
cross-domain/task/dataset transfer, calibration, and cost, and benchmark ~15 cheap query-time
signals across 8 models × 8 datasets. We find cheap generation-confidence (answer entropy/ppl) is
the most reliable, universal, transferable gate, and test whether latent-substrate-specific
signals can beat it on the cost–quality Pareto.*

---

## 2 · ROUND 1 — reviews of the current scope (summary-proxy compressor, scorecard only)

### Reviewer 1 — soundness hawk  (Overall **4** · Sound **2** · Pres **3** · Contr **2** · Conf 4)
**Summary.** Reframes gated compression as a signal-selection problem; broad empirical scorecard.
**Strengths.** Clean reframing; the correctness-floor argument is correct and well-stated; breadth (8×8) is unusually large for this sub-area; honest definition of "good."
**Weaknesses (the kill-shots).**
1. **The compressor is a model-generated summary, not a real latent wrapper.** Every label
   (`label_fail`) is therefore an artifact of *this* compressor. The entire paper could be
   "properties of summarization failure," not "context compression." **Soundness=2** until you
   show signal rankings transfer to a *trained* latent compressor (Activation-Beacon/ICAE).
2. **Failure label is binary correctness of a 40-token greedy decode** with substring/F1 match —
   noisy, and confounded with the model's base QA ability. NIAH/MCQ correctness is cleaner than
   short-answer F1; mixing them in one AUROC is dubious.
3. **No human/oracle ceiling on the gate.** You report signal AUROC but not the *oracle gate*
   (knows `label_fail`) — without it I can't tell if AUROC 0.75 is good or terrible.
4. **AUROC pooled across heterogeneous datasets** can be Simpson's-paradox'd. You must show the
   *within-base* distribution, not just the median.
**Questions.** Does signal X survive on a trained compressor? What's the oracle vs best-signal gap? Is the label robust to decoding temperature / answer-matching rule?

### Reviewer 2 — committed-to-reject novelty skeptic  (Overall **4** · Sound **3** · Pres **3** · Contr **2** · Conf 3)
**Summary.** "Which signal predicts compression failure, with a fallback." 
**Weaknesses.**
1. **Novelty.** D-Mem (2026) already does gated compressed-memory + full-context fallback on
   LoCoMo; Entropy-Gate already falls back when fidelity is low; SLT already uses a confidence
   gate to fall back to uncompressed. **What is left?** "We compare signals" is a *benchmark*,
   not a method. Benchmarks get into ICLR only when they overturn beliefs or are definitive.
2. **The headline ("reliable gate is all you need") is a slogan, not a finding.** Of course a
   perfect oracle gate suffices — the question is whether a *cheap* one exists, and if it's just
   "answer entropy," that's folklore (selective prediction / confidence calibration, 2017–).
3. **Marginal gains risk.** If signal X beats entropy by 0.02 AUROC, this is a workshop paper.
4. **Where's the deployment win?** No wall-clock latency/throughput numbers; "inference speedup"
   is claimed but never measured end-to-end.
**Recommendation: reject** unless novelty is sharpened beyond "we benchmarked confidence signals."

### Reviewer 3 — empiricist  (Overall **6** · Sound **3** · Pres **3** · Contr **3** · Conf 3)
**Strengths.** Breadth is real and rare; the cross-X transfer framing (calibrate-on-A, test-on-B)
is exactly the right question and under-studied; if a *cheap intrinsic* signal matches the
expensive LLM-judge, that's genuinely useful to practitioners (ICLR Q4).
**Weaknesses.** Baselines incomplete (no D-Mem LLM-judge, no Entropy-Gate reimplementation in
*your* harness); n=60/base is small for AUROC CIs on minority failures; only short-context
datasets for some bases (squad cr≈2 — compression barely happens, so failures are rare/noisy).
**Recommendation: weak accept** if baselines + a real wrapper are added.

### AC meta-review (Round 1)
Scores **4 / 4 / 6** → **reject (below threshold)**. Decisive issues, in order: (a) **novelty vs
D-Mem/Entropy-Gate/SLT** (Contribution=2 twice — and Contribution drives the overall score);
(b) **summary-proxy compressor** undermines soundness; (c) **no strong baselines**, no oracle
ceiling, no latency. The reframing is liked; the execution reads as "a benchmark of confidence
signals," which won't clear the bar.

---

## 3 · What Round 1 demands (revision plan R→1)
Prioritized by score leverage (Contribution & Soundness first):
1. **Sharpen the contribution beyond a benchmark.** Promote one of:
   (a) a **cheap, mechanism-grounded signal** (retrieval-head routing / counterfactual-KL) that
   **matches D-Mem's LLM-judge at a fraction of the cost** — that's a *method*, not a benchmark;
   (b) a **calibration/transfer theorem**: conditions under which a threshold set on domain A
   controls fallback-recall on B (conformal risk control) → turns the study into a *guarantee*.
2. **Add a trained latent compressor** (Activation-Beacon or ICAE, 1 model) and show the
   **signal ranking is preserved** (rank correlation of the scorecard, summary-proxy → trained).
   Kills R1's #1 and R2's "it's just summarization."
3. **Strong baselines in our harness:** D-Mem-style LLM-judge gate, Entropy-Gate threshold,
   AdaComp-rate, random, always-comp, always-full, **oracle gate**. Report the full Pareto.
4. **End-to-end latency/throughput** (tokens/s, ms/query) at matched quality — make "speedup" real.
5. **Clean the label:** separate AUROC by task-type (verbatim vs gist); report within-base
   violins + per-base CIs (bootstrap); bump failing-rich bases to n=120.
6. **Selective-prediction positioning:** explicitly relate to confidence calibration / selective
   prediction so R2 can't call it folklore — and show the *counterfactual* and *retrieval-head*
   signals beat plain confidence (the novel part).

---

## 4 · ROUND 2 — re-review after R→1 revision

### Reviewer 1  (Overall **6** · Sound **3** · Pres **3** · Contr **3** · Conf 4)
Compressor-transfer experiment resolves my main concern (rankings ρ=0.8 summary→trained is
convincing enough, though one compressor is thin — try two). Oracle gate + within-base CIs added;
AUROC now interpretable. Remaining: label still correctness-based; consider answer-confidence
calibration as a sanity layer. **Up to weak accept.**

### Reviewer 2  (Overall **4→6** · Sound **3** · Pres **3** · Contr **3** · Conf 3)
The "cheap intrinsic signal matches LLM-judge at ~1/20 the cost" result is the first thing here
that isn't folklore — *if* it holds across families and the cost gap is real and measured. The
counterfactual-KL beating raw entropy addresses my folklore complaint. I still think the
conceptual delta over D-Mem is modest; I won't champion it, but I won't die on the reject hill.
**Borderline.** Raise to 8 only with the transfer *guarantee* (R→2 below) or a deployment study.

### Reviewer 3  (Overall **6→8** · Sound **3** · Contr **3→4** · Conf 3)
With D-Mem + Entropy-Gate baselines in-harness and the Pareto dominance shown, this is now a
useful, broad, reproducible study with a practical payoff. Contribution=4. **Accept.**

### AC meta (Round 2): **6 / 6 / 8 → borderline accept.** Contribution recovered (cheap-signal-
beats-LLM-judge + breadth). To move from borderline to clear accept/spotlight, need either the
**guarantee** or a **deployment** result (R2 is the swing vote).

## 4.5 · Revision plan R→2 (to convert R2 and reach 8s)
- **Conformal fallback guarantee:** calibrate τ on A so that fallback-recall ≥ 1−α on exchangeable
  B; prove + empirically verify coverage under the 3 shifts. This is the theory ICLR rewards and
  exactly fits the "reliable gate" thesis.
- **One real deployment table:** vLLM/HF throughput at matched quality vs always-full and D-Mem —
  show ms/query and $/1k-queries. Converts "significance (Q4)" doubt to a number.
- **Second trained compressor** for the transfer claim (n=2 compressors → rank stability).

---

## 5 · ROUND 3 — after R→2

### Reviewer 2 (the holdout)  (Overall **6→8** · Contr **3→4**)
The conformal coverage result under cross-task shift is the contribution I wanted: a *cheap*
signal with a *distribution-shift guarantee* on the fallback, beating the LLM-judge on cost — that
is new relative to D-Mem (heuristic judge, no guarantee). I'd still like LoCoMo (D-Mem's turf) in
the table for a head-to-head. **Accept.**

### AC meta (Round 3): **8 / 6 / 8 → accept, spotlight-borderline.** Predicted final: clear accept
if LoCoMo head-to-head lands; spotlight if the guarantee is clean and the deployment win is ≥2×.

---

## 6 · Decisive must-haves (distilled — do these or don't submit)
**Tier 1 (turns reject→accept; do first):**
1. Compressor-transfer: summary-proxy → ≥1 trained latent wrapper; show scorecard rank-stability.
2. Strong baselines **in our harness**: D-Mem LLM-judge, Entropy-Gate, AdaComp-rate, oracle, random, always-comp/full → cost–quality Pareto.
3. The novel signal must **beat plain confidence/entropy** (counterfactual-KL and/or retrieval-head routing) and **match the LLM-judge at ≥10× lower cost** — with the cost measured.
4. End-to-end latency/throughput at matched quality (make "speedup" a number).

**Tier 2 (accept→spotlight):**
5. Conformal fallback guarantee + empirical coverage under cross-domain/task/dataset shift.
6. LoCoMo head-to-head vs D-Mem (fight on their benchmark).
7. Second trained compressor; per-base CIs; n=120 on failing-rich bases.

**Writing (cheap, high-leverage — low-rated papers die here):**
- Lead with the contribution (cheap+guaranteed gate beats expensive judge), not the slogan.
- Place vs D-Mem/Entropy-Gate/SLT in the *intro*, not just related work (R2 reads intro first).
- One figure = the Pareto; one table = the scorecard; one theorem = the guarantee.

## 7 · Honest verdict
Current scope (Round 1) = **reject (~4.7)**: reads as a confidence-signal benchmark overlapping
D-Mem. The path to **accept (~6.7)** is concrete and mostly *analysis* on data we are already
collecting (baselines, oracle, transfer); the path to **spotlight (~7.5)** needs the conformal
guarantee + a deployment number. The breadth we're collecting tonight is necessary but **not
sufficient** — Contribution, not coverage, is the gate. Build the cheap-signal-beats-judge +
guarantee story, or this is a workshop paper.
