# Gated Context Compression — pipeline spec, setup & "good gating signal" definition

> Created 2026-06-05. New line (call it **Paper B** = the v1.5 wrapper/gating line, Plan 08
> v1.5), distinct from Paper A
> (gradient-shielding) and the protection-sweep. Working title:
> **"Know When to Fall Back: Selecting Reliable Gating Signals for Context Compression."**
> Code: `janus-methods/gate_study.py` (collection) + `gate_analyze.py` (analysis).
>
> **Scope (set 2026-06-05): parametric / latent-space *agentic* memory.** The memory is compressed
> latent vectors integrated with the model (MemoryLLM / Memory³ / ICAE family), not text/vector RAG.
> Full scope + literature review + novelty defense:
> [`gate-scope-litreview-2026-06-05.md`](gate-scope-litreview-2026-06-05.md). §6 below is the short
> novelty table; the scope doc is the authoritative version (parametric substrate is the wedge).

---

## 1 · The pipeline under test

```
                          ┌─────────────────────────────────────────┐
   long context  ───────► │  COMPRESS  (wrapper → k latent slots)    │ ──► compressed memory M
   (N tokens)             └─────────────────────────────────────────┘
                                            │
   query q ──────────────┐                  ▼
                         ┌──────────────────────────────┐   s ≤ τ   ┌────────────────────┐
                         │  GATE  s = signal(M, q)       │ ────────► │ answer from M (fast)│
                         │  (cheap, query-time)          │           └────────────────────┘
                         └──────────────────────────────┘
                                            │ s > τ  (gate fires)
                                            ▼
                         ┌──────────────────────────────────────────────┐
                         │  FALL BACK: answer from FULL context (slow)   │
                         └──────────────────────────────────────────────┘
```

**The one idea that makes this worth a paper.** Fallback restores full-context quality,
so the system has a **correctness floor = full-context accuracy** *by construction*. The
compressor no longer has to be lossless; it only has to be cheap-on-average. Therefore the
**entire** cost/quality trade-off is set by one thing: **how reliably the gate predicts that
compression will lose this particular answer.** Everything reduces to *signal selection*.

That reframing is what we sell. Not "a better compressor" (crowded, see §6), not "a gate
exists" (also taken, see §6) — but a **rigorous, broad, mechanism-grounded study of which
query-time signal is the reliable one**, with a definition of "reliable" that survives
cross-domain / cross-task / cross-dataset shift.

---

## 2 · Formal setup & labels

For base *b* = (model, dataset) and item *i* with context *c_i*, query *q_i*, gold *a_i*:
- **full_correct** *f_i* = 1 if the model answers correctly from the **full** context.
- **comp_correct** *g_i* = 1 if it answers correctly from the **compressed** memory.
- **Recoverable pool** = {*i* : *f_i* = 1}. Fallback can only ever help here (if full is also
  wrong, re-reading does not save it).
- **Gate target** *Y_i* = 1 iff *f_i* = 1 ∧ *g_i* = 0  (**compression-induced failure** — the
  gate *must* fire). On the recoverable pool this is just *Y_i = ¬g_i*.

A **gating signal** is any cheap function *s_i = σ(M_i, q_i)* computable **without** *c_i*.
A **gate** is (σ, τ): fire fallback iff *s_i > τ*. Cost model: fraction of queries that fall
back = β; expected cost ≈ β·cost_full + (1−β)·cost_comp; quality = full-context quality on
fired queries + comp quality on the rest.

**Compressor in this study (training-free proxy).** Model-generated extractive summary to
*k* tokens ("keep all names/numbers/dates"), amortized over many queries on the same context
(the agent-memory / repeated-query regime). It reproduces the documented gist-loss failure
modes (lost-by-boundary / lost-if-surprise / lost-along-the-way) so the gate has a real signal
to predict. The trained latent wrapper (Activation-Beacon / ICAE style) is the **follow-up**;
signal rankings are expected to transfer because the failure modes are compressor-agnostic
(to be verified — see ablations).

---

## 3 · Definition of a **GOOD** gating signal  *(the thing the user asked to nail down)*

A signal is graded on six axes; "good" = passes the bar on **all** of D1–D5 and is competitive
on D6. Each axis has an explicit metric and threshold so "good" is falsifiable.

| ID | Axis | Metric | "good" bar |
|----|------|--------|-----------|
| **D1** | **In-domain predictiveness** (black-box, train+test same base) | AUROC and AUPRC for predicting *Y* on held-out items; **recall@β** (failures caught at fallback budget β=20%) | AUROC ≥ **0.70**; recall@20% ≥ **0.6** |
| **D2** | **Statistical significance** | point-biserial / Spearman(s, Y) with **bootstrap 95% CI**; **BH-FDR** across signals×bases | CI excludes 0; q < 0.05 |
| **D3** | **Universality across bases** | **coverage@0.7** = frac. of bases with AUROC≥0.7 **and consistent sign**; report median AUROC + IQR + worst base | coverage ≥ **0.8**, sign 100% consistent |
| **D4** | **Cross-X generalization** (calibrate τ on A, test on B) | AUROC(B) and recall@β(B) using τ_A; **ECE** under shift; report Δ vs in-domain for **cross-domain, cross-task, cross-dataset, (cross-model)** | AUROC drop ≤ **0.05**; ECE ≤ 0.1 |
| **D5** | **Cheapness / availability** | extra FLOPs & latency to compute *s* vs cost saved; must NOT need *c_i* | overhead ≤ **10%** of full-context cost |
| **D6** | **System dominance** | cost–quality Pareto vs baselines (always-comp / always-full / random / entropy-thresh / LLM-judge); **quality@β** and **β@quality=95%/99%** | Pareto-dominates random & entropy-thresh; ≥ LLM-judge at lower cost |

**Plain-language "good":** a good signal (1) tells failures from successes inside a base
(D1), (2) for real, not by chance (D2), (3) on *almost every* model×dataset with the *same
direction* (D3), (4) with a threshold you can set on one slice and reuse on another — domain,
task, dataset, ideally model — without recalibration (D4), (5) for nearly free (D5), and (6)
yields a system that beats the obvious gates (D6).

**Composite ranking score** (for the headline table, all sub-scores in [0,1]):
`G = 0.30·medAUROC + 0.25·coverage@0.7 + 0.20·(1 − crossX_drop) + 0.15·recall@20% + 0.10·(1 − cost_overhead)`.
We report the full scorecard *and* G; G only orders ties.

---

## 4 · Signal catalogue (candidates) — ≥ 12 angles

Query-time signals (implemented in `gate_study.py`, prefix `sig_`); each is a distinct *angle*
on "is the compressed memory enough for this query?"

| # | angle | signal(s) | intuition (high ⇒ likely failure?) |
|---|-------|-----------|-----------|
| 1 | generation confidence | `ans_entropy`, `ans_margin`, `ans_ppl` | unsure answer ⇒ memory insufficient |
| 2 | self-consistency | `self_consistency` (greedy vs sampled) | unstable answer ⇒ guessing |
| 3 | memory counterfactual | `kl_mem_vs_q` (P(ans\|M,q) ‖ P(ans\|q)) | memory didn't move the answer ⇒ unused/missing |
| 4 | query difficulty | `q_len`, `q_has_number`, `q_n_entities` | numeric/entity/long queries ⇒ verbatim need |
| 5 | summary↔query coverage | `summary_query_overlap` | summary lacks query terms ⇒ info dropped |
| 6 | summary grounding | `summary_context_overlap` | hallucinated summary ⇒ unreliable |
| 7 | compression load | `compression_ratio` | more squeezed ⇒ more loss |
| 8 | attention routing | `attn_to_mem` (last layer) | gen ignores memory ⇒ memory not used |
| 9 | **retrieval-head routing** *(Janus, planned)* | `rh_mem_score` | retrieval heads not finding evidence in M |
| 10 | **sink displacement** *(Janus, planned)* | `sink_mass_gen` | attention dumped to sink ⇒ no real evidence |
| 11 | **reconstruction residual** *(planned)* | `recon_gap` (NLL of c given M) | summary can't regenerate context ⇒ lossy |
| 12 | answer degeneracy | `answer_len` | empty/runaway answer ⇒ failure |

**Expanding the pool to "量大".** Two extensions give ~50 candidates without new passes:
(a) every signal in angles 8–11 has a **per-layer** variant → multi-layer profile (also the
wrapper-design ablation, §5); (b) the **39 intrinsic metrics** (`metrics-reference.md`) computed
on the *compressed* forward are all admissible gating signals. We pool all of them and let
D1–D6 decide — this is the "再扩大测一次相关性" the user asked for, now with a *purpose*
(predict *Y*) rather than generic correlation.

---

## 5 · Main tables & ablations

**Main Table 1 — signal scorecard** (the headline). Rows = signals; cols = the D1–D6 metrics
(medAUROC, coverage@0.7, in-domain recall@20%, cross-domain/task/dataset AUROC-drop, ECE,
cost-overhead, composite G). Bold the winner; this *is* the paper's claim.

**Main Table 2 — system Pareto.** Rows = gates (best-signal gate, random, always-comp,
always-full, entropy-threshold [Entropy-Gate], LLM-judge [D-Mem-style], oracle gate). Cols =
quality@β∈{0,10,20,40%}, β@quality=95%/99%, avg tokens, latency. Shows the best signal turns
into a system that dominates.

**Ablations (each isolates one design axis):**
1. **Where knowledge + gate plug in (multi-layer wrapper design).** Input-only soft prompt vs
   per-layer KV injection (Activation-Beacon style) × gate reading from {last / mid / per-layer
   pooled} attention. Which layer's routing signal (angle 8–11 per-layer) is most predictive?
2. **Generalization matrices.** 3 shift types, each a train→test grid:
   - cross-**domain** (wiki / books / web / synthetic / gutenberg),
   - cross-**task** (extractive / multihop / narrative / mcq / reasoning / passage / niah),
   - cross-**dataset** (squad↔hotpot↔trivia↔marco within QA),
   - cross-**model** (Qwen3 0.6→14B, Qwen3.5, GLM-4 9B/32B) — bonus axis for D4.
3. **Signal selection / combination.** Single best vs small logistic combo (2–3 signals);
   does a cheap combo beat the best single without overfitting (nested CV)?
4. **Compression ratio sweep** (`summary_tokens` ∈ {32,64,96,160}) — does the winning signal
   stay winning as compression hardens (where the gate matters most)?
5. **Compressor transfer.** Summary-proxy → trained latent wrapper: do signal *rankings* hold?
6. **Calibration method.** raw-threshold vs Platt vs conformal — which gives D4 transfer + ECE.

---

## 6 · Novelty position (from the 2026 literature audit)

| prior work | what it has | what it is missing (our seam) |
|---|---|---|
| **D-Mem** (2603.18631, agent memory, LoCoMo) | quality gate (LLM-judge: relevance/faithful/complete) + **full-deliberation fallback** | gate is an **expensive LLM judge**, prompt-level, **per-method heuristic**; no signal study, no cross-X calibration, no cheap intrinsic signal |
| **Entropy-Gate** (2606.03739) | "graceful degradation": if below fidelity thresh, forward full prompt | **single heuristic** (entropy), prompt-level, no comparison of signals, no shift study |
| **SLT / CoLaR** (2605.25745) | latent compression + **confidence gate → fall back to explicit** | for **CoT reasoning spans**, not input/memory context; one signal (confidence) |
| **AdaComp** (2409.01579) | learned predictor of compression **rate** | extractive RAG, predicts *rate* not *failure*; no fallback floor; no cross-X |
| **RazorAttention / CompressKV** | retrieval-head-guided KV keep/evict | **architecture**, not a *failure predictor*; no calibrated gate, no fallback semantics |

**Our differentiators (the defensible contribution):** (i) treat the **gate signal as the
object of study**, not the architecture; (ii) a **falsifiable definition of "reliable"** with
explicit cross-domain/task/dataset(/model) transfer + calibration; (iii) **cheap intrinsic
signals** (confidence / counterfactual-KL / retrieval-head routing) benchmarked head-to-head
against the expensive LLM-judge of D-Mem; (iv) **breadth** — N models × M datasets × the full
signal pool, the kind of universality evidence none of the above provide. Risk: incremental vs
D-Mem unless (ii)+(iii)+(iv) clearly land. Mitigation: the universality + cheap-signal + shift
results are exactly what a measurement paper can own.

---

## 7 · Experimental scope ("量大、广泛、针对性")

- **Models (8, scale + family):** Qwen3-{0.6,1.7,4,8,14}B, Qwen3.5-{4,9}B, GLM-4-9B. (GLM-4-32B
  planned but not cached offline on ray → dropped.) 8 × 8 = **64 cells**.
- **Datasets (8, cross-task + cross-domain):** squad (extractive/wiki), hotpotqa (multihop/wiki),
  narrativeqa (narrative/books), triviaqa (trivia), msmarco (passage/web), quality (mcq/gutenberg),
  musr (reasoning-mcq/synthetic), niah (verbatim/synthetic — the canonical failure case).
- **n** = 60 items/base (→ enough *Y*=1 to estimate AUROC; bump failing-rich bases to 120).
- **Signals:** 13 now + 4 planned (angles 9–12) + 39 intrinsic (pooled) ≈ 50 candidates.
- **Compute:** ray 4×H100, `orch_gate.sh` (56 model×dataset jobs queued). sam-dev on v1 (leave).

**Deliverables:** Tables 1–2 + the 6 ablations + the scorecard CSV; figs = per-signal AUROC
violin across bases, cross-X transfer heatmaps, system Pareto, per-layer routing curves.
