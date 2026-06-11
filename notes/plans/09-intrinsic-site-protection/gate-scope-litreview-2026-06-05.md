# Scope, literature review & novelty defense — Paper B (parametric agentic memory)

> Created 2026-06-05. Sets the **scope**, does the **literature review**, and mounts the
> **novelty defense** for the gated-compression line, now scoped (per directive) to
> **parametric / latent-space agentic memory**. Pairs with
> [`gate-pipeline-2026-06-05.md`](gate-pipeline-2026-06-05.md) and the ICLR self-review.

---

## 1 · Scope (chosen: appropriate + popular + defensible)

**One sentence.** *Reliability-gated parametric memory for LLM agents: an agent stores its
long-term memory as compressed latent vectors integrated with the model, and a cheap, calibrated
gate decides per query whether that latent memory suffices — falling back to the full raw context
when it does not.*

**Where this sits in the field's taxonomy** (using the two 2026 surveys —
[From Human Memory to AI Memory, 2504.15965] and [Memory for Autonomous LLM Agents, 2603.07670],
whose axes are *substrate × temporal scope × control policy*):
- **Substrate = parametric / latent-space** (memory lives in the model's activation/parameter
  space: memory tokens, soft prompts, compressed KV) — *not* non-parametric text/vector stores,
  and *not* weight-level fine-tuning. This is the MemoryLLM / Memory³ / ICAE / Activation-Beacon
  family.
- **Temporal scope = long-term** (cross-turn / cross-session agent memory).
- **Control policy = read-path gating** (a learned decision of *when the latent memory is enough*),
  the "policy-learned management" mechanism family.

**Why this scope is popular & timely.** Agent memory is one of the hottest 2026 areas
(Mem0/Letta/Zep ecosystems; surveys 2504.15965, 2603.07670; benchmarks LoCoMo, LongMemEval,
BEAM). Within it, **latent/parametric memory is the rising research substrate** (MemoryLLM 2024,
M+ ICML'25, Memory³ 2024, Memory Layers) because it promises RAG-free, low-latency, on-device,
externally-DB-free memory. It is *under-served on reliability*: latent memory **forgets silently**
(MemoryLLM has an explicit exponential forgetting curve; HyMem notes "compression inevitably leads
to permanent information loss"). That gap — *no one tells you when the latent memory has dropped
the fact you need* — is exactly what we own.

**In scope.** Latent-memory wrappers (compress context→slots); the read-path gate; fallback to
full context; the *signal* that drives the gate; cost/quality under shift.
**Out of scope.** Building a better compressor (Activation-Beacon et al. already do this); weight
editing / continual fine-tuning; non-parametric vector/graph memory engineering; the *write* path
(what to store) — we take a wrapper off the shelf and study the *read-time reliability gate*.

**Problem we own (the gap).** Parametric/latent memory trades fidelity for speed and then fails
silently on verbatim/numeric/entity queries ([Silver Bullet 2412.17483]: gist compression is
near-lossless on RAG/QA but fails on synthetic recall; [xRAG]: loses dates/numbers/entities). In
an *agent* this is dangerous: a wrong recalled fact propagates. A correctness floor via fallback
fixes it — **iff** the gate is reliable. So: *which gate signal is reliable for latent memory?*

---

## 2 · Literature review (organized by the taxonomy)

### A. Parametric / latent-space memory — our substrate (nearest in-scope)
- **MemoryLLM** (Wang+ 2024): interleaves a 1B latent **memory-token pool** across layers,
  self-updates, **exponential forgetting**; effective ≤16–20k tokens.
- **M+** (ICML 2025, 2502.00592): adds CPU **long-term latent store** + **co-trained retriever**
  fetching latent states once/layer; retention 20k→160k. *Closest prior work.*
- **Memory³** (2407.01178): "explicit memory" = sparse KV pairs retrieved into attention; a 3rd
  memory form between weights (implicit) and context-KV (working). RAG-free, faster decode.
- **Memory Layers** (Meta 2024): trainable key–value memory layers (parametric).
- **Context compression as latent memory** (the wrapper we'd use): Gisting (2304.08467),
  ICAE (2307.06945), AutoCompressor, **Activation-Beacon** (2401.03462, 2× speed/8× KV, ~lossless
  to 128k incl. NIAH), UltraGist (2405.16635), 500×Compressor, CCM (online chat KV).
- **Take:** this family compresses into latents and *retrieves more latents*; **none has a
  fallback to the full raw context, and none studies a failure-prediction gate.** They forget
  silently.

### B. Non-parametric agentic memory — the popular practice (contrast)
- **MemGPT** (2310.08560) OS-style paging; **Mem0** (2504.19413, ECAI'25) hierarchical
  extraction+graph, the LoCoMo head-to-head baseline; **Letta** (editable core + archival
  blocks); **Zep** (temporal knowledge graph); **A-Mem**, **MemoryBank**, **Cognee**.
- **Take:** memory = **text/vectors/graphs**, retrieved into the prompt. Inspectable & updatable,
  but adds retrieval latency and lives outside the model. Different substrate from us.

### C. Gated / adaptive memory control — the contested ground (the crowded part)
- **D-Mem** (2603.18631): **Quality-Gating** (LLM-judge: relevance/faithfulness/completeness) →
  **Full-Deliberation fallback** reads raw history; on **LoCoMo**, gate recovers 96.7% of full at
  35.8% tokens. *Non-parametric; expensive LLM-judge; one heuristic.*
- **Oblivion** (2604.00131): read/write-decoupled; **uncertainty-gated read path** triggers
  retrieval only when context insufficient. *Non-parametric; one uncertainty signal.*
- **RF-Mem** (2603.09250): Familiarity (mean score + **list entropy**) → one-shot top-K, else
  **Recollection** expansion; beats one-shot and full-context. *Non-parametric; list-entropy gate.*
- **HyMem** (2602.13933): two tiers (summaries / raw text); **query-complexity** picks granularity.
  *Non-parametric; complexity gate.*
- **AdaComp** (2409.01579): predicts compression **rate** (RAG, extractive). **Entropy-Gate**
  (2606.03739): below-fidelity → forward full prompt (prompt-level heuristic). **SLT/CoLaR**
  (2605.25745): confidence gate → fall back to explicit **CoT** (reasoning, not memory).
- **Uncertainty gating / selective prediction** (Han+ 2024; long line since 2017): escalate to
  tool/human when uncertain. The generic prior we must beat with *latent-specific* signals.
- **Take:** gated escalation/fallback with *a* signal is now common — but (i) **all in the
  non-parametric substrate** (gate on retrieved *text*), and (ii) **each picks one signal**
  (LLM-judge / uncertainty / list-entropy / complexity); none does a systematic signal-selection
  study with cross-shift transfer + calibration.

### D. Why compression fails (motivation science)
- **Silver Bullet** (2412.17483): gist compression near-lossless on RAG/QA/summarization, **fails
  on synthetic recall/reranking**; failure modes *lost-by-boundary / lost-if-surprise /
  lost-along-the-way*. **xRAG**: single-token compression keeps topic, drops dates/numbers/entities.

### E. Mechanism → latent-specific gate signals (our edge)
- **RazorAttention** (ICLR'25, 2407.15891) & **CompressKV** (2508.02401): **retrieval heads**
  decide which tokens to keep under compression. → We repurpose retrieval-head *routing onto the
  latent slots* as a gate signal — only available in the parametric substrate.
- **Janus intrinsic metrics** (`metrics-reference.md`): 39 metrics computable on the compressed
  forward → an extra gate-signal pool a text-RAG gate cannot use.

### F. Benchmarks for the agentic scope
- **LoCoMo** (multi-session conversational; D-Mem/Mem0 turf — we have it integrated),
  **LongMemEval** (knowledge-update/temporal/multi-session), **BEAM** (scalability). We add LoCoMo
  for the head-to-head; QA suite (squad/hotpot/narrativeqa/quality/musr/niah/msmarco/triviaqa)
  for the broad signal study.

---

## 3 · Novelty defense

### 3.1 The three claims we defend
- **N1 — substrate.** First to bring **gated fallback-to-full-context** to **parametric/latent**
  memory. Every gated-memory system (D-Mem, Oblivion, RF-Mem, HyMem) is non-parametric; every
  latent-memory model (MemoryLLM, M+, Memory³) lacks a fallback/reliability gate. We fill the
  empty cell: *latent memory × read-path reliability gate × full-context fallback*.
- **N2 — systematic signal study.** First to treat the **gate signal as the object of study** —
  a falsifiable "good signal" definition (D1–D6) and a head-to-head of ~50 signals across
  8 models × 8 datasets with **cross-domain/task/dataset transfer + calibration**, turning the
  field's one-off heuristics (LLM-judge / uncertainty / list-entropy / complexity) into a benchmark
  with a winner and a transfer guarantee.
- **N3 — latent-specific signals.** New signals only available in the parametric substrate —
  **counterfactual-KL** (latent-memory vs no-memory), **retrieval-head routing onto latent slots**,
  **latent reconstruction residual** — shown to beat the generic uncertainty/LLM-judge gates **at a
  fraction of the cost**.

### 3.1b EMPIRICAL STATUS (2026-06-05, clean scorecard, 13 bases / 663 items) — read this
- **Thesis SUPPORTED:** a reliable, universal, transferable cheap gate exists.
  `ans_entropy` predicts compression failure with **median AUROC 0.88, coverage@0.7 = 1.0,
  sign-consistent, LODO 0.87 / LOMO 0.87** (ppl & margin nearly identical). So D1/D3/D4/D5 pass.
- **N3 AT RISK:** the winner is **generation confidence (entropy/ppl/margin)** — i.e.
  selective-prediction-adjacent — exactly R2's "folklore" worry. The *latent-specific* signals as
  currently implemented (`kl_mem_vs_q` first-token proxy, `attn_to_mem` last-layer summary span)
  are **at the bottom** (AUROC ~0.60). N3 is **not yet supported**.
- **But N3 is not yet fairly tested:** (a) the real signal (retrieval-head routing onto memory
  *slots*) is not implemented; (b) we're on a **summary-text** proxy, where "attention to memory"
  is weak — a true latent wrapper exposes real slots; (c) `kl_mem_vs_q` should be over the answer
  span, not first token. → Implement these and re-test before accepting/rejecting N3.
- **Fallback if N3 dies:** novelty rests on **N1 (parametric substrate) + N2 (systematic study +
  conformal guarantee)**, honestly reporting *confidence is the best gate and here is its calibrated
  cross-shift transfer for latent memory* — still a paper, weaker but defensible. Caveat to address:
  entropy on the compressed run conflates question difficulty with compression loss (we restrict to
  the full-correct pool, but Δ-style signals need full context → unavailable at gate time).

### 3.2 Threat-by-threat
| nearest threat | what it does | our differentiator |
|---|---|---|
| **M+ / MemoryLLM** | latent memory + retrieve *more latents* | no fallback to full raw context; no failure gate; **silent exponential forgetting** — we add the reliability layer they lack |
| **Memory³** | explicit sparse-KV memory into attention | architecture, always-on; no per-query sufficiency gate / fallback |
| **D-Mem** | LLM-judge gate + full-deliberation fallback | **non-parametric** (vector text); gate is an **expensive LLM judge**; single heuristic; no signal study / no guarantee |
| **Oblivion** | uncertainty-gated read path | non-parametric; one uncertainty signal; no cross-shift calibration; not latent |
| **RF-Mem** | familiarity (list entropy) → recollection | non-parametric retrieval list; can't use latent signals; single signal |
| **HyMem** | query-complexity picks summary vs raw | non-parametric tiers; coarse single signal |
| **Activation-Beacon/ICAE** | the compressor itself | inference-time, not agentic memory; no gate/fallback; we *use* one as the wrapper |
| **SLT/CoLaR** | confidence gate → explicit CoT | reasoning chains, not memory; one signal |

### 3.3 Anticipated reviewer attacks → rebuttals
- *"Gated memory escalation is well-trodden (Oblivion/RF-Mem/D-Mem)."* → All **non-parametric**;
  the parametric substrate is where compression (hence failure) and the speedup actually live, and
  enables **latent-specific signals** they cannot use. Plus we provide the **first systematic
  signal study + guarantee**, not another one-off heuristic. (N1+N2+N3.)
- *"Parametric vs non-parametric is a mere substrate swap."* → It changes the signal space (N3),
  the failure modes, and the deployment story (no external DB, lower latency); and latent memory
  models currently have **no** reliability mechanism — a real, unfilled gap.
- *"Isn't this just selective prediction / confidence calibration?"* → We *include* confidence as a
  baseline and show **counterfactual-KL / retrieval-head routing beat it**; the contribution is
  *which* signal transfers across shift for *latent memory*, with calibration — not "use entropy."
- *"Why not always use M+'s retriever?"* → Retrieving more latents doesn't recover info the
  compressor *destroyed*; only fallback to the raw context can. We quantify exactly when.
- *"Novelty is incremental."* → Honest risk. We de-risk with (a) **LoCoMo head-to-head vs D-Mem**
  at lower cost, (b) a **conformal fallback guarantee** under shift, (c) latent-signal wins —
  any one is a contribution; together they clear the bar (see ICLR self-review trajectory).

### 3.4 What we must run to make the defense real (ties to the ICLR review's must-haves)
1. Instantiate on a **real latent wrapper** (Activation-Beacon / ICAE) — not only the summary proxy.
2. **LoCoMo** (and ideally LongMemEval) for the agentic head-to-head **vs D-Mem / Mem0 / M+**.
3. Show **latent-specific signals (counterfactual-KL, retrieval-head routing) beat** LLM-judge &
   entropy on the D1–D6 scorecard, at ≥10× lower gate cost.
4. **Conformal fallback guarantee** + cross-domain/task/dataset transfer.
5. End-to-end **latency/throughput** (the parametric-memory selling point).

---

## 4 · Intro positioning paragraph (drop-in)
> LLM agents increasingly store long-term memory in *parametric, latent* form — compressed memory
> tokens or KV integrated with the model (MemoryLLM, M+, Memory³) — trading fidelity for RAG-free,
> low-latency recall. But latent memory forgets silently: it drops the exact names, numbers, and
> dates that compression cannot preserve, and unlike non-parametric stores it offers no signal that
> this has happened. We argue that the right fix is not a better compressor but a **reliable gate**:
> if the agent can tell when its latent memory is insufficient and fall back to the full context,
> the system inherits a correctness floor and the entire problem reduces to gate reliability. We
> therefore study the **gating signal** itself — defining reliability via in-domain accuracy,
> cross-domain/task/dataset transfer, calibration, and cost — and benchmark ~50 signals across
> 8 models and 8 datasets. We find that latent-substrate-specific signals (counterfactual-KL and
> retrieval-head routing onto memory slots) predict compression failure more reliably and far more
> cheaply than the LLM-judge and uncertainty gates used by current agent-memory systems, and that a
> conformally-calibrated threshold transfers across distribution shift — turning silently-forgetting
> latent memory into memory that knows when to fall back.
>
> *(Empirical status, see §3.1b: a cheap generation-confidence signal already delivers this
> reliably and universally; whether the latent-substrate-specific signals beat it is under test —
> do not state the latent-signal win as settled until N3 lands.)*
