# Paper A literature review — compression failure, latent memory, and long-context reliability

> Cutoff: 2026-07-16. Sources were checked against Google Scholar search, ACL Anthology, PMLR/ICML,
> NeurIPS proceedings, OpenReview, arXiv, and publisher pages. Archival venue records take precedence over
> aggregation pages. Preprint-only work is labeled.

## 0. Executive verdict

The broad story “compress context, estimate reliability, and fall back when compression is unsafe” is
**not new in 2026**. Direct prior work already covers:

- query-specific failure detection for soft compressed tokens;
- input-adaptive compression ratios and soft-token budgets;
- first-token confidence gates over latent memories;
- summary-to-raw fallback for agents;
- validation and repair of agent compaction;
- finite-sample selective routing.

The defensible paper is narrower:

> **Execute and evaluate a paired route between a recurrent learned soft memory and a specified feasible
> raw-evidence path, using one frozen backbone, with finite-sample positive-harm control and full
> accuracy–coverage–cost accounting across model families and heterogeneous tasks.**

This exact intersection is not clearly occupied, but the current empirical thresholding does not yet
establish the finite-sample guarantee.

## 1. Closest direct precedents

### 1.1 Detecting Overflow in Compressed Token Representations for Retrieval-Augmented Generation

- **Authors:** Julia Belikova, Danila Rozhevskii, Dennis Svirin, Konstantin Polev, Alexander Panchenko.
- **Venue:** EACL 2026 Student Research Workshop.
- **Links:** [ACL Anthology](https://aclanthology.org/2026.eacl-srw.59/) ·
  [arXiv:2602.12235](https://arxiv.org/abs/2602.12235).
- **Method:** defines compression overflow as the reference path succeeding while xRAG soft compression
  fails; trains query–compressed-context probes.
- **Result:** approximately 0.70–0.73 ROC-AUC on TriviaQA, SQuAD-v2, and HotpotQA; query-agnostic statistics
  are near random.
- **Overlap:** directly preempts “first compression-failure detector,” “first query-aware soft-token
  reliability signal,” and the motivation that compressed representations should be gated.
- **Remaining gap:** one xRAG-7B setting, three datasets used in compressor training, cross-validation but
  no executed fallback policy, no finite-sample risk control, and no broad cost/competence map.

### 1.2 Context Distillation as Latent Memory Management

- **Authors:** Ziyang Zheng, Zeju Li, Xiangyu Wen, Jianyuan Zhong, Junhua Huang, Lei Chen, Mingxuan Yuan,
  Qiang Xu.
- **Status:** 2026 preprint.
- **Link:** [arXiv:2605.28889](https://arxiv.org/abs/2605.28889).
- **Method:** distills documents into independent LoRA memories; first-token entropy decides whether to
  keep the latent adapter active or deactivate it and use the frozen base.
- **Overlap:** preempts first-token uncertainty gating, latent-memory activation, and adapter-toggle fallback.
- **Difference:** detects whether memory is relevant, not specifically whether a lossy representation omitted
  query-critical evidence; fallback is the no-context base rather than raw evidence.

### 1.3 PoC: Performance-oriented Context Compression via Performance Prediction

- **Authors:** Runsong Zhao et al.
- **Status:** 2026 preprint.
- **Link:** [arXiv:2603.19733](https://arxiv.org/abs/2603.19733).
- **Method:** predicts a sample-specific performance–compression curve and chooses the strongest LLMLingua-2
  ratio satisfying a requested performance floor.
- **Overlap:** directly occupies input-dependent, performance-constrained compression and adaptive ratios.
- **Difference:** predicts a floor rather than selecting between recurrent soft memory and raw evidence; no
  finite-sample guarantee.

### 1.4 Selective Latent Thinking

- **Authors:** Hui Xie, Jie Liu, Ziyue Qiao, Joaquin Vanschoren.
- **Status:** 2026 preprint.
- **Link:** [arXiv:2605.25745](https://arxiv.org/abs/2605.25745).
- **Method:** confidence-gates compressed latent reasoning spans and falls back to explicit chain-of-thought.
- **Overlap:** “trust latent execution when confident; otherwise use an explicit path.”
- **Difference:** compresses generated reasoning rather than input context; math-only evaluation and no
  calibrated selective-risk treatment.

### 1.5 TierMem / verified summary-to-raw fallback

- **Title:** *From Lossy to Verified: A Provenance-Aware Tiered Memory for Agents*.
- **Authors:** Qiming Zhu, Shunian Chen, Rui Yu, Zhehao Wu, Benyou Wang.
- **Venue:** ICLR 2026 MemAgents Workshop.
- **Links:** [arXiv:2602.17913](https://arxiv.org/abs/2602.17913) ·
  [OpenReview](https://openreview.net/forum?id=dJgeY3Awrv).
- **Method:** labels summary-only failures that raw evidence repairs, trains a cost-aware sufficiency router,
  escalates to provenance-linked raw logs, and can write verified findings back.
- **Overlap:** strongly preempts the broad “compression-failure signal as an agent control plane” framing.
- **Difference:** textual summaries with provenance, not learned continuous memory inside one frozen backbone.

### 1.6 Formal selective routing

- **Wynn et al., “Controlling the Risk of Corrupted Contexts for Language Models via Early-Exiting.”**
  ICML 2026. [arXiv:2510.02480](https://arxiv.org/abs/2510.02480).
- **Wang et al., “A Linear Expectation Constraint for Selective Prediction and Routing.”**
  ICML 2026. [ICML page](https://icml.cc/virtual/2026/poster/63016).
- **Learn then Test.** Angelopoulos et al., *Annals of Applied Statistics* 2025.
  [arXiv:2110.01052](https://arxiv.org/abs/2110.01052).
- **Conformal Risk Control.** Angelopoulos, Bates, Fisch, Lei, Schuster, ICLR 2024.
  [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html).

Generic “risk-controlled routing” is therefore not novel. Paper A must define a compression-specific harm
event and implement a valid finite-sample procedure.

## 2. Latent context-compression lineage

| work | venue | mechanism | key relevance / limitation |
|---|---|---|---|
| Recurrent Memory Transformer — Bulatov et al. | NeurIPS 2022 | recurrent memory tokens across segments | recurrence ancestor; fixed capacity and BPTT cost |
| Gist Tokens — Mu, Li, Goodman | NeurIPS 2023 | gist attention mask and tuned LM activations | up to 26× prompt compression; requires model tuning |
| AutoCompressor — Chevalier et al. | EMNLP 2023 | recursively accumulated soft summaries | closest recurrent-summary ancestor; often below longer raw passages |
| ICAE — Ge et al. | ICLR 2024 | LoRA encoder, frozen decoder, autoencoding/LM/instruction objectives | close frozen-decoder design; above 4× becomes difficult |
| CCM — Kim et al. | ICLR 2024 | conditional-LoRA recurrent layerwise KV memory | fixed-size merging loses diverse information |
| xRAG — Cheng et al. | NeurIPS 2024 | retriever embedding projected into one LLM token | extreme compression; overflow motivates Belikova et al. |
| Activation Beacon — Zhang et al. | ICLR 2025 | progressive activation condensation | strong speed/memory result; model-specific training, no safety decision |
| 500xCompressor — Li, Su, Collier | ACL 2025 | frozen decoder + LoRA encoder + layerwise KV | materially lossy at extreme ratios |
| Cramming 1568 — Kuratov et al. | ACL 2025 | per-text optimization into one input vector | capacity upper bound, not an online compressor |
| Cartridges — Eyuboglu et al. | ICLR 2026 | corpus-specific optimized KV prefix | strong reuse efficiency but expensive offline self-study |
| End-to-End Context Compression at Scale — Li et al. | 2026 manuscript | 0.6B encoder + 4B decoder, pooled latent windows | huge training budget; decoder not frozen; selective source expansion |
| Semi-Dynamic Context Compression — Jiong et al. | 2026 preprint | mean-pooled latent tokens with discrete 2–32× ratio selection | latest adaptive soft-token baseline; released training range is short |

Verified links:

- [RMT](https://arxiv.org/abs/2207.06881)
- [Gist Tokens](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3d77c6dcc7f143aa2154e7f4d5e22d68-Abstract.html)
- [AutoCompressor](https://aclanthology.org/2023.emnlp-main.232/)
- [ICAE](https://openreview.net/forum?id=uREj4ZuGJE)
- [CCM](https://openreview.net/forum?id=64kSvC4iPg)
- [xRAG](https://openreview.net/forum?id=6pTlXqrO0p)
- [Activation Beacon](https://openreview.net/forum?id=45zwTmts0G)
- [500xCompressor](https://aclanthology.org/2025.acl-long.1219/)
- [Cramming 1568](https://aclanthology.org/2025.acl-long.948/)
- [Cartridges](https://openreview.net/forum?id=0k5w8O0SNg)
- [LCLM](https://arxiv.org/abs/2606.09659)
- [Semi-Dynamic Context Compression](https://arxiv.org/abs/2603.25926)

### New baselines that matter

- **No Mean Feat: Simple, Strong Baselines for Context Compression**
  ([arXiv:2510.20797](https://arxiv.org/abs/2510.20797)): mean pooling and bidirectional compression tokens
  challenge standard causal compression. Add a mean-pooling baseline.
- **Frozen LLMs are Native Decoders for High-Norm Semantic Vectors**
  ([ACL 2026](https://aclanthology.org/2026.acl-long.1717/)): high-norm latent vectors can be two orders
  larger than ordinary embeddings; directly challenges GCM's hard match to average embedding norm.
- **Bridging the Memorization-Utilization Gap**
  ([ACL 2026](https://aclanthology.org/2026.acl-long.682/)): outcome-based RL plus pooling is a strong accuracy
  baseline but trains substantial model capacity.
- **IndexMem** ([ICML 2026](https://icml.cc/virtual/2026/poster/63943)): online latent compression of evicted
  KV entries with gated residual readout.

## 3. Adaptive compression and augmentation gates

The following mechanisms are established and should be cited as precedents, not presented as Paper A
novelties:

- **FLARE** — low-confidence generation triggers retrieval and regeneration.
  [EMNLP 2023](https://aclanthology.org/2023.emnlp-main.495/)
- **Self-RAG** — learned retrieval and critique tokens.
  [ICLR 2024](https://openreview.net/forum?id=hSyW5go0v8)
- **Adaptive-RAG** — classifier chooses no/single/iterative retrieval.
  [NAACL 2024](https://aclanthology.org/2024.naacl-long.389/)
- **Self-Route** — retrieved chunks when sufficient, full context otherwise.
  [EMNLP Industry 2024](https://aclanthology.org/2024.emnlp-industry.66/)
- **SeaKR** — hidden-state uncertainty triggers and reranks retrieval.
  [ACL 2025](https://aclanthology.org/2025.acl-long.1312/)
- **ACC-RAG** — progressively supplies learned compressed embeddings and learns a stopping selector.
  [Findings EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.1307/)
- **TARG / Retrieval as a Decision** — query-only entropy, margin, and sample variance.
  [arXiv:2511.09803](https://arxiv.org/abs/2511.09803)
- **QGC, AdaComp, Provence, DAST, AttnComp, Compactor, COMI, CompilerKV, SARA, RAM, ATACompressor,
  CONF-KV** — adaptive ratios, token budgets, or raw/latent allocation.

Consequences for claims:

- adaptive \(K\), adaptive compression ratio, and raw-span retention are not new;
- first-token confidence and margin are not new gate signals;
- “one signal as an agent control plane” is an implication, not a demonstrated contribution, unless the
  resulting policies are implemented and evaluated.

## 4. Agent-memory and long-horizon context management

### Canonical lineage

- **Generative Agents** — Park et al., UIST 2023.
  [DOI](https://doi.org/10.1145/3586183.3606763)
- **MemGPT** — Packer et al., 2023 preprint.
  [arXiv:2310.08560](https://arxiv.org/abs/2310.08560)
- **A-Mem** — Xu et al., NeurIPS 2025.
  [Proceedings](https://papers.neurips.cc/paper_files/paper/2025/hash/19909c36f51abc4856b4560aff3d36d6-Abstract-Conference.html)
- **AgeMem** — Yu et al., ACL 2026.
  [ACL](https://aclanthology.org/2026.acl-long.981/)
- **Memory-R1** — Yan et al., ACL 2026.
  [ACL](https://aclanthology.org/2026.acl-long.583/)
- **RecMem** — Dai et al., Findings ACL 2026.
  [ACL](https://aclanthology.org/2026.findings-acl.1619/)
- **Titans** — Behrouz et al., NeurIPS 2025.
  [arXiv:2501.00663](https://arxiv.org/abs/2501.00663)
- **ATLAS** — Behrouz et al., 2025 preprint/ICLR submission.
  [arXiv:2505.23735](https://arxiv.org/abs/2505.23735)

### Verification, consolidation, and fallback

Prior work already includes confidence- or evidence-triggered memory actions:

- TierMem: summary → linked raw pages → wider search.
- Slipstream: validates candidate compactions against continuation steps, repairs rejected summaries, then
  falls back.
- TrustMem: coverage/preservation/faithfulness verifier for memory transitions.
- Self-Compacting Language Model Agents: rubric-gated compaction tool.
- FlashMem: attention-entropy-triggered latent consolidation.
- D-Mem: fast vector memory → exhaustive raw-history deliberation.
- AgentAsk / VeriMAP: reliability checks and clarification at inter-agent handoffs.

The remaining differentiator is not “verified memory” or “reliability-aware handoff” in general. It is a
**continuous latent-memory-specific risk value tied to an executed raw recovery policy**, if Paper A actually
implements and evaluates it.

## 5. Why long context matters and fails

### Strong claims with archival support

- **Lost in the Middle** — Liu et al., TACL 2024.
  [ACL Anthology](https://aclanthology.org/2024.tacl-1.9/)
- **Context Length Alone Hurts LLM Performance Despite Perfect Retrieval** — Du et al., Findings EMNLP 2025.
  [ACL Anthology](https://aclanthology.org/2025.findings-emnlp.1264/)
- **Large Language Models Can Be Easily Distracted by Irrelevant Context** — Shi et al., ICML 2023.
  [PMLR](https://proceedings.mlr.press/v202/shi23a.html)
- **Long-Context LLMs Meet RAG** — Jin et al., ICLR 2025.
  [OpenReview](https://openreview.net/forum?id=oU3tpaR8fm)

Careful wording:

- lost-in-the-middle is real but task-dependent, not universal;
- “context rot” is descriptive, with Chroma 2025 as a technical report, not a settled mechanism;
- a fixed recurrent state is primarily an information-capacity/recall bottleneck, whereas the dense-attention
  KV cache is a serving-memory bottleneck;
- accepting a sequence is not equivalent to using it effectively.

### Cost citations

- FlashAttention — Dao et al., NeurIPS 2022.
  [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- PagedAttention — Kwon et al., SOSP 2023.
  [DOI](https://dl.acm.org/doi/10.1145/3600006.3613165)
- Sarathi-Serve — Agrawal et al., OSDI 2024.
  [USENIX](https://www.usenix.org/conference/osdi24/presentation/agrawal)

## 6. Benchmark audit and Paper A suite

### Verified benchmark references

| benchmark | archival source | role | caution |
|---|---|---|---|
| RULER | COLM 2024 | controlled length/retrieval/aggregation/QA | vanilla NIAH alone is weak |
| LongBench | ACL 2024 | broad moderate-length comparability | lexical shortcuts and noisy metrics |
| LongBench v2 | ACL 2025 | realistic hard long-context MC | 503 examples; length buckets confounded by category |
| ∞Bench | ACL 2024 | >100K heterogeneous stress | En.MC alone is narrow |
| NoLiMa | ICML 2025 | semantic retrieval without literal overlap | synthetic single-needle component |
| HELMET | ICLR 2025 | realistic controlled suite through 128K | some judge-based metrics |
| BABILong | NeurIPS 2024 D&B | recurrent-state/reasoning stress | filter tasks with weak short-context competence |
| LongMemEval | ICLR 2025 | persistent conversational/agent memory | requires a dedicated loader/protocol |

Links:

- [RULER](https://openreview.net/forum?id=kIoBbc76Sy)
- [LongBench](https://aclanthology.org/2024.acl-long.172/)
- [LongBench v2](https://aclanthology.org/2025.acl-long.183/)
- [∞Bench](https://aclanthology.org/2024.acl-long.814/)
- [NoLiMa](https://proceedings.mlr.press/v267/modarressi25a.html)
- [HELMET](https://openreview.net/forum?id=293V3bJbmE)
- [BABILong](https://openreview.net/forum?id=u7m2CG84BQ)
- [LongMemEval](https://openreview.net/forum?id=pZiyCaVuti)

### Recommended headline protocol

1. harder RULER tasks and positions at 4K/8K/16K/32K;
2. NoLiMa one-hop/two-hop subsets;
3. HELMET RAG at controlled lengths;
4. full LongBench-v2 on primary bases;
5. ∞Bench En.MC plus En.QA or Code.Debug;
6. BABILong QA1–QA3 for recurrent-state stress;
7. LongMemEvalS for the claimed persistent-agent relevance.

Current implementation status:

- included: RULER length sweep, LongBench-v2, ∞Bench-choice, LongBench-v1 transfer tasks, BABILong QA1–QA3;
- blocked: NoLiMa fetch currently fails without an authenticated/local dataset;
- missing loader: HELMET and LongMemEval;
- narrow: ∞Bench currently uses En.MC only and should add En.QA/Code.Debug on primary bases.

## 7. Statistical correction for the gate

The current policy-level signed excess loss is:

$$
R_{\mathrm{policy}}(\tau)
=
\mathbb{E}\left[
A_\tau(\ell_{\mathrm{mem}}-\ell_{\mathrm{raw}})
\right].
$$

This is insufficient because:

- memory wins can cancel compression-induced harms;
- low coverage mechanically reduces unconditional risk;
- a two-point unconditional allowance at 20% coverage can hide approximately ten points of accepted-set harm;
- selecting a threshold from empirical calibration loss is not a finite-sample guarantee.

Paper A must additionally report:

1. raw correct, memory wrong (compression harm);
2. memory correct, raw wrong (rescue);
3. both correct and both wrong;
4. conditional accepted-set positive regret;
5. policy-level signed performance;
6. coverage, latency, and expected cost.

For formal control, use Learn-then-Test or a suitable LEC/SCoRE procedure with a stated \(\delta\), upper
confidence bound, and raw fallback if no threshold is certified. Otherwise use the phrase
**held-out empirically calibrated routing**, not **risk-controlled**.

## 8. Novelty boundary

### Already known

- latent/soft-token context compression;
- recurrent memory tokens;
- frozen decoder with small memory components;
- reconstruction and downstream-task objectives;
- compression-failure probing;
- first-token confidence/margin gates;
- input-adaptive compression budgets;
- summary/latent-to-raw fallback;
- verified memory writes and consolidation;
- reliability-aware agent handoffs;
- risk–coverage and selective routing.

### Potentially defensible

1. **Paired positive-regret control specifically for recurrent soft context versus a specified feasible raw
   path.**
2. **The exact shared-base implementation:** recurrent query-conditioned memory tokens, one frozen
   encoder/teacher/reader, and adapter-toggle fallback.
3. **Executed policy plus full cost accounting**, not probe AUROC alone.
4. **A broad competence-boundary study** across seven bases, task regimes, lengths, and architectures.
5. **One signal driving multiple agent actions**, only if those actions are implemented and evaluated rather
   than listed as implications.

Recommended positioning:

> Prior work has established soft-token, recurrent, and KV-based compression, and recent work directly
> detects soft-token overflow or predicts safe compression levels. Paper A does not claim the first latent
> compressor or compression gate. It studies a specific deployment decision: held-out routing between a
> recurrent learned memory and a defined feasible raw-evidence path, with paired harm, coverage, and
> end-to-end cost measured across frozen model families.

## 9. Naming

- **Gated Latent Memory** collides with G-MemLLM and related GRU-Mem terminology.
- **GCM** is easily confused with ICLR 2024's CCM.
- Safer title candidates:
  - *Calibrated Latent Context Routing for Frozen Language Models*
  - *When Should a Frozen LM Trust Compressed Context?*
  - *Selective Latent Context with Raw-Evidence Fallback*

Do not rename until the final claim and statistical procedure are locked.

## 10. Required direct baselines

1. Belikova-style joint query–memory overflow probe;
2. TARG query-only margin;
3. first-token entropy from latent-memory management;
4. PoC-style performance predictor;
5. Self-Route-style sufficiency judgment;
6. faithful Gisting;
7. mean pooling / No Mean Feat;
8. full+SFT, same-input raw, extended-access GCM, and source-matched raw controls.

## 11. Introduction citation map

- Books/papers/codebases: LongBench, ∞Bench, LongBench-v2.
- Long conversations and persistent agents: LongMemEval and very-long-term conversational-memory work.
- Prefill/KV cost: FlashAttention, PagedAttention, Sarathi-Serve.
- Accepted length versus effective length: Lost in the Middle, RULER, NoLiMa, Du et al.
- Distractors: Shi et al. and Long-Context LLMs Meet RAG.
- Compression lineage: Gist, AutoCompressor, ICAE, CCM, Activation Beacon, 500xCompressor, Cartridges.
- Compression-failure detection: Belikova et al., PoC, Context Distillation as Latent Memory Management.
- Selective routing/risk: Wynn et al., LEC, Learn then Test, Conformal Risk Control.
- Agent implications: TierMem, Slipstream, TrustMem, Self-Compacting Agents.

## 12. Immediate paper changes

1. Replace “comparatively under-studied” with “recently studied but not yet broadly evaluated as an executed
   latent-memory-versus-raw routing policy.”
2. Do not claim first compression-failure detection, first confidence gate, first adaptive compression, first
   raw fallback, or first agent context control plane.
3. Treat agent-system designs as implications until implemented.
4. Rename “risk-controlled” to “held-out calibrated” unless finite-sample control is added.
5. Add positive-harm and four-outcome routing metrics.
6. Distinguish same-input compression fidelity from extended-access utility.
7. Add direct overflow, safe-ratio, and sufficiency baselines.
