# Paper A: Introduction, Methodology, and Experiments

> New section draft, 2026-08-04. This file does not replace the existing manuscript.
> It is written for later migration into the ICLR LaTeX version after review.

## 1 Introduction

Large language model agents are becoming capable enough to revolutionize how people work. In software engineering, coding agents are used to navigate large repositories, connect early requirements to later test failures, and resume unfinished work from execution traces. In research, agents are used to synthesize evidence distributed across large document collections while preserving the provenance behind their conclusions. Both are fundamentally long-context tasks. Code, documents, tool traces, user requirements, observations, and intermediate state collectively form the agent's working memory \citep{wu2024longmemeval,maharana2024locomo,mem0_2025}.

This dependence on long context creates a growing systems and reliability bottleneck. An agent's history expands with every observation and action, but the context that can be processed at each step remains bounded by model and hardware limits. Keeping the entire history increases prefill cost and reader-state memory at every subsequent decision. Truncating it can remove an early constraint, a failed attempt, or one piece of evidence needed to interpret a later observation. The resulting errors are not isolated: a missing fact can change the next action, whose output then becomes part of the history and influences the rest of the trajectory. Reliable memory management is therefore not merely an efficiency optimization. It is a prerequisite for agents that must remain consistent over long-running tasks.

Existing approaches manage this budget by selecting, retrieving, summarizing, or compressing the history. Retrieval preserves raw text but can miss evidence that is distributed across passages or only becomes relevant in combination. Text compression produces a readable prompt, but deletion and rewriting are difficult to reverse once a needed detail has been removed \citep{llmlingua2023,longllmlingua2023,llmlingua2_2024}. Soft compression instead maps context into a shorter sequence of continuous states \citep{mu2023gist,autocompressor2023,icae_2024,ccm2024}. Such states can preserve information that is awkward to express in a short textual summary, but their contents are opaque: a model may answer fluently even when compression has removed the evidence required by the current query.

This exposes a limitation in how context compression is usually formulated. Average accuracy and compression ratio describe a method over a dataset, but they do not tell a downstream agent whether a particular compressed memory is adequate for its present decision. Reliability is inherently query-dependent. The same history may preserve enough information for one query and omit a critical relation for another. A practical memory interface must therefore solve two coupled problems: it must preserve useful information under a bounded reader budget, and it must identify when the compressed representation should not be used.

We introduce **Gated Context Memory (GCM)**, a query-conditioned, self-assessing soft-memory interface. GCM encodes successive context chunks with recurrent conditioning, concatenates the resulting per-chunk states, and reads them through a lightweight adapter. A held-out threshold applied to memory-path confidence determines whether to use the compressed representation or a bounded raw view of the same source. Memory construction, memory reading, and raw fallback share one frozen language-model backbone; GCM requires neither a separate verifier nor a second language model. The resulting interface learns not only what to remember, but also when its memory may be inadequate for the current query. The present system rebuilds memory for each query; it is a per-decision long-context interface rather than a persistent agent state.

Our experiments are organized around this paired objective. First, we test whether the compressed state retains task-relevant information under controlled reader budgets. Second, we measure both sides of compression: cases where memory rescues an answer that bounded raw context misses, and cases where compression destroys evidence that raw context retains. Third, we evaluate whether held-out confidence ranks examples on which raw context obtains a higher task score. Finally, we test whether the same recipe generalizes across backbones and transfers without target-specific training. On one single-seed Qwen3.5-9B to 2WikiMQA transfer, the memory path scores 27.6 versus 2.9 for bounded raw input, a 24.7-point difference; broader transfer results remain task-dependent. On BFCL, confidence ranks raw-better examples with up to 84.1 AUROC. Under the fixed recipe, compressed-path BFCL accuracy lies between 71.2 and 73.1 across seven backbones, although four backbones currently have two rather than three runs. The separate finite-sample analysis certifies no nonzero memory coverage and therefore returns the all-raw policy: the confidence result is an empirical failure-ranking result, not a formal reliability guarantee.

These results support a broader view of memory for language-model agents. No single representation is best for every input: raw context preserves exact evidence, retrieval exploits sparse relevance, and soft memory can integrate information distributed beyond a bounded window. The central requirement is therefore not compression at any cost, but a selective interface that exposes its own operating boundary. Reliable agent memory should not merely shorten the history; it should make the quality of what survives visible to the system.

Our contributions are:

1. **A reliability formulation of context compression.** We formulate memory management as a per-query decision between a compact soft state and bounded raw evidence, and evaluate both compression rescue and compression harm.
2. **A shared-backbone memory interface.** GCM combines recurrent soft compression, lightweight memory reading, empirical failure assessment, and raw fallback around one frozen language model.
3. **An evaluation protocol for self-assessing compression.** We use disjoint calibration and test data, paired per-example outcomes, accepted-set positive harm, risk--coverage curves, and a separate document-level finite-sample analysis.
4. **Evidence for a task-dependent operating boundary.** Across document reading, tool use, multi-hop reasoning, long-context transfer, and a seven-backbone BFCL study, we identify settings in which compact memory helps and settings in which raw context remains necessary.

## 2 Methodology

### 2.1 Problem formulation

Let \(F_\theta\) be a pretrained language model with frozen backbone parameters \(\theta\). For a context \(C\), query \(q\), and target answer \(y\), the bounded raw path reads at most \(B\) context tokens:

\[
p_{\mathrm{raw}}(y \mid C_{\leq B}, q)
= F_\theta(C_{\leq B}, q).
\]

We call this path *bounded raw* rather than *full context*: \(C_{\leq B}\) may omit relevant evidence when the source exceeds the available reader budget. A compressor \(E_\phi\) instead maps the available context and query to a sequence of continuous memory states,

\[
M = E_\phi(C,q) \in \mathbb{R}^{K_{\mathrm{eff}}\times d},
\qquad
p_{\mathrm{mem}}(y \mid M,q)
= F_{\theta,\psi}(M,q),
\]

where \(d\) is the backbone hidden size and \(\psi\) is a lightweight read adapter. The backbone remains frozen in both paths. The compressed path is useful when it preserves information unavailable to \(C_{\leq B}\), but it can also remove information that the raw path retains.

GCM therefore treats compression as a selective decision. A scalar memory signal \(s(M,q)\) and threshold \(\tau\) define

\[
A_\tau(M,q)=\mathbf{1}\{s(M,q)\geq\tau\},
\]

where \(A_\tau=1\) accepts the memory path and \(A_\tau=0\) invokes bounded raw fallback. The final predictor is

\[
p_{\mathrm{GCM}} =
A_\tau\, p_{\mathrm{mem}}
+(1-A_\tau)\,p_{\mathrm{raw}}.
\]

This formulation separates three questions that are often conflated: whether a state is compact, whether it is accurate on average, and whether accepting it satisfies a defined per-query harm criterion.

### 2.2 Query-conditioned recurrent memory

We split \(C\) into \(S\) chunks \(C_1,\ldots,C_S\). For chunk \(C_i\), learned memory queries \(W\in\mathbb{R}^{K\times d}\), the current query \(q\), and the projected summaries from earlier chunks, the compressor applies the first \(D\) layers of the frozen backbone:

\[
X_i=[M_{<i};C_i;q;W],
\qquad
H_i=F_\theta^{1:D}(X_i)_W,
\qquad
M_i=P_\phi(H_i).
\]

The subscript \(W\) selects hidden states at the learned memory-query positions, and \(P_\phi\) is a two-layer projection with a nonlinear activation. Earlier summaries condition the next chunk in the forward pass and are detached before that update, preserving recurrent information flow while bounding cross-chunk backpropagation. In the default model, the reader receives the concatenation \([M_1;\ldots;M_S]\); recurrence conditions chunk encoding but does not keep the final state fixed.

The default configuration uses chunks of 4,096 source tokens and \(K=128\) states per chunk:

\[
K_{\mathrm{eff}} = S K,
\qquad
S=\left\lceil |C|/4096 \right\rceil.
\]

The default representation is therefore **length-adaptive**, not a fixed 128-state document vector. It reduces the sequence seen by the reader while allowing memory capacity to grow with the amount of available context. During training, variable-length examples are constructed by sampling prefixes of the available chunks. A separate capped-state ablation tests a genuinely fixed recurrent state rather than truncating the final memory post hoc.

The projection output is normalized to the mean norm of the backbone's input embeddings. This aligns the scale of continuous memory with the reader's embedding space without forcing memory states to equal vocabulary embeddings. Each backbone retains its native positional mechanism. Token order within chunks and chunk order are preserved, but exact original token positions are not.

### 2.3 Shared-backbone reader

The memory reader replaces the raw context with \([M;q]\) and activates a rank-64 LoRA adapter \(\psi\). The raw path disables this adapter and reads \(C_{\leq B}\) with the same frozen backbone. This design has two consequences. First, a difference between the paths is attributable to representation and the small read interface rather than to two unrelated language models. Second, fallback does not require an external verifier or a separately maintained reader.

The trainable components are the memory-query vectors, the projection network, one reconstruction slot, and the read adapter. Each backbone learns its own adapter and latent space; GCM does not assume that continuous memories transfer directly between models.

### 2.4 Training objectives

GCM is trained with

\[
\mathcal{L}
= \mathcal{L}_{\mathrm{task}}
+ \lambda_{\mathrm{distill}}\mathcal{L}_{\mathrm{distill}}
+ \lambda_{\mathrm{recon}}\mathcal{L}_{\mathrm{recon}},
\]

with \(\lambda_{\mathrm{distill}}=\lambda_{\mathrm{recon}}=0.5\) by default.

**Task loss.** \(\mathcal{L}_{\mathrm{task}}\) is teacher-forced answer cross-entropy under the memory path. Gradients flow through both the read adapter and the memory construction path, so the compressor is optimized for the downstream decision rather than only for reconstruction.

**Bounded-raw distillation.** \(\mathcal{L}_{\mathrm{distill}}\) matches the compressed reader to the bounded-raw path over the gold answer span. We retain the teacher's top-64 logits per position. This objective transfers answer-relevant behavior while keeping the teacher consistent with the raw budget used at evaluation.

**Slot reconstruction.** \(\mathcal{L}_{\mathrm{recon}}\) asks the memory states to reconstruct up to the first 512 context tokens through the tied language-model head. It provides a content-preservation signal complementary to answer supervision. Ablations remove each auxiliary objective separately and also test whether the answer loss is detached from memory construction.

### 2.5 Self-assessment and routing

The primary self-assessment signal is compressed-path first-token confidence,

\[
s_{\mathrm{conf}}(M,q)
= \max_v p_{\mathrm{mem}}(v\mid M,q).
\]

We compare it with margin, TARG, a joint probe, and a performance predictor. Signals are evaluated against the paired event that the bounded raw path obtains a higher task score than the memory path. This definition applies to both binary accuracy and continuous task metrics; binary both-correct/raw-only/memory-only/both-wrong outcomes are reported separately. AUROC measures failure ranking independently of a deployment threshold.

For empirical routing, each model--task run is divided by document into a 25% calibration subset and a disjoint 75% test subset. We repeat this split 20 times. Let \(u(\hat y,y)\in[0,1]\) denote the task score, with larger values better. The threshold is selected on calibration data to maximize memory coverage subject to accepted-set positive compression harm,

\[
R_{\mathrm{harm}}(\tau)
=
\frac{
\mathbb{E}\left[
A_\tau
\{u(\hat y_{\mathrm{raw}},y)
-u(\hat y_{\mathrm{mem}},y)\}_+
\right]
}{
\mathbb{E}[A_\tau]
}
\leq \epsilon,
\]

with \(\epsilon=0.02\). Positive harm prevents memory wins from canceling memory failures. We report the frozen threshold on the held-out test subset, including coverage, fallback rate, final score, and all four paired correctness outcomes.

Empirical calibration is not a formal safety certificate. We therefore conduct a separate document-level Learn-then-Test analysis over a fixed, pre-specified threshold family. If no threshold certifies nonzero coverage, the formal policy returns all raw. Keeping these analyses separate prevents a useful empirical ranking signal from being presented as a finite-sample guarantee.

### 2.6 Computation and memory accounting

For an input of \(L\) tokens, the default reader consumes \(SK\) memory states rather than \(L\) raw tokens. The recurrent encoder, however, still scans the available source. Moreover, because memory is conditioned on \(q\), it generally must be rebuilt for a different query. We therefore report encoder latency, reader latency, realized state length, and incremental peak allocation separately. Reader-state reduction is not treated as evidence of end-to-end speedup.

## 3 Experimental Design

### 3.1 Research questions

The experiments answer five questions:

- **RQ1: Compression competence.** Does GCM preserve task-relevant information better than matched compact controls?
- **RQ2: Complementarity.** When does memory recover evidence lost by bounded raw context, and when does compression destroy evidence that raw context retains?
- **RQ3: Self-assessment.** Can a held-out signal identify raw-better examples and improve the final memory-or-raw decision?
- **RQ4: Generality and transfer.** Does the same recipe work across backbone families and source-to-target shifts without target-specific training?
- **RQ5: Mechanism and cost.** Which objectives and architectural choices matter, and what reader-state benefit remains after accounting for encoder cost?

### 3.2 Models

The main experiments use Qwen3-8B and Qwen3.5-9B. Fixed-configuration generality additionally evaluates Qwen3.5-4B, GLM-4-9B, Ministral-8B-Instruct, Llama-xLAM-2-8B, and ToolACE-2-8B. Every backbone trains an independent GCM adapter with the same default recipe. Cross-model comparisons therefore test algorithmic generality, not transfer of one shared continuous memory space.

### 3.3 Tasks

The main task suite covers distinct evidence structures:

- **QuALITY** evaluates multiple-choice comprehension of long documents \citep{pang2022quality}.
- **BFCL-live-multiple** evaluates multi-step tool selection and argument construction.
- **HotpotQA** evaluates multi-hop question answering over distributed evidence \citep{yang2018hotpotqa}.
- **SQuAD-v2** is retained as a short-passage exact-text diagnostic rather than a headline long-context task \citep{rajpurkar2018squad2}.

For long-context transfer, each source adapter is trained once and then reused unchanged on its target tasks. We evaluate six LongBench subsets spanning single-document, extractive, narrative, and multi-hop QA, together with LongBench-v2 and InfiniteBench-choice. This design tests whether memory learned on one source distribution captures a transferable compression strategy rather than target-specific supervision.

The benchmark pipeline uses corrected zero-based QuALITY labels, removes one exact BFCL train--evaluation overlap, records source and visible-token lengths per item, and disables gold-substring fallback. Results produced by earlier invalid labels or silent fallback paths are excluded rather than repaired through post hoc score transformations.

### 3.4 Comparisons

We separate references from compression competitors.

**References.**

- *No context* measures task priors available from the query alone.
- *Bounded raw* uses the raw-token budget available to the reader.
- *Available raw* reads all available input where the model and hardware permit it.
- *Bounded-raw SFT-LoRA* uses matched training data, steps, seeds, and adapter rank but pays the bounded raw reader cost. Separate available-input variants are reported where they fit. SFT estimates adaptation performance when compression is removed and is not presented as a same-cost compression baseline.

**Matched compact controls.**

- A raw window retains a fixed compact token budget.
- BM25 retrieve-then-read selects raw spans under the same target budget.
- LLMLingua-2 performs hard text compression to the same target token count.
- Exact mean pooling provides a simple continuous-state control.

The original main-table controls use fixed operating points: 256 reader tokens for QuALITY and 128 for the other main tasks. They are compact controls, not exact per-item matches to a multi-chunk \(S\times K\) state. For the corrected long-context target controls, the budget is computed per item as \(\lceil L_{\mathrm{visible}}/4096\rceil\times128\), exactly matching the state produced by the default GCM chunking rule. All methods in that comparison see the same bounded source before compression or selection.

Published soft-memory systems with different released backbones or native task interfaces are evaluated with authors' code and checkpoints in a separate native-base table. We record their released base, realized state length, raw and no-context references, and repository/checkpoint provenance. Absolute scores across unrelated backbones are not used to claim a same-base ranking.

### 3.5 Training and evaluation protocol

The default GCM configuration uses \(K=128\) states per 4,096-token chunk, a half-depth encoder, a two-layer projection, hard norm matching, rank-64 read LoRA, and 2,000 optimization steps with AdamW at \(3\times10^{-4}\). The effective batch is eight through gradient accumulation. Trainable main cells use seeds 42, 43, and 44.

Multiple-choice tasks use answer-label log likelihood. BFCL uses official tool accuracy. Open-ended QA uses the benchmark's token-overlap F1 or native metric. Generation is deterministic, and no gold-answer fallback is permitted. The current evaluator logs item-level encoder, generation, and scoring exceptions and omits affected items. We therefore report the realized number of per-item records and do not infer full-split completion from the presence of an output file.

Result artifacts store per-item records containing the model, seed, item and document identity, source length, encoder-visible length, bounded-raw length, realized memory length, predictions from each path, paired scores, and gate signals. Reported sample counts are computed from these records. Runtime status files are operational metadata and are not treated as evidence that an expected full split was evaluated.

### 3.6 Statistical and reliability reporting

For trainable main paths, we report mean and standard deviation over three seeds together with paired bootstrap intervals where available. Reliability tables include:

- memory-only and raw-only correctness;
- both-correct and both-wrong outcomes;
- failure-ranking AUROC;
- accepted memory coverage and raw fallback rate;
- accepted-set positive harm;
- final held-out routed accuracy;
- the result of the separate document-level finite-sample test.

We also report realized state tokens rather than nominal compression settings. Cost profiles separate recurrent encoding from compressed reading and bounded-raw reading, with synchronized CUDA timing and incremental peak allocation. This accounting makes accuracy, reliability, and systems cost independently inspectable.

### 3.7 Result organization

The results follow the research questions rather than the order in which experiments were run:

1. **Main matched-budget comparison:** whether the compressed path learns a useful state.
2. **Paired reliability analysis:** where memory helps, where it harms, and whether confidence separates the two.
3. **Long-context transfer:** the task-dependent operating boundary under source-to-target shift.
4. **Seven-backbone generality:** stability of the fixed training recipe.
5. **Mechanism and fixed-state ablations:** what creates the effect and whether memory can remain truly bounded.
6. **Measured cost and official baselines:** reader-state savings, encoder overhead, and placement relative to released soft-memory systems.
