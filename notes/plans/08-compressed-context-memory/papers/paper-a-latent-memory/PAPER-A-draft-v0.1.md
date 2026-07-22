# When to Trust Soft Memory: Mapping the Reliability Boundary of Long-Context Compression in Frozen Language Models

> Working draft v0.1 · 2026-07-16  
> Target format: paper prose in Markdown before migration to Overleaf.  
> Method name in this draft: **GCM (Gated Context Memory)**.  
> **Naming/statistics note:** the method name remains provisional because it is close to earlier
> G-MemLLM and Compressed Context Memory (CCM) names.
> Current gate claims are **held-out empirically calibrated**. A finite-sample claim is allowed only if the
> corrected document-level test certifies nonzero coverage.  
> **Narrative status:** structure scaffold only. The Introduction storyline, contribution ordering, and final
> experiment hierarchy have not been approved and must be revised from the author's direction before Overleaf.  
> Evidence labels: **Established** = usable now; **Provisional** = existing v1.8 result under fair re-evaluation;
> **Pending** = new paper-grade experiment currently running.

## Abstract

Long context is becoming the working memory of language-model systems: it holds documents, code, tool
descriptions, conversation history, observations, and the unfinished work of an agent. Two problems remain:
the input may exceed the available budget, and even an input that fits may not be used reliably. Compression
shortens the input but does not guarantee that the fact or relation needed for the next decision survives. We
introduce **Gated Context Memory (GCM)**, a recurrent soft-memory interface with a shared-backbone raw
fallback. The same frozen language model builds memory, answers through a small adapter, estimates
confidence, and can return to bounded raw evidence; no second language model or external verifier is
required. Across long-document reading and tool use, GCM learns a useful compact state and improves over
matched window and text-compression controls. In zero-shot long-context transfer, compact memory helps on
multi-hop targets where bounded raw input loses evidence, while empirical routing falls back on targets that
favor raw context. These results support a robustness-centered view: the value of soft memory depends not
only on how much it compresses, but on whether the needed information survives and whether the system can
recover when it does not.

## 1. Introduction

Long context is becoming the working memory of language-model systems. It lets a model read books and
papers, navigate a codebase, follow a long conversation, use many tools, and continue an agent trajectory
without discarding earlier observations. This state is not background metadata: it contains the information
from which the next answer or action must be produced. As language models move from single questions to
persistent assistants, the ability to carry this state efficiently becomes a basic systems requirement.

Long context has two distinct failure modes. First, the input may exceed the model or hardware budget,
forcing truncation or selection. Second, even when the entire input fits, inclusion does not guarantee robust
use: answers still change with evidence position, distractors, wording, and reasoning depth. Longer inputs
also require more computation and memory throughout inference. The problem is therefore not only how to fit
more tokens, but how to preserve reliable access to the information that matters.

Soft memory offers a different interface. Instead of asking the answering model to read every raw token, an
encoder condenses the input into a shorter sequence of continuous vectors. This addresses length and reader
cost, but it does not guarantee robustness: a compact reader can still produce an answer after a needed fact
or relation has been lost. Most compression evaluations report average task quality at a fixed ratio, which
does not tell a system whether memory is sufficient for a particular input. Detecting compression failure is
therefore part of the memory interface, not a separate deployment detail.

An agent does not need to use one context representation for every step. It can use compact memory for
routine decisions and return to raw evidence when the memory signal is weak. This two-path design keeps the
benefit of a short state on accepted inputs without removing access to the original context. The system
question is no longer only “how much should we compress?” but also “when should the agent trust the
compressed state?”

We introduce **Gated Context Memory (GCM)** to implement this design with one pretrained model whose main
weights remain fixed. GCM recurrently encodes successive chunks, carries earlier memory forward, and teaches
the same backbone to answer from the resulting vectors through a small low-rank adapter. Disabling the
adapter recovers a bounded raw-context path, while answer confidence supports held-out routing between the
two. In the same-backbone evaluation, GCM substantially improves over matched window and text-compression
controls on long-document multiple choice and remains stable on tool use. Across tasks, the results identify
where compact memory is useful and where the raw path remains necessary. In zero-shot long-context transfer,
GCM improves over bounded raw input on multi-hop targets, while the empirical route returns to raw context
on single-document and extractive targets. The result is a task-structured reliability boundary rather than
one universal compression setting.

We make four contributions. First, we introduce a recurrent short-state interface that uses one fixed base
model for encoding, answering, confidence, and raw fallback. Second, we evaluate compression as a paired
decision, measuring both harm (`raw succeeds, memory fails`) and rescue (`memory succeeds, raw fails`). Third,
we report answer quality together with memory-use rate, realized state length, latency, and fallback cost.
Fourth, we evaluate this robustness boundary across base models, controlled retrieval, public long-context
transfer, tool use, document reading, and multi-hop question answering.

## 2. Problem Setup

Let \(F_\theta\) be a pretrained language model with frozen backbone parameters \(\theta\). For a context
\(C\), query \(q\), and answer \(y\), the feasible raw path predicts

$$
p_{\mathrm{raw}}(y \mid C_{\le B}, q)
= F_\theta(C_{\le B}, q),
$$

where \(B\) is the raw-token budget that fits the selected hardware and evaluation protocol. We deliberately
write **feasible raw** rather than **full context**: \(C_{\le B}\) may be a truncation of the original input.

A compressor \(E_\phi\) maps the available input into a latent memory:

$$
M = E_\phi(C, q), \qquad M \in \mathbb{R}^{K_{\mathrm{eff}}\times d}.
$$

The compressed reader predicts

$$
p_{\mathrm{mem}}(y \mid M, q)
= F_{\theta,\psi}(M, q),
$$

where \(\psi\) is a small read adapter and \(\theta\) remains frozen.

The gate observes a scalar signal \(s(M,q)\) and selects

$$
\hat y =
\begin{cases}
\hat y_{\mathrm{mem}}, & s(M,q)\ge \tau,\\
\hat y_{\mathrm{raw}}, & s(M,q)<\tau.
\end{cases}
$$

The target is not simply maximum accuracy. Let \(A_\tau=\mathbf{1}\{s(M,q)\ge\tau\}\) denote acceptance of
the memory path. Signed policy-level excess loss is useful for end-to-end utility,

$$
R_{\mathrm{policy}}(\tau)
=
\mathbb{E}\left[
A_\tau\left(\ell(\hat y_{\mathrm{mem}},y)-\ell(\hat y_{\mathrm{raw}},y)\right)
\right]
$$

but memory wins can cancel compression harms in this signed quantity, and low coverage mechanically shrinks
it. Our primary safety target is therefore accepted-set positive compression harm:

$$
R_{\mathrm{harm}}(\tau)
=
\frac{
\mathbb{E}\left[
A_\tau
\left(
\ell(\hat y_{\mathrm{mem}},y)-\ell(\hat y_{\mathrm{raw}},y)
\right)_+
\right]
}{
\mathbb{E}[A_\tau]
}
\le \epsilon .
$$

We use a 2% harm target and report both risks and all four raw/memory outcome types. The main route selects
\(\tau\) on held-out calibration data. A separate document-level fixed-family test returns all-raw unless
it certifies nonzero coverage.

## 3. Gated Context Memory

### 3.1 One Frozen Base, Three Roles

GCM contains one pretrained backbone. The same frozen base is used as:

1. an **encoder**, whose intermediate features at learned memory-query positions summarize the context;
2. a **teacher**, whose feasible raw-context distribution supplies a distillation target;
3. a **reader**, which answers from the latent memory through a toggleable LoRA adapter.

The trainable state is limited to:

- \(K\) learned memory-query vectors \(W\);
- a projection multilayer perceptron (MLP) \(P_\phi\);
- a read-LoRA adapter \(\psi\);
- a content-free reconstruction slot used only during training.

The fallback disables the read adapter and recovers the original frozen base on the feasible raw input.

### 3.2 Chunk Encoder

For context chunk \(C_i\), query \(q\), prior projected memories \(M_{<i}\), and learned memory queries \(W\),
the encoder processes

$$
X_i = [M_{<i}; C_i; q; W].
$$

A causal mask lets the memory-query rows attend to all preceding positions. For the query-agnostic memory
\(M_0\), only memory-query-to-query attention is blocked.

The frozen base runs to a selected intermediate depth \(D\) (half depth in the default configuration).
We retain the hidden states at the final \(K\) positions:

$$
H_i = F_\theta^{1:D}(X_i)_{W}.
$$

The implementation temporarily truncates the base layer list during encoding, avoiding computation and
activation storage in layers that cannot affect the selected readout.

### 3.3 Recurrent, Length-Adaptive Memory

Each projected chunk memory is carried into the next chunk:

$$
M_i = P_\phi(H_i), \qquad
M_{<i+1} = [M_1;\ldots;M_i].
$$

The recurrent prefix is detached before the next chunk. This preserves the forward dependency while
bounding cross-chunk backpropagation. The final task loss still reads the concatenation of all chunk memories.

The default configuration uses 4,096 raw tokens per chunk and \(K=128\) memory tokens **per chunk**.
Consequently,

$$
K_{\mathrm{eff}} = S K,\qquad
S=\left\lceil \frac{|C|}{4096}\right\rceil.
$$

This is a length-adaptive memory, not a fixed 128-token document representation. A median QuALITY article
uses approximately two chunks and therefore about 256 reader-side memory tokens.

During training, variable-length augmentation randomly selects a prefix containing between one and \(S\)
chunks. This reduces sensitivity to one fixed training length.

### 3.4 Projection and Embedding-Scale Alignment

The default projection uses a Gaussian error linear unit (GELU):

$$
P_\phi(h)
= W_2\,\mathrm{GELU}(W_1 h+b_1)+b_2,
$$

with dimensions \(d\rightarrow 2d\rightarrow d\).

The result is normalized to the average norm of the base model's input embeddings:

$$
\widetilde M
=
\frac{M}{\|M\|_2+\varepsilon}
\cdot
\mathbb{E}_{v\in V}\|E_v\|_2.
$$

This is a scale alignment, not proof that \(M\) corresponds to real vocabulary tokens or lies in a shared
cross-model coordinate system.

### 3.5 Position Handling

GCM adds no learned position embedding. Each base retains its native positional mechanism, principally
rotary position embeddings (RoPE) in full-attention layers and recurrent order in gated delta network (GDN)
layers.

Within one chunk, the sequence \([M_{<i};C_i;q;W]\) receives contiguous positions \(0,\ldots,L_i-1\).
Each recurrent call restarts from zero. The final reader re-indexes \([M_1;\ldots;M_S;q]\) contiguously.
Therefore GCM preserves local token order and coarse chunk order, but it does not preserve every source
token's original absolute index.

### 3.6 Latent-Memory Reader

The compressed path replaces the raw context with

$$
[\widetilde M; q].
$$

A rank-64 LoRA adapter is active only on this path. It covers attention projections, GDN input/output
projections, and MLP gate/up/down projections when those modules exist. The raw path disables the adapter.

Each model receives its own memory queries, projection, and read adapter. Tokenizer identity or equal hidden
dimension does not imply a shared embedding coordinate system. Cross-model results therefore test a shared
algorithm and configuration, not direct transfer of latent vectors.

## 4. Training Objectives

### 4.1 Answer Loss

The primary loss is teacher-forced answer cross-entropy:

$$
\mathcal{L}_{\mathrm{task}}
=
-\log p_{\mathrm{mem}}(y\mid M,q).
$$

The gold answer includes a trailing newline. This provides termination supervision and fixes the earlier
artifact in which a correct first answer was followed by extra same-line text and scored as wrong.

### 4.2 Self-Distillation

The feasible raw path acts as a teacher over the gold positions. We minimize the Kullback--Leibler (KL)
divergence between its output distribution and the memory reader:

$$
\mathcal{L}_{\mathrm{distill}}
=
\mathrm{KL}
\left(
p_{\mathrm{raw}}(\cdot\mid C_{\le B},q,y_{<t})
\;\|\;
p_{\mathrm{mem}}(\cdot\mid M,q,y_{<t})
\right).
$$

We retain the teacher's top-64 logits and use weight \(\lambda_d=0.5\).

For the true-input QuALITY condition, the encoder may read up to 16k tokens while the teacher remains bounded
to the feasible 8k raw path. The paper reports this asymmetry explicitly.

### 4.3 Slot Reconstruction

A repeated content-free slot vector is appended after \(M\). Native positional encoding makes each slot
position distinct. The model's existing output layer predicts the context tokens:

$$
\mathcal{L}_{\mathrm{recon}}
=
-\sum_t \log p(c_t\mid M,\mathrm{slot}_{\le t}).
$$

The current run uses \(\lambda_r=0.5\) and reconstructs at most the first 512 raw tokens. Read-LoRA parameters
are frozen for this backward pass; gradients update the memory encoder and reconstruction slot.

### 4.4 Joint Training

The full objective is

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{task}}
+\lambda_d\mathcal{L}_{\mathrm{distill}}
+\lambda_r\mathcal{L}_{\mathrm{recon}}.
$$

Unlike the superseded phase-split implementation, \(M\) is not detached from the answer loss. Answer
supervision therefore trains both encoder and reader.

Deviation regularization, variational-autoencoder (VAE) KL, adversarial alignment, contrastive alignment,
encoder-LoRA, and
repeat-prompt reconstruction are retained as ablations or negative results and are not part of the main
method.

## 5. Held-Out Calibrated Read Gate

### 5.1 Signal

The current primary signal is the compressed path's next-token confidence:

$$
s_{\mathrm{conf}}(M,q)
=
\max_v p_{\mathrm{mem}}(v\mid M,q).
$$

Margin and TARG (query-only base margin) are pre-registered baselines. Reconstruction signals are not allowed
to replace confidence after observing test performance.

### 5.2 Calibration

For each model/benchmark run, we repeat a document-disjoint 25% calibration / 75% test split twenty times.
We report two routes. The first is an **empirically calibrated** comparison: choose the threshold with maximum
coverage subject to calibration-set policy excess loss at most two points, then freeze it on test. The second
is a conditional formal analysis: Bonferroni Learn-then-Test over fixed confidence thresholds, using source
documents rather than questions as independent units. We use a 2% harm target and 10% family-wise error
level. If no nonzero-coverage threshold passes, the formal result is all-raw and does not support
a finite-sample compression claim.

We report:

- test score and delta relative to feasible raw;
- signed policy excess loss;
- accepted-set positive harm and accepted-set signed excess;
- memory-harm rate (`raw better than memory`) and memory-benefit rate;
- the four binary outcomes: both correct, raw-only correct, memory-only correct, and both wrong;
- compressed-path coverage and fallback rate;
- always-raw and always-compress policies;
- random routing at matched coverage;
- TARG, margin, and oracle routing.

The empirical gate remains an analysis baseline. A risk-control claim is made only for thresholds certified
by Learn-then-Test with nonzero coverage; useful deployment additionally requires at least 20% mean coverage.

### 5.3 Why the Earlier Gate Result Is Excluded

The original v1.8 evaluator selected both the best signal and threshold on the evaluation set, initializing
the search at the raw-path score. “Gated accuracy is at least full accuracy” was therefore true by
construction. Those values are not used as paper evidence.

## 6. Computational Cost

GCM reduces the reader-side state from \(L\) raw tokens to \(K_{\mathrm{eff}}=SK\) latent tokens, but it does
not avoid reading the input:

1. the encoder scans the available context;
2. the reader processes the latent memory and query;
3. the gate performs a compressed first-token read;
4. fallback inputs additionally pay the feasible raw-path cost.

We therefore report encoder latency, compressed-reader latency, fallback latency, peak allocated memory,
actual memory-token count, and end-to-end expected cost under the gate. Compression ratios alone are
insufficient.

The current memory is query-conditioned, so encoder cost cannot automatically be amortized across unrelated
queries. Query-agnostic reuse is left to future work.

## 7. Experimental Setup

### 7.1 Models

Primary models:

- Qwen3-8B (dense quadratic attention);
- Qwen3.5-9B (hybrid GDN/full attention).

Fixed-configuration generality:

- Ministral-8B-Instruct;
- Qwen3.5-4B.

Existing BFCL pilots additionally cover ToolACE-2-8B, Llama-xLAM-2-8B, Qwen2.5-7B, and GLM-4-9B.
All memory adapters are trained separately per base.

### 7.2 Benchmarks

The full benchmark, status, and baseline matrix is maintained in
[paper-a-full-benchmark-table-2026-07-21.md](paper-a-full-benchmark-table-2026-07-21.md).

The main table uses:

- **QuALITY**: long literary multiple choice;
- **BFCL-live-multiple**: tool selection;
- **HotpotQA**: multi-hop QA.

SQuAD-v2 is retained only as an exact-text diagnostic. It is a short-passage extractive QA benchmark built
from Wikipedia paragraphs; answers are usually text spans, and some questions are unanswerable. It is not a
long-context benchmark and is therefore removed from the main table.

The RULER needle-in-a-haystack (NIAH) task is the exact-retrieval counterexample in the budget study.

The three-row development table is not sufficient evidence for a long-context compressor: BFCL is short,
HotpotQA is medium-length, and only QuALITY is consistently several thousand tokens. We therefore add
a distinct **real long-context transfer track**:

- QuALITY-trained adapters → LongBench-v2 and ∞Bench-choice;
- SQuAD-trained adapters → LongBench MultiFieldQA and Qasper;
- HotpotQA-trained adapters → LongBench HotpotQA, 2WikiMQA, and MuSiQue;
- NarrativeQA-trained adapters → LongBench NarrativeQA;
- RULER at controlled lengths as the exact-retrieval boundary.

These are transfer evaluations rather than per-target tuning: the target LongBench/∞Bench validation data are
never used to train the adapter. All eight bases run source-trained GCM, feasible raw, and no context; the two
primary bases additionally run source-trained full+SFT on every target. ∞Bench also tests a smaller K per
chunk because a fixed per-chunk K otherwise grows to thousands of final memory tokens on 100k-token books.

Full public validation sizes are 1,595 (QuALITY), 316 (BFCL-live-multiple), 5,928 (SQuAD-v2), and 7,405
(HotpotQA). SQuAD and Hotpot training are capped at 2,000 examples for matched-cost experiments. One exact
BFCL train/validation duplicate is removed before training.

### 7.3 Baselines

- no context;
- feasible bounded raw context;
- true raw context where it fits;
- task-trained full-context SFT-LoRA;
- LCLM-4x/8x/16x as the latest general long-context soft-token baseline;
- Semi-Dynamic Context Compression with automatic and fixed-32x latent budgets;
- faithful Gisting (LoRA + gist attention mask);
- In-Context Autoencoder (ICAE), AutoCompressor, Activation Beacon, 500xCompressor, Cartridges, ComprExIT,
  and Latent Context Compilation
  on architectures where their required KV/interface exists;
- mean pooling as the strong simple compression control;
- LLMLingua-2 at an approximately matched reader-token budget;
- raw sliding window at the same budget;
- always-compress, always-raw, random gate, TARG, margin/entropy, and oracle routing;
- a Belikova-style joint query–memory overflow probe and a lightweight PoC-style performance predictor,
  both trained only on the calibration split.

The historical `Gist-lite` and `Cartridge-lite` results are retained only as internal controls and are not
presented as faithful implementations of the published methods.

Cramming 1568 is reported only as a per-sample one-vector capacity control. It optimizes a new vector for
every text and is not an online context-compression baseline.

Full-context SFT is an adaptation reference, not a compression competitor. It matches the base model,
training data, steps, seeds, and adapter rank, but reads raw context at inference. It answers how much
accuracy is available when the compression and reader-state requirements are removed.

### 7.4 Metrics and Statistics

- multiple choice: length-normalized log-likelihood over answer letters;
- BFCL: tool accuracy;
- Hotpot: token-overlap F1 on the first non-empty answer line;
- RULER: exact value.

No gold-substring fallback is used. Trainable main-table methods use seeds 42, 43, and 44. We report
mean, standard deviation, paired bootstrap intervals, and per-item gate records.

## 8. Results

### 8.1 Evidence Audit

The audit changes the interpretation of the original QuALITY result. Qwen3-8B QuALITY contexts have median
length 6,511 tokens, p95 8,017, and maximum 8,413. Only 1.9% exceed 8,192 tokens. For Qwen3.5-9B, 5.2% exceed
8,192. Thus the earlier statement that QuALITY generally overruns an 8k window is false.

The old QuALITY comparison also used a task-trained read-LoRA on the compressed path and a frozen raw path.
The new main table adds both true raw and matched full+SFT controls.

### 8.2 Historical v1.8 Pilot

These values explain why the paper was started, but they are not used as final main-table evidence.
On BFCL-live-multiple, compressed accuracy was 74% for Qwen3-8B and 71% for Qwen3.5-9B,
compared with frozen raw references of 92% and 83%. Across eight bases and per-base K tuning, BFCL
compressed accuracy lies between approximately 65% and 75%.

The pilot also showed the negative boundary on HotpotQA: Qwen3-8B obtains 36% with memory versus 56% with
raw context, while Qwen3.5-9B obtains 34% versus 58%. Latent memory is therefore not a universal replacement
for raw evidence.

The original QuALITY result was 39% vs 11% for Qwen3-8B and 50% vs 14% for Qwen3.5-9B. It is replaced
by the clean three-seed results in Section 8.6.

### 8.3 Corrected Held-Out Gate Results

The primary confidence signal is evaluated on 24 model/task/seed groups with 20 exact-context-disjoint
repeated holdouts per group. The table averages test results over seeds and splits.

| model | metric | QuALITY | BFCL | Hotpot |
|---|---|---:|---:|---:|
| Qwen3-8B | GCM | 54.4% | 72.3% | 28.9% |
|  | route | **54.6%** | **88.5%** | **50.9%** |
|  | gain | +0.2 pp | +16.2 pp | +22.0 pp |
|  | FB AUC | 57.2 | 82.8 | 63.9 |
|  | FB rate | 0.2% | 46.6% | 68.3% |
|  | Δ raw | +47.4 pp | -3.5 pp | -2.4 pp |
| Qwen3.5-9B | GCM | **51.5%** | 72.0% | 30.5% |
|  | route | 51.4% | **80.5%** | **51.7%** |
|  | gain | -0.1 pp | +8.5 pp | +21.2 pp |
|  | FB AUC | 54.9 | 84.1 | 67.4 |
|  | FB rate | 0.3% | 22.0% | 52.1% |
|  | Δ raw | +44.4 pp | -3.8 pp | -2.2 pp |

Fallback AUROC measures whether low GCM confidence ranks examples where bounded raw scores higher than GCM.
QuALITY is an easy routing case because memory is much stronger than the bounded frozen raw path, so the
gate uses memory almost always. On BFCL and HotpotQA, empirical routing remains below raw. Confidence is
therefore an analysis signal, not a safe deployment rule on these tasks.

The corrected document-level fixed-family test certifies 0/24 groups at
a 2% harm target and 10% family-wise error level. Every formal route is all-raw. The paper makes no realized finite-sample
risk-control claim.

### 8.4 Existing Cross-Model Result

With per-base K tuning, BFCL compressed accuracy is:

| base | K64 | K128 | K256 |
|---|---:|---:|---:|
| ToolACE-2-8B | 73.4% | 74.2% | 74.2% |
| Llama-xLAM-2-8B | 72.7% | 71.9% | 71.9% |
| Ministral-8B | 67.2% | 75.0% | 73.4% |
| Qwen2.5-7B | 65.6% | 71.9% | 70.3% |
| Qwen3-8B | 69.5% | 71.9% | 74.0% |
| Qwen3.5-9B | 64.8% | 71.1% | 64.8% |
| GLM-4-9B | -- | 67.2% | 64.8% |
| Qwen3.5-4B | 64.8% | 66.4% | 67.2% |

K=128 wins or ties on six of eight bases. Because the table tunes K per model and uses independently trained
adapters, it is a generality pilot rather than evidence that one latent space transfers across models.

### 8.5 Full-Cost Adaptation and Raw-Window Boundaries

SFT-LoRA is a strong accuracy reference when full read-state cost and task adaptation are allowed:
95% on BFCL and 69% on Hotpot for Qwen3-8B. A matched raw window reaches 26% on Hotpot and remains below
bounded raw context.

These comparisons sharpen the scope of the method. GCM is useful only when a short reusable/readable state
is itself valuable and when the evidence can be represented by the latent carrier.

### 8.6 Paper-Grade Rerun

The 88-cell clean main grid is complete. The main result has two panels. Panel A asks whether the compressor
itself learns a useful short state. Panel B in Section 8.3 asks how much a shared-backbone raw fallback adds
at inference. Scores are native metrics; GCM and SFT show mean ± sample standard deviation over three seeds.

Benchmarks are columns and methods are rows. Raw and SFT use the same gray text color because both are
references rather than compressed-path competitors.

| model | method | QuALITY | BFCL | Hotpot |
|---|---|---:|---:|---:|
| Qwen3-8B | Raw | 7.2% | 92.4% | 53.7% |
|  | SFT | 81.7 ± 1.7%§ | 95.4 ± 0.3% | 68.8 ± 0.6% |
|  | Window | 15.7% | 55.7% | 26.2% |
|  | LLMLingua | 14.3% | 70.3% | 22.1% |
|  | GCM | **54.4 ± 0.2%** | **72.3 ± 0.5%** | **28.9 ± 0.2%** |
| Qwen3.5-9B | Raw | 7.1% | 84.5% | 53.9% |
|  | SFT | 85.0 ± 0.4% | 94.9 ± 1.0% | 71.7 ± 0.6% |
|  | Window | 16.7% | 52.8% | 24.8% |
|  | LLMLingua | 20.3% | 60.8% | 28.9% |
|  | GCM | **51.5 ± 1.7%** | **72.0 ± 0.8%** | **30.5 ± 0.3%** |

§ The six-cell same-config SFT reaudit saves outputs and adapters. Bounded SFT scores are 32.8/79.4/81.3%;
true-input SFT scores are 83.7/80.6/80.9%.

Three conclusions are stable. First, BFCL memory accuracy is about 72% on both main bases with small seed
variation, close to LLMLingua-2 but below raw context and SFT. Second, the Qwen3-8B QuALITY main-grid score
is \(54.4\pm0.2\)% , but independent reruns are run-sensitive: \(44.2\pm11.2\)% in the fixed-config grid
and \(48.7\pm15.1\)% in the replicate grid. Reaudited SFT reaches \(64.5\pm27.5\)% at the bounded input and
\(81.7\pm1.7\)% with true input. On Qwen3.5, GCM reaches \(51.5\pm1.7\)% while SFT reaches 84.7%.
The valid QuALITY claim is positive compressed competence with material run variance, not an accuracy
advantage over matched adaptation. Third, GCM is not a replacement for raw evidence on HotpotQA.

All 88 cells contain the expected number of validation records. Ten LongLLMLingua/original-LLMLingua cells
are excluded because the compressor raised legacy KV-cache errors and silently used fallback text. They will
be rerun in a compatible official environment with fallback disabled. Eight duplicate tags are treated as
technical repeats within seed; the largest same-tag spread is 0.95 percentage points.

## 9. Analysis

### 9.1 What Is Load-Bearing?

Existing flip-one BFCL ablations move compressed accuracy by at most approximately three points. Half-depth
encoding is as effective as full-depth encoding, K=128--256 is robust, and removing reconstruction has a
small effect on BFCL accuracy. The new minimal ablation repeats only five scientifically interpretable axes:
joint task training, distillation, reconstruction, recurrence, and memory budget.

### 9.2 Exact Retrieval Is a Boundary

Earlier latent-memory systems, including our v1 wrapper and matched Gist baselines, collapse on exact
needle retrieval even when raw context is nearly perfect. GCM compresses information into continuous
summaries rather than retaining source tokens verbatim. RULER is therefore treated as a falsifying boundary,
not a benchmark to hide.

### 9.3 Position Information

The recurrent memory preserves chunk order but re-indexes each chunk and the final memory sequence. It does
not preserve source-token absolute positions. Current benchmarks test content retention more than exact
position recovery; an explicit absolute-position probe is future work.

### 9.4 Cross-Model Meaning

The algorithm applies across architectures, but latent vectors are model-specific. ToolACE and xLAM have
almost identical sampled input embeddings, while Qwen2.5/Qwen3 and Qwen3.5-9B/4B do not share directly
compatible embedding dimensions or geometry. Cross-model claims refer to reproducibility after per-base
training, not adapter transfer.

## 10. Related Work

### Latent Context Compression

Recurrent Memory Transformer (RMT), Gist Tokens, AutoCompressor, In-Context Autoencoder (ICAE), Compressed
Context Memory (CCM), Activation Beacon, 500xCompressor,
xRAG, Cartridges, and other encoder--decoder compressors map context into recurrent states, soft tokens,
latent activations, or persistent key-value memories (Bulatov et al., 2022; Mu et al., 2023; Chevalier et al., 2023; Ge et al.,
2024; Kim et al., 2024; Cheng et al., 2024; Zhang et al., 2025; Li et al., 2025; Eyuboglu et al., 2026).
GCM is closest to recurrent soft memories that use a model's own hidden states as the compression interface.
Its focus is the shared-backbone interface that joins recurrent memory construction, memory reading,
confidence, and raw fallback. Recent mean-pooling baselines and evidence that frozen LMs can decode unusually
high-norm semantic vectors also motivate direct pooling and norm-scale controls.

Two 2026 methods are direct required comparisons. Latent Context Language Models (LCLMs) train a 0.6B
encoder and 4B decoder at 4x, 8x, and 16x compression and release RULER/LongBench checkpoints.
Semi-Dynamic Context Compression uses
Qwen3 and selects among discrete 2x--32x mean-pooling ratios per input. Cramming 1568 demonstrates
one-vector reconstruction capacity through per-sample optimization; it is a capacity bound, not a
deployment-equivalent compressor.

### Compression-Failure Detection and Adaptive Compression

Belikova et al. (2026) define overflow as raw/reference success followed by xRAG soft-compression failure and
train query--memory probes. An adaptive compressed-embedding retrieval method learns
when progressively supplied embeddings are sufficient; PoC predicts a sample-specific
performance--compression curve; and Compactor estimates context-specific KV-compression tolerance. Context
Distillation as Latent Memory Management uses first-token entropy to deactivate a retrieved LoRA memory, while
Selective Latent Thinking confidence-gates latent versus explicit reasoning. These studies establish failure
detection and adaptive compression as separate problems. Paper A studies them inside an executed
recurrent-soft-memory-versus-raw route, including paired harm outcomes and full cost.

### Retrieval and Agent-Memory Controllers

Retrieval-augmented generation (RAG) controllers such as TARG, Self-RAG, FLARE, Adaptive-RAG, Self-Route,
CRAG, and SeaKR use uncertainty, complexity, or sufficiency to control retrieval. TierMem routes from
summaries to linked raw evidence; Slipstream validates
and repairs trajectory compactions; TrustMem verifies memory consolidation; and self-compacting agents make
context maintenance an explicit action. Consequently, adaptive \(K\), compressor routing, verified memory
writes, and reliability-aware handoffs provide the broader system setting for GCM's compact-versus-raw route.

### Selective Prediction

Risk--coverage and abstention provide the correct evaluation language for a compress-or-fallback policy.
Learn-then-Test, Conformal Risk Control, early-exit routing under corrupted context, and Linear Expectation
Constraints provide finite-sample tools for selective decisions. We evaluate held-out empirical routing and
separately test whether a fixed policy family supports finite-sample control. The compression-specific
setting is one shared frozen-base recurrent memory versus a defined feasible raw path, evaluated end-to-end
across model families and task regimes.

## 11. Conclusion

Long-context systems need more than larger raw windows; they need a compact state that the answering model
can use directly. GCM shows that one fixed language model can recurrently build a short memory and answer
from it on tool use and long-document multiple choice.

The cross-task results reveal the central design principle: soft memory is an interface with a clear
operating region, not a universal replacement for evidence. Systems can use memory inside that region and
preserve raw context beyond it. This reframes context compression from choosing one global ratio to deciding
what information survives, whether the model can use it, and which path should answer next.

---

## Appendix A. Evidence Status for This Draft

| claim | current status | source / replacement |
|---|---|---|
| BFCL compresses to 65--75% across bases | Established pilot | v1.8 cross-model table |
| Gate preserves strong raw context | Rejected | held-out route remains below raw on BFCL and Hotpot |
| Formal nonzero risk-controlled coverage | Rejected at current setting | 0/24 groups certified; all-raw |
| Gate rescues degraded raw path | Established pilot | Qwen3.5-4B BFCL +52.3 points |
| QuALITY compression beats matched SFT | Rejected | GCM is stable but reaudited SFT is stronger |
| QuALITY generally exceeds 8k | Rejected | only 1.9% / 5.2% exceed 8,192 |
| Fixed 128-token document memory | Rejected | current method uses S×128 |
| Reconstruction is the best gate | Rejected | repeat-reconstruction follow-up is null |
| One embedding space transfers across models | Rejected | model-specific dimensions and geometry |
| GCM replaces raw evidence on multi-hop QA | Rejected | Hotpot raw and SFT are stronger |
| Cost savings at matched quality | Pending | measured latency + risk/coverage grid |

## Appendix B. Current Main Configuration

```text
base                  frozen, per model
memory queries        K=128 per 4096-token chunk
encoder depth         half
projection            d -> 2d -> d, GELU
normalization         hard embedding-norm match
recurrent prefix      on; prior memories detached
variable-length train on
read LoRA             rank 64
task loss             on, joint encoder+reader
distillation          0.5
slot reconstruction   0.5, context cap 512
gate signal           confidence
calibration/test       25/75, repeated 20 times
risk target            <=2 points
minimum coverage       >=20%
optimizer              AdamW, lr 3e-4
batch                  1 x accumulation 8
training steps         2000
seeds                  42, 43, 44
```

## Appendix C. Results Excluded from the Main Claims

- all in-sample `gated >= full` and `gated > full` values;
- pre-fix multiple-choice option-text scoring;
- pre-termination-fix low compressed scores;
- Gist-lite/Cartridge-lite labeled as the published methods;
- internal root-cause-analysis SFT-LoRA=1.00, which is likely memorization;
- the internal 12-item benchmark as a headline result;
- NarrativeQA method rankings at a near-floor ceiling;
- reconstruction-conformal and KVzip-unification claims that were designed but not validated.

## Appendix D. Scope and Future Work

The remaining long-context and official-baseline grids will broaden the current two-model result. GCM memory
length is \(S K\), not a fixed \(K\), and encoding still scans the available input. Because memory depends on
the query, reuse across unrelated questions is limited. Continuous states can lose exact identifiers and do
not preserve original token positions. Every base model needs its own adapter and latent space.

The current confidence route is empirical, and the formal test returns all-raw. Future work should combine
the reliability signal with adaptive memory budgets, raw-span retrieval, and step-level recovery. Some
official baselines require a different architecture or unavailable checkpoints; these rows are reported as
unavailable rather than replaced by internal replicas.
