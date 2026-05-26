# Inference-Time Training: Ideas & Notes

> Brainstorm notebook on **inference-time training (ITT)**, organized around the central framing:
> the model output $y = f(X; W)$ has two levers — **$X$** (context / prompts / tools / reasoning,
> tunable at inference) and **$W$** (weights, classically tuned only at training).
> What if we *exhaust $X$ first*, then go look for the best $W$?
>
> Author: Mingjia Shi · Last updated: 2026-05-26

---

## 0. TL;DR

- The interesting frontier in 2026 is no longer "more $X$" (longer context, more CoT tokens, more search).
  It's the **$X \leftrightarrow W$ exchange** at inference time.
- Two paradigms are emerging:
  - **Hypernet-amortized updates** (e.g. Doc-to-LoRA, Text-to-LoRA) — *no backprop, one forward pass.*
  - **Inference-time gradient updates** (TTT, online RFT, sparse memory finetuning) — *real backprop, per-instance / per-stream.*
- The two are complementary. The most under-explored regime is **"do all the $X$-side work first, then update $W$ on the residual"** — this gives a clean curriculum, a clean dataset definition, and a clean evaluation protocol.

---

## 1. The Framing: bilevel $X$–$W$ optimization

Classical ML separates the two:
- **Training**: $\min_W \mathbb{E}_{(q, y^\*)} \mathcal{L}(f(X_q; W), y^\*)$ with $X_q$ fixed
- **Inference**: choose $X$ to maximize quality, $W$ frozen

The bilevel view:

$$
W^\star \;=\; \arg\max_W \, \mathbb{E}_q \big[\, V\big(f(X^\star(q,W); W)\big)\,\big],
\qquad
X^\star(q,W) \;=\; \arg\max_X V\big(f(X; W)\big)
$$

i.e. *inner loop* exhausts $X$ at inference, *outer loop* updates $W$ on the residual.

Key derived quantities (all worth defining + measuring):
- **$X$-saturation curve** for a query: accuracy vs $X$-budget (tokens / search depth / retrieval depth).
- **$W$-demand**: $1 - \max_X V(f(X; W))$ — the residual that no amount of $X$ can solve.
- **$X{\to}W$ exchange rate**: marginal accuracy gain per FLOP, swapping $X$ for $W$ updates.
- **$X$-saturated examples**: training data filter — only update $W$ on examples whose $X$-curve has plateaued.

---

## 2. Ideas — full list

Ideas are grouped by lever. ★ marks my personal top picks. Cross-refs to the "framing" section above.

### A. Supervision signal (what to use as the loss for $W$)

- **A1** ★ **Self-consistency as loss** — N CoT samples → majority vote → soft label → update $W$ on the difference. Distill the model's own ensemble into its weights.
- **A2** ★ **Reasoning-trace bootstrap** — verifier-validated CoT trajectories become training samples for an online RFT step. "Sleep-time" or "between-turn" updates. Refs: [STaR](#refs), [Sleep-time Compute](#refs).
- **A3** **Reconstruction loss on context** — the classic TTT self-supervised aux loss adapted to LLM inputs (mask & reconstruct).
- **A4** **Contrastive on retrieved docs** — InfoNCE between relevant vs irrelevant retrieved chunks; updates embedding/MLP layers.
- **A5** **User implicit feedback** — accept/reject/rewrite signals become DPO pairs, applied as a single in-session update.
- **A6** **Tool-call outcome reward** — tool return value or environment feedback drives one online RL step.
- **A7** ★ **Contrastive gradient between $X$-levels** — same $(q, y^\*)$ under `x_simple` (fail) vs `x_CoT` (succeed). $\nabla_W [\log p(y^\*|q,x_\text{simple}) - \log p(y^\*|q,x_\text{CoT})]$ tells you *which weight to change so simple prompts work*. (Same family as long-CoT → short-CoT distillation, but with a clean signal.)

### B. Parameterization (what part of $W$ to update)

- **B1** ★ **Per-request LoRA** — a from-scratch tiny LoRA, lifecycle = single request, discarded on completion. Privacy-friendly, no cross-contamination.
- **B2** **Steering vectors at hidden states** — no weight update at all; learn a residual vector $h \leftarrow h + v$. Faster than LoRA.
- **B3** **KV-as-weight** — SVD-compress the KV cache into a low-rank weight update (turns *read* memory into *learnt* memory). Doc-to-LoRA implicitly does this.
- **B4** **Trainable memory tokens** — add $M$ learnable tokens; do TTT only on them.
- **B5** **Layer-selective TTT** — only update last-$K$ layers or specific attention heads, chosen by Fisher info / activation magnitude.
- **B6** **Sparse memory finetuning** — update only a small subset of MLP rows (refs: [Lin et al., 2025](#refs)).

### C. Trigger (when to update)

- **C1** ★ **Uncertainty-gated TTT** — only fire a gradient step when token entropy / verifier disagreement crosses a threshold. The compute story is the biggest bottleneck of TTT — gating is the unlock.
- **C2** **Error-driven TTT** — self-critique detects an error → contrastive update.
- **C3** **Streaming TTT** — continuous in-place updates across a long dialog / agent rollout, with forgetting.
- **C4** **Checkpoint & rollback** — TTT must not degrade base capability; gate via a probe held-out set.

### D. Search / exploration

- **D1** ★★ **$W$-space Best-of-N** — generate $N$ LoRA perturbations $\{\Delta W_i\}$, verifier picks the best. The orthogonal axis to the usual $X$-space Best-of-N. (Massively under-explored.)
- **D2** **MCMC over $W$** — Langevin-style sampler in weight space, accepting / rejecting by verifier reward. Test-time RLHF on a single instance.
- **D3** **Joint $(X, W)$ search** — alternating updates between an outer prompt search and an inner weight update. Bilevel made explicit.

### E. Application domains

- **E1** ★ **Codebase-adaptive coding agent** — entering a new repo triggers self-supervised TTT on the repo's source: API names, idioms, style. Then start the task. Hypothesis: dramatic drop in hallucinated API calls. Directly applicable to Cursor / coding agents.
- **E2** ★ **Long-document QA via Doc-to-LoRA** — internalize a 100K-token doc into a 50 MB adapter, free the context window. (Sakana already did this. Open: how to combine with backprop for the last few %.)
- **E3** **Personalization on-device** — nightly TTT on a single user's history; weights never leave the device. Federated aggregation across users for shared improvements.
- **E4** **Safety / jailbreak defense** — detect adversarial pattern → contrastive update reinforcing refusal behavior, scoped to the session.
- **E5** **Robotics sim-to-real** — deploying into new env triggers self-supervised dynamics-loss update on the policy / world-model.
- **E6** **Diffusion test-time tuning** — CLIP / reward model backprops to U-Net once per sample. (Already exists; needs a systematic study.)
- **E7** **GUI agent online learning** — Computer-use rollouts; every verified success is an online RFT example.

### F. Systems / serving

- **F1** ★★ **TTT serving infra** — per-request weight delta inside batched serving. Beyond S-LoRA: needs runtime *write* support, not just read.
- **F2** **Reusable TTT cache** — cluster per-request LoRAs by task / user / domain; reuse / merge → dynamic expert pool.
- **F3** **Diff-quantized weight deltas** — sparse + quantized for cheap storage and transfer.
- **F4** **Shared optimizer state proxy** — avoid per-request Adam state allocation.

### G. Theory / science

- **G1** **ICL ↔ TTT equivalence boundary** — when does in-context = implicit gradient descent? when does it diverge?
- **G2** ★ **Information-theoretic $X{+}W$ budget** — $I_\text{needed} = I(y \mid q)$. If $I(X) + I(W) < I_\text{needed}$, no inference strategy works. Quantify how many bits D2L / TTT actually transfer per update.
- **G3** **Sample complexity of TTT** — how many gradient steps at test time approximate $E$ epochs of offline training?
- **G4** **Stability / catastrophic forgetting** — TTT update directions that *preserve* base capability (capacity-preserving updates, null-space edits — see [AlphaEdit](#refs)).
- **G5** **Privacy of TTT** — does a per-instance update leak data into subsequent requests sharing the same instance?
- **G6** ★ **$X$-irreducible problems** — characterize tasks for which no $X$ suffices and $W$ *must* change. Test for them empirically (e.g. capabilities absent at all temperatures with infinite context).

### H. Cross-paradigm / "wild" ideas

- **H1** ★ **Inference-time RL alignment** — per-conversation lightweight RLHF using a small reward model and the system prompt as preference signal. Personalized alignment.
- **H2** **TTT × World Model** — agent updates its world model online, then re-plans, then updates again. Bi-level inference.
- **H3** ★ **Reasoning trace → weight delta** — instead of generating CoT tokens, generate a weight delta that produces the same output. The model "compiles thinking into weights" on the fly.
- **H4** **TTT as compression** — internalize retrieval-augmented prompts into a LoRA over time. Memory → muscle.
- **H5** **Multi-model collaborative TTT** — small model generates supervision for the large model; reverse-distillation at test time.
- **H6** ★★★ **Model outputs $\Delta W$ as part of generation** — every assistant turn produces $(y, \Delta W)$ where $\Delta W$ describes "what I learned from this turn"; verifier accepts/rejects/scales. Closest path to true self-modifying LLMs. Builds on Doc-to-LoRA's proof that hypernets can output usable LoRAs.
- **H7** ★ **Inference-time $\Delta W$ as a tool call** — the model emits `<learn>...</learn>` spans during reasoning; the content triggers a D2L-like submodule that generates a temporary LoRA. Tool-use, but the tool is the weights.

### I. Ideas from the $X$–$W$ exchange framing (latest brainstorm)

- **I1** ★★★ **$X$-saturation curve as dataset curator** — for each candidate training example, sweep $X$-budget; keep only those with positive $W$-demand. Sharper than hard-negative mining; principled difficulty notion.
- **I2** ★★ **$X$ as virtual training data** — Idea A7's diff gradient applied at population scale: builds a SFT/RL pipeline that systematically converts "things long-$X$ can do" into "things short-$X$ can do".
- **I3** ★★★ **$X$-then-$W$ curriculum** — the production pipeline of the future: (i) train a strong $X$-saturating model, (ii) distill only the $X$-residual into $W$, (iii) repeat. Amortizes o1-style inference cost into training. (Different from generic distillation because the source is "*same model under more $X$-budget*", and only the bits *not* already in-context-recoverable are distilled.)
- **I4** ★★ **Inverse problem on $X$-trajectories** — collect $(q, \text{long-}X\text{-trajectory}, y)$ triplets. Solve $\min_W \|f(\text{short-}X; W) - f(\text{long-}X; W_0)\|$. The "compile thinking" objective made concrete.
- **I5** **Prompt ↔ LoRA equivalence map** — D2L is doc→LoRA. The *reverse* (LoRA→prompt) gives interpretability of TTT updates and a detector for "stealth jailbreaks" (a prompt that secretly behaves like a malicious weight delta).
- **I6** **Inference-time compute allocator** — a tiny controller that, per query, decides "spend the next FLOP on more $X$ or on $\Delta W$?" Trained via bandit / RL.

---

## 3. Reading notes: Doc-to-LoRA (Sakana AI, 2026)

[arxiv 2602.15902](https://arxiv.org/abs/2602.15902) · [project page](https://pub.sakana.ai/doc-to-lora/) · [code](https://github.com/SakanaAI/doc-to-lora)

### One-liner
Meta-learn the **Context Distillation (CD)** procedure into a hypernetwork. A single forward pass replaces 40–100s of backprop, producing a context-specific LoRA in <1s.

### Objective
$$
\min_\phi \;\mathrm{KL}\!\big(\, p_\theta(y\mid x, c) \;\big\|\; p_{\theta + H_\phi(c)}(y\mid x) \,\big)
$$
- $\phi$: hypernet params (309M, Perceiver-style, 8 cross-attn blocks)
- $H_\phi(c) = \Delta W$: rank-8 LoRA on MLP `down_proj`
- Teacher = base LLM with $c$ in context. Student = base LLM with LoRA, no $c$ in context.

### Architecture
- Context $c$ → frozen base LLM → per-layer activations $Z_l$
- Perceiver cross-attn: variable-length $Z_l$ → fixed-size latents → $(A_l, B_l)$ LoRA matrices
- **Chunking** is the key trick: long context → $K$ chunks → concat along rank dim → effective rank $r \cdot K$. Enables zero-shot generalization from 256-token training to 40K+ token inference.

### Results
| Metric | Full context | CD (oracle) | CD (gen-query) | **D2L** |
|---|---|---|---|---|
| SQuAD rel. perf | 100% | ~99% | ~70% | **82.5%** |
| 2WikiMHQA rel. perf | 100% | 90% | 70% | **85.7%** |
| Update latency | – | 40s | 100s+ | **<0.5s** |
| 128K-doc memory | 12 GB KV | 7.8 GB | 80 GB | **<50 MB** |
| Length generalization | – | – | – | **256 tokens → 40K tokens** |

### Striking ablation
VLM (Gemma-3-4b) as the encoder → D2L generates LoRA for text-only LLM (Gemma-2-2b) → text model classifies images at **75% on Imagenette**, despite *neither* model having seen images in D2L training. The hypernet learns a generic *modality-agnostic* context→weight map.

### Limitations (= idea gaps)
1. **Knowledge interference** (Table 8): SQuAD perf drops 0.81 → 0.10 when context is unrelated to query. D2L has an implicit "subsequent queries are about the internalized content" prior.
2. **Hypernet must be retrained for each new base LLM** (5 days × 8 H200 for Gemma-2-2b).
3. **Performance gap vs ICL still exists** (~82% vs 100%).
4. **Add-only**: no unlearn / rollback.
5. **LoRA rank ceiling**: composing many chunks may interfere; multi-doc compositionality not fully validated.

### How D2L maps to ideas above
- Implements **B1** (per-request LoRA), **B3** (KV→weight), **E2** (long-doc QA), **H4** (memory→muscle).
- Open gaps: **C1** (when to fire), **D1** ($W$-space search), **G4** (interference), **G5** (privacy), **H6** (model outputs $\Delta W$), **I3** ($X$-then-$W$ curriculum).

### My top D2L-adjacent follow-ups
1. **D2L + K-step backprop hybrid** — use D2L's LoRA as initialization for a tiny gradient-descent finetune. Closes the 82→100% gap.
2. **Code-D2L** — same recipe with code repos as the context corpus. Per-repo LoRAs for coding agents. (Idea E1 made concrete via D2L.)
3. **Unlearn-aware D2L** — augment training with irrelevant-query examples + a regularizer that drives $\|\Delta W\|$ → 0 when query is unrelated. Kills the interference (Table 8) problem.
4. **D2L → prompt inversion** — train an inverse model LoRA→prompt for interpretability + safety auditing.
5. **Mixture-of-LoRA routing** — when composing $K$ chunk LoRAs, route per-query to a relevant subset rather than always concat-rank.

---

## 4. Top picks (if forced to choose three)

Ranked by (novelty × feasibility × impact) and personal taste:

| Rank | Idea | Why |
|---|---|---|
| 1 | **I1** + **I3**: $X$-saturation curve + $X$-then-$W$ curriculum | Reshapes the SFT/RL data pipeline. Clean theory, measurable. Connects directly to the o1-cost-amortization story. |
| 2 | **D1**: $W$-space Best-of-N | Genuinely new axis for test-time compute. Could be one focused paper. Easy ablations vs $X$-space Best-of-N. |
| 3 | **E1**/**Code-D2L**: codebase-adaptive coding agent | Highest industrial $/research alignment. Cursor / coding agent integration is obvious. Concrete metric (pass@k, API hallucination rate). |
| Honourable | **H6**: model outputs $\Delta W$ | Most ambitious. Builds on D2L proof that hypernets can output usable LoRAs. Probably the next paradigm if it works. |

---

## 5. Open questions to investigate

1. Is there a universal *exchange rate* between $X$-tokens and $W$-bits across tasks, or is it task-specific?
2. Does TTT update on query $q_t$ help / hurt query $q_{t+1}$ in a stream? Under what conditions?
3. Can we learn a *single* hypernet that produces both Doc-LoRA and Text-LoRA (Sakana's "foundation update generator" vision)?
4. Is there a "minimum rank" theorem — for a task with $W$-demand $b$ bits, the LoRA rank must be at least $\Omega(b/d)$ to be effective?
5. What is the *prompt-injection* analogue of a malicious LoRA, and can D2L's hypernet itself be attacked?
6. Privacy: can a per-request LoRA at inference leak into a concurrent / subsequent batched request?
7. How does $X$-saturation curve interact with model scale — does larger $W$ shift the saturation point left (need less $X$)?

---

## 6. References <a id="refs"></a>

### Inference-time training, TTT, TTA
- Sun, Y., Wang, X., Liu, Z., Miller, J., Efros, A., & Hardt, M. (2020). **Test-time training with self-supervision for generalization under distribution shifts.** ICML.
- Sun, Y., Li, X., Dalal, K., Xu, J., Vikram, A., Zhang, G., Dubois, Y., Chen, X., Wang, X., Koyejo, S., et al. (2024). **Learning to (Learn at Test Time): RNNs with Expressive Hidden States.** arXiv:2407.04620.
- Wang, D., Shelhamer, E., Liu, S., Olshausen, B., & Darrell, T. (2021). **Tent: Fully test-time adaptation by entropy minimization.** ICLR.
- Zhang, M., Levine, S., & Finn, C. (2022). **MEMO: Test time robustness via adaptation and augmentation.** NeurIPS.
- Wang, Y., Ma, D., & Cai, D. (2024). **With greater text comes greater necessity: Inference-time training helps long text generation.** COLM.

### Hypernetworks / context → weights
- Ha, D., Dai, A., & Le, Q. V. (2016). **Hypernetworks.** arXiv:1609.09106.
- Jaegle, A., Gimeno, F., Brock, A., Vinyals, O., Zisserman, A., & Carreira, J. (2021). **Perceiver: General Perception with Iterative Attention.** ICML.
- Charakorn, R., Cetin, E., Uesaka, S., & Lange, R. T. (2026). **Doc-to-LoRA: Learning to Instantly Internalize Contexts.** arXiv:2602.15902. ([code](https://github.com/SakanaAI/doc-to-lora))
- Charakorn, R., Cetin, E., Tang, Y., & Lange, R. T. (2025). **Text-to-LoRA: Instant Transformer Adaption.** ICML.
- Chen, T., Fang, H., Xia, P., Liu, X., et al. (2025). **Generative Adapter: Contextualizing Language Models in Parameters with a Single Forward Pass.** ICLR.
- Phang, J., Mao, Y., He, P., & Chen, W. (2023). **HyperTuning: Toward adapting large language models without back-propagation.** ICML.
- Ivison, H., Bhagia, A., Wang, Y., Hajishirzi, H., & Peters, M. E. (2023). **HINT: Hypernetwork instruction tuning.** ACL.
- Lv, C., Li, L., et al. (2024). **HyperLoRA: Efficient cross-task generalization via constrained low-rank adapters generation.** ACL Findings.
- Zhao, D., Kobayashi, S., Sacramento, J., & von Oswald, J. (2020). **Meta-learning via hypernetworks.** NeurIPS MetaLearn Workshop.
- von Oswald, J., Henning, C., Grewé, B. F., & Sacramento, J. (2020). **Continual learning with hypernetworks.** ICLR.

### Context distillation / prompt internalization
- Askell, A., Bai, Y., et al. (2021). **A general language assistant as a laboratory for alignment.** arXiv:2112.00861.
- Snell, C. V., Klein, D., & Zhong, R. (2023). **Learning by distilling context.**
- Padmanabhan, S., Onoe, Y., et al. (2023). **Propagating knowledge updates to LMs through distillation.** NeurIPS.
- Bhargava, A., Witkowski, C., Detkov, A., & Thomson, M. (2024). **Prompt baking.** arXiv:2409.13697.
- Caccia, L., Ansell, A., Ponti, E., Vulić, I., & Sordoni, A. (2025). **Training plug-and-play knowledge modules with deep context distillation.** COLM.
- Eyuboglu, S., Ehrlich, R., Arora, S., et al. (2025). **Cartridges: Lightweight and general-purpose long context representations via self-study.** arXiv:2506.06266.
- Kujanpää, K., Marttinen, P., Valpola, H., & Ilin, A. (2025). **Efficient knowledge injection in LLMs via self-distillation.** TMLR.
- Qi, S., Yang, B., et al. (2025). **In-context editing: Learning knowledge from self-induced distributions.** ICLR.
- Shin, H., Ji, L., et al. (2025). **Generative prompt internalization.** NAACL.
- Choi, E., Jo, Y., Jang, J., Jang, J., & Seo, M. (2023). **Fixed input parameterization for efficient prompting.** ACL Findings.

### Prompt compression / token-space compression
- Mu, J., Li, X., & Goodman, N. (2024). **Learning to compress prompts with gist tokens.** NeurIPS.
- Pan, Z., Wu, Q., et al. (2024). **LLMLingua-2: Data distillation for efficient and faithful task-agnostic prompt compression.** ACL Findings.
- Chevalier, A., Wettig, A., Ajith, A., & Chen, D. (2023). **Adapting language models to compress contexts.** EMNLP.
- Zhang, P., Liu, Z., Xiao, S., et al. (2025). **Long context compression with activation beacon.** ICLR.
- Cao, B., Cai, D., & Lam, W. (2025). **InfiniteICL: Breaking the limit of context window size via long short-term memory transformation.** arXiv:2504.01707.
- Li, X. L. & Liang, P. (2021). **Prefix-tuning: Optimizing continuous prompts for generation.** ACL.
- Li, Y., Ma, X., Lu, S., Lee, K., Liu, X., & Guo, C. (2024). **MEND: Meta demonstration distillation for efficient and effective in-context learning.** ICLR.

### Model editing / continual learning / memory
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). **LoRA: Low-rank adaptation of large language models.** ICLR.
- Fang, J., Jiang, H., et al. (2025). **AlphaEdit: Null-space constrained model editing for language models.** ICLR.
- Lin, J., Zettlemoyer, L., et al. (2025). **Continual learning via sparse memory finetuning.** arXiv:2510.15103.
- Mitchell, E., Lin, C., Bosselut, A., Finn, C., & Manning, C. D. (2022). **Fast model editing at scale.** ICLR.
- Bourtoule, L., et al. (2021). **Machine unlearning.** IEEE S&P.

### Inference-time compute / reasoning
- OpenAI (2024). **Introducing OpenAI o1-preview.**
- DeepSeek-AI (2025). **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.** arXiv:2501.12948.
- Zelikman, E., Wu, Y., Mu, J., & Goodman, N. D. (2022). **STaR: Bootstrapping Reasoning With Reasoning.** NeurIPS.
- Snell, C., Lee, J., Xu, K., & Kumar, A. (2024). **Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters.** arXiv:2408.03314.
- Lin, K., et al. (2025). **Sleep-time compute: Beyond inference scaling at test time.**

### Long context limitations
- Liu, N. F., Lin, K., Hewitt, J., et al. (2024). **Lost in the middle: How language models use long contexts.** TACL.
- Hsieh, C.-P., Sun, S., et al. (2024). **RULER: What's the real context size of your long-context language models?** COLM.
- Li, T., Zhang, G., et al. (2025). **Long-context LLMs struggle with long in-context learning.** TMLR.
- Hong, K., Troynikov, A., & Huber, J. (2025). **Context rot: How increasing input tokens impacts LLM performance.** Chroma Tech Report.
- Ye, T., Dong, L., et al. (2025). **Differential transformer.** ICLR.

### LoRA serving / weight-space
- Sheng, Y., Cao, S., et al. (2024). **S-LoRA: Serving thousands of concurrent LoRA adapters.** MLSys.
- Olah, C., Turner, N. L., & Conerly, T. (2025). **A toy model of interference weights.** transformer-circuits.pub.

### Personalization / Surveys
- Zhang, Z., Rossi, R. A., et al. (2025). **Personalization of large language models: A survey.** TMLR.
- Shi, H., Xu, Z., et al. (2025). **Continual learning of large language models: A comprehensive survey.** ACM Comput. Surv.

---

*This is a living document. PRs / issues welcome.*
