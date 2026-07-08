# Knowledge mother nest

> Running **chronological intake log** of papers, blogs, codebases, and conversations that seed ideas in this repo.
> Every entry has: ID, citation, why it matters here, and the tags that connect it to topics.
>
> Distinct from [`../../known/`](../../known/) — that folder is the *curated, by-category* knowledge base.
> Entries here are referenced by `[id]` from `known/<category>/README.md` files.

Format:
```
## [id] Short title — Author, Year
- **Type**: paper / blog / code / talk / conversation
- **Link**: …
- **Why it matters**: 1–2 sentences in our voice
- **Used in**: list of notes / plans that reference this
- **Tags**: #topic-1 #topic-2
```

---

## Inference-time training & TTT

### [d2l] Doc-to-LoRA: Learning to Instantly Internalize Contexts — Charakorn et al., Sakana AI, 2026
- **Type**: paper
- **Link**: https://arxiv.org/abs/2602.15902 · [code](https://github.com/SakanaAI/doc-to-lora) · [project](https://pub.sakana.ai/doc-to-lora/)
- **Why it matters**: Direct proof that a hypernetwork can produce a useful LoRA from a document in <1s. Establishes the "amortize the slow update into a forward pass" pattern. Sets a baseline + limitations (interference, add-only, perf gap) that several of our ideas attack.
- **Used in**: `notes/ideas/inference-time-training.md`, plan 08, plan 01 (motivation)
- **Tags**: #inference-time-training #hypernetwork #context-distillation #lora

### [t2l] Text-to-LoRA: Instant Transformer Adaption — Charakorn et al., ICML 2025
- **Type**: paper
- **Link**: https://openreview.net/forum?id=zWskCdu3QA
- **Why it matters**: Sibling to D2L. Maps natural-language task descriptions → LoRA. Confirms that hypernet output of usable weights generalizes beyond docs.
- **Used in**: plan 08
- **Tags**: #hypernetwork #adaptation #lora

### [cartridges] Cartridges: Lightweight and general-purpose long context representations — Eyuboglu et al., 2025
- **Type**: paper
- **Link**: https://arxiv.org/abs/2506.06266
- **Why it matters**: Trains a small **KV cache** per corpus via "self-study" — synthetic Q&A + KL distillation. Naive NTP on the corpus fails; the self-study recipe is the contribution. Matches ICL at **38.6× less memory**, **26.4× throughput**. Extends 128K → 484K effective tokens (MTOB). Cartridges compose at inference without retraining. Direct comparable baseline for plan 01 and plan 08; the self-study Q&A pipeline is directly transferable to v0's teacher-student training. Read in detail 2026-05-28.
- **Used in**: matrix/2026-05-26, matrix/2026-05-28; plan 08 v0 training recipe.
- **Tags**: #context-distillation #kv-cache #sleep-time-compute #long-context

### [sleep] Sleep-time Compute: Beyond Inference Scaling at Test-Time — Lin, K. et al., 2025
- **Type**: paper
- **Why it matters**: Names the "compute between turns" regime that plan 08 lives in.
- **Tags**: #inference-time-compute #self-improvement

### [ttt-layers] Learning to (Learn at Test Time): RNNs with Expressive Hidden States — Sun, Li, et al., 2024
- **Type**: paper
- **Link**: arXiv:2407.04620
- **Why it matters**: Revives TTT as an architectural primitive. Important reference for any inference-time-gradient method.
- **Tags**: #ttt #architecture

### [star] STaR: Bootstrapping Reasoning With Reasoning — Zelikman et al., NeurIPS 2022
- **Type**: paper
- **Why it matters**: Original "verifier-validated CoT → SFT" loop. Plan 08's RL signal is a direct descendant.
- **Tags**: #self-improvement #reasoning

### [genadapter] Generative Adapter: Contextualizing Language Models in Parameters with a Single Forward Pass — Chen et al., ICLR 2025
- **Type**: paper / code
- **Link**: arXiv:2411.05877 · [code](https://github.com/chentong0/generative-adapter)
- **Why it matters**: Bi-linear hypernet generates a **LoRA delta** per context chunk; key recurrence $S_t = S_{t-1} + A_2 H_t^\top H_t B_1$ with $S_t \in \mathbb{R}^{d_r \times d_r}$, $d_r \ll d_h$ — **constant memory** across stream length. SVD normalization stabilizes training and naturally produces low-rank LoRA. Pretrain: 1B SlimPajama tokens, chunk size 1024. Mistral-7B / Llama2-7B. StreamingQA F1 19.5 (SFT) → 31.5 (genadapter) at 32K context. **Essentially plan 08's north star minus verifier gating** — the gap is now identifiable. v0 should treat as primary LoRA-side baseline. Read in detail 2026-05-28.
- **Used in**: matrix/2026-05-26, matrix/2026-05-28; plan 08, plan 08 v0.
- **Tags**: #hypernetwork #context-distillation #lora #recurrent-state

### [hint] HINT: Hypernetwork instruction tuning — Ivison et al., ACL 2023
- **Type**: paper
- **Why it matters**: Early hypernet for instruction-following. Foundational for plan 08.
- **Tags**: #hypernetwork #instruction-tuning

### [mend-edit] Fast Model Editing at Scale — Mitchell et al., ICLR 2022
- **Type**: paper
- **Why it matters**: Hypernet for *editing*. Different goal from D2L but same machinery.
- **Tags**: #model-editing #hypernetwork

### [alphaedit] AlphaEdit: Null-space constrained model editing — Fang et al., ICLR 2025
- **Type**: paper
- **Why it matters**: Method for editing weights *without* breaking other capabilities. Direct mitigation for the interference problem (D2L Table 8 / plan 08 risk).
- **Tags**: #model-editing #capacity-preserving

### [sparse-mem] Continual learning via sparse memory finetuning — Lin, J. et al., 2025
- **Type**: paper
- **Link**: arXiv:2510.15103
- **Why it matters**: Update only a small subset of MLP rows. Plan 08 should consider using this as the W-parameterization.
- **Tags**: #continual-learning #sparse-update

### [ttt-2020] Test-Time Training with Self-Supervision for Generalization under Distribution Shifts — Sun, Y. et al., ICML 2020
- **Type**: paper
- **Link**: arXiv:1909.13231
- **Why it matters**: Original TTT paper. Cited from `known/test-time-training` and `known/inference-time-training`.
- **Tags**: #ttt #classic

### [tent] Tent: Fully Test-time Adaptation by Entropy Minimization — Wang, D. et al., ICLR 2021
- **Type**: paper
- **Why it matters**: Entropy-minimization variant of TTT. Reference baseline for any "unsupervised inference-time update" method.
- **Tags**: #ttt #test-time-adaptation

### [memo] MEMO: Test time robustness via adaptation and augmentation — Zhang, M., Levine, S., Finn, C., NeurIPS 2022
- **Type**: paper
- **Why it matters**: Marginal entropy on augmented copies of a test input. Cousin of TENT.
- **Tags**: #ttt #augmentation

### [von-oswald-2020] Continual learning with hypernetworks — von Oswald, J. et al., ICLR 2020
- **Type**: paper
- **Why it matters**: One hypernet conditioned on task ID across a stream. Architectural precedent for plan 08's $H_\phi$ producing per-turn $\Delta W$.
- **Tags**: #hypernetwork #continual-learning

## Hypernet machinery (architectural roots)

### [ha-2016] HyperNetworks — Ha, D., Dai, A., Le, Q., 2016
- **Type**: paper
- **Link**: arXiv:1609.09106
- **Why it matters**: The original. Cited as the foundational paper from `known/hypernetworks`.
- **Tags**: #hypernetwork #classic

### [hypertune] HyperTuning: Toward Adapting Large Language Models without Back-propagation — Phang, J. et al., ICML 2023
- **Type**: paper
- **Why it matters**: Task-conditioned hypernet for LLM adaptation without test-time backprop. Direct ancestor of D2L / T2L architecturally.
- **Tags**: #hypernetwork #adaptation

### [mend-demo] MEND (Meta-Demonstration Distillation) — Li, Y. et al., ICLR 2024
- **Type**: paper
- **Why it matters**: Hypernet that compresses few-shot demonstrations into a usable representation. Sibling pattern to D2L for *demos* rather than docs.
- **Tags**: #hypernetwork #demo-compression

### [gisting] Learning to Compress Prompts with Gist Tokens — Mu, J., Li, X., Goodman, N., NeurIPS 2024
- **Type**: paper
- **Link**: arXiv:2304.08467
- **Why it matters**: LM is its own gist predictor — no separate hypernet. Insert `k` gist tokens between prompt and input; attention-mask trick forces information into gist activations. **No additional training cost** over instruction tuning. Up to **26× prompt compression**, 40% FLOPs reduction. **Designed for short prompts (≤ 30 tokens), not long context** — soft-prompt ceiling. Baseline that v0 must beat to justify the wrapper architecture. Read in detail 2026-05-28.
- **Used in**: matrix/2026-05-26, matrix/2026-05-28; plan 08 v0 baseline.
- **Tags**: #hypernetwork #prompt-compression #attention-masking

## Inference-time compute (X-axis)

### [o1] OpenAI o1 — 2024
- **Type**: model release
- **Why it matters**: Defined the "scale test-time compute via reasoning" narrative. Our framing extends this to the W-axis.
- **Tags**: #inference-time-compute #reasoning

### [r1] DeepSeek-R1 — DeepSeek-AI, 2025
- **Type**: paper
- **Link**: arXiv:2501.12948
- **Why it matters**: Open-weights replication of o1-style RL on reasoning traces. Useful baseline for plan 01.
- **Tags**: #rl #reasoning

### [scaling-tt] Scaling LLM Test-Time Compute Optimally — Snell et al., 2024
- **Type**: paper
- **Link**: arXiv:2408.03314
- **Why it matters**: Quantifies the *exchange rate* between X-budget and model size. Direct ancestor of plan 01's framing.
- **Tags**: #inference-time-compute #scaling-laws

### [self-cons] Self-Consistency Improves Chain of Thought Reasoning in Language Models — Wang, X., Wei, J. et al., 2022
- **Type**: paper
- **Link**: arXiv:2203.11171
- **Why it matters**: The original Best-of-N along $X$. Plan 03 must beat or complement this baseline.
- **Tags**: #inference-time-compute #best-of-n

### [bon-prm] Let's Verify Step by Step (PRM800k) — Lightman, H. et al., 2023
- **Type**: paper / dataset
- **Link**: arXiv:2305.20050
- **Why it matters**: Process Reward Model BoN — the gold-standard X-axis BoN baseline for plan 03 and the verifier substrate for plan 01.
- **Tags**: #verifier #prm #best-of-n

### [tot] Tree of Thoughts — Yao, S. et al., 2023
- **Type**: paper
- **Link**: arXiv:2305.10601
- **Why it matters**: Search-based X-axis reasoning. Comparator for plan 03 when extending BoN to richer search.
- **Tags**: #search #reasoning

### [restmcts] ReST-MCTS\* — Chen, Z. et al., 2024
- **Type**: paper
- **Why it matters**: Tree search + reward model for self-improving reasoning. Bridge between BoN and full search.
- **Tags**: #search #self-improvement

### [alphacode] AlphaCode — Li, Y. et al., 2022
- **Type**: paper / system
- **Why it matters**: Massive sampling + filtering for code. Demonstrates the asymptotic of X-axis BoN; plan 03 wants to undercut its FLOPs.
- **Tags**: #best-of-n #code

### [math-shep] Math-Shepherd: Verify and Reinforce LLMs Step-by-Step without Human Annotations — Wang, Y. et al., 2024
- **Type**: paper
- **Why it matters**: Process reward model trained without human step labels. The cheap-PRM channel for plan 03's verifier.
- **Tags**: #verifier #prm #math

## Self-improvement (training loops on model-generated data)

### [self-refine] Self-Refine: Iterative Refinement with Self-Feedback — Madaan, A. et al., NeurIPS 2023
- **Type**: paper
- **Why it matters**: Model critiques and rewrites its own outputs. Prompt-side analog of plan 08's `<learn>` loop.
- **Tags**: #self-improvement #self-critique

### [reflexion] Reflexion: Language Agents with Verbal Reinforcement Learning — Shinn, N. et al., NeurIPS 2023
- **Type**: paper
- **Why it matters**: Verbal RL via self-reflection on episode outcomes. Prompt-side ancestor of plan 08's verifier-gated `<learn>` segment.
- **Tags**: #self-improvement #agent

### [vstar] V-STaR: Training Verifiers for Self-Taught Reasoners — Hosseini, A. et al., 2024
- **Type**: paper
- **Why it matters**: Adds a verifier on top of STaR. Closer to plan 08's verifier-gated updates than plain STaR.
- **Tags**: #self-improvement #verifier

### [restem] Beyond Human Data: Scaling Self-Training (ReST^EM) — Singh, A. et al., 2023
- **Type**: paper
- **Why it matters**: EM-style self-training loop. Operational template for plan 01's "train on the W-residual" iteration.
- **Tags**: #self-improvement #self-training

### [self-reward] Self-Rewarding Language Models — Yuan, W. et al., 2024
- **Type**: paper
- **Why it matters**: Model produces both responses and reward judgments. Plausible inner loop for plan 08's $\alpha_t$ gating signal.
- **Tags**: #self-improvement #reward-model

### [dpo] Direct Preference Optimization — Rafailov, R. et al., NeurIPS 2023
- **Type**: paper
- **Why it matters**: Pairwise preference → policy update. Likely loss surface for plan 08's `<learn>`-emit training.
- **Tags**: #rlhf #preference

## Data selection / curriculum (plan 01 substrate)

### [deita] What Makes Good Data for Alignment? — Liu, W., Zeng, W. et al., 2024
- **Type**: paper
- **Link**: arXiv:2312.15685
- **Why it matters**: Complexity × quality × diversity selection. Plan 01's main "data-selection-as-heuristic" baseline.
- **Tags**: #data-selection #instruction-tuning

### [less] LESS: Selecting Influential Data for Targeted Instruction Tuning — Xia, M., Malladi, S. et al., 2024
- **Type**: paper
- **Link**: arXiv:2402.04333
- **Why it matters**: Influence-function based data selection. Direct comparator for plan 01.
- **Tags**: #data-selection #influence

### [calib] Language Models (Mostly) Know What They Know — Kadavath, S. et al., 2022
- **Type**: paper
- **Why it matters**: Models can self-estimate whether more X will help — informs the $\epsilon$ threshold for $X$-saturation detection in plan 01.
- **Tags**: #calibration #self-prediction

## Model editing (the closest W-update prior)

### [rome] Locating and Editing Factual Associations in GPT (ROME) — Meng, K. et al., NeurIPS 2022
- **Type**: paper
- **Why it matters**: Locate-then-edit foundational paper. Cited from `known/model-editing`. Methodological ancestor for capacity-aware $\Delta W$ in plan 08.
- **Tags**: #model-editing #classic

### [memit] Mass-Editing Memory in a Transformer (MEMIT) — Meng, K. et al., 2023
- **Type**: paper
- **Why it matters**: Scales ROME to thousands of edits at once. The "many small edits" regime plan 08 enters at deployment.
- **Tags**: #model-editing #scaling

### [grace] GRACE: Continual editing via adaptive codebook — Hartvigsen, T. et al., NeurIPS 2023
- **Type**: paper
- **Why it matters**: Continual editing without retraining the editor. Sequential-edit stability reference.
- **Tags**: #model-editing #continual

## Theory (X-W, ICL, calibration)

### [icl-bayes] An Explanation of In-context Learning as Implicit Bayesian Inference — Xie, S. et al., 2022
- **Type**: paper
- **Link**: arXiv:2111.02080
- **Why it matters**: Theoretical bridge between ICL and Bayesian updates — relevant for the "when does ICL = TTT" debate.
- **Tags**: #theory #icl

## Agentic memory & personalization (plan 08 application surface)

### [memgpt] MemGPT: Towards LLMs as Operating Systems — Packer, C. et al., 2023
- **Type**: paper / system
- **Why it matters**: Prompt-side memory tiering. The bolt-on alternative plan 08 wants to replace with native weight updates.
- **Tags**: #agentic-memory #long-context-alt

### [voyager] Voyager: An Open-Ended Embodied Agent with Large Language Models — Wang, G. et al., 2023
- **Type**: paper
- **Why it matters**: Skill-library agent — *prompt-side* analog of self-modifying weights. Reference target for plan 08 Phase 3.
- **Tags**: #agent #self-improvement

### [a-mem] A-MEM: Agentic Memory for LLM Agents — Xu, F. et al., 2025
- **Type**: paper
- **Why it matters**: Modern agent-memory architecture survey + system. Comparator for plan 08's "memory in weights" claim.
- **Tags**: #agentic-memory

### [lamp] LaMP: When Large Language Models Meet Personalization — Salemi, A. et al., 2024
- **Type**: paper / benchmark
- **Why it matters**: Primary personalization benchmark for plan 08 Phase 2.
- **Tags**: #benchmark #personalization

### [perltqa] PerLTQA: Long-Term Memory in Personalized Dialogue — Du, Y. et al., 2024
- **Type**: paper / benchmark
- **Why it matters**: Long-term personalized QA. Phase-2 channel for plan 08.
- **Tags**: #benchmark #personalization #long-term-memory

## Long context & memory

### [lost-mid] Lost in the Middle — Liu et al., TACL 2024
- **Type**: paper
- **Why it matters**: Foundational evidence that more X is not free. Justifies the search for W-side alternatives.
- **Tags**: #long-context #limitations

### [ruler] RULER: What's the real context size of your long-context LLMs? — Hsieh et al., COLM 2024
- **Type**: paper / benchmark
- **Why it matters**: Validation channel for plan 01 and plan 08 (long-context). Goes beyond NIAH.
- **Tags**: #benchmark #long-context

### [context-rot] Context rot: How increasing input tokens impacts LLM performance — Hong et al., Chroma 2025
- **Type**: tech report
- **Why it matters**: Empirical case for why we can't just keep growing X.
- **Tags**: #long-context

### [act-beacon] Long Context Compression with Activation Beacon — Zhang, P. et al., 2024
- **Type**: paper / system
- **Link**: arXiv:2401.03462 · [code](https://github.com/FlagOpen/FlagEmbedding/)
- **Why it matters**: Argues soft-token compression (Gisting, ICAE, AutoCompressor) is the *bottleneck* for long-context, and proposes compressing into KV activations at every layer via a `<b>` beacon token. Chunked progressive workflow, random compression-ratio training. 8× KV cache reduction, 2× speedup; 128K tested at 20K train. Direct architectural counter-proposal to plan 08 v0's soft-token default. Read in detail 2026-05-28.
- **Used in**: matrix/2026-05-28; plan 08 v0 design ablation (proposed J5).
- **Tags**: #long-context #compression #ttt-adjacent #plan08-v0

### [longbench] LongBench: A bilingual, multitask benchmark for long context — Bai et al., 2023
- **Type**: benchmark
- **Why it matters**: Validation channel for any "internalize a document" idea.
- **Tags**: #benchmark #long-context

## Parameterization / serving

### [lora] LoRA: Low-rank adaptation — Hu et al., ICLR 2022
- **Type**: paper
- **Why it matters**: Substrate for all plans. Read once, cite forever.
- **Tags**: #peft #lora

### [s-lora] S-LoRA: Serving thousands of concurrent LoRA adapters — Sheng et al., MLSys 2024
- **Type**: paper / system
- **Why it matters**: Serving infra prior. Plan 08 inherits constraints from here.
- **Tags**: #serving #lora

### [hyperlora] HyperLoRA: Efficient cross-task generalization via constrained low-rank adapters generation — Lv et al., ACL 2024
- **Type**: paper
- **Tags**: #hypernetwork #lora

### [punica] Punica: Multi-Tenant LoRA Serving — Chen, L. et al., 2023
- **Type**: paper / system
- **Why it matters**: Companion to S-LoRA for serving many LoRAs concurrently. Substrate for any WBoN deployment (plan 03).
- **Tags**: #serving #lora

### [lorahub] LoRAHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition — Huang, C. et al., 2023
- **Type**: paper
- **Link**: arXiv:2307.13269
- **Why it matters**: Training-time LoRA combination. Closest spiritual cousin to plan 03's WBoN-Library stage.
- **Tags**: #lora #composition

### [mole] MoLE: Mixture of LoRA Experts — Wu, X. et al., 2024
- **Type**: paper
- **Why it matters**: Routing among LoRAs at inference. Plan 03 contrasts: *sample* + verify rather than *route*.
- **Tags**: #lora #moe

### [lots-of-loras] Lots-of-LoRAs (dataset) — community / HuggingFace
- **Type**: dataset
- **Link**: https://huggingface.co/datasets/Lots-of-LoRAs/Lots-of-LoRAs
- **Why it matters**: ~479 task-specific LoRAs ready to use. Off-the-shelf $\{\Delta W_i\}$ library for plan 03's WBoN-Library stage.
- **Tags**: #dataset #lora

## Verifiers / benchmarks

### [gsm8k] GSM8K — Cobbe et al., 2021
- **Type**: benchmark
- **Why it matters**: Cheap, rule-based verifier. Primary channel for plan 01 + plan 03 pilots.
- **Tags**: #benchmark #math #verifier

### [math] MATH — Hendrycks et al., 2021
- **Type**: benchmark
- **Why it matters**: Harder math; better signal for X-saturation curve.
- **Tags**: #benchmark #math

### [humaneval+] HumanEval+ / EvalPlus — Liu et al., 2023
- **Type**: benchmark
- **Why it matters**: Execution-based verifier for code. Channel for plan 03.
- **Tags**: #benchmark #code #verifier

### [swebench] SWE-Bench — Jimenez et al., 2024
- **Type**: benchmark
- **Why it matters**: Agentic, repo-scale code. Channel for Code-D2L (idea E1).
- **Tags**: #benchmark #code #agentic

### [arena-hard] Arena-Hard — Li et al., 2024
- **Type**: benchmark
- **Why it matters**: General quality. Use only after rule-based channels are saturated.
- **Tags**: #benchmark #open-ended

## 2026 prior-work landscape (added 2026-05-28)

These five clusters were scanned during the 2026-05-28 brainstorm. They define the field's current frontier and directly affect the uniqueness column of our idea evaluation.

### TTT for LLMs (2026 cluster)

### [in-place-ttt] In-Place Test-Time Training — Anonymous, ICLR 2026 Oral
- **Type**: paper
- **Link**: arXiv:2604.06169
- **Why it matters**: Treats MLP final projection matrix as fast weights — drop-in TTT enhancement for pretrained LLMs. Replaces generic reconstruction with NTP-aligned objective. Chunk-wise update, context-parallel compatible. 4B model handles 128K context. **Directly preempts plan 08's "what to update" and J5's parameterization choice.** Plan 08 must explicitly differentiate.
- **Tags**: #ttt #fast-weights #plan08-prior

### [tempo] TEMPO: Scaling Test-time Training for Large Reasoning Models — 2026
- **Type**: paper
- **Link**: arXiv:2604.19295
- **Why it matters**: TTT for reasoning models via EM — alternating M-step policy refinement with E-step critic recalibration on labeled rewards. OLMO3-7B on AIME 2024: 33.0% → 51.1%. **Partially preempts J2 (verifier-gated genadapter) and any "RL-style inference-time learning" idea.** Our angle must be either non-reasoning, single-pass, or recurrent-state.
- **Tags**: #ttt #rl #reasoning #plan03-prior

### [lact] Large Chunk Test-Time Training (LaCT) — ICLR 2026
- **Type**: paper
- **Why it matters**: 1M-token chunk updates; high hardware utilization. Generalizes TTT across modalities. Frames "chunk size as a state-capacity / hardware-utilization trade-off". Useful baseline for plan 08 v0's chunk-size sweep.
- **Tags**: #ttt #scaling #efficiency

### [ttc-rl] Test-Time Curriculum RL — ICLR 2026
- **Type**: paper
- **Why it matters**: Model auto-selects task-relevant data at inference and trains on it. **Adjacent to plan 01's X-saturation curriculum** — but TTC-RL does it online during inference, plan 01 offline as data curation.
- **Tags**: #ttt #curriculum #plan01-adjacent

### Sequential editing stability (2026 cluster — now crowded)

### [mose] Multiplicative Orthogonal Sequential Editing — AAAI 2026
- **Type**: paper
- **Why it matters**: Replaces additive $W' = W_0 + \Delta W$ with **multiplicative** $W' = R \cdot W_0$ where $R^\top R = I$ — strictly preserves Frobenius norm and condition number. 12.08% sequential-editing improvement, 95.73% general-capability retention. **Cleanest formulation of "edits without drift" — direct alternative to AlphaEdit / additive null-space methods.** Strong candidate for plan 08 north-star ΔW parameterization.
- **Tags**: #model-editing #lifelong #stability #plan08-relevant

### [crispedit] CRISPEDIT: Curvature-Restricted In-Situ Parameter Editing — 2026
- **Type**: paper
- **Why it matters**: Gauss-Newton Hessian eigenspace identifies low-curvature directions where capability loss is invariant. K-FAC + matrix-free projector for LLM-scale efficiency. Bregman-divergence constraint avoids base-convergence requirement. **Most principled current answer to plan 08's "what update directions are safe" question.**
- **Tags**: #model-editing #curvature #k-fac #stability

### [rlsedit] RLSEdit: Recursive Least-Squares for Lifelong Editing — 2026
- **Type**: paper
- **Link**: arXiv:2601.15686
- **Why it matters**: Online quadratic optimization via Woodbury identity. Per-edit cost independent of history length; scales with current edit rank only. Soft constraints (deviation from pre-trained + anchor mapping). Has deviation bounds. **The mathematical model of "lifelong edits" most reusable for plan 08's verifier-gated $\Delta W$ stream.**
- **Tags**: #model-editing #online #woodbury

### [revive] REVIVE: Dominant Singular Subspace Protection — 2026
- **Type**: paper
- **Link**: arXiv:2601.11042
- **Why it matters**: Identifies that model collapse correlates with cumulative corruption of dominant singular subspace. Plug-and-play filter removes update components interfering with top singular vectors. **Scales to 20,000 sequential edits on LLaMA-3.**
- **Tags**: #model-editing #svd #lifelong

### [stable-edit] StableEdit: LN as a self-reinforcing stability loop — 2026
- **Type**: paper
- **Link**: arXiv:2605.11836
- **Why it matters**: Theoretical analysis of why LayerNorm-based normalization stabilizes lifelong edits. Refines ULTRAEDIT with warm-up + full whitening. **Asymptotically orthogonal updates, bounded norms — million-scale streams.** The current best-understood "why LN works for editing" result.
- **Tags**: #model-editing #ln #theory

### [beta-edit] BetaEdit: Null-Space Constrained Sequential Editing — 2026
- **Type**: paper
- **Link**: arXiv:2605.09285
- **Why it matters**: Reintroduces penalty term to compensate for pseudo-null-space imperfection in AlphaEdit-style methods. History-aware updates. **10,000 sequential edits stable.** Extends the null-space line.
- **Tags**: #model-editing #null-space #history-aware

### [ultra-edit] ULTRAEDIT — Gu et al., 2026
- **Type**: paper
- **Why it matters**: LN-style normalization achieves long-horizon stability. The model that StableEdit theoretically explains. Million-scale edit streams.
- **Tags**: #model-editing #ln #scaling

### Hypernet → LoRA (still expanding)

### [shine] SHINE: Scalable In-Context Hypernetwork for Context → LoRA — 2026
- **Type**: paper
- **Link**: arXiv:2602.06358
- **Why it matters**: Transformer-based hypernet via self-attention; M2P (memory-to-parameter) transformer with alternating row/column bidirectional attention. Generates layer-specific LoRA from base LM's own internal representations. **Most expressive in-context hypernet of 2026; sits between [genadapter] and plan 08 north star.** Plan 08 north star reduces to "[shine] + verifier gating" at first order.
- **Tags**: #hypernetwork #lora #context

### [ouroboros] OUROBOROS: Controller hypernet for recursive transformers — 2026
- **Type**: paper
- **Link**: arXiv:2604.02051
- **Why it matters**: 9.2M-param Controller hypernet modulates frozen **SVD-initialized** LoRA bases recurrently. Per-step diagonal modulation. Replaces standard residual with gated recurrence (88% retention init). 43.4% loss reduction on 17-layer Qwen2.5-3B. Architectural cousin to genadapter; suggests "freeze the LoRA bases, only train the modulator" as a third design point alongside our v0 wrapper and Genadapter.
- **Tags**: #hypernetwork #lora #recurrent

### LoRA merging interference (2026)

### [pico] Pico: Pre-merge Interference Calibration in Output-space — 2026
- **Type**: paper
- **Link**: arXiv:2604.16826
- **Why it matters**: Identifies the output-side matrix B of LoRA as the interference source; calibrates B (data-free) before merge. Plugs into Task Arithmetic, TIES, TSV-M. Relevant if plan 03 (W-space BoN) wants to compose its $\{\Delta W_i\}$ rather than search.
- **Tags**: #lora #merging #calibration

### [iso-c] Isotropic Merging — Marczak et al., 2025
- **Type**: paper
- **Why it matters**: SVD on summed task vectors, equalize singular values. Spectral harmonization of merged models. Baseline for any merging-based plan 03 variant.
- **Tags**: #merging #svd

### Representation engineering (matured into a field)

### [repe-survey] Representation Engineering for LLMs — Survey, 2025
- **Type**: survey
- **Link**: arXiv:2502.17601
- **Why it matters**: Comprehensive map of activation-steering family. Establishes vocabulary (Representation Reading vs Representation Control). **The starting point for any RepE work** — read before B2 (steering vectors) or I5 (prompt↔LoRA equivalence).
- **Tags**: #repe #survey

### [odesteer] ODESteer: ODE-Based Activation Steering via Barrier Functions — 2026
- **Type**: paper
- **Link**: arXiv:2602.17560
- **Why it matters**: First principled framework for activation steering as ODE solution; steering direction = barrier function from control theory. Multi-step adaptive steering. +5.7% TruthfulQA. **Promotes activation steering from heuristic to theoretical.** Closest path to making I5 / G2 / B2 theory-respectable.
- **Tags**: #repe #theory #ode

### [w-a-equiv] Weight-Activation Equivalence Mapping — 2026
- **Type**: paper
- **Link**: arXiv:2603.00425
- **Why it matters**: First-order equivalence between weight-space updates and activation-space interventions. Identifies **post-block output** as theoretically grounded highly expressive intervention point. Steering accuracy within 0.2%-0.9% of full SFT while training only 0.04% of params. **Operationalizes I5 (Prompt ↔ LoRA equivalence) — and immediately suggests J3 (hybrid X+W router) can be reframed as a weight↔activation router with theoretical backing.**
- **Tags**: #repe #theory #peft #plan08-relevant

### [austeer] AUSteer: Fine-grained Attribute Unit steering — 2026
- **Type**: paper
- **Why it matters**: Moves from block-level to fine-grained Attribute Unit steering. Targeted intervention on fewer activations beats coarse intervention. Reference for "what granularity should B2 use?".
- **Tags**: #repe #granularity

## Public benchmarks used in plan-08 v1 (added 2026-06-03)

These appear in `latent-mem-paper` Tables 1-2 and in `notes/plans/08-compressed-context-memory/history/v1-results-2026-06-03.md`. Stable IDs assigned now so all v1 + v2 docs can cite them consistently. Plan-08 headline numbers cited in this section use setting [`P08-S2`](../../notes/plans/08-compressed-context-memory/settings/settings.md#p08-s2--v1-phase-y-three-regime-benchmark-cells).

### [quality] QuALITY: Question Answering with Long Input Texts, Yes! — Pang, Parrish, Lal, et al., NAACL 2022
- **Type**: benchmark
- **Link**: arXiv:2112.08608 · HF `emozilla/quality`
- **Why it matters**: 4-way MC reading comprehension on ~7K-token articles. **The Regime A canonical benchmark** in plan-08 v1: lossy compression actually *helps* here because the answer is reconstructible from a gist. Phase Y headline under [`P08-S2`](../../notes/plans/08-compressed-context-memory/settings/settings.md#p08-s2--v1-phase-y-three-regime-benchmark-cells): OURS 0.193 ± 0.032 (n=4 seeds) vs no_context 0.141.
- **Used in**: plan 08 v1 paper Table 1; mem-X Phase V/X/Y
- **Tags**: #benchmark #long-context #mc

### [musr] MuSR: Multistep Soft Reasoning — Sprague, Zhang, Sanford, et al., 2024
- **Type**: benchmark
- **Link**: arXiv:2310.16049
- **Why it matters**: Closed-MC reasoning over 1000+ token narratives with multi-step temporal logic. **Regime B canonical** in plan-08 v1 under [`P08-S2`](../../notes/plans/08-compressed-context-memory/settings/settings.md#p08-s2--v1-phase-y-three-regime-benchmark-cells): both wrappers at chance (≈ 0.495), full_context wins. Shows the wrapper learned compression but no transferable reasoning signal. Forces honest "limitations" discussion.
- **Used in**: plan 08 v1 paper Table 1; mem-X Phase X/Y
- **Tags**: #benchmark #reasoning #long-context

### [ruler] (already exists) — Hsieh et al., COLM 2024

Plan-08 v1 promotes this to Regime C canonical under [`P08-S2`](../../notes/plans/08-compressed-context-memory/settings/settings.md#p08-s2--v1-phase-y-three-regime-benchmark-cells): extract one specific string from a 2K-token WikiText haystack. Both wrappers = 0.000 ± 0.000 (4 seeds, exactly zero); full_context = 0.995. Cleanest evidence for "lossy compression destroys the needle string".

### [hotpotqa] HotpotQA: A Dataset for Diverse, Explainable Multi-hop QA — Yang, Qi, Zhang, et al., EMNLP 2018
- **Type**: benchmark
- **Link**: arXiv:1809.09600 · HF `hotpotqa/hotpot_qa`
- **Why it matters**: Multi-hop QA with distractor paragraphs; free-form span answers. Plan-08 v1 used as cross-distribution generalization probe (Phase V). Free-form (token_f1) rather than MC — different metric domain from QuALITY.
- **Used in**: plan 08 v1 paper Table 1; mem-X Phase V
- **Tags**: #benchmark #multi-hop #long-context

### [narrativeqa] The NarrativeQA Reading Comprehension Challenge — Kočiský, Schwarz, Blunsom, et al., TACL 2018
- **Type**: benchmark
- **Link**: arXiv:1712.07040 · HF `deepmind/narrativeqa`
- **Why it matters**: Open-ended QA over long narratives (books / movie scripts). Generous metric (`contains_match` over multiple references). Plan-08 v1 Phase V cross-distribution probe.
- **Tags**: #benchmark #long-context #open-ended

### [triviaqa] TriviaQA: A Large Scale Distantly Supervised Challenge — Joshi, Choi, Weld, Zettlemoyer, ACL 2017
- **Type**: benchmark
- **Link**: arXiv:1705.03551
- **Why it matters**: Open-domain QA with Wikipedia evidence. Tests whether wrapper trained on synthetic NIAH transfers to real Wikipedia text. Plan-08 v1 Phase X.
- **Tags**: #benchmark #open-domain

### [msmarco] MS MARCO v2.1: A Human Generated MAchine Reading COmprehension Dataset — Nguyen, Rosenberg, Song, et al., 2016
- **Type**: benchmark
- **Link**: arXiv:1611.09268
- **Why it matters**: Passage-QA at scale. Plan-08 v1 Phase X cross-distribution probe; tests "short context, real text" failure mode.
- **Tags**: #benchmark #passage-qa

### [squad2] SQuAD v2: Know What You Don't Know — Rajpurkar, Jia, Liang, ACL 2018
- **Type**: benchmark
- **Link**: arXiv:1806.03822
- **Why it matters**: Short-context RC with unanswerable questions. Plan-08 v1 Phase X — important because most wrapper benchmarks are long-context; SQuAD v2 measures whether the wrapper hurts in the short-context regime.
- **Tags**: #benchmark #rc

### [wikitext103] WikiText-103 — Merity, Xiong, Bradbury, Socher, ICLR 2017
- **Type**: dataset
- **Link**: arXiv:1609.07843
- **Why it matters**: The haystack substrate for plan-08 v1's RULER-NIAH evaluation. Standard LM benchmark; here used as text background, not as a metric.
- **Tags**: #dataset #lm

## Long-context compression / memory architectures (related work for plan 08, added 2026-06-03)

### [icae] In-Context Auto-Encoder for Context Compression in a Large Language Model — Ge, Hu, et al., ICLR 2024
- **Type**: paper
- **Link**: arXiv:2307.06945
- **Why it matters**: Encoder-decoder with learned compression of context into a small set of memory slots, trained via reconstruction. **Closest neighbour to Direction D (infilling objective)** in plan-08 v1's pivot menu.
- **Tags**: #compression #soft-prompt #plan08-relevant

### [autocompressor] Adapting Language Models to Compress Contexts — Chevalier, Wettig, Ajith, Chen, EMNLP 2023
- **Type**: paper
- **Link**: arXiv:2305.14788
- **Why it matters**: Summary-token-style recurrent context compression. Plan-08 v1 explicitly positions itself alongside this lineage (and Gist, ICAE). Different write-side recurrence; same overall problem.
- **Tags**: #compression #soft-prompt #recurrent

### [h2o] H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs — Zhang, Sheng, Zhou, et al., NeurIPS 2023
- **Type**: paper
- **Link**: arXiv:2306.14048
- **Why it matters**: KV-cache eviction policy; the canonical reference for plan-08 v1's **Direction F (KV-cache compression instead of soft prompt)** pivot.
- **Tags**: #kv-cache #long-context

### [streamingllm] Efficient Streaming Language Models with Attention Sinks — Xiao, Tian, Chen, Han, ICLR 2024
- **Type**: paper
- **Link**: arXiv:2309.17453
- **Why it matters**: Attention-sink + sliding-window for streaming long contexts. Plan-08 v1 Direction F neighbour.
- **Tags**: #kv-cache #streaming

### [tokmem] TokMem — token-memory baseline (citation TBD, v2 target)
- **Type**: paper
- **Why it matters**: Plan-08 v2's planned **head-to-head baseline** for suffix memory. Reimplementation on top of `mem-test/mem-embedding` is one of the v2 work items (~1 focused week).
- **Tags**: #v2 #baseline #memory-paper

## RCA / transformer-intrinsic prior-work audit (added 2026-06-03)

Closest-preempting work found in the 2022-2026 lit review for the RCA brainstorm ([`notes/ideas/rca-transformer-intrinsic-2026-06-03.md`](../../notes/ideas/rca-transformer-intrinsic-2026-06-03.md) §10). Each killed one of our R1-R12 angles.

### Forgetting / capability-preserving fine-tuning

- **[oplora]** OPLoRA — Orthogonal Projection LoRA (Xiong & Xie, AAAI 2026, arXiv:2510.13003). SVD-decompose frozen W, constrain LoRA to orthogonal complement of top-k singular subspace ($P_L=I-U_kU_k^\top$, $P_R=I-V_kV_k^\top$); math+code+commonsense; LLaMA-2-7B + Qwen2.5-7B. **= our R8 verbatim.**
- **[clora]** CLoRA — Controlled LoRA with subspace/null-space regularization (ACL 2025). Orthogonal regularization on LoRA dirs. R8 neighbour.
- **[smf]** Sparse Memory Finetuning (Meta, arXiv:2510.15103) + follow-ups (2605.03229, 2604.05248). Update only memory slots high-activated by new data vs background; NQ F1 drop 89% full-FT / 71% LoRA / **11% SMF**. **= our R10 / B6 verbatim.**
- **[mofo]** MoFO — Momentum-Filtered Optimizer (arXiv:2407.20999). Update only large-momentum params → stay near pretrained. **= our R6.**
- **[migu]** MIGU — Magnitude-based Gradient Updating (arXiv:2406.17245). Task-label-free; update large-magnitude-output params. **= our R6.**
- **[mech-forget]** Mechanistic Analysis of Catastrophic Forgetting (arXiv:2601.18699). Freezing attention → forgetting ↓64%; 15-23% of lower-layer heads severely disrupted. **= our R2** (and the main threat to the unifying-observation framing).
- **[ft-no-forget-icl]** Fine-Tuning Without Forgetting In-Context Learning (arXiv:2602.23197). Theory: restricting updates to the value matrix preserves ICL. **= our R2.**
- **[abft]** Attention Behavior Fine-Tuning. Directly manipulates induction heads for ICL. **= our R2.**
- **[selfaug]** SelfAug (EMNLP 2025 findings.763). Input-logit self-distribution alignment vs forgetting. **= our R5.**
- **[tmkl]** Target-Masked KL (arXiv:2605.29498). Replay-free LoRA forgetting regularizer on non-target logits. **= our R5.**
- **[logit-lens-loss]** Logit Lens Loss (arXiv:2602.01530, VLM) · **[distill-lens]** DistillLens (arXiv:2602.13567). Internal-readout consistency aux losses. **= our R5.**
- **[sae-ft]** SAE-FT (arXiv:2605.15961, CLIP) · **[sae-fd]** SAE-FD (arXiv:2605.25525) · **[sae-tuning]** SAE-Tuning (OpenReview vUrZaERt8b). Freeze/gate SAE features to preserve capability. **= our R11.**

### Attention temperature / softmax sharpening

- **[ssa]** Selective Self-Attention (arXiv:2411.12892). Learnable per-head/token temperature on Q/V; <0.5% params; fine-tunable on existing LLMs. **= our R4.**
- **[ssmax]** Scalable-Softmax (arXiv:2501.19399). Per-layer/head learnable scalar; long-context retrieval. **= our R4.**
- **[focal-attn]** Focal Attention (arXiv:2511.06818). Learned per-layer softmax temperature. **= our R4.**

### Steering / capability addition on frozen base

- **[steer-reason]** Trained steering vectors to unlock reasoning (OpenReview URrDgCHA1i). Layer-wise additive biases, frozen weights (BitFit-like). **= our R7.**
- **[instruct-steer]** Instruction-following via activation steering (arXiv:2410.12877). Difference-in-means task directions. **= our R7.**
- **[cache-steer]** KV Cache Steering (arXiv:2507.08799). One-time KV-cache steering vectors, frozen base. **= our R7 / R3.**

### Frozen-base KV / memory adapters (long context)

- **[reasoncache]** ReasonCache (arXiv:2602.02366). Prefix-tuning KV vectors at each layer, frozen base; KV cache as learnable memory interface. **= our R3 / R1.**
- **[kv-packet]** KV Packet (arXiv:2604.13226). Trainable soft-token KV adapters wrapping document blocks, frozen base, self-supervised distillation; **explicitly fixes disrupted attention-sink boundary artifacts**. **= our R3 / R1.**
- **[rlkv]** RLKV (arXiv:2510.08525). RL-trained gating adapters identify reasoning-critical heads for KV compression. R3 neighbour.
- **[persist-mem-dec]** Trained Persistent Memory for Frozen Decoder-Only LLMs (arXiv:2603.22329) + encoder-decoder variant (2603.16413). Prefix / KV-extension / Hebbian / slot-write memory adapters, frozen base. **= our R1** (closest persistent-memory neighbour).
- **[kvm]** Key-Value Means (arXiv:2605.09877). Block-recurrent compressed memory; chunked RNN over attention state. **= our R1** (recurrent-memory neighbour).
- **[lcirc]** LCIRC (arXiv:2502.06139) · **[memcom]** MemCom (arXiv:2510.16092). Recurrent / layer-wise compression injected into a frozen LLM. R1 neighbours.

### Attention-sink science (the phenomenon R1 builds on)

- **[sink-streaming]** Xiao et al. 2023 — StreamingLLM / attention sinks (arXiv:2309.17453). Origin; learnable sink token for streaming stability.
- **[sink-emerge]** When Attention Sink Emerges (arXiv:2410.10781). Sink ≈ **key bias storing extra attention scores** — basis for the §10.2 key-bias gap.
- **[sink-first-token]** Why do LLMs attend to the first token? (arXiv:2504.02732). Sink = over-mixing avoidance.
- **[sink-ctr]** Attention Sinks: Catch-Tag-Release (NeurIPS 2025, arXiv:2502.00919). Sinks tag tokens with semantic directions; low-rank-capturable.
- **[massive-act]** Sun et al. 2024 — Massive Activations in LLMs. Few channels with 100-1000× activation act as implicit bias; deleting breaks the model. Basis for R6.

## Janus / Plan-09 audit additions (added 2026-06-04)

New IDs surfaced by the 2026-06-04 deeper novelty audit (brainstorm §5.5–5.6) + Janus Phase-0/cohort. Used in `notes/plans/09-intrinsic-site-protection/`.

### Retrieval / sink heads (read-side site definitions)
- **[retrieval-head]** Retrieval Head Mechanistically Explains Long-Context Factuality (arXiv:2404.15574, ICLR'25). Retrieval heads: universal, sparse (<5%), intrinsic, causal for NIAH. The canonical read-side site for Janus.
- **[ret-dyn]** Retrieval Heads are Dynamic (arXiv:2602.11162). Retrieval heads vary per-context, irreplaceable → detector must average over inputs (informs Janus Phase-0 stability).
- **[duo-attn]** DuoAttention (arXiv:2410.10819). Clean split: retrieval heads vs streaming/sink heads; uses it for KV compression (inference). Janus reuses the split as the *site oracle* for training-time protection. (Janus Phase-0 confirms sink≠retrieval, Jaccard 0.)

### Sink-tuning / sink-forgetting (the single-leg fixes)
- **[focusft]** FocusFT (arXiv:2605.09932). Sink-dilution fix for long-context SFT (read-leg only).
- **[sink-forget]** (arXiv:2410.05648). Attention-sink → continual-learning forgetting + pre-scaling fix (write-leg only). Janus must beat the **[focusft]+[sink-forget]** stack.
- **[zerotuning]** ZeroTuning — inference-time sink-scalar knob. **[act-sink]** ACT (arXiv:2406.15765). **[pasta]** PASTA special-token PEFT (EACL'23). **[sinklora]** sink-aware LoRA. **[sink-survey]** attention-sink survey (arXiv:2604.10098).

### Long-context continual / forgetting dynamics
- **[lccp-dyn]** Learning Dynamics of Long-Context Continual Pre-training (arXiv:2604.02650). Retrieval heads stable (>93%) during long-ctx CPT; descriptive (continual *pre-training*, not downstream SFT).
- **[rl-circuits]** Mechanistic origins of forgetting: RL vs SFT (arXiv:2605.28860). Differential circuit vulnerability at head level.

### MoE expert-selection competitors (Janus super-expert headline differentiators)
- **[das-moe]** DAS (OpenReview zBgjWTWgCh). Two-stage, freeze all but top-k experts by **domain affinity** for MoE forgetting. Criterion competitor.
- **[expert-condenser]** ExpertCondenser / MoECondenser (arXiv:2604.23036). Preserve **long-tail** experts via always-on gated condensers (opposite end from super-experts). MoE-SFT baseline.

## MoE prior-work audit (added 2026-06-03)

For the MoE-specific RCA angles (`notes/ideas/rca-transformer-intrinsic-2026-06-03.md` §11). MoE = forgetting lever (crowded), not a long-context lever.

### The genuine bridge — experts ↔ attention sinks (the one interesting find)

- **[super-experts]** Unveiling Super Experts in MoE LLMs (arXiv:2507.23279). A *tiny* set of experts **induce the attention sinks**; pruning them → sink decay rate >90% → near-zero on some tasks (Qwen3-30B-A3B). MoE analog of `[massive-act]`; the load-bearing site. **Studies compression, NOT fine-tuning → the gap for our "super-expert-anchored adaptation" (§11.3).**
- **[sink-native-moe]** Attention Sink Forges Native MoE in Attention Layers (arXiv:2602.01203). Sink attention weight = implicit gating factor → heads form a "native MoE"; freeze top-m "shared heads" + route the rest to fix head collapse; better long-context retrieval after removing vanilla sinks.
- **[gated-attn-sinkfree]** Gated Attention for LLMs: Attention-Sink-Free (arXiv:2505.06708). Query-dependent sparse gating at SDPA output eliminates attention sinks; +10 on RULER length-generalization; <2M extra params; dense + MoE variants.

### MoE continual-learning / PEFT (preempts our M1-M9)

- **[esft]** ESFT — Expert-Specialized Fine-Tuning / "Let the Expert Stick to His Last" (arXiv:2407.01906, EMNLP 2024). Tune only highest-task-affinity experts, freeze rest + router; routing for a task is highly concentrated; -90% storage, -30% train time, better general-task retention. **= our M4 verbatim; §6.3 studies shared-vs-non-shared (= our M1).**
- **[des-moe]** DES-MoE — Dynamic Expert Specialization (arXiv:2509.16882, EMNLP 2025). Adaptive router (distillation) + real-time expert-domain correlation + 3-phase progressive freezing; 6 domains incl. math+code; **89% less forgetting** than full FT, 98% general-cap retention. **= our M4/M8.**
- **[loramoe]** LoRAMoE (arXiv:2312.09979, ACL 2024). Freeze backbone, add parallel LoRA experts + router, localized balancing splits experts into world-knowledge vs task groups → alleviates world-knowledge forgetting under large SFT data. **= our M3 verbatim.**
- **[same-moe]** Same — Stabilized MoE for Multimodal Continual Instruction Tuning (arXiv:2602.01990). Models **router drift + expert drift**; spectral-aware routing + curvature-aware Riemannian expert scaling + adaptive expert freezing. **= our M8 verbatim.**
- **[lifelong-moe]** Lifelong-MoE (arXiv:2305.12281, ICML 2023). Expand experts + gating dims, freeze old experts/gatings, output-level regularization → lifelong pretraining without forgetting. **= our M3 (expansion variant).**

## Conversations / internal

### [conv-2026-05-26] First brainstorm session
- **Type**: conversation
- **Why it matters**: Origin point of the X-W framing.
- **Used in**: everything in this session's `docs/matrix/` entry.
- **Tags**: #origin
