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

## Conversations / internal

### [conv-2026-05-26] First brainstorm session
- **Type**: conversation
- **Why it matters**: Origin point of the X-W framing.
- **Used in**: everything in this session's `docs/matrix/` entry.
- **Tags**: #origin
