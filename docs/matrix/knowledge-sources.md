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
- **Why it matters**: Self-study CD with prefix tuning. Shares the "internalize context" goal with D2L but uses real gradient descent + sleep-time compute. Direct comparable baseline for plan 01 and plan 08.
- **Tags**: #context-distillation #prefix-tuning #sleep-time-compute

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
- **Type**: paper
- **Why it matters**: Closest prior to D2L. Uses NTP loss instead of KL — D2L explicitly outperforms it. Read carefully before plan 08.
- **Tags**: #hypernetwork #context-distillation

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
