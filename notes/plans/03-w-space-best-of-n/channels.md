# Plan 03 — Validation channels

## Primary channels (cheap, programmatic verifiers)

### MATH (especially level 4–5)
- **Why**: hard enough that BoN matters; gold verifier exists.
- **Verifier**: SymPy-based grader; LLM-judge fallback only when SymPy errors out.
- **Test stratification**: by subject + level. Report per-subject if WBoN wins by topic.

### HumanEval+ / MBPP+
- **Why**: execution-based verifier is unambiguous.
- **Verifier**: EvalPlus extended test suites.
- **Stratification**: by problem category.

### AIME-2024 / AIME-2025
- **Why**: ultra-hard math; even strong models miss most of them; BoN moves the needle.
- **Scale**: only ~30 problems but high signal.

## Secondary channels

### MMLU-Pro
- **Why**: multiple-choice ⇒ trivial verifier. Diverse subjects. Tests whether WBoN helps on *knowledge* tasks (not just reasoning).
- **Caveat**: little room for diversity-of-output; BoN selects from a discrete option set. Use as a stress test of *whether WBoN even helps* here.

### Arena-Hard
- **Why**: open-ended quality, judged by GPT-4o or Claude-Sonnet. Use as a final-paper signal, *after* the rule-based channels confirm gains.
- **Caveat**: judge bias is known; report alongside rule-based.

### GSM8K (only as sanity)
- Many models near saturation. Use to confirm the experimental setup, not as the headline.

## Channels we explicitly avoid

- Anything where the verifier and the generator are the same model family — risks circular gains.
- Long-context benchmarks — orthogonal axis; would muddle the headline.

## Base models

In priority order:

| Model | Size | Notes |
|---|---|---|
| **Qwen2.5-7B-Instruct** | 7B | Strong math + tractable LoRA budget |
| **Llama-3.1-8B-Instruct** | 8B | Industry-standard sanity check |
| **Qwen2.5-Math-7B** | 7B | Pre-tuned for math; tests whether WBoN still helps a specialist |
| **DeepSeek-R1-Distill-7B** | 7B | Reasoning model — does WBoN still help when CoT is already long? Important sanity |
| **Qwen2.5-72B-Instruct** | 72B | Final paper-scale runs |

## Verifier infrastructure

- **SymPy grader** + standard MATH evaluation harness (e.g., from [MARIO_EVAL](https://github.com/MARIO-Math-Reasoning/MARIO_EVAL) or PRM800k).
- **EvalPlus** for code.
- **Process Reward Model (PRM)** as a learned-verifier baseline (e.g., Math-Shepherd). Test WBoN under both rule-based and learned verifiers.

## Hypernet starting point

- Fork [SakanaAI/doc-to-lora](https://github.com/SakanaAI/doc-to-lora). Reuse the Perceiver architecture and training scaffold.
- Strip the document encoder; replace with a query encoder + noise injection.

## Comparable published baselines

- **Self-consistency** (Wang et al., 2022) — the canonical $X$-BoN with majority voting.
- **Best-of-N with reward model** (Lightman et al., PRM-800k 2023; Cobbe et al., 2021).
- **Tree-of-Thoughts** (Yao et al., 2023) — search along $X$ at higher depth.
- **MCTS-style search** (Chen et al., 2024, ReST-MCTS*).
- **LoRA ensembling** — (closest spiritual cousin) LoRA-ensembles, e.g., [LoRA-Hub](https://arxiv.org/abs/2307.13269) — different goal (training-time combination) but methodologically nearby.

## Open-source code to fork

- [vLLM](https://github.com/vllm-project/vllm) — supports LoRA hot-swap, low-overhead batched serving.
- [S-LoRA](https://github.com/S-LoRA/S-LoRA) reference for serving thousands of LoRAs.
- [Sakana doc-to-lora](https://github.com/SakanaAI/doc-to-lora) — hypernet scaffolding.
- [PRM800k](https://github.com/openai/prm800k) — verifier infra and training data.
- [TRL](https://github.com/huggingface/trl) — for the RL stage.
