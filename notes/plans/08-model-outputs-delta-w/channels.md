# Plan 08 — Validation channels

## Phase 1 channels (synthetic / controlled)

### Sequential SQuAD
- Stream of (passage, question) pairs from different topics.
- After each turn, model emits `<learn>` summarizing the new fact.
- Probe with held-out questions from the same passage.
- **Why**: easy to construct, cheap, has a clear gold answer and learning signal.

### Sequential ARC / OpenBookQA
- Similar idea, knowledge-intensive.
- Probe by held-out questions requiring the just-learned fact.

### Sequential rule-induction synthetic tasks
- Generate stream of input-output pairs from a hidden rule. Model must induce + remember the rule.
- **Why**: clean signal for "did the model actually internalize the rule into $\theta$?"

## Phase 2 channels (real continual / personalization)

### LaMP (Language Models for Personalization)
- 7 sub-tasks (citation prediction, movie tag generation, product review, etc.).
- Each user has a profile (long history). Models must adapt.
- **Use**: primary channel for personalization.
- **Verifier**: per-task ground truth; usually F1 or BLEU.

### PerLTQA (Personalized Long-Term QA)
- Multi-session conversational QA where facts from session 1 must be remembered in session 5.
- **Use**: stress test for *cross-session* memory via weight updates.

### MSC (Multi-Session Chat)
- Open-domain personalization across multiple sessions.

### CLAM (Continual Language Adaptation Benchmark)
- Sequential domain adaptation (e.g., a sequence of legal → medical → financial Q&A).
- **Use**: tests catastrophic forgetting under our setup.

## Phase 3 channels (agentic, long horizon)

### SWE-Bench Verified
- Real GitHub issue resolution requiring multi-step code edits.
- **Why**: the agent must understand the repo within a single task — perfect substrate for inside-task weight updates.
- **Verifier**: hidden test suite. Execution-based, unambiguous.

### WebArena
- Multi-step web agent tasks.
- **Verifier**: programmatic checks of final web state.

### OSWorld
- Multi-step desktop/OS agent tasks.
- **Verifier**: programmatic.

### MLE-Bench (long-horizon ML engineering)
- Multi-day-equivalent ML pipelines.
- **Why**: extreme case — does within-task learning materialize?

### Computer-Use Agent (Claude / Operator-style)
- For qualitative demos.

## Base models

| Model | Size | Used for |
|---|---|---|
| **Gemma-2-2B-IT** | 2B | Phase 1 iteration (matches D2L) |
| **Qwen2.5-7B-Instruct** | 7B | Phase 2 primary |
| **Llama-3.1-8B-Instruct** | 8B | Phase 2 secondary |
| **Qwen2.5-Coder-7B** | 7B | Phase 3 (SWE-Bench) |
| **DeepSeek-R1-Distill-7B** | 7B | Phase 3 (agentic, reasoning-heavy) |

Avoid 70B+ at first — Phase 3 cost would explode.

## Verifiers

- **Rule-based** for SQuAD / ARC / SWE-Bench.
- **Process Reward Model** (Math-Shepherd / OmegaPRM-style) for stepwise checks where needed.
- **Reward model** (Skywork-Reward, Llama-Guard variants) for personalization quality scoring.
- **Self-verifier** as a last resort (use a stronger model to check) — flag if used.

## Hypernet starting points

- [SakanaAI/doc-to-lora](https://github.com/SakanaAI/doc-to-lora) — Perceiver hypernet, well-engineered.
- [SakanaAI/text-to-lora](https://github.com/SakanaAI/text-to-lora) — task-description → LoRA; complementary baseline + possible co-training.
- [Generative Adapter](https://github.com/ChenT0518/Generative-Adapter) — Chen et al., ICLR 2025; closest peer.

## Comparable published baselines

- **D2L-as-memory** — apply Doc-to-LoRA on conversation history at each turn.
- **MEMIT / ROME** — knowledge editing baselines. Adjacent, slower, but established.
- **AlphaEdit** — capacity-preserving editor; primary stability baseline.
- **MemGPT** / **MemoryBank** / **LangChain memory tools** — prompt-side memory; the dominant industry alternative.
- **Sparse memory finetuning** (Lin et al., 2025) — closest TTT competitor.
- **Cartridges** (Eyuboglu et al., 2025) — sleep-time CD; uses real backprop, but the comparison clarifies the "no backprop" claim of our method.

## Datasets / streams to construct

We may need to *build* the Phase-1 stream. Plan:
- Take SQuAD, partition by topic, schedule a curriculum.
- Generate synthetic rules (function-like: $f(a, b) = ab + 7$) and stream training examples.
- Release these synthetic streams as a benchmark contribution.

## Open-source forks needed

- vLLM + LoRA hot-swap for inference.
- TRL/GRPO for RL training.
- PEFT for LoRA application.
- Sakana D2L scaffold (architecture + training code).
- A custom agent loop for Phase 3 (one of [SWE-Agent](https://github.com/princeton-nlp/SWE-agent), [Aider](https://github.com/Aider-AI/aider), or write our own).
