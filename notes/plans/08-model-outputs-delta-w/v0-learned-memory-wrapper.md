# Plan 08 v0 — Learned Memory Wrapper

> Scope freeze for the first implementable version of Plan 08.
> The full self-modifying-LLM idea remains the north star; v0 proves the
> smaller claim that a wrapper can compress long context into an updateable
> memory state.
>
> **Settings / provenance:** this is a v0 design document. Any result cells or
> later citations should resolve through [`settings.md`](settings.md); the
> implemented v1 recipe is [`P08-S1`](settings.md#p08-s1--v1-canonical-wrapper-recipe).

## V0 Claim

Given a long context stream `c_1 ... c_n`, learn a wrapper `G_phi` that maintains
a compact memory state:

```text
m_t = G_phi(m_{t-1}, c_t)
y_t = F_theta(q_t, m_t)
```

The memory must be small enough to reduce context/KV cost, but expressive enough
to preserve the information needed by downstream answers.

## Model Assumption

V0 uses an existing open-source model around the 8B scale as the frozen base.
The wrapper is the trainable part.

Recommended starting bases:

- Qwen3-8B or Qwen3-8B-Instruct for reasoning and long-context experiments.
- Llama-3.1-8B-Instruct as a secondary baseline if licensing / tooling makes it
  easier.
- Qwen2.5-Coder-7B or a similar 7B/8B coder for the later public debug-trace extension.

The base model should remain frozen in v0. This is intentional: direct SFT of
the base risks catastrophic forgetting because we do not have the original
pretraining data distribution, so dataset mixing cannot fully protect the base
capabilities. V0 asks whether a trained wrapper can absorb new context while
preserving the base model.

## Expected Outcomes

There are two expected outcomes, and they should be evaluated separately:

1. **Research paper**: a new wrapper architecture framed as next-state
   prediction over memory states. The core contribution is not another RCA
   benchmark result; it is a general method for learning `m_t = G_phi(m_{t-1},
   c_t)` and using `m_t` as compact context.
2. **Open-source model / community impact**: an RCA foundation model built on a
   public 7B/8B base plus trained wrapper. This should be useful for structured
   RCA, evidence-heavy incident analysis, and later public debug-trace reasoning.

## RCA-Demo Findings Behind the RCA Foundation Model Direction

The RCA foundation-model direction is motivated by what the `rca-demo`
experiments already showed. SFT can teach RCA behavior, but it is not stable
enough to be the main long-context strategy by itself.

| Demo finding | Evidence / symptom | Interpretation | Impact on v0 design |
|---|---|---|---|
| RCA capability is learnable through SFT, but only partially. | `q25_mix` reached `recommendation_f1 = 0.733` on RCAEval, far above `q25_base`; post-fix `q25_curr_4_lincyaw` improved lincyaw `service_set_f1` from 0.272 to 0.563. | SFT can inject RCA structure, service identification, and recommendation behavior. | Treat SFT as initialization / capability probing, not the final architecture. |
| SFT is not always a win. | Mix and curriculum results flip depending on dataset, base, and evaluation path; base-mismatch re-evaluation changed the closure narrative. | More SFT or more data mixing does not reliably solve RCA. | Compare against compression baselines instead of only extending SFT recipes. |
| Qwen3 is format-fragile under the current LoRA recipe. | q3 cumulative checkpoints had low `json_parseable` after depth 2/3; the issue was not only `q3_mix`. | The base + LoRA delta + JSON-only output constraint is unstable. | Use q25-class checkpoints for the first demo; keep q3 as a separate recipe problem. |
| Anti-forgetting quick-win failed. | v2 oversample + 5% general anchor degraded lincyaw `service_set_f1` from 0.563 to 0.233 and collapsed Nezha `evidence_overlap` from 0.578 to 0.000. | Well-formed JSON can mask empty or ungrounded RCA evidence. | Track evidence-grounding metrics, not only parseability. |
| General capability regresses under direct SFT. | `q25_mix` dropped on GSM8K and TruthfulQA versus `q25_base`. | We do not have the original pretraining distribution, so small RCA mixes cannot fully protect the base. | Freeze the base model and train only the wrapper in v0. |
| Evaluation and release paths can silently fail. | Adapter/base mismatch, dataset-cache version mismatch, and final-step merge issues changed or invalidated results. | RCA claims need strict provenance and post-conditions. | Keep the v0 demo small, reproducible, and metric-backed. |
| The compression path is scaffolded but not yet end-to-end verified. | Scripts exist for long-context item building and full/summary/retrieval/learned-memory prompt preparation, but generation + scoring still needs a verified demo run. | The next milestone is a concrete public case, not a broad benchmark sweep. | First demo: full context vs summary vs retrieval vs wrapper memory on a public RCA case. |

## Default Memory Type

V0 starts with explicit memory tokens:

- `m_t` is a fixed-length sequence of learned/generated memory-token embeddings.
- The base model consumes these tokens before the current query or RCA prompt.
- The wrapper updates memory after each new chunk.

This is the default because it is:

- easiest to train with teacher-student distillation;
- easiest to inspect and ablate by memory length;
- compatible with any decoder model that accepts `inputs_embeds`;
- safer than writing directly into model weights.

## Later Ablations

Hidden-state memory and LoRA/weight memory are not v0 blockers.

| Memory form | V0 role | Why not first |
|---|---|---|
| Explicit memory tokens | Default | Smallest engineering surface and easiest evaluation. |
| Hidden-state features | v0.1 ablation | Needs layer-specific injection points and stronger model coupling. |
| LoRA / weight deltas | v1 | Closest to Plan 08 north star, but requires hypernet training, verifier gating, rollback, and interference control. |

## Training Objective

Use a full-context teacher and compressed-memory student.

```text
teacher = F_theta(q, full_context)
student = F_theta(q, memory_tokens)
loss = supervised_loss(student, gold) + lambda * distill_loss(student, teacher)
```

For sequential chunks:

```text
m_0 = init_memory
for chunk in chunks:
    m_t = G_phi(m_{t-1}, chunk)
```

The probe set should ask about facts that appeared in early, middle, and late
chunks so the forgetting curve is visible.

## Minimum Baselines

- full context prompt;
- naive summary prompt;
- retrieval over chunks;
- fixed/gist tokens;
- memory-wrapper tokens.

Doc-to-LoRA/Text-to-LoRA are comparable baselines for later weight-memory
experiments, but not required for v0.

## Base Long-Context Dataset Track

Before extending to public debug traces, run wrapper experiments on general
long-context and memory benchmarks:

| Dataset | Use | Why |
|---|---|---|
| LoCoMo | primary memory benchmark | Multi-session / long-context memory and conversation QA. |
| LongBench | broad long-context QA | Standard mix of retrieval, summarization, and reasoning over long inputs. |
| RULER | controlled length stress test | Separates nominal context length from effective context use. |
| Needle-in-a-Haystack variants | smoke test | Fast check that memory preserves planted evidence. |
| HotpotQA / 2WikiMultihopQA long-form variants | multi-hop reasoning | Tests whether compressed memory keeps cross-document links. |
| Qasper / NarrativeQA | document QA | Good teacher-student setup against full-context answers. |

These datasets establish whether the wrapper works before introducing RCA
domain complexity.

## Pass Criteria

V0 passes if it demonstrates all of the following on a controlled long-context
task and one RCA task:

- quality is within 5-10 points of full-context prompting;
- input tokens or KV footprint are reduced by at least 4x;
- memory updates across chunks do not erase early evidence by more than 5 points;
- learned memory beats naive summary at matched token budget.

## Kill Criteria

Do not escalate to LoRA/weight deltas if explicit memory tokens fail to beat a
summary baseline or if the wrapper cannot preserve early evidence under
sequential updates.
