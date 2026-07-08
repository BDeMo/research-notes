# Plan 08 v0 — 可学习记忆 Wrapper

> 这是 Plan 08 第一个可落地版本的范围说明。
> 完整的 self-modifying LLM 仍然是长期目标；v0 先证明一个更小的问题：
> wrapper 能否把长上下文压缩成一个可更新的记忆状态。
>
> **Settings / provenance:** 这是 v0 设计文档。这里或后续引用的结果 cell
> 统一通过 [`settings.md`](../settings/settings.md) 追溯；已经实现的 v1 recipe 是
> [`P08-S1`](../settings/settings.md#p08-s1--v1-canonical-wrapper-recipe)。

## v0 要证明什么

给定一个长上下文流 `c_1 ... c_n`，训练一个 wrapper `G_phi` 维护压缩记忆：

```text
m_t = G_phi(m_{t-1}, c_t)
y_t = F_theta(q_t, m_t)
```

这里 `m_t` 要足够小，能降低上下文/KV cache 成本；同时又要保留回答问题所需的信息。

## 模型假设

v0 使用 7B/8B 左右的开源模型作为 frozen base，只训练 wrapper。

推荐起点：

- Qwen3-8B 或 Qwen3-8B-Instruct：用于 long-context memory 和 reasoning。
- Llama-3.1-8B-Instruct：作为跨模型验证备选。
- Qwen2.5-Coder-7B 或类似 7B/8B coder：用于后续 public debug trace 扩展。

v0 暂时不直接 finetune base model。原因是我们没有原模型的预训练数据分布；
直接 SFT 容易造成 catastrophic forgetting，只靠混合我们自己的小数据集，很难最大程度保护原有能力。
因此 v0 的核心问题是：能不能让 wrapper 吸收新上下文，同时保持 base model 不变。

## 预期产出

这个方向有两个产出，应该分开评估：

1. **Research paper**：提出一种新的 wrapper architecture，把问题定义成 memory state 的 next-state prediction。核心贡献不是 RCA 分数，而是通用的 `m_t = G_phi(m_{t-1}, c_t)` 学习方法。
2. **开源模型 / 社区影响力**：基于公开 7B/8B base + trained wrapper，发布 RCA foundation model，用于结构化 RCA、长上下文 evidence reasoning，以及后续 public debug trace reasoning。

## RCA-Demo 发现如何推出 RCA Foundation Model 路线

RCA foundation-model 路线不是从空白开始，而是基于 `rca-demo` 已经暴露的问题。当前结论是：
SFT 可以让模型学到 RCA 行为，但单靠 SFT 不够稳定，不能作为 long-context RCA 的主路线。

| Demo 发现的问题 | 证据 / 现象 | 说明 | 对 v0 设计的影响 |
|---|---|---|---|
| RCA 能力可以通过 SFT 学到，但只是部分能力。 | `q25_mix` 在 RCAEval 上达到 `recommendation_f1 = 0.733`，明显高于 `q25_base`；base-mismatch 修复后，`q25_curr_4_lincyaw` 的 lincyaw `service_set_f1` 从 0.272 到 0.563。 | SFT 能注入 RCA 结构、service identification 和 recommendation 行为。 | 把 SFT 作为 initialization / capability probing，而不是最终架构。 |
| SFT 不是 always win。 | mix 和 curriculum 的结论会随数据集、base、评测路径变化；base-mismatch re-eval 后，closure narrative 被改写。 | 继续加 SFT 或调整 mix 比例不能稳定解决 RCA。 | v0 要比较 compression baselines，而不是只延长 SFT recipe。 |
| Qwen3 在当前 LoRA recipe 下有格式脆弱性。 | q3 cumulative checkpoints 在 depth 2/3 后 `json_parseable` 很低；问题不是只有 `q3_mix`。 | base + LoRA delta + JSON-only 输出约束不稳定。 | 第一个 demo 先用 q25-class checkpoint；q3 单独作为训练 recipe 问题处理。 |
| Anti-forgetting quick-win 失败。 | v2 oversample + 5% general anchor 让 lincyaw `service_set_f1` 从 0.563 掉到 0.233，Nezha `evidence_overlap` 从 0.578 掉到 0.000。 | JSON 合法不代表 evidence grounded；模型可能输出格式正确但 evidence 为空。 | 指标必须看 evidence grounding，不能只看 parseability。 |
| 直接 SFT 会损伤 general capability。 | `q25_mix` 相比 `q25_base` 在 GSM8K 和 TruthfulQA 上下降。 | 我们没有原始 pretraining distribution，只靠小规模 RCA mix 很难保护 base 能力。 | v0 冻结 base model，只训练 wrapper。 |
| 评测和发布路径可能 silent fail。 | adapter/base mismatch、dataset cache 版本、final-step merge 都曾导致结果变化或无效。 | RCA 结论必须有 provenance 和 post-condition。 | v0 demo 要小、可复现、每一步都有 metric。 |
| compression 路径已有脚手架，但还没 end-to-end verified。 | 已有 long-context item builder 和 full/summary/retrieval/learned-memory prompt preparation scripts；但 generation + scoring 还需要一次 verified demo run。 | 下一步不是大规模 benchmark sweep，而是先做一个公开 case 的 concrete demo。 | first demo：public RCA case 上比较 full context、summary、retrieval、wrapper memory。 |

## v0 默认记忆形式

v0 先做 explicit memory tokens：

- `m_t` 是固定长度的 memory-token embeddings。
- base model 在当前 query 或 RCA prompt 前读取这些 memory tokens。
- wrapper 在每个新 chunk 到来后更新 memory。

先选 memory tokens 的原因：

- 最容易用 teacher-student distillation 训练；
- 最容易通过 memory length 做 ablation；
- 只要 decoder 支持 `inputs_embeds` 就能接；
- 比直接写模型权重安全。

## 后续 ablation

Hidden-state memory 和 LoRA/weight memory 不作为 v0 阻塞项。

| 记忆形式 | v0 角色 | 为什么不先做 |
|---|---|---|
| Explicit memory tokens | 默认方案 | 工程面最小，最容易评估。 |
| Hidden-state features | v0.1 ablation | 需要 layer-specific injection，和模型耦合更强。 |
| LoRA / weight deltas | v1 | 最接近 Plan 08 长期目标，但需要 hypernet、verifier、rollback 和 interference control。 |

## 训练目标

用 full-context teacher 和 compressed-memory student：

```text
teacher = F_theta(q, full_context)
student = F_theta(q, memory_tokens)
loss = supervised_loss(student, gold) + lambda * distill_loss(student, teacher)
```

对于连续 chunks：

```text
m_0 = init_memory
for chunk in chunks:
    m_t = G_phi(m_{t-1}, chunk)
```

probe set 要覆盖 early / middle / late chunks，这样才能看到 forgetting curve。

## 最小 baseline

- full context prompt；
- naive summary prompt；
- retrieval over chunks；
- fixed/gist tokens；
- memory-wrapper tokens。

Doc-to-LoRA / Text-to-LoRA 是后续 weight-memory 版本的重要 baseline，但不是 v0 必须项。

## 基础 long-context 数据集线

在扩展到 public debug traces 之前，先用通用 long-context / memory benchmark 验证 wrapper：

| 数据集 | 用途 | 原因 |
|---|---|---|
| LoCoMo | primary memory benchmark | 多轮 / 长期记忆 QA。 |
| LongBench | broad long-context QA | 标准长上下文 QA、summarization、reasoning 混合 benchmark。 |
| RULER | controlled length stress test | 区分 nominal context length 和 effective context use。 |
| Needle-in-a-Haystack variants | smoke test | 快速验证 planted evidence 是否被保留。 |
| HotpotQA / 2WikiMultihopQA long-form variants | multi-hop reasoning | 测试压缩记忆能否保留跨文档联系。 |
| Qasper / NarrativeQA | document QA | 适合 full-context teacher-student setup。 |

这些数据集用来先证明 wrapper 有通用 long-context 价值，再进入 RCA domain complexity。

## 通过标准

v0 至少要在一个 controlled long-context task 和一个 RCA task 上满足：

- 质量距离 full-context prompting 在 5-10 points 内；
- 输入 tokens 或 KV footprint 至少降低 4x；
- 后续 chunks 不会让 early evidence 掉超过 5 points；
- matched token budget 下，learned memory 要超过 naive summary。

## Kill Criteria

如果 explicit memory tokens 不能超过 summary baseline，或者 sequential updates 无法保留 early evidence，
就不要升级到 LoRA/weight deltas。
