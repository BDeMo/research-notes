# Plan 08 v0 — Budget 预算

> 这里计算的是 practical v0：8B 左右 frozen base model + trainable memory
> wrapper。它和 `budget.md` 里完整 self-modifying LLM 的预算分开。

## 预算假设

- Base model：7B/8B 级别开源模型，保持 frozen。
- 可训练部分：只训练 memory wrapper。
- 硬件价格：H100 按 `$3 / GPU-hour` 估算，和 Plan 08 主预算一致。
- 主要交付：research prototype、RCA foundation model、技术报告、demo。
- Private RCA / Nokia DTS 数据不计入公开 release 成本。

## TL;DR

| 范围 | 日历时间 | GPU-hours | H100 费用 | 能得到什么 |
|---|---:|---:|---:|---|
| Lean MVP | 4-7 周 | 1,420-2,950 | $4.3k-$8.9k | 一个 8B base、一个 wrapper、smoke + RCA demo。 |
| Paper-quality v0 | 7-13 周 | 2,820-5,750 | $8.5k-$17.3k | 通用 long-context benchmark、RCA foundation model、RCA domain eval、ablations。 |
| Strong paper / model release | 10-16 周 | 5,000-9,000 | $15k-$27k | 多 seed、更完整 sweep、model card 和技术报告。 |

如果有一台 dedicated 8xH100，paper-quality v0 的 GPU 总量大约是 15-30 天的满载时间。
实际日历时间更长，因为数据整理、debug、评测和写报告不能完全并行。

## 分阶段预算

| 阶段 | 日历时间 | GPU-hours | 费用 | 决策点 |
|---|---:|---:|---:|---|
| 0. Scope + smoke | 3-5 天 | 120-250 | $360-$750 | wrapper 能否在 synthetic / NIAH smoke 上训练起来。 |
| 1. 基础 long-context track | 2-3 周 | 600-1,200 | $1.8k-$3.6k | LoCoMo / LongBench / RULER 上是否超过 summary baseline。 |
| 2. Public debug traces + wrapper | 2-4 周 | 900-1,800 | $2.7k-$5.4k | public debug trace 是否能产生有用的 RCA prediction。 |
| 3. RCA domain transfer + demo | 1-2 周 | 400-900 | $1.2k-$2.7k | RCA 压缩是否能在 Nezha / OpenRCA / RCAEval / lincyaw 上工作。 |
| 4. Ablations + release polish | 1-2 周 | 800-1,600 | $2.4k-$4.8k | memory length、chunk order、summary/retrieval baseline、model card。 |
| Paper-quality v0 总计 | 7-13 周 | 2,820-5,750 | $8.5k-$17.3k | 决定是否写 paper 和 release RCA foundation model。 |

## Phase 0 — Scope + smoke

目标：在真实 benchmark 前确认实现路线可行。

包含：

- frozen 8B base setup；
- memory-wrapper forward pass；
- NIAH 或 synthetic planted-evidence smoke set；
- 一次短 teacher-student training run。

预算：`120-250 GPU-hours`。

通过条件：learned memory 在 smoke task 上超过同长度 naive summary。

## Phase 1 — 基础 long-context track

目标：先验证 wrapper 的通用 long-context 能力，再进入 RCA。

数据集：

- LoCoMo；
- LongBench；
- RULER；
- Needle-in-a-Haystack variants；
- 可选 Qasper / NarrativeQA / 2WikiMultihopQA。

包含：

- full-context teacher generation；
- full-context / summary / retrieval / wrapper evaluation；
- memory-state next prediction 的 wrapper training；
- memory length sweep，例如 16 / 32 / 64 / 128 memory tokens。

预算：`600-1,200 GPU-hours`。

通过条件：至少两个 benchmark family 上，wrapper memory 距离 full context 在 5-10 points 内，
并且 matched token budget 下超过 summary。

## Phase 2 — Public debug traces + wrapper

目标：把 RCA foundation model 路线扩展到公开 coding/debug traces。

数据集：

- SWE-bench / SWE-bench Verified；
- BugsInPy；
- Defects4J；
- QuixBugs；
- 可选 curated CodeNet / APPS wrong-submission traces。

包含：

- 把 public traces 转成 long-context RCA/debug items；
- direct prediction evaluation；
- 小规模 agentic trace collection；
- 在 trace / test-output / code-context chunks 上继续训练 wrapper。

预算：`900-1,800 GPU-hours`。

通过条件：在不更新 frozen base 的前提下，wrapper 的 root-cause / fix recommendation
质量超过 summary 和 retrieval baselines。

## Phase 3 — RCA domain transfer + demo

目标：把 research method 接到 RCA domain。

数据集：

- Nezha；
- OpenRCA-500；
- RCAEval；
- lincyaw/rca；
- private Liang/Nokia DTS 只做 internal qualitative probe。

包含：

- long-context RCA benchmark generation；
- prompt strategy comparison；
- demo incident；
- 初版技术报告图表。

预算：`400-900 GPU-hours`。

通过条件：learned memory 保留 evidence grounding 和 RCA output format，同时 prompt/KV 成本至少降低 4x。

## Phase 4 — Ablations + release polish

目标：把 prototype 变成可以写 paper / release model 的版本。

包含：

- memory size ablation；
- chunk order ablation；
- early/middle/late evidence retention；
- direct vs agentic public-debug RCA comparison；
- model card；
- technical report；
- public-data-only release package。

预算：`800-1,600 GPU-hours`。

## 费用公式

按 `$3 / H100 GPU-hour`：

```text
cost = GPU_hours * 3

Lean MVP:          1,420-2,950 GPUh  => $4,260-$8,850
Paper-quality v0: 2,820-5,750 GPUh  => $8,460-$17,250
Strong release:   5,000-9,000 GPUh  => $15,000-$27,000
```

## 为什么比完整 Plan 08 便宜

完整 Plan 08 包含 model-emitted weight deltas、verifier-gated apply、RL、rollback
和 agent integration。v0 先避开这些：

- 不 finetune base model；
- 第一版不输出 LoRA/weight；
- 第一版不做 RL；
- wrapper training 比 full model training 便宜；
- RCA foundation-model release 可以基于 public debug data 和 public RCA datasets。

## Off-Ramps

- Phase 0：如果 learned memory 在 smoke task 上打不过 summary，停止。
- Phase 1：如果 LoCoMo / LongBench / RULER 上 matched budget 打不过 retrieval 或 summary，停止。
- Phase 2：如果 public-debug direct prediction 没有超过 summary / retrieval baseline，停止。
- explicit memory tokens 没有明显收益前，不升级到 LoRA/weight memory。

## 人力

最低配置：

- 1 个 researcher/engineer：负责 wrapper architecture、training、report。
- 1 个 engineer：负责 dataset conversion、evaluation plumbing、demo。

可选：

- systems support：加速 batched inference 和 wrapper serving；
- RCA/domain reviewer：帮助做 qualitative failure analysis。
