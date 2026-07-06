# v1.7 改进方案总表（提升压缩成功率 + 判别器准确率）

**确诊问题**：① 压缩失败的根因 = `M` 对冻结 base 是 **OOD**（`m_raw` 是 encoder 过完 final-norm 的 hidden，norm≈405，却当作 input-embedding 注入第 0 层；`m_proj` eye-init 不改尺度）+ 冻结 base 从没学过读软前缀。② 门弱 = 对抗判别器是被训来"被骗"的，且 batch=1/均值池化/单层，目标也和"压缩会不会掉点"错位。

**命名规则**：组 A = 压缩成功率（`v1.7.1.x`）；组 B = 判别器/门（`v1.7.2.x`）。每个方案一个 md：`v<ver>-<id>-<slug>.md`，含 **详细做法 / 流程图 / 结果**。

## 执行序 / 决策树（2026-06-11 夜，按用户指示）
1. **Phase 1 — 跑完 faithful baselines**（已 coding+smoke）：Cartridge=**逐层可训 KV-cache**、Gist=**LoRA 微调+gist mask**。
   **RUNNING** on sam-dev / **Qwen3-8B**（标准注意力；Qwen3.5 是 hybrid linear-attn，KV-cache 不一定干净）。squad+hotpot，1500 steps。
2. **Phase 2 — 在「原设置」(vanilla GCM) 上调参，尽量覆盖关键参数**。**RUNNING** `queue_hp.py`（15 配置：
   lr{1e-4,2e-4,3e-4}×steps{2400}×lam_dev{0}×lam_rec{0,10+m_only}×lam_distill{1}×N{8,12}×K{64}×n_chunks{3}+combos）
   on ray/**Qwen3.5**(squad) + test/**Qwen3-8B**(squad)。
3. **Phase 3 — 仅当 Phase-2 调参后的 best GCM 仍打不过 no_ctx/baselines** → 在 solution 上调参，**A→B→C 依次**。
   （A/C 组之前的"无效"是默认超参，Phase-2 会先排除这个；C 组损失权重已并入 Phase-2 的 lam_rec/lam_distill。）

**判定**：Phase-1+2 出数后 → 若 best-tuned vanilla GCM ≥ no_ctx 且接近 faithful baselines/full ⇒ 之前是调参问题，收工调参线；否则进 Phase-3。

**统一对照协议**（公平比较）：Qwen3.5-9B（newest）；in-task `squad_v2` + `hotpot_qa`（context-dependent，full≫no_ctx，有压缩空间）；n=96；bf16；N=16/K=128；steps 800；seed 42（趋势）→ 关键点补 3 seed。每个方案只改一个变量，其余对齐 baseline GCM。**主指标**：`comp`（压缩-only 准确率，越接近 full 越好）、`gAcc`（GCM+fallback 可实现）、门 `AUROC/F1`、真实 `FLOPs`。

---

## 大表（看这个；细节点进各自 md）

| ver | id | 方案 | 改哪里 | 复杂度 | 预期 | 状态 | comp(squad/hotpot) | gateAUROC | 备注 |
|---|---|---|---|---|---|---|---|---|---|
| — | base | 现状 GCM (soft-prefix, frozen base) | — | — | — | DONE | 0.146 / 0.146 | 0.50–0.65 | full=0.617/0.447；comp<no_ctx |
| **1.7.1.1** | **A1** | **M 归一化到 embed 尺度**（修 OOD，最便宜） | `compressor.encode` 出口加 norm-match | ⭐ | 中 | **DONE≈null** | 0.171 / 0.095 | – | squad +0.025/hotpot −0.05，都在噪声内；范数对齐不够 |
| 1.7.1.2 | A2 | **M 投影到 token-embedding 流形**（soft-nearest-embedding，构造性 in-dist） | `encode` 出口 `softmax(M·Eᵀ/τ)·E` | ⭐⭐ | 中-高 | **DONE≈null** | 0.139 (squad) | – | 比修复前(0.146)还略低；流形投影没用 |
| 1.7.1.3 | A3 | **base 加小 LoRA 学读 M**（放开冻结，忠于 Gist 原版） | `svc/lora.py` + method 联合训 | ⭐⭐⭐ | **高** | **DONE✗** | 0.11 (squad) | – | **LoRA 把 full 抬到 0.88 却不抬 comp**→瓶颈是压缩丢信息，不是 base 读不了。假设证伪 |
| 1.7.1.4 | A4 | **KV-cache 注入**（每层 K/V，而非第0层 embed） | encoder→per-layer KV；base past_kv | ⭐⭐⭐⭐ | 高 | **DEFER** | – | – | past_kv 增量解码与 cache-free runtime 冲突，风险高；A3 已覆盖同假设 |
| 1.7.2.1 | B1 | **监督门**（真实标签 y=1[comp≥full]，5-fold CV 离线） | `disc_gate --fit` | ⭐ | 中 | **DONE** | n/a | **0.48 (held-out, 随机)** | 组合 9 信号仍≈随机；门**当前不可学**，先修压缩 |
| 1.7.2.2 | B2 | **多层+attn-pool 富特征门** | `signals` 记多层 hidden | ⭐⭐⭐ | 中-高 | **DEFER** | – | – | B1 已证门不可学(0.48)；压缩变强前无意义 + 存储重 |
| — | — | **C 组：攻"M 丢 span"（A 组已证瓶颈在压缩本身，不在读）** | | | | | | | |
| 1.7.3.1 | C1 | **无损重建目标**（decoder 仅从 M 重建 ctx，封 ctx→ctx） | `compressor.reconstruct(m_only)`+lam_rec↑ | ⭐⭐ | 中 | **RUNNING** | – | – | 已实现；squad 跑（C0 baseline / C1 / C2） |
| 1.7.3.2 | C2 | **全分布蒸馏**（[M;q]→[ctx;q] 答案分布，开 lam_distill） | `--lam-distill` | ⭐ | 中 | **RUNNING** | – | – | 已有实现，配置即可 |
| 1.7.3.3 | C3 | **规模化**（n_items/steps↑，查欠拟合） | 配置 | ⭐ | 中 | TODO | – | – | 待 C0/C1/C2 出数后排 |
| 1.7.3.4 | C4 | **混合：保留 top-m 原始 token + 压缩其余** | 新代码 | ⭐⭐⭐ | 高 | TODO | – | – | span 在保留 token 里，直接解决 |
| 1.7.4.1 | C5 | **prompt 压缩(LLMLingua 式，保留真 token)** | 新代码/外部 | ⭐⭐⭐ | 高 | TODO | – | – | 文献里真 work；但非"可学压缩器" |
| 1.7.4.2 | C6 | **KV 驱逐(H2O/SnapKV)** | 新代码 | ⭐⭐⭐⭐ | 高 | TODO | – | – | 同上，保留真 KV |
| P1 | P1 | **gated-LoRA / forgetting（绕开压缩，门控权重 delta）** | LoRA+门 | ⭐⭐⭐ | **高** | TODO | – | – | 最高 ROI 的转向；压缩做反例 |

复杂度 ⭐=改几行/离线，⭐⭐⭐⭐=大改。**建议执行序**：A1 → B1（都最便宜）→ A3（最高预期）→ A2 → A4 → B2。

## 修复前 vs 修复后 vs baselines（compress-only `comp`，Qwen3.5-9B in-task，seed42）

**SQuAD-v2**（no_ctx 0.21 · **full 0.617**）　|　**HotpotQA**（no_ctx 0.25 · **full 0.447**）

| 方法 | squad `comp` | hotpot `comp` | 备注 |
|---|---|---|---|
| **full（天花板）** | **0.617** | **0.447** | 冻结 base，无 LoRA/SFT |
| Gist (baseline) | 0.215 | 0.117 | 贴 no_ctx；hotpot 还更低 |
| Cartridge (baseline) | 0.196 | 0.235 | 最强压缩器，但也只 ≈ no_ctx |
| **GCM 修复前**(soft-prefix) | 0.146 | 0.146 | OOD；垫底 |
| **GCM +A1**(norm-match) | 0.171 | 0.095 | ≈ 无变化（+0.025 / −0.05，噪声内） |
| **GCM +A2**(manifold) | 0.139 | _seed43跑_ | 流形投影；比修前还低 |
| **GCM +A3**(LoRA r16) | 0.115 | _seed43跑_ | LoRA→**full 0.843** 但 comp 不动 |

> ⚠️ **重要 caveat（tuning）**：A1–A3 全用**固定未调超参**（lr5e-5/steps800/lam_rec1/lam_dev0.05），损失权重还和 C 组重叠。"无效"可能是**欠拟合/坏超参**而非方法本身。**已开 HP sweep**（`queue_hp.py`：lr{1e-4,2e-4,3e-4}×steps{2400}×lam_dev{0}×lam_rec{0,10+m_only}×distill{1}×combos，subsumes C1/C2）on ray(squad)/sam-dev(hotpot)/test(Q3-8B)。**best-tuned comp 出来前，下面的结论都是 preliminary。**

**一句话（更新，preliminary）**：A1/A2/A3 在**默认超参下**都没修动 comp（squad 仍 0.11–0.17 ≪ full）。**A3 是关键反例**：给 base 加 LoRA 把 **full 从 0.62 抬到 0.88**，但 **comp 纹丝不动（~0.11）**——证明瓶颈是 **M 压缩本身丢了 span 信息**，不是"base 读不了 M"。➡️ 修"注入/读"（A1/A2/A3/A4）这条路被证伪；要救只能**改压缩目标/容量本身**，或承认"单 query span-QA 上 K-token 压缩不可行"并把贡献移到别处（见 gated-LoRA/forgetting 讨论）。
（注：A3 seed42 的 LoRA 是 fp32，已改 bf16；sam-dev seed43 队列出 bf16 版复核。）

## 链接

- [A1 v1.7.1.1 — M 归一化到 embed 尺度](./v1.7.1.1-A1-norm-match.md)
- [A2 v1.7.1.2 — soft-nearest-embedding 投影](./v1.7.1.2-A2-embed-manifold.md)
- [A3 v1.7.1.3 — base LoRA 学读 M](./v1.7.1.3-A3-base-lora.md)
- [A4 v1.7.1.4 — KV-cache 注入](./v1.7.1.4-A4-kv-inject.md)
- [B1 v1.7.2.1 — 监督门](./v1.7.2.1-B1-supervised-gate.md)
- [B2 v1.7.2.2 — 富特征门](./v1.7.2.2-B2-rich-features.md)

主结果文档（setting/矩阵/门/成本）：[../results-v1.7.md](../results-v1.7.md)
