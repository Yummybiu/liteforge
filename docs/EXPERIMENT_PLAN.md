# LiteForge 实验计划（W1-W2 主线 + 完整实验矩阵）

## 目标

一张表回答面试官的核心问题："压缩率换了多少质量？实际部署能快多少？"
即 **压缩率-质量-速度 三角 trade-off** 的完整证据链。

## 第一阶段：本机实验（✅ 已完成 2026-09-03，13 组真实数据见 README/results/）

| 模型 | 方法 | 覆盖点 |
|---|---|---|
| Qwen2.5-0.5B | dense | fp16 基线 PPL 13.21 + 生成速度 |
| Qwen2.5-0.5B | Wanda | 25% / 50% / 60% / 2:4（稀疏度扫描）|
| Qwen2.5-0.5B | Magnitude | 50%（激活感知对照，PPL 485.6 vs Wanda 25.5）|
| Qwen2.5-0.5B | RTN | W8 g128 / W4 g128 / W4 per-ch / W3 g128（分组消融 + 崩溃点）|
| Qwen2.5-1.5B | dense + Wanda 50% + RTN W4 | 规模效应对照 |

已产出：results/table.md（主对照表）、results/tradeoff.png（等效比特-PPL 曲线）。
核心发现三条：激活感知 19 倍差距；分组粒度决定量化质量；模型越大越耐压。

## 第二阶段：系统实验矩阵（A5000 24G 可独立完成）

模型 × 方法 × 评测的网格（每格一次 CLI 调用，全自动）：

- **模型**：Qwen2.5-0.5B / 1.5B / 3B（本机）；7B（实验室卡，排期）
- **剪枝**：Wanda {0.25, 0.5, 0.6, 2:4} × Magnitude {0.5}（对照）
- **量化**：RTN {W8, W4g128, W4g32, W3g128(观察崩溃)}；GPTQ W4g128；AWQ W4g128
- **组合**：GPTQ W4 → +Wanda 2:4（压缩叠加，观察交互损失）
- **评测**：
  - 质量：wikitext2 PPL（2048 窗口，≥256 blocks）、MMLU 子集（lm-eval，可选）
  - 速度：本机 generate tok/s；vLLM serve 实测（dense vs GPTQ-Int4 vs AWQ-Int4）
  - 显存：vLLM 启动日志中的 weight memory

## 关键产出物

1. `results/table.md`：主对照表（PPL × 方法 × 模型）
2. `results/tradeoff.png`：等效比特 vs PPL 曲线
3. `docs/vllm_bench_report.md`：部署吞吐/TTFT 实测报告
4. 博客《我压了三个 Qwen：Wanda/GPTQ/AWQ 的真实 trade-off》

## 口径纪律（诚实性是面试生命线）

- 剪枝模型评测的是**质量损失**（权重置零的稠密推理），不声称速度收益；
  速度收益只在 vLLM（GPTQ/AWQ 整数内核）和 2:4 sparse tensor core 场景声称；
- RTN 是伪量化口径（量化误差模拟），GPTQ/AWQ 是真实可部署口径，表格中分列；
- 所有数字均可由仓库脚本一键复现。
