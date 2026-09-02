# LiteForge 设计说明

## 定位

LiteForge 回答一个具体的问题：**把一个开源 LLM 压缩下去，质量掉多少、部署能快多少？**
它不是又一个"微调框架"，也不是 RAG demo——它是一条从算法（剪枝/量化）到系统
（vLLM/llama.cpp 部署实测）的完整证据链。

## 架构

```
                 ┌──────────────────────────────────────────┐
                 │                cli.py（统一入口）          │
                 └──────┬───────────┬───────────┬───────────┘
                        │           │           │
        ┌───────────────▼──┐  ┌─────▼─────┐  ┌──▼────────────┐
        │ prune/           │  │ quant/    │  │ eval/         │
        │ MagnitudePruner  │  │ RTN(从零) │  │ perplexity    │
        │ WandaPruner      │  │ GPTQ 封装 │  │ speed         │
        │ BasePruner(可插拔)│  │ AWQ 封装  │  └──┬────────────┘
        └───────┬──────────┘  └─────┬─────┘     │
                │    data/text.py   │           │
                │  (wikitext2/分块) │           │
                └────────┬──────────┘           │
                         ▼                      ▼
                 results/*.json ──► report/ ──► table.md + tradeoff.png
                                            └► deploy/vllm_bench.py（端侧实测）
```

## 关键设计决策

### 1. 剪枝评估的诚实口径
剪枝后的权重以**稠密形式置零存储**，评测回答的是"方法学的质量损失"；
速度收益只在两类场景声称：GPTQ/AWQ 的整数内核（vLLM 实测）与 2:4 半结构化
（Ampere sparse tensor core）。混用两类口径是这类项目最常见的注水点，本仓库
从 schema 上就分开（`prune` 记录无速度指标列，`vllm-bench` 单独 task 类型）。

### 2. RTN 从零实现 vs GPTQ/AWQ 封装
RTN（Round-To-Nearest）是所有现代量化方法的基线，从零实现的价值：
- 面试可现场推导 scale/zero-point（本仓库即参考实现）；
- 伪量化可在无整数内核环境（如本机 Windows + CPU torch）评估量化误差；
- 与 GPTQ/AWQ 的真实打包形成"方法学上限 vs 可部署下限"的对照。

### 3. Pruner 可插拔接口
`BasePruner` 只要求子类实现 `compute_scores()`，激活统计（钩子）、掩码生成、
层遍历、报告全部由基类承担。新剪枝方法（包括未发表的方法）约 30 行代码接入，
自动获得全链路评测与对照表。**未发表论文的方法不进本公开仓库**，以私有分支接入。

### 4. 统一结果 schema
所有命令输出同一 JSON 结构（task/model/method/params/metrics/env），
report 命令即可聚合。env 字段带 torch/CUDA/时间戳——可复现性是一等公民。

### 5. 离线可测试
12 个单元测试用 48-vocab 微型 Llama + 字符级 mock tokenizer，CPU 10 秒跑完，
不依赖任何下载。这是仓库可信度的基础设施。

## 已知边界（诚实清单）

- [x] 剪枝无稀疏内核加速（见口径纪律）
- [ ] 滑动重叠窗口 PPL（当前为非重叠协议，与 lm-eval 一致）
- [ ] GPTQ/AWQ 实测依赖可选库安装（wrappers 已就绪）
- [ ] MMLU 子集评测（预留 lm-eval 接入点）
