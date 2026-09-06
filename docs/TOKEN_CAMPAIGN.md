# Token 攻坚战台账（2026-09-06，额度当日有效）

> 目标：把 2 亿 token 额度全部转化为 LiteForge/RLForge/求职的持久资产。
> 纪律：每个子代理必须产出落盘文件（docs/papers/、docs/reviews/、rlforge/assets/ 等），
> 禁止无产出烧 token；汇报只写 3-5 行摘要保持主上下文精简；限流则降并行。

## 战果登记（按波次）

### Wave 1 — 论文实现级精读军团（进行中）
| 代理 | 主题 | 产物 | 状态 |
|---|---|---|---|
| A | 二阶误差补偿族：GPTQ/SparseGPT/OBC | docs/papers/second_order_family.md | 🔄 |
| B | 剪枝族：Wanda/LLM-Pruner/Sheared-LLaMA | docs/papers/pruning_family.md | 🔄 |
| C | 激活离群族：SmoothQuant/AWQ/LLM.int8 | docs/papers/activation_family.md | 🔄 |

### 待发波次
- Wave 2: 极低比特与压缩联合族（QuIP#/AQLM/SpQR/SqueezeLLM）+ 结构化（Sheared-LLaMA 补充）
- Wave 3: RL 族精读（DeepSeek-R1/DAPO/Dr.GRPO/投机解码 Leviathan+EAGLE/HAWQ）
- Wave 4: 代码审查军团 ×8 视角（数值正确性/协议健壮性/CLI一致性/测试缺口/文档一致性/安全/性能/打包）
- Wave 5: RLForge 弹药工厂（验证任务工厂 500+ 题 / veRL 源码集成指南 / GRPO 理论深潜）
- Wave 6: 面试题库军团（八股 ×8 主题，每题三层深度）
- Wave 7: 历史实验数据的二次深挖（results/*.json 全量再分析）

## 事后统计
（日终填写：代理数、产物清单、估算 token 消耗）

## 日终统计（2026-09-06 18:29）

| 波次 | 代理数 | 消耗(约) | 产物 |
|---|---|---|---|
| Wave 1 论文精读 | 3 | 4.0M | papers/second_order·pruning·activation_family.md |
| Wave 2 极低比特族 | 1 | 0.7M | papers/extreme_lowbit_family.md（D21 文献印证+菜单升级路径）|
| Wave 3 RL 族 | 1 | 0.9M | rlforge/docs/papers_rl_family.md（新增 2 个消融臂）|
| Wave 4 代码审查 | 2 | 6.6M | reviews/×2（P0×2/P1×6/P2×18，已修复+回归测试）|
| Wave 5 RLForge 弹药 | 1 | 1.1M | Countdown 600 + GSM8K 变体 1845 + 课程 7473 题 |
| Wave 6 面试题库 | 2 | 4.8M | bagugu_quant_infer.md(20题) + bagugu_rl_transformer.md(27题)|
| Wave 7 数据深挖 | 1 | 1.4M | analysis_deepdive.md（每比特效率排名/分配可解释性）|
| **合计** | **11** | **~19.5M** | **14 份持久资产 + 5 个代码修复** |

**结论**：子代理串行+限流的现实上限约 20M/日（目标的 10%）——但全部转化为
持久资产，零空烧。剩余额度价值最高的用法是：下次 GPU 实验批的自动分析管线（A 方向），
而非今日强堆数量。
