# 版本历史 / Changelog

本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。
全部实验数字可复现：对应 tag 的 `scripts/` + `results/*.json`。

## [0.3.0] — 2026-09-04

### 新增
- **V2 研究层**：统一损失计价器 `lossmeter`（剪枝/量化同尺 `tr(ΔW·H·ΔWᵀ)`）、
  预算分配器 `allocate`（**精确 DP** + 贪心 + uniform 三策略消融）、
  `apply-alloc` 预测-实际闭环命令
- **SmoothQuant W8A8**（从零）：离线等价变换、α 跷跷板网格搜索、
  蒙特卡洛损失口径（激活误差按 batch 计入）
- **投机解码**（从零）：draft-target 贪心验证，单测证明输出与 target 纯贪心
  逐 token 一致（k=1/2/8）
- **MMLU-mini 似然评测**（lm-eval 口径，STEM/人文/社科分桶）
- **压缩报告卡**：单文件 Markdown 汇总一个模型的全部压缩证据
- examples/ 六个可运行示例 · Dockerfile · 跨库交叉验证脚本 ·
  `export`（压缩模型 HF 导出 + 压缩清单）

### 修复
- DP 回溯位置变量污染（achieved_bits 恒 0）
- 贪心分配方向反转（目标高于最省点时永不触发）
- report/card 相对导入深度错误

## [0.2.0] — 2026-09-04

### 新增
- 自研 **OBC/SparseGPT 族剪枝**：Hessian 引擎（XᵀX 采集→阻尼求逆→Cholesky），
  静态预选与**块级动态重评分**两种掩码（`--obc-mask`）
- 自研 **GPTQ**（逐列量化 + 误差反馈，组起点重估 scale）
- SGMix 敏感度混合稀疏（**负结果**入库：OBS 干跑敏感度不迁移 + 损失凸性）
- 动态 OBC 三规模实验：0.5B/1.5B/3B = 17.52/11.67/9.81，**全胜 Wanda**
- CI（GitHub Actions，py3.10/3.12 矩阵）

### 修复
- 全模型预存 Cholesky 因子导致的 3B OOM（改逐层用完即释放）

## [0.1.0] — 2026-09-03

### 首个可用版本
- Wanda / Magnitude 剪枝（从零，含 2:4 半结构化）
- RTN 伪量化（从零，对称/非对称/分组，可 restore）
- 滑窗困惑度（lm-eval 口径）+ 吞吐基准 + vLLM 压测客户端
- 统一 JSON 结果 schema + Markdown/CSV 聚合 + trade-off 图
- 16 组真实实验（Qwen2.5-0.5B/1.5B/3B）· 12 项离线单测
