# 相关工作与定位（Related Work）

> 目的：V2 的"压缩预算分配"不声称真空原创——这个方向有活跃文献。
> 本文如实记录前人工作与本项目的 delta。面试被问"这和 HAWQ/DayPQ 什么关系"时，
> 这里的每一行都是答案。

## 直接相关文献

| 工作 | 做了什么 | 与本项目的关系 |
|---|---|---|
| [Kuzmin et al., NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/c48bc80aa5d3cbbdd712d1cc107b8319-Paper-Conference.pdf)《Pruning vs Quantization》 | 在固定预算下正面比较剪枝与量化（208+ 引用） | 我们复现其核心结论之一（同比特下量化保真度更高），并把比较做成**开源测量框架** |
| [SmoothQuant (mit-han-lab, arXiv:2211.10438)](https://github.com/mit-han-lab/smoothquant) | W8A8 激活量化：等价变换把离群从激活迁移进权重，α 平衡两侧难度 | 本仓库从零实现 + α 网格搜索（蒙特卡洛损失口径），W8A8 补齐 weight-only 之外的技术纵深 |
| [Speculative Decoding (Leviathan et al., ICML 2023)](https://arxiv.org/abs/2211.17192) | draft-target 投机解码，贪心验证输出分布不变 | 从零实现并以单测证明精确性定理；对齐 JD"解码加速/投机解码"关键词 |
| [Revisiting Pruning vs Quantization for SLMs](https://casszhao.github.io/cass/QP.pdf) | 逐层 SNR 视角比较两者；发现量化逐层保真度始终更高 | 直接可复现假设：损失表应在逐层粒度验证同一结论 |
| HAWQ 系列（ICML'19 等） | Hessian 迹敏感度 → 混合精度**量化**位宽分配 | 分配思想的鼻祖，但只覆盖量化菜单；我们扩展到"量化∪剪枝"统一菜单 |
| [BESA (arXiv 2402.16880)](https://arxiv.org/html/2402.16880v2) | 块级重建损失驱动的**非均匀稀疏度**分配 | 剪枝侧的非均匀分配先例；我们的菜单含量化选项，损失度量跨方法族 |
| [DayPQ (IEEE 2026)](https://www.computer.org/csdl/journal/si/2026/06/11419640/2eyKHh9FQSQ) | 动态逐层"剪枝+量化"联合分配 | 最接近的同期工作（期刊版）；我们提供开源可复现管线 + 精确 DP 对照 + 预测力验证 |
| [Joint Structural Pruning & MPQ (arXiv 2502.16638, CVPR 2025)](https://arxiv.org/html/2502.16638v1) | 端到端联合结构剪枝与混合精度量化 | 面向结构化/训练感知，路线不同；我们是 PTQ/免训练范畴 |
| [ApiQ](https://github.com/pprp/awesome-llm-quantization/blob/main/README.md) 等 | 联合压缩+恢复的信息损失框架 | 恢复侧互补；我们不训练 |
| [Discovering Sparsity Allocation (2025)](https://www.researchgate.net/publication/397217245_Discovering_Sparsity_Allocation_for_Layer-wise_Pruning_of_Large_Language_Models) | 放弃均匀稀疏假设、搜索稀疏分配 | 剪枝内部分配；我们跨方法族并用 DP 求精确解 |

## 本项目的 delta（面试口径）

1. **统一度量**：把 OBC 剪枝损失与 GPTQ/RTN 量化损失放进同一货币（校准输出 L2，
   tr(ΔW·H·ΔWᵀ)），前人方法各用各的度量，跨族比较历来是各论文自己说了算；
2. **精确分配**：分离式背包的**动态规划精确解**（非启发式），与贪心/均匀分配做成
   可复现消融——文献里的分配多为敏感度启发式；
3. **预测力验证**：损失表的预测值与实际 PPL 变化的相关性检验——"成本模型可信"
   本身是被验证的结论而非假设；
4. **全开源可复现**：上述文献均无完整开源的"测量→分配→应用→验证"管线
   （截至 2026-09 检索），本仓库补齐。

## 诚实声明

- 本项目是**工程-科学框架**，不声称算法新颖性超越上述任一论文；
- 分配实验的结论（无论正负）都以"在统一度量下"为限定语。
