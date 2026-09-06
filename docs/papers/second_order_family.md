# 二阶误差补偿家族：GPTQ / SparseGPT / OBC 实现级精读与 LiteForge 对照

> 依据：arXiv abs 页 + ar5iv 全文（2210.17323、2301.00774、2208.11580）+ GPTQ v2（ICLR 2023 camera-ready）PDF + 官方代码（IST-DASLab/gptq `gptq.py`、IST-DASLab/sparsegpt `sparsegpt.py`、IST-DASLab/OBC `trueobs.py` 等）。所有公式与默认值均有出处；查证不到的细节明确标注"未获取/未证实"。对照的本地实现：`liteforge/utils/hessian.py`、`liteforge/quant/gptq.py`、`liteforge/prune/obc.py`、`liteforge/lossmeter.py`。

## 1. 统一框架：一切从 tr(ΔW·H·ΔWᵀ) 开始

三篇论文共用同一条层式损失（GPTQ Eq.1 / SparseGPT Eq.1 / OBC Eq.2）：给定层权重 W 与校准输入 X，找压缩权重 Ŵ 最小化

  argmin_Ŵ ‖WX − ŴX‖²₂

二次目标的 Hessian 为 H = 2·E[xxᵀ]（实现里常数 2 与均值/求和口径等价缩放，我们的 `collect_xtx` 用 sum 形式）。展开即 LiteForge 统一损失货币 L = tr(ΔW·H·ΔWᵀ)，ΔW = W − Ŵ。三者的全部差异只在于**如何在这个度量下选"动谁"与"怎么补"**：

- OBC：贪心地逐个选损失最小的权重动，动一个精确重算 H⁻¹（剪枝）或固定网格（量化）；
- SparseGPT：把"选谁删"（掩码）与"怎么补"（固定掩码下的闭式最优重建）解耦，块级流水线；
- GPTQ：把 OBC 的量化版从"逐权贪心"改成"任意固定顺序 + 逐列误差反馈"，换取大模型可扩展性。

## 2. GPTQ（arXiv 2210.17323，ICLR 2023）

**核心思想一句话**：基于近似二阶信息（Hessian）的逐列量化，把每一列的舍入误差按 H⁻¹ 的 Cholesky 因子反馈补偿给后续未量化列，从而 4 GPU 小时内量化 175B 模型到 3/4 bit。

**可复现级细节**：
- 基座 OBQ：对每行独立，Hessian H_F=2X_F X_Fᵀ；单权更新 δ_F = −w_q·(H_F⁻¹)_{:,q}/(H_F⁻¹)_qq（Eq.2）；H⁻¹ 用一步高斯消元更新（Eq.3）。
- **Step 1 任意顺序洞察**：贪心顺序相比任意固定顺序提升很小（大误差权重的代价会被"留到最后、可调整的权重少"抵消）。由此所有行共用同一列顺序与同一份 H⁻¹，运行时间从 O(d_row·d_col³) 降到 O(max{d_row·d_col², d_col³})。注意：论文只说"任意固定顺序"，**没有**提出按 H 对角降序排序——那是官方代码的 act-order（见下）。
- **Step 2 惰性批更新**：blocksize B=128。块内逐列：q=quant(w_j)，err=(w−q)/d（d=(H⁻¹)_jj），块内后续列 W[:,j+1:B] −= err·H⁻¹_{j,j+1:B}；整块完成后一次性把累积误差泼溅给块外所有剩余列 W[:,B:] −= E·H⁻¹_{block,B:}（Algorithm 1）。
- **Step 3 Cholesky 形式**：H⁻¹ ← Cholesky(H⁻¹)ᵀ（上三角 U），补偿只需 U 的第 j 行；数值稳定且内存友好。阻尼 λ = 平均对角 × 1%（percdamp=0.01）。
- **实验设置**：校准 = C4 随机 128 条 2048 token 片段；per-row 非对称 min-max 均匀量化；单张 A100-80GB（含 175B）；**逐 Transformer block 顺序处理**：量化完当前 block 的 7 个层后，把（部分量化的）输入前传给下一 block 采 Hessian——H 反映"已量化模型的真实激活"。
- **分组**：论文明确分组与 GPTQ 交互极好，"组参数可在量化过程中用最新已补偿权重现算"；g=1024 约 +0.02 bit 提升 ~0.2 ppl，g=128（+0.15 bit）再提升 ~0.1。

**论文/官方代码有、我们没覆盖的**：
1. **act-order（desc_act）**：不在论文正文（v1 与 ICLR v2 均已 grep 验证，"Additional Tricks"一节只讲 grouping），出自官方代码 `actorder=True`：按 diag(H) 降序 permute 列、量化后逆 permute。对低比特+分组尤其有效，是"任意顺序洞察"的工程化收敛加速。我们未实现。
2. **static_groups**：官方代码 `static_groups=True` 时先在**原始权重**上预计算每组 scale/zero（通常与 act-order 同开，因为 act-order 打乱列后动态分组会跨组污染）。我们做的是动态组（官方默认路径，`group_params` 在每组起点用当前补偿后权重重算——与官方 `find_params(W[:, ...])` 一致，这个是对的）。
3. **死列处理**：官方 `H[dead,dead]=1; W[:,dead]=0`（把从未激活的输入维对应权重直接清零）；我们 `hessian.py` 把死对角替换为均值但不清零权重——小模型上死列罕见，影响近零，但语义不同。
4. **逐 block 顺序重前向**：我们的 `collect_xtx` 一次全模型前向采齐所有层的 H（在**未压缩**激活上），官方是压缩完一层 block 重前传。这导致深层误差累积不被修正，是我们数字劣于论文口径的最大结构性差异之一。

## 3. SparseGPT（arXiv 2301.00774，ICML 2023）

**核心思想一句话**：把大规模一次性剪枝改写为"掩码选择 + 固定掩码下的闭式最优重建"，用块级流水线把 OBS 精确补偿做到 175B 规模（OPT-175B/BLOOM-176B 4.5 小时内 50-60% 稀疏）。

**可复现级细节**：
- 层式损失同 Eq.1；OBS 单权删除：δ_m = −w_m·(H⁻¹)_{:,m}/(H⁻¹)_mm，损失 ε_m = w_m²/(H⁻¹)_mm（Eq.3）。
- **Theorem 1（固定掩码 M 的闭式重建）**：Ŵ = W − (W − W(M))·diag(M)·(H⁻¹diag(M))⁻¹，其中 W(M) := W·diag(M)·(H⁻¹diag(M))⁻¹·H⁻¹。只需对 H⁻¹ 做 Cholesky（上三角）即可逐列执行。
- **Algorithm 1**：列按 Bs=128 分块；每块开始时用**当前已补偿权重**按 OBS 分数 w_c²/[H⁻¹]_cc² 重选掩码（这就是我们 dynamic 模式的出处）；块内逐列顺序补偿，块结束把累积误差 E 批量泼溅给所有剩余列（W[:,(i+B):] −= E·H⁻¹_{i:(i+B),(i+B):}）。社区所谓"泼溅策略"即此批量更新（论文原文无 "spilling" 一词——标注）。
- **blocksize 消融**：B=1（纯逐列）与接近全列（4096/8192）都显著变差；几百最优，取 128。
- **阻尼消融**：percdamp 在 0.001–0.1 间结果相近，默认 0.01；死列同 GPTQ 官方处理。
- **近似质量**：与精确 OBS 相比平均层误差约差 20%；种子敏感性 13.52±0.075 ppl。
- **实验设置**：校准 = C4 首个 shard 的 128×2048 token；WikiText2/PTB ppl；50%/60% 非结构化 + 2:4/4:8（n:m 时每 prunem 列按当前权重重新 top-k）；可与 GPTQ 量化复合。

**官方代码有、论文正文没有充分强调的**：非结构化掩码的官方实现是 `tmp = W1²/diag(Hinv)²` 后在**整块跨所有行 flatten 取分位数阈值**（`mask1 = tmp <= thresh`），即块内跨行全局分配删留名额，**不是逐行 top-k**；只有 n:m 分支才逐行选择。这是与我们实现最大的行为差异（见 §5）。另外 Hessian 采集官方是运行均值（`H *= n/(n+k)`，`sqrt(2/n)` 折入系数），逐 block 重前向与 GPTQ 相同。

## 4. OBC（arXiv 2208.11580，NeurIPS 2022）

**核心思想一句话**：把经典 OBS 做成精确且高效的层式框架（ExactOBS/OBQ），统一覆盖后训练剪枝与量化，并支持二者复合（BOP 约束混合压缩）。

**可复现级细节**：
- **逐行可分**（Section 3 关键观察）：损失可写成逐行平方误差之和，"删一个权重只影响对应输出行的误差，行与行之间没有 Hessian 交互"；H=2XXᵀ 全行共享。这从理论上支持了逐行独立选掩码（我们 static 模式的合法性）。
- 单权 OBS：δ_F = −w_q·(H_F⁻¹)_{:,q}/(H_F⁻¹)_qq；损失增量 ΔL_q = w_q²/(H_F⁻¹)_qq；贪心选 ΔL 最小者；多权推广即固定删除集的批量最优更新（与 SparseGPT Theorem 1 等价，`obc.py` 文档串里的 (H⁻¹)_{K,S}((H⁻¹)_{S,S})⁻¹ 形式同源）。
- **量化（OBQ）**：同一贪心框架作用到"舍入到网格"；**outlier 启发式**——舍入误差 > Δ/2 的离案权重一旦出现立即量化，防止留到最后无法补偿（GPTQ 为扩展性放弃了该贪心与启发式）。
- **数值工程**：H 不可逆来自样本太少或死/线性相关输入——前者用数据增广累积进 H（"增广样本只需累积进 Hessian 一次，非常便宜"），后者加对角阻尼；GPU 上用批操作把多行一起处理（官方 sample implementation）；对已稀疏权重按非零元素稠密化，复杂度随行密度三次方。
- **实验设置**：所有校准集 = 1024 条随机训练样本；ImageNet 用翻转/裁剪增广 ×10；ResNet 做 BN 统计重置，其余模型在归一化层后做**均值/方差修正**（AdaQuant 式统计校正）；逐层非均匀目标用"model database + SPDY 动态规划"求解；量化网格：主实验非对称 per-channel（LAPQ 定网格），BOP 混合实验对称 per-channel。
- **重要边界**：OBC 论文实验只有 ResNet/YOLOv5/BERT，**没有 decoder LLM**；结论明确把"very large-scale language models"列为未来工作（即后来的 SparseGPT/GPTQ）。
- **Bessel 校正：未获取/未证实**。论文全文与官方代码（trueobs.py、spdy.py、quant.py、main_trueobs.py）grep 均无 "Bessel"/无偏修正字样；官方 Hessian 采集为运行均值（H *= n/(n+k)），最接近的记忆点可能是上述"均值/方差修正"（统计校正），它作用于归一化层统计量而非 Hessian。

## 5. 与 LiteForge 实现逐条对照

| 项 | 论文/官方 | 我们 | 结论 |
|---|---|---|---|
| H 采集 | H=2XXᵀ，运行均值 | `collect_xtx` sum 形式 XᵀX（fp32 累加） | 等价缩放，正确 |
| 阻尼 | 平均对角×0.01，消融 0.001-0.1 | `damp_inverse` 同款，percdamp=0.01 默认 | 正确（官方同款） |
| 求逆 | float32：Cholesky(H)→cholesky_inverse→Cholesky upper | float64 同流程 | 更稳，正确 |
| 死列 | H 对角置 1 且 W[:,dead]=0 | 对角置均值、不清零 | 语义不同；小模型影响≈0 |
| GPTQ 误差反馈 | err=(w−q)/d，块内 + 块间泼溅 | `_gptq_layer` 完全一致（块内更新 j+1..b1，块末 block_err@U） | 正确 |
| GPTQ 分组 | 动态组：组起点用当前权重 find_params | 同款（`group_params` 重算） | 正确 |
| act-order / static_groups | 官方代码选项，论文无 | 未实现 | 未覆盖 |
| 逐 block 重前向 | 官方 GPTQ/SparseGPT 均是 | 一次全模型前向采 H | 未覆盖（结构性差距来源） |
| SparseGPT 块级重评分 | 每块用当前权重重选掩码 | `_sequential_zero_dynamic` 同款 | 正确 |
| 非结构化掩码粒度 | 官方：块内**跨行全局**阈值（w²/d²） | per-row top-k（dynamic 块内、static 全列） | 不同（我们更贴 OBC 逐行理论；官方让大权重行多留名额） |
| 打分分母 | Algorithm 1/官方用 (diag)² | dynamic 用 (diag)²；static 用 diag（OBC ε 形式） | 两种口径各自与出处一致 |
| OBC 静态消融 | OBC：行间无 Hessian 交互 | static 模式 + `score_dry` | 方向正确 |
| 损失货币 | L=‖ΔW X‖² | `compression_loss`=tr(ΔW·H·ΔWᵀ)（fp64） | 正确，同一把尺 |

**"做错了可能吗"的显式判定**：核对后未发现数学错误——误差反馈方向（用 U 的第 j 行、除以对角 d）、块间泼溅（E·U[block, rest:]）、动态分组重算均与官方逐步一致；`_sequential_zero` 静态版把误差更新写到行尾（j+1..n_in）而非官方的"块内+块末批量"，两者数学等价，只是少了块级批量化的缓存友好性。真正算"错"风险的只有两处：① static 模式在 50% 以上稀疏度会系统性**高估**自己的质量（删留决策不随补偿刷新，decision_log D9 记录的失败模式即其表现，dynamic 模式已修复）；② 死列不清零会让从未激活的输入维权重照常参与量化，若上游出现死特征会白白消耗位宽——语义偏差而非实现 bug。

**数字差异来源**（GPTQ W4g128=14.13 为 Qwen2.5-0.5B，稠密 13.21；OBC dynamic 50% = 17.52/11.67/9.81 对应 0.5B/1.5B/3B，稠密 13.21/9.26/8.01）：
1. **模型规模**：论文最小到 OPT-125M 时增益同样明显收窄；0.5B 级冗余少，稠密基线 13.21 本身高，任何一次性压缩的相对恶化都放大。
2. **无逐 block 重前向**：我们的 H 全部来自未压缩前向，24 层（0.5B）误差逐层累积无修正；论文设置里后续层"看到"的是已压缩激活。
3. **校准分布**：论文用 C4 校准（GPTQ/SparseGPT），评测 WikiText2；我们校准与评测同为 WikiText2 文本，且 32×4×2048≈262K token 恰与论文 128×2048 同量级——量不是问题，分布是（同分布校准对 GPTQ 类方法偏乐观/偏悲观取决于评测口径，此处需注明）。未获取：论文在 Qwen 系架构上的数字（论文不含 GQA/SiLU 模型）。
4. **无 act-order**：对 W4g128 官方经验有可观提升（代码注释/后续工作口径），我们未开。
5. **OBC 50% 的 17.52**：官方 SparseGPT 的块内跨行全局阈值会把删留名额向"大权重行"倾斜，我们逐行强制 50% 更刚性；加上第 2 条，0.5B 上仍比 Wanda（25.52）好 31%。

## 6. 可借鉴设计清单

**GPTQ**（1）act-order 列重排 + 逆重排：~50 行，0.5 天，预期 W4/W3g128 在 0.5B-1.5B 上显著收益；（2）static_groups（与 act-order 配套）：~40 行，0.5 天；（3）逐 Transformer block 顺序压缩+重前向：~120-150 行，1-2 天，最大结构性收益（也惠及 OBC 剪枝）；（4）死列官方语义：~5 行，10 分钟。

**SparseGPT**（1）块内跨行全局阈值掩码开关（`--obc-mask global`）：~20 行，0.5 小时，可直接对照 17.52 是否再降；（2）把每层 Σ(w−q)²/d² 的过程量写进 PruneResult，作为免费的逐层敏感度（与 `score_dry` 互补）：~20 行，0.5 小时；（3）n:m 分支（每 prunem 列重选 top-n）接入 2:4 配置：~50 行，1 天（`group_2to4_mask` 已有静态版）。

**OBC**（1）校准增广累积进 H（数据增强只加 XᵀX，近零成本）：~15 行，1 小时；（2）model database + 动态规划非均匀分配（`lossmeter.measure_menus` 已产出逐层 (bits_eff, loss) 菜单，`allocate.py` 是雏形）：补 SPDY 式 DP ~150 行，2-3 天，是 V2 预算分配的学术正名；（3）二次剪枝时的非零元素稠密化（对已剪模型再剪）：~40 行，0.5 天；（4）均值/方差修正：对 Qwen RMSNorm（无 running stats）不适用——不落地，仅记录。

## 7. 统一损失货币：三方法在同一把尺下的表达

对任意压缩算子 A 产生的 ΔW_A，层损失恒为 L_A = tr(ΔW_A·H·ΔW_Aᵀ)（`lossmeter.compression_loss` 已实现）。三篇论文的方法可视为**在同一 H 度量下不同的 ΔW 生成过程**：

- **RTN**：ΔW 第 j 列即独立舍入误差 e_j，L 为各列平方误差按 H 对角加权和——无人管理的噪声。
- **GPTQ**：第 j 列误差被立即摊派到未量化列：W[:,j+1:] −= e_j/d·U[j,j+1:]，等价于在"剩余列可自由调整"的约束下把 e_j 沿 H 度量最陡方向分摊；其单列即时损失 (w−q)²/d² 正是 OBS ε 在量化下的对应物。因此 GPTQ 后的 ΔW 在 H 度量下逐列被"预对冲"，`lossmeter` 用 gptq impl 实测的 loss 应系统性低于 RTN——这正是 14.13 vs 15.69 的来源。
- **SparseGPT/OBC 剪枝**：固定掩码 M 时 Theorem 1 给出 ΔW 的闭式最优解，此时 L = tr(W diag(M)(H⁻¹diag(M))⁻¹diag(M) Wᵀ)（剪枝残差的 H 范数）；掩码选择即在这个量上做组合优化，SparseGPT 的对角近似 → w²/d² 分数。static 掩码用"原始权重"估计该量，dynamic 用"已补偿权重"逐块刷新——后者是把 O(全搜索) 的联合优化用块级贪心逼近。
- **统一推论**：量化与剪枝在同一 L 下可直接比价（`bits_eff` 与 loss 构成菜单），逐层最优分配 = 在 Σ L_i ≤ 预算下最大化质量，OBC 的 database+DP 是其精确解法；跨层非线性项（H 依赖上游压缩）则由"逐 block 重前向"吸收——我们当前缺失的那一块恰好是三篇论文工程实现的共同骨架。

## 8. 抓取来源与未获取项

已核实：三篇 abs 页；ar5iv 全文（GPTQ/SparseGPT/OBC）；GPTQ v2 PDF（ICLR camera-ready，验证无 act-order）；官方代码 gptq.py/sparsegpt.py/trueobs.py/spdy.py/quant.py/main_trueobs.py。未获取/未证实：OBC 论文的数学公式在 PDF 提取中乱码（以 ar5iv 渲染为准）；"Bessel 校正"在三篇论文与官方代码中均不存在；"泼溅"为社区称谓，论文原文无此词；GPTQ/SparseGPT 在 Qwen 类模型上的官方数字未获取。
