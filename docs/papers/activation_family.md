# 激活离群家族精读：SmoothQuant / AWQ / LLM.int8()

> 依据：arXiv 2211.10438（v7，ICML'23）、2306.00978（v6，MLSys'24 最佳论文）、2208.07339（v2，NeurIPS'22）的 arXiv HTML 全文，并以官方代码（mit-han-lab/smoothquant、mit-han-lab/llm-awq 的 `auto_scale.py`/`auto_clip.py`）交叉验证。
> 对照实现：`liteforge/smooth.py`（从零 SmoothQuant：α 网格搜索 + 蒙特卡洛 W8A8 损失）、`liteforge/lossmeter.py`（统一损失货币）。
> 实测锚点：双模型（0.5B / 1.5B）最优 α=0.7（非文献常用 0.5），无平滑→最优 α 的 W8A8 损失下降 3.65× / 3.98×（`docs/decision_log.md` D20、`docs/TECHNICAL_REPORT.md`）。

**结论速览**：
- 三篇论文是同一条"离群轴"上的三个坐标：LLM.int8 实证刻画离群（涌现于 ~6.7B、~0.1% 维、幅度 >6）；SmoothQuant 用等价变换把激活侧离群"搬"进权重（W8A8）；AWQ 把同一变换用于 weight-only，只压权重侧误差（激活侧免费）。
- SmoothQuant 与 AWQ 在数学上是同一等价变换族的两个实例，差别只在目标函数：前者是两侧都量化的零和跷跷板，后者是单向受益的加权误差最小化。
- 我们的 α=0.7 偏离论文 0.5 可以被四条机制解释（模型规模低于涌现阈值 / 权重 per-channel 更鲁棒 / max 统计噪声 / 损失口径不同），且论文自身对 Llama-2-7B 也取 0.85——"α 随模型漂移、给搜索不给定值"是文献内部一致的结论。

## 1. SmoothQuant（Xiao & Lin et al., ICML 2023）

### 1.1 核心思想
LLM 权重天然易量化，激活含系统性离群通道难量化。SmoothQuant 用数学等价变换把激活侧的量化难度**离线迁移**到权重侧：

    Y = X·Wᵀ = (X·diag(s)^-1)·(diag(s)·W)ᵀ，    s_j = max|X_j|^α / max|W_j|^(1-α)    (Eq.3/4)

α 是"跷跷板"：α→1 时 s_j≈max|X_j|，难度全压给权重（坏）；α→0 时 s_j≈1/max|W_j|，难度全留在激活（坏）；目标是两侧难度均衡。激活范围动态，s 必须在校准集上离线估计。

### 1.2 可复现级细节
- **α 的论文取值并非常数 0.5**：OPT/BLOOM 全系 α=0.5；GLM-130B α=0.75（约 30% 通道为离群）；LLaMA-1 α=0.8；Llama-2-7B/13B α=0.85、70B α=0.9；Falcon-7B/40B α=0.6/0.7；Mistral、Mixtral α=0.8。OPT-175B 消融：α<0.4 激活难量化，α>0.6 权重难量化，甜点 0.4–0.6。
- **搜索协议**：在 Pile 验证集子集上做"快速网格搜索"（论文未列网格点），选出的是**全局单 α**（非逐层）。
- **离线/在线分工**：用 512 条 Pile 句子一次性校准平滑因子 s 与静态量化步长；s 离线融合进前一层参数（前一 Linear 或 LN；输入含残加时需在**残差支路**上加同样缩放），运行时激活"天然平滑"，零额外 kernel。GLM-130B 特例：校准静态步长时裁掉 top 2% token。
- **粒度（Table 2，O1/O2/O3）**：O1=权重 per-tensor、激活 per-token 动态；O2=双 per-tensor 动态；O3=双 per-tensor 静态。**激活 per-channel 精度好（OPT-6.7B 64.8% vs FP16 64.9%）但与 INT8 GEMM 不兼容**——GEMM 后只能沿 token 维/输出通道维乘 scale（Eq.2），逐输入通道的激活 scale 非法。全程对称量化；非对称（如 ReLU 后）加 zero-point。
- **引擎**：PyTorch（HuggingFace）与 FasterTransformer 双后端，线性层与 attention BMM 用 CUTLASS INT8 GEMM（PyTorch 侧封装为 torch-int 的 `Int8OPTForCausalLM`，权重与 scale 打包导出）；Softmax/LayerNorm/ReLU 等逐元素算子保持 FP16。最高 1.51×（PyTorch）/1.56×（FT）加速、~2× 省内存、解码 1.42×；与标准 Tensor Parallelism 兼容，MT-NLG 530B 压进单节点 8×A100（73.1% 精度不变）。
- **MCP**：**未获取**——arXiv v1/v7、ar5iv 及官方 README 均未出现该术语，无法核实其定义与机制（不排除来自其他来源的记忆混淆）。

### 1.3 与我们实现的对照（liteforge/smooth.py）
- `smooth_scale` 与 Eq.4 一致（eps=1e-5 防零）；`collect_act_max` 用前向预钩子取逐通道 running max，对应论文的 max|X_j|。
- 粒度：权重 **per-channel（每输出行）对称 W8**、激活 **per-token 动态对称 A8**——比论文 O1 的"权重 per-tensor"更细，对更高 α 更鲁棒（见 4.3）。
- 搜索：我们做**逐层 α** 扫描（`w8a8_alpha_sweep`，网格 {0.4,…,0.8}，no_smooth 以 s=1 并入同一循环），目标是逐层**蒙特卡洛** W8A8 输出误差 ‖Q(X')Q(W')ᵀ−X'W'ᵀ‖²（校准 batch 累加，在等价空间计算，数学上与原空间相同；两遍校准——先采 act_max，再逐 α 重放，权重侧量化每配置只做一次）。论文是全局单 α + 端到端指标。
- `apply_smoothquant` 只固化权重侧（W←W·diag(s)），激活侧除以 s 留给部署引擎——即论文"离线融合进前层"那一步未做。
- 实现上有意保守的点：`w8a8_alpha_sweep` 只重放 ≤8 个校准 batch、逐层钩子串行累加（精度换可行性）；`no_smooth` 以 s=1 并入同一循环保证可比性；`apply_smoothquant` 每层返回 s 列表供导出，但激活静态步长、残差支路缩放、前层融合均未落码。与论文口径的差距集中在"部署工程"而非"变换数学"。

## 2. AWQ（Lin et al., MLSys 2024）

### 2.1 核心思想与误差分析
Weight-only（W4/W3A16）低比特量化。观察：仅 1% 显著通道贡献大部分量化误差，且**显著性由激活幅度而非权重大小决定**（按权重范数或随机选 1% 通道保 FP16，PPL 与 RTN 持平甚至更差：OPT-1.3B 108.71/98.55/98.08 vs 随机 119.76）。缩放技巧：显著通道权重乘 s>1、激活除以 s，输出不变但相对误差下降：

    Err(Q(w·s)(x/s)) / Err(Q(w)x) = (Δ'/Δ)·(1/s)

依据：round 误差近似均匀分布于 [0,0.5]（均值 0.25）；缩放单通道几乎不改变组内 max（Δ'≈Δ）。实测（OPT-6.7B，1% 显著通道）：s=1 PPL 23.54；**s=2 最优 11.92**；s=4 反弹 12.36（伤及非显著通道）。Δ'≠Δ 的组占比随 s 增大：0%/2.8%/4.4%/8.2%/21.2%（s=1/1.25/1.5/2/4）。

### 2.2 可复现级细节
- **层目标函数**：s* = argmin_s ‖Q(W·diag(s))·(diag(s)^-1·X) − WX‖，X 为小校准集（Pile）缓存激活；无反传（Round 不可导，STE 收敛不稳）。
- **Auto Scale**：参数化 s = s_X^α，s_X 为**逐通道平均激活幅度**（官方代码 `x.abs().view(-1, x.shape[-1]).mean(0)`；注意 SmoothQuant 用 max，口径不同），α 在 [0,1] 网格搜索（论文 grid size 20；代码 `ratio = i/20` 即 0~0.95），目标为块输出 MSE（`(org_out-out).pow(2).mean()`），并做 `scales/(scales.max()*scales.min()).sqrt()` 归一。
- **搜索单位是层组**（官方 `auto_scale_block`）：LN→{q,k,v}（inspect self_attn）、v_proj→o_proj、后置 LN→{gate,up}、up_proj→down_proj，组内共享 s 向量；融合经 `scale_ln_fcs`/`scale_fc_fc`，GELU/SiLU 用 `ScaledActivation` 包装；缓存激活同步 `div_(s)`。
- **Auto Clip**（论文一句带过，细节在官方 `auto_clip.py`）：逐组搜索截断值 max_val = 原组 absmax × (1 − i_s/n_grid)，n_grid=20、max_shrink=0.5 → 10 个候选；目标 = token 级输出 MSE（`(input_feat·q_w).sum(-1)` 与原输出之差按 token 维平方平均）；逐通道选最优；输出通道按 256/64 分批防 OOM；**q_/k_/query/key/Wqkv 投影跳过 clip**（qk BMM 难以精确 clip）。流程是 scale 搜索 → clip 搜索的单向流水（同一文件内未见交替迭代）。
- **量化口径**：全程 group size 128 的 per-group absmax 量化；缩放按输入通道施加（W·diag(s)、x/s）；scale 离线融合进前一算子。TinyChat：LN 多算子单 kernel、QKV 合并投影、位置编码即时计算、KV cache 预分配；桌面/移动 GPU 3× 于 HF FP16，70B Llama-2 可上手机 GPU。

## 3. LLM.int8()（Dettmers et al., NeurIPS 2022）

### 3.1 离群特征的实证刻画
- **涌现**：~6.7B 出现"相变"——受影响层从 65%→100%、受影响序列维从 35%→75%；~6B 起出现比其他维大 20× 的特征并在 ~25% 层中扩散。2048 token 序列约 15 万个离群值，**集中在仅 6 个特征维**（13B 至多 7 维），占全部输入特征 ~0.1%。
- **幅度**：阈值 α=6（经验上"把性能退化降到近零"）；普通维约在 [−3.5, 3.5]，13B 离群维四分位数 (−63, −58, −45)。
- **位置**：仅出现在 attention 投影（q/k/v/o）与 FFN 扩张层（第一子层）；attention 函数内部与 FFN 收缩层未纳入分析。
- **破坏机制**：离群单侧偏斜 + absmax 缩放 → 大多数量化 bin 空置、小值塌缩为 0。把离群维置零：top-1 softmax 概率 40%→20%，PPL +600~1000%；置零 7 个随机维仅 0.02–0.3% / ~0.1%。离群数量随 C4 PPL 下降单调增、与参数量非单调。这组对照（置零离群 vs 置零随机维）是论文最锋利的实验设计：把"离群重要"从相关性陈述升级为因果陈述，任何 SmoothQuant/AWQ 类方法的消融都值得抄这个"随机对照"模板。

### 3.2 可复现级细节
- **Vector-wise 量化**：X（s×h）逐行常数 c_x、W（h×o）逐列常数 c_w（c = 127/absmax）；INT8×INT8 → INT32 累加，反量化 C_f16 ≈ (1/(c_x⊗c_w))·C_i32 = S·Q(X)·Q(W)。仅靠它 ~2.7B 内无损。
- **混合精度分解**（Eq.8）：O = {i：该维出现过 |x|>6}；C ≈ Σ_{h∈O} X_f16^h·W_f16^h + S·Σ_{h∉O} X_i8^h·W_i8^h——离群列走 FP16 matmul，与反量化后的 INT8 结果在 FP16 输出相加；|O|≤7 → 额外 ~0.1% 显存，>99.9% 乘法仍在 INT8；BLOOM-176B 显存 1.96×。分解后 zero-point 的优势消失，但 vector-wise 仍优于 row-wise（权重精度仍有意义）。
- **Kernel 与速度**：PyTorch+NVIDIA 默认量化 kernel 几乎全线变慢，需自定义 CUDA kernel（bitsandbytes）；裸 INT8 cuBLASLt 仅在模型维 ≥5140 时约 2× 于 FP16 cuBLAS。相对 FP16 的 matmul 吞吐：0.14×(768)→0.86×(4096)→1.22×(5140)→1.81×(175B 尺度)——**<6.7B 模型变慢是硬约束**。BLOOM-176B 每 token：bf16 8 卡 239ms vs int8 8 卡 253ms，但 int8 3 卡 247ms——单卡略慢、换卡数省内存。8×RTX3090 跑 175B，12GB Colab 跑 11B。

## 4. 关键对照

### 4.1 AWQ 的"激活感知缩放" vs SmoothQuant 的"难度迁移"
**同**：二者是同一等价变换族 Y=(X·diag(s)^-1)(W·diag(s))ᵀ 的两个实例；都用逐输入通道 scale；都用"闭式参数化 + [0,1] 网格搜索（grid≈20）"而非反传；scale 都离线融合进前层。
**异**（目标函数不同）：
- SmoothQuant 两侧都量化（W8A8），损失同时含 Q(X') 与 Q(W') 的误差，α 调节两侧难度配比，是**零和跷跷板**；统计口径取 **max**。
- AWQ 只有权重侧量化（激活 FP16），X/s 只是"分母"，激活侧无代价，最优解是"给高激活通道的权重放大以降误差"，是**单向受益**（激活侧免费）；统计口径取 **mean**，且叠加 clip 搜索压量化 MSE。
一句话：这解释了 AWQ 敢用更激进的逐通道放大（s=2 最优），而 SmoothQuant 必须在两侧找平衡点（α 压在 0.4–0.9 之间）。我们 lossmeter 对 weight-only 用闭式 tr(ΔW·H·ΔWᵀ)、对 W8A8 退回蒙特卡洛，正是同一分野：激活量化误差依赖具体 token，无法用 H 闭式表达。

### 4.2 LLM.int8 离群分析 vs 我们的合成实验
LLM.int8 为 SmoothQuant 的假设提供了微观证据：离群维极少（~0.1%）、幅度远超普通维（>6 vs ~3.5）、单侧偏斜。我们 D16 的合成实验（构造离群通道主导的合成 w_max/act_max 分布）踩过的两个坑与此吻合：其一，离群主导时 per-token absmax 把量化预算浪费在少数维——即 LLM.int8 的"bin 空置"机制；其二，`smooth_scale` 参数传反（s 整个倒置）时平滑灾难性失效——s 的方向性就是 Eq.4 的方向性，等价变换不保护错误方向。可操作推论：离群越极端（大模型/LLaMA 系），α 应越高或需 outlier 分解兜底；离群温和（0.5B/1.5B），适度 α 即可。
另一层对照值得显式记下：LLM.int8 的分解方案（离群列走 FP16）与 SmoothQuant 的平滑方案是**同一问题的两条解法**——前者保留离群的数值身份（运行时分流），后者抹平离群的数值身份（离线迁移）。SmoothQuant 之所以是更优的部署答案，是因为 INT8 GEMM kernel 不允许逐输入通道的激活 scale（论文 Table 2 的灰格），而离群恰恰落在输入通道维；分解方案绕开了这个 kernel 约束，代价是两条数据通路。我们的合成实验可以作为这两条路线的共同测试床：合成"离群维数 × 离群幅度"的二维网格，分别测平滑最优 α 与分解最优阈值的行为差异——若按 LLM.int8 的刻画（≤7 维、幅度 >6），一个 0.5B 模型的合成分布应当落在"平滑足够"区域，这正是 4.3 解释 1 的可检验形式。

### 4.3 我们 α=0.7 vs 论文 0.5 的可能解释
1. **模型规模**：0.5 是 OPT/BLOOM（130B+ 尺度）的取值；LLM.int8 的涌现相变在 ~6.7B，我们的 0.5B/1.5B 远在阈值之下、离群温和，可安全多迁移。论文自己在 Llama-2-7B 用 0.85——"小/新模型 α 偏高"与文献自身一致。
2. **量化器差异**：论文 O1 权重 per-tensor（更脆，压更多难度到权重会先崩），我们权重 per-channel（更鲁棒，容得起更高 α）。
3. **校准统计**：我们 act_max 是 ≤16 batch 的 running max（max 统计噪声大、随 batch 数增长），论文 512 句并与静态步长联合校准；max 偏小 ⇒ 实际平滑不足 ⇒ 搜索推向更大 α 补偿。
4. **损失口径**：我们以逐层蒙特卡洛输出 L2 为目标（统一损失货币），论文以端到端 PPL/任务精度为准；逐层最优的组合 ≠ 端到端最优；且网格 {0.4..0.8} 步长 0.1，0.65/0.75 未测。
5. **部署口径的残余差异**：lossmeter 的 W8A8 损失把"等价空间中的 Q(X')Q(W')ᵀ−X'W'ᵀ"整体计价，隐含"激活侧 scale 已被引擎完美吸收"的理想化假设；真实引擎还有融合精度、残差支路缩放等实现误差，会轻微改变最优 α 的位置。
6. D20 结论依然成立：**给固定 0.5 不如给搜索**——α 本身是模型/校准/口径的函数。

## 5. 我们实现未覆盖的细节清单
SmoothQuant：
- [ ] s 融合进前层的完整离线导出（前一 Linear/LN；**残差支路额外缩放**）；`apply_smoothquant` 目前只固化权重侧，激活侧除法依赖部署引擎。
- [ ] O1/O2/O3 粒度选项与静态激活步长校准（含 GLM-130B 式 top 2% token 裁剪）。
- [ ] CUTLASS INT8 GEMM/BMM 引擎与非线性层 FP16 混合执行；**MCP：未获取**（arXiv v1/v7、ar5iv、官方 README 均无该术语）。
- [ ] α 搜索的端到端复核（论文：Pile 子集快速网格 + 全局单 α）。

AWQ：
- [ ] auto_scale 的**层组共享 s**（LN→{q,k,v}、v→o、LN→{gate,up}、up→down）与 `scales/(max·min)^0.5` 归一；GELU/SiLU 的 ScaledActivation 融合分支。
- [ ] **auto_clip 全套**：逐组 shrink 网格（n_grid=20、max_shrink=0.5 → 10 候选）、token 输出 MSE 目标、q/k 投影豁免。
- [ ] per-group g128 口径与 lossmeter 有效比特记账的对齐核对。

LLM.int8：
- [ ] 混合精度 split kernel（离群列 FP16 + 其余 INT8/INT32 累加 + 外积反量化 + FP16 相加）。
- [ ] 自定义 INT8 GEMM kernel 及"<6.7B 会变慢"的工程边界。
- [ ] vector-wise（行/列 absmax 外积反量化）对照基线。

## 6. 可借鉴设计清单（标工作量）
SmoothQuant：
1. 补齐 `apply_smoothquant` 的激活侧导出：s 融合进前层 + 残差支路缩放，产出可部署的 W8A8 权重包（0.5–1 天）。
2. 把 O2/O3（per-tensor 权重/静态激活）加进 lossmeter 菜单，用统一货币比较粒度-开销权衡；静态激活步长还能去掉运行时 absmax 的访存开销（1 天）。
3. 校准规模消融：act_max 从 16 batch → 512 句 / 分位数统计，检验 α=0.7 是否漂移；论文的"512 句一次校准、s 与步长联合估计"是更稳的口径（0.5 天）。
4. α 搜索加端到端 PPL 复核，防逐层过拟合；同时把网格加密到 0.05 步长覆盖 0.6–0.9（1 天）。

AWQ：
1. auto_clip 移植进 weight-only 通道（作为 w4g128/w3g128 菜单项的配套优化，目标与 lossmeter 同口径）（1 天）。
2. 消融：AWQ 的 mean 口径 s_X^α vs SmoothQuant 的 max 口径 act_max^α——同构搜索、不同统计的对照（0.5 天）。
3. 层组共享 s（LN→{q,k,v}、up→down），降低搜索自由度与逐层过拟合（0.5 天）。
4. 直接采纳"q/k 投影豁免 clip"的经验（0.1 天）。

LLM.int8：
1. **离群维度分析器**：按层统计 |x|>6 的维度数/占比/幅度分位，作为 α 选择与"哪些层需更强处理"的证据源（0.5 天）。
2. W8A8 的 outlier 分解推理原型：极端层离群列走 FP16，替代对全模型无差别平滑（2–3 天）。
3. 小模型离群涌现检查：验证 0.5B/1.5B 是否处于 6.7B 涌现阈值之下，为 α=0.7 提供文献级解释（0.5 天）。
4. vector-wise 对照基线进 lossmeter（1 天）。

## 来源
- SmoothQuant: https://arxiv.org/abs/2211.10438 （HTML v7/v1、ar5iv）；代码 https://github.com/mit-han-lab/smoothquant
- AWQ: https://arxiv.org/abs/2306.00978 （HTML v6/v3）；代码 https://github.com/mit-han-lab/llm-awq （auto_scale.py / auto_clip.py）
- LLM.int8(): https://arxiv.org/abs/2208.07339 （HTML）
- 本仓库：`liteforge/smooth.py`、`liteforge/lossmeter.py`、`docs/decision_log.md`（D16/D20）、`docs/TECHNICAL_REPORT.md`
