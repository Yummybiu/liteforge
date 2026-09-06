# LiteForge 双视角代码审查：数值正确性 × API 协议健壮性

> 审查日期：2026-09-05。范围：`liteforge/` 全部 33 个源文件（约 3140 行），逐行通读。
> 视角一：数值正确性（dtype 混用、累加精度、边界条件、数学公式 vs 文档）。
> 视角二：API 协议健壮性（tokenizer/模型返回形式、pad 方向、attention_mask、HF 接口版本、设备/dtype 一致性）。
> 纪律：只报能指出具体行号与触发路径的问题；无法在本机复现的点标注"待验证"。
> 历史项 D9（动态掩码）/D11（合成测试分布）/D14（DP 回溯污染、贪心方向）/D16（smooth_scale 参数序）已修复，不再重复报。

---

## 一、发现清单

严重度定义：P0 = 崩溃；P1 = 数值错误（结果静默变错）；P2 = 健壮性/协议缺口；P3 = 风格/文档/度量。

### P0（1 条）

**F1. `torch_dtype` 关键字在 transformers 5.x 已移除，显式 dtype 加载路径崩溃**
- 位置：`liteforge/models.py:24`（`kwargs["torch_dtype"] = torch_dtype`），配合 `pyproject.toml:16`（`transformers>=4.44` 无上界）。
- 问题：transformers 4.56 起 `from_pretrained` 的 `torch_dtype` 弃用并改名为 `dtype`（PR #39782），v5.0 按迁移指南移除旧名。`load_pair`（cli.py:38-42）把 `--dtype fp32|fp16|bf16` 传进来，凡显式指定 dtype 的加载都会走到 models.py:24。
- 触发条件：新装环境（2026 年必然解析到 transformers 5.x）+ `--dtype` 非 auto 的任何命令（eval-ppl / prune / quant-* / loss-report / apply-alloc / speed / eval-mmlu）。
- 建议修复：改为 `kwargs["dtype"] = torch_dtype`；如需兼容旧版可 try `TypeError` 回退 `torch_dtype`。同时给 `transformers` 加上界或版本探测。
- 待验证：v5 是否保留兼容别名（若保留则降级为 P2 警告噪音）；本机无 Python 无法实测，依据为官方迁移指南与 vLLM/SGLang/diffusers 的跟进 issue。
- 附带（P2）：models.py:26-32 的 `dtype="auto"` 语义在 v5 下改变——v4 默认加载 fp32、再于 CUDA 上转 bf16/fp16（models.py:30-32）；v5 默认按 checkpoint config dtype 加载（bf16 权重不再先过 fp32），CPU 上的"auto"不再保证 fp32。数值口径静默漂移，需在 CHANGELOG 声明或显式传 dtype 消除歧义。

### P1（2 条）

**F2. `quantize_tensor` 在 n_in 不被 group_size 整除时静默退化为 per-row 量化（pad 后忘了 reshape）**
- 位置：`liteforge/quant/rtn.py:88-94`（非整除分支），对照组：正确实现 `rtn.py:96-98`（整除分支）。
- 问题：`wf = F.pad(wf, (0, pad))` 之后直接调 `_quant_groups(wf, bits, symmetric)`——`_quant_groups` 把**最后一维整体当一个组**（rtn.py:34-35 "w: (..., G)"），传入的 2D 张量被当作 (n_out, 整行) 做 per-output-channel 量化，而非 (n_out, n_groups, group_size) 分组量化；pad 的 0 还进入了该行的 min/max。pad 变得毫无意义，量化粒度静默变粗。
- 触发条件：任何 `n_in % group_size != 0`。CLI `quant-rtn --group-size` 任意可达（如 `--group-size 100`）；真实模型也存在不被 128 整除的维度（如 Falcon d_model=4544）。`tests/test_quant.py` 全部用整除尺寸（group_size=16/32/128 vs 维度整除），未覆盖。
- 建议修复：pad 后 `g = wf.reshape(n_out, -1, group_size)` 再量化、再切片；或与 `group_params`（rtn.py:73 注释的"不可整除→per-row 降级"）统一口径并打 warning。
- 严重度：P1（结果静默错误，且错误方向是"量化比用户要求的更粗"，PPL 会变差而非崩溃）。

**F3. `_gptq_layer` 的量化区间写死为非对称 `[0, qpeak]`，`symmetric=True` 时所有负权重被置零**
- 位置：`liteforge/quant/gptq.py:53`（`q = clamp(round(w/s) + z, 0, qpeak)`）；配合 `rtn.py:64-65, 74-76`（symmetric 时 `z = zeros_like(s)`）。
- 问题：非对称区间对 z=0 的对称量化会把 `round(w/s) < 0` 的全部 clamp 成 0，`dq = (0-0)*s = 0`——每一列所有负权重直接归零，量化误差爆炸。对照：RTN 的对称路径是正确的（rtn.py:40-42 clamp `[-qmax, qmax]`）。
- 触发条件：程序化调用 `GPTQQuantizer(model, RTNConfig(bits=4, group_size=128, symmetric=True)).quantize_(calib)`。`quant/__init__.py:1` 公开导出 GPTQQuantizer，gptq.py:71 明确声明 config 兼容 `symmetric` 字段；CLI `quant-gptq` 未暴露 `--symmetric`（cli.py:488-500），故仅库调用可达。仓库内损失表路径恒传 `symmetric=False`（lossmeter.py:133、allocate.py:208），不受影响。现有 GPTQ 测试全部 `symmetric=False`（tests/test_gptq.py:24,37），未覆盖。
- 建议修复：`_gptq_layer` 内按 symmetric 分支：`q = clamp(round(w/s), -qmax, qmax); dq = q*s`（z 恒 0）。
- 严重度：P1（文档承诺支持的配置下静默产出垃圾权重）。

### P2（6 条）

**F4. dict 型校准批次（带 attention_mask）的支持分支是死路——到达前就 AttributeError**
- 位置：`liteforge/utils/hessian.py:46-51`；同型复制在 `liteforge/prune/base.py:130-135`。
- 问题：`collect_xtx` 先执行 `batch = batch.to(device)`（:46）——dict 没有 `.to`，直接 AttributeError；`prune/base.py:131` 的 `batch.dim()` 同理。即使传入 3D tensor，else 分支 `batch["input_ids"]`（hessian.py:50）也是 TypeError。写了 attention_mask 分支（:50-51、base.py:134-135）却没有任何输入形态能走到那里。
- 触发条件：调用方按代码契约传入 HF tokenizer `padding=True` 产出的 dict 批次（含 `attention_mask`）做校准。
- 建议修复：`if isinstance(batch, dict): batch = {k: v.to(device) for k, v in batch.items()}` 前置；3D tensor 分支语义明确化（或删除）。
- 严重度：P2（仓库内部 BlockBatcher 恒为 2D 等长张量所以现网不炸，但公开 API 契约破裂）。

**F5. 钩子统计不排除 padding 位置（与 F4 同根，修复 F4 后即触发）**
- 位置：`liteforge/utils/hessian.py:33`（`reshape(-1, last)` 后全量 `xᵀx`）；`liteforge/prune/base.py:116`（x_sum 同样全量）。
- 问题：模型拿到了 attention_mask（:50-51），但钩子仍把 pad 位置的隐状态计入 H / Wanda 平方和——右 padding 的尾部、左 padding 的头部都是垃圾激活。`collect_act_max`（smooth.py:62）同理。
- 触发条件：F4 修复后传入 padded 批次；或任何不等长 2D 手工拼接批次（当前 API 拒绝 dict、2D 只能等长，故现网未触发）。
- 建议修复：pre_hook 里接收 mask（functools.partial 传入或用 `with_kwargs=True`），`x = x[mask]` 后再统计。
- 严重度：P2（条件触发）。

**F6. `damp_inverse` 在"全部对角为 0"时 damp=0 → cholesky 崩溃（MoE 未激活 expert / 全零输入层）**
- 位置：`liteforge/utils/hessian.py:61-66`。
- 问题：`dead = H.diagonal() == 0` 后用 `H.diagonal().mean()` 回填——若**所有**对角为零（该层校准期间输入恒零），mean=0，回填无效，`damp = percdamp * 0 = 0`，`torch.linalg.cholesky(零矩阵)` 抛 LinalgError。官方 SparseGPT 对 dead 对角填 1.0 正是为避免此边界。
- 触发条件：MoE 模型（`find_linears` 会收集全部 expert，未收到任何 token 的 expert H 全零）；或任何全零输入层。伴生崩溃点：`smooth.py:124/178` 的 `act_max_map[name]` 为 None → `smooth_scale(None, ...)` AttributeError。
- 建议修复：`H[dead, dead] = 1.0 if dead.all() else mean`（或 raise 带明确信息的 ValueError）；smooth 侧对 None 显式跳过并告警。
- 严重度：P2（仓库现役 dense 模型不触发，接口对 MoE 不设防）。

**F7. MMLU 选项打分未 cast fp32，bf16 下 logsumexp 近平局可翻转 argmax**
- 位置：`liteforge/eval/mmlu.py:89-91`（单 token 快路径）、`:96-103`（多 token 回退）。
- 问题：bf16 模型的 logits 直接 `logsumexp`、`logits[t] - lse`，尾数只有 8 bit，四个选项 logprob 差在 1e-2 量级时排序可翻转；同仓库 perplexity.py:65 已做 `.float()`，此处漏了。该评测是"压缩前后 acc 对比"的证据链，翻转直接污染结论。
- 触发条件：`--dtype bf16`（默认 auto 在 Ampere+ CUDA 上正是 bf16）跑 `eval-mmlu`。
- 建议修复：`logits = model(**enc).logits[0, -1, :].float()`（回退路径同样对 `logits` 整体 `.float()`）。
- 严重度：P2（数值口径，无崩溃）。

**F8. `speculative_generate` 隐含 batch=1 契约：B>1 时 `int(t)` 崩溃、EOS 只看第 0 条**
- 位置：`liteforge/speculative.py:84`（`int(t)`，t 为 [B,1]）、`:89`（`int(ids[0, -1])`）。
- 问题：函数签名接受任意 input_ids；B>1 时 `int(t)` 抛 "only one element tensors..."，且即便绕过，终止判断也只检查 batch 0。另注：多 batch 下首个 mismatch 的 break 会对所有元素统一截断（丢弃 batch 内其它序列已被接受的 draft），虽然不破坏贪心等价性（丢的 token 会被重新 draft 出来），但效率口径失真。
- 触发条件：`speculative_generate(target, draft, ids.shape[0] > 1 的批次, ...)`。
- 建议修复：入口 `assert input_ids.shape[0] == 1` 或向量化终止判断。
- 严重度：P2。

**F9. `greedy_allocate` 对不可行预算静默返回超预算方案，与 DP 的失败语义不一致**
- 位置：`liteforge/allocate.py:135-161`（`best is None` 即 break 后直接 return）。
- 问题：`target_bits` 低于全场最省平均比特时，循环内所有升级被 `bits + d_bits > target` 过滤 → 返回 min-bit 配置，`achieved_bits > target` 无任何告警；同场景 `dp_allocate` 明确 raise（allocate.py:96-97, 100-102）。且 CLI 的 DP 失败回退链（cli.py:287-301）只捕 MemoryError，MemoryError 回退到 greedy 时用户拿到的是"超预算且无提示"的结果。
- 触发条件：`--strategy greedy --target-bits` 低于 min 桶平均比特（如 2.0 bit 而最省选项为 p75≈4.0 bit）。
- 建议修复：return 前比较 `bits > target_bits + eps` 则 `logger.warning`（或 raise，与 DP 对齐）。
- 严重度：P2（achieved_bits 字段本身诚实，属契约不一致而非数据错误）。

### P3（风格/文档/度量，7 条）

**F10. `apply_allocation` 给 dynamic 模式传了语义错误的 `Hinv_diag`（未被使用，纯误导）**
- 位置：`liteforge/allocate.py:204-206`。`prune_copy(..., U.diagonal().clamp(...), U, mode="dynamic")`——dynamic 分支（lossmeter.py:80-81）不消费该参数；且 `U.diagonal()`（=U_jj，SparseGPT 块打分口径）与 static 路径用的 `(H⁻¹)_jj`（lossmeter.py:124-125）不是同一个量。将来若有人把 mode 改成 static，会静默用错打分。建议传 `Hinv.diagonal()` 或删参。

**F11. DP 的状态数守卫没算上层数维度；死变量 OPT**
- 位置：`liteforge/allocate.py:72-78`。守卫只约束 `budget+1 > 4M`，但 `choice` 是 `(n_layers, budget+1)` int8——200 层 × 4M ≈ 800MB。`allocate.py:75` 的 `OPT` 未使用。

**F12. fp32 模型下 GPTQ/OBC 的 `mean_abs_weight_err` 度量失真（`.to(float32)` 别名）**
- 位置：`liteforge/quant/gptq.py:83,89`；`liteforge/prune/obc.py:127`。fp32 模型时 `.to(torch.float32)` 返回原张量，`_gptq_layer`/`_sequential_zero_*` 的误差反馈就地改写原始权重：最终 `copy_` 结果仍正确（量化值覆盖），但 gptq.py:89 的 err 是"补偿后权重 vs 量化值"而非对原始权重的误差，报告字段口径随模型 dtype 漂移。建议 `W = m.weight.data.detach().clone().to(torch.float32)`（或固定在 clone 上算 err）。

**F13. OBC docstring 与实现的打分公式口径不一致**
- 位置：`liteforge/prune/obc.py:16`（docstring 称 score = w²/(H⁻¹)_jj）vs `:50`（dynamic 实现用 U_jj²）。实现是 SparseGPT 官方口径（正确），docstring 描述的是 static 口径；两口径并存但文档未区分。

**F14. OBCPruner 对 `config.structure == "2:4"` 静默忽略；`compute_scores` 是 NotImplementedError 地雷**
- 位置：`liteforge/prune/obc.py:104-105, 117-129`。CLI `prune --method obc --structure 2:4` 会按非结构化跑且无提示（cli.py:120-126）。

**F15. smooth 的权重侧量化每 batch 重做，与 docstring "只做一次/α" 不符**
- 位置：`liteforge/smooth.py:104` vs `:135`。确定性计算、数值无损，纯性能+文档问题（每钩子调用 `quant_w8_perchannel(W_s)`）。

**F16. `apply_smoothquant` 只固化 W·diag(s)，直接 float 评测返回模型会得到错误 PPL**
- 位置：`liteforge/smooth.py:166-182`。1/s 未融合进前置 LayerNorm（SmoothQuant 标准做法是缩放 LN 的 affine），docstring 已声明激活侧由部署引擎承担——但"保存后的模型不可直接运行"这一后果建议升级为显式运行时警告或提供 LN 融合选项，否则用户拿去 eval 会静默踩坑。

**F17. 杂项**
- `liteforge/eval/mmlu.py:53`：docstring 称"分层抽样（按 subject 比例）"，实现是 shuffle+head 的均匀抽样（:60-62）。
- 死变量/死导入：`models.py:29`（tok_dtype 赋值后未用）、`eval/speed.py:23`（vocab）、`cli.py:247, 273`（numpy import 未用）。
- `liteforge/data/text.py:53-54`：空文本时 `ids[0]` IndexError；`text.py:70` `if max_blocks:` 使 `--max-blocks 0` 退化为不限量。
- `liteforge/allocate.py:190, 213`：不在 `find_linears` 里的层也被计入 `skipped_fp16`；`allocate.py:197` 对非 `DEFAULT_MENU` 选项名直接 KeyError（自产自销链路不受影响）。

---

## 二、已验证正确的关键路径（正面证据）

1. **lossmeter 的 tr(ΔW·H·ΔWᵀ)**（lossmeter.py:61-64）：全程 float64，`compression_loss(H, Wc, W)` 的参数序与 ΔW=W−W_c 一致（:140 调用点核对无误）；tr(ΔW·XᵀX·ΔWᵀ)=‖XΔWᵀ‖²_F 恒等式有 float64 单测（tests/test_lossmeter.py:13，D12）。
2. **Hessian 链路**（hessian.py:17-66）：阻尼 `percdamp·mean(diag)` 官方同款；部分死列处理正确（mean>0 时）；返回 float64；`H⁻¹ = UᵀU` 的 `cholesky(Hinv, upper=True)` 与官方 GPTQ/SparseGPT 链路一致（lossmeter.py:126、obc.py:110、gptq.py:85 三处一致）。float32 累加 XᵀX 是已声明的设计（docstring :23），D12 已知其误差量级。
3. **GPTQ 逐列误差反馈**（gptq.py:36-61）：块内 `err=(w−dq)/U_jj` 传播到 `j+1..b1`（:57-58），块间 `block_errᵀ @ U[b0:b1, b1:]`（:60-61），与官方 GPTQ 完全一致；分组 scale 在组起点按**当前已补偿**权重重算、组可跨块（:41-45），官方同款；group_size=0 per-channel 正确（:46-47）；非对称 zero-point 的 clamp 链正确。D8 的数字（W4g128 14.13 vs RTN 15.69）与此实现自洽。
4. **OBC 顺序补偿**（obc.py:70-91）：static 打分用 `(H⁻¹)_jj`（:131-132，OBS 单权重损失口径），`err=w_j/U_jj`、后续列 `-= err⊗U[j, j+1:]` 与多权重 OBS 的顺序形式等价；dynamic 块级重评分（:32-67）块内定秩（:48-51，U_jj²=SparseGPT 官方口径）、块间传播（:65-66）正确。D9 的 static→dynamic 修复在位。
5. **Wanda**（wanda.py:42-44 + base.py:107-118）：float64 累加平方和（与 hessian 的 float32 不同、更稳），RMS=sqrt(x_sum/n)，`|W|·RMS` 广播方向正确（act_norm.unsqueeze(0) 沿输出行广播）；n==0 有护栏（wanda.py:40-41）。
6. **SmoothQuant 等价变换**（smooth.py:85-96）：`(X/s)·(Ws)ᵀ = XWᵀ` 恒等；`s_j = act_max^α / w_max^(1-α)`（:45-49），`(w_max, act_max)` 参数序与 D16 修复一致，两处调用（:124、:178）均先 w 后 act、均按 dim=0 取输入通道 max，核对无误；per-token 激活 / per-channel 权重的量化算子正确；`best_alpha` 只在 α 之间比较、no_smooth 不参选（:157-159）；CLI 聚合键为 float 与库内 float 一致（cli.py:367-372）。
7. **DP 分配器**（allocate.py:56-117）：预算按平均比特、GCD 缩放（:62-69），步长 `d_u[i]*bucket` 与预算单位一致；回溯前捕获终点 `pos_end`（:104，D14①修复在位），`choice`/`best` 的更新与回溯自洽（:90-95, 106-114）；`achieved_bits = pos_end·granularity·d_gcd/total_d` 量纲正确；不可行预算 raise（:96-97, 100-102）；状态数守卫 + CLI 三档降粒度再回退贪心（cli.py:287-301）；同桶合并保留最小损失（:34-44，含 p50/w8 同桶的诚实注释）。`test_dp_matches_brute_force` 在位。
8. **Greedy**（allocate.py:121-161）：从最省可行点出发、按"每 bit 损失改善"排序升级（:147-152，D14②修复在位）；增量按平均比特计。
9. **投机解码**（speculative.py:41-90）：teacher-forcing 位置对齐 `n_ctx-1+i`（:62）；修正采纳、for-else 红利 token（:66-73）；`min(k, 剩余)` 防超发（:45）；EOS 与 max_new_tokens 截断（:78-90）；单测覆盖 exactness（tests/test_speculative.py）。D18 的 bf16 位级差异是已知浮点现实、非代码缺陷，CLI 已输出 exact 字段（cli.py:400）。
10. **PPL**（perplexity.py:60-71）：shift 正确；`logits...float()` 后 cross_entropy；reduction=sum + 全局 token 平均；空输入有明确 ValueError（:57-58）。BlockBatcher 等长块 → 无 padding → 不需要 attention_mask，口径成立。
11. **RTN 本体**（rtn.py:34-50, 96-101）：对称 clamp [-qmax,qmax]、非对称 zero=round(-wmin/s)∈[0,qpeak]、死组 scale clamp(min=1e-8)；整除分组路径正确（pad 分支见 F2）。
12. **有效比特记账**（lossmeter.py:39-45）与 allocate 文档、plots.effective_bits 三处口径一致；剪枝 16·(1−s) 诚实口径贯通。
13. **per_row_topk_mask / 2:4**（base.py:52-71）：round+max(1,·) 边界合理；2:4 组内 top2、n_cols%4 校验在位；Wanda 2:4 强制 sparsity=0.5（wanda.py:51-55）。
14. **设备/dtype 一致性**：collect_xtx/smooth/prune 的钩子均在模型 device 上累加，`next(model.parameters()).device` 取法一致；calib 批次 `.to(device)` 统一（hessian.py:46 等三处）；models.py 的 bf16 能力探测（Ampere+）与 pad→eos 回退正确。
15. **协议兼容面**：`tokenize_to_ids`（text.py:48-55）对 list[int]/list[list[int]]/tensor 三种 tokenizer 返回形式均处理（mmlu.py:82-87 的局部实现覆盖了 list 两种，letter 场景够用）；`tokenizer(prompt, return_tensors="pt").to(device)` 用法正确；generate 传 pad_token_id 显式（speed.py:52,57）；报卡/聚合对坏 JSON 有 try/except（collect.py:23-26, card.py:24-27）。

---

## 三、建议补的测试清单（对应 P0/P1，兼 P2）

| 编号 | 对应发现 | 测试内容 |
|---|---|---|
| T1 | F1 (P0) | monkeypatch `AutoModelForCausalLM.from_pretrained` 捕获 kwargs，断言 `load_model_and_tokenizer(dtype="fp16")` 传出的是 `dtype=` 而非 `torch_dtype=`；并在 CI 里对 `transformers` 做主版本断言或兼容探测 |
| T2 | F2 (P1) | `quantize_tensor(W(4,100), bits=4, group_size=32)`：断言每组 32 列的 scale 在组内为常数（真分组量化）且逐组误差 == 手工分组 RTN 误差；同时断言与 `group_params` 在不可整除时的行为一致 |
| T3 | F3 (P1) | `_gptq_layer(W.clone(), U, bits, gs, symmetric=True)` 对照 RTN 对称：断言输出误差 ≤ RTN 对称误差，且 `(Wq[W<0] == 0).float().mean() < 0.5`（排掉"负权重全零"回归） |
| T4 | F6 (P2) | `damp_inverse(torch.zeros(n,n))` 不抛 LinalgError（或抛带信息的 ValueError）；`measure_menus`/`score_dry` 对 H 全零层不崩 |
| T5 | F7 (P2) | bf16 微型模型下 `_choice_logprobs` 输出与 fp32 对照的 pred 一致（或断言内部 logsumexp 的 dtype==float32） |
| T6 | F8 (P2) | `speculative_generate` 传 B=2：断言显式报错信息（固化 assert）或全 batch 正确终止 |
| T7 | F4/F5 (P2) | 构造 padded dict 批次 + attention_mask 跑 `collect_xtx`：断言与无 pad 的 H 逐元素一致（pad 位置不计入） |
| T8 | F9 (P2) | `greedy_allocate(target_bits < 最省平均比特)` 应 warning/raise，与 DP 的 ValueError 语义对齐 |

---

## 四、统计

- 发现合计 16 项：P0×1（待验证报错形态）、P1×2、P2×6、P3×7。
- 最严重两条：F1（transformers 5.x 移除 `torch_dtype`，显式 dtype 的所有加载命令崩溃/语义漂移）；F2（RTN 不可整除分组静默退化为 per-row 量化，数值静默变错）。
- 已知历史项 D9/D11/D14/D16 核对：修复均在位，未重复报。
