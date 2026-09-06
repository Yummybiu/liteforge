# LiteForge 四视角代码审查：测试覆盖 × 文档一致性 × CLI 一致性 × 安全打包

> 审查日期：2026-09-05。范围：`tests/`（13 文件）、`liteforge/` 全部源码、`cli.py` 14 个子命令、
> `examples/` 6 脚本、`scripts/` 11 个工具、`docs/` 全部文档、`pyproject.toml` / `Dockerfile` /
> `.gitignore` / `Makefile` / CI workflow、`results/*.json`（43 条记录）。
> 方法：逐行通读 + 实机复现（本机 conda env：Python 3.12.14 / transformers 5.16.1 / torch 2.14.0+cu126；
> 55 项离线单测实跑 23.14s 全部通过）。
> 纪律：只报有具体行号/出处的；所有标注"已复现"的问题均在本机实际触发过。
> 与 [review_numerics_protocol.md](review_numerics_protocol.md) 的关系：该报告的 F1–F17 不重复；
> 本文编号 R1 起。特别注意 **R1 是 F1 修复本身引入的新 P0 回归**。

严重度定义：P0 = 崩溃；P1 = 功能失效/命令跑不通/结果无效；P2 = 健壮性、一致性、覆盖缺口；P3 = 文档/风格/打包卫生。

---

## 一、发现清单

### P0（1 条，已复现）

**R1. `models.py:25` — 显式 `--dtype` 的所有加载路径 NameError（CLI 全线可达，无任何测试覆盖）**
- 位置：`liteforge/models.py:24-26`；影响 `cli.py:38-42`（load_pair）与 `cli.py:392-393`（spec-bench）。
- 问题：`from transformers import AutoModelForCausalLM, AutoTokenizer`（:19）**不会**绑定名字
  `transformers`，随后 `major = int(transformers.__version__.split(".")[0])` 直接
  `NameError: name 'transformers' is not defined`。该行是 review_numerics_protocol F1
  （torch_dtype→dtype 改名）的修复代码，修复本身引入了新崩溃。
- 已复现：`load_model_and_tokenizer('definitely/not/a/model', dtype='fp16')` →
  `NameError : name 'transformers' is not defined`（NameError 发生在 from_pretrained 之前）。
- 触发条件：任何子命令 `--dtype fp32|fp16|bf16`（eval-ppl / prune / quant-* / loss-report /
  apply-alloc / smooth-alpha / speed / eval-mmlu / spec-bench）。默认 `auto` 恰好绕过该分支，
  所以全部离线单测与日常冒烟都测不到。**直接后果：decision_log D18 / README:24 /
  TECHNICAL_REPORT:81 声称的 "fp32 重跑 exact=True 1.96×"（results/spec_qwen15b_k4_fp32.json）
  在当前代码下无法复现**——`spec-bench --dtype fp32` 必然 NameError。
- 修复建议：`import transformers`（模块级或函数内）后再取 `transformers.__version__`；
  同时按 F1 的 T1 建议补测试（见第三节 #1）。

### P1（4 条）

**R2. `examples/04_budget_allocation.py:39` — `alloc` 模式 NameError（已复现）**
- `run_alloc` 使用 `np.prod`，但 `import numpy as np` 只在 `run_loss` 函数体内（:20）。
  `python examples/04_budget_allocation.py alloc --losses results/losses_qwen0.5B.json --target-bits 2.5`
  → `NameError: name 'np' is not defined`。示例 04 的第二步从未能按原样跑通。
- 修复：把 `import numpy as np` 提到模块级；顺带删除 ：16 与 ：15 的重复 `setup_logging` 导入。

**R3. `cli.py:470-471` — `prune --include` 是静默 no-op（声明了参数，从未消费）**
- 全文检索 `args.include` 零消费；`cmd_prune` 构造 `PruneConfig(sparsity=..., structure=...)`
  （:123、:130）从不传 `include=`。`BasePruner.__init__`（prune/base.py:98）与
  `find_linears`（utils/common.py:45-64）都支持 include，唯独 CLI 链路断接。
- 触发：`python -m liteforge.cli prune --model X --include mlp ...` → 全模型照剪，无任何提示。
  "只剪 mlp 层"的实验者会得到完全错误的结论。
- 修复：`PruneConfig(sparsity=sparsity, structure=args.structure, include=tuple(args.include))`；
  在补齐前先从 parser 删除该参数。

**R4. `Dockerfile:20` — CMD 跑 pytest，但镜像里没有 pytest**
- `pip install -e .`（:12）只装核心依赖；pytest 在 `dev` extra（pyproject.toml:27）未被安装。
  `docker run liteforge` 默认命令 `python -m pytest tests/ -q` → `No module named pytest`。
- 修复：`RUN pip install --no-cache-dir -e ".[dev]"`（或显式 `pip install pytest`）。

**R5. README:44 与 examples/04:50-51 的 `apply-alloc --eval` 命令必然失败（已复现）**
- `apply-alloc` parser（cli.py:545-553）没有 `--eval` 参数，且 cmd_apply_alloc（:337）
  本来就无条件评测。实跑：`liteforge: error: unrecognized arguments: --eval`（exit 2）。
  README 六步示例、examples/04 的"下一步"提示都是这条死命令。
- 修复（二选一）：文档删去 `--eval`；或给 apply-alloc 增加 `--eval`/`--no-eval` 开关并让
  :337 条件化（后者与其它命令的 `--eval` 惯例更一致）。

### P2（12 条）

**R6. allocate 记录缺 `strategy` 顶层键 → 报告卡与预测力分析脚本输出 None（已复现，含已入库产物）**
- 根因：`cmd_allocate`（cli.py:305）把策略写进 `method` 字段，记录没有顶层 `strategy`。
  连锁：① `cmd_apply_alloc`（cli.py:340）`alloc_rec.get("strategy")` 恒 None → apply 记录的
  `params.strategy` 为 None；② `report/card.py:69` `a.get("strategy")` 恒 None → 报告卡打印
  `- None @ target 2.25 bit`（已入库的 `results/card_0.5B.md:29-36` 即如此）；③
  `scripts/analyze_prediction.py:31` strategy 列恒 None（实跑验证输出全为 None）。
- 修复：`cmd_allocate` 在 rec 上加 `rec["strategy"] = res["strategy"]`（或统一读
  `a.get("method")`）；重新生成 card_*.md。

**R7. `smooth.py:100-163` `w8a8_alpha_sweep` + `collect_act_max`（:54-80）零测试**
- 这是 smooth-alpha 命令的核心（两遍校准、钩子逐 batch 累加、α 循环内缓存权重），复杂度
  远超已测的 `_batch_w8a8_loss`/`smooth_scale`。test_smooth.py 只测了数学算子；钩子机制、
  `by_alpha`/`no_smooth` 路由（:151-155）、`best_alpha` 聚合全部无测试。tiny_model 夹具完全
  够跑——纯离线可测而未测，属"真机才暴露"高危缺口。
- 另：CLI `--alphas` 传空列表（nargs="*" 允许）时 `min(by_alpha, key=...)`（cli.py:372）
  对空 dict 抛 `ValueError: arg is an empty sequence`。

**R8. `report/card.py:31-98` `build_report_card` 零测试**
- V1 行拼装（:46-55）、allocs/applies 段（:63-81）、三个任务段（:84-91）全无测试。
  R6 的 None 污染、`_fmt` 对 None 的处理（:17-18）、`rows` 为空时缺章节等，只有 golden 测试
  才能守住。它还是 Makefile `card` target 与 README 宣传的"一文件报告卡"。

**R9. `cli.py:287-301` DP 降级/回退链零测试**
- `dp → granularity 0.5 → 1.0 → greedy` 的 MemoryError 回退链、`g != args.granularity` 的
  warning 分支（:293-294）、res=None 回退贪心（:298-301）只有真机大模型才走到。
  test_allocate 只测 dp/greedy/uniform 函数本体。monkeypatch `dp_allocate` 抛 MemoryError
  即可离线覆盖。

**R10. pyproject 依赖与 extras 三处失真；configs/ 是无消费者的死配置**
- `pyproject.toml:17-19`：`pyyaml`、`tqdm` 在 liteforge/ scripts/ examples/ tests/ 中零 import
  （grep 验证）；:26 `deploy = ["requests"]` 而 `deploy/vllm_bench.py:14` 明说"仅依赖标准库
  （urllib）"且确实零 requests import；README:133 也写"仅标准库"——三处自相矛盾。
- `configs/smoke_qwen05b.yaml`、`configs/gptq_deploy_qwen15b.yaml`：仓库内无任何代码加载
  YAML（grep "configs/" 零引用，无 yaml import），文件头自称"供程序化调用参考"但消费方
  不存在；且 `gptq_deploy_qwen15b.yaml` 给 `quant-gptq` 写了 `save:` 步骤——quant-gptq
  parser（cli.py:488-500）**没有 `--save` 参数**，照抄必然失败。
- 修复：删 pyyaml/tqdm/requests；configs/ 要么补一个 config loader，要么改名为
  `docs/examples/` 下的说明文档并修正不存在的参数。

**R11. Dockerfile 未装 `datasets`，:19 注释里的示例命令跑不通**
- `docker run --rm liteforge python examples/01_quickstart_ppl.py --model <model>` 需要
  `datasets` 加载 wikitext2（data/text.py:25），而镜像只装了核心依赖（datasets 在 `eval`
  extra，pyproject.toml:24）。→ ImportError。examples/02-06 同理（全部用 load_eval_text）。
- 修复：`pip install -e ".[dev,eval]"`；或 README/Dockerfile 注明镜像内只能跑 pytest。

**R12. `export.py`（export_compressed）是死模块：无 CLI、无调用方、无测试，但 CHANGELOG 声称已交付**
- grep 全仓库零调用（cli.py 无 export 子命令；examples/scripts/tests 均未引用）。
  CHANGELOG:19 把"export（压缩模型 HF 导出 + 压缩清单）"列为 0.3.0 新增——名不副实。
- 修复：加 `export` CLI 子命令（与 prune 的 `--save` 打通），或从 CHANGELOG 移除/标注"库函数（未接线）"。

**R13. `.gitignore` 漏了根目录日志：`*.log`**
- 工作区已有 `budget15b.log`、`budget_resume.log`、`deep_batch.log`、`v21_batch.log` 四个
  根目录日志，.gitignore 只忽略 `runs/`、`outputs/`（:12-13）。这些挂机批日志含本机路径
  （F:/MyProgram/...）与几万行输出，不宜入库。修复：加 `*.log`。

**R14. spec-bench / eval-ppl 记录的 params 缺关键口径字段，fp32 声称不可追溯**
- `cmd_spec_bench`（cli.py:402-404）params 只写 draft/k/max_new/prompt_words，不记
  device/dtype——README:24 与 TECHNICAL_REPORT:80-83 的核心结论"bf16 位级差异翻转、fp32
  恢复 exact"无法从 `results/spec_qwen15b_k4_fp32.json` 验证（该记录 params 里无 dtype）。
- `cmd_eval_ppl`（cli.py:80）params 硬编码 `{}`：dataset/seqlen/batch_size/max_blocks 全部
  不入记录。实际后果已显现：`20260903_011442_dense.json`（13.21，32 块 65K tokens）与
  `20260906_051031_dense.json`（13.0282，145 块 297K tokens）是两种口径，schema 上却无法区分。
- 修复：把 dtype/device/max_blocks 等并入 params。

**R15. `report/collect.py:42-49` 透视键缺 `obc_mask`，静态/动态 OBC 记录互相覆盖；results/table.md 已过期**
- to_markdown 的 key 只含 (model, method, sparsity, structure, bits, group_size)。0.5B 的
  obc 静态（54.16）与动态（17.52）两条 params 几乎相同（后者仅多 obc_mask="dynamic"），
  "相同设置取最新一条"会让静态记录在表里消失。README:67-68 是手工维护才同时列出两者。
- 另外 `results/table.md`（25 行）生成于预算批之前：无 12 条 apply/mixed 记录、dense 仍为
  13.2064（会取到 20260906 的 13.0282）——重新生成即漂移。修复：key 加入 obc_mask 与
  task 相关字段，并重生成 table.md。

**R16. CLI 子命令参数/默认值互相不一致（清单）**
- `spec-bench`（cli.py:566-574）：自建 `--device/--dtype`（无 choices、无 auto 语义说明），
  缺 `--seed`（main() 靠 hasattr 兜底，:589）与 `--trust-remote`，与 add_model_args 惯例相悖。
- `speed --seqlen` 默认 1024（:514）vs eval 系命令默认 2048（:57）。
- `smooth-alpha` 默认 batch-size 8 / calib-size 8（:559-560）vs eval 系 4 / calib-size 16——
  若是有意为之应在 help 里注明。
- `quant-awq`（:502-510）解析 `--device/--dtype/--trust-remote` 但 `cmd_quant_awq`（:208-217）
  把模型加载完全交给 AutoAWQ（wrappers.py:39），三者均为静默 no-op。
- `quant-gptq --impl lib` 分支（:194-205）接受 `--restore` 但不执行（restore 仅 ：166/:187
  两个 scratch/rtn 分支消费）。

### P3（文档/风格，并入第三节逐条给改法；此处仅列代码侧两条）

**R17. `cli.py:247` `import numpy as np`（cmd_loss_report 内）函数级死导入**
- cmd_loss_report 内无 np 使用。`scripts/dead_imports.py:22` 的 `src.count(i) <= 1` 启发式
  被 cmd_allocate 的 `np.prod`（:280）稀释，lint 抓不到；且该扫描器只覆盖 liteforge/，
  examples/04 的 R2 同类问题在盲区。建议扫描器按"函数作用域"分析并纳入 examples/。

**R18. `report/card.py:90` `__import__("json")` 内联导入 hack**
- 文件顶部无 json 导入却用 `__import__("json").dumps(...)`。功能等价但属反模式（也是本次
  安全视角唯一命中点，非风险）。修复：顶部 `import json`。

---

### 视角一补充：eval/quant/prune 命令"参数组合 × 测试"矩阵缺口

图例：✅ 有测试（单测或夹具级）；⚠️ 仅真机跑过、无回归测试；❌ 完全无测试。
核心算法层（RTN/GPTQ/OBC/Wanda/allocate/lossmeter/speculative/mmlu 评分）覆盖良好，
缺口集中在 **CLI 接线层、模型加载层、smooth-alpha 核心与记录 schema 层**。

| 命令 | 已测路径 | 从未被测的参数组合/路径 |
|---|---|---|
| eval-ppl | compute_perplexity 数学（test_eval_report） | `--dtype` 任意非 auto（❌，R1 P0）；`--device cuda/cpu`；`--dataset file:`（text.py:41-42）；`--max-blocks 0` 退化为不限量（text.py:70）；`--speed` 组合（benchmark_generate 零测试，speed.py:45-67）；`--trust-remote` |
| prune | wanda/magnitude/obc static+dynamic、2:4（test_prune/test_obc） | `--sparsity-map` dict 全链路（❌，layer_sparsity_target dict 分支 base.py:74-78 仅被 run_sgmix.py 真机用）；`--dry-score` CLI 接线（cli.py:109-118）；`--save`（:142-145）；`--include`（❌ no-op，R3）；`--method obc --structure 2:4` 静默忽略（前审 F14）；`--sparsity-map`+`--method magnitude` 冗余校准 |
| quant-rtn | quantize_tensor/restore/pad 分支/对称网格（test_quant） | CLI `--restore --eval` 组合；`--group-size 0` CLI 路径（真机跑过，仅⚠️） |
| quant-gptq | scratch 量化+restore（test_gptq） | `--impl lib` 全路径（❌，含 wrappers.py:16-24 ImportError 指引文案）；lib 分支 `--restore` no-op（R16）；`--calib-chars`（仅 lib 消费） |
| quant-awq | 无（❌） | 全部；至少应测 ImportError 指引（wrappers.py:33-36）与参数 no-op（R16） |
| speed | 无（❌） | benchmark_forward/generate（speed.py:20-67，需 fake tokenizer）；`vocab` 死变量（前审 F17） |
| report | collect_results/to_markdown/effective_bits（test_eval_report） | `--plot`（plot_ppl_tradeoff，❌）；`write_csv`（❌）；`--out` 非 .md 后缀时 ：238 的 `.replace(".md",".csv")` 静默不产 CSV；R15 键覆盖问题 |
| loss-report | measure_menus 全量/chunk/static（test_lossmeter） | `--quant-impl gptq`（❌）；`--prune-mode dynamic` 在 measure_menus 入口的组合（prune_copy dynamic 经 apply_allocation 间接✅） |
| allocate | DP=暴力枚举/贪心/uniform/不可行（test_allocate，质量高） | CLI MemoryError 降级链（R9）；`--plot`（plot_allocation_map ❌）；`--strategy uniform:缺选项` 的 KeyError 文案 |
| apply-alloc | apply_allocation 端到端（test_allocate:108，历史教训项✅） | CLI 层 record schema（R6 的 strategy=None 链）；`--chunk/--percdamp` 透传 |
| smooth-alpha | 仅数学算子（test_smooth） | `w8a8_alpha_sweep` 核心（R7）；CLI 聚合（best_alpha/improvement_ratio）；`--alphas` 空表崩溃（R7 附） |
| spec-bench | 精确性定理 k∈{1,2,8}×3 种子（test_speculative，质量高） | `--dtype fp32`（❌ R1 P0，D18 复现路径断裂）；params 缺 dtype（R14） |
| eval-mmlu | 评分/分桶/prompt（test_mmlu，mock 完整） | 多 token 回退路径（mmlu.py:92-104 ❌）；`load_mmlu` 本地 parquet/在线与 n>len（❌）；`--dtype bf16` logit 精度（前审 F7） |
| report-card | 无（❌） | 全部（R8；R6 已在真机产物中显形） |
| deploy/vllm_bench | 无（❌） | build_prompt/run_bench worker 循环（可用本地 http.server 线程测）；--out schema |
| （库）export.py | 无（❌） | 全部（R12 死模块） |

---

## 二、测试补齐优先级清单（按"真机才暴露"风险排序，前 10 条）

> 原则：先堵"CI 绿但真机炸"的（R1 即现实案例），再堵"结论/产物静默污染"的。
> 全部可用现有 tiny_model/tiny_tokenizer 夹具离线实现，预计合计 <150 行。

1. **dtype 加载契约**（堵 R1/P0）：monkeypatch `AutoModelForCausalLM.from_pretrained` 捕获
   kwargs，断言 `load_model_and_tokenizer(dtype in {fp32,fp16,bf16})` 正常执行且按
   transformers 主版本传出正确 kwarg；顺带断言 fp32 真加载 tiny model 可前向。这是 D18
   fp32 结论可复现性的唯一守卫。
2. **CLI parser↔handler 冒烟矩阵**：对全部 14 个子命令用 monkeypatch 后的
   load_model_and_tokenizer 走 `main(argv)`，断言不抛 SystemExit/"unrecognized arguments"，
   并对每个声明参数断言被消费（一次性堵 R3 `--include`、R5、R16 全部 no-op 与死参数）。
3. **w8a8_alpha_sweep 端到端**（堵 R7）：tiny_model + 离群注入夹具（复用
   test_smooth.py:49-56 的构造），断言 best_alpha 收敛、no_smooth > best、模型权重不被修改、
   act_max 为 None 时报错清晰；再加 `--alphas` 空表的显式报错。
4. **V2 CLI 三连端到端**：tiny 模型跑 loss-report → allocate(dp/greedy/uniform) →
   apply-alloc，断言记录 schema（task/model/method/params/metrics）、`strategy` 字段非空、
   predicted_total_loss 与 achieved_bits 存在（堵 R6 的传播链）。
5. **build_report_card golden 测试**（堵 R8/R6）：tmp_path 写入合成记录，断言输出含各章节
   且 **"None" 不出现在文本中**、strategy 行显示真实策略名。
6. **allocate 降级链**（堵 R9）：monkeypatch dp_allocate 抛 MemoryError，断言 CLI 回退
   greedy、日志含"自动放宽/回退"字样、结果记录 strategy=greedy。
7. **prune --sparsity-map**：构造 {layer: s, "__default__": 0.5} 走 WandaPruner 与
   OBCPruner，断言逐层稀疏度命中 dict 目标（覆盖 base.py:74-78 分支与 run_sgmix.py 真机路径）。
8. **cmd_prune --dry-score / --save 接线**：score_dry JSON 落盘可回读、save_pretrained+
   tokenizer 落盘后可重新加载并前向（这两条 CLI 路径目前完全无守卫）。
9. **eval-mmlu 多 token 回退 + load_mmlu**：mock tokenizer 返回两 token 选项走 mmlu.py:92-104；
   本地 parquet fixture（tmp parquet）验证 load_mmlu 的缓存分支与 n 裁剪。
10. **记录 schema 回归**：固定一组"命令 → 期望 params/metrics 键集合"的断言表（eval-ppl
    的 dataset/seqlen/max_blocks、spec-bench 的 dtype/device），防止 R14 类口径信息再次丢失。

（备选 11-13：export_compressed 往返测试或接线；write_csv + report --plot Agg 冒烟；
vllm_bench 用本地线程 http server 测 run_bench 的统计字段。）

---

## 三、文档修正清单（逐条改法）

### README.md
1. :44 删除 `--eval`（apply-alloc 无此参数），或按 R5 第二方案给 CLI 加开关后保留。
2. :123 "12 passed" → "55 passed"（实测 55 项 23 秒；12 是 v0.1 时代的数字）。
3. :135 功能矩阵 "12 个离线单测" → "55 个离线单测"。
4. :152-158 Roadmap："MMLU 子集评测（lm-eval 接入）"与 ：26 已报告的 MMLU-mini 结果矛盾——
   改为 "[x] MMLU-mini 似然评测（自研）；[ ] lm-eval 全量接入"，其余未勾项核对后保留。
5. :12-13 与 ：51-52 完全重复的一段话，删除后者。
6. :57 "16 组实验" 与下方 25 行表格、results/ 43 条记录不符；与 TECHNICAL_REPORT:53 "40 余组"
   统一口径（建议："首轮 16 组 + 预算/V2.1 批共 40+ 组记录"）。
7. :56 口径注明 dense=13.21 是 32 块/65K tokens 冒烟口径（CROSSCHECK.md:41 的全 test 集
   口径为 13.0282），避免与 report 重生成后的 13.03 打架。

### docs/ARCHITECTURE.md
8. :40、:55、:84、:101 GSM8K 四处均为"已实现 runner"口吻，代码零实现（grep 验证）——
   全部改为"规划中（rlforge 移植，未开始）"或删除。
9. :50 "33+ 离线单测"、:103 "33 测试" → "55 项离线单测"。
10. :21、:24、:98-99 "实测待 GPU" 状态已过期（smooth_qwen*.json、spec_qwen*.json、
    mmlu_qwen*.json 均已落盘）→ 更新为 ✅ 实测（2026-09-05）。
11. :66 "α 网格搜索 {0.4–0.9}" 与 CLI 默认 [0.4..0.8]（cli.py:562）不一致 → 统一为 0.4–0.8。

### docs/TECHNICAL_REPORT.md
12. :16 与 ：109 "53 项离线单测" → "55 项"。
13. 其余数字全部核对通过（ρ=1.000 n=6 实跑 scripts/analyze_prediction.py 复现 ✓；OBC
    54.16/17.52、GPTQ 14.13/9.91/8.34、smooth 3.65×/3.98×、spec 1.96× 均与 results/*.json 一致）。

### docs/DESIGN.md
14. :54 "12 个单元测试" → 55；:11-28 架构图缺 V2 层（lossmeter/allocate/eval-mmlu/spec），
    补一行"详见 ARCHITECTURE.md"或重绘。
15. :62 已知边界 "MMLU 子集评测（预留 lm-eval 接入点）" → 已有自研 eval-mmlu，改述。

### docs/V2_PLAN.md
16. :3 "通过 33 项单测" → 55 项（或去掉具体数字，写"全部离线单测"以免再腐化）。

### CHANGELOG.md
17. :19 "export（压缩模型 HF 导出 + 压缩清单）" → 按实际情况改（"export.py 库函数，
    暂无 CLI/测试"）或先接线再宣称。

### examples/README.md
18. :17 "所有示例输出统一 JSON 到 results/" → 只有 01/04 落盘；02/03/05/06 仅打印。
    改述，或给 02/03/05 补 save_json。

### scripts/CROSSCHECK.md
19. :37 "结果统一记录到 results/crosscheck_*.json 并进报告卡（核对区）"——仓库无该产物、
    报告卡无核对区 → 改为"计划"，或给 report-card 增加该节。

### configs/
20. 两个 YAML 无任何加载器（R10）；gptq_deploy_qwen15b.yaml 的 `save:` 步骤对应不存在的
    `quant-gptq --save` → 若保留为"人工操作剧本"，文件头注明"非机器可读配置，quant-gptq
    无 --save，导出请改用 prune --save / export 模块"。

### 打包卫生（pyproject.toml / .gitignore / Dockerfile / Makefile）
21. pyproject.toml:17-19 删 `pyyaml`、`tqdm`；:26 删 `deploy = ["requests"]`（或改
    `deploy = []` 并注明纯标准库）。
22. .gitignore 增加 `*.log`（R13）。
23. Dockerfile:12 改 `pip install -e ".[dev,eval]"`（R4+R11 一并解决），:19 注释同步。
24. Makefile `card` target 正常，但产物含 R6 的 "None @" 行——修复 R6 后重生成
    results/card_*.md 与 results/table.md（table.md 现为预算批前的过期版本，R15）。

---

## 四、正面证据（本次核对确认无误项）

1. README 结果表 25 行数字与 results/*.json 全部吻合（含 OBC 静/动、三模型 GPTQ、smooth/spec/mmlu 摘要）。
2. `scripts/analyze_prediction.py` 实跑复现 ρ=1.000（0.5B n=6 / 1.5B n=6，12 个点），TECHNICAL_REPORT 4.3 成立。
3. 55 项离线单测 23.14s 全绿；核心数学层（trace 恒等式、DP=暴力、GPTQ 对称网格、2:4、投机精确性）测试质量高。
4. 安全面干净：无 eval/exec/pickle/yaml.load/torch.load/subprocess；导出走
   `save_pretrained(safe_serialization=True)`（export.py:23）；`trust_remote_code` 默认
   False 且需显式 `--trust-remote`（models.py:22）；vllm_bench 仅 urllib 无 SSRF 面扩大。
5. scripts/run_* 系列（smoke/deep_batch/budget/v21/resume）命令参数与 cli.py parser 逐条
   核对全部合法（唯一例外是文档侧 README/examples/04 的 `--eval`，见 R5）。

## 五、统计

- 新发现合计 18 项：P0×1（已复现）、P1×4（R2/R5 已复现）、P2×12、P3 两条代码侧 + 文档清单 24 条。
- 最危险组合：R1（--dtype 全线 NameError）× 零测试（models.py 从未被任何测试 import 执行）——
  与历史上 apply_allocation 漏 import 同类，正是"每个 CLI 可达路径都要有测试"教训的再次显形。
- 文档数字三处各说各话：12（README/DESIGN）/33（ARCHITECTURE/V2_PLAN）/53（TECH_REPORT），
  实际 55——建议统一为"55（随 CI 增长）"或干脆不写死。
