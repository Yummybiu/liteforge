# Examples

按序号即学即用（模型路径可换成 cache/ 下的本地路径或任意 HF CausalLM）：

| # | 脚本 | 内容 | 算力 |
|---|---|---|---|
| 01 | [quickstart_ppl.py](01_quickstart_ppl.py) | 困惑度口径与结果记录 | CPU 可跑（慢） |
| 02 | [pruning_wanda.py](02_pruning_wanda.py) | Wanda vs Magnitude，激活感知的价值 | 单卡分钟级 |
| 03 | [quantization_gptq.py](03_quantization_gptq.py) | RTN → GPTQ，误差反馈的收益 | 单卡分钟级 |
| 04 | [budget_allocation.py](04_budget_allocation.py) | V2：统一损失 → 精确 DP 预算分配 | 单卡（loss 步 ~1h/0.5B） |
| 05 | [smoothquant_alpha.py](05_smoothquant_alpha.py) | W8A8 激活量化，α 跷跷板搜索 | 单卡 |
| 06 | [speculative_decoding.py](06_speculative_decoding.py) | 投机解码：零质量损失加速 | 单卡 |

前置：`pip install -e .`；模型下载 `bash scripts/download_models.sh 0.5B`；
国内环境 `export HF_ENDPOINT=https://hf-mirror.com`。

所有示例输出统一 JSON 到 `results/`，可用 `python -m liteforge.cli report` 聚合，
或 `python -m liteforge.cli report-card --model-key 0.5B` 生成单文件报告卡。
