# LiteForge

**LLM 压缩-评测-部署一站式工具箱：剪枝 / 量化 / 困惑度评测 / 部署基准，附带完整真实实验数据。**

它回答一个具体的问题：*把一个开源 LLM 压下去，质量掉多少？部署能快多少？*
—— 不做 RAG demo，不做微调套壳，只做一条从算法到系统的完整 trade-off 证据链。

## 仓库自带的真实实验数据（可一键复现）

Qwen2.5-0.5B / 1.5B / 3B，WikiText-2 困惑度（2048-token 非重叠窗口，32 块 ≈ 65K tokens），
单卡 RTX A5000，bf16，16 组实验全部由本仓库 CLI 产出（记录见 `results/*.json`）：

| 模型 | 方法 | 稀疏度/位宽 | PPL↓ | vs dense |
|---|---|---|---|---|
| Qwen2.5-0.5B | dense (fp16) | - | **13.21** | - |
| | Wanda | 25% | 13.72 | +0.51 |
| | Wanda | 50% | 25.52 | +12.32 |
| | Wanda | 50% (2:4) | 70.14 | +56.93 |
| | Wanda | 60% | 76.71 | +63.50 |
| | Magnitude | 50% | 485.58 | +472.38 |
| | OBC 静态掩码 | 50% | 54.16 | +40.95 |
| | **OBC 动态重评分** | 50% | **17.52** | +4.31 |
| | RTN | W8 g128 | 13.22 | ≈无损 |
| | RTN | W4 g128 | 15.69 | +2.48 |
| | RTN | W4 per-ch | 23.99 | +10.78 |
| | RTN | W3 g128 | 51.37 | +38.17 |
| | GPTQ（自研） | W4 g128 | 14.13 | +0.92 |
| Qwen2.5-1.5B | dense (fp16) | - | **9.26** | - |
| | Wanda | 50% | 13.34 | +4.08 |
| | OBC 静态 | 50% | 13.73 | +4.47 |
| | **OBC 动态** | 50% | **11.67** | +2.41 |
| | RTN | W4 g128 | 10.37 | +1.11 |
| | GPTQ（自研） | W4 g128 | 9.91 | +0.65 |
| Qwen2.5-3B | dense (fp16) | - | **8.01** | - |
| | Wanda | 50% | 10.89 | +2.88 |
| | **OBC 动态** | 50% | **9.81** | +1.80 |
| | RTN | W4 g128 | 9.00 | +0.99 |

### 四个可直接讲的核心发现

1. **激活感知是剪枝的生死线**：同样 50% 稀疏度，Wanda（PPL 25.5）与幅度剪枝
  （PPL 485.6）相差 **19 倍**——不看激活分布的剪枝在小模型上等于摧毁模型。
2. **误差补偿 + 块级动态重评分是剪枝的最优解（三点曲线全胜）**：自研 OBC 的
  演化三部曲——静态预选掩码崩塌（54.16，"互相保护的列簇被集体删除"），
  SparseGPT 忠实版块级动态重评分在 0.5B/1.5B/3B 全部反超 Wanda
  （**17.52 / 11.67 / 9.81** vs 25.52 / 13.34 / 10.89）。高稀疏度下，删谁的决定
  必须跟着补偿动态走。
3. **量化分组粒度决定质量，误差反馈白赚一个点**：同为 W4，g128（15.69）比
  per-channel（23.99）好 8.3 个 PPL；自研 GPTQ 在 RTN 之上再赚 1.55（→14.13）。
4. **模型越大越耐压（三点曲线）**：50% 剪枝的 PPL 代价，0.5B → 1.5B → 3B
  为 +12.3 → +4.1 → +2.9（Wanda）；RTN W4 为 +2.5 → +1.1 → +1.0。

> 口径纪律：剪枝模型的 PPL 是**稠密权重置零**口径（方法学质量损失，同 Wanda
> 论文协议），不声称速度收益；部署加速口径见 `deploy/vllm_bench.py`（GPTQ/AWQ
> 整数内核 + vLLM 实测）。`生成 tok/s` 列为 HF `generate` 未批处理口径，
> 受 Windows WDDM 同步开销影响，不代表部署吞吐。

## 快速开始

```bash
# 环境：Python ≥3.10，torch ≥2.1（GPU 推荐）
pip install -e .

# 模型与评测数据（国内走 hf-mirror）
export HF_ENDPOINT=https://hf-mirror.com
bash scripts/download_models.sh 0.5B

# 一键冒烟：dense → Wanda 50% → Magnitude → RTN W4/W8 → 汇总表 + trade-off 图
bash scripts/run_smoke.sh
```

单测（离线，10 秒）：

```bash
pip install pytest && python -m pytest tests/ -q   # 12 passed
```

## 功能矩阵

| 模块 | 内容 | 依赖 |
|---|---|---|
| `prune/` | Magnitude、**Wanda 从零实现**（激活感知打分 + 2:4 半结构化）；`BasePruner` 可插拔接口 | 核心依赖 |
| `quant/` | **RTN 伪量化从零实现**（对称/非对称/分组，可 restore）；GPTQ/AWQ 薄封装 | 可选 |
| `eval/` | 困惑度（非重叠滑窗协议）、前向/生成吞吐 | 核心依赖 |
| `deploy/` | vLLM OpenAI 兼容端点压测（TTFT / decode 吞吐 / 成功率），仅标准库 | 可选 |
| `report/` | 统一 JSON schema 聚合 → Markdown/CSV 对照表 + 等效比特 trade-off 图 | matplotlib 可选 |
| `tests/` | 12 个离线单测（微型 Llama + mock tokenizer） | pytest |

## 架构

```
cli.py ─┬─ prune/ ──┐                       ┌─ results/*.json（统一 schema）
        ├─ quant/ ──┼─ data/text.py ────────┤
        ├─ eval/ ───┘  (wikitext2/分块/校准)  └─ report/ ─► table.md + tradeoff.png
        └─ deploy/vllm_bench.py（在部署机运行）
```

## 文档

- [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md)：完整实验矩阵（0.5B/1.5B/3B/7B × 全方法）与口径纪律
- [docs/DESIGN.md](docs/DESIGN.md)：架构与设计决策（含诚实清单）
- [docs/decision_log.md](docs/decision_log.md)：每个"为什么这么做"

## Roadmap

- [x] Wanda / Magnitude / RTN / 滑窗 PPL / 冒烟实验（本仓库自带数据）
- [ ] GPTQ / AWQ 真实打包量化 + vLLM 端到端吞吐实测（`quant/wrappers.py` 已就绪）
- [ ] MMLU 子集评测（lm-eval 接入）
- [ ] 剪枝+量化组合实验（GPTQ W4 → +2:4）
- [ ] 3B/7B 扩展与博客《我压了两个 Qwen》

## License

MIT
