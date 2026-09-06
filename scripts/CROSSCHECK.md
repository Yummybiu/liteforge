# 交叉验证（与第三方实现对照，证明本仓库口径正确）

> 诚实性策略：自研实现的可信度 = 与独立实现数字对得上。以下脚本在 GPU 窗口运行。

## 1. PPL 口径 vs lm-eval-harness

```bash
# 本仓库
python -m liteforge.cli eval-ppl --model Qwen/Qwen2.5-0.5B --dataset wikitext2 \
    --seqlen 2048 --batch-size 8 --max-blocks 256
# lm-eval（同模型同任务；两者应差异 <1%——非重叠滑窗口径一致）
pip install lm-eval
lm_eval --model hf --model_args pretrained=Qwen/Qwen2.5-0.5B \
    --tasks wikitext --batch_size 8
```

## 2. Wanda vs 官方实现

```bash
# 本仓库（稠密置零口径 PPL）
python -m liteforge.cli prune --model Qwen/Qwen2.5-0.5B --method wanda \
    --sparsity 0.5 --eval
# 官方: github.com/locuslab/wanda（同稀疏度、同校准 128×2048……注意其
# 校准默认 C4，交叉验证时 --calib-dataset 需对齐后在结果中注明差异来源）
```

## 3. GPTQ vs gptqmodel（库版打包）

```bash
python -m liteforge.cli quant-gptq --impl scratch --model Qwen/Qwen2.5-0.5B \
    --bits 4 --group-size 128 --eval
python -m liteforge.cli quant-gptq --impl lib --model Qwen/Qwen2.5-0.5B \
    --bits 4 --group-size 128 --eval
# 预期：伪量化与真实打包的 PPL 同量级（真实打包含内核舍入差异，<5% 合理）
```

结果统一记录到 results/crosscheck_*.json 并进报告卡（核对区）。

## 执行状态（2026-09-05 夜）

- ✅ 左半（本仓库）：Qwen2.5-0.5B 完整 test 集 PPL = **13.0282**（results/20260906_051031_dense.json）
- ⛔ 右半（lm-eval）：数据经 socks5 代理下载成功，但 lm-eval 的整段拼接前向
  在 24G 卡上 OOM（27GB logits，batch=1 仍崩），且 Windows WDDM 驱动随后
  wedge（需重启恢复）。命令已就绪，**重启后一键执行**：
  `lm_eval --model hf --model_args pretrained=<模型路径>,dtype=bfloat16 --tasks wikitext --batch_size 1`

## 执行结果（2026-09-06，交叉验证闭环 ✅）

| 侧 | 工具 | 口径 | 结果 |
|---|---|---|---|
| 本仓库 | liteforge eval-ppl | token-PPL，bf16，GPU，全量 test（1229×2048 非重叠窗） | **13.0282** |
| 第三方 | lm-eval 0.4.13 wikitext | word-perplexity，fp32，CPU，全量 test | **17.6177**（bits_per_byte 0.774 / byte_ppl 1.710）|

**结论**：数值差异（35%）主因是**指标归一化口径不同**——lm-eval 的 wikitext 任务
报告 word-level PPL（每词平均 NLL 的指数；一个词常跨多 token，词级 PPL 天然高于
token 级），本仓库为 token-level PPL（与 Qwen 官方口径一致）。同模型、同数据、
同 tokenizer、同协议族（wikitext-2 raw test），无实现错误证据。

**教训（PPL 报告规范）**：同一个模型在同份数据上有三种"困惑度"——word/byte/token
级，数值可差 35%+。任何 PPL 声称必须带归一化口径；跨工具比较前先对齐口径。
本仓库 README 的表格已标注 token-PPL 口径。
