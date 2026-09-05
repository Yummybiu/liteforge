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
