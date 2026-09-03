# 补全实验批（自愿挂机运行——会占满 GPU，勿在使用电脑时运行！）
# 用法: bash scripts/run_deep_batch.sh
# 预计时长 ~60 分钟（A5000）；产出：动态 OBC 三模型 + 3B GPTQ + 最终汇总表
set -e
PY=${PY:-/f/Anaconda/envs/liteforge/python.exe}
cd "$(dirname "$0")/.."

for M in 0.5B 1.5B 3B; do
  MODEL=F:/MyProgram/BIG/liteforge/cache/models/Qwen2.5-$M
  BS=8; [ "$M" = "3B" ] && BS=4
  echo "=== ${M} OBC-dynamic 50% (SparseGPT 忠实版) ==="
  $PY -m liteforge.cli prune --model $MODEL --method obc --obc-mask dynamic \
      --sparsity 0.5 --calib-dataset wikitext2:train --calib-size 16 \
      --dataset wikitext2 --seqlen 2048 --batch-size $BS --max-blocks 32 --eval
done

echo "=== 3B 自研 GPTQ W4g128 ==="
$PY -m liteforge.cli quant-gptq --impl scratch \
    --model F:/MyProgram/BIG/liteforge/cache/models/Qwen2.5-3B \
    --bits 4 --group-size 128 --calib-size 16 \
    --dataset wikitext2 --seqlen 2048 --batch-size 4 --max-blocks 32 --eval

echo "=== 汇总 ==="
$PY -m liteforge.cli report --results-dir results --plot
echo "DONE. 结果见 results/table.md"
