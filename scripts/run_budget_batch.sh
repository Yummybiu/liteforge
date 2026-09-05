# V2 主实验批（GPU，等空闲窗口挂机运行——会占满 GPU，勿在使用电脑时运行！）
# 用法: bash scripts/run_budget_batch.sh
# 预计: 0.5B ~1h / 1.5B ~2.5h / 3B ~6h（loss 表为大头，rtn+static 快口径）
# 产出: 三模型损失表、三目标×三策略分配、应用后 PPL 验证、分配地图
set -e
PY=${PY:-/f/Anaconda/envs/liteforge/python.exe}
cd "$(dirname "$0")/.."

for M in 0.5B 1.5B 3B; do
  MODEL=F:/MyProgram/BIG/liteforge/cache/models/Qwen2.5-$M
  BS=8; [ "$M" = "3B" ] && BS=4
  CHUNK=6

  echo "=== ${M} 1/3 损失表 ==="
  $PY -m liteforge.cli loss-report --model $MODEL --calib-dataset wikitext2:train \
      --calib-size 16 --seqlen 2048 --batch-size $BS --chunk $CHUNK \
      --quant-impl rtn --prune-mode static \
      --out results/losses_qwen${M}.json

  for T in 2.75 2.5 2.25; do
    echo "=== ${M} 2/3 分配 target=${T} ==="
    $PY -m liteforge.cli allocate --losses results/losses_qwen${M}.json \
        --target-bits $T --strategy dp --plot \
        --out results/alloc_qwen${M}_t${T}_dp.json
    $PY -m liteforge.cli allocate --losses results/losses_qwen${M}.json \
        --target-bits $T --strategy greedy \
        --out results/alloc_qwen${M}_t${T}_greedy.json
    $PY -m liteforge.cli allocate --losses results/losses_qwen${M}.json \
        --target-bits $T --strategy uniform:w4g128 \
        --out results/alloc_qwen${M}_t${T}_unifw4.json

    echo "=== ${M} 3/3 应用+验证 target=${T} ==="
    $PY -m liteforge.cli apply-alloc --model $MODEL --alloc results/alloc_qwen${M}_t${T}_dp.json \
        --calib-dataset wikitext2:train --calib-size 16 --seqlen 2048 \
        --batch-size $BS --chunk $CHUNK --max-blocks 32 \
        --out results/apply_qwen${M}_t${T}_dp.json
    $PY -m liteforge.cli apply-alloc --model $MODEL --alloc results/alloc_qwen${M}_t${T}_unifw4.json \
        --calib-dataset wikitext2:train --calib-size 16 --seqlen 2048 \
        --batch-size $BS --chunk $CHUNK --max-blocks 32 \
        --out results/apply_qwen${M}_t${T}_unifw4.json
  done
done

echo "=== 汇总 ==="
$PY -m liteforge.cli report --results-dir results --plot
echo "DONE. 结果见 results/table.md；对照 docs/V2_PLAN.md 的验证清单"
