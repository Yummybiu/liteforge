# V2 预算分配批（0.5B+1.5B 今晚版；3B 损失表待 p50-only 快口径后单跑）
# 用法: bash scripts/run_budget_15b.sh   （GPU 挂机，预计 ~3.5h）
set -e
PY=${PY:-/f/Anaconda/envs/liteforge/python.exe}
cd "$(dirname "$0")/.."

for M in 0.5B 1.5B; do
  MODEL=F:/MyProgram/BIG/liteforge/cache/models/Qwen2.5-$M
  BS=8
  echo "=== ${M} 1/3 损失表 ==="
  $PY -m liteforge.cli loss-report --model $MODEL --calib-dataset wikitext2:train \
      --calib-size 16 --seqlen 2048 --batch-size $BS --chunk 6 \
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

    echo "=== ${M} 3/3 应用+验证 target=${T}（dp vs uniform-w4）==="
    $PY -m liteforge.cli apply-alloc --model $MODEL --alloc results/alloc_qwen${M}_t${T}_dp.json \
        --calib-dataset wikitext2:train --calib-size 16 --seqlen 2048 \
        --batch-size $BS --chunk 6 --max-blocks 32 \
        --out results/apply_qwen${M}_t${T}_dp.json
    $PY -m liteforge.cli apply-alloc --model $MODEL --alloc results/alloc_qwen${M}_t${T}_unifw4.json \
        --calib-dataset wikitext2:train --calib-size 16 --seqlen 2048 \
        --batch-size $BS --chunk 6 --max-blocks 32 \
        --out results/apply_qwen${M}_t${T}_unifw4.json
  done
done

echo "=== 报告卡 ==="
$PY -m liteforge.cli report-card --model-key 0.5B --out results/card_0.5B.md
$PY -m liteforge.cli report-card --model-key 1.5B --out results/card_1.5B.md
echo "BUDGET_15B_DONE"
