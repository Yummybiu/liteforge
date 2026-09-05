# 幂等续跑：只跑缺失的产物（损失表→分配→应用），已存在的跳过
# 用法: bash scripts/run_budget_15b_resume.sh
set -e
PY=${PY:-/f/Anaconda/envs/liteforge/python.exe}
cd "$(dirname "$0")/.."

need() { [ ! -f "$1" ]; }

for M in 0.5B 1.5B; do
  MODEL=F:/MyProgram/BIG/liteforge/cache/models/Qwen2.5-$M
  BS=8; LOSSES=results/losses_qwen${M}.json
  if need $LOSSES; then
    echo "=== ${M} 损失表 ==="
    $PY -m liteforge.cli loss-report --model $MODEL --calib-dataset wikitext2:train \
        --calib-size 16 --seqlen 2048 --batch-size $BS --chunk 6 \
        --quant-impl rtn --prune-mode static --out $LOSSES
  fi

  for T in 2.75 2.5 2.25; do
    for S in dp greedy; do
      A=results/alloc_qwen${M}_t${T}_${S}.json
      if need $A; then
        echo "=== ${M} allocate ${S} t=${T} ==="
        $PY -m liteforge.cli allocate --losses $LOSSES --target-bits $T --strategy $S \
            $( [ "$S" = "dp" ] && echo --plot ) --out $A
      fi
    done
    U=results/alloc_qwen${M}_t${T}_unifw4.json
    if need $U; then
      $PY -m liteforge.cli allocate --losses $LOSSES --target-bits $T \
          --strategy uniform:w4g128 --out $U
    fi
    for S in dp unifw4; do
      P=results/apply_qwen${M}_t${T}_${S}.json
      if need $P; then
        echo "=== ${M} apply ${S} t=${T} ==="
        $PY -m liteforge.cli apply-alloc --model $MODEL --alloc results/alloc_qwen${M}_t${T}_${S}.json \
            --calib-dataset wikitext2:train --calib-size 16 --seqlen 2048 \
            --batch-size $BS --chunk 6 --max-blocks 32 --out $P
      fi
    done
  done
done

$PY -m liteforge.cli report-card --model-key 0.5B --out results/card_0.5B.md
$PY -m liteforge.cli report-card --model-key 1.5B --out results/card_1.5B.md
echo "RESUME_DONE"
