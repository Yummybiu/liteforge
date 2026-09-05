# V2.1 实验批（GPU，等空闲窗口挂机——会占满 GPU，勿在使用电脑时运行！）
# 用法: bash scripts/run_v21_batch.sh
# 内容: SmoothQuant α 扫描 + 投机解码基准 + MMLU-mini 下游（0.5B/1.5B）
# 预计: ~2.5h（0.5B ~40min, 1.5B ~1.5h）
set -e
PY=${PY:-/f/Anaconda/envs/liteforge/python.exe}
cd "$(dirname "$0")/.."

for M in 0.5B 1.5B; do
  MODEL=F:/MyProgram/BIG/liteforge/cache/models/Qwen2.5-$M
  BS=8; [ "$M" = "1.5B" ] && BS=6
  echo "=== ${M} SmoothQuant W8A8 α 扫描 ==="
  $PY -m liteforge.cli smooth-alpha --model $MODEL \
      --calib-dataset wikitext2:train --calib-size 8 \
      --seqlen 2048 --batch-size $BS --out results/smooth_qwen${M}.json

  echo "=== ${M} MMLU-mini（dense 基线）==="
  $PY -m liteforge.cli eval-mmlu --model $MODEL --n 500 \
      --out results/mmlu_qwen${M}_dense.json
done

echo "=== 投机解码：target=1.5B draft=0.5B ==="
$PY -m liteforge.cli spec-bench \
    --model F:/MyProgram/BIG/liteforge/cache/models/Qwen2.5-1.5B \
    --draft F:/MyProgram/BIG/liteforge/cache/models/Qwen2.5-0.5B \
    --k 4 --max-new 256 \
    --out results/spec_qwen15b_k4.json
$PY -m liteforge.cli spec-bench \
    --model F:/MyProgram/BIG/liteforge/cache/models/Qwen2.5-1.5B \
    --draft F:/MyProgram/BIG/liteforge/cache/models/Qwen2.5-0.5B \
    --k 8 --max-new 256 \
    --out results/spec_qwen15b_k8.json

echo "=== 报告卡 ==="
$PY -m liteforge.cli report-card --model-key 0.5B --out results/card_0.5B.md
$PY -m liteforge.cli report-card --model-key 1.5B --out results/card_1.5B.md
echo "DONE. 验收：smooth 的 improvement_ratio>1；spec 的 exact=true 且 speedup>1；"
echo "      mmlu acc 与 dense 基线合理；card_*.md 为面试展示文件。"
