# LiteForge 冒烟实验：Qwen2.5-0.5B 本机 A5000（或 CPU）快速验证全链路
# 产出：results/ 下的 JSON 记录 + results/table.md 对照表
set -e
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=${HF_HOME:-F:/MyProgram/BIG/liteforge/cache/hf}
PY=${PY:-/f/Anaconda/envs/liteforge/python.exe}
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B}
BLOCKS=${BLOCKS:-32}        # 评测块数：32×2048≈65K tokens，冒烟足够
cd "$(dirname "$0")/.."

echo "=== 1/5 dense 基线 PPL ==="
$PY -m liteforge.cli eval-ppl --model $MODEL --dataset wikitext2 \
    --seqlen 2048 --batch-size 8 --max-blocks $BLOCKS --speed

echo "=== 2/5 Wanda 50% 非结构化剪枝 + PPL ==="
$PY -m liteforge.cli prune --model $MODEL --method wanda --sparsity 0.5 \
    --calib-dataset wikitext2:train --calib-size 16 \
    --dataset wikitext2 --seqlen 2048 --batch-size 8 --max-blocks $BLOCKS --eval

echo "=== 3/5 Magnitude 50%（对照组）==="
$PY -m liteforge.cli prune --model $MODEL --method magnitude --sparsity 0.5 \
    --dataset wikitext2 --seqlen 2048 --batch-size 8 --max-blocks $BLOCKS --eval

echo "=== 4/5 RTN W4 g128（非对称组量化）+ PPL ==="
$PY -m liteforge.cli quant-rtn --model $MODEL --bits 4 --group-size 128 \
    --dataset wikitext2 --seqlen 2048 --batch-size 8 --max-blocks $BLOCKS --eval

echo "=== 4b/5 RTN W8（对照组）==="
$PY -m liteforge.cli quant-rtn --model $MODEL --bits 8 --group-size 128 \
    --dataset wikitext2 --seqlen 2048 --batch-size 8 --max-blocks $BLOCKS --eval

echo "=== 5/5 汇总对照表 ==="
$PY -m liteforge.cli report --results-dir results --plot
