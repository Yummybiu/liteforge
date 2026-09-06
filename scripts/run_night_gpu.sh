# GPU 恢复后的一键实验序列（驱动 wedge 恢复后运行）
# 用法: bash scripts/run_night_gpu.sh   （预计 ~2.5h）
set -e
PY=${PY:-/f/Anaconda/envs/liteforge/python.exe}
cd "$(dirname "$0")/.."
M=F:/MyProgram/BIG/liteforge/cache/models/Qwen2.5-0.5B

echo "=== 1/5 GPU 健康探针 ==="
$PY -c "import torch; x=torch.randn(1000,1000,device='cuda'); print('cuda ok', (x@x).sum().item()>0)"

echo "=== 2/5 act-order GPTQ 真机验证（vs 基线 14.13）==="
$PY -m liteforge.cli quant-gptq --impl scratch --model $M --bits 4 --group-size 128 \
    --act-order --calib-size 16 --dataset wikitext2 --seqlen 2048 --batch-size 8 \
    --max-blocks 32 --eval --out results/gptq_scratch_actorder_w4.json

echo "=== 3/5 菜单 v2（GPTQ 口径 + 动态剪枝）→ 分配 → 应用（检验预算下限）==="
$PY -m liteforge.cli loss-report --model $M --calib-dataset wikitext2:train \
    --calib-size 16 --seqlen 2048 --batch-size 8 --chunk 6 \
    --quant-impl gptq --prune-mode dynamic \
    --out results/losses_qwen0.5B_v2.json
$PY -m liteforge.cli allocate --losses results/losses_qwen0.5B_v2.json \
    --target-bits 2.75 --strategy dp --plot --out results/alloc_v2_t275_dp.json
$PY -m liteforge.cli allocate --losses results/losses_qwen0.5B_v2.json \
    --target-bits 2.5 --strategy dp --out results/alloc_v2_t25_dp.json
$PY -m liteforge.cli apply-alloc --model $M --alloc results/alloc_v2_t275_dp.json \
    --calib-dataset wikitext2:train --calib-size 16 --seqlen 2048 --batch-size 8 \
    --chunk 6 --max-blocks 32 --save outputs/qwen05b_v2_t275 \
    --out results/apply_v2_t275_dp.json
$PY -m liteforge.cli apply-alloc --model $M --alloc results/alloc_v2_t25_dp.json \
    --calib-dataset wikitext2:train --calib-size 16 --seqlen 2048 --batch-size 8 \
    --chunk 6 --max-blocks 32 --save outputs/qwen05b_v2_t25 \
    --out results/apply_v2_t25_dp.json

echo "=== 4/5 压缩模型 MMLU 下游（能力损失地图）==="
$PY -m liteforge.cli eval-mmlu --model outputs/qwen05b_v2_t275 --n 500 \
    --out results/mmlu_qwen05b_v2_t275.json
$PY -m liteforge.cli eval-mmlu --model outputs/qwen05b_v2_t25 --n 500 \
    --out results/mmlu_qwen05b_v2_t25.json

echo "=== 5/5 报告卡与预测力分析 ==="
$PY scripts/analyze_prediction.py > results/prediction_analysis.txt 2>&1 || true
$PY -m liteforge.cli report-card --model-key 0.5B --out results/card_0.5B.md
echo "NIGHT_GPU_DONE"
