# 模型下载（走 hf-mirror，国内直连）
# 用法: bash scripts/download_models.sh [0.5B|1.5B|3B|all]
set -e
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=${HF_HOME:-F:/MyProgram/BIG/liteforge/cache/hf}
PY=${PY:-/f/Anaconda/envs/liteforge/python.exe}
TARGET=${1:-0.5B}

download() {
  echo "== downloading $1 =="
  $PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('$1'))"
}

case $TARGET in
  0.5B) download "Qwen/Qwen2.5-0.5B" ;;
  1.5B) download "Qwen/Qwen2.5-1.5B" ;;
  3B)   download "Qwen/Qwen2.5-3B" ;;
  7B)   download "Qwen/Qwen2.5-7B" ;;
  all)  download "Qwen/Qwen2.5-0.5B"; download "Qwen/Qwen2.5-1.5B" ;;
  *) echo "unknown target: $TARGET"; exit 1 ;;
esac
