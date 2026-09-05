"""压缩模型导出：HF 格式保存 + 压缩清单（manifest）。

诚实口径：
- 剪枝/伪量化模型 = 权重置零/伪量化的 fp16 权重，可直接 HF 加载与继续评测；
- 真实 int4 打包（vLLM/GPTQModel 可加载的 compressed-tensors/GPTQ 格式）
  走 quant/wrappers.py 的库版路线（quantize_awq 已含 save_quantized）；
  本模块导出时在 manifest 中明确标注 pseudo=True，避免口径混淆。
"""

import json
import os
import time

from .utils import find_linears


def export_compressed(model, tokenizer, out_dir: str, compressions: list) -> str:
    """保存模型与 tokenizer，并写 manifest.json（压缩清单）。

    compressions: [{"type": "prune", "method": "wanda", "sparsity": 0.5, ...}, ...]
    """
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)

    linears = find_linears(model, exclude=("lm_head", "embed_out"))
    sparsity = sum((m.weight.data == 0).float().sum().item() for _, m in linears)
    total = sum(m.weight.data.numel() for _, m in linears)
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2

    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pseudo": True,   # True = 权重仍以浮点存储（方法学口径），非整数打包
        "compressions": compressions,
        "linear_sparsity": round(sparsity / max(total, 1), 4),
        "size_mb_fp16": round(size_mb, 1),
        "note": ("剪枝/伪量化权重以浮点存储；部署用整数打包请走 "
                 "liteforge.quant.wrappers（gptqmodel/autoawq）→ vLLM"),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return out_dir
