"""示例 03：量化 —— RTN 基线 vs 自研 GPTQ，看误差反馈赚回多少。

运行：
    python examples/03_quantization_gptq.py --model Qwen/Qwen2.5-0.5B

你将学到：RTN 的 scale/zero-point 推导级实现、GPTQ 的 Hessian 误差反馈
（本仓库实测 W4g128 上 15.69 → 14.13）、位宽-质量单调性。
"""

import argparse

from liteforge.data.text import BlockBatcher, load_eval_text
from liteforge.eval import compute_perplexity
from liteforge.models import load_model_and_tokenizer
from liteforge.quant import GPTQQuantizer, RTNConfig, RTNQuantizer
from liteforge.utils import seed_everything, setup_logging


def main():
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--calib-size", type=int, default=16)
    args = ap.parse_args()
    seed_everything()

    calib = BlockBatcher(None, load_eval_text("wikitext2:train"),
                         block_size=2048, batch_size=8)  # tokenizer 下面填充

    results = {}
    for name in ("rtn_w4", "gptq_w4"):
        model, tokenizer = load_model_and_tokenizer(args.model)
        calib.tokenizer = tokenizer
        if name == "rtn_w4":
            q = RTNQuantizer(model, RTNConfig(bits=4, group_size=128))
            q.quantize_()
        else:
            q = GPTQQuantizer(model, RTNConfig(bits=4, group_size=128))
            q.quantize_(calib, max_batches=args.calib_size)
        results[name] = compute_perplexity(model, tokenizer, dataset="wikitext2",
                                           max_blocks=32).ppl
        print(f"{name}: PPL = {results[name]:.2f}")

    print(f"\n=== 误差反馈的收益: {results['rtn_w4']:.2f} → {results['gptq_w4']:.2f} "
          f"（GPTQ 相对 RTN）===")


if __name__ == "__main__":
    main()
