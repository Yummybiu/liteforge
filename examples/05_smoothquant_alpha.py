"""示例 05：SmoothQuant W8A8 —— α 跷跷板网格搜索。

运行（GPU 推荐）：
    python examples/05_smoothquant_alpha.py --model Qwen/Qwen2.5-0.5B

你将学到：激活离群为什么让 W8A8 崩坏、等价变换如何把难度迁移进权重、
α 作为"激活/权重难度跷跷板"的搜索口径（蒙特卡洛损失，非 trace 闭式）。
"""

import argparse

from liteforge.data.text import BlockBatcher, load_eval_text
from liteforge.models import load_model_and_tokenizer
from liteforge.smooth import w8a8_alpha_sweep
from liteforge.utils import find_linears, seed_everything, setup_logging


def main():
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--alphas", type=float, nargs="*",
                    default=[0.4, 0.5, 0.6, 0.7, 0.8])
    ap.add_argument("--calib-size", type=int, default=8)
    args = ap.parse_args()
    seed_everything()

    model, tokenizer = load_model_and_tokenizer(args.model)
    linears = find_linears(model, exclude=("lm_head", "embed_out"))
    calib = BlockBatcher(tokenizer, load_eval_text("wikitext2:train"),
                         block_size=2048, batch_size=8)

    sweep = w8a8_alpha_sweep(model, linears, calib, alphas=tuple(args.alphas),
                             max_batches=args.calib_size)

    tot = {"no_smooth": 0.0, **{a: 0.0 for a in args.alphas}}
    for s in sweep.values():
        tot["no_smooth"] += s["no_smooth"]
        for a, v in s["by_alpha"].items():
            tot[a] += v
    print("\n=== W8A8 总损失（越小越好）===")
    print(f"无平滑: {tot['no_smooth']:.4g}")
    for a in args.alphas:
        print(f"α={a}:  {tot[a]:.4g}")
    best = min(args.alphas, key=lambda a: tot[a])
    print(f"→ 最优 α = {best}（无平滑/最优 = {tot['no_smooth']/tot[best]:.2f}×）")


if __name__ == "__main__":
    main()
