"""示例 01：十分钟上手 —— 加载模型并计算 WikiText-2 困惑度。

运行（仓库根目录）：
    python examples/01_quickstart_ppl.py --model Qwen/Qwen2.5-0.5B

你将学到：LiteForge 的 PPL 口径（非重叠滑窗，与 lm-eval 一致）、
统一 JSON 结果记录的产出方式。
"""

import argparse

from liteforge.data.text import load_eval_text
from liteforge.eval import compute_perplexity
from liteforge.models import load_model_and_tokenizer
from liteforge.utils import save_json, seed_everything, setup_logging


def main():
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--max-blocks", type=int, default=32, help="冒烟用；正式实验 ≥256")
    args = ap.parse_args()
    seed_everything()

    model, tokenizer = load_model_and_tokenizer(args.model)
    res = compute_perplexity(model, tokenizer, dataset="wikitext2",
                             max_blocks=args.max_blocks)
    print(f"\n=== {args.model} ===")
    print(f"PPL = {res.ppl:.4f}  ({res.n_tokens} tokens, {res.n_blocks} blocks)")

    save_json({"task": "eval-ppl", "model": args.model, "method": "dense",
               "params": {}, "metrics": res.to_dict()},
              "results/example01_ppl.json")
    print("记录已写入 results/example01_ppl.json")


if __name__ == "__main__":
    main()
