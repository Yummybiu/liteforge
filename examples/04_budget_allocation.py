"""示例 03：预算分配 —— LiteForge V2 的研究问题。

运行（两步，第二步需要 GPU 更快）：
    python examples/04_budget_allocation.py loss --model Qwen/Qwen2.5-0.5B
    python examples/04_budget_allocation.py alloc --target-bits 2.5

你将学到：剪枝与量化如何放进同一损失货币（tr(ΔW·H·ΔWᵀ)）、
精确 DP 分配如何回答"2.5 bit 预算下每层该剪还是该量化"。
"""

import argparse
import sys

from liteforge.allocate import build_bucket_menus, dp_allocate, greedy_allocate
from liteforge.utils import load_json, save_json, setup_logging
from liteforge.utils.common import setup_logging as _sl


def run_loss(args):
    import numpy as np
    from liteforge.data.text import BlockBatcher, load_eval_text
    from liteforge.lossmeter import measure_menus
    from liteforge.models import load_model_and_tokenizer
    from liteforge.utils import find_linears

    model, tokenizer = load_model_and_tokenizer(args.model)
    linears = find_linears(model, exclude=("lm_head", "embed_out"))
    calib = BlockBatcher(tokenizer, load_eval_text("wikitext2:train"),
                         block_size=2048, batch_size=8)
    table = measure_menus(model, linears, calib, max_batches=16, chunk=6)
    save_json({"model": args.model, "table": table}, args.losses)
    n = len(table)
    mean_loss = float(np.mean([v["options"]["w4g128"]["loss"] for v in table.values()]))
    print(f"损失表完成：{n} 层 × 8 选项 → {args.losses}（w4g128 平均损失 {mean_loss:.3g}）")


def run_alloc(args):
    table = load_json(args.losses)["table"]
    dims = {l: int(np.prod(v["shape"])) for l, v in table.items()}
    menus = build_bucket_menus(table, granularity=0.25)
    dp = dp_allocate(menus, dims, args.target_bits)
    gr = greedy_allocate(menus, dims, args.target_bits)
    from collections import Counter
    print(f"=== target = {args.target_bits} bit/权重 ===")
    print(f"DP 精确解:   {dp['achieved_bits']:.3f} bit, 预测损失 {dp['total_loss']:.4g}")
    print(f"  分布: {dict(Counter(dp['layers'].values()))}")
    print(f"贪心对照:   {gr['achieved_bits']:.3f} bit, 预测损失 {gr['total_loss']:.4g}")
    save_json({"strategy": "dp", "target_bits": args.target_bits,
               "layers": dp["layers"]}, "results/example04_alloc.json")
    print("下一步: python -m liteforge.cli apply-alloc --model <模型> "
          "--alloc results/example04_alloc.json --eval")


def main():
    _sl()
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["loss", "alloc"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--losses", default="results/losses.json")
    ap.add_argument("--target-bits", type=float, default=2.5)
    args = ap.parse_args()
    {"loss": run_loss, "alloc": run_alloc}[args.mode](args)


if __name__ == "__main__":
    main()
