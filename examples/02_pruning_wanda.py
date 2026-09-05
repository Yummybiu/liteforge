"""示例 02：剪枝 —— Wanda 激活感知剪枝，并对比幅度剪枝基线。

运行：
    python examples/02_pruning_wanda.py --model Qwen/Qwen2.5-0.5B

你将学到：激活感知打分（|W|·‖X‖）为什么比幅度剪枝好得多（本仓库实测
约 19 倍 PPL 差距）、剪枝的诚实评测口径（稠密置零，方法学质量）。
"""

import argparse

from liteforge.data.text import BlockBatcher, load_eval_text
from liteforge.eval import compute_perplexity
from liteforge.models import load_model_and_tokenizer
from liteforge.prune import MagnitudePruner, WandaPruner
from liteforge.prune.base import PruneConfig
from liteforge.utils import seed_everything, setup_logging


def main():
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--sparsity", type=float, default=0.5)
    ap.add_argument("--calib-size", type=int, default=16)
    args = ap.parse_args()
    seed_everything()

    model, tokenizer = load_model_and_tokenizer(args.model)
    base_ppl = compute_perplexity(model, tokenizer, dataset="wikitext2",
                                  max_blocks=32).ppl
    calib = BlockBatcher(tokenizer, load_eval_text("wikitext2:train"),
                         block_size=2048, batch_size=8)

    results = {}
    for name, cls in (("wanda", WandaPruner), ("magnitude", MagnitudePruner)):
        # 每种方法从原始权重重新剪枝，保证可比
        model, tokenizer = load_model_and_tokenizer(args.model)
        pruner = cls(model, PruneConfig(sparsity=args.sparsity))
        result = pruner.run(calib_batches=calib, max_batches=args.calib_size)
        ppl = compute_perplexity(model, tokenizer, dataset="wikitext2",
                                 max_blocks=32).ppl
        results[name] = ppl
        print(f"\n{name} @ {result.overall_sparsity:.0%}: PPL = {ppl:.2f}")

    print(f"\n=== 汇总（base={base_ppl:.2f}）===")
    print(f"Wanda 激活感知: {results['wanda']:.2f}")
    print(f"Magnitude 对照: {results['magnitude']:.2f}"
          f"  （差 {results['magnitude']/results['wanda']:.1f} 倍——激活感知的价值）")


if __name__ == "__main__":
    main()
