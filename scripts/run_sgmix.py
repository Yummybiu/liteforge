"""SGMix 端到端：敏感度评分 → 混合稀疏分配 → 剪枝 → 评测。

用法：
  python scripts/run_sgmix.py --model F:/.../Qwen2.5-0.5B --budget 0.5

流程：
  1) OBC dry-score：一次校准前向，取每层 OBS 理论删除损失（敏感度）；
  2) allocate：同预算生成逐层稀疏度表（α=1 反比分配）；
  3) Wanda 按表剪枝（用 Wanda 打分保证与均匀组可比——分配策略是唯一变量）；
  4) PPL 评测，与 uniform 记录对照。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from liteforge.cli import build_record, default_out, eval_ppl_and_speed  # noqa: E402
from liteforge.data.text import BlockBatcher, load_eval_text  # noqa: E402
from liteforge.models import load_model_and_tokenizer  # noqa: E402
from liteforge.prune import OBCPruner, WandaPruner  # noqa: E402
from liteforge.prune.base import PruneConfig  # noqa: E402
from liteforge.prune.sgmix import allocate  # noqa: E402
from liteforge.utils import save_json, seed_everything, setup_logging, torch_env_info  # noqa: E402
import logging  # noqa: E402

logger = logging.getLogger("sgmix")


def main():
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--budget", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--calib-size", type=int, default=16)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-blocks", type=int, default=32)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    args = p.parse_args()
    seed_everything(42)

    model, tokenizer = load_model_and_tokenizer(args.model, args.device, args.dtype)

    calib = BlockBatcher(tokenizer, load_eval_text("wikitext2:train"),
                         block_size=args.seqlen, batch_size=args.batch_size)

    # 1) 敏感度
    logger.info("[1/3] OBC dry-score 敏感度评估 ...")
    pruner = OBCPruner(model, PruneConfig(sparsity=args.budget))
    sens = pruner.score_dry(calib, max_batches=args.calib_size)

    # 2) 分配
    sp_map = allocate(sens, budget=args.budget, alpha=args.alpha)
    lo = min(sp_map.values()); hi = max(sp_map.values())
    logger.info("[2/3] 分配完成：稀疏度范围 [%.2f, %.2f]，均值 %.3f",
                lo, hi, sum(sp_map.values()) / len(sp_map))

    # 3) 按表 Wanda 剪枝
    logger.info("[3/3] Wanda 按逐层表剪枝 ...")
    wpruner = WandaPruner(model, PruneConfig(sparsity=sp_map))
    result = wpruner.run(calib_batches=calib, max_batches=args.calib_size)

    # 4) 评测
    class A: pass
    a = A(); a.dataset = "wikitext2"; a.seqlen = args.seqlen
    a.batch_size = args.batch_size; a.max_blocks = args.max_blocks; a.speed = True
    metrics = eval_ppl_and_speed(model, tokenizer, a)

    rec = build_record(
        "prune-sgmix", args.model, "wanda-sgmix",
        params={"budget": args.budget, "alpha": args.alpha,
                "mean_sparsity": result.overall_sparsity,
                "min": round(lo, 3), "max": round(hi, 3),
                "sparsity_map": sp_map},
        metrics=metrics)
    rec["env"] = torch_env_info()
    path = save_json(rec, default_out("sgmix"))
    logger.info("SGMix 完成：PPL=%.4f（uniform Wanda 50% 对照 25.52@0.5B）\n%s",
                metrics["ppl"], path)
    # 敏感度表落盘
    save_json(sens, path.replace(".json", "_sensitivity.json"))


if __name__ == "__main__":
    main()
