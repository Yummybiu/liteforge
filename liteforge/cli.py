"""LiteForge CLI：python -m liteforge.cli <command>

常用命令：
  eval-ppl   困惑度评测（wikitext2 或任意文本）
  prune      剪枝（wanda / magnitude），可接 --eval 直接产出剪枝后 PPL
  quant-rtn  RTN 伪量化（从零实现），可接 --eval
  quant-gptq / quant-awq   可部署整数权重量化（需可选依赖）
  speed      前向 / 生成吞吐
  report     聚合 results/*.json 为 Markdown 对照表（可加 --plot 出 trade-off 图）

所有命令输出统一的 JSON 记录到 --out（默认 results/ 下按时间命名）。
"""

import argparse
import time

from .utils import save_json, seed_everything, setup_logging, torch_env_info
import logging

logger = logging.getLogger("liteforge")


def build_record(task, model_id, method, params, metrics):
    return {
        "task": task,
        "model": model_id,
        "method": method,
        "params": params,
        "metrics": metrics,
        "env": torch_env_info(),
    }


def default_out(prefix: str) -> str:
    return "results/{}_{}.json".format(time.strftime("%Y%m%d_%H%M%S"), prefix)


def load_pair(args):
    from .models import load_model_and_tokenizer
    return load_model_and_tokenizer(args.model, device=args.device,
                                    dtype=args.dtype,
                                    trust_remote=args.trust_remote)


def add_model_args(p: argparse.ArgumentParser):
    p.add_argument("--model", required=True, help="HF 模型 ID 或本地路径")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--dtype", default="auto",
                   choices=["auto", "fp32", "fp16", "bf16"])
    p.add_argument("--trust-remote", action="store_true")
    p.add_argument("--seed", type=int, default=42)


def add_eval_args(p: argparse.ArgumentParser):
    p.add_argument("--dataset", default="wikitext2",
                   help="wikitext2[:split] 或 file:/path/to.txt")
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-blocks", type=int, default=None,
                   help="限制评测块数（快速冒烟用）")
    p.add_argument("--speed", action="store_true", help="附带生成吞吐测量")


def eval_ppl_and_speed(model, tokenizer, args) -> dict:
    from .eval import compute_perplexity
    res = compute_perplexity(model, tokenizer, dataset=args.dataset,
                             block_size=args.seqlen, batch_size=args.batch_size,
                             max_blocks=args.max_blocks)
    metrics = res.to_dict()
    if args.speed:
        from .eval import benchmark_generate
        metrics.update(benchmark_generate(model, tokenizer, max_new_tokens=128))
    return metrics


# ---------------------------------------------------------------- commands
def cmd_eval_ppl(args):
    model, tokenizer = load_pair(args)
    metrics = eval_ppl_and_speed(model, tokenizer, args)
    rec = build_record("eval-ppl", args.model, "dense", {}, metrics)
    path = save_json(rec, args.out or default_out("dense"))
    logger.info("PPL=%.4f  (%s)", metrics["ppl"], path)
    return rec


def cmd_prune(args):
    from .data.text import BlockBatcher, load_eval_text
    from .prune import MagnitudePruner, WandaPruner

    model, tokenizer = load_pair(args)

    sparsity = args.sparsity
    if args.sparsity_map:
        from .utils import load_json
        sparsity = load_json(args.sparsity_map)
        vals = [v for v in sparsity.values()]
        params = {"sparsity_map": args.sparsity_map,
                  "mean_sparsity": round(sum(vals) / len(vals), 4)}
    else:
        params = {"sparsity": args.sparsity, "structure": args.structure}

    needs_calib = args.method in ("wanda", "obc") or bool(args.sparsity_map)
    calib = None
    if needs_calib:
        text = load_eval_text(args.calib_dataset)
        calib = BlockBatcher(tokenizer, text, block_size=args.seqlen,
                             batch_size=args.batch_size)

    if args.dry_score:
        # 只算敏感度不落刀（SGMix 第一步）
        from .prune import OBCPruner
        from .prune.base import PruneConfig
        pruner = OBCPruner(model, PruneConfig(sparsity=args.sparsity),
                           percdamp=args.percdamp)
        sens = pruner.score_dry(calib, max_batches=args.calib_size)
        save_json(sens, args.dry_score)
        logger.info("敏感度已写入 %s（%d 层）", args.dry_score, len(sens))
        return None

    if args.method == "obc":
        from .prune import OBCPruner
        from .prune.base import PruneConfig
        cfg = PruneConfig(sparsity=sparsity, structure=args.structure)
        pruner = OBCPruner(model, cfg, percdamp=args.percdamp,
                           mask_mode=args.obc_mask)
        result = pruner.run(calib_batches=calib, max_batches=args.calib_size)
    else:
        pruner_cls = WandaPruner if args.method == "wanda" else MagnitudePruner
        from .prune.base import PruneConfig
        cfg = PruneConfig(sparsity=sparsity, structure=args.structure)
        pruner = pruner_cls(model, cfg)
        result = pruner.run(calib_batches=calib, max_batches=args.calib_size)
    params.update({"layer_reports_n": len(result.layer_reports),
                   "overall_sparsity": result.overall_sparsity})
    if args.method == "obc":
        params["obc_mask"] = args.obc_mask

    metrics = {}
    if args.eval:
        metrics = eval_ppl_and_speed(model, tokenizer, args)

    if args.save:
        model.save_pretrained(args.save)
        tokenizer.save_pretrained(args.save)
        params["saved_to"] = args.save

    rec = build_record("prune", args.model, args.method, params, metrics)
    path = save_json(rec, args.out or default_out(f"prune_{args.method}"))
    logger.info("剪枝后 overall_sparsity=%.2f%%  %s", result.overall_sparsity * 100, path)
    return rec


def cmd_quant_rtn(args):
    from .quant import RTNConfig, RTNQuantizer

    model, tokenizer = load_pair(args)
    cfg = RTNConfig(bits=args.bits, group_size=args.group_size,
                    symmetric=args.symmetric)
    q = RTNQuantizer(model, cfg)
    report = q.quantize_()
    params = {"bits": args.bits, "group_size": args.group_size,
              "symmetric": args.symmetric}
    metrics = {}
    if args.eval:
        metrics = eval_ppl_and_speed(model, tokenizer, args)
    if args.restore:
        q.restore()
    rec = build_record("quant-rtn", args.model, "rtn", params, metrics)
    rec["quant_report"] = report
    path = save_json(rec, args.out or default_out(f"rtn_w{args.bits}"))
    logger.info("RTN W%d g%d 完成：%s", args.bits, args.group_size, path)
    return rec


def cmd_quant_gptq(args):
    from .utils import save_json
    if args.impl == "scratch":
        from .data.text import BlockBatcher, load_eval_text
        from .quant import GPTQQuantizer, RTNConfig
        model, tokenizer = load_pair(args)
        cfg = RTNConfig(bits=args.bits, group_size=args.group_size)
        calib = BlockBatcher(tokenizer, load_eval_text(args.calib_dataset),
                             block_size=args.seqlen, batch_size=args.batch_size)
        q = GPTQQuantizer(model, cfg, percdamp=args.percdamp)
        report = q.quantize_(calib, max_batches=args.calib_size)
        metrics = eval_ppl_and_speed(model, tokenizer, args) if args.eval else {}
        if args.restore:
            q.restore()
        rec = build_record("quant-gptq-scratch", args.model, "gptq",
                           {"bits": args.bits, "group_size": args.group_size,
                            "impl": "scratch"}, metrics)
        rec["quant_report"] = report
        return save_json(rec, args.out or default_out("gptq_scratch"))
    # 库版（可部署打包格式）
    from .quant.wrappers import quantize_gptq
    model, tokenizer = load_pair(args)
    from .data.text import load_eval_text
    calib_texts = [load_eval_text(args.calib_dataset)[:args.calib_chars]]
    quantize_gptq(model, tokenizer, calib_texts, bits=args.bits,
                  group_size=args.group_size)
    metrics = eval_ppl_and_speed(model, tokenizer, args) if args.eval else {}
    rec = build_record("quant-gptq-lib", args.model, "gptq",
                       {"bits": args.bits, "group_size": args.group_size,
                        "impl": "lib"}, metrics)
    return save_json(rec, args.out or default_out("gptq_lib"))


def cmd_quant_awq(args):
    from .quant.wrappers import quantize_awq
    from .data.text import load_eval_text
    calib_texts = [load_eval_text(args.calib_dataset)[:args.calib_chars]]
    out_dir = quantize_awq(args.model, args.save or "outputs/awq_model",
                           calib_texts, bits=args.bits, group_size=args.group_size)
    rec = build_record("quant-awq", args.model, "awq",
                       {"bits": args.bits, "group_size": args.group_size,
                        "saved_to": out_dir}, {})
    return save_json(rec, args.out or default_out("awq"))


def cmd_speed(args):
    from .eval import benchmark_forward, benchmark_generate
    model, tokenizer = load_pair(args)
    metrics = benchmark_forward(model, tokenizer, seqlen=args.seqlen)
    metrics.update(benchmark_generate(model, tokenizer, max_new_tokens=128))
    rec = build_record("speed", args.model, "dense", {}, metrics)
    return save_json(rec, args.out or default_out("speed"))


def cmd_report(args):
    from .report import collect_results, plot_ppl_tradeoff, to_markdown, write_csv
    records = collect_results(args.results_dir)
    if not records:
        logger.warning("results/ 下没有可识别的记录")
        return
    table = to_markdown(records)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# LiteForge 实验对照表\n\n" + table)
    write_csv(records, args.out.replace(".md", ".csv"))
    logger.info("\n%s", table)
    if args.plot:
        png = plot_ppl_tradeoff(records)
        logger.info("trade-off 图：%s", png)


# ---------------------------------------------------------------- parser
def main(argv=None):
    setup_logging()
    p = argparse.ArgumentParser(prog="liteforge",
                                description="LLM 压缩-评测-部署工具箱")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("eval-ppl", help="困惑度评测")
    add_model_args(sp); add_eval_args(sp)
    sp.add_argument("--out", default=None)
    sp.set_defaults(fn=cmd_eval_ppl)

    sp = sub.add_parser("prune", help="剪枝（wanda/magnitude/obc）")
    add_model_args(sp); add_eval_args(sp)
    sp.add_argument("--method", default="wanda",
                    choices=["wanda", "magnitude", "obc"])
    sp.add_argument("--sparsity", type=float, default=0.5)
    sp.add_argument("--structure", default="unstructured",
                    choices=["unstructured", "2:4"])
    sp.add_argument("--sparsity-map", default=None,
                    help="JSON：逐层稀疏度（SGMix 流程产物）")
    sp.add_argument("--percdamp", type=float, default=0.01,
                    help="OBC/GPTQ 的 Hessian 阻尼比例")
    sp.add_argument("--obc-mask", default="dynamic",
                    choices=["dynamic", "static"],
                    help="dynamic=SparseGPT 块级重评分（忠实版）；static=静态预选（消融）")
    sp.add_argument("--dry-score", default=None,
                    help="只计算 OBS 敏感度并写入该 JSON，不剪枝")
    sp.add_argument("--calib-dataset", default="wikitext2:train")
    sp.add_argument("--calib-size", type=int, default=16,
                    help="校准批数（wanda/obc 用）")
    sp.add_argument("--include", nargs="*", default=[],
                    help="只剪名字含这些子串的层")
    sp.add_argument("--save", default=None, help="保存剪枝后模型目录")
    sp.add_argument("--eval", action="store_true", help="剪枝后立即评 PPL")
    sp.add_argument("--out", default=None)
    sp.set_defaults(fn=cmd_prune)

    sp = sub.add_parser("quant-rtn", help="RTN 伪量化（从零实现）")
    add_model_args(sp); add_eval_args(sp)
    sp.add_argument("--bits", type=int, default=4)
    sp.add_argument("--group-size", type=int, default=128)
    sp.add_argument("--symmetric", action="store_true")
    sp.add_argument("--eval", action="store_true")
    sp.add_argument("--restore", action="store_true",
                    help="评测后还原权重（内存内连续实验用）")
    sp.add_argument("--out", default=None)
    sp.set_defaults(fn=cmd_quant_rtn)

    sp = sub.add_parser("quant-gptq", help="GPTQ（--impl scratch=自研 | lib=可部署打包）")
    add_model_args(sp); add_eval_args(sp)
    sp.add_argument("--impl", default="scratch", choices=["scratch", "lib"])
    sp.add_argument("--bits", type=int, default=4)
    sp.add_argument("--group-size", type=int, default=128)
    sp.add_argument("--percdamp", type=float, default=0.01)
    sp.add_argument("--calib-dataset", default="wikitext2:train")
    sp.add_argument("--calib-size", type=int, default=16)
    sp.add_argument("--calib-chars", type=int, default=200_000)
    sp.add_argument("--restore", action="store_true")
    sp.add_argument("--eval", action="store_true")
    sp.add_argument("--out", default=None)
    sp.set_defaults(fn=cmd_quant_gptq)

    sp = sub.add_parser("quant-awq", help="AWQ（需 autoawq）")
    add_model_args(sp)
    sp.add_argument("--bits", type=int, default=4)
    sp.add_argument("--group-size", type=int, default=128)
    sp.add_argument("--calib-dataset", default="wikitext2:train")
    sp.add_argument("--calib-chars", type=int, default=200_000)
    sp.add_argument("--save", default="outputs/awq_model")
    sp.add_argument("--out", default=None)
    sp.set_defaults(fn=cmd_quant_awq)

    sp = sub.add_parser("speed", help="前向/生成吞吐基准")
    add_model_args(sp)
    sp.add_argument("--seqlen", type=int, default=1024)
    sp.add_argument("--out", default=None)
    sp.set_defaults(fn=cmd_speed)

    sp = sub.add_parser("report", help="聚合 results/*.json 出对照表")
    sp.add_argument("--results-dir", default="results")
    sp.add_argument("--out", default="results/table.md")
    sp.add_argument("--plot", action="store_true")
    sp.set_defaults(fn=cmd_report)

    args = p.parse_args(argv)
    seed_everything(args.seed if hasattr(args, "seed") else 42)
    return args.fn(args)


if __name__ == "__main__":
    main()
