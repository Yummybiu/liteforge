"""结果聚合：扫描 results/ 下的统一 JSON 记录，输出 Markdown/CSV 对照表。

统一记录 schema（由 CLI 写出）：
{
  "task": "eval-ppl | prune | quant-rtn | ...",
  "model": "Qwen/Qwen2.5-0.5B",
  "method": "dense | wanda | magnitude | rtn | gptq | awq",
  "params": {...},          # sparsity / bits / group_size ...
  "metrics": {"ppl": .., "generate_tokens_per_s": .., ...},
  "env": {...}
}
"""

import glob
import os

from ..utils import load_json


def collect_results(results_dir: str = "results") -> list:
    records = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            rec = load_json(path)
        except Exception:
            continue
        if "model" not in rec or "method" not in rec:
            continue
        rec["_file"] = os.path.basename(path)
        records.append(rec)
    return records


def _short_model(model: str) -> str:
    return model.split("/")[-1]


def to_markdown(records: list) -> str:
    """按 模型×方法 透视出 PPL 对照表（相同设置取最新一条）。"""
    best = {}
    for rec in records:
        key = (
            rec.get("model", "?"),
            rec.get("method", "?"),
            str(rec.get("params", {}).get("sparsity", "")),
            str(rec.get("params", {}).get("structure", "")),
            str(rec.get("params", {}).get("bits", "")),
            str(rec.get("params", {}).get("group_size", "")),
        )
        best[key] = rec  # collect 按文件名排序，后写覆盖

    header = (
        "| 模型 | 方法 | 稀疏度/位宽 | PPL↓ | 生成 tok/s | 记录文件 |\n"
        "|---|---|---|---|---|---|\n"
    )
    lines = []
    for key, rec in sorted(best.items()):
        model, method, sparsity, structure, bits, group = key
        parts = [sparsity + ("(2:4)" if structure == "2:4" else "") if sparsity else "",
                 f"W{bits}" if bits else "", f"g{group}" if group else ""]
        param = "/".join(p for p in parts if p) or "-"
        ppl = rec.get("metrics", {}).get("ppl")
        tok_s = rec.get("metrics", {}).get("generate_tokens_per_s")
        lines.append(
            f"| {_short_model(model)} | {method} | {param} | "
            f"{ppl if ppl is not None else '-'} | {tok_s if tok_s is not None else '-'} | "
            f"`{rec['_file']}` |"
        )
    return header + "\n".join(lines) + "\n"


def write_csv(records: list, path: str) -> None:
    import csv
    cols = ["model", "method", "sparsity", "bits", "group_size", "ppl",
            "forward_tokens_per_s", "generate_tokens_per_s"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for rec in records:
            params, metrics = rec.get("params", {}), rec.get("metrics", {})
            w.writerow([
                rec.get("model"), rec.get("method"),
                params.get("sparsity"), params.get("bits"), params.get("group_size"),
                metrics.get("ppl"), metrics.get("forward_tokens_per_s"),
                metrics.get("generate_tokens_per_s"),
            ])
