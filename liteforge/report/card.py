"""压缩报告卡：扫描 results/*.json，生成单文件 Markdown 报告。

面试展示形态：一个文件看懂一个模型的全部压缩证据——V1 对照表、
V2 分配方案与预测/实际对照、（将来）α 扫描 / 投机解码 / 下游任务。
"""

import glob
import os

from ..utils import load_json


def _short(model: str) -> str:
    return os.path.basename(model.rstrip("/"))


def _fmt(x, nd=2):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else (x or "-")


def collect_all(results_dir: str = "results") -> list:
    recs = []
    for p in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            recs.append(load_json(p))
        except Exception:
            continue
    return recs


def build_report_card(model_key: str, results_dir: str = "results",
                      out_path: str | None = None) -> str:
    """model_key: 模型名子串（如 '0.5B'）。生成该模型的报告卡 Markdown。"""
    recs = [r for r in collect_all(results_dir)
            if model_key in str(r.get("model", ""))]
    if not recs:
        raise ValueError(f"results 下没有匹配 {model_key} 的记录")

    lines = [f"# LiteForge 压缩报告卡 — {model_key}", "",
             f"> 自动生成于 {len(recs)} 条实验记录；全部数字可由仓库脚本一键复现。"
             f"口径：剪枝/伪量化为方法学质量（稠密推理），部署速度单独由 vLLM 实测。",
             ""]

    # ---- V1：压缩对照表
    rows = []
    for r in recs:
        if r.get("task") in ("eval-ppl", "prune", "quant-rtn", "quant-gptq-scratch",
                             "prune-sgmix"):
            p, m = r.get("params", {}), r.get("metrics", {})
            if "ppl" not in m:
                continue
            tag = r.get("method", "?")
            extra = "/".join(str(p[k]) for k in ("sparsity", "structure") if k in p) \
                or (f"W{p['bits']}/g{p['group_size']}" if "bits" in p else "")
            rows.append((tag, extra, m["ppl"]))
    if rows:
        lines += ["## 一、压缩-质量对照（WikiText-2 PPL）", "",
                  "| 方法 | 设置 | PPL ↓ |", "|---|---|---|"]
        lines += [f"| {t} | {e or '-'} | {_fmt(p)} |" for t, e, p in rows]
        lines.append("")

    # ---- V2：分配
    allocs = [r for r in recs if r.get("task") == "allocate"]
    applies = [r for r in recs if r.get("task") == "apply-alloc"]
    if allocs:
        lines += ["## 二、预算分配（V2）", ""]
        for a in allocs:
            p, m = a.get("params", {}), a.get("metrics", {})
            lines.append(f"- {a.get('strategy')} @ target {p.get('target_bits')} bit："
                         f"实际 {_fmt(m.get('achieved_bits'), 3)} bit，"
                         f"预测总损失 {_fmt(m.get('predicted_total_loss'), 4)}，"
                         f"分布 {m.get('distribution')}")
        lines.append("")
    if applies:
        lines += ["### 预测 vs 实际（损失模型的可信度）", "",
                  "| 方案 | 预测损失 | 实际 PPL |", "|---|---|---|"]
        for r in applies:
            p, m = r.get("params", {}), r.get("metrics", {})
            lines.append(f"| {os.path.basename(str(p.get('alloc_file', '')))} | "
                         f"{_fmt(p.get('predicted_total_loss'), 4)} | {_fmt(m.get('ppl'))} |")
        lines.append("")

    # ---- SmoothQuant / 投机解码 / 下游
    for task, title in (("smooth-alpha", "三、SmoothQuant W8A8 α 扫描"),
                        ("spec-bench", "四、投机解码"),
                        ("eval-mmlu", "五、MMLU-mini 下游")):
        rs = [r for r in recs if r.get("task") == task]
        if rs:
            lines += [f"## {title}", "", "```json",
                      __import__("json").dumps(rs[-1], ensure_ascii=False, indent=1)[:1500],
                      "```", ""]

    text = "\n".join(lines) + "\n"
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
    return text
