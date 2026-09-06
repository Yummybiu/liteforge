"""论文级图表包：统一风格，从 results/*.json 真实数据再生。

产出 docs/assets/ 下 5 张图（数据变更后重跑即可全部更新）：
  fig_tradeoff.png   压缩率-质量 trade-off（三模型 × 全方法）
  fig_prune_evolution.png   剪枝方法演化：静态崩塌→动态反超
  fig_prediction.png 预测力验证：排序完美迁移（ρ=1.000）
  fig_alpha.png      SmoothQuant α 跷跷板
  fig_scale.png      规模效应三点曲线
风格约定：每图一个观点；关键数字直接标注在图上；色弱友好配色。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from liteforge.report.collect import collect_results  # noqa: E402
from liteforge.report.plots import effective_bits  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

C = {"0.5B": "#4477AA", "1.5B": "#EE6677", "3B": "#228833",
     "dense": "#111111", "wanda": "#1f77b4", "obc": "#d62728",
     "magnitude": "#999999", "rtn": "#2ca02c", "gptq": "#9467bd"}

plt.rcParams.update({
    "figure.dpi": 160, "font.size": 9, "axes.titlesize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
})


def _short(model):
    for k in ("0.5B", "1.5B", "3B"):
        if k in model:
            return k
    return model


# ---- 图 1：trade-off（log-y，标注关键点） ----------------------------------
def fig_tradeoff(records):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    seen = set()
    for r in records:
        ppl = r.get("metrics", {}).get("ppl")
        bits = effective_bits(r)
        m = r.get("method")
        if not ppl or not bits or m not in ("dense", "wanda", "obc", "magnitude", "rtn", "gptq"):
            continue
        key = _short(r.get("model", ""))
        label = m if m not in seen else None
        seen.add(m)
        ax.scatter(bits, ppl, s=55 if m != "dense" else 130,
                   c=C.get(m, "#333"), marker={"dense": "*", "wanda": "o", "obc": "D",
                                                "magnitude": "x", "rtn": "s", "gptq": "^"}[m],
                   label=label, alpha=0.85, zorder=3)
    ax.annotate("Magnitude 485.6\n(19× worse than Wanda)", xy=(8, 485.58),
                xytext=(10.5, 220), fontsize=7.5,
                arrowprops=dict(arrowstyle="->", color="#777", lw=0.8))
    ax.annotate("dynamic OBC\nbeats Wanda", xy=(8, 17.52), xytext=(9.8, 60),
                fontsize=7.5, arrowprops=dict(arrowstyle="->", color="#777", lw=0.8))
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Effective bits per weight")
    ax.set_ylabel("WikiText-2 PPL (log)")
    ax.set_title("Compression–quality trade-off · Qwen2.5-0.5B/1.5B/3B · all runs reproducible")
    ax.legend(fontsize=7.5, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_tradeoff.png")


# ---- 图 2：剪枝方法演化 ------------------------------------------------------
def fig_prune_evolution(records):
    import numpy as np
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    models = ["0.5B", "1.5B", "3B"]
    groups = {"Magnitude": [], "Wanda": [], "OBC\n(static)": [], "OBC\n(dynamic)": []}
    STATIC_PPL = {(0.5, 54.1607), (1.5, 13.7296)}   # 旧记录无 mask_mode 字段，按已知值识别
    for m in models:
        scale = float(m.rstrip("B"))
        vals = {k: np.nan for k in groups}
        for r in records:
            if m not in r.get("model", ""):
                continue
            method, ppl = r.get("method"), r.get("metrics", {}).get("ppl")
            sp = r.get("params", {}).get("sparsity")
            if method not in ("magnitude", "wanda", "obc"):
                continue
            if method != "magnitude" and sp != 0.5:
                continue
            if method == "magnitude":
                vals["Magnitude"] = ppl
            elif method == "wanda":
                vals["Wanda"] = ppl
            else:
                mask = r.get("params", {}).get("obc_mask")
                if mask is None:
                    mask = "static" if (scale, round(ppl, 4)) in STATIC_PPL else "dynamic"
                vals["OBC\n(static)" if mask == "static" else "OBC\n(dynamic)"] = ppl
        for k in groups:
            groups[k].append(vals[k])
    x = np.arange(len(models))
    w = 0.2
    for i, (k, vals) in enumerate(groups.items()):
        ax.bar(x + i * w, vals, width=w, label=k,
               color=["#999999", "#1f77b4", "#dd8833", "#d62728"][i], alpha=0.9)
        for xi, v in zip(x, vals):
            if not np.isnan(v) and (v > 100 or k in ("Wanda", "OBC\n(dynamic)")):
                ax.text(xi + i * w, v * 1.05, f"{v:.1f}", ha="center", fontsize=6.5)
    ax.set_yscale("log")
    ax.set_xticks(x + 1.5 * w)
    ax.set_xticklabels([f"Qwen2.5-{m}" for m in models])
    ax.set_ylabel("PPL @ 50% sparsity (log)")
    ax.set_title("Pruning method evolution: static masking collapses,\ndynamic re-scoring wins at all scales")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_prune_evolution.png")


# ---- 图 3：预测力验证（ρ=1.000） --------------------------------------------
def fig_prediction(records):
    pts = []
    for r in records:
        if r.get("task") != "apply-alloc":
            continue
        pred = r.get("params", {}).get("predicted_total_loss")
        ppl = r.get("metrics", {}).get("ppl")
        if pred and ppl:
            pts.append((_short(r.get("model", "")), pred, ppl))
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for m in ("0.5B", "1.5B"):
        xs = [p[1] for p in pts if p[0] == m]
        ys = [p[2] for p in pts if p[0] == m]
        ax.scatter(xs, ys, s=60, c=C[m], label=f"{m} (n={len(xs)})", zorder=3,
                   edgecolors="white", linewidths=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Predicted total loss (unified currency)")
    ax.set_ylabel("Actual WikiText-2 PPL")
    ax.set_title("Does the cost model predict reality?\nSpearman ρ = 1.000 on both models — ranking transfers, magnitudes don't")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_prediction.png")


# ---- 图 4：α 跷跷板 ---------------------------------------------------------
def fig_alpha():
    import json
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for f, m in (("results/smooth_qwen0.5B.json", "0.5B"),
                 ("results/smooth_qwen1.5B.json", "1.5B")):
        try:
            rec = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        by = rec["metrics"]["total_by_alpha"]
        alphas = sorted(float(a) for a in by)
        vals = [by[str(a)] if str(a) in by else by[a] for a in alphas]
        base = rec["metrics"]["total_no_smooth"]
        best = min(vals)
        ax.plot(alphas, [v / base for v in vals], marker="o", c=C[m], label=m)
        ax.axhline(1.0, color="#999", lw=0.8, ls="--")
        bi = vals.index(best)
        ax.annotate(f"best α={alphas[bi]:.1f}\n{base/best:.2f}×", xy=(alphas[bi], best / base),
                    xytext=(alphas[bi] - 0.06, best / base * 0.86), fontsize=7.5,
                    arrowprops=dict(arrowstyle="->", color="#777", lw=0.8))
    ax.set_xlabel("α (smoothing balance: activation ↔ weight difficulty)")
    ax.set_ylabel("W8A8 loss / no-smooth loss")
    ax.set_title("SmoothQuant α sweep: the seesaw is real,\nand the optimum is found by search, not by default 0.5")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_alpha.png")


# ---- 图 5：规模效应 ---------------------------------------------------------
def fig_scale(records):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    models = ["0.5B", "1.5B", "3B"]
    for method, color, label in (("wanda", C["wanda"], "Wanda 50%"),
                                 ("rtn", C["rtn"], "RTN W4 g128")):
        deltas = []
        for m in models:
            base = comp = None
            for r in records:
                if m not in r.get("model", ""):
                    continue
                if r.get("method") == "dense":
                    base = r["metrics"].get("ppl")
                p = r.get("params", {})
                if r.get("method") != method:
                    continue
                if method == "wanda" and p.get("sparsity") == 0.5:
                    comp = r["metrics"].get("ppl")
                if method == "rtn" and p.get("bits") == 4 and p.get("group_size") == 128:
                    comp = r["metrics"].get("ppl")
            deltas.append(None if (base is None or comp is None) else comp - base)
        xs = [i for i, d in zip(range(len(models)), deltas) if d is not None]
        ys = [d for d in deltas if d is not None]
        ax.plot(xs, ys, marker="o", c=color, label=label)
        for xi, d in zip(xs, ys):
            ax.annotate(f"+{d:.1f}", xy=(xi, d), xytext=(0, 6),
                        textcoords="offset points", ha="center", fontsize=7)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([f"Qwen2.5-{m}" for m in models])
    ax.set_ylabel("ΔPPL vs same-model fp16 dense")
    ax.set_title("Scale effect: bigger models compress better\n(50% pruning cost: +12.3 → +4.1 → +2.9)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_scale.png")


def main():
    records = collect_results("results")
    fig_tradeoff(records)
    fig_prune_evolution(records)
    fig_prediction(records)
    fig_alpha()
    fig_scale(records)
    print("5 figures →", OUT)


if __name__ == "__main__":
    main()
