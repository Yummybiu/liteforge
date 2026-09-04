"""Trade-off 图：等效压缩率 vs 困惑度。

等效比特口径（用于把剪枝和量化画进同一张图）：
- fp16 基线 = 16 bit；
- 量化 W4 = 4 bit；W8 = 8 bit；
- 剪枝 sparsity s ≈ 16·(1-s) bit（诚实口径：只算存储，不含稀疏内核加速收益）。
"""

import os


def effective_bits(rec: dict) -> float | None:
    method = rec.get("method")
    params = rec.get("params", {})
    if method == "dense":
        return 16.0
    if method in ("rtn", "gptq", "awq") and params.get("bits"):
        return float(params["bits"])
    if method in ("wanda", "magnitude", "obc") and params.get("sparsity") is not None:
        return 16.0 * (1.0 - float(params["sparsity"]))
    return None


def plot_ppl_tradeoff(records: list, out_path: str = "results/tradeoff.png") -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("绘图需要 matplotlib：pip install matplotlib")

    models = sorted({r.get("model", "?").split("/")[-1] for r in records})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in models:
        pts = []
        for r in records:
            if r.get("model", "?").split("/")[-1] != m:
                continue
            ppl = r.get("metrics", {}).get("ppl")
            bits = effective_bits(r)
            if ppl and bits:
                pts.append((bits, ppl))
        if not pts:
            continue
        pts.sort()
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", label=m)

    ax.set_xlabel("Effective bits per weight (fp16=16)")
    ax.set_ylabel("WikiText-2 PPL ↓")
    ax.set_title("LiteForge: compression-quality trade-off")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    return out_path
