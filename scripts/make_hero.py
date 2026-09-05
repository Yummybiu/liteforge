"""从 results/*.json 真实数据再生 README hero 图（数据变更后重跑即可）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from liteforge.report.collect import collect_results  # noqa: E402
from liteforge.report.plots import effective_bits  # noqa: E402


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = collect_results("results")
    models = ["Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen2.5-3B"]
    colors = {"dense": "#111111", "wanda": "#1f77b4", "obc": "#d62728",
              "magnitude": "#bbbbbb", "rtn": "#2ca02c", "gptq": "#9467bd"}
    markers = {"dense": "*", "wanda": "o", "obc": "D", "magnitude": "x",
               "rtn": "s", "gptq": "^"}

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    seen = set()
    for m in models:
        pts = []
        for r in records:
            if not m in r.get("model", ""):
                continue
            ppl = r.get("metrics", {}).get("ppl")
            bits = effective_bits(r)
            if ppl and bits:
                pts.append((bits, ppl, r.get("method")))
        for bits, ppl, method in sorted(pts):
            label = method if method not in seen else None
            seen.add(method)
            ax.scatter(bits, ppl, s=64 if method != "dense" else 140,
                       c=colors.get(method, "#333"), marker=markers.get(method, "o"),
                       label=label, zorder=3, alpha=0.9)
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Effective bits per weight（→ 越靠右压缩越狠）")
    ax.set_ylabel("WikiText-2 PPL（log 轴）")
    ax.set_title("LiteForge: compression–quality trade-off\n"
                 "Qwen2.5-0.5B / 1.5B / 3B · 28 real runs · one-line reproducible")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig("docs/assets/hero.png", dpi=160)
    print("docs/assets/hero.png")


if __name__ == "__main__":
    main()
