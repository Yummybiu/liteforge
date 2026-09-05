"""损失模型预测力分析：predicted_total_loss vs 实际 PPL 的秩相关。

用法: python scripts/analyze_prediction.py [--results-dir results]
对全部 apply-alloc 记录计算 Spearman 秩相关（分配器只消费排序，
所以秩相关是成本模型可信度的正确度量——幅度差 46× 也不影响分配决策）。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from liteforge.report.collect import collect_results  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    pts = []
    for r in collect_results(args.results_dir):
        if r.get("task") != "apply-alloc":
            continue
        pred = r.get("params", {}).get("predicted_total_loss")
        ppl = r.get("metrics", {}).get("ppl")
        if pred and ppl:
            model = str(r.get("model", "")).split("\\")[-1].split("/")[-1]
            pts.append((model, r["params"].get("target_bits"),
                        r["params"].get("strategy"), pred, ppl))

    if len(pts) < 2:
        print("样本不足")
        return

    def spearman(xs, ys):
        def rank(v):
            order = sorted(range(len(v)), key=lambda i: v[i])
            rk = [0.0] * len(v)
            for r_, i in enumerate(order):
                rk[i] = r_
            return rk
        rx, ry = rank(xs), rank(ys)
        n = len(xs)
        mx, my = sum(rx) / n, sum(ry) / n
        num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
        return num / den if den else float("nan")

    print(f"{'model':<16}{'target':>7}{'strategy':>10}{'pred_loss':>14}{'PPL':>12}")
    for model, t, s, p, ppl in sorted(pts):
        print(f"{model:<16}{str(t):>7}{str(s):>10}{p:>14.3e}{ppl:>12.2f}")

    for model in sorted({p[0] for p in pts}):
        sub = [p for p in pts if p[0] == model]
        preds, ppls = [p[3] for p in sub], [p[4] for p in sub]
        rho = spearman(preds, ppls)
        print(f"\n{model}: Spearman 秩相关 = {rho:.3f} (n={len(sub)})")


if __name__ == "__main__":
    main()
