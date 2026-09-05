"""压缩预算分配器（V2 核心）：给定逐层损失菜单，在平均比特预算下分配策略。

问题形式化：每层 l 有离散菜单 {option → (bits_eff, loss)}（lossmeter 产出）。
目标：min Σ_l loss_l(b_l)  s.t.  Σ_l d_l·b_l ≤ target · Σ_l d_l
（d_l 为该层输入维，即权重元素数 / 输出维；预算按**平均每权重比特**计）。

三种求解策略：
- **dp**     : 精确解。菜单比特值离散且全模型共享 → 背包 DP；位宽按 granularity
               量化、维度按 GCD 缩放后状态数可控（0.5B/0.25bit ≈ 数万状态）。
- **greedy** : 边际代价贪心（每 bit 节省的损失增量排序）——文献常用启发式，
               作为 DP 的对照。
- **uniform**: 全层同一选项（all-W4 / all-P50 …）——部署常用基线。

有效比特的诚实口径（与 lossmeter 一致）：分组非对称量化 b_eff = b + 32/gs，
剪枝 b_eff = 16·(1−s)。注意 p50（8.0 bit）与 w8g128（8.0625 bit）几乎同桶——
"同比特下谁更划算"会在这里自动显形。
"""

import logging
import math

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- 菜单预处理
def _options_of(layer_entry: dict) -> dict:
    """兼容两种 schema：lossmeter 新格式 {"shape","options"} 与旧的扁平 {opt: {...}}。"""
    return layer_entry["options"] if "options" in layer_entry else layer_entry


def build_bucket_menus(loss_table: dict, granularity: float = 0.25) -> dict:
    """loss_table → {layer: {bucket_int: {"loss", "opt"}}}（同桶保留损失最小者）。"""
    menus = {}
    for layer, entry in loss_table.items():
        buckets = {}
        for opt_name, o in _options_of(entry).items():
            b = int(round(o["bits_eff"] / granularity))
            if b not in buckets or o["loss"] < buckets[b]["loss"]:
                buckets[b] = {"loss": float(o["loss"]), "opt": opt_name}
        menus[layer] = buckets
    return menus


def _dims_from_loss_table(loss_table: dict, model_linears_dim: dict) -> dict:
    """层 → 输出维×输入维（权重元素数比例，用 d_in·d_out 或 d_in 均可，只差常数倍）。

    allocate 关心的是 Σ d_l·b_l 的相对量，直接用权重元素数 d_out×d_in。
    """
    return {k: model_linears_dim[k] for k in loss_table}


# ---------------------------------------------------------------- DP
def dp_allocate(bucket_menus: dict, dims: dict, target_bits: float,
                granularity: float = 0.25) -> dict:
    """精确 DP。返回 {"layers": {...}, "achieved_bits", "total_loss", "strategy"}。"""
    layers = list(bucket_menus)
    if not layers:
        raise ValueError("空菜单")
    d = np.array([float(dims[l]) for l in layers])
    d_gcd = np.gcd.reduce(d.astype(np.int64))
    d_u = d / d_gcd                       # 缩放后的整型权重
    total_d = d.sum()

    max_bucket = max(max(bm) for bm in bucket_menus.values())
    min_bucket = min(min(bm) for bm in bucket_menus.values())
    budget = int(math.floor(target_bits * total_d / granularity / d_gcd))

    # 状态数保护（异常大的模型/粒度组合时回退贪心）
    if budget + 1 > 4_000_000:
        raise MemoryError(f"DP 状态数 {budget} 过大，调大 granularity 或用 greedy")

    OPT = np.iinfo(np.int8).max
    dp = np.full(budget + 1, np.inf)
    dp[0] = 0.0
    choice = np.full((len(layers), budget + 1), -1, dtype=np.int8)

    for i, layer in enumerate(layers):
        bm = bucket_menus[layer]
        buckets = np.array(sorted(bm))
        losses = np.array([bm[b]["loss"] for b in buckets])
        step = (d_u[i] * buckets).astype(np.int64)   # 该层各选项的预算步长
        new_dp = np.full(budget + 1, np.inf)
        best = np.full(budget + 1, -1, dtype=np.int8)
        for j, b in enumerate(buckets):
            if step[j] > budget:
                continue
            cand = dp[: budget + 1 - step[j]] + losses[j]
            mask = cand < new_dp[step[j]:]
            new_dp[step[j]:][mask] = cand[mask]
            best[step[j]:][mask] = j
        dp = new_dp
        choice[i] = best
        if not np.isfinite(dp).any():
            raise ValueError("预算不足以覆盖所有层（检查 target_bits 下限）")

    # 回溯：从损失最小的可行预算位置出发
    finite = np.isfinite(dp)
    if not finite.any():
        raise ValueError("预算不足以覆盖所有层（检查 target_bits 下限）")
    pos = int(np.argmin(np.where(finite, dp, np.inf)))
    pos_end = pos                      # 终点即"用掉的预算"——回溯会把它减到 0
    alloc, total_loss = {}, 0.0
    for i in range(len(layers) - 1, -1, -1):
        j = int(choice[i, pos])
        if j < 0:
            raise ValueError(f"DP 回溯失败（layer {i}, pos {pos}）")
        bm = bucket_menus[layers[i]]
        b = sorted(bm)[j]
        alloc[layers[i]] = bm[b]["opt"]
        total_loss += bm[b]["loss"]
        pos -= int(d_u[i] * b)
    achieved = pos_end * granularity * d_gcd / total_d
    return {"layers": alloc, "achieved_bits": round(achieved, 4),
            "total_loss": total_loss, "strategy": "dp"}


# ---------------------------------------------------------------- Greedy
def greedy_allocate(bucket_menus: dict, dims: dict, target_bits: float,
                    granularity: float = 0.25) -> dict:
    """边际价值贪心：从全层最省比特的可行点出发，反复执行"每花 1 bit 换回损失
    最多"的升级，直到预算用满或无可行升级。DP 的对照基线（文献常用启发式）。"""
    layers = list(bucket_menus)
    total_d = sum(dims.values())
    cur = {}
    for l in layers:
        bm = bucket_menus[l]
        b_min = min(bm)
        cur[l] = (b_min, bm[b_min])
    bits = sum(dims[l] * cur[l][0] for l in layers) / total_d * granularity

    eps = 1e-9
    while True:
        best = None  # (saved_per_bit, layer, b_next, new_bits)
        for l in layers:
            bm = bucket_menus[l]
            b_now, entry_now = cur[l]
            ups = [b for b in bm if b > b_now]
            if not ups:
                continue
            b_next = min(ups)
            d_bits = (b_next - b_now) * dims[l] * granularity / total_d
            if bits + d_bits > target_bits + eps:
                continue
            saved = entry_now["loss"] - bm[b_next]["loss"]
            if d_bits <= 0:
                continue
            ratio = saved / d_bits
            if best is None or ratio > best[0]:
                best = (ratio, l, b_next, d_bits)
        if best is None:
            break
        _, l, b_next, d_bits = best
        cur[l] = (b_next, bucket_menus[l][b_next])
        bits += d_bits
    return {"layers": {l: cur[l][1]["opt"] for l in layers},
            "achieved_bits": round(bits, 4),
            "total_loss": sum(cur[l][1]["loss"] for l in layers),
            "strategy": "greedy"}


# ---------------------------------------------------------------- Uniform
def uniform_assign(loss_table: dict, opt_name: str) -> dict:
    """基线：全层同一选项（要求菜单含该选项）。"""
    for l, entry in loss_table.items():
        if opt_name not in _options_of(entry):
            raise KeyError(f"层 {l} 菜单缺选项 {opt_name}")
    return {"layers": {l: opt_name for l in loss_table},
            "achieved_bits": None,   # 由调用方按 opt 查
            "total_loss": sum(_options_of(entry)[opt_name]["loss"]
                              for entry in loss_table.values()),
            "strategy": f"uniform-{opt_name}"}


# ---------------------------------------------------------------- 应用
def apply_allocation(model, alloc: dict, calib_batches, max_batches: int = 16,
                     chunk: int = 0, percdamp: float = 0.01):
    """把分配方案应用到模型（原地）。逐块重算 H，仅处理被压缩的层。

    返回 {"applied": {layer: opt}, "skipped_fp16": n}
    """
    from .lossmeter import DEFAULT_MENU, _chunk_linears, prune_copy, quantize_copy
    from .utils import find_linears
    from .utils.hessian import damp_inverse

    menu_by_name = {m["name"]: m for m in DEFAULT_MENU}
    linears = dict(find_linears(model, exclude=("lm_head", "embed_out")))
    todo = {l: o for l, o in alloc.items() if o != "fp16" and l in linears}
    applied = {}
    for _, linears_c in enumerate(_chunk_linears(list(todo), chunk)):
        H_dict = collect_xtx(model, [(l, linears[l]) for l in linears_c],
                             calib_batches, max_batches)
        for l in linears_c:
            opt = todo[l]
            item = menu_by_name[opt]
            m = linears[l]
            H = H_dict[l]
            W = m.weight.data.detach().to(torch.float32)
            Hinv = damp_inverse(H, percdamp)
            U = torch.linalg.cholesky(Hinv, upper=True).to(torch.float32)
            del Hinv
            if item["kind"] == "prune":
                Wc = prune_copy(W, item["sparsity"], U.diagonal().to(torch.float64).clamp(min=1e-12), U,
                                mode="dynamic")
            else:
                Wc = quantize_copy(W, item["bits"], item["group_size"],
                                   symmetric=False, impl="gptq", U=U)
            m.weight.data.copy_(Wc.to(m.weight.data.dtype))
            applied[l] = opt
            del H, U
    skipped = len(alloc) - len(applied)
    return {"applied": applied, "skipped_fp16": skipped}


def _collect_for(model, named_linears, calib_batches, max_batches):
    return collect_xtx(model, named_linears, calib_batches, max_batches)
