"""分配器测试：DP 精确性（对暴力枚举）、贪心对照、预算可行性。"""

import itertools

import numpy as np
import pytest

from liteforge.allocate import build_bucket_menus, dp_allocate, greedy_allocate, uniform_assign


def _synthetic_loss_table(n_layers=4, seed=0):
    """合成损失表：凸增的量化损失 + 随机剪枝损失；形状各异（非平凡加权背包）。"""
    rng = np.random.RandomState(seed)
    shapes = [[64, 32], [32, 64], [16, 32], [48, 16], [32, 16], [64, 64]]
    table = {}
    for i in range(n_layers):
        scale = rng.uniform(0.5, 2.0) * (1 + 0.1 * i)
        table[f"layer_{i}"] = {"shape": shapes[i % len(shapes)], "options": {
            "fp16": {"bits_eff": 16.0, "loss": 0.0, "kind": "keep"},
            "w8g128": {"bits_eff": 8.0625, "loss": 0.02 * scale * rng.uniform(0.9, 1.1), "kind": "quant"},
            "p50": {"bits_eff": 8.0, "loss": 0.05 * scale * rng.uniform(0.9, 1.1), "kind": "prune"},
            "w4g128": {"bits_eff": 4.25, "loss": 0.2 * scale * rng.uniform(0.9, 1.1), "kind": "quant"},
            "p75": {"bits_eff": 4.0, "loss": 0.35 * scale * rng.uniform(0.9, 1.1), "kind": "prune"},
            "w2g128": {"bits_eff": 2.25, "loss": 1.2 * scale * rng.uniform(0.9, 1.1), "kind": "quant"},
        }}
    return table


def _dims(table):
    return {l: int(np.prod(v["shape"])) for l, v in table.items()}


def _brute_force_min_loss(bucket_menus, dims, budget):
    """小规模实例的暴力最优（与 DP 同一口径：Σ (d/gcd)·b ≤ budget）。"""
    layers = list(bucket_menus)
    g = np.gcd.reduce(np.array([dims[l] for l in layers], dtype=np.int64))
    best = np.inf
    for combo in itertools.product(*[sorted(bucket_menus[l]) for l in layers]):
        total_b = sum(dims[l] // int(g) * b for l, b in zip(layers, combo))
        if total_b <= budget:
            loss = sum(bucket_menus[l][b]["loss"] for l, b in zip(layers, combo))
            best = min(best, loss)
    return best


def _budget(menus, dims, target, granularity=0.25):
    g = np.gcd.reduce(np.array([dims[l] for l in menus], dtype=np.int64))
    return int(target * sum(dims.values()) / granularity / int(g))


def test_bucket_merging_same_bit():
    """p50(8.0bit) 与 w8(8.0625bit) 在 0.25 粒度下同桶，保留损失最小者。"""
    table = _synthetic_loss_table(n_layers=1)
    menus = build_bucket_menus(table, granularity=0.25)
    bm = menus["layer_0"]
    b8 = int(round(8.0 / 0.25))  # 32
    assert b8 in bm
    winner = bm[b8]["opt"]
    assert winner in ("p50", "w8g128")
    assert bm[b8]["loss"] == min(table["layer_0"]["options"]["p50"]["loss"],
                                 table["layer_0"]["options"]["w8g128"]["loss"])


def test_dp_matches_brute_force():
    for seed in range(3):
        table = _synthetic_loss_table(n_layers=3, seed=seed)
        menus = build_bucket_menus(table, granularity=0.25)
        dims = _dims(table)
        budget = _budget(menus, dims, 2.5)
        res = dp_allocate(menus, dims, 2.5, granularity=0.25)
        best = _brute_force_min_loss(menus, dims, budget)
        assert abs(res["total_loss"] - best) < 1e-9 * max(best, 1e-9), \
            f"seed={seed}: DP {res['total_loss']} != 暴力最优 {best}"


def test_dp_beats_or_ties_greedy():
    table = _synthetic_loss_table(n_layers=6, seed=7)
    menus = build_bucket_menus(table, granularity=0.25)
    dims = _dims(table)
    dp = dp_allocate(menus, dims, 2.5)
    gr = greedy_allocate(menus, dims, 2.5)
    assert dp["total_loss"] <= gr["total_loss"] + 1e-12
    assert dp["achieved_bits"] <= 2.5 + 1e-6
    assert gr["achieved_bits"] <= 2.5 + 1e-6
    # 一致性：DP 报告的损失 = 所分选项损失之和（回溯无账目错位）
    opts = {l: v["options"] for l, v in table.items()}
    accounted = sum(opts[l][o]["loss"] for l, o in dp["layers"].items())
    assert abs(accounted - dp["total_loss"]) < 1e-9 * max(accounted, 1e-9)


def test_uniform_and_errors():
    table = _synthetic_loss_table(n_layers=2)
    u = uniform_assign(table, "w4g128")
    assert set(u["layers"].values()) == {"w4g128"}
    with pytest.raises(KeyError):
        uniform_assign(table, "w5g128")


def test_infeasible_budget_raises():
    table = _synthetic_loss_table(n_layers=2)
    menus = build_bucket_menus(table, granularity=0.25)
    dims = _dims(table)
    with pytest.raises((ValueError, MemoryError)):
        dp_allocate(menus, dims, 0.5, granularity=0.25)  # 低于全 w2 的平均比特
