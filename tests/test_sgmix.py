"""SGMix 分配器测试：预算约束与反敏感单调性。"""

from liteforge.prune.sgmix import allocate


def test_mean_equals_budget():
    sens = {f"layer_{i}": v for i, v in enumerate([10, 5, 1, 0.5, 0.2])}
    sp = allocate(sens, budget=0.5)
    assert abs(sum(sp.values()) / len(sp) - 0.5) < 0.02


def test_more_sensitive_gets_less_pruned():
    sens = {"sensitive": 100.0, "robust": 0.1}
    sp = allocate(sens, budget=0.5)
    assert sp["sensitive"] < sp["robust"]


def test_alpha_zero_uniform():
    sens = {f"l{i}": v for i, v in enumerate([100, 1, 1, 1])}
    sp = allocate(sens, budget=0.5, alpha=0.0)
    assert max(sp.values()) - min(sp.values()) < 1e-6


def test_clipping_respected():
    sens = {"v_sensitive": 1e9, "v_robust": 1e-9}
    sp = allocate(sens, budget=0.5, min_s=0.25, max_s=0.65)
    for v in sp.values():
        assert 0.25 - 1e-6 <= v <= 0.65 + 1e-6
