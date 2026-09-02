"""SGMix：敏感度引导的混合稀疏（本项目的原创实验贡献）。

思想：不是所有层都同样耐压。用 OBC 的理论删除损失（被删权重的
score = w²/(H⁻¹)_jj 之和）作为每层的敏感度，在同一平均稀疏预算下，
给敏感层分配更低的稀疏度、给鲁棒层更高的稀疏度，对比均匀剪枝。

分配策略（启发式，诚实地说是启发式）：
    s_i = budget · (S_mean / S_i)^α，再裁剪到 [min_s, max_s] 并重归一到均值。
α=1 为标准反比分配；α=0 退化为均匀。项目实验对比两者。
"""

import numpy as np


def allocate(sensitivity: dict, budget: float = 0.5, alpha: float = 1.0,
             min_s: float = 0.25, max_s: float = 0.65) -> dict:
    """sensitivity: {layer: obs_loss 越大越敏感} → {layer: sparsity}。

    裁剪到边界后用迭代投影收敛回平均预算（水填法投影，20 轮足够）。
    """
    assert sensitivity, "sensitivity 为空"
    names = list(sensitivity)
    vals = np.array([max(sensitivity[k], 0.0) for k in names], dtype=np.float64)
    inv = (vals.mean() / np.maximum(vals, vals.mean() * 1e-6)) ** alpha  # 越敏感 → 越小
    sp = budget * inv / inv.mean()
    for _ in range(20):  # 迭代投影：逼近"边界内均值=budget"的解
        sp = np.clip(sp * (budget / sp.mean()), min_s, max_s)
    return {k: round(float(v), 4) for k, v in zip(names, sp)}
