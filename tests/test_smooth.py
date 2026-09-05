"""SmoothQuant 测试：等价变换数学、平滑有效性（离群注入）、量化算子。"""

import torch

from liteforge.smooth import (
    _batch_w8a8_loss,
    quant_a8_pertoken,
    quant_w8_perchannel,
    smooth_scale,
)


def test_transform_equivalence():
    """(X/s)·(W·s)ᵀ == X·Wᵀ（平滑变换数学等价，float64 验证）。"""
    g = torch.Generator().manual_seed(0)
    X = torch.randn(64, 32, generator=g, dtype=torch.float64)
    W = torch.randn(16, 32, generator=g, dtype=torch.float64)
    s = torch.rand(32, generator=g, dtype=torch.float64) + 0.5
    lhs = (X / s) @ (W * s.unsqueeze(0)).T
    rhs = X @ W.T
    assert torch.allclose(lhs, rhs, rtol=1e-12)


def test_quant_operators_on_grid():
    W = torch.randn(8, 64) * 0.3
    Wq = quant_w8_perchannel(W)
    X = torch.randn(4, 64)
    Xq = quant_a8_pertoken(X)
    # 各自的界：反量化最大值不超过输入幅度（127 级对称网格）
    assert Wq.abs().max() <= W.abs().max() + 1e-6
    assert Xq.abs().max() <= X.abs().max() + 1e-6
    # 反量化值应落在格点附近（scale 步长的整数倍）
    scale_w = W.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127
    ratio = Wq / scale_w
    assert (ratio.round() - ratio).abs().max() < 5e-3
    scale_x = X.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127
    ratio_x = Xq / scale_x
    assert (ratio_x.round() - ratio_x).abs().max() < 5e-3


def test_smooth_beats_no_smooth_with_outliers():
    """文献核心结论的复现：激活离群（真实 regime：离群通道权重列小、信号贡献低）下，
    最优 α 的 W8A8 损失应远低于无平滑。

    注意参数序：smooth_scale(w_max, act_max, alpha)——w 在前。
    （本测试曾因 act/w 传反而"发现"平滑有害的假象——D16 教训：
    实现与理论矛盾时，第一步核对调用契约，而非构造理论解释。）
    """
    g = torch.Generator().manual_seed(1)
    n_in, n_out, T = 64, 32, 512
    X = torch.randn(T, n_in, generator=g)
    outlier_channels = [5, 17, 40]
    X[:, outlier_channels] *= 50.0
    W = torch.randn(n_out, n_in, generator=g) * 0.3
    W[:, outlier_channels] *= 0.005   # 离群激活通道的权重列很小（信号贡献可忽略）

    def total_loss(alpha):
        w_max = W.abs().amax(dim=0)
        act_max = X.abs().amax(dim=0)
        if alpha is None:
            s = torch.ones(n_in)
        else:
            s = smooth_scale(w_max, act_max, alpha)   # w 在前！
        return float(_batch_w8a8_loss(X, W, s).item())

    loss_no = total_loss(None)
    losses = {a: total_loss(a) for a in (0.4, 0.5, 0.6, 0.7)}
    best = min(losses.values())
    assert best < loss_no * 0.2, \
        f"平滑应大幅降低 W8A8 损失: best={best:.1f} vs no_smooth={loss_no:.1f}"


def test_smooth_scale_extremes():
    """α=1 → s 完全由激活决定；α=0 → 完全由权重决定。"""
    act = torch.tensor([1.0, 100.0])
    w = torch.tensor([1.0, 1.0])
    s1 = smooth_scale(w, act, alpha=1.0)
    s0 = smooth_scale(w, act, alpha=0.0)
    assert torch.allclose(s1, act)
    assert torch.allclose(s0, torch.ones_like(s0))
