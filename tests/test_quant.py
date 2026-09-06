"""RTN 伪量化测试：误差单调性、对称/非对称、分组、restore。"""

import torch

from liteforge.quant import RTNConfig, RTNQuantizer, quantize_tensor


def _rand_weight():
    g = torch.Generator().manual_seed(0)
    return torch.randn(64, 128, generator=g)


def test_quant_error_monotonic_in_bits():
    w = _rand_weight()
    err4 = (quantize_tensor(w, 4, 0, False) - w).abs().mean()
    err8 = (quantize_tensor(w, 8, 0, False) - w).abs().mean()
    assert err8 < err4, "8bit 误差应小于 4bit"


def test_group_quant_closer_than_per_channel_at_same_bits():
    """分组量化（group_size=128, 非对称）应比 per-channel 非对称误差更小。"""
    w = _rand_weight()
    err_pc = (quantize_tensor(w, 4, 0, False) - w).abs().mean()
    err_g = (quantize_tensor(w, 4, 32, False) - w).abs().mean()
    assert err_g < err_pc


def test_symmetric_per_channel_small_error_int8():
    w = _rand_weight() * 0.01
    deq = quantize_tensor(w, 8, 0, True)
    rel = ((deq - w).abs().mean() / w.abs().mean()).item()
    assert rel < 0.02, f"int8 对称量化相对误差 {rel:.3%} 应 <2%"


def test_quantizer_quantize_and_restore(tiny_model):
    from liteforge.utils import find_linears
    name, m = find_linears(tiny_model, exclude=("lm_head",))[0]
    orig = m.weight.data.clone()
    q = RTNQuantizer(tiny_model, RTNConfig(bits=4, group_size=16))
    report = q.quantize_()
    assert report["n_layers"] > 0
    assert not torch.equal(m.weight.data, orig), "量化后权重应发生变化"
    q.restore()
    assert torch.equal(m.weight.data, orig), "restore 后应回到原始权重"


def test_rtn_pad_branch_stays_groupwise():
    """回归：n_in 不整除 group_size 时，pad 分支曾静默退化成 per-row（审查 F2）。"""
    g = torch.Generator().manual_seed(0)
    w = torch.randn(8, 100, generator=g)          # 100 % 32 = 4 → pad 到 128
    deq = quantize_tensor(w, 4, 32, False)
    err_pad = (deq - w).abs().mean().item()
    ref = quantize_tensor(w, 4, 0, False)          # per-row 参照
    err_ref = (ref - w).abs().mean().item()
    assert err_pad < err_ref * 1.05, "pad 分支的分组量化不应差于 per-row"


def test_gptq_symmetric_grid():
    """回归：对称模式曾把负权重全部置零（审查 F3）。"""
    from liteforge.quant.gptq import _gptq_layer
    from liteforge.utils.hessian import damp_inverse
    g = torch.Generator().manual_seed(2)
    n_in, n_out, T = 64, 16, 256
    A = torch.randn(n_in, n_in, generator=g) / (n_in ** 0.5)
    X = torch.randn(T, n_in, generator=g) @ A.T
    W = torch.randn(n_out, n_in, generator=g) * 0.3
    Hinv = damp_inverse(X.T @ X)
    U = torch.linalg.cholesky(Hinv, upper=True).float()
    Wq = _gptq_layer(W.clone(), U, bits=4, group_size=32, symmetric=True)
    neg_kept = (Wq[W < 0] != 0).float().mean().item()
    assert neg_kept > 0.9, f"对称量化应保留负权重（保留率 {neg_kept:.2f}）"
