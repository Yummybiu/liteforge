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
