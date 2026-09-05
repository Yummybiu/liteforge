"""统一损失计价器测试：trace 恒等式、菜单单调性、模型不被修改。"""

import torch

from liteforge.lossmeter import (
    compression_loss,
    effective_bits_prune,
    effective_bits_quant,
    measure_menus,
)


def test_trace_identity():
    """核心数学：||X ΔWᵀ||²_F == tr(ΔW H ΔWᵀ)，H = XᵀX。

    全程 float64（float32 的累积误差在 1e-6 量级，达不到 1e-9 的验证要求——
    这本身就是一次口径教训：数值验证必须在正确精度下做）。
    """
    g = torch.Generator().manual_seed(3)
    n_in, n_out, T = 32, 16, 256
    A = torch.randn(n_in, n_in, generator=g, dtype=torch.float64) / (n_in ** 0.5)
    X = torch.randn(T, n_in, generator=g, dtype=torch.float64) @ A.T
    W = torch.randn(n_out, n_in, generator=g, dtype=torch.float64) * 0.5
    from liteforge.quant.rtn import quantize_tensor
    Wq = quantize_tensor(W, 4, 8, False)
    dW = W - Wq
    empirical = (X @ dW.T).pow(2).sum().item()
    H = X.T @ X
    analytic = compression_loss(H, Wq, W)
    rel = abs(empirical - analytic) / max(empirical, 1e-9)
    assert rel < 1e-10, f"trace 恒等式偏差 {rel:.2e}"


def test_effective_bits_accounting():
    assert effective_bits_quant(4, 128, symmetric=False) == 4.25   # +32/128
    assert effective_bits_quant(4, 128, symmetric=True) == 4.125   # +16/128
    assert effective_bits_quant(4, 0, symmetric=False) == 4.0      # per-ch 无开销
    assert effective_bits_prune(0.5) == 8.0
    assert effective_bits_prune(0.75) == 4.0


def test_measure_menus_tiny_model(tiny_model, tiny_tokenizer, sample_text):
    from liteforge.data.text import BlockBatcher
    from liteforge.utils import find_linears
    linears = find_linears(tiny_model, exclude=("lm_head", "embed_out"))
    batcher = BlockBatcher(tiny_tokenizer, sample_text, block_size=64, batch_size=2)

    # 快照一组权重，测完核对未被修改
    w0 = linears[0][1].weight.data.clone()

    table = measure_menus(tiny_model, linears, batcher, max_batches=3,
                          prune_mode="static")

    assert torch.equal(linears[0][1].weight.data, w0), "measure_menus 不得修改模型"
    name = linears[0][0]
    entry = table[name]
    assert "shape" in entry and "options" in entry
    o = entry["options"]
    assert o["fp16"]["loss"] == 0.0
    # 位宽单调：bit 越低损失越大（RTN 误差单调性在真实 H 下成立）
    assert o["w8g128"]["loss"] < o["w4g128"]["loss"] < o["w3g128"]["loss"]
    assert o["p50"]["loss"] > 0
    # 剪枝 50%(8bit) 与量化 w8(8.0625bit) 近同比特——它们的相对大小由数据决定，
    # 这里只验证字段完整性
    for k in ("w8g128", "w4g128", "w3g128", "w2g128", "p25", "p50", "p75"):
        assert k in o and "bits_eff" in o[k] and "loss" in o[k]


def test_measure_menus_chunked(tiny_model, tiny_tokenizer, sample_text):
    """chunk 模式与全量模式结果一致（分块只是内存策略）。"""
    from liteforge.data.text import BlockBatcher
    from liteforge.utils import find_linears
    linears = find_linears(tiny_model, exclude=("lm_head", "embed_out"))
    batcher = BlockBatcher(tiny_tokenizer, sample_text, block_size=64, batch_size=2)
    t_full = measure_menus(tiny_model, linears, batcher, max_batches=3,
                           prune_mode="static")
    t_chunk = measure_menus(tiny_model, linears, batcher, max_batches=3,
                            chunk=3, prune_mode="static")
    for l in t_full:
        for o in t_full[l]["options"]:
            assert abs(t_full[l]["options"][o]["loss"]
                       - t_chunk[l]["options"][o]["loss"]) < 1e-6 * max(
                t_full[l]["options"][o]["loss"], 1e-9)
